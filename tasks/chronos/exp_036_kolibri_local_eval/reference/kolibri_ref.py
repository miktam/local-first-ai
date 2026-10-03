# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""Independent numpy reference of the Kolibri-1 forward pass.

This module is the numerical oracle that gates the exp_036 MLX port. It was
written from the vendor vLLM plugin (aleph_alpha_inference/kolibri1.py at
commit 049a6a7), the vLLM v0.29 sources it builds on, and the numbered
forward-pass specification ("spec item N" in the comments). It shares no code
with the port.

Modes (HYPOTHESIS "Reference", BUILD_SPEC 5.1b):

* fp32 (the reference proper): every activation and every matmul is
  float32. Checkpoint weights (BF16) are upcast exactly to float32 before
  use; nothing is rounded back to bf16. It is a higher-precision version of
  the model that both the vendor's bf16/FP8 serving and the MLX port
  approximate.
* bf16 emulation (emulate_bf16=True): rounds to bf16 (nearest even) wherever
  vLLM stores bf16 -- norm outputs, q/k/v, RoPE output, attention output,
  o_proj, each expert output, the residual after the fp32 add -- and keeps
  router logits, routing weights and the head in fp32. The residual is
  threaded as vLLM's fused_add_rms_norm does: the sum is formed in fp32, the
  stored residual is that sum rounded to bf16, and the following norm
  normalises the unrounded sum (spec item 5). It calibrates what a correct
  bf16 implementation achieves on given weights (the gate's "emu").
* dequantised (dequant_dir=...): the weights come from a converted MLX
  directory, unpacked by mlx_affine_np (written from the format, no mlx), each
  dequantised as the MLX op that consumes it does. Combines with either
  precision mode. Used by G2q.

Memory: weights are streamed one layer at a time and, inside the MoE, one
expert at a time; only experts that at least one token routes to are read.
Peak RAM is a few hundred MB plus activations, logits [T, V] and (optional)
hidden states, independent of the 156 GB checkpoint size. forward_streams
carries several model variants (the G3 mutants) through one pass and keeps
one layer's weights (up to ~6.2 GB fp32 on the real model) while it does.
"""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass

import numpy as np

try:  # package import (python -m reference.kolibri_ref, tests)
    from .mlx_affine_np import ConvertedCheckpoint, round_bf16
    from .safetensors_np import INDEX_FILE, SINGLE_FILE, open_checkpoint, read_header
except ImportError:  # direct script execution from inside reference/
    from mlx_affine_np import ConvertedCheckpoint, round_bf16
    from safetensors_np import INDEX_FILE, SINGLE_FILE, open_checkpoint, read_header

REF_VERSION = "exp036-ref-2"

SLIDING = "sliding_attention"
FULL = "full_attention"

# Score-buffer budget for full-attention query chunks (bytes of fp32 scores;
# the buffer is reused in place for every chunk).
_ATTN_SCORE_BUDGET = 128 * 1024 * 1024
# LM-head rows converted to fp32 at a time (16384 x 2560 x 4 B = 168 MB).
_LM_HEAD_CHUNK = 16384


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KolibriConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    sliding_window: int | None
    layer_types: tuple[str, ...]
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, d: dict) -> "KolibriConfig":
        """Parse a Kolibri-1 config.json dict and reject anything the
        reference does not implement (rather than silently diverging)."""
        if d.get("model_type") not in (None, "kolibri1"):
            raise ValueError(f"model_type {d.get('model_type')!r} is not kolibri1")
        if d.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"hidden_act {d['hidden_act']!r} (only silu)")
        if d.get("attention_bias", False):
            raise NotImplementedError("attention_bias=True (Kolibri-1 has no attention bias)")

        # RoPE base: transformers moves top-level rope_theta into
        # rope_parameters; accept either, require agreement (spec item 2).
        theta = d.get("rope_theta")
        rp = d.get("rope_parameters") or {}
        if "rope_theta" in rp:
            if theta is not None and float(theta) != float(rp["rope_theta"]):
                raise ValueError("rope_theta and rope_parameters.rope_theta disagree")
            theta = rp["rope_theta"]
        if rp.get("rope_type", "default") != "default":
            raise NotImplementedError(f"rope_type {rp['rope_type']!r} (only default)")
        rs = d.get("rope_scaling")
        if rs and (rs.get("rope_type") or rs.get("type")) not in (None, "default"):
            raise NotImplementedError(f"rope_scaling {rs!r}")
        if float(d.get("partial_rotary_factor", rp.get("partial_rotary_factor", 1.0))) != 1.0:
            raise NotImplementedError("partial_rotary_factor != 1.0")
        if theta is None:
            raise ValueError("config has no rope_theta")

        n_layers = int(d["num_hidden_layers"])
        layer_types = d.get("layer_types")
        if layer_types is None:
            raise ValueError("config has no layer_types; the reference needs them explicit")
        layer_types = tuple(layer_types)
        if len(layer_types) != n_layers:
            raise ValueError(f"{len(layer_types)} layer_types for {n_layers} layers")
        bad = sorted(set(layer_types) - {SLIDING, FULL})
        if bad:
            raise ValueError(f"unknown layer_types {bad}")

        # Sliding layers need a positive window. Qwen3MoeConfig (the vendor
        # config's base class) defaults use_sliding_window to False and then
        # nulls sliding_window, and the vendor attention raises (kolibri1.py,
        # spec item 2). So a missing key means false, as there.
        window = d.get("sliding_window")
        if SLIDING in layer_types:
            if not d.get("use_sliding_window", False):
                raise ValueError("sliding layers present but use_sliding_window is false or missing")
            if window is None or int(window) <= 0:
                raise ValueError(f"sliding layers need a positive sliding_window, got {window}")
            window = int(window)

        n_heads = int(d["num_attention_heads"])
        n_kv = int(d["num_key_value_heads"])
        if n_heads % n_kv:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        head_dim = int(d.get("head_dim") or int(d["hidden_size"]) // n_heads)
        if head_dim % 2:
            raise ValueError("head_dim must be even for half-rotation RoPE")

        return cls(
            hidden_size=int(d["hidden_size"]),
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv,
            head_dim=head_dim,
            vocab_size=int(d["vocab_size"]),
            rms_norm_eps=float(d.get("rms_norm_eps", 1e-6)),
            rope_theta=float(theta),
            max_position_embeddings=int(d.get("max_position_embeddings", 1 << 62)),
            sliding_window=window,
            layer_types=layer_types,
            num_experts=int(d["num_experts"]),
            num_experts_per_tok=int(d["num_experts_per_tok"]),
            moe_intermediate_size=int(d["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(d["shared_expert_intermediate_size"]),
            norm_topk_prob=bool(d.get("norm_topk_prob", False)),
            tie_word_embeddings=bool(d.get("tie_word_embeddings", False)),
        )


def load_config(model_dir: str) -> KolibriConfig:
    with open(os.path.join(model_dir, "config.json")) as f:
        return KolibriConfig.from_dict(json.load(f))


def expected_shapes(cfg: KolibriConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the reference reads, with its checkpoint shape (spec item 3)."""
    H, D = cfg.hidden_size, cfg.head_dim
    q_dim = cfg.num_attention_heads * D
    kv_dim = cfg.num_key_value_heads * D
    E, I, Is = cfg.num_experts, cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size
    shapes = {
        "model.embed_tokens.weight": (cfg.vocab_size, H),
        "model.norm.weight": (H,),
    }
    if not cfg.tie_word_embeddings:
        shapes["lm_head.weight"] = (cfg.vocab_size, H)
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}"
        for norm in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            shapes[f"{p}.{norm}.weight"] = (H,)
        shapes[f"{p}.self_attn.q_proj.weight"] = (q_dim, H)
        shapes[f"{p}.self_attn.k_proj.weight"] = (kv_dim, H)
        shapes[f"{p}.self_attn.v_proj.weight"] = (kv_dim, H)
        shapes[f"{p}.self_attn.o_proj.weight"] = (H, q_dim)
        shapes[f"{p}.self_attn.q_norm.weight"] = (D,)
        shapes[f"{p}.self_attn.k_norm.weight"] = (D,)
        shapes[f"{p}.mlp.gate.weight"] = (E, H)
        shapes[f"{p}.moe.router.expert_bias"] = (E,)
        for e in range(E):
            ep = f"{p}.mlp.experts.{e}"
            shapes[f"{ep}.gate_proj.weight"] = (I, H)
            shapes[f"{ep}.up_proj.weight"] = (I, H)
            shapes[f"{ep}.down_proj.weight"] = (H, I)
        sp = f"{p}.mlp.shared_experts"
        shapes[f"{sp}.gate_proj.weight"] = (Is, H)
        shapes[f"{sp}.up_proj.weight"] = (Is, H)
        shapes[f"{sp}.down_proj.weight"] = (H, Is)
    return shapes


def param_count(cfg: KolibriConfig) -> int:
    return sum(int(np.prod(s, dtype=np.int64)) for s in expected_shapes(cfg).values())


# ---------------------------------------------------------------------------
# Pure numerical building blocks (all float32 in, float32 out)
# ---------------------------------------------------------------------------


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """y = w * x / sqrt(mean(x^2) + eps) over the last axis (spec item 6).
    Plain w, NOT (1 + w): the vendor uses vLLM RMSNorm, not GemmaRMSNorm."""
    x = np.asarray(x, dtype=np.float32)
    var = np.mean(x * x, axis=-1, keepdims=True)
    y = x * (np.float32(1.0) / np.sqrt(var + np.float32(eps)))
    return y * np.asarray(w, dtype=np.float32)


def rope_inv_freq(head_dim: int, theta: float) -> np.ndarray:
    """inv_freq[j] = 1 / theta^(2j / head_dim), computed in fp32 as vLLM's
    RotaryEmbedding._compute_inv_freq does (spec item 8)."""
    exponents = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    return np.float32(1.0) / (np.float32(theta) ** exponents)


def rope_neox(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """NeoX (half-rotation) RoPE, the vLLM get_rope default is_neox_style=True
    (spec item 8; MLX nn.RoPE(traditional=False) is the same layout).

    x: [T, n_heads, head_dim]; positions: [T] absolute 0-based positions.
    With x = [x1, x2] split into halves and angle a[t, j] = pos[t] * inv_freq[j]:
        out = [x1 * cos(a) - x2 * sin(a),  x2 * cos(a) + x1 * sin(a)]
    Angles, cos and sin are fp32, like vLLM's fp32 cos/sin cache.
    """
    x = np.asarray(x, dtype=np.float32)
    D = x.shape[-1]
    half = D // 2
    inv_freq = rope_inv_freq(D, theta)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1)
    ang = pos[:, None] * inv_freq[None, :]  # [T, D/2], one fp32 rounding as in vLLM
    cos = np.cos(ang)[:, None, :]  # broadcast over heads
    sin = np.sin(ang)[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention_mask(q_pos: np.ndarray, k_pos: np.ndarray, window: int | None) -> np.ndarray:
    """Boolean [Tq, Tk] mask, True where query position i may attend key j.

    Full layers (window None): plain causal, j <= i (spec item 9).
    Sliding layers: i - (window - 1) <= j <= i, i.e. `window` keys including
    the query itself. vLLM FlashAttention turns per_layer_sliding_window=W
    into window_size=(W-1, 0) (flash_attn.py), so W=513 means 512 previous
    tokens plus the current one."""
    q = np.asarray(q_pos)[:, None]
    k = np.asarray(k_pos)[None, :]
    allowed = k <= q
    if window is not None:
        allowed &= k >= q - (window - 1)
    return allowed


def sdpa_ref(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    window: int | None,
    q_chunk: int | None = None,
) -> np.ndarray:
    """Masked softmax attention over a whole sequence with GQA.

    q: [T, n_heads, D]; k, v: [T, n_kv, D]. Query head h reads kv head
    h // (n_heads // n_kv) (vLLM / flash-attn grouping, spec item 7).
    Token index = position (sequence starts at 0). Queries are processed in
    chunks so the score matrix stays bounded; each chunk only looks at the key
    range that can be unmasked for it, and the explicit mask is applied inside
    that range. Masked scores are -inf, so they contribute exactly zero.
    Returns [T, n_heads, D] float32.
    """
    T, n_heads, D = q.shape
    n_kv = k.shape[1]
    rep = n_heads // n_kv
    # [n_kv, rep, T, D] for queries, [n_kv, 1, T, D] for keys/values.
    qg = np.ascontiguousarray(q.reshape(T, n_kv, rep, D).transpose(1, 2, 0, 3))
    kg = np.ascontiguousarray(k.transpose(1, 0, 2))[:, None]
    vg = np.ascontiguousarray(v.transpose(1, 0, 2))[:, None]
    if q_chunk is None:
        if window is None:
            # rows * T keys * heads * 4 bytes <= budget
            q_chunk = max(16, _ATTN_SCORE_BUDGET // (n_heads * T * 4))
        else:
            # a chunk of `window` rows sees at most 2 * window - 1 keys
            q_chunk = window
    q_chunk = min(T, max(1, int(q_chunk)))
    max_keys = T if window is None else min(T, q_chunk + window - 1)
    out = np.empty((n_kv, rep, T, D), dtype=np.float32)
    # One score buffer reused for every chunk. Allocating a fresh array of a
    # different size per chunk made the macOS allocator hold on to the freed
    # blocks (peak RSS 5.6 GB instead of 0.4 GB at T=8192).
    s_buf = np.empty((n_kv, rep, q_chunk, max_keys), dtype=np.float32)
    scale = np.float32(scale)
    for a in range(0, T, q_chunk):
        b = min(T, a + q_chunk)
        k0 = 0 if window is None else max(0, a - (window - 1))
        mask = attention_mask(np.arange(a, b), np.arange(k0, b), window)  # [Tc, Tk]
        s = s_buf[:, :, : b - a, : b - k0]  # [n_kv, rep, Tc, Tk]
        np.matmul(qg[:, :, a:b], kg[:, :, k0:b].swapaxes(-1, -2), out=s)
        s *= scale
        np.copyto(s, np.float32(-np.inf), where=~mask)
        s -= s.max(axis=-1, keepdims=True)  # every row has >= 1 allowed key (itself)
        np.exp(s, out=s)  # masked entries become exactly 0
        s /= s.sum(axis=-1, keepdims=True)
        np.matmul(s, vg[:, :, k0:b], out=out[:, :, a:b])
    # back to [T, n_heads, D] with head h = g * rep + r
    return out.transpose(2, 0, 1, 3).reshape(T, n_heads, D)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function in fp32."""
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, np.float32(1.0) / (np.float32(1.0) + e), e / (np.float32(1.0) + e))


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x * sigmoid(x)


def route_ref(
    logits_f32: np.ndarray, bias_f32: np.ndarray, k: int, renormalize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Kolibri-1 router, the vendor's sigmoid_logit_add_routing (spec items 11-12).

    Selection is top-k over (logits + expert_bias); the weights are the
    UNBIASED sigmoid(logits) of the selected experts. With the shipped config
    (norm_topk_prob=False) the weights are not renormalised, and the routed
    scale is 1.0. This differs from vLLM's built-in sigmoid scoring and from
    DeepSeek-V3/afmoe, which select on sigmoid(logits) + bias.

    logits_f32: [T, E] fp32 router logits; bias_f32: [E].
    Returns (weights [T, k] float32, ids [T, k] int64), ids ordered by
    descending biased score (as torch.topk(sorted=True); ties -> lower id).
    """
    logits = np.asarray(logits_f32, dtype=np.float32)
    choice = logits + np.asarray(bias_f32, dtype=np.float32)[None, :]
    ids = np.argsort(-choice, axis=-1, kind="stable")[:, :k].astype(np.int64)
    weights = sigmoid(np.take_along_axis(logits, ids, axis=-1))
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20))
    return weights.astype(np.float32), ids


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x [N, in] @ w[out, in]^T in fp32, always as a matrix-matrix product.

    BLAS runs a one-row product as gemv, whose summation order differs from
    gemm (observed with Accelerate); for N >= 2 each output row was observed
    to be bit-identical whatever N is. Running a single row twice therefore
    keeps every token's result independent of how many other tokens share the
    product (an expert group, a batch), which makes the reference exactly
    causal and batch-invariant on the same machine."""
    if x.shape[0] == 1:
        return (np.repeat(x, 2, axis=0) @ w.T)[:1]
    return x @ w.T


def expert_mlp(x: np.ndarray, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> np.ndarray:
    """SwiGLU expert: down(silu(gate(x)) * up(x)) (spec item 13). Used for both
    the routed experts and the ungated shared expert. x: [N, H]; weights in
    checkpoint layout [out, in]."""
    g = linear(x, w_gate)
    u = linear(x, w_up)
    return linear(silu(g) * u, w_down)




def forced_route_ref(logits_f32: np.ndarray, ids, renormalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Weights for an externally fixed selection (the gate's "forced"
    comparisons): sigmoid of the raw logits at the given ids, renormalised
    only if asked, exactly as route_ref weights its own selection (spec item 12).
    Returns (weights [T, k] float32, ids [T, k] int64)."""
    logits = np.asarray(logits_f32, dtype=np.float32)
    ids = np.asarray(ids).astype(np.int64)
    if ids.ndim != 2 or ids.shape[0] != logits.shape[0]:
        raise ValueError(f"force_ids must be [T, k] with T = {logits.shape[0]}, got {ids.shape}")
    weights = sigmoid(np.take_along_axis(logits, ids, axis=-1))
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20))
    return weights.astype(np.float32), ids


def reference_tree_sha256(root: str | None = None) -> str:
    """sha256 over the sorted lines "relpath<TAB>sha256<LF>" of every file in
    the reference/ directory: the identity of the reference code a dump was
    made with. The rule is tools/hash_tree.py's tree rule (bytecode caches,
    .pytest_cache and .DS_Store excluded), computed without git, so it equals
    the HYPOTHESIS REFERENCE_SHA256 value while reference/ holds no
    git-ignored files."""
    root = os.path.dirname(os.path.abspath(__file__)) if root is None else root
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", ".pytest_cache"))
        for fn in filenames:
            if fn.endswith((".pyc", ".pyo")) or fn == ".DS_Store":
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            lines.append(f"{rel}\t{_sha256_file(path)}\n")
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Layer-streaming model
# ---------------------------------------------------------------------------


class KolibriReference:
    """Kolibri-1 forward pass in numpy, streamed from a safetensors checkpoint
    one layer (and one expert) at a time.

        ref = KolibriReference(model_dir)                    # fp32
        emu = KolibriReference(model_dir, emulate_bf16=True) # bf16 emulation
        deq = KolibriReference(model_dir, dequant_dir=K8)    # converted weights
        logits = ref.forward(ids)                       # [T, V] float32
        logits, layers, final = ref.forward(ids, return_hidden=True)
        # layers: list of num_hidden_layers arrays [T, H], the residual
        #         stream after each decoder layer; final: model.norm output.
        results = ref.forward_batch([ids_a, ids_b])     # one weight pass
        logits, hidden, final, lengths = ref.forward_packed([ids_a, ids_b])
        out = ref.layer_branches(i, h, force_ids=top6)  # one layer, all branches
        streams = ref.forward_streams(ids, {"m": MutantClass})

    forward_batch / forward_packed run several independent sequences (each
    from position 0) through one pass over the checkpoint: attention is per
    sequence, while the per-token parts (norms, MoE, LM head) run on all
    tokens together, so each layer and expert is read once per batch rather
    than once per sequence. Results are bit-identical to separate forwards
    (checked on Accelerate; see linear()).

    After every forward, `self.last_routing` holds (weights, ids) [N, k] per
    layer for the N tokens of the batch (sequences concatenated in order) and
    `self.last_stats` holds per-layer timing and expert counts.

    Hooks. Each semantic choice the G3 mutants (reference/mutants.py) vary is
    one method, so a mutant is a subclass that overrides exactly one:
    norm, rope, uses_rope, qk_norm_rope, route, layer_norm_names.
    """

    def __init__(self, model_dir: str, attn_chunk: int | None = None, emulate_bf16: bool = False,
                 dequant_dir: str | None = None, verbose: bool = False):
        self.model_dir = model_dir
        self.cfg = load_config(model_dir)
        self.emulate_bf16 = bool(emulate_bf16)
        self.dequant_dir = dequant_dir
        if dequant_dir is None:
            self.ckpt = open_checkpoint(model_dir)
        else:
            self.ckpt = ConvertedCheckpoint(dequant_dir)
            other = KolibriConfig.from_dict(self.ckpt.config)
            if other != self.cfg:
                raise ValueError(f"{dequant_dir}: architecture differs from {model_dir}'s config.json")
        self.attn_chunk = attn_chunk
        self.verbose = verbose
        self._shapes = expected_shapes(self.cfg)
        self._check_checkpoint()
        self.last_routing: list[tuple[np.ndarray, np.ndarray]] = []
        self.last_stats: list[dict] = []
        self.last_layer_info: dict | None = None  # layer info of the latest layer
        # Weight cache shared with the stream clones of forward_streams; None
        # outside a streamed layer.
        self._shared = {"cache": None}

    @property
    def mode(self) -> str:
        return "bf16_emulation" if self.emulate_bf16 else "fp32"

    # ---- checkpoint access ---------------------------------------------

    def _check_checkpoint(self) -> None:
        """Header-only check that every tensor the forward reads exists with
        the expected shape and a float dtype. Reads no tensor data."""
        missing, wrong = [], []
        for name, shape in self._shapes.items():
            if name not in self.ckpt:
                missing.append(name)
            elif tuple(self.ckpt.shape_of(name)) != shape:
                wrong.append(f"{name}: {self.ckpt.shape_of(name)} != {shape}")
            elif self.ckpt.dtype_of(name) not in ("BF16", "F16", "F32"):
                wrong.append(f"{name}: dtype {self.ckpt.dtype_of(name)}")
        if missing or wrong:
            raise ValueError(
                f"checkpoint does not match config: {len(missing)} missing "
                f"(e.g. {missing[:3]}), {len(wrong)} mismatched (e.g. {wrong[:3]})"
            )

    def _w(self, name: str) -> np.ndarray:
        """One tensor as a float32 array (exact upcast from BF16, or the
        dequantised value). Inside forward_streams the array is cached for the
        layer and shared, read-only, by every stream."""
        cache = self._shared["cache"]
        if cache is None:
            return self.ckpt.get(name)
        if name not in cache:
            arr = self.ckpt.get(name)
            arr.flags.writeable = False
            cache[name] = arr
        return cache[name]

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _store(self, x: np.ndarray) -> np.ndarray:
        """A value vLLM stores in bf16: rounded in emulation mode, unchanged in fp32."""
        return round_bf16(x) if self.emulate_bf16 else x

    # ---- hooks (one semantic choice each; see reference/mutants.py) ------

    def norm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """RMSNorm over the last axis, plain w (spec item 6). In emulation,
        vLLM's ir rms_norm: normalise in fp32, round to bf16, multiply by the
        bf16 weight, round again."""
        if not self.emulate_bf16:
            return rms_norm(x, w, self.cfg.rms_norm_eps)
        x = np.asarray(x, dtype=np.float32)
        var = np.mean(x * x, axis=-1, keepdims=True)
        y = round_bf16(x * (np.float32(1.0) / np.sqrt(var + np.float32(self.cfg.rms_norm_eps))))
        return round_bf16(y * np.asarray(w, dtype=np.float32))

    def rope(self, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """NeoX half-rotation RoPE with fp32 angles (spec item 8)."""
        return self._store(rope_neox(x, positions, self.cfg.rope_theta))

    def uses_rope(self, i: int) -> bool:
        """RoPE on sliding layers only; full layers are NoPE (spec item 9)."""
        return self.cfg.layer_types[i] == SLIDING

    def qk_norm_rope(self, i: int, q: np.ndarray, k: np.ndarray, positions: np.ndarray):
        """Per-head q/k RMSNorm in every layer, BEFORE RoPE (spec item 7)."""
        p = f"model.layers.{i}.self_attn"
        q = self.norm(q, self._w(f"{p}.q_norm.weight"))
        k = self.norm(k, self._w(f"{p}.k_norm.weight"))
        if self.uses_rope(i):
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        return q, k

    def route(self, logits: np.ndarray, bias: np.ndarray, k: int):
        """Selection and weights (spec items 11-12): route_ref, looked up at
        call time so tests can wrap it."""
        return route_ref(logits, bias, k, self.cfg.norm_topk_prob)

    def layer_norm_names(self) -> dict:
        """Checkpoint names of the four layer norms by role (spec item 5).
        NAMING TRAP: post_attention_layernorm is the PRE-MoE norm (Qwen
        naming); post_attn_norm and post_ffn_norm are the sandwich norms."""
        return {
            "input": "input_layernorm",
            "post_attn": "post_attn_norm",
            "pre_moe": "post_attention_layernorm",
            "post_ffn": "post_ffn_norm",
        }

    # ---- model pieces ---------------------------------------------------

    def embed(self, ids: np.ndarray) -> np.ndarray:
        """Plain row lookup, no sqrt(hidden) scaling (spec item 4)."""
        return self.ckpt.get_rows("model.embed_tokens.weight", ids)

    def _proj(self, x: np.ndarray, name: str) -> np.ndarray:
        """A bf16 GEMM output in vLLM (fp32 accumulation, stored bf16)."""
        return self._store(linear(x, self._w(name)))

    def expert(self, x: np.ndarray, prefix: str) -> np.ndarray:
        """SwiGLU expert down(silu(gate(x)) * up(x)) (spec item 13). In
        emulation each stored intermediate is rounded: the gate and up GEMM
        outputs, silu (vLLM's silu kernel returns bf16), the product, and
        the expert output."""
        wg = self._w(f"{prefix}.gate_proj.weight")
        wu = self._w(f"{prefix}.up_proj.weight")
        wd = self._w(f"{prefix}.down_proj.weight")
        if not self.emulate_bf16:
            return expert_mlp(x, wg, wu, wd)
        g = round_bf16(linear(x, wg))
        u = round_bf16(linear(x, wu))
        a = round_bf16(round_bf16(silu(g)) * u)
        return round_bf16(linear(a, wd))

    def attention_block(self, i: int, x: np.ndarray, segments: list[tuple[int, int]] | None = None) -> np.ndarray:
        """Self-attention of layer i on the normed input x [N, H] (spec item 7).
        segments: (start, end) row ranges of independent sequences, each
        starting at position 0; default one sequence covering all rows."""
        cfg = self.cfg
        p = f"model.layers.{i}.self_attn"
        N = x.shape[0]
        segments = [(0, N)] if segments is None else segments
        nh, nkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = self._proj(x, f"{p}.q_proj.weight").reshape(N, nh, D)
        k = self._proj(x, f"{p}.k_proj.weight").reshape(N, nkv, D)
        v = self._proj(x, f"{p}.v_proj.weight").reshape(N, nkv, D)
        # absolute positions, restarting at 0 for every sequence
        positions = np.concatenate([np.arange(b - a) for a, b in segments])
        # qk-norm per head over head_dim, in every layer, BEFORE RoPE; v is not normed.
        q, k = self.qk_norm_rope(i, q, k, positions)
        # The window is a property of the layer type; full layers: plain causal mask.
        window = cfg.sliding_window if cfg.layer_types[i] == SLIDING else None
        o = np.empty_like(q)
        for a, b in segments:
            o[a:b] = sdpa_ref(q[a:b], k[a:b], v[a:b], D**-0.5, window, self.attn_chunk)
        o = self._store(o)
        return self._proj(o.reshape(N, nh * D), f"{p}.o_proj.weight")

    def moe_block(self, i: int, x: np.ndarray, force_ids=None) -> tuple[np.ndarray, dict]:
        """Routed experts + ungated shared expert on the normed input x [N, H]
        (spec items 11-13). Purely per token. Returns (output [N, H], info):

            logits   [N, E] fp32  raw router logits
            biased   [N, E] fp32  logits + expert_bias, the selection scores
            top6     [N, k] int64 ids used: by descending biased score, or
                                  force_ids as given (k = 6 for Kolibri-1)
            ids      the same array as top6 (older name)
            weights  [N, k] fp32  sigmoid(logits[top6]), not renormalised
            forced   bool         whether force_ids replaced the selection
            experts_loaded        number of distinct experts read

        force_ids [N, k] (optional) fixes the selection; the weights still
        come from these logits (the gate's forced comparisons, G2q)."""
        cfg = self.cfg
        p = f"model.layers.{i}"
        k = cfg.num_experts_per_tok
        # Router logits in fp32 from fp32 inputs and fp32-upcast weights; never
        # rounded, in either mode (GateLinear out_dtype fp32).
        logits = linear(x, self._w(f"{p}.mlp.gate.weight"))  # [N, E]
        bias = self._w(f"{p}.moe.router.expert_bias")  # BF16 in the checkpoint, exact upcast
        if force_ids is None:
            weights, ids = self.route(logits, bias, k)
        else:
            weights, ids = forced_route_ref(logits, force_ids, cfg.norm_topk_prob)
        # The same fp32 sum route_ref selects on; kept for gates (G2 near-ties).
        biased = logits + bias[None, :]

        shared = self.expert(x, f"{p}.mlp.shared_experts")

        # Group (token, slot) pairs by expert; each selected expert is read once
        # and run on the matrix of its tokens. A token selects an expert at most
        # once, so the rows inside one group are distinct and the scatter-add
        # below touches each row once.
        kk = ids.shape[1]
        routed = np.zeros_like(x)
        flat = ids.reshape(-1)
        order = np.argsort(flat, kind="stable")
        experts, starts = np.unique(flat[order], return_index=True)
        ends = np.append(starts[1:], flat.size)
        for e, s, t in zip(experts.tolist(), starts.tolist(), ends.tolist()):
            sel = order[s:t]
            rows, slots = sel // kk, sel % kk
            y = self.expert(x[rows], f"{p}.mlp.experts.{e}")
            routed[rows] += weights[rows, slots][:, None] * y
        # Shared and routed outputs are added unscaled (moe_runner: shared + fused).
        # In emulation the fused output and the sum are stored bf16.
        out = self._store(self._store(routed) + shared) if self.emulate_bf16 else routed + shared
        info = {
            "logits": logits,
            "biased": biased,
            "top6": ids,
            "ids": ids,
            "weights": weights,
            "forced": force_ids is not None,
            "experts_loaded": len(experts),
        }
        return out, info

    def layer_branches(self, i: int, h: np.ndarray, segments: list[tuple[int, int]] | None = None,
                       force_ids=None) -> dict:
        """One decoder layer on the residual stream h [N, H] (spec items 5, 29),
        with every intermediate the gate compares:

            r_attn  post_attn_norm(attn(input_layernorm(h)))   branch output
            h_mid   h + r_attn
            r_moe   post_ffn_norm(moe(post_attention_layernorm(h_mid)))
            h_out   h_mid + r_moe
            info    moe_block's info

        In emulation the stream h is the unrounded fp32 sum (what vLLM's
        fused_add_rms_norm normalises); vLLM stores round_bf16 of it, and the
        next add starts from that stored value. Loads layer i's weights, runs
        all tokens, and drops the weights."""
        p = f"model.layers.{i}"
        names = self.layer_norm_names()

        def norm(x, role):
            return self.norm(x, self._w(f"{p}.{names[role]}.weight"))

        residual = self._store(h)
        a = self.attention_block(i, norm(h, "input"), segments)
        r_attn = norm(a, "post_attn")
        h_mid = residual + r_attn
        m, info = self.moe_block(i, norm(h_mid, "pre_moe"), force_ids)
        r_moe = norm(m, "post_ffn")
        h_out = self._store(h_mid) + r_moe
        return {"r_attn": r_attn, "h_mid": h_mid, "r_moe": r_moe, "h_out": h_out, "info": info}

    def layer_forward(self, i: int, h: np.ndarray, segments: list[tuple[int, int]] | None = None,
                      force_ids=None) -> np.ndarray:
        """One decoder layer on the residual stream h [N, H] (spec item 5):
            h += post_attn_norm(attn(input_layernorm(h)))
            h += post_ffn_norm(moe(post_attention_layernorm(h)))
        NAMING TRAP: post_attention_layernorm is the PRE-MoE norm (Qwen naming);
        post_attn_norm and post_ffn_norm are the sandwich norms on the outputs.
        Returns a new array; the branches are in self.last_layer_info."""
        t0 = time.perf_counter()
        cfg = self.cfg
        out = self.layer_branches(i, h, segments, force_ids)
        info = out["info"]
        dt = time.perf_counter() - t0
        self.last_layer_info = dict(info, r_attn=out["r_attn"], h_mid=out["h_mid"], r_moe=out["r_moe"])
        self.last_routing.append((info["weights"], info["ids"]))
        self.last_stats.append({"layer": i, "type": cfg.layer_types[i], "seconds": dt, "experts_loaded": info["experts_loaded"]})
        self._log(
            f"[ref] layer {i + 1}/{cfg.num_hidden_layers} {cfg.layer_types[i]:<17} "
            f"experts {info['experts_loaded']:>3}/{cfg.num_experts}  {dt:6.2f}s  peak RSS {_peak_rss_bytes() / 2**30:.2f} GiB"
        )
        return out["h_out"]

    def final_norm(self, h: np.ndarray) -> np.ndarray:
        return self.norm(h, self._w("model.norm.weight"))

    def lm_head(self, hn: np.ndarray) -> np.ndarray:
        """fp32 logits = hn @ W^T with the bf16 weight upcast (spec item 14),
        computed in vocab-row chunks so the fp32 copy of W never exists whole.
        Never rounded, in either mode."""
        name = "model.embed_tokens.weight" if self.cfg.tie_word_embeddings else "lm_head.weight"
        V = self.cfg.vocab_size
        logits = np.empty((hn.shape[0], V), dtype=np.float32)
        for a in range(0, V, _LM_HEAD_CHUNK):
            b = min(V, a + _LM_HEAD_CHUNK)
            logits[:, a:b] = linear(hn, self.ckpt.get_rows(name, slice(a, b)))
        return logits

    def _check_ids(self, ids) -> np.ndarray:
        cfg = self.cfg
        ids = np.asarray(ids)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("ids must be a 1-D integer array")
        if ids.size == 0:
            raise ValueError("ids is empty")
        if ids.min() < 0 or ids.max() >= cfg.vocab_size:
            raise ValueError(f"token id out of range [0, {cfg.vocab_size})")
        if ids.size > cfg.max_position_embeddings:
            raise ValueError(f"{ids.size} tokens exceed max_position_embeddings {cfg.max_position_embeddings}")
        return ids.astype(np.int64)

    def _segments(self, seqs):
        seqs = [self._check_ids(s) for s in seqs]
        if not seqs:
            raise ValueError("no sequences")
        lengths = np.array([s.size for s in seqs], dtype=np.int64)
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        segments = [(int(bounds[j]), int(bounds[j + 1])) for j in range(len(seqs))]
        return seqs, lengths, segments

    def _num_layers(self, num_layers) -> int:
        n = self.cfg.num_hidden_layers if num_layers is None else int(num_layers)
        if not 0 <= n <= self.cfg.num_hidden_layers:
            raise ValueError(f"num_layers must be in [0, {self.cfg.num_hidden_layers}]")
        return n

    def forward_packed(self, seqs, return_hidden: bool = False, num_layers: int | None = None, on_layer=None):
        """Forward several independent sequences (each from position 0) in one
        pass over the weights, returning packed arrays without per-sequence
        copies: (logits [N, V], hidden [n_layers, N, H] or None,
        final_norm [N, H], seq_lengths [S]), N = total tokens, sequences
        concatenated in order. Split with np.cumsum(seq_lengths)[:-1].

        on_layer(i, h_in, h_out, info), if given, is called after each layer
        with the residual stream entering and leaving layer i and the layer's
        info (moe_block's keys plus r_attn, h_mid and r_moe); the CLI's
        --dump writes its files from it."""
        cfg = self.cfg
        seqs, lengths, segments = self._segments(seqs)
        n = self._num_layers(num_layers)

        self.last_routing, self.last_stats = [], []
        h = self.embed(np.concatenate(seqs))
        hidden = np.empty((n, h.shape[0], cfg.hidden_size), dtype=np.float32) if return_hidden else None
        for i in range(n):
            h_out = self.layer_forward(i, h, segments)  # a new array; h is unchanged
            if on_layer is not None:
                on_layer(i, h, h_out, self.last_layer_info)
            h = h_out
            if return_hidden:
                hidden[i] = h
        hn = self.final_norm(h)
        logits = self.lm_head(hn)
        return logits, hidden, hn, lengths

    def forward_batch(self, seqs, return_hidden: bool = False, num_layers: int | None = None) -> list:
        """Forward several independent sequences in one pass over the weights.

        seqs: list of 1-D integer token-id arrays, each starting at position 0.
        Returns one entry per sequence, as forward() would return it (views
        into the packed arrays of forward_packed)."""
        logits, hidden, hn, lengths = self.forward_packed(seqs, return_hidden, num_layers)
        bounds = np.concatenate([[0], np.cumsum(lengths)])
        out = []
        for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
            if return_hidden:
                out.append((logits[a:b], [layer[a:b] for layer in hidden], hn[a:b]))
            else:
                out.append(logits[a:b])
        return out

    def forward(self, ids, return_hidden: bool = False, num_layers: int | None = None):
        """Full-sequence forward from position 0, no KV cache.

        ids: 1-D integer token ids [T].
        num_layers: run only the first n decoder layers, then model.norm and
            the LM head (for partial gates; the logits are then not the model's).
        Returns logits [T, V] float32, or (logits, layers, final_norm) when
        return_hidden: layers[i] is the residual stream after layer i [T, H].
        """
        return self.forward_batch([ids], return_hidden, num_layers)[0]

    # ---- several model variants in one pass (G3) ------------------------

    def _clone_as(self, cls) -> "KolibriReference":
        """An instance of `cls` (a KolibriReference subclass) sharing this
        instance's checkpoint, config, mode and weight cache, with its own
        routing and statistics lists. No file is opened again."""
        if not (isinstance(cls, type) and issubclass(cls, KolibriReference)):
            raise TypeError(f"{cls!r} is not a KolibriReference subclass")
        clone = cls.__new__(cls)
        clone.__dict__.update(self.__dict__)
        clone.last_routing, clone.last_stats, clone.last_layer_info = [], [], None
        return clone

    def forward_streams(self, ids, variants, include_base: bool = True, num_layers: int | None = None,
                        on_logits=None, on_layer=None) -> dict:
        """Carry one hidden stream per model variant through a single
        layer-streamed pass (BUILD_SPEC 5.1b item 6): each layer's weights are
        read once, cached for the layer, and used by every stream.

        ids: one sequence [T] or a list of sequences (packed as in
            forward_packed).
        variants: {name: KolibriReference subclass}, e.g. mutants.MUTANTS.
            Each stream runs its class's own methods on the shared weights.
        include_base: also carry this instance's own stream, under "ref".
        on_logits(name, logits [N, V]) -> anything: if given, its result is
            stored per stream instead of the logits (e.g. per-token NLL, so
            k logit matrices are never held at once).
        on_layer(i, h_in, h_out, info): as in forward_packed, for the base
            stream only (the CLI dumps it while the mutants ride along).

        Returns {name: logits or on_logits(...)}, plus "_lengths" [S]. Each
        stream's result is bit-identical to that variant's own forward_packed.
        Afterwards self.last_final_norm holds the base stream's model.norm
        output (or None without a base stream)."""
        seqs, lengths, segments = self._segments(_as_sequences(ids))
        n = self._num_layers(num_layers)
        streams = {}
        if include_base:
            streams["ref"] = self
        for name, cls in dict(variants).items():
            if name in streams:
                raise ValueError(f"duplicate stream name {name!r}")
            streams[name] = self._clone_as(cls)
        for inst in streams.values():
            inst.last_routing, inst.last_stats = [], []

        h0 = self.embed(np.concatenate(seqs))
        hs = {name: h0 for name in streams}
        self.last_final_norm = None
        try:
            for i in range(n):
                self._shared["cache"] = {}
                for name, inst in streams.items():
                    h_out = inst.layer_forward(i, hs[name], segments)
                    if on_layer is not None and inst is self:
                        on_layer(i, hs[name], h_out, self.last_layer_info)
                    hs[name] = h_out
                self._shared["cache"] = None
            # The final norm weight is shared like a layer's; the head is
            # read in row chunks once per stream (from the page cache after
            # the first).
            self._shared["cache"] = {}
            results = {}
            for name, inst in streams.items():
                hn = inst.final_norm(hs[name])
                if inst is self:
                    self.last_final_norm = hn
                logits = inst.lm_head(hn)
                results[name] = logits if on_logits is None else on_logits(name, logits)
                hs[name] = None
        finally:
            self._shared["cache"] = None
        results["_lengths"] = lengths
        return results


def _as_sequences(ids) -> list:
    """One sequence (1-D int array or list of ints) -> [it]; a list of
    sequences is returned as a list."""
    if isinstance(ids, np.ndarray):
        return [ids] if ids.ndim == 1 else list(ids)
    ids = list(ids)
    if ids and all(isinstance(t, (int, np.integer)) for t in ids):
        return [np.asarray(ids)]
    return ids


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _peak_rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024  # Linux reports KiB


def _checkpoint_fingerprint(model_dir: str) -> str:
    """sha256 of the index file (sharded) or of the single file's header:
    ties an output to the exact checkpoint layout without hashing 156 GB."""
    index = os.path.join(model_dir, INDEX_FILE)
    if os.path.isfile(index):
        with open(index, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    path = os.path.join(model_dir, SINGLE_FILE)
    _, _, data_start = read_header(path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(data_start)).hexdigest()


def _read_ids(path: str) -> list[np.ndarray]:
    """One sequence ([1, 2, 3]) or several ([[1, 2], [3, 4, 5]]), optionally
    wrapped as {"ids": ...}. Returns a list of 1-D int64 arrays."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("ids", "input_ids", "token_ids"):
            if key in obj:
                obj = obj[key]
                break
        else:
            raise ValueError(f"{path}: expected a list of ints or a dict with 'ids'")
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"{path}: expected a non-empty list")
    seqs = obj if isinstance(obj[0], list) else [obj]
    out = []
    for s in seqs:
        a = np.asarray(s)
        if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError(f"{path}: every sequence must be a non-empty list of ints")
        out.append(a.astype(np.int64))
    return out


def log_softmax_f32(logits: np.ndarray, rows: int = 128) -> np.ndarray:
    """Row-wise log-softmax of fp32 logits [N, V], computed in float64 in
    blocks of `rows` and stored as float32."""
    out = np.empty(logits.shape, dtype=np.float32)
    for a, block in _log_softmax_blocks(logits, rows):
        out[a : a + block.shape[0]] = block
    return out


def _log_softmax_blocks(logits: np.ndarray, rows: int = 128):
    """(start row, float32 log-softmax block) pairs, computed in float64."""
    for a in range(0, logits.shape[0], rows):
        x = logits[a : a + rows].astype(np.float64)
        x -= x.max(axis=-1, keepdims=True)
        x -= np.log(np.exp(x).sum(axis=-1, keepdims=True))
        yield a, x.astype(np.float32)


def next_token_nll(logits: np.ndarray, ids: np.ndarray, lengths) -> np.ndarray:
    """Per-position NLL of the next token, nats, float32 [N]: position t of a
    sequence scores ids[t + 1]; each sequence's last position is NaN."""
    ids = np.asarray(ids, dtype=np.int64)
    nll = np.full(ids.shape[0], np.nan, dtype=np.float32)
    bounds = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64))])
    target = np.full(ids.shape[0], -1, dtype=np.int64)
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        target[a : b - 1] = ids[a + 1 : b]
    for a, block in _log_softmax_blocks(logits):
        t = target[a : a + block.shape[0]]
        ok = t >= 0
        rows = np.flatnonzero(ok)
        nll[a + rows] = -block[rows, t[ok]]
    return nll


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


class Dump:
    """The --dump DIR files, written while the forward runs (one layer in
    memory at a time). Per decoder layer i (two-digit index):

        layer{i:02d}.h_in.npy           [N, H] fp32   residual stream entering layer i
        layer{i:02d}.r_attn.npy         [N, H] fp32   attention branch after post_attn_norm
        layer{i:02d}.h_mid.npy          [N, H] fp32   h_in + r_attn
        layer{i:02d}.r_moe.npy          [N, H] fp32   MoE branch after post_ffn_norm
        layer{i:02d}.h_out.npy          [N, H] fp32   residual stream leaving layer i
        layer{i:02d}.router_logits.npy  [N, E] fp32   raw router logits
        layer{i:02d}.biased.npy         [N, E] fp32   router logits + expert_bias (selection scores)
        layer{i:02d}.top6.npy           [N, k] int64  selected ids, by descending biased score
        layer{i:02d}.top6_gap.npy       [N]    fp32   k-th minus (k+1)-th biased score:
                                                      the G2 near-tie margin

    and once: logits.npy [N, V] fp32, logprobs.npy [N, V] fp32,
    final_norm.npy [N, H] fp32, ids.npy [N] int64, seq_lengths.npy [S] int64.
    With --mutants, stream_<name>.nll.npy [N] fp32 per stream (next-token NLL,
    NaN at each sequence's last position). ref_record.json is written last,
    so a directory without it is an incomplete dump. layer{i}.h_in equals
    layer{i-1}.h_out; both are kept so each layer's files stand alone.
    """

    def __init__(self, directory: str):
        self.dir = directory
        if os.path.exists(directory) and (not os.path.isdir(directory) or os.listdir(directory)):
            raise SystemExit(f"--dump {directory}: exists and is not an empty directory")
        os.makedirs(directory, exist_ok=True)
        self.files: dict[str, dict] = {}

    def _entry(self, name: str, shape, dtype) -> None:
        path = os.path.join(self.dir, name)
        self.files[name] = {
            "sha256": _sha256_file(path),
            "bytes": os.path.getsize(path),
            "shape": [int(s) for s in shape],
            "dtype": str(np.dtype(dtype)),
        }

    def save(self, name: str, arr: np.ndarray) -> None:
        np.save(os.path.join(self.dir, name), arr)
        self._entry(name, arr.shape, arr.dtype)

    def save_blocks(self, name: str, shape, dtype, blocks) -> None:
        """Write a C-order .npy from consecutive row blocks, so the whole array
        never exists in memory (the [N, V] log-probs of T9: 8.4 GB)."""
        path = os.path.join(self.dir, name)
        header = {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)), "fortran_order": False,
                  "shape": tuple(int(s) for s in shape)}
        rows = 0
        with open(path, "wb") as f:
            np.lib.format.write_array_header_1_0(f, header)
            for block in blocks:
                block = np.ascontiguousarray(block, dtype=dtype)
                f.write(block.tobytes())
                rows += block.shape[0]
        if rows != shape[0]:
            raise RuntimeError(f"{name}: wrote {rows} rows, expected {shape[0]}")
        self._entry(name, shape, dtype)

    def layer(self, i: int, h_in: np.ndarray, h_out: np.ndarray, info: dict) -> None:
        biased = info["biased"]
        k = info["top6"].shape[-1]
        if biased.shape[-1] > k:
            s = -np.sort(-biased, axis=-1)  # descending, fp32
            # fp32 difference of two close fp32 values is exact (Sterbenz).
            gap = (s[:, k - 1] - s[:, k]).astype(np.float32)
        else:
            gap = np.full(biased.shape[0], np.inf, dtype=np.float32)
        p = f"layer{i:02d}"
        self.save(f"{p}.h_in.npy", h_in)
        self.save(f"{p}.r_attn.npy", info["r_attn"])
        self.save(f"{p}.h_mid.npy", info["h_mid"])
        self.save(f"{p}.r_moe.npy", info["r_moe"])
        self.save(f"{p}.h_out.npy", h_out)
        self.save(f"{p}.router_logits.npy", info["logits"])
        self.save(f"{p}.biased.npy", biased)
        self.save(f"{p}.top6.npy", info["top6"])
        self.save(f"{p}.top6_gap.npy", gap)

    def finish(self, logits, final, ids, lengths, record: dict) -> str:
        self.save("logits.npy", logits)
        self.save_blocks("logprobs.npy", logits.shape, np.float32,
                         (block for _, block in _log_softmax_blocks(logits)))
        self.save("final_norm.npy", final)
        self.save("ids.npy", ids)
        self.save("seq_lengths.npy", lengths)
        record = dict(record, files=self.files, peak_rss_bytes=_peak_rss_bytes())
        path = os.path.join(self.dir, "ref_record.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, sort_keys=True)
            f.write("\n")
        return path


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m reference.kolibri_ref",
        description="Kolibri-1 numpy reference: teacher-forced logits for one or more token sequences.",
    )
    ap.add_argument("--model-dir", help="checkpoint dir (default: $EXP036_MODELS/Kolibri-1-BF16)")
    ap.add_argument(
        "--ids-file", required=True,
        help="JSON: one sequence [ids...], several [[ids...], ...] (one weight pass), or {\"ids\": either}",
    )
    ap.add_argument("--out", help="output .npz (.npz is appended when missing)")
    ap.add_argument(
        "--dump", metavar="DIR",
        help="also write per-layer h_in/r_attn/h_mid/r_moe/h_out/router_logits/biased/top6 .npy, "
        "logprobs.npy and ref_record.json into DIR (must not exist or be empty)",
    )
    ap.add_argument("--hidden", action="store_true", help="also save per-layer residuals, final norm and routing")
    ap.add_argument("--num-layers", type=int, default=None, help="run only the first N layers (partial gate)")
    ap.add_argument("--attn-chunk", type=int, default=None, help="query rows per attention chunk")
    ap.add_argument("--emulate-bf16", action="store_true",
                    help="bf16-emulation mode: round wherever vLLM stores bf16 (the gate's 'emu')")
    ap.add_argument("--dequant-dir", metavar="DIR",
                    help="dequantised mode: weights from this converted MLX directory (port/convert.py output)")
    ap.add_argument("--mutants", metavar="NAMES",
                    help="with --dump: also carry the G3 reference mutants (comma-separated names from "
                    "reference/mutants.py, or 'all') through the same pass and write stream_<name>.nll.npy")
    ap.add_argument("--gate-text-sha", metavar="HEX",
                    help="the gate-text sha256 to record (default: the sha256 of --ids-file)")
    ap.add_argument("--quiet", action="store_true", help="no per-layer progress on stderr")
    args = ap.parse_args(argv)
    if not args.out and not args.dump:
        ap.error("give --out, --dump or both")
    if args.mutants and not args.dump:
        ap.error("--mutants needs --dump")
    if args.mutants and args.hidden:
        ap.error("--mutants and --hidden cannot be combined (the dump holds the per-layer states)")
    # np.savez appends .npz to a name without it; say so up front and print the real path.
    out_path = None
    if args.out:
        out_path = args.out if args.out.endswith(".npz") else args.out + ".npz"

    model_dir = args.model_dir
    if model_dir is None:
        root = os.environ.get("EXP036_MODELS")
        if not root:
            ap.error("--model-dir not given and EXP036_MODELS not set")
        model_dir = os.path.join(root, "Kolibri-1-BF16")

    mutants = {}
    if args.mutants:
        try:
            from .mutants import MUTANTS
        except ImportError:
            from mutants import MUTANTS
        names = sorted(MUTANTS) if args.mutants == "all" else [m.strip() for m in args.mutants.split(",") if m.strip()]
        unknown = sorted(set(names) - set(MUTANTS))
        if unknown:
            ap.error(f"unknown mutants {unknown}; known: {sorted(MUTANTS)}")
        mutants = {name: MUTANTS[name] for name in names}

    t_start = _utc()
    t0 = time.perf_counter()
    seqs = _read_ids(args.ids_file)
    ids_file_sha = _sha256_file(args.ids_file)
    ref = KolibriReference(model_dir, attn_chunk=args.attn_chunk, emulate_bf16=args.emulate_bf16,
                           dequant_dir=args.dequant_dir, verbose=not args.quiet)
    with open(os.path.join(model_dir, "config.json"), "rb") as f:
        config_sha = hashlib.sha256(f.read()).hexdigest()
    dump = Dump(args.dump) if args.dump else None  # before the forward: a bad DIR fails fast

    # Sequences are stored concatenated along the token axis; seq_lengths
    # splits them (np.split(x, np.cumsum(seq_lengths)[:-1])).
    ids = np.concatenate(seqs)
    stream_names = []
    if not mutants:
        logits, hidden, final, lengths = ref.forward_packed(
            seqs, return_hidden=args.hidden, num_layers=args.num_layers,
            on_layer=dump.layer if dump is not None else None,
        )
    else:
        # One pass over the weights: the reference's own stream (dumped as
        # usual) plus one stream per mutant, each reduced to next-token NLL.
        lengths = np.array([s.size for s in seqs], dtype=np.int64)
        kept = {}

        def nll_of(name, stream_logits):
            if name == "ref":
                kept["logits"] = stream_logits
            return next_token_nll(stream_logits, ids, lengths)

        streams = ref.forward_streams(seqs, mutants, include_base=True, num_layers=args.num_layers,
                                      on_logits=nll_of, on_layer=dump.layer)
        logits, hidden, final = kept["logits"], None, ref.last_final_norm
        for name in ["ref"] + list(mutants):
            dump.save(f"stream_{name}.nll.npy", streams[name])
            stream_names.append(name)
    out = {}
    if args.hidden:
        k = ref.cfg.num_experts_per_tok
        N = logits.shape[0]
        out["hidden"] = hidden  # [layers, N, H]
        out["final_norm"] = final  # [N, H]
        out["router_weights"] = (
            np.stack([w for w, _ in ref.last_routing]) if ref.last_routing else np.zeros((0, N, k), np.float32)
        )  # [layers, N, k]
        out["router_ids"] = (
            np.stack([i for _, i in ref.last_routing]) if ref.last_routing else np.zeros((0, N, k), np.int64)
        )  # [layers, N, k]
    wall = time.perf_counter() - t0
    peak = _peak_rss_bytes()

    n_layers = ref.cfg.num_hidden_layers if args.num_layers is None else args.num_layers
    run = dict(
        ref_version=REF_VERSION,
        model_dir_name=os.path.basename(os.path.normpath(model_dir)),
        config_sha256=config_sha,
        checkpoint_fingerprint=_checkpoint_fingerprint(model_dir),
        num_layers=n_layers,
        layer_seconds=[s["seconds"] for s in ref.last_stats],
        experts_loaded=[s["experts_loaded"] for s in ref.last_stats],
        wall_seconds=wall,
        mode=ref.mode,
        dequantised_from=(os.path.basename(os.path.normpath(args.dequant_dir)) if args.dequant_dir else ""),
    )
    written = []
    if out_path is not None:
        out.update({key: np.array(value) for key, value in run.items()})
        out["layer_seconds"] = out["layer_seconds"].astype(np.float64)
        out["experts_loaded"] = out["experts_loaded"].astype(np.int64)
        out.update(
            logits=logits,  # [N, V] float32
            ids=ids,  # [N]
            seq_lengths=lengths,
            peak_rss_bytes=np.array(peak),  # before saving; the line printed below is after
        )
        np.savez(out_path, **out)
        written.append(out_path)
    if dump is not None:
        dequant_manifest = None
        if args.dequant_dir:
            rec = os.path.join(args.dequant_dir, "exp036_convert_record.json")
            if os.path.isfile(rec):
                with open(rec) as f:
                    dequant_manifest = json.load(f).get("manifest_sha256")
        record = dict(
            run,
            num_tokens=int(ids.size),
            seq_lengths=[int(n) for n in lengths],
            ids_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
            ids_file_sha256=ids_file_sha,
            gate_text_sha256=args.gate_text_sha or ids_file_sha,
            reference_tree_sha256=reference_tree_sha256(),
            dequantised_manifest_sha256=dequant_manifest,
            streams=stream_names,
            t_start=t_start,
            t_end=_utc(),
            utc=_utc(),
        )
        written.append(dump.finish(logits, final, ids, lengths, record))
    print(
        f"{REF_VERSION} ({ref.mode}{', dequantised' if args.dequant_dir else ''}): {len(seqs)} sequence(s), "
        f"{logits.shape[0]} tokens, {n_layers} layers -> "
        f"{', '.join(written)}  wall {time.perf_counter() - t0:.1f} s  peak RSS {_peak_rss_bytes() / 2**30:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    # Run main() from the importable module, not from __main__, so the classes
    # of reference/mutants.py (which import this module) are the same classes.
    try:
        from reference.kolibri_ref import main as _main  # python -m reference.kolibri_ref
    except ImportError:
        from kolibri_ref import main as _main  # python kolibri_ref.py inside reference/
    sys.exit(_main())
