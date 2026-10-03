# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""Kolibri-1 (model_type "kolibri1") for mlx_lm 0.31.3.

Loaded through config.json `"model_file": "kolibri1.py"`: mlx_lm's
utils.load_model executes this file and uses its `Model` / `ModelArgs`.
Nothing in site-packages is edited.

Forward pass (spec items refer to the exp_036 port specification, items 1-25):
  h = embed_tokens[ids]                                   (no scaling, item 4)
  per layer:
    h = h + post_attn_norm(attn(input_layernorm(h)))      (item 5)
    h = h + post_ffn_norm(moe(post_attention_layernorm(h)))
  logits = lm_head(fp32(model.norm(h)))                   (item 14; fp32 out)

Sliding layers: RoPE (NeoX half-rotation, theta 1e4), window of
`sliding_window` keys including the query itself (items 8-10).
Full layers: no positional encoding, plain causal mask (item 9).
Router: fp32 logits from an fp32 weight (exact upcast of the bf16
checkpoint); select top-k on logits + expert_bias; weight by the unbiased
sigmoid(logits); no renormalisation, scale 1.0 (items 11-12).

Head and embedding policy (item 14, item 18, BUILD_SPEC 5.1 deltas 1-3):
by default (the pre-registered "quantised_head" policy) embed_tokens and
lm_head stay bf16 through sanitize and the --dtype cast, so convert.py
quantises them from their bf16 values (bf16 scales and biases, as for the
peers); the head always receives fp32 activations, so its logits are fp32.
convert.py --no-quantize-lm-head writes "exp036_quantize_lm_head": false into
the config; sanitize then stores lm_head as an exact fp32 upcast (the
"vendor_faithful" head).

Inspection hooks for the gate (BUILD_SPEC 5.1 delta 5): Model.forward_hidden,
Model.compute_logits, Kolibri1Model.make_masks and DecoderLayer.branches.
DecoderLayer.__call__ is branches(...)[3] and Model.__call__ is
compute_logits(forward_hidden(...)), so a patch of either hook reaches the
normal forward too.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

SPEC_VERSION = "exp036-port-2"

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    sliding_window: int
    layer_types: List[str]
    rms_norm_eps: float = 1e-6
    # The released config.json carries a top-level rope_theta; configs re-saved
    # by transformers 5 carry rope_parameters instead. Either is accepted.
    rope_theta: Optional[float] = None
    rope_parameters: Optional[dict] = None
    rope_scaling: Optional[dict] = None
    norm_topk_prob: bool = False
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144
    head_dtype: Optional[str] = "float32"
    hidden_act: str = "silu"
    attention_bias: bool = False
    # Default False, as in transformers' Qwen3MoeConfig, which the vendor's
    # Kolibri1Config subclasses: a config without the key gets
    # sliding_window None there, and the vendor attention refuses sliding
    # layers. The released config.json sets it to true.
    use_sliding_window: bool = False
    # Conversion policy, written into the staging config by convert.py
    # (BUILD_SPEC 5.1 deltas 2-3). Absent from the released config.json, which
    # means the pre-registered default: both quantised at the arm's bits.
    exp036_quantize_embeddings: bool = True
    exp036_quantize_lm_head: bool = True

    def __post_init__(self):
        if self.rope_theta is None:
            params = self.rope_parameters or {}
            self.rope_theta = float(params.get("rope_theta", 10000.0))
        for name, params in (
            ("rope_parameters", self.rope_parameters),
            ("rope_scaling", self.rope_scaling),
        ):
            rope_type = (params or {}).get("rope_type", (params or {}).get("type"))
            if rope_type not in (None, "default"):
                # Item 8: no rope scaling at any length.
                raise ValueError(f"kolibri1: unsupported {name} type {rope_type!r}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"kolibri1: layer_types has {len(self.layer_types)} entries, "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
        unknown = set(self.layer_types) - {SLIDING, FULL}
        if unknown:
            raise ValueError(f"kolibri1: unknown layer_types {sorted(unknown)}")
        if SLIDING in self.layer_types and (
            not self.use_sliding_window
            or self.sliding_window is None
            or self.sliding_window <= 0
        ):
            # Same refusal as the vendor plugin's Kolibri1Attention.
            raise ValueError(
                "kolibri1: sliding layers need use_sliding_window: true and a "
                "positive sliding_window"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("kolibri1: num_attention_heads % num_key_value_heads != 0")
        if self.hidden_act != "silu":
            raise ValueError(f"kolibri1: hidden_act {self.hidden_act!r}, expected silu")
        if self.attention_bias:
            raise ValueError("kolibri1: attention_bias is not part of the architecture")
        if self.tie_word_embeddings:
            raise ValueError("kolibri1: the released model has an untied lm_head")
        if self.head_dtype not in (None, "float32"):
            raise ValueError(
                f"kolibri1: head_dtype {self.head_dtype!r}; this port computes "
                "logits in float32 only (item 14)"
            )


def route(logits_f32: mx.array, expert_bias_f32: mx.array, top_k: int, renormalize: bool = False):
    """Kolibri-1 routing (items 11-12), the vendor's sigmoid_logit_add_routing.

    ids     = top-k of (logits + expert_bias)        selection uses the RAW logits
    weights = sigmoid(logits[ids])                   the bias never enters the weights

    afmoe / deepseek_v3 select on sigmoid(logits) + bias, which picks different
    experts once the bias is large; that is wrong for this model.
    `renormalize` mirrors config.norm_topk_prob (False for Kolibri-1).
    Returns (weights fp32 [..., k], ids uint32 [..., k]); the order inside the
    k selected experts is unspecified (it does not change the weighted sum).
    """
    logits = logits_f32.astype(mx.float32)
    choice = logits + expert_bias_f32.astype(mx.float32)
    ids = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.sigmoid(mx.take_along_axis(logits, ids, axis=-1))
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return weights, ids


def forced_route(logits_f32: mx.array, ids: mx.array, renormalize: bool = False):
    """Weights for an externally fixed expert selection (the gate's "forced"
    comparisons, BUILD_SPEC 5.1 delta 5): sigmoid of the port's own fp32
    logits at the given ids, exactly as route() weights its own selection
    (item 12). `ids` [..., k] broadcasts against logits[..., E]."""
    logits = logits_f32.astype(mx.float32)
    ids = mx.broadcast_to(ids.astype(mx.uint32), logits.shape[:-1] + (ids.shape[-1],))
    weights = mx.sigmoid(mx.take_along_axis(logits, ids, axis=-1))
    if renormalize:
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return weights, ids


class Router(nn.Module):
    """`mlp.gate`: the router weight [E, H] and the selection bias [E].

    weight: fp32, an exact upcast of the BF16 checkpoint tensor
    mlp.gate.weight (item 11; sanitize upcasts, cast_predicate keeps it).
    expert_bias: fp32; checkpoint name moe.router.expert_bias, stored BF16
    and upcast exactly in sanitize (item 12).

    Deliberately not an nn.Linear: it has no `to_quantized`, so neither the
    model's quant_predicate nor an mlx_lm mixed recipe can quantise it.
    """

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.weight = mx.zeros((num_experts, hidden_size), dtype=mx.float32)
        self.expert_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Item 11: fp32 logits from bf16 operands. The weight holds bf16 values
        # in fp32, and bf16*bf16 products are exact in fp32, so this equals
        # vLLM's GateLinear (bf16 GEMM, fp32 accumulation, fp32 output) up to
        # summation order. Only x is cast; there is no per-call weight cast
        # (it cost about 0.5 GB of memory traffic per token over 50 layers).
        return x.astype(mx.float32) @ self.weight.T


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x)). Used for the shared expert, which is
    ungated and unscaled (item 13)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    """Every layer is MoE (item 13): routed experts + one shared expert."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = Router(args.hidden_size, args.num_experts)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_experts = MLP(args.hidden_size, args.shared_expert_intermediate_size)

    def forward_routed(self, x: mx.array, force_ids: Optional[mx.array] = None):
        """(output, router logits fp32 [..., E], selected ids [..., k]).

        force_ids [..., k] (optional) replaces the selection; the weights are
        still sigmoid of this block's own fp32 logits (forced_route)."""
        logits = self.gate(x)
        if force_ids is None:
            weights, ids = route(logits, self.gate.expert_bias, self.top_k, self.norm_topk_prob)
        else:
            weights, ids = forced_route(logits, force_ids, self.norm_topk_prob)
        y = self.switch_mlp(x, ids)  # [..., k, H], one row per selected expert
        # fp32 routing weights, fp32 sum, then back to the residual dtype.
        y = (y * weights[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x), logits, ids

    def __call__(self, x: mx.array) -> mx.array:
        return self.forward_routed(x)[0]


class Attention(nn.Module):
    """GQA with per-head q/k RMSNorm in every layer, applied before RoPE
    (item 7). RoPE only when `use_rope` (sliding layers); full layers are NoPE.
    No biases, no output gate, no softcap, no sinks."""

    def __init__(self, args: ModelArgs, use_rope: bool):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim**-0.5
        dim = args.hidden_size

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Item 8: traditional=False is the NeoX / half-rotation layout.
        self.rope = (
            nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)
            if use_rope
            else None
        )

    def __call__(self, x: mx.array, mask: Any = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # Norm over head_dim on the [B, L, heads, D] view, then to [B, heads, L, D].
        q = self.q_norm(q).transpose(0, 2, 1, 3)
        k = self.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.rope is not None:
            # Absolute positions: cache.offset counts every token seen, also
            # past the window (RotatingKVCache.offset is unbounded).
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    """Sandwich norms (item 5). Attribute names mirror the checkpoint:
    input_layernorm      pre-attention norm
    post_attn_norm       norm on the attention output
    post_attention_layernorm  pre-MoE norm (Qwen naming; NOT the post-attention norm)
    post_ffn_norm        norm on the MoE output
    """

    def __init__(self, args: ModelArgs, layer_type: str):
        super().__init__()
        self.use_sliding = layer_type == SLIDING
        self.self_attn = Attention(args, use_rope=self.use_sliding)
        self.mlp = SparseMoeBlock(args)
        eps = args.rms_norm_eps
        # nn.RMSNorm is w * x / sqrt(mean(x^2) + eps) in fp32, rounded to bf16
        # before the weight multiply, as vLLM's rms_norm does (item 6). Not (1 + w).
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attn_norm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.post_ffn_norm = nn.RMSNorm(args.hidden_size, eps=eps)

    def branches(self, h: mx.array, mask: Any = None, cache: Optional[Any] = None,
                 force_ids: Optional[mx.array] = None):
        """The layer with its intermediate values (item 29; the G2 harness):
        (r_attn, h_mid, r_moe, h_out, router_logits fp32, ids).

        r_attn and r_moe are the branch outputs before the residual adds
        (after the sandwich norms); h_mid = h + r_attn; h_out = h_mid + r_moe.
        force_ids [..., k] fixes the MoE selection (weights still from this
        layer's own fp32 logits)."""
        r_attn = self.post_attn_norm(self.self_attn(self.input_layernorm(h), mask, cache))
        h_mid = h + r_attn
        m, logits, ids = self.mlp.forward_routed(self.post_attention_layernorm(h_mid), force_ids)
        r_moe = self.post_ffn_norm(m)
        return r_attn, h_mid, r_moe, h_mid + r_moe, logits, ids

    def __call__(self, x: mx.array, mask: Any = None, cache: Optional[Any] = None) -> mx.array:
        return self.branches(x, mask, cache)[3]


class Kolibri1Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, t) for t in args.layer_types]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # First layer of each kind; its cache decides the mask for all layers
        # of that kind (all caches of one kind hold the same offset).
        self.swa_idx = args.layer_types.index(SLIDING) if SLIDING in args.layer_types else None
        self.fa_idx = args.layer_types.index(FULL) if FULL in args.layer_types else None

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Final-norm hidden states [B, T, H].

        Memory: without a cache, a call with T > sliding_window gets a dense
        [T, T] boolean window mask from mlx_lm's create_attention_mask, plus
        intermediates of the same size, so memory grows with T^2 (about 0.4 GB
        at 16k tokens, 17 GB at 128k for the mask alone). With a cache the
        mask is [chunk, chunk + window - 1]. Score long sequences through
        model.make_cache() in chunks (as generate, server and evaluate do)."""
        # Item 4: plain lookup, no sqrt(hidden) scaling.
        h = input_embeddings if input_embeddings is not None else self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask, swa_mask = self.make_masks(h, cache)
        for layer, c in zip(self.layers, cache):
            h = layer(h, swa_mask if layer.use_sliding else fa_mask, c)

        return self.norm(h)

    def make_masks(self, h: mx.array, cache: Optional[List[Any]] = None):
        """(full-attention mask, sliding-window mask) for hidden states h
        [B, T, H] and the per-layer caches (or None): what __call__ passes to
        each layer, exposed for the gate's per-layer harness."""
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        swa_mask = None
        if self.swa_idx is not None:
            # Item 10: mlx's window_size W keeps keys j with i - W < j <= i, i.e.
            # W keys including the query. Kolibri's sliding_window=513 means
            # exactly that (vLLM FlashAttention window (512, 0)), so pass it verbatim.
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )
        return fa_mask, swa_mask


class SlidingKVCache(RotatingKVCache):
    """RotatingKVCache for the sliding layers.

    mlx_lm's generate calls to_quantized() on every cache when --kv-bits is
    set, and RotatingKVCache raises NotImplementedError there (item 19).
    Returning self keeps the 513-token sliding caches in bf16 and lets only the
    full-attention KVCaches be quantised.

    Prefill memory with --kv-bits: a quantised full-attention cache sends
    every prefill chunk through mlx_lm's unfused
    quantized_scaled_dot_product_attention, which materialises the
    [heads, chunk, context] score matrix. The transient is about
    100 B x prefill_step_size x context (measured 96-103 B with 48 heads):
    about 52 GB at 256k with the default step 2048. Without --kv-bits the
    fused kernel keeps it flat. Long prefills with --kv-bits need a small
    --prefill-step-size (256 gives about 6.5 GB at 256k and 26 GB at 1M).
    exp_036 does not use --kv-bits.
    """

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        return self


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Kolibri1Model(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        return self.compute_logits(self.forward_hidden(inputs, cache, input_embeddings))

    def forward_hidden(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """model.norm output [B, T, H] (the head's input), in the model dtype."""
        return self.model(inputs, cache, input_embeddings)

    def compute_logits(self, h: mx.array) -> mx.array:
        """fp32 logits from final-norm hidden states h (item 14).

        The head always receives fp32 activations; logits are never computed
        from bf16 activations (BUILD_SPEC 5.1 delta 1):
          * quantised head (default policy): quantized_matmul with an fp32
            input and bf16 scales/biases returns fp32, from the dequantised
            weights scale * q + bias in fp32;
          * vendor-faithful head: the weight is an exact fp32 upcast (sanitize);
          * a bf16 head (the BF16 source loaded directly, unquantised): the
            weight is upcast per call. That path is for tests and the
            gate's layer-wise checks, never for a measured run.
        """
        x = h.astype(mx.float32)
        if hasattr(self.lm_head, "scales"):
            return self.lm_head(x)
        return x @ self.lm_head.weight.astype(mx.float32).T

    def sanitize(self, weights: dict) -> dict:
        """Map checkpoint names to this module (item 21). Idempotent: a second
        call, or a call on an already converted / quantised MLX checkpoint,
        changes nothing."""
        if any(k.endswith("weight_scale_inv") for k in weights):
            raise ValueError(
                "kolibri1: this is the FP8 checkpoint (weight_scale_inv tensors); "
                "this port loads the BF16 repo (Aleph-Alpha/Kolibri-1-BF16) only"
            )

        n_experts = self.args.num_experts
        for layer in range(self.args.num_hidden_layers):
            p = f"model.layers.{layer}"

            # Routing bias: torchtitan name -> the router module; exact upcast
            # BF16 -> fp32 (vLLM holds e_score_correction_bias in fp32).
            src = f"{p}.moe.router.expert_bias"
            dst = f"{p}.mlp.gate.expert_bias"
            if src in weights:
                weights[dst] = weights.pop(src).astype(mx.float32)
            elif dst in weights:
                weights[dst] = weights[dst].astype(mx.float32)

            # Router weight: exact fp32 upcast of the bf16 checkpoint (item 11).
            key = f"{p}.mlp.gate.weight"
            if key in weights:
                weights[key] = weights[key].astype(mx.float32)

            # Stack per-expert tensors into SwitchGLU tensors [E, out, in],
            # explicitly in numeric order 0..E-1 (never lexicographic).
            for proj in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    if f"{p}.mlp.experts.0.{proj}.{part}" not in weights:
                        continue
                    stack = []
                    for e in range(n_experts):
                        key = f"{p}.mlp.experts.{e}.{proj}.{part}"
                        if key not in weights:
                            raise ValueError(f"kolibri1: missing {key}")
                        stack.append(weights.pop(key))
                    weights[f"{p}.mlp.switch_mlp.{proj}.{part}"] = mx.stack(stack)

        leftover = [k for k in weights if ".mlp.experts." in k]
        if leftover:
            raise ValueError(
                f"kolibri1: {len(leftover)} expert tensors outside 0..{n_experts - 1} "
                f"or of an unexpected kind, e.g. {leftover[0]}"
            )

        # lm_head and embed_tokens stay as stored (bf16 in the source), so a
        # quantised head or embedding is quantised from its bf16 values
        # (BUILD_SPEC 5.1 delta 3). Only the vendor-faithful head
        # (exp036_quantize_lm_head false) is stored as an exact fp32 upcast.
        if (
            not self.args.exp036_quantize_lm_head
            and "lm_head.weight" in weights
            and "lm_head.scales" not in weights
        ):
            weights["lm_head.weight"] = weights["lm_head.weight"].astype(mx.float32)

        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self) -> List[Any]:
        # Item 24: keep=0, because vLLM's sliding window has no attention sinks.
        return [
            SlidingKVCache(max_size=self.args.sliding_window, keep=0)
            if layer.use_sliding
            else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        """Used by mlx_lm.convert when it casts parameters to --dtype. Kept
        out of the cast: the routing bias and the router weight (fp32, items
        11-12) and, for the vendor-faithful head only, lm_head (fp32, item 14).
        embed_tokens and a to-be-quantised lm_head are cast to bf16, a no-op on
        the bf16 source, so they are quantised from bf16 (BUILD_SPEC 5.1 delta 3)."""
        keep_head_fp32 = not self.args.exp036_quantize_lm_head

        def predicate(path: str) -> bool:
            if path.endswith("expert_bias") or path.endswith("mlp.gate.weight"):
                return False
            if keep_head_fp32 and path.startswith("lm_head."):
                return False
            return True

        return predicate

    @property
    def quant_predicate(self):
        """Policy (item 18): quantise attention projections, routed and shared
        experts, and (by default, BUILD_SPEC 5.1 delta 2) embed_tokens and
        lm_head; never the router, expert_bias or norms. The config keys
        exp036_quantize_embeddings / exp036_quantize_lm_head (written by
        convert.py) turn the last two off. mlx_lm.utils.quantize_model also
        skips any module whose weight.shape[-1] is not divisible by the group size."""
        return make_quant_predicate(
            quantize_embeddings=self.args.exp036_quantize_embeddings,
            quantize_lm_head=self.args.exp036_quantize_lm_head,
        )


def make_quant_predicate(
    group_size: Optional[int] = None,
    quantize_embeddings: bool = True,
    quantize_lm_head: bool = True,
):
    """Build the quant_predicate(path, module) used by Model.quant_predicate and
    by convert.py. Decisions use paths and duck typing only (no isinstance on
    classes of this file), so the predicate works on a model loaded from a
    different copy of kolibri1.py."""

    def predicate(path: str, module: nn.Module) -> bool:
        if not hasattr(module, "to_quantized"):
            return False  # Router (mlp.gate), RMSNorm, anything without a quantised form
        last = path.rsplit(".", 1)[-1]
        if last == "gate" or "expert_bias" in path or "norm" in last:
            return False
        if path == "lm_head":
            return quantize_lm_head
        if last == "embed_tokens":
            return quantize_embeddings
        weight = getattr(module, "weight", None)
        if group_size is not None and weight is not None and weight.shape[-1] % group_size != 0:
            return False
        return True

    return predicate
