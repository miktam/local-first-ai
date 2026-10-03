# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""Tiny random-weight Kolibri 1 checkpoints for the exp_036 tests.

Re-expresses the vendor's tests/checkpoints.py without torch. The weights are
drawn with numpy and written as BF16 safetensors through mlx
(mx.save_safetensors), so the writer shares no code with the numpy reference's
reader (reference/safetensors_np.py) or with the port's loader.

Tensor names, dtypes and config fields mirror the released checkpoint
(Aleph-Alpha/Kolibri-1-BF16): spec items 2, 3 and 21. The checkpoint is split
into shards with a model.safetensors.index.json, as the real one is.

Three presets:

  "vendor"   the vendor test config (hidden 256, 6 layers, 8 q / 2 kv heads,
             head_dim 32, 8 experts top-2, sliding_window 65, layers 4-5
             full), with the vocab cut from 96000 to 1024 for speed.
  "pattern5" the real layer layout at small scale: 10 layers, full attention
             at i % 5 == 4, 16 experts top-6, sliding_window 17. It uses
             12 q / 2 kv heads so that n_heads * head_dim (384) differs from
             hidden_size (256), as in the real model (6144 vs 2560).
  "w513"     the vendor preset with the real window, sliding_window 513, and
             max_position_embeddings 8192, for cache and batching tests
             whose sequences run to several windows (BUILD_SPEC 6).

Scales are chosen so that the semantic traps in the spec are visible:
norm weights are 1 + N(0, 0.1), so (1 + w) instead of w roughly doubles a
norm's output (item 6); expert_bias is N(0, 2), large next to the router
logits, so selecting on sigmoid(logits) + bias picks different experts than
logits + bias (item 12).
"""

from __future__ import annotations

import json
import os
import shutil
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

# The port module copied into each checkpoint dir, so mlx_lm can load the dir
# through config["model_file"] (spec item 17, option A).
# Mutation-testing hook: $EXP036_PORT_FILE, when set, names another copy of
# kolibri1.py to copy instead (a deliberately broken port), so the suite can be
# run against it without editing port/kolibri1.py. Unset in normal runs. It
# reaches every test that loads the port from a tiny checkpoint; tests that
# import port.kolibri1 or port.convert directly still see the real file.
PORT_MODEL_FILE = Path(
    os.environ.get("EXP036_PORT_FILE")
    or Path(__file__).resolve().parent.parent / "port" / "kolibri1.py"
).resolve()

VOCAB_SIZE = 1024
# Mirrors the real config: no BOS, EOS <|im_end|>, PAD <|endoftext|>, both
# near the top of the vocabulary. Prompts from random_ids() stay below these.
EOS_TOKEN_ID = 1022
PAD_TOKEN_ID = 1021
# random_ids() draws from [0, VOCAB_SIZE - RESERVED_TOP_IDS).
RESERVED_TOP_IDS = 16

# Standard deviations of the random weights.
STD_EMBED = 1.0
STD_PROJ = 0.05  # q/k/v/o, routed and shared experts, lm_head
STD_ROUTER = 0.1  # mlp.gate.weight; router logits come out with std ~1.6
STD_NORM = 0.1  # norm weights are 1 + N(0, STD_NORM)
STD_EXPERT_BIAS = 2.0


def _layer_types(n_layers: int, full_every: int) -> list[str]:
    # Real layout: full attention at i % 5 == 4 (spec item 2).
    return [
        "full_attention" if i % full_every == full_every - 1 else "sliding_attention"
        for i in range(n_layers)
    ]


def _base_config() -> dict:
    # Field set and order follow the real config.json (spec item 2).
    return {
        "architectures": ["Kolibri1ForCausalLM"],
        "model_type": "kolibri1",
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "hidden_act": "silu",
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB_SIZE,
        "rope_theta": 10000.0,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
        "norm_topk_prob": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
        "sliding_window": 65,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ],
        "bos_token_id": None,
        "eos_token_id": EOS_TOKEN_ID,
        "pad_token_id": PAD_TOKEN_ID,
        "dtype": "bfloat16",
        "head_dtype": "float32",
        "model_file": "kolibri1.py",
    }


def preset_config(preset: str) -> dict:
    """Return the config.json dict for a preset (a fresh copy)."""
    cfg = _base_config()
    if preset == "vendor":
        return cfg
    if preset == "pattern5":
        cfg.update(
            {
                "num_hidden_layers": 10,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_experts": 16,
                "num_experts_per_tok": 6,
                "moe_intermediate_size": 128,
                "shared_expert_intermediate_size": 128,
                "sliding_window": 17,
                "layer_types": _layer_types(10, 5),
            }
        )
        return cfg
    if preset == "w513":
        cfg.update({"sliding_window": 513, "max_position_embeddings": 8192})
        return cfg
    raise ValueError(f"unknown preset {preset!r}; use 'vendor', 'pattern5' or 'w513'")


def _normal(seed: int, name: str, shape: tuple[int, ...], std: float, mean: float = 0.0) -> np.ndarray:
    # One generator per tensor, keyed by (seed, name). A tensor's values do not
    # depend on which other tensors exist, so a test can drop or edit layers
    # and still get the same weights everywhere else.
    rng = np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])
    return (mean + std * rng.standard_normal(shape)).astype(np.float32)


def make_tensors(cfg: dict, seed: int = 0) -> dict[str, np.ndarray]:
    """All checkpoint tensors as float32 numpy arrays, keyed by the real names."""
    H = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    E = cfg["num_experts"]
    moe_inter = cfg["moe_intermediate_size"]
    shared_inter = cfg["shared_expert_intermediate_size"]
    V = cfg["vocab_size"]

    def proj(name, shape):
        return _normal(seed, name, shape, STD_PROJ)

    def norm(name, dim):
        return _normal(seed, name, (dim,), STD_NORM, mean=1.0)

    t: dict[str, np.ndarray] = {}
    t["model.embed_tokens.weight"] = _normal(seed, "model.embed_tokens.weight", (V, H), STD_EMBED)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        # Attention (spec item 7): q/k/v/o without bias, per-head q/k norms.
        t[f"{p}.self_attn.q_proj.weight"] = proj(f"{p}.self_attn.q_proj.weight", (n_heads * hd, H))
        t[f"{p}.self_attn.k_proj.weight"] = proj(f"{p}.self_attn.k_proj.weight", (n_kv * hd, H))
        t[f"{p}.self_attn.v_proj.weight"] = proj(f"{p}.self_attn.v_proj.weight", (n_kv * hd, H))
        t[f"{p}.self_attn.o_proj.weight"] = proj(f"{p}.self_attn.o_proj.weight", (H, n_heads * hd))
        t[f"{p}.self_attn.q_norm.weight"] = norm(f"{p}.self_attn.q_norm.weight", hd)
        t[f"{p}.self_attn.k_norm.weight"] = norm(f"{p}.self_attn.k_norm.weight", hd)
        # The four layer norms (spec item 5; 'post_attention_layernorm' is the
        # PRE-MoE norm, 'post_attn_norm' / 'post_ffn_norm' are the sandwich norms).
        for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm"):
            t[f"{p}.{n}.weight"] = norm(f"{p}.{n}.weight", H)
        # Router (spec item 11) and its correction bias under the torchtitan
        # name (spec item 12). Both are BF16 in the real checkpoint.
        t[f"{p}.mlp.gate.weight"] = _normal(seed, f"{p}.mlp.gate.weight", (E, H), STD_ROUTER)
        t[f"{p}.moe.router.expert_bias"] = _normal(seed, f"{p}.moe.router.expert_bias", (E,), STD_EXPERT_BIAS)
        # Ungated shared expert (spec item 13).
        sp = f"{p}.mlp.shared_experts"
        t[f"{sp}.gate_proj.weight"] = proj(f"{sp}.gate_proj.weight", (shared_inter, H))
        t[f"{sp}.up_proj.weight"] = proj(f"{sp}.up_proj.weight", (shared_inter, H))
        t[f"{sp}.down_proj.weight"] = proj(f"{sp}.down_proj.weight", (H, shared_inter))
        # Routed experts, one tensor per expert as in the HF checkpoint.
        for e in range(E):
            ep = f"{p}.mlp.experts.{e}"
            t[f"{ep}.gate_proj.weight"] = proj(f"{ep}.gate_proj.weight", (moe_inter, H))
            t[f"{ep}.up_proj.weight"] = proj(f"{ep}.up_proj.weight", (moe_inter, H))
            t[f"{ep}.down_proj.weight"] = proj(f"{ep}.down_proj.weight", (H, moe_inter))
    t["model.norm.weight"] = norm("model.norm.weight", H)
    # Untied (tie_word_embeddings false).
    t["lm_head.weight"] = proj("lm_head.weight", (V, H))
    return t


def _shard_of(name: str, n_layers: int, n_shards: int) -> int:
    # Embedding with the first layers, final norm and lm_head with the last,
    # as in the real 32-shard layout.
    if name.startswith("model.embed_tokens"):
        return 0
    if name.startswith("model.norm") or name.startswith("lm_head"):
        return n_shards - 1
    layer = int(name.split(".")[2])
    return min(layer * n_shards // max(n_layers, 1), n_shards - 1)


def write_tiny_checkpoint(
    out_dir,
    *,
    seed: int = 0,
    preset: str = "vendor",
    copy_port: bool = True,
    overrides: dict | None = None,
    edit_tensors: Callable[[dict[str, np.ndarray]], None] | None = None,
    n_shards: int = 2,
) -> Path:
    """Write a tiny Kolibri 1 checkpoint directory and return its path.

    out_dir      directory to create (may exist; files in it are overwritten)
    seed         weight seed
    preset       "vendor", "pattern5" or "w513"
    copy_port    copy port/kolibri1.py next to config.json (config["model_file"])
    overrides    config fields to replace before the weights are drawn, e.g.
                 {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
    edit_tensors called with the float32 tensor dict before the BF16 cast; may
                 modify, add or delete entries in place
    n_shards     number of model-0000k-of-0000n.safetensors files
    """
    import mlx.core as mx  # lazy: only the writer needs mlx

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = preset_config(preset)
    if overrides:
        cfg.update(overrides)
    if len(cfg["layer_types"]) != cfg["num_hidden_layers"]:
        raise ValueError(
            f"layer_types has {len(cfg['layer_types'])} entries, "
            f"num_hidden_layers is {cfg['num_hidden_layers']}"
        )

    tensors = make_tensors(cfg, seed=seed)
    if edit_tensors is not None:
        edit_tensors(tensors)

    n_layers = cfg["num_hidden_layers"]
    n_shards = max(1, min(n_shards, n_layers + 1))
    shards: list[dict[str, "mx.array"]] = [{} for _ in range(n_shards)]
    weight_map: dict[str, str] = {}
    total_size = 0
    names = [f"model-{k + 1:05d}-of-{n_shards:05d}.safetensors" for k in range(n_shards)]
    for name, value in tensors.items():
        k = _shard_of(name, n_layers, n_shards)
        # Every tensor is BF16 in the real checkpoint, expert_bias included
        # (spec item 3). mlx rounds float32 -> bfloat16 to nearest even.
        shards[k][name] = mx.array(value).astype(mx.bfloat16)
        weight_map[name] = names[k]
        total_size += value.size * 2
    for k in range(n_shards):
        mx.save_safetensors(str(out / names[k]), shards[k], metadata={"format": "pt"})
    index = {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    # mlx_lm.utils.load_config merges this EOS list into the config (spec item 15).
    gen = {"eos_token_id": [EOS_TOKEN_ID, PAD_TOKEN_ID], "pad_token_id": PAD_TOKEN_ID}
    (out / "generation_config.json").write_text(json.dumps(gen, indent=2) + "\n")

    if copy_port:
        if not PORT_MODEL_FILE.exists():
            raise FileNotFoundError(f"copy_port=True but {PORT_MODEL_FILE} does not exist yet")
        shutil.copyfile(PORT_MODEL_FILE, out / "kolibri1.py")
    return out


def load_checkpoint_f32(model_dir) -> dict[str, np.ndarray]:
    """Read a tiny checkpoint back as float32 numpy arrays, through mlx.

    For tests that need the stored (BF16-rounded) values without going through
    the reference's reader.
    """
    import mlx.core as mx

    out: dict[str, np.ndarray] = {}
    for f in sorted(Path(model_dir).glob("model*.safetensors")):
        for name, arr in mx.load(str(f)).items():
            out[name] = np.array(arr.astype(mx.float32))
    return out


def random_ids(n: int, vocab: int = VOCAB_SIZE, seed: int = 0) -> list[int]:
    """n token ids drawn uniformly from [0, vocab - RESERVED_TOP_IDS).

    The top ids (EOS, PAD) are excluded so a prompt never contains a stop token.
    """
    rng = np.random.default_rng(seed)
    high = vocab - RESERVED_TOP_IDS if vocab > RESERVED_TOP_IDS else vocab
    return [int(x) for x in rng.integers(0, high, size=n)]
