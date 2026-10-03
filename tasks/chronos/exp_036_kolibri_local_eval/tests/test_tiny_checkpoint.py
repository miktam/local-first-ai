"""The tiny checkpoints look like the released one where it matters.

Checks the writer in tiny_checkpoint.py against the real layout (spec items
2, 3, 15, 17): tensor names, shapes and dtypes read straight from the
safetensors headers, the shard index, the config fields, and that the weight
scales make the routing and RMSNorm traps live. Two further checks run once
the sibling modules exist: the reference's reader returns the same values as
mlx, and mlx_lm loads the directory through config["model_file"].
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling

PRESETS = ["vendor", "pattern5"]

# Keys of the released config.json (Aleph-Alpha/Kolibri-1-BF16 @ 7a8f290e).
REAL_CONFIG_KEYS = {
    "architectures", "model_type", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
    "max_position_embeddings", "rms_norm_eps", "vocab_size", "rope_theta",
    "num_experts", "num_experts_per_tok", "moe_intermediate_size",
    "shared_expert_intermediate_size", "norm_topk_prob", "attention_bias",
    "attention_dropout", "tie_word_embeddings", "use_cache", "use_sliding_window",
    "sliding_window", "layer_types", "bos_token_id", "eos_token_id",
    "pad_token_id", "dtype", "head_dtype",
}  # fmt: skip


@pytest.fixture(params=PRESETS)
def preset_dir(request, tiny_vendor_dir, tiny_pattern5_dir):
    return request.param, (tiny_vendor_dir if request.param == "vendor" else tiny_pattern5_dir)


def _header(path: Path) -> dict:
    # safetensors: 8-byte little-endian header length, then a JSON header.
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def _expected_shapes(cfg: dict) -> dict[str, tuple[int, ...]]:
    """Spec item 3, written out independently of the writer."""
    H, V = cfg["hidden_size"], cfg["vocab_size"]
    q = cfg["num_attention_heads"] * cfg["head_dim"]
    kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    E, I, S = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"]
    out = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,), "lm_head.weight": (V, H)}
    for n in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{n}"
        out.update({
            f"{p}.self_attn.q_proj.weight": (q, H),
            f"{p}.self_attn.k_proj.weight": (kv, H),
            f"{p}.self_attn.v_proj.weight": (kv, H),
            f"{p}.self_attn.o_proj.weight": (H, q),
            f"{p}.self_attn.q_norm.weight": (cfg["head_dim"],),
            f"{p}.self_attn.k_norm.weight": (cfg["head_dim"],),
            f"{p}.input_layernorm.weight": (H,),
            f"{p}.post_attn_norm.weight": (H,),
            f"{p}.post_attention_layernorm.weight": (H,),
            f"{p}.post_ffn_norm.weight": (H,),
            f"{p}.mlp.gate.weight": (E, H),
            f"{p}.moe.router.expert_bias": (E,),
            f"{p}.mlp.shared_experts.gate_proj.weight": (S, H),
            f"{p}.mlp.shared_experts.up_proj.weight": (S, H),
            f"{p}.mlp.shared_experts.down_proj.weight": (H, S),
        })  # fmt: skip
        for e in range(E):
            out[f"{p}.mlp.experts.{e}.gate_proj.weight"] = (I, H)
            out[f"{p}.mlp.experts.{e}.up_proj.weight"] = (I, H)
            out[f"{p}.mlp.experts.{e}.down_proj.weight"] = (H, I)
    return out


def test_tensors_match_real_layout(preset_dir):
    _, d = preset_dir
    cfg = json.loads((d / "config.json").read_text())
    want = _expected_shapes(cfg)
    # Per layer: 15 fixed tensors + 3 per routed expert (1167 for the real model).
    assert len(want) == 3 + cfg["num_hidden_layers"] * (15 + 3 * cfg["num_experts"])

    index = json.loads((d / "model.safetensors.index.json").read_text())
    shards = sorted(d.glob("model-*-of-*.safetensors"))
    assert len(shards) >= 2
    seen: dict[str, dict] = {}
    for f in shards:
        for name, meta in _header(f).items():
            assert name not in seen, f"{name} in two shards"
            seen[name] = meta
            assert index["weight_map"][name] == f.name
    assert set(seen) == set(want)
    assert set(index["weight_map"]) == set(want)
    for name, meta in seen.items():
        assert meta["dtype"] == "BF16", f"{name} is {meta['dtype']}"  # every tensor, expert_bias included
        assert tuple(meta["shape"]) == want[name], name
    assert index["metadata"]["total_size"] == sum(2 * int(np.prod(s)) for s in want.values())


def test_config_mirrors_real_config(preset_dir):
    preset, d = preset_dir
    cfg = json.loads((d / "config.json").read_text())
    assert REAL_CONFIG_KEYS <= set(cfg)
    assert set(cfg) - REAL_CONFIG_KEYS == {"model_file"}
    assert cfg["model_file"] == "kolibri1.py"
    assert cfg["model_type"] == "kolibri1"
    assert "torch_dtype" not in cfg and cfg["dtype"] == "bfloat16"
    assert cfg["head_dtype"] == "float32"
    assert cfg["norm_topk_prob"] is False and cfg["tie_word_embeddings"] is False
    assert cfg["bos_token_id"] is None
    assert 0 <= cfg["pad_token_id"] < cfg["vocab_size"] and 0 <= cfg["eos_token_id"] < cfg["vocab_size"]
    assert len(cfg["layer_types"]) == cfg["num_hidden_layers"]
    full = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    if preset == "vendor":
        assert full == [4, 5] and cfg["sliding_window"] == 65 and cfg["num_experts_per_tok"] == 2
    else:
        assert full == [i for i in range(10) if i % 5 == 4]
        assert cfg["sliding_window"] == 17 and (cfg["num_experts"], cfg["num_experts_per_tok"]) == (16, 6)
        # q width differs from hidden, as in the real model (6144 vs 2560).
        assert cfg["num_attention_heads"] * cfg["head_dim"] != cfg["hidden_size"]
    gen = json.loads((d / "generation_config.json").read_text())
    assert gen["eos_token_id"] == [cfg["eos_token_id"], cfg["pad_token_id"]]
    assert (d / "kolibri1.py").exists() == tc.PORT_MODEL_FILE.exists()


def test_weight_scales_make_the_traps_live(preset_dir):
    """Norm weights far enough from 0 and 1 that (1 + w) vs w matters; the
    expert bias large enough that score-add and logit-add routing disagree."""
    _, d = preset_dir
    cfg = json.loads((d / "config.json").read_text())
    w = tc.load_checkpoint_f32(d)
    norms = np.concatenate([v for k, v in w.items() if k.endswith("norm.weight") or k.endswith("layernorm.weight")])
    assert 0.95 < norms.mean() < 1.05 and 0.07 < norms.std() < 0.13
    bias = np.concatenate([v for k, v in w.items() if k.endswith("expert_bias")])
    assert 1.5 < bias.std() < 2.5

    # Router on unit-RMS inputs, the scale the post_attention_layernorm hands it.
    k = cfg["num_experts_per_tok"]
    x = np.random.default_rng(0).standard_normal((256, cfg["hidden_size"])).astype(np.float32)
    logits = x @ w["model.layers.0.mlp.gate.weight"].T
    b = w["model.layers.0.moe.router.expert_bias"]

    def topk_sets(z):
        return np.sort(np.argsort(-z, axis=-1, kind="stable")[:, :k], -1)

    logit_add = topk_sets(logits + b)
    score_add = topk_sets(1.0 / (1.0 + np.exp(-logits)) + b)
    no_bias = topk_sets(logits)
    assert (logit_add != score_add).any(-1).mean() > 0.2
    assert (logit_add != no_bias).any(-1).mean() > 0.2
    # And the logits still matter: tokens do not all pick the same experts.
    assert len({tuple(r) for r in logit_add}) > 4


def test_writer_is_deterministic(tmp_path):
    a = tc.write_tiny_checkpoint(tmp_path / "a", seed=3, preset="pattern5", copy_port=False)
    b = tc.write_tiny_checkpoint(tmp_path / "b", seed=3, preset="pattern5", copy_port=False)
    c = tc.write_tiny_checkpoint(tmp_path / "c", seed=4, preset="pattern5", copy_port=False)
    for f in sorted(a.glob("model-*.safetensors")):
        assert f.read_bytes() == (b / f.name).read_bytes()
        assert f.read_bytes() != (c / f.name).read_bytes()


def test_overrides_and_edits(tmp_path):
    def zero_bias(t):
        t["model.layers.0.moe.router.expert_bias"][:] = 0.0

    d = tc.write_tiny_checkpoint(
        tmp_path / "one",
        preset="vendor",
        copy_port=False,
        overrides={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
        edit_tensors=zero_bias,
    )
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["num_hidden_layers"] == 1 and cfg["layer_types"] == ["full_attention"]
    w = tc.load_checkpoint_f32(d)
    assert not any(k.startswith("model.layers.1.") for k in w)
    assert np.all(w["model.layers.0.moe.router.expert_bias"] == 0.0)
    # Per-tensor seeding: layer 0 is the same as in the 6-layer checkpoint.
    full = tc.make_tensors(tc.preset_config("vendor"), seed=0)
    np.testing.assert_array_equal(
        w["model.layers.0.self_attn.q_proj.weight"],
        np.array(_bf16_round(full["model.layers.0.self_attn.q_proj.weight"])),
    )
    with pytest.raises(ValueError):
        tc.write_tiny_checkpoint(tmp_path / "bad", copy_port=False, overrides={"num_hidden_layers": 2})


def _bf16_round(x: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))


def test_random_ids():
    ids = tc.random_ids(500, seed=1)
    assert ids == tc.random_ids(500, seed=1)
    assert ids != tc.random_ids(500, seed=2)
    assert all(isinstance(i, int) for i in ids)
    assert min(ids) >= 0 and max(ids) < tc.VOCAB_SIZE - tc.RESERVED_TOP_IDS
    assert tc.EOS_TOKEN_ID not in ids and tc.PAD_TOKEN_ID not in ids


def test_reference_reader_matches_mlx(tiny_pattern5_dir):
    """The reference's own safetensors reader decodes the BF16 values exactly."""
    safetensors_np = import_sibling("reference.safetensors_np")
    ckpt = safetensors_np.open_checkpoint(str(tiny_pattern5_dir))
    w = tc.load_checkpoint_f32(tiny_pattern5_dir)
    for name in (
        "model.embed_tokens.weight",
        "model.layers.3.moe.router.expert_bias",
        "model.layers.4.self_attn.k_norm.weight",
        "model.layers.9.mlp.experts.15.down_proj.weight",
        "lm_head.weight",
    ):
        got = np.asarray(ckpt.get(name))
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, w[name], err_msg=name)


def test_mlx_lm_loads_through_model_file(tiny_vendor_dir):
    """mlx_lm.utils.load_model execs <dir>/kolibri1.py (spec item 17, option A)."""
    if not (tiny_vendor_dir / "kolibri1.py").exists():
        pytest.skip("port/kolibri1.py not written yet, so the checkpoint has no model file")
    import mlx.core as mx
    from mlx_lm.utils import load_model

    model, config = load_model(Path(tiny_vendor_dir))
    assert config["model_type"] == "kolibri1"
    ids = tc.random_ids(12, seed=5)
    logits = model(mx.array([ids]))
    mx.eval(logits)
    assert logits.shape == (1, 12, tc.VOCAB_SIZE)
    assert logits.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(logits)))


def test_w513_preset_is_the_vendor_preset_with_the_real_window():
    """BUILD_SPEC 6: a preset at the real window, 513, for the batching and
    cache tests; everything else is the vendor preset."""
    vendor, w513 = tc.preset_config("vendor"), tc.preset_config("w513")
    assert w513["sliding_window"] == 513 and w513["max_position_embeddings"] >= 6 * 513
    assert {k: v for k, v in w513.items() if k not in ("sliding_window", "max_position_embeddings")} == {
        k: v for k, v in vendor.items() if k not in ("sliding_window", "max_position_embeddings")
    }
    with pytest.raises(ValueError, match="unknown preset"):
        tc.preset_config("w512")
