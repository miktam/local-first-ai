# SPDX-License-Identifier: MIT
"""The tiny real-layout checkpoints of exp_037 (tests/tiny_real_layout.py; DESIGN §5.1, W0b).

* the config is §5.1's: preset w513, 10 layers with full attention at i % 5 == 4,
  48 q / 4 kv heads of head_dim 128, window 513, 64 experts top-6, hidden 256,
  moe and shared intermediate 64, and q/k-norm weights x 3;
* the writer is deterministic per seed. At seed 23 it reproduces exp_036's
  bughunt build ckv_s3 byte for byte (the shas exp_036's
  diagnostics/gate1/mini/MANIFEST.json records). The conversions are
  reproducible too;
* the converted shards load strictly under mlx-lm 0.32.0, through
  port.convert.model_file_trust, and the BF16 shards load strictly into the port.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint as tc
import tiny_real_layout as trl
from exp036_helpers import import_sibling

EXP = Path(__file__).resolve().parents[1]
E36 = EXP.parent / "exp_036_kolibri_local_eval"
E36_MANIFEST = E36 / "diagnostics" / "gate1" / "mini" / "MANIFEST.json"

# exp_036's bughunt build ckv_s3 (build_ck.py: seed 23, sharpen 3), as recorded in
# E36 diagnostics/gate1/mini/MANIFEST.json (sources $SCRATCH/bughunt_verify/ckv_s3/*).
CKV_S3_SHA256 = {
    "config.json": "3ee351c7403329386518e3f0ad36f1b73c4597052190829c5bf281363622e7f3",
    "generation_config.json": "99176bb81e9428eff00663c0bd1a9a75914108bedd1dbe68a8e1c894fa37901d",
    "model.safetensors.index.json": "bed133df21f2e1bd4aff2c9f508f6def756dc0a16011940a609639eb7e21045d",
    "model-00001-of-00002.safetensors": "52b676b4b53ae081d3aa3ef86144ed4b97341f1c024a1c733279cebb866412a3",
    "model-00002-of-00002.safetensors": "ed2f74e5adeb4e89103bc2b1ac060870614bff3a620c93eeef1443f10f36299c",
}
WEIGHT_FILES = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                "model.safetensors.index.json")

# DESIGN §5.1, written out independently of the writer.
SECTION_5_1 = {
    "num_hidden_layers": 10,
    "num_attention_heads": 48,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "sliding_window": 513,
    "num_experts": 64,
    "num_experts_per_tok": 6,
    "hidden_size": 256,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
}
SETS = {"cal": ("tiny_real_cal", 23), "val": ("tiny_real_val", 29)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _set(request, which: str) -> "trl.RealLayoutSet":
    return request.getfixturevalue(SETS[which][0])


def _bf16_round(x: np.ndarray) -> np.ndarray:
    import mlx.core as mx

    return np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))


@pytest.mark.parametrize("which", sorted(SETS))
def test_config_matches_section_5_1(request, which):
    s = _set(request, which)
    assert (s.seed, s.sharpen) == (SETS[which][1], 3.0)
    assert (s.bf16, s.k8, s.k4) == (s.root / "Kolibri-1-BF16", s.root / "Kolibri-1-MLX-8bit-g64",
                                     s.root / "Kolibri-1-MLX-4bit-g64")
    cfg = json.loads((s.bf16 / "config.json").read_text())
    for key, want in SECTION_5_1.items():
        assert cfg[key] == want, key
    assert [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"] == [4, 9]
    assert len(cfg["layer_types"]) == 10 and set(cfg["layer_types"]) == {"full_attention", "sliding_attention"}
    # q width differs from hidden, as in the real model (6144 vs 2560).
    assert cfg["num_attention_heads"] * cfg["head_dim"] == 6144 != cfg["hidden_size"]
    # Everything else is the w513 preset (build_ck.py adds max_position_embeddings 32768).
    w513 = tc.preset_config("w513")
    w513.pop("model_file")
    rest = {k: v for k, v in cfg.items() if k not in SECTION_5_1 and k != "layer_types"}
    assert rest == {k: v for k, v in w513.items() if k not in SECTION_5_1 and k != "layer_types"} | {
        "max_position_embeddings": 32768}
    # The source layout of the released Kolibri-1-BF16: no model_file, no port copy.
    assert "model_file" not in cfg and not (s.bf16 / "kolibri1.py").exists()
    assert cfg == trl.real_layout_config(hf_layout=True)
    # §5.1's token ids are drawn from [0, 1008): every id below the reserved top ids.
    assert tc.VOCAB_SIZE - tc.RESERVED_TOP_IDS == 1008

    # q/k-norm weights x 3 (in float32, then the BF16 cast); every other tensor as drawn.
    w = tc.load_checkpoint_f32(s.bf16)
    for i in (0, 4, 9):
        for name in ("q_norm", "k_norm"):
            key = f"model.layers.{i}.self_attn.{name}.weight"
            drawn = tc._normal(s.seed, key, (128,), tc.STD_NORM, mean=1.0)
            np.testing.assert_array_equal(w[key], _bf16_round(drawn * np.float32(3.0)), err_msg=key)
        key = f"model.layers.{i}.input_layernorm.weight"
        np.testing.assert_array_equal(w[key], _bf16_round(tc._normal(s.seed, key, (256,), tc.STD_NORM, mean=1.0)))
    assert w["model.layers.0.mlp.experts.63.down_proj.weight"].shape == (256, 64)
    assert w["model.layers.9.self_attn.q_proj.weight"].shape == (6144, 256)

    # The conversions: affine, group 64, at 8 and 4 bits, with the pre-registered head policy.
    for bits, d in ((8, s.k8), (4, s.k4)):
        assert s.build(bits) == d
        out = json.loads((d / "config.json").read_text())
        assert out["model_file"] == "kolibri1.py"
        assert out["quantization"] == {"group_size": 64, "bits": bits, "mode": "affine"}
        assert out["exp036_quantize_embeddings"] is True and out["exp036_quantize_lm_head"] is True
        # mlx_lm's save merges generation_config.json's EOS list into the config.
        assert out["eos_token_id"] == [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]
        assert {k: v for k, v in out.items() if k in cfg and k != "eos_token_id"} == {
            k: v for k, v in cfg.items() if k != "eos_token_id"}
        record = json.loads((d / "exp036_convert_record.json").read_text())
        assert record["head_policy"] == "quantised_head" and record["bits"] == bits
        assert record["src_config_sha256"] == _sha256(s.bf16 / "config.json")


def test_seed_23_reproduces_exp036_bughunt_build(tmp_path, tiny_real_cal):
    """write_real_layout is build_ck.py's recipe: at seed 23 and sharpen 3 every
    file of exp_036's ckv_s3 comes out byte-identical, and the session's
    calibration checkpoint holds the same weights."""
    d = trl.write_real_layout(tmp_path / "ckv_s3", 23)
    for name, sha in CKV_S3_SHA256.items():
        assert _sha256(d / name) == sha, name
    assert (d / "kolibri1.py").read_bytes() == tc.PORT_MODEL_FILE.read_bytes()
    assert {p.name for p in d.iterdir()} == set(CKV_S3_SHA256) | {"kolibri1.py"}
    for name in WEIGHT_FILES:
        assert _sha256(tiny_real_cal.bf16 / name) == CKV_S3_SHA256[name], name

    # The pinned values are exp_036's record (read-only, by relative path).
    manifest = json.loads(E36_MANIFEST.read_text())
    recorded = {}
    for entry in manifest["included"] + manifest["excluded"]:
        src = entry.get("source") or ""
        if src.startswith("$SCRATCH/bughunt_verify/ckv_s3/"):
            recorded[src.rsplit("/", 1)[1]] = entry["sha256"]
    assert {k: v for k, v in recorded.items() if k in CKV_S3_SHA256} == CKV_S3_SHA256


def test_deterministic_per_seed(tmp_path, tiny_real_cal, tiny_real_val):
    """Same seed, same bytes (checkpoint and conversion); another seed, other weights."""
    convert = import_sibling("port.convert")
    again = trl.write_real_layout(tmp_path / "val", 29, hf_layout=True)
    for name in WEIGHT_FILES + ("config.json", "generation_config.json"):
        assert (again / name).read_bytes() == (tiny_real_val.bf16 / name).read_bytes(), name
    for name in WEIGHT_FILES[:2]:
        assert _sha256(tiny_real_val.bf16 / name) != _sha256(tiny_real_cal.bf16 / name), name
    assert (tiny_real_val.bf16 / "config.json").read_bytes() == (tiny_real_cal.bf16 / "config.json").read_bytes()

    # The 8-bit conversion of the same source is the same tensors, bit for bit.
    trl.write_stub_tokenizer(again)
    with contextlib.redirect_stdout(io.StringIO()):
        record = convert.convert(again, tmp_path / "k8", 8, 64, guard=False)
    first = json.loads((tiny_real_val.k8 / "exp036_convert_record.json").read_text())
    assert record["files"] == first["files"] and "model.safetensors" in record["files"]
    assert record["manifest_sha256"] == first["manifest_sha256"]


@pytest.mark.parametrize("bits", trl.BITS)
@pytest.mark.parametrize("which", sorted(SETS))
def test_converted_shards_load_strictly(request, which, bits):
    """Each conversion loads strictly through mlx_lm under mlx-lm 0.32.0, with
    the trust model_file_trust grants after check_port_file, and runs."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import load_model

    convert = import_sibling("port.convert")
    d = _set(request, which).build(bits)
    assert convert.check_port_file(d) == _sha256(convert.PORT_FILE)
    trust = convert.model_file_trust(d)
    assert trust == {"trust_remote_code": True}
    model, cfg = load_model(d, strict=True, **trust)
    assert cfg["quantization"]["bits"] == bits
    layer = model.layers[4]
    assert isinstance(layer.self_attn.q_proj, nn.QuantizedLinear) and layer.self_attn.q_proj.bits == bits
    assert type(layer.mlp.switch_mlp.down_proj).__name__ == "QuantizedSwitchLinear"
    assert layer.mlp.gate.weight.dtype == mx.float32
    assert isinstance(model.lm_head, nn.QuantizedLinear) and model.lm_head.bits == bits
    logits = model(mx.array([tc.random_ids(16, seed=7)]))
    mx.eval(logits)
    assert logits.shape == (1, 16, tc.VOCAB_SIZE) and logits.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(logits)))


@pytest.mark.parametrize("which", sorted(SETS))
def test_bf16_shards_load_strictly_into_the_port(request, which):
    """The BF16 source names no model_file, so model_file_trust grants nothing
    and the port is built from port/kolibri1.py directly (as the gate loads)."""
    import mlx.core as mx
    from mlx_lm.utils import load_model

    convert = import_sibling("port.convert")
    d = _set(request, which).bf16
    assert convert.model_file_trust(d) == {}
    port = convert.load_port_module(convert.PORT_FILE, "exp037_test_tiny_real_layout_port")
    model, _ = load_model(d, strict=True, get_model_classes=lambda config: (port.Model, port.ModelArgs))
    assert len(model.layers) == 10 and model.layers[0].self_attn.q_proj.weight.shape == (6144, 256)
    logits = model(mx.array([tc.random_ids(16, seed=7)]))
    mx.eval(logits)
    assert logits.shape == (1, 16, tc.VOCAB_SIZE) and bool(mx.all(mx.isfinite(logits)))

