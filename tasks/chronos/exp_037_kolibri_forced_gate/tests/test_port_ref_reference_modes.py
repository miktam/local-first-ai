# SPDX-License-Identifier: MIT
"""The reference's modes and gate-facing APIs (BUILD_SPEC 5.1b; named
test_reference_modes.py in BUILD_SPEC 6, test_port_ref_* here):

* round_bf16 is round-to-nearest-even, bit for bit as mlx converts;
* mlx_affine_np unpacks and dequantises MLX affine tensors to 0 ulp of
  mx.dequantize (fp32 and bf16 outputs) at 8, 4 and 2 bits;
* the dequantised mode presents a converted directory under the checkpoint
  names (census, shapes, refusal of a different architecture) and through
  the CLI;
* the bf16-emulation mode reproduces INTEGRATION_LOG entry 1's emulation
  numbers within their sampling noise, and rounds exactly where it says;
* force_ids fixes the selection without changing anything else;
* forward_streams equals separate passes and reads each tensor once.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling

PRESETS = ("vendor", "pattern5")


def _bf16_exact(x) -> bool:
    kolibri_ref = import_sibling("reference.kolibri_ref")
    x = np.asarray(x, dtype=np.float32)
    return bool(np.array_equal(kolibri_ref.round_bf16(x), x))


# --- round_bf16 and the MLX affine format ----------------------------------------


def test_round_bf16_matches_mlx_bit_for_bit():
    import mlx.core as mx

    affine = import_sibling("reference.mlx_affine_np")
    rng = np.random.default_rng(0)
    # Random bit patterns cover every exponent (normals, subnormals, inf, NaN);
    # add exact ties (lower half 0x8000) with even and odd kept bits.
    bits = rng.integers(0, 1 << 32, size=200_000, dtype=np.uint64).astype(np.uint32)
    ties = (rng.integers(0, 1 << 16, size=20_000, dtype=np.uint64).astype(np.uint32) << 16) | 0x8000
    special = np.array([0x00000000, 0x80000000, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F800000, 0x7FC00000,
                        0x00008000, 0x00018000, 0x3F808000, 0x3F818000], dtype=np.uint32)
    x = np.concatenate([bits, ties, special]).view(np.float32)
    got = affine.round_bf16(x)
    # mlx's CPU conversion is IEEE round-to-nearest-even, subnormals included.
    want = np.array(mx.array(x).astype(mx.bfloat16, stream=mx.cpu).astype(mx.float32, stream=mx.cpu))
    ok = ~np.isnan(want)
    np.testing.assert_array_equal(got[ok].view(np.uint32), want[ok].view(np.uint32))
    assert np.isnan(got[~ok]).all()
    # The GPU (Metal) conversion agrees except that it flushes fp32 subnormals
    # (|x| < 2^-126) to zero, which no activation of this model comes near.
    gpu = np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    normal = ok & (np.abs(x) >= np.float32(2.0**-126))
    np.testing.assert_array_equal(got[normal].view(np.uint32), gpu[normal].view(np.uint32))
    assert affine.round_bf16(np.float32(3.4e38)) == np.inf  # overflow rounds to inf, as in hardware


@pytest.mark.parametrize("bits", [8, 4, 2])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_unpack_and_dequantize_equal_mx_dequantize(bits, group_size):
    """0 ulp against mx.dequantize on the GPU: with bf16 scales (out=bfloat16,
    QuantizedEmbedding's values) and with fp32 scales (out=float32, the
    values quantized_matmul multiplies with). 3-D like a stacked expert tensor."""
    import mlx.core as mx

    affine = import_sibling("reference.mlx_affine_np")
    rng = np.random.default_rng(bits * 1000 + group_size)
    w = mx.array(rng.standard_normal((3, 40, 256)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=group_size, bits=bits)
    packed = np.array(wq)
    s32, b32 = np.array(s.astype(mx.float32)), np.array(b.astype(mx.float32))
    codes = affine.unpack(packed, bits)
    assert codes.shape == (3, 40, 256) and codes.max() <= (1 << bits) - 1
    # Re-packing the codes gives the stored words back.
    per_word = 32 // bits
    words = (codes.reshape(3, 40, -1, per_word) << (np.arange(per_word, dtype=np.uint32) * bits)).sum(
        -1, dtype=np.uint64).astype(np.uint32)
    np.testing.assert_array_equal(words, packed)

    got_bf16 = affine.dequantize(packed, s32, b32, bits, group_size, out="bfloat16")
    want_bf16 = np.array(mx.dequantize(wq, s, b, group_size=group_size, bits=bits).astype(mx.float32))
    np.testing.assert_array_equal(got_bf16.view(np.uint32), want_bf16.view(np.uint32))
    got_f32 = affine.dequantize(packed, s32, b32, bits, group_size, out="float32")
    want_f32 = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32),
                                      group_size=group_size, bits=bits))
    np.testing.assert_array_equal(got_f32.view(np.uint32), want_f32.view(np.uint32))


def test_quantized_matmul_uses_the_unrounded_fp32_values():
    """Why linear weights are dequantised to fp32 (mlx_affine_np docstring):
    quantized_matmul with an fp32 input agrees with the fp32 dequantised
    weight to fp32 accumulation error, and not with the bf16-rounded one."""
    import mlx.core as mx

    affine = import_sibling("reference.mlx_affine_np")
    rng = np.random.default_rng(1)
    w = mx.array(rng.standard_normal((512, 256)).astype(np.float32) * 0.05).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    x = rng.standard_normal((16, 256)).astype(np.float32)
    y = np.array(mx.quantized_matmul(mx.array(x), wq, s, b, transpose=True, group_size=64, bits=8))
    s32, b32 = np.array(s.astype(mx.float32)), np.array(b.astype(mx.float32))
    w32 = affine.dequantize(np.array(wq), s32, b32, 8, 64, out="float32").astype(np.float64)
    wbf = affine.dequantize(np.array(wq), s32, b32, 8, 64, out="bfloat16").astype(np.float64)
    scale = np.abs(y).max()
    assert np.abs(y - x @ w32.T).max() / scale < 1e-5
    assert np.abs(y - x @ wbf.T).max() / scale > 1e-4


def test_dequantize_refusals():
    affine = import_sibling("reference.mlx_affine_np")
    words = np.zeros((2, 8), dtype=np.uint32)
    with pytest.raises(NotImplementedError):
        affine.unpack(words, 3)
    with pytest.raises(TypeError):
        affine.unpack(words.astype(np.int64), 4)
    with pytest.raises(ValueError):
        affine.dequantize(words, np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32), 4, 64)
    with pytest.raises(ValueError):
        affine.dequantize(words, np.zeros((2, 1), np.float32), np.zeros((2, 1), np.float32), 4, 64, out="fp16")


# --- dequantised mode ---------------------------------------------------------------


@pytest.fixture(scope="module")
def converted_8bit(tmp_path_factory):
    """The tiny vendor checkpoint converted to 8 bits with the default policy."""
    import contextlib
    import io

    from test_convert import _write_minimal_tokenizer

    convert = import_sibling("port.convert")
    src = tc.write_tiny_checkpoint(tmp_path_factory.mktemp("modes_src"), seed=0, preset="vendor", copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _write_minimal_tokenizer(src)
    out = tmp_path_factory.mktemp("modes_conv") / "Kolibri-tiny-MLX-8bit-g64"
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(src, out, 8, 64)
    return src, out


def test_converted_checkpoint_presents_the_checkpoint_names(converted_8bit):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    affine = import_sibling("reference.mlx_affine_np")
    src, out = converted_8bit
    view = affine.ConvertedCheckpoint(str(out))
    cfg = kolibri_ref.load_config(str(src))
    shapes = kolibri_ref.expected_shapes(cfg)
    assert set(view.names()) == set(shapes)  # every name, nothing extra
    for name, shape in shapes.items():
        assert view.shape_of(name) == shape, name
    assert view.dtype_of("model.layers.0.self_attn.q_proj.weight") == "F32"  # dequantised
    assert view.dtype_of("model.layers.0.mlp.gate.weight") == "F32"  # stored fp32 router
    assert view.dtype_of("model.layers.0.input_layernorm.weight") == "BF16"
    assert view.is_quantised("lm_head.weight") and not view.is_quantised("model.norm.weight")
    # The reference accepts it (header census) against the source config.
    ref = kolibri_ref.KolibriReference(str(src), dequant_dir=str(out))
    assert ref.dequant_dir == str(out)


def test_dequantised_mode_refuses_another_architecture(converted_8bit, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    _, out = converted_8bit
    other = tc.write_tiny_checkpoint(tmp_path / "other", seed=0, preset="pattern5", copy_port=False)
    with pytest.raises(ValueError, match="architecture differs"):
        kolibri_ref.KolibriReference(str(other), dequant_dir=str(out))


def test_dequant_cli_records_the_build(converted_8bit, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    src, out = converted_8bit
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps(tc.random_ids(30, seed=9)))
    dump = tmp_path / "dump"
    kolibri_ref.main(["--model-dir", str(src), "--dequant-dir", str(out), "--ids-file", str(ids),
                      "--dump", str(dump), "--emulate-bf16", "--quiet"])
    record = json.loads((dump / "ref_record.json").read_text())
    convert_record = json.loads((out / "exp036_convert_record.json").read_text())
    assert record["mode"] == "bf16_emulation" and record["dequantised_from"] == out.name
    assert record["dequantised_manifest_sha256"] == convert_record["manifest_sha256"]
    ref = kolibri_ref.KolibriReference(str(src), dequant_dir=str(out), emulate_bf16=True)
    np.testing.assert_array_equal(np.load(dump / "logits.npy"), ref.forward(np.load(dump / "ids.npy")))


# --- bf16 emulation ------------------------------------------------------------------

# INTEGRATION_LOG entry 1: the earlier independent emulation (rounding where
# the port stores bf16, bf16 residual add before the norm) against the fp32
# reference, 160 tokens, ids seed 100 + seed. Re-run with that script on
# 2026-10-03; its range end points are the logged 0.863-0.975 and
# 2.0e-4-3.9e-3.
ENTRY1 = {  # (preset, seed): (top-1, mean KL)
    ("vendor", 0): (0.969, 1.27e-3),
    ("vendor", 1): (0.975, 2.03e-4),
    ("pattern5", 0): (0.887, 3.73e-3),
    ("pattern5", 1): (0.863, 3.85e-3),
}


def test_emulation_reproduces_integration_log_entry1(tiny_checkpoint_factory):
    """BUILD_SPEC 5.1b item 3: the emulation mode reproduces entry 1's
    emulation numbers on the same checkpoints and sequences.

    The two emulations round at the same stored values; this one threads the
    residual as vLLM's fused add + norm does (normalise the unrounded fp32
    sum, store it rounded), entry 1's as MLX does (round, then normalise),
    which spec item 5 bounds by 1 bf16 ulp at the norm input. Their routing
    flips therefore fall on different tokens, and a single 160-token case
    moves by binomial noise (sigma = sqrt(p (1 - p) / 160) = 0.027 at
    p = 0.86). The checks:
      * pooled over the four cases (640 positions), top-1 agrees with entry 1
        within 3 sigma of the pooled proportion (3 * 0.0105 = 0.032);
      * pooled mean KL within a factor 2 of entry 1's;
      * every case inside entry 1's band widened by 3 case sigmas (top-1
        >= 0.863 - 0.08) and within its KL band x/ 2.
    Measured: top-1 0.969, 0.981, 0.900, 0.856 (pooled 0.927 vs 0.924); KL
    8.8e-4, 2.7e-4, 2.7e-3, 3.5e-3 (pooled 1.9e-3 vs 2.3e-3)."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    top1, kl = [], []
    for (preset, seed), (t_entry, kl_entry) in ENTRY1.items():
        d = tiny_checkpoint_factory(preset, seed)
        ids = np.array(tc.random_ids(160, seed=100 + seed))
        fp32 = kolibri_ref.KolibriReference(str(d)).forward(ids)
        emu = kolibri_ref.KolibriReference(str(d), emulate_bf16=True).forward(ids)
        stats = ph.compare(fp32, emu)
        top1.append(stats["top1"])
        kl.append(stats["kl_mean"])
        assert 0.863 - 0.08 <= stats["top1"] < 1.0, (preset, seed, ph.fmt(stats))
        assert 2.0e-4 / 2 <= stats["kl_mean"] <= 3.9e-3 * 2, (preset, seed, ph.fmt(stats))
    entry_top1 = np.mean([v[0] for v in ENTRY1.values()])
    entry_kl = np.mean([v[1] for v in ENTRY1.values()])
    assert abs(np.mean(top1) - entry_top1) <= 0.032, (top1, entry_top1)
    assert entry_kl / 2 <= np.mean(kl) <= entry_kl * 2, (kl, entry_kl)


def test_emulation_rounds_exactly_where_it_says(tiny_pattern5_dir, monkeypatch):
    """In emulation every GEMM operand that vLLM keeps in bf16 is a bf16
    value (normed inputs, attention output, the SwiGLU product, the final
    norm), q/k after RoPE and v are bf16, the branch outputs are bf16, and
    the router logits and the head output are not rounded. In fp32 mode
    none of these is rounded."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    seen = {"linear_in": [], "qkv": []}
    real_linear, real_sdpa = kolibri_ref.linear, kolibri_ref.sdpa_ref

    def linear(x, w):
        seen["linear_in"].append(x)
        return real_linear(x, w)

    def sdpa(q, k, v, *args, **kwargs):
        seen["qkv"] += [q, k, v]
        return real_sdpa(q, k, v, *args, **kwargs)

    monkeypatch.setattr(kolibri_ref, "linear", linear)
    monkeypatch.setattr(kolibri_ref, "sdpa_ref", sdpa)
    ids = np.array(tc.random_ids(40, seed=12))
    for emulate in (True, False):
        seen["linear_in"].clear()
        seen["qkv"].clear()
        ref = kolibri_ref.KolibriReference(str(tiny_pattern5_dir), emulate_bf16=emulate)
        h = ref.embed(ids)
        out = ref.layer_branches(4, h)  # a full-attention layer
        out2 = ref.layer_branches(3, h)  # a sliding (RoPE) layer
        logits = ref.lm_head(ref.final_norm(out2["h_out"]))
        operands = seen["linear_in"]
        assert len(operands) > 10
        if emulate:
            assert all(_bf16_exact(x) for x in operands)
            assert all(_bf16_exact(x) for x in seen["qkv"])
            for o in (out, out2):
                assert _bf16_exact(o["r_attn"]) and _bf16_exact(o["r_moe"])
                assert not _bf16_exact(o["info"]["logits"])  # router logits fp32 (item 11)
            assert not _bf16_exact(logits)  # head fp32 (item 14)
        else:
            assert not all(_bf16_exact(x) for x in seen["qkv"])
            assert not _bf16_exact(out["r_attn"]) and not _bf16_exact(out["r_moe"])


# --- force_ids, branches, streams -------------------------------------------------


@pytest.mark.parametrize("emulate", [False, True], ids=["fp32", "emu"])
def test_force_ids(tiny_pattern5_dir, emulate):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ref = kolibri_ref.KolibriReference(str(tiny_pattern5_dir), emulate_bf16=emulate)
    h = ref.embed(np.array(tc.random_ids(30, seed=13)))
    natural = ref.layer_branches(2, h)
    top6 = natural["info"]["top6"]
    # Forcing the layer's own selection changes nothing, bit for bit.
    same = ref.layer_branches(2, h, force_ids=top6)
    assert same["info"]["forced"] and not natural["info"]["forced"]
    for key in ("r_attn", "h_mid", "r_moe", "h_out"):
        np.testing.assert_array_equal(same[key], natural[key])
    # Another selection: weights still sigmoid of this layer's own logits.
    other = np.roll(top6, 1, axis=0)
    forced = ref.layer_branches(2, h, force_ids=other)
    np.testing.assert_array_equal(forced["info"]["top6"], other)
    want = 1.0 / (1.0 + np.exp(-np.take_along_axis(natural["info"]["logits"].astype(np.float64), other, -1)))
    np.testing.assert_allclose(forced["info"]["weights"], want, rtol=1e-6)
    np.testing.assert_array_equal(forced["r_attn"], natural["r_attn"])  # attention is untouched
    assert np.abs(forced["r_moe"] - natural["r_moe"]).max() > 1e-2
    np.testing.assert_array_equal(ref.layer_forward(2, h, force_ids=other), forced["h_out"])
    with pytest.raises(ValueError, match="force_ids"):
        ref.moe_block(2, h, force_ids=top6[:5])


@pytest.mark.parametrize("emulate", [False, True], ids=["fp32", "emu"])
def test_forward_streams_equal_separate_passes(tiny_vendor_dir, emulate):
    """BUILD_SPEC 5.1b item 6: k hidden streams through one layer-streamed
    pass; each stream bit-identical to its own pass; every tensor read from
    the checkpoint at most once (the head's row chunks, read with get_rows,
    are read once per stream)."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    ref = kolibri_ref.KolibriReference(str(tiny_vendor_dir), emulate_bf16=emulate)
    seqs = [np.array(tc.random_ids(70, seed=14)), np.array(tc.random_ids(33, seed=15))]
    chosen = {k: mutants.MUTANTS[k] for k in ("rope_on_full", "renorm_topk", "swap_sandwich_norms")}

    reads = []
    real_get = ref.ckpt.get

    def counting_get(name):
        reads.append(name)
        return real_get(name)

    ref.ckpt.get = counting_get
    out = ref.forward_streams(seqs, chosen)
    ref.ckpt.get = real_get
    assert len(reads) == len(set(reads)), "a tensor was read twice in one streamed pass"
    assert ref._shared["cache"] is None
    np.testing.assert_array_equal(out["_lengths"], [70, 33])
    np.testing.assert_array_equal(out["ref"], ref.forward_packed(seqs)[0])
    for name, cls in chosen.items():
        separate = kolibri_ref.KolibriReference(str(tiny_vendor_dir), emulate_bf16=emulate)._clone_as(cls)
        np.testing.assert_array_equal(out[name], separate.forward_packed(seqs)[0], err_msg=name)
    # on_logits replaces the stored result; include_base=False drops "ref".
    reduced = ref.forward_streams(seqs[0], chosen, include_base=False, on_logits=lambda n, lg: lg.argmax(-1))
    assert set(reduced) == set(chosen) | {"_lengths"}
    np.testing.assert_array_equal(reduced["renorm_topk"],
                                  ref._clone_as(chosen["renorm_topk"]).forward(seqs[0]).argmax(-1))


def test_layer_forward_is_layer_branches(tiny_pattern5_dir):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ref = kolibri_ref.KolibriReference(str(tiny_pattern5_dir))
    h = ref.embed(np.array(tc.random_ids(25, seed=16)))
    out = ref.layer_branches(0, h)
    np.testing.assert_array_equal(out["h_mid"], h + out["r_attn"])
    np.testing.assert_array_equal(out["h_out"], out["h_mid"] + out["r_moe"])
    np.testing.assert_array_equal(ref.layer_forward(0, h), out["h_out"])
    info = ref.last_layer_info
    for key in ("r_attn", "h_mid", "r_moe"):
        np.testing.assert_array_equal(info[key], out[key])
    assert kolibri_ref.REF_VERSION == "exp036-ref-2"


def test_reference_imports_only_numpy_and_stdlib():
    """HYPOTHESIS "Reference": numpy only, never mlx or the port."""
    import ast
    import sys
    from pathlib import Path

    root = Path(import_sibling("reference.kolibri_ref").__file__).parent
    allowed = set(sys.stdlib_module_names) | {"numpy", "reference"}  # reference: its own package
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: inside reference/
                    continue
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                assert top in allowed or (root / f"{top}.py").exists(), f"{path.name} imports {name}"
