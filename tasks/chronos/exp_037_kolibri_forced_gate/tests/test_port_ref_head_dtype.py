# SPDX-License-Identifier: MIT
"""Logit dtypes of the port (spec items 11 and 14; BUILD_SPEC 5.1 deltas 1-4;
`test_head_dtype.py` in BUILD_SPEC 6, named test_port_ref_* here).

* router logits are fp32 and essentially never bf16-representable;
* head output is fp32 for the bf16 head of a direct BF16 load, for the
  quantised head (8 and 4 bits) and for the vendor-faithful fp32 head, and
  its bf16-exact fraction is ~0 (the G0 value-level check, threshold 1 %);
* on a large-logit fixture (|logit| >= 16, where one bf16 ulp is >= 0.125)
  the quantised head fed fp32 activations differs from the same head fed
  bf16 activations, the defect of exp036-port-1 (mutant 11);
* the bf16 variants (mutants 10 and 11) give a bf16-exact fraction of 1.

"bf16-exact" means x == float32(bfloat16(x)), i.e. the low 16 bits of the
fp32 pattern are zero. For fp32 compute the expected fraction is about 2^-16
plus the exact zeros.
"""

from __future__ import annotations

import contextlib
import io
import json

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling


def bf16_exact_fraction(x) -> float:
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
    a = a[np.isfinite(a)]
    return float(np.mean((a.view(np.uint32) & np.uint32(0xFFFF)) == 0))


def _source(tmp_path_factory, name: str, lm_head_scale: float = 1.0):
    from test_convert import _write_minimal_tokenizer

    def edit(t):
        t["lm_head.weight"] = t["lm_head.weight"] * np.float32(lm_head_scale)

    src = tc.write_tiny_checkpoint(tmp_path_factory.mktemp(name), seed=0, preset="vendor", copy_port=False,
                                   edit_tensors=edit if lm_head_scale != 1.0 else None)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _write_minimal_tokenizer(src)
    return src


@pytest.fixture(scope="module")
def builds(tmp_path_factory):
    """The tiny source (BF16, loaded directly) and its conversions: 8-bit and
    4-bit with the quantised head, 8-bit vendor-faithful. The lm_head is
    scaled x40 so the logits reach |logit| ~ 16-100 (std ~25), the
    large-logit fixture of HYPOTHESIS "Port mutants"."""
    convert = import_sibling("port.convert")
    src = _source(tmp_path_factory, "head_src", lm_head_scale=40.0)
    out = {"bf16-direct": src}
    for name, bits, q in (("q8", 8, True), ("q4", 4, True), ("vendor8", 8, False)):
        d = tmp_path_factory.mktemp(f"head_{name}") / name
        with contextlib.redirect_stdout(io.StringIO()):
            convert.convert(src, d, bits, 64, quantize_embeddings=q, quantize_lm_head=q)
        out[name] = d
    # The direct load needs the port file next to config.json.
    (src / "kolibri1.py").write_bytes((convert.PORT_FILE).read_bytes())
    cfg = json.loads((src / "config.json").read_text())
    cfg["model_file"] = "kolibri1.py"
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out


@pytest.mark.parametrize("build", ["bf16-direct", "q8", "q4", "vendor8"])
def test_head_and_router_logits_are_fp32(builds, build):
    import mlx.core as mx

    model = ph.load_port(builds[build])
    ids = mx.array([tc.random_ids(96, seed=50)])
    h = model.forward_hidden(ids)
    assert h.dtype == mx.bfloat16  # the trunk runs in bf16
    logits = model.compute_logits(h)
    assert logits.dtype == mx.float32
    np.testing.assert_array_equal(np.array(model(ids)), np.array(logits))
    lg = np.array(logits[0])
    assert np.abs(lg).max() >= 16, "the fixture must have large logits"
    assert bf16_exact_fraction(lg) <= 0.01, bf16_exact_fraction(lg)
    # Router logits of every layer, from the layer's own branches.
    x = model.model.embed_tokens(ids)
    fa, swa = model.model.make_masks(x)
    for layer in model.layers:
        out = layer.branches(x, swa if layer.use_sliding else fa)
        router_logits = out[4]
        assert router_logits.dtype == mx.float32
        assert bf16_exact_fraction(np.array(router_logits)) <= 0.01
        x = out[3]


@pytest.mark.parametrize("build", ["q8", "q4"])
def test_quantised_head_gets_fp32_activations(builds, build):
    """Delta 1: the quantised head is fed h.astype(float32). Feeding it bf16
    activations (exp036-port-1's h.astype(scales.dtype), mutant 11) yields
    bf16 logits, which on this fixture differ by up to half an ulp at
    |logit| >= 16 (>= 0.0625) plus the input rounding."""
    import mlx.core as mx

    model = ph.load_port(builds[build])
    ids = mx.array([tc.random_ids(96, seed=51)])
    h = model.forward_hidden(ids)
    good = np.array(model.compute_logits(h)[0])
    bad = np.array(model.lm_head(h.astype(mx.bfloat16)).astype(mx.float32)[0])
    assert bf16_exact_fraction(bad) == 1.0 and bf16_exact_fraction(good) <= 0.01
    assert np.abs(good - bad).max() >= 0.0625
    # The fp32 path is the dequantised fp32 weight times fp32 activations.
    affine = import_sibling("reference.mlx_affine_np")
    lm = model.lm_head
    w = affine.dequantize(np.array(lm.weight), np.array(lm.scales.astype(mx.float32)),
                          np.array(lm.biases.astype(mx.float32)), lm.bits, lm.group_size, out="float32")
    exact = np.array(h[0].astype(mx.float32)).astype(np.float64) @ w.astype(np.float64).T
    assert np.abs(good - exact).max() / np.abs(exact).max() <= 1e-5


def test_vendor_faithful_head_is_fp32_upcast(builds):
    import mlx.core as mx

    model = ph.load_port(builds["vendor8"])
    assert model.lm_head.weight.dtype == mx.float32
    stored = tc.load_checkpoint_f32(builds["bf16-direct"])["lm_head.weight"]
    np.testing.assert_array_equal(np.array(model.lm_head.weight), stored)


def test_bf16_mutants_are_fully_bf16_exact(builds):
    """The value-level check can see mutants 10 and 11: computed in bf16, every
    router logit and every head logit is bf16-exact (fraction 1), against
    ~0 for the port. Inline versions of the two defects; the gate's own
    mutants (gate/port_mutants.py) are checked too when that module exists."""
    import mlx.core as mx

    model = ph.load_port(builds["q8"])
    ids = mx.array([tc.random_ids(64, seed=52)])
    h = model.forward_hidden(ids)
    head_bf16 = model.lm_head(h.astype(mx.bfloat16)).astype(mx.float32)
    assert bf16_exact_fraction(np.array(head_bf16)) == 1.0
    gate = model.layers[0].mlp.gate
    x = model.model.embed_tokens(ids)
    router_bf16 = (x.astype(mx.bfloat16) @ gate.weight.astype(mx.bfloat16).T).astype(mx.float32)
    assert bf16_exact_fraction(np.array(router_bf16)) == 1.0
    assert bf16_exact_fraction(np.array(gate(x))) <= 0.01


def test_gate_port_mutants_10_and_11_fail_the_value_check(builds):
    """Cross-check with the gate's mutants 10 and 11 (BUILD_SPEC 5.3)."""
    import mlx.core as mx

    try:
        gate_common = import_sibling("gate.common")
        import_sibling("gate.port_mutants")
    except Exception as e:  # noqa: BLE001 - another builder's area, maybe mid-edit
        pytest.skip(f"gate/port_mutants.py not usable yet: {e}")
    ids = mx.array([tc.random_ids(64, seed=53)])
    for mutant in (None, "bf16_router_logits", "bf16_head_logits"):
        model, _, _ = gate_common.load_port(builds["q8"], mutant=mutant)
        h = model.forward_hidden(ids)
        head = bf16_exact_fraction(np.array(model.compute_logits(h)))
        x = model.model.embed_tokens(ids)
        router = bf16_exact_fraction(np.array(model.layers[0].mlp.gate(x)))
        if mutant is None:
            assert head <= 0.01 and router <= 0.01
        elif mutant == "bf16_router_logits":
            assert router == 1.0 and head <= 0.01
        else:
            assert head == 1.0 and router <= 0.01
