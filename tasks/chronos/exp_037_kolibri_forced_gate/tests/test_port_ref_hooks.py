# SPDX-License-Identifier: MIT
"""The port's exp037-port-1 surface (BUILD_SPEC 5.1 deltas 2-5; SPEC_VERSION per exp_037 DESIGN §2.10):

* inspection hooks: Model.forward_hidden, Model.compute_logits,
  Kolibri1Model.make_masks and DecoderLayer.branches, consistent with the
  normal forward bit for bit, with force_ids fixing the MoE selection only;
* the branches of each port layer against the reference's layer_branches in
  fp32, forced onto the same experts (the G2 fp32 forced row on tiny);
* sanitize / cast_predicate / quant_predicate under both head policies;
* invariants: SPEC_VERSION, no import of reference/.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling


def _np(a):
    import mlx.core as mx

    return np.array(a.astype(mx.float32))


def test_spec_version_and_no_reference_import():
    kolibri1 = import_sibling("port.kolibri1")
    assert kolibri1.SPEC_VERSION == "exp037-port-1"
    tree = ast.parse(Path(kolibri1.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] == "reference" for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "reference" and not node.level


@pytest.mark.parametrize("float32", [True, False], ids=["fp32", "bf16"])
def test_hooks_reproduce_the_forward(tiny_pattern5_dir, float32):
    import mlx.core as mx

    model = ph.load_port(tiny_pattern5_dir, float32=float32)
    ids = mx.array([tc.random_ids(50, seed=60)])
    logits = model(ids)
    h = model.forward_hidden(ids)
    np.testing.assert_array_equal(_np(model.compute_logits(h)), _np(logits))
    np.testing.assert_array_equal(_np(h), _np(model.model(ids)))
    # Layer by layer with make_masks and branches gives the same final state.
    x = model.model.embed_tokens(ids)
    fa, swa = model.model.make_masks(x)
    for layer in model.layers:
        r_attn, h_mid, r_moe, h_out, router_logits, sel = layer.branches(x, swa if layer.use_sliding else fa)
        np.testing.assert_array_equal(_np(h_mid), _np(x + r_attn))
        np.testing.assert_array_equal(_np(h_out), _np(h_mid + r_moe))
        np.testing.assert_array_equal(_np(layer(x, swa if layer.use_sliding else fa)), _np(h_out))
        assert router_logits.dtype == mx.float32 and sel.shape[-1] == model.args.num_experts_per_tok
        x = h_out
    np.testing.assert_array_equal(_np(model.model.norm(x)), _np(h))


def test_force_ids_fixes_only_the_selection(tiny_pattern5_dir):
    import mlx.core as mx

    model = ph.load_port(tiny_pattern5_dir, float32=True)
    ids = mx.array([tc.random_ids(40, seed=61)])
    x = model.model.embed_tokens(ids)
    fa, swa = model.model.make_masks(x)
    layer = model.layers[1]
    mask = swa if layer.use_sliding else fa
    natural = layer.branches(x, mask)
    own = layer.branches(x, mask, force_ids=natural[5])
    for a, b in zip(natural[:5], own[:5]):
        np.testing.assert_array_equal(_np(a), _np(b))
    other = mx.roll(natural[5], 1, axis=1)
    forced = layer.branches(x, mask, force_ids=other)
    np.testing.assert_array_equal(np.array(forced[5]), np.array(other))
    np.testing.assert_array_equal(_np(forced[0]), _np(natural[0]))  # attention untouched
    np.testing.assert_array_equal(_np(forced[4]), _np(natural[4]))  # same router logits
    assert np.abs(_np(forced[2]) - _np(natural[2])).max() > 1e-3
    # Weights of a forced selection: sigmoid of the layer's own fp32 logits.
    kolibri1 = import_sibling("port.kolibri1")
    w, sel = kolibri1.forced_route(natural[4], other)
    want = 1 / (1 + np.exp(-np.take_along_axis(np.array(natural[4]).astype(np.float64), np.array(other).astype(np.int64), -1)))
    np.testing.assert_allclose(np.array(w), want, rtol=1e-6)


@pytest.mark.parametrize("preset", ["vendor", "pattern5"])
def test_port_branches_match_reference_branches_fp32(tiny_checkpoint_factory, preset):
    """The G2 fp32 forced row on tiny: port layer l fed the reference's h_l,
    forced onto the reference's experts; branch errors relative to the
    reference branch, per token: max <= 1e-4 (G2 thresholds: median <= 1e-4,
    max <= 1e-3). Measured: max 1.1e-6 to 1.3e-6 (fp32 summation order).
    The port's natural selection equals the reference's at every token whose
    6th-7th biased gap exceeds 1e-4 (the G2 fp32 natural row)."""
    import mlx.core as mx

    kolibri_ref = import_sibling("reference.kolibri_ref")
    d = tiny_checkpoint_factory(preset, 1)
    ref = kolibri_ref.KolibriReference(str(d))
    model = ph.load_port(d, float32=True)
    ids = np.array(tc.random_ids(100, seed=62))
    h = ref.embed(ids)
    k = ref.cfg.num_experts_per_tok
    worst = 0.0
    for i, layer in enumerate(model.layers):
        out = ref.layer_branches(i, h)
        top6 = out["info"]["top6"]
        x = mx.array(h)[None]
        fa, swa = model.model.make_masks(x)
        mask = swa if layer.use_sliding else fa
        port = layer.branches(x, mask, force_ids=mx.array(top6.astype(np.uint32))[None])
        for name, got in (("r_attn", port[0]), ("r_moe", port[2])):
            want = out[name]
            e = np.linalg.norm(_np(got[0]) - want, axis=-1) / np.linalg.norm(want, axis=-1)
            worst = max(worst, float(e.max()))
            assert e.max() <= 1e-4, (i, name, float(e.max()))
        natural = np.sort(np.array(layer.branches(x, mask)[5][0]), -1)
        gap = ph.selection_margin(out["info"]["biased"], k)
        differ = np.flatnonzero(np.any(natural != np.sort(top6, -1), -1) & (gap > 1e-4))
        assert differ.size == 0, (i, differ.tolist())
        h = out["h_out"]
    assert worst > 0  # a real comparison, not two copies of one computation


def test_policy_keys_drive_sanitize_cast_and_quant_predicates(tiny_vendor_dir):
    import mlx.core as mx
    import mlx.nn as nn

    kolibri1 = import_sibling("port.kolibri1")
    cfg = json.loads((tiny_vendor_dir / "config.json").read_text())
    stored = {}
    for f in sorted(tiny_vendor_dir.glob("model*.safetensors")):
        stored.update(mx.load(str(f)))

    for keys, head_fp32 in (({}, False),
                            ({"exp036_quantize_embeddings": False, "exp036_quantize_lm_head": False}, True)):
        args = kolibri1.ModelArgs.from_dict(dict(cfg, **keys))
        model = kolibri1.Model(args)
        w = model.sanitize(dict(stored))
        assert w["model.layers.0.mlp.gate.weight"].dtype == mx.float32  # router: exact fp32 upcast
        assert w["model.layers.0.mlp.gate.expert_bias"].dtype == mx.float32
        assert w["lm_head.weight"].dtype == (mx.float32 if head_fp32 else mx.bfloat16)
        assert w["model.embed_tokens.weight"].dtype == mx.bfloat16
        again = model.sanitize(dict(w))  # idempotent
        assert set(again) == set(w) and all(again[k].dtype == w[k].dtype for k in w)
        cast = model.cast_predicate
        assert not cast("model.layers.0.mlp.gate.weight") and not cast("model.layers.0.mlp.gate.expert_bias")
        assert cast("model.layers.0.self_attn.q_proj.weight") and cast("model.embed_tokens.weight")
        assert cast("lm_head.weight") is (not head_fp32)
        pred = model.quant_predicate
        assert pred("lm_head", model.lm_head) is (not head_fp32)
        assert pred("model.embed_tokens", model.model.embed_tokens) is (not head_fp32)
        assert pred("model.layers.0.self_attn.q_proj", model.layers[0].self_attn.q_proj) is True
        assert pred("model.layers.0.mlp.gate", model.layers[0].mlp.gate) is False
        assert pred("model.layers.0.input_layernorm", model.layers[0].input_layernorm) is False
        assert not hasattr(model.layers[0].mlp.gate, "to_quantized")
        assert isinstance(model.lm_head, nn.Linear)
