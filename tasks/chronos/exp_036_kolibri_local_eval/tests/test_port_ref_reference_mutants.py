# SPDX-License-Identifier: MIT
"""reference/mutants.py, the seven G3 reference mutants (BUILD_SPEC 5.2,
`test_reference_mutants.py`; named test_port_ref_* here).

* the registry has exactly the seven keys of HYPOTHESIS G3;
* each mutant is a KolibriReference subclass overriding exactly one method,
  so the frozen reference file is never edited;
* on both tiny presets each mutant changes the logits by more than 1e-2
  (max-abs). Measured: 1.1 to 7.6 against an fp32 noise floor of ~1e-6,
  i.e. every mutant is a gross change on these weights;
* each mutant targets what its name says (a probe per mutant);
* streamed together they equal their separate passes.
"""

from __future__ import annotations

import numpy as np
import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling

EXPECTED = {
    "sigmoid_bias_select", "rope_on_full", "one_plus_w_norm", "renorm_topk",
    "swap_sandwich_norms", "rope_traditional", "qknorm_after_rope",
}  # fmt: skip
HOOKS = {"norm", "rope", "uses_rope", "qk_norm_rope", "route", "layer_norm_names"}


def test_registry_has_exactly_the_seven_mutants():
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    assert set(mutants.MUTANTS) == EXPECTED and len(mutants.MUTANTS) == 7
    for name, cls in mutants.MUTANTS.items():
        assert issubclass(cls, kolibri_ref.KolibriReference), name
        assert cls.__mro__[1] is kolibri_ref.KolibriReference, name
        overridden = {k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")}
        assert len(overridden) == 1 and overridden <= HOOKS, (name, overridden)


@pytest.mark.parametrize("preset", ["vendor", "pattern5"])
def test_each_mutant_changes_the_logits(preset, tiny_checkpoint_factory):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    d = tiny_checkpoint_factory(preset, 0)
    ids = np.array(tc.random_ids(120, seed=40))
    ref = kolibri_ref.KolibriReference(str(d))
    base = ref.forward(ids)
    for name, cls in mutants.MUTANTS.items():
        got = cls(str(d)).forward(ids)
        diff = float(np.abs(got - base).max())
        assert diff > 1e-2, f"{name} moves the logits by only {diff:.2e}"
    # A mutant constructed directly equals the same mutant cloned from a reference.
    for name, cls in mutants.MUTANTS.items():
        np.testing.assert_array_equal(cls(str(d)).forward(ids), ref._clone_as(cls).forward(ids), err_msg=name)


def test_each_mutant_does_what_its_name_says(tiny_pattern5_dir):
    """Probes of the single behaviour each mutant changes."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    M = mutants.MUTANTS
    d = str(tiny_pattern5_dir)
    ref = kolibri_ref.KolibriReference(d)
    rng = np.random.default_rng(0)
    E = ref.cfg.num_experts
    logits = rng.standard_normal((32, E)).astype(np.float32)
    bias = (rng.standard_normal(E) * 2).astype(np.float32)
    k = ref.cfg.num_experts_per_tok

    w, ids = M["sigmoid_bias_select"](d).route(logits, bias, k)
    score = 1 / (1 + np.exp(-logits)) + bias
    np.testing.assert_array_equal(np.sort(ids, -1), np.sort(np.argsort(-score, -1)[:, :k], -1))
    _, ids_ref = ref.route(logits, bias, k)
    assert (np.sort(ids, -1) != np.sort(ids_ref, -1)).any()

    w, _ = M["renorm_topk"](d).route(logits, bias, k)
    np.testing.assert_allclose(w.sum(-1), 1.0, rtol=1e-6)

    full = [i for i, t in enumerate(ref.cfg.layer_types) if t == "full_attention"]
    assert not any(ref.uses_rope(i) for i in full)
    assert all(M["rope_on_full"](d).uses_rope(i) for i in full)

    x = rng.standard_normal((4, 64)).astype(np.float32)
    wn = (1 + 0.1 * rng.standard_normal(64)).astype(np.float32)
    np.testing.assert_allclose(M["one_plus_w_norm"](d).norm(x, wn), kolibri_ref.rms_norm(x, wn + 1, 1e-6), rtol=1e-6)

    names = M["swap_sandwich_norms"](d).layer_norm_names()
    assert names["post_attn"] == "post_attention_layernorm" and names["pre_moe"] == "post_attn_norm"
    assert names["input"] == "input_layernorm" and names["post_ffn"] == "post_ffn_norm"

    q = rng.standard_normal((5, 2, 32)).astype(np.float32)
    pos = np.arange(5)
    trad = M["rope_traditional"](d).rope(q, pos)
    # Interleaved pairs rotate together: |(x0, x1)| is preserved per pair.
    np.testing.assert_allclose(np.hypot(trad[..., 0::2], trad[..., 1::2]), np.hypot(q[..., 0::2], q[..., 1::2]),
                               rtol=1e-5)
    assert np.abs(trad - ref.rope(q, pos)).max() > 1e-2

    sliding = ref.cfg.layer_types.index("sliding_attention")
    kq = rng.standard_normal((5, 2, 32)).astype(np.float32) * 3
    a = ref.qk_norm_rope(sliding, kq, kq, pos)[0]
    b = M["qknorm_after_rope"](d).qk_norm_rope(sliding, kq, kq, pos)[0]
    assert np.abs(a - b).max() > 1e-3
    # On a full (NoPE) layer the order cannot matter.
    np.testing.assert_array_equal(ref.qk_norm_rope(full[0], kq, kq, pos)[0],
                                  M["qknorm_after_rope"](d).qk_norm_rope(full[0], kq, kq, pos)[0])


def test_all_mutants_stream_together(tiny_vendor_dir):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    ref = kolibri_ref.KolibriReference(str(tiny_vendor_dir))
    ids = np.array(tc.random_ids(90, seed=41))
    out = ref.forward_streams(ids, mutants.MUTANTS)
    assert set(out) == set(mutants.MUTANTS) | {"ref", "_lengths"}
    np.testing.assert_array_equal(out["ref"], ref.forward(ids))
    for name, cls in mutants.MUTANTS.items():
        np.testing.assert_array_equal(out[name], cls(str(tiny_vendor_dir)).forward(ids), err_msg=name)
