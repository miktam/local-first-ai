# SPDX-License-Identifier: MIT
"""The MLX port against the independent numpy reference, on tiny checkpoints.

Both tiny presets ("vendor": 6 layers, window 65, 8 experts top-2;
"pattern5": the real 4-sliding/1-full layout, window 17, 16 experts top-6)
at two weight seeds, on a 160-token random sequence, which is longer than
either window, so every sliding layer actually drops keys.

(a) fp32 port vs fp32 reference: logits agree to 1e-4 relative and the
    argmax is identical at every position. This is the semantic gate: any
    deviation from the spec (window off by one, RoPE layout, norm formula,
    routing rule, residual order) moves the logits by 0.2-1.0 relative, see
    the mutation checks in the reference and fixtures reports.
(b) bf16 port (as loaded) vs fp32 reference, in two parts:
    (b1) forced routing: precision of the bf16 arithmetic itself;
    (b2) natural routing: the end-to-end bf16 path, smoke bounds.
(c) fp32 hidden states after every layer, and the selected experts of every
    layer, port vs reference.

Why (b) is split, and where its thresholds come from: INTEGRATION_LOG.md,
entry 1, and the docstrings of the two tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling

PRESETS = ("vendor", "pattern5")
SEEDS = (0, 1)
SEQ_LEN = 160


@pytest.fixture(
    scope="module",
    params=[(p, s) for p in PRESETS for s in SEEDS],
    ids=lambda c: f"{c[0]}-seed{c[1]}",
)
def case(request, tiny_checkpoint_factory):
    """One checkpoint, one random sequence, and the reference's results."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    import_sibling("port.kolibri1")  # skip while the port does not exist
    preset, seed = request.param
    model_dir = tiny_checkpoint_factory(preset, seed)
    cfg = tc.preset_config(preset)
    assert SEQ_LEN > cfg["sliding_window"], "the sequence must be longer than the window"
    ids = tc.random_ids(SEQ_LEN, seed=1000 + seed)

    with pytest.MonkeyPatch.context() as mp:
        scores = ph.record_reference_scores(mp, kolibri_ref)
        ref = kolibri_ref.KolibriReference(str(model_dir))
        logits, layers, final = ref.forward(np.array(ids), return_hidden=True)
    k = cfg["num_experts_per_tok"]
    return SimpleNamespace(
        preset=preset,
        seed=seed,
        dir=model_dir,
        ids=ids,
        cfg=cfg,
        logits=logits,
        layers=layers,
        routing_ids=[i for _, i in ref.last_routing],
        margins=[ph.selection_margin(s, k) for s in scores],
    )


def test_fp32_logits_match_reference(case):
    """(a) fp32 port vs fp32 reference: max|d| / max|ref| <= 1e-4, same argmax
    everywhere. Measured: 1.3e-6 to 1.8e-6 (summation order only)."""
    model = ph.load_port(case.dir, float32=True)
    got = ph.port_logits(model, case.ids)
    assert got.shape == case.logits.shape
    stats = ph.compare(case.logits, got)
    assert stats["rel_max"] <= 1e-4, ph.fmt(stats)
    mismatch = np.flatnonzero(got.argmax(-1) != case.logits.argmax(-1))
    assert mismatch.size == 0, f"argmax differs at positions {mismatch.tolist()}; {ph.fmt(stats)}"


def test_bf16_arithmetic_vs_reference_forced_routing(case, monkeypatch):
    """(b1) bf16 port, expert ids forced to the reference's, vs fp32 reference.

    Forcing the ids removes routing flips (see b2), so what is left is bf16
    rounding of activations: every port op given the same bf16 input is within
    about one unit roundoff u = 2^-8 of its fp32 result (INTEGRATION_LOG entry 1).

    * KL(ref || port) mean <= 1e-3 (the original bound, unchanged). A logit
      error delta costs about delta^2 / 2 nats. Typical |delta| is 0.01-0.04
      (logit std 0.8), so the expected KL is 5e-5 to 8e-4. Measured: 5.7e-5
      to 1.1e-4.
    * max|d| / max|ref| <= 5e-2. Rounding errors of relative size <= u
      accumulate like a random walk: sqrt(n) * u for n sequential roundings.
      There are about 16 per layer (two sublayers of norm, projection,
      qk-norm, RoPE, attention, output projection, post-norm, residual add)
      and 10 layers, so n ~ 160 and sqrt(160) * 2^-8 = 4.9e-2. Measured: 1.6e-2
      to 2.3e-2.
    * Top-1 agreement >= 0.98 on decisive positions (reference lead >=
      DECISIVE_GAP = 0.1). The original ">= 0.98 everywhere" cannot hold: these
      random-weight logits are flat (std 0.8 over 1024 ids), and 6-11% of
      positions have a lead below 0.02, which is the size of the bf16 error
      itself. Measured: 1.000 on 99-111 decisive positions and 0.92-0.99
      everywhere; every disagreement is at a lead of 0.031 or less.
    """
    model = ph.load_port(case.dir)  # bf16 as loaded; expert_bias and lm_head fp32
    ph.force_routing(monkeypatch, model, case.routing_ids)
    got = ph.port_logits(model, case.ids)
    stats = ph.compare(case.logits, got)
    assert stats["kl_mean"] <= 1e-3, ph.fmt(stats)
    assert stats["rel_max"] <= 5e-2, ph.fmt(stats)
    assert stats["top1_decisive"] >= 0.98, ph.fmt(stats)


def test_bf16_logits_vs_reference_natural_routing(case):
    """(b2) bf16 port as it runs in the experiment (own routing) vs fp32 reference.

    The original bounds (top-1 >= 0.98, mean KL <= 1e-3) assume well-separated
    logits and routing. Tiny random weights have neither, so they would fail a
    correct bf16 implementation. An independent numpy bf16 emulation of the
    reference, with no MLX code, scores top-1 0.86-0.98 and KL 2e-4 to 3.9e-3
    against the fp32 reference (INTEGRATION_LOG entry 1). The bounds below follow
    from the bf16 error model:

    * The k-th and (k+1)-th biased router scores are near-tied for some tokens
      (1st percentile of the margin: 0.002-0.04). Router-logit noise from the
      bf16 residual stream is about 1-2% of the router logit scale (1.6), or
      0.02-0.03, so those tokens flip. A flip swaps one expert of k at similar
      weight. After post_ffn_norm that moves the token's residual by about
      sqrt(2/(k+1)) of a sublayer's RMS, which is 10-25% of the residual. Its
      logits then move by delta ~ 0.1-0.2 and the token pays KL ~ delta^2 / 2,
      about 0.005-0.02. Measured on flipped positions: a mean of 0.014-0.027
      and a maximum of 0.12. Changed tokens also perturb later tokens through
      attention, and on pattern5 (16 experts, top-6, 10 layers) 40-52% of
      tokens flip in at least one layer.
    * Both bounds are worst cases in which every position is flip-affected.
      The same bounds serve the cached bf16 path and the 8-bit checkpoint,
      whose noise is larger.
    * Mean KL <= 3e-2: every position paying the flipped-token KL of <= 0.03.
      Measured: 1.6e-3 to 7.5e-3.
    * Top-1 >= 0.60: give every position a logit change of
      delta_rms = sqrt(2 * 0.03) = 0.24. It then loses its argmax with
      probability Q(lead / (sqrt(2) * delta_rms)). Averaged over the
      reference's own leads, that is a 25-29% expected loss, including the
      coin flips at tiny leads. Three binomial standard deviations over 160
      positions (3 * 0.036) give 0.60. Measured: 0.81-0.94.

    For contrast, a gross routing bug (expert_bias dropped) scores top-1
    0.18-0.24 and mean KL 0.13-0.18 on the same checkpoints, so these bounds
    still separate a broken bf16 path from a correct one. The real-weight
    gate is a separate step and does not use these numbers.
    """
    model = ph.load_port(case.dir)
    got = ph.port_logits(model, case.ids)
    stats = ph.compare(case.logits, got)
    assert np.isfinite(got).all()
    assert stats["kl_mean"] <= 3e-2, ph.fmt(stats)
    assert stats["top1"] >= 0.60, ph.fmt(stats)


def test_router_logits_are_fp32_of_bf16_operands(tiny_vendor_dir):
    """Spec item 11: in the bf16 model the router weight is stored as an exact
    fp32 upcast of the bf16 checkpoint tensor (BUILD_SPEC 5.1 delta 4; it was
    bf16 with a per-call cast in exp036-port-1), and the logits are
    x.f32 @ W.f32^T, fp32 and never rounded to bf16.

    The end-to-end bf16 tests cannot see this deviation. Rounding the logits
    to bf16 changes them by about u * |logit| ~ 6e-3, below the
    routing-flip noise of the bf16 path (INTEGRATION_LOG entry 1).
    test_routing only feeds route() fp32 logits that it builds itself. So the
    module is checked directly: a float64 product of the same bf16 operands
    must match it to fp32 accumulation error, about 1e-7 relative over 256
    products. The bound is 1e-5. Rounding to bf16 would be off by more than
    1e-3 relative."""
    import mlx.core as mx

    import_sibling("port.kolibri1")
    model = ph.load_port(tiny_vendor_dir)  # bf16 as loaded
    gate = model.layers[0].mlp.gate
    assert gate.weight.dtype == mx.float32
    # Exactly the bf16 checkpoint values, read with mlx's plain loader.
    stored = tc.load_checkpoint_f32(tiny_vendor_dir)["model.layers.0.mlp.gate.weight"]
    np.testing.assert_array_equal(np.array(gate.weight), stored)
    assert gate.expert_bias.dtype == mx.float32
    assert model.layers[0].self_attn.q_proj.weight.dtype == mx.bfloat16  # the trunk is bf16
    hidden = tc.preset_config("vendor")["hidden_size"]
    x = mx.array(np.random.default_rng(0).standard_normal((1, 64, hidden)).astype(np.float32)).astype(mx.bfloat16)
    got = gate(x)
    assert got.dtype == mx.float32
    exact = np.array(x.astype(mx.float32))[0].astype(np.float64) @ np.array(
        gate.weight.astype(mx.float32)
    ).astype(np.float64).T
    err = np.abs(np.array(got[0]).astype(np.float64) - exact).max() / np.abs(exact).max()
    bf16_err = np.abs(
        np.array(mx.array(exact.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)) - exact
    ).max() / np.abs(exact).max()
    assert err <= 1e-5, f"router logits off by {err:.2e} relative (bf16 rounding would be {bf16_err:.2e})"
    assert bf16_err > 1e-3  # the check above can see bf16 rounding


def test_fp32_hidden_states_and_routing_per_layer(case, monkeypatch):
    """(c) fp32 port, layer by layer, vs the reference.

    Each layer's output (the residual stream after the layer) agrees to
    max|d| / max|ref| <= 1e-4 (measured: <= 2.0e-6). Each layer selects the
    same expert set for every token whose reference selection margin exceeds
    1e-3. The guard is about 200x the fp32 router-logit noise (~1e-6 relative
    on logits up to ~5). Below it, summation order may legitimately decide on
    other hardware. At these seeds margins go down to 1.2e-4, and even those
    tokens select identically.
    """
    model = ph.load_port(case.dir, float32=True)
    hidden = ph.record_layer_outputs(monkeypatch, model)
    routing = ph.record_routing(monkeypatch, model)
    ph.port_logits(model, case.ids)

    n_layers = case.cfg["num_hidden_layers"]
    assert len(hidden) == n_layers and len(routing) == n_layers
    worst = []
    for i in range(n_layers):
        ref_h = case.layers[i]
        rel = float(np.abs(hidden[i] - ref_h).max() / np.abs(ref_h).max())
        worst.append(rel)
        assert rel <= 1e-4, f"layer {i} ({case.cfg['layer_types'][i]}): hidden rel {rel:.2e}"

        ref_sel = np.sort(case.routing_ids[i], axis=-1)
        clear = case.margins[i] > 1e-3
        differ = np.flatnonzero(np.any(routing[i] != ref_sel, axis=-1) & clear)
        assert differ.size == 0, (
            f"layer {i}: expert sets differ at tokens {differ.tolist()} "
            f"(margins {case.margins[i][differ].round(4).tolist()})"
        )
    assert max(worst) <= 1e-4
