# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""Router semantics: the vendor's test_routing_semantics, in numpy.

Kolibri routes like torchtitan's TokenChoiceTopKRouter with
router_score_fn=sigmoid_logit_add, route_norm=False, route_scale=1.0
(spec items 11-12):

    ids     = top-k of (logits + expert_bias)        selection on the RAW logits
    weights = sigmoid(logits[ids])                   unbiased, NOT renormalised

The trap is the DeepSeek-V3 / afmoe rule, which selects on
sigmoid(logits) + bias. The vendor test asserts the two selections differ on
its setup; so does this one.

The setup matches the vendor test (64 tokens x 384 experts, logits randn x 3,
bias randn x 5, seed 0) except that the draws come from numpy's generator, not
torch's. Same distribution, different numbers.

Both the MLX port (port.kolibri1.route) and the numpy reference
(reference.kolibri_ref.route_ref) are checked against the closed form here and
against each other. The order of the k selected experts is irrelevant (the
MoE output is a weighted sum), so results are compared after sorting each row
by expert id.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp036_helpers import import_sibling

NUM_TOKENS, NUM_EXPERTS, TOP_K = 64, 384, 6


def vendor_setup(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = (rng.standard_normal((NUM_TOKENS, NUM_EXPERTS)) * 3).astype(np.float32)
    # Magnitudes as observed in a trained Kolibri 1 checkpoint (up to ~20).
    bias = (rng.standard_normal(NUM_EXPERTS) * 5).astype(np.float32)
    return logits, bias


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def expected_route(logits: np.ndarray, bias: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed form (torchtitan router.py), ids in descending score order."""
    choice = logits.astype(np.float32) + bias.astype(np.float32)
    ids = np.argsort(-choice, axis=-1, kind="stable")[..., :k]
    weights = np.take_along_axis(_sigmoid(logits), ids, axis=-1)
    return weights, ids


def canonical(weights, ids) -> tuple[np.ndarray, np.ndarray]:
    """Sort each row by expert id, carrying the weights along."""
    ids = np.asarray(ids).astype(np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    assert ids.shape == weights.shape, (ids.shape, weights.shape)
    order = np.argsort(ids, axis=-1, kind="stable")
    return np.take_along_axis(weights, order, axis=-1), np.take_along_axis(ids, order, axis=-1)


def port_route(logits: np.ndarray, bias: np.ndarray, k: int):
    import mlx.core as mx

    kolibri1 = import_sibling("port.kolibri1")
    w, ids = kolibri1.route(mx.array(logits, dtype=mx.float32), mx.array(bias, dtype=mx.float32), k)
    mx.eval(w, ids)
    # Spec item 12: routing weights are float32.
    assert w.dtype == mx.float32, f"port routing weights are {w.dtype}, expected float32"
    return np.array(w), np.array(ids)


def reference_route(logits: np.ndarray, bias: np.ndarray, k: int):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    w, ids = kolibri_ref.route_ref(logits.copy(), bias.copy(), k)
    return np.asarray(w), np.asarray(ids)


# --- the setup itself --------------------------------------------------------


def test_setup_is_not_vacuous():
    """logit-add and score-add must select different experts on this setup."""
    logits, bias = vendor_setup()
    _, ref_ids = expected_route(logits, bias, TOP_K)
    score_add_ids = np.argsort(-(_sigmoid(logits) + bias), axis=-1, kind="stable")[:, :TOP_K]
    differs = (np.sort(ref_ids, -1) != np.sort(score_add_ids, -1)).any(-1)
    # The vendor test only requires one differing row; on this draw all 64 differ.
    assert differs.sum() >= NUM_TOKENS // 2, (
        f"test is weak: logit-add and score-add differ on only {differs.sum()} of {NUM_TOKENS} rows"
    )
    # The bias must matter too: top-k of the logits alone is a different set.
    no_bias_ids = np.argsort(-logits, axis=-1, kind="stable")[:, :TOP_K]
    assert (np.sort(ref_ids, -1) != np.sort(no_bias_ids, -1)).any(-1).sum() >= NUM_TOKENS // 2
    # No near-ties at the k / k+1 boundary that fp32 rounding could flip.
    z = -np.sort(-(logits + bias), axis=-1)
    assert (z[:, TOP_K - 1] - z[:, TOP_K]).min() > 1e-3
    # Unnormalised: the weights do not sum to one.
    w, _ = expected_route(logits, bias, TOP_K)
    assert np.abs(w.sum(-1) - 1.0).min() > 0.5


# --- implementations against the closed form ---------------------------------


@pytest.mark.parametrize("impl", ["port", "reference"])
def test_routing_semantics(impl):
    logits, bias = vendor_setup()
    route = port_route if impl == "port" else reference_route
    w, ids = route(logits, bias, TOP_K)
    assert np.asarray(ids).shape == (NUM_TOKENS, TOP_K)

    exp_w, exp_ids = canonical(*expected_route(logits, bias, TOP_K))
    got_w, got_ids = canonical(w, ids)
    np.testing.assert_array_equal(got_ids, exp_ids, err_msg=f"{impl}: expert selection mismatch")
    np.testing.assert_allclose(got_w, exp_w, rtol=0, atol=1e-6, err_msg=f"{impl}: routing weights mismatch")

    # Not the score-add rule (DeepSeek-V3 / afmoe), not renormalised.
    score_add_ids = np.sort(np.argsort(-(_sigmoid(logits) + bias), axis=-1)[:, :TOP_K], -1)
    assert not np.array_equal(got_ids, score_add_ids), f"{impl} selects on sigmoid(logits) + bias"
    assert np.abs(got_w.sum(-1) - 1.0).min() > 0.5, f"{impl} renormalises the routing weights"


def test_port_matches_reference():
    """Port and reference agree exactly on ids and to 1e-6 on weights."""
    logits, bias = vendor_setup()
    pw, pids = port_route(logits, bias, TOP_K)
    rw, rids = reference_route(logits, bias, TOP_K)
    pw, pids = canonical(pw, pids)
    rw, rids = canonical(rw, rids)
    np.testing.assert_array_equal(pids, rids)
    np.testing.assert_allclose(pw, rw, rtol=0, atol=1e-6)


def test_port_route_batched_shape():
    """The MoE block calls route on [B, L, E] logits; selection is per token."""
    logits, bias = vendor_setup()
    batched = logits.reshape(2, NUM_TOKENS // 2, NUM_EXPERTS)
    w, ids = port_route(batched, bias, TOP_K)
    assert np.asarray(ids).shape == (2, NUM_TOKENS // 2, TOP_K)
    got_w, got_ids = canonical(np.asarray(w).reshape(NUM_TOKENS, TOP_K), np.asarray(ids).reshape(NUM_TOKENS, TOP_K))
    exp_w, exp_ids = canonical(*expected_route(logits, bias, TOP_K))
    np.testing.assert_array_equal(got_ids, exp_ids)
    np.testing.assert_allclose(got_w, exp_w, rtol=0, atol=1e-6)


@pytest.mark.parametrize("impl", ["port", "reference"])
def test_routing_bf16_bias_upcast_is_exact(impl):
    """expert_bias is BF16 in the checkpoint and float32 at routing time (item 12).

    The upcast is exact, so routing on the BF16-rounded bias in float32 must
    match the closed form evaluated on the same rounded values.
    """
    import mlx.core as mx

    logits, bias = vendor_setup(seed=1)
    bias_bf16 = np.array(mx.array(bias).astype(mx.bfloat16).astype(mx.float32))
    assert not np.array_equal(bias_bf16, bias)  # the rounding is real
    z = -np.sort(-(logits + bias_bf16), axis=-1)
    assert (z[:, TOP_K - 1] - z[:, TOP_K]).min() > 1e-4  # no near-ties on this draw
    route = port_route if impl == "port" else reference_route
    exp_w, exp_ids = canonical(*expected_route(logits, bias_bf16, TOP_K))
    got_w, got_ids = canonical(*route(logits, bias_bf16, TOP_K))
    np.testing.assert_array_equal(got_ids, exp_ids)
    np.testing.assert_allclose(got_w, exp_w, rtol=0, atol=1e-6)
