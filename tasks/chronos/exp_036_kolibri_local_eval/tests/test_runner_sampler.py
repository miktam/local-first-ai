"""runner/sampler.py: the vLLM-order sampler (BUILD_SPEC §4 item 26, §5.4;
HYPOTHESIS C5, C6).

- mask parity with filter_np, the numpy transcription of vLLM v0.29
  apply_top_k_top_p, on 1,000 random vectors, ties included (ties at the
  top-k threshold, heavy-tie rows where the top-p boundary is hit exactly);
- the kept set differs from mlx_lm's make_sampler order on a constructed case;
- chi-square sanity on 20,000 seeded draws;
- the fp32 wrapper returns fp32 and exposes layers / make_cache.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from runner.sampler import filter_np, filter_top_k_top_p, fp32_logits, make_vllm_sampler, top_p_cum_np  # noqa: E402

BOUNDARY_TOL = 1e-5


def _vector(rng, kind: int) -> np.ndarray:
    V = int(rng.integers(5, 400))
    if kind == 0:
        x = rng.normal(size=V) * rng.uniform(0.5, 4.0)
    elif kind == 1:
        x = np.round(rng.normal(size=V) * 2, 1)  # many ties
    elif kind == 2:
        x = rng.integers(-3, 3, size=V).astype(float)  # heavy ties
    else:
        x = rng.normal(size=V) * 8  # peaked
    return x.astype(np.float32)


def test_mask_parity_with_vllm_transcription_on_1000_vectors():
    rng = np.random.default_rng(36)
    exact = boundary = 0
    for t in range(1000):
        x = _vector(rng, t % 4)
        V = x.size
        k = int(rng.choice([1, 2, 5, 20, 64, 128, V, V + 5, 0]))
        p = float(rng.choice([0.1, 0.5, 0.9, 0.95, 0.97, 1.0]))
        ref = filter_np(x, k, p)
        got = np.array(filter_top_k_top_p(mx.array(x), k, p))
        kr, kg = np.isfinite(ref), np.isfinite(got)
        if np.array_equal(kr, kg):
            exact += 1
            np.testing.assert_array_equal(got[kg], x[kg])  # kept values are the inputs, unchanged
            continue
        # Only allowed where the cumulative probability sits on 1 - p within
        # float rounding (summation order differs between numpy and Metal).
        diff = np.flatnonzero(kr != kg)
        cum = top_p_cum_np(x, k)[0]
        assert np.all(np.abs(cum[diff] - (np.float32(1) - np.float32(p))) < BOUNDARY_TOL), (t, k, p, diff)
        boundary += 1
    assert exact + boundary == 1000
    assert boundary <= 20, f"{boundary} boundary-ambiguous vectors"


def test_batched_rows_match_row_by_row():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(6, 300)).astype(np.float32) * 3
    got = np.array(filter_top_k_top_p(mx.array(X), 20, 0.9))
    for i in range(6):
        np.testing.assert_array_equal(np.isfinite(got[i]), np.isfinite(filter_np(X[i], 20, 0.9)))


def test_top_k_keeps_ties_at_the_threshold():
    x = np.array([5.0, 3.0, 3.0, 3.0, 1.0, 0.0], dtype=np.float32)
    got = np.array(filter_top_k_top_p(mx.array(x), 2, 1.0))
    assert np.isfinite(got).tolist() == [True, True, True, True, False, False]
    assert np.array_equal(np.isfinite(got), np.isfinite(filter_np(x, 2, 1.0)))


def test_top_p_runs_on_the_renormalised_top_k_distribution():
    # Two equal top tokens hold 0.6 of the full distribution. vLLM: top-k 2
    # renormalises them to 0.5 / 0.5; top-p 0.5 masks cum <= 0.5, so only the
    # higher-index one survives (stable ascending order, largest always kept).
    logits = np.log(np.array([0.3, 0.3] + [0.1] * 4, dtype=np.float32))
    got = np.isfinite(np.array(filter_top_k_top_p(mx.array(logits), 2, 0.5)))
    assert got.tolist() == [False, True, False, False, False, False]
    assert np.array_equal(got, np.isfinite(filter_np(logits, 2, 0.5)))


def test_kept_set_differs_from_mlx_lm_order():
    """mlx_lm make_sampler: top-p on the full distribution, then top-k (C5)."""
    from mlx_lm.sample_utils import apply_top_k, apply_top_p

    logits = np.log(np.array([0.3, 0.3] + [0.1] * 4, dtype=np.float32))
    lp = mx.array(logits)[None]
    mlx_lm_kept = np.isfinite(np.array(apply_top_k(apply_top_p(lp, 0.5), 2)))[0]
    ours = np.isfinite(np.array(filter_top_k_top_p(lp, 2, 0.5)))[0]
    assert mlx_lm_kept.sum() == 2 and ours.sum() == 1
    assert not np.array_equal(mlx_lm_kept, ours)


def _chi2_crit_999(df: int) -> float:
    """Wilson-Hilferty approximation of the 0.999 chi-square quantile."""
    z = 3.090232
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def test_chi_square_sanity_on_20000_seeded_draws():
    rng = np.random.default_rng(3)
    logits = (rng.normal(size=60) * 1.5).astype(np.float32)
    k, p = 12, 0.9
    masked = filter_np(logits, k, p)
    keep = np.isfinite(masked)
    probs = np.zeros_like(logits, dtype=np.float64)
    e = np.exp(masked[keep].astype(np.float64) - masked[keep].max())
    probs[keep] = e / e.sum()
    sampler = make_vllm_sampler(1.0, p, k)
    mx.random.seed(20261003)
    lp = mx.array(logits - np.log(np.exp(logits.astype(np.float64)).sum()).astype(np.float32))
    draws = np.array(sampler(mx.broadcast_to(lp[None], (20000, logits.size))))
    assert draws.shape == (20000,)
    assert keep[draws].all(), "a draw outside the kept set"
    counts = np.bincount(draws, minlength=logits.size)[keep]
    expected = probs[keep] * 20000
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    df = int(keep.sum()) - 1
    assert df >= 3
    assert chi2 < _chi2_crit_999(df), (chi2, df)


def test_same_seed_same_draws_and_greedy_is_argmax():
    rng = np.random.default_rng(11)
    lp = mx.array(rng.normal(size=(8, 500)).astype(np.float32))
    s = make_vllm_sampler(1.0, 0.97, 128)
    mx.random.seed(5)
    a = np.array(s(lp))
    mx.random.seed(5)
    b = np.array(s(lp))
    assert np.array_equal(a, b)
    g = make_vllm_sampler(0.0, 0.97, 128)
    assert np.array_equal(np.array(g(lp)), np.argmax(np.array(lp), axis=-1))
    assert s.config == {"order": "vllm", "temperature": 1.0, "top_p": 0.97, "top_k": 128}


def test_temperature_scales_before_filtering():
    x = np.array([2.0, 1.0, 0.0, -1.0], dtype=np.float32)
    # At T = 0.5 the top token holds 0.865 of the mass, so top-p 0.8 keeps only it.
    s = make_vllm_sampler(0.5, 0.8, 0)
    mx.random.seed(1)
    draws = np.array(s(mx.broadcast_to(mx.array(x)[None], (200, 4))))
    assert set(draws.tolist()) == {0}
    assert np.isfinite(filter_np(x / 0.5, None, 0.8)).tolist() == [True, False, False, False]


def test_fp32_wrapper_returns_fp32_and_exposes_layers_and_cache():
    import mlx.nn as nn

    class Bf16Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(4, 4)]
            self.emb = nn.Embedding(10, 4)

        def __call__(self, inputs, cache=None):
            return self.emb(inputs).astype(mx.bfloat16)

        def make_cache(self):
            return ["c"]

    m = Bf16Model()
    w = fp32_logits(m)
    out = w(mx.array([[1, 2, 3]]))
    assert out.dtype == mx.float32
    assert w.layers is m.layers
    assert w.make_cache() == ["c"]
    assert fp32_logits(w) is w

    class NoCache(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []

        def __call__(self, inputs, cache=None):
            return inputs.astype(mx.bfloat16)

    assert not hasattr(fp32_logits(NoCache()), "make_cache")
