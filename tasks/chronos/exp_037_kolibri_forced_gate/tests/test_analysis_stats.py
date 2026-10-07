"""analysis/stats.py (BUILD_SPEC §5.8, HYPOTHESIS Statistics and Common rules).

Known-answer cases: seeds, the p-value rule with ties on the null side, the
t-test against hand-computed values, Holm on textbook examples (including a case
where a later raw p clears its own threshold but the adjusted p does not),
post-stratification, the two-pass variance, the bootstraps and escalation.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from analysis import stats


# ---------------------------------------------------------------- seeds

def test_seed_is_first_8_bytes_of_sha256_big_endian():
    # exp_037: the prefix is "exp037|" (exp_036 used "exp036|"); H2 and the protocol control seed the tripwire.
    assert stats.SEED_PREFIX == b"exp037|"
    for h in ("H1", "H2", "H8", "H3|truncation", "H2|protocol"):
        want = int.from_bytes(hashlib.sha256(b"exp037|" + h.encode()).digest()[:8], "big")
        assert stats.seed_of(h) == want
        assert stats.seed_of(h) != int.from_bytes(hashlib.sha256(b"exp036|" + h.encode()).digest()[:8], "big")
    first = stats.rng("H2").integers(0, 2 ** 31, size=4)
    assert np.array_equal(first, np.random.Generator(np.random.PCG64(stats.seed_of("H2"))).integers(0, 2 ** 31, size=4))


def test_seed_per_hypothesis_independent_of_order():
    a = stats.rng("H3").integers(0, 1000, size=20)
    _ = stats.rng("H2").integers(0, 1000, size=500)     # another H drawing first
    b = stats.rng("H3").integers(0, 1000, size=20)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, stats.rng("H4").integers(0, 1000, size=20))


# ---------------------------------------------------------------- p-values

def test_p_one_sided_counts_ties_on_the_null_side():
    s = np.array([-0.04, -0.04, -0.03, -0.02])
    # null theta <= -0.04: two samples on the null side, both ties
    assert stats.p_one_sided(s, -0.04, "le") == pytest.approx((1 + 2) / (4 + 1))
    # reverse null theta >= -0.04: all four
    assert stats.p_one_sided(s, -0.04, "ge") == pytest.approx((1 + 4) / (4 + 1))
    # float noise on a tie still counts on the null side
    assert stats.p_one_sided(np.array([-0.04 + 1e-15]), -0.04, "le") == pytest.approx(1.0)


def test_p_one_sided_bounds():
    s = np.zeros(9999)
    assert stats.p_one_sided(s, -1.0, "le") == pytest.approx(1 / 10000)
    assert stats.p_one_sided(s, 1.0, "le") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        stats.p_one_sided(s, 0.0, "lt")


def test_mc_se():
    assert stats.mc_se(0.01, 10_000) == pytest.approx(math.sqrt(0.01 * 0.99 / 10_000))
    assert stats.mc_se(0.5, 0) == 0.0


def test_percentile_ci_and_bounds():
    s = np.arange(1001, dtype=float)
    assert stats.percentile_ci(s, 0.95) == pytest.approx((25.0, 975.0))
    assert stats.one_sided_bound(s, 0.05, "lower") == pytest.approx(50.0)
    assert stats.one_sided_bound(s, 0.05, "upper") == pytest.approx(950.0)


# ---------------------------------------------------------------- t distribution

def test_t_quantiles_known_values():
    assert stats.t_ppf(0.975, 9) == pytest.approx(2.2621571628, abs=1e-8)
    assert stats.t_ppf(0.95, 9) == pytest.approx(1.8331129327, abs=1e-8)
    assert stats.t_ppf(0.975, 1) == pytest.approx(12.7062047362, abs=1e-6)
    assert stats.t_cdf(0.0, 5) == pytest.approx(0.5)


def test_t_cdf_matches_closed_form_df3():
    # F_3(t) = 1/2 + (1/pi) [ (t/sqrt3) / (1 + t^2/3) + atan(t/sqrt3) ]
    for t in (-4.0, -1.3, 0.2, 1.0, 3.87):
        u = t / math.sqrt(3)
        want = 0.5 + (u / (1 + t * t / 3) + math.atan(u)) / math.pi
        assert stats.t_cdf(t, 3) == pytest.approx(want, abs=1e-12)


def test_t_test_hand_computed():
    x = [0.1, 0.2, 0.3, 0.4]                      # mean 0.25, sd 0.129099, se 0.0645497
    p, p_rev, (lo, hi) = stats.t_test_one_sample(x, 0.0)
    t = 0.25 / (math.sqrt(sum((v - 0.25) ** 2 for v in x) / 3) / 2)
    assert t == pytest.approx(3.872983346, rel=1e-9)
    u = t / math.sqrt(3)
    F = 0.5 + (u / (1 + t * t / 3) + math.atan(u)) / math.pi
    assert p == pytest.approx(1 - F, abs=1e-12)
    assert p_rev == pytest.approx(F, abs=1e-12)
    half = 3.182446305 * 0.0645497224
    assert lo == pytest.approx(0.25 - half, abs=1e-8)
    assert hi == pytest.approx(0.25 + half, abs=1e-8)


def test_t_test_zero_variance():
    p, p_rev, _ = stats.t_test_one_sample([1.0, 1.0, 1.0], 0.0)
    assert (p, p_rev) == (0.0, 1.0)
    p, p_rev, _ = stats.t_test_one_sample([0.0, 0.0], 0.0)
    assert (p, p_rev) == (1.0, 1.0)


def test_t_bound():
    x = [0.1, 0.2, 0.3, 0.4]
    lo = stats.t_bound(x, 0.025, "lower")
    _, _, (lo95, hi95) = stats.t_test_one_sample(x, 0.0)
    assert lo == pytest.approx(lo95, abs=1e-9)
    assert stats.t_bound(x, 0.025, "upper") == pytest.approx(hi95, abs=1e-9)


# ---------------------------------------------------------------- Holm

def test_holm_textbook():
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert [h[k]["step"] for k in "dacb"] == [1, 2, 3, 4]
    assert h["d"]["adjusted"] == pytest.approx(0.02)
    assert h["a"]["adjusted"] == pytest.approx(0.03)
    assert h["c"]["adjusted"] == pytest.approx(0.06)
    assert h["b"]["adjusted"] == pytest.approx(0.06)     # running maximum
    assert [h[k]["reject"] for k in "dacb"] == [True, True, False, False]
    assert h["b"]["threshold"] == pytest.approx(0.05)
    assert h["d"]["threshold"] == pytest.approx(0.0125)


def test_holm_later_bound_clears_its_threshold_but_adjusted_does_not():
    # sorted: 0.001 (x8), 0.0072 (x7 = 0.0504 > 0.05: stop), 0.0075 (x6 = 0.045 <= 0.05, but adjusted 0.0504)
    p = {"H1": 0.001, "H2": 0.0072, "H3": 0.0075, "H4": 0.5, "H5": 0.6, "H6": 0.7, "H7": 0.8, "H8": 0.9}
    h = stats.holm(p)
    assert h["H3"]["raw"] <= h["H3"]["threshold"]                     # 0.0075 <= 0.05/6
    assert h["H3"]["adjusted"] == pytest.approx(0.0504)
    assert not h["H3"]["reject"]
    assert h["H1"]["reject"] and not h["H2"]["reject"]


def test_holm_ties_follow_mapping_order_and_m_counts_not_run():
    h = stats.holm({"H1": 0.004, "H2": 1.0, "H3": 1.0, "H4": 1.0, "H5": 1.0, "H6": 1.0, "H7": 1.0, "H8": 1.0})
    assert h["H1"]["adjusted"] == pytest.approx(8 * 0.004)          # m = 8 even with 7 at p = 1
    t = stats.holm({"x": 0.01, "y": 0.01})
    assert t["x"]["step"] == 1 and t["y"]["step"] == 2
    assert t["x"]["adjusted"] == t["y"]["adjusted"] == pytest.approx(0.02)


def test_holm_caps_at_one():
    h = stats.holm({"a": 0.6, "b": 0.7})
    assert h["a"]["adjusted"] == 1.0 and h["b"]["adjusted"] == 1.0


# ---------------------------------------------------------------- point estimates

def test_post_stratified_mean():
    scores = {"a": [1, 1, 0, 0], "b": [1, 1, 1, 1], "c": [0, 0, 0, 0]}
    w = {"a": 2, "b": 1, "c": 1}
    assert stats.post_stratified_mean(scores, w) == pytest.approx((2 * 0.5 + 1 * 1 + 1 * 0) / 4)
    with pytest.raises(ValueError):
        stats.post_stratified_mean(scores, {"a": 1, "b": 1})            # c has no weight
    with pytest.raises(ValueError):
        stats.post_stratified_mean({"a": [1]}, {"a": 1, "b": 1})        # b absent from the sample


def test_post_stratified_mean_with_unequal_n_per_category():
    """Amendment 4: categories hold different numbers of items (proportional, not 42 each). Each category's own mean
    is weighted: by hand, 0.5 * 1/3 + 0.3 * 4/5 + 0.2 * 1/2 = 1/6 + 6/25 + 1/10 = 76/150, not the item mean 6/10."""
    scores = {"a": [1, 0, 0], "b": [1, 1, 1, 1, 0], "c": [0, 1]}
    w = {"a": 50, "b": 30, "c": 20}
    assert stats.post_stratified_mean(scores, w) == pytest.approx(76 / 150, abs=1e-15)
    assert stats.post_stratified_mean(scores, w) != pytest.approx(6 / 10)
    # the stratified bootstrap weights strata, not items: constant strata give the weighted mean in every replicate
    const = {"a": [1.0, 1.0, 1.0], "b": [0.0] * 5, "c": [1.0, 1.0]}
    draws = stats.boot_strata(const, 200, stats.rng("test|unequal"), weights=w)
    assert np.allclose(draws, 0.5 * 1 + 0.3 * 0 + 0.2 * 1)


def test_two_pass_row_var():
    x1 = [1, 0, 1, 1]
    x2 = [1, 1, 0, 1]
    assert stats.two_pass_row_var(x1, x2) == pytest.approx(2 / (4 * 16))
    with pytest.raises(ValueError):
        stats.two_pass_row_var([1], [1, 0])


# ---------------------------------------------------------------- bootstraps

def test_boot_strata_deterministic_and_centred():
    v = np.array([1.0] * 200 + [0.0] * 200)
    a = stats.boot_strata({"s": v}, 5000, stats.rng("T"))
    b = stats.boot_strata({"s": v}, 5000, stats.rng("T"))
    assert np.array_equal(a, b)
    assert a.mean() == pytest.approx(0.5, abs=0.002)
    assert a.std() == pytest.approx(0.025, rel=0.05)                 # sqrt(.25/400)


def test_boot_strata_weights_and_joint_columns():
    st = {"x": np.array([[1.0, 0.0]] * 10), "y": np.array([[0.0, 1.0]] * 30)}
    out = stats.boot_strata(st, 100, stats.rng("T"))
    assert out.shape == (100, 2)
    assert np.allclose(out[:, 0], 0.25) and np.allclose(out[:, 1], 0.75)   # n_s / N weights
    outw = stats.boot_strata(st, 100, stats.rng("T"), weights={"x": 1, "y": 1})
    assert np.allclose(outw[:, 0], 0.5)
    # constant strata: zero variance; a paired difference of identical columns is exactly 0
    d = stats.boot_paired({"all": np.zeros(50)}, 200, stats.rng("T"))
    assert np.all(d == 0.0)


def test_boot_strata_chunking_is_stable():
    v = np.arange(1000, dtype=float) % 7
    old = stats.MAX_DRAWS_PER_CHUNK
    try:
        a = stats.boot_strata({"s": v}, 3000, stats.rng("T"))
        stats.MAX_DRAWS_PER_CHUNK = 1000 * 7       # a different chunking of the same stream
        b = stats.boot_strata({"s": v}, 3000, stats.rng("T"))
    finally:
        stats.MAX_DRAWS_PER_CHUNK = old
    assert np.array_equal(a, b)


def test_boot_mean_rows_vendor_term_and_two_pass():
    rows = [
        {"name": "r1", "strata": {"all": np.ones(100)}, "weights": None, "vendor": 0.9, "vendor_sd": 0.0},
        {"name": "r2", "two_pass": (0.7, 0.0001), "vendor": 0.8, "vendor_sd": 0.01},
    ]
    out = stats.boot_mean_rows(rows, 20000, stats.rng("H2"))
    assert np.allclose(out["rows"]["r1"], 0.1)
    assert out["rows"]["r2"].mean() == pytest.approx(-0.1, abs=0.001)
    assert out["rows"]["r2"].std() == pytest.approx(math.sqrt(0.0001 + 0.0001), rel=0.03)
    assert out["mean"].mean() == pytest.approx(0.0, abs=0.001)


def test_blocks_point_and_bootstrap():
    num = [np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([[3.0, 3.0], [3.0, 3.0]])]
    den = [np.array([[10.0, 10.0], [10.0, 10.0]]), np.array([[30.0, 30.0], [30.0, 30.0]])]
    w = np.array([[0.25, 0.75], [0.25, 0.75]])
    pt = stats.blocks_point(num, den, w)
    assert pt == pytest.approx([0.25 * 0.1 + 0.75 * 0.1, 0.25 * 0.2 + 0.75 * 0.1])
    bs = stats.boot_blocks_stratified(num, den, w, 50, stats.rng("H8"))
    assert bs.shape == (50, 2) and np.allclose(bs, pt)            # constant blocks


def test_boot_docs_ratio():
    num = np.array([[2.0], [4.0], [6.0]])
    den = np.array([[1.0], [2.0], [3.0]])
    out = stats.boot_docs_ratio(num, den, 100, stats.rng("H5"))
    assert np.allclose(out, 2.0)


# ---------------------------------------------------------------- escalation

def test_near_threshold():
    assert stats.near_threshold(0.006, 0.00625)
    assert stats.near_threshold(0.0125, 0.00625)
    assert not stats.near_threshold(0.0126, 0.00625)
    assert not stats.near_threshold(0.003, 0.00625)


def test_escalate_reruns_with_100k_only_near_the_threshold():
    calls = []

    def compute(B):
        calls.append(B)
        return {"p": 0.006, "p_rev": 0.9, "B": B}

    res, esc = stats.escalate("H4", compute, threshold=0.05 / 8, threshold_rev=0.05 / 8)
    assert esc and res["B"] == 100_000 and res["escalated_from_B"] == 10_000
    assert calls == [10_000, 100_000]
    calls.clear()

    def far(B):
        calls.append(B)
        return {"p": 0.0001, "p_rev": 0.9, "B": B}

    res, esc = stats.escalate("H4", far, threshold=0.05 / 8, threshold_rev=0.05)
    assert not esc and calls == [10_000]
