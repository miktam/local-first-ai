"""exp_036 statistics (Tier 1): bootstrap, t-test, Holm, post-stratification.

BUILD_SPEC §5.8; HYPOTHESIS "Fixed before any run" (Statistics row) and
"Confirmatory hypotheses", Common rules.

Rules implemented here, word for word from HYPOTHESIS:
- Bootstrap B = 10,000 per hypothesis (100,000 near a threshold), numpy PCG64
  seeded with the first 8 bytes of sha256("exp036|" + H_id), big-endian, so
  results do not depend on evaluation order or on which hypotheses ran.
- Percentile intervals.
- Bootstrap p-values: p = (1 + #{theta*_b on the null side of theta0}) / (B + 1);
  theta*_b = theta0 counts on the null side (every null includes equality).
- H1 uses a one-sample t-test (df = n - 1).
- Holm step-down with a running maximum, at family-wise alpha = 0.05, run
  separately for the CONFIRMED family (p) and the REFUTED family (p_rev).
- A hypothesis whose raw p or p_rev lies within a factor of 2 of its Holm
  threshold is recomputed with B = 100,000 before Holm is final.

numpy and the standard library only (no scipy): the Student t distribution is
computed from the regularised incomplete beta function (continued fraction).

Draw order is part of the frozen method: bootstrap replicates are drawn in
chunks whose size depends only on the data shapes (MAX_DRAWS_PER_CHUNK), and
within a chunk strata are visited in sorted key order.
"""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist
from typing import Callable, Mapping, Sequence

import numpy as np

SEED_PREFIX = b"exp036|"
B_DEFAULT = 10_000
B_ESCALATED = 100_000
ESCALATION_FACTOR = 2.0
ALPHA = 0.05

# Bootstrap index draws per chunk (B rows x n items). Fixed: it sets how the
# replicate matrix is cut, so it is part of the frozen draw order.
MAX_DRAWS_PER_CHUNK = 4_000_000

# theta*_b == theta0 counts on the null side. Bootstrap statistics are means of
# rationals computed in floating point, so an exact tie can come out 1 ulp off;
# this tolerance keeps such ties on the null side (conservative).
TIE_EPS = 1e-12


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def seed_of(h_id: str) -> int:
    """First 8 bytes of sha256("exp036|" + h_id), big-endian (HYPOTHESIS, Statistics)."""
    digest = hashlib.sha256(SEED_PREFIX + h_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def rng(h_id: str) -> np.random.Generator:
    """The per-hypothesis generator: PCG64(seed_of(h_id))."""
    return np.random.Generator(np.random.PCG64(seed_of(h_id)))


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------

def _normalised_weights(keys: Sequence[str], weights: Mapping[str, float]) -> list[float]:
    missing = [k for k in keys if k not in weights]
    if missing:
        raise ValueError(f"no post-stratification weight for strata {missing}")
    w = [float(weights[k]) for k in keys]
    if any(x < 0 for x in w) or sum(w) <= 0:
        raise ValueError(f"invalid weights {w}")
    total = sum(w)
    return [x / total for x in w]


def post_stratified_mean(scores_by_category: Mapping[str, Sequence[float]],
                         weights: Mapping[str, float]) -> float:
    """sum_c w_c * mean(scores_c), with w normalised over the categories sampled.

    HYPOTHESIS H2 Measurement: each category's mean is weighted by that
    category's share of the full test split. Every sampled category needs a
    weight; a weighted category with no items is an error (the n_M ladder keeps
    every category present), so a silent re-weighting can never happen.
    Categories may hold different numbers of items: each category's own mean
    is weighted (Amendment 4: the Lite allocation is proportional, not 42 per
    category; every category has at least 5 items at n_M = 154).
    """
    keys = sorted(scores_by_category)
    empty = [k for k in keys if len(scores_by_category[k]) == 0]
    if empty:
        raise ValueError(f"categories with no items: {empty}")
    unsampled = sorted(set(weights) - set(keys))
    if unsampled:
        raise ValueError(f"weighted categories absent from the sample: {unsampled}")
    w = _normalised_weights(keys, weights)
    return float(sum(wk * float(np.mean(np.asarray(scores_by_category[k], dtype=np.float64)))
                     for wk, k in zip(w, keys)))


def two_pass_row_var(x1: Sequence[float], x2: Sequence[float]) -> float:
    """sum (x1 - x2)^2 / (4 n^2): the variance of a two-pass row mean (HYPOTHESIS H2, B10)."""
    a = np.asarray(x1, dtype=np.float64)
    b = np.asarray(x2, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("two_pass_row_var needs two equal-length non-empty vectors")
    n = a.size
    return float(np.sum((a - b) ** 2) / (4.0 * n * n))


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def _chunk_rows(B: int, n_total: int) -> int:
    return max(1, min(B, MAX_DRAWS_PER_CHUNK // max(1, n_total)))


def boot_strata(values_by_stratum: Mapping[str, Sequence], B: int, gen: np.random.Generator,
                weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Stratified item bootstrap of a (weighted) mean.

    values_by_stratum: stratum -> array of shape (n_s,) or (n_s, k). Rows are
    items; with k columns the same resampled rows are used for every column
    (a joint, paired resample across arms).
    weights: stratum -> weight (post-stratification); default n_s / N, which
    makes the statistic the plain mean over all items.
    Returns shape (B,) or (B, k): sum_s w_s * mean(resample of stratum s).
    """
    keys = sorted(values_by_stratum)
    if not keys:
        raise ValueError("boot_strata: no strata")
    vals = [np.asarray(values_by_stratum[k], dtype=np.float64) for k in keys]
    for k, v in zip(keys, vals):
        if v.shape[0] == 0:
            raise ValueError(f"boot_strata: stratum {k!r} is empty")
    tails = {v.shape[1:] for v in vals}
    if len(tails) != 1:
        raise ValueError("boot_strata: strata disagree on the number of columns")
    tail = tails.pop()
    ns = [v.shape[0] for v in vals]
    n_total = sum(ns)
    if weights is None:
        w = [n / n_total for n in ns]
    else:
        w = _normalised_weights(keys, weights)
    out = np.empty((B,) + tail, dtype=np.float64)
    chunk = _chunk_rows(B, n_total)
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        acc = np.zeros((b,) + tail, dtype=np.float64)
        for v, n, wk in zip(vals, ns, w):
            idx = gen.integers(0, n, size=(b, n))
            acc += wk * v[idx].mean(axis=1)
        out[start:start + b] = acc
    return out


def boot_paired(diffs_by_stratum: Mapping[str, Sequence], B: int,
                gen: np.random.Generator) -> np.ndarray:
    """Paired item bootstrap of the mean paired difference, stratified (BUILD_SPEC §5.8).

    diffs_by_stratum: stratum -> per-item differences (already paired). With a
    single stratum this is the plain paired item bootstrap.
    """
    return boot_strata(diffs_by_stratum, B, gen)


def boot_mean_rows(rows: Sequence[Mapping], B: int, gen: np.random.Generator) -> dict:
    """H2: the bootstrap of D-bar = mean_r (ours_r - vendor_r).

    Each row is a mapping with:
      name        row id
      strata      stratum -> per-item scores (one stratum for an unstratified row;
                  categories for the MMLU rows)
      weights     stratum -> post-stratification weight, or None (plain mean)
      vendor      the vendor value (fraction)
      vendor_sd   sd of the vendor's avg@N term, sqrt(p_v (1 - p_v) / (N_v n_v)); 0 if none
      two_pass    optional (estimate, variance): a B10 GPQA row, drawn as
                  estimate + N(0, variance) instead of resampling items

    Draw order: every row's own draws in row order, then every row's vendor
    term in row order. Returns {"mean": (B,), "rows": {name: (B,) of D_r*}}.
    """
    row_draws: list[np.ndarray] = []
    for row in rows:
        tp = row.get("two_pass")
        if tp is not None:
            est, var = float(tp[0]), float(tp[1])
            row_draws.append(est + gen.normal(0.0, math.sqrt(max(var, 0.0)), size=B))
        else:
            row_draws.append(boot_strata(row["strata"], B, gen, weights=row.get("weights")))
    out_rows: dict[str, np.ndarray] = {}
    for row, draws in zip(rows, row_draws):
        sd = float(row.get("vendor_sd") or 0.0)
        vendor = float(row["vendor"])
        if sd > 0:
            vendor_draw = vendor + gen.normal(0.0, sd, size=B)
        else:
            vendor_draw = np.full(B, vendor)
        out_rows[row["name"]] = draws - vendor_draw
    mean = np.mean(np.stack([out_rows[r["name"]] for r in rows], axis=0), axis=0)
    return {"mean": mean, "rows": out_rows}


def blocks_point(per_text_num: Sequence[np.ndarray], per_text_den: Sequence[np.ndarray],
                 text_weights: np.ndarray) -> np.ndarray:
    """Per model: sum_t w_mt * (sum_b num[m,t,b] / sum_b den[m,t,b]). Returns (M,)."""
    w = np.asarray(text_weights, dtype=np.float64)
    out = np.zeros(w.shape[0], dtype=np.float64)
    for t, (num, den) in enumerate(zip(per_text_num, per_text_den)):
        num = np.asarray(num, dtype=np.float64)
        den = np.asarray(den, dtype=np.float64)
        out += w[:, t] * num.sum(axis=1) / den.sum(axis=1)
    return out


def boot_blocks_stratified(per_text_num: Sequence[np.ndarray], per_text_den: Sequence[np.ndarray],
                           text_weights: np.ndarray, B: int, gen: np.random.Generator) -> np.ndarray:
    """H8: stratified block bootstrap, joint across models.

    per_text_num[t]: (M, n_t) per-block KL sums for M models.
    per_text_den[t]: (M, n_t) per-block bytes (or tokens, for the per-token
                     sensitivity); for bytes all rows are equal.
    text_weights:    (M, T) fixed text weights (byte shares), rows summing to 1.
    The n_t blocks of each text are resampled with replacement, with the same
    block indices for every model (joint). Returns (B, M):
    sum_t w_mt * (sum_b num*[m,t,b] / sum_b den*[m,t,b]).
    """
    w = np.asarray(text_weights, dtype=np.float64)
    M = w.shape[0]
    nums = [np.asarray(x, dtype=np.float64) for x in per_text_num]
    dens = [np.asarray(x, dtype=np.float64) for x in per_text_den]
    n_total = sum(x.shape[1] for x in nums)
    out = np.empty((B, M), dtype=np.float64)
    chunk = _chunk_rows(B, n_total * M)
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        acc = np.zeros((b, M), dtype=np.float64)
        for t, (num, den) in enumerate(zip(nums, dens)):
            n = num.shape[1]
            idx = gen.integers(0, n, size=(b, n))
            s_num = num[:, idx].sum(axis=2).T   # (b, M)
            s_den = den[:, idx].sum(axis=2).T
            acc += w[:, t][None, :] * s_num / s_den
        out[start:start + b] = acc
    return out


def boot_docs_ratio(num: np.ndarray, den: np.ndarray, B: int, gen: np.random.Generator) -> np.ndarray:
    """Document bootstrap of sum(num) / sum(den), joint across columns.

    num, den: (n_docs, k). The same documents are resampled for every column.
    Returns (B, k). Used for H5 (token-count ratios on the same documents).
    """
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.ndim == 1:
        num = num[:, None]
    if den.ndim == 1:
        den = den[:, None]
    n = num.shape[0]
    k = max(num.shape[1], den.shape[1])
    out = np.empty((B, k), dtype=np.float64)
    chunk = _chunk_rows(B, n)
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        idx = gen.integers(0, n, size=(b, n))
        out[start:start + b] = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return out


# --------------------------------------------------------------------------
# p-values, intervals, Monte Carlo SE
# --------------------------------------------------------------------------

def p_one_sided(samples: np.ndarray, theta0: float, null_side: str) -> float:
    """(1 + #{theta* on the null side}) / (B + 1); theta* = theta0 is on the null side.

    null_side "le": the null is theta <= theta0. "ge": the null is theta >= theta0.
    """
    s = np.asarray(samples, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("p_one_sided: no samples")
    if null_side == "le":
        count = int(np.sum(s <= theta0 + TIE_EPS))
    elif null_side == "ge":
        count = int(np.sum(s >= theta0 - TIE_EPS))
    else:
        raise ValueError(f"null_side must be 'le' or 'ge', not {null_side!r}")
    return (1 + count) / (s.size + 1)


def mc_se(p: float, B: int) -> float:
    """Monte Carlo SE of a bootstrap p-value: sqrt(p (1 - p) / B)."""
    if B <= 0:
        return 0.0
    return math.sqrt(max(p * (1.0 - p), 0.0) / B)


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval: the (1 - level)/2 and (1 + level)/2 quantiles (linear interpolation)."""
    s = np.asarray(samples, dtype=np.float64).ravel()
    lo, hi = np.quantile(s, [(1.0 - level) / 2.0, (1.0 + level) / 2.0], method="linear")
    return float(lo), float(hi)


def one_sided_bound(samples: np.ndarray, alpha: float, side: str) -> float:
    """Percentile bound: "lower" = the alpha quantile, "upper" = the 1 - alpha quantile."""
    s = np.asarray(samples, dtype=np.float64).ravel()
    q = alpha if side == "lower" else 1.0 - alpha
    if side not in ("lower", "upper"):
        raise ValueError(f"side must be 'lower' or 'upper', not {side!r}")
    return float(np.quantile(s, q, method="linear"))


def z_quantile(q: float) -> float:
    return NormalDist().inv_cdf(q)


def normal_cdf(x: float) -> float:
    return NormalDist().cdf(x)


# --------------------------------------------------------------------------
# Student t (no scipy)
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 20000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            return h
    raise RuntimeError("betacf did not converge")


def betainc_reg(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """CDF of Student's t with df degrees of freedom."""
    if math.isnan(t):
        raise ValueError("t_cdf: nan")
    if t == math.inf:
        return 1.0
    if t == -math.inf:
        return 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc_reg(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_ppf(q: float, df: float) -> float:
    """Quantile of Student's t, by bisection on t_cdf (to ~1e-12)."""
    if not 0.0 < q < 1.0:
        raise ValueError("t_ppf: q must be in (0, 1)")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > q:
        lo *= 2.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


def t_stats(x: Sequence[float], theta0: float) -> dict:
    """mean, sd (ddof 1), se, t, df of a one-sample t-test against theta0."""
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n < 2:
        raise ValueError("t-test needs at least 2 observations")
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1))
    se = sd / math.sqrt(n)
    if se > 0:
        t = (mean - theta0) / se
    elif mean > theta0:
        t = math.inf
    elif mean < theta0:
        t = -math.inf
    else:
        t = math.nan   # zero variance at exactly theta0: no evidence either way
    return {"n": n, "df": n - 1, "mean": mean, "sd": sd, "se": se, "t": t}


def t_test_one_sample(x: Sequence[float], theta0: float, level: float = 0.95):
    """One-sample t-test (H1). Returns (p, p_rev, (lo, hi)).

    p = 1 - F_df(t) for the null mean <= theta0; p_rev = F_df(t) for the reverse
    null mean >= theta0; t = (mean - theta0) / (sd / sqrt(n)), df = n - 1. The
    interval is mean -/+ t_{df,(1+level)/2} * se (HYPOTHESIS H1 Measurement).
    """
    s = t_stats(x, theta0)
    if math.isnan(s["t"]):
        p = p_rev = 1.0
    else:
        F = t_cdf(s["t"], s["df"])
        p, p_rev = 1.0 - F, F
    half = t_ppf((1.0 + level) / 2.0, s["df"]) * s["se"]
    return p, p_rev, (s["mean"] - half, s["mean"] + half)


def t_bound(x: Sequence[float], alpha: float, side: str) -> float:
    """One-sided t bound at level alpha: mean -/+ t_{df,1-alpha} * se."""
    s = t_stats(x, 0.0)
    q = t_ppf(1.0 - alpha, s["df"]) * s["se"]
    return s["mean"] - q if side == "lower" else s["mean"] + q


# --------------------------------------------------------------------------
# Holm
# --------------------------------------------------------------------------

def holm(pvals: Mapping[str, float], alpha: float = ALPHA) -> dict:
    """Holm step-down with running maximum.

    pvals: hypothesis -> raw p; m = len(pvals) (a NOT RUN hypothesis is passed
    with p = 1 so that m stays frozen). Ties are ordered by the mapping's order.
    Returns hypothesis -> {"raw", "adjusted", "step" (1 = smallest p),
    "threshold" (alpha / (m - step + 1)), "reject" (adjusted <= alpha)}.
    """
    keys = list(pvals)
    m = len(keys)
    order = sorted(range(m), key=lambda i: (float(pvals[keys[i]]), i))
    out: dict[str, dict] = {}
    running = 0.0
    for step0, i in enumerate(order):
        k = keys[i]
        p = float(pvals[k])
        running = max(running, min(1.0, (m - step0) * p))
        out[k] = {
            "raw": p,
            "adjusted": running,
            "step": step0 + 1,
            "threshold": alpha / (m - step0),
            "reject": running <= alpha,
        }
    return {k: out[k] for k in keys}


def near_threshold(p: float, threshold: float, factor: float = ESCALATION_FACTOR) -> bool:
    """True if p lies within a factor of `factor` of the threshold (either side)."""
    return threshold / factor <= p <= threshold * factor


def escalate(h_id: str, compute: Callable[[int], dict], threshold: float,
             threshold_rev: float | None = None, current: dict | None = None,
             factor: float = ESCALATION_FACTOR, B: int = B_ESCALATED) -> tuple[dict, bool]:
    """Rerun `compute` with B = 100,000 if p (or p_rev) is within `factor` of its Holm threshold.

    compute(B) -> a result dict with "p" and "p_rev"; it draws from rng(h_id)
    afresh on every call, so the escalated result is reproducible.
    `current` is the B = 10,000 result (computed here if None).
    Returns (result, escalated).
    """
    res = current if current is not None else compute(B_DEFAULT)
    hit = near_threshold(float(res["p"]), threshold, factor)
    if threshold_rev is not None and "p_rev" in res:
        hit = hit or near_threshold(float(res["p_rev"]), threshold_rev, factor)
    if not hit:
        return res, False
    new = compute(B)
    new = dict(new)
    new["escalated_from_B"] = int(res.get("B", B_DEFAULT))
    return new, True
