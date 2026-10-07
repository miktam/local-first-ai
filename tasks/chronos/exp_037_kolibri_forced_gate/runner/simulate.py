"""Continuous-batching projection (HYPOTHESIS "Pilot …", rule 4; BUILD_SPEC
§5.4 runner/simulate.py; plan_rules.json "projection").

Step model, fitted per arm by least squares (coefficients constrained to be
non-negative) on the decode steps of the last third of that arm's pilot wall
time:

    step_seconds = a + b · n_live + c · n_live · padded_len

simulate_cell() then replays one cell: B slots, items admitted in order as
slots free, each admission paying its prompt tokens at the measured prefill
rate (prefill blocks decoding, as in mlx_lm's BatchGenerator), every decode
step advancing every live sequence by one token, and the cell's tail (the
last sequences decoding with fewer than B live) included. padded_len is the
longest live sequence (prompt + generated): BatchKVCache left-pads every row
to it and trims when the longest leaves.

Between two events (a sequence finishing, an admission) n_live is constant
and padded_len grows by one per step, so the time of m steps has the closed
form m·(a + b·n + c·n·L0) + c·n·m(m+1)/2; the replay is event-driven and
exact for the model, and deterministic for a given seed.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def select_last_third(steps: Sequence[dict], t_start: float | None = None, t_end: float | None = None) -> list[dict]:
    """Decode steps in the last third of the arm's pilot wall time.

    Steps carry "t" (epoch seconds at the end of the step). Steps that also
    processed prompt tokens (admissions) are excluded: their time is mostly
    prefill, which is projected separately at the prefill rate."""
    pure = [s for s in steps if not s.get("prompt_tokens") and s.get("n_live", 0) > 0]
    if not pure:
        return []
    ts = [float(s["t"]) for s in steps if "t" in s]
    t0 = float(t_start) if t_start is not None else min(ts)
    t1 = float(t_end) if t_end is not None else max(ts)
    cut = t0 + (t1 - t0) * 2.0 / 3.0
    return [s for s in pure if cut <= float(s["t"]) <= t1]


def fit_step_model(steps: Iterable[dict]) -> tuple[float, float, float]:
    """Least-squares (a, b, c) for step_seconds = a + b·n + c·n·padded_len, with a, b, c >= 0.

    A decode step cannot get faster as more or longer sequences are live, so the fit is constrained to
    non-negative coefficients (exact active-set search over the 7 column subsets). Whenever the unconstrained
    least-squares solution is already non-negative it is returned unchanged; a noisy last third with little
    spread in padded_len can otherwise give c < 0, and an extrapolation to longer completions then predicts
    non-positive step times and makes every plan unprojectable (integration finding, BUILD_LOG.md)."""
    rows = [(1.0, float(s["n_live"]), float(s["n_live"]) * float(s["padded_len"]), float(s["step_seconds"])) for s in steps]
    if len(rows) < 3:
        raise ValueError(f"need at least 3 decode steps to fit the step model, got {len(rows)}")
    A = np.array([r[:3] for r in rows], dtype=np.float64)
    y = np.array([r[3] for r in rows], dtype=np.float64)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    if np.all(coef >= 0):
        a, b, c = (float(v) for v in coef)
        return a, b, c
    best, best_rss = None, math.inf
    for cols in ((0, 1, 2), (0, 1), (0, 2), (1, 2), (0,), (1,), (2,)):
        sub, *_ = np.linalg.lstsq(A[:, cols], y, rcond=None)
        if np.any(sub < 0):
            continue
        full = np.zeros(3)
        full[list(cols)] = sub
        rss = float(np.sum((A @ full - y) ** 2))
        if rss < best_rss - 1e-18:
            best, best_rss = full, rss
    if best is None:  # every subset fit is negative: step times are not positive at all
        raise ValueError("no non-negative step model fits the decode steps")
    a, b, c = (float(v) for v in best)
    return a, b, c


def step_time(model: tuple[float, float, float], n_live: int, padded_len: float) -> float:
    a, b, c = model
    return a + b * n_live + c * n_live * padded_len


def resample(values: Sequence[float], n: int, rng: np.random.Generator) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("cannot resample from an empty pilot distribution")
    return v[rng.integers(0, v.size, size=n)]


def simulate_cell(
    n_items: int,
    lengths: Sequence[float],
    prompt_lengths: Sequence[float],
    B: int,
    step_model: tuple[float, float, float],
    prefill_rate: float,
    seed: int = 36,
    *,
    scale: float = 1.0,
    cap: int | None = None,
) -> float:
    """Projected wall hours of one cell (before the 1.15 margin).

    lengths: the pilot completion lengths of the source (arm, task), already
    adjusted for the cap rule (items truncated at the initial cap counted at
    the post-raise cap); they are resampled with replacement (seed) and
    multiplied by `scale` ((mean + 1 SE)/mean, the GPQA-D factor, an effort
    ratio), rounded up and clipped to `cap`. prompt_lengths are resampled with
    the same draw when they pair with lengths, else independently."""
    if n_items <= 0:
        return 0.0
    if B < 1:
        raise ValueError("B must be >= 1")
    if prefill_rate <= 0:
        raise ValueError("prefill_rate must be > 0")
    rng = np.random.default_rng(seed)
    lens = np.asarray(lengths, dtype=np.float64)
    if lens.size == 0:
        raise ValueError("no pilot lengths")
    idx = rng.integers(0, lens.size, size=n_items)
    L = np.ceil(lens[idx] * float(scale))
    if cap is not None:
        L = np.minimum(L, float(cap))
    L = np.maximum(L, 1.0)
    pl = np.asarray(prompt_lengths, dtype=np.float64)
    if pl.size == lens.size:
        P = pl[idx]
    else:
        P = resample(pl, n_items, rng)

    a, b, c = step_model
    t = 0.0
    live: list[list[float]] = []  # [remaining tokens, current length]
    nxt = 0
    while nxt < n_items or live:
        admitted = 0.0
        while len(live) < B and nxt < n_items:
            admitted += float(P[nxt])
            live.append([float(L[nxt]), float(P[nxt])])
            nxt += 1
        if admitted:
            t += admitted / prefill_rate
        m = min(e[0] for e in live)
        n = len(live)
        L0 = max(e[1] for e in live)
        inc = m * (a + b * n + c * n * L0) + c * n * m * (m + 1) / 2.0
        if not inc > 0:
            raise ValueError(f"step model {step_model} gives a non-positive time for {m} steps at n_live {n}")
        t += float(inc)
        for e in live:
            e[0] -= m
            e[1] += m
        live = [e for e in live if e[0] > 0]
    return float(t) / 3600.0


def length_scale(values: Sequence[float]) -> float:
    """(mean + 1 SE) / mean of a pilot length sample (plan rule 4)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or v.mean() <= 0:
        return 1.0
    se = v.std(ddof=1) / math.sqrt(v.size) if v.size > 1 else 0.0
    return float((v.mean() + se) / v.mean())
