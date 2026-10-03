"""runner/simulate.py: the step-model fit and the continuous-batching projection
(HYPOTHESIS rule 4; BUILD_SPEC §5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from runner import simulate


def _trace(a, b, c, n=300, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        nl = int(rng.integers(1, 17))
        pl = int(rng.integers(50, 30_000))
        s = a + b * nl + c * nl * pl + (rng.normal() * noise if noise else 0.0)
        out.append({"type": "step", "t": 1000.0 + i, "n_live": nl, "padded_len": pl, "step_seconds": s,
                    "prompt_tokens": 0})
    return out


def test_fit_recovers_known_coefficients():
    a, b, c = 0.012, 0.0031, 2.2e-8
    fa, fb, fc = simulate.fit_step_model(_trace(a, b, c))
    assert fa == pytest.approx(a, rel=1e-9) and fb == pytest.approx(b, rel=1e-9) and fc == pytest.approx(c, rel=1e-6)
    na, nb, nc = simulate.fit_step_model(_trace(a, b, c, n=5000, seed=1, noise=1e-4))
    assert na == pytest.approx(a, abs=2e-5) and nb == pytest.approx(b, abs=5e-6) and nc == pytest.approx(c, rel=0.05)


def test_last_third_selection_drops_admission_steps():
    st = _trace(0.01, 0.002, 0.0, n=90)
    st[80]["prompt_tokens"] = 500
    sel = simulate.select_last_third(st, 1000.0, 1089.0)
    assert all(s["t"] >= 1000.0 + 89 * 2 / 3 for s in sel)
    assert all(not s["prompt_tokens"] for s in sel)
    assert len(sel) == 29


def _brute_force(lengths, prompts, B, model, rate):
    """Step-by-step replay, written independently of the event-driven one."""
    a, b, c = model
    t = 0.0
    queue = list(zip(lengths, prompts))
    live = []  # [remaining, current length]
    while queue or live:
        adm = 0.0
        while len(live) < B and queue:
            L, P = queue.pop(0)
            live.append([L, P])
            adm += P
        t += adm / rate
        n = len(live)
        pad = max(x[1] for x in live) + 1
        t += a + b * n + c * n * pad
        for x in live:
            x[0] -= 1
            x[1] += 1
        live = [x for x in live if x[0] > 0]
    return t / 3600


@pytest.mark.parametrize("B", [1, 3, 8])
def test_event_driven_equals_step_by_step(B):
    rng = np.random.default_rng(B)
    lengths = rng.integers(5, 400, size=23).astype(float)
    prompts = rng.integers(10, 900, size=23).astype(float)
    model = (0.02, 0.003, 1e-6)
    # n_items == len(lengths), so resampling draws a permutation-with-replacement;
    # replay exactly the same draw in the brute force.
    rs = np.random.default_rng(36)
    idx = rs.integers(0, lengths.size, size=23)
    want = _brute_force(list(lengths[idx]), list(prompts[idx]), B, model, 900.0)
    got = simulate.simulate_cell(23, lengths, prompts, B, model, 900.0, seed=36)
    assert got == pytest.approx(want, rel=1e-12)


def test_constant_throughput_model_gives_tokens_over_rate_plus_prefill():
    rate, pre = 180.0, 1500.0
    h = simulate.simulate_cell(198, [5000.0] * 8, [300.0] * 8, 8, (0.0, 1 / rate, 0.0), pre)
    assert h == pytest.approx((198 * 5000 / rate + 198 * 300 / pre) / 3600, rel=1e-12)


def test_tail_and_scale_and_cap():
    m = (0.05, 0.001, 0.0)
    full = simulate.simulate_cell(16, [1000.0], [100.0], 8, m, 1000.0)
    # 2 waves of 1000 steps at n_live 8 plus prefill
    assert full == pytest.approx((2 * 1000 * (0.05 + 0.008) + 1600 / 1000) / 3600, rel=1e-12)
    scaled = simulate.simulate_cell(16, [1000.0], [100.0], 8, m, 1000.0, scale=1.5)
    assert scaled > full
    capped = simulate.simulate_cell(16, [1000.0], [100.0], 8, m, 1000.0, scale=1.5, cap=1000)
    assert capped == pytest.approx(full)


def test_determinism_and_seed_dependence():
    rng = np.random.default_rng(4)
    L = rng.integers(100, 9000, size=8).astype(float)
    P = rng.integers(100, 500, size=8).astype(float)
    m = (0.03, 0.004, 3e-8)
    a = simulate.simulate_cell(198, L, P, 8, m, 1200.0, seed=36)
    assert a == simulate.simulate_cell(198, L, P, 8, m, 1200.0, seed=36)
    assert a != simulate.simulate_cell(198, L, P, 8, m, 1200.0, seed=37)


def test_length_scale_is_mean_plus_one_se_over_mean():
    v = [100.0, 200.0, 300.0, 400.0]
    se = np.std(v, ddof=1) / 2
    assert simulate.length_scale(v) == pytest.approx((250 + se) / 250)
    assert simulate.length_scale([5.0] * 8) == 1.0


def test_bad_inputs():
    with pytest.raises(ValueError):
        simulate.fit_step_model([{"n_live": 1, "padded_len": 1, "step_seconds": 1}])
    with pytest.raises(ValueError):
        simulate.simulate_cell(3, [10.0], [1.0], 2, (-1.0, 0.0, 0.0), 100.0)
    assert simulate.simulate_cell(0, [10.0], [1.0], 2, (0.1, 0.0, 0.0), 100.0) == 0.0
