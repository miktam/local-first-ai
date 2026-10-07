# SPDX-License-Identifier: MIT
"""G0k, the kernel probe of P3 (gate/checks/g0k_kernels.py, gate/preconditions.p3;
DESIGN §3.1 P3, §3.3 G1 item 5; decision F2).

* the rule (gate/rules.p3_g0k through preconditions.p3) on synthetic arrays:
  a pass; a sorted error 3.01 x the unsorted one; one bad row; a sorted call
  that does not repeat bitwise; and the factor's boundary (equality passes,
  one ulp beyond fails);
* the shape set is §3.1 P3's, read from thresholds.json "P3_G0k": every count
  the run issues up to 196,608 rows, the 52,752-row first wave and
  mlx_repro_min.py's boundary sweep, at both projections and both bit widths;
* the inputs follow the routing of the calls they stand for;
* the probe's statistics equal an independent whole-array computation, and a
  wrong sorted kernel (a block of rows off, a non-repeating call) is caught;
* the probe at every judged shape, on this machine. The M4 has no NAX path, so
  this tests the code, not the M5 kernel; the mbp's G1 runs it on the M5.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from gate import preconditions as pre
from gate.checks import g0k_kernels as gk

EXP = Path(__file__).resolve().parents[1]
THRESHOLDS = json.loads((EXP / "gate" / "thresholds.json").read_text())
P3 = THRESHOLDS["P3_G0k"]
KEY = "gate_up|8|rows:4096"
SHAPE = {"projection": "gate_up", "bits": 8, "rows": 4096}
# the thresholds with one judged shape: gate_up, 8 bits, 4,096 rows
TH_ONE = {**THRESHOLDS, "P3_G0k": {**P3, "bits": [8], "judged_rows": {"boundary_mlx_repro_min": [4096]},
                                   "judged_rows_one_expert": [],
                                   "projections": {"gate_up": P3["projections"]["gate_up"]}}}

# §3.1 P3, written out once more from the specification's text.
SPEC_COUNTS = {
    6, 12, 24, 48, 96, 8, 16, 32, 64, 128,                # decode
    12288, 2048, 16384,                                   # one prefill chunk (2,048 on one expert)
    24576, 49152, 98304, 52752, 32768, 65536, 131072,     # batched prefill under min(B, 8)
    73728, 196608,                                        # G2's call; the B = 16 bound
    16384, 24576, 30000, 32000, 32704, 32760, 32767, 32768, 32769, 32770, 32776, 32832, 33000,
    34000, 40000, 49152, 65536,                           # mlx_repro_min.py's boundary sweep
}


def _noisy(ref, rel_err, seed):
    """ref plus noise of the same relative size in every row, so no row stands out."""
    noise = np.random.default_rng(seed).standard_normal(ref.shape)
    noise *= (rel_err * np.linalg.norm(ref, axis=1) / np.linalg.norm(noise, axis=1))[:, None]
    return ref + noise


def _synthetic(rows=4096, cols=512, rel_u=0.002, seed=0):
    ref = np.random.default_rng([seed, 1]).standard_normal((rows, cols))
    return ref, _noisy(ref, rel_u, [seed, 2])


def _judge(measured: dict) -> dict:
    return pre.p3({"results": {KEY: {**measured, **SHAPE}}}, TH_ONE)


# --------------------------------------------------------------------------- the rule


def test_rule_passes_on_a_correct_call():
    ref, yu = _synthetic()
    ys = _noisy(ref, 0.0023, 3)  # an independent error of about the same size, as the M5 shows
    m = gk.measure(ys, ys.copy(), yu, ref)
    assert m["bad_rows"] == 0 and m["repeat_bitwise"] is True
    assert 0.5 < m["ratio_sorted_to_unsorted"] < 2.0
    assert m["rel_f64_unsorted"] == pytest.approx(0.002, rel=1e-9)
    r = _judge(m)
    assert r["ok"] and r["reason"] is None
    assert r["values"]["n_judged"] == 1 and r["values"]["failing"] == [] and r["values"]["absent"] == []


def test_rule_refuses_a_ratio_of_3_01():
    ref, yu = _synthetic()
    ys = ref + 3.01 * (yu - ref)
    m = gk.measure(ys, ys.copy(), yu, ref)
    assert m["ratio_sorted_to_unsorted"] == pytest.approx(3.01, rel=1e-9)
    assert m["bad_rows"] == 0  # 2.01 x 0.2 % per row, far below 5 %
    r = _judge(m)
    assert not r["ok"] and "['gate_up', 8, 4096] (rel)" in r["reason"]
    ok = _judge(gk.measure(ref + 2.99 * (yu - ref), ref + 2.99 * (yu - ref), yu, ref))
    assert ok["ok"]


def test_rule_refuses_one_bad_row():
    ref, yu = _synthetic()
    ys = yu.copy()
    ys[1234] = yu[1234] * 1.06  # 6 % off in one row: > 5 % of ||unsorted||
    m = gk.measure(ys, ys.copy(), yu, ref)
    assert m["bad_rows"] == 1 and m["first_bad_row"] == 1234
    assert m["ratio_sorted_to_unsorted"] < 3.0  # the aggregate alone would pass
    r = _judge(m)
    assert not r["ok"] and "(bad_rows)" in r["reason"]
    assert r["values"]["failing"][0]["bad_rows"] == 1
    ys[1234] = yu[1234] * 1.04  # 4 %: not a bad row
    assert gk.measure(ys, ys, yu, ref)["bad_rows"] == 0


def test_rule_refuses_a_non_repeating_sorted_call():
    ref, yu = _synthetic()
    ys = yu.astype(np.float32)
    yr = ys.copy()
    yr[7, 3] = np.nextafter(yr[7, 3], np.float32(np.inf))  # one ulp in one element
    m = gk.measure(ys, yr, yu, ref)
    assert m["repeat_bitwise"] is False and m["bad_rows"] == 0
    r = _judge(m)
    assert not r["ok"] and "(repeat)" in r["reason"]
    assert gk.measure(ys, ys.copy(), yu, ref)["repeat_bitwise"] is True
    assert not gk.bitwise_equal(ys, ys.astype(np.float64))  # a dtype change is not a repeat


def test_rule_boundary_equality_passes_one_ulp_beyond_fails():
    factor = P3["rel_sorted_max_factor_vs_unsorted"]
    assert factor == 3.0
    base = {"bad_rows": 0, "repeat_bitwise": True, "rel_f64_unsorted": 0.25}
    assert _judge({**base, "rel_f64_sorted": 0.75})["ok"]
    assert not _judge({**base, "rel_f64_sorted": math.nextafter(0.75, 1.0)})["ok"]
    assert not _judge({**base, "rel_f64_sorted": None})["ok"]
    assert not _judge({**base, "rel_f64_sorted": 0.1, "bad_rows": 1})["ok"]


def test_a_judged_shape_without_a_measurement_fails():
    r = pre.p3({"results": {}}, TH_ONE)
    assert not r["ok"] and "not measured" in r["reason"] and r["values"]["absent"] == [["gate_up", 8, 4096]]
    full = pre.p3({"results": {}}, THRESHOLDS)  # every judged (projection, bits, rows) of P3_G0k
    assert len(full["values"]["absent"]) == 4 * len(gk.p3_row_counts()) and full["values"]["measured"] == 0
    other = pre.p3({"results": {KEY: {**SHAPE, "rows": 4097, "rel_f64_sorted": 0.1, "rel_f64_unsorted": 0.1,
                                      "bad_rows": 0, "repeat_bitwise": True}}}, TH_ONE)
    assert not other["ok"] and other["values"]["absent"] == [["gate_up", 8, 4096]]


def test_streamed_blocks_equal_the_whole_array_measurement():
    ref, yu = _synthetic(rows=1000, cols=64)
    ys = ref + 1.7 * (yu - ref)
    ys[10] *= 1.2
    whole = gk.measure(ys, ys, yu, ref)
    acc = gk.Accumulator()
    for a in range(0, 1000, 37):
        acc.add(ys[a:a + 37], yu[a:a + 37], ref[a:a + 37])
    streamed = acc.result()
    for k in ("rows", "bad_rows", "first_bad_row"):
        assert streamed[k] == whole[k]
    for k in ("rel_f64_sorted", "rel_f64_unsorted", "max_row_rel_sorted_vs_unsorted"):
        assert streamed[k] == pytest.approx(whole[k], rel=1e-12)


# --------------------------------------------------------------------------- the shape set


def test_the_judged_set_is_section_3_1_p3():
    assert set(gk.p3_row_counts()) == SPEC_COUNTS
    assert max(SPEC_COUNTS) == 196608 and 52752 in SPEC_COUNTS
    assert set(gk.BOUNDARY_SWEEP) <= SPEC_COUNTS
    assert sum(s.rows for s in gk.JUDGED) == pytest.approx(1.2e6, rel=0.05)  # "about 1.2 M rows"
    by = {(s.rows, s.routing) for s in gk.JUDGED}
    assert (2048, "hot") in by
    assert {(n, "tokens") for n in (6, 12, 24, 48, 96, 12288, 24576, 49152, 98304, 52752, 73728, 196608)} <= by
    assert {(n, "rows") for n in gk.BOUNDARY_SWEEP} <= by
    assert all(s.rows % gk.K == 0 for s in gk.JUDGED if s.routing == "tokens")


def test_thresholds_p3_g0k_is_read_as_the_judged_set():
    cfg = gk.config_from_thresholds(P3)
    assert {(s.rows, s.routing) for s in cfg["shapes"]} == {(s.rows, s.routing) for s in gk.JUDGED}
    assert cfg["bits"] == (8, 4) and set(cfg["projections"]) == {"gate_up", "down"}
    assert cfg["seed"] == gk.SEED == 37 and cfg["bad_row_fraction"] == gk.BAD_ROW_FRACTION == 0.05
    assert P3["bad_rows_max"] == 0 and P3["repeat_bitwise"] is True and P3["recorded_only"] == []
    assert gk.PROJECTIONS == {"gate_up": (512, 2560), "down": (2560, 512)} and gk.E == 384
    with pytest.raises(ValueError):
        gk.config_from_thresholds({**P3, "judged_rows": {**P3["judged_rows"], "something_new": [7]}})
    with pytest.raises(ValueError):
        gk.config_from_thresholds({**P3, "group_size": 32})
    with pytest.raises(ValueError):
        gk.config_from_thresholds({**P3, "recorded_only": [32769]})
    assert len(gk.expected_keys(cfg["shapes"], cfg["bits"], cfg["projections"])) == 4 * len(gk.JUDGED)


def test_lifted_sources_are_the_pinned_ones():
    import hashlib

    confirm = EXP.parent / "exp_036_kolibri_local_eval" / "diagnostics" / "gate1" / "confirm"
    for name, prefix in P3["lifted_from"].items():
        assert hashlib.sha256((confirm / name).read_bytes()).hexdigest().startswith(prefix)


# --------------------------------------------------------------------------- inputs


def test_inputs_follow_the_routing_and_are_deterministic():
    tok = next(s for s in gk.JUDGED if s.routing == "tokens" and s.rows == 12288)
    xg, ig = gk.make_inputs(tok, "gate_up", 2560)
    xd, idd = gk.make_inputs(tok, "down", 512)
    assert xg.shape == (12288, 2560) and xd.shape == (12288, 512) and xg.dtype == np.uint16
    assert np.array_equal(ig, idd)  # the same expert ids for both projections
    assert np.all(np.diff(ig) >= 0) and ig.min() >= 0 and ig.max() < gk.E
    assert np.bincount(ig, minlength=gk.E).sum() == 12288
    # each token's hidden state is gate/up's input once per expert: 2,048 distinct rows
    assert np.unique(xg, axis=0).shape[0] == 12288 // gk.K
    assert np.unique(xd, axis=0).shape[0] == 12288
    x2, i2 = gk.make_inputs(tok, "gate_up", 2560)
    assert np.array_equal(x2, xg) and np.array_equal(i2, ig)
    hot = next(s for s in gk.JUDGED if s.routing == "hot")
    _, ih = gk.make_inputs(hot, "down", 512)
    assert hot.rows == 2048 and set(ih.tolist()) == {gk.HOT_EXPERT}
    rows = next(s for s in gk.JUDGED if s.routing == "rows" and s.rows == 32769)
    xr, ir = gk.make_inputs(rows, "gate_up", 2560)
    assert xr.shape == (32769, 2560) and np.all(np.diff(ir) >= 0)
    # bf16 bit patterns of standard-normal values
    v = gk.bf16_bits_to_f64(xr[:4])
    assert np.all(np.isfinite(v)) and 0.5 < float(np.std(gk.bf16_bits_to_f64(xr[:64]))) < 2.0


def test_tokens_routing_gives_k_distinct_experts_per_token():
    rng = np.random.default_rng(0)
    t = gk.route_uniform(rng, 100)
    assert t.shape == (100, gk.K) and all(len(set(r)) == gk.K for r in t.tolist())
    idx, tok = gk.flatten_sorted(t)
    assert np.all(np.diff(idx) >= 0)
    assert sorted(zip(tok.tolist(), idx.tolist())) == sorted((i, int(e)) for i in range(100) for e in t[i])


# --------------------------------------------------------------------------- the probe on mlx


@pytest.fixture(scope="module")
def weights8():
    pytest.importorskip("mlx.core")
    return gk.make_weights("gate_up", 8)


def test_probe_statistics_equal_a_whole_array_computation(weights8):
    import mlx.core as mx

    W, deq = weights8
    shape = gk.merge_shapes([(600, "rows", "test")])[0]
    r = gk.probe_one(shape, "gate_up", 8, W, deq)
    assert "pass" not in r and "ok" not in r  # measurement only (DESIGN §9.2)
    x_bits, idx = gk.make_inputs(shape, "gate_up", 2560)
    x = mx.array(x_bits).view(mx.bfloat16)[:, None, :]
    ri = mx.array(idx.astype(np.uint32))
    ys = np.array(gk.call(x, ri, W, True, 8).astype(mx.float32))[:, 0, :]
    yu = np.array(gk.call(x, ri, W, False, 8).astype(mx.float32))[:, 0, :]
    x64 = gk.bf16_bits_to_f64(x_bits)
    ref = np.stack([x64[i] @ deq[idx[i]].astype(np.float64).T for i in range(len(idx))])
    assert r["rel_f64_sorted"] == pytest.approx(gk.rel(ys, ref), rel=1e-9)
    assert r["rel_f64_unsorted"] == pytest.approx(gk.rel(yu, ref), rel=1e-9)
    assert r["bad_rows"] == 0 and r["repeat_bitwise"] is True and r["rows"] == 600
    # the quantised weights are what deq says: dequantise expert 5 on the device
    q, s, b = W
    d5 = mx.dequantize(q[5], s[5].astype(mx.float32), b[5].astype(mx.float32), group_size=64, bits=8)
    assert np.array_equal(np.array(d5), deq[5])


def test_a_wrong_sorted_kernel_is_caught(weights8, monkeypatch):
    import mlx.core as mx

    W, deq = weights8
    real_call = gk.call
    shape = gk.merge_shapes([(4096, "rows", "test")])[0]

    def rows_off(x, idx, W_, sorted_flag, bits):  # a block of 40 rows 10 % off on the sorted path only
        y = real_call(x, idx, W_, sorted_flag, bits)
        if sorted_flag:
            y = mx.concatenate([y[:1000], y[1000:1040] * 1.1, y[1040:]])
            mx.eval(y)
        return y

    monkeypatch.setattr(gk, "call", rows_off)
    r = gk.probe_one(shape, "gate_up", 8, W, deq)
    assert r["bad_rows"] == 40 and r["first_bad_row"] == 1000
    judged = pre.p3({"results": {KEY: r}}, TH_ONE)
    assert not judged["ok"] and "bad_rows" in judged["values"]["failing"][0]["failed"]

    calls = {"n": 0}

    def flaky(x, idx, W_, sorted_flag, bits):  # the second sorted call differs in one element
        y = real_call(x, idx, W_, sorted_flag, bits)
        if sorted_flag:
            calls["n"] += 1
            if calls["n"] == 2:
                y = mx.concatenate([y[:1], y[1:2] * 1.0078125, y[2:]])
                mx.eval(y)
        return y

    monkeypatch.setattr(gk, "call", flaky)
    r2 = gk.probe_one(shape, "gate_up", 8, W, deq)
    assert r2["repeat_bitwise"] is False and r2["bad_rows"] == 0
    assert not pre.p3({"results": {KEY: r2}}, TH_ONE)["ok"]


def test_g0k_at_every_judged_shape_on_this_machine():
    """G1 item 5: the probe at §3.1 P3's shapes (to 196,608 rows), both
    projections, 8 and 4 bits, judged by P3's rule. About 2 minutes on the M4."""
    pytest.importorskip("mlx.core")
    cfg = gk.config_from_thresholds(P3)
    res = gk.probe(**cfg)
    assert len(res["results"]) == 4 * len(gk.JUDGED)
    assert max(r["rows"] for r in res["results"].values()) == 196608
    assert all("pass" not in r for r in res["results"].values())
    assert json.loads(json.dumps(res, default=str))["seed"] == 37
    judged = pre.p3(res, THRESHOLDS)
    assert judged["ok"], judged["reason"]
    assert judged["values"]["failing"] == [] and judged["values"]["absent"] == []
    assert judged["values"]["n_judged"] == 4 * len(gk.p3_row_counts())
    assert judged["values"]["measured"] == 4 * len(gk.JUDGED)
    for r in res["results"].values():
        assert r["repeat_bitwise"] and r["bad_rows"] == 0
        assert 0 < r["rel_f64_unsorted"] < 0.02 and r["rel_f64_sorted"] <= 3 * r["rel_f64_unsorted"]
