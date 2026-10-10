"""W7 analysis: the tests check each statistic against an independent oracle (brute force or closed form)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as A  # noqa: E402


def test_signflip_matches_brute_force():
    vals = [2, 1, 0, -1, 1, 2, 0, 1, -2, 1]
    obs = sum(vals)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=len(vals)) if sum(s * v for s, v in zip(signs, vals)) >= obs)
    assert abs(A.signflip_upper(vals) - hits / 2 ** len(vals)) < 1e-12


def test_h3_p_rev_uses_the_shift():
    # all differences 0: mean 0 < 0.20, so the reverse test (H0: mean >= 0.20) should reject at n = 33
    r = A.h3_test([0] * 33, tested=True)
    assert r["p"] == 1.0 and r["p_rev"] < 0.05 and r["state"] == "REFUTED"
    # all +2: H0 mean <= 0 rejected
    r = A.h3_test([2] * 33, tested=True)
    assert r["p"] < 1e-6 and r["state"] == "CONFIRMED"
    assert A.h3_test([2] * 33, tested=False)["state"] == "not tested"


def test_h1_bootstrap_is_seeded_and_sane():
    diffs = [1, 0, 0, 1, 2, 0, -1, 1] * 7 + [0, 1, 0, 1]
    strata = (["en"] * 31 + ["pl"] * 17 + ["es"] * 12)
    a, b = A.h1_test(diffs, strata, reps=2000), A.h1_test(diffs, strata, reps=2000)
    assert a == b  # deterministic seed
    assert a["state"] == "CONFIRMED" and a["ci95"][0] > 0
    z = A.h1_test([0] * 60, strata, reps=2000)
    assert z["state"] == "REFUTED" and z["p"] == 1.0


def test_kappa_and_agreement():
    assert A.quadratic_kappa([0, 1, 2, 2, 1], [0, 1, 2, 2, 1]) == 1.0
    r = A.reliability({"a": 2, "b": 1, "c": 0, "d": 2}, {"a": 2, "b": 0, "c": 0, "d": 2})
    assert r["exact_agreement"] == 0.75 and r["double_judging_required"]


def test_ceiling_and_harm_guard():
    rows = [{"type": "simple", "substance_in_context": True}, {"type": "simple", "substance_in_context": False},
            {"type": "oos", "substance_in_context": False}]
    assert abs(A.ceiling(rows) - 5 / 3) < 1e-12
    k4 = {"x": {"usefulness": 1}, "y": {"usefulness": 5}}
    g = {"x": {"usefulness": 5}, "y": {"usefulness": 1}}
    assert A.harm_guard(k4, g, ["x", "y"])["holds"]


def test_verdict_map_order():
    h3 = {"state": "not tested"}
    d4b = {"point": 0.0, "ci95": (-0.2, 0.2)}
    harm = {"holds": True}
    conf = {"state": "CONFIRMED", "point": 0.25, "ci95": (0.1, 0.4), "n": 60}
    assert A.verdict(conf, True, [], harm, "CONFIRMED", None, False, h3, d4b).startswith("Earns a live-bot trial")
    assert A.verdict(conf, False, ["ES language"], harm, "CONFIRMED", None, False, h3, d4b).startswith("Better, not drop-in (ES language")
    assert A.verdict(dict(conf, point=0.12), True, [], harm, "CONFIRMED", None, False, h3, d4b).startswith("Better by less")
    assert A.verdict(conf, True, [], harm, "INCONCLUSIVE", None, False, h3, d4b).startswith("Better, not drop-in (E38-D1")
    assert A.verdict({"state": "REFUTED", "point": 0, "ci95": (0, 0), "n": 60}, True, [], harm, "CONFIRMED", None, True, h3, d4b).startswith("A +0.20 gain is not attainable")
    assert A.verdict(conf, True, [], harm, "CONFIRMED", "gate", False, h3, d4b) == "NOT RUN"
