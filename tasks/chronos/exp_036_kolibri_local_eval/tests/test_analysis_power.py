"""analysis/power.py: reproduces the HYPOTHESIS power tables and the budget/scenario tables.

BUILD_SPEC §5.8: power tables to +/- 0.01 and the scenario picks exactly. The
H8 figures in HYPOTHESIS come from a 1,500-replicate simulation (Monte Carlo SE
about 0.013), so H8 is checked within three combined Monte Carlo SEs instead.
"""

from __future__ import annotations

import math

import pytest

from analysis import power as P

TOL = 0.01

# HYPOTHESIS H2 power table: n_M -> (SE pp, {D: (a8, a3)})
H2_TABLE = {
    588: (1.09, {0: (0.88, 0.94), -1: (0.60, 0.74), -2: (0.25, 0.39), -3: (0.06, 0.11)}),
    406: (1.14, {0: (0.84, 0.92), -1: (0.55, 0.69), -2: (0.23, 0.35), -3: (0.05, 0.11)}),
    294: (1.20, {0: (0.80, 0.89), -1: (0.50, 0.64), -2: (0.20, 0.32), -3: (0.05, 0.10)}),
    196: (1.31, {0: (0.71, 0.83), -1: (0.42, 0.57), -2: (0.17, 0.28), -3: (0.04, 0.09)}),
    154: (1.39, {0: (0.65, 0.78), -1: (0.37, 0.51), -2: (0.15, 0.25), -3: (0.04, 0.08)}),
}
H2_P10 = (1.57, {0: (0.52, 0.66), -1: (0.28, 0.41), -2: (0.11, 0.20), -3: (0.03, 0.07)})
H3_TABLE = {588: ((0.98, 0.99), (0.53, 0.67)), 406: ((0.89, 0.94), (0.36, 0.50)),
            294: ((0.74, 0.85), (0.25, 0.38)), 196: ((0.53, 0.67), (0.16, 0.26)),
            154: ((0.41, 0.56), (0.12, 0.21))}
H7_TABLE = {588: ((0.70, 0.81), (0.31, 0.45)), 406: ((0.61, 0.75), (0.26, 0.39)),
            294: ((0.53, 0.67), (0.22, 0.34)), 196: ((0.41, 0.56), (0.16, 0.27)),
            154: ((0.34, 0.49), (0.14, 0.23))}


@pytest.mark.parametrize("n_M", sorted(H2_TABLE))
def test_h2_table(n_M):
    se, rows = H2_TABLE[n_M]
    assert 100 * P.se_h2(n_M) == pytest.approx(se, abs=0.005 + 1e-9)
    for D, (a8, a3) in rows.items():
        assert P.power("H2", n_M, "a8", {"true": D / 100}) == pytest.approx(a8, abs=TOL)
        assert P.power("H2", n_M, "a3", {"true": D / 100}) == pytest.approx(a3, abs=TOL)


def test_h2_p10_four_rows_and_refute_power():
    se, rows = H2_P10
    assert 100 * P.se_h2(154, four_rows=True) == pytest.approx(se, abs=0.005 + 1e-9)
    for D, (a8, a3) in rows.items():
        assert P.power("H2", 154, "a8", {"true": D / 100, "four_rows": True}) == pytest.approx(a8, abs=TOL)
        assert P.power("H2", 154, "a3", {"true": D / 100, "four_rows": True}) == pytest.approx(a3, abs=TOL)
    # "REFUTED needs a true shortfall well beyond the margin: power 0.88 at D = -8 (n_M = 588), 0.71 at n_M = 196"
    assert P.power("H2_refute", 588, "a8") == pytest.approx(0.88, abs=TOL)
    assert P.power("H2_refute", 196, "a8") == pytest.approx(0.71, abs=TOL)


def test_design_effect():
    assert P.mmlu_deff() == pytest.approx(1.11, abs=0.005)


@pytest.mark.parametrize("n_M", sorted(H3_TABLE))
def test_h3_table(n_M):
    (v8, v3), (t8, t3) = H3_TABLE[n_M]
    assert P.power("H3", n_M, "a8") == pytest.approx(v8, abs=TOL)
    assert P.power("H3", n_M, "a3") == pytest.approx(v3, abs=TOL)
    assert P.power("H3", n_M, "a8", {"true": -0.03}) == pytest.approx(t8, abs=TOL)
    assert P.power("H3", n_M, "a3", {"true": -0.03}) == pytest.approx(t3, abs=TOL)


@pytest.mark.parametrize("n_M", sorted(H7_TABLE))
def test_h7_table(n_M):
    (z8, z3), (m8, m3) = H7_TABLE[n_M]
    assert P.power("H7", n_M, "a8") == pytest.approx(z8, abs=TOL)
    assert P.power("H7", n_M, "a3") == pytest.approx(z3, abs=TOL)
    assert P.power("H7", n_M, "a8", {"true": -0.01}) == pytest.approx(m8, abs=TOL)
    assert P.power("H7", n_M, "a3", {"true": -0.01}) == pytest.approx(m3, abs=TOL)


def test_h1_t_test_power():
    # "1.00 at a true 0.9; 0.93 / 0.98 at 0.80; 0.47 / 0.67 at 0.78. At SD 0.06: 0.62 / 0.80 at 0.80."
    assert P.power("H1", None, "a8", {"true": 0.9}) == pytest.approx(1.00, abs=TOL)
    assert P.power("H1", None, "a8", {"true": 0.8}) == pytest.approx(0.93, abs=TOL)
    assert P.power("H1", None, "a3", {"true": 0.8}) == pytest.approx(0.98, abs=TOL)
    assert P.power("H1", None, "a8", {"true": 0.78}) == pytest.approx(0.47, abs=TOL + 1e-6)
    assert P.power("H1", None, "a3", {"true": 0.78}) == pytest.approx(0.67, abs=TOL)
    assert P.power("H1", None, "a8", {"true": 0.8, "sd": 0.06}) == pytest.approx(0.62, abs=TOL)
    assert P.power("H1", None, "a3", {"true": 0.8, "sd": 0.06}) == pytest.approx(0.80, abs=TOL)


def test_t_test_power_at_zero_effect_is_alpha():
    assert P.t_test_power(0.0, 9, 0.05) == pytest.approx(0.05, abs=1e-4)


def test_h4_h5_h6():
    assert P.power("H4", None, "a8") == pytest.approx(0.90, abs=TOL)
    assert P.power("H4", None, "a3") == pytest.approx(0.95, abs=TOL)
    assert P.power("H4", None, "a8", {"true": 0.06}) == pytest.approx(0.27, abs=TOL)
    assert P.power("H4", None, "a3", {"true": 0.06}) == pytest.approx(0.41, abs=TOL)
    assert P.power("H5", None, "a8") == pytest.approx(0.96, abs=TOL)
    assert P.power("H5", None, "a8", {"true": 1.16}) == pytest.approx(0.22, abs=TOL)
    assert P.power("H6", None, "a8") == pytest.approx(1.0, abs=TOL)
    assert P.power("H6", None, "a8", {"true": -0.15}) == pytest.approx(0.39, abs=TOL)
    assert P.power("H6", None, "a3", {"true": -0.15}) == pytest.approx(0.54, abs=TOL)


@pytest.mark.parametrize("kw,want", [({}, 0.99), ({"tau": 0.3}, 0.88), ({"true": 1.2}, 0.85),
                                     ({"true": 1.2, "tau": 0.3}, 0.52)])
def test_h8_simulation_within_monte_carlo_error(kw, want):
    got = P.power("H8", None, "a8", kw)
    se = math.sqrt(want * (1 - want) / 1500 + got * (1 - got) / P.DEFAULTS["H8"]["R"])
    assert abs(got - want) <= max(3 * se, TOL)


def test_h8_power_is_seeded():
    assert P.power("H8", None, "a8", {"R": 300}) == P.power("H8", None, "a8", {"R": 300})


def test_p10_sentence():
    # "At P10, power is 0.41 / 0.56 for H3 at the vendor's gap and 0.34 / 0.49 for H7 at a true cost of 0"
    assert P.power("H3", 154, "a8") == pytest.approx(0.41, abs=TOL)
    assert P.power("H3", 154, "a3") == pytest.approx(0.56, abs=TOL)
    assert P.power("H7", 154, "a8") == pytest.approx(0.34, abs=TOL)
    assert P.power("H7", 154, "a3") == pytest.approx(0.49, abs=TOL)


def test_power_at_plan_shape():
    r = P.power_at_plan(196)
    assert r["H3_vendor_gap"]["a8"] == pytest.approx(0.53, abs=TOL)
    r4 = P.power_at_plan(154, h2_rows=4)
    assert r4["H2_D0"]["a8"] == pytest.approx(0.52, abs=TOL)
    assert "GPQA EN n = 197 as registered" in r["assumptions"]


def test_h2_power_gpqa_en_197_vs_198_under_0001():
    """Amendment 3: GPQA EN runs all 198 Diamond items, but the H2 power arithmetic stays at the
    registered n = 197. Check that this moves no registered H2 design point by 0.001 or more."""
    rows = P.DEFAULTS["H2"]["rows_full"]
    assert rows[0][1] == 197
    rows198 = [[rows[0][0], 198, rows[0][2]]] + [list(r) for r in rows[1:]]
    worst = 0.0
    for n_M in P.NM_LADDER:
        for four in (False, True):
            for true in (0.0, -0.01, -0.02, -0.03):
                for a in ("a8", "a3"):
                    p197 = P.power("H2", n_M, a, {"true": true, "four_rows": four})
                    p198 = P.power("H2", n_M, a, {"true": true, "four_rows": four, "rows_full": rows198})
                    worst = max(worst, abs(p197 - p198))
    assert 0.0 < worst < 0.001


# ---------------------------------------------------------------- budget

LADDER = {"P0": (25.1, 56.4, 74.3), "P1": (23.4, 52.8, 69.9), "P2": (21.3, 48.3, 64.0), "P3": (18.9, 43.0, 57.2),
          "P4": (17.4, 40.0, 53.3), "P5": (16.0, 37.0, 49.4), "P6": (15.0, 34.7, 46.5), "P7": (13.6, 31.7, 42.6),
          "P8": (12.7, 29.4, 39.6), "P9": (11.7, 27.1, 36.6), "P10": (9.9, 22.9, 30.3)}
TIER_B = {"B1": (1.5, 3.2), "B2": (2.2, 4.1), "B3": (2.9, 6.1), "B4": (1.2, 1.4), "B5": (0.9, 2.4),
          "B6": (2.9, 6.1), "B7": (2.0, 4.3), "B8": (2.7, 6.4), "B9": (4.5, 9.6), "B10": (3.5, 8.5)}


def test_budget_ladder_and_tier_b_tables():
    b = P.budget()
    for name, want in LADDER.items():
        assert tuple(b["ladder_totals"][name]) == want, name
    for name, want in TIER_B.items():
        assert tuple(b["tier_b_totals"][name][:2]) == want, name


def test_budget_scenario_picks_exactly():
    s = P.budget()["scenarios"]
    assert s["nominal"] == {"S1": 5.4, "B_main": 31.0, "plan": "P0", "n_M": 588, "forced": True, "negfc": True,
                            "tier_b": ["B1", "B2", "B5"], "S2": 14.0, "S3": 15.7, "main": 29.7, "run_total": 35.1}
    assert s["pessimistic"] == {"S1": 9.1, "B_main": 30.9, "plan": "P8", "n_M": 196, "forced": False,
                                "negfc": False, "tier_b": [], "S2": 15.8, "S3": 13.5, "main": 29.4,
                                "run_total": 38.5}
    assert s["adverse"] == {"S1": 9.6, "B_main": 30.4, "plan": "P10", "n_M": 154, "forced": False,
                            "negfc": False, "tier_b": [], "S2": 15.3, "S3": 15.0, "main": 30.3, "run_total": 39.9}


def test_budget_slightly_worse_than_adverse_stops():
    s = P.budget({"s1_hours": {"nominal": 5.4, "pessimistic": 9.1, "adverse": 10.0}})["scenarios"]
    assert s["adverse"]["plan"] == "STOP"


def test_tier_a_queue_order():
    q = P.tier_a_queue(588, negfc=True, forced=True)
    assert q[:6] == [("K8", "GPQA", 198), ("K8", "GPQA", 198), ("K8", "MMLU", 588), ("K8", "MMLU", 588),
                     ("K8", "IFB", 300), ("K8", "CB", 400)]
    assert q[-6:] == [("K8", "NEG", 300), ("K8", "FC", 100), ("G8", "NEG", 300), ("G8", "FC", 100),
                      ("Q36-8", "NEG", 300), ("Q36-8", "FC", 100)]
    assert ("K8", "GPQA", 198) not in P.tier_a_queue(154, False, False, k8_gpqa_de=False)[1:]
