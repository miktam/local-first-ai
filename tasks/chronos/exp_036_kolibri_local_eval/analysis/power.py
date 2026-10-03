"""exp_036 power and planning budget (Tier 1).

BUILD_SPEC §5.8: power(h, n, alpha_level, assumptions) using the HYPOTHESIS
formulas, plus budget(assumptions), which reproduces the HYPOTHESIS ladder and
scenario tables ("Sessions & budget"). runner/plan_fix.py calls power_at_plan()
to recompute power at the fixed n for the plan amendment.

Formulas (normal approximations, as in the design arithmetic of 2026-10-03):
- one-sided test at level a: z_a = Phi^-1(1 - a); a = alpha/8 (worst case,
  primary) or alpha/3 (if five others are confirmed first).
- "claim theta > margin" power: 1 - Phi((margin + z_a SE - true) / SE).
- "claim theta < threshold" power: Phi((threshold - z_a SE - true) / SE).
- H1: exact power of the one-sample t-test (df = n_blocks - 1) on the block
  log-ratios, by numerical integration over the chi-square distribution.
- H2: SE(D-bar) = sqrt(sum_r var_r) / k, with var_r = p(1-p)/n + p(1-p)/(N_v n)
  for GPQA EN/DE and IFBench, and deff * p(1-p)/n_M for the two MMLU rows,
  deff = sum_c (14 w_c)^2 / 14 = 1.114 from the MMLU-Pro 12,032 category shares.
- H3: SE = sd(d) / sqrt(2 n_M); H4: sd / sqrt(300); H6: sd / sqrt(400);
  H7: SE = sqrt(2 * 0.42^2 / n_M + 0.45^2 / 300 + 0.35^2 / 400) / 4;
  H5: SE = 0.5 % of the true ratio (CI about +/- 1 %).
- H8: a seeded simulation (6 texts x 8 blocks, between-text SD tau, within-block
  SD sigma of the log ratio) with a nested stratified block bootstrap
  (B_inner = 1,000) and the per-text condition (>= 5 of 6 texts).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis import stats

ALPHA = 0.05
ANALYSIS_DIR = Path(__file__).resolve().parent
NM_LADDER = (588, 504, 406, 350, 294, 252, 196, 154)


def alpha_of(level) -> float:
    """'a8' -> alpha/8, 'a3' -> alpha/3, or a float alpha."""
    if level == "a8":
        return ALPHA / 8
    if level == "a3":
        return ALPHA / 3
    return float(level)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power_greater(true: float, margin: float, se: float, a: float) -> float:
    """Power of a one-sided test of theta > margin (null theta <= margin)."""
    z = stats.z_quantile(1.0 - a)
    return 1.0 - _phi((margin + z * se - true) / se)


def power_less(true: float, threshold: float, se: float, a: float) -> float:
    """Power of a one-sided test of theta < threshold (null theta >= threshold)."""
    z = stats.z_quantile(1.0 - a)
    return _phi((threshold - z * se - true) / se)


def mmlu_deff(counts: dict | None = None) -> float:
    """Design effect of post-stratifying a category-balanced sample: sum (K w_c)^2 / K."""
    if counts is None:
        vv = json.loads((ANALYSIS_DIR / "vendor_values.json").read_text(encoding="utf-8"))
        counts = vv["mmlu_pro_category_counts"]["counts"]
    c = np.array(list(counts.values()), dtype=np.float64)
    w = c / c.sum()
    K = len(c)
    return float(np.sum((K * w) ** 2) / K)


DEFAULTS = {
    "H1": {"true": 0.80, "sd": 0.042, "n_blocks": 10, "margin": 0.75},
    "H2": {"true": 0.0, "margin": -0.04, "four_rows": False,
           "rows_full": [[0.843, 197, 8], [0.813, 198, 8], [0.781, 300, 5]],
           "rows_mmlu": [0.80, 0.755]},
    "H2_refute": {"true": -0.08, "margin": -0.04, "four_rows": False},
    "H3": {"true": -0.052, "sd": 0.40, "threshold": 0.0},
    "H4": {"true": 0.12, "sd": 0.55, "n": 300, "threshold": 0.0},
    "H5": {"true": 1.175, "rel_se": 0.005, "threshold": 1.15},
    "H6": {"true": -0.28, "sd": 0.45, "n": 400, "threshold": -0.10},
    "H7": {"true": 0.0, "sd_mmlu": 0.42, "sd_ifbench": 0.45, "n_ifbench": 300, "sd_rgb": 0.35, "n_rgb": 400,
           "margin": -0.03},
    "H8": {"true": 1.0, "tau": 0.15, "sigma": 0.3, "n_texts": 6, "n_blocks": 8, "threshold": 1.5,
           "texts_min": 5, "R": 4000, "B_inner": 1000, "seed": "power|H8"},
}


def se_h2(n_M: int, four_rows: bool = False, a: dict | None = None) -> float:
    a = {**DEFAULTS["H2"], **(a or {})}
    full = a["rows_full"] if not four_rows else [a["rows_full"][0], a["rows_full"][2]]
    deff = a.get("deff", mmlu_deff())
    v = 0.0
    for p, n, nv in full:
        v += p * (1 - p) / n + p * (1 - p) / (nv * n)
    for p in a["rows_mmlu"]:
        v += deff * p * (1 - p) / n_M
    return math.sqrt(v) / (len(full) + len(a["rows_mmlu"]))


def se_h3(n_M: int, sd: float = 0.40) -> float:
    return sd / math.sqrt(2 * n_M)


def se_h7(n_M: int, a: dict | None = None) -> float:
    a = {**DEFAULTS["H7"], **(a or {})}
    v = 2 * a["sd_mmlu"] ** 2 / n_M + a["sd_ifbench"] ** 2 / a["n_ifbench"] + a["sd_rgb"] ** 2 / a["n_rgb"]
    return math.sqrt(v) / 4


def _chi2_pdf(v: np.ndarray, k: int) -> np.ndarray:
    return np.exp((k / 2 - 1) * np.log(v) - v / 2 - (k / 2) * math.log(2) - math.lgamma(k / 2))


def t_test_power(delta: float, df: int, a: float) -> float:
    """P(T > t_{df,1-a}) for T noncentral t with noncentrality delta (numerical integration)."""
    c = stats.t_ppf(1.0 - a, df)
    hi = df + 80.0 * math.sqrt(2.0 * df) + 80.0
    v = np.linspace(1e-9, hi, 200_001)
    f = _chi2_pdf(v, df)
    x = c * np.sqrt(v / df) - delta
    sf = 0.5 * np.array([math.erfc(xx / math.sqrt(2.0)) for xx in x])
    g = sf * f
    return float(np.sum((g[1:] + g[:-1]) * np.diff(v)) / 2.0)


def h8_power(a: dict, alpha: float) -> float:
    """Seeded simulation of H8's decision with a nested stratified block bootstrap.

    Per replicate: the log ratio of block b in text t is log(true) + u_t + e_tb,
    u_t ~ N(0, tau), e_tb ~ N(0, sigma); the statistic is the equal-weight mean;
    the 8 blocks of each text are resampled B_inner times; the test rejects if
    (1 + #{mean* >= log 1.5}) / (B_inner + 1) <= alpha and the per-text mean is
    <= log 1.5 on >= texts_min texts. This is the design simulation behind the
    HYPOTHESIS H8 power figures (which used 1,500 replicates, MC SE about 0.013).
    """
    gen = stats.rng(a["seed"])
    R, T, nb, Bi = int(a["R"]), int(a["n_texts"]), int(a["n_blocks"]), int(a["B_inner"])
    thr = math.log(a["threshold"])
    hits = 0
    chunk = 50
    for start in range(0, R, chunk):
        r = min(chunk, R - start)
        te = gen.normal(0.0, a["tau"], size=(r, T, 1))
        x = math.log(a["true"]) + te + gen.normal(0.0, a["sigma"], size=(r, T, nb))
        idx = gen.integers(0, nb, size=(r, Bi, T, nb))
        boot = np.take_along_axis(np.broadcast_to(x[:, None], (r, Bi, T, nb)), idx, axis=3).mean(axis=(2, 3))
        p = (1 + np.sum(boot >= thr - stats.TIE_EPS, axis=1)) / (Bi + 1)
        per_text = (x.mean(axis=2) <= thr).sum(axis=1) >= a["texts_min"]
        hits += int(np.sum((p <= alpha) & per_text))
    return hits / R


def power(h: str, n: int | None = None, alpha_level="a8", assumptions: dict | None = None) -> float:
    """Power of hypothesis h at n (n_M for H2, H3, H7; ignored otherwise) and level alpha_level."""
    a_ = alpha_of(alpha_level)
    A = {**DEFAULTS.get(h, {}), **(assumptions or {})}
    if h == "H1":
        se = A["sd"] / math.sqrt(A["n_blocks"])
        delta = (math.log(A["true"]) - math.log(A["margin"])) / se
        return t_test_power(delta, int(A["n_blocks"]) - 1, a_)
    if h == "H2":
        return power_greater(A["true"], A["margin"], se_h2(int(n), A.get("four_rows", False), A), a_)
    if h == "H2_refute":
        A2 = {**DEFAULTS["H2"], **A}
        return power_less(A["true"], A["margin"], se_h2(int(n), A.get("four_rows", False), A2), a_)
    if h == "H3":
        return power_less(A["true"], A["threshold"], se_h3(int(n), A["sd"]), a_)
    if h == "H4":
        return power_greater(A["true"], A["threshold"], A["sd"] / math.sqrt(A["n"]), a_)
    if h == "H5":
        return power_greater(A["true"], A["threshold"], A["rel_se"] * A["true"], a_)
    if h == "H6":
        return power_less(A["true"], A["threshold"], A["sd"] / math.sqrt(A["n"]), a_)
    if h == "H7":
        return power_greater(A["true"], A["margin"], se_h7(int(n), A), a_)
    if h == "H8":
        return h8_power(A, a_)
    raise ValueError(f"unknown hypothesis {h!r}")


def power_tables() -> dict:
    """Every power figure quoted in HYPOTHESIS, recomputed (keys mirror the tables)."""
    out: dict = {"H2": {}, "H3": {}, "H7": {}}
    for n in NM_LADDER:
        out["H2"][n] = {"se_pp": 100 * se_h2(n),
                        **{f"D={d:g}": (power("H2", n, "a8", {"true": d / 100}), power("H2", n, "a3", {"true": d / 100}))
                           for d in (0, -1, -2, -3)}}
        out["H3"][n] = {"vendor_-5.2": (power("H3", n, "a8"), power("H3", n, "a3")),
                        "-3": (power("H3", n, "a8", {"true": -0.03}), power("H3", n, "a3", {"true": -0.03}))}
        out["H7"][n] = {"0": (power("H7", n, "a8"), power("H7", n, "a3")),
                        "-1": (power("H7", n, "a8", {"true": -0.01}), power("H7", n, "a3", {"true": -0.01}))}
    out["H2"]["154_four_rows"] = {"se_pp": 100 * se_h2(154, True),
                                  **{f"D={d:g}": (power("H2", 154, "a8", {"true": d / 100, "four_rows": True}),
                                                  power("H2", 154, "a3", {"true": d / 100, "four_rows": True}))
                                     for d in (0, -1, -2, -3)}}
    out["H2_refute"] = {n: power("H2_refute", n, "a8") for n in (588, 196)}
    out["H1"] = {
        "sd0.042": {t: (power("H1", None, "a8", {"true": t}), power("H1", None, "a3", {"true": t}))
                    for t in (0.9, 0.8, 0.78)},
        "sd0.06": {0.8: (power("H1", None, "a8", {"true": 0.8, "sd": 0.06}),
                         power("H1", None, "a3", {"true": 0.8, "sd": 0.06}))},
    }
    out["H4"] = {"+12": (power("H4", None, "a8"), power("H4", None, "a3")),
                 "+6": (power("H4", None, "a8", {"true": 0.06}), power("H4", None, "a3", {"true": 0.06}))}
    out["H5"] = {"1.175": power("H5", None, "a8"), "1.16": power("H5", None, "a8", {"true": 1.16})}
    out["H6"] = {"-28": power("H6", None, "a8"),
                 "-15": (power("H6", None, "a8", {"true": -0.15}), power("H6", None, "a3", {"true": -0.15}))}
    out["H8"] = {"1.0_tau0.15": power("H8", None, "a8"), "1.0_tau0.3": power("H8", None, "a8", {"tau": 0.3}),
                 "1.2_tau0.15": power("H8", None, "a8", {"true": 1.2}),
                 "1.2_tau0.3": power("H8", None, "a8", {"true": 1.2, "tau": 0.3})}
    return out


def power_at_plan(n_M: int, h2_rows: int = 5) -> dict:
    """Power at the fixed n for the plan amendment (alpha/8 and alpha/3), at the HYPOTHESIS design points."""
    four = h2_rows == 4
    return {
        "n_M": n_M,
        "H2_D0": {"a8": power("H2", n_M, "a8", {"four_rows": four}), "a3": power("H2", n_M, "a3", {"four_rows": four})},
        "H2_D-2": {"a8": power("H2", n_M, "a8", {"true": -0.02, "four_rows": four}),
                   "a3": power("H2", n_M, "a3", {"true": -0.02, "four_rows": four})},
        "H3_vendor_gap": {"a8": power("H3", n_M, "a8"), "a3": power("H3", n_M, "a3")},
        "H3_-3": {"a8": power("H3", n_M, "a8", {"true": -0.03}), "a3": power("H3", n_M, "a3", {"true": -0.03})},
        "H7_0": {"a8": power("H7", n_M, "a8"), "a3": power("H7", n_M, "a3")},
        "H7_-1": {"a8": power("H7", n_M, "a8", {"true": -0.01}), "a3": power("H7", n_M, "a3", {"true": -0.01})},
        "H4_+12": {"a8": power("H4", None, "a8"), "a3": power("H4", None, "a3")},
        "H6_-15": {"a8": power("H6", None, "a8", {"true": -0.15}), "a3": power("H6", None, "a3", {"true": -0.15})},
        "assumptions": "HYPOTHESIS power tables (SDs and true effects as pre-registered)",
    }


# --------------------------------------------------------------------------
# Budget: the HYPOTHESIS ladder and scenario tables ("Sessions & budget")
# --------------------------------------------------------------------------

# Planning assumptions (nominal, pessimistic, adverse). Mean completion tokens and
# decode rates are the HYPOTHESIS "Planning assumptions" table; prompt tokens per
# item, per-arm prefill rates and the effort none/low/medium lengths are the
# design arithmetic behind its ladder and scenario tables.
PLANNING = {
    "tokens": {
        "K": {"GPQA": (5000, 8000, 10000), "MMLU": (2000, 3000, 3500), "IFB": (1500, 2500, 3000),
              "CB": (600, 1000, 1200), "CBF": (400, 800, 1000), "NEG": (700, 1000, 1200), "FC": (800, 1200, 1400),
              "AIME": (14000, 20000, 24000), "NONE_GPQA": (300, 600, 800), "NONE_MMLU": (250, 500, 600),
              "NONE_IFB": (500, 800, 900), "NONE_CB": (60, 150, 200), "LOW": (2000, 3000, 3500),
              "MED": (3500, 5000, 6000)},
        "G": {"GPQA": (4000, 6000, 7000), "MMLU": (1500, 2500, 2800), "IFB": (1200, 2000, 2200),
              "CB": (600, 1000, 1000), "CBF": (400, 800, 900), "NEG": (600, 1000, 1000), "FC": (600, 1000, 1000),
              "AIME": (12000, 15000, 18000)},
        "Q36": {"GPQA": (7000, 9000, 10000), "MMLU": (3000, 3500, 4000), "IFB": (2000, 3000, 3200),
                "CB": (900, 1200, 1300), "CBF": (600, 1000, 1100), "NEG": (900, 1200, 1300), "FC": (900, 1200, 1300),
                "AIME": (15000, 18000, 20000)},
        "Q38": {"GPQA": (7000, 9000, 10000)},
    },
    "prompt_tokens": {"GPQA": 300, "MMLU": 400, "IFB": 150, "CB": 50, "CBF": 60, "NEG": 1500, "FC": 1500,
                      "AIME": 200, "NONE_GPQA": 300, "NONE_MMLU": 400, "NONE_IFB": 150, "NONE_CB": 50,
                      "LOW": 300, "MED": 300},
    "decode_tps": {"K8": (180, 120, 100), "K4": (240, 160, 140), "G8": (220, 150, 140), "Q36-8": (250, 160, 150),
                   "Q38-8": (100, 60, 55)},
    "prefill_tps": {"K8": (1500, 800, 700), "K4": (1700, 900, 800), "G8": (1500, 800, 700),
                    "Q36-8": (1800, 900, 800), "Q38-8": (500, 300, 280)},
    "family": {"K8": "K", "K4": "K", "G8": "G", "Q36-8": "Q36", "Q38-8": "Q38"},
    "margin": 1.15,
    "session_cap_h": 16.0,
    "main_cap_h": 31.0,
    "total_h": 40.0,
    "b4_hours": (1.2, 1.4, 1.6),
    "s1_hours": {"nominal": 5.4, "pessimistic": 9.1, "adverse": 9.6},
}
SCENARIOS = ("nominal", "pessimistic", "adverse")

LADDER = [("P0", 588, True, True, True), ("P1", 588, False, True, True), ("P2", 504, False, True, True),
          ("P3", 406, False, True, True), ("P4", 350, False, True, True), ("P5", 294, False, True, True),
          ("P6", 252, False, True, True), ("P7", 196, False, True, True), ("P8", 196, False, False, True),
          ("P9", 154, False, False, True), ("P10", 154, False, False, False)]

TIER_B = [("B1", [("K8", "AIME", 60)]), ("B2", [("G8", "AIME", 60), ("Q36-8", "AIME", 60)]),
          ("B3", [("G8", "GPQA", 198), ("Q36-8", "GPQA", 198)]), ("B4", None),
          ("B5", [("K8", "NONE_GPQA", 396), ("K8", "NONE_MMLU", 600), ("K8", "NONE_IFB", 300), ("K8", "NONE_CB", 400)]),
          ("B6", [("G8", "GPQA", 198), ("Q36-8", "GPQA", 198)]), ("B7", [("K8", "LOW", 198), ("K8", "MED", 198)]),
          ("B8", [("K4", "GPQA", 396)]), ("B9", [("Q38-8", "GPQA", 198)]), ("B10", [("K8", "GPQA", 396)])]


def cell_hours(arm: str, task: str, n: int, s: int, P: dict = PLANNING) -> float:
    """Raw (unmargined) hours of a cell: decode + prefill, scenario index s."""
    tok = P["tokens"][P["family"][arm]][task][s] * n
    return tok / P["decode_tps"][arm][s] / 3600 + P["prompt_tokens"][task] * n / P["prefill_tps"][arm][s] / 3600


def tier_a_queue(n_M: int, negfc: bool, forced: bool, k8_gpqa_de: bool = True) -> list[tuple]:
    """HYPOTHESIS "Tier A cells, in queue order"."""
    q = [("K8", "GPQA", 198)] + ([("K8", "GPQA", 198)] if k8_gpqa_de else [])
    q += [("K8", "MMLU", n_M), ("K8", "MMLU", n_M), ("K8", "IFB", 300), ("K8", "CB", 400)]
    q += [("G8", "MMLU", n_M), ("G8", "MMLU", n_M), ("Q36-8", "MMLU", n_M), ("Q36-8", "MMLU", n_M),
          ("Q36-8", "IFB", 300), ("G8", "IFB", 300), ("G8", "CB", 400), ("Q36-8", "CB", 400)]
    q += [("K4", "MMLU", n_M), ("K4", "MMLU", n_M), ("K4", "IFB", 300), ("K4", "CB", 400)]
    if forced:
        q += [("K8", "CBF", 400), ("G8", "CBF", 400), ("Q36-8", "CBF", 400)]
    if negfc:
        q += [(a, t, n) for a in ("K8", "G8", "Q36-8") for t, n in (("NEG", 300), ("FC", 100))]
    return q


def _hours(cell, s, P):
    if len(cell) == 4:            # a fixed-hours cell (B4, the ladder)
        return cell[3] * P["margin"]
    return cell_hours(cell[0], cell[1], cell[2], s, P) * P["margin"]


def _total(cells, s, P):
    return sum(_hours(c, s, P) for c in cells)


def _split(cells, s, P):
    """Greedy split of the ordered queue at cell boundaries into at most two sessions."""
    cap = P["session_cap_h"]
    segs = [0.0]
    for c in cells:
        h = _hours(c, s, P)
        if h > cap:
            return None
        if segs[-1] + h > cap:
            segs.append(0.0)
        segs[-1] += h
    return segs if len(segs) <= 2 else None


def budget(assumptions: dict | None = None) -> dict:
    """Reproduce HYPOTHESIS "Ladder totals", the Tier-B table and "Plan picked by the rule"."""
    P = {**PLANNING, **(assumptions or {})}
    ladder_totals = {name: [round(_total(tier_a_queue(nM, nf, fo, de), s, P), 1) for s in range(3)]
                     for name, nM, nf, fo, de in LADDER}
    tier_b_totals = {}
    for bn, cells in TIER_B:
        if cells is None:
            tier_b_totals[bn] = list(P["b4_hours"])
        else:
            tier_b_totals[bn] = [round(_total(cells, s, P), 1) for s in range(3)]
    scenarios = {}
    for s, label in enumerate(SCENARIOS):
        S1 = P["s1_hours"][label]
        Bm = min(P["main_cap_h"], P["total_h"] - S1)
        pick = None
        for name, nM, nf, fo, de in LADDER:
            q = tier_a_queue(nM, nf, fo, de)
            if _total(q, s, P) <= Bm and _split(q, s, P):
                pick = (name, nM, fo, nf, q)
                break
        if pick is None:
            scenarios[label] = {"S1": S1, "B_main": Bm, "plan": "STOP"}
            continue
        name, nM, fo, nf, q = pick
        added = []
        for bn, cells in TIER_B:
            extra = [("ladder", "B4", 0, P["b4_hours"][s])] if cells is None else cells
            q2 = q + extra
            if _total(q2, s, P) <= Bm and _split(q2, s, P):
                q = q2
                added.append(bn)
        sp = _split(q, s, P)
        main = _total(q, s, P)
        scenarios[label] = {"S1": S1, "B_main": Bm, "plan": name, "n_M": nM, "forced": fo, "negfc": nf,
                            "tier_b": added, "S2": round(sp[0], 1), "S3": round(sp[1] if len(sp) > 1 else 0.0, 1),
                            "main": round(main, 1), "run_total": round(S1 + main, 1)}
    return {"ladder_totals": ladder_totals, "tier_b_totals": tier_b_totals, "scenarios": scenarios}


if __name__ == "__main__":
    print(json.dumps({"power": power_tables(), "budget": budget()}, indent=1, default=str))
