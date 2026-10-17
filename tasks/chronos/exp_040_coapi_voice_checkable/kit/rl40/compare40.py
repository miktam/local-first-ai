#!/usr/bin/env python3
"""
exp_040 — the code-scored hypotheses on v3 (HYPOTHESIS.md H1, H3, H4, D1). No model call; reads run_arm40 outputs.

  python rl40/compare40.py RUN_X RUN_Y [--json OUT]      # X = treatment (R3), Y = control (B0i); paired over v3 rows

H1  exact match X − Y, paired over rows. Gain shown iff the stratified paired bootstrap 95 % lower bound > 0
    (10,000 resamples, seed 40, strata = language × kind) AND the exact one-sided McNemar p < 0.05.
    The robustness cell is the same call on R3q8 vs B0q8.
H3  contrast pairs with both twins right: X − Y over pairs, same two tests over pairs; registered effect ≥ +0.10.
H4  abstention rows (kind abstain + oos): rate of final lines that abstain, X − Y ≥ +0.20;
    false abstention = abstaining final lines on answerable rows, X ≤ 0.05.
Rows are matched by id; the two runs must be on the same set file (sha256) and cover the same ids.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

B, SEED, ALPHA = 10_000, 40, 0.05
H3_MIN, H4_RISE, H4_FALSE_MAX = 0.10, 0.20, 0.05


def load(run: Path) -> tuple[dict, dict]:
    s = json.loads((run / "summary.json").read_text())
    rows = {r["id"]: r for r in (json.loads(l) for l in (run / "rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    return s, rows


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(X ≥ b) for X ~ Binomial(b + c, 1/2): exact, one-sided (treatment better)."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def paired(units: list[tuple[str, int, int]]) -> dict:
    """units = (stratum, x, y) with x, y ∈ {0, 1}. Mean difference, stratified bootstrap CI, McNemar."""
    n = len(units)
    diff = sum(x - y for _, x, y in units) / n
    strata = defaultdict(list)
    for s, x, y in units:
        strata[s].append(x - y)
    rng = random.Random(SEED)
    boots = []
    for _ in range(B):
        tot = 0
        for ds in strata.values():
            tot += sum(rng.choice(ds) for _ in ds)
        boots.append(tot / n)
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    b = sum(1 for _, x, y in units if x and not y)
    c = sum(1 for _, x, y in units if y and not x)
    p = mcnemar_one_sided(b, c)
    return {"n": n, "x": round(sum(x for _, x, _ in units) / n, 4), "y": round(sum(y for _, _, y in units) / n, 4),
            "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)], "discordant": {"x_only": b, "y_only": c},
            "mcnemar_p_one_sided": round(p, 5), "shown": lo > 0 and p < ALPHA}


def compare(run_x: Path, run_y: Path) -> dict:
    sx, rx = load(run_x)
    sy, ry = load(run_y)
    if sx["sha256"]["set"] != sy["sha256"]["set"] or set(rx) != set(ry):
        sys.exit("REFUSED: the two runs are not on the same set / rows")
    ids = sorted(rx)
    right = lambda r: int(r["verdict"] == "right")
    h1 = paired([(f"{rx[i]['lang']}/{rx[i]['kind']}", right(rx[i]), right(ry[i])) for i in ids])

    pairs, seen = [], set()
    for i in ids:
        t = rx[i].get("twin_of")
        if t and t in rx and (t, i) not in seen:
            seen.add((i, t))
            pairs.append((rx[i]["lang"], right(rx[i]) * right(rx[t]), right(ry[i]) * right(ry[t])))
    h3 = paired(pairs) if pairs else None
    if h3:
        h3["meets_registered_effect"] = h3["shown"] and h3["diff"] >= H3_MIN

    abst = [i for i in ids if rx[i]["kind"] in ("abstain", "oos")]
    ans = [i for i in ids if rx[i]["kind"] not in ("abstain", "oos")]
    rate = lambda rows, sel: round(sum(rows[i]["abstained"] for i in sel) / len(sel), 4) if sel else None
    h4 = {"abstain_rows": len(abst), "abstain_rate": {"x": rate(rx, abst), "y": rate(ry, abst)},
          "answerable_rows": len(ans), "false_abstention": {"x": rate(rx, ans), "y": rate(ry, ans)},
          "abstain_paired": paired([(rx[i]["lang"], int(rx[i]["abstained"]), int(ry[i]["abstained"])) for i in abst]) if abst else None}
    h4["rise"] = round(h4["abstain_rate"]["x"] - h4["abstain_rate"]["y"], 4) if abst else None
    h4["pass"] = bool(abst) and h4["rise"] >= H4_RISE and h4["false_abstention"]["x"] <= H4_FALSE_MAX
    return {"x": {"arm": sx["arm"], "run": run_x.name, "exact": sx["exact"]}, "y": {"arm": sy["arm"], "run": run_y.name, "exact": sy["exact"]},
            "set_sha256": sx["sha256"]["set"], "H1_exact": h1, "H3_pairs_both_right": h3, "H4_abstention": h4,
            "rules": {"bootstrap": f"{B} stratified paired resamples, seed {SEED}, strata language × kind", "mcnemar": "exact, one-sided",
                      "alpha": ALPHA, "H3_min": H3_MIN, "H4_rise": H4_RISE, "H4_false_max": H4_FALSE_MAX}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_x")
    ap.add_argument("run_y")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = compare(Path(a.run_x), Path(a.run_y))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
