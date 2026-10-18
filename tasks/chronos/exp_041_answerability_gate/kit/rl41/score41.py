#!/usr/bin/env python3
"""
exp_041 — score the gates on v3 (HYPOTHESIS H1–H4, D1). No model call; reads the frozen thresholds, each gate's
v3 probabilities, and exp_040's B0i and A13 v3 rows.

  python rl41/score41.py [--json OUT]

Gated system = gate + B0i: a row the gate blocks (P(answerable) < τ) becomes an abstention — right on v3's
abstention and out-of-scope rows, wrong (a false abstention) on answerable rows; every other row keeps B0i's
exp_040 verdict. Paired statistics are exp_040's (rl40/compare40.paired: stratified bootstrap, seed 40, exact
one-sided McNemar).
  H1  primary gate: abstention on the 38 abstention rows rises ≥ 0.20 over B0i AND McNemar p < 0.05
  H2  primary gate: false abstention on the 112 answerable rows ≤ 0.05
  H3  primary gate: exact match over 150, paired bootstrap 95 % lower bound of (gated − B0i) > −0.02
  H4  primary gate: dev ECE ≤ 0.10 (from thresholds.json)
  D1  each gate over A13 (the deployed glue) instead of B0i; descriptive
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV / "rl40")]
import compare40 as C   # noqa: E402

H1_RISE, H2_MAX, H3_FLOOR, H4_MAX = 0.20, 0.05, -0.02, 0.10
ABST = ("abstain", "oos")


def base_rows(arm: str) -> dict:
    d = glob.glob(str(CV / f"runs/exp040_{arm}_v3_*"))
    assert len(d) == 1, f"expected one exp040 {arm} v3 run, found {d}"
    return {r["id"]: r for r in (json.loads(l) for l in open(Path(d[0]) / "rows.jsonl"))}


def gated(base: dict, probs: dict, tau: float) -> dict:
    out = {}
    for i, r in base.items():
        if probs[i] < tau:
            out[i] = {**r, "abstained": True, "verdict": "right" if r["kind"] in ABST else "wrong", "gate_blocked": True}
        else:
            out[i] = {**r, "gate_blocked": False}
    return out


def evaluate(x: dict, y: dict) -> dict:
    ids = sorted(y)
    right = lambda r: int(r["verdict"] == "right")
    ab = [i for i in ids if y[i]["kind"] in ABST]
    an = [i for i in ids if y[i]["kind"] not in ABST]
    h1 = C.paired([(y[i]["lang"], int(x[i]["abstained"]), int(y[i]["abstained"])) for i in ab])
    exact = C.paired([(f"{y[i]['lang']}/{y[i]['kind']}", right(x[i]), right(y[i])) for i in ids])
    fa_x = sum(x[i]["abstained"] for i in an) / len(an)
    return {"abstention": {"x": h1["x"], "y": h1["y"], "rise": h1["diff"], "ci95": h1["ci95"],
                           "mcnemar_p": h1["mcnemar_p_one_sided"], "discordant": h1["discordant"]},
            "false_abstention": {"x": round(fa_x, 4), "y": round(sum(y[i]["abstained"] for i in an) / len(an), 4),
                                 "n": len(an), "x_count": sum(x[i]["abstained"] for i in an)},
            "exact": {"x": exact["x"], "y": exact["y"], "diff": exact["diff"], "ci95": exact["ci95"]},
            "blocked": sum(x[i].get("gate_blocked", False) for i in ids)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    th = json.loads((CV / "data/gate41/thresholds.json").read_text())
    b0, a13 = base_rows("B0i"), base_rows("A13")
    res = {"primary": th["primary"], "gates": {}}
    for gate, g in th["gates"].items():
        path = CV / "runs" / f"exp041_v3_{gate}" / "probs.jsonl"
        if not g.get("complete") or not path.exists():
            continue
        probs = {json.loads(l)["id"]: json.loads(l)["p"] for l in path.read_text().splitlines() if l.strip()}
        if set(probs) != set(b0):
            res["gates"][gate] = {"complete": False}
            continue
        main_ = evaluate(gated(b0, probs, g["tau"]), b0)
        res["gates"][gate] = {"tau": g["tau"], "over_B0i": main_, "D1_over_A13": evaluate(gated(a13, probs, g["tau"]), a13),
                              "dev_ece": g["ece"]}
    p = th["primary"]
    if p in res["gates"] and "over_B0i" in res["gates"][p]:
        m = res["gates"][p]["over_B0i"]
        h = {"H1": m["abstention"]["rise"] >= H1_RISE and m["abstention"]["mcnemar_p"] < 0.05,
             "H2": m["false_abstention"]["x"] <= H2_MAX,
             "H3": m["exact"]["ci95"][0] > H3_FLOOR,
             "H4": res["gates"][p]["dev_ece"] <= H4_MAX}
        verdict = ("No gate earns its place" if not (h["H1"] and h["H2"]) else
                   "Abstains, but costs answers" if not h["H3"] else "Gate adopted for the glue")
        res.update({"hypotheses": h, "verdict": verdict,
                    "rules": {"H1_rise": H1_RISE, "H2_max": H2_MAX, "H3_floor": H3_FLOOR, "H4_max": H4_MAX}})
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
