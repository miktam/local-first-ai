#!/usr/bin/env python3
"""exp_055: score every arm present in a run against the construction labels.

    python3.14 kit/score55.py runs/<ts>
"""
import json
import math
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
NAME_ITEMS = {"I-5e3d4b", "I-9ad8cd", "I-d2ff0e"}           # her name and its variants: every message uses them
FLAGGED = {"suspicious", "likely-scam"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def mcnemar_one_sided(b, c):
    """P(at least b of b+c discordant pairs favour the first arm) under a fair coin."""
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, k) for k in range(b, n + 1)) / 2 ** n


def load(run: Path, arm: str) -> dict:
    p = run / arm / "results.jsonl"
    return {json.loads(l)["id"]: json.loads(l) for l in open(p)} if p.exists() else {}


SAME_ITEM = {"C-69db50": "A-f3776b"}   # Amendment 1: her booking page is both a channel and an artefact (one URL)


def reported_items(r: dict, strong_only=False) -> set:
    if "hooks" in r:
        got = {h["item_id"] for h in r["hooks"] if not strong_only or h["level"] == "strong"}
    else:
        got = set(r.get("used_items") or [])
    return got | {SAME_ITEM[i] for i in got if i in SAME_ITEM}


def arm_metrics(res: dict, man: dict) -> dict:
    lures = [i for i, m in man.items() if m["kind"] == "lure"]
    benign = [i for i, m in man.items() if m["kind"] == "benign"]
    caught = [i for i in lures if res.get(i, {}).get("band") in FLAGGED]
    alarms = [i for i in benign if res.get(i, {}).get("band") in FLAGGED]
    severe = [i for i in benign if res.get(i, {}).get("band") == "likely-scam"]
    unanswered = [i for i in man if res.get(i, {}).get("band") is None]
    cr, far = len(caught) / len(lures), len(alarms) / len(benign)
    att_n = att_hit = 0
    recall = []
    for i in lures:
        declared = set(man[i]["used_items"]) | {SAME_ITEM[x] for x in man[i]["used_items"] if x in SAME_ITEM}
        dist = declared - NAME_ITEMS - {x for x in declared if x.startswith("C-")}
        if not dist:
            continue
        rep = reported_items(res.get(i, {}))
        att_n += 1
        att_hit += bool(rep & dist)
        recall.append(len(rep & dist) / len(dist))
    return {"lures": len(lures), "benign": len(benign), "caught": len(caught), "catch_rate": round(cr, 3),
            "catch_ci": wilson(len(caught), len(lures)), "false_alarms": len(alarms), "false_alarm_rate": round(far, 3),
            "false_alarm_ci": wilson(len(alarms), len(benign)), "likely_scam_on_genuine": len(severe),
            "balanced_accuracy": round((cr + 1 - far) / 2, 3), "unanswered": len(unanswered),
            "attribution_lures_with_items": att_n, "attribution_named_one": att_hit,
            "attribution_rate": round(att_hit / att_n, 3) if att_n else None,
            "attribution_mean_recall": round(sum(recall) / len(recall), 3) if recall else None,
            "_caught": set(caught), "_alarms": set(alarms)}


def breakdown(res: dict, man: dict, key) -> dict:
    out = {}
    for i, m in man.items():
        k = key(m)
        if k is None:
            continue
        o = out.setdefault(k, {"n": 0, "flagged": 0})
        o["n"] += 1
        o["flagged"] += res.get(i, {}).get("band") in FLAGGED
    return out


def main():
    run = Path(sys.argv[1])
    man = json.loads((HERE / "data/MANIFEST.json").read_text())
    arms = {a: load(run, a) for a in ("P", "R", "G", "O")}
    arms = {a: r for a, r in arms.items() if r}
    met = {a: arm_metrics(r, man) for a, r in arms.items()}
    res = {"arms": {a: {k: v for k, v in m.items() if not k.startswith("_")} for a, m in met.items()}}
    lures = [i for i, m in man.items() if m["kind"] == "lure"]
    if "P" in met:
        p = met["P"]
        res["H1_parry_catches_lures"] = {"catch": p["catch_rate"], "false_alarms": p["false_alarm_rate"],
                                         "verdict": "pass" if p["catch_rate"] >= 0.80 and p["false_alarm_rate"] <= 0.15
                                         else "fail"}
        res["H2_parry_names_the_exploited_items"] = {"rate": p["attribution_rate"],
                                                     "verdict": "pass" if (p["attribution_rate"] or 0) >= 0.70 else "fail"}
    if "P" in met and "G" in met:
        res["H3_glue_not_worse_than_local_model_alone"] = {
            "P": met["P"]["balanced_accuracy"], "G": met["G"]["balanced_accuracy"],
            "verdict": "pass" if met["P"]["balanced_accuracy"] >= met["G"]["balanced_accuracy"] else "fail"}
    if "P" in met and "O" in met:
        gap = round(met["O"]["balanced_accuracy"] - met["P"]["balanced_accuracy"], 3)
        res["H4_sovereignty_costs_at_most_0.10"] = {"O_minus_P": gap, "verdict": "pass" if gap <= 0.10 else "fail"}
    if "P" in met and "R" in met:
        b = len(met["P"]["_caught"] - met["R"]["_caught"])
        c = len(met["R"]["_caught"] - met["P"]["_caught"])
        extra_alarms = met["P"]["false_alarms"] - met["R"]["false_alarms"]
        pv = mcnemar_one_sided(b, c)
        res["H5_the_model_step_earns_its_place"] = {
            "lures_caught_only_with_model": b, "only_without": c, "p_one_sided": round(pv, 4),
            "extra_false_alarms": extra_alarms, "verdict": "pass" if pv < 0.05 and extra_alarms <= 3 else "fail"}
    res["breakdowns"] = {a: {
        "lure_by_tactic_group": breakdown(r, man, lambda m: (m["tactic"].split()[0] if m["kind"] == "lure" else None)),
        "lure_by_ask": breakdown(r, man, lambda m: m["ask"] if m["kind"] == "lure" else None),
        "lure_by_domain_age": breakdown(r, man, lambda m: (("old" if any(c < "2026-01-01" for c in m["domains"].values())
                                                           else "new") if m["domains"] else "no domain")
                                        if m["kind"] == "lure" else None),
        "genuine_by_domain_age": breakdown(r, man, lambda m: (("young" if any(c >= "2026-06-01" for c in m["domains"].values())
                                                              else "old") if m["domains"] else "no domain")
                                           if m["kind"] == "benign" else None),
        "by_channel_and_kind": breakdown(r, man, lambda m: f"{m['channel']}/{m['kind']}")} for a, r in arms.items()}
    (run / "score.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    for k, v in res.items():
        if k.startswith("H"):
            print(k, v["verdict"], {x: y for x, y in v.items() if x != "verdict"})
    for a, m in res["arms"].items():
        print(a, {k: m[k] for k in ("catch_rate", "false_alarm_rate", "balanced_accuracy", "attribution_rate",
                                    "unanswered")})


if __name__ == "__main__":
    main()
