#!/usr/bin/env python3
"""exp_053: wait and code checks per arm, each candidate against the live build (G4); with judge/score.json
present, the registered decision.

    python3.14 -I kit/score53.py data/frozen.jsonl runs/<ts>
"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks53 as C  # noqa: E402

ARMS = ["G4", "GQ", "G8"]


def final(path: Path) -> dict:
    last = {}
    for r in map(json.loads, open(path)):
        last[r["id"]] = r
    return last


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    frozen = {r["id"]: r for r in map(json.loads, open(sys.argv[1])) if r["static"] is None}
    run = Path(sys.argv[2])
    A = {a: final(run / a / "answers.jsonl") for a in ARMS}
    ids = sorted(i for i in frozen if all(i in A[a] for a in ARMS))
    rows = []
    for i in ids:
        q = frozen[i]
        ctx = q["messages"][-1]["content"] if len(q["messages"]) > 1 else ""
        row = {"id": i, "lang": q["lang"], "intent": q["intent_routed"]}
        for a in ARMS:
            r = A[a][i]
            row[a] = {"wait_s": r["user_wait_s"], "gen_tokens": r.get("gen_tokens"), "contaminated": r["contaminated"],
                      "error": r["answer"] == C.LLM_ERROR,
                      **C.check(r["prefix"] + r["answer"], q["lang"], q["text"], ctx, r.get("finish"))}
        rows.append(row)
    res = {"n": len(rows), "arms": {}}
    for a in ARMS:
        clean = [r for r in rows if not any(r[x]["contaminated"] for x in ARMS)]
        w = [r[a]["wait_s"] for r in clean]
        res["arms"][a] = {"median_wait_s": statistics.median(w), "p90_wait_s": statistics.quantiles(w, n=10)[-1],
                          "median_gen_tokens": statistics.median(r[a]["gen_tokens"] for r in rows if r[a]["gen_tokens"]),
                          "flagged": sum(r[a]["flagged"] for r in rows), "errors": sum(r[a]["error"] for r in rows),
                          "by_check": {k: sum(k in r[a]["violations"] for r in rows) for k in ("V1", "V2", "V3", "V4", "V5")}}
    g4 = res["arms"]["G4"]
    for a in ("GQ", "G8"):
        x = res["arms"][a]
        x["wait_ratio_to_G4"] = x["median_wait_s"] / g4["median_wait_s"]
        b = sum(r["G4"]["flagged"] and not r[a]["flagged"] for r in rows)
        c = sum(r[a]["flagged"] and not r["G4"]["flagged"] for r in rows)
        x["flags_only_G4"], x["flags_only_this"], x["mcnemar_p"] = b, c, mcnemar(b, c)
    js = run / "judge" / "score.json"
    if js.exists():
        j = json.loads(js.read_text())
        dec = {}
        for a in ("G8", "GQ"):
            x, q = res["arms"][a], j[f"{a}_vs_G4"]
            dec[a] = {"H_quality": q["quality_better"], "speed_ok": x["wait_ratio_to_G4"] <= 1.10,
                      "checks_ok": x["flagged"] <= g4["flagged"] + 3}
            dec[a]["adopt"] = all(dec[a].values())
        cands = [a for a in ("GQ", "G8") if dec[a]["adopt"]]
        res["decision"] = {"per_candidate": dec,
                           "adopt": min(cands, key=lambda a: res["arms"][a]["median_wait_s"]) if cands else "keep G4"}
    (run / "score.json").write_text(json.dumps({**res, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
