#!/usr/bin/env python3
"""exp_052: score H1 and H2 from a run, and H3 once Andrei's marks are in runs/<ts>/READ30_marks.json
({"1": {"pick": "A"|"B"|"=", "wrong": ["A"]}, ...}).

    python3.14 -I kit/score52.py data/frozen.jsonl runs/<ts>
"""
import json
import statistics
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks52 as C  # noqa: E402


def final(path: Path) -> dict:
    last = {}
    for r in map(json.loads, open(path)):
        last[r["id"]] = r
    return last


def mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    frozen = {r["id"]: r for r in map(json.loads, open(sys.argv[1])) if r["static"] is None}
    run = Path(sys.argv[2])
    G, M = final(run / "G" / "answers.jsonl"), final(run / "M" / "answers.jsonl")
    ids = sorted(i for i in frozen if i in G and i in M)
    rows = []
    for i in ids:
        q = frozen[i]
        ctx = q["messages"][-1]["content"] if len(q["messages"]) > 1 else ""
        row = {"id": i, "lang": q["lang"], "intent": q["intent_routed"]}
        for arm, d in (("G", G), ("M", M)):
            r = d[i]
            row[arm] = {"wait_s": r["user_wait_s"], "gen_tokens": r.get("gen_tokens"),
                        "thinking_chars": r.get("thinking_chars"), "contaminated": r["contaminated"],
                        **C.check(r["prefix"] + r["answer"], q["lang"], q["text"], ctx, r.get("finish"))}
        rows.append(row)
    clean = [r for r in rows if not r["G"]["contaminated"] and not r["M"]["contaminated"]]
    wg, wm = [r["G"]["wait_s"] for r in clean], [r["M"]["wait_s"] for r in clean]
    ratio = statistics.median(wm) / statistics.median(wg)
    fg, fm = sum(r["G"]["flagged"] for r in rows), sum(r["M"]["flagged"] for r in rows)
    b = sum(r["G"]["flagged"] and not r["M"]["flagged"] for r in rows)
    c = sum(r["M"]["flagged"] and not r["G"]["flagged"] for r in rows)
    res = {"n_reached": len(frozen), "n_compared": len(rows), "n_clean_for_time": len(clean),
           "H1_mlx_10pct_faster": {"median_wait_G": statistics.median(wg), "median_wait_M": statistics.median(wm),
                                   "ratio_M_over_G": ratio, "p90_G": statistics.quantiles(wg, n=10)[-1],
                                   "p90_M": statistics.quantiles(wm, n=10)[-1],
                                   "M_faster_pairs": sum(m < g for g, m in zip(wg, wm)),
                                   "verdict": "pass" if ratio <= 0.90 else "fail"},
           "H2_checks_not_worse": {"flagged_G": fg, "flagged_M": fm, "only_G": b, "only_M": c,
                                   "mcnemar_p_two_sided": mcnemar_two_sided(b, c),
                                   "verdict": "pass" if fm <= fg + 3 else "fail"}}
    by = {}
    for arm in ("G", "M"):
        for r in rows:
            for k in r[arm]["violations"]:
                by.setdefault(arm, {}).setdefault(k, 0)
                by[arm][k] += 1
    res["violations_by_check"] = by
    res["gen_tokens_median"] = {a: statistics.median(r[a]["gen_tokens"] for r in rows if r[a]["gen_tokens"])
                                for a in ("G", "M")}
    marks = run / "READ30_marks.json"
    if marks.exists():
        key, mk = json.loads((run / "READ30_key.json").read_text()), json.loads(marks.read_text())
        better = {"G": 0, "M": 0, "=": 0}
        wrong = {"G": 0, "M": 0}
        for n, m in mk.items():
            k = key[n]
            better["=" if m["pick"] == "=" else k[m["pick"]]] += 1
            for w in m.get("wrong", []):
                wrong[k[w]] += 1
        ok = better["M"] >= better["G"] - 2 and wrong["M"] <= wrong["G"] + 1
        res["H3_read30_not_worse"] = {"preferred": better, "wrong": wrong, "verdict": "pass" if ok else "fail"}
    else:
        res["H3_read30_not_worse"] = {"verdict": "waiting for READ30_marks.json"}
    (run / "score.json").write_text(json.dumps({**res, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    for k, v in res.items():
        if k.startswith("H"):
            print(k, v["verdict"], {x: y for x, y in v.items() if x != "verdict"})
    print("violations", by)


if __name__ == "__main__":
    main()
