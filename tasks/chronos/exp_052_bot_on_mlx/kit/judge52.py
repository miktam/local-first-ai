#!/usr/bin/env python3
"""exp_052 Amendment 1: blind judge packets for all 120 pairs, and the unblinding.

    python3.14 -I kit/judge52.py build  data/frozen.jsonl runs/<ts>     # judge/batch_{1..4}.md + judge/KEY.json
    python3.14 -I kit/judge52.py score  data/frozen.jsonl runs/<ts>     # reads judge/verdicts_{1..4}.jsonl
"""
import json
import random
import sys
from math import comb
from pathlib import Path


def final(path: Path) -> dict:
    last = {}
    for r in map(json.loads, open(path)):
        last[r["id"]] = r
    return last


def build(frozen: dict, run: Path):
    G, M = final(run / "G" / "answers.jsonl"), final(run / "M" / "answers.jsonl")
    ids = sorted(i for i in frozen if i in G and i in M)
    rng = random.Random(5201)
    rng.shuffle(ids)
    jd = run / "judge"
    jd.mkdir(exist_ok=True)
    key = {}
    for b in range(4):
        part = ids[b * 30:(b + 1) * 30]
        md = [f"# exp_052 judge batch {b + 1}: 30 pairs\n"]
        for k, i in enumerate(part, 1):
            n = b * 30 + k
            a_is_g = rng.random() < 0.5
            key[str(n)] = {"id": i, "A": "G" if a_is_g else "M", "B": "M" if a_is_g else "G"}
            q = frozen[i]
            ga, ma = G[i]["prefix"] + G[i]["answer"], M[i]["prefix"] + M[i]["answer"]
            A, B = (ga, ma) if a_is_g else (ma, ga)
            system = q["messages"][0]["content"]
            user = q["messages"][-1]["content"]
            md += [f"\n=== PAIR {n} ===\n", f"Language the user writes in: {q['lang']}\n",
                   f"THE BOT'S RULES (its system prompt):\n<<<\n{system}\n>>>\n",
                   f"WHAT THE BOT RECEIVED (retrieved context and the user's question):\n<<<\n{user}\n>>>\n",
                   f"ANSWER A:\n<<<\n{A}\n>>>\n", f"ANSWER B:\n<<<\n{B}\n>>>\n"]
        (jd / f"batch_{b + 1}.md").write_text("\n".join(md))
    (jd / "KEY.json").write_text(json.dumps(key, indent=2) + "\n")
    print("built", jd)


def sign_p(x: int, y: int) -> float:
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)


def score(frozen: dict, run: Path):
    jd = run / "judge"
    key = json.loads((jd / "KEY.json").read_text())
    pref, wrong, rows = {"G": 0, "M": 0, "=": 0}, {"G": 0, "M": 0}, []
    by_intent = {}
    for b in range(1, 5):
        for v in map(json.loads, open(jd / f"verdicts_{b}.jsonl")):
            k = key[str(v["pair"])]
            p = "=" if v["better"] == "=" else k[v["better"]]
            pref[p] += 1
            w = [k[x] for x in v.get("wrong", [])]
            for x in w:
                wrong[x] += 1
            it = frozen[k["id"]]["intent_routed"]
            by_intent.setdefault(it, {"G": 0, "M": 0, "=": 0})[p] += 1
            rows.append({"pair": v["pair"], "id": k["id"], "preferred": p, "wrong": w, "why": v.get("why", "")})
    ok = pref["M"] >= pref["G"] - 8 and wrong["M"] <= wrong["G"] + 4
    res = {"n": len(rows), "preferred": pref, "wrong": wrong, "sign_test_p": sign_p(pref["G"], pref["M"]),
           "by_intent": by_intent, "verdict": "pass" if ok else "fail"}
    (jd / "score.json").write_text(json.dumps({**res, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    fz = {r["id"]: r for r in map(json.loads, open(sys.argv[2])) if r["static"] is None}
    (build if sys.argv[1] == "build" else score)(fz, Path(sys.argv[3]))
