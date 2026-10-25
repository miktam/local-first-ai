#!/usr/bin/env python3
"""exp_053: blind judge packets and unblinding. Two comparisons against the live build: G8 vs G4 and GQ vs G4,
120 pairs each. Each comparison gets its own shuffle and its own random A/B order per pair (seed 5301 and 5302).
The 240 pairs are mixed across 8 batches of 30, so a judge cannot tell which comparison a pair belongs to.

    python3.14 -I kit/judge53.py build data/frozen.jsonl runs/<ts>    # judge/batch_{1..8}.md + judge/KEY.json
    python3.14 -I kit/judge53.py score data/frozen.jsonl runs/<ts>    # reads judge/verdicts_{1..8}.jsonl
"""
import json
import random
import sys
from math import comb
from pathlib import Path

COMPARISONS = [("G8", "G4", 5301), ("GQ", "G4", 5302)]


def final(path: Path) -> dict:
    last = {}
    for r in map(json.loads, open(path)):
        last[r["id"]] = r
    return last


def build(frozen: dict, run: Path):
    arms = {a: final(run / a / "answers.jsonl") for a in ("G4", "GQ", "G8")}
    pairs = []
    for cand, base, seed in COMPARISONS:
        rng = random.Random(seed)
        for i in sorted(frozen):
            if i in arms[cand] and i in arms[base]:
                a_is_cand = rng.random() < 0.5
                pairs.append({"id": i, "A": cand if a_is_cand else base, "B": base if a_is_cand else cand})
    random.Random(5300).shuffle(pairs)
    jd = run / "judge"
    jd.mkdir(exist_ok=True)
    key = {}
    for b in range(8):
        md = [f"# exp_053 judge batch {b + 1}: 30 pairs\n"]
        for k, p in enumerate(pairs[b * 30:(b + 1) * 30], 1):
            n = b * 30 + k
            key[str(n)] = p
            q = frozen[p["id"]]
            ans = {x: arms[p[x]][p["id"]]["prefix"] + arms[p[x]][p["id"]]["answer"] for x in ("A", "B")}
            md += [f"\n=== PAIR {n} ===\n", f"Language the user writes in: {q['lang']}\n",
                   f"THE BOT'S RULES (its system prompt):\n<<<\n{q['messages'][0]['content']}\n>>>\n",
                   f"WHAT THE BOT RECEIVED (retrieved context and the user's question):\n<<<\n{q['messages'][-1]['content']}\n>>>\n",
                   f"ANSWER A:\n<<<\n{ans['A']}\n>>>\n", f"ANSWER B:\n<<<\n{ans['B']}\n>>>\n"]
        (jd / f"batch_{b + 1}.md").write_text("\n".join(md))
    (jd / "KEY.json").write_text(json.dumps(key, indent=2) + "\n")
    print("built", jd, len(pairs), "pairs")


def sign_p(x: int, y: int) -> float:
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n)


def score(frozen: dict, run: Path):
    jd = run / "judge"
    key = json.loads((jd / "KEY.json").read_text())
    res = {f"{c}_vs_{b}": {"preferred": {c: 0, b: 0, "=": 0}, "wrong": {c: 0, b: 0}, "by_intent": {}, "rows": []}
           for c, b, _ in COMPARISONS}
    for bt in range(1, 9):
        for v in map(json.loads, open(jd / f"verdicts_{bt}.jsonl")):
            k = key[str(v["pair"])]
            cand = k["A"] if k["A"] != "G4" else k["B"]
            r = res[f"{cand}_vs_G4"]
            p = "=" if v["better"] == "=" else k[v["better"]]
            r["preferred"][p] += 1
            for w in v.get("wrong", []):
                r["wrong"][k[w]] += 1
            it = frozen[k["id"]]["intent_routed"]
            r["by_intent"].setdefault(it, {cand: 0, "G4": 0, "=": 0})[p] += 1
            r["rows"].append({"pair": v["pair"], "id": k["id"], "preferred": p, "wrong": [k[w] for w in v.get("wrong", [])],
                              "why": v.get("why", "")})
    for name, r in res.items():
        cand = name.split("_")[0]
        r["n"] = len(r["rows"])
        r["sign_test_p"] = sign_p(r["preferred"][cand], r["preferred"]["G4"])
        r["quality_better"] = (r["preferred"][cand] > r["preferred"]["G4"] and r["sign_test_p"] < 0.05
                               and r["wrong"][cand] <= r["wrong"]["G4"])
    (jd / "score.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    for name, r in res.items():
        print(name, {k: r[k] for k in ("n", "preferred", "wrong", "sign_test_p", "quality_better")})


if __name__ == "__main__":
    fz = {r["id"]: r for r in map(json.loads, open(sys.argv[2])) if r["static"] is None}
    (build if sys.argv[1] == "build" else score)(fz, Path(sys.argv[3]))
