#!/usr/bin/env python3
"""exp_052: build Andrei's blind read. 30 questions that reached the model, 6 per routed intent (property_search,
area_knowledge, legal_fiscal, process, market; an intent with fewer than 6 is filled at random from the rest), seed
52. In each pair the two answers are shown as A and B in an order drawn at random per pair (seed 52). The key goes
to a separate file. The retrieved context is attached folded, for reference.

    python3.14 -I kit/read52.py data/frozen.jsonl runs/<ts>
"""
import json
import random
import sys
from pathlib import Path

INTENTS = ["property_search", "area_knowledge", "legal_fiscal", "process", "market"]


def final(path: Path) -> dict:
    last = {}
    for r in map(json.loads, open(path)):
        last[r["id"]] = r                            # the last try is the one used
    return last


def main():
    frozen = {r["id"]: r for r in map(json.loads, open(sys.argv[1])) if r["static"] is None}
    run = Path(sys.argv[2])
    G, M = final(run / "G" / "answers.jsonl"), final(run / "M" / "answers.jsonl")
    ids = sorted(i for i in frozen if i in G and i in M)
    rng = random.Random(52)
    pick = []
    for it in INTENTS:
        pool = [i for i in ids if frozen[i]["intent_routed"] == it]
        pick += rng.sample(pool, min(6, len(pool)))
    rest = [i for i in ids if i not in pick]
    pick += rng.sample(rest, 30 - len(pick)) if len(pick) < 30 else []
    rng.shuffle(pick)
    key, md = {}, ["# exp_052: read 30 pairs blind\n",
                   "Each question went to the bot twice, through two runtimes. For every pair, mark which answer "
                   "you would rather the bot send (**A**, **B**, or **=** for no real difference), and add "
                   "**wrong A** / **wrong B** if an answer says something false or breaks the bot's rules "
                   "(invented property, price or place; wrong language; legal advice presented as binding).\n",
                   "Reply in one line per pair, e.g. `1 A`, `2 =`, `3 B wrong A`.\n"]
    for n, i in enumerate(pick, 1):
        a_is_g = rng.random() < 0.5
        key[str(n)] = {"id": i, "A": "G" if a_is_g else "M", "B": "M" if a_is_g else "G"}
        q = frozen[i]
        ga, ma = G[i]["prefix"] + G[i]["answer"], M[i]["prefix"] + M[i]["answer"]
        A, B = (ga, ma) if a_is_g else (ma, ga)
        ctx = q["messages"][-1]["content"] if len(q["messages"]) > 1 else ""
        md += [f"\n---\n\n## {n}. ({q['lang']}, {q['intent_routed']}) {q['text']}\n",
               f"**A**\n\n{A}\n", f"**B**\n\n{B}\n",
               f"<details><summary>What the bot retrieved</summary>\n\n```\n{ctx}\n```\n</details>\n"]
    (run / "READ30.md").write_text("\n".join(md))
    (run / "READ30_key.json").write_text(json.dumps(key, indent=2) + "\n")
    print("wrote", run / "READ30.md")


if __name__ == "__main__":
    main()
