#!/usr/bin/env python3
"""exp_040 — writer and checker batch files for the training bank (BUILD.md §3).

  python3.14 batches.py writer  NAME POOL [N_PER_BATCH] [--pilot K] [--wave k/of]   # writer batches from pools.json
  python3.14 batches.py checker NAME WRITER_OUT.jsonl [N_PER_BATCH]   # blind checker batches from writer output

Only train/valid pool chunks can be written out (holdout and m6 are refused). Checker batches carry the EN
question, the chunk text and the candidate names shuffled (seed "40:chk:<item>"), never the label, rule or twin data.
Output under ~/REPOS/casasol/coapi_voice/data/bank40/<NAME>/.
"""
import json
import random
import sys
from pathlib import Path

CV = Path.home() / "REPOS/casasol/coapi_voice"
BANK = CV / "data/bank40"


def chunks():
    idx = {}
    for lang in ("en", "pl"):
        for c in json.loads((CV / f"index/guide_{lang}.json").read_text())["chunks"]:
            idx[c["id"]] = c
    return idx


def writer(name, pool, n=10, pilot=None, wave=None):
    assert pool in ("train", "valid"), "holdout and m6 are for the v3 session only"
    P = json.loads((BANK / "pools.json").read_text())["pools"][pool]
    if wave:
        k, of = wave   # wave k of `of`: pool minus the pilot's chunks, shuffled (seed "40:waves"), split in order
        done = {json.loads(l)["chunk"] for l in (BANK / "pilot/writer_out.jsonl").read_text().splitlines() if l.strip()}
        P = sorted((p for p in P if p["id"] not in done), key=lambda p: p["id"])
        random.Random("40:waves").shuffle(P)
        size = -(-len(P) // of)
        P = P[(k - 1) * size: k * size]
    if pilot:
        rng = random.Random("40:pilot")
        mods = sorted({p["module"] for p in P})
        P = [rng.choice([p for p in P if p["module"] == m]) for m in rng.sample(mods, pilot)]
    idx = chunks()
    out = BANK / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in P:
        c = idx[p["id"]]
        r = {"id": c["id"], "title": c["title"], "text_en": c["text"]}
        if p["pl_id"]:
            r["text_pl"] = idx[p["pl_id"]]["text"]
        rows.append(r)
    for i in range(0, len(rows), n):
        (out / f"w_{i // n + 1:02d}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + n]))
    print(f"{len(rows)} chunks -> {(len(rows) + n - 1) // n} writer batches in {out}")


def checker(name, src, n=40):
    idx = chunks()
    items = [json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()]
    items = [it for it in items if "item" in it]
    random.Random("40:chk:order").shuffle(items)   # twins and same-chunk items land apart, not side by side
    out = BANK / name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it in items:
        r = {"item": it["item"], "q": it["q"]["en"], "chunk_text": idx[it["chunk"]]["text"]}
        if it["type"] == "numeric":
            r["numeric_unit"] = it["numeric"]["unit"]
        else:
            names = sorted(it["candidates"])
            random.Random(f"40:chk:{it['item']}").shuffle(names)
            r["candidates"] = names
        rows.append(r)
    for i in range(0, len(rows), n):
        (out / f"c_{i // n + 1:02d}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + n]))
    print(f"{len(rows)} items -> {(len(rows) + n - 1) // n} checker batches in {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    pilot = int(a[a.index("--pilot") + 1]) if "--pilot" in a else None
    wave = tuple(map(int, a[a.index("--wave") + 1].split("/"))) if "--wave" in a else None
    a = [x for i, x in enumerate(a) if x not in ("--pilot", "--wave") and (i == 0 or a[i - 1] not in ("--pilot", "--wave"))]
    if a[0] == "writer":
        writer(a[1], a[2], int(a[3]) if len(a) > 3 else 10, pilot, wave)
    else:
        checker(a[1], a[2], int(a[3]) if len(a) > 3 else 40)
