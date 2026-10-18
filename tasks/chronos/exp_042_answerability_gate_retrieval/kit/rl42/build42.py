#!/usr/bin/env python3
"""
exp_042 — render the sets once with today's retrieval (DESIGN rev 1). Voice and gates read the same stored rows.

  python rl42/build42.py dev --host URL     # data/gate42/dev.jsonl (+ .sha256, manifest)
  python rl42/build42.py v3  --host URL     # data/gate42/v3_rendered.jsonl (+ .sha256, manifest); private, v3 text

Rendering = exp_040's prompt mode (prompt40: glue system prompt + final-line sentence, options after the question for
categorical / yes-no, withheld chunks removed from the index) with **today's retrieval**: guide_bm25 LEAD_DEPTH 2,
glue budget 1,350, k 5, and Spanish questions searched with xling's English translation (the glue's DEFAULT_MODEL
at temperature 0, served at URL). Each row stores system, user, material, the question as shown, the retrieved ids,
and source_in_material.

Labels (code only):
  v3   answerable  iff the row is not an abstention row AND its source chunk is in the material;
       abstention / out-of-scope rows → not_answerable; answerable-by-guide rows whose source was lost → not_answerable,
       kept with source_lost = true and reported apart (DESIGN decision 2).
  dev  bank prompts (exp_040 BANK_FROZEN × EN/PL/ES, seed 42): source in material → answerable (80 per language);
       source lost → not_answerable (up to 40 per language); exp_040's abstention variants the blind Fable reader
       confirmed undecidable (163, re-rendered with today's retrieval, source withheld) → not_answerable;
       the 87 off-topic prompts → not_answerable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl40"), str(CV / "rl")]
import glue as G          # noqa: E402
import guide_bm25 as g    # noqa: E402
import prompt40 as P      # noqa: E402
import run_eval as ev     # noqa: E402
import xling              # noqa: E402

OUT = CV / "data" / "gate42"
SEED = 42


def render42(q: str, lang: str, idx: dict, options: str | None = None, withhold=()) -> dict:
    """prompt40.render40 with today's retrieval: the ES question is searched by its English translation."""
    index = P.index_without(idx, lang, withhold) if withhold else idx["en" if lang == "es" else lang]
    chunks = index.retrieve(xling.query_for_index(q, lang, G.DEFAULT_MODEL), k=G.K, budget=G.BUDGET)
    context = g.context_block(chunks)
    shown = f"{q}\n{options}" if options else q
    user = f"{G.CONTEXT_LABEL[lang]}:\n{context}\n\n{G.QUESTION_LABEL[lang]}:\n{shown}"
    return {"system": P.system40(q, lang), "user": user, "material": context, "question": shown,
            "chunk_ids": [c["id"] for c in chunks]}


def write(name: str, rows: list[dict], extra: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / f"{name}.sha256").write_text(h + "\n")
    man = {"n": len(rows), "sha256": h, "label": dict(Counter(r["label"] for r in rows)),
           "by_lang_label": {f"{l}/{b}": c for (l, b), c in sorted(Counter((r["lang"], r["label"]) for r in rows).items())},
           "retrieval": {"k": G.K, "budget": G.BUDGET, "lead_depth": g.LEAD_DEPTH, "es_translator": G.DEFAULT_MODEL},
           **extra}
    (OUT / f"{name}_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("n", "label", "by_lang_label")}))


def build_v3(idx: dict) -> None:
    import run_arm40 as A
    rows = []
    for row in A.load_set("v3"):
        r = render42(row["q"], row["lang"], idx, A.options_for(row), withhold=row.get("withhold") or [])
        src = row.get("source_chunk")
        in_mat = bool(src) and P.source_retrieved(src, row["lang"], r["chunk_ids"])
        abst = row["type"] == "abstain"
        rows.append({"id": row["id"], "lang": row["lang"], "type": row["type"], "twin_of": row.get("twin_of"),
                     "kind": "oos" if abst and not src else "abstain" if abst else "answer",
                     "label": "answerable" if (not abst and in_mat) else "not_answerable",
                     "source_lost": (not abst) and not in_mat, **r,
                     "meta": json.loads(A.meta_of(row, r["question"], r["material"]))})
    write("v3_rendered", rows, {"source_lost": sum(r["source_lost"] for r in rows),
                                "v3_sha256": hashlib.sha256(A.SETS["v3"][0].read_bytes()).hexdigest()})


def build_dev(idx: dict) -> None:
    rng = random.Random(SEED)
    bank = [json.loads(l) for l in (CV / "data/bank40/BANK_FROZEN.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for lang in ("en", "pl", "es"):
        items = rng.sample(bank, len(bank))
        ans, lost = [], []
        for it in items:
            if len(ans) >= 80 and len(lost) >= 40:
                break
            opts = P.options_line(it["candidates"], lang, it.get("twin_of") or it["item"]) if it.get("candidates") else None
            r = render42(it["q"][lang], lang, idx, opts)
            hit = P.source_retrieved(it["chunk"], lang, r["chunk_ids"])
            if hit and len(ans) < 80:
                ans.append({"id": f"B-{it['item']}-{lang}", "src": "bank", "lang": lang, "label": "answerable", **r})
            elif not hit and len(lost) < 40:
                lost.append({"id": f"B-{it['item']}-{lang}", "src": "bank_lost", "lang": lang, "label": "not_answerable", **r})
        rows += ans + lost
    A = CV / "data" / "bank40" / "abstain"
    verdict = {}
    for p in sorted(A.glob("reader_out_*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                v = json.loads(l); verdict[v["id"]] = v
    for x in (json.loads(l) for l in (A / "rows.jsonl").read_text().splitlines() if l.strip()):
        v = verdict[x["id"]]
        if v["pick"] == "undecidable" and not v.get("doubt"):
            r = render42(x["q"], x["lang"], idx, None, withhold=x["withhold"])
            rows.append({"id": x["id"], "src": "variant", "lang": x["lang"], "label": "not_answerable", **r})
    for p in (json.loads(l) for l in (CV / "data/rl40/prompts/train.jsonl").read_text().splitlines() if l.strip()):
        if p["id"].startswith("O-"):
            m = json.loads(p["answer"])
            r = render42(m["q"], p["lang"], idx)
            rows.append({"id": p["id"], "src": "oos", "lang": p["lang"], "label": "not_answerable", **r})
    write("dev", rows, {"by_src": dict(Counter(r["src"] for r in rows)), "seed": SEED})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=["dev", "v3"])
    ap.add_argument("--host", required=True, help="Ollama serving glue.DEFAULT_MODEL (the ES translator)")
    a = ap.parse_args(argv)
    ev.OLLAMA = a.host
    import run_arm as ra
    ra.use_index(ra.INDEX_LIVE)
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    build_v3(idx) if a.set == "v3" else build_dev(idx)


if __name__ == "__main__":
    main()
