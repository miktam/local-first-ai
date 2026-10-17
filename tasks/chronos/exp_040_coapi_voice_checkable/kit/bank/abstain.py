#!/usr/bin/env python3
"""exp_040 — abstention variants of training-bank items (BUILD.md §3.4). Code only; the readers are Fable subagents.

  ~/REPOS/casasol/.venv/bin/python3 abstain.py batches [--share 0.15] [N_PER_BATCH]   # select items, write reader batches
  ~/REPOS/casasol/.venv/bin/python3 abstain.py collect                                # reader verdicts → ABSTAIN_FROZEN.jsonl

Selection (seed "40:abstain"): about 15 % of BANK_FROZEN, at most one item per twin group (twins share a chunk).
Each item × EN/PL/ES is one candidate row, rendered exactly as training renders it (prompt40.render40) but with
withheld chunks removed from the index, as v3 does: EN and ES withhold the item's EN chunk; PL withholds every
PL chunk of the same section (EN↔PL ids are positional). No options are shown (v3's abstention rows have none).

A blind Fable reader sees the row's question, that material and the candidate names (or the unit), never the
label, and either picks or says "undecidable". Only "undecidable" rows without doubt become ABSTAIN rows; a pick of any
candidate, right or wrong, drops the row (the material still bears on it). One reader per row.
Output under ~/REPOS/casasol/coapi_voice/data/bank40/abstain/.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

CV = Path.home() / "REPOS/casasol/coapi_voice"
sys.path[:0] = [str(CV), str(CV / "rl40")]
import guide_bm25 as g   # noqa: E402
import prompt40 as P     # noqa: E402

BANK = CV / "data/bank40"
OUT = BANK / "abstain"
BANK_PIN = "6de249dd"
LANGS = ("en", "pl", "es")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def batches(share=0.15, n=15):
    src = BANK / "BANK_FROZEN.jsonl"
    if not sha(src).startswith(BANK_PIN):
        sys.exit("REFUSED: BANK_FROZEN.jsonl does not match its freeze")
    items = load(src)
    roots = sorted({i.get("twin_of") or i["item"] for i in items})
    random.Random("40:abstain").shuffle(roots)
    pick = set(roots[: round(share * len(items))])
    chosen = [i for i in items if i["item"] in pick]
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    OUT.mkdir(parents=True, exist_ok=True)
    rows, chars = [], 0
    for it in chosen:
        for lang in LANGS:
            withhold = P.pl_section(it["chunk"], idx) if lang == "pl" else [it["chunk"]]
            if not withhold:
                continue
            r = P.render40(it["q"][lang], lang, idx, withhold=withhold)
            rows.append({"id": f"X-{it['item']}-{lang}", "item": it["item"], "lang": lang, "type": it["type"],
                         "chunk": it["chunk"], "withhold": withhold, "q": it["q"][lang], "context": r["context"],
                         "candidates": sorted(it["candidates"]) if it.get("candidates") else None,
                         "numeric_unit": (it.get("numeric") or {}).get("unit")})
            chars += len(r["context"])
    (OUT / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rng = random.Random("40:abstain:order")
    order = rows[:]
    rng.shuffle(order)                     # a batch mixes items and languages
    for b in range(0, len(order), n):
        with open(OUT / f"r_{b // n + 1:02d}.jsonl", "w", encoding="utf-8") as f:
            for r in order[b: b + n]:
                rec = {"id": r["id"], "q": r["q"], "material": r["context"]}
                rec.update({"candidates": random.Random(f"40:abs:{r['id']}").sample(r["candidates"], len(r["candidates"]))}
                           if r["candidates"] else {"numeric_unit": r["numeric_unit"]})
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(chosen), "rows": len(rows), "by_lang": dict(Counter(r["lang"] for r in rows)),
                      "by_type": dict(Counter(r["type"] for r in rows)), "batches": -(-len(rows) // n),
                      "material_chars": chars, "material_tokens_est": chars // 4}))


def collect(partial=False):
    rows = {r["id"]: r for r in load(OUT / "rows.jsonl")}
    verdict = {}
    for p in sorted(OUT.glob("reader_out_*.jsonl")):
        for v in load(p):
            if v["id"] not in rows:
                sys.exit(f"REFUSED: {p.name} has an unknown id {v['id']}")
            verdict[v["id"]] = v
    missing = sorted(set(rows) - set(verdict))
    if partial:
        rows = {i: r for i, r in rows.items() if i in verdict}
    elif missing:
        sys.exit(f"REFUSED: {len(missing)} rows have no reader verdict (first: {missing[0]})")
    kept = [rows[i] for i in rows if verdict[i]["pick"] == "undecidable" and not verdict[i].get("doubt")]
    out = BANK / "ABSTAIN_FROZEN.jsonl"
    out.write_text("".join(json.dumps({k: r[k] for k in ("id", "item", "lang", "type", "chunk", "withhold", "q")},
                                      ensure_ascii=False, sort_keys=True) + "\n" for r in kept), encoding="utf-8")
    h = sha(out)
    (BANK / "ABSTAIN_FROZEN.sha256").write_text(h + "\n")
    man = {"rows": len(rows), "kept_undecidable": len(kept), "sha256": h,
           "kept_by_lang": dict(Counter(r["lang"] for r in kept)), "kept_by_type": dict(Counter(r["type"] for r in kept)),
           "dropped_decidable": dict(Counter(rows[i]["lang"] for i in rows if verdict[i]["pick"] != "undecidable")),
           "dropped_undecidable_with_doubt": sum(1 for i in rows if verdict[i]["pick"] == "undecidable" and verdict[i].get("doubt")),
           "rows_sha256": sha(OUT / "rows.jsonl")}
    (OUT / "ABSTAIN_MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "batches":
        share = float(rest[rest.index("--share") + 1]) if "--share" in rest else 0.15
        nums = [a for a in rest if a.isdigit()]
        batches(share, int(nums[0]) if nums else 15)
    elif cmd == "collect":
        collect(partial="--partial" in rest)
    else:
        sys.exit(__doc__)
