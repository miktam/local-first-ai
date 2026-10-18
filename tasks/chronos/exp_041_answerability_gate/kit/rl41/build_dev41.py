#!/usr/bin/env python3
"""
exp_041 — the dev set for prompt and threshold (DESIGN §3). Code only; every label comes from exp_040.

  python rl41/build_dev41.py        # writes data/gate41/dev.jsonl, dev.sha256, dev_manifest.json

Rows {id, src, lang, label, material, question}; label ∈ answerable / not_answerable:
  variant     exp_040's abstention variants (data/bank40/abstain/rows.jsonl), blind Fable reader verdicts:
              "undecidable" → not_answerable, a pick → answerable; doubt-flagged verdicts dropped. Material = the
              context retrieved with the source withheld; question as shown (no options, like v3's abstention rows)
  oos         exp_040's off-topic training prompts (O-*): not_answerable
  bank        exp_040 training prompts whose source chunk was retrieved (B-*, categorical / yes-no / numeric):
              answerable; 80 per language, seed 41; question as shown (options line included)
Material and question are taken from each prompt's reward meta (its rendered context and shown question), so they
are exactly what the voice saw.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

CV = Path(__file__).resolve().parent.parent
A = CV / "data" / "bank40" / "abstain"
PROMPTS = CV / "data" / "rl40" / "prompts" / "train.jsonl"
OUT = CV / "data" / "gate41"
N_BANK = 80
SEED = 41


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    rows = []
    verdict = {}
    for p in sorted(A.glob("reader_out_*.jsonl")):
        for v in load(p):
            verdict[v["id"]] = v
    for r in load(A / "rows.jsonl"):
        v = verdict[r["id"]]
        if v.get("doubt"):
            continue
        rows.append({"id": r["id"], "src": "variant", "lang": r["lang"],
                     "label": "not_answerable" if v["pick"] == "undecidable" else "answerable",
                     "material": r["context"], "question": r["q"]})
    prompts = load(PROMPTS)
    for r in prompts:
        if r["id"].startswith("O-"):
            m = json.loads(r["answer"])
            rows.append({"id": r["id"], "src": "oos", "lang": r["lang"], "label": "not_answerable",
                         "material": m["context"], "question": m["q"]})
    rng = random.Random(SEED)
    for lang in ("en", "pl", "es"):
        pool = [r for r in prompts if r["id"].startswith("B-") and r["lang"] == lang and r["source_retrieved"]
                and r["type"] in ("categorical", "yesno", "numeric")]
        for r in rng.sample(sorted(pool, key=lambda x: x["id"]), N_BANK):
            m = json.loads(r["answer"])
            rows.append({"id": r["id"], "src": "bank", "lang": lang, "label": "answerable",
                         "material": m["context"], "question": m["q"]})
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "dev.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / "dev.sha256").write_text(h + "\n")
    man = {"n": len(rows), "sha256": h, "seed": SEED,
           "by_src_label": {f"{s}/{l}": c for (s, l), c in sorted(Counter((r["src"], r["label"]) for r in rows).items())},
           "by_lang_label": {f"{s}/{l}": c for (s, l), c in sorted(Counter((r["lang"], r["label"]) for r in rows).items())},
           "inputs": {"abstain_rows": hashlib.sha256((A / "rows.jsonl").read_bytes()).hexdigest(),
                      "prompts": hashlib.sha256(PROMPTS.read_bytes()).hexdigest()}}
    (OUT / "dev_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: man[k] for k in ("n", "by_src_label", "by_lang_label")}))


if __name__ == "__main__":
    main()
