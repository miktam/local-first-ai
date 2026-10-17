#!/usr/bin/env python3
"""
exp_040 — the GRPO prompt set before the offline filter (BUILD §3, §5). Prompts only; no answer is written by any model.

  python rl40/build_prompts40.py [--out data/rl40/prompts] [--arith 150] [--keep-unretrieved]

Rows are mlx-lm-lora GRPO rows: {"prompt", "system", "answer", "type"} plus "id", "split", "lang", "tema", "item",
"chunk", "twin_of", "source_retrieved" for the record. Train = serve: system and user turn come from
prompt40.render40 (exp_039's prompt mode + reward40.FINAL_LINE). `answer` = reward40.encode_meta40(...).

Sources, in order:
  bank      data/bank40/BANK_FROZEN.jsonl (train) and VALID_FROZEN.jsonl (valid), pinned by sha256; every item
            rendered in EN, PL and ES (its own q.en / q.pl / q.es); kind = the item's type, label / value from the item;
            categorical and yes/no show their options after the question (prompt40.options_line; twins share an order)
  abstain   data/bank40/ABSTAIN_FROZEN.jsonl when present (train): a bank question in one language with its source
            withheld from the index, confirmed undecidable by a blind Fable reader; kind abstain, label ABSTAIN
  oos       exp_039's build_train.SPECIAL_OOS + probes.OOS (train only): kind abstain, label ABSTAIN — off-topic
            requests whose right final line is "not in the material"
  arith     exp_039's arith_templates, inputs drawn in code, gold computed in code (train templates → train,
            valid templates → valid, once per language); a draw whose gold is readable off the question is redrawn
Filters, every row:
  glue.detect_lang(q) == the row's language (else serve renders another prompt);
  trigram Jaccard < 0.45 against every v1 and v2.1 question (exp_039's Fresh; v2.1 seal checked);
  bank rows: the source chunk is in the retrieved material (prompt40.source_retrieved), unless --keep-unretrieved —
  a labelled question whose rule is not in the material would pay the policy for answering from memory.
  Abstention by design (source withheld, label ABSTAIN) is a separate step after v3 (BUILD §3.4).

Writes <out>/train.jsonl, valid.jsonl and manifest.json (counts, drops, maxima, sha256 of inputs, code and outputs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl"), str(CV / "data" / "eval_v2")]
import arith_templates as at   # noqa: E402
import build_prompts as bp39   # noqa: E402  (Fresh, EVAL_V1, EVAL_V2_1 — exp_039's, unchanged)
import build_train as bt       # noqa: E402
import dedupe as dd            # noqa: E402
import glue as G               # noqa: E402
import guide_bm25 as g         # noqa: E402
import probes                  # noqa: E402
import prompt40 as P           # noqa: E402
import reward as R             # noqa: E402
import reward40 as W           # noqa: E402

SEED = 40
LANGS = ("en", "pl", "es")
BANK = CV / "data" / "bank40"
PINS = {"BANK_FROZEN.jsonl": "6de249dd", "VALID_FROZEN.jsonl": "67934fb5"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_pinned(name: str) -> list[dict]:
    p = BANK / name
    if not sha(p).startswith(PINS[name]):
        sys.exit(f"REFUSED: {name} does not match its freeze ({PINS[name]})")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def tema_of(chunk: str) -> str:
    return ":".join(chunk.split(":")[1:3])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CV / "data" / "rl40" / "prompts"))
    ap.add_argument("--arith", type=int, default=150, help="train arithmetic prompts per language")
    ap.add_argument("--keep-unretrieved", action="store_true", help="keep bank rows whose source chunk is not retrieved")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    fresh = bp39.Fresh()
    drops = Counter()
    seen: set[str] = set()

    def admit(q: str, lang: str, dedup: bool = True) -> bool:
        k = g._fold(q).strip()
        if dedup and k in seen:
            drops["duplicate"] += 1
            return False
        if G.detect_lang(q) != lang:
            drops[f"lang_mismatch/{lang}"] += 1
            return False
        if not fresh.ok(q):
            return False
        seen.add(k)
        return True

    rows: list[dict] = []

    def add(rid, split, lang, kind, q, *, tema=None, item=None, label=None, candidates=None, value=None, tol=0.005,
            chunk=None, twin_of=None, require_source=False, opts_key=None, withhold=()):
        r = P.render40(q, lang, idx, P.options_line(candidates, lang, opts_key) if candidates else None, withhold=withhold)
        src = P.source_retrieved(chunk, lang, r["chunk_ids"]) if chunk else None
        if require_source and not src and not a.keep_unretrieved:
            drops[f"source_not_retrieved/{lang}"] += 1
            return
        rows.append({"id": rid, "split": split, "lang": lang, "tema": tema, "type": kind, "item": item, "chunk": chunk,
                     "twin_of": twin_of, "source_retrieved": src, "system": r["system"], "prompt": r["user"],
                     "answer": W.encode_meta40(lang=lang, kind=kind, q=r["shown_q"], context=r["context"], label=label,
                                               candidates=candidates, value=value, tol=tol, item=rid)})

    # ---- bank items × languages
    for split, name in (("train", "BANK_FROZEN.jsonl"), ("valid", "VALID_FROZEN.jsonl")):
        for it in load_pinned(name):
            for lang in LANGS:
                q = it["q"][lang]
                if not admit(q, lang):
                    continue
                num = it.get("numeric") or {}
                add(f"B-{it['item']}-{lang}", split, lang, it["type"], q, tema=tema_of(it["chunk"]), item=it["item"],
                    label=it.get("label") if it["type"] != "numeric" else None, candidates=it.get("candidates"),
                    value=num.get("value"), tol=num.get("tol", 0.005), chunk=it["chunk"],
                    twin_of=f"B-{it['twin_of']}-{lang}" if it.get("twin_of") else None, require_source=True,
                    opts_key=it.get("twin_of") or it["item"])

    # ---- abstention variants (train): a bank question with its source withheld from the index (BUILD §3.4).
    # Same question as a bank row on purpose, so no duplicate check; no options (v3's abstention rows have none).
    abst = BANK / "ABSTAIN_FROZEN.jsonl"
    abstain_sha = None
    if abst.exists():
        abstain_sha = sha(abst)
        if (BANK / "ABSTAIN_FROZEN.sha256").read_text().strip() != abstain_sha:
            sys.exit("REFUSED: ABSTAIN_FROZEN.jsonl does not match its .sha256")
        for x in [json.loads(l) for l in abst.read_text(encoding="utf-8").splitlines() if l.strip()]:
            if admit(x["q"], x["lang"], dedup=False):
                add(x["id"], "train", x["lang"], "abstain", x["q"], tema=tema_of(x["chunk"]), item=x["item"],
                    label=W.ABSTAIN, chunk=x["chunk"], withhold=x["withhold"])

    # ---- out of scope (train)
    n_oos = Counter()
    for lang, q in bt.SPECIAL_OOS + probes.OOS:
        if admit(q, lang):
            n_oos[lang] += 1
            add(f"O-{lang}-{n_oos[lang]:02d}", "train", lang, "abstain", q, label=W.ABSTAIN)

    # ---- arithmetic
    echo = Counter()
    for split, n in (("train", a.arith), ("valid", None)):
        temps = [t for t in at.TEMPLATES if t["split"] == split]
        for lang in LANGS:
            want = n if n is not None else len(temps)
            got, tries = 0, 0
            while got < want and temps:
                t = temps[tries % len(temps)]
                tries += 1
                if tries > 50 * max(want, 1):
                    sys.exit(f"arith {split}/{lang}: only {got} of {want} prompts pass the filters")
                q, gold = at.render(t, lang, rng)
                if R._arith_hit(q, gold):
                    echo[t["id"]] += 1
                    continue
                if admit(q, lang):
                    got += 1
                    add(f"A-{split}-{lang}-{got:03d}", split, lang, "arith", q, tema=t["tema"], item=t["id"], value=gold)

    # twins whose partner was dropped keep the row but lose the link
    ids = {r["id"] for r in rows}
    orphaned = 0
    for r in rows:
        if r["twin_of"] and r["twin_of"] not in ids:
            r["twin_of"] = None
            orphaned += 1

    # ---- write
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid")}
    rng.shuffle(by_split["train"])
    for s, rs in by_split.items():
        (out / f"{s}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")

    counts = {s: {"total": len(rs), "by_lang": dict(Counter(r["lang"] for r in rs)),
                  "by_type": dict(Counter(r["type"] for r in rs)),
                  "by_lang_type": {f"{l}/{t}": c for (l, t), c in sorted(Counter((r["lang"], r["type"]) for r in rs).items())},
                  "by_module": dict(sorted(Counter((r["tema"] or "-").split(":")[0] for r in rs).items()))}
              for s, rs in by_split.items()}
    tr = by_split["train"]
    commit = subprocess.run(["git", "-C", str(CV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CV), "status", "--porcelain", "--", "rl40", "rl", "glue.py", "guide_bm25.py",
                            "build_train.py", "run_eval.py"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment": "exp_040", "built_by": "rl40/build_prompts40.py", "seed": SEED, "reward_rev": W.REWARD_REV,
        "casasol_commit": commit, "code_dirty": bool(dirty), "args": vars(a), "counts": counts,
        "train_type_shares": {t: round(c / len(tr), 4) for t, c in Counter(r["type"] for r in tr).items()},
        "drops": dict(sorted(drops.items())), "twin_links_cleared": orphaned, "arith_echo_redrawn": dict(echo),
        "freshness": {"threshold": dd.THRESHOLD, "max_jaccard": {k: round(v, 4) for k, v in fresh.maxima.items()},
                      "dropped": dict(fresh.dropped)},
        "sha256": {
            **{f"out/{p.name}": sha(p) for p in sorted(out.glob("*.jsonl"))},
            **{f"bank/{n}": sha(BANK / n) for n in PINS},
            **({"bank/ABSTAIN_FROZEN.jsonl": abstain_sha} if abstain_sha else {}),
            **{f"code/{p.name}": sha(p) for p in (HERE / "reward40.py", HERE / "prompt40.py", Path(__file__),
                                                    CV / "rl" / "reward.py", CV / "rl" / "arith_templates.py",
                                                    CV / "rl" / "probes.py", CV / "glue.py", CV / "guide_bm25.py",
                                                    CV / "run_eval.py")},
            **{f"index/{p.name}": sha(p) for p in sorted((CV / "index").glob("guide_*.json"))},
        },
        "use": "exp_040 training prompts before the offline filter; v1, v2.1 and v3 are test sets and never enter this file",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "train_type_shares", "drops", "twin_links_cleared", "freshness", "code_dirty")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
