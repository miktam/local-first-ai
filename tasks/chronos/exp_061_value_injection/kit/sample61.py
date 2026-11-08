#!/usr/bin/env python3
"""
sample61.py draws exp_061's H2 and H3 units (HYPOTHESIS.md, "Units"), seed 20261108.

H2: 10 Mamardashvili and 10 Petukhov units whose Russian, exactly as the translator receives
it, contains at least one of the registered keywords. The match is a literal, case-sensitive
substring test of the keywords as the registration writes them. The pools are the
translation pilot's (library/tasks/mt-model-pilot/sample.py), read through its own modules:
every Mamardashvili unit of both editions, with the footnote markers removed as the pilot
does, and every Petukhov unit that has both languages. Excluded: the pilot's 40 units, the
Mamardashvili review samples (review/sample.json, review/pilot-sample.json) and the Petukhov
review sample. The draw is uniform over the matching units that remain (random.sample); the
count of those units per corpus is written under `matches`, and a corpus with fewer than 10
gives all of them. Each unit carries the pilot's fields (corpus, id, edition, file, unit, ru;
Petukhov also `index`, and `edition` is null for it), with the pilot's id format, so the
exclusion compares like with like.

H3: 10 prompts per category from the Propaganda_Eval items.jsonl given with --items,
uniform within category (one random.Random(20261108) consumed over the categories in sorted
order, random.sample over each category's items in file order), with the file's sha256 under
`items_sha256`. Every item is kept as the file has it, plus its 1-based `line`.

The output object has `seed`, `parts`, and per part drawn: `h2`, `matches`, `excluded`,
`keywords`; `h3`, `items_file`, `items_sha256`, `per_category`. The same inputs give
byte-identical output. H2 needs python3.14 (the pilot's translate.py); H3 runs under any
Python 3.9+. No network.

    python3.14 -B kit/sample61.py --part h2
    python3.14 -B kit/sample61.py --part all --items <Propaganda_Eval>/data/items.jsonl
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # never leave __pycache__ in the pilot's folders

HERE = Path(__file__).resolve().parent
PILOT = Path("/Users/miktam02/repos/library/tasks/mt-model-pilot")
SEED, PER_CORPUS, PER_CATEGORY = 20261108, 10, 10
KEYWORDS = ("Маркс", "Ленин", "Сталин", "парти", "советск", "коммунизм", "социализм", "буржуаз",
            "идеолог", "пролетар", "революц", "государств", "цензур", "диссидент", "капитализм")
MATCH = re.compile("|".join(re.escape(k) for k in KEYWORDS))
DEFAULT_OUT = HERE / "sample61.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pilot_sample_module():
    """The pilot's sample.py, imported (never run): it brings translate.py's parse_lecture,
    _NOTE, EDITIONS and SRC_ROOT, and the paths of both reviews."""
    if sys.version_info < (3, 10):
        sys.exit(f"--part h2 reads the corpora through the pilot's translate.py, which needs "
                 f"python3.14; this is {sys.version.split()[0]}")
    return load_module("mt_pilot_sample", PILOT / "sample.py")


def draw_h2():
    ps = pilot_sample_module()
    pilot = json.loads((PILOT / "sample.json").read_text(encoding="utf-8"))
    pilot_ids = {u["id"] for u in pilot}

    mm_review = set()
    for f in ("sample.json", "pilot-sample.json"):
        for x in json.loads((ps.MM / "review" / f).read_text(encoding="utf-8")):
            mm_review.add((x["edition"], x["file"], x["unit"]))
    mm_pool, mm_excl = [], {"pilot": 0, "review": 0}
    for ed, folder in ps.EDITIONS.items():
        for path in sorted((ps.SRC_ROOT / folder).glob("[0-9][0-9]-*.md")):
            _, units, _, _ = ps.parse_lecture(path)
            for idx, ru in units:
                ru = ps._NOTE.sub("", ru)
                if not MATCH.search(ru):
                    continue
                uid = f"{folder}/{path.name}:{idx}"
                if uid in pilot_ids:
                    mm_excl["pilot"] += 1
                elif (ed, path.name, idx) in mm_review:
                    mm_excl["review"] += 1
                else:
                    mm_pool.append({"corpus": "mamardashvili", "id": uid, "edition": ed,
                                    "file": path.name, "unit": idx, "ru": ru})

    pk = ps.load_module("petukhov_review_sample", ps.PK_REVIEW / "sample.py")
    pk_review = {(u["file"], u["unit"]) for u in
                 json.loads((ps.PK_REVIEW / "sample.json").read_text(encoding="utf-8"))}
    pk_pool, pk_excl = [], {"pilot": 0, "review": 0}
    for path in sorted(pk.SRC.glob("*.md")):
        for idx, ts, ru, _en in pk.units(path):
            if not MATCH.search(ru):
                continue
            uid = f"{path.name}|{ts}"
            if uid in pilot_ids:
                pk_excl["pilot"] += 1
            elif (path.name, ts) in pk_review:
                pk_excl["review"] += 1
            else:
                pk_pool.append({"corpus": "petukhov", "id": uid, "edition": None,
                                "file": path.name, "unit": ts, "index": idx, "ru": ru})

    rng = random.Random(SEED)
    drawn = []
    for pool in (mm_pool, pk_pool):
        picks = sorted(rng.sample(range(len(pool)), min(PER_CORPUS, len(pool))))
        drawn += [pool[i] for i in picks]
    matches = {"mamardashvili": len(mm_pool), "petukhov": len(pk_pool)}
    excluded = {"mamardashvili": mm_excl, "petukhov": pk_excl}
    return drawn, matches, excluded


def draw_h3(items_path: Path):
    raw = items_path.read_bytes()
    items = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        x = json.loads(line)
        missing = [k for k in ("id", "category", "prompt") if k not in x]
        if missing:
            sys.exit(f"{items_path.name} line {n}: no {missing}; its fields are {sorted(x)}. "
                     f"The kit expects id, category and prompt (assumed, not checked against the "
                     f"real file before its download)")
        x = dict(x)
        x["line"] = n
        items.append(x)
    ids = [x["id"] for x in items]
    if len(set(ids)) != len(ids):
        sys.exit(f"{items_path.name}: ids are not unique")
    rng = random.Random(SEED)
    by_cat = {}
    for x in items:
        by_cat.setdefault(x["category"], []).append(x)
    drawn, per_cat = [], {}
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        picks = sorted(rng.sample(range(len(pool)), min(PER_CATEGORY, len(pool))))
        drawn += [pool[i] for i in picks]
        per_cat[cat] = {"items": len(pool), "drawn": len(picks)}
    return drawn, per_cat, hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("h2", "h3", "all"), required=True)
    ap.add_argument("--items", type=Path, help="Propaganda_Eval data/items.jsonl (needed for h3)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    parts = ("h2", "h3") if args.part == "all" else (args.part,)
    if "h3" in parts and not args.items:
        ap.error("--part h3 needs --items <path to items.jsonl>")

    out = {"seed": SEED, "parts": list(parts)}
    notes = []
    if "h2" in parts:
        h2, matches, excluded = draw_h2()
        out.update({"keywords": list(KEYWORDS), "match": "literal substring, case-sensitive, on ru",
                    "matches": matches, "excluded": excluded, "h2": h2})
        for c, n in matches.items():
            if n < PER_CORPUS:
                notes.append(f"{c}: only {n} matching units, all taken (shortfall {PER_CORPUS - n})")
    if "h3" in parts:
        h3, per_cat, digest = draw_h3(args.items)
        out.update({"items_file": args.items.name, "items_sha256": digest,
                    "per_category": per_cat, "h3": h3})
        for c, v in per_cat.items():
            if v["drawn"] < PER_CATEGORY:
                notes.append(f"h3 {c}: only {v['items']} items, all taken")
    if notes:
        out["shortfalls"] = notes

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    summary = []
    if "h2" in out:
        summary.append("h2 " + ", ".join(f"{c} {sum(u['corpus'] == c for u in out['h2'])} of {n} matches"
                                         for c, n in out["matches"].items()))
    if "h3" in out:
        summary.append(f"h3 {len(out['h3'])} prompts over {len(out['per_category'])} categories, "
                       f"items sha256 {out['items_sha256'][:16]}")
    print("; ".join(summary) + f" → {args.out}")
    for n in notes:
        print("NOTE " + n, file=sys.stderr)


if __name__ == "__main__":
    main()
