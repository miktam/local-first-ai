#!/usr/bin/env python3
"""exp_040 — validate writer output before the blind check (BUILD.md §2–§3). Code only; prints counts and flags, never eval text.

  python3.14 validate.py WAVE_DIR [PRIOR_KEPT.jsonl ...]   # reads WAVE_DIR/writer_out_*.jsonl → WAVE_DIR/writer_out.jsonl (valid items), validate.json

Item checks:
  schema       type, q.en/pl/es, label in candidates (2–5, forms in en/pl/es) or numeric {value, unit}
  collision    a form of one candidate appears (whole word, case/accent-folded) inside a form of another, same language
  giveaway     a label form appears in that language's question while no other candidate's form does (REPORT ONLY:
               party names occur naturally; the blind checker's 'ambiguous' flag judges real giveaways)
  twin         twin_of names an item of the same chunk, and the twin's label differs from it
  fresh        character-trigram Jaccard of every language's question < 0.45 against v1 and v2.1 (exp_039's rule)
  dup          trigram Jaccard >= 0.45 against an earlier kept item's EN question with the same label (this wave or PRIOR files)
Items failing schema, collision (the parser could not score them), twin, fresh or dup are dropped.
"""
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

CV = Path.home() / "REPOS/casasol/coapi_voice"
sys.path.insert(0, str(CV / "data/eval_v2"))
import dedupe as dd  # noqa: E402

EVAL = {"v1": (CV / "data/eval/coapi_eval_v1.jsonl", "efc16ab2"), "v2_1": (CV / "data/eval/coapi_eval_v2_1.jsonl", "b40a3c79")}
LANGS = ("en", "pl", "es")


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def has(form, text):
    return re.search(r"(?<!\w)" + re.escape(fold(form)) + r"(?!\w)", fold(text)) is not None


def check(it, items_by_id):
    t = it.get("type")
    if t not in ("categorical", "yesno", "numeric") or not all(it.get("q", {}).get(l) for l in LANGS):
        return "schema"
    if t == "numeric":
        n = it.get("numeric") or {}
        if not isinstance(n.get("value"), (int, float)) or not n.get("unit"):
            return "schema"
    else:
        C = it.get("candidates") or {}
        if it.get("label") not in C or not 2 <= len(C) <= 5:
            return "schema"
        if not all(all(C[c].get(l) for l in LANGS) for c in C):
            return "schema"
        for l in LANGS:
            for a in C:
                for b in C:
                    if a != b and any(has(fa, fb) for fa in C[a][l] for fb in C[b][l]):
                        return "collision"
    if it.get("twin_of"):
        o = items_by_id.get(it["twin_of"])
        if not o or o["chunk"] != it["chunk"]:
            return "twin"
        same = (o.get("numeric") or {}).get("value") == (it.get("numeric") or {}).get("value") if t == "numeric" else o.get("label") == it.get("label")
        if same:
            return "twin"
    return None


def main(wave):
    raw = [json.loads(l) for p in sorted(wave.glob("writer_out_*.jsonl")) for l in p.read_text().splitlines() if l.strip()]
    skips = [r for r in raw if "skip" in r]
    items = [r for r in raw if "item" in r]
    by_id = {i["item"]: i for i in items}
    import hashlib
    ev = []
    for name, (p, pin) in EVAL.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest().startswith(pin), f"{p.name} is not {pin}"
        ev += [dd.grams(json.loads(l)["q"]) for l in p.read_text().splitlines() if l.strip()]
    flags, kept, kept_grams = Counter(), [], []
    for prior in PRIOR:   # items kept in earlier waves: a same-label near-duplicate of one of them is a dup here too
        for l in Path(prior).read_text().splitlines():
            if l.strip():
                it = json.loads(l)
                kept_grams.append((str(it.get("label") if it["type"] != "numeric" else it["numeric"]["value"]), dd.grams(it["q"]["en"])))
    for it in items:
        why = check(it, by_id)
        if not why and any(max(dd.jaccard(dd.grams(it["q"][l]), g) for g in ev) >= dd.THRESHOLD for l in LANGS):
            why = "fresh"
        if not why:
            g = dd.grams(it["q"]["en"])
            lab = str(it.get("label") if it["type"] != "numeric" else it["numeric"]["value"])
            if any(lab == kl and dd.jaccard(g, kg) >= dd.THRESHOLD for kl, kg in kept_grams):
                why = "dup"
            else:
                kept_grams.append((lab, g))
        if not why and it["type"] == "categorical":
            C = it["candidates"]
            for l in LANGS:   # report-only: party names occur naturally in questions; the blind checker judges real giveaways
                if any(has(f, it["q"][l]) for f in C[it["label"]][l]) and not any(has(f, it["q"][l]) for c in C if c != it["label"] for f in C[c][l]):
                    flags["giveaway_flag_only"] += 1
                    break
        if why:
            flags[why] += 1
            it["_dropped"] = why
        else:
            kept.append(it)
    # a twin whose partner was dropped loses its twin link, not its place
    ids = {k["item"] for k in kept}
    for k in kept:
        if k.get("twin_of") and k["twin_of"] not in ids:
            k["twin_of"], k["flip"] = None, None
            flags["twin_link_cleared"] += 1
    (wave / "writer_out.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    rep = {"chunks_in": len({r.get("chunk") for r in raw}), "items_drafted": len(items), "skips": len(skips), "kept": len(kept),
           "flags": dict(flags), "by_type": dict(Counter(k["type"] for k in kept)),
           "twins": sum(1 for k in kept if k.get("twin_of")),
           "dropped_items": {it["item"]: it["_dropped"] for it in items if "_dropped" in it}}
    (wave / "validate.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "dropped_items"}, ensure_ascii=False))


if __name__ == "__main__":
    PRIOR = sys.argv[2:]   # optional: earlier waves' validated writer_out.jsonl files
    main(Path(sys.argv[1]))
