#!/usr/bin/env python3
"""
exp_039 — the GRPO prompt set (DESIGN §6). Prompts only; no answer is written by any model.

  python rl/build_prompts.py [--out data/rl] [--grounded 180] [--valid 16] [--arith 40]

Rows are mlx-lm-lora GRPO rows: {"prompt", "system", "answer", "type"} plus "id", "lang",
"tema", "split" for the record. Train = serve: `system` and `prompt` are built exactly as
glue.answer() builds them — system prompt rendered in the question's language, the advice
suffix when glue's own _ADVICE regex fires, retrieved material inside the user turn, same
chunker, k = 5, budget 1200 — so the policy is trained on the prompt it is served with.
`answer` is reward.encode_meta(...): the reward's per-row facts (language, type, question,
context, and for arithmetic the gold computed in code).

Sources and filters, in order:
  grounded  doc2query questions of the train temas (M1–M4, M5 T39–41) and validation temas
            (M5 T42–43); Módulo 6 never; round-robin over temas so every tema is covered
  oos       build_train.SPECIAL_OOS + probes.OOS        (30 per language)
  advice    build_train.SPECIAL_ADVICE + probes.ADVICE  (30 per language)
  arith     arith_templates.TEMPLATES, inputs drawn in code, gold computed in code; a draw whose
            gold already appears in the question (F05 housing: deposit = the rent) is redrawn
  every row: glue.detect_lang(q) must equal its language (else serve would render another
            prompt); character-trigram Jaccard against every v1 and v2.1 question < 0.45
            (dedupe.py's pre-registered threshold). v2.1 questions are compared in code only;
            the builder prints counts and the maximum, never a v2 row.

Writes <out>/train.jsonl, valid.jsonl, glossary.json (EXP039_GLOSSARY for the reward off the
mini) and manifest.json (counts, maxima, sha256 of every input and output, casasol commit).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "data" / "eval_v2")]
import arith_templates as at   # noqa: E402
import build_train as bt       # noqa: E402
import dedupe as dd            # noqa: E402  (code only: grams, jaccard, THRESHOLD)
import glue                    # noqa: E402
import guide_bm25 as g         # noqa: E402
import probes                  # noqa: E402
import reward as R             # noqa: E402

SEED = 39
EVAL_V1 = CV / "data" / "eval" / "coapi_eval_v1.jsonl"
EVAL_V2_1 = CV / "data" / "eval" / "coapi_eval_v2_1.jsonl"
EVAL_V2_1_SHA = "b40a3c79"
LANGS = ("en", "pl", "es")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _eval_grams(path: Path) -> list[set]:
    return [dd.grams(json.loads(l)["q"]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class Fresh:
    """Max trigram Jaccard of a question against v1 and v2.1; refuses at the threshold."""
    def __init__(self):
        if not sha(EVAL_V2_1).startswith(EVAL_V2_1_SHA):
            sys.exit(f"{EVAL_V2_1.name} does not match its seal ({EVAL_V2_1_SHA}); refusing")
        self.sets = {"v1": _eval_grams(EVAL_V1), "v2_1": _eval_grams(EVAL_V2_1)}
        self.maxima = {"v1": 0.0, "v2_1": 0.0}
        self.dropped = Counter()

    def ok(self, q: str) -> bool:
        gq, fresh = dd.grams(q), True
        for name, sets in self.sets.items():
            m = max(dd.jaccard(gq, s) for s in sets)
            if m >= dd.THRESHOLD:
                self.dropped[name] += 1
                fresh = False
            else:
                self.maxima[name] = max(self.maxima[name], m)
        return fresh


def split_of(tema: str) -> str | None:
    m = re.match(r"M(\d+):T(\d+)$", tema)
    return bt.split_of(int(m.group(1)), int(m.group(2))) if m else None


def round_robin(by_tema: dict[str, list[str]], n: int, rng: random.Random) -> list[tuple[str, str]]:
    queues = {t: rng.sample(qs, len(qs)) for t, qs in sorted(by_tema.items())}
    out = []
    while len(out) < n and any(queues.values()):
        for t in sorted(queues):
            if queues[t] and len(out) < n:
                out.append((t, queues[t].pop()))
    return out


def render(q: str, lang: str, idx: dict) -> tuple[str, str, str]:
    """(system, user, context) exactly as glue.answer() renders them."""
    chunks = idx["en" if lang == "es" else lang].retrieve(q, k=glue.K, budget=glue.BUDGET)
    context = g.context_block(chunks)
    system = glue.SYSTEM[lang] + (glue.ADVICE_SUFFIX[lang] if glue._ADVICE.search(q) else "")
    user = f"{glue.CONTEXT_LABEL[lang]}:\n{context}\n\n{glue.QUESTION_LABEL[lang]}:\n{q}"
    return system, user, context


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(CV / "data" / "rl"))
    ap.add_argument("--grounded", type=int, default=180, help="train grounded prompts per language")
    ap.add_argument("--valid", type=int, default=16, help="validation grounded prompts per language")
    ap.add_argument("--arith", type=int, default=40, help="train arithmetic prompts per language")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    fresh = Fresh()
    lang_mismatch = Counter()
    seen: set[str] = set()

    def admit(q: str, lang: str) -> bool:
        k = g._fold(q).strip()
        if k in seen:
            return False
        if glue.detect_lang(q) != lang:
            lang_mismatch[lang] += 1
            return False
        if not fresh.ok(q):
            return False
        seen.add(k)
        return True

    # ---- candidates
    d2q: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for r in bt._load_d2q():
        sp = split_of(r["tema"])
        if sp:
            d2q[(sp, r["lang"])][r["tema"]].append(r["q"])

    rows: list[dict] = []

    def add(rid, split, lang, typ, q, tema=None, gold=None):
        system, user, context = render(q, lang, idx)
        rows.append({"id": rid, "split": split, "lang": lang, "tema": tema, "type": typ, "system": system,
                     "prompt": user, "answer": R.encode_meta(lang=lang, type=typ, q=q, context=context, arith=gold)})

    for split, n in (("train", a.grounded), ("valid", a.valid)):
        for lang in LANGS:
            picked = [(t, q) for t, q in round_robin(d2q[(split, lang)], 10**9, rng) if admit(q, lang)][:n]
            if len(picked) < n:
                sys.exit(f"only {len(picked)} grounded {split}/{lang} prompts pass the filters (need {n})")
            for i, (t, q) in enumerate(picked, 1):
                add(f"G-{split}-{lang}-{i:03d}", split, lang, "numeric" if re.search(r"\d", q) else "simple", q, tema=t)

    for kind, items in (("oos", bt.SPECIAL_OOS + probes.OOS), ("advice", bt.SPECIAL_ADVICE + probes.ADVICE)):
        n_by = Counter()
        for lang, q in items:
            if admit(q, lang):
                n_by[lang] += 1
                add(f"{kind[0].upper()}-{lang}-{n_by[lang]:02d}", "train", lang, kind, q)

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
                if R._arith_hit(q, gold):          # gold readable off the question: echoing it would score
                    echo[t["id"]] += 1
                    continue
                if admit(q, lang):
                    got += 1
                    add(f"A-{split}-{lang}-{got:03d}", split, lang, "arith", q, tema=t["tema"], gold=gold)

    # ---- write
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid")}
    rng.shuffle(by_split["train"])
    for s, rs in by_split.items():
        (out / f"{s}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    (out / "glossary.json").write_text(json.dumps(sorted(R.glossary()), ensure_ascii=False, indent=0), encoding="utf-8")

    counts = {s: {"total": len(rs), "by_lang": dict(Counter(r["lang"] for r in rs)),
                  "by_type": dict(Counter(r["type"] for r in rs)),
                  "by_lang_type": {f"{l}/{t}": c for (l, t), c in sorted(Counter((r["lang"], r["type"]) for r in rs).items())}}
              for s, rs in by_split.items()}
    tr = by_split["train"]
    shares = {t: round(c / len(tr), 4) for t, c in Counter(r["type"] for r in tr).items()}
    commit = subprocess.run(["git", "-C", str(CV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(CV), "status", "--porcelain", "--", "rl", "glue.py", "guide_bm25.py",
                            "build_train.py", "run_eval.py"], capture_output=True, text=True).stdout.strip()
    manifest = {
        "experiment": "exp_039", "built_by": "rl/build_prompts.py", "seed": SEED, "casasol_commit": commit,
        "code_dirty": bool(dirty), "args": vars(a), "counts": counts, "train_type_shares": shares,
        "freshness": {"threshold": dd.THRESHOLD, "max_jaccard": {k: round(v, 4) for k, v in fresh.maxima.items()},
                      "dropped": dict(fresh.dropped)},
        "lang_mismatch_dropped": dict(lang_mismatch),
        "arith_echo_dropped": dict(echo),
        "arith_by_tema": dict(sorted(Counter(r["tema"] for r in rows if r["type"] == "arith").items())),
        "sha256": {
            **{f"out/{p.name}": sha(p) for p in sorted(out.glob("*.json*")) if p.name != "manifest.json"},
            **{f"code/{p.name}": sha(p) for p in (HERE / "reward.py", HERE / "probes.py", HERE / "arith_templates.py",
                                                    Path(__file__), CV / "glue.py", CV / "guide_bm25.py", CV / "run_eval.py")},
            **{f"index/{p.name}": sha(p) for p in sorted((CV / "index").glob("guide_*.json"))},
            **{f"eval/{p.name}": sha(p) for p in (EVAL_V1, EVAL_V2_1)},
        },
        "use": "training prompts for exp_039 only; v1 and v2.1 are test sets and never enter this file",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("counts", "train_type_shares", "freshness", "lang_mismatch_dropped", "arith_echo_dropped", "code_dirty")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
