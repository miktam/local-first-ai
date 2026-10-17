#!/usr/bin/env python3
"""
exp_040 — run one arm on one set through the Ollama 0.33.0 side server (HYPOTHESIS.md). Started by rl40/measure40.sh,
which owns the side server and runs exp_039's ruler check first.

  python rl40/run_arm40.py --host URL ARM SET        # ARM: B0i R3 B0q8 R3q8 A13 R3G · SET: v3 valid40

Arms
  B0i, R3, B0q8, R3q8   "prompt" mode = training's prompt: prompt40.render40 (glue's system prompt + the advice suffix
                        when it fires + reward40.FINAL_LINE; retrieval on the bare question, k 5, budget 1200, ES from
                        EN; withheld chunks removed from the index; options after the question for categorical and
                        yes/no), run_eval.OPTIONS (serve: greedy, seed 42, 512 tokens), thinking off. No glue.
  A13, R3G              "glue" mode: glue.answer() unchanged (b1a648c) with three patches, all in this file:
                        the final-line sentence is appended last to glue's own answer system prompt (as system40);
                        retrieval and the scope gate see the bare question (the options line follows it in the user
                        turn only); withheld chunks are removed from the index. A glue decline (scope "out") is
                        scored as an abstention: right on ABSTAIN rows, wrong elsewhere. The final line is read from
                        the model's raw reply to glue's answer call (glue.clean() drops a period-less final line);
                        glue's served text is kept beside it ("served") for the judged prose.
Scoring: reward40's answer verdict on the final line (exact match = "right"); each row also records whether the final
line abstained (H4).

Pins checked before any row: glue / run_eval / guide_bm25 identical to b1a648c; rl40/ committed (the scorer is the
registered one); live index f1e28120 / 9a422b0e; server 0.33.0; the arm's model digest (R3 digests from
models/r3/MANIFEST.json, written by fuse_quant40.sh); v3's seal 2c6cbad1. A v3 cell runs once per arm, and no v3
cell runs before R3 is registered (all arms are scored after training, together). Output:
runs/exp040_<ARM>_<SET>_<stamp>/rows.jsonl + summary.json. This script prints aggregates only, never a row.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl")]
import glue as G          # noqa: E402
import guide_bm25 as g    # noqa: E402
import prompt40 as P      # noqa: E402
import reward40 as W      # noqa: E402
import run_arm as ra      # noqa: E402  (pins, digests, server and code checks — exp_039's, unchanged)
import run_eval as ev     # noqa: E402

R3_MANIFEST = CV / "models" / "r3" / "MANIFEST.json"
ARMS = {"B0i": ("prompt", ra.BASE_OURS_IMAT), "B0q8": ("prompt", ra.BASE_OURS_Q8),
        "R3": ("prompt", "coapi-voice:r3"), "R3q8": ("prompt", "coapi-voice:r3-q8"),
        "A13": ("glue", ra.BASE_MODEL), "R3G": ("glue", "coapi-voice:r3")}
SETS = {"v3": (CV / "data/eval/coapi_eval_v3.jsonl", "2c6cbad1"),
        "valid40": (CV / "data/bank40/VALID_FROZEN.jsonl", "67934fb5")}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- rows
def kind_of(row: dict, twins: set[str]) -> str:
    if row["type"] == "abstain":
        return "abstain" if row.get("source_chunk") else "oos"
    return "pair" if (row.get("twin_of") or row["id"] in twins) else "single"


def load_set(name: str) -> list[dict]:
    """v3 rows as sealed; valid40 = VALID_FROZEN items in v3's row shape, one row per language (smoke only)."""
    path, pin = SETS[name]
    if not sha(path).startswith(pin):
        sys.exit(f"REFUSED: {path.name} does not match its seal {pin}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if name == "valid40":
        rows = [{"id": f"{it['item']}-{lang}", "lang": lang, "q": it["q"][lang], "type": it["type"],
                 "candidates": it.get("candidates"), "label": it.get("label"), "numeric": it.get("numeric"),
                 "source_chunk": it["chunk"], "withhold": [],
                 "twin_of": f"{it['twin_of']}-{lang}" if it.get("twin_of") else None} for it in rows for lang in ("en", "pl", "es")]
    return rows


def meta_of(row: dict, shown_q: str, context: str) -> str:
    t = row["type"]
    num = row.get("numeric") or {}
    return W.encode_meta40(lang=row["lang"], kind=t, q=shown_q, context=context,
                           label=W.ABSTAIN if t == "abstain" else (row.get("label") if t != "numeric" else None),
                           candidates=row.get("candidates") if t in ("categorical", "yesno") else None,
                           value=num.get("value"), tol=num.get("tol", 0.005), item=row["id"])


def options_for(row: dict) -> str | None:
    if row["type"] in ("categorical", "yesno") and row.get("candidates"):
        return P.options_line(row["candidates"], row["lang"], row.get("twin_of") or row["id"])
    return None


# ---------------------------------------------------------------- glue mode patches
class _BareIndex:
    """Retrieves on the question without its options line, from the (possibly withheld) index."""
    def __init__(self, base, options: str | None):
        self.base, self.options = base, options

    def retrieve(self, query, k=5, budget=None):
        if self.options and query.endswith("\n" + self.options):
            query = query[: -len(self.options) - 1]
        return self.base.retrieve(query, k=k, budget=budget)


@contextlib.contextmanager
def glue40(lang: str, options: str | None, raw: dict | None = None):
    chat, scope = G.ev.ollama_chat, G.in_scope

    def ollama_chat(model, messages, options_):
        main = bool(messages) and messages[0]["role"] == "system" and messages[0]["content"].startswith(G.SYSTEM[lang])
        if main:
            messages = [{**messages[0], "content": messages[0]["content"] + W.FINAL_LINE[lang]}] + messages[1:]
        resp = chat(model, messages, options_)
        if main and raw is not None:     # the model's own reply to glue's answer call, before clean()/scrub()
            raw["text"] = resp["message"]["content"]
        return resp

    def in_scope(question, model):
        if options and question.endswith("\n" + options):
            question = question[: -len(options) - 1]
        return scope(question, model)

    G.ev.ollama_chat, G.in_scope = ollama_chat, in_scope
    try:
        yield
    finally:
        G.ev.ollama_chat, G.in_scope = chat, scope


# ---------------------------------------------------------------- one row
def answer_row(row: dict, mode: str, model: str, idx: dict) -> dict:
    lang, opts = row["lang"], options_for(row)
    withhold = row.get("withhold") or []
    served = None
    if mode == "prompt":
        r = P.render40(row["q"], lang, idx, opts, withhold=withhold)
        resp = ev.ollama_chat(model, [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                              ev.OPTIONS)
        text, scope, chunk_ids, context, shown = resp["message"]["content"], "in", r["chunk_ids"], r["context"], r["shown_q"]
        out_tok = resp.get("eval_count")
    else:
        shown = f"{row['q']}\n{opts}" if opts else row["q"]
        ilang = "en" if lang == "es" else lang
        base = P.index_without(idx, lang, withhold) if withhold else idx[ilang]
        raw: dict = {}
        with glue40(lang, opts, raw):
            res = G.answer(shown, model=model, lang=lang, index={ilang: _BareIndex(base, opts)})
        scope, chunk_ids, context, out_tok = res["scope"], res["chunk_ids"], res["context"], res.get("output_tokens")
        served = res["answer"]
        # Scored: the raw reply. glue.clean() drops a final "Answer: 4%" line as a cut-off sentence (no full stop),
        # so the served text would make the score depend on punctuation (smoke 2026-10-16). The served text is kept
        # for the judged prose (H2/D1). A decline (scope "out") is the glue's abstention.
        text = raw.get("text") if scope == "in" else f"{served}\n{W.ANSWER_KEY[lang]}: {W.ABSTAIN_LINE[lang]}"
    meta = meta_of(row, shown, context)
    W.R.set_token_counter(lambda s, n=out_tok: n if n is not None else int(len(s.split()) * 1.4))
    a = W.analyse(text, meta)
    line = a["facts"]["final_line"]
    return {"id": row["id"], "lang": lang, "type": row["type"], "scope": scope, "verdict": a["facts"]["verdict"],
            "abstained": bool(line) and W.ABSTAIN in W.candidate_hits(line, {}), "final_line": line,
            "total": a["total"], "components": a["components"], "chunk_ids": chunk_ids, "output_tokens": out_tok,
            "detect_lang": G.detect_lang(row["q"]), "completion": text,
            **({"served": served} if mode == "glue" else {})}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--limit", type=int, default=0, help="valid40 only: first N rows")
    a = ap.parse_args(argv)
    mode, model = ARMS[a.arm]

    ra.check_code()
    if subprocess.run(["git", "-C", str(CV.parent), "status", "--porcelain", "--", "coapi_voice/rl40"],
                      capture_output=True, text=True).stdout.strip():
        sys.exit("REFUSED: coapi_voice/rl40 has uncommitted changes; the scorer must be the committed one")
    ra.use_index(ra.INDEX_LIVE)
    r3 = json.loads(R3_MANIFEST.read_text()) if R3_MANIFEST.exists() else None
    if r3:
        ra.DIGEST.update({"coapi-voice:r3": r3["digest"]["coapi-voice:r3"], "coapi-voice:r3-q8": r3["digest"]["coapi-voice:r3-q8"]})
    elif model.startswith("coapi-voice:r3"):
        sys.exit("REFUSED: models/r3/MANIFEST.json missing — run fuse_quant40.sh first")
    if a.set == "v3":
        if not r3:
            sys.exit("REFUSED: no v3 cell before R3 is registered (all arms are scored after training, together)")
        if a.limit:
            sys.exit("REFUSED: v3 is scored whole, once")
        if list((CV / "runs").glob(f"exp040_{a.arm}_v3_*/summary.json")):
            sys.exit(f"REFUSED: {a.arm} already has its one v3 pass")
    ev.OLLAMA = a.host
    digest = ra.check_server(a.host, model)

    rows = load_set(a.set)
    if a.limit:
        rows = rows[: a.limit]
    twins = {r["twin_of"] for r in rows if r.get("twin_of")}
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    out = CV / "runs" / f"exp040_{a.arm}_{a.set}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    t0 = time.time()
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            rec = answer_row(row, mode, model, idx)
            rec["kind"] = kind_of(row, twins)
            rec["twin_of"] = row.get("twin_of")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"{a.arm}/{a.set} {i}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    recs = [json.loads(l) for l in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    by = lambda key: {k: {"n": n, "exact": round(sum(r["verdict"] == "right" for r in recs if key(r) == k) / n, 4)}
                      for k, n in sorted(Counter(key(r) for r in recs).items())}
    summary = {"arm": a.arm, "set": a.set, "mode": mode, "model": model, "digest": digest[:12], "n": len(recs),
               "exact": round(sum(r["verdict"] == "right" for r in recs) / len(recs), 4),
               "verdicts": dict(Counter(r["verdict"] for r in recs)), "by_kind": by(lambda r: r["kind"]),
               "by_lang": by(lambda r: r["lang"]), "by_type": by(lambda r: r["type"]),
               "glue_declined": sum(r["scope"] == "out" for r in recs),
               "detect_lang_mismatch": sum(r["detect_lang"] != r["lang"] for r in recs),
               "options": ev.OPTIONS, "think": False, "server": "ollama 0.33.0", "code_pin": ra.CODE_PIN,
               "sha256": {"set": sha(SETS[a.set][0]), "reward40.py": sha(HERE / "reward40.py"), "prompt40.py": sha(HERE / "prompt40.py"),
                          "run_arm40.py": sha(Path(__file__))},
               "r3_manifest": r3, "seconds": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("arm", "set", "n", "exact", "verdicts", "by_kind", "glue_declined")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
