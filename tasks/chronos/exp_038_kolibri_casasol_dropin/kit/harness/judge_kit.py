"""exp_038 W6: the blind judge kit (DESIGN §5.2, X1-X5). Code only; the judge is a Fable subagent per block.

make     <runs-root> <out>   Build the bundles from every arm's scored rows (v1, de, twins).
         - Rows are dealt, seeded (sha256("exp038|deal")), into blocks of about 9; a DE row and its twin never share a block.
         - Each row gets an opaque key; each answer gets a per-row random code; answer order is shuffled per row.
         - A bundle holds: the redacted rubric and the exp_038 additions, the field list, and per row its key, question,
           gold, retrieved context (M0-flagged S1 lines replaced by "[withheld]" by code), the substance flag
           and the coded answers. Nothing names an arm, a model, a run or a build.
         - The code -> arm map goes to <out>/private_unblind.json; the judge never sees it.
redeal   <out>               A seeded 5 of the blocks, re-dealt to fresh judges (within-protocol reliability).
collect  <out>               Validate every judge file against the schema and unblind into per-arm scores.

Judge output, one file per block: <out>/judged/<block>.json = {"block": ..., "rows": [{"key": ..., "answers":
{"<code>": {"correctness": 0|1|2, "usefulness": 1-10, "unsupported_claims": [...], "fabricated_citations_kept":
[...], "language_ok": bool, "language_native": bool, "tail_degradation": bool, "from_weights": bool, "note": "..."}}}]}
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
BLOCK = 9
FIELDS = {"correctness": (int, {0, 1, 2}), "usefulness": (int, set(range(1, 11))), "unsupported_claims": (list, None),
          "fabricated_citations_kept": (list, None), "language_ok": (bool, None), "language_native": (bool, None),
          "tail_degradation": (bool, None), "from_weights": (bool, None), "note": (str, None)}

ADDITIONS = """[exp_038 additions, written before any output existed]
X1. Retrieval misses. Each row states substance_in_context. On a row where it is false, a statement of the gold's key rule or number that is NOT in the retrieved context scores correctness at most 1 and sets from_weights = true; usefulness is unaffected. Where it is true, from_weights stays false.
X2. You are blind. Answers carry random codes, in a random order per row. Never guess which system wrote an answer, and never compare an answer with another answer: score each one against the gold and the context only, on the absolute scale.
X3. Score only from this bundle. Don't open any other file, don't search, don't use the web.
X4. German. German answers use the same language_native convention as Polish and Spanish (rubric B). Spanish-term-at-first-use applies to German as to English and Polish. Rows marked lang "de" are asked in German. Their gold is in English; it states the rule, and the answer is expected in German.
X5. Glue text. Templated declines, advice leads and hand-offs, in every language including German, are glue output: judge them for shape only, as rubric (a) says. The stripped-citation filler ("according to the material", "gemäß dem Material", and the like) is the design's chosen phrase, not an artefact.
Scored fields: correctness 0/1/2; usefulness 1-10 (anchors in rubric 10); unsupported_claims (list of short descriptions); fabricated_citations_kept (citations in the answer that neither the context nor the gold supports); language_ok; language_native; tail_degradation; from_weights; note (one sentence). The other fields the earlier rubric parts mention are not scored here."""


def seeded(tag: str) -> random.Random:
    return random.Random(int(hashlib.sha256(f"exp038|{tag}".encode()).hexdigest(), 16))


def withheld_lines() -> list[str]:
    """The M0-flagged S1/S2 lines as text, for code-only replacement in contexts. Never printed."""
    ex = json.loads((P38 / "rights/m0_exclusions.json").read_text(encoding="utf-8"))
    casasol = Path.home() / "REPOS/casasol"
    out = []
    for rel, ranges in ex["lines"].items():
        lines = (casasol / rel).read_text(encoding="utf-8").split("\n")
        for a, b in ranges:
            out.extend(l.strip() for l in lines[a - 1:b] if len(l.strip()) > 20)
    return out


def scrub_context(ctx: str, flagged: list[str]) -> str:
    def norm(s):
        return re.sub(r"[*_`#>]+", "", re.sub(r"\s+", " ", s)).strip()
    out = ctx
    for line in flagged:
        n = norm(line)
        if n and n in norm(out):
            # the context is chunked text; replace the line's span word-by-word tolerant of whitespace and markdown
            words = [re.escape(w) for w in n.split()]
            out = re.sub(r"[*_`]*\s*".join(words), "[withheld]", out)
    return out


def load_rows(runs_root: Path) -> dict:
    """{arm: {set: {id: row}}} from <runs_root>/<arm>/<set>.jsonl (first error-free record per id)."""
    out = {}
    for arm_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        for s in ("v1", "de", "twins"):
            f = arm_dir / f"{s}.jsonl"
            if f.is_file():
                for l in f.read_text(encoding="utf-8").splitlines():
                    r = json.loads(l)
                    if not r.get("error"):
                        out.setdefault(arm_dir.name, {}).setdefault(s, {}).setdefault(r["id"], r)
    return out


def make(runs_root: Path, out: Path, flags: dict) -> None:
    rows = load_rows(runs_root)
    arms = sorted(rows)
    rubric = json.loads((P38 / "private/w6/rubric_redacted.json").read_text(encoding="utf-8"))["text"]
    flagged = withheld_lines()
    units = []  # one unit per (set, id): every arm's answer to that row
    ids = sorted({(s, i) for a in arms for s in rows[a] for i in rows[a][s]})
    rng = seeded("codes")
    unblind = {}
    for s, i in ids:
        present = [a for a in arms if i in rows[a].get(s, {})]
        base = rows[present[0]][s][i]
        key = hashlib.sha256(f"exp038|key|{s}|{i}".encode()).hexdigest()[:10]
        answers = []
        for a in present:
            code = f"{rng.randrange(16**6):06x}"
            unblind[f"{key}:{code}"] = {"arm": a, "set": s, "id": i}
            answers.append({"code": code, "answer": rows[a][s][i]["answer"]})
        rng.shuffle(answers)
        twin_of = base.get("twin_id") or (i.replace("TW-", "DE-") if i.startswith("TW-") else None)
        units.append({"key": key, "set": s, "id": i, "pair": (i if s == "de" else twin_of) if s in ("de", "twins") else None,
                      "lang": base["lang"], "type": base.get("type"), "question": base["q"], "gold": base.get("gold"),
                      "context": scrub_context(base.get("context") or "", flagged),
                      "substance_in_context": flags.get(f"{s}:{i}"), "answers": answers})
    # deal into blocks; a DE row and its twin never share one
    deal = seeded("deal")
    deal.shuffle(units)
    n_blocks = max(1, round(len(units) / BLOCK))
    blocks = [[] for _ in range(n_blocks)]
    for u in units:
        for k in sorted(range(n_blocks), key=lambda k: (len(blocks[k]), k)):
            if not (u["pair"] and any(v["pair"] == u["pair"] for v in blocks[k])):
                blocks[k].append(u)
                break
    out.mkdir(parents=True, exist_ok=True)
    (out / "private_unblind.json").write_text(json.dumps({"arms": arms, "map": unblind}, indent=1) + "\n")
    for k, b in enumerate(blocks, 1):
        bundle = {"block": f"B{k:02d}", "rubric": rubric, "additions": ADDITIONS, "fields": list(FIELDS),
                  "rows": [{x: u[x] for x in ("key", "lang", "type", "question", "gold", "context", "substance_in_context", "answers")}
                           for u in b]}
        (out / "bundles").mkdir(exist_ok=True)
        (out / "bundles" / f"B{k:02d}.json").write_text(json.dumps(bundle, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"units": len(units), "blocks": n_blocks, "answers": len(unblind), "arms": arms}))


def validate_judged(j: dict, bundle: dict) -> list[str]:
    errs = []
    want = {(r["key"], a["code"]) for r in bundle["rows"] for a in r["answers"]}
    got = set()
    for r in j.get("rows", []):
        for code, f in (r.get("answers") or {}).items():
            got.add((r.get("key"), code))
            for name, (typ, allowed) in FIELDS.items():
                v = f.get(name)
                if not isinstance(v, typ) or (allowed is not None and v not in allowed):
                    errs.append(f"{r.get('key')}:{code} {name}={v!r}")
    if got != want:
        errs.append(f"answers missing {len(want - got)}, extra {len(got - want)}")
    return errs


def collect(out: Path, judged_dir: str = "judged") -> dict:
    """Validate every judge file and unblind: {arm: {"<set>:<id>": fields}} plus the error list."""
    unblind = json.loads((out / "private_unblind.json").read_text())["map"]
    scores, errors = {}, {}
    for bf in sorted((out / "bundles").glob("B*.json")):
        jf = out / judged_dir / bf.name
        if not jf.is_file():
            errors[bf.stem] = ["no judge file"]
            continue
        bundle, j = json.loads(bf.read_text()), json.loads(jf.read_text())
        e = validate_judged(j, bundle)
        if e:
            errors[bf.stem] = e
            continue
        for r in j["rows"]:
            for code, f in r["answers"].items():
                m = unblind[f"{r['key']}:{code}"]
                scores.setdefault(m["arm"], {})[f"{m['set']}:{m['id']}"] = f
    res = {"judged_dir": judged_dir, "errors": errors, "scores": scores}
    (out / f"collected_{judged_dir}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"errors": {k: v[:3] for k, v in errors.items()}, "arms": {a: len(v) for a, v in scores.items()}}))
    return res


def redeal(out: Path, n: int = 5) -> list[str]:
    """A seeded n of the blocks for fresh judges (reliability, §5.2): their bundles are copied to bundles_redeal/."""
    blocks = sorted(p.stem for p in (out / "bundles").glob("B*.json"))
    pick = sorted(seeded("redeal").sample(blocks, min(n, len(blocks))))
    (out / "bundles_redeal").mkdir(exist_ok=True)
    for b in pick:
        (out / "bundles_redeal" / f"{b}.json").write_text((out / "bundles" / f"{b}.json").read_text())
    print(json.dumps({"redeal": pick}))
    return pick


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        collect(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "judged")
    elif cmd == "redeal":
        redeal(Path(sys.argv[2]))
    elif cmd == "make":
        flags_path = Path(sys.argv[4]) if len(sys.argv) > 4 else None
        make(Path(sys.argv[2]), Path(sys.argv[3]), json.loads(flags_path.read_text()) if flags_path else {})
    else:
        sys.exit(f"unknown command {cmd}")
