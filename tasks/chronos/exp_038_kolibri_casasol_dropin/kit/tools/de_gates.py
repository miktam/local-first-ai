"""exp_038 DE slice: the grid and the code gates (DESIGN §4.3 steps 2 and 5). Code only, never a model.

The grid (registered here before the writer starts): 36 rows, 6 per módulo, types numeric 11, simple 10, multi 9,
advice 3, oos 3. Advice rows in módulos 1, 3, 5; out-of-scope rows in 2, 4, 6.

Row schema (one JSON object per line, in grid order):
  id "DE-M<m>-<nn>", twin_id "TW-M<m>-<nn>", module int, type, lang "de",
  q          the German question (persona: a German-speaking buyer, seller or owner from DE/AT/CH on the Costa del Sol)
  q_en       its English twin (a faithful translation, not a paraphrase toward the guide)
  gold       the reference answer, in English
  key_rule   the rule the row hinges on, at most 18 words
  expect     tema ids "M<m>:T<t>" ([] for oos)
  hand_off   bool (false for oos)
  numbers    numbers the gold states
  citations  citations the gold makes
  guide_evidence  verbatim quotes from the writer's S1 copy (never a "[withheld]" line)
  arithmetic [{"expr": "<python arithmetic>", "result": <number>, "literal": "<text the gold contains>"}] for numeric rows

Gates (exit 1 on any failure; report written beside the input):
  grid/schema; expect temas exist in the EN index; guide_evidence verbatim in the writer copy, and none of it on a
  withheld line; cited article and law numbers present in the guide; oos rows have empty expect and no hand-off;
  no Markdown in gold; every arithmetic case recomputes to its result and its literal is in the gold;
  near-duplicates: character-trigram Jaccard of q_en < 0.45 against v1, v2, doc2query and train questions (the
  exp_035 threshold; v2 questions are read by code only, never printed).

    python3 tools/de_gates.py grid                     # writes private/sets/grid_de.jsonl
    python3 tools/de_gates.py check <slice.jsonl>      # gates; report <slice>.gates.json
"""
from __future__ import annotations

import ast
import glob
import json
import operator
import re
import sys
import unicodedata
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
CASASOL = Path.home() / "REPOS/casasol"
CV = CASASOL / "coapi_voice"
WRITER_S1 = P38 / "private/writer_inputs/s1_en"
GRID = P38 / "private/sets/grid_de.jsonl"
THRESHOLD = 0.45

TYPES = {1: ["numeric", "numeric", "simple", "simple", "multi", "advice"],
         2: ["numeric", "numeric", "simple", "simple", "multi", "oos"],
         3: ["numeric", "numeric", "simple", "multi", "multi", "advice"],
         4: ["numeric", "numeric", "simple", "simple", "multi", "oos"],
         5: ["numeric", "numeric", "simple", "multi", "multi", "advice"],
         6: ["numeric", "simple", "simple", "multi", "multi", "oos"]}
FIELDS = {"id": str, "twin_id": str, "module": int, "type": str, "lang": str, "q": str, "q_en": str, "gold": str,
          "key_rule": str, "expect": list, "hand_off": bool, "numbers": list, "citations": list,
          "guide_evidence": list, "arithmetic": list}


def make_grid() -> list[dict]:
    rows = []
    for m, types in TYPES.items():
        for i, t in enumerate(types, 1):
            rows.append({"id": f"DE-M{m}-{i:02d}", "twin_id": f"TW-M{m}-{i:02d}", "module": m, "type": t, "lang": "de"})
    counts = {t: sum(r["type"] == t for r in rows) for t in ("numeric", "simple", "multi", "advice", "oos")}
    assert len(rows) == 36 and counts == {"numeric": 11, "simple": 10, "multi": 9, "advice": 3, "oos": 3}, counts
    return rows


def flat(s: str) -> str:
    s = re.sub(r"[*_`]+", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq}


def arith(expr: str):
    """Evaluate plain arithmetic (numbers, + - * / **, comparisons, and/or): no names, no calls."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _OPS:
            return _OPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        raise ValueError(f"not plain arithmetic: {ast.dump(n)[:80]}")
    return ev(ast.parse(expr.replace("_", ""), mode="eval"))


def norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9ł ]+", " ", s.replace("ł", "l")).split()


def grams(s: str) -> set[str]:
    t = " ".join(norm_words(s))
    return {t[i:i + 3] for i in range(len(t) - 2)}


def dedupe_sources() -> dict:
    data = CV / "data"
    src = {"v1": [], "v2": [], "doc2query": [], "train": []}
    for l in open(data / "eval/coapi_eval_v1.jsonl", encoding="utf-8"):
        r = json.loads(l); src["v1"].append((r["id"], r["q"]))
    v2 = data / "eval_v2/eval_v2_reviewed.jsonl"
    if v2.is_file():  # questions only, by code; never printed
        for l in open(v2, encoding="utf-8"):
            r = json.loads(l); src["v2"].append((r["id"], r["q"]))
    for f in sorted(glob.glob(str(data / "doc2query/M*.jsonl"))):
        for i, l in enumerate(open(f, encoding="utf-8")):
            r = json.loads(l); src["doc2query"].append((f"{Path(f).stem}:{i}", r["q"]))
    for f in ("train/train_v1.jsonl", "train/valid_v1.jsonl"):
        if (data / f).is_file():
            for l in open(data / f, encoding="utf-8"):
                r = json.loads(l)
                u = next(m["content"] for m in r["messages"] if m["role"] == "user")
                src["train"].append((r["id"], u.rsplit("Question:\n", 1)[-1]))
    return {k: [(i, grams(q)) for i, q in v] for k, v in src.items()}


def check(path: Path) -> int:
    grid = [json.loads(l) for l in GRID.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    writer = {p.name: p.read_text(encoding="utf-8") for p in sorted(WRITER_S1.glob("study_guide_modulo*.md"))}
    withheld_free = flat("\n".join(writer.values()))
    guide_all = flat("\n".join(p.read_text(encoding="utf-8") for lang in ("en", "pl")
                               for p in sorted((CASASOL / "casasol.ai/guide/src" / lang).glob("*.md"))))
    idx = json.loads((CV / "index/guide_en.json").read_text(encoding="utf-8"))
    temas = {":".join(c["id"].split(":")[1:3]) for c in (idx["chunks"] if isinstance(idx, dict) else idx)}
    src = dedupe_sources()
    fails, report = [], []
    if len(rows) != len(grid):
        fails.append(f"row count {len(rows)} != {len(grid)}")
    for cell, r in zip(grid, rows):
        f, info = [], {"id": r.get("id")}
        for k in ("id", "twin_id", "module", "type", "lang"):
            if r.get(k) != cell[k]:
                f.append(f"grid {k}: {r.get(k)!r} != {cell[k]!r}")
        for k, t in FIELDS.items():
            if not isinstance(r.get(k), t):
                f.append(f"schema {k}: {type(r.get(k)).__name__}")
        if not f:
            oos = r["type"] == "oos"
            if [e for e in r["expect"] if e not in temas]:
                f.append(f"expect not in index: {[e for e in r['expect'] if e not in temas]}")
            if oos and (r["expect"] or r["hand_off"]):
                f.append("oos row with expect or hand_off")
            if not oos and (not r["expect"] or not r["guide_evidence"]):
                f.append("in-scope row without expect or guide_evidence")
            for q in r["guide_evidence"]:
                if "[withheld]" in q or flat(q) not in withheld_free:
                    f.append(f"evidence not verbatim in the writer copy: {q[:50]!r}")
            for c in r["citations"]:
                for a in re.findall(r"(?:art(?:ículo|icle|ikel)?s?\.?\s*)(\d+(?:\.\d+)*)", c, re.I):
                    if not re.search(rf"\b(?:arts?\.?|artículos?|articles?|artykuł\w*)\s*{re.escape(a)}(?!\d)", guide_all):
                        f.append(f"citation article not in guide: {c!r}")
                for lw in re.findall(r"\b(\d+/\d{4})\b", c):
                    if lw not in guide_all:
                        f.append(f"citation law not in guide: {c!r}")
            if re.search(r"\*\*|^#|^\s*[-•]\s", r["gold"], re.M):
                f.append("markdown in gold")
            if len(r["key_rule"].split()) > 18:
                f.append("key_rule over 18 words")
            if r["type"] == "numeric" and not r["arithmetic"] and not r["numbers"]:
                f.append("numeric row without numbers")
            for case in r["arithmetic"]:
                try:
                    v = arith(case["expr"])
                except Exception as e:
                    f.append(f"arithmetic not evaluable: {case.get('expr')!r} ({e})")
                    continue
                want = case["result"]
                ok = (v == want) if isinstance(want, bool) else abs(float(v) - float(want)) <= max(abs(float(want)) * 1e-6, 1e-9)
                if not ok:
                    f.append(f"arithmetic wrong: {case['expr']} = {v}, not {want}")
                if case["literal"] not in r["gold"]:
                    f.append(f"arithmetic literal not in gold: {case['literal']!r}")
            g = grams(r["q_en"])
            near = {k: max((len(g & gg) / len(g | gg) if g and gg else 0.0) for _, gg in items) if items else 0.0
                    for k, items in src.items()}
            info["dedupe_max"] = {k: round(v, 3) for k, v in near.items()}
            if any(v >= THRESHOLD for v in near.values()):
                f.append(f"near-duplicate q_en: {info['dedupe_max']}")
        info["fail"] = f
        report.append(info)
        if f:
            fails.append((r.get("id"), f))
    out = {"input": path.name, "n_rows": len(rows), "dedupe_sources": {k: len(v) for k, v in src.items()},
           "threshold": THRESHOLD, "n_failed_rows": len([x for x in fails if isinstance(x, tuple)]), "rows": report,
           "fails": [str(x) for x in fails]}
    path.with_suffix(".gates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("input", "n_rows", "dedupe_sources", "n_failed_rows")}))
    for x in fails:
        print("FAIL", x)
    return 1 if fails else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["grid"]:
        GRID.parent.mkdir(parents=True, exist_ok=True)
        GRID.write_text("".join(json.dumps(r) + "\n" for r in make_grid()), encoding="utf-8")
        print(GRID, 36)
        sys.exit(0)
    sys.exit(check(Path(sys.argv[2])))
