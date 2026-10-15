#!/usr/bin/env python3
"""exp_039 D1 — device fidelity: R2 at Q8_0 vs Q4_K_M on E35's 10 pre-named v1 rows (descriptive).

  ~/REPOS/casasol/.venv/bin/python3 tasks/exp039/scripts/d1_compare.py RESULTS_DIR

Rule (E35 H5's field list, applied 2026-10-14 and written down here 2026-10-15 after the review flagged
that it had no provenance): a row agrees when all five fields are equal between the two answers —
answer language (run_eval.detect_answer_lang), refusal (_REFUSAL and < 80 words), hand-off (_HANDOFF),
the set of citations (_cites), and the set of numbers (_nums minus 1/2/3). Writes RESULTS_DIR/d1.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "REPOS/casasol/coapi_voice"))
import run_eval as rv  # noqa: E402

O = Path(sys.argv[1])
cells = json.loads((O / "cells.json").read_text())


def load(p):
    return {json.loads(l)["id"]: json.loads(l) for l in (Path(p) / "outputs.jsonl").read_text().splitlines() if l.strip()}


def fields(r):
    a = r["answer"]
    return (rv.detect_answer_lang(a), bool(rv._REFUSAL.search(a)) and len(a.split()) < 80, bool(rv._HANDOFF.search(a)),
            tuple(rv._cites(a)), tuple(sorted(rv._nums(a) - {"1", "2", "3"})))


NAMES = ["language", "refusal", "handoff", "citations", "numbers"]
q8, q4 = load(cells["R1q8:d1"]), load(cells["R1:v1"])
rows = []
for rid, a in q8.items():
    fa, fb = fields(a), fields(q4[rid])
    rows.append({"id": rid, "agree": fa == fb, "differs": [n for n, x, y in zip(NAMES, fa, fb) if x != y]})
d1 = {"rule": "all five E35 fields equal (language, refusal, handoff, citation set, number set)",
      "agree": sum(r["agree"] for r in rows), "of": len(rows), "rows": rows}
(O / "d1.json").write_text(json.dumps(d1, indent=1) + "\n")
print(d1["agree"], "/", d1["of"])
