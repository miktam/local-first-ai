"""exp_038 W6: the mechanical redaction of exp_035's rubric chain (DESIGN §5.2).

Source: local-first-ai exp_035 results/designA_A12_judge_20260922.json. Kept: rubric_notes (1-18) and the additions
(A-E), designA (a)-(e), A6 (f)-(g), A8 (h)-(i), A9b (j)-(k), A12 (l)-(o), in that order. Dropped: judge, model, run,
rubric_source, field_notes, rows, summary.

Rules, applied in order, by code only:
  1. a parenthetical that contains a row id (M1-08, V2-M3-01, ...) -> "([example withheld])"
  2. a row id with a quoted span on either side, a ": figure" after it, or its own parenthetical description\n     -> "[example withheld]" (single quotes count only at word boundaries, so apostrophes are never quotes)
  3. any remaining row id -> "[example withheld]"
  4. build and model names (B2m, B3, A4 ... A12, A9b, G3, Design-A, designA, 4B, Qwen, Gemma, gemma-3-4b) -> neutral words
  5. section labels "rubric_additions_<build>" are not reproduced; sections are numbered in order instead
The sha256 of the source sections (joined) and of the redacted text are recorded.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
SRC = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_035_coapi_voice/results/designA_A12_judge_20260922.json"
SECTIONS = ["rubric_notes", "rubric_additions", "rubric_additions_designA", "rubric_additions_A6", "rubric_additions_A8",
            "rubric_additions_A9b", "rubric_additions_A12"]
ROW = r"(?:V2-)?M\d-\d{2}"
SQUOTE = r"(?:(?<!\w)'[^']{1,160}'(?!\w)|\"[^\"]{1,160}\"|«[^»]{1,160}»|‘[^’]{1,160}’)"  # apostrophes inside words are not quotes
BUILD_NAMES = [
    (r"\bthe (?:A9b|A1[0-2]|A[4-8]|G3|B3) (judge|score)\b", r"an earlier \1"), (r"\bthe (?:A9b|A1[0-2]|A[4-8]) judge's\b", "an earlier judge's"),
    (r"\bB2m\b", "the base rubric"), (r"\bB3-specific\b", "Additional"), (r"\bB3\b", "the earlier build"),
    (r"\bDesign-A-specific\b", "Glue-specific"), (r"\bDesign-A\b", "the glue"), (r"\bdesignA\b", "the glue"),
    (r"\bA9b\b", "an earlier build"), (r"\bA1[0-2]\b", "an earlier build"), (r"\bA[4-8]\b", "an earlier build"),
    (r"\bG3\b", "another"), (r"\b4B\b", "small"), (r"\bQwen[\w.-]*", "a model"), (r"\bgemma[\w.-]*", "a model"),
    (r"\bGemma[\w.-]*", "a model"),
]


def redact(text: str) -> str:
    paren = r"\((?:[^()]|\([^()]*\))*"  # a parenthetical, one level of nesting allowed
    t = re.sub(rf"{paren}\b{ROW}\b(?:[^()]|\([^()]*\))*\)", "([example withheld])", text)
    t = re.sub(rf"\b{ROW}\b\s*\([^()]*\)", "[example withheld]", t)                     # id + its description
    t = re.sub(rf"{SQUOTE}\s*\b{ROW}\b", "[example withheld]", t)                          # 'fragment' id
    t = re.sub(rf"\b{ROW}\b(?:\s*:\s*[^\s,;)]+)?\s*{SQUOTE}?", "[example withheld]", t)   # id 'fragment' / id: figure
    t = re.sub(rf"\b{ROW}\b", "[example withheld]", t)
    for pat, rep in BUILD_NAMES:
        t = re.sub(pat, rep, t)
    return t


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    src = "\n\n".join(d[k] for k in SECTIONS)
    parts = [redact(d[k]) for k in SECTIONS]
    out_text = "\n\n".join(f"[Rubric part {i + 1}]\n{p}" for i, p in enumerate(parts))
    left = sorted(set(re.findall(rf"\b{ROW}\b", out_text)))
    rec = {"source": "exp_035 results/designA_A12_judge_20260922.json (" + ", ".join(SECTIONS) + ")",
           "source_sha256": hashlib.sha256(src.encode("utf-8")).hexdigest(),
           "redacted_sha256": hashlib.sha256(out_text.encode("utf-8")).hexdigest(),
           "row_ids_left": left, "withheld_count": out_text.count("[example withheld]"), "text": out_text}
    out = P38 / "private/w6/rubric_redacted.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("source_sha256", "redacted_sha256", "row_ids_left", "withheld_count")}))


if __name__ == "__main__":
    main()
