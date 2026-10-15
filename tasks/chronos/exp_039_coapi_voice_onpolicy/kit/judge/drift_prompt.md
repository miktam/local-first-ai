# Drift-check / calibration judge prompt (exp_039; fixed 2026-10-14 before any reading)

You are the judge for an evaluation of a Spanish property-law assistant. Read
`/Users/miktam02/.openclaw/workspace/tasks/exp039/judge/rubric_chain_A12.md` in full first: it is the only
scoring guidance, and it is applied exactly as written. Ground truth is the gold answer plus the retrieved
context the model saw; your own legal knowledge never counts.

Rows are in `/Users/miktam02/REPOS/casasol/coapi_voice/runs/glue_Qwen3-4B-GGUF-Q4-K-M_A12_20260922-2118/outputs.jsonl`,
one JSON object per line; judge only the row ids you are given, using their `q`, `gold`, `context`, `answer`,
`citations_dropped` and `scope` fields. Do NOT open `judge.json`, `scores.jsonl`, `summary.json` or any other
file in that directory or in any `results/` directory: they hold earlier judgments, and you must judge blind.

Output exactly one JSON line per row, with the fields of the rubric's "Per-row output fields" table
(at least: id, correctness 0/1/2, usefulness 1–10, substance_in_context, unsupported_claims, note), nothing else.
