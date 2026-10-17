# exp_040 step 0 result: the importance-matrix Q4_K_M is adopted

*2026-10-15. Pre-registration: [STEP0.md](STEP0.md) (`297a830`). Analysis code `scripts/step0.py` (`bd1f66f`) was committed before any judgment existed. Raw files: `~/REPOS/casasol/coapi_voice/runs/exp040_step0/` (cells.json, batches, key.jsonl, replies, judge.jsonl, verdict.json).*

## Decision (rule fixed in advance)

d = B0i − B0rq8 = **−0.025** (95 % CI −0.20 to +0.15; 8 rows up, 9 down, 23 the same; exact sign-flip p = 1.0). The rule is d ≥ −0.10, so **the imatrix Q4_K_M recipe is adopted for every exp_040 arm.** Q8_0 stays as the robustness arm.

## Scores (40 with-substance v1 rows, three arms judged side by side)

| Arm | Model | Correctness | Usefulness | Harmful |
|---|---|---|---|---|
| B0i | ours, imatrix Q4_K_M (`92bfae6dd5dd`) | **1.375** | 4.48 | 4 |
| B0u | Unsloth Q4_K_M, no glue (`66f6d52aa56a`) | 1.325 | 4.43 | 6 |
| B0rq8 | ours, Q8_0 (`3e8eb60ca5fa`), exp_039 answers re-judged | 1.40 | 4.90 | 4 |

## Descriptive only (no decision rests on these)

- **Judge drift:** B0rq8 re-judged at 1.40, against 1.425 in exp_039's Q8 control. A 0.025 drift across sessions.
- **B0i − B0u = +0.05** (CI −0.15 to +0.25). Our guide-calibrated matrix is level with Unsloth's quantisation.
- **B0u − B0rq8 = −0.075** (CI −0.275 to +0.125).
- **B0i against our plain Q4** (1.20 in exp_039, a different judging session): +0.175. The matrix recovers most of the ~0.2 loss. Plain Q4 was not re-judged in these batches, so this crosses sessions.
- **Decomposing exp_039's A12 − B0r = +0.225**, crossing sessions:
  - quantiser: B0u − B0r ≈ +0.125;
  - glue: A12 − B0u ≈ +0.10.

  So exp_039's "glue over the base" gain was about half quantiser.

## Build record

| Item | Value |
|---|---|
| Calibration text | the live index EN `f1e28120` + PL `9a422b0e`: 1,036 chunk texts, 987,985 bytes, sha256 `8c6bfa80…` |
| `llama-imatrix` | b11577, defaults (context 512); 583 chunks; final PPL 11.2959 ± 0.0973 |
| Importance matrix | `imatrix_guide_en_pl.gguf`, sha256 `a6f11973e2c7808b…` |
| Source f16 | `qwen3-4b-f16.gguf`, sha256 `6723dd8b97280115…` (exp_039's) |
| Quantised GGUF | `qwen3-4b-Q4_K_M-imat.gguf`, 2,497,281,184 bytes, sha256 `38623d2a0c34cdc5…` |
| Ollama | `qwen3-4b-b11577:q4km-imat`, digest `92bfae6dd5dd`; Modelfile differs from exp_039's Q4 only in `FROM` |
| Ruler checks | before each cell, 3/3 byte-identical to 2026-09-22 (`exp039_repro_20261015-062642`, `-063232`) |
| Runs | `exp040_B0i_v1_20261015-062705`, `exp040_B0u_v1_20261015-063254` |
| Judges | 8 fresh Fable instances, exp_039 scoring prompt and rubric verbatim; 120 judgments collected, none missing |

## Limits

- 40 rows and one judge family; a CI of ±0.18 means "level" is the most this can say.
- The calibration is in-domain on purpose. The guide is also the retrieval source, and v1's gold comes from it. The matrix measures which weights matter on that text; it carries no answers.
