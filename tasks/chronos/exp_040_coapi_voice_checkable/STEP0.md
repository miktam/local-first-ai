# exp_040 step 0: the quantiser (pre-registration)

*Committed 2026-10-15, before the importance matrix is built and before any B0-i or B0-u answer exists. The decision rule is DESIGN.md §8's, which Andrei approved on 2026-10-15 ("yes, ok, let's go"). This file adds only the procedure.*

## Question

exp_039's Q4_K_M recipe (llama.cpp b11577, no importance matrix) cost the untrained base about 0.2 points: 1.20 at Q4 against 1.425 at Q8_0 on the 40 with-substance v1 rows.

Does an importance matrix calibrated on the guide recover that loss? And is Unsloth's Q4_K_M, without the glue, as good as Q8_0?

## Procedure

1. **Calibration text.** Every chunk text of the live index, EN `f1e28120` then PL `9a422b0e`, in index order, joined by blank lines. Its sha256 goes in the result. It is guide text only: no eval question, gold or prompt template.
2. **Importance matrix.**
   - Tool: `llama-imatrix` from llama.cpp b11577 (built from the same tree as exp_039's `llama-quantize`), with its defaults (context 512, all chunks).
   - Input: `qwen3-4b-f16.gguf`, the same f16 that exp_039's B0r and B0r-Q8 came from.
3. **Quantise.**
   - `llama-quantize --imatrix <matrix> qwen3-4b-f16.gguf Q4_K_M`.
   - Register it as `qwen3-4b-b11577:q4km-imat` on the Ollama 0.33.0 side server, with exp_039's Q4 Modelfile changed only in `FROM`.
   - Pin its digest in `run_arm.py` before any answer is generated.
4. **Arms (all in "prompt" mode, exactly exp_039's B0r pipeline):**

   | Arm | Model | Answers |
   |---|---|---|
   | B0i | `qwen3-4b-b11577:q4km-imat` | generated now |
   | B0u | `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M` (`66f6d52aa56a`): A12's model without the glue | generated now |
   | B0rq8 | `qwen3-4b-b11577:q8` (`3e8eb60ca5fa`) | **reused** from `runs/exp039_B0rq8_v1_20261015-051952`, not regenerated |

5. **Generation.**
   - `measure.sh` runs the ruler check first (A12 on v1 rows 1–3, archived index, byte-identical to 2026-09-22), then cells `B0i:v1` and `B0u:v1`.
   - Pins: glue `b1a648c`, live index, server 0.33.0, set sha `efc16ab2`.
6. **Rows.** The 40 v1 rows flagged `substance_in_context` in `exp039_results/judge_rows.jsonl`. Rows are fixed by exp_039, not chosen here.
7. **Judging.**
   - The exp_039 scoring prompt and `rubric_chain_A12.md`, verbatim.
   - Three answers per row, shuffled per row (seed `"40:s0:<id>"`), unlabelled.
   - 5 rows per batch, 8 batches, one fresh Fable instance per batch. The key is kept apart.

## Rule (DESIGN.md §8, fixed before any answer)

Let d = mean correctness(B0i) − mean correctness(B0rq8), both judged in the same batches.

- **d ≥ −0.10 → the imatrix Q4_K_M recipe is adopted** for every exp_040 arm. Q8_0 stays as a robustness arm.
- **d < −0.10 → Q8_0 becomes exp_040's primary quantisation.**

The rule uses the point estimate: step 0 is an engineering choice, not a hypothesis test. The stratified paired bootstrap 95 % CI (lang × type, 10,000 resamples, seed 40) and the exact sign-flip p are reported alongside it.

## Descriptive only (no decision rests on these)

- **B0u − B0rq8:** whether Unsloth's Q4 alone matches Q8_0.
- **B0i − B0u:** our importance matrix against Unsloth's.
- **A12 − B0u:** the glue's effect at a fixed quantiser. A12's scores come from exp_039's judging session, so this crosses sessions.
- **Judge drift:** B0rq8's re-judged mean against its 1.425 from exp_039.

## Limits, stated before the result

- 40 rows and one judge family. A 0.10 threshold is coarse at this size.
- **Calibration is in-domain on purpose,** since the guide is what the model serves. A recipe adopted here says nothing about other domains.
- **This session is not blind to v1's judge notes.** No step-0 choice depends on v1 content: the rows are exp_039's, the calibration text is the whole guide, and the judges are fresh.
