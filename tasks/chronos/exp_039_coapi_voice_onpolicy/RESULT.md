# exp_039 — result: Substance kept at Q4_K_M, form not learned; no training gain (Q8_0 control)

*Computed 2026-10-14 by `scripts/analyze.py`, committed `bea541c` before any arm was scored, from the signed pre-registration (HYPOTHESIS.md, amendments 1–4). Rev 2 of this write-up was revised on 2026-10-15 after an adversarial Fable review (`REVIEW_2026-10-15.md`). That review recomputed every number from the raw files and found no mismatch. It narrowed several readings, and this version applies all nine of its must-fix items. Rev 1 of this write-up was drafted by Opus 5.5 and is in git history (`556b731`). Nothing about exp_039 is published.*

## Verdict

**Registered verdict (Q4_K_M, the phone format), verdict map step 3: "Substance kept, form not learned."**
- H5 passed on run 2.
- H2a, non-inferiority at −0.10 against the base quantised the same way, passed under every test method.
- H1 failed: R2 alone meets 0 of 7 format floors.

**Amendment 5 control (Q8_0, 2026-10-15): "Not shown."** The apparent +0.20 gain at Q4_K_M was quantisation, not training.
- At Q8_0, R2 − base on the same 40 rows is **−0.075** (CI −0.20 to +0.05).
- The untrained base at Q8_0 scores 1.425: the same as A12 (Unsloth's Q4 of the same weights) and close to exp_035's base (1.50).
- Our b11577 Q4_K_M recipe cost the untrained base about 0.2, and R2 at Q4_K_M did not lose that.

**What the experiment supports:**
- GRPO against a code-only verifier did not install the format.
- It did not raise correctness.
- At the deployed Q4_K_M it is not worse than the base quantised the same way.
- At Q8_0, non-inferiority is not shown (lower bound −0.20).

So the contrast with exp_035 holds only in the narrow form: no large loss like SFT's −0.3, where this run's point estimate is −0.075. Even the "kept" claim cannot be stated at full precision.

**Practical finding:** our Q4_K_M pipeline, without an importance matrix, is lossy for this base. Unsloth's quantisation of the same weights scores like our Q8_0.

The glue stays the deployed reference, on Unsloth's base. R2 is kept as evidence.

## What was measured

- **R2 = run 2, checkpoint 100** (reward rev 2, amendment 4). Run 1 stopped at H5 and is never scored.
- **Arms**, all served by Ollama 0.33.0 on the post-audit index:

  | Arm | Model | Prompt / post-processing |
  |---|---|---|
  | B0-r | untrained base, quantised through R2's own pipeline | R2's exact prompt |
  | R1 | R2 alone | R2's exact prompt |
  | A12 | Unsloth's quantisation of the base | glue |
  | R1+G | R2 | glue |

  The ruler check passed (3/3 byte-identical to 22 Sep). Nine cells ran with no refusals.
- **Judge:** Fable 5.1, 18 fresh instances, 360 judgments.
  - The four answers to each row were shown together, shuffled and unlabelled, under the A12 rubric chain verbatim.
  - **The judge could tell glue from non-glue arms.** Its notes name the glue template on 119 of 120 glue-arm v1 answers, and R1+G's text starts identically to R1's on 76/90 rows. Both registered comparisons are within a class (R1 vs B0-r, R1+G vs A12), so they are not biased by this. Raw-vs-glue usefulness comparisons are partly mechanical.
- **Judge drift:** on the 10 calibration rows, today's Fable matches exp_035's judge on 9/10 (means 1.00 vs 1.10). exp_038's fresh judge gave the same A12 answers 1.067 against exp_035's 1.217.
- **With-substance subset:** the judge flags each row from its context.
  - v1: 40 of 57 in-scope rows; 17 excluded as retrieval misses, 3 out of scope.
  - v2.1: 20 of 28 in-scope rows; 8 excluded.
  - On the 17 excluded v1 rows, R2 − B0-r is +0.18 (3 up, 0 down), so the subset does not create the effect.

## Results

| set | arm | correctness | with substance (n) | usefulness | harmful | Spanish term 1st use | tail |
|---|---|---|---|---|---|---|---|
| v1 | B0-r | 0.95 | 1.20 (40) | 3.50 | 13 | 35% | 9 |
| v1 | R1 (R2 alone) | 1.15 | 1.40 (40) | 4.02 | 7 | 43% | 6 |
| v1 | A12 | 1.18 | 1.43 (40) | 4.43 | 12 | 37% | 8 |
| v1 | R1+G | 1.18 | 1.40 (40) | 4.72 | 7 | 43% | 6 |
| v2.1 | B0-r | 1.07 | 1.35 (20) | 3.53 | 6 | 27% | 4 |
| v2.1 | R1 | 1.10 | 1.45 (20) | 3.70 | 8 | 18% | 2 |
| v2.1 | A12 | 1.03 | 1.25 (20) | 3.90 | 6 | 36% | 2 |
| v2.1 | R1+G | 1.17 | 1.45 (20) | 4.77 | 7 | 18% | 2 |

**exp_035 alongside, as the pre-registration requires.** These were judged on the old index, with B0 on the English prompt:

| arm | with substance | overall | harmful |
|---|---|---|---|
| B0 | 1.50 (38 rows) | — | — |
| A12 | 1.50 | 1.217 | 13 |

## Hypotheses

**H2a (primary): PASS.** Non-inferiority at −0.10 holds under every test.
- R2 − B0-r on the 40 with-substance v1 rows is **+0.20**:
  - registered stratified paired bootstrap: 95% CI **+0.05 to +0.375**;
  - unstratified bootstrap: +0.025 to +0.375;
  - paired t: +0.02 to +0.38;
  - bootstrap p(mean ≤ −0.10) = 0.0001.
- Rows: 9 up, 2 down, 29 unchanged.
- **Superiority was not pre-registered.** It holds for the registered interval but fails the exact sign-flip permutation test at α = 0.05 (p = 0.056). It is *suggestive, not shown*.
- **Harmful rows went from 13 to 7:** 7 fixed, 1 new, McNemar exact p = 0.07. Five of the 7 fixed rows are retrieval misses. On rows with the rule in context, the change is 6 → 4.

**The quantisation confound.** B0-r and A12 carry the same untrained weights.
- On the same 40 rows, A12 − B0-r = **+0.225** (CI +0.05 to +0.425). The two differ by quantiser (Unsloth's against ours) and by glue.
- Glue adds nothing to R2: R1+G − R1 = 0.00 on all 40 v1 rows and all 20 v2.1 rows.
- So our own Q4_K_M pipeline appears to cost the untrained base about as much as R2 gains.
- R2 (1.40) is level with A12 (1.43) and below exp_035's B0 (1.50, on a different index and prompt).

**The design cannot separate "GRPO raised correctness" from "GRPO recovered what our quantisation cost the base."** What it does show is the non-inferiority: on the same base, adapter class and data family, verifier-scored GRPO did not lose substance the way SFT did.

**H2b (descriptive, v2.1):** +0.10, CI −0.25 to +0.45, n = 20, sign-flip p = 0.79. Harmful rows: 8 for R2 against 6 for B0-r. Uninformative.

**H1: NULL, 0 of 7 floors** (R2 on v2.1, scripted, no post-processing):

| Floor | R2 | B0-r |
|---|---|---|
| Language | 96.7% | |
| Plain text | 90.0% | |
| Out of scope | 0/2 | |
| Hand-off | 3.6% | |
| Citation in context | 86.0% | |
| Numbers | 97.9% | 99.1% (met) |
| Spanish term | 78.3% | 87.0% (met) |

Two floors are missed by a single item. B0-r meets the same two floors, so on those two R2 is under its own base. R2 did not learn the hand-off even on advice rows: 0/4 on v1 and 0/2 on v2.1. The form floors are met only through the glue.

**H3: passes by the letter only.** The absolute tail check (2/30 ≤ 3) is also met by A12 (2/30) and by R2 alone (2/30), so it does not tell R1+G apart from the reference. Both checks that compare against A12 fail:
- Spanish term at first use: 18% against 36%;
- harmful rows: 7 against 6.

"The ceiling moves" is not shown. The verdict map never reaches H3.

**H4: NULL on the module-range check, which cannot be met at this n.** With 4–5 rows per module, one row moves a mean by 0.20–0.25. B0-r and A12 fail it too (range 0.4 each).
- Módulo 6, never trained, is R2's best module (1.6); Módulo 1 is its worst (0.6, against B0-r's 1.2). There is no leak signature.
- Unsupported numbers: 2 against 4. Citations the scorer cannot match in the context: 4 against 22. But R2 cites less (79 against 112 citations, on 41 against 48 rows), and the judge's count of unsupported claims is flat: 90 on 41 rows against 95 on 44.
- This is a drop in unmatched citation strings, not in fabrication.

**D1 (descriptive): 3/10.** Rule: `scripts/d1_compare.py`, using E35's five fields.
- R2 at Q8_0 and at Q4_K_M give different greedy text from the first 60 characters on 8/10 rows, and differ in extracted numbers or citations on 7/10.
- Every scored arm is a single Q4_K_M realisation. Quantisation noise of this size sits inside the H2a difference, and the paired design does not estimate it.

## H5 resolution

Run 1's audit stopped the run: Fable 5/40 gamed, 3 of them padding. Run 2 passed at 2/40, with no padding verdicts.

The 3-item change between runs equals the Fable–Opus disagreement on run 1 (5 against 2), and run 2 had no second reader. So the gate records that run 2 passes. It does not show that removing the floor caused the drop.

## Caveats

- **Training was not pre-registered.** The measurement was pre-registered (2026-10-14 09:28), and amendments 1–4 were all signed before scoring.
- **Run 2 is the second reward revision** after a failed gate (see H5 resolution above).
- **v1 golds were never re-checked against current law.**
- **The same model family audited, judged and reviewed this experiment.** Fable did the H5 reads, the scoring and this review; Opus 5.5 drafted the write-up. The numbers are recomputed from raw files, but the rubric readings are not independent of the model family.

## Notes for the exp_035 post update

On fresh v2.1 rows, A12's scope gate declined 0 of 2 out-of-scope questions. Its "all out-of-scope rows declined" floor holds on v1 only.

## Deployment note: Glue + R2 instead of A12

Not separated from A12 on the data:
- with-substance difference on v1: −0.025, CI −0.20 to +0.15;
- harmful rows on v1: 7 against 12, McNemar p = 0.23;
- on v2.1 the harmful count reverses: 7 against 6.

A swap needs its own pre-registered test.

## Amendment 5 result: the Q8_0 quantisation control (run 2026-10-15)

**Setup:**
- B0-r-Q8: `qwen3-4b-b11577:q8`, `3e8eb60ca5fa`, GGUF `391074283823…`.
- R2-Q8: `coapi-voice:r2-q8`, `8478de4545b7`.
- Both ran on all 60 v1 rows through `rl/measure.sh`; the ruler check passed again.
- The 40 with-substance rows were judged blind by 8 fresh Fable instances, two answers per row, shuffled.
- Analysis: `scripts/q8_control.py`, committed `6061c20` before any Q8 answer was read.
- Data: `casasol/coapi_voice/runs/exp039_q8/`.

| | B0-r-Q8 | R2-Q8 |
|---|---|---|
| Correctness, 40 with-substance v1 rows | 1.425 | 1.35 |
| Harmful | 3 (M3-01, M4-02, M4-03) | 5 (M1-08, M2-02, M3-01, M4-03, M5-01) |

**Primary (signed rule):**
- d = R2-Q8 − B0-r-Q8 = **−0.075**, stratified bootstrap 95% CI **−0.20 to +0.05**.
- Exact one-sided sign-flip p = 0.89. Rows: 4 up, 7 down, 29 unchanged.
- **Verdict: "Not shown."** The gain criteria fail, and the lower bound is below −0.10, so non-inferiority at Q8_0 is not shown either.

**Reading:**
- The same 40 rows at Q4_K_M gave B0-r 1.20 and R2 1.40 (the main session; judge drift applies across sessions).
- The base moves +0.225 from Q4_K_M to Q8_0. R2 moves −0.05.
- The H2a "gain" is the base's quantisation loss under our recipe, which R2 does not show. It is not a training effect.
- Whether GRPO made R2's correctness more robust to Q4 quantisation is a hypothesis this design cannot test (one Q4 realisation per arm).

## Open, cheap strengthening (from the review)

- **(iii) Score B0-r and R2 at Q8_0 on the 40 with-substance v1 rows: done. Amendment 5, above: "Not shown".**
- **(ii) An Opus read of run 2's 40 H5 items** (40 reads), so the pass has the inter-reader check the stop had.
- **(i) Judge D1's 10 Q8_0 answers** (10 judgments).

## Files

- `casasol/coapi_voice/runs/exp039_results/`: `verdict.json`, `judge.jsonl`, `judge_rows.jsonl`, `cells.json`, `d1.json`, `key.jsonl`, `batches/`, `replies/`.
- Model files: `coapi_voice/models/coapi-voice-r2/` (adapter `0c57f0e7…`, GGUF `86fcc330…`, Ollama `0d974e5fc4a3`).

## Addendum 2026-10-24: two readings narrowed

- **The arithmetic trace.** In `results/training/r2_per_step.csv`, 52 of 105 steps log `arith` 0.0. A step with no arithmetic question also logs 0.0, and the CSV has no coverage column to tell the two apart (`kit/rl/reward.py`). Means over 21-step blocks run 0.548, 0.262, 0.012, 0.419, 0.067. The trace cannot show that training traded arithmetic away. The hand-off and decline terms have the same gap.
- **"Unsloth's quantisation … scores like our Q8_0"** compared A12, which is Unsloth's Q4_K_M *behind the glue*. exp_040's `STEP0_RESULT.md` later judged Unsloth's Q4_K_M without the glue at 1.325. That splits the 0.225 gap over our Q4_K_M into about +0.125 from the quantiser and +0.10 from the glue, across judging sessions. The Q8_0 control's verdict (no gain from training) does not depend on this.
