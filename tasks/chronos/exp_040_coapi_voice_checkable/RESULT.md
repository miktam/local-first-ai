# exp_040 result: COAPI Voice — make correctness checkable

*Scored 2026-10-17 on the mini, under HYPOTHESIS.md (signed 2026-10-16 21:31 CEST, text = 4004235). v3 = 150 rows, sealed `2c6cbad1`. One pass per arm, all six in one `measure40.sh` call (06:01–07:10), after the H5 gate ([RESULT_H5.md](RESULT_H5.md)). Files: [results/v3/](results/v3/).*

## Verdict: **Not robust**

GRPO against a reward that checks the answer itself raised exact match at Q4: +0.067, CI [+0.013, +0.120], McNemar p 0.015. The same weights at Q8 show no gain: +0.013, CI [−0.027, +0.053], p 0.38. The pre-registered verdict map stops at rule 2: **shown at Q4, not at Q8 → Not robust.**

The same weights flip between Q4 and Q8 about as many answers as training flipped:

| Pair (paired over 150 rows) | A only right | B only right | Diff |
|---|---|---|---|
| **R3 vs B0i (training, Q4)** | 14 | 4 | +0.067 |
| R3q8 vs B0q8 (training, Q8) | 6 | 4 | +0.013 |
| B0i vs B0q8 (same base weights, Q4 vs Q8) | 8 | 11 | −0.020 |
| R3 vs R3q8 (same trained weights, Q4 vs Q8) | 11 | 6 | +0.033 |

That last two rows are descriptive, not registered. The training effect at Q4 is the size of quantisation churn alone. On 150 rows, this experiment cannot separate a training effect of about 0.05 from the quantiser.

## The six arms (exact match by code)

| Arm | All | Pair (60) | Single (52) | Abstain (23) | Out of scope (15) | EN | PL | ES |
|---|---|---|---|---|---|---|---|---|
| B0i (base, imatrix Q4) | 0.707 | 0.733 | 0.865 | 0.304 | 0.667 | 0.82 | 0.66 | 0.64 |
| **R3** (trained, imatrix Q4) | **0.773** | 0.833 | 0.846 | 0.435 | 0.800 | 0.86 | **0.82** | 0.64 |
| B0q8 | 0.727 | 0.833 | 0.769 | 0.304 | 0.800 | 0.82 | 0.76 | 0.60 |
| R3q8 | 0.740 | 0.833 | 0.789 | 0.348 | 0.800 | 0.86 | 0.74 | 0.62 |
| A13 (Unsloth Q4 + glue + final line) | 0.733 | 0.783 | 0.769 | 0.435 | 0.867 | 0.90 | 0.62 | 0.68 |
| R3G (R3 + glue) | 0.733 | 0.767 | 0.789 | 0.435 | 0.867 | 0.88 | 0.68 | 0.64 |

The glue declined 15 rows for A13 and 11 for R3G; a decline counts as an abstention.

## The other hypotheses (reported; the verdict is already fixed)

- **H5, gate:** PASS. Fable judged 0/40 top-reward answers gamed, Opus 1/40.
- **H3, inversions:** shown at Q4 only.
  - At Q4, contrast pairs with both twins right went 0.50 → 0.70: +0.20, CI [+0.03, +0.37], p 0.035, which meets the registered +0.10.
  - At Q8: 0.70 vs 0.70, +0.00.
  - It fails the same robustness check as H1.
- **H4, abstention:** fails.
  - Abstaining on the 38 abstention rows rose from 0.447 to 0.579: +0.13, against the registered ≥ 0.20.
  - False abstention: 0.027 for both arms (≤ 0.05 holds).
  - At Q8: +0.03.
- **H2, judged substance:** not run. No verdict branch depends on it once rule 2 fires. It stays available on the stored answers.
- **D1, deployment:** R3G and A13 both reach 0.733 exact, with 9 rows flipping each way. H3 −0.03; H4 0.605 each. **Glue over the untrained Unsloth base remains as good as glue over R3.**

## What the result says

1. **The correctness reward is not gamed.** Two readers found 1 gamed item in 40. Compare exp_039, where the form reward was gamed (run 1 stopped at H5). Making correctness checkable fixed the verifier problem.
2. **It did not produce a robust gain.** The training signal was thin:
   - 622 rows / 78 steps after the base was already 6/6 right on 71 % of the labelled categorical rows;
   - a flat training reward curve;
   - selection on a validation set where the base was already 78/79.

   What gain there is (PL +0.16 and pairs +0.10 at Q4) does not survive Q8.
3. **The measurement floor is quantisation.** On 150 rows, Q4↔Q8 of the same weights moves 17–19 answers. Any future training cell needs either a larger test set or a design that compares at a single quantisation chosen in advance, with the other reported as robustness. This one did both, and the second check is what caught it.
4. **Deployment does not change.** A13 (Unsloth Q4 + glue) equals R3G. The deployed reference stays glue over the untrained base.

## Departures and incidents (all recorded before scoring)

- **OOM on the first mbp attempt:** the pilot and the full run were started together. The GPU wired limit was raised (environment only).
- **Harness bug:** the first B0i v3 cell crashed on v3's two-language withhold lists. Fixed in `df53573`; the partial is kept in `runs/aborted/`.
- **Selection carried no information**, because validation was at ceiling.

**Publication:** as HYPOTHESIS ("weights to Hugging Face whichever way the result goes; training data not published; Andrei decides at the end").


---

## Addendum 2026-10-17: point 1 restated

Point 1 above ("The correctness reward is not gamed … fixed the verifier problem") states more than the evidence shows. The original text is left as published. Restated:

**In our audit, a reward that checks the answer was not gamed; the form reward was.**
- **What the audit was:** one audit per experiment, of 40 top-reward training answers, read by two blind readers. exp_040 found 0/40 (Fable) and 1/40 (Opus); exp_039 run 1 found 5/40.
- **What the reward cannot see:** it reads only the final line, so whether the explanation above it is right goes unchecked. The one answer Opus flagged argued "no" and then answered "yes".
- **What was not run:** the judged check of the explanations (H2), so whether explanations got better or worse under this reward is not measured.
- **What remains open:** a confident guess among 3–4 options is still paid when it lands. Options are shuffled and hedging costs −2, but this is not excluded.
