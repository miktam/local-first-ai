# exp_042: the answerability gate with fixed retrieval (design, rev 0)

*Rev 1, 2026-10-18 — Andrei decided: 1 reuse v3; 2 and 3 as recommended (my call); 4 the mini. Drafted 2026-10-18. It follows exp_041 ("No gate earns its place": Clef 27B raised abstention 0.447 → 0.947 and exact match 0.707 → 0.820, but blocked 10.7 % of answerable rows, all of which had lost their source in retrieval) and the 2026-10-17 retrieval fixes (`tasks/coapi-voice-fixes-2026-10-17.md`).*

## The question

With retrieval fixed (Spanish translated before search, two chunks from the leading tema, budget 1,350), does the gate make the system abstain when the rule is missing without blocking questions whose rule **is** in the material?

## What changes from exp_041

1. **Retrieval:** today's glue retrieval. Bank source-chunk hit: EN 0.871, ES 0.771, PL 0.899 (was 0.782 / 0.489 / 0.888).
2. **What counts as a false abstention (decision 2).** exp_041 counted every row whose source exists in the guide as answerable, even when retrieval had not returned it. The glue's own rule is "never fill the gap from memory", so the proposal is: a row is **answerable** iff its source chunk is in the material the voice saw. Blocking a row whose source was lost is then counted as correct, not as a false abstention. Source-lost rows are reported separately, with how often the voice got them right from memory.
3. **The gate is fixed in advance:** Clef 27B (G4) as primary, Kev-4B (G2) as the fast alternative. No gate selection; thresholds re-fitted on a rebuilt dev set (same rule: dev false abstention ≤ 0.03).
4. **The dev set is rebuilt with today's retrieval and the new label.** It includes answerable rows whose source was retrieved for every reason, not only bank rows.

## Arms

| Arm | System |
|---|---|
| S0 | untrained voice (B0i) with today's retrieval, no gate |
| S4 | S0 + Clef 27B gate |
| S2 | S0 + Kev-4B gate |
| D1 | the deployed glue (A13 style) with and without Clef 27B (descriptive) |

The voice's answers are regenerated with today's retrieval: about 150 greedy answers, 10 minutes.

## Test set (decision 1)

- **(a) A fresh v4 (recommended).** Built like v3 in a separate blind session from guide sections the bank did not use: 150 rows, with your 30-row read and a seal. Fable cost is about v3's.
- **(b) Reuse v3.** Cheaper, but every exp_041 gate has already had its one pass on it, and this session has read the blocked rows' analysis. A gain on v3 would be weaker evidence.

## Hypotheses (thresholds to fix in HYPOTHESIS.md)

- **H1:** S4 abstention on abstention rows ≥ S0 + 0.20, McNemar p < 0.05.
- **H2:** S4 false abstention on answerable rows (source in material) ≤ 0.05.
- **H3:** S4 exact match ≥ S0 (paired 95 % lower bound > −0.02).
- **H4:** dev ECE ≤ 0.10.
- **Reported:** the share of rows whose source was lost, per language, and the voice's accuracy on them.

## Cost

- Your time: a 30-row read for v4, plus the 2 h dev run and 30 min of v3/v4 runs, which you start.
- Fable: v4 drafting and checking, about v3's spend.
- Mini: the gate runs as in exp_041; the two models are already downloaded.

## Decisions (answered 2026-10-18)

**Known limit of reusing v3:** every exp_041 gate has already had one pass on v3, and this session has read exp_041's analysis of the blocked rows. A gain on v3 is therefore weaker evidence than one on a fresh set; it is reported as such.

**D1 (the deployed glue) is dropped:** with today's retrieval, the glue's own option and withhold handling would need new patches. The gates are measured over S0 only.


1. **Test set:** (a) fresh v4 *(recommended)* / (b) reuse v3.
2. **Answerable = source in the material** *(recommended; it follows the glue's "never from memory")* / answerable = source exists in the guide (exp_041's rule).
3. **Gates:** Clef 27B primary + Kev-4B alternative, fixed in advance *(recommended)*.
4. **Deployment target if it passes:** the mini (Clef 27B fits) *(recommended)*; the phone would need Kev-4B or smaller and is out of scope here.
