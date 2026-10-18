# exp_041 result: an answerability gate

*Scored 2026-10-17 on the mini, under HYPOTHESIS.md (signed 2026-10-17 16:09 CEST, text = d4e7778). Thresholds were frozen from dev before any v3 call (casasol 50b4956). Each gate made one v3 pass. Files: [results/v3/](results/v3/), [results/dev/](results/dev/).*

## Verdict: **No gate earns its place** (H2 fails)

The primary gate, chosen on dev, is G4 (Clef 27B, τ 0.487). Layered over the untrained voice (B0i) on v3:

| | B0i alone | B0i + G4 | Rule | |
|---|---|---|---|---|
| **H1:** abstains on the 38 abstention rows | 0.447 | **0.947** | rise ≥ +0.20 and p < 0.05 | +0.50, CI [+0.34, +0.66], p < 0.001: **pass** |
| **H2:** false abstention on the 112 answerable rows | 0.027 (3) | **0.107 (12)** | ≤ 0.05 | **fail** |
| **H3:** exact match over 150 | 0.707 | **0.820** | 95 % lower bound > −0.02 | +0.113, CI [+0.073, +0.153]: **pass** |
| **H4:** calibration on dev (ECE) | — | 0.065 | ≤ 0.10 | **pass** |

The verdict map's first rule fires on H2.

## All four gates (each over B0i; D1 = each over A13, the deployed glue)

| Gate | Abstention (38) | False abstention (112) | Exact (150) | D1 exact over A13 (0.733) |
|---|---|---|---|---|
| G1 Qwen3-4B prompted | 0.632 | 0.054 (6) | 0.747 | 0.753 |
| G2 Kev-4B | 0.895 | 0.125 (14) | 0.800 | 0.780 |
| G3 Clef-Flash 9B | 0.763 | 0.098 (11) | 0.773 | 0.773 |
| **G4 Clef 27B** | **0.947** | 0.107 (12) | **0.820** | **0.800** |

- Every gate raises exact match. G4 adds 17 right answers on the 150, over B0i and also over the deployed glue.
- No gate holds false abstention at 0.05. G1 comes closest (0.054) and catches the fewest.

## Why false abstention tripled from dev (2.7 %) to v3 (10.7 %)

*Descriptive, after the verdict; it does not change it.*

All 11 answerable v3 rows that G4 newly blocked had **lost their source chunk in retrieval**. The material the voice saw did not contain the answer.
- B0i got 9 of those 11 wrong anyway.
- Only 2 right answers were lost, both Spanish, both answered from memory.
- Retrieval missed the source on 22 of the 112 answerable v3 rows, mostly Spanish. Spanish questions are matched against the English index, the same weakness exp_040 found in training (53 % of ES bank rows lost their source).
- The dev set's answerable rows were drawn only from prompts whose source *was* retrieved. So dev measured false abstention on easy rows, and v3 on realistic ones.

So the gate is right about the **material** on all 11. The H2 label "answerable" is about the guide, not about what retrieval returned.

**What this means:**
1. **As a component, the gate works.** On the material it is given, Clef 27B separates answerable from not with AUROC 0.990 on dev, and catches 95 % of the abstention rows on v3. Net correctness rises by 11 points.
2. **The failure H2 counts sits upstream.** Spanish retrieval drops the source on about a fifth of answerable questions. A gate that refuses those turns a likely wrong answer into "not in the material": 9 of 11 times a gain, 2 of 11 a loss.
3. **The next step is retrieval, not the gate.** Fix ES retrieval, then re-test the gate. That needs a new pre-registration and, for a clean claim, a fresh test set: v3 has now had a pass from every gate.

## Incidents

- **Dropped before registration** ([SMOKE.md](SMOKE.md)): pplx-decider (third-party bridge), Vega-27B (llama.cpp fork), OpenJev (non-commercial license).
- **G1's smoke separation did not hold.** 8/8 rows in the smoke; on dev it caught only 35 % of withheld-source variants.

---

## Follow-up 2026-10-17: Spanish retrieval fixed (before any new experiment)

Andrei: fix it before starting a new experiment. Done in casasol `1257a46` (`coapi_voice/xling.py`, one call in `glue.answer`).
- **The fix:** a Spanish question is translated to English by the deployed model (Qwen3-4B, temperature 0, Spanish legal terms kept), and the EN index is searched with the translation. The model still sees and answers the Spanish question.
- **Measured on training-side data only** (no test set touched). Source chunk retrieved:

| ES query | Bank (734) | Validation (29) |
|---|---|---|
| original Spanish (before) | 0.489 | 0.621 |
| **English translation (adopted)** | **0.703** | **0.931** |
| translation + original | 0.666 | 0.897 |
| ceiling: the bank's own EN question | 0.782 | 1.000 |

- **Cost:** about 0.8 s per Spanish question.
- **Remaining gap:** EN itself retrieves the source only 0.782 of the time (PL 0.888), so about a fifth of questions still miss their rule in any language. That is general retrieval, not a Spanish problem.
- **Ruler note:** `glue.py` now differs from b1a648c, so the exp_039/040 runners refuse by design. They reproduce from that commit.
