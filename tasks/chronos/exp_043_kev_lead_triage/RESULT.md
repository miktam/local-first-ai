# exp_043 result: Kev-4B lead triage

*Scored 2026-10-18 on the mini, under HYPOTHESIS.md (signed 2026-10-18 12:01 CEST). One pass on the sealed test set: 235 synthetic leads in 7 languages, `8495fbd6`. Code: `kev_knife.py` rev 3, casasol `d581267` ([kit/kev_knife_rev3_measured.py](kit/kev_knife_rev3_measured.py)); runner guard `626e48e`. Files: [results/test/](results/test/), [results/dev_runs/](results/dev_runs/), [data/](data/) (all synthetic, all public), [kit/](kit/).*

## Verdict: **Blade usable for triage** (H1, H2, H3 pass)

| Hypothesis | Test | Bar | Majority baseline | |
|---|---|---|---|---|
| **H1 intent** (8 classes) | **0.894** | ≥ 0.85 | 0.243 | pass |
| **H2 spam recall / false spam** | **0.906 / 0.015** (29/32, 3/203) | ≥ 0.90 / ≤ 0.02 | — | pass, narrowly on both |
| **H3 next action** | **0.898** | ≥ 0.75 | 0.379 | pass |
| H4 qualified | 0.783 | ≥ 0.80 | 0.574 | **fail** |
| H5 property type / timeline | 0.885 / 0.732 | ≥ 0.80 / ≥ 0.75 | — / 0.451 | pass / **fail** |
| H6 financing / foreign buyer | 1.00 (n 50) / 0.887 (n 80) | ≥ 0.80 | — | pass |
| H7 urgency within ±1 | 0.930 (exact 0.759) | ≥ 0.85 | — | pass |

- **Intent by language:** SV 1.00, ES 0.97, DE 0.90, FR 0.89, EN 0.87, RU 0.87, **PL 0.77**.
- **Speed:** median 1.6 s per lead, every judgement in one call.
- **Dev → test:** intent 0.933 → 0.894; action 0.958 → 0.898. The usual drop from tuning on dev.

## What went wrong (descriptive, after the verdict)

- **All 3 false spams came from the "business pitching the agency" rule.** Each scored b2b 0.60–0.63, just over the cut of 0.6. The cut was set on dev, where the margin was already noted as thin (no client above 0.55). The three were a developer looking for building plots, a company booking a villa for a team retreat, and **an owner listing a €1.9M villa**. Auto-discarding would have thrown away real business.
- **Buy → holiday rental (10 of 25 intent errors).** These are buyers of a holiday home, or buyers in town for a few days ("over from Leeds for the week", "flying in Thursday for four days"). The holiday and short-stay cues pull Kev toward holiday rental.
- **Qualified (0.78) and timeline (0.73)** are the weak fields. Next action still reaches 0.90, because the rules use them together.

## Known limits (as registered)

- **Synthetic leads.** Ambiguous cases were filtered out by construction (85 of 440 dropped on checker doubt), so real leads will be harder.
- **One model, one pass.**

## SO WHAT (2026-10-18)

**Adopted:** nothing in production yet. The verdict licenses use as **agent assist**: intent, next action, urgency, property type and financing/foreign flags shown as suggestions. It does not license automatic discarding. 1.5 % false spam is within the bar, but each one is a lost client, and one of them was a seller.

**Proposed cascade, for Andrei to decide:**
1. **Run lead triage in shadow on CasaSolGuide's inbound messages,** the same pattern as the gate: log Kev's labels per turn, act on nothing. This gives the first real-lead distribution.
2. **Never auto-discard.** "Spam" goes to a review folder.
3. **Show only the fields that passed.** Qualified and timeline stay internal.

**Seeds:**
1. **A safer spam cut.** Raise b2b to ≥ 0.7, or require b2b and Kev's own spam choice to agree, then re-measure false spam. It needs a fresh test, because this one is used.
2. **Holiday home vs holiday rental:** sharpen the descriptions, or add a "buying or renting?" yes/no fact.
3. **Real leads.** An anonymised sample, with consent, from agencies we work with is the test that matters next.
4. **Polish is weakest (0.77),** the same pattern as the COAPI gate.

**After the result (2026-10-18, Andrei):** spam is never discarded. Its action is `review`, a human review folder ([kit/kev_knife_current.py](kit/kev_knife_current.py); not measured on this test). "We don't want to miss any business."
