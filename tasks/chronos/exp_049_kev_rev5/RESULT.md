# exp_049 result: lead triage rev 5 against rev 4.2, on a fresh set

*Scored 2026-10-21, 14:21, on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-21 12:53 CEST, `e5664b6`; text as in `e273857`). Andrei ran the test pass in tmux, 12:53–14:21 ([test_run/exp049_test_20261021-1253.sh](results/test/exp049_test_20261021-1253.sh) and `.log`). Set: 350 fresh synthetic enquiries in seven languages (`de3e1710…`; 188 standard, 54 in each hard stratum). Code: rev 4.2 `5b79950f…`, rev 5 `8d268660…` (casasol `bfcd0d6`), both with Kev-4B and Clef-Flash 9B; "today" = 21 October 2026. Files: [results/test/](results/test/), including the predictions, `compare.json`, and gemma4:26b's labels and their agreement. No model outside the mini was called during the test pass; the data was written by Fable and checked by Opus.*

## Verdict: **adopt F1 (intermediaries), F3 (dated timeline) and F4 (financing wording); F2 (price only) stays off**

| Hypothesis | rev 4.2 | rev 5 | Bar | |
|---|---|---|---|---|
| **H1** standard (188): request · next step | 0.931 · 0.931 | **0.941 · 0.936** (+0.010 · +0.005) | ≥ −0.02 each | pass |
| **H2** genuine clients flagged as spam, intermediary stratum · all genuine · spam recall | 2 · 2 · 0.963 | **0 · 0 · 0.963** | ≤ half · ≤ rev 4.2 · ≥ 0.85 and ≥ −0.05 | pass → **F1 adopted** |
| **H3** valuation stratum (54): request | 0.852 | 0.926 (+0.074; 4 vs 0 discordant, one-sided p 0.0625) | ≥ +0.10 and p < 0.05 | **fail → F2 off** |
| **H4** timing stratum (54): timeline · standard timeline | 0.444 · 0.737 | **0.870 · 0.817** (+0.426, 23 vs 0, p < 0.001 · +0.080) | ≥ +0.10, p < 0.05 · ≥ −0.02 | pass → **F3 adopted** |

**F4 (financing asks "use") comes with F3.** Financing where stated: 0.985 → 1.000, no worse.

**The adopted set is rev 5 with F2's cut off.** Call it rev 5.1. Its test numbers are exact, not estimated: recombining rev 5's recorded answers reproduces rev 5 on 350/350, and switching F2 off is a cut over those same answers.

| | rev 4.2 | **rev 5.1 (F1 + F3 + F4)** | rev 5 (all four) |
|---|---|---|---|
| request, standard | 0.931 | 0.931 | 0.941 |
| next step, standard | 0.931 | 0.936 | 0.936 |
| request, intermediary stratum | 0.907 | 0.944 | 0.944 |
| request, valuation stratum | 0.852 | 0.852 | 0.926 |
| request, all 350 | 0.923 | 0.929 | 0.946 |
| genuine flagged as spam · spam recall | 2 · 0.963 | 0 · 0.963 | 0 · 0.963 |

## What changed (descriptive)

- **F3 is the large win.**
  - Timing given only as a month, date, season or holiday is read right 0.870 of the time, against 0.444. All 23 changed enquiries went from wrong to right; none went the other way.
  - Ordinary enquiries gain too: standard timeline 0.737 → 0.817 (20 vs 5).
  - Over all 350, timeline 0.723 → 0.838.
- **F1 removes the false spam.**
  - The two genuine clients rev 4.2 flagged were Y5-06 and Y8-52, both in the intermediary stratum. With F1, none are flagged.
  - Spam caught is unchanged at 52 of 54, so the new question let no pitch through.
  - Flash's `on_behalf` was right on 0.991 of the test enquiries.
- **F2 corrected 6 enquiries and broke none,** 4 in the valuation stratum and 2 in standard. It still falls short of the registered bar.
  - Rev 5's remaining valuation-stratum errors include two owners Flash still read as selling (Y1-04, Y8-22) and one read as long-term rent (Y6-02).
  - That stratum is also where the readers disagree most. gemma4:26b agrees with only 0.859 of its request labels, against 0.975 for standard. Y5-25 is labelled `sell`; both versions and gemma read it as valuation.
- **The cross-family check.** gemma4:26b (Google; neither the writers', the checker's nor the triage models' family) labelled all 513 kept leads, with no errors.
  - It agreed with the kept request labels on 0.963 of them, `on_behalf` 0.990, action 0.932, and timeline 0.838.
  - On the 335 test enquiries where it agrees with the request label too, rev 5 reads 0.946 against rev 4.2's 0.922. The gain doesn't come from leads only Fable and Opus would label that way.
- **Request by language, rev 4.2 → rev 5:**
  - up: German 0.909 → 0.955, French 0.911 → 0.956, Russian 0.841 → 0.886, Swedish 0.889 → 0.911, English 0.941 → 0.953;
  - unchanged: Spanish 0.976, Polish 0.978.
- **Unchanged or near it:** qualified 0.873 → 0.867, property type 0.947 both.
- **Speed:** both versions ask both models, so about 5.5 s per enquiry.

## Known limits (as registered)

- **Synthetic enquiries, with hard strata written to contain the errors.** This shows the fixes work on these patterns, not how often the patterns occur in an inbox.
- **One pass per version.** The cuts were tuned on 163 dev enquiries.
- **Labels from Fable writers and an Opus checker who agreed.** The ambiguous 8 % was dropped. The independent third reader agrees on 96 % of requests.
- **Real enquiries remain the test that matters.**

## SO WHAT (2026-10-21)

**To adopt** (Andrei's steps, as with rev 4.2):
1. Rev 5.1 (rev 5 with `PRICE_CUT = None`) becomes `kev_knife.py`, and rev 4.2 is frozen as `kev_knife_rev42.py`. The `price_only` question stays in the schema, as measured, with no rule reading it.
2. The regression gate on exp_044's sealed test.
3. A bot deploy.

**In the bot after that:**
- the timeline question carries the real date (`TODAY = None`), which fixes demo 4 ("в ноябре") and TRIAGE.md weakness 12;
- a client writing through a lawyer, relocation firm or relative is no longer sent to spam;
- financing asks "use".

**Seeds:**
1. **F2, re-tested with power.** It made 6 corrections and no errors here, but 54 valuation-stratum enquiries can't show +0.10. A dedicated valuation set of about 150 would. Its labels need a reader agreement study first, since readers disagree most on valuation versus sell.
2. **Andrei's call-request rule** (READ30): "an explicit call request with a number → call now, whatever the timeline". TRIAGE.md weakness 7.
3. **Russian is now the weakest language** (0.886).

**Adopted (2026-10-21):**
- rev 5.1 is in casasol `kev_knife.py` (`897c0ac2…`), with rev 4.2 frozen as `kev_knife_rev42.py`;
- its regression gate on exp_044's sealed test passed: request 0.912 = rev 4, next step 0.952 (0.933), false spam 0 (2), spam recall 0.918 (0.939);
- the one pitch lost there is X2-02, an advance-fee scam that claims a client, which the on-behalf rule overturned. It is TRIAGE.md weakness 13, with a guard as a seed;
- the bot was deployed by Andrei the same day.

**Published (2026-10-21):** local-first-ai `tasks/chronos/exp_049_kev_rev5/`, with every writer and checker file, the gemma4:26b labels and every prediction, and no blog post. Before the push, the data was scanned for realistic numbers in digits and in words, and the privacy check ran.
