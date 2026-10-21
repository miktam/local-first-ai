# exp_048 result: the middle-sized decision model

*Scored 2026-10-21, 06:55, on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-21 06:19 CEST, `267d620`; text as in `20cdc55`). Andrei ran the test pass in tmux, 06:19–06:55; Claude wrote it up, as Andrei asked. Set: the same 566 enquiries as exp_047 (`a5f823d9…`). Code: casasol `3f695d1` (`flash48.py` `77d18ef2…`); triage = rev 4.1 as frozen (`2acc7a1d…`). Model: Clef-Flash 9B Q8_0 (`55c222dfc9f5…`, matching exp_041's file) on llama.cpp b11711 (`b1-9d1e036`), with no server errors. The Kev-4B and Clef 27B arms are exp_047's recorded test passes. Files: [results/test/](results/test/), [results/dev/](results/dev/). No cloud model was used anywhere.*

## Verdict: **Split the work: Kev reads the request, Flash the facts** (verdict map rule 2)

| Hypothesis | Kev-4B | **Flash 9B** | hybrid | Clef 27B | Bar | |
|---|---|---|---|---|---|---|
| **H1** qualified (565) | 0.821 | **0.901** | 0.901 | 0.936 | Flash ≥ 0.80 | pass |
| **H2** timeline (532) | 0.712 | **0.773** | 0.773 | 0.833 | Flash ≥ 0.75 | pass |
| **H3** request (566) · next step (565) | 0.919 · 0.924 | **0.965 · 0.943** | 0.919 · 0.933 | 0.991 · 0.973 | Flash ≥ Kev − 0.02 | pass |
| **H4** genuine flagged as spam (485) · spam caught (81) | 2 · 75 | **9 · 81** | 2 · 75 | 0 · 81 | Flash ≤ Kev + 1 · ≥ 0.85 | **fail** |
| **H5** Flash's share of Clef 27B's gain: request, qualified, timeline | — | **0.64, 0.70, 0.50** | — | — | ≥ 0.5 each | pass |
| **H6** hybrid: request, next step, false spam | — | — | **0.919 (±0), 0.933 (+0.009), 2** | — | ≥ Kev − 0.02; ≤ Kev + 1 | pass |

**Rule 1, Flash alone, needs H1–H5, and H4 fails.** Rule 2 needs H1, H2 and H6, and they pass. So the candidate is the hybrid: Kev-4B answers the request side, Clef-Flash answers every other question, and rev 4.1's rules combine them. Adoption is Andrei's decision.

## What changed (descriptive)

- **Flash alone reads the request better than Kev on test,** 0.965 against 0.919 (39 vs 13 discordant, p 0.0002). It loses the verdict only on spam. **Dev said the opposite** (0.867 against 0.933). Those 60 dev enquiries were too few to call it, which is why the test decides.
- **Flash's 9 false spams are genuine clients writing through an intermediary.**
  - Lawyers for a family office or a client: X1-53, X3-57, X8-48, X6-41.
  - Relocation firms: X1-05, X3-09.
  - A developer buying land: X4-27.
  - Two other buyers: X5-25 and X2-03.
  - 7 are Flash's own spam reading. 2 (X4-27, X3-09) are rev 4.1's business-pitch rule, firing because Flash scores those firms at 0.82–0.89 against a cut tuned for Kev at 0.60.
  - This is the business look-alike group: it has caused false spam in exp_043, exp_044 and now here. Spam goes to review, never the bin, but the bar existed to fail this.
- **The hybrid gets Flash's facts at Kev's request quality and spam safety:**
  - qualified 0.901 and timeline 0.773, both showable;
  - property type 0.963, from abroad 0.951, urgency within one step 0.979;
  - next step 0.933 against Kev's 0.924.

  It doesn't get Flash's request gain (0.919, Kev's own).
- **Flash's other fields:**
  - property type 0.963, from abroad 0.951 and urgency 0.976, all close to Clef 27B;
  - financing 0.985 (2 wrong), against Clef 27B's 0.923 and Kev's 1.00.
- **Request by language, Flash against Kev:** Polish 0.941 against 0.853, German 0.971 against 0.900, Russian 0.972 against 0.901. French dips, 0.972 against 0.986.
- **Speed and memory:**
  - Flash: median 3.8 s per enquiry, about 10 GB.
  - Kev: 1.7 s, 4.5 GB.
  - Clef 27B: 13.5 s, 29 GB.
  - Hybrid: both small models, about 14 GB. The scorer computed it offline, so its time wasn't measured: about 3.8 s with the two calls running at once, 5.5 s one after the other.

## Known limits (as registered)

- **Synthetic enquiries,** labelled by two Fable readers who agreed. The ambiguous fifth was dropped.
- **Cuts tuned for Kev.** Two of Flash's false spams come from the pitch cut alone.
- **One pass.** The comparison arms are exp_047's recorded passes, and Kev's probabilities move by about ±0.001 between identical calls.
- **The hybrid was designed after looking at dev,** and the test decided it.

## SO WHAT (2026-10-21)

**Nothing changes on its own.** The bot still runs rev 4.1 on Kev-4B. There are now three measured options for Andrei:

| Option | Request | Qualified · timeline | False spam | Per enquiry | Memory |
|---|---|---|---|---|---|
| Kev-4B (today) | 0.919 | 0.821 · 0.712 (not showable) | 2 | 1.7 s | 4.5 GB |
| **Hybrid Kev + Flash** (this verdict) | 0.919 | 0.901 · 0.773 (showable) | 2 | ~3.8 s | ~14 GB |
| Clef 27B (exp_047) | 0.991 | 0.936 · 0.833 | 0 | 13.5 s | 29 GB |

The hybrid needs two decision servers. Kev's service exists; Flash's would be a new one. The bot would also need a two-call triage built from this experiment's `recombine()`.

**Seeds:**
1. **Intermediaries acting for a client.** Add a closed question ("is the sender writing on behalf of a buyer or tenant?") and let a yes block the spam reading. This is the weakness every model shows here, and it would also make Flash alone spam-safe. For rev 5.
2. **Re-tune the business-pitch cut for Flash** on dev. Its pitch scores run higher than Kev's.
3. **Flash alone with seed 1** could beat the hybrid: request 0.965, at Flash's cost only.
4. **From earlier:** the financing wording (exp_047), and labels from a different model family.

**Adopted (2026-10-21, Andrei: "let's use combination"):**
- the hybrid is rev 4.2 in casasol `kev_knife.py`;
- the Clef-Flash service is installed (`ai.casasol.triage-flash9b`);
- its regression gate passed on exp_044's sealed test: request 0.912 = rev 4, next step 0.945, false spam 2, Flash answered all 331;
- the bot was deployed the same day;
- live, the two calls take about 5.5 s, because both servers share one GPU.

Rev 5.1 (exp_049) superseded it later that day.

**Published (2026-10-21):** local-first-ai `tasks/chronos/exp_048_flash_hybrid/`, with no blog post. A privacy check ran before the push.
