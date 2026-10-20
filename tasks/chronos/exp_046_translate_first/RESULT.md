# exp_046 result: translate before triage

*Scored 2026-10-20 on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-20 17:47 CEST, `a83f9d9`; text as in `12b259b`). Andrei ran the test pass in tmux, 18:43–19:28 ([results/test/exp046_test_20261020-1843.sh](results/test/exp046_test_20261020-1843.sh) and `.log`). Set: 425 non-English enquiries from exp_043's and exp_044's test sets, public copies (`a5f823d9…`). Code: casasol `5f0a1c4` (`translate46.py` `99aa0869…`); triage = rev 4 as frozen (`b1940235…`); translator gemma4:26b (`777b927e3b6b…`, Ollama 0.40.2). Files: [results/test/](results/test/) (predictions, every translation, `done` records with hashes, `score.json`), [results/dev/](results/dev/). No cloud model was used anywhere.*

## Verdict: **No gain. Kev reads the original** (H1 fails; H2 and H3 pass)

| Hypothesis | native | translated | Bar | |
|---|---|---|---|---|
| **H1** request (425) | 0.927 | 0.925 (−0.002; 12 vs 13 discordant, one-sided p 0.66) | ≥ +0.02 and p < 0.05 | **fail** |
| **H2** next step (425) | 0.873 | 0.868 (−0.005; 10 vs 12) | ≥ −0.01 | pass |
| **H3** genuine flagged as spam (365) · spam caught (60) | 1 · 55 | 2 · 58 | ≤ native + 1 · ≥ 0.85 | pass |
| **H4** Polish request (68) | 0.853 | 0.868 (+0.015; 4 vs 3) | ≥ +0.05 | fail |

The verdict map stops at rule 2. H4 fails too, so there is no Polish-only seed.

**Correction (2026-10-20, after publication):** the next-step numbers above are understated in both arms. exp_043's labels call spam's next step "discard", and the code has called it "review" since 2026-10-18 (the same decision, renamed). `translate46.py` scored the 32 exp_043 spam enquiries as wrong in both arms. Read with "discard" = "review", next step is **0.922 native and 0.920 translated** (−0.002). H2 still passes, and the verdict is unchanged. `score.json` is kept as computed. Found while building exp_047, whose scorer converts the label.

## What changed (descriptive)

- **Translation trades errors rather than removing them.** 25 enquiries change their request: 12 get it right only when translated, 13 only natively.
  - **Helped:**
    - **valuation read as sell, 4 fixed:** X5-63 German, X3-63 and X3-48 Polish, X8-24 Russian. This is rev 4's top confusion (9 in exp_044).
    - **pitches read as a client, 5 caught:** W4-42 German, W6-06 French, X5-18 German, X4-15 Spanish, X3-62 Polish. Two other pitches were missed by translation (W5-12 Swedish, X8-50 Russian), so spam caught went from 55 to 58.
  - **Hurt:**
    - buyers read as holiday renters (W6-01, X4-18, X3-54);
    - questions read as transactions (W2-24, W2-43, W4-15, X6-51);
    - a genuine buyer read as spam (X8-48, the one extra false spam).
- **By language, request native → translated:**
  - German 0.900 → 0.914; Polish 0.853 → 0.868;
  - Spanish 0.945 → 0.932; Swedish 0.972 → 0.944 (0 vs 2);
  - French (0.986) and Russian (0.901) unchanged.
  - The source sets disagree too: exp_043 −0.011, exp_044 +0.004.
- **Next step by language:** Polish 0.765 → 0.809, German 0.871 → 0.843, Spanish 0.849 → 0.822.
- **The translations themselves were sound.**
  - 423 of 425 passed every code check. The 2 fallbacks were the translator adding a `[phone]` or `[email]` marker the source didn't have, as on dev.
  - Median 1.7 s per translation, so translation would add about 1.7 s to a triage that takes 1.7 s.
- **Kev's "English only" model card understates it.** Natively, Kev read 92.7 % of these non-English requests right. Translating them into English did not improve that.
- **Dev predicted the direction but not the size:** −0.028 on 209 enquiries, against −0.002 on test's 425.

## Known limits (as registered)

- **Synthetic enquiries,** labelled by two Fable readers who agreed (exp_043/044), and reused from earlier verdicts.
- **One translator, one prompt, one pass.** Kev's probabilities move by about ±0.001 between identical calls, so a case sitting exactly on a cut could flip from this alone.
- **Real enquiries mix languages and slang** more than these.

## SO WHAT (2026-10-20)

**Decided** (nothing to wire):
- **No translation layer in triage.** Kev keeps reading the original text. No exception to invariant 1 is needed, and gemma4:26b doesn't need to stay loaded for triage.
- **Claim wording:** "On 425 synthetic enquiries in six languages, translating into English first did not improve a local decision model's triage (0.927 native, 0.925 translated)."

**Seeds:**
1. **Valuation vs sell** (exp_047, rev 5). The English wording fixed 4 of native's valuation → sell errors. That suggests the cue is in the wording of the question, and a closed question asked separately ("do they want a price only, or to sell?") may do the same without translating.
2. **Pitches read as clients.** Translation caught 5 more pitches but cost one genuine buyer. The business-pitch question deserves a look in rev 5.
3. **Polish is still the weakest language** (0.853), and translation doesn't fix it.

**E, done in the same run (not part of this experiment):** rev 4.1's regression gate **passed** on exp_044's sealed test, with request 0.912, next step 0.933, false spam 2, spam caught 0.939 and budgets in words 1.00, all identical to rev 4. Rev 4.1 (`kev_knife.py` sha256 `2acc7a1d…`) is frozen in `leads44.VERSIONS` and can be deployed.

**Published (2026-10-20):** local-first-ai `tasks/chronos/exp_046_translate_first/`: both records, the kit, the run script and log, and every prediction and translation (test and dev). Re-scoring the published files reproduces `score.json` exactly. No blog post (Andrei). Rev 4.1 is deployed in the bot (Andrei, 2026-10-20).
