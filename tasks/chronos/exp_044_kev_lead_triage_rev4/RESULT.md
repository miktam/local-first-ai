# exp_044 result: lead triage rev 4 vs rev 3.1

*Scored 2026-10-19 on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-19 17:36 CEST, `bbacce9`). One pass per version on the same sealed test set: 331 synthetic leads in seven languages (`bf54b4d0`). Code: rev 3.1 `kev_knife_rev31.py` sha256 `bc2ae147…`; rev 4 `kev_knife.py` sha256 `b1940235…` (casasol `abcfacf`). Files: [results/test/](results/test/) (predictions, `done` records with code and set hashes, per-version scores, `compare.json`), [results/dev_runs/](results/dev_runs/), [data/](data/) (all synthetic, all public; see [data/PUBLIC_COPY.md](data/PUBLIC_COPY.md)), [kit/](kit/).*

## Verdict: **adopt rev 4** (H1 and H2 pass; H4 and H5 pass; H3 fails)

| Hypothesis | rev 3.1 | rev 4 | Bar | |
|---|---|---|---|---|
| **H1** request, standard stratum (241) | 0.896 | **0.909** (+0.013) | ≥ −0.02 | pass |
| **H1** next step, standard stratum | 0.938 | **0.938** (0) | ≥ −0.02 | pass |
| **H2** genuine leads flagged as spam (282) | 5 (0.018) | **2 (0.007)** | ≤ 0.01 and ≤ rev 3.1 | pass |
| **H2** spam caught (49) | 47 (0.959) | 46 (0.939) | ≥ 0.85 | pass |
| **H3** holiday stratum, request (31) | 0.871 | 0.935 (+0.064; 2 vs 0 discordant, one-sided p 0.25) | ≥ +0.10, p < 0.05 | **fail** |
| **H4** budgets in words, within 1 % (29) | 0.00 | **1.00** | ≥ 0.90 | pass |
| **H5** request accuracy by band | — | Sure 1.00 (147) · Fairly sure 0.889 (90) · Unsure 0.798 (94) | Sure ≥ 0.90, falling | pass |

**Replication:** on fresh messages, rev 3.1 read the standard-stratum request right 0.896 of the time, against exp_043's 0.894. Next step was 0.938 against 0.898 (a different writer set, so the rise is descriptive).

## What changed (descriptive, after the verdict)

- **The agreement rule removed exactly exp_043's failure mode.**
  - **rev 3.1's five false spams:** four were genuine businesses acting for clients, flagged by the pitch rule at 0.61–0.70 (three relocation firms, one law firm). The fifth was a buyer Kev itself called spam.
  - **rev 4 keeps two of them:**
    - the law firm (X3-57): Kev read it as buy, but its spam probability passed the second cut, so the pitch rule still fired;
    - the buyer (X1-38): Kev's own spam reading, which no rule touches.
  - **The cost is one more missed spam:** 2 → 3. X4-15 is a pitch Kev read as "sell" (pitch score 0.80); Kev's own spam probability was under 0.15, so no override.
- **Holiday home vs holiday rent.**
  - **Across all 331 test leads,** buy→holiday-rent confusions fell from 11 to 4.
  - **Within the 31-lead holiday stratum,** 27 became 29 correct: the right direction, but not significant. Hence the H3 fail.
- **Overall** (all 331 leads): request 0.885 → 0.912, next step 0.927 → 0.933. Business look-alike stratum, request: 0.833 → 0.900.
- **Budgets:** words 0 of 29 → 29 of 29. All leads, including "no budget stated" (n 328): 0.655 → 0.768.
- **Standard stratum by language,** rev 3.1 → rev 4: Spanish 0.906 → 0.969, Russian 0.871 → 0.903; French 1.00, Swedish 0.933, German 0.897 and English 0.860 unchanged. **Polish 0.833 is again the weakest.**
- **Calibration:** the bands are ordered as registered, but the raw confidence is not calibrated (ECE 0.262). Sure / Fairly sure / Unsure are fair labels; the number behind them is not a probability to quote.
- **A new top confusion in both versions:** valuation read as sell, 9 times. These are owners asking for a price who also mention selling.
- **Unchanged,** since the questions and rules are the same (standard stratum): property type 0.891, qualified 0.822, timeline 0.707, urgency within one step 0.944.
- **Speed:** median 1.59 s → 1.67 s per lead (one more question).

## Known limits (as registered)

- **Synthetic again,** from the same family of writers as exp_043. The hard strata were written to target the fixes. H1 and the replication are the guard against that.
- **One pass per version, small hard strata** (29–31 leads each).
- **Real leads remain the test that matters** (option B).

## SO WHAT (2026-10-19)

**Adopted** (code path checked):
- **rev 4 is now the lead triage.**
  - The CasaSol Guide `/triage` demo switches from the pinned rev 3.1 to `kev_knife.py` (rev 4), in the casasol commit after this result.
  - All seven demos, including the two examples on casasol.ai, give identical chips, urgency and next step under rev 4 (checked 2026-10-19). The website stays true without a change.
- **Budgets written as words are read** (`scripts/budget_words.py`, seven languages).
- **Bug fix:** a bare "m" is no longer read as millions above 50 ("300 m de la plage").
- **`TRIAGE.md` is updated** to rev 4 and these results.

**Seeds:**
1. **Valuation vs sell** (9 errors): a closed question such as "do they only want a price, or do they want to sell?", or a rule.
2. **Kev's own low-confidence spam readings** (X1-38 on test; X4-69 on dev, at a spam probability near 0.28): require a minimum probability before Kev's own spam pick stands. Needs a fresh test.
3. **The agreement rule's price:** a pitch Kev reads as a seller (X4-15) gets through to the agent. A pitch-specific question might close it.
4. **Polish,** the weakest language in both experiments.
5. **Real leads (option B),** the test this synthetic series cannot replace.
