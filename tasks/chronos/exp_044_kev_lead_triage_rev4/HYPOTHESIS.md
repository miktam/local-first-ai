# exp_044 pre-registration: lead triage rev 4 on a fresh synthetic set

*Written 2026-10-19. **Signed by Andrei 2026-10-19 17:36 CEST.** Andrei chose option A ("go with A"): fix what exp_043 exposed, and compare both versions on the same new test set. Nothing runs on the test set before Andrei signs. Builds on exp_043 (`tasks/exp043/`, public record `local-first-ai/tasks/chronos/exp_043_kev_lead_triage/`). Code: casasol `scripts/kev_knife.py`; reference: casasol `TRIAGE.md`.*

## The question

Can four changes fix the weaknesses exp_043 found, without making anything else worse? The changes are a safer spam rule, a direct "buy or rent?" question, budgets written as words, and an honest confidence label. Both versions are scored on the same fresh enquiries.

## The two versions

- **rev 3.1, the baseline:** today's `kev_knife.py`, the same decisions exp_043 measured. Since then the spam action was renamed to `review` (same decision), the budget reader learned more units, and the probabilities are passed through. **Frozen 2026-10-19 as casasol `scripts/kev_knife_rev31.py`**, a byte-for-byte copy of `kev_knife.py` at `c1efd0b`, sha256 `bc2ae14723530d6893206bf80dcb16844e79034eff77b93fb3de776a7bf9b16c`. The test runner refuses any other bytes.
- **rev 4:** rev 3.1 plus these changes, each tuned on the dev set only:
  1. **Safer spam rule.** The business-pitch question may override Kev's own reading only when both agree:
     - the business-pitch score is at or above a cut, **and** Kev's own spam probability is at or above a second cut;
     - if Kev already picked spam, nothing changes.
     - The two cuts are set on dev, with a constraint: no genuine dev lead may be flagged, then spam caught is maximised.
  2. **"Buy or rent?"** A new yes/no question to Kev: *"Does the sender want to buy a property to own, even a holiday home, rather than rent one?"*
     - If Kev picks holiday rent and this answer is at or above a cut, the request becomes buy.
     - The reverse is allowed too: buy with this answer at or below 1 − cut becomes holiday rent.
     - The cut is set on dev.
  3. **Budgets written as words, in code.** Examples: "un millón", "medio millón", "half a million", "eine Million", "pół miliona", "два миллиона", "900 mil", "1,2 millones". A fixed word list per language, with no model involved.
  4. **No change to urgency or the next-step rules**, unless Andrei decides otherwise below.

## Data (synthetic only; no client data)

**Writers.** Eight Fable writers, as in exp_043:
- two English batches (one deliberately messy), and one each in Spanish, Polish, German, Swedish, French and Russian;
- written fresh, with no access to the exp_043 messages.

Each batch has two strata:

| Stratum | Per batch | What it is |
|---|---|---|
| **standard** | 55 | exp_043's quotas and rules exactly: buy 14, rent_long 7, rent_holiday 5, sell 7, valuation 4, invest 5, info 6, spam 7. About a quarter carry fake contacts, and at least 1 in 5 are hard cases. |
| **hard: business look-alikes** | 6 | 3 genuine clients who sound like businesses (a relocation firm renting for staff, a developer looking for land, a buyer of several units, a lawyer or family office acting for a buyer) and 3 real business pitches |
| **hard: holiday home vs holiday rent** | 6 | 3 buyers of a holiday or second home who use holiday wording ("over for the week", "flying in", "for our summers") and 3 holiday renters who use buying wording |
| **hard: budgets in words** | 6 | genuine clients whose budget is written in words in their language; buy, sell, invest or rent |

That makes 73 per batch and 584 in total.

**Labels v2.** The same as exp_043's `LABELS.md`, plus `budget_eur`:
- the largest budget stated, in euros: the total price for buy, sell and invest, the monthly rent for rentals;
- `null` if none is stated.

**Agreement, as in exp_043.**
- A blind Fable checker labels from the text alone.
- A lead is dropped if writer and checker disagree on the request, or the checker flags doubt.
- A field they disagree on is excluded for that lead.
- `budget_eur` counts as agreed when the two are within 1 %.

**Split.** Seeded, stratified by language and stratum: about 150 leads for dev, the rest for test. Andrei reads 30 test leads before the set is sealed; any he rejects are dropped and a new sample is drawn.

## As built (2026-10-19, before any Kev call on these leads)

- **Writers:** 8 batches × 73 = **584 leads**, no schema errors, exact quotas: standard 440 (buy 112, rent_long 56, sell 56, spam 56, info 48, invest 40, rent_holiday 40, valuation 32); hard_b2b 48, hard_holiday 48, hard_budget 48. Raw files: [data/raw/](data/raw/) (public copy).
- **Label clarifications** (dated note in `LABELS.md`, written after the writers and before any checker label): a holiday rental's `budget_eur` is the amount stated for the stay (all eight writers did this); a January-2027-or-later date is `later` (the existing rule taken literally; the Swedish batch called December–February `months`, so those leads lose their timeline label where the checker disagrees).
- **A rev 4 fix found in the writer data, before any split or model run:** the digit reader's `m` unit read "300 m de la plage" (metres) as €300M, a bug rev 3.1 also has. In rev 4, a bare `m` counts as million only below 50. rev 3.1 keeps the bug, as frozen.
- **Checker:** 8 blind batches of 73 (ids and text only), all done 2026-10-19.
- **Agreement:** 584 written; **98 dropped**, all on checker doubt (no request-type disagreements remained); **486 kept** (`agreed.jsonl`, `57aa2dd9`). Kept by stratum: standard 351, hard_b2b 44, hard_budget 45, hard_holiday 46. Field agreement among kept: request 1.00, next step 0.998, budget 0.994, property type 0.981, foreign 0.957, timeline 0.938, urgency 0.92.
- **Split:** dev **155** (`d237a277`); test **331** (`bf54b4d0`).
  - Test by stratum: standard 241, hard_holiday 31, hard_b2b 30, hard_budget 29.
  - Test by language: EN 81, FR 44, ES 43, PL 42, RU 41, SV 41, DE 39.
  - Test standard by intent: buy 60, spam 34, info 29, sell 28, rent_holiday 26, rent_long 25, invest 22, valuation 17.
- **Andrei's read (2026-10-19):** `data/READ30.md` (18 standard, 5 business look-alikes, 4 holiday, 3 budgets in words). His six notes (X7-31, X4-27, X7-38, X7-48, X7-73, X4-04) were all fields shown as `None`/`disagree`. Every one is a writer–checker disagreement, which the agreement rule excludes from scoring for that lead. In all six, request type and next step agree. None of the excluded fields (from abroad, timeline, urgency, property type) enters H1–H5. **No lead rejected** (confirmed with the signature).
- **Dev runs (155 leads, both versions, 2026-10-19):**

  | Dev | rev 3.1 | rev 4 untuned | rev 4 tuned |
  |---|---|---|---|
  | request, standard | 0.882 | 0.891 | — |
  | next step, standard | 0.891 | 0.891 | — |
  | request, all dev | — | — | 0.923 |
  | next step, all dev | — | — | 0.916 |
  | holiday stratum, request (15) | 0.933 | 0.933 | 0.933 |
  | budgets in words, within 1 % | 0.00 | 0.938 | 0.938 |
  | spam caught / genuine flagged | 23 of 24 / 1 of 131 | the same | the same |
  | Sure / Fairly sure / Unsure accuracy | 1.00 / 0.95 / 0.72 | 1.00 / 0.95 / 0.74 | — |
- **Tuning (offline, on Kev's recorded dev answers, 504 candidates):**
  - **No candidate reached zero dev false spam.** The one flagged lead (X4-69, a genuine buyer) is Kev's own spam reading at probability 0.28, not a rule, so no cut can fix it. No new rule was added for one dev lead.
  - **The spam cuts tied at the best value;** the most cautious of the ties was kept: `B2B_CUT` 0.60 and `SPAM_AGREE` 0.15.
  - **`OWN_CUT` is 0.65** (request on all dev: 0.903 with the rule off, 0.910 at 0.70, 0.923 at 0.65).
  - **`OWN_REVERSE` stays off,** because it changed nothing.
  - **What the dev numbers suggest:** rev 3.1 already reads the dev holiday stratum well (14 of 15), so H3 may be hard to pass on test.
- **rev 4 frozen 2026-10-19:** casasol `scripts/kev_knife.py` at `abcfacf`, sha256 `b19402352632b676861477284fbbde06fdd2e4a65b8139509c2b199f11c9a815`. The test runner refuses any other bytes.
- **The bot demo (`/triage`) is pinned to rev 3.1** until this experiment decides (casasol commit after `abcfacf`).

## Procedure

1. Write, check and agree; split; Andrei reads 30. No Kev call has touched the test set at this point.
2. **Dev, tuning allowed (rev 4 only).** The three cuts, the wording of the new question, and the word lists. Then commit and freeze rev 4 by sha256.
3. **Andrei signs.** This text plus the as-built section are frozen.
4. **Test, one pass per version:** rev 3.1 and rev 4 on the same sealed test set. The runner refuses unfrozen code and a second pass.
5. Score both. Then write `RESULT.md` with a SO WHAT section.

## Hypotheses (test set; paired: same leads, two versions)

| | Measure | Bar |
|---|---|---|
| **H1 no regression** | standard stratum: request type and next step, rev 4 − rev 3.1 | ≥ −0.02 on each |
| **H2 safer spam** | all genuine test leads: rev 4's false-spam rate; spam caught on all spam test leads | false spam ≤ 0.01 **and** ≤ rev 3.1's; spam caught ≥ 0.85 |
| **H3 holiday-home fix** | holiday stratum: request accuracy, rev 4 − rev 3.1 | ≥ +0.10, one-sided McNemar p < 0.05 |
| **H4 budgets in words** | budget stratum: rev 4's budget within 1 % of the label | ≥ 0.90 (rev 3.1 reported) |
| **H5 confidence means something** | rev 4, all test leads: request accuracy by band, Sure (≥ 0.75), Fairly sure (0.5–0.75), Unsure (< 0.5) | Sure ≥ 0.90, and Sure > Fairly sure > Unsure |

**Also reported:**
- **Replication:** rev 3.1's standard-stratum scores against exp_043's (0.894 request, 0.898 next step). Does the result hold on fresh messages?
- rev 4's expected calibration error for request confidence (descriptive; ≤ 0.10 would be good);
- per-language request accuracy, the top confusions, and the time per lead;
- the business-look-alike stratum on its own.

## Verdict map

1. **H1 and H2 pass → adopt rev 4.**
   - If H3 also passes: "rev 4 fixes the holiday-home confusion".
   - If H4 passes: budgets in words are read.
   - If H5 passes: the demo keeps Sure / Fairly sure / Unsure as they are; if it fails, the labels change.
2. **H1 fails → keep rev 3.1.** Any change that cost accuracy is dropped, and the others are re-tested.
3. **H2 fails but H1 passes → adopt rev 4 without the new spam rule,** and keep rev 3.1's.

## Decided by Andrei, 2026-10-19 (before the writers run)

- **Urgency: (a) keep the policy.** Urgency and the next-step rules are unchanged in rev 4.
- **Plan: go, build the data** with this text ("Go, build the data"). He signs after dev tuning, as with exp_043.

### The decision as it was put

**Urgency for a call request.** The written policy gives urgency 0 to "a timeline of later or unspecified with no hot signal". So a seller who asks for a call tomorrow but gives no selling date gets 0. That's the policy, not a model error. Two ways to go:
- **(a) Keep the policy** (default). Urgency is not part of rev 4.
- **(b) Change it in labels v2:** a phone number or call request from a genuine client means at least urgency 1. Rev 4 implements it.
  - Scored against the v2 labels, rev 3.1 would lose on those leads by definition.
  - So with (b), urgency is reported as a policy change, never as an improvement.

## Known limits

- **Synthetic again,** from the same family of writers as exp_043. The hard strata are written to target the fixes, so H3 and H4 test the fixes on the cases they were made for. H1 and the replication check are the guard against overfitting to them.
- **One model family, one pass per version.**
- **Real leads remain the real test** (option B, the next experiment).

## SO WHAT (to be filled in with the result)

---

**Signature:** Andrei, 2026-10-19 17:36 CEST, "Sign". This also confirms his read: no lead rejected. The text above, including the as-built section, is frozen from here; any change is a dated amendment below.
