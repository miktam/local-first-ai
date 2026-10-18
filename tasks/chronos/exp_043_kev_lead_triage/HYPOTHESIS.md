# exp_043 pre-registration: Kev-4B lead triage (the first Kev-knife blade)

*Written 2026-10-18, before any Kev call on these leads. **Signed by Andrei: 2026-10-18 12:01 CEST** ("sign"); text as committed in 13e3fd1. Andrei asked to measure the lead-triage prototype on synthetic leads ("go 1"). Code: casasol `scripts/kev_knife.py` (triage), `scripts/leads43.py` (pipeline). Labels: [LABELS.md](LABELS.md).*

## The question

Can one local decision model, Kev-4B with no generative model, triage inbound agency leads well enough to be used: the intent, whether the sender is qualified, and what to do next?

## Data (synthetic only; no real client data)

- **Writing:** 440 leads, written by 8 Fable writers. Each writer had one language, 7 languages in all (EN ×2 batches, one deliberately messy, plus ES, PL, DE, SV, FR, RU), and the same intent quotas: buy 14, rent_long 7, rent_holiday 5, sell 7, valuation 4, invest 5, info 6, spam 7.
  - About a quarter of the leads carry fake contact details.
  - At least 1 in 5 per batch are hard cases.
- **Agreement:** a blind Fable checker labels every lead from the text alone. A lead is dropped if writer and checker disagree on intent, or the checker flags doubt. Any other field where they disagree is excluded for that lead.
- **Split:** seeded, stratified by language: about 120 dev leads, the rest test.
- **Andrei reads 30 test leads** before the test set is sealed. Leads he rejects are dropped, and a new sample is drawn.

## As built (2026-10-18, before any test call)

- **Agreement:** 440 written; 85 dropped on checker doubt; **355 kept** (`agreed.jsonl`, `71014cc8`).
  - Writer–checker agreement over all 440: intent 0.986, action 0.989.
- **Split:** dev 120 (`614141b1`); **test 235** (`8495fbd6`).
  - Test by language: EN 60, DE 31, ES 30, RU 30, SV 30, FR 28, PL 26.
  - Test by intent: buy 57, info 37, spam 32, rent_long 29, invest 24, sell 22, rent_holiday 22, valuation 12.
- **Dev tuning, recorded** (predictions in `results/dev_runs/`):

  | | rev 1 | rev 2 | rev 3 (frozen) |
  |---|---|---|---|
  | intent | 0.858 | 0.908 | **0.933** |
  | spam recall / false spam | 0.857 / 0.01 | 0.857 / 0.01 | **1.00 / 0.01** |
  | action | 0.808 | 0.933 | **0.958** |
  | urgency ±1 | 0.765 | 0.924 | **0.933** |
  | qualified | 0.825 | 0.825 | 0.825 |

  - **rev 2:** sharper intent descriptions. Kev answers yes/no facts (hot, specific listing, call request, asks worth), and code derives urgency and action from LABELS.md.
  - **rev 3:** adds a "business pitching the agency" fact; ≥ 0.6 makes the lead spam. The cut was set on dev: 0.5 flagged 5 clients as spam.
- **Andrei's read (2026-10-18 12:00 CEST):** the 30-lead test sample (`data/READ30.md`) was accepted with no lead rejected ("yes, sample looks good"). The test set is sealed at `8495fbd6`.
- **Frozen:** `scripts/kev_knife.py` rev 3, casasol `d581267`. The test runner refuses any other version (`626e48e`), and refuses a second pass.

## Procedure

1. **Dev, tuning allowed.** `kev_knife.py` question wording, the code action rules, and the yes/no cut-off (default 0.5) may change, using dev only. Then the code is committed and frozen.
2. **Test, one pass:** `leads43.py run test`. The runner refuses a second.
3. **Score:** `leads43.py score test`.

## Hypotheses (test set)

| Field | Measure | Bar |
|---|---|---|
| **H1 intent** (8 classes) | accuracy | ≥ 0.85 |
| **H2 spam** | recall / false-spam rate on non-spam | ≥ 0.90 / ≤ 0.02 (never discard a real client) |
| **H3 action** (after the code rules) | accuracy | ≥ 0.75 |
| **H4 qualified** | accuracy | ≥ 0.80 |
| H5 property type, timeline | accuracy | ≥ 0.80, ≥ 0.75 |
| H6 financing, foreign buyer | accuracy on leads where the label is not null | ≥ 0.80 each |
| H7 urgency (0–3) | share within ±1 | ≥ 0.85 |

Reported alongside:
- intent accuracy by language (Kev's card says English only);
- the top intent confusions;
- the majority-class baseline for each field;
- milliseconds per lead.

## Verdict map

1. H1, H2 and H3 pass → **Blade usable for triage** (as a suggestion to an agent; no automatic action other than discarding spam). H4–H7 decide which extra fields can be shown.
2. H1 passes but H2 or H3 fails → **Classifies, but can't act yet.** Show the intent only; no discard, no routing.
3. H1 fails → **Not usable as is.**

## Known limits

- **The leads are synthetic.** Real leads are messier and carry history. A pass here licenses a trial on real leads, not a product claim.
- **Labels are two Fable readers' agreement.** Hard, ambiguous cases are filtered out by construction, so the test is easier than reality.
- **One model family, one run.**

## SO WHAT (to be filled in with the result)

---

**Signature:** Andrei, 2026-10-18 12:01 CEST — "sign". Frozen at 13e3fd1; any change from here is a dated amendment below.
