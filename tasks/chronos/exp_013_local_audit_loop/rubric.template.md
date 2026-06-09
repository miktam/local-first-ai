# rubric.template.md

Copy to `rubric.md` and fill in BEFORE running any model. Commit it before the first run.
Do not edit after seeing any output — that is the pre-registration discipline that makes
the score meaningful. Scoring stays manual.

## Task definition

- **Task name:**
- **Persona given to the model:**
- **Context bundle (exact files passed via --code / --policy):**
- **Output schema:** JSON gaps array, severity-ranked (matches the instrument)
- **Max score:**

## Scoring rule

- +1 per correct identification of a pre-registered item below
- −1 per confirmed false positive (a fabricated gap absent from the current codebase)
- Above-rubric findings: recorded separately, not scored, unless promoted in a later
  pre-registered version

## Pre-registered items

| id | item (the real gap) | where it lives | how a correct answer is recognized |
|----|---------------------|----------------|------------------------------------|
| 1 | | | |
| 2 | | | |
| 3 | | | |

> Keep item wording in the model's prompt OUT of this table's phrasing — the prompt must
> never quote these descriptions. See `intervention_ledger.md` on rubric-leakage.

## Reps

- Reps per model: 3 (consistency check — an item found in 1/3 reps is noted as unstable)

## Pre-registration metadata

- Committed (date / git sha):
- Models under test:
- num_ctx:
