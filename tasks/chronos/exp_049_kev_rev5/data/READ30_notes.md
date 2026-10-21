# Andrei's read of READ30 (2026-10-21)

Two notes. Both are about the action rules, not about how a lead was labelled under them:

- **Y8-12** (ru, standard): `send_listings` → he suggested `call_now`. The sender asks to be called back and gives a number. But the timeline is `months` ("до конца года"), and rule 4 gives `call_now` only with timeline `now`. Rule 5 → `send_listings`.
- **Y1-34** (en, hard_valuation): `book_valuation` → he suggested `call_now`. The sender gives a number. But rule 3 (sell + asks what it's worth → `book_valuation`) comes before rule 4, and the first rule that fits decides.

**Decision (Andrei, "ok, keep it"):**
- the rules stay as written for exp_049;
- no lead is rejected, and the sample is not redrawn;
- the test set stays sealed at `de3e1710`.

**Seed:** "an explicit call request with a phone number → call now, whatever the timeline and ahead of a valuation". This is TRIAGE.md weakness 7. It is a rule change to make deliberately and measure on its own, after exp_049.
