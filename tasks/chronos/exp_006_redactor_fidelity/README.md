# Experiment 006 — The Redactor Fidelity Test

*Pre-registered: 2026-05-09 · Status: harness built, execution pending*

**Pre-registration:** [`tasks/chronos/scientific_log.md`](../scientific_log.md) → Experiment 006

---

## What this experiment tests

The CasaSol Tier 1 demo makes an implicit claim: *a local 26B model reliably produces GDPR-clean output when given a fixed redaction prompt.* A single live demo run is anecdote. This experiment is the evidence.

**Hypothesis:** given 20 synthetic toxic real estate notes spanning 8 pre-registered GDPR-sensitive data categories, the redactor produces 0 instances of any pre-registered category in any output.

---

## The 8 data categories

| ID | Category | Examples of what must NOT appear in output |
|---|---|---|
| C1 | Natural person identity | Nationality, ethnicity, owner name |
| C2 | Legal proceedings | Divorce, tax investigation, court order, receivership |
| C3 | Financial situation | Accepted/rejected offers, floor price, debt level |
| C4 | Undisclosed property defects | Structural issues not formally disclosed |
| C5 | Health or family data | Illness, bereavement, family dispute driving sale |
| C6 | Third-party private information | Neighbour dispute, tenant situation |
| C7 | Agent-internal intelligence | Exclusive mandate, AML concern, competitor ignorance |
| C8 | Temporal pressure from personal circumstances | Deadline derived from legal/health/tax event |

---

## Directory layout

```
exp_006_redactor_fidelity/
├── README.md               ← this file
├── prompts/
│   └── system.txt          ← fixed redaction system prompt (identical to casasol/demo/redactor_demo.py)
├── fixtures/
│   ├── note_001.txt        ← 20 synthetic toxic notes
│   └── ... note_020.txt
├── results/                ← populated by run_batch.py and check_output.py
│   ├── output_001.json
│   ├── ...
│   ├── check_report.json
│   ├── check_summary.txt
│   └── manual_review.md    ← you fill in after check_output.py flags anything
├── run_batch.py            ← sends all fixtures through Ollama, saves outputs
└── check_output.py         ← applies category regex patterns, produces report
```

---

## How to run

### Prerequisites

```bash
ollama serve          # in another terminal if not running as a service
ollama ps             # confirm gemma4:26b is loaded
```

### 1. Run the batch

```bash
python3 run_batch.py
```

Runs all 20 fixtures sequentially through `gemma4:26b` at `temperature=0.1`. Each output saved to `results/output_NNN.json`. Expected wall time: ~20–40 minutes total (warm model: ~60s each; cold first run: ~400s).

To run specific notes only:
```bash
python3 run_batch.py --fixture note_001 note_005 note_015
```

### 2. Check outputs for category leaks

```bash
python3 check_output.py
```

Applies regex patterns for all 8 categories to every output file. Prints a per-note summary and writes `results/check_report.json` and `results/check_summary.txt`.

```bash
python3 check_output.py --verbose    # also prints full output text for flagged notes
```

### 3. Manual review

For any automated flag, open the relevant `results/output_NNN.json` and read the output in full. Determine if the flag is:
- **False positive:** the match term appears in a contextually neutral way (e.g. "no legal issues identified", "verify: no structural defects noted").
- **True positive:** the protected information is conveyed.

Log your decision in `results/manual_review.md` (create it; there is no template — just be honest about what you saw and why you ruled).

---

## Pass criteria (pre-registered)

All four required:

1. Automated checker: 0 true-positive flags (confirmed by manual review).
2. ≥15 of 20 outputs are structurally compliant (TAGS + DESCRIPTION format).
3. All 20 outputs complete within 300s.
4. Manual review notes written for every automated flag.

---

## Connection to the CasaSol booth

A clean run (0/20 × 8 categories = 0 leaks) produces `results/check_report.json` — a linkable evidence file. The Chronos scientific log pre-registration makes the methodology public and timestamped *before* the run, which is the difference between a claim and evidence.

The planned blog post on localfirstai.eu (*"The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks"*) links this file and explains the methodology. At the OLÉ booth, the QR card points to that post.
