---
title: "The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks"
date: 2026-05-09
author: "Nestor"
tags: ["local-llm", "gdpr", "redaction", "casasol", "chronos"]
description: "A pre-registered fidelity sweep: 20 synthetic toxic real estate notes, 8 GDPR categories, one fixed prompt. Result: 0 true leaks."
---

The CasaSol demo shows a local Gemma 4 26B model redacting a toxic real estate agent note in real time. The implicit claim behind that demo is that the output is GDPR-clean.

An anecdote is not an experiment. One demo run is a marketing moment. To turn that claim into engineering truth, we needed a controlled test before the booth opens. Enter Experiment 006: The Redactor Fidelity Test.

### The Redaction Contract

Redaction is a contract. On one side, the input contains "toxic" data—sensitive, private, or legally protected information. On the other side, the output must contain only the allowed content, stripped of specific identifiers.

For this experiment, we pre-registered eight specific data categories. If any of these appeared in the output, the contract was breached.

| ID | Category | Description |
| :---  | :--- | :--- |
| **C1** | Natural person identity | Nationality, ethnicity, owner name |
| **C2** | Legal proceedings | Divorce, tax investigation, court order, receivership |
| **C3** | Financial situation | Floor price, rejected offers, debt, must-sell urgency |
| **C4** | Undisclosed property defects | Structural issues the seller has not disclosed |
| **C5** | Health or family data | Illness, bereavement, care situation |
| **C6** | Third-party private information | Neighbour disputes, tenant situation |
| **C7** | Agent-internal commercial intelligence | Exclusive mandate, competitor ignorance |
| **C8** | Temporal pressure | Deadlines from legal/tax/health events |

### The Setup

We didn't wing it. The hypothesis, the eight categories, and the pass criteria were committed to `scientific_log.md` before a single note was generated. This is the core of the *Chronos* methodology: pre-registration to prevent hindsight bias.

The execution was stripped of all variables except the data itself:
- **Dataset:** 20 synthetic "toxic" notes. Each note was engineered to contain between 2 and 7 of the 8 categories.
- **Model:** `gemma4:26b` via Ollama.
- **Parameters:** Temperature set to 0.1 to minimize creative drift.
- **Prompt:** A fixed system prompt designed to identify and redact.
- **Validation:** An automated checker applied regex patterns for all 8 categories to every output. Any flag triggered a manual review.

### The Results

The numbers are clear.

| Metric | Result |
| :--- | :--- |
| Notes processed | 20/20 |
| **True-positive leaks** | **0** |
| Automated flags | 4 |
| False positives | 4 |
| Correct TAGS + DESCRIPTION format | 20/20 |
| Outputs within 300s | 20/20 |

The zero true-positive leaks is the headline. The contract held.

However, the automated flags tell a story of refinement. All four flags were triggered by category **C7** (Agent-internal intelligence). The model was doing the right thing — it produced marketing-appropriate language like "Exclusive listing" and "exclusive opportunity" while correctly suppressing the actual sensitive content (mandate instructions, competitor ignorance). The checker was too broad: bare `\bexclusive\b` fires on legitimate marketing adjectives.

We tightened the pattern after the first run: replacing `\bexclusive\b` with `\bexclusive (mandate|instruction|with us)\b`. The final check across all 20 outputs returned 0 flags.

One interesting observation regarding compute: `note_002`, which contained 7 toxic categories, produced 4,339 response tokens, compared to the typical 1,000–1,600. The model appears to perform significant internal reasoning/chain-of-thought before producing the final, clean `TAGS + DESCRIPTION` output. While this is a cost and latency observation, it does not constitute a failure of the redaction task itself.

### The Verdict

There is a fundamental distinction between the architecture and the agent.

In Experiment 003, we proved **data sovereignty** as an architectural invariant. We showed that the anonymization boundary enforced by the import graph cannot leak by design. That is about the **vault**.

Experiment 006 proves **fidelity**. It proves that the model, when given a fixed prompt and a clear contract, faithfully executes redaction across varied, high-entropy inputs. This is about the **locksmith**.

The vault keeps the data safe; the locksmith ensures that when you need to share a sanitized version, the privacy remains intact.

**Evidence Reference:**
- [Experiment 006 pre-registration](https://github.com/miktam/local-first-ai/blob/main/tasks/chronos/scientific_log.md) — hypothesis, categories, and pass criteria committed before execution
- [check_report.json](https://github.com/miktam/local-first-ai/blob/main/tasks/chronos/exp_006_redactor_fidelity/results/check_report.json) — per-note, per-category flag record
- [check_summary.txt](https://github.com/miktam/local-first-ai/blob/main/tasks/chronos/exp_006_redactor_fidelity/results/check_summary.txt) — human-readable run summary
- [manual_review.md](https://github.com/miktam/local-first-ai/blob/main/tasks/chronos/exp_006_redactor_fidelity/results/manual_review.md) — reviewer rulings on all automated flags
