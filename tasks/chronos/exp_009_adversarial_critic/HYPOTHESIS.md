# Experiment 009 — Adversarial Project Critic

*Pre-registered: 2026-06-06 · Status: pre-registered, execution pending*

**Pre-registration:** [`tasks/chronos/scientific_log.md`](../scientific_log.md) → Experiment 009

---

## What this experiment tests

Project reviews are slow because human reviewers must first reconstruct context before they can critique. This experiment tests whether an adversarial LLM critic — given recent git commits, a BUILD_LOG, a BRIEF, and a TODO — can surface high-severity issues faster than a human and whether a local 26B model can match a frontier model at this task.

Two critics are compared:
- **Option A (Claude):** frontier model, full reasoning, API cost per run
- **Option B (gemma4:26b):** local model, zero marginal cost, no network dependency

---

## Hypothesis

**Primary:** A local gemma4:26b adversarial critic, given the same project context (recent commits, BUILD_LOG, BRIEF, TODO), identifies ≥60% of the high-severity issues that a Claude Sonnet adversarial critic identifies.

**Secondary:** The local critic has a false-positive rate (issues flagged by gemma4 but not Claude) of ≤30%.

**Null hypothesis:** The local critic identifies <40% of Claude-flagged high-severity issues — not useful as a standalone QA gate.

---

## Adversarial personas (fixed across both critics)

| ID | Persona | Focus |
|----|---------|-------|
| P1 | Sceptical investor | Is the moat real or just marketing copy? Are the numbers auditable? What would a competing founder attack? |
| P2 | DPO / AEPD auditor | Will the compliance architecture survive a real audit? Where are the gaps in the RoPA / LIA / DPA? |
| P3 | Competing engineer | How would I replicate this in a weekend? What is the actual technical barrier vs. the claimed one? |

---

## Context inputs (fixed across both critics)

1. `git log --oneline -20` — recent commit history
2. Last 100 lines of `BUILD_LOG.md` — what was done and why
3. `BRIEF.md` §1–§2 or equivalent strategic document
4. `strategy/TODO.md` or `tasks/CURRENT_TASK.md` — open items

---

## Output schema (JSON, fixed across both critics)

```json
{
  "ts": "ISO-8601",
  "critic": "claude-sonnet-4-6 | gemma4:26b",
  "project": "casasol | ...",
  "personas": {
    "investor": {
      "holds_up": ["..."],
      "weak": ["..."],
      "missing": ["..."]
    },
    "dpo": {
      "holds_up": ["..."],
      "weak": ["..."],
      "missing": ["..."]
    },
    "engineer": {
      "holds_up": ["..."],
      "weak": ["..."],
      "missing": ["..."]
    }
  },
  "top_actions": ["...", "...", "..."],
  "severity_counts": {"high": N, "medium": N, "low": N}
}
```

---

## Scoring

**Issue overlap rate:** for each high-severity issue in the Claude output, does an equivalent issue appear in the gemma4 output? Scored by human reviewer (Andrei) on a binary match per issue.

**False positive rate:** issues in gemma4 output not present in Claude output — also scored by human reviewer.

**Pass threshold:** ≥60% overlap + ≤30% false positives → gemma4 is viable as a primary gate, Claude as a periodic deep-review.

---

## Evidence artefacts

- `results/claude_YYYYMMDD_HHMMSS.json` — Option A output
- `results/gemma4_YYYYMMDD_HHMMSS.json` — Option B output
- `results/comparison_YYYYMMDD.json` — overlap scores, scored by human reviewer
- This HYPOTHESIS.md — pre-registered before any runs (git timestamp is the proof)

---

## localfirstai.eu post hook

If gemma4 passes the ≥60% threshold, the post title is:
*"Can a local 26B model replace a frontier critic? We ran the adversarial experiment."*

If gemma4 fails, the post title is:
*"We tried to replace Claude with a local critic. Here's exactly where it fell short."*

Either result is publishable. The experiment is the content.

---

*Experiment design: Andrei + Claude Sonnet 4.6 · 2026-06-06*
*Inspired by MS Build 2026 BRK250 ASSERT framework and the Bastion SDLC loop.*
