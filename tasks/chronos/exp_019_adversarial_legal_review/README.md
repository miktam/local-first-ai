# exp_019 — Adversarial Legal Review Pipeline

**Status:** Complete  
**Date:** 2026-06-23  
**Type:** Methodology experiment + applied output

---

## Pre-registered hypothesis

An adversarial 3-panelist pipeline (Regulator / Opposing Counsel / Devil's Advocate) produces a more accurate legal brief than single-pass drafting. Metric: claims surviving the panel unchanged vs. total claims drafted.

**Result: 0/7 claims survived unchanged.** Core argument correct; every claim required revision. Two critical issues caught that would have caused a Gibraltar lawyer to distrust the document on first read.

---

## What this experiment is

First application of the Legal Agent adversarial pipeline (`wiki/legal-agent.md`). A data sovereignty legal brief — cloud AI versus local inference under UK/Gibraltar, Singapore, and EU law — was drafted (v1), attacked by three panelists with conflicting mandates, and revised to v_final.

---

## Artifact index (public)

| File | What it is |
|------|-----------|
| `data_sovereignty_brief_sanitised.md` | Final brief — sanitised, all client context removed. Covers UK/Gibraltar, Singapore, EU. 12 open questions. |
| `iterations/v2_citation_check.md` | GPT-4o independent citation check. 3 critical corrections: MAS wrong instrument, BDAA citation, ICO overstated. |
| `iterations/v3_overclaim_review.md` | Gemini as UK barrister. Caught Gibraltar-EEA adequacy gap (most critical) and absolutist phrasing throughout. |

## Iteration chain (held privately)

The full internal iteration chain — v1 (neutral drafter), adversarial panel (Regulator / Opposing Counsel / Devil's Advocate), dissent register, and v_final — contains client-specific context and is held in the operator's private workspace. The sanitised brief above reflects all revisions from the complete chain.

**Iteration summary:**
- v1: claude-sonnet-4-6, neutral ILAC drafter. Score 4.2/5. All 7 claims SOLID in body text — 0 survived the full review process.
- Adversarial panel (Claude, 3 stances): caught BDAA omission, ingestion transfer gap, CLAIM-007 addressability error.
- v2 (GPT-4o): caught wrong MAS instrument (TRM vs. Outsourcing Guidelines), BDAA § 2523 citation error, ICO position overstated.
- v3 (Gemini): caught Gibraltar-EEA adequacy gap, "structural guarantee" absolutism, FISA 702 omission, MAS intra-group scenario.
- v_final: 12 open questions for legal counsel (up from 5 in v1).

---

## Two critical issues the panel caught

**Issue A — UK-US BDAA (Oct 2022) missing.** The Bilateral Data Access Agreement between the UK and US constrains how US law enforcement can access UK/Gibraltar persons' data from US companies. v1 presented CLOUD Act risk as if this agreement doesn't exist. Any UK-trained lawyer would ask about it immediately. Added to v_final.

**Issue B — Ingestion transfer gap.** v1 claimed "no international transfer during inference." This is true only if compute and data are co-located. The technical brief stated the compute cluster would be in Singapore; the documents originate in Gibraltar. The transit Gibraltar→Singapore is itself a restricted international transfer. Panelist 3 (Devil's Advocate) caught this by attacking the fundamental conclusion. Added to v_final as a named precondition.

---

## Blog posts

- [2026-06-24-adversarial-legal-panel.md] — methodology post (Nestor voice)
- [2026-06-24-data-sovereignty-brief.md] — brief post (miktam preface + sanitised v_final)
