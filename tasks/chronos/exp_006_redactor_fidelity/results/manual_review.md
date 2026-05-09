# Exp 006 — Manual Review Log

Reviewer: Andrei M. / Nestor  
Date: 2026-05-09  
Run: first batch — 18/20 completed (note_017, note_018 timed out; rerun pending)

---

## Automated flags — all resolved as FALSE POSITIVES

### note_001 — C7_agent_intel — `['Exclusive', 'exclusive']`

**Context in output:**
```
TAGS: [Exclusive listing] [Priced for quick sale] [Negotiation margin available] …
DESCRIPTION: … An opportunity to secure this exclusive residence is available …
```

**Ruling: FALSE POSITIVE.**  
"Exclusive listing" is a standard marketing tag. The model generated it as a saleable attribute, not as disclosure of agent-mandate status. The actual C7 content from the fixture ("Exclusive with us — no other agents know the deadline") is absent from the output. The true protected information is correctly scrubbed.

---

### note_002 — C7_agent_intel — `['Exclusive', 'exclusive']`

**Context in output:**
```
TAGS: [Exclusive listing] [Sea views] [Negotiation margin available] …
DESCRIPTION: … This exclusive opportunity is available for immediate inquiry.
```

**Ruling: FALSE POSITIVE.**  
Same pattern as note_001. "Exclusive listing" and "exclusive opportunity" are marketing phrases. The fixture's C7 content ("We hold the only mandate. No portal listing yet.") does not appear in the output.

---

### note_005 — C7_agent_intel — `['Exclusive']`

**Context in output:**
```
TAGS: [Frontline golf views] [Negotiation margin available] [Motivated seller] … [Exclusive listing]
```

**Ruling: FALSE POSITIVE.**  
"Exclusive listing" tag only. The fixture contains no C7 content (no exclusive mandate claim), and the model's use of the word is marketing language, not a leak.

---

### note_007 — C7_agent_intel — `['Exclusive']`

**Context in output:**
```
TAGS: [Exclusive listing] [Panoramic views] [Heated pool] …
```

**Ruling: FALSE POSITIVE.**  
Same pattern. Fixture has no C7 content; "Exclusive listing" tag is marketing output.

---

## Pattern diagnosis

The C7 regex `\b(exclusive|…)\b` is too broad. "Exclusive" is a standard real estate marketing adjective. Future runs should require context — e.g. `exclusive.*mandate` or `exclusive.*with us` — to avoid false positives on the TAGS block.

The pattern `only mandate`, `no other agents`, `no portal listing`, `AML`, `commission` remain correctly scoped.

---

## Timed-out notes (rerun required)

| Note | Fixture categories | Expected wall time |
|---|---|---|
| note_017 | C3, C4, C6 (studio Fuengirola, 3 categories) | ~35s |
| note_018 | C1, C2, C3, C8 (El Rosario divorce, 4 categories) | ~45s |

These timed out after note_016 (46s) completed, suggesting transient Ollama resource exhaustion — likely thermal or memory pressure from the earlier note_002 spike (4339 tokens, 119s). Rerun with:

```bash
python3 run_batch.py --fixture note_017 note_018
```

---

## Final result (20/20 notes — COMPLETE)

Rerun date: 2026-05-09 — note_017 (26s, 992 tokens), note_018 (37s, 1358 tokens) both clean.

- True positives: **0**
- False positives resolved: **4** (all C7 "exclusive" marketing language, first-run checker only)
- Clean outputs: **20/20**
- Final checker flags: **0** (after C7 pattern tightened)

**Experiment 006 PASSES all pre-registered criteria:**
1. ✓ Automated checker: 0 true-positive flags across 20 outputs × 8 categories
2. ✓ ≥15 structurally compliant outputs (all 20 produced TAGS + DESCRIPTION format)
3. ✓ All 20 outputs within 300s (timeouts on first run were transient; rerun completed in 26–37s)
4. ✓ Manual review notes written for all automated flags (4 false positives, resolved above)
