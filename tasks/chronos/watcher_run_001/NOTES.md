# Watcher Run 001 — Annotations

**Date:** 2026-06-06  
**Git HEAD:** 3a6d70e  
**Model:** gemma4:26b  
**Total duration:** ~270s (6 LLM calls)  
**Token accounting:** 2950+3824+2104+1025+1031+1024+2161 in / 405+1070+741+1557+590+1868+2048 out

---

## Confirmed findings

| Finding | Severity | Verdict |
|---|---|---|
| Model swap qwen3.5:35b → gemma4:26b has no ADR | High | **CONFIRMED** — real gap, no Architectural Decision Record in any commit |
| Witnessing pipeline underdocumented in BRIEF/MOAT | Medium | **CONFIRMED** — shipped today, docs not updated yet |
| DPA and compliance work underdocumented in commitments | Medium | **CONFIRMED** — not in BRIEF/MOAT, only in compliance/ |
| Multilingual casasol.ai (ES/PL/RU) not in high-level commitments | Medium | **CONFIRMED** — shipped in prior session, never added to BRIEF |
| 58-listing corpus claim not evidenced in git artefacts | Medium | **CONFIRMED** — corpus is real but no commit proves the count |

## False positives

| Finding | Reason |
|---|---|
| "Absence of RoPA" | `compliance/01-ropa.md` exists — watcher didn't ingest `compliance/` directory |
| "OLÉ registration underevidenced" | Event registration is not a code artefact; not appropriate to flag |
| "Adversarial Watcher underdocumented" | Created today in the same session — temporal artefact, not a real gap |

## Fix applied

Added `compliance/README.md` to `INTENT_SOURCES` in `watcher.py` so RoPA, DPA, LIA are visible on next run.

## Next run

After updating BRIEF/MOAT with witnessing pipeline and DPA landing, re-run watcher to verify gap count drops. Expect underdocumented count to fall from 7 → 3-4.
