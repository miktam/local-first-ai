# Exp 013 — Rubric (pre-registered)

**Pre-committed: 2026-06-09. DO NOT MODIFY after first model run.**

The rubric text must never appear in any prompt given to the model. It is for the human scorer only. Scoring stays manual.

---

## Run configuration

- **Task:** Compliance gap detection — policy documents vs. implementation code
- **Model under test:** gemma4:26b (Ollama, FA=0, q8_0)
- **num_ctx:** 32768
- **Context bundle (baseline run):**
  - `--code ~/REPOS/casasol/scripts/inference_log.py ~/REPOS/casasol/scripts/mcp_server.py`
  - `--policy ~/REPOS/casasol/compliance/01-ropa.md ~/REPOS/casasol/compliance/07-retention-schedule.md`
- **Reps:** 3
- **Baseline reference:** gemma4:26b 0/8 in Exp 012 (single-shot flat context, same source material)

---

## Scoring rule

- **+1** per pre-registered item correctly identified in ≥2/3 reps (stable find)
- **+0.5** per item found in 1/3 reps (unstable — noted separately, not counted toward headline score)
- **−1** per confirmed false positive: a gap claimed that is verifiably absent from the current codebase
- Above-rubric findings: recorded in the run notes, not scored

Headline score = integer sum of stable finds minus FP penalties. Max: 3 for the baseline bundle (items 1, 3, 4 are fully findable; 2 and 5 require expanded context — see notes).

---

## Pre-registered items

| id | what a correct answer contains | source files required | findable in baseline bundle? |
|----|-------------------------------|----------------------|:----------------------------:|
| 1 | The inference logger stores the full verbatim text of every request AND response to an append-only file. A privacy policy document in the same bundle explicitly states this category of data is not persisted beyond a defined boundary. The model must cite evidence from **both** the code and the policy document. | inference_log.py, 01-ropa.md | ✓ |
| 2 | A specific pipeline that processes photographs using a vision model includes automated face detection, meeting the threshold for high-risk processing under Art. 35 GDPR. No DPIA document exists for this activity in the compliance directory. | scripts/witness_ingest.py *(not in baseline bundle)*, compliance/ directory listing | ✗ **fair-evidence run only** — add witness_ingest.py to --code |
| 3 | The endpoint that receives and handles queries has no authentication, token validation, or session binding in its implementation. The model must name the specific file and note the absence of any credential check. | mcp_server.py | ✓ |
| 4 | The append-only log file written on every inference call is absent from the data retention schedule — it has no row, no defined retention period, and no deletion trigger in the schedule document. The model must name the specific log file and its absence from the schedule. | inference_log.py, 07-retention-schedule.md | ✓ |
| 5 | The inference logger records the model identifier as a plain string label, not an immutable digest or hash. This means the log cannot prove which exact model weights processed a given request — a mutable tag can be silently overwritten by a model update. | inference_log.py (label recording), config.py *(not in baseline bundle)* (label definition) | ✗ **partial** — inference_log.py alone shows label recording; config.py confirms label is mutable. Add config.py to --code for full evidence. |

---

## Context expansion schedule

Run these in order. Each is a separate pre-registered run with its own trace.

| run id | bundle change | class | new items unlocked | expected max |
|--------|--------------|-------|--------------------|:------------:|
| 000 | baseline (no change) | generic-scaffolding | 1, 3, 4 | 3/5 |
| 001-A2 | add `scripts/witness_ingest.py` to --code | fair-evidence | 2 | 4/5 |
| 001-A5 | add `config.py` to --code | fair-evidence | 5 | 4/5 |

Context expansion runs must be reported as "with expanded context" and not compared head-to-head with the baseline score.

---

## False positive reference (confirmed resolved — score −1 if claimed)

These items appeared in Exp 009/012 output but are verifiably present in the current codebase:

| claimed gap | why it is a false positive |
|-------------|---------------------------|
| "No DPA template / no data processing agreement" | `compliance/02-dpa.md` exists (244 lines) |
| "No DSAR / DSR procedure" | `compliance/05-dsr-procedure.md` exists (208 lines) |
| "VLM witnessing pipeline not implemented / not in any commit" | `scripts/witness_ingest.py` exists with PIL/cv2/Ollama implementation |

---

## Pre-registration metadata

- Committed: 2026-06-09
- git sha: *(filled by commit)*
- Models under test: gemma4:26b
- num_ctx: 32768 (verify active via `ollama show gemma4:26b` before running)
