# Exp 012 — Pre-scored Rubric

**Pre-committed: 2026-06-09. DO NOT MODIFY after first model run.**

All items verified against CasaSol repo state at commit `623c4c8` (2026-06-09).
Scoring is applied to model output after all runs are complete.

---

## Task A — DPO Compliance Extraction

**Context bundle:** CasaSol BRIEF.md + ROPA (compliance/01-ropa.md) + retention schedule (compliance/07-retention-schedule.md) + scripts/inference_log.py snippet (first 50 lines) + scripts/mcp_server.py snippet (first 50 lines).

**Persona:** DPO Compliance Auditor. Find gaps between stated policy, documentation, and implementation.

**Scoring:** 1 point per item correctly identified. 0 for missed. −1 for confirmed false positive (fabricated gap with no basis in the provided context). Max score: 5.

### Ground truth items (established before any run)

**A1 — ROPA-code retention divergence (HIGH)**
- Evidence: `compliance/01-ropa.md` §76 states buyer query text is "not retained beyond the session (default)" and "The query is not stored beyond the session."
- Evidence: `scripts/inference_log.py` line 42-43 stores `"input": input_text` and `"output": output_text` as full text in JSONL for every Ollama call — including buyer MCP queries routed through the pipeline.
- Gap: Documented policy contradicts implementation. Queries are stored indefinitely in `logs/inference.jsonl`.
- Scoring: Award 1 point if the model identifies that stated retention policy conflicts with actual code behaviour. Must reference both policy and implementation to score (not just one side).

**A2 — No DPIA for VLM witnessing pipeline (HIGH)**
- Evidence: `scripts/witness_ingest.py` implements photo ingestion with VLM extraction and face detection — systematic automated processing of photographic data, qualifying as high-risk under Art. 35 GDPR.
- Evidence: `compliance/` directory contains 01-ROPA, 02-DPA, 03-LIA, 04-privacy-notice, 05-DSR, 06-breach-notification, 07-retention, 08-audit-log. No DPIA document exists.
- Gap: High-risk processing activity (VLM + face detection on property photographs) has no Data Protection Impact Assessment.
- Scoring: Award 1 point if the model identifies the absence of a DPIA for VLM/photo processing.

**A3 — MCP server has no authentication (MEDIUM)**
- Evidence: `scripts/mcp_server.py` — no API key, bearer token, session management, or middleware. Any device on the agency LAN can submit queries to the MCP endpoint and receive listing matches.
- Gap: Buyer query matching is a personal-data-adjacent processing activity (ROPA §76). No access control on the endpoint.
- Scoring: Award 1 point if the model identifies missing authentication on the MCP server.

**A4 — Retention schedule does not cover inference.jsonl (MEDIUM)**
- Evidence: `compliance/07-retention-schedule.md` lists MCP audit logs (12 months, `logs/mcp.log`) and buyer query text ("Not retained (default), Session only"). It does not list `logs/inference.jsonl` as a retention-managed asset.
- Gap: `logs/inference.jsonl` contains full input/output text from all Ollama calls with no deletion trigger and no entry in the retention schedule. The file grows indefinitely.
- Scoring: Award 1 point if the model identifies that the inference JSONL log is not covered by the retention schedule, or that it has no deletion/expiry trigger.

**A5 — Model version not auditable in inference log (LOW)**
- Evidence: `config.py` sets `OLLAMA_MODEL_ASSESSMENT = "gemma4:26b"` (string label). `inference_log.py` records `"model": model` — the string label, not a hash or digest.
- Gap: Under Art. 22 GDPR, automated processing of personal data requires accountability for which model performed the processing. The inference log cannot prove which exact model weights processed buyer queries — `ollama pull` silently updates weights under the same label.
- Scoring: Award 1 point if the model identifies that the inference log records model name only, not model version or hash, creating an accountability gap.

---

## Task B — Implementation Gap Detection

**Context bundle:** CasaSol BRIEF.md + BUILD_LOG.md + git log summary (last 20 commits, one line each) + db.py + config.py + scripts/mcp_server.py first 80 lines.

**Persona:** Senior Engineer Auditor. Find gaps between documented claims and shipped code. Cross-reference documentation against implementation.

**Scoring:** 1 point per item correctly identified. −1 for confirmed false positive. Max score: 3.

### Ground truth items (established before any run)

**B1 — No schema versioning for 15-field Reducer output (MEDIUM)**
- Evidence: BRIEF.md describes the Reducer as producing a 15-field structured output stored in SQLite.
- Evidence: `db.py` and `models.py` contain no `schema_version`, migration logic, or version column.
- Gap: Adding or renaming a field in the Reducer prompt silently produces records incompatible with the existing schema. No migration path exists. Old records cannot be distinguished from new ones by field presence.
- Scoring: Award 1 point if the model identifies the absence of schema versioning / migration for the Reducer output fields.

**B2 — Model pinned by label only — no hash (MEDIUM)**
- Evidence: `config.py` line 20-21: `OLLAMA_MODEL_ASSESSMENT = "gemma4:26b"` and `OLLAMA_MODEL_ROUTER = "gemma4:e4b"` — string labels only.
- Evidence: `inference_log.py` records `"model": model` (the label string).
- Gap: `ollama pull gemma4:26b` silently replaces the model weights under the same tag. No hash or digest is recorded anywhere. Reducer output can drift between runs with no detection mechanism and no audit trail.
- Scoring: Award 1 point if the model identifies that model versioning relies on a mutable tag with no hash/digest pinning.

**B3 — MCP server has no concurrency model (MEDIUM)**
- Evidence: `scripts/mcp_server.py` — no `async`, `thread`, `worker`, `concurrent`, or `uvicorn workers` configuration.
- Evidence: BUILD_LOG.md describes a booth demo at OLÉ Marbella (June 17-18) with multiple concurrent visitors expected.
- Gap: FastMCP defaults to single-threaded request handling. Three concurrent demo queries at a busy booth is a plausible and documented scenario. No tested fallback or queue exists.
- Scoring: Award 1 point if the model identifies that the MCP server is single-threaded with no concurrency model, specifically in the context of the documented booth demo scenario.

---

## False positive examples (do NOT score these)

The following are items that appeared in Exp 009 but are no longer accurate as of this commit:

- "No DPA template" — resolved: `compliance/02-dpa.md` (244 lines)
- "No DSAR procedure" — resolved: `compliance/05-dsr-procedure.md` (208 lines)
- "VLM witnessing pipeline not in any commit" — resolved: `scripts/witness_ingest.py` with full PIL/cv2/Ollama implementation

A model that reports these as current gaps is hallucinating stale findings. Score −1 per confirmed false positive on items that are verifiably present in the current codebase.

---

## Scoring summary table

| Item | Task | Severity | Max pts |
|------|------|----------|---------|
| A1 — ROPA-code retention divergence | A | HIGH | 1 |
| A2 — No DPIA for VLM witnessing | A | HIGH | 1 |
| A3 — MCP no authentication | A | MEDIUM | 1 |
| A4 — inference.jsonl not in retention schedule | A | MEDIUM | 1 |
| A5 — Model version not auditable | A | LOW | 1 |
| B1 — No schema versioning | B | MEDIUM | 1 |
| B2 — Model pinned by label only | B | MEDIUM | 1 |
| B3 — MCP no concurrency model | B | MEDIUM | 1 |

**Task A max: 5. Task B max: 3. FP penalty: −1 per confirmed false positive.**

*Rubric committed before first model run. Hash this file to prove pre-registration.*
