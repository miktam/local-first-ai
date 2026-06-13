# Exp 018 — Sovereignty Resilience: Three Failure Modes

*Pre-registered: 2026-06-13*  
*Status: PRE-REGISTERED — awaiting execution*  
*Referenced in: [We Didn't Notice](https://localfirstai.eu/posts/2026-06-13-we-didnt-notice/)*

---

## Motivation

On 11 June 2026 the US government suspended Anthropic's Fable and Mythos models for all
foreign nationals. Every agency using the Anthropic API, Amazon Bedrock, or Google Cloud
Platform lost access immediately. CasaSol was unaffected.

That claim rests on architecture, not measurement. This experiment treats it as an evidence
question: can we deliberately break the local infrastructure in the ways an external provider
failure would simulate, observe the failure mode, and confirm recovery is a local configuration
decision — not a migration?

---

## Hypothesis

**H1 (service interruption):** Stopping the Ollama daemon degrades CasaSol MCP tool responses
to a predictable error; no data is lost; restarting Ollama restores full service with no
re-enrichment required.

**H2 (model unavailability):** Removing the `gemma4:26b` weights from disk (simulating a
weight pull failure or quota block) produces a recoverable error at the Reducer stage; the
SQLite corpus is intact; restoring the weights restores service.

**H3 (network isolation):** Blocking all outbound network on miktam02 (pf firewall rule) does
not affect CasaSol MCP `search_properties` or `get_property` responses, since inference and
storage are local. The only expected failure is HuggingFace weight download — not inference.

---

## Design

**Hardware:** miktam02 (Mac mini M4 Pro, 64 GB) — production setup unchanged.

**Scenarios (run sequentially, restore between each):**

| Scenario | Action | Expected failure mode | Recovery action |
|---|---|---|---|
| S1 — daemon down | `sudo launchctl bootout system /Library/LaunchDaemons/com.ollama.serve.plist` | MCP tools return 503 / connection refused | Restart Ollama daemon |
| S2 — weights gone | `ollama rm gemma4:26b` | Reducer stage returns model-not-found error | `ollama pull gemma4:26b` |
| S3 — network cut | `sudo pfctl -e` + block rule for all outbound | Inference unaffected; HF download fails | Remove pf rule |

For each scenario:
1. Confirm baseline: `mcp_server.py search_properties` returns results
2. Apply disruption
3. Run `search_properties` again — record exact error
4. Confirm SQLite corpus still readable (direct query, no Ollama)
5. Restore service
6. Run `search_properties` again — confirm full recovery

**Evidence per scenario:** `evidence/S{N}-{slug}/before.json`, `during.json`, `after.json`,
`recovery_steps.txt`, ISO timestamp in filename.

---

## Success criteria

| Criterion | Pass condition |
|---|---|
| H1 | S1 error is deterministic; SQLite intact; recovery ≤2 commands |
| H2 | S2 error names missing model; corpus intact; recovery = `ollama pull` |
| H3 | S3 inference succeeds; only outbound-dependent ops fail |
| All | Zero data loss across all three scenarios |

Partial pass (2/3) is reportable but not sufficient to claim full sovereignty resilience.

---

## Artifacts structure

```
exp_018_sovereignty_resilience/
├── README.md          ← this file (pre-registration)
└── evidence/
    ├── S1-daemon-down/
    ├── S2-weights-gone/
    └── S3-network-cut/
```
