# ADR 002 — Cliff-aware operational ceilings for the cascade

**Status:** Accepted
**Date:** 2026-05-02
**Context:** Experiment 005 Phase 0
**Supersedes:** *(amends ADR-001's silence on operational ceilings; doesn't supersede)*
**Related:** [Incident 003 — The Memory Bandwidth Cliff](https://localfirstai.eu/posts/incident_003_alpha_post/) (2026-04-28)

## Context

On 2026-05-01, Phase 0's first compound query (workout signature probe over 4,460 workouts) failed at the Describer stage. The cascade plumbing was correct end-to-end: Dicer routed cleanly, the extractor produced an aggregated 347-row slice, the token guard reported ~21K tokens — well under the previously-set 200K limit. But `gemma4:26b` produced no first byte in 90 seconds, and continued to spin at 900% CPU after the request was abandoned, requiring an `ollama serve` restart to recover.

Re-reading [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/) explained the symptom completely. On miktam02, with `gemma4-think:26b`, prefill is empirically super-quadratic between 25K and 35K on-the-wire tokens. The bottleneck is memory bandwidth, not compute — both GPU and CPU power drop simultaneously as the KV cache crawls across the unified memory bus. The mitigation Incident 003 prescribed was: *treat 25K tokens as a soft ceiling on the on-the-wire prompt.*

Phase 0 had not propagated that constraint into the cascade's defaults. The 200K limit was protecting the model's *advertised context window* (262K), not the *cliff zone*. The cascade's architectural value depends on respecting the cliff — that's the constraint that makes a small-Dicer + bounded-Describer cascade meaningfully different from naively calling a frontier model with whatever context it accepts.

## Decisions

### 1. Bundle-level token ceiling: 22K (was 200K)

`DESCRIBER_CONTEXT_LIMIT_TOKENS` in `cascade.py` is now `22_000`. This sits below Incident 003's 25K cliff onset with margin for the chat template wrapper and the model's own response generation. The bundle-level token guard downsamples slices proportionally if the composed prompt (system + user + bundle) exceeds this limit.

**Why 22K and not 25K.** Two reasons. First, our token-counting is a heuristic (`chars / 4`); under-counting by 10-15% is plausible for JSON-heavy content with lots of structural tokens. Second, the cliff's onset isn't a hard line — Incident 003 measured 13.5 ms/token at 25K, already 1.6× the 15K baseline, so the slowdown is gradual into the cliff. 22K leaves margin for both.

### 2. Per-slice operational caps in `extract.py`

`AGG_WORKOUT_ROW_CAP = 200` and `AGG_RECORD_DAILY_CAP = 1500` enforce per-slice ceilings *during extraction*, before the bundle-level guard ever runs. This is structurally cleaner than catching everything at the bundle level: the extractor is the boundary that already enforces ADR-001's strict-validation contract, so cliff-awareness lives in the same enforcement layer.

If aggregated workouts exceed the cap, the extractor coarsens monthly aggregation to yearly aggregation (`_coarsen_workout_aggregates_to_yearly`) before truncating. A 4,460-workout corpus across 8 years collapses to ~95 yearly rows (15 activity types × 8 years, sparsely populated), well under the cap. Coarsening preserves more signal than truncation.

The schema's `MAX_ROWS_CEILING = 5000` remains as the absolute schema limit (Dicer can't request more even theoretically). The new operational caps (200, 1500) are *defaults* enforced silently when the Dicer doesn't set `max_rows` explicitly.

### 3. First-byte timeout vs idle timeout, separated

The Describer call now has three timeouts:

- `DESCRIBER_FIRST_BYTE_TIMEOUT_S = 600` — generous (10 min) wait for the first event, accommodating prefill latency on local hardware.
- `DESCRIBER_IDLE_TIMEOUT_S = 90` — gap between any two events once streaming has begun; if the model goes silent mid-generation, give up.
- `DESCRIBER_TOTAL_BUDGET_S = 1800` — absolute ceiling regardless of progress.

This distinction matters because *time-to-first-token* and *inter-token-interval* have structurally different causes on local hardware. First-token includes prefill (memory-bandwidth-bound on long inputs); inter-token is generation (compute-bound, predictable). Conflating them produces either premature aborts on long prefill (yesterday's bug at 30s and 90s) or hung sessions on actually-stuck generation.

### 4. The cascade's value is the cliff

This is the architectural framing the implementation is meant to express. The cascade's compute economy comes from:

- **Small Dicer** producing **small plans** (JSON, hundreds of tokens at most).
- **Deterministic extractor** producing **bounded slices** (capped per ADR-002).
- **Describer** working on inputs **structurally constrained to live below the cliff**.

A frontier model that doesn't have a memory-bandwidth cliff doesn't need this architecture. The cascade exists *because* local inference has the cliff, and *because* the architecture turns the cliff from a hazard into an invariant. That's the demand-signal asymmetry made operational: you don't beat frontier capability by being faster, you beat frontier capability by *fitting under a constraint they don't have to think about, with data they can't see*.

## Alternatives considered

### Switch to a non-thinking model for the Describer

Considered and rejected for now. All gemma4 variants in the local library are thinking models; switching to a non-thinking model would mean leaving the family. Worth Phase 1 investigation but not Phase 0 work.

### Use Ollama's `num_ctx` to clamp context window per request

Ollama allows per-request context window override. Setting `num_ctx: 22000` would tell the runtime not to allocate KV cache for the model's full 262K context, potentially saving memory. Worth investigating in Phase 1 — the question is whether smaller `num_ctx` actually changes prefill cost or only memory allocation. ADR-002 doesn't depend on it.

### Streaming with cancellation

The wedged-runner problem (abandoned request keeps spinning) is the strongest motivation for proper cancellation. Ollama's HTTP API may or may not honour client disconnects mid-stream; needs investigation. Out of scope for ADR-002; tracked as a Phase 0 finding.

## Consequences

- The cascade is now *architecturally constrained* to bundles below the cliff. If a query genuinely needs more data than fits in 22K tokens, the answer is to coarsen aggregation, narrow the date range, or refuse — not to give the Describer a bigger bundle.
- Slice truncation is observable (the slice's `truncated` flag, the optional `_workout_coarsened_to` field, the bundle's `_orchestrator_note`). The Describer prompt should be updated to surface these flags in answers when relevant. Pending.
- Phase 1's pre-registration must respect ADR-002's ceilings. Comparator runs against frontier models will use the same query set, but the local cascade's constraints (bounded bundle, coarsened workouts) become part of *what's being measured*, not noise to control for.

## Revisit triggers

This ADR should be reopened if:

- A query that should fit (under 22K tokens, reasonable row counts) still triggers a first-byte timeout. That would mean 22K is still inside the cliff zone for some prompt shapes; lower the ceiling.
- A non-thinking 26b-class model becomes available locally and produces first tokens fast on 25K+ inputs. The cliff may not apply equally to non-thinking variants; ceilings could relax.
- Ollama gains a server-side cancellation API and the wedged-runner problem disappears. Some of the urgency around tight ceilings comes from "if we hit the cliff we can't recover"; if recovery is cheap, the ceiling could be probed harder.
- Hardware changes (M4 Mac mini Pro, dedicated GPU, etc.) shift the cliff to a different token count. Ceilings would need re-measurement.
