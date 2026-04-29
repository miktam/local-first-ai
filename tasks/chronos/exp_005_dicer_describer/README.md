# Experiment 005 — Multi-Model Cascade (Dicer / Describer)

**Phase 0 — build first. Falsifiable experiment design follows the build.**

Pre-registered in [`../scientific_log.md`](../scientific_log.md) on 2026-04-29.

## What this is

A two-stage local cascade that takes a question about a large private corpus and produces a grounded answer without leaving miktam02.

- **Dicer** (`gemma4:e4b`): reads the query, decides which slice of the corpus the Describer needs, returns a routing plan. Small, fast, cheap to invoke. Runs on a deliberately smaller model so heavy retrieval reasoning never blocks the larger model's compute budget.
- **Describer** (`gemma4:26b`): receives the routed slice plus the original query, synthesises an answer. May emit a clarifying question to the human rather than guess when the input is ambiguous.

Both models served by Ollama on miktam02. Orchestration via OpenClaw. No write surface. No network egress. No frontier model. No NemoClaw.

## Why Phase 0 exists

Pre-registering criteria for a system that has never run is speculation. The honest sequence is *build → use → notice → measure*. Phase 0 produces the working cascade and a notebook of observations. Phase 1 (separate scientific-log entry, written later) is the falsifiable experiment, designed once the build has revealed what is worth measuring.

This separation preserves the Chronos contract: every entry labelled "experiment" comes with pre-registered pass criteria. Builds get their own label.

## Strategic anchor

The local-first thesis rests on the claim that personal context, held privately, is a moat strong enough to overcome the capability gap between local and frontier models for a meaningful class of tasks. The Hobbesian counter — *frontier intelligence is so much better that sovereignty must be surrendered, the way violence is delegated to the state* — is what Phase 1 will eventually test.

Phase 0 is the prerequisite: a cascade actually has to exist before its competitiveness can be measured.

## Architecture (Phase 0)

```
[user query]
     │
     ▼
[Dicer: gemma4:e4b]  ── reads corpus index, produces routing plan
     │
     ▼
[corpus slice]       ── deterministic fetch, no model in this step
     │
     ▼
[Describer: gemma4:26b]  ── synthesises answer; may emit clarifying question
     │
     ▼
[answer or question to user]
```

## Corpus

Apple Watch health export, eleven years, ~6GB.
Location on miktam02: `~/REPOS/apple_health_export`.

The corpus is real, personal, non-public, and not memorised by either Gemma or any frontier model. This is the right shape for the strategic claim being explored — the asymmetry the experiment is meant to demonstrate is precisely that this data should never leave the device.

Phase 1 will need a synthetic shadow corpus to enable any frontier-comparator runs without sharing real health data. Design deferred.

## Open engineering questions for the build week

These are the things Phase 0 needs to *answer*, not pre-register against. Capture findings inline in this README or in a `build_notes.md`.

1. **OpenClaw configuration for two simultaneous models.** How is `gemma4:e4b` set up alongside `gemma4:26b` so the small model can be invoked cheaply without unloading the large one? What is the cost of warm vs. cold model swaps on miktam02?
2. **Corpus indexing.** What index does the Dicer read against to produce useful routing plans on 6GB of heterogeneous timeseries? A flat file listing? A structured manifest? An embedding store? Pick the simplest thing that lets the Dicer route across years and metrics; only add complexity when a seed prompt makes it necessary.
3. **Routing plan format.** What does the Dicer actually return? File paths and time ranges? A natural-language description? A structured JSON object? The Describer needs to be able to consume it deterministically.
4. **Clarifying-question protocol.** When and how does the Describer emit a question to the user instead of an answer? This is architecturally load-bearing — a cascade that models its own uncertainty is materially different from one that always answers. Worth treating as a first-class behaviour, not a UX nicety.
5. **Failure surfaces.** Where does the cascade go wrong, and is the failure recoverable? A Dicer that hallucinates a non-existent slice, a Describer that ignores its routed input, a clarifying question that loops — all worth logging.

## Out of scope (Phase 0)

- Frontier-model comparator → Phase 1
- Formal pass criteria, queries, rubric → Phase 1
- Escalation logic ("call the general") → Experiment 006
- Sandboxing harness (NemoClaw) → later experiment. Evaluated and deferred 2026-04-29: alpha software, alters inference path, Landlock-only sandboxing on Linux while miktam02 is Apple Silicon.

## Build observations

*(Capture during build week. Surprises, failure modes, behaviour the cascade exhibited that wasn't expected. These observations are the input to Phase 1's pre-registration.)*

## Seed prompts

See [`seed_prompts.md`](./seed_prompts.md). These are exploration probes, not pre-registered queries.
