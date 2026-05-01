# ADR 001 — Routing-plan schema for the Dicer

**Status:** Accepted
**Date:** 2026-04-30
**Context:** Experiment 005 Phase 0
**Artefact:** [`routing_plan.schema.json`](./routing_plan.schema.json)

## Context

The Dicer (`gemma4:e4b`) reads a user query plus the corpus manifest and produces a structured output that the deterministic extractor consumes to assemble a slice for the Describer. The shape of that output is load-bearing: the extractor is built against it, the Dicer prompt is constrained by it, and the Describer reads what it produces. Get this wrong and three things break together.

This ADR captures four design decisions that fixed the schema, with the alternatives considered and the reasoning. The schema itself lives in [`routing_plan.schema.json`](./routing_plan.schema.json).

## Decisions

### 1. Multiple slices per plan, not single

A plan contains a `slices` array (1–8 items), not a single slice.

**Why.** Compound queries — *"compare my fitness in 2019 to now"* — naturally want two slices with different date ranges. Allowing the Dicer to emit both in one pass avoids round-tripping for compound queries, which would double Dicer latency without buying clarity. Gemma e4b can plausibly handle two-to-three slices in one structured output; if it can't, that's a Phase 0 finding worth logging.

**Alternative considered.** Single slice per plan, orchestrator calls the Dicer N times for compound queries. Cleaner per-call shape but doubles latency and creates a coordination problem (does the orchestrator know it's a compound query before the first Dicer call?).

**Bound.** `maxItems: 8`. A plan with more than 8 slices is almost certainly a Dicer pathology — overdecomposition or a malformed loop. Hard cap, fail the plan if exceeded.

### 2. Workouts as a separate field on each slice, not a record type

Workouts have their own `WorkoutFilter` object on each slice, parallel to `record_types`. A slice must specify at least one of the two.

**Why.** Workouts aren't in the `record_types` namespace in `manifest.json` — they live in their own JSONL file with a different shape (activity-type, duration, distance, energy, source). Overloading `record_types` to also accept workout activity types would conflate two different schemas and force the extractor to demultiplex by string-matching `HKWorkoutActivityType*` prefixes, which is fragile.

A separate field makes the dichotomy explicit and lets the Dicer signal *"workouts only"*, *"records only"*, or *"both"* unambiguously. The extractor switches on which field is populated.

**Alternative considered.** Treat workouts as if they were a record type, distinguishing by name prefix. Rejected as overloading.

### 3. Plan or question — never both — via discriminated union

The Dicer's top-level output is a discriminated union: either `{kind: "plan", slices: [...]}` or `{kind: "question", question: "..."}`. The two branches share no fields beyond `kind`.

**Why.** A clarifying question implies the Dicer cannot responsibly route yet. If the Dicer returns a plan *and* a question, the extractor's behaviour is ambiguous — does it execute the plan and surface the question, or hold the plan pending a user reply? Forcing the Dicer to choose one or the other puts the decision in the right place (the model with the query in front of it) and keeps the orchestrator simple.

**Alternative considered.** Single object with both a plan and an optional `clarifying_question` field. Rejected — see ambiguity above. The model is free to *suggest* a clarifying question after producing an answer; that's a Describer concern, not a Dicer concern.

**Implication for the Dicer prompt.** It needs explicit instruction that ambiguous queries should be answered with `kind: "question"` rather than a low-confidence plan. Otherwise the Dicer will default to plans, because that's what most of its instruction is about.

### 4. Strict schema validation, no repair layer

The extractor validates the Dicer's output against the JSON schema with hard rejection. Malformed output fails the plan; the failure is logged to `build_notes.md` with the raw output and the validator error, and the orchestrator either retries the Dicer (Phase 0: at most once) or surfaces the failure to the user.

**Why.** Repair layers (parse-fix-retry, partial extraction, fuzzy field matching) hide problems by absorbing them. In Phase 0 we are explicitly trying to *find* failure modes — what kinds of queries make the Dicer produce malformed output, what tends to go wrong, whether it's recoverable. A repair layer would mask exactly the data Phase 1's pre-registration depends on.

If structured-output failures turn out to be common, that's itself a finding worth measuring before deciding whether the cure (repair, schema-constrained decoding, function-calling-style enforcement) is worth its cost.

**Alternative considered.** Tolerant parser with auto-repair. Deferred to Phase 1 or later, conditional on Phase 0 evidence that strict validation fails too often to be usable.

## Other choices, briefly

These weren't load-bearing enough to merit their own decisions, but worth noting:

- **`aggregation_level` is a closed enum** of `monthly | daily | raw`. Forces the Dicer to pick coarsest-that-works rather than inventing intermediate levels. The Dicer prompt will instruct it to prefer monthly unless the query needs finer resolution.
- **Date range is optional.** Omitting it means *whole history of the requested types*, which is correct for trend questions like *"how has my resting heart rate changed since 2018."*
- **`max_rows` is a per-slice safety cap** with a hard ceiling of 5000. Defensive; the extractor enforces it and flags truncation in the slice it hands to the Describer. Stops a buggy plan with `aggregation_level: "raw"` and no date range from dumping millions of records into the Describer's context.
- **`label` is a Dicer-author hint to the Describer.** The Describer sees labels like *"2019 baseline"* and *"recent year"* in compound plans, which keeps slices distinguishable in synthesis. The Dicer's `rationale` field, by contrast, is for logging only — the Describer never sees it, so the Dicer can think out loud there without polluting the synthesis context.
- **Local time, not UTC.** Date ranges use the local-time prefix from Apple's strings, matching the indexer's choice. Documented on the schema field. *"May 14"* in your data is what you mean when you ask about May 14.

## Consequences

- The extractor (next task) is built against a stable schema. Field names will not change without an ADR amendment.
- The Dicer prompt has a constrained output target — the JSON schema doubles as the prompt's instruction surface ("here is the shape your output must take"). Schema-as-spec.
- Strict validation will produce real failures during the build week. Those failures are the data, not noise.
- Phase 1 may want a repair layer or constrained decoding. Don't pre-empt that decision; let Phase 0 evidence drive it.

## Revisit triggers

This ADR should be reopened if:

- Strict validation rejects more than ~30% of Dicer outputs across the build week's seed prompts. That would mean the schema or the Dicer prompt is mis-shaped relative to what `gemma4:e4b` can reliably emit.
- A compound query naturally wants more than 8 slices. The cap was a guess; if real queries hit it, raise it or rethink decomposition.
- The "plan or question, never both" rule turns out to mean the Dicer routes confidently when it shouldn't, because mixing the two is the only way it can express partial uncertainty. That would shift the decision from architecture to model behaviour and merit a re-examination.
