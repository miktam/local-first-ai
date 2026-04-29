# Experiment 005 — Multi-Model Cascade (Dicer / Describer)

**Phase 0 — build first. Falsifiable experiment design follows the build.**

Pre-registered in [`../scientific_log.md`](../scientific_log.md) on 2026-04-29.

## What this is

A two-stage local cascade that takes a question about a large private corpus and produces a grounded answer without leaving miktam02.

- **Dicer** (`gemma4:e4b`): reads the question and the corpus *manifest*, decides which slice the Describer needs, returns a structured routing plan. Small, fast, cheap to invoke. Runs on a deliberately smaller model so heavy retrieval reasoning never blocks the larger model's compute budget.
- **Describer** (`gemma4:26b`): receives the routed slice plus the original question, synthesises an answer. May emit a clarifying question to the human rather than guess when the input is ambiguous.

Both models served by Ollama on miktam02. Orchestration via OpenClaw. No write surface. No network egress. No frontier model. No NemoClaw.

## Why Phase 0 exists

Pre-registering criteria for a system that has never run is speculation. The honest sequence is *build → use → notice → measure*. Phase 0 produces the working cascade and a notebook of observations. Phase 1 (separate scientific-log entry, written later) is the falsifiable experiment, designed once the build has revealed what is worth measuring.

This separation preserves the Chronos contract: every entry labelled "experiment" comes with pre-registered pass criteria. Builds get their own label.

## Strategic anchor

The local-first thesis rests on the claim that personal context, held privately, is a moat strong enough to overcome the capability gap between local and frontier models for a meaningful class of tasks. The Hobbesian counter — *frontier intelligence is so much better that sovereignty must be surrendered, the way violence is delegated to the state* — is what Phase 1 will eventually test.

Phase 0 is the prerequisite: a cascade actually has to exist before its competitiveness can be measured.

## Architecture (Phase 0)

```
[offline, run once: build_index.py]
                │
                ▼
       [index/manifest.json]      ── small, ~13K, fits in any prompt
       [index/monthly_aggregates.json]
       [index/daily_aggregates.jsonl]
       [index/workouts.jsonl]
       [index/ecg_inventory.json]
                │
                │  (Dicer reads manifest + monthlies)
                ▼
[user query]
                │
                ▼
[Dicer: gemma4:e4b]  ── produces structured routing plan (JSON)
                │
                ▼
[deterministic extractor]  ── reads the plan, pulls slice from export.xml or aggregates
                │
                ▼
[corpus slice]   ── JSONL the Describer can ingest
                │
                ▼
[Describer: gemma4:26b]  ── synthesises answer; may emit clarifying question
                │
                ▼
[answer or question to user]
```

Two stages of LLM, but three offline pieces around them: the indexer (run once, deterministic, no model in the loop), the extractor (runs per query, deterministic, no model in the loop), and the model orchestration. The LLMs do language work; everything else is parsing and IO.

## Corpus

Apple Watch health export, 8 years (2018-04-21 onward), 3.5 GB of HealthKit records in `export.xml` plus 1.1 GB of GPX route files in `workout-routes/`.

Location on miktam02: `~/REPOS/apple_health_export` (absolute, not symlinked into the workspace — keeps published output traces clean).

The corpus is real, personal, non-public, and not memorised by either Gemma or any frontier model. This is the right shape for the strategic claim being explored — the asymmetry the experiment is meant to demonstrate is precisely that this data should never leave the device.

The manifest at `index/manifest.json` is the canonical inventory of what the corpus actually contains: 7,743,796 records across ~46 record types, 4,460 workouts (15 activity types, fencing being the most distinctive personal signature with 394 sessions), and 7 ECGs. See [`build_notes.md`](./build_notes.md) for corpus realities the indexing surfaced — partial coverage on some types, multi-source recording on others, late-feature data discontinuities.

Phase 1 will need a synthetic shadow corpus to enable any frontier-comparator runs without sharing real health data. Design deferred.

## Index (the offline parse)

Run once, deterministic, regenerable from `~/REPOS/apple_health_export/export.xml`. Output is gitignored.

```bash
python3 build_index.py \
    --export ~/REPOS/apple_health_export/export.xml \
    --routes ~/REPOS/apple_health_export/workout-routes \
    --ecg    ~/REPOS/apple_health_export/electrocardiograms \
    --out    index/
```

Produces:

- `index/manifest.json` — record types, counts, date ranges, sources, units, numeric flag. Small enough to feed to the Dicer in full.
- `index/monthly_aggregates.json` — count + min/max/mean per (type, month). Dicer-readable in a single prompt.
- `index/daily_aggregates.jsonl` — count + min/max/mean per (type, day). Larger; deterministic-query target, not fed wholesale to any model.
- `index/workouts.jsonl` — one row per workout, no GPX content (routes inventoried by filename only).
- `index/ecg_inventory.json` — filename + date per ECG, no waveform parsing.

The indexer is a single-pass `xml.etree.ElementTree.iterparse` stream. Memory is bounded regardless of XML size. Re-run after a fresh Apple Health export.

## Open engineering questions for the build week

These are the things Phase 0 needs to *answer*, not pre-register against. Findings accumulate in `build_notes.md`.

1. **OpenClaw configuration for two simultaneous models.** How is `gemma4:e4b` set up alongside `gemma4:26b` so the small model can be invoked cheaply without unloading the large one? What is the cost of warm vs. cold model swaps on miktam02?
2. ~~**Corpus indexing.**~~ **Answered 2026-04-29** — see *Index (the offline parse)* above. `build_index.py` produces a manifest the Dicer reads, plus monthly and daily aggregates the Describer can consume by date-typed slice. The LLMs never parse XML.
3. **Routing plan format.** What does the Dicer return? Pinned candidates: a JSON object with `record_types[]`, `date_range`, `aggregation_level` ("monthly" | "daily" | "raw"), optional `clarifying_question`. The Describer needs to consume the slice deterministically; the extractor needs to consume the plan deterministically. Schema design is the next concrete task.
4. **Clarifying-question protocol.** When and how does the Describer emit a question to the user instead of an answer? This is architecturally load-bearing — a cascade that models its own uncertainty is materially different from one that always answers. Worth treating as a first-class behaviour, not a UX nicety.
5. **Failure surfaces.** Where does the cascade go wrong, and is the failure recoverable? A Dicer that hallucinates a record type not in the manifest, a Describer that ignores its routed input, a clarifying question that loops — all worth logging.

## Out of scope (Phase 0)

- Frontier-model comparator → Phase 1
- Formal pass criteria, queries, rubric → Phase 1
- Escalation logic ("call the general") → Experiment 006
- Sandboxing harness (NemoClaw) → later experiment. Evaluated and deferred 2026-04-29: alpha software, alters inference path, Landlock-only sandboxing on Linux while miktam02 is Apple Silicon.
- Multi-source deduplication (multiple sources logging the same metric in parallel) → flagged in `build_notes.md`; a known limitation of the Phase 0 indexer, exposed to the cascade as a probe (see seed prompt #10).

## Next

In order, smallest dependency first:

1. **Routing-plan schema.** Pin the JSON shape the Dicer must produce. Lives in `routing_plan.schema.json` once written.
2. **Extractor.** Reads a routing plan, pulls the slice from `export.xml` (raw) or `daily_aggregates.jsonl` (aggregated), writes JSONL the Describer can ingest.
3. **Dicer prompt.** System instructions plus the manifest, designed so the model produces routing plans the extractor can consume without retries.
4. **Describer prompt.** System instructions for synthesis-from-slice, including the clarifying-question protocol.
5. **First seed prompt run.** Tier 1, #1 (resting heart rate trend) — see [`seed_prompts.md`](./seed_prompts.md). Log to `build_notes.md`.

## Build observations

See [`build_notes.md`](./build_notes.md). Captures dated entries on what the build surfaced — corpus realities, surprises, failure modes. These observations are the input to Phase 1's pre-registration.

## Seed prompts

See [`seed_prompts.md`](./seed_prompts.md). Four tiers — start, expand, adversarial, meta. Each prompt is grounded in record types confirmed present in `index/manifest.json`. These are exploration probes, not pre-registered queries.
