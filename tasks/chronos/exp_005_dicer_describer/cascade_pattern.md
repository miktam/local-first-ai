# The Dicer/Describer cascade — an architectural pattern for big private data on local hardware

**Status:** Phase 0 deliverable, internal source-of-truth.
**Date:** 2026-05-02
**Context:** Experiment 005 Phase 0, drawn from build observations 2026-04-29 to 2026-05-02.
**Companion artefacts:** [`README.md`](./README.md), [`build_notes.md`](./build_notes.md), [`ADR-001-routing-plan-schema.md`](./ADR-001-routing-plan-schema.md), [`ADR-002-cliff-aware-ceilings.md`](./ADR-002-cliff-aware-ceilings.md).

This document captures the architectural pattern that emerged from Phase 0. It is the source for any future public writeup. It does not claim the pattern is novel; it claims the pattern works under specific local-inference constraints, and names the constraints.

## 1. The problem shape

A class of useful queries shares three properties at once:

- **The corpus is large.** Larger than any practical context window, often by orders of magnitude. (Phase 0 corpus: 7.7M records, 3.5GB of HealthKit XML, 4,460 workouts.)
- **The corpus is private.** It cannot be sent to a frontier model without surrendering data sovereignty. (Phase 0 corpus: 8 years of personal health data.)
- **The hardware is consumer-grade.** Local inference happens on unified-memory machines (Apple Silicon, similar) where prefill goes super-quadratic somewhere between 25K and 35K on-the-wire tokens. This is the *memory bandwidth cliff* documented in [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/).

Each of these constraints alone has known answers. Big corpus alone: retrieval-augmented generation. Private alone: a frontier model running locally on hosted hardware (defeats the privacy point, but architecturally simple). Local hardware alone: small models with short prompts. Take all three together and the conventional answers fail. Naive RAG sends candidate chunks to a model that may push past the cliff. Frontier-model-with-RAG breaks data sovereignty if the chunks leave the device. Local-model-with-everything-in-context fails the moment the corpus exceeds the cliff zone.

The cascade is what fits in this intersection.

## 2. A note on the demand-signal asymmetry

This document refers throughout to *demand-signal evidence* and the *demand-signal asymmetry*. The terms are worth defining briefly because the rest of the pattern is, in part, an attempt to make them operational.

The framing comes from a recurring question in the local-first AI conversation: *given that frontier models have so much more capability than local models, what could a local-first architecture possibly offer that the frontier doesn't?* The conventional answer is "privacy" — and that answer is true, but it's defensive. It says the local model is *not worse on a dimension users care about*, but it doesn't say anything is *better*.

The demand-signal asymmetry is the affirmative version. It says: *a local model running inside the user's own data perimeter can see things a frontier model fundamentally cannot, and that visibility is not just incidental — it changes what the model can answer.*

Made concrete with an example from this corpus: the user has 394 fencing sessions over eight years, the second-largest category in their workout history. A frontier model with no access to the corpus cannot know this. It cannot say *"fencing is a defining activity for you."* It cannot ask *"are you asking about VO2 max, resting heart rate, or activity-type consistency?"* It can only ask generic questions and produce generic answers. The local cascade, with the corpus visible, can do both. That difference — what the local architecture can produce that the frontier architecture cannot — is the demand signal.

The asymmetry has two surfaces in this pattern, and Phase 0 produced evidence for both:

- **In answers** — the Describer surfaced fencing as a personal signature with three years of volume data, grounded in the actual records. Capability the frontier can match only by being given the data, which is the privacy violation the architecture exists to avoid.
- **In clarifying questions** — the Dicer offered concrete disambiguation options (*"VO2 max, resting heart rate, or activity-type consistency"*) drawn from what data exists in the manifest. A frontier model with no manifest access can only ask abstract questions (*"what do you mean by fitness?"*).

When this document says a finding is *demand-signal evidence*, it means: *evidence that the architecture surfaces, asks about, or grounds answers in user-specific content the frontier model cannot see.* That class of evidence is what makes the cascade architecturally interesting beyond the privacy argument.

## 3. The architecture

Five components. Two of them are LLMs; the other three are deterministic.

**Indexer (offline, deterministic, run once).** Streams the raw corpus and writes a manifest plus aggregate tables. The manifest answers *what types of records exist, over what date ranges, in what counts*. Aggregate tables answer *what are the typical values per (type, period)*. The indexer never runs at query time. Its output is the substrate every other component reads.

**Dicer (small LLM, runs per query).** Reads the user's question and a trimmed view of the manifest. Returns one of two structured outputs: a *plan* describing which slice of the corpus the answer needs, or a *clarifying question* when the query is too ambiguous to plan responsibly. Never produces the answer.

**Extractor (deterministic, runs per query).** Reads a plan, validates it strictly against a JSON schema and against the manifest's known-good values, and pulls the requested slice from the indexer's output (or from the raw corpus for `aggregation_level: "raw"`). Applies operational caps that bound the slice's contribution to the bundle. Returns a *slice bundle* the Describer can ingest.

**Bundle guard (deterministic, runs per query).** Estimates the on-the-wire prompt size from the composed (system prompt + user query + slice bundle). If it exceeds the configured ceiling — set below the cliff zone — proportionally downsamples slices to fit, logging what was dropped. Catches what the per-slice extractor caps miss, particularly compound queries where each slice fits individually but the sum overflows.

**Describer (large LLM, runs per query).** Reads the user's question and the slice bundle. Produces the answer. Never sees the raw corpus, never sees the manifest, never makes routing decisions. Its job is synthesis from bounded input.

```
[user query] ─→ [Dicer] ─→ plan or question
                    ↓ (plan)
              [Extractor] ─→ slice bundle
                    ↓
              [Bundle guard] ─→ slice bundle, possibly downsampled
                    ↓
              [Describer] ─→ answer
```

## 4. The invariants the architecture enforces

The pattern's value isn't its components; it's the invariants they jointly enforce. Each invariant addresses a specific failure mode that breaks naive cascades.

**Invariant 1 — The Describer's input always fits below the cliff.** The bundle guard is the last line of defence; the per-slice caps and the Dicer's aggregation choices are the first two. No path produces a Describer call with a prompt past the cliff zone. This is the architectural commitment that makes the cascade viable on local hardware.

**Invariant 2 — The Dicer chooses or asks, never both.** The plan-or-question discriminated union (ADR-001) means the Dicer either commits to a routing decision or admits it can't responsibly commit yet. There is no half-confident plan. Ambiguity is observable; clarification is a first-class output, not a fallback.

**Invariant 3 — The extractor is the only validation boundary.** All plan validation — schema shape, manifest references, range checks, operational caps — happens in the extractor. The orchestrator does not interpret plans; the Describer does not see plans. If a plan is invalid, the failure has a single, structured cause logged to the run trace.

**Invariant 4 — The LLMs are stateless.** Neither the Dicer nor the Describer maintains turn history within or across queries. Same input produces the same output (modulo sampling temperature). This makes traces fully observable and makes the cascade reproducible in a way conversational architectures aren't.

**Invariant 5 — Routing cost stays small even when answering doesn't happen.** A clarifying question costs one Dicer call (~3-5s warm) and zero Describer calls. The cascade's compute economy comes from the small model handling decisions the large model would have wasted minutes on.

## 5. Failure modes Phase 0 surfaced

The architecture is defined as much by the failures it bounds as by the components it composes. Each failure observed during Phase 0 has a documented mitigation; some are still open. Listing them honestly is part of what makes this a pattern rather than a sales pitch.

**Failure: cliff overflow despite the guard.** The VO2/RHR query produced a 75K-token bundle that the guard correctly downsampled to 21,989 tokens, just under our 22K ceiling. The Describer still produced no first byte in 600 seconds. The cliff zone is empirically wider than Incident 003's measurement suggested for *some* prompt shapes, possibly because thinking-model generation expands effective KV cache utilisation past the prefill estimate. Mitigation: lower the ceiling; investigate empirically in Phase 1. Open: cliff measurement on thinking models specifically.

**Failure: small-model prose ignored.** The Dicer prompt's prose guidance — *"Prefer monthly for trends spanning years"*, *"Set max_rows for workouts"* — was ignored by `gemma4:e4b` across multiple runs while the model dutifully copied fixture patterns instead. Few-shot beats prose for this model class. Mitigation: encode constraints as fixture examples, not prose instructions. Open: extent to which this generalises to other small models and other constraint types.

**Failure: streaming-cancel does not work in Ollama 0.20.2.** When the orchestrator abandoned a streaming request that had hit the first-byte timeout, the runner kept spinning at 903% CPU. Recovery required killing `ollama serve`. Mitigation: tighter ceilings to make timeouts rarer. Open: investigate whether a later Ollama version honours client disconnect; possible upstream contribution.

**Failure: structured output produced but malformed.** `gemma4:e4b` occasionally wrapped valid JSON in markdown code fences (` ```json ... ``` `) despite being asked for JSON-only output via Ollama's `format: json` parameter. The wrapping was a *protocol*-level deviation, not a content error. Mitigation: normalisation of the output (strip outer fences) before strict validation. Captured in ADR-001 amendment as boundary-fix-not-repair.

**Failure: stateless cascade can't compose follow-up fragments.** A user answering *"By peak physiological metrics"* to the Dicer's prior clarifying question produced a *second* clarifying question, because the Dicer received the fragment with no memory of the first turn. The architecture is doing the right thing; the user contract is the design question. Two framings for Phase 1: turn-history (matches user expectation) or orchestrator-level fusion (preserves model-layer statelessness).

**Failure: synthesis quality varies on identical inputs.** The same RHR query produced three different answers across three runs, all grounded but with varying coverage and arc-telling. Architectural claim (data sovereignty + grounding) is robust; synthesis claim (useful answer per query) is variance-bounded. Phase 1 evaluation requires N-of-M sampling, not single-shot scoring.

## 6. What the architecture does not solve

Equally important. Naming the limits keeps the pattern honest and frames the open questions Phase 1 will address.

**Memorisation and retrieval drift.** The Dicer routes against a manifest, not against semantic content. Questions that need to find a *specific* episode by description (*"the day I felt unusually tired"*) are out of reach without an embedding index, which we deliberately did not build for Phase 0. The pattern handles structured-retrieval well and free-text-retrieval not at all.

**Causal claims.** The Describer can describe trends, peaks, and changes. It cannot establish cause. *"Why did my heart rate spike on March 14"* is correctly declined by a well-prompted Describer; the cascade has no signal about life events outside the watch's recordings. This is a property of the corpus, not the architecture, but it bounds what the cascade can responsibly answer.

**Multi-source deduplication.** The Phase 0 indexer treats every record as it appears. Apple Health records are written by multiple sources in parallel — watch, phone, third-party apps — and totals can double-count. The cascade can flag the ambiguity but cannot resolve it without a deduplication pass we have not yet designed.

**Hardware variance.** The 22K ceiling, the cliff onset, the wall-clock numbers — all are specific to one machine (Mac mini, unified memory, gemma4:26b). The pattern is portable; the constants are not. Anyone applying this needs to measure their own cliff.

**Streaming reliability.** Until Ollama (or whatever inference runtime) honours streaming-cancel cleanly, the cascade has a one-cliff-hit-per-restart reliability ceiling. The architecture works; the substrate it runs on has rough edges.

## 7. What is transferable

This section is the seed for any external writeup. It lists the parts of the pattern that are not specific to health data, Apple watches, Gemma 4, or Mac mini.

**The cliff is a constraint, not an obstacle.** The most important architectural commitment is that the cascade is *defined* to stay below the local-inference cliff. Frontier-model architectures don't have to think about this. Local-first architectures that don't think about this fail the same way ours did before ADR-002. The pattern's value comes from making the cliff explicit and making the architecture respect it by construction.

**Routing is a small-model job.** A small fast model producing a structured plan against a manifest is a much better use of compute than a large model producing an answer over an unbounded context. The compute economy is *the cascade's value proposition*. This generalises to any system where retrieval-and-routing decisions can be separated from synthesis.

**Determinism is the spine.** The two LLMs do language work; everything else is parsing, schema enforcement, and IO. Most failure modes that hit small-LLM systems become tractable when the surrounding pipeline is deterministic and strictly validated. Few-shot beats prose; structured output beats free-form; protocol normalisation beats repair.

**Clarification is a first-class output, not a fallback.** Treating "I cannot responsibly route this" as a structured outcome — distinct from "I will route this with low confidence" — is what keeps the cascade from producing confidently wrong answers on ambiguous queries. The latency advantage is real (3.7 seconds for clarification vs. 90+ seconds for a wrong answer), and the user-experience advantage is also real (specific options grounded in what data actually exists, vs. generic *"what do you mean?"*).

**Statelessness is a feature, with a cost.** Stateless models are reproducible, observable, and don't drift across calls. Stateless cascades cannot follow up on user fragments. For research and pre-registration, the trade is worth it; for production user experience, the orchestrator can fuse history without weakening model-layer statelessness. Pick the trade deliberately.

**The moat shows up in two places.** Phase 0 produced two kinds of demand-signal evidence. The first is in answers: the Describer surfaced fencing as a personal signature with three years of volume data — content a frontier model with no access cannot produce. The second is in clarifying questions: the Dicer's options were grounded in the user's actual data (*"VO2 max, resting heart rate, or activity-type consistency"*), not generic categories — questions a frontier model with no access cannot ask. Both are the demand-signal asymmetry made operational.

## 8. Prior art and what's different

This document does not claim the cascade is novel. The components are familiar; the framing is older than this experiment. Naming the prior art is part of being honest about what was actually contributed.

**Plan-and-Solve prompting** ([Wang et al., 2023](https://aclanthology.org/2023.acl-long.147/)) split a single LLM call into two phases — devise a plan, then execute it — to reduce missing-step errors in multi-step reasoning. The Dicer/Describer split is the same shape, separated across two models instead of two phases of one model.

**LLM cascading and routing** ([RouterBench and related work](https://arxiv.org/abs/2410.10347), [FrugalGPT, AutoMix](https://towardsdatascience.com/llm-routing-intuitively-and-exhaustively-explained-5b0789fe27aa/)) describe the broader pattern of using a smaller, cheaper model first and escalating to a larger model only when needed. The Dicer being smaller than the Describer reflects this thinking; the Phase 0 cascade is *not* a routing system in the strict sense (we don't evaluate Dicer answers and escalate; we use the small model for routing and the large model for synthesis), but the cost-economy intuition is identical.

**Decomposed prompting** ([Khot et al., 2023](https://arxiv.org/abs/2210.02406)) generalises plan-and-solve into a modular framework where sub-tasks are dispatched to specialised handlers. The deterministic extractor in this pattern is essentially a non-LLM specialised handler dispatched by the Dicer; the architecture is one instance of decomposed prompting where the "specialist" happens to be parsing code rather than another model.

**Planner/executor or planner/critic agent patterns** (e.g., [APEX-Searcher](https://arxiv.org/pdf/2603.13853), [D²Plan](https://arxiv.org/pdf/2601.08282), and many recent agentic systems) split high-level decision-making from local execution. The Dicer is a planner; the extractor + Describer are the execution path. Same shape again, with the constraint that this Phase 0 implementation is single-pass — the Dicer plans once, the Describer answers once, no iterative replanning.

Given all of that, what is the contribution here, if anything?

The honest answer: the contribution is not the architecture. It is the framing of the architecture as *the response to a specific local-hardware constraint*. Most prior-art cascades and planner/executor systems were designed to optimise *cost* or *quality* on hosted inference, where neither prefill latency nor memory bandwidth is the dominant variable. They split work across models because larger models cost more API tokens, not because the larger model literally cannot finish a long-context job on the hardware that's available.

This pattern is built around a different binding constraint: the memory bandwidth cliff documented in [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/). The cascade doesn't exist to save money; it exists because, on consumer-grade unified-memory hardware, the larger model genuinely cannot operate at scale on naive inputs. ADR-002's invariants — bounded slices, tight ceilings, structurally-enforced cliff awareness — make the cascade an *expression of* the constraint rather than an optimisation around a soft preference.

That's the small contribution worth claiming: not "we invented a cascade," but "we re-derived the cascade pattern from first principles starting from a constraint that prior art mostly didn't have to consider, and the constraint turned out to dictate which invariants matter most." If frontier-model hosted inference had the same memory-bandwidth profile as consumer unified-memory hardware, every cascade in the literature would probably look more like ours: tighter, more deterministic, more aggressive about routing decisions, and explicitly committed to staying below a hardware ceiling rather than optimising on a cost curve.

Reinventing the wheel on purpose, by understanding why the wheel is the right shape — that was the point of Phase 0 working from first principles. The patterns in the literature show what we converged toward; the cliff explains why the convergence wasn't optional.

## 9. The recipe — what to do with this if you have your own corpus

This is the "so what" of the document. If you have a large private corpus and want to apply the cascade pattern, here is the structural recipe and the parameters you need to measure for your own setup. The numbers from this Phase 0 are illustrative; the *steps* generalise.

**Step 1 — Measure your hardware's cliff.**

This is non-negotiable and comes before any other architectural choice. On the hardware you'll deploy on, with the *Describer model* you intend to use, sweep prefill latency at three input sizes (e.g., 5K, 15K, 25K tokens) and find the point where ms/token starts climbing super-linearly. That point — minus a margin of 10-20% — is your operational bundle ceiling.

For this Phase 0: Mac mini, gemma4:26b, cliff onset around 25K-35K tokens (Incident 003), operational ceiling 22K. Your numbers will differ. Anyone who skips this step will re-experience our wedged-Ollama hour.

**Step 2 — Index offline. Decide your aggregate granularities.**

Build a manifest that lists every type of thing in the corpus, its date range (or other primary axis), and its volume. Build aggregate tables at two granularities: a *coarse* one small enough to fit in any prompt (the Dicer's view), and a *fine* one large enough to support specific queries (the Describer's substrate when needed).

The numbers depend on your corpus. For 7.7M time-series records across 8 years and 46 record types, monthly aggregation produced 2,356 cells (~225KB) — small enough to feed to the Dicer in full. Daily aggregation produced 59,224 cells (~5.8MB) — too big for any prompt, but cheap to query deterministically. Your axes might not be temporal; your aggregates will follow whatever structure your corpus actually has. Both granularities matter; neither alone is enough.

**Step 3 — Pick model sizes that respect the role asymmetry, not the relative size.**

The Dicer should be small enough that routing is cheap (warm latency under ~5 seconds on your hardware) and large enough to follow JSON-shaped structured-output instructions. The Describer should be large enough to synthesise from a slice and small enough to run locally on your hardware below the cliff.

In this Phase 0: Dicer = `gemma4:e4b` (9.6GB), Describer = `gemma4:26b` (17GB). The Dicer is *not* an order of magnitude smaller than the Describer; it's about 56% the size. The cost saving comes from the Dicer producing a small *output* (a structured plan, hundreds of tokens) rather than from the Dicer being trivially fast. Pick by role, not by size ratio.

**Step 4 — Make routing structured.**

The Dicer's output must be a JSON object validated against a schema before anything downstream consumes it. The schema describes a *plan* — which slice of the corpus the Describer needs — with a few key fields:
- which logical types/fields/tables to pull from
- what range or filter applies
- what aggregation level (coarse, fine, raw)
- an optional row cap

Plus a discriminated alternative: the Dicer can return *kind: question* instead of a plan, when the query is too ambiguous to route responsibly. Treating ambiguity as a first-class structured outcome — not a low-confidence plan — is what keeps the cascade from producing confidently wrong answers on under-specified queries.

**Step 5 — Make the extractor deterministic and the only validation boundary.**

A single piece of code reads the plan, validates it strictly (schema shape, references actually exist in the manifest, ranges parse), pulls the requested slice, and applies operational caps that bound any one slice's contribution. No LLM in this step. No fuzzy matching, no plan repair, no fallback. If the Dicer produces an invalid plan, the failure has a single structured cause logged to disk.

The caps are where you encode the cliff awareness. For this Phase 0: aggregated workouts capped at 200 rows (with yearly coarsening as a fallback before truncation), daily record aggregates capped at 1500 rows. These are the operational ceilings — *below* the schema's absolute maximum (5000) — that the architecture enforces silently when the Dicer doesn't ask explicitly. Pick yours based on the cliff measurement from Step 1.

**Step 6 — Add a bundle-level guard for compound queries.**

The per-slice caps in the extractor handle simple queries. Compound queries — multiple slices in one plan — can fit each slice individually and overflow when concatenated. A token-count estimate (roughly chars / 4 for English-ish JSON, but measure your own ratio for non-English content or heavily-structured JSON) on the composed prompt before sending. If it exceeds the ceiling from Step 1, downsample slices proportionally with a minimum floor (~20 rows per slice) and flag the downsampling in the bundle so the Describer knows.

**Step 7 — Stream the Describer with three timeouts.**

Treat *time-to-first-token* and *time-between-tokens* as structurally different. On local hardware with thinking models, prefill can take minutes; once tokens start, the gap between them is much shorter. A single read timeout conflates the two and either aborts prefill prematurely or hangs forever on stuck generation. Three timeouts: first-byte (generous, e.g., 600s), idle (tight, e.g., 90s), total (absolute ceiling, e.g., 1800s). Each fires for a different failure mode; each writes a different error code to the trace.

**Step 8 — Keep both LLMs stateless.**

Same input → same output (modulo sampling). No turn history inside the model layer. If the user contract requires conversational follow-ups, manage that state in the orchestrator and re-pose the fully-fused question to the Dicer. Statelessness in the model layer is what makes traces reproducible and failures observable; it's a feature even when it makes follow-ups awkward.

### What this recipe does not give you

It does not give you a magic constant. *"Index every 1GB to ~XMB"* is meaningless without knowing what your data actually contains. A gigabyte of source code is not a gigabyte of timeseries records is not a gigabyte of email. The recipe gives you *what to measure* and *what each measurement constrains*. The constants are yours to derive.

It does not give you an embedding-based retrieval layer. The Dicer routes against a structured manifest, not against semantic similarity. If your queries genuinely need free-text retrieval ("the email where I mentioned cancelling that subscription"), you need an embedding index in addition to or instead of the manifest. The cascade pattern composes with embeddings; this Phase 0 deliberately did not, because we wanted to characterise the structured-routing path on its own.

It does not give you reliability past one cliff hit. As of Ollama 0.20.2, abandoned streaming requests wedge the runner. The cascade can produce a clean failure trace, but the inference runtime needs a manual restart afterward. Until that's fixed upstream or worked around, plan for the operational reality: *iteration speed is bounded by how often you hit the cliff and have to recover.*

### What it does give you

A discipline. Every architectural decision in the cascade points at the same constraint: *the on-the-wire prompt to the synthesiser must stay below the local-inference cliff, by construction, regardless of what the user asks.* Every component — the Dicer's routing, the extractor's caps, the bundle guard's downsampling, the schema's absolute ceiling — is a different mechanism for enforcing that one invariant.

That discipline is what the pattern transfers. Apply it to your corpus, on your hardware, with your model — and the architecture will respect your cliff the same way ours respects ours. The numbers change. The shape doesn't.

## 10. What Phase 1 measures

Phase 0 produced a working pattern. Phase 1 will test whether the pattern is competitive with a frontier model on a defined query class. Three open questions Phase 1 must close:

- **Cliff measurement on thinking models.** Incident 003 measured prefill on a 35K-token input and found the bandwidth cliff. Phase 0 hit a cliff at 22K with thinking-phase generation in play. The mechanism is plausibly different — prefill plus initial-thinking-token KV expansion — and the operational ceiling depends on the answer.

- **Comparator design.** The frontier comparator has been pinned (Claude Opus 4.7); the context shape has not. Spec A (synthetic shadow corpus, full parity) and Spec B (aggregate-only, lived asymmetry) produce different findings. Both are defensible. Phase 1 picks one as primary, optionally reports the other.

- **Scoring rubric.** Today's variance results show single-shot scoring is unreliable. The rubric needs to score the *distribution* of answer quality across N-of-M samples, plus a separate reliability axis (does the cascade complete? is it within wall-clock budget?). This is a design question Phase 1 has to settle before any comparator runs happen.

These three are the spine of the next experiment. The cascade is the prerequisite. The experiment is the point.
