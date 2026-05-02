# Phase 0 — build notes

## 2026-04-29 — corpus realities

- Corpus is 8 years (2018-04-21 onward), not 11 as initially framed.
- Several record types are partial: cycling stopped Oct 2023, swimming stopped Sep 2021, sleeping wrist temp / environmental audio / walking steadiness all started 2021–2023. Dicer must respect manifest date ranges or it will route to empty.
- Multi-source recording (watch + phone + Balbus STEP + Health) creates a deduplication problem for total-style queries. Not solved in Phase 0 indexer; flag in any answer involving sums.

## 2026-05-01 — first end-to-end cascade run

**Query:** "How has my resting heart rate changed since 2018?"

**Outcome:** Grounded answer; every cited number matches monthly_aggregates.json.
End-to-end works on first attempt; no Dicer output rejection, no retry needed.

**Wall-clock surprises:**
- Dicer (gemma4:e4b): 27.45s. Heavier than expected for a "small fast" routing
  model. Suspect cold-start dominates first call. Need warm-cache measurement.
- Extractor: 0.072s. Negligible, as expected.
- Describer (gemma4:26b): 136.67s for 96 monthly rows. Sets the rhythm of the
  build week — ~2 min per query.

**Behavioural observations:**
- Describer answer is grounded and specific but flat — lists numbers without
  surfacing the trend arc (RHR ~55 → 47 by 2020, drift back to ~57 by 2022,
  stable since). Synthesis-quality issue, not routing issue.
- Dicer chose monthly aggregation, single slice, no date range — exactly the
  fixture pattern. e4b followed the few-shot example faithfully.
- format: "json" did not produce malformed output. ADR-001 strict-validation
  contract held without retries.

**Open questions surfaced:**
- Cold-start vs steady-state Dicer cost.
- Whether "tell the arc, not just the numbers" is a Describer prompt issue
  or a model capability ceiling.
- Whether 2 minutes per query is a Phase 0 bottleneck worth fixing or just
  the cost of doing business.

## 2026-05-01 — second run, same query

**Wall-clock:**
- Dicer cold: 27.45s (run 1) → warm: 5.47s (run 2). Cold-start theory confirmed.
- Describer: 136.67s → 129.80s. Stable; not cache-dependent at the model level.

**Schema violation observed:**
- Run 2 attempt 0: Dicer emitted a label > 40 chars. ADR-001 retry path
  triggered cleanly; attempt 1 succeeded. First real evidence that strict
  validation catches and recovers from e4b's tight-constraint violations.
- Suggests the 40-char cap on `label` is borderline tight; consider revisiting
  if violation rate stays elevated across the build week.

**Describer non-determinism:**
- Identical query and slice produced meaningfully different answers across
  two runs.
- Run 1 cited 5 distinct datapoints; run 2 cited 3 and used softer language
  ("stayed relatively low") that's more inferential than the data warrants.
- Implications for Phase 1: single-shot scoring is unreliable. Either set
  temperature=0 for deterministic decoding, or run N trials per query and
  report distribution. Decision deferred but flagged.

## 2026-05-01 — three-run variance characterisation

Same query, same slice (96 monthly RHR rows), three runs.

| Run | Cited points | Notable |
|-----|-------------:|---------|
| 1 | 5 | Cold start, 164s total. Mentioned 2022 peak, both recent months, range. |
| 2 | 3 | Schema retry on label cap. Skipped 2022 peak. Soft inferential language. |
| 3 | 5 | Fastest run (102s), full arc 2018→2020 trough→2022 peak→2026 plateau. |

Architectural claim holds across all three: numbers cited are real, no
hallucination, slice round-trips correctly, retry recovered from constraint
violation. Synthesis quality varies. Implications:

- Phase 1 cannot score answers single-shot. Either temperature=0 (loses the
  occasional good run 1/3 in exchange for stability) or N-of-M with quality
  distribution reported.
- Variance bounds may matter more than means. A cascade that's great half the
  time and weak half the time has a different value proposition than one
  that's mediocre every time.

## 2026-05-01 — model size reality check

`gemma4:e4b` is 9.6 GB on disk, `gemma4:26b` is 17 GB. The Dicer is ~56% the
Describer's size, not order-of-magnitude smaller. This explains the warm Dicer
cost (~3–5s rather than sub-second) and means the cascade's compute savings
come from "small model produces small output (JSON plan)" more than "small
model is fundamentally fast." If Phase 1 wants a true fast-router, candidates
are smaller models in the same family or a Qwen 3.5b-class model.

## 2026-05-01 — workout query exposed slice-volume problem

**Query:** "Look at my workout patterns over the last few years..."

**What happened:** Dicer routed correctly (workouts-only, all activity types,
monthly aggregation — matched fixture pattern). But omitted `max_rows`. The
extractor returned all 4,460 workouts. Describer hit the 300s timeout
generating synthesis from 53,536 lines of JSON.

**Architectural lesson:** The Dicer's job is not just "what data does the
question want" — it's also "what data can the Describer actually consume."
Slice volume is a load-bearing routing decision, not an afterthought.

**Fix attempted:** Added volume guidance to Dicer prompt (see commit), raised
Describer timeout to 1800s, added clean timeout handling so partial state
gets logged.

**Open question:** Does e4b reliably follow defensive volume guidance from
the prompt, or does enforcement need to live in the orchestrator? Re-run
will tell us.

## 2026-05-01 — context-window truncation, prompt guidance ignored

**Query:** workout signature probe, second attempt with volume guidance
in Dicer prompt.

**What happened:**
- Dicer ignored the prose volume guidance. No max_rows field on the
  workouts slice. Few-shot fixtures (which don't show max_rows) won
  over the prose section that told it to set one.
- Slice bundle was 717,827 tokens. gemma4:26b has a 262,144 ctx limit.
  Ollama truncated 63% of the input silently (only a WARN in the log).
- Truncation kept the first 262k tokens and dropped the rest. With
  Apple's chronological ordering, this means the Describer sees
  2018-2020 and never sees 2021-2026 in this query.

**Architectural lessons:**
- Prompt-as-defence is unreliable for e4b. Defensive constraints belong
  in the orchestrator, not the system prompt.
- Ollama truncates silently. Without watching the log we'd have gotten
  back a confident answer based on partial data and not known. The
  orchestrator should compute approximate token count before calling the
  Describer and refuse if it exceeds context (or downsample).
- Few-shot examples teach behaviour more reliably than prose
  instructions, at least for e4b. Update fixtures to show max_rows.

**Decision implication:** Volume enforcement moves to the orchestrator.
Dicer prompt guidance stays as a hint but is no longer load-bearing.

## 2026-05-01 — schema-meaning gap, not just prompt issue

Reviewing the Dicer's output: rationale says "A monthly aggregation provides a
good trend overview without being overwhelming." The Dicer chose monthly
*because* it understood volume mattered. The problem is that the extractor
doesn't actually aggregate workouts when aggregation_level="monthly" — that
field only affects record_types (monthly vs daily vs raw). For workouts, the
extractor returns every row regardless.

The Dicer obeyed the schema. The schema doesn't deliver what it promises.

**Decision:** extend the extractor to aggregate workouts by month/day when
the Dicer asks for it, matching the schema's implicit contract. Keep
prose-guidance updates secondary — the real fix is making the architecture
honest, not patching prompts to work around it.

## 2026-05-01 — Dicer output-format unreliability

Two distinct failure modes observed for gemma4:e4b producing structured JSON:

1. Schema violations inside valid JSON (e.g. label > 40 chars). Recoverable
   by ADR-001's retry path. Observed in run 20260501T113735Z.
2. Output-format violations: model wraps valid JSON in markdown code fences
   (```json ... ```). Not recoverable by retry alone — second attempt also
   wrapped in fences. Observed in run 20260501T134013Z.

Format violations are unrelated to the JSON's correctness; the content is
fine, only the wrapper is wrong. Adding a normalisation step (strip code
fences before parsing) handles this without weakening ADR-001's strict-
validation contract on the actual plan structure. ADR-001 amendment to follow.

Pattern recognised: e4b is unreliable at strict structured output. Across
three documented runs, three distinct ways the output deviated from the
schema — label too long, fences around JSON, or correct. Phase 1 should
either pin temperature=0, switch to a stricter constrained-decoding mode,
or accept that retry+normalise is permanent overhead.

## 2026-05-01 — urllib.urlopen timeout is per-read, not connect-only

Streaming Describer call failed at 30s with "timed out" before any byte
arrived. Cause: urllib.request.urlopen(req, timeout=30) treats `timeout`
as a per-read deadline, not a connect-only deadline. On a streaming
endpoint where the server holds the connection open while the model
thinks, this means urllib aborts at 30s if the model hasn't produced
its first token yet. 26b on the workout bundle takes longer than 30s
to start emitting tokens.

Fix: set urlopen timeout to DESCRIBER_IDLE_TIMEOUT_S (90s) so that
urllib's read deadline aligns with our queue-based idle policy.
The queue idle timeout remains the real enforcement; urllib's now
redundant rather than premature.

## 2026-05-01 — re-encountering the memory bandwidth cliff (cf. Incident 003)

Workout query failed at the Describer because the on-the-wire prompt
exceeded the 25K-token soft ceiling identified in Incident 003 (April 28).
The token guard's 200K limit was an order of magnitude too high — it was
protecting the model's *advertised context window*, not the *cliff zone*.

Bundle was 21K tokens. Add the describer prompt (~1K), the chat template,
and the user-message scaffolding, and we land in the 25-30K cliff zone.
At that size, prefill becomes memory-bandwidth-bound; both GPU and CPU
power drop while the KV cache crawls across the bus. No tokens emerge
in tractable wall-clock.

Mitigation per Incident 003 is to keep on-the-wire prompts below 22K
tokens with margin. Phase 0 needs:

1. Token guard limit dropped from 200K → 22K.
2. Workout monthly aggregation capped further (347 rows → ~100 rows).
3. Server-side cancellation: when our client abandons a streaming call,
   Ollama keeps the runner spinning at ~900% CPU. Wedges the model.
   Phase 0 cannot recover from this without an ollama serve restart.
   Worth investigating whether Ollama exposes a cancel API.

Bigger lesson: the cliff isn't a Phase 0 obstacle, it's a Phase 0
*constraint that should have been built in from the start*. The
findings from Incident 003 weren't propagated into cascade.py's
defaults. They are now.

## 2026-05-02 — RHR run with streaming + thinking visible (success)

**Wall-clock decomposition (RHR query, 96 monthly rows, 3.4K tokens bundle):**
- Dicer: 31s (cold start; warm should be ~3-5s).
- Extractor: 0.07s.
- Describer: 127s total. First content token at 78s.
- Thinking-to-answer ratio: 19:1 (6,411 chars thinking, 336 chars answer).

**Cascade behaviour (working as designed):**
- Cliff guard: 3,366 tokens, well under 22K. No truncation.
- Streaming exposed the Describer's deliberation in real time. Thinking
  visible from t=0; content arrived at t=78s; full answer by t=127s.
- ADR-001 strict validation passed without retry; no normalisation needed.

**Synthesis quality observation (most important):**
The Describer's thinking trace shows the describer_prompt is actively
shaping the answer. The model:
- Restated constraints from the prompt
- Drafted, self-corrected, redrafted four times
- Caught a misstatement ("dropped to" → "reached a low of")
- Verified every cited number against the slice

The result is a meaningfully better answer than yesterday's non-streaming
runs — traces an arc instead of listing numbers, uses careful language,
respects what the data does and doesn't show.

**Implication for Phase 1:** thinking traces are evaluable. The thinking
content itself is potentially scoreable (does the model reason against
the prompt? does it self-correct? does it verify?), independent of the
final answer. This adds a measurement axis that wasn't on the table
before today.

## 2026-05-02 — workout query succeeds with ADR-002 caps

**Wall-clock decomposition:**
- Dicer: 4.3s (warm).
- Extractor: 0.09s (coarsening included).
- Describer: 90s total, first content at 51s. Thinking-to-answer 8:1
  (4,836 chars thinking, 588 chars answer).

**Cascade behaviour (working as designed per ADR-002):**
- Extractor coarsened workouts from monthly (347 rows yesterday) to
  yearly (62 rows today). 4,460 raw sessions → 62 yearly aggregates.
- Cliff guard: 4,415 tokens. 5x below the 22K ceiling. No bundle-level
  truncation needed.
- Where yesterday wedged Ollama at 900% CPU for an hour, today
  produced a grounded answer in 94 seconds.

**Demand-signal surfacing (architectural claim evidence):**
The Describer named fencing with three specific year-volume datapoints
(2021: 6,502 min; 2024: 3,937 min; 2025: 6,225 min). This is exactly
the kind of personal-context-grounded answer a frontier model with no
data access cannot produce. First piece of evidence for the
demand-signal asymmetry hypothesis from a working cascade run.

**Two limitations exposed:**
1. The Describer claimed "data was truncated" — but the cliff cap
   produced *coarsening* (yearly aggregation), not truncation. Slices
   include `_workout_coarsened_to: yearly` annotation, but the
   describer_prompt doesn't yet teach the model the distinction.
   Update describer_prompt to handle coarsening as a different shape
   of partial data.
2. The Describer surfaced "a significant shift in 2025 toward
   functional strength training and mixed cardio." Genuine pattern
   or artefact (new source, classification change)? Worth verifying
   manually — it's the kind of plausible-sounding claim that exposes
   whether the cascade can distinguish behavioural shifts from
   measurement shifts. Phase 1 candidate measurement.

## 2026-05-02 — clarifying-question protocol working end-to-end

**Query:** "What were my best fitness years?"

**Outcome:** Dicer returned kind="question" with three concrete
disambiguating options (training volume, peak physiological metrics,
activity-type consistency). Each option is grounded in record types
present in dicer_view.json. Total cascade wall-clock: 3.7s. Describer
not called.

**Significance:**
- ADR-001's discriminated union (plan-or-question) is now exercised
  end-to-end. The Dicer's question branch was defensible on paper but
  unproven before today. Architecture now has full coverage.
- The clarifying question is well-formed: specific, options-not-prose,
  each option corresponds to a plan the Dicer could have produced.
  Better than I expected from e4b on a routing-quality task.
- Latency argument concretised: ambiguous query cost 3.7s. The
  Describer (90+s, much more compute) was correctly never invoked.
  This is the cascade's value proposition working — small fast model
  handles routing decisions that would have been wasted on the big
  model.

**Demand-signal note:**
The three options the Dicer offered are themselves a hint of the
moat. A frontier model without access to dicer_view would have to
ask much more generic clarifying questions ("what do you mean by
fitness?"). Because the Dicer sees what data actually exists, its
clarifications can be *concrete suggestions about what to compute*,
not abstract requests for definitions. Worth filing as a separate
demand-signal touchpoint: even the question-asking benefits from
local context.

## 2026-05-02 — clarification-of-clarification, stateless cascade

After the Dicer asked for clarification on "best fitness years", I
answered "By peak physiological metrics." The Dicer asked a *second*
clarifying question, requesting which metric (running speed, max HR,
VO2 max, or general summary).

Two findings:

**Finding 1: cascade is stateless.** Confirmed by inspection of
cascade.py — no turn history is passed between calls. The Dicer
received the fragment with no memory of its own prior question.
This is architecturally clean (deterministic Dicer, fully observable
traces, no hidden state) but means user fragments don't compose
across turns.

**Finding 2: the Dicer asked the right second question anyway.**
Without any context that "peak physiological metrics" was a follow-up,
e4b correctly identified that the noun phrase itself contains multiple
distinct metrics and asked which one. All four offered options grounded
in real record types in dicer_view.

Cost of a clarification chain so far: 3.7s + 4.1s = 7.8s.
Cost of the alternative (Describer routing on bad context, then
producing a wrong answer): would have been 90s+ minimum.

The architecture is doing the right thing; the *user contract* is
the design question. Two framings:

  A) Add turn-history to make follow-ups work.
  B) Keep Dicer stateless; orchestrator fuses follow-up fragments
     with prior question context before calling Dicer.

Phase 1 decision. B preserves architectural cleanliness; A matches
user expectation. For Phase 0, statelessness is the right default —
users restate fully on retry, ambiguity reduction is observable.
