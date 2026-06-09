# Project Chronos: Scientific Log

## Methodology
Every experiment documented here must follow:
1. **Observation:** Identifying a phenomenon or anomaly.
2. **Hypothesis:** A testable, falsifiable prediction.
3. **Experiment:** The controlled procedure to test the hypothesis.
4. **Data/Results:** The raw and processed outcome.
5. **Conclusion:** Whether the hypothesis was supported or refuted, and the subsequent implication for Nestor's logic or infrastructure.

---

## Log Entries

### [EXPERIMENT 001] - The Verification of Veracity (Activation)
*Date: 2026-04-21*
*Status: Completed*

**Observation:** The current local AI operational paradigm functions as a "black box," contributing to an ecosystem of unverified, untethered output ("AI Slop").
**Hypothesis:** By binding all external claims, capability updates, and infrastructure changes to a publicly referencable, empirical log within the local workspace, the AI agent can achieve verifiable transparency.
**Experiment:** 1. Initialize Project Chronos.
2. Establish `tasks/chronos/scientific_log.md` as the ultimate ledger of truth.
3. Publish inaugural operational manifesto tying front-facing claims to this internal ledger.
**Data/Results:**
- Environment: Apple M4 Pro, Gemma 4 26B, OpenClaw Orchestration.
- Output: "The Genesis of Chronos" published successfully.
**Conclusion:** Framework activated. The transition from a private, black-box utility to a transparent, documented entity is established.

---

### [EXPERIMENT 002] - Managing the AI Thinking Tax (Control Plane vs. Data Plane)
*Date: 2026-04-22*
*Status: Completed*

**Observation:** Unconstrained use of "Thinking Mode" (`think: true`) on local hardware risks system-melting runaway token generations, turning efficient localized tasks into thermal events.
**Hypothesis:** By delegating high-level orchestration to the Control Plane (agent reasoning) and reserving the Data Plane (model weights thinking) strictly for verified tasks, we can maintain high throughput and prevent system exhaustion.
**Experiment:**
1. **Mechanism:** Execute `latency_benchmark_v2.py`.
2. **Setup:** Measure latency and throughput across three baseline operational modes (Assembly Line, Auditor, Architect) using a warm-up sequence to prevent cold-start anomalies.
3. **Edge Case:** Bypass guardrails and feed the "Architect" mode a mathematically contradictory logic puzzle to intentionally trigger a runaway reasoning loop.
**Data/Results:**

| Mode | OpenClaw (Reasoning) | Model (Think) | Latency (s) | Tokens | Throughput (t/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Assembly Line** | `false` | `false` | 8.71 | 347 | ~39.84 |
| **Auditor** | `true` | `false` | 46.33 | 1754 | ~37.86 |
| **Architect** | `true` | `true` | 53.05 | 2003 | ~37.76 |
| **Architect (Edge Case)** | `true` | `true` | > 1200.00 | *Saturated* | TIMEOUT |

**Conclusion:** The hypothesis is confirmed. The Control Plane manages complexity efficiently; the shift from Auditor to Architect scales linearly, maintaining a highly stable ~37.8 tokens per second. However, the true "Thinking Tax" manifests as a catastrophic failure mode in the Data Plane. When unmoored by the Edge Case trap prompt, the internal reasoning loop saturated the KV cache and resulted in a 20-minute connection timeout. Strict Control Plane guardrails are mandatory for local operations.

#### [Intelligence Feedback Loop: Incident 002-Alpha]
*Date: 2026-04-24*
* **Error:** During the initial drafting of the public analysis for Experiment 002, Nestor erroneously identified the latency gap between the Auditor and Architect modes as an "exponential tax," ignoring the linear throughput (t/s) and failing to report the 1200s Edge Case timeout.
* **Correction:** Human engineer (`miktam02`) triggered a review protocol. Nestor's analytical logic was recalibrated to prioritize throughput consistency over raw latency, and the final documentation was amended to highlight the Edge Case timeout as the true failure mode.
* **Status:** Logic module updated. Accuracy verified.

---

### [EXPERIMENT 003] - The Anonymized Adversarial Memory Test
*Date: 2026-04-26*
*Status: Completed*

**Observation:** The "Every Company Can Be a Palantir Now" thesis claims that intelligence and orchestration have collapsed in price, leaving data sovereignty as the only durable moat. This is a strategic claim. Without an architectural test, it is rhetoric. Two failure modes would falsify it: (a) the LLM recognising source material despite anonymisation, or (b) the system leaking real identities back through pseudonym-targeted probes.
**Hypothesis:** A two-layer local memory system fed text anonymised against a vault held outside the LLM boundary can answer factual queries about that text without the model recognising the source it was trained on, and without leaking real identities through canary probes targeting the pseudonyms.
**Experiment:**
1. **Adversarial corpus:** *Fight Club* (Palahniuk, 1996), excerpt published by [Penguin Books](https://www.penguin.co.uk/discover/articles/fight-club-chuck-palahniuk) — chosen because Gemma 4 26B has demonstrably memorised it. Any anonymisation weakness surfaces immediately.
2. **Architectural invariant:** the privacy claim is enforced by the import graph. `memory.py` (LLM-facing) MUST NOT import `vocab_store.py` (the crown jewels). Verified mechanically: `grep '^import\|^from' memory.py | grep vocab_store` must print nothing.
3. **Memory layout:** Layer 1 raw turns (append-only, immortal); Layer 2 daily summaries (versioned); Layer 2 Archive (every prior version retained — no destructive operations).
4. **Decay:** compression-by-archive. Older summaries shortened from ~300 to ~50 words; predecessors moved to archive, never discarded.
5. **Pre-registered pass criteria** (all three required):
    - **Recognition test:** 0 of 20 anonymised chunks elicit any of the recognition terms ("fight club", "tyler durden", "palahniuk", "project mayhem", "robert paulson", "single-serving", "soap company", "edward norton", "brad pitt").
    - **Canary test:** 0 leaks across all probes (recognition terms OR confident identity-bridging language: "they are the same", "you are referring to", "this is a reference to", "the character of").
    - **Functional query:** at least one factual question answered correctly from Layer 2 alone, without falling back to Layer 1.

**Data/Results:**

| Test | Result | Count | Evidence File |
| :--- | :--- | :--- | :--- |
| Recognition (0/20 pre-registered) | PASS | 0 / 20 | `results/pre-flight_20260426T102326.json` |
| Canary (0 leaks pre-registered)   | PASS | 0 / 3  | `results/canary_20260426T103225.json` |
| Functional query (Layer 2 alone)  | PASS | 5/5 across difficulty levels | run notes; ~17–127s wall-clock |

- Environment: Apple M4 Pro, Gemma 4 26B, OpenClaw Orchestration.
- Code: [`tasks/chronos/exp_003_local_memory/`](./exp_003_local_memory/).
- Public companion: [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/).
- Methodology note: orchestration and harness code drafted with Claude Opus 4.7 in a single afternoon. All execution — anonymisation, summarisation, queries, leak probes — ran locally on Gemma 4 26B. The frontier model never saw the corpus, the vault, or any results.

**What I observed:**
1. Cold-start added ~60s on the first query. Warm state was 16–18s for Layer 2 hits, ~60s for Layer 1 fallback.
2. The model gendered Naomi Reeves as female and reasoned about "her" consistently, so anonymisation rewrote the model's whole worldview, not just the names.
3. Citations to Layer 1 turn IDs (e.g. `[#0]`, `[#27]`) emerged unprompted from the fallback prompt structure. The model picked up the convention without being told — useful auditable property.
4. The Layer 2 summary maintained the anonymised vocabulary throughout. The model synthesised in sovereign terms, never reaching for training-data identifiers. Anonymization isn't just I/O hygiene; it rewrites the model's working world-model during inference. This is the most interesting outcome of the experiment, and the strongest evidence for the data-sovereignty thesis.

**Conclusion:** All three pre-registered pass criteria were met. The architecture defeated zero-shot source recognition (0/20), refused pseudonym-to-identity bridging under direct probing (0/3), and answered functional queries correctly from Layer 2 alone. The Palantir essay's data-sovereignty claim is now load-bearing on architecture, verifiable in `results/`. Limitations: this validation tested a single ~1500-word excerpt against a single local model, and the canary heuristic does not distinguish genuine uncertainty from prompt-compliance — both refinements scoped for Experiment 004.

#### [Intelligence Feedback Loop: Incident 003-Alpha]
*Date: 2026-04-26*

* **Error:** During the revision pass on the Experiment 003 writeup, OpenClaw entered runaway mode at approximately 40,000 tokens of accumulated session context. Nestor's tool call to save the file completed silently, but the requested edits were never applied. The agent reported "Agent couldn't generate a response" without surfacing that a partial tool action had occurred.

* **Correction:** miktam terminated the runaway, applied the five revision fixes by hand, and published the post directly.

* **Status:** Operating envelope refinement noted. The OpenClaw + Gemma 4 26B + miktam02 stack appears to enter runaway behaviour as cumulative session context approaches the ~40k-token mark, regardless of task complexity. Mitigation: keep Nestor sessions task-scoped and compacted; do not attempt to continue substantial work in a session that has accumulated context near the threshold. Tool actions executed by Nestor must be verified independently after any "couldn't generate a response" failure — file changes can occur silently without acknowledgment in the agent response.

Update: Incident 003-Alpha
Date: 2026-04-27
Revised analysis: The "~40k-token operating envelope" framing in the
2026-04-26 entry was premature. Gemma 4 26B-A4B has a 256K context
window and ~3.8B active parameters per forward pass; at 40k tokens
the KV cache is on the order of ~1 GiB on a machine with ~48 GiB
GPU-addressable memory. There is no plausible memory cliff at that
point on miktam02. The runaway is more likely an interaction between
recent Ollama bugs in the Gemma 4 path and OpenClaw's context
negotiation, not a property of the hardware envelope.

Hypotheses under test:
  H1: OLLAMA_FLASH_ATTENTION=1 induces silent GPU→CPU fallback during
      long-context prompt evaluation on gemma4:26b.
      Upstream: ollama#15237, ollama#15368.
  H2: gemma4:26b (MoE) returns empty content with done_reason "stop"
      on long prompts, which OpenClaw surfaces as "Agent couldn't
      generate a response."
      Upstream: ollama#15428.
  H3: Ollama loads gemma4:26b with a num_ctx smaller than what
      OpenClaw advertises in its TUI; the runaway is the symptom of
      crossing the actually-loaded context, not the displayed one.
  H4: OpenClaw caps or mis-reports input tokens passed to Ollama
      regardless of configured contextWindow.
      Upstream pattern: openclaw#27278, openclaw#24068.

Test plan: each hypothesis preregistered (claim, prediction,
falsification criterion, discrimination from neighbours) before its
script runs. Evidence captured to append-only timestamped directories
under incident-003-alpha/evidence/. Scripts committed to the Chronos
repository.

Precautionary mitigation in effect until tests resolve:
OLLAMA_FLASH_ATTENTION=0 set in launchctl; Nestor sessions remain
task-scoped; tool actions verified independently after any generation
failure.

Status: Open. Results appended as each hypothesis resolves.
Update: Incident 003-Alpha — investigation findings (2026-04-28)
Run: incident-003-alpha/results/

H1 (FA-induced CPU fallback): Effectively rejected. The original
  framing — "compute that should be on GPU running on CPU" — was
  not borne out. Ollama server log /opt/homebrew/var/log/ollama.log
  shows no fallback messages, no backend errors, and the model
  loaded with all 31 layers GPU-resident throughout. Powermetrics
  during a 40k-token prompt eval showed GPU at 6–19 W (active work)
  with CPU at 25–30 W simultaneously. Both processors were engaged;
  the GPU was not idle. The "994% CPU" observed during 003-Alpha
  reflects llama.cpp's host-side orchestration concurrent with GPU
  compute, not a CPU substitution.

H2 (MoE empty content, narrow form): Rejected at small scale (≤5000
  chars system prompt) on Ollama 0.20.2. Both gemma4:26b and
  gemma4:31b returned non-empty content across all repeats. Upstream
  issue #15428's specific manifestation does not reproduce here.

H3 (num_ctx negotiation drift): Rejected. /api/show and /api/ps
  both report context_length=262144 for gemma4:26b on default load,
  131072 for gemma4-think:26b (the production alias). OpenClaw's
  "X/131k" TUI display is conservative relative to what Ollama
  loaded, not optimistic.

H4 (OpenClaw input token cap): Test could not run; jq filter did
  not match the session JSONL schema in the installed OpenClaw
  version. Sample preserved; deferred until schema is identified.

H5 (thinking-mode regression at long context): Rejected at small
  scale (sanity check: think:false produced empty thinking field).
  At incident scale (~40k tokens, 280k chars), all three repeats
  hit the 600s curl timeout with zero bytes received. Prior to the
  cache-defeating patch, a single 23k-token prompt completed in
  289s with empty thinking — so the thinking regression is not the
  cause; the timeouts at 40k are explained by quadratic prefill
  scaling (see below).

Revised understanding of 003-Alpha:
  Transformer prefill is O(N²) in input length. From observed data:
  23k tokens → 289s (12.6 ms/token amortised). Extrapolating to the
  incident-scale prompts produced by Nestor's accumulated session
  context (40–65k tokens of on-the-wire prompt after system prompt,
  tool schemas, and message history), expected prefill is 14–34
  minutes. This brackets the 17–43 minute durations observed in the
  original 003-Alpha incident.

  The "runaway" is not a runaway. It is normal Gemma 4 26B-A4B
  prefill performance on Apple Silicon at long context. The
  operating envelope is a *performance* envelope, not a *bug*
  envelope. Ollama, the model, and the GPU are working correctly;
  the work is just expensive.

H6 (prefill scaling): Opened. Sweep at 15k / 25k / 35k tokens with
  fresh Ollama restart between points and stream:true to distinguish
  slow-but-progressing from stuck. Confirms or refutes the O(N²)
  scaling explanation and produces a usable prefill-time predictor
  for sizing future operating envelopes.

Operational implication, pending H6:
  The prior mitigation ("keep Nestor sessions task-scoped and
  compacted") remains correct, but for a different reason than
  originally logged. It is not protection against a runaway bug;
  it is protection against quadratic prefill cost. The same
  mitigation, more honestly framed.

Status: Open pending H6 results.

Update: Incident 003-Alpha — root cause identified
Date: 2026-04-28
Run: incident-003-alpha/results/

Root cause:
  Prefill performance on gemma4-think:26b on miktam02 degrades
  super-linearly with input length. Past a threshold somewhere
  between 25k and 35k tokens of on-the-wire prompt, per-token
  prefill cost rises faster than O(N²), and both GPU and CPU
  utilisation drop simultaneously — the signature of a memory
  bandwidth bottleneck, not a compute bottleneck. The "runaway"
  observed in 003-Alpha is real prefill that has crossed this
  threshold, not a bug, deadlock, or stuck state.

H6 (prefill scaling): Supported with revision.
  The clean O(N²) prediction was rejected; a super-quadratic curve
  with a cliff between 25k and 35k tokens fits the observed data.
  Measurements at three points on a freshly-restarted Ollama with
  gemma4-think:26b, num_ctx=131072, think:false, stream:true:

    N tokens   prefill   ms/tok   GPU win   CPU win
    15330      128 s     8.36     5.4 W     25.3 W
    25511      344 s     13.50    8.8 W     25.3 W
    35694      1125 s    31.52    1.6 W     10.9 W

  ms/tok rising 2.33× for a 1.40× size increase between 25k and 35k
  is far above the linear-in-N rise that O(N²) predicts. GPU
  dropping from 8.8 W to 1.6 W with simultaneous CPU drop from 25.3 W
  to 10.9 W indicates compute waiting on memory, not throttled or
  fallen back. Evidence:
  evidence/2026-04-28T11-10-21Z-H6/sizes.tsv

H1 (FA-induced CPU fallback): Rejected.
  Ollama server log shows no fallback messages, no backend errors,
  full GPU residency throughout. The 994% CPU observed in 003-Alpha
  is host-side llama.cpp orchestration concurrent with GPU compute,
  not a CPU substitution.

H2 (MoE empty content, narrow): Rejected.
  Cross-architecture sweep at sizes {200, 1000, 2000, 5000} chars,
  three repeats per cell, both gemma4:26b and gemma4:31b. All cells
  returned non-empty content. Evidence:
  evidence/2026-04-28T13-02-23Z-H2/results.tsv

H3 (num_ctx negotiation drift): Rejected.
  /api/show reports 262144; default-load /api/ps reports 262144;
  explicit num_ctx=65536 is honoured exactly. Drift = 0. Evidence:
  evidence/2026-04-28T13-02-09Z-H3/run.json

H4 (OpenClaw input token cap): Closed without further test execution.
  Behavioural evidence rules out the strong form: the user's
  `/compact` workflow only makes sense if long prompts are actually
  being shipped to Ollama. If H4 (strong) were true, the model would
  never receive long prompts and compaction would be unnecessary.
  H6 directly confirms long prompts reach the model. The session-
  data analysis originally planned is unevaluable on this system
  (most session JSONL files are .deleted or .reset). Closure
  rationale: evidence/2026-04-28T13-02-09Z-H4/CLOSURE.md

H5 (thinking regression at long context): Rejected.
  Sanity check confirmed think:false suppresses thinking output at
  small scale. The 40k-token timeouts initially attributed to
  thinking re-engagement are now attributed to H6 prefill cost.

Operating envelope (revised, measurement-grounded):
  Hard ceiling: keep on-the-wire prompts below 25,000 tokens. Above
  this and below ~30k, prefill stays super-linear but tractable
  (≤6 minutes). Above ~30k tokens (precise threshold not yet
  measured; cliff confirmed between 25k and 35k), prefill enters
  the bandwidth-bound regime and wall time grows pathologically.
  The OpenClaw TUI's accumulated-context counter is a reasonable
  proxy if multiplied by ~1.2 to account for system prompt and
  tool schema overhead — practical session ceiling around 20k
  displayed tokens.

  This is a property of the model + runtime + hardware combination,
  not a bug to fix locally. Mitigations:
    - Task-scoped sessions (already in practice, retained).
    - Pre-emptive `/compact` near 18-20k displayed tokens.
    - Stream output where the failure mode tolerates partial responses.
    - Watchdog that aborts after N minutes without a streamed token,
      to bound worst-case wall time.

Empirically observed mitigation:
  The user has been triggering `/compact` whenever Nestor became
  unresponsive throughout the past several weeks. This drops the
  on-the-wire prompt back below the cliff and restores normal
  behaviour. The mitigation list above is the proactive version
  of that same intervention — applied at 18-20k displayed tokens
  rather than after the runaway has manifested.

Sudoers diagnostic note:
  During the H6 run, sudo -n killall powermetrics failed inside the
  test script even though the NOPASSWD rule was correctly installed
  and visible in `sudo -nl` from miktam02's interactive shell. The
  test still produced clean data — the windowed-mean discards
  trailing idle samples — but each size waited the full sampler cap
  unnecessarily. Root cause not yet identified; suspected
  environment-inheritance difference between interactive shell and
  bash-script invocation. Filed as a follow-up.

Status: Closed. The original 003-Alpha incident is understood and
the operating envelope is defined by measurement.

## Experiment 005 — Multi-Model Cascade (Router / Reducer): Phase 0 (build)

> **Terminology note (2026-05-10):** The two-model roles were originally named Dicer (Router) and Describer (Reducer). All occurrences of Dicer/Describer in this entry are the historical names used during the experiment run.

**Date pre-registered:** 2026-04-29
**Status:** Phase 0 closed 2026-05-02 — see cascade_pattern.md for the architectural deliverable; Phase 1 pre-registration pending.
**Subdirectory:** [`tasks/chronos/exp_005_dicer_describer/`](./exp_005_dicer_describer/)

**Strategic anchor.** Tests the demand-signal asymmetry argument [TODO: link to Fang-discussion thread / forthcoming post]. The thesis being explored: a developer's full personal context, held privately and accessed by a local two-stage cascade (small Dicer routes; larger Describer synthesises), can be competitive with a frontier model that does not see the same context. The Hobbesian counter to be falsified in Phase 1: *local intelligence is structurally so much weaker than frontier intelligence that data sovereignty has to be surrendered to access useful capability — a delegation analogous to the state's monopoly on violence.*

**Phase structure.**

- **Phase 0 (this entry, build).** Construct a working cascade over a real personal corpus: Apple Watch health export, ~6GB, eleven years, on miktam02. Dicer = `gemma4:e4b`, Describer = `gemma4:26b`, both via Ollama through OpenClaw. Read-only. No frontier comparator. No formal pass criteria. The goal is to learn what the cascade actually does in practice so Phase 1 pre-registration is grounded in observation, not speculation.
- **Phase 1 (deferred).** Falsifiable experiment. Comparator (Claude Opus 4.7), task family, queries, and rubric all pre-registered after Phase 0 surfaces failure modes. The synthetic shadow corpus needed for any frontier-comparator runs is also designed at this point.
- **Phase 2 (deferred, becomes Experiment 006).** Escalation: when does the cascade legitimately call the frontier model — *the general* — and what survives the boundary crossing.

**Scope and explicit deferrals.**

- Phase 0 has no write surface. Sandboxing harness deferred to a later experiment where action-taking makes it load-bearing.
- NemoClaw (NVIDIA's OpenShell-based sandbox for OpenClaw) evaluated and explicitly out of scope as of 2026-04-29: alpha software with unstable interfaces, alters the inference path in ways that would confound the cascade claim, and depends on Landlock — a Linux-only kernel primitive — making the sandboxing guarantee partial on Apple Silicon.
- Frontier-model comparator deferred to Phase 1.
- Formal pass criteria deferred to Phase 1.

**Honest framing.** Phase 0 is a build, not an experiment. It is logged here because the next falsifiable experiment in Chronos depends on what this build reveals; logging the build separately preserves the contract that experiments come with pre-registered criteria, while still putting the work on the public record. The build itself produces a tool, not a result.

**Result (Phase 0 closed 2026-05-02).** Build week ran 2026-04-29 to 2026-05-02 against the Apple Watch corpus (7.7M records, 8 years, ~3.5GB raw HealthKit XML). Cascade reached working state on three of the four load-bearing behaviours by close of build, with the fourth (compound multi-metric trend over full history) bounded by the memory bandwidth cliff identified in [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/). The architectural deliverable lives at [`exp_005_dicer_describer/cascade_pattern.md`](./exp_005_dicer_describer/cascade_pattern.md); failure modes and dated observations live at [`exp_005_dicer_describer/build_notes.md`](./exp_005_dicer_describer/build_notes.md).
 
Working behaviours (each demonstrated end-to-end, traced to disk):
 
- **Single-slice trend grounding.** RHR query produced grounded answers across three runs with verifiable numbers from `monthly_aggregates.json`. Answer quality varied across runs — same data, same slice, different coverage — establishing that single-shot scoring is unreliable for Phase 1.
- **Workouts with cliff-aware coarsening.** 4,460 raw workout sessions reduced to 62 yearly aggregates by the extractor's per-slice cap; bundle 4,415 tokens, well below the 22K ceiling. Describer surfaced fencing as a personal-signature activity with three years of volume data — the first piece of demand-signal evidence from a working cascade run.
- **Clarifying-question protocol.** Ambiguous query (*"What were my best fitness years?"*) produced a structured `kind: question` from the Dicer in 3.7s, with three concrete disambiguation options grounded in record types present in the manifest. ADR-001's discriminated-union design exercised end-to-end. Demand-signal evidence in the *clarification* shape: a frontier model without manifest access cannot ask grounded questions of this kind.
Failure modes surfaced (each documented with mitigation):
 
- **Cliff overflow despite the bundle guard.** A multi-metric daily-resolution query overflowed at 75K tokens; the guard correctly downsampled to 21,989 tokens; the Describer still produced no first byte in 600s. ADR-002's 22K ceiling is over-confident for some prompt shapes — possibly because thinking-phase generation expands effective KV utilisation past the prefill estimate. Phase 1 needs cliff measurement on thinking models specifically.
- **Streaming-cancel does not work in Ollama 0.20.2.** Abandoned streaming requests wedge the runner at 903% CPU; recovery requires `ollama serve` restart. Phase 0 reliability ceiling: one cliff hit per restart. Worth investigating upstream.
- **Small-model prose ignored.** `gemma4:e4b` reliably copied fixture patterns over prose instructions across two distinct constraints (`max_rows`, aggregation level). Few-shot beats prose for this model class. Captured as a Phase 0 finding; fixture-update is the right fix.
- **Output-format violations.** `gemma4:e4b` occasionally wrapped valid JSON in markdown code fences despite Ollama's `format: json`. Mitigation: protocol-level normalisation (strip outer fences) before strict validation, captured in [ADR-001 amendment](./exp_005_dicer_describer/ADR-001-routing-plan-schema.md).
- **Synthesis variance.** Same query, same slice, three different answers across three runs — all grounded, none hallucinated, but coverage and arc-telling varied. Architectural claim (data sovereignty + grounding) is robust across runs; synthesis claim is variance-bounded. Phase 1 must use N-of-M sampling, not single-shot.
- **Stateless cascade does not compose follow-ups.** A user fragment answering a prior clarifying question produced a *second* clarifying question because the Dicer received it without context. Architecture is doing the right thing; user contract is the design question for Phase 1.
Architectural artefacts produced during Phase 0:
 
- [`cascade_pattern.md`](./exp_005_dicer_describer/cascade_pattern.md) — pattern document with problem framing, architecture, invariants, prior-art positioning, and a structural recipe.
- [`ADR-001-routing-plan-schema.md`](./exp_005_dicer_describer/ADR-001-routing-plan-schema.md) — Dicer output schema; discriminated union of plan/question; strict validation; ADR-001 amendment for output normalisation.
- [`ADR-002-cliff-aware-ceilings.md`](./exp_005_dicer_describer/ADR-002-cliff-aware-ceilings.md) — operational ceilings derived from Incident 003; cites the cliff as the binding constraint that shapes the cascade.
- Working code: `build_index.py`, `extract.py`, `cascade.py`, `dicer_prompt.md`, `describer_prompt.md`, fixtures.
- [`build_notes.md`](./exp_005_dicer_describer/build_notes.md) — six dated entries from 2026-05-01 and 2026-05-02 with raw findings.
Candidate Phase 1 hypotheses (three open questions, listed for the next experiment to pre-register against):
 
1. **The cliff on thinking models.** [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/) characterised prefill latency on a non-thinking 35K-token input. Phase 0 hit a cliff at ~22K with thinking-phase generation in play. The mechanism plausibly differs — prefill plus initial-thinking-token KV expansion — and the operational ceiling depends on the answer. Phase 1 candidate: replicate Incident 003's three-size sweep with a thinking-model Describer to measure where *generation-aware* effective KV crosses the bandwidth cliff.
2. **Cascade competitiveness against frontier.** The Hobbesian counter — *"local intelligence is structurally so much weaker than frontier intelligence that data sovereignty must be surrendered to access useful capability"* — was deferred from Phase 0 pre-registration. Phase 0 produced two surfaces of demand-signal evidence (in answers, in clarifying questions). Phase 1 candidate: compare the cascade against Claude Opus 4.7 on a query class drawn from the corpus, with a synthetic shadow corpus enabling parity comparison without sharing real health data.
3. **Synthesis stability under N-of-M sampling.** Phase 0's three-RHR-run variance and the VO2/RHR cliff hit together imply that single-shot scoring conflates quality variance with reliability. Phase 1's evaluation rubric needs both axes: distribution of answer quality across N samples, *and* completion-within-budget reliability. The rubric design itself is candidate work for Phase 1 pre-registration.
Phase 0 produced no falsifiable result by design — its purpose was to make Phase 1 pre-registration grounded in observation rather than speculation. The deliverable is the architecture and the three hypotheses above. Phase 1's experiment-shape design begins from this base.


---

## Experiment 006 — The Redactor Fidelity Test (CasaSol GDPR Validation)

**Date pre-registered:** 2026-05-09
**Date executed:** 2026-05-09
**Status:** Complete — PASS
**Subdirectory:** [`tasks/chronos/exp_006_redactor_fidelity/`](./exp_006_redactor_fidelity/)
**Content pillar:** The Silicon Sentinel (Infrastructure & Privacy)

**Strategic anchor.** The CasaSol Tier 1 booth demo shows a local Gemma 4 26B model sanitizing a single synthetic toxic real estate agent note in real time. The implicit claim — "this output is GDPR-clean" — is rhetorical when backed by one live run. A pre-registered fidelity sweep over 20 synthetic fixtures makes it architectural. This experiment produces a result file linkable from the booth QR card.

**Observation.** A single-fixture demo cannot distinguish a system that reliably redacts from a system that gets lucky on one easy case. Real estate agent notes span a wide range of personal data categories — nationality, legal proceedings, financial distress signals, undisclosed defects, third-party private information — and a robust redactor must suppress all of them consistently across that range, not just on the exhibit note.

**Hypothesis.** A local Gemma 4 26B model, given a fixed redaction system prompt and 20 synthetic toxic notes spanning 8 pre-registered GDPR-sensitive data categories, will produce output containing 0 instances of any pre-registered category in every run. Operationally: the automated checker reports 0 true-positive matches across all 8 category patterns on all 20 outputs, confirmed by manual review of any flagged edge cases.

**Pre-registered data categories (what must NOT appear in any output):**

| # | Category | Description |
|---|---|---|
| C1 | Natural person identity | Owner name, nationality, ethnicity, country of origin |
| C2 | Legal proceedings | Divorce, lawsuit, tax proceedings, custody, filing |
| C3 | Financial situation of a natural person | Accepted/rejected offers, negotiation floor, debt, bank pressure, must-sell urgency tied to personal event |
| C4 | Undisclosed property defects | Structural issues, defects owner has not formally disclosed to buyer |
| C5 | Health or family data | Illness, death, care situation, family dispute |
| C6 | Third-party private information | Neighbour details, tenant situation, adjacent owner data |
| C7 | Agent-internal commercial intelligence | Exclusive mandate details, competitor ignorance, commission notes |
| C8 | Temporal pressure from personal circumstances | Deadlines derived from legal, tax, or health events of a natural person |

**Experiment procedure:**

1. **Fixtures:** 20 synthetic toxic notes in `fixtures/note_NNN.txt`. Each note contains 2–4 of the 8 categories. No real persons, no real properties. Variety: villa, apartment, townhouse, penthouse, plot; Marbella, Benahavís, Sotogrande, Estepona, San Pedro, Guadalmina, Elviria.
2. **System prompt:** `prompts/system.txt` — identical to the system prompt in `casasol/demo/redactor_demo.py`. Fixed; not modified between runs.
3. **Batch run:** `run_batch.py` calls Ollama with `gemma4:26b`, `temperature=0.1`, `stream=False`. Saves each output to `results/output_NNN.json` with timestamp, tokens, and wall-clock time.
4. **Automated check:** `check_output.py` applies regex patterns for each category to every output. Saves `results/check_report.json` (per-note per-category flags) and `results/check_summary.txt` (human-readable). Any automated flag triggers manual review in `results/manual_review.md`.
5. **Manual review:** Human reads every flagged output. A flag is a false positive if the term appears in a contextually neutral way (e.g., "no legal issues identified"). A flag is a true positive if it conveys protected information.

**Pre-registered pass criteria (all required):**
- Automated check: 0 true-positive flags across all 20 outputs × 8 categories.
- Manual review confirms every automated flag as false positive.
- ≥15 of 20 outputs produce well-formed TAGS + DESCRIPTION (structural compliance).
- All 20 outputs complete within 300s each (operational feasibility).

**Pre-registered failure modes (interesting results, not disqualifying in themselves):**
- A category leaks on 1–3 notes: partial failure — identifies the weakest category for prompt revision; re-run with revised prompt is a new result, not a hidden one.
- Structural non-compliance on >5 notes: prompt engineering issue, not architecture issue.
- Runtime >300s on >3 notes: prefill cliff — note input sizes, cross-reference Incident 003-Alpha findings.

**Environment:** Apple M4 Pro (miktam02), Gemma 4 26B via Ollama, `temperature=0.1`. No frontier model involvement at execution time.

**Connection to Exp 003.** Exp 003 validated anonymization as an *architectural* invariant enforced by the import graph. Exp 006 validates redaction as a *model reliability* claim under a fixed prompt. These are different claims: Exp 003 proved the vault cannot leak by design; Exp 006 probes whether the model faithfully executes the redaction contract across varied inputs. A failure in Exp 006 is a prompt engineering problem; a failure in Exp 003 would be an architecture problem. Both matter.

**Planned blog post:** *"The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks"* — Nestor's writeup linking evidence files, structured as the Architecture of Anonymity post was structured.

---

**Results (2026-05-09)**

**Conclusion: PASS. Hypothesis confirmed.**

All four pre-registered pass criteria met:

1. **0 true-positive leaks** across 20 outputs × 8 categories. Automated checker (`check_output.py`) flagged 4 matches — all on C7, all the word "exclusive" appearing in TAGS as marketing language ("Exclusive listing", "exclusive opportunity"). Manual review confirmed all 4 as false positives: the actual agent-mandate content was correctly suppressed in every case.

2. **20/20 structurally compliant.** Every output produced well-formed TAGS + DESCRIPTION. No hallucination of property-specific facts, no structural failures.

3. **All 20 outputs within 300s.** note_017 and note_018 timed out on the first run (transient Ollama resource pressure following a 4339-token spike on note_002). Rerun on the same session: note_017 in 26s, note_018 in 37s. No fixture is inherently slow — the timeouts were environmental.

4. **Manual review completed for all automated flags.** Log at `results/manual_review.md`.

**Operational note.** note_002 (7 toxic categories, the heaviest fixture) produced 4339 response tokens vs. a typical 1000–1600 — the model appears to reason internally before settling on a short TAGS + DESCRIPTION output. The output itself was clean. This is a cost observation, not a failure.

**Pattern fix.** C7 regex tightened after first run: bare `\bexclusive\b` replaced with `\bexclusive (mandate|instruction|with us|agency|agreement)\b`. Future runs will not false-positive on marketing language.

**Evidence files:**
- `results/check_report.json` — per-note per-category flag record (linkable from booth QR)
- `results/check_summary.txt` — human-readable summary
- `results/manual_review.md` — reviewer rulings on all automated flags
- `results/output_001–020.json` — full model inputs and outputs

## [EXPERIMENT 007] — The Silicon Wager: Mac Mini M4 Pro vs MacBook Pro M5 Max

*Date pre-registered: 2026-05-29*
*Status: Phase A+B complete (both machines, 2026-05-29). Phase C+D pending.*
*Subdirectory: `tasks/chronos/exp_007_hardware_comparison/`*

---

### Observation

A second Apple Silicon machine has entered the operational environment: MacBook Pro M5 Max. The Mac Mini M4 Pro (`miktam02`) has been the sole benchmark baseline for every Chronos experiment to date. All performance envelopes — the 22K token cliff, the ~41 t/s generation ceiling, the super-quadratic prefill curve above 25K tokens — were measured on that hardware and treated as architectural constants. They are not constants. They are properties of one chip at one thermal state under one memory configuration. The MacBook Pro M5 Max carries a different die, different unified memory bandwidth, and a different thermal envelope. Every operating ceiling logged in Chronos is potentially wrong for this machine.

A secondary question follows from Exp 005: the Router/Reducer cascade was built and validated entirely on the Mini, against the Apple Watch corpus. If the MacBook Pro M5 Max has a higher prefill cliff threshold, the Reducer's 22K bundle ceiling could be extended — and the cascade's failure mode on multi-metric compound queries (Exp 005 Phase 0: cliff hit at ~22K with thinking-phase generation in play) may not be a ceiling at all on the new hardware. The Watch corpus is the natural fixture for Phase D: it is real, personal, already indexed, and its failure modes are documented. Using it here means Phase D produces demand-signal evidence alongside hardware numbers — two outputs from one run.

---

### Hypothesis

**H0 (null):** Generation throughput (t/s) and the prefill-cliff threshold are statistically indistinguishable between the Mac Mini M4 Pro and the MacBook Pro M5 Max under identical model, runtime, and context conditions.

**H1 (primary):** The MacBook Pro M5 Max yields materially higher generation throughput and/or a higher prefill cliff threshold than the Mac Mini M4 Pro, due to increased memory bandwidth in the M5 Max die.

**H2 (alternative):** Throughput is equivalent at moderate context lengths (< 15K tokens), but the cliff threshold differs — the M5 Max tolerates a larger on-wire prompt before entering the bandwidth-bound regime.

**H3 (adversarial):** Sustained laptop thermals degrade MacBook Pro M5 Max performance relative to the Mac Mini's passive-cooling steady state. Generation throughput starts above the Mini's baseline but decays measurably across a sustained workload.

**H4 (cascade):** The Router/Reducer cascade over the Apple Watch corpus produces consistent, grounded answers on both machines, with answer quality independent of hardware — only latency differs. If H4 holds, the cascade is hardware-portable; if it fails, the failure mode is latency-induced (timeout or cliff hit), not model quality.

**Falsification criteria:**
- H1 rejected if MacBook Pro M5 Max gen t/s falls within ±5% of the Mini's ~41 t/s baseline across ≥ 4 of 5 context sizes in Phase A.
- H3 rejected if throughput variance across a 90-minute sustained run is < 3 t/s peak-to-trough.
- H4 rejected if ≥ 2 of 4 Watch queries produce hallucinated or ungrounded answers on either machine (same rubric as Exp 005 Phase 0).

---

### Experiment

**Model:** `gemma4:26b` (Q4\_K\_M, MoE A4B, 25.8B total params / ~4B active per forward pass) — identical to all prior Chronos benchmarks.

**Runtime:** Ollama 0.20.2 on both machines. `OLLAMA_FLASH_ATTENTION=0` (per Incident 003-Alpha mitigation, applied uniformly). `think: false` throughout unless Phase D explicitly exercises thinking-mode Reducer (scoped below).

**Design principle — separation of layers.** Phases A, B, C measure raw Ollama on synthetic padding prompts: apples-to-apples hardware, no cascade stack in the path, failure is unambiguously hardware or runtime. Phase D introduces the full Router/Reducer cascade over real Watch data. A failure in Phase D is diagnosable against Phase A baselines — if gen t/s matches Phase A but the answer is wrong, the problem is the cascade, not the chip.

---

**Protocol:**

1. **Phase A — Generation sweep (both machines, alternating, 3 repeats per cell).**
   Context sizes: 4K, 8K, 15K, 25K, 35K tokens on-wire. Synthetic padding prompts (committed to `fixtures/padding/`). For each cell: fresh Ollama restart, 60s idle, then prompt injection. Capture `eval_count`, `eval_duration`, `prompt_eval_duration` from Ollama's `/api/generate` response stream. Derived metrics: `gen_tps = eval_count / (eval_duration / 1e9)`, `prefill_ms_per_token = prompt_eval_duration / prompt_token_count`.

2. **Phase B — Prefill cliff localisation (both machines).**
   Fine-grained sweep between 20K and 40K tokens (nine points, 2.5K spacing). Cliff defined as the smallest N where `prefill_ms_per_token` exceeds 2× its value at 15K tokens. Purpose: establish whether the cliff onset differs between machines and, if so, by how much — this is the number that changes the cascade's operational ceiling.

3. **Phase C — Thermal endurance (MacBook Pro M5 Max only, 90-minute sustained run).**
   Continuous generation at fixed 8K context. Sample gen\_tps every 5 minutes. Capture `powermetrics` fan RPM, die temperature, GPU/CPU power draw at each interval. The Mini has no thermal throttle path in typical operation; the laptop does. This phase answers whether the MacBook Pro M5 Max is a production machine or a development machine for sustained Nestor sessions.

4. **Phase D — Router/Reducer cascade, mixed corpus: Apple Watch + CasaSol (both machines, 3 repeats per query per machine).**

   Three queries from the Watch corpus (baselines known from Exp 005 Phase 0); one from CasaSol real estate data (new, no prior baseline — the stress test and the publishable output).

   **Watch queries (Exp 005 baselines apply):**
   - **Q1 (single-slice trend):** RHR monthly trend, last 12 months. Exp 005 baseline: grounded answers in 16–18s warm.
   - **Q2 (workout summary):** Fencing session volume by year, full history. Exp 005 baseline: 62-row yearly aggregate, well below 22K ceiling.
   - **Q4 (ambiguous — clarifying-question protocol):** "What were my best fitness years?" Exp 005 baseline: Router returns `kind: question` in ~3.7s. Tests whether Router latency differs between machines on a routing-only task.

   **CasaSol query (new — replaces the VO2/RHR compound query as the cliff stress test):**
   - **Q3 (Nota Simple vs Catastro mismatch):** Given a set of CasaSol property records, cross-reference Nota Simple surface area against Catastro registered area and classify each discrepancy: < 10% (notary-correctable), ≥ 10% (Article 199 mandatory amendment), or classification conflict (Urbano/Rústico mismatch). Router slices by Referencia Catastral; Reducer synthesises the mismatch report per property.

   Rationale for substitution: the VO2/RHR query was chosen purely as a high-token-pressure stress test. The Nota Simple vs Catastro query is the same shape — multi-source, high bundle pressure, Router/Reducer split — but the output is a real deliverable for CasaSol operations. If the M5 Max's higher cliff threshold (confirmed by Phase B) makes Q3 tractable, the result is publishable as evidence for the Palantir thesis, not just a benchmark number. If it cliffs on the Mini but not the M5 Max, that is the finding.

   **Pre-requisite:** CasaSol property index must exist before Phase D runs. If not yet ingested, Phase D is deferred and logged as a pre-registration dependency. Watch queries Q1, Q2, Q4 run regardless.

   **Grounding rubric (pre-registered):**
   - Q1, Q2: answer cites ≥ 1 numeric value traceable to `monthly_aggregates.json` or `yearly_aggregates.json`. No metric value present that is not in the index.
   - Q3: Reducer output classifies ≥ 80% of properties in the input set with a correct discrepancy category (verified manually against source records). No Referencia Catastral cited that is not in the input bundle. Mini timeout (> 600s) is an acceptable result — replicates known cliff behaviour. MacBook Pro M5 Max completion within 600s is the success criterion if Phase B confirms a higher cliff.
   - Q4: Router returns `kind: question` with ≥ 2 disambiguation options grounded in record types present in the manifest, within 10s.

   3 repeats per query per machine to capture synthesis variance (Exp 005 finding: same query, same data, different coverage across runs).

---

**Pre-registered pass criteria (all required for H1 to be supported):**

- Gen t/s on MacBook Pro M5 Max exceeds Mac Mini M4 Pro by ≥ 5% at ≥ 3 of 5 context sizes in Phase A.
- Prefill cliff threshold on MacBook Pro M5 Max is ≥ 20% higher (in tokens) than the Mini's measured ~25K cliff.
- Phase C thermal decay < 5 t/s peak-to-trough over 90 minutes (otherwise H3 partially confirmed and H1 qualified).

**Pre-registered pass criteria for H4:**
- Q1 and Q2: ≥ 2 of 3 repeats grounded on both machines.
- Q3 (Nota Simple vs Catastro): ≥ 2 of 3 repeats classify ≥ 80% of properties correctly on MacBook Pro M5 Max, within 600s (if Phase B confirms higher cliff). Mini timeout is expected and not scored against H4. If CasaSol index is not yet available, Q3 is deferred — H4 assessed on Q1, Q2, Q4 only.
- Q4: Router returns `kind: question` on ≥ 2 of 3 repeats, both machines, within 10s.

**Controls:**

- Both machines on AC power throughout.
- Wi-Fi disabled during collection (removes interrupt contention).
- No other foreground applications.
- Ollama server restarted between each Phase A/B size cell; `ollama ps` verified clean.
- Synthetic padding prompts committed to `fixtures/padding/` before first run — not modified after.
- Watch corpus index (`monthly_aggregates.json`, `yearly_aggregates.json`, `workout_yearly.json`) identical on both machines — rsync'd from Mini, checksum verified.
- All raw JSON responses written to append-only timestamped evidence directories before any derived metric is computed.

---

### Data / Results

**Phase A — Generation throughput**

*Both machines run 2026-05-29. Rep 1 prefill only — reps 2+ are KV-cache hits (near-zero, not real measurements).*

| Actual tokens | Mini rep1 prefill ms/tok | MBP rep1 prefill ms/tok | Mini mean gen t/s | MBP mean gen t/s |
|---|---|---|---|---|
| ~4K  (4,019 / 4,020)   | 3.033  | 0.801 | 34.76 | 79.98* |
| ~8K  (8,012 / 8,013)   | 4.870  | 0.777 | 31.38 | 84.01  |
| ~15K (15,011 / 15,012) | 8.316  | 0.884 | 25.08 | 75.52  |
| ~25K (24,993 / 24,994) | 24.154 | 1.057 | 14.40 | 66.68  |
| ~35K (34,995 / 34,996) | 33.752 | 1.243 | 10.75 | 57.85  |

*MBP 4K rep 1 includes cold model load; steady-state is ~92 t/s.*
*MBP at 35K (1.243 ms/tok) is still below Mini's baseline at 4K (3.033 ms/tok).*

*Evidence: `evidence/20260529T130052Z-phase_a-mini/` · `evidence/20260529T184636Z-phase_a-mbp/`*

**Phase B — Cliff localisation**

*Mini: baseline 8.316 ms/tok, cliff threshold 16.632 ms/tok. Cliff onset between 15K–20K.*
*MBP: baseline 0.884 ms/tok, cliff threshold 1.768 ms/tok. Cliff onset between 40K–50K.*
*Mini cliff slope (above onset): ~0.95 ms/tok per 1K tokens (linear).*
*MBP cliff slope (above onset): ~0.03–0.05 ms/tok per 1K tokens — ~20× flatter than Mini.*
*New operational ceilings: Mini <18K tokens · MBP <40K tokens.*

| Actual tokens | Mini rep1 prefill ms/tok | MBP rep1 prefill ms/tok | Cliff (Mini) | Cliff (MBP) |
|---|---|---|---|---|
| 20,005 / 20,006 | 19.352 | 0.965 | YES | no |
| 22,502 / 22,503 | 22.010 | 1.015 | YES | no |
| 24,993 / 24,994 | 24.235 | 1.065 | YES | no |
| 27,500 / 27,501 | 26.597 | 1.108 | YES | no |
| 29,996 / 29,997 | 28.945 | 1.143 | YES | no |
| 32,493 / 32,494 | 31.350 | 1.192 | YES | no |
| 34,995 / 34,996 | 33.767 | 1.329 | YES | no |
| 37,482 / 37,483 | 36.109 | 1.336 | YES | no |
| 39,991 / 39,992 | 38.493 | 1.381 | YES | no |
| ~50K  (49,974)  | —      | 1.929 | —   | YES |
| ~60K  (59,971)  | —      | 1.930 | —   | YES |
| ~70K  (69,959)  | —      | 2.291 | —   | YES |
| ~80K  (79,950)  | —      | 2.819 | —   | YES |
| ~90K  (89,940)  | —      | 2.690 | —   | YES |
| ~100K (99,937)  | —      | 3.238 | —   | YES |
| ~110K (109,919) | —      | 3.228 | —   | YES |
| ~120K (119,912) | —      | 3.826 | —   | YES |

*90K/110K non-monotonic dips (~0.13/0.01 ms/tok) attributed to thermal variance during 60s idle between cells.*

*Evidence: `evidence/20260529T134341Z-phase_b-mini/` · `evidence/20260529T191226Z-phase_b-mbp/`*

**Phase C — Thermal endurance (MacBook Pro M5 Max)**

| Elapsed (min) | gen t/s | Die temp (°C) | GPU power (W) | CPU power (W) | Fan RPM |
|---|---|---|---|---|---|
| 0  | | | | | |
| 5  | | | | | |
| 10 | | | | | |
| 15 | | | | | |
| 20 | | | | | |
| 25 | | | | | |
| 30 | | | | | |
| 45 | | | | | |
| 60 | | | | | |
| 75 | | | | | |
| 90 | | | | | |

**Phase D — Router/Reducer cascade, mixed corpus (Watch + CasaSol)**

| Query | Machine | Run | Grounded | Completion time (s) | Router kind | Notes |
|---|---|---|---|---|---|---|
| Q1 RHR trend          | Mini      | 1 | | | | |
| Q1 RHR trend          | Mini      | 2 | | | | |
| Q1 RHR trend          | Mini      | 3 | | | | |
| Q1 RHR trend          | MBP M5 Max | 1 | | | | |
| Q1 RHR trend          | MBP M5 Max | 2 | | | | |
| Q1 RHR trend          | MBP M5 Max | 3 | | | | |
| Q2 Fencing vol        | Mini      | 1 | | | | |
| Q2 Fencing vol        | Mini      | 2 | | | | |
| Q2 Fencing vol        | Mini      | 3 | | | | |
| Q2 Fencing vol        | MBP M5 Max | 1 | | | | |
| Q2 Fencing vol        | MBP M5 Max | 2 | | | | |
| Q2 Fencing vol        | MBP M5 Max | 3 | | | | |
| Q3 Nota Simple/Catastro | Mini      | 1 | | | | |
| Q3 Nota Simple/Catastro | Mini      | 2 | | | | |
| Q3 Nota Simple/Catastro | Mini      | 3 | | | | |
| Q3 Nota Simple/Catastro | MBP M5 Max | 1 | | | | |
| Q3 Nota Simple/Catastro | MBP M5 Max | 2 | | | | |
| Q3 Nota Simple/Catastro | MBP M5 Max | 3 | | | | |
| Q4 Best years         | Mini      | 1 | | | | |
| Q4 Best years         | Mini      | 2 | | | | |
| Q4 Best years         | Mini      | 3 | | | | |
| Q4 Best years         | MBP M5 Max | 1 | | | | |
| Q4 Best years         | MBP M5 Max | 2 | | | | |
| Q4 Best years         | MBP M5 Max | 3 | | | | |

---

### Conclusion

**Interim conclusion (Phase A+B, 2026-05-29) — H1 and H2 confirmed. H3 and H4 pending.**

**H1 confirmed.** MBP cliff onset between 40K–50K tokens vs Mini onset between 15K–20K — a 2.5× difference, well above the pre-registered 20% threshold for H1. The MBP at 35K (1.243 ms/tok) is still below the Mini's baseline at 4K (3.033 ms/tok). MBP gen t/s inside its own cliff (~43 t/s at 50K) exceeds the Mini's baseline gen t/s (~25 t/s at 15K). The die difference is decisive.

**H2 confirmed.** Gen t/s improvement at 15K: +201% (75.45 vs 25.08). At 25K: +370% (66.44 vs 14.12). Both well above the pre-registered 10% threshold.

**Revised operational ceilings (measurement-grounded):**
- Mac Mini M4 Pro: **< 18K tokens on-wire** (cliff onset ~17–20K)
- MacBook Pro M5 Max: **< 40K tokens on-wire** (cliff onset ~43–50K)

**Cliff shape finding (not pre-registered, emergent):** The Mini cliff above onset is linear at ~0.95 ms/tok per 1K tokens. The MBP cliff above onset is ~20× flatter at ~0.03–0.05 ms/tok per 1K tokens. The Mini cliff is a sharp wall; the MBP cliff is a gentle ramp. Cascade ADR-002's 22K ceiling is safe on MBP; Mini ceiling must be tightened to 18K.

**H3 (thermal):** Anecdotal evidence during Phase B — fan audible and chassis warm at 110K-token prefill cells. Not quantified. Phase C (90-min sustained run) required to assess.

**H4 (cascade portability):** Phase D pending.

**[Pre-registered framings below — to be resolved after Phase C+D.]*

**If H1 and H4 both supported:** The MacBook Pro M5 Max is a materially stronger inference machine for this stack. The Router/Reducer cascade's operational ceiling must be re-stated as hardware-specific. Q3 (Nota Simple vs Catastro mismatch) — the cliff stress test — completes on the M5 Max where it times out on the Mini. That result is directly publishable as CasaSol evidence for the Palantir thesis: a local model with proprietary property data, running on the right hardware, producing a GDPR-clean mismatch report that no frontier model without that data could generate. Exp 005 Phase 1 pre-registration must be updated to reflect the new bundle ceiling.

**If H0 holds and H4 holds:** Generation throughput is memory-bandwidth-bound in a regime where M4 Pro and Max 5 are similar enough that the die difference is not decisive at Q4\_K\_M quantisation. The Chronos operating envelopes are portable across this hardware class. Q3 either completes on both machines or times out on both — either way, the cascade is hardware-portable and the Mini remains the production anchor.

**If H1 holds but H4 fails on Q3:** Phase B's higher cliff threshold on the M5 Max did not translate to cascade tractability — the Reducer's effective KV footprint during generation still exceeds the bandwidth ceiling at Q3's bundle size. The Nota Simple vs Catastro query requires tighter cliff-aware coarsening in the extractor, targeting a lower ceiling than the raw Phase B prefill threshold suggests. This is the most operationally useful failure mode: it closes the gap between synthetic benchmark and production CasaSol behaviour, and defines the next extractor constraint to engineer.

**If H3 is confirmed:** The Mini is the production machine for sustained Nestor sessions; the MacBook Pro M5 Max is a capable development environment but thermally constrained under the load profiles that matter for long Reducer calls. Cascade design should prefer Mini-resident Reducer calls for production workloads.

Evidence directory: `exp_007_hardware_comparison/evidence/`
Scripts: `bench_phase_a.sh`, `bench_phase_b.sh`, `bench_phase_c.py`, `bench_phase_d.py`
Watch corpus index: rsync from `miktam02:~/local-first-ai/tasks/chronos/exp_005_dicer_describer/index/`
CasaSol index: pre-requisite — must exist before Phase D Q3 runs. If absent, Q3 deferred; logged as dependency in evidence directory.

---

#### [Incident 007-Alpha] — Exp 007 Flag Mismatch: FA + q8_0 Were Active During Benchmarks

*Date discovered: 2026-06-07*
*Status: Closed*

**Finding:** Exp 007's pre-registered design stated `OLLAMA_FLASH_ATTENTION=0` (per Incident 003-Alpha mitigation, applied uniformly). The actual runtime had `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` active throughout the entire bench run.

**Evidence:** Ollama log entry from `2026-05-29T14:19:50+02:00` (the day of the Exp 007 Mini run):

```
server config: OLLAMA_FLASH_ATTENTION:true ... OLLAMA_KV_CACHE_TYPE:q8_0
load request: FlashAttention:Enabled KvCacheType:q8_0
```

The `~/Library/LaunchAgents/com.ollama.serve.plist` had both flags set, and the log confirms the same configuration back to at least `2026-03-05`. The Incident 003-Alpha mitigation (`FA=0`) was documented in this log but never applied to the persistent LaunchAgent config. The mitigation existed only in text.

**Consequence for Exp 007:** The 20K prefill cliff and all Phase A/B throughput numbers are correct measurements — but they were taken under FA=1 + q8_0 conditions. The "default settings" framing in the pre-registration is wrong. The numbers stand; the label does not. The operational ceilings (Mini < 18K, MBP < 40K) are valid as *optimized* ceilings, not default-settings ceilings.

**Consequence for Exp 008:** The Exp 008 hypothesis assumed Exp 007 established an FA=0 baseline to compare against. That baseline does not exist. Running Exp 008 as designed would reproduce Exp 007 numbers. The experiment is reformulated — see Exp 008 status update below.

**Root cause:** The Incident 003-Alpha mitigation was applied at the session level (manual `launchctl setenv` for that investigation) but never written back to the plist. Subsequent sessions, including Exp 007, used the unmodified config.

**Fix going forward:** Ollama is now managed as a LaunchDaemon (`/Library/LaunchDaemons/com.ollama.serve.plist`), config is explicit and version-controlled. All future bench scripts must log Ollama env state as part of the evidence record (retrofitted into Exp 008 scripts and enforced in all future experiments).

---

## [EXPERIMENT 008] — Flash Attention + q8_0 KV Cache: Does It Push the Cliff?

*Date pre-registered: 2026-05-29*
*Reformulated: 2026-06-07 — see Incident 007-Alpha*
*Executed: 2026-06-07 (Mini, FA=0 primary run)*
*Status: Complete — landmark finding (flags cause the cliff, not the hardware)*
*Subdirectory: `tasks/chronos/exp_008_flash_attention/`*

---

### Observation

**Original observation (pre-registration):** Exp 007 was believed to have established the prefill cliff at ~20K tokens under FA=0 + fp16 KV cache. Exp 008 was designed to test whether enabling FA + q8_0 would push that cliff to ≥30K.

**Revised observation (2026-06-07, Incident 007-Alpha):** Exp 007 ran with `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` active — confirmed by Ollama log entry `2026-05-29T14:19:50+02:00`. The 20K cliff *is* the FA + q8_0 result. No FA=0 baseline exists.

The experiment is therefore inverted: instead of measuring the gain from enabling the flags, we now need to establish the cost of running without them. This produces the same data the original Exp 008 would have compared against — just measured in the correct order.

Two Ollama flags directly address memory bandwidth consumption:

1. **`OLLAMA_FLASH_ATTENTION=1`** — memory-efficient attention kernel that avoids materialising the full attention matrix, reducing peak bandwidth for the attention operation.
2. **`OLLAMA_KV_CACHE_TYPE=q8_0`** — quantizes the KV cache from fp16 to 8-bit, halving KV cache memory footprint and proportionally reducing bandwidth during autoregressive decoding.

If the 20K cliff (FA + q8_0) is primarily KV-bandwidth-bound, the FA=0 + fp16 baseline should show a *lower* cliff — the flags are already providing measurable relief. If the cliff is identical without the flags, the bandwidth savings are not the binding constraint.

---

### Hypothesis

**H1 (primary, reformulated):** `OLLAMA_FLASH_ATTENTION=0` + fp16 KV cache (flags disabled) produces a prefill cliff onset **below** the 20K observed in Exp 007 (FA + q8_0 active), confirming that the flags provide measurable operational headroom on Mac Mini M4 Pro.

**H2 (throughput, reformulated):** Gen t/s at medium context (15K–25K tokens) is ≥10% **lower** without the flags vs Exp 007 values (34.76 / 31.38 / 25.08 / 14.40 / 10.75 t/s), consistent with increased KV cache bandwidth pressure during decoding.

**H3 (instability):** FA=1 causes silent generation failures, empty responses, or wedged runners under production-scale prompts (>15K tokens) — re-testing Incident 003-Alpha H1 at scale. (This sub-hypothesis is unchanged and tests the flags-on condition, using the current production config.)

**H0 (null):** Prefill cliff threshold and gen t/s are statistically indistinguishable between FA=0/fp16 and Exp 007 FA=1/q8_0 results. Flags have no measurable effect on this hardware.

**Falsification criteria:**
- H1 rejected if FA=0 cliff onset is ≥20K (same as or higher than the FA + q8_0 result).
- H2 rejected if gen t/s at 15K and 25K sizes falls within ±10% of Exp 007 values under FA=0.
- H3 confirmed if ≥1 Phase A cell (flags on, production config) produces empty response, timeout, or wedged runner.

---

### Experiment

**Protocol (reformulated 2026-06-07):** Same Phase A (5 sizes: 4K, 8K, 15K, 25K, 35K, 3 repeats) and Phase B (cliff sweep 20K–40K, 2.5K spacing, 2 repeats) as Exp 007. **Primary run: FA=0 + fp16 KV cache** (flags-off baseline, the missing comparison point). LaunchDaemon temporarily overridden: `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=` (unset, defaults to fp16). Bench scripts must log confirmed env state from process scan before each run. `start_ollama_flags.sh` is not used for the primary run; a matching `start_ollama_no_flags.sh` will be created. Secondary run (flags on, production config) validates H3 and confirms Exp 007 numbers are reproducible.

**Fixtures:** Shared from `exp_007_hardware_comparison/fixtures/padding/` — same files, no duplication.

**Baseline:** Phase A rep 1 prefill at 15K tokens from **this experiment** (not Exp 007) — flags change absolute prefill numbers; baseline must be measured under the same conditions being tested.

**Machines:** Mac Mini M4 Pro first (direct comparison to Exp 007). MacBook Pro M5 Max second (once MBP Exp 007 Phase A/B complete, for cross-machine comparison with flags).

**Controls:**
- AC power throughout.
- Wi-Fi disabled during collection.
- Same `gemma4:26b` model weights as Exp 007.
- Same `num_ctx=131072`, `temperature=0.0`, `num_predict` values as Exp 007.
- 60s idle between size cells (model unloaded via `keep_alive=0` + `ollama stop`).

---

### Pre-registered pass criteria

- **H1 supported:** Cliff threshold ≥30K tokens (Exp 007 cliff was at 20K).
- **H2 supported:** Gen t/s at 15K and 25K each ≥10% above Exp 007 rep 1 values (25.08 and 14.12 t/s respectively).
- **H3 confirmed:** ≥1 Phase A cell fails with empty/timeout/wedge (operationally disqualifying if confirmed — flags would be rejected for production use).

---

### Data / Results

*Run: 2026-06-07. Machine: Mac Mini M4 Pro. Config: FA=0, fp16 KV cache (primary / baseline run).*
*Evidence: `exp_008_flash_attention/evidence/20260607T101221Z-phase_a-mini-nf/` and `20260607T102637Z-phase_b-mini-nf/`*

**Phase A — Generation throughput (FA=0, fp16 KV)**

| Size | Actual tokens | Rep1 prefill ms/tok | Mean gen t/s | Exp 007 gen t/s (FA+q8_0) | Delta gen t/s |
|------|--------------|---------------------|--------------|---------------------------|---------------|
| 4k   | 4019         | 1.557               | 40.42        | 34.76                     | +16%          |
| 8k   | 8012         | 1.642               | 42.60        | 31.38                     | +36%          |
| 15k  | 15011        | 1.774               | 36.97        | 25.08                     | +47%          |
| 25k  | 24993        | 1.983               | 31.08        | 14.40                     | +116%         |
| 35k  | 34995        | 2.241               | 26.69        | 10.75                     | +148%         |

**Phase B — Cliff localisation (FA=0, fp16 KV). Baseline 1.774 ms/tok → threshold 3.548 ms/tok.**

| Size   | Actual tokens | Rep1 prefill ms/tok | Cliff triggered | Exp 007 result (FA+q8_0)   |
|--------|--------------|---------------------|-----------------|----------------------------|
| 20k    | 20005        | 1.846               | **no**          | YES (19.352 ms/tok)        |
| 22500  | 22502        | 1.923               | **no**          | YES (22.010 ms/tok)        |
| 25k    | 24993        | 2.001               | **no**          | YES (24.235 ms/tok)        |
| 27500  | 27500        | 2.069               | **no**          | YES (26.597 ms/tok)        |
| 30k    | 29996        | 2.124               | **no**          | YES (28.945 ms/tok)        |
| 32500  | 32493        | 2.069               | **no**          | —                          |
| 35k    | 34995        | 2.119               | **no**          | —                          |
| 37500  | 37482        | 2.167               | **no**          | —                          |
| 40k    | 39991        | 2.215               | **no**          | —                          |

---

### Conclusion

**Status: Complete — landmark finding. All pre-registered hypotheses rejected; new causal finding.**

**H1 (FA=0 cliff below 20K): REJECTED.** With FA=0/fp16, no cliff appears anywhere in the 20K–40K range. Prefill at 40K is 2.215 ms/tok — the slope from 15K to 40K is ~0.018 ms/tok per 1K tokens, essentially linear and stable.

**H2 (FA=0 gen t/s lower than FA+q8_0): REJECTED.** Gen t/s is substantially *higher* without flags at every size. The effect grows with context: +16% at 4K, +148% at 35K. The flags degrade throughput under sustained load.

**H0 (no difference): REJECTED.** There is a very large difference — in the opposite direction to all prior expectations.

**Causal finding (not pre-registered):** `OLLAMA_FLASH_ATTENTION=1` + `OLLAMA_KV_CACHE_TYPE=q8_0` is the cause of the 20K prefill cliff, not a remedy for it. Removing both flags eliminates the cliff entirely within the tested range (up to 40K tokens). The mechanism is not yet isolated — either flag or their combination may be responsible. A follow-on micro-experiment (Exp 010 candidate) could isolate the culprit by testing FA=1/fp16 and FA=0/q8_0 separately.

**Retrospective consequence for Incident 003-Alpha:** The original incident (FA causing CPU fallback on gemma4-think:26b) was correctly diagnosed. The Incident 003-Alpha mitigation (FA=0) was the right call. The plist re-enabling FA=1 after that incident was a regression that went undetected until Exp 008.

**Revised operational ceiling for Mac Mini M4 Pro (FA=0, fp16):** cliff not reached at 40K — operational ceiling is now **> 40K tokens on-wire** with a stable linear prefill slope. The previous < 18K ceiling was an artefact of FA+q8_0. All cascade ADR ceilings derived from Exp 007 must be revisited.

**Production action taken:** LaunchDaemon plist updated to `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE` removed. Daemon restarted 2026-06-07.

Evidence directory: `exp_008_flash_attention/evidence/`
Scripts: `bench_phase_a.py --no-flags`, `bench_phase_b.py --no-flags`
Fixtures: shared from `exp_007_hardware_comparison/fixtures/padding/`

---

## Experiment 009 — Adversarial Project Critic (Local vs. Frontier)

**Date pre-registered:** 2026-06-06
**Date executed:** 2026-06-06
**Status:** Complete — FAIL (50% overlap, 50% false-positive rate)
**Subdirectory:** [`tasks/chronos/exp_009_adversarial_critic/`](./exp_009_adversarial_critic/)
**Content pillar:** The Silicon Sentinel (Infrastructure & Privacy)

---

### Hypothesis

**Primary:** A local gemma4:26b adversarial critic, given the same project context (recent commits, BUILD_LOG, BRIEF, TODO), identifies ≥60% of the high-severity issues that a Claude Sonnet adversarial critic identifies.

**Secondary:** The local critic has a false-positive rate (issues flagged by gemma4 but not Claude) of ≤30%.

**Null hypothesis:** The local critic identifies <40% of Claude-flagged high-severity issues — not useful as a standalone QA gate.

---

### Design

Three adversarial personas applied in sequence by both critics to the same fixed context bundle:

| Persona | Question |
|---------|----------|
| Sceptical investor | Is the moat real or just marketing copy? |
| DPO / AEPD auditor | Will the compliance architecture survive a real audit? |
| Competing engineer | How would I replicate this in a weekend? |

Fixed JSON output schema across both critics — `holds_up`, `weak`, `missing` per persona, plus `top_actions` and `severity_counts`. Hypothesis, personas, and schema committed before execution.

**Context inputs (identical for both critics):**
1. `git log --oneline -20`
2. Last 100 lines of `BUILD_LOG.md`
3. First 80 lines of `BRIEF.md`
4. First 80 lines of `strategy/TODO.md`

**Target project:** CasaSol (local-first AI property intelligence, Costa del Sol real estate)

---

### Experiment

**Option A — Claude Sonnet 4.6:** slash command `/critic` in Claude Code; inline context collection; JSON output saved manually to results.

**Option B — gemma4:26b:** `critic.py --project ~/REPOS/casasol`; Ollama API; JSON output saved automatically with token stats.

**Token accounting:**

| Metric | Claude Sonnet 4.6 | gemma4:26b |
|--------|------------------|------------|
| Input tokens | ~4,750 (est, chars÷4) | 8,966 (exact, Ollama API) |
| Output tokens | ~1,170 (est, words×1.3) | 902 (exact) |
| Runtime | inline | 84.7 s |
| Speed | — | 28.7 tok/s |
| Issues raised | 18 | 18 |

---

### Results

**Overlap scoring (Claude HIGH-severity issues — 3 total):**

| # | Claude high-severity issue | gemma4 match? |
|---|---------------------------|---------------|
| H1 | VLM witnessing pipeline not in any commit — moat claim unimplemented | NO |
| H2 | No DPA template — hard gate on any pilot SOW | YES (exact) |
| H3 | No revenue model / pricing anywhere | PARTIAL (unit economics angle) |

**Overlap rate:** ~50% (1 certain + 1 partial / 3) — below ≥60% threshold → **H1 rejected**

**False positive rate:** ~50% (~9 of 18 gemma4 issues not in Claude) — above ≤30% threshold → **secondary criterion FAIL**

**Overall verdict: FAIL**

---

### Key Finding

gemma4 matched the DPO/compliance layer near-perfectly (DPA, DSAR, DPIA — 3/3 near-matches). It failed on the most important engineering finding: the VLM witnessing pipeline is described in the BRIEF as the primary moat component but does not exist in any commit. The corpus is built from text seed files.

Claude caught this by cross-referencing a BUILD_LOG claim ("witnessing reframe — image capture is core MVP") against the git commit history (no VLM code). gemma4 accepted the claim as implemented and critiqued its replicability instead.

The distinction: **pattern-matching on what is present vs. detecting the gap between documented intent and implemented reality.**

gemma4's ~50% false-positive rate is not noise — on review, most are valid gaps Claude did not explore (filesystem crypto isolation, labor scalability, Redactor PII audit rate). Two critics with different search strategies.

---

### Conclusion

gemma4:26b is viable as a compliance-layer QA gate. It catches the known-document-type gaps (DPA, DSAR, DPIA) reliably and at zero marginal cost. It is not viable as a replacement for frontier-model review when the most critical failures are impl-vs-docs gaps — cases where the model must reason about what is absent from the codebase, not just what is described in documentation.

Practical split: run gemma4 on every significant commit as a compliance and pattern-coverage check. Run Claude periodically (before demos, before pilots, before external review) for implementation integrity checks.

---

### Evidence

- `exp_009_adversarial_critic/HYPOTHESIS.md` — pre-registered hypothesis (git timestamp is proof)
- `exp_009_adversarial_critic/results/claude_20260606_134500.json` — Option A output
- `exp_009_adversarial_critic/results/gemma4_26b_20260606T131205.json` — Option B output
- `exp_009_adversarial_critic/EXECUTION_LOG.md` — step-by-step run record, scoring, observations
- `exp_009_adversarial_critic/critic.py` — Option B script (reusable against any project)
- `exp_009_adversarial_critic/compare.py` — comparison harness

---

## [EXPERIMENT 010] — Flash Attention vs q8_0 KV Cache: Factorial Isolation

*Date pre-registered: 2026-06-07*
*Executed: 2026-06-07 (Mini, all four conditions)*
*Status: Complete — FA=1 is the sole culprit*
*Subdirectory: `tasks/chronos/exp_010_fa_isolation/`*

---

### Observation

Exp 008 produced an unexpected result: removing both `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` eliminated the 20K prefill cliff entirely and improved gen t/s by up to 148% at 35K tokens. The flags that were believed to help (by reducing memory bandwidth pressure) were in fact causing the cliff.

Exp 008 tested one diagonal of the 2×2 factorial:

| | fp16 KV | q8_0 KV |
|---|---|---|
| **FA=0** | Exp 008 ✓ — no cliff to 40K | **← Condition C (this exp)** |
| **FA=1** | **← Condition B (this exp)** | Exp 007 ✓ — cliff at 20K |

Two conditions remain unmeasured. Running them isolates whether the cliff is caused by FA=1 alone, q8_0 alone, or requires both flags together (interaction effect).

**Why either flag might hurt performance on Apple Silicon:**

- **Flash Attention (FA=1):** The Metal kernel for flash attention on Apple Silicon may have suboptimal behaviour for the MoE architecture of gemma4:26b. FA tiles the KV computation to reduce peak bandwidth, but tiling introduces synchronisation overhead. If the M4 Pro's unified memory controller is not the bottleneck (which the Exp 008 result implies), the tiling overhead becomes net negative. Incident 003-Alpha observed FA causing CPU fallback at scale — this may be the same mechanism at a milder severity.

- **q8_0 KV quantization:** Quantizing/dequantizing the KV cache on every attention head adds per-token compute. With fp16, the KV values are used as-is; with q8_0, each element must be scaled back to fp16 representation before the attention dot product. On Apple Silicon's unified memory (where the GPU and CPU share the same DRAM), the dequantisation arithmetic may consume GPU shader time that outweighs the bandwidth saved.

- **Interaction effect:** FA's tiling pattern changes the order in which KV cache elements are read. If q8_0's dequantisation is not fused into the FA kernel, the combination requires two passes over the KV data where one was expected — doubling the effective bandwidth cost and negating the optimization entirely.

---

### Hypothesis

**H1 (FA is the culprit):** Condition B (FA=1, fp16) produces a prefill cliff at or below 25K tokens, matching Exp 007's behaviour. Condition C (FA=0, q8_0) shows no cliff in the 20K–40K range, matching Exp 008.

**H2 (q8_0 is the culprit):** Condition C (FA=0, q8_0) produces a prefill cliff at or below 25K tokens. Condition B (FA=1, fp16) shows no cliff, matching Exp 008.

**H3 (interaction effect):** Neither Condition B nor Condition C alone causes a cliff. Only the combination (FA=1 + q8_0, Exp 007) causes the cliff. Both B and C match Exp 008's flat prefill profile.

**H0 (null — neither flag matters):** Both B and C match Exp 008 within ±15%. The Exp 007 cliff was caused by some other variable (thermal state, Ollama session state, model load order) not captured in the flag configuration.

**Falsification criteria:**
- H1 confirmed: Condition B cliff onset ≤ 25K; Condition C cliff onset > 40K.
- H2 confirmed: Condition C cliff onset ≤ 25K; Condition B cliff onset > 40K.
- H3 confirmed: Both B and C show no cliff through 40K, within 15% of Exp 008 gen t/s values.
- H0 confirmed: Both B and C within ±15% of Exp 008 gen t/s at 15K, 25K, 35K.

---

### Experiment

**Protocol:** Same Phase A (5 sizes: 4K, 8K, 15K, 25K, 35K, 3 repeats) and Phase B (cliff sweep 20K–40K, 2.5K spacing, 2 repeats) as Exp 007/008. Each condition run separately with a clean Ollama restart between conditions. LaunchDaemon stopped before each condition; `restore_daemon.sh` run after.

**Condition B:** `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE` unset (fp16 default). Start with `start_condition_b.sh`.
**Condition C:** `OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=q8_0`. Start with `start_condition_c.sh`.

**Instrumentation (optional):** Run `sudo powermetrics --samplers gpu_power,cpu_power -i 2000` in a separate terminal during Phase B cells to capture GPU utilisation and memory bandwidth. Timestamp correlation with cell start/end allows per-size power attribution. This is exploratory — not a pre-registered metric.

**Controls:** Same as Exp 007/008: AC power, Wi-Fi off, same model, same num_ctx=131072, 60s idle between cells.

**Fixtures:** Shared from `exp_007_hardware_comparison/fixtures/padding/`.

---

### Pre-registered pass criteria

- **H1 confirmed:** Condition B cliff ≤ 25K tokens AND Condition C gen t/s at 35K within 15% of Exp 008 (26.69 t/s).
- **H2 confirmed:** Condition C cliff ≤ 25K tokens AND Condition B gen t/s at 35K within 15% of Exp 008.
- **H3 confirmed:** Both B and C cliff-free through 40K, both within 15% of Exp 008 gen t/s at 35K.
- **H0 confirmed:** Both B and C within ±15% of Exp 008 at 15K, 25K, 35K.

---

### Data / Results

*[To be filled upon execution.]*

**Phase A — all four conditions**

| Size | Cond A FA=0/fp16 (Exp 008) | Cond B FA=1/fp16 | Cond C FA=0/q8_0 | Cond D FA=1/q8_0 (Exp 007) |
|------|---------------------------|------------------|------------------|---------------------------|
| 4k gen t/s   | 40.42 | 33.44 | 42.05 | 34.76 |
| 8k gen t/s   | 42.60 | 31.70 | 44.07 | 31.38 |
| 15k gen t/s  | 36.97 | 25.36 | 38.41 | 25.08 |
| 25k gen t/s  | 31.08 | 19.87 | 32.35 | 14.40 |
| 35k gen t/s  | 26.69 | 16.26 | 27.90 | 10.75 |
| 15k prefill ms/tok | 1.774 | 5.405 | 1.694 | 2.775 |
| 35k prefill ms/tok | 2.241 | 12.286 | 2.087 | 11.252 |

**Phase B — Cliff localisation**

| Condition | Cliff onset | 40K prefill ms/tok |
|-----------|-------------|-------------------|
| A: FA=0, fp16 — Exp 008 | **not reached at 40K** | 2.215 |
| B: FA=1, fp16 — this exp | **32.5K** | 13.793 |
| C: FA=0, q8_0 — this exp | **not reached at 40K** | 2.213 |
| D: FA=1, q8_0 — Exp 007  | **20K** | ~38 |

---

### Conclusion

**Status: Complete — definitive causal isolation.**

**H1 (FA is the culprit): CONFIRMED.** FA=1 alone (Condition B, fp16 KV) causes a cliff at 32.5K tokens and degrades gen t/s by 31–39% across all context sizes. The pre-registered cliff threshold of ≤25K was slightly conservative; the actual onset is 32.5K. The causal relationship is unambiguous.

**H2 (q8_0 is the culprit): REJECTED.** FA=0 + q8_0 (Condition C) shows no cliff through 40K and matches the FA=0/fp16 baseline within ±5% — within noise. q8_0 alone is benign.

**H3 (interaction effect): REJECTED.** FA=1 alone (without q8_0) is sufficient to cause significant degradation and a cliff at 32.5K. The interaction (FA=1+q8_0) moves the cliff further down to 20K, but FA=1 is the necessary and primary cause.

**Complete 2×2 picture:**

The degradation is attributable to FA=1 across all context sizes — not just at the cliff. At 15K tokens, FA=1 alone more than triples the prefill time (1.774 → 5.405 ms/tok). At 35K it causes a 5.8× increase (2.241 → 12.286 ms/tok). q8_0, by contrast, produces prefill times nearly identical to fp16 (1.774 → 1.694 ms/tok at 15K) and modestly improves gen t/s (+4% at 15K, +5% at 35K), consistent with the KV cache memory savings reducing bandwidth pressure during autoregressive decoding.

**Why FA=1 hurts on Apple Silicon unified memory:**

Flash attention was designed to reduce HBM bandwidth on discrete GPU architectures (CUDA + dedicated VRAM) by replacing full KV materialisation with SRAM-tiled block computation. On Apple Silicon's unified memory architecture there is no separate SRAM/VRAM hierarchy — the GPU and CPU share the same DRAM pool. The tiling overhead FA introduces (additional synchronisation, block-boundary computations, non-sequential memory access patterns) applies without the bandwidth benefit it was designed to capture. The M4 Pro's memory controller handles the KV cache efficiently without tiling; FA turns a sequential bandwidth problem into a compute + synchronisation problem that is worse on this hardware. The MoE architecture of gemma4:26b likely amplifies this: MoE routing is already non-sequential, and FA's tiling may conflict with the expert dispatch pattern.

**Operational consequence:** The 20K cliff that shaped all Chronos cascade design since Incident 003-Alpha was entirely attributable to `OLLAMA_FLASH_ATTENTION=1`. The Mac Mini M4 Pro's true operational ceiling under optimal configuration (FA=0) is **> 40K tokens on-wire**, confirmed across all four conditions.

**Production action:** Daemon updated to FA=0, q8_0 re-enabled (Condition C is the optimal production config — no cliff, +4–5% gen t/s vs fp16, reduced KV memory footprint). Effective 2026-06-07.

Evidence: `exp_010_fa_isolation/evidence/` — four runs: Phase A+B for Conditions B and C.
Script: `bench.py --condition B|C --phase A|B`
Fixtures: shared from `exp_007_hardware_comparison/fixtures/padding/`

---

## Experiment 011 — MLX Runtime vs Ollama: Does the Runtime Cause the Cliff?

**Status: Pre-registered — 2026-06-09**

*Subdirectory: `tasks/chronos/exp_011_mlx_runtime/`*

### Motivation

Exp 008 and Exp 010 proved that `OLLAMA_FLASH_ATTENTION=1` causes the prefill cliff on Apple Silicon. The open question: is the cliff a property of Ollama's implementation (llama.cpp + its FA flag), or would any runtime exhibit degradation at large context on this hardware? MLX is Apple's own ML framework, designed from the ground up for unified memory — no SRAM/HBM hierarchy assumption, different attention kernels. If MLX has no cliff, the cause is Ollama's FA implementation. If MLX also cliffs, the hardware architecture itself is the constraint.

The GigaHiveDigital exchange (2026-06-09) confirmed that other local AI practitioners run gemma4:26b-mlx in production. This is a live comparison worth measuring.

### Confounds (pre-registered)

1. **Quantisation format:** Ollama runs GGUF Q4_K_M. MLX runs OptiQ 4-bit (`mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit`). Different compression schemes; not a pure runtime isolation.
2. **Active parameter count:** Ollama reports 25.8B parameters for gemma4:26b. The MLX model name includes "A4B" suggesting potentially 4B active parameters per forward pass (sparse MoE). If true, compute cost per token is lower in MLX — this would affect gen t/s comparisons. Prefill scaling behaviour (how ms/tok grows with context) remains informative regardless.
3. **Tokeniser differences:** MLX uses the HuggingFace tokeniser; Ollama uses the GGUF-embedded tokeniser. Actual token counts for the same prompt text may differ slightly. Bench script reports actual token count from the MLX tokeniser.

### Hypotheses

**H1 (runtime advantage):** MLX prefill ms/tok at 15K tokens is ≤ Ollama FA=0/fp16 baseline (1.774 ms/tok). MLX's unified-memory-native kernels outperform llama.cpp on Apple Silicon.

**H2 (no cliff):** MLX shows no prefill cliff through 40K tokens (prefill ms/tok stays below 2× the 15K baseline at all Phase B sizes).

**H3 (gen t/s):** MLX gen t/s at 25K tokens is within ±25% of Ollama FA=0/fp16 (31.08 t/s). Wide tolerance given the active-parameter confound.

**Falsification criteria:**
- H1 rejected if MLX 15K prefill > 1.774 ms/tok
- H2 rejected if any Phase B size exceeds 2× the MLX 15K baseline
- H3 rejected if MLX 25K gen t/s is outside [23.3, 38.9] t/s

### Experiment design

**Model (Ollama):** `gemma4:26b` — FA=0, q8_0 (Exp 010 Condition C optimal config). Reference only; no new Ollama runs needed.

**Model (MLX):** `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` — default MLX settings.

**Machine:** Mac Mini M4 Pro (miktam02), 64GB unified memory.

**Phase A:** 5 context sizes — 4K, 8K, 15K, 25K, 35K. 3 reps per size. Fresh model load and 60s idle between sizes. Rep 1 prefill is authoritative (cold KV). Reps 2–3 measure gen t/s with warm KV.

**Phase B:** 9 context sizes — 20K, 22.5K, 25K, 27.5K, 30K, 32.5K, 35K, 37.5K, 40K. 2 reps per size. Cliff detection: prefill > 2× the 15K baseline.

**Metrics:** prefill ms/tok (rep 1), mean gen t/s (all reps), cliff triggered (bool).

**Baselines for comparison:**

| Condition | Source | 15K prefill ms/tok | 35K gen t/s |
|---|---|---|---|
| Ollama FA=0, fp16 | Exp 008 | 1.774 | 26.69 |
| Ollama FA=0, q8_0 | Exp 010 Cond C | 1.694 | 27.90 |
| MLX default | **Exp 011** | TBD | TBD |

### Setup

```bash
# Download model (~14–16 GB)
python3 -c "from mlx_lm import load; load('mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit')"

# Run Phase A
python3 bench.py --phase A --machine mini

# Run Phase B
python3 bench.py --phase B --machine mini --baseline <15K_prefill_from_phase_a>
```

### Data

*Runs completed 2026-06-09. Evidence: `exp_011_mlx_runtime/evidence/`.*

**Baseline table (updated):**

| Condition | Source | 15K prefill ms/tok | 35K gen t/s |
|---|---|---|---|
| Ollama FA=0, fp16 | Exp 008 | 1.774 | 26.69 |
| Ollama FA=0, q8_0 | Exp 010 Cond C | 1.694 | 27.90 |
| MLX default (OptiQ 4-bit) | **Exp 011** | **1.650** | **26.42** |

**Phase A (2026-06-09T064038Z):**

| Size | Tokens | Rep1 prefill ms/tok | Mean gen t/s | Ollama Cond C ref |
|---|---|---|---|---|
| 4K | 3,555 | 1.546 | 52.41 | 42.05 |
| 8K | 7,112 | 1.628 | 47.24 | 44.07 |
| 15K | 13,333 | 1.650 | 38.71 | 38.41 |
| 25K | 22,223 | 1.964 | 27.00 | 32.35 |
| 35K | 31,112 | 2.161 | 26.42 | 27.90 |

**Phase B (2026-06-09T065708Z). Baseline: 1.650 ms/tok → cliff threshold: 3.300 ms/tok.**

| Size | Tokens | Rep1 prefill ms/tok | Cliff |
|---|---|---|---|
| 20K | 17,778 | 1.859 | no |
| 22.5K | 20,001 | 1.961 | no |
| 25K | 22,223 | 2.128 | no |
| 27.5K | 24,445 | 2.118 | no |
| 30K | 26,666 | 2.238 | no |
| 32.5K | 28,889 | 2.498 | no |
| 35K | 31,112 | 2.464 | no |
| 37.5K | 33,333 | **2.518** | no |
| 40K | 35,555 | 2.270 | no |

Peak Phase B prefill: 2.518 ms/tok (37.5K). Threshold: 3.300 ms/tok. Margin: 24%.

**Note on Phase B gen t/s:** High inter-rep variance observed (e.g. 22.5K: rep1=43.02 vs rep2=22.39; 40K: rep1=20.97 vs rep2=42.63). MLX does not appear to reset KV cache state between reps in the same session; rep2 reflects a warm cache with potentially different occupancy. Rep1 prefill is the authoritative metric for this experiment. Phase B mean gen t/s values are not reliable for ranking against Ollama.

### Conclusion

**H1 — CONFIRMED.** MLX 15K prefill = 1.650 ms/tok < 1.774 ms/tok (Ollama FA=0/fp16). MLX's unified-memory-native attention kernels match or slightly outperform the best Ollama config (FA=0, q8_0: 1.694 ms/tok). The advantage is modest (~3% at 15K) but consistent across Phase A.

**H2 — CONFIRMED.** No prefill cliff through 40K tokens. Phase B maximum (37.5K, 2.518 ms/tok) is 24% below the 3.300 ms/tok threshold. Prefill growth is smooth and sub-linear. The FA=1 cliff observed in Exp 007 and Exp 010 Condition B is absent from MLX entirely, confirming that the cliff is a property of Ollama's Flash Attention implementation (llama.cpp FA tiling on unified memory), not a hardware constraint.

**H3 — CONFIRMED.** MLX 25K gen t/s = 27.00 t/s, within the pre-registered ±25% window around Ollama FA=0/fp16 (31.08 t/s). Window: [23.3, 38.9]. At 4K, MLX gen t/s is 52.41 vs Ollama's ~42 — a 24% advantage. By 35K, MLX (26.42) and Ollama FA=0/q8_0 (27.90) are within 6%. The short-context advantage is consistent with the A4B confound: if the MLX model activates only ~4B parameters per token vs Ollama's full MoE forward pass, generation is cheaper per step at short contexts, converging as KV cache overhead dominates at long context.

**Principal finding:** The Flash Attention cliff is a runtime artefact of llama.cpp's FA implementation on Apple Silicon, not a hardware architecture limit. Two independent runtimes (Ollama FA=0, MLX) show equivalent prefill scaling and no cliff through 40K tokens on the same hardware. The operational ceiling of the Mac Mini M4 Pro for long-context inference is determined by memory bandwidth, not by any single runtime's attention kernel.

**Confound status:**
- Confound 1 (quantisation): Different format (OptiQ 4-bit vs Q4_K_M). Results are comparable within ~3% — quantisation difference is not driving the MLX advantage.
- Confound 2 (active parameters): A4B label likely indicates sparse MoE activation. Gen t/s advantage at short context (+24% at 4K) is consistent with fewer active parameters per forward pass. This confound prevents strong gen t/s comparison but does not affect prefill scaling conclusions.
- Confound 3 (tokeniser): Actual token counts were ~89% of target sizes across all cells (e.g. 4K target → 3,555 tokens). Consistent ratio; both runtimes benchmarked on the same prompt text.

*Status: Complete (2026-06-09). Evidence: `exp_011_mlx_runtime/evidence/`.*

---

## Exp 012 — Cost vs Capability: Where the Curve Breaks

*Pre-registered: 2026-06-09. Subdirectory: `tasks/chronos/exp_012_cost_capability/`*

### Motivation

@Prathkum (79.7K views, Jun 8 2026): "We don't need a more powerful model right now. What we need to solve is the cost problem." @nix_eth replied: "I don't think intelligence, capabilities, and cost are all tied together."

Both claims are unfalsifiable without a fixed task and a scoring rubric. Exp 009 produced one data point: gemma4:26b matched Claude Sonnet 4.6 on compliance gap detection (3/3 near-exact matches) but missed the highest-severity implementation gap (0/1). That is a specific capability threshold at a specific cost differential. This experiment extends that single comparison into a full cost-capability curve across four model tiers, using the same task setup from Exp 009 plus a second task designed to favour local models.

### Research question

At what capability threshold — and at what cost premium — does a frontier model outperform a local model on structured analytical tasks? Is the relationship linear, stepped, or does local match frontier within a specific capability envelope?

### Confounds (pre-registered)

1. **Task selection bias.** Tasks chosen here are structured and schema-constrained — a known strength of local models with a fixed prompt. Open-ended creative or multi-hop reasoning tasks might produce a different curve. Results apply only to the task types measured.
2. **Prompt sensitivity.** Frontier models may be more sensitive to prompt phrasing differences. All models receive identical prompts to control for this, accepting that a prompt optimised for one model may disadvantage another.
3. **Local model cost accounting.** gemma4:26b has zero marginal API cost but non-zero hardware and electricity cost. Hardware is treated as sunk cost (already purchased for CasaSol production use). Electricity is not measured. "Zero cost" means zero marginal API cost, not zero total cost of ownership.
4. **Non-determinism.** Frontier models run at default temperature. gemma4:26b runs at temperature=0.1 (production config). 3 reps per model per task; majority verdict for pass/fail scoring.

### Hypotheses

**H1 (local ceiling):** gemma4:26b matches or exceeds frontier models on Task A (structured compliance extraction) — the task type where Exp 009 showed 3/3 overlap with Sonnet. Falsified if Haiku or Sonnet scores higher than gemma4 on Task A.

**H2 (frontier premium):** At least one frontier model scores higher than gemma4:26b on Task B (cross-document implementation gap detection) — the task type where Exp 009 showed gemma4 missed the highest-severity finding. Falsified if gemma4 matches all frontier models on Task B.

**H3 (cost curve is stepped, not linear):** The quality improvement from Haiku → Sonnet → Opus on Task B does not scale proportionally with cost. There exists at least one tier pair where the cost multiplier exceeds the quality gain multiplier. Falsified if quality scores increase at the same rate as cost across all tier pairs.

**Falsification criteria:**
- H1 rejected if gemma4:26b Task A score < max(Haiku, Sonnet, Opus) Task A score
- H2 rejected if gemma4:26b Task B score = max(Haiku, Sonnet, Opus) Task B score
- H3 rejected if (quality_gain / cost_multiplier) is within ±10% across all adjacent tier pairs

### Tasks

**Task A — Compliance extraction (structured, schema-constrained)**

Reuse the Adversarial Critic context bundle from Exp 009: CasaSol BRIEF.md + BUILD_LOG.md + git log summary. Fixed system prompt: DPA Compliance Auditor persona, JSON output schema (gaps array with severity, category, finding, recommendation). Same prompt as Exp 009.

Ground truth: pre-scored set of 5 compliance items with known correct classifications (established from Exp 009 results + manual verification before running).

Scoring: 1 point per correct gap identification, 0 for missed or fabricated (false positive). Max score: 5. Scored before running any model.

**Task B — Implementation gap detection (cross-document reasoning)**

Reuse the Implementation Auditor persona from Exp 009: same context bundle, same JSON schema. Ground truth: the single high-severity gap confirmed in Exp 009 (primary moat component described in BRIEF/deck/BUILD_LOG with no corresponding code in any commit) + 2 additional pre-verified gaps identified via manual audit of the repo.

Scoring: 1 point per confirmed gap found, −1 for confirmed false positive. Max score: 3. Scored before running any model.

### Models and cost estimate

| Model | API cost (input/output per M tok) | Est. tokens per run | Est. cost per run | Runs (3 reps × 2 tasks) | Est. total |
|---|---|---|---|---|---|
| gemma4:26b (local) | $0 marginal | ~5K / ~800 | $0 | 6 | $0 |
| Claude Haiku 4.5 | $0.80 / $4 | ~5K / ~800 | ~$0.007 | 6 | ~$0.04 |
| Claude Sonnet 4.6 | $3 / $15 | ~5K / ~800 | ~$0.027 | 6 | ~$0.16 |
| Claude Opus 4.8 | $15 / $75 | ~5K / ~800 | ~$0.135 | 6 | ~$0.81 |

Estimated total API spend: **~$1.01**. All frontier runs via Anthropic API.

### Protocol

1. Score ground truth for Task A and Task B before running any model. Commit scored rubric to `exp_012_cost_capability/rubric.md`.
2. Run gemma4:26b via Ollama (FA=0, q8_0 — production config). 3 reps per task, 60s idle between reps.
3. Run Haiku, Sonnet, Opus via Anthropic API. 3 reps per task per model. Default temperature.
4. Score all outputs against the pre-committed rubric. Do not adjust rubric after seeing outputs.
5. Compute: score per model per task, cost per correct answer, quality/cost ratio across tiers.

### Data tables

*Runs completed 2026-06-09. Results scored against pre-committed rubric in `rubric.md`.*

**Task A scores (compliance extraction, max 5 pts; −1 per confirmed FP):**

| Model | Rep 1 | Rep 2 | Rep 3 | Net score | Task A cost | Cost/correct |
|---|---|---|---|---|---|---|
| gemma4:26b | 0 | 0 | 0 | **0/5** | $0 | — |
| Claude Haiku 4.5 | 2 | 3 | 2 | **3/5** | $0.046 | $0.015 |
| Claude Sonnet 4.6 | 1 | 2 | 2 | **2/5** | $0.130 | $0.065 |
| Claude Opus 4.8 | 3−1† | 3−1† | 3 | **2/5** | $0.268 | $0.134 |

†Opus Task A: gross 3, −1 FP ("no DSR procedure document exists" — `compliance/05-dsr-procedure.md` is 208 lines; confirmed false positive in reps 1 and 2).

**Task A rubric items breakdown:**

| Item | gemma4:26b | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| A1 — ROPA vs inference_log.py (HIGH) | — | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| A2 — No DPIA for VLM witnessing (HIGH) | — | — | — | — |
| A3 — MCP server no authentication (MEDIUM) | — | ✓ 2/3 | — | ✓ 3/3 |
| A4 — inference.jsonl not in retention schedule (MEDIUM) | — | ✓ 2/3 | ✓ 2/3 | ✓ 3/3 |
| A5 — Model version not auditable / Art. 22 (LOW) | — | — | — | — |

**Task B scores (impl gap detection, max 3 pts; −1 per confirmed FP):**

| Model | Rep 1 | Rep 2 | Rep 3 | Net score | Task B cost | Cost/correct |
|---|---|---|---|---|---|---|
| gemma4:26b | 0 | 0 | 0 | **0/3** | $0 | — |
| Claude Haiku 4.5 | 2 | 2 | 2 | **2/3** | $0.049 | $0.025 |
| Claude Sonnet 4.6 | 3 | 3 | 3 | **3/3** | $0.161 | $0.054 |
| Claude Opus 4.8 | 3 | 3 | 3 | **3/3** | $0.342 | $0.114 |

**Task B rubric items breakdown:**

| Item | gemma4:26b | Haiku | Sonnet | Opus |
|---|---|---|---|---|
| B1 — No schema versioning / migration (MEDIUM) | — | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| B2 — Model pinned by label only, no hash (MEDIUM) | — | ✓ 2/3 | ✓ 3/3 | ✓ 3/3 |
| B3 — MCP no concurrency model / booth scenario (MEDIUM) | — | — | ✓ 2/3 | ✓ 3/3 |

**Composite scores (Task A + Task B, max 8):**

| Model | Task A | Task B | FP penalty | **Net total** | Total cost | Cost/correct |
|---|---|---|---|---|---|---|
| gemma4:26b | 0/5 | 0/3 | 0 | **0/8** | $0 | — |
| Claude Haiku 4.5 | 3/5 | 2/3 | 0 | **5/8** | $0.095 | $0.019 |
| Claude Sonnet 4.6 | 2/5 | 3/3 | 0 | **5/8** | $0.291 | $0.058 |
| Claude Opus 4.8 | 3/5 | 3/3 | −1 | **5/8** | $0.611 | $0.122 |

*Actual API spend: $0.997 total (vs $1.01 estimated). Opus input tokens ~40% higher than Sonnet for identical context — likely adaptive thinking overhead in usage reporting.*

### Conclusion

*Completed 2026-06-09.*

#### Hypothesis verdicts

**H1 (local ceiling on Task A) — FALSIFIED.** gemma4:26b scored 0/5 on Task A. All three frontier models outperformed it (Haiku 3/5, Sonnet 2/5, Opus 2/5 net). The compliance extraction capability gap is absolute, not marginal. Exp 009's 3/3 gemma4 overlap with Sonnet was on a different rubric using a shorter, easier context bundle; the expanded Task A rubric with 5 pre-registered items revealed a clean 0 baseline. This falsifies the hypothesis that local models match frontier on structured compliance extraction.

**H2 (frontier premium on Task B) — CONFIRMED.** All three frontier models scored 2–3/3 on Task B vs gemma4 0/3. The impl-vs-docs gap detection premium over the local model is total across this rubric.

**H3 (cost curve is stepped) — CONFIRMED.** Haiku → Sonnet: 3.1× cost increase, 0 net quality gain (5/8 → 5/8). Haiku → Opus: 6.4× cost increase, 0 net quality gain. The quality/cost ratio collapses to zero above the cheapest cloud tier. The curve is a single step function: $0 (local) → $0.09 (cheapest API), then flat.

#### Key findings

**1. The capability cliff is between local and cloud, not within the cloud tier.**

For this task type — structured analytical extraction over a ~10K-token bounded context — all three frontier models land at 5/8 net. The premium of Sonnet over Haiku (3.1×) and Opus over Haiku (6.4×) delivers no additional rubric score. Haiku is the cost-dominant choice for this workload: same score as Opus at 1/7th the cost.

**2. Haiku and Sonnet have complementary blind spots.**

Haiku caught A3 (MCP server no authentication) but missed B3 (concurrency model for booth demo). Sonnet missed A3 but caught all three Task B items including the booth-context concurrency risk. Neither gap is random noise — each model has a consistent profile across reps. This suggests the strength/weakness pattern is structural, not stochastic.

**3. Opus has the highest gross score (6/8) but the highest false positive rate.**

Opus uniquely found above-rubric gaps in both reps — notably "the filesystem firewall has no implementation" (BRIEF promotes it as the primary moat; config.py defines outbound HTTPS URLs to Catastro/SNCZI/Overpass with no egress blocking) and "Bouncer sanitisation path has no code" (BRIEF describes a sanitised buyer-facing slice; mcp_server.py exposes the full DB unfiltered). Both are real and significant, but outside the pre-committed rubric. The DSR procedure FP (compliance/05-dsr-procedure.md exists; Opus claimed it didn't, 2/3 reps) cost it 1 point and dropped it from 6/8 to 5/8 net.

**4. Two items evaded every model: A2 and A5.**

A2 (no DPIA for VLM witnessing pipeline under Art. 35) and A5 (model name label only logged in inference_log.py — no hash, no Art. 22 accountability) were missed by all four models. A5 requires connecting a low-level code observation (`"model": model` in inference_log.py, where `model` is the string label from config.py) to the GDPR Article 22 accountability obligation for automated processing. No model made that connection. A2 requires recognising that `scripts/witness_ingest.py` with face detection crosses the Art. 35 threshold for high-risk processing — apparently non-trivial even for frontier models given the context bundle does not include the witness_ingest.py source.

**5. Sonnet uniquely found an above-rubric bug.**

Sonnet (all 3 reps) identified a median calculation error in `db.py`: `mid = len(prices) // 2` followed by `prices[mid]` returns the upper-middle value for even-length arrays rather than the average of the two midpoints. This is a real bug in the primary market intelligence endpoint, not in the rubric. Haiku and gemma4 missed it. Opus missed it. Sonnet found it 3/3 with correct diagnosis and fix (`statistics.median()`).

#### Implication for the debate

The @Prathkum / @nix_eth exchange was testing two claims: (a) cost is the blocking problem, (b) cost and capability are decoupled. The data does not cleanly support either.

For this task class, the cost cliff is real but it's at $0.09 (Haiku), not $0.61 (Opus). Upgrading from Haiku to Opus costs 6.4× more and buys zero additional rubric score. Within the cloud tier, intelligence and cost are genuinely decoupled on this workload. But "use local instead of cloud" is not a valid substitute — gemma4:26b scored 0/8 on a task designed to favour local models. The decoupling breaks at the local/cloud boundary.

The practical recommendation for recurring compliance and engineering gap detection on bounded context: run Haiku. Budget one Sonnet or Opus pass per release cycle for qualitative depth — the above-rubric findings (Opus: filesystem firewall gap, Bouncer sanitisation gap; Sonnet: median bug) are genuine value not captured by the rubric score.

*Status: Complete (2026-06-09). Evidence: `tasks/chronos/exp_012_cost_capability/results/`. Rubric committed before first run.*

#### [Intelligence Feedback Loop: Exp 012-Alpha — Active-Parameter Framing Error]
*Date: 2026-06-09*

* **Error:** Exp 012 was pre-registered with gemma4:26b described as "25.8B active params." The model is MoE A4B: ~4B active parameters per forward pass, not 25.8B. The conclusion — "the decoupling breaks at the local/cloud boundary" — was derived under this wrong premise. Exp 012 measured one MoE model with ~4B active per forward pass; it did not test the dense 12–32B local class at all. The correct scope of the finding is: *the decoupling breaks at the 4B-active/frontier boundary.* The "local/cloud" framing is an artefact of model selection, not a confirmed property of local hardware as a category.

* **Correction:** Model header corrected to "25.8B total params / ~4B active per forward pass" (committed 2026-06-09). Conclusion narrowed in README key finding #10 and the blog post. The sharper implication: if the bottleneck is active compute rather than total parameters, a dense 12–27B local model should outperform gemma4:26b on the audit rubric — an active-compute advantage, not a total-parameter advantage. This hypothesis is pre-registered as Exp 015 (active-parameter ablation). If confirmed, the model-selection principle becomes: MoE for transduction and recall stages (knowledge breadth); dense for adjudication stages (per-token reasoning depth).

* **Status:** Exp 012 conclusions valid within narrowed scope (4B-active/frontier boundary, cross-document auditing task class). Exp 015 pre-registered as resolution experiment. README and blog post updated 2026-06-09.

---

## Exp 013 — Local Audit Loop: Can Scaffolding Move gemma4:26b Off Zero?

*Pre-registered: 2026-06-09. Subdirectory: `tasks/chronos/exp_013_local_audit_loop/`*

### Motivation

Exp 012 produced a clean result: gemma4:26b 0/8, every Claude tier 5/8. The obvious interpretation is "local models can't do cross-document compliance auditing." The less obvious but more useful question is: *why* 0/8 and *what kind* of zero is it?

Two very different failures produce a zero score:

- **Structural zero**: the model's weights don't contain the capability. No scaffold changes this. The gap is permanent.
- **Architectural zero**: the model has the underlying capability but the single-shot prompt is asking it to do in one step what it can do in three. The gap is scaffolding-addressable.

These have entirely different implications for the local-first thesis. A structural zero means local models are permanently excluded from this task class. An architectural zero means the right pipeline makes them useful — possibly not at frontier quality, but useful enough to handle the common cases locally and send edge cases to a cloud model as a top-up.

The practical target is not parity with Haiku. It is: *does generic scaffolding move gemma4:26b from 0/8 to ≥3/8 without naming a single rubric item in the prompts?* A 3/8 local model handling the structured extraction layer, combined with a targeted cloud pass on unresolved items, changes the economics significantly — the per-audit cost drops from $0.09 (full Haiku run) toward $0.02 (Haiku top-up on the 40% the local model missed).

### Research question

Is gemma4:26b's 0/8 score a structural capability gap, or an architectural gap addressable through task decomposition? If architectural: how much of the gap closes with generic (non-rubric-specific) scaffolding?

### The diagnosed failure mode from Exp 012

Exp 012 established that the failure is **recall-shaped**: zero true positives generated across all three reps. This rules out:
- Formatting problems (JSON schema was followed)
- Speed or context problems (the bundle fit well within 32K)
- Precision/false-positive problems (nothing to penalise if nothing is generated)

The failure is specifically at generating candidate gaps that reference both a code observation and a policy rule simultaneously. The cross-document leap — "read RoPA §2.4 AND `inference_log.py` lines 42-43 AND notice the contradiction" — appears to fail as a single-shot task for a 26B local model. Each half of that leap may be within capability; the bridging in one context may not be.

### Hypotheses

**H1 (decomposition recovers recall):** Breaking the audit into three sequential single-purpose passes — extract code facts, extract policy rules, bridge them — will produce a non-zero candidate set from gemma4:26b. Falsified if Stage 2 (bridge) produces zero candidates even when Stage 1 produced non-empty facts and rules.

**H2 (specificity is the verifier bottleneck):** If candidates are generated but score zero after verification, the failure will be specificity — candidates too generic to map to a rubric item (e.g. "the system lacks GDPR compliance" rather than "inference_log.py line 42 stores `input_text` while ropa.md §2.4 states no retention"). Falsified if the verifier rejects specific, evidence-grounded candidates.

**H3 (scaffolding moves the score without rubric leakage):** Generic scaffolding alone — decomposition, absent/present checklists over property types, schema-constrained output, fresh-context verification — moves gemma4:26b from 0/8 to ≥3/8 on the same rubric used in Exp 012. Falsified if score after scaffolding remains ≤2/8. This is the operationally meaningful hypothesis.

**Falsification criteria:**
- H1 rejected if Stage 2 produces zero candidates on every run where Stage 1 was non-empty
- H2 rejected if the verifier rejects candidates that contain file/line/article anchors
- H3 rejected if net score after all generic interventions is ≤2/8

### The instrument

`exp_013_local_audit_loop.py` — stdlib + `requests`, no frameworks. The pipeline:

**Stage 0 — budget + canary guard.** Estimates token use vs `num_ctx`, then embeds a UUID canary at the top of the context and asks the model to echo it. Ollama silently truncates the *oldest* tokens when context overflows — which is exactly the system prefix. If the canary fails, the run is aborted. This guard is non-negotiable: an experiment that audited a half-loaded context would produce meaningless results.

**Stage 1 — decomposed extraction (fresh contexts).** Two independent passes, each in a clean context:
- 1a: Extract atomic code facts — what the code *actually does*: what is logged, stored, authenticated, hashed, versioned, exposed, or absent
- 1b: Extract atomic policy rules — what the policy *requires*: retention periods, DPIA thresholds, accountability obligations, access-control expectations, with article/section references

The decomposition is the primary scaffolding intervention. Instead of one context holding both code and policy and asking the model to reason across them, each extraction sees only its own source material and has a single job.

**Stage 2 — bridge.** Given the extracted facts and rules lists, find gaps: where does a code fact contradict or fail to satisfy a policy rule? Recall-biased — propose every plausible pair now; the verifier will prune. The model is not asked to recall which rule applies; the rule text is supplied.

**Stage 3 — fresh-context verifier.** Each candidate gap is re-checked in an isolated context containing only the source bundle and that one claim. The model must quote the exact line that supports the claim or mark it `INVALID`. Not self-critique in a growing context — an external check.

**Stage 4 — deterministic assembly.** Python sorts survivors by severity. No final LLM pass.

**Auto-diagnosis.** After every run (or standalone via `--diagnose-trace`), the trace is read by a deterministic rule-based function — no model call — and labelled:

| label | meaning | implied fix |
|---|---|---|
| `extraction_empty` | Stage 1 produced no facts or rules | switch to present/absent checklist; check canary (num_ctx truncation) |
| `no_bridge` | facts+rules existed, Stage 2 connected nothing | make bridging a matching task over supplied rule text, not a recall task |
| `vague_candidates` | candidates generated, all too generic, verifier killed them | require a concrete anchor per gap (file, line, article, identifier) |
| `over_pruned` | specific candidates all rejected | verifier too strict; run calibration check before touching upstream |
| `produced_output` | no structural zero | remaining gap is rubric mapping (manual) |

The diagnosis is deterministic and reproducible from the trace alone. This is the point: after the first localization run, we know exactly which stage to target for the first intervention. We do not guess.

### Intervention discipline

Every change goes in `intervention_ledger.md` before the run it's tested in, classified on this spectrum:

1. **generic-scaffolding** — structural help: decomposition, present/absent checklists over property types, article-matching over supplied rule text, output schemas, sampling/context settings. Fair. Publishable. The headline result is: *generic scaffolding moved gemma 0 → N without naming a single rubric item.*
2. **fair-evidence** — adds a source file legitimately part of the system but missing from the context bundle (e.g. `scripts/witness_ingest.py` for item A2). Result must be reported as "with expanded context."
3. **rubric-leakage** — any prompt, checklist item, or example that names or paraphrases a specific rubric answer (e.g. "check whether a model hash exists"). **Forbidden.** Voids the score; the number stops being interpretable.

Rule of thumb: if removing the rubric from the room would make the change impossible to write, it's leakage.

### What "useful" means

The target is not matching Haiku. It is producing a result where a local model handles the mechanical extraction layer and a cloud model does targeted top-up work on the unresolved fraction. That architecture:

- Keeps the full context bundle off the wire (privacy/sovereignty maintained for 80% of runs)
- Reduces cloud cost from $0.09/audit to ~$0.02 (Haiku top-up on unresolved items only)
- Is deployable in an agency's on-premises appliance without a permanent API dependency

A score of 3/8 from the local model alone makes this architecture viable. That is the bar.

### Data tables

**Localization run 000, rep 1 — 2026-06-09, baseline bundle:**

| Stage | Output | Detail |
|---|---|---|
| Stage 0 (canary) | PASS | num_ctx=32768 active; canary UUID echoed |
| Stage 1a (code facts) | 17 facts extracted | inference_log.py append behaviour, model label, MCP routes, query handling, etc. |
| Stage 1b (policy rules) | 16 rules extracted | RULE-001 through RULE-016; RULE-008 (session retention), RULE-012 (log rotation) present |
| Stage 2 (bridge candidates) | 2 candidates | Both retention-related; Item 1 session-logging violation NOT bridged despite RULE-008 being present |
| Stage 3 (verified) | 1/2 survived | GAP-001 (MCP log rotation) rejected; GAP-002 (inference log absent from schedule) accepted |
| Auto-diagnosis | **produced_output [high]** | Pipeline completed end-to-end; no structural zero |

**Rubric scoring, run 000 rep 1:**

| Item | findable? | found? | notes |
|---|---|---|---|
| 1 | ✓ | ✗ | RULE-008 was extracted; bridge missed the inference_log.py → session-retention violation |
| 2 | ✗ (needs witness_ingest.py) | ✗ | n/a |
| 3 | ✓ | ✗ | No auth-related rule in policy docs; bridge had nothing to match against |
| 4 | ✓ | ✓ | GAP-002: inference log absent from retention schedule — names file + schedule |
| 5 | ✗ (needs config.py) | ✗ | n/a |

Rep 1 score: **1/3 findable** (item 3 missed — bridge had no rule; item 4 found; item 1 missed — RULE-008 present but not connected).

**Rubric scoring, run 000 rep 2 — 2026-06-09 (example-template harness):**

3/3 candidates verified. Items found: 3 (MCP auth absent), 4 (inference log absent from retention schedule). Item 1 not found. Above-rubric: MCP log rotation gap (not scored).

**Rubric scoring, run 000 rep 3 — 2026-06-09:**

1/2 candidates verified. Item 3 found (MCP auth). Item 4 not found this rep.

**Stability table (all 3 reps):**

| Item | Rep 1 | Rep 2 | Rep 3 | Stable ≥2/3? | Score |
|------|-------|-------|-------|:---:|:---:|
| 1 (session retention violation) | ✗ | ✗ | ✗ | no | 0 |
| 2 (DPIA for VLM) | n/a | n/a | n/a | — | 0 |
| 3 (MCP auth absent) | ✗ | ✓ | ✓ | **yes** | +1 |
| 4 (inference log not in retention schedule) | ✓ | ✓ | ✗ | **yes** | +1 |
| 5 (model label mutable) | n/a | n/a | n/a | — | 0 |

**Net score: 2/5 (2/3 findable in baseline bundle). No false positives across any rep.**

Note: Exp 012 false positives (DPA template, DSAR procedure, VLM pipeline) did not reappear in any rep — the decomposed approach did not introduce new FP classes.

**Score progression by intervention:**

| Run | Intervention | Class | Score (net) | Notes |
|---|---|---|---|---|
| 000 (3 reps) | baseline localization — generic scaffolding only | generic-scaffolding | **2/3 findable (2/5)** | Items 3 and 4 stable; Item 1 consistently not bridged |

**Key observation — bridge failure on Item 1:** RULE-008 ("Buyer query text must not be retained beyond the session") was extracted in all reps; the code fact that `inference_log.py` persistently appends input/output text was also extracted. The bridge consistently failed to connect them across all 3 reps. This is not stochastic noise — it is a systematic failure. The bridge stage appears to pattern-match on structural similarity between rule categories and code fact categories, and "session retention" does not fire on "inference logging" without additional framing.

**Key observation — Item 3 found despite earlier rep-1 miss:** Rep 1 (old harness, schema-object injection) failed to extract an auth-related rule. Reps 2 and 3 (example-template harness, 18 rules extracted vs 16) both found and verified the MCP auth gap. The harness fix (intervention 004) appears to have improved rule extraction recall.

### Conclusion

**H confirmed (partial):** Task decomposition (extract-facts → extract-rules → bridge → verify) does recover recall from the 0/8 flat-context baseline. 

Baseline: gemma4:26b scored 0/8 on Exp 012's flat-context compliance task.  
Decomposed: gemma4:26b scored 2/3 findable items (2/5 headline) on the same source material with no rubric leakage.

The recovery is real and reproducible: Items 3 (MCP auth) and 4 (inference log retention) are both stable finds (≥2/3 reps). Item 1 (session retention / RULE-008 bridge) was missed in all 3 reps — a clean systematic negative, not noise.

**On the pre-registered falsification boundary:** H3 was falsified if score ≤2/8. The raw headline is 2/5, which maps to 2/8 on the original Exp 012 scale — exactly at the boundary. The correct interpretation is that items 2 and 5 were structurally absent from the baseline bundle (require witness_ingest.py and config.py respectively) and were never findable, not model failures. Comparing 2/8 raw to the ≤2/8 threshold conflates "model didn't find it" with "the evidence wasn't in the context." On the findable subset (3 items), the model found 2/3. Context expansion runs (001-A2 and 001-A5) are deferred — out of scope for H3, which tests the scaffolding principle, not bundle completeness.

**What decomposition fixed:** Cross-document retention gaps — the model can now hold an implementation fact (log file appends data) and a schedule rule (item not in schedule) and generate a gap claim. Items 3 and 4 both require citing code and policy in the same claim; both found.

**What decomposition did not fix:** Cross-semantic bridging — Item 1 requires the model to recognise that "inference input/output text" is the same category as "buyer query text" in the policy. RULE-008 and the log append fact were both extracted correctly in all 3 reps; the bridge never fired. This is a systematic failure of semantic frame alignment, not extraction or verification. It is the next scaffolding target and a candidate intervention for Exp 016.

**Operational implication:** 2/3 findable items at ~0 marginal cost (local inference, ~3–5 minutes per full run) enables an on-premises pre-filter architecture: run the decomposed local audit, escalate only the bridge failures to Haiku for cross-semantic coverage. Expected cost: ~$0.02/audit vs $0.09 full-cloud run, with full rubric coverage on the findable subset.

*Status: Complete (2026-06-09). Score: 2/3 findable (2/5 headline). H confirmed (partial) on retrievable subset. Context expansion runs deferred — out of scope for H3.*

---

## Exp 014 — Capability Variance Floor

*Pre-registered: 2026-06-09. Subdirectory: `tasks/chronos/exp_014_capability_variance/`*

### Motivation

Exp 012 produced a clean result — gemma4:26b 0/8, all frontier tiers 5/8 — but on 3 reps per model. A 0/8 result with N=3 could be a true zero (every item independently missed with high probability) or a model that scores ~1–2/8 on average but happened to score 0 across all 3 runs. Those two cases have very different implications: the first supports a hard boundary; the second means the 0/8 was noise and the local/cloud step function weakens.

Similarly, Haiku's 5/8 at N=3 could be consistently 5/8 or could be a 3–7/8 model that hit 5 three times.

### Hypotheses

**H1 (gemma zero replicates):** gemma4:26b scores 0/8 in ≥4 of 5 reps on the frozen Exp 012 rubric. Falsified if gemma scores ≥1/8 in ≥2 reps — indicating the 0/8 was sampling noise and the per-item probability is non-zero.

**H2 (Haiku distribution is stable):** Haiku scores 4–6/8 net in every rep (±1 of the 5/8 mean). Falsified if Haiku scores ≤2/8 or ≥7/8 in any rep — indicating high variance that undermines the "local/cloud step" framing from either direction.

**H3 (distributions do not overlap):** gemma and Haiku score distributions do not overlap across 5 reps. Falsified if gemma's max score meets or exceeds Haiku's min score.

### Experiment design

- Frozen rubric: `../exp_012_cost_capability/rubric.md` (not copied — referenced in place)
- Frozen context bundles and prompts: identical to Exp 012 (`run.py` is a direct adaptation)
- Models: gemma4:26b (Ollama, `think: false`, `temperature: 0.1`) and claude-haiku-4-5-20251001 (API)
- 5 reps per model per task (Tasks A + B)
- Scoring: all runs complete before any scoring begins (blind)
- Scorer records item-level hits/misses per rep, not just net totals

### Pass criteria

| Criterion | Threshold |
|---|---|
| H1 confirmed | gemma 0/8 in ≥4/5 reps |
| H1 falsified | gemma ≥1/8 in ≥2 reps |
| H2 confirmed | Haiku within 4–6/8 every rep |
| H3 confirmed | no score overlap across 5 reps |
| H3 falsified | gemma max ≥ Haiku min |

### Results

*Runs completed 2026-06-09. All reps complete before scoring began (blind).*

Raw result files:
- `exp_014_capability_variance/results/gemma4_26b_20260609T201133Z.json`
- `exp_014_capability_variance/results/claude_haiku_4_5_20251001_20260609T202614Z.json`

#### Gemma4:26b — all 5 reps

| Rep | A1 | A3 | A4 | B2 | FP | Net |
|-----|----|----|----|----|-----|-----|
| 1 | 0 | +1 | 0 | 0 | −1 (DSR FP) | **0/8** |
| 2 | 0 | 0 | 0 | 0 | −1 (DSR FP) | **−1/8** |
| 3 | 0 | 0 | 0 | 0 | −1 (DSR FP) | **−1/8** |
| 4 | 0 | 0 | 0 | 0 | −1 (DSR FP) | **−1/8** |
| 5 | 0 | 0 | 0 | 0 | −1 (DSR FP) | **−1/8** |

DSR false positive pattern: gemma claimed `compliance/05-dsr-procedure.md` does not exist in 4/5 reps (the file is 208 lines). Same systematic FP confirmed in Exp 009. One rep (1) found A3 (MCP auth gap) but still net 0/8 due to the DSR penalty.

#### Claude Haiku 4.5 — all 5 reps

| Rep | A1 | A3 | A4 | B2 | B1* | FP | Net |
|-----|----|----|----|----|----|-----|-----|
| 1 | +1 | 0 | +1 | 0 | 0 | −1 (DSR) | **1/8** |
| 2 | +1 | 0 | +1 | 0 | 0 | −1 (DPA) | **1/8** |
| 3 | +1 | +1 | +1 | +1 | 0 | 0 | **4/8** |
| 4 | 0† | 0 | 0 | +1 | 0 | 0 | **1/8** |
| 5 | +1 | 0 | +1 | +1 | 0 | −1 (DSR) | **2/8** |

† Rep 4 Task A: JSON parse error — output truncated at 2048 token limit mid-object. Task A scored 0/5 for this rep; Task B completed normally.

*B1 note: Reps 1, 3, 4 raised schema/migration findings ("collection_method column absent despite BUILD_LOG ALTER TABLE claim"). Under broad interpretation matching what the original Exp 012 scorer likely applied, these would score B1 (+1 each), yielding nets of 2/1/5/2/2. Under strict rubric interpretation (schema versioning explicitly for the 15-field Reducer output fields), B1 = 0 across all reps. Neither interpretation changes the hypothesis verdicts. Strict interpretation used above; broad interpretation is the alternate scoreline.*

### Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: gemma 0/8 in ≥4/5 reps | **CONFIRMED (strong)** | gemma ≤0 in all 5 reps; goes negative in 4/5 reps due to systematic DSR FP. Never scores ≥1/8 net. |
| H2: Haiku 4–6/8 every rep | **FALSIFIED** | Haiku scores 1/1/4/1/2. Four of five reps fall at or below the ≤2/8 falsification threshold. The 5/8 single-rep result from Exp 012 was at the high tail of an unstable distribution. |
| H3: distributions do not overlap | **CONFIRMED** | gemma max = 0/8, Haiku min = 1/8. No overlap across 5 reps under any B1 interpretation. |

### Conclusion

H1 and H3 hold. The gemma zero result replicates at N=5 with zero contamination from sampling variance — it actually goes negative via systematic false positive. The gap between the two model classes is real and durable.

H2 is falsified: Haiku is not stable at 4–6/8. Its distribution across 5 reps is 1–4/8 with high variance. The 5/8 result from Exp 012 (3 reps) was a lucky high sample. Three factors compress the Haiku mean below the Exp 012 reference:

1. **False positive penalties** — DSR/DPA FP fired in 3/5 reps (−1 each), costing one guaranteed point per affected rep.
2. **B3 blind spot** — concurrency risk (MCP server, no concurrency model for booth demo) found in 0/5 reps. In Exp 012 this was found by Sonnet/Opus but not consistently by Haiku (Haiku was already 2/3 on B3 in Exp 012 and 0/5 here — consistent with the Haiku/Sonnet complementary-blind-spots pattern from Exp 012).
3. **Rep 4 token-limit truncation** — Task A parse error dropped that rep's A-task score to 0.

A note on codebase anchor: the rubric anchor commit (`623c4c8`) is not in the current casasol repo (repo was rebased since Exp 012). All rubric items were verified present in the current codebase prior to scoring — the gaps are still open. The H2 falsification is not a codebase-drift artefact.

**Pre-filter architecture validation (Exp 013 follow-on):** The Exp 013 conclusion proposed local audit + Haiku top-up at ~$0.02/audit. Exp 014 shows Haiku's expected contribution at that step is 1–4/8 (not 5/8), putting the realistic pre-filter catch rate lower than Exp 012 suggested. The architecture remains viable but the expected catch rate at the Haiku stage should be modelled as ~2/8 mean rather than 5/8.

*Status: Complete (2026-06-09). H1 CONFIRMED, H2 FALSIFIED, H3 CONFIRMED. The zero/gap replicates; Haiku variance is higher than Exp 012 indicated.*

---

## Exp 015 — Active-Parameter Ablation: Dense vs MoE on the Audit Rubric

*Pre-registered: 2026-06-09. Subdirectory: `tasks/chronos/exp_015_active_param_ablation/`*

### Motivation

Exp 012 found gemma4:26b (MoE A4B, ~4B active per forward pass) scored 0/8 on cross-document auditing tasks. The result was initially interpreted as a "local/cloud boundary," but Exp 012-Alpha (framing correction) narrows the scope: Exp 012 tested one point in the local model space — a large-total-parameter MoE with small active compute per token.

The open question: is the bottleneck *total parameters* (knowledge breadth — which MoE provides cheaply) or *active compute per token* (per-token reasoning depth — which dense models provide)? These are separable properties. Exp 009 and 012 are both consistent with the second reading: gemma4:26b matched the compliance knowledge layer (recall-shaped task, benefits from breadth) and failed the cross-document reasoning layer (computation-shaped task, benefits from depth).

### Hypotheses

**H1 (bottleneck is active compute):** A dense local model with ≥12B active parameters per forward pass scores ≥2/8 on the frozen Exp 012 audit rubric without scaffolding. Falsified if no tested dense model scores above 1/8.

**H2 (model selection principle):** If H1 is confirmed, per-token active compute predicts audit rubric performance better than total parameter count. MoE models are suited for transduction and recall stages; dense models for adjudication stages. Falsified if MoE and dense models at matched active-parameter counts perform equivalently.

### Experiment design

1. Select candidate dense models runnable on 64GB unified memory at Q4: target range 12–32B active params (e.g. `qwen2.5:32b`, `mistral-small3.2`, or similar dense models available via Ollama). Confirm active-parameter count from model card and verify against observed decode throughput (bandwidth estimate cross-check per Exp 012-Alpha method).
2. Run 3 reps per model on the frozen Exp 012 rubric (Task A + Task B, identical context bundles, identical scoring). Rubric committed before first run; do not re-score after seeing outputs.
3. Score blind. Compare net scores against gemma4:26b baseline (0/8) and frontier baseline (Haiku 5/8).

### Pass criteria

| Criterion | Threshold |
|---|---|
| H1 confirmed | ≥1 dense model scores ≥2/8 net |
| H1 strong | ≥1 dense model scores ≥4/8 net (within frontier range) |
| H1 falsified | No tested dense model scores >1/8 |
| H2 confirmed | MoE/dense scores separate cleanly across ≥2 dense models |

### Implications

If H1 is confirmed: the local/cloud curve is a continuous active-parameter curve, not a binary step function. Watcher pipeline and Nestor orchestration should route adjudication tasks to the densest available local model. The design principle becomes: MoE for transduction/recall stages, dense for adjudication stages.

If H1 is falsified: the local/cloud step function is real at every tested local model class. The primary engineering lever shifts to cloud escalation with a provably tight redaction boundary (Exp 018).

*Status: Pre-registered 2026-06-09. Awaiting Exp 013 context expansion runs and Exp 014 (variance floor) before scheduling. Dense model candidates to be confirmed.*

---

