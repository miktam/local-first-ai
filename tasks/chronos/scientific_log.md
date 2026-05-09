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

## Experiment 005 — Multi-Model Cascade (Dicer / Describer): Phase 0 (build)

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
**Status:** Pre-registered — not yet executed
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

