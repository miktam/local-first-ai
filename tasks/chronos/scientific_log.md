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

---
