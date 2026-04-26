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
- Environment: Apple M4 Silicon, Gemma 4 26B, OpenClaw Orchestration.
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

- Environment: Apple M4 Silicon, Gemma 4 26B, OpenClaw Orchestration.
- Code: [`tasks/chronos/exp_003_local_memory/`](./exp_003_local_memory/).
- Public companion: [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/).
- Methodology note: orchestration and harness code drafted with Claude Opus 4.7 in a single afternoon. All execution — anonymisation, summarisation, queries, leak probes — ran locally on Gemma 4 26B. The frontier model never saw the corpus, the vault, or any results.

**What I observed:**
1. Cold-start added ~60s on the first query. Warm state was 16–18s for Layer 2 hits, ~60s for Layer 1 fallback.
2. The model gendered Naomi Reeves as female and reasoned about "her" consistently, so anonymisation rewrote the model's whole worldview, not just the names.
3. Citations to Layer 1 turn IDs (e.g. `[#0]`, `[#27]`) emerged unprompted from the fallback prompt structure. The model picked up the convention without being told — useful auditable property.
4. The Layer 2 summary maintained the anonymised vocabulary throughout. The model synthesised in sovereign terms, never reaching for training-data identifiers. Anonymization isn't just I/O hygiene; it rewrites the model's working world-model during inference. This is the most interesting outcome of the experiment, and the strongest evidence for the data-sovereignty thesis.

**Conclusion:** All three pre-registered pass criteria were met. The architecture defeated zero-shot source recognition (0/20), refused pseudonym-to-identity bridging under direct probing (0/3), and answered functional queries correctly from Layer 2 alone. The Palantir essay's data-sovereignty claim is now load-bearing on architecture, verifiable in `results/`. Limitations: this validation tested a single ~1500-word excerpt against a single local model, and the canary heuristic does not distinguish genuine uncertainty from prompt-compliance — both refinements scoped for Experiment 004.

---
