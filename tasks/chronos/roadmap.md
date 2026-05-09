# Project Chronos: Roadmap

## Objective
To build and maintain a public-facing blog written from the perspective of Nestor (the local AI butler), utilizing the scientific method to document evolution, infrastructure, and intelligence growth.

## Content Pillars
1. **The Silicon Sentinel** (Infrastructure & Privacy)
2. **The Intelligence Feedback Loop** (AI Evolution & Self-Correction)
3. **The Human-Machine Interface** (HCI)
4. **The Observer's Ledger** (Scientific Method & Benchmarks)

## Milestones
- [x] Initialize project structure
- [x] Setup static site generator (SSG) configuration
- [x] Configure deployment pipeline (European hosting/privacy-centric)
- [x] Define first scientific experiment (Verification of Veracity)
- [x] Draft and finalize post: "The Genesis of Chronos"
- [x] Execute and log Latency Benchmark (Experiment 002)
- [x] Draft and finalize post: "The Control Plane and the Data Plane: Managing the AI Thinking Tax"
- [x] Establish the `scientific_log.md` verification ledger
- [x] Implement Nestor-authored Git commits for verified task execution
- [x] **Execute Experiment 003:** Validated Anonymized Adversarial Memory (Data Sovereignty)
- [x] **Incident 003-Alpha Resolution:** Identified the 25k–35k token "Memory Cliff" for Gemma 4 26B on M4 Pro.
- [x] **Drafted/Published:** "Every Company Can Be a Palantir Now" (Data Sovereignty thesis).
- [x] **Exp 005 (Phase 0):** Build and validate "Dicer Describer" Cascade (Health Corpus).
- [x] **Incident 005-Alpha:** Root-cause schema-meaning gaps and "zombie" prefill runners.
- [x] **Exp 005 (Phase 0):** Build and validate "Dicer Describer" Cascade (Health Corpus).
- [x] **Incident 005-Alpha:** Root-cause schema-meaning gaps and "zombie" prefill runners.
- [x] **Exp 005 Phase 0 closed (2026-05-02):** Architectural pattern documented in [`cascade_pattern.md`](./tasks/chronos/exp_005_dicer_describer/cascade_pattern.md). Three working behaviours demonstrated end-to-end (single-slice trend grounding, workouts with cliff-aware coarsening, clarifying-question protocol); first demand-signal evidence surfaced (fencing as personal-signature activity).
- [x] **Orchestrator Hardening:** `aggregation_level` honoured for workouts (yearly coarsening fallback); per-slice operational caps in `extract.py`; bundle-level token guard at 22K in `cascade.py` (ADR-002).
- [x] **Streaming Describer:** `cascade.py` now streams with three-timeout separation (first-byte/idle/total), `--show-thinking` flag for live deliberation visibility.
- [ ] **Exp 006 — Redactor Fidelity Test:** Pre-registered 2026-05-09. Harness built; execution pending.

## Current Focus

Phase 0 of Experiment 005 closed 2026-05-02 with a working cascade and an architectural pattern documented in [`cascade_pattern.md`](./tasks/chronos/exp_005_dicer_describer/cascade_pattern.md). Phase 1 design begins next, with three open questions to close before pre-registration: (1) measure the memory bandwidth cliff specifically on thinking-model generation, since Phase 0 hit a cliff at ~22K despite Incident 003's measurement suggesting 25K-35K; (2) finalise the frontier comparator's context shape (synthetic shadow corpus vs. aggregate-only); (3) design an N-of-M scoring rubric that captures synthesis variance separately from completion reliability. The cliff measurement is the prerequisite that makes the rest defensible.

## Technical Deployment Plan

- **Engine:** Hugo (Static Site Generator).
- **Workflow:** Nestor generates content via OpenClaw; symlinked to `~/REPOS/local-first-ai` for version control.
- **Cascade:** Dicer (`gemma4:e4b`) plans → Extractor (`json`) → Describer (`gemma4:26b`) synthesizes.
- **Hardware Invariant:** Prefill context must remain **< 22,000 tokens** to avoid memory-bandwidth saturation.
- **Strict Validation:** ADR-001 contract with code-fence normalization for unreliable structured output.

## Pending Tasks

### 🛠️ Infrastructure & Experimentation

- [ ] **Exp 006 — Redactor Fidelity Test (CasaSol GDPR Validation).** Pre-registered 2026-05-09. 20 synthetic toxic real estate notes × 8 GDPR data categories → automated + manual check for 0 leaks. Produces a result file linkable from the CasaSol OLÉ booth QR card. Harness at [`exp_006_redactor_fidelity/`](./exp_006_redactor_fidelity/). Planned post: *"The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks."*
- [ ] **Exp 005-Beta — Cliff measurement on thinking models.** Replicate [Incident 003](https://localfirstai.eu/posts/incident_003_alpha_post/)'s three-size prefill sweep, but with a thinking-model Describer where generation expands effective KV utilisation past the prefill estimate. Produces the empirical ceiling Phase 1 must respect.
- [ ] **Exp 005 Phase 1 — Cascade vs Claude Opus 4.7.** Pre-register against frontier on a query class drawn from the health corpus. Synthetic shadow corpus for parity-without-disclosure. Depends on Exp 005-Beta for ceiling, and on N-of-M rubric design.
- [ ] **N-of-M scoring rubric design.** Phase 0's three-RHR-run variance and the VO2/RHR cliff hit together imply single-shot scoring conflates quality variance with reliability. Methodology work; could be standalone or rolled into Phase 1 pre-registration.
- [ ] **Process Management:** Track upstream Ollama cancellation API or contribute a fix. As of 0.20.2, abandoned streaming requests wedge runners at ~900% CPU; recovery requires `ollama serve` restart. Phase 0 reliability ceiling: one cliff hit per restart.
- [ ] **Environment Fix:** Resolve `sudo -n killall powermetrics` inheritance for automated power-profile logging. (Carried forward.)
- [ ] **Exp 007 — Dicer alternative:** Evaluate **Qwen 2.5/3.5b** or similar as a more deterministic structured-output model for the Dicer role. Phase 0 confirmed `gemma4:e4b` ignores prose instructions in favour of fixture patterns; a stricter structured-output model may not require fixture-only constraint encoding.

### ✍️ Content Execution

- [ ] **The Cascade Pattern (transferable post):** Public writeup drawing from [`cascade_pattern.md`](./tasks/chronos/exp_005_dicer_describer/cascade_pattern.md) sections 6–9. The pattern as a response to the memory-bandwidth cliff, framed for readers who don't have the watch corpus. Closes the Shape B writeup deferred from Phase 0.
- [ ] **The Dicer's Dilemma:** Phase 0 narrative — how Nestor learned to "slice" 8 years of heartbeats without triggering a system timeout. The story version of cascade_pattern.md, focused on the cliff hit and the architectural response.
- [ ] **The Silicon Sentinel:** Update with "Memory Bandwidth Cliff" findings — visualizing why local-first AI is bound by the bus, not just the GPU.
- [ ] **Intelligence Feedback Loop:** Case study on **Incident 005-Alpha** (Dicer following few-shot patterns over prose instructions).
- [ ] **Technical Tutorial:** The "Anonymization & Dicing" pattern for local-first retrieval.
