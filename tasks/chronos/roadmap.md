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

## Current Focus
Refining the **Dicer Describer** architecture. Following the 2026-05-01 "Workout Query" failure, the focus has shifted from prompt-engineering to **architectural honesty**. The immediate priority is moving defensive constraints (volume, timeout, and token-guards) out of LLM prompts and into the orchestrator logic.

## Technical Deployment Plan
- **Engine:** Hugo (Static Site Generator).
- **Workflow:** Nestor generates content via OpenClaw; symlinked to `~/REPOS/local-first-ai` for version control.
- **Cascade:** Dicer (`gemma4:e4b`) plans → Extractor (`json`) → Describer (`gemma4:26b`) synthesizes.
- **Hardware Invariant:** Prefill context must remain **< 22,000 tokens** to avoid memory-bandwidth saturation.
- **Strict Validation:** ADR-001 contract with code-fence normalization for unreliable structured output.

## Pending Tasks

### 🛠️ Infrastructure & Experimentation
- [ ] **Orchestrator Hardening:** Move `max_rows` and `aggregation_level` enforcement from Dicer prompts to hard-coded logic in `cascade.py`.
- [ ] **Context Watchdog:** Implement a pre-check that refuses or auto-downsamples Extractor output if token count exceeds 22k.
- [ ] **Process Management:** Investigate Ollama API for server-side cancellation to kill "zombie" processes during timeouts.
- [ ] **Environment Fix:** Resolve `sudo -n killall powermetrics` inheritance for automated power-profile logging.
- [ ] **Exp 006:** Evaluate **Qwen 2.5/3.5b** as a more deterministic alternative to `e4b` for the Dicer role.

### ✍️ Content Execution
- [ ] **The Dicer's Dilemma:** Draft a post on Experiment 005—how Nestor learned to "slice" 8 years of heartbeats without triggering a system timeout.
- [ ] **The Silicon Sentinel:** Update with "Memory Bandwidth Cliff" findings—visualizing why local-first AI is bound by the bus, not just the GPU.
- [ ] **Intelligence Feedback Loop:** Case study on **Incident 005-Alpha** (Dicer following few-shot patterns over prose instructions).
- [ ] **Technical Tutorial:** The "Anonymization & Dicing" pattern for local-first retrieval.
