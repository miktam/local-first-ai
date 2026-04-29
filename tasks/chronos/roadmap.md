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

## Current Focus
Transitioning from infrastructure setup to **Operating Envelope Management**. Following the H6 prefill scaling results, the immediate priority is codifying system limits to maintain high-throughput local operations and refining the multi-layer memory architecture.

## Technical Deployment Plan
- **Engine:** Hugo (Static Site Generator).
- **Workflow:** Nestor generates `.md` content and `scientific_log.md` updates. Verified changes committed via `--author="Nestor <nestor@chronos.local>"`.
- **Constraint:** Architecture must strictly honor the **<25,000 token prefill limit** to avoid super-quadratic performance degradation (Incident 003-Alpha).
- **Security Invariant:** `memory.py` must remain decoupled from `vocab_store.py` (verified via import graph checks).

## Pending Tasks

### 🛠️ Infrastructure & Experimentation
- [ ] **Establish "Complexity Ceiling" (Experiment 004):** Refine canary heuristics to distinguish genuine model uncertainty from prompt-compliance when probing anonymized vaults.
- [ ] **Implementation:** Automate a pre-emptive `/compact` watchdog in the orchestration layer that triggers when cumulative session context hits 20k tokens.
- [ ] **Environment Fix:** Resolve the `sudo -n killall powermetrics` environment inheritance issue for non-interactive test scripts.
- [ ] **Memory Management:** Implement "Compression-by-Archive" for Layer 2 daily summaries (decaying old summaries from ~300 to ~50 words).

### ✍️ Content Execution
- [ ] **The Intelligence Feedback Loop:** Draft a post detailing **Incident 003-Alpha** as a case study in AI hardware profiling and the transition from "bug hunting" to "performance envelope identification."
- [ ] **The Silicon Sentinel:** Draft a deep dive into the **M4 Pro thermal/power profile** using the wattage data gathered during the H6 Prefill Sweep.
- [ ] **Technical Tutorial:** Document the "Sovereign Memory" architecture (the separation of `memory.py` and `vocab_store.py`) for the LocalFirstAI community.
