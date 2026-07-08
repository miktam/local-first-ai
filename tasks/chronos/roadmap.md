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
- [x] **Exp 005 Phase 0 closed (2026-05-02):** Router / Reducer cascade validated on 8-year Apple Watch corpus.
- [x] **Exp 006 — Redactor Fidelity Test:** Complete 2026-05-09. 0/20 × 8 categories — zero true-positive leaks.
- [x] **Exp 007 Phase A+B — The Silicon Wager:** Complete 2026-05-29. MBP gen t/s +200–370% vs Mini; cliff at ~45K vs ~18K.
- [x] **Exp 008/010 — Flash Attention cliff resolved:** FA=1 is the sole culprit. FA=0 has no cliff through 40K tokens.
- [x] **Exp 009 — Adversarial Critic:** gemma4:26b matches compliance layer; misses impl-vs-docs gap. FAIL as drop-in.
- [x] **Exp 011 — MLX runtime:** No cliff through 40K on MLX. Confirms cliff is Ollama/llama.cpp artefact, not hardware.
- [x] **Exp 012–014 — Cost/capability curve:** Step function confirmed at 4B-active/frontier boundary. Haiku cost-dominant.
- [x] **Exp 019 — Adversarial Legal Review Pipeline:** Complete 2026-06-24. 0/7 claims survived unchanged. Adversarial 3-panelist pipeline caught 2 critical issues single-pass drafting missed (UK-US BDAA omission; ingestion transfer gap). First application of the Legal Agent pattern.

## Current Focus

*Updated: 2026-06-24*

**Adversarial Legal Review Pipeline (Exp 019) complete.** First application of the Legal Agent pattern. 0/7 claims survived the adversarial panel unchanged. Two critical issues caught that single-pass drafting missed: UK-US BDAA omission and the ingestion transfer gap (data and compute in different jurisdictions). Methodology post drafted for blog 2026-06-24 (pending deploy). Full iteration chain in Chronos (internal); sanitised brief held privately.

**Flash Attention cliff resolved (Exp 008/010, June 2026).** `OLLAMA_FLASH_ATTENTION=1` was the sole cause of the prefill cliff. FA=0/fp16 has no cliff through 40K tokens — confirmed by Exp 010's 2×2 factorial and independently by MLX on the same hardware (Exp 011). The Mini's true operational ceiling is >40K tokens on-wire.

**Pre-registered and awaiting execution:** Exp 015 (active-parameter ablation, dense vs MoE), Exp 017 (Argos Phase 0 feed reconnaissance), Exp 018 (sovereignty resilience — triggered by Fable/Mythos export control suspension June 11).

**OLÉ Marbella (June 17–18).** CasaSol booth demo. Exp 015/017/018 execution scheduled post-event.

## Technical Deployment Plan

- **Engine:** Hugo (Static Site Generator).
- **Workflow:** Nestor generates content via OpenClaw; symlinked to `~/REPOS/local-first-ai` for version control.
- **Cascade:** Router (`gemma4:e4b`) plans → Extractor (`json`) → Reducer (`gemma4:26b`) synthesizes.
- **Hardware Invariant:** Prefill context must remain **< 40,000 tokens** (revised upward from 22K after Exp 008/010).
- **Strict Validation:** ADR-001 contract with code-fence normalization for unreliable structured output.

## Pending Tasks

### Infrastructure & Experimentation

- [ ] **Exp 016 Phase B — TTFT benchmark.** 10 reps mini→local vs mini→miktam-mbp.local. Gate: median overhead <500 ms. Produce `phase_b_lan_latency.json`. [`exp_016_two_mac_orchestration/`](./exp_016_two_mac_orchestration/)
- [ ] **Exp 016 Phase C — Tiered pipeline end-to-end.** Plan on smart (Qwen3-Coder-Next), execute on cheap (gemma4:26b). Deterministic-glue invariants enforced. Gate: fewer human interventions than single-model baseline.
- [ ] **Exp 015 — Active-parameter ablation (dense vs MoE).** Does a dense 12–32B local model score ≥2/8 on the frozen Exp 012 audit rubric? Tests whether the capability boundary is total-parameter or active-compute. [`exp_015_active_param_ablation/`](./exp_015_active_param_ablation/)
- [ ] **Exp 017 — Argos Phase 0 feed reconnaissance.** DGT NAP (DATEX II) + OpenChargeMap, Málaga–Gibraltar EV corridor. 24h cadence measurement required. [`exp_017_argos_phase0/`](./exp_017_argos_phase0/)
- [ ] **Exp 018 — Sovereignty resilience.** Three failure modes: Ollama down, weights removed, network cut. Triggered by Fable/Mythos export control event (June 11). [`exp_018_sovereignty_resilience/`](./exp_018_sovereignty_resilience/)
- [ ] **Exp 007 Phase C — Thermal endurance (MBP, 90 min sustained).** Fan audible during Phase B at 110K tokens. Deferred; lower priority now that Exp 016 characterises the two-machine pipeline.
- [ ] **Router alternative evaluation.** `gemma4:e4b` follows fixture patterns over prose instructions. Evaluate Qwen 2.5/3.5b as a more deterministic structured-output router.
- [x] **Exp 020 — Run the one-time Ollama proxy setup.** Applied and verified 2026-07-08 — H2 post-fix confirmed (401 unauth, 200 auth). [`exp_020_miktam_mini_hardening/`](./exp_020_miktam_mini_hardening/)
- [x] **Exp 020 — Confirm SSH password-auth setting.** `PasswordAuthentication no` confirmed 2026-07-08 — key-only, no action needed.
- [ ] **Exp 020 — Tighten screen-lock delay.** Currently 3,600s (1 hour); recommend ~5s. No known downside (doesn't affect background daemons), just wasn't in the originally negotiated hardening scope — pending confirmation of intent.

### Content Execution

- [ ] **Exp 019 — Publish methodology post.** `2026-06-24-adversarial-legal-panel.md` — drafted, in `local-first-ai-blog/content/posts/`, pending `hugo deploy`. Brief post removed — the analysis is a product, not marketing content.
- [ ] **Exp 020 — Publish hardening post.** `2026-07-08-hardening-the-inference-node.md` — drafted, fully anonymized per the exp_019 precedent (no client/product names). Pending local Hugo preview check and `hugo deploy`.
- [ ] **"What was in the tank?"** Hindenburg analogy essay (US helium ban → hydrogen → 1937 fire as rhetorical frame for Fable/Mythos sovereignty event). Post-OLÉ.
- [ ] **Exp 015 writeup.** "The Active Compute Boundary" — does the local/cloud step function hold at every model class, or is it an artefact of testing only MoE-A4B? Post-execution.
- [ ] **Chronos README narrative.** Add "Chronos tests the 29%, not the 981%" framing (NBER WP 35275: Claude Code +981% lines written, +29% shipped releases).
- [ ] **The Cascade Pattern.** Transferable architecture writeup from [`exp_005_dicer_describer/cascade_pattern.md`](./exp_005_dicer_describer/cascade_pattern.md). Deferred since May; still worth writing post-OLÉ.
