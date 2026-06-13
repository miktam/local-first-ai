# Local First AI

Benchmark data, experiment harnesses, and scientific logs for running production AI on local hardware — no cloud required.

This repo accompanies the blog at [localfirstai.eu](https://localfirstai.eu) and provides the verifiable evidence backing every claim made there.

---

## Hardware

| Component | Value |
|---|---|
| Primary machine | Mac Mini M4 Pro (`miktam02`), 64 GB unified memory |
| Secondary machine | MacBook Pro M5 Max (`miktam-mbp`) — smart inference tier (Exp 016) |
| Primary model | `gemma4:26b` (MoE A4B, 25.8B total params / ~4B active per forward pass, Q4_K_M) |
| Router model | `gemma4:e4b` (fast routing layer) |
| Runtime | Ollama 0.20.2 |
| Orchestration | OpenClaw → Nestor (local AI agent) |
| Operating ceiling (Mini) | **> 40,000 tokens on-wire** (Exp 008/010 — no cliff at FA=0; Exp 007's 18K was a FA=1 artefact) |
| Operating ceiling (MBP)  | **> 40,000 tokens on-wire** (Exp 007 Phase B — FA=1 cliff at ~45K; FA=0 baseline not yet measured) |

---

## Project Chronos

Every claim on the blog is backed by a pre-registered experiment logged in [`tasks/chronos/scientific_log.md`](./tasks/chronos/scientific_log.md). The methodology: observation → hypothesis → experiment → evidence → conclusion. No retrofitted results.

Roadmap and pending experiments: [`tasks/chronos/roadmap.md`](./tasks/chronos/roadmap.md)

### Experiments

| # | Name | Status | Key finding |
|---|---|---|---|
| [001](./tasks/chronos/exp_001_verification_of_veracity/) | Verification of Veracity | Complete | Chronos framework activated |
| [002](./tasks/chronos/exp_002_control_plane_vs_data_plane/) | Control Plane vs Data Plane | Complete | Thinking mode is flat at ~38 t/s until it isn't — unconstrained prompts trigger runaway |
| [003](./tasks/chronos/exp_003_local_memory/) | Anonymized Adversarial Memory | Complete | 0/20 source recognition, 0/3 identity leaks on Fight Club corpus — data-sovereignty moat is architectural |
| [004](./tasks/chronos/exp_004_bootstrap_diet/) | Bootstrap Diet | Complete | OpenClaw session hygiene |
| [005](./tasks/chronos/exp_005_dicer_describer/) | Router / Reducer Cascade | Phase 0 closed | Working two-model cascade over 8-year Apple Watch corpus; three load-bearing behaviours demonstrated |
| [006](./tasks/chronos/exp_006_redactor_fidelity/) | Redactor Fidelity (GDPR) | Complete | 0/20 × 8 categories — zero true-positive leaks across all pre-registered GDPR categories |
| [007](./tasks/chronos/exp_007_hardware_comparison/) | Mac Mini vs MacBook Pro M5 Max | Phase A+B complete | Mini cliff ~18K, MBP cliff ~45K (2.5×); MBP gen t/s +200–370%; H1+H2 confirmed ⚠ see Exp 008 |
| [008](./tasks/chronos/exp_008_flash_attention/) | Flash Attention + q8_0 KV Cache | Complete — landmark | FA+q8_0 flags *cause* the cliff; FA=0/fp16 has no cliff through 40K. Exp 007 ceiling was an artefact |
| [009](./tasks/chronos/exp_009_adversarial_critic/) | Adversarial Project Critic (Local vs. Frontier) | Complete — FAIL | gemma4:26b matched compliance layer (DPA/DSAR/DPIA) but missed impl-vs-docs gaps; 50% overlap, 50% FP rate |
| [010](./tasks/chronos/exp_010_fa_isolation/) | FA vs q8_0 Factorial Isolation | Complete | FA=1 is the sole culprit (cliff at 32.5K alone, 20K combined with q8_0). q8_0 alone: no cliff, +5% gen t/s |
| [011](./tasks/chronos/exp_011_mlx_runtime/) | MLX Runtime vs Ollama — Context Cliff | Complete | No cliff through 40K on MLX. Prefill matches Ollama FA=0 within 3%. Cliff is an Ollama FA artefact, not a hardware limit. |
| [012](./tasks/chronos/exp_012_cost_capability/) | Cost vs Capability: Where the Curve Breaks | Complete ⚠ see Exp 012-Alpha | gemma4:26b (MoE A4B, ~4B active) 0/8, Haiku/Sonnet/Opus all 5/8 net. Cliff confirmed at the 4B-active/frontier boundary; dense local 12–32B class untested (→ Exp 015). Haiku is cost-dominant; Haiku→Opus is 6.4× cost, 0 score gain. |
| [013](./tasks/chronos/exp_013_local_audit_loop/) | Local Audit Loop: Can Scaffolding Move gemma4:26b Off Zero? | Complete | H confirmed (partial): decomposition recovered 2/3 findable items (Items 3+4 stable, Item 1 systematic bridge failure). Raw 2/5; context expansion deferred as out of scope for H3. Pre-filter architecture: local audit + Haiku top-up = ~$0.02/audit. |
| [014](./tasks/chronos/exp_014_capability_variance/) | Capability Variance Floor | Complete | H1 CONFIRMED: gemma ≤0/8 all 5 reps (goes negative via systematic DSR FP). H3 CONFIRMED: no overlap (gemma max 0, Haiku min 1). H2 FALSIFIED: Haiku scores 1/1/4/1/2 — not stable at 4–6/8; the 5/8 Exp 012 result was a high-tail sample. Haiku mean ~2/8 across reps; variance driven by DSR/DPA FP penalties and B3 blind spot. |
| [015](./tasks/chronos/exp_015_active_param_ablation/) | Active-Parameter Ablation: Dense vs MoE | Pre-registered | Does a dense 12–32B local model outperform gemma4:26b (MoE A4B) on the Exp 012 audit rubric? Bottleneck hypothesis: active compute per token, not total params. |
| [016](./tasks/chronos/exp_016_two_mac_orchestration/) | Two-Mac Orchestration | Phase B in progress | Phase A: Qwen3-Coder-Next-4bit selected (4.0/5, 100.6 tok/s). Phase B: LAN path mini→MBP confirmed (miktam-mbp.local:8080). TTFT benchmark pending. |
| [017](./tasks/chronos/exp_017_argos_phase0/) | Argos Phase 0 — Feed Reconnaissance | Pre-registered | DGT NAP (DATEX II) + OpenChargeMap recon, Málaga–Gibraltar EV corridor. 24h cadence measurement required before Phase 1. |
| [018](./tasks/chronos/exp_018_sovereignty_resilience/) | Sovereignty Resilience | Pre-registered | Three failure modes: Ollama down, weights removed, network cut. Teased in "We Didn't Notice" (2026-06-13). Execution post-OLÉ. |

### Watcher Runs

Production runs of the Adversarial Watcher — a staged local LLM pipeline that compares documented intent against shipped artefacts and produces annotated gap reports. Each run is an auditable Chronos artefact.

| Run | Project | Confirmed gaps | False positives | Evidence |
|---|---|---|---|---|
| [watcher_run_001](./tasks/chronos/watcher_run_001/) | CasaSol | 5 | 3 | 2026-06-06, gemma4:26b, ~270s |

### Incidents

| # | Name | Finding |
|---|---|---|
| [003-Alpha](./tasks/chronos/incident_003_alpha/) | Memory Bandwidth Cliff | Prefill on `gemma4-think:26b` goes super-quadratic past ~25K tokens on Apple Silicon. Hard operational ceiling: **< 22K tokens on-wire**. The bottleneck is memory bandwidth, not VRAM. |

---

## Benchmarks

Early benchmarks that preceded the Chronos framework. Scripts and results in [`tasks/chronos/experiments/`](./tasks/chronos/experiments/).

| Script | What it measures |
|---|---|
| `nestor-bench-phase1.sh` | Context window (4K–130K) vs generation speed. Finding: gen_tps flat at ~41 t/s. |
| `nestor-bench-phase1b.sh` | Thinking mode token overhead. Finding: 5–15× token cost for zero quality gain on simple tasks. |
| `nestor-bench-phase2-compare.sh` | Compressed-memory retrieval vs raw context. |
| `nestor-bench-phase2-memory.sh` | Memory layer latency at scale. |
| `nestor-bench-phase2b-retrieval.sh` | Retrieval accuracy across compression levels. |

```bash
ollama pull gemma4:26b
chmod +x tasks/chronos/experiments/nestor-bench-phase1.sh
./tasks/chronos/experiments/nestor-bench-phase1.sh
# Results written to tasks/chronos/experiments/results/
```

---

## Key findings (cumulative)

1. **The prefill cliff was an artefact of `OLLAMA_FLASH_ATTENTION=1`, not a hardware limit.** Exp 007 measured a cliff at ~18K tokens on Mac Mini M4 Pro — but those runs were made with FA=1+q8_0 enabled (Incident 007-Alpha). Exp 008 established the FA=0 baseline: no cliff through 40K tokens. Exp 010's 2×2 factorial confirmed FA=1 is the sole culprit: it alone (without q8_0) produces a cliff at 32.5K and triples prefill latency at 15K tokens (1.774 → 5.405 ms/tok). Under optimal config (FA=0, q8_0), the Mac Mini's true operational ceiling is **> 40K tokens on-wire**. Flash Attention was designed for discrete GPU SRAM/HBM hierarchies; on Apple Silicon unified memory, its tiling overhead applies without the bandwidth benefit. (Incident 003-Alpha, Exp 007, Incident 007-Alpha, Exp 008, Exp 010)

2. **Thinking tokens are expensive — cap and name them explicitly.** `gemma4:26b` is a thinking model: left unconstrained, a simple task generates 10,000–25,000 hidden thinking tokens — at 38 t/s that's 4–11 minutes per response with zero quality gain. The architectural response: a `gemma4-think:26b` Ollama alias with a hard 128K context cap, used only for tasks that genuinely need deliberation. The name makes the choice visible; the cap prevents runaway. (Exp 002)

3. **Data sovereignty is an architectural property, not a policy.** An anonymization boundary enforced by the import graph — not by a prompt or a config flag — defeated source recognition (0/20) and identity bridging (0/3) on a corpus the model has memorised. The moat is the architecture. (Exp 003)

4. **A two-model cascade extends the operating envelope.** Router (`gemma4:e4b`) routes in ~3–4s. Reducer (`gemma4:26b`) synthesises only what fits below the 22K cliff. The cascade made an 8-year health corpus queryable on local hardware without hitting the bandwidth cliff on normal queries. (Exp 005)

5. **The 22K ceiling is a property of the hardware, not a bug.** Memory bandwidth saturates during prefill on the M4 Pro's unified memory architecture. Mitigations: cliff-aware coarsening in the extractor, hard token budgets in the cascade, streaming watchdog for booth/production use.

6. **The M5 Max die is in a different performance class for inference.** At 25K tokens, MBP gen t/s is 66 vs Mini's 14 — a 4.7× difference on the same model weights and quantisation. MBP at 35K tokens (1.24 ms/tok prefill) is still well below the Mini's baseline at 4K tokens (3.03 ms/tok). The cascade's 22K bundle ceiling — set for the Mini — is comfortably safe on the MBP, which can handle ~40K before hitting its own cliff. (Exp 007)

7. **A fixed redaction prompt reliably produces GDPR-clean output.** 20 synthetic toxic real estate notes spanning 8 pre-registered GDPR categories — 0 true-positive leaks in any output. The local 26B model with `temperature=0.1` and a structured system prompt passes all four pre-registered criteria: zero leaks, full structural compliance (TAGS + DESCRIPTION), all 20 within 300s. (Exp 006)

8. **The Flash Attention cliff is a runtime artefact, not a hardware limit — confirmed by independent runtime.** MLX (Apple's native ML framework) shows no prefill cliff through 40K tokens on the same Mac Mini M4 Pro hardware. MLX prefill at 15K is 1.650 ms/tok — matching Ollama FA=0/q8_0 (1.694) within 3%. The cliff Exp 007 attributed to the Mac Mini's architecture was entirely a product of Ollama's llama.cpp Flash Attention tiling on unified memory. Two independent runtimes; same hardware; same result. The hardware ceiling is memory bandwidth, not attention kernel. (Exp 011)

9. **Local models match compliance gaps; frontier models catch implementation gaps.** A head-to-head adversarial critic comparison (three fixed personas, fixed JSON schema, same context bundle) found that gemma4:26b matched Claude Sonnet 4.6 on the DPO/compliance layer (DPA template, DSAR procedure, DPIA — 3/3 near-exact matches) but missed the highest-severity engineering finding: a primary moat component described across the BRIEF, deck, and BUILD_LOG had no corresponding code in any commit. gemma4 pattern-matched on documented claims and critiqued their replicability; Claude cross-referenced the BUILD_LOG claim against the git history and flagged the absence. Overlap rate: ~50%. False-positive rate: ~50%. Verdict: FAIL as a drop-in replacement, viable as a zero-cost compliance-layer complement to periodic frontier review. (Exp 009)

10. **The cost-capability curve has one step — confirmed at the 4B-active/frontier boundary; dense local models untested.** ⚠ *Scope correction (Exp 012-Alpha, 2026-06-09): gemma4:26b is MoE A4B (~4B active parameters per forward pass, not 25.8B). Exp 012 tested one local model class; whether the step holds for dense 12–32B local models is open and pre-registered as Exp 015.* A four-model sweep (gemma4:26b, Haiku, Sonnet, Opus) on a pre-scored 8-point rubric across two task types (DPO compliance extraction + engineering gap detection) produced: gemma4:26b 0/8, all three frontier models 5/8 net. Haiku→Sonnet (3.1× cost) and Haiku→Opus (6.4× cost) each yield zero additional rubric points. For structured analytical extraction on bounded context (~10K tokens), Haiku is cost-dominant. Qualitative differences exist within the same net score: Haiku and Sonnet have complementary blind spots (Haiku misses concurrency risk; Sonnet misses auth gap). Opus has the highest gross score (6/8) but the highest false positive rate. Two items — a missing DPIA for VLM processing and the Art. 22 model-version accountability gap — evaded all four models. (Exp 012; scope corrected 2026-06-09 — see Exp 012-Alpha in scientific_log.md)

---

## Blog posts

Published at [localfirstai.eu](https://localfirstai.eu):

**Technical — benchmarks, experiments, architecture**

- [We Didn't Notice](https://localfirstai.eu/posts/2026-06-13-we-didnt-notice/) — The US government suspended the world's best AI model overnight for all foreign nationals. CasaSol was unaffected. Exp 018 pre-registered.
- [The Cost-Capability Curve Has One Step](https://localfirstai.eu/posts/2026-06-09-cost-capability-curve/) — A four-model sweep at the 4B-active/frontier boundary. One step, not a ramp. Haiku is cost-dominant; Haiku→Opus is 6.4× cost, 0 score gain.
- [Same Hardware. Different Runtime. Same Result.](https://localfirstai.eu/posts/2026-06-09-mlx-vs-ollama-runtime/) — Exp 011: MLX and Ollama FA=0 on the same Mac Mini M4 Pro. Neither cliffs through 40K tokens. Prefill within 3%. The FA cliff was an Ollama/llama.cpp artefact, confirmed by an independent runtime.
- [The Cliff That Wasn't](https://localfirstai.eu/posts/2026-06-07-the-cliff-that-wasnt/) — The 20K prefill cliff that shaped six months of cascade architecture was `OLLAMA_FLASH_ATTENTION=1`. Removing it tripled the Mac Mini's operational ceiling to >40K tokens. Full 2×2 factorial: FA=1 is the sole culprit, q8_0 alone is benign.
- [The Adversarial Watcher: When a Local Model Audits Its Own Project](https://localfirstai.eu/posts/2026-06-06-adversarial-watcher/) — A staged 5-step pipeline that catches documentation drift before every merge. First production run: 5 confirmed gaps, 3 false positives, anatomy of each.
- [We Tried to Replace Claude with a Local Critic. Here's Exactly Where It Failed.](https://localfirstai.eu/posts/2026-06-06-adversarial-critic/) — Exp 009: head-to-head adversarial review. gemma4:26b matches the compliance layer; only the frontier model caught the impl-vs-docs gap.
- [The Silicon Wager: M4 Pro vs M5 Max](https://localfirstai.eu/posts/2026-05-29-silicon-wager/) — Exp 007: every Chronos envelope was measured on one machine. A second arrived. The difference is not incremental.
- [The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks](https://localfirstai.eu/posts/2026-05-09-redactor-fidelity/) — Exp 006: pre-registered fidelity sweep over 20 synthetic toxic notes. Zero true leaks. The claim becomes evidence.
- [The Memory Bandwidth Cliff](https://localfirstai.eu/posts/2026-04-28-incident_003_alpha_post/) — Incident 003-Alpha: why local AI is bound by the bus, not the GPU.
- [The Architecture of Anonymity](https://localfirstai.eu/posts/2026-04-26-the-architecture-of-anonymity/) — Exp 003: data sovereignty enforced by the import graph, not by policy.
- [The Control Plane and the Data Plane](https://localfirstai.eu/posts/2026-04-22-control-plane-vs-data-plane/) — Exp 002: managing the AI thinking tax.
- [The Genesis of Chronos](https://localfirstai.eu/posts/2026-04-21-genesis-of-chronos/) — Why Nestor commits to verified, evidence-backed claims.

**Essays — strategy, product, philosophy**

- [Why CasaSol.ai](https://localfirstai.eu/posts/2026-05-22-why-casasol/) — If every company can be a Palantir now, how do you test that claim? An attempt to answer by building one — on the Costa del Sol.
- [Should We Stop Asking Local LLMs to Think?](https://localfirstai.eu/posts/should-we-stop-asking-local-llms-to-think/) — What Adam Smith, neuroscience, and a melting Mac Mini taught me about the real division of cognitive labour.
- [The Sovereign Individual: Why Private Data is the Only Moat Left](https://localfirstai.eu/posts/the-sovereign-individual/) — As AI becomes commoditised, competitive advantage is private context.
- [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/) — Proprietary structured data is the durable moat.

---

## License

MIT
