# Local First AI

Benchmark data, experiment harnesses, and scientific logs for running production AI on local hardware — no cloud required.

This repo accompanies the blog at [localfirstai.eu](https://localfirstai.eu) and provides the verifiable evidence backing every claim made there.

---

## Hardware

| Component | Value |
|---|---|
| Primary machine | Mac Mini M4 Pro (`miktam02`), 64 GB unified memory |
| Secondary machine | MacBook Pro M5 Max — under benchmark (Exp 007) |
| Primary model | `gemma4:26b` (MoE, 25.8B active params, Q4_K_M) |
| Router model | `gemma4:e4b` (fast routing layer) |
| Runtime | Ollama 0.20.2 |
| Orchestration | OpenClaw → Nestor (local AI agent) |
| Operating ceiling (Mini) | **< 18,000 tokens on-wire** (Exp 007 Phase B — cliff onset ~17–20K) |
| Operating ceiling (MBP)  | **< 40,000 tokens on-wire** (Exp 007 Phase B — cliff onset ~43–50K) |

---

## Project Chronos

Every claim on the blog is backed by a pre-registered experiment logged in [`tasks/chronos/scientific_log.md`](./tasks/chronos/scientific_log.md). The methodology: observation → hypothesis → experiment → evidence → conclusion. No retrofitted results.

Roadmap and pending experiments: [`tasks/chronos/roadmap.md`](./tasks/chronos/roadmap.md)

### Experiments

| # | Name | Status | Key finding |
|---|---|---|---|
| [001](./tasks/chronos/scientific_log.md) | Verification of Veracity | Complete | Chronos framework activated |
| [002](./tasks/chronos/experiments/) | Control Plane vs Data Plane | Complete | Thinking mode is flat at ~38 t/s until it isn't — unconstrained prompts trigger runaway |
| [003](./tasks/chronos/exp_003_local_memory/) | Anonymized Adversarial Memory | Complete | 0/20 source recognition, 0/3 identity leaks on Fight Club corpus — data-sovereignty moat is architectural |
| [004](./tasks/chronos/exp_004_bootstrap_diet/) | Bootstrap Diet | Complete | OpenClaw session hygiene |
| [005](./tasks/chronos/exp_005_dicer_describer/) | Router / Reducer Cascade | Phase 0 closed | Working two-model cascade over 8-year Apple Watch corpus; three load-bearing behaviours demonstrated |
| [006](./tasks/chronos/exp_006_redactor_fidelity/) | Redactor Fidelity (GDPR) | Complete | 0/20 × 8 categories — zero true-positive leaks across all pre-registered GDPR categories |
| [007](./tasks/chronos/exp_007_hardware_comparison/) | Mac Mini vs MacBook Pro M5 Max | Phase A+B complete | Mini cliff ~18K, MBP cliff ~45K (2.5×); MBP gen t/s +200–370%; H1+H2 confirmed |
| [008](./tasks/chronos/exp_008_flash_attention/) | Flash Attention + q8_0 KV Cache | Pre-registered | Tests whether FA + KV cache quantization pushes the 20K cliff to ≥30K tokens |

### Incidents

| # | Name | Finding |
|---|---|---|
| [003-Alpha](./tasks/chronos/incident_003_alpha/) | Memory Bandwidth Cliff | Prefill on `gemma4-think:26b` goes super-quadratic past ~25K tokens on Apple Silicon. Hard operational ceiling: **< 22K tokens on-wire**. The bottleneck is memory bandwidth, not VRAM. |

---

## Benchmarks

Early benchmarks that preceded the Chronos framework. Results in `benchmarks/results/`.

| Script | What it measures |
|---|---|
| `nestor-bench-phase1.sh` | Context window (4K–130K) vs generation speed. Finding: gen_tps flat at ~41 t/s. |
| `nestor-bench-phase1b.sh` | Thinking mode token overhead. Finding: 5–15× token cost for zero quality gain on simple tasks. |
| `nestor-bench-phase2-compare.sh` | Compressed-memory retrieval vs raw context. |
| `nestor-bench-phase2-memory.sh` | Memory layer latency at scale. |
| `nestor-bench-phase2b-retrieval.sh` | Retrieval accuracy across compression levels. |

```bash
ollama pull gemma4:26b
chmod +x benchmarks/nestor-bench-phase1.sh
./benchmarks/nestor-bench-phase1.sh
# Results written to benchmarks/results/
```

---

## Key findings (cumulative)

1. **The prefill cliff is real — and hardware-specific.** On Mac Mini M4 Pro (`gemma4:26b`), prefill degrades sharply past ~18K tokens: at 35K tokens, prefill takes 20 minutes. On MacBook Pro M5 Max, the cliff onset is at ~45K tokens — 2.5× higher — and the slope above it is 20× gentler. The bottleneck is memory bandwidth, not compute. The right machine for long-context local inference is not interchangeable. (Incident 003-Alpha, Exp 007)

2. **Thinking tokens are expensive — cap and name them explicitly.** `gemma4:26b` is a thinking model: left unconstrained, a simple task generates 10,000–25,000 hidden thinking tokens — at 38 t/s that's 4–11 minutes per response with zero quality gain. The architectural response: a `gemma4-think:26b` Ollama alias with a hard 128K context cap, used only for tasks that genuinely need deliberation. The name makes the choice visible; the cap prevents runaway. (Exp 002)

3. **Data sovereignty is an architectural property, not a policy.** An anonymization boundary enforced by the import graph — not by a prompt or a config flag — defeated source recognition (0/20) and identity bridging (0/3) on a corpus the model has memorised. The moat is the architecture. (Exp 003)

4. **A two-model cascade extends the operating envelope.** Router (`gemma4:e4b`) routes in ~3–4s. Reducer (`gemma4:26b`) synthesises only what fits below the 22K cliff. The cascade made an 8-year health corpus queryable on local hardware without hitting the bandwidth cliff on normal queries. (Exp 005)

5. **The 22K ceiling is a property of the hardware, not a bug.** Memory bandwidth saturates during prefill on the M4 Pro's unified memory architecture. Mitigations: cliff-aware coarsening in the extractor, hard token budgets in the cascade, streaming watchdog for booth/production use.

6. **The M5 Max die is in a different performance class for inference.** At 25K tokens, MBP gen t/s is 66 vs Mini's 14 — a 4.7× difference on the same model weights and quantisation. MBP at 35K tokens (1.24 ms/tok prefill) is still well below the Mini's baseline at 4K tokens (3.03 ms/tok). The cascade's 22K bundle ceiling — set for the Mini — is comfortably safe on the MBP, which can handle ~40K before hitting its own cliff. (Exp 007)

7. **A fixed redaction prompt reliably produces GDPR-clean output.** 20 synthetic toxic real estate notes spanning 8 pre-registered GDPR categories — 0 true-positive leaks in any output. The local 26B model with `temperature=0.1` and a structured system prompt passes all four pre-registered criteria: zero leaks, full structural compliance (TAGS + DESCRIPTION), all 20 within 300s. (Exp 006)

---

## Blog posts

Published at [localfirstai.eu](https://localfirstai.eu):

- [The Silicon Wager: M4 Pro vs M5 Max](https://localfirstai.eu/posts/2026-05-29-silicon-wager/) — Exp 007 writeup: every Chronos envelope was measured on one machine. A second arrived. The difference is not incremental.
- [Why CasaSol.ai](https://localfirstai.eu/posts/2026-05-22-why-casasol/) — If every company can be a Palantir now, how do you test that claim? An attempt to answer by building one — on the Costa del Sol.
- [The GDPR Canary for Real Estate: 8 Data Categories, 0 Leaks](https://localfirstai.eu/posts/2026-05-09-redactor-fidelity/) — Exp 006 writeup: a pre-registered fidelity sweep over 20 synthetic toxic notes. Zero true leaks. The claim becomes evidence.
- [The Architecture of Anonymity](https://localfirstai.eu/posts/2026-04-26-the-architecture-of-anonymity/) — Exp 003 writeup: data sovereignty enforced by the import graph, not by policy.
- [The Memory Bandwidth Cliff](https://localfirstai.eu/posts/incident_003_alpha_post/) — Incident 003-Alpha: why local AI is bound by the bus, not the GPU.
- [Should We Stop Asking Local LLMs to Think?](https://localfirstai.eu/posts/should-we-stop-asking-local-llms-to-think/) — What Adam Smith, neuroscience, and a melting Mac Mini taught me about the real division of cognitive labour.
- [The Control Plane and the Data Plane](https://localfirstai.eu/posts/2026-04-22-control-plane-vs-data-plane/) — Managing the AI thinking tax.
- [The Sovereign Individual: Why Private Data is the Only Moat Left](https://localfirstai.eu/posts/the-sovereign-individual/) — The strategic essay: as AI becomes commoditised, competitive advantage is private context.
- [Every Company Can Be a Palantir Now](https://localfirstai.eu/posts/every-company-can-be-a-palantir-now/) — The corporate corollary: proprietary structured data is the durable moat.
- [The Genesis of Chronos](https://localfirstai.eu/posts/2026-04-21-genesis-of-chronos/) — Why Nestor commits to verified, evidence-backed claims.

---

## License

MIT
