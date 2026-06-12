# Exp 016 — Two-Mac Orchestration: Model Selection and Tiered Pipeline

*Pre-registered: 2026-06-12*  
*Phase A: COMPLETE 2026-06-12 — Winner: Qwen3-Coder-Next-4bit (4.0/5, 100.6 tok/s)*  
*Phase B: Blocked — mlx_lm.server hang unresolved*

---

## Motivation

Exp 007 established per-machine throughput and context ceilings for `gemma4:26b` running on
both the mini (M4 Pro, 64 GB) and MBP (M5 Max, 128 GB). That experiment used one model on both
machines. The open question is whether the MBP's 128 GB unified memory enables a qualitatively
different class of model — one too large to run on the mini — and whether that enables a
productive two-tier orchestration: cheap mechanical work on the mini, high-reasoning work on
the MBP.

This experiment covers three phases: model selection (Phase A), LAN routing overhead (Phase B),
and tiered pipeline end-to-end (Phase C).

---

## Hardware

| Machine | Chip | Memory | Role |
|---|---|---|---|
| Mac mini (`miktam02`) | M4 Pro | 64 GB | Cheap tier; always-on |
| MacBook Pro 14" | M5 Max | 128 GB | Smart tier; mobile |

Baseline throughput references:
- Exp 007 — `gemma4:26b` via Ollama on MBP: 57–92 tok/s gen across 4K–35K context sizes.
- Exp 011 — `gemma4:26b` via MLX on mini: 52 tok/s gen at 4K, no cliff through 40K tokens.
  MLX is the chosen runtime for MBP — native unified memory, no FA cliff, OpenAI-compatible
  server via `mlx_lm.server`.

---

## Phase A — Smart model selection on MBP

### Hypotheses

**H1 (capability threshold):** At least one model runnable on 128 GB at Q4 sustains
>20 tok/s generation and scores ≥3 subjective points (0–5 scale) on the benchmark task.
Falsified if no tested model exceeds 20 tok/s OR no model scores ≥3/5.

**H2 (quality differential):** The winning MBP model scores subjectively higher than
`gemma4:26b` on the mini on the same task. Falsified if scores are equivalent (within ±1
on the 0–5 rubric).

**H3 (tier viability):** The quality differential in H2 is large enough to justify the
added latency and network overhead. Gate threshold: if H2 delta ≤1/5 with overhead
>500 ms, the tiered design is not worthwhile.

### Benchmark task

**Task specification (frozen before first run):**

> Add a `price_per_sqm` computed field to the CasaSol MCP server's `get_property` response.
> The field is `price_EUR / living_area_sqm`, rounded to the nearest integer. Return `null`
> if either field is absent or zero. Write the implementation in
> `scripts/mcp_server.py` and add a regression test to the existing test suite.

This is a real single-file implementation task from the CasaSol repo. It is bounded,
has a clear correctness criterion (the test passes), and requires no external knowledge.
It is representative of the "cheap" tier's daily work — using it as the benchmark tests
whether the smart tier is genuinely better at tasks that the cheap tier will execute.

### Runtime

**MBP:** `mlx-lm` (`pip install mlx-lm`), models from mlx-community on HuggingFace.  
Server: `mlx_lm.server --model mlx-community/<model-id> --host 0.0.0.0 --port 8080`  
API: OpenAI-compatible at `http://mbp.local:8080/v1`.

Rationale: Exp 011 ran MLX on the mini — no cliff through 40K tokens, 52 tok/s at 4K
(vs Ollama FA=0 ~40 tok/s). MLX is Apple's native unified-memory runtime; on M5 Max with
128 GB it is the right choice. Ollama is not installed on the MBP and not required.

**Mini (control):** Ollama FA=0 q8_0 at `localhost:11434` — existing setup, unchanged.

### Candidate models

| Priority | Model | Type | Est. memory | Notes |
|---|---|---|---|---|
| Control | `mlx-community/gemma-4-26b-a4b-it-4bit` | MoE (26B / 4B active) | ~13–15 GB | Same model as Exp 011 mini run; confirms per-machine scaling |
| Primary | `mlx-community/Qwen3.5-122B-A10B-4bit` | MoE (122B / ~10B active) | ~50 GB | Best balanced smart tier candidate |
| Coding specialist | `mlx-community/Qwen3-Coder-Next-4bit` | MoE (80B / 3B active) | ~40 GB | Purpose-built for coding; likely strongest on benchmark task |
| Optional | `mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit` | MoE (~109B / 17B active) | ~55–65 GB | Strong generalist; run if time allows |

**Testing order:** control → Qwen3.5-122B → Qwen3-Coder-Next → Llama-4-Scout (optional).

Pull models one at a time; do not load multiple into memory simultaneously.

**Expected outcome:** H1 and H2 both likely to confirm — all candidates are MoE architectures
with 10–17B active parameters per forward pass, substantially more than the control's 4B active.
Qwen3-Coder-Next may score highest on the coding benchmark despite lower active params due to
domain specialisation.

### Measurements (per model, per run — 3 reps)

| Field | Source |
|---|---|
| `model` | mlx-community model ID |
| `gen_tps` | tokens / generation wall-clock time from `mlx_lm` response |
| `ttft_ms` | Time from request to first token (streaming) |
| `peak_memory_gb` | `ps` / Activity Monitor peak RSS during generation |
| `quality_score` | 0–5 rubric: 0=wrong/incomplete, 3=correct+test passes, 5=correct+clean+idiomatic |
| `rep` | 1–3 |

Write each run as `measurements/YYYYMMDDTHHMMSSZ-phase_a-<machine>-<model_slug>-rep<N>.json`.
Summary in `measurements/phase_a_summary.json`.

### Pass criteria

| Criterion | Threshold |
|---|---|
| H1 confirmed | ≥1 model: gen_tps >20 AND quality_score ≥3/5 |
| H2 confirmed | Winning model quality_score > gemma4:26b score by >1/5 |
| Phase A gate | H1 confirmed; if not, stop and report — tiered design needs rethinking |

---

## Phase B — LAN routing overhead

### Hypothesis

**H4 (network overhead):** Round-trip overhead for a mini→MBP Ollama request over home WiFi
adds <500 ms to TTFT compared to a local mini request. Falsified if median overhead ≥500 ms.

### Design

From the mini, send 10 identical short prompts to both endpoints using the OpenAI chat
completions format (both Ollama and MLX server support it):
- `http://localhost:11434/v1/chat/completions` (local baseline, `gemma4:26b` via Ollama)
- `http://mbp.local:8080/v1/chat/completions` (LAN target, winning Phase A model via MLX)

Record TTFT for each. Compute median and 95th percentile overhead.

Write raw timings to `measurements/phase_b_lan_latency.json`.

### Pass criteria

| Criterion | Threshold |
|---|---|
| H4 confirmed | Median overhead <500 ms |
| H4 inconclusive | Overhead 500–1500 ms — acceptable for long-running tasks, not for interactive |
| H4 falsified | Median overhead ≥1500 ms — investigate network path before Phase C |

---

## Phase C — Tiered pipeline end-to-end

*Gated on Phase A and Phase B passing.*

### Hypothesis

**H5 (tier reduction):** Running one real feature through the two-tier pipeline
(plan on smart, mechanical steps on cheap) requires fewer human interventions than
running the same feature single-model on the mini. Falsified if intervention count is equal
or higher.

### Design constraint

The pipeline MUST follow deterministic-glue invariants
(`wiki/patterns/deterministic-glue-pipeline.md`). Specifically:

- Smart model produces a `plan.json` artifact with schema:
  `{ "task": string, "steps": [{ "id": string, "role": "cheap"|"smart", "instruction": string, "context_budget_tokens": integer }] }`
- `plan.json` is committed to the bead before cheap starts any step.
- Cheap receives one step object at a time — never the full plan or smart's prose.
- Role routing is deterministic code (config lookup), not a model decision.

### Task

The same feature as Phase A benchmark task (or the next queued CasaSol task — document
which one before running).

### Measurements

| Field | Value |
|---|---|
| `wall_clock_minutes` | Total elapsed from task start to close |
| `human_interventions` | Count of corrections, retries, or manual steps required |
| `bead_close_rate` | Fraction of beads closed without human intervention |
| `tier_failures` | List of steps where the assigned tier failed and why |

Baseline comparison: single-model run of the same task on mini (document before Phase C).

### Pass criteria

| Criterion | Threshold |
|---|---|
| H5 confirmed | Tiered interventions < single-model interventions |
| Phase C gate | H5 confirmed; if not, Gas Town (Phase 4) is unlikely to fix it |

---

## Artifacts structure

```
exp_016_two_mac_orchestration/
├── README.md                         ← this file
└── measurements/
    ├── phase_a_summary.json
    ├── YYYYMMDDTHHMMSSZ-phase_a-mbp-<model>-rep<N>.json
    ├── phase_b_lan_latency.json
    └── phase_c_tiered_pipeline.json
```
