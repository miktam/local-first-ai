# Experiment 023 — Generation Efficiency Across the Local Model Family

*Pre-registered: 2026-08-22 · Status: pre-registered, execution pending*

**Inspired by:** [terminalbytes.com — "Run Qwen 3.8 27B Locally"](https://terminalbytes.com/run-qwen-3-8-27b-locally/), which benchmarked Qwen3.8 27B against its predecessor Qwen3.6 27B on fixed hardware (Mac Studio M3 Ultra), running five timed generations per model through `ollama run --verbose` across varied technical prompts, and found that the newer model generated at roughly half the per-token speed but used roughly half as many tokens per answer — netting similar wall-clock latency despite looking slower on the headline tok/s number.

---

## What this experiment tests

Chronos has accumulated five local models on miktam-mini (M4 Pro, 64GB) across prior experiments and production use, but has never directly compared their **generation efficiency** against each other under one consistent methodology:

| Model | Size (Q4_K_M) | Role in prior experiments |
|---|---|---|
| `gemma4:e4b` | 9.6 GB | Router — intent classification (production, casasol) |
| `gemma4:26b` | 17 GB | Reducer — primary response/enrichment model (production, casasol; exp_005–exp_022) |
| `gemma4:31b` | 19 GB | Larger gemma variant, pulled but not yet used in a shipped pipeline |
| `qwen3.5:35b` | 23 GB | Dense candidate model, referenced but untested against exp_012's audit rubric (exp_015 candidate pool) |
| `qwen3.6:35b` | 23 GB | Newer dense candidate, same family as above |

This experiment applies the terminalbytes methodology to this exact set — not new models, only ones already pulled and used somewhere in this project — to build a token-economy/speed baseline that complements exp_012's capability-focused cost/capability curve and feeds exp_015 (dense vs MoE ablation, still pre-registered and unexecuted) with real efficiency numbers for its candidate models.

---

## Hypotheses

**H1 — Router is fastest per-token:** `gemma4:e4b` (9.6GB, smallest) has the highest mean generation tok/s of the five models.

*Null:* a larger model matches or beats `gemma4:e4b`'s generation tok/s.

**H2 — Token economy, not just speed, drives wall-clock latency (the terminalbytes finding, replicated or not):** among the two same-family pairs tested here — `qwen3.5:35b` vs `qwen3.6:35b` (predecessor vs newer, direct parallel to the source article's Qwen3.6-vs-Qwen3.8 comparison) — the model with the lower generation tok/s does not necessarily have the higher wall-clock time per answer, because token economy (mean tokens generated per answer) can offset a raw speed disadvantage.

*Null:* wall-clock time ranks the same as generation tok/s ranks within the pair — i.e., token economy has no measurable offsetting effect here.

**H3 — Wall-clock time is not monotonic in model size:** across all five models, ranking by wall-clock time per answer does not match ranking by model size (GB). A bigger model is not always slower end-to-end.

*Null:* wall-clock time increases monotonically with model size.

---

## Design

**Hardware:** miktam-mini (Mac Mini M4 Pro, 64GB unified memory) — fixed across all runs, same machine as every other Chronos generation-speed measurement (exp_007, exp_008, exp_010, exp_011).

**Prompts:** five fixed, varied technical prompts, identical across every model (a deliberate improvement on the source article's per-run-varied prompts — this removes prompt choice as a confound when comparing across models, at the cost of not testing per-model robustness to prompt variety, which is out of scope here):

1. Explain how a hash table resolves collisions, comparing chaining and open addressing.
2. Write a function that finds the longest palindromic substring in a string, and explain its time complexity.
3. Explain the difference between TCP and UDP, and give one real-world scenario where each is the better choice.
4. A user reports that their API returns 200 OK but the response body is empty about 1% of the time under load. Walk through how you'd debug this.
5. Explain what a Bloom filter is, how it works, and one real-world use case.

Target answer length ~200–500 words, matching the source article's range. No system prompt — raw model behavior, temperature 0.2 (matches this project's existing `ollama_chat` default elsewhere).

**Procedure per model:**
1. One untimed warm-up call (any prompt) to force the model into memory — `OLLAMA_KEEP_ALIVE=-1` means it stays resident until evicted by memory pressure from loading the *next* model in the sequence, so running all 5 timed prompts for one model consecutively (before moving to the next) keeps it warm throughout and isolates cold-load time to the warm-up call only.
2. Five timed calls via the Ollama HTTP API (`/api/chat`, non-streaming) — one per prompt above, in fixed order.
3. Record directly from each response's JSON fields (no CLI `--verbose` text parsing — the API exposes the same numbers as typed fields, more reliably): `total_duration`, `load_duration`, `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` (all nanoseconds from the API; converted to seconds/tok-per-second for reporting).

**Derived metrics per run:**
- `gen_tok_s` = eval_count / (eval_duration / 1e9)
- `prompt_tok_s` = prompt_eval_count / (prompt_eval_duration / 1e9)
- `wall_clock_s` = total_duration / 1e9 (includes load_duration only on the untimed warm-up, not on the 5 timed runs)
- `tokens_per_answer` = eval_count

Per model: mean ± range across the 5 timed runs for each metric above.

**Out of scope for this pass** (unlike the source article): power draw and RAM-by-quantization-level. `asitop`/`pmset` are available passwordless on this machine but `asitop` is an interactive TUI, not built for scripted single-shot sampling, and RAM-by-quantization is a static fact about each already-downloaded GGUF rather than something this run needs to measure. Neither blocks the core token-economy/speed comparison. A 1-bit-quantization sub-experiment (present in the source article) is also out of scope — it would require pulling a new model variant, which breaks the "previously run LLMs only" framing of this experiment.

---

## Evidence artefacts

```
results/
  run_YYYYMMDDTHHMMSSZ.json   — one file, all 5 models × 5 prompts, per-run + per-model summary
```

Each result file:
```json
{
  "ts": "ISO-8601",
  "hardware": "miktam-mini M4 Pro 64GB",
  "prompts": ["...", "...", "...", "...", "..."],
  "runs": [
    {"model": "gemma4:e4b", "prompt_idx": 0, "gen_tok_s": 0.0, "prompt_tok_s": 0.0,
     "wall_clock_s": 0.0, "tokens_per_answer": 0, "reply_excerpt": "..."}
  ],
  "summary": {
    "gemma4:e4b": {"mean_gen_tok_s": 0.0, "mean_prompt_tok_s": 0.0,
                   "mean_wall_clock_s": 0.0, "mean_tokens_per_answer": 0.0}
  },
  "verdicts": {"H1": "CONFIRMED|REFUTED", "H2": "CONFIRMED|REFUTED", "H3": "CONFIRMED|REFUTED"}
}
```

---

## Publication angle

Ties directly into exp_012's cost/capability curve and exp_015's still-pending dense-vs-MoE ablation: this gives real speed/token-economy numbers for the exact candidate models exp_015 will later score on capability, so a future post can pair "how fast/token-efficient" (this experiment) with "how capable" (exp_015) for the same model set — the two axes the terminalbytes article conflated into a single wall-clock number, kept separate here on purpose.

---

*Experiment design: Andrei + Claude Sonnet 5 · 2026-08-22*
