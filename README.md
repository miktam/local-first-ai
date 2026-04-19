# Local First AI

Benchmark data, scripts, and configuration for running production AI on local hardware.

This repo accompanies the blog at [localfirstai.eu](https://localfirstai.eu).

## What's here

**Benchmarks** for Gemma 4 26B (MoE, Q4_K_M) on Apple Silicon M4 Pro with 64GB unified memory, running via Ollama.

- **Phase 1** — Context window vs inference speed (4K–130K tokens). Result: generation speed is flat at ~41 t/s across all context sizes. No memory cliff.
- **Phase 1b** — Thinking mode cost isolation. Result: thinking adds 5–15x token overhead on simple tasks for zero quality improvement.

Full analysis: [Should We Stop Asking Local LLMs to Think?](https://localfirstai.eu/posts/should-we-stop-asking-local-llms-to-think/)

## Setup

| Component | Value |
|---|---|
| Hardware | Mac Mini M4 Pro, 64GB unified memory |
| Model | gemma4-think:26b (MoE, 25.8B params, Q4_K_M) |
| Runtime | Ollama 0.20.2, KEEP_ALIVE=-1, FLASH_ATTENTION=1 |
| Context | 130,000 tokens |
| Gateway | OpenClaw → Telegram |

## Running the benchmarks

```bash
# Phase 1: context window vs performance
chmod +x benchmarks/nestor-bench-phase1.sh
./benchmarks/nestor-bench-phase1.sh

# Phase 1b: thinking token cost
chmod +x benchmarks/nestor-bench-phase1b.sh
./benchmarks/nestor-bench-phase1b.sh
```

Requires: `ollama`, `jq`, `python3`, and a model pulled via `ollama pull gemma4:26b`.

Results are saved as CSV to `benchmarks/results/`.

## Key findings

1. **Context size doesn't matter on Apple Silicon.** gen_tps holds at ~41 t/s from 4K to 130K.
2. **Thinking mode is the bottleneck.** A JSON-to-table conversion: 31 tokens (1s) without thinking, 451 tokens (11s) with thinking. Same output.
3. **Ambiguous prompts + thinking = runaway.** In production with accumulated session context, thinking generates 10,000–25,000 hidden tokens. At 38 t/s, that's 4–11 minutes per response.
4. **The fix is architectural.** Default to think=off. Decompose tasks. Let the human be the reasoning layer.

## License

MIT
