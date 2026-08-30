# Experiment 025 — Context-Allocation Isolation: Why gemma4:31b Specifically Collapses

*Pre-registered: 2026-08-30 · Status: H1 + H2 CONFIRMED (Part 1 completed cleanly); H3 (Part 2) superseded by a direct fix-and-verify*

**Update, same day:** Part 1 (the context sweep) completed in full — 2 reps × 5 context sizes × 2 models — before the run was stopped to prioritize fixing the actual daemon config over collecting more comparison data. That data is clean and worth keeping as-is:

- **H1 CONFIRMED, and it's a genuine cliff, not a decline:** `gemma4:31b` holds flat at ~13.2-13.4 tok/s across `num_ctx` ∈ {2048, 4096, 8192, 16384}, then collapses to ~0.09-0.13 tok/s at 262144 — over 100x, between adjacent tested points, not a smooth degradation.
- **H2 CONFIRMED:** `gemma4:26b` stays flat at ~60-64 tok/s across the *entire* range including 262144 — no cliff anywhere in the same sweep.

Before Part 2 (the corrected 5-model re-run) started, the root cause was found directly: `OLLAMA_CONTEXT_LENGTH` had never been set, so Ollama auto-selected the top 256k tier given this Mac's unified memory — a daemon-wide default applied to every call that doesn't explicitly pass `num_ctx`, including production. Fixed by setting `OLLAMA_CONTEXT_LENGTH=8192` in the launchd plist (Andrei applied it directly via sudo). Verified post-fix: `gemma4:31b` now runs at ~13-14 tok/s by default, no caller needs to change anything; `gemma4:26b` unregressed. H3 (the corrected exp_023 re-run) is superseded by this — the practical question it was designed to answer is resolved by the fix itself. Worth running later only if a formally corrected exp_023 dataset is wanted for publication.

*Evidence: `results/sweep_20260830T030214Z.json` (Part 1, complete).*

**Follows from:** exp_023 (generation efficiency). A quick diagnostic probe (not yet a controlled experiment — logged here, formalized below) found that `gemma4:31b`'s ~400x slowdown vs `gemma4:26b` (0.13 vs ~55 tok/s) mostly disappears when `num_ctx` is explicitly set to 4096 instead of left at the model's default (262144) — generation jumped to ~12.6 tok/s, prompt eval from ~0.85 to ~17-19 tok/s. That single probe isn't sufficient evidence on its own (one data point, no sweep, no comparison against sibling models) — this experiment formalizes it properly.

---

## What this experiment tests

`ollama show` reports the **same** 262144-token default context for `gemma4:26b`, `gemma4:31b`, `qwen3.5:35b`, and `qwen3.6:35b` (only `gemma4:e4b` differs, at 131072). In exp_023, all four of the 262144-default models ran at that default with no explicit `num_ctx` override — yet only `gemma4:31b` collapsed to ~400x slower than `gemma4:26b`; `qwen3.5/3.6:35b` were "only" ~7x slower, plausibly just a normal size-driven gap for a bigger, differently-architected model family. **Something about `gemma4:31b` specifically makes it disproportionately sensitive to a large context allocation, when three same-default-context siblings aren't.** This experiment isolates that, and produces a context-controlled, non-confounded re-run of exp_023's efficiency comparison as a byproduct.

---

## Hypotheses

**H1 — Context-size cliff is real and specific to gemma4:31b:** A sweep of `num_ctx` ∈ {2048, 4096, 8192, 16384, 262144(default)} for `gemma4:31b`, holding prompt/output fixed, shows generation tok/s dropping by at least one order of magnitude somewhere in that range — not a smooth, proportionate decline.

*Null:* tok/s declines smoothly and modestly across the sweep, consistent with normal larger-KV-cache overhead, not a cliff.

**H2 — The cliff is not shared by gemma4:26b at the same nominal default:** the identical `num_ctx` sweep run against `gemma4:26b` does not show a comparable order-of-magnitude collapse at any tested point.

*Null:* gemma4:26b shows a comparable cliff, meaning the effect is a general large-context Ollama/llama.cpp issue, not specific to gemma4:31b's architecture.

**H3 — Fixing `num_ctx` corrects exp_023's efficiency ranking materially only for gemma4:31b:** re-running exp_023's original 5-prompt comparison across all 5 models with `num_ctx` explicitly fixed at 4096 (matching Pharos's real production `STAGE_CONTEXT_LIMIT`) changes `gemma4:31b`'s standing dramatically (from ~400x slower to within ~10x of gemma4:26b) while leaving the relative ranking of `gemma4:e4b`, `gemma4:26b`, `qwen3.5:35b`, and `qwen3.6:35b` materially unchanged from the original exp_023 numbers.

*Null:* fixing `num_ctx` changes the other four models' relative standing by more than a small margin, meaning exp_023's original ranking was confounded more broadly than just gemma4:31b.

---

## Design

**Hardware:** miktam-mini (M4 Pro, 64GB), same Ollama 0.33.0 instance as exp_023, config unchanged (`OLLAMA_FLASH_ATTENTION=0`, no KV cache quantization — post exp_015's forced fix).

**Part 1 — context sweep (H1, H2):** for `gemma4:31b` and `gemma4:26b` (control), run the identical short prompt ("say hi", `num_predict=10`) at `num_ctx` ∈ {2048, 4096, 8192, 16384, 262144}, 2 reps each to catch noise, recording `prompt_eval_duration`, `eval_duration`, derived tok/s, and `load_duration` (expected to spike whenever `num_ctx` changes between calls, since Ollama reloads the model — reported separately, not counted as inference cost).

**Part 2 — corrected 5-model re-run (H3):** re-run exp_023's exact 5 prompts against all 5 original models, this time with `num_ctx=4096` explicitly set in every request's `options`. Same derived metrics as exp_023 (`gen_tok_s`, `prompt_tok_s`, `wall_clock_s`, `tokens_per_answer`), directly diffable against the original `exp_023_local_model_efficiency/results/` files.

**Out of scope:** the Flash-Attention/KV-cache-quantization factorial (FA=1+q8_0 vs FA=0+fp16 vs FA=1+fp16) that exp_015's forced config fix also raised as an open question — that needs multiple daemon restarts per condition and is a distinct, heavier experiment; not conflated with this one, which is pure per-request `num_ctx` variation with no daemon changes needed.

---

## Evidence artefacts

```
results/
  sweep_YYYYMMDDTHHMMSSZ.json    — Part 1: context sweep, both models
  corrected_YYYYMMDDTHHMMSSZ.json — Part 2: 5-model re-run at fixed num_ctx=4096
```

---

*Experiment design: Andrei + Claude Sonnet 5 · 2026-08-30*
