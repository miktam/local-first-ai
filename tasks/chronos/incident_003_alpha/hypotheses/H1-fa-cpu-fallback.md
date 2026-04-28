---
id: H1
title: Flash Attention causes silent GPU→CPU fallback on Gemma 4 MoE
status: open
test_script: tests/h1_fa_cpu_fallback.sh
related_upstream:
  - https://github.com/ollama/ollama/issues/15237
  - https://github.com/ollama/ollama/issues/15368
created: 2026-04-27
---

## Claim

With `OLLAMA_FLASH_ATTENTION=1`, Gemma 4 26B-A4B MoE inference on
miktam02 silently falls back from GPU to CPU during long-context
prompt evaluation. `ollama ps` continues to report 100% GPU while
the actual work is on CPU, producing the observed ~994% CPU + idle
GPU signature of Incident 003-Alpha.

**Production target**: `gemma4-think:26b` (the local alias of
`gemma4:26b` with explicit `num_ctx=131072`, used because OpenClaw's
name-based heuristic only sends `"think": false` when the model name
contains "r1" / "reasoning" / "think"). The FA bug, if present, lives
at the Ollama runtime layer and should manifest identically against
the alias and the base — but we test the alias to match production
faithfully. `"think": false` is sent to match production.

## Prediction (if H1 is true)

With a ≥10k-token prompt to `gemma4:26b`:

- **FA=1 condition**: mean GPU power < 500 mW sustained during prompt
  eval; CPU > 700% on the ollama process; `ollama ps` reports "100%
  GPU" throughout.
- **FA=0 condition**: mean GPU power > 2000 mW sustained; CPU stays
  below ~300%.
- The delta between the two conditions is the signal.

## Prediction (if H1 is false / null)

GPU power tracks compute load throughout in both conditions. CPU
stays below 300% on the ollama process regardless of FA setting.

## Falsification

A single clean run with `FA=1`, a ≥10k-token prompt, sustained GPU
power above 2000 mW, and CPU below 300% — falsifies H1 cleanly.

The reverse (FA=0 showing GPU idle) would also falsify H1 but
indicate a different bug.

## Discrimination

- **vs H2** (MoE empty content): H1 is about *where* compute runs
  (CPU vs GPU). H2 is about *what comes back* (empty content vs real
  content). They can both be true simultaneously and produce
  overlapping symptoms.
- **vs H3** (num_ctx drift): H3 should not be sensitive to the FA
  flag. If toggling FA changes the symptom, H1 is implicated; if not,
  H3 is more likely.
- **vs H4** (OpenClaw token drift): H1 reproduces against raw
  `ollama run` with no OpenClaw involved.

## Method (summary)

For each value of `OLLAMA_FLASH_ATTENTION` in {1, 0}:

1. Set the env var via `launchctl setenv`.
2. Restart the Ollama app so the new env is applied.
3. Issue a `gemma4:26b` `/api/generate` request with a deterministic
   ~12k-token prompt and `num_ctx=65536`.
4. Sample `powermetrics --samplers gpu_power,cpu_power` at 1s
   intervals throughout the run.
5. Snapshot `ollama ps` every 5s while the request is in flight.

Decision rule encoded in the test script.

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
