---
id: H4
title: OpenClaw caps input tokens to Ollama regardless of configured contextWindow
status: open
test_script: tests/h4_openclaw_token_drift.sh
related_upstream:
  - https://github.com/openclaw/openclaw/issues/27278
  - https://github.com/openclaw/openclaw/issues/24068
created: 2026-04-27
---

## Claim

OpenClaw's session JSONL files record an `input` token count per
turn that should grow with conversation length. Upstream report
#27278 documents a regression where every turn was hard-capped at
exactly 4096 tokens regardless of the configured `contextWindow`. If
that regression (or a variant) is present on miktam02, the model
never sees most of the accumulated context, and the runaway at ~40k
TUI-displayed tokens is unrelated to what was actually shipped to
Ollama.

## Prediction (if H4 is true)

- Across recent OpenClaw sessions, `input` token counts cluster
  tightly around a single value (4096, 8192, or another suspicious
  power of two), even on long conversations where the TUI counter
  showed tens of thousands.
- The median input token count across turns is much smaller than the
  configured `contextWindow`.
- Coefficient of variation across turns is low (< 0.1) despite
  variable conversation lengths.

## Prediction (if H4 is false / null)

`input` token counts grow roughly proportionally with cumulative
conversation length, capped only by the configured `contextWindow`
and by OpenClaw's pruning heuristics. Coefficient of variation is
high; max input count approaches `contextWindow` in long sessions.

## Falsification

A single session where the recorded input count exceeds 8192 in any
turn falsifies the strong form of H4 (4096 hard cap). The weak form
(some lower-than-configured cap) requires sweeping across more data.

## Discrimination

- **vs H1, H2, H3**: H4 is purely an OpenClaw-side claim. It can be
  evaluated from session logs alone, without sending any new
  requests.
- This test is therefore the cheapest to run and the safest to run
  first — passive analysis only, no Ollama load.

## Method (summary)

1. Locate OpenClaw session directory (default
   `~/.openclaw/agents/main/sessions/`).
2. Parse all `*.jsonl` files. For each line that contains an `input`
   token count, record (session_id, turn_index, input_tokens).
3. Compute per-session statistics: min, median, max, std dev,
   coefficient of variation of input token counts.
4. Compute global histogram of input token values.
5. Flag sessions where input counts cluster tightly around a single
   value or where max < 16000 across many turns.

Decision rule: H4 (strong form) supported if the global mode is
exactly 4096 and ≥80% of turns are at the mode. H4 (weak form)
supported if median < configured `contextWindow / 4` across the
dataset.

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
