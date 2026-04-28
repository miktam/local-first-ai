---
id: H2
title: gemma4:26b MoE returns empty content on long prompts
status: open
test_script: tests/h2_moe_empty_content.sh
related_upstream:
  - https://github.com/ollama/ollama/issues/15428
created: 2026-04-27
---

## Claim

`gemma4:26b` (the MoE variant) returns a structurally valid response
with empty `content`, populated `eval_count`, and `done_reason: "stop"`
when the input prompt exceeds a threshold (reported in upstream as
~500 chars of system prompt; in agentic workloads the threshold may
arrive via accumulated turns + tool definitions rather than a single
long system prompt). OpenClaw surfaces this as "Agent couldn't
generate a response" because there is literally no text to display.

## Prediction (if H2 is true)

Holding everything else constant and sweeping system prompt size
across {200, 1000, 2000, 5000} characters of filler text, with a
trivial user message:

- `gemma4:26b` returns `content: ""` and `done_reason: "stop"` for at
  least one of the larger sizes.
- `gemma4:31b` (dense, if available) returns non-empty content for the
  same prompts.
- The transition for `gemma4:26b` happens at a reproducible threshold
  on repeat runs.

## Prediction (if H2 is false / null)

`gemma4:26b` returns non-empty content for all tested sizes. Or both
26b and 31b return empty content (which would indicate a different
bug, not the MoE-specific one).

## Falsification

Three consecutive runs at the largest tested size (5000 chars) where
`gemma4:26b` returns non-empty content and a non-trivial `eval_count`
falsifies H2.

## Discrimination

- **vs H1** (FA fallback): H2 should reproduce regardless of the FA
  setting. The test runs with FA in its default state and does not
  toggle it. If H2 reproduces only with FA=1, the two are conflated
  and we should retest with FA=0.
- **vs H3/H4** (context negotiation): H2 reproduces against raw
  `/api/chat` with explicit `num_ctx`, no OpenClaw.

## Method (summary)

For each model in {`gemma4:26b`, `gemma4:31b` (if pulled)} and each
size in {200, 1000, 2000, 5000}:

1. POST to `/api/chat` with a system message of `size` characters of
   deterministic filler and a fixed short user message.
2. Record full response: `content` length, `eval_count`,
   `done_reason`, `prompt_eval_count`, `total_duration`.
3. Three repeats per (model, size) cell to test for stochasticity.

Decision rule: H2 supported if `gemma4:26b` returns empty content at
≥1 size in ≥2 of 3 repeats AND `gemma4:31b` does not (or, if 31b is
not pulled, the threshold is reproducible across repeats for 26b
alone).

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
