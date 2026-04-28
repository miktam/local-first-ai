---
id: H3
title: Ollama loads gemma4:26b with a smaller num_ctx than OpenClaw advertises
status: open
test_script: tests/h3_num_ctx_negotiation.sh
related_upstream:
  - https://github.com/openclaw/openclaw/issues/24068
created: 2026-04-27
---

## Claim

Ollama's `/api/show` reports `gemma4.context_length` as 256000 (or
similar large value), and OpenClaw reads that field to populate its
TUI's "X/131k" display. But when Ollama actually loads the model on a
generation request, it does so with a `num_ctx` set by the request
options or by Ollama's default (often 4096 or 8192) — much smaller
than what OpenClaw believes is available. The runaway is the symptom
of crossing the actually-loaded context, not the displayed one.

## Prediction (if H3 is true)

- `/api/show` for `gemma4:26b` reports a `context_length` in the
  high tens or hundreds of thousands.
- After loading the model with no explicit `num_ctx`, `/api/ps`
  reports a much smaller `context_length` (e.g. 4096 or 8192).
- Loading the model with explicit `options: {num_ctx: 65536}` makes
  `/api/ps` report 65536.
- The numerical gap between displayed (`/api/show`) and loaded
  (`/api/ps`) without explicit override is large.

## Prediction (if H3 is false / null)

`/api/ps` reports the same large value as `/api/show` (or close to
it) on a default load. Then there is no negotiation drift to explain
the runaway, and we look elsewhere.

## Falsification

`/api/ps` reporting a `context_length` ≥ 65536 on a default load
(no explicit `num_ctx` in the request) falsifies H3.

## Discrimination

- **vs H1** (FA fallback): H3 makes no claim about CPU/GPU placement.
  Both could be true.
- **vs H2** (MoE empty content): H2 reproduces with explicit
  `num_ctx`; H3 is about what happens when no override is set.
- **vs H4** (OpenClaw token cap): H3 is about Ollama's loaded context.
  H4 is about the input token count OpenClaw actually sends. Both
  could be operating: OpenClaw advertises 131k, sends 4096 anyway, and
  Ollama loaded only 8192 — orthogonal failures.

## Method (summary)

1. Query `/api/show` for `gemma4:26b`; record reported context length.
2. Send a minimal generation request with no `num_ctx` option to
   force a load. Then query `/api/ps`; record the loaded
   `context_length`.
3. Send a second request with explicit `options: {num_ctx: 65536}`,
   wait for the model to reload, query `/api/ps`; confirm the value
   is honoured.
4. Compute the drift = displayed − loaded (default).

Decision rule: H3 supported if drift > 8000 tokens.

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
