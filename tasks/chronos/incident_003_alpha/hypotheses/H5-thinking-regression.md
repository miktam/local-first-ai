---
id: H5
title: Thinking mode re-engages at long context despite think:false
status: open
test_script: tests/h5_thinking_regression.sh
related_upstream: []
created: 2026-04-27
opened_in_response_to:
  - H2 narrow form rejected at small scale
  - Symptom signature of Incident 003-Alpha matches the April 12
    thinking-runaway bug (CPU pegged, multi-tens-of-minutes hang,
    no output to stream) more closely than any FA-fallback or
    empty-content pattern.
---

## Background

On 2026-04-12, an unrelated incident traced a Nestor runaway to
Gemma 4's renderer defaulting `think` to ON unless the chat request
explicitly passed `"think": false`. The fix was a local Ollama alias
`gemma4-think:26b` (`FROM gemma4:26b`, `PARAMETER num_ctx 131072`).
The "think" substring in the alias name triggers OpenClaw's
name-based reasoning-model heuristic, which then sends
`"think": false` on chat requests. This worked at the time and Nestor
became responsive.

Incident 003-Alpha (2026-04-26) reproduces the same symptom signature
— pegged CPU, multi-tens-of-minutes hang, no output reaching the
agent stream — but at a higher scale (~40k cumulative tokens of
session context). H1, H2, H3 do not predict that signature
specifically; H4 might, but only if the session-log analysis lands.

## Claim

At cumulative context near the 003-Alpha threshold (~40k tokens),
`gemma4-think:26b` re-engages thinking-mode generation despite
`"think": false` being explicitly set on the request. The renewed
thinking generation either (a) produces a populated `message.thinking`
field in the response, (b) emits inline `<think>...</think>` markers
in `message.content`, or (c) fails to converge within a reasonable
wall-clock bound, reproducing the runaway.

Possible mechanisms (not mutually exclusive):

- A renderer-side bug where `"think": false` is honoured at small
  context but ignored once accumulated input exceeds some threshold.
- An OpenClaw-side regression where `"think": false` is dropped from
  certain turns at long context (this overlaps with H4 but is
  testable independently because we can replay the request directly
  to Ollama).
- A model-state bug where thinking is re-enabled after a particular
  number of tokens have been processed in a single Ollama session,
  regardless of request flags.

## Prediction (if H5 is true)

With a ~40k-token user message sent to `gemma4-think:26b` with
`"think": false` and `num_ctx=131072`, in 3 repeats:

- At least one repeat returns a non-empty `message.thinking` field, OR
- At least one repeat contains `<think>` or similar markers in
  `message.content`, OR
- At least two repeats fail to complete within a 600s timeout (a
  weaker signal — runaway-consistent but not proof of thinking
  specifically).

A small-scale sanity check (~200 chars, `think:false`) returns clean
content with empty thinking, validating that the test setup correctly
suppresses thinking at small scale.

## Prediction (if H5 is false / null)

All 3 incident-scale repeats complete within timeout, return clean
content, and have an empty `message.thinking` field with no `<think>`
markers in content. Then thinking-mode regression is not the cause of
003-Alpha, and the runaway has another origin (FA fallback, KV-cache
bug, OpenClaw control flow) — investigation continues with H1 and H4.

## Falsification

Three consecutive incident-scale runs that all complete within
timeout with empty `thinking` field and no `<think>` markers in
content. This is a strong falsification because the test directly
reproduces production conditions.

## Discrimination

- **vs H1** (FA CPU fallback): independent. H5 is about *what content
  the model generates*; H1 is about *where compute runs*. A confirmed
  H5 with FA off would explain the runaway without invoking H1.
- **vs H2** (MoE empty content, narrow): orthogonal. H2 tested base
  `gemma4:26b` with thinking ON at small scale and was rejected. H5
  tests the alias with thinking nominally OFF at large scale.
- **vs H4** (OpenClaw input cap): H5 reproduces against raw Ollama
  with no OpenClaw involved. If H5 confirms but the runaway only
  shows up via OpenClaw, H4 may still be a contributing factor.

## Method (summary)

1. Sanity check: small prompt (~200 chars), `think:false`, single
   request to `gemma4-think:26b`. Verify response has empty
   `thinking` and non-empty `content`. Aborts test if sanity fails.
2. Generate deterministic ~40k-token user message (~160k chars).
3. 3 repeats: POST to `/api/chat` with the long message, `think:false`,
   `num_ctx=131072`. Per-request timeout of 600s.
4. For each response, record: content length, thinking length,
   eval_count, total_duration, presence of `<think>` markers,
   completion vs timeout.

Decision rule encoded in the test script.

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
