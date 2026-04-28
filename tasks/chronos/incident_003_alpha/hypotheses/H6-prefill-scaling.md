---
id: H6
title: Prefill time scales as O(N²) in input length on gemma4-think:26b
status: open
test_script: tests/h6_prefill_scaling.sh
related_upstream: []
created: 2026-04-28
opened_in_response_to:
  - H1 rejected (no CPU fallback evidence in Ollama logs)
  - H5 timeouts at 40k tokens explainable by O(N²) extrapolation
    from the single 23k completion (289s) observed before the
    cache-defeating patch
---

## Background

Transformer prefill (the prompt-evaluation phase) scales as O(N²)
in input length because each new token's self-attention attends to
all previously-processed tokens. On hardware that does not throttle
or fault, prefill time should fit a curve t(N) ≈ aN² + bN + c with
the quadratic term dominant for large N.

We have one data point so far: 23305 tokens completed in 289s
(2026-04-27, before the cache-defeating patch on H5). If the stack
is healthy, smaller and larger sizes should fit the same curve.

The original 003-Alpha incident reported 17–43 minute hangs at ~40k
tokens of accumulated session context. Naive O(N²) extrapolation
from the single data point predicts 14 minutes at 40k and ~30 minutes
at 60k of on-the-wire prompt. The brackets match. If H6 is supported,
003-Alpha was never a bug — it was the cost of long-context inference
on this hardware, mislabelled as a runaway.

## Claim

Prefill time on `gemma4-think:26b` running on miktam02 fits a
quadratic curve in input token count, with no inflection point or
non-quadratic component up to at least 35k tokens. Specifically:

- Prefill rate (ms/token) increases approximately linearly with N.
- Total prefill time fits t(N) ≈ aN² to within 30% across the
  measured range.
- Tokens stream out normally once prefill completes.
- GPU is active throughout prefill (not idle, not occasional blips).

## Prediction (if H6 is true)

Sweeping {15k, 25k, 35k} input tokens, with one fresh attempt per
size (Ollama restarted between):

- All three completions occur within their respective timeouts
  (1800s ceiling generous for any size).
- Prefill durations roughly proportional to N²: t(35k) / t(15k) ≈
  (35/15)² = 5.4, t(25k) / t(15k) ≈ 2.8.
- ms/token at 35k is roughly 35/15 ≈ 2.3× the ms/token at 15k.
- GPU windowed-mean power > 5 W in all three runs (clear GPU work).
- First streamed token arrives within ~5 s of prefill completion
  (eval phase healthy).

## Prediction (if H6 is false / null)

One or more of:

- Prefill at 35k or 25k fails to complete within 1800s while 15k
  completes — an inflection somewhere in the range.
- ms/token does not scale linearly with N (e.g. flat across sizes,
  or jumps non-linearly).
- GPU drops to idle for extended stretches during prefill.
- A request stalls completely with no token stream after prefill.

Any of these means the simple O(N²) explanation does not cover all
of the observed behaviour, and there is some additional bug we
have not yet identified.

## Falsification

The 35k case must complete and produce a streamed response. If it
does not, H6 is rejected and we are looking at a real cliff, not
just expensive scaling.

## Discrimination

- **vs all prior hypotheses**: H6 predicts that prefill duration
  is the *only* effect at long context. If H6 fits, no runaway
  needs explaining.
- **vs an undiscovered bug above 35k**: H6 cannot distinguish "the
  curve continues smoothly to 60k" from "there is a cliff between
  35k and 60k." A second sweep above 35k would be needed. We start
  at 35k because it is the largest size that still fits in a
  reasonable wall-clock budget.

## Method (summary)

For each size in {15000, 25000, 35000}:

1. Stop OpenClaw gateway if running.
2. Restart Ollama via launchd (fresh model load, empty KV cache).
3. Generate a unique seeded prompt of the target size using the
   calibrated 7 chars/token ratio, with a brief end instruction
   to elicit a short response.
4. Start powermetrics sampler (cap covers expected wall time).
5. Issue `/api/chat` with `stream: true`, capture every event line.
6. Record:
   - First-byte time (when streaming starts)
   - First-token time (when the first non-empty content chunk arrives)
   - Last-token time (when `done: true`)
   - prompt_eval_count, prompt_eval_duration, eval_count,
     eval_duration from the final event
   - Windowed mean GPU and CPU power
7. Stop on the first failure to complete: do not run subsequent
   sizes if a smaller size already failed.

Decision rule (encoded in script):

- **Supported**: all three sizes complete; prefill time fits aN² to
  within 30%.
- **Rejected (cliff)**: the largest size fails to complete or shows
  prolonged GPU idle without progress.
- **Inconclusive**: completes but does not fit O(N²) cleanly, or
  GPU/CPU pattern is anomalous in a way that warrants a deeper look.

## Result

(Filled in by `results/<date>-summary.md` after the test runs.)
