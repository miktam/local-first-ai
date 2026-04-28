# Experiment 004: Bootstrap diet

Cut Nestor's per-turn prompt overhead by separating per-turn essentials
from on-demand reference material. Motivated by the prefill cliff
identified in [Incident 003-Alpha](../incident_003_alpha/).

## Background

The 003-Alpha investigation closed with the prefill cliff measured at
roughly 25,000 tokens of on-the-wire prompt. What 003-Alpha did not
quantify was how much of that budget was already spent on fixed
overhead before any user conversation began.

Inspecting a captured `prompt.submitted` trajectory record on
2026-04-28 revealed that every turn shipped:

- ~11,000 chars (~1,571 tokens) of OpenClaw-assembled system prompt
  (tool catalogue, skill list, runtime block)
- ~28,000 chars (~4,020 tokens) of seven workspace MD files loaded
  by the bootstrap path
- Combined: ~5,591 tokens of overhead per turn, before any
  conversation history

That left ~19,400 tokens of practical conversation headroom under
the conservative cliff — much less than the nominal 131,072-token
context window suggested.

## Hypothesis

Most of the per-turn overhead was reference material that did not
need to load on every turn. Specifically: heartbeat protocols,
group-chat etiquette, log-formatting rules, and biographical context
that only mattered when the conversation entered a relevant domain.

Splitting these into on-demand files — loaded by trigger, not by
default — would reduce per-turn overhead substantially without
losing behavioural coverage.

## Method

Two diagnostic tools written first, both read-only:

- [`tools/bootstrap_tokens.sh`](../tools/bootstrap_tokens.sh) —
  measures per-file token cost, reports total bootstrap overhead,
  computes remaining headroom against the 003-Alpha cliff.
- [`tools/bootstrap_lint.sh`](../tools/bootstrap_lint.sh) —
  heuristic linter flagging long files, list-heaviness, deep
  nesting, UI-paste dumps, emoji density, cross-file duplicates.
  Stack of findings on one file is the signal to cut, not any
  single warning.

Tools tuned the chars/token ratio to 7 based on the H6 measurement
(see `CALIBRATION.md` in the incident directory).

The cut itself, in two passes:

1. **AGENTS.md split.** Lean AGENTS.md (~1.2K, identity + memory
   model + safety + pointers) plus three protocol files in
   `workspace/protocols/`:
   - `heartbeat.md` — read when a heartbeat poll fires
   - `log-presentation.md` — read before showing structured output
   - `developer-mode.md` — read when starting any 3+ step
     engineering task
2. **USER.md split.** Lean USER.md (~1.5K, identity + behavioural
   rules + pointers) plus four background files in
   `workspace/background/`:
   - `history.md` — long-term biographical context, rarely loaded

The lean AGENTS.md and USER.md include explicit triggers for each
on-demand file, so Nestor can decide when to read them.

## Result

| Metric | Before | After | Delta |
|---|---|---|---|
| AGENTS.md tokens | 1,587 | 168 | -89% |
| USER.md tokens | 1,445 | 213 | -85% |
| Tracked files total | 4,020 | 1,369 | -66% |
| Bootstrap overhead per turn | 5,591 | 2,940 | -47% |
| Linter findings | 7 | 0 | — |
| Conversation headroom (conservative cliff at 25k) | ~19,400 tok | ~22,000 tok | +13% |
| Conversation headroom (practical ceiling at 35k) | ~29,400 tok | ~32,000 tok | +9% |

See `measurements/` for raw output. The "before" snapshot was
captured during interactive investigation and not preserved by
script; the after-snapshot captures are reproducible by running the
tools against the current `workspace_snapshot/`.

## Pending verification

The token savings are measured. The behavioural wiring is not.
Three sanity checks remain to confirm the on-demand files actually
load when triggered:

1. Trigger a heartbeat → check the trajectory for a read of
   `protocols/heartbeat.md`.
2. Ask Nestor to display structured tool output → check for a read
   of `protocols/log-presentation.md`.
3. Give Nestor a multi-step engineering task → check for a read of
   `protocols/developer-mode.md`.

Each check uses the trajectory log under
`~/.openclaw/agents/main/sessions/*.trajectory.jsonl`. If the
expected protocol read is missing, the trigger language in the lean
files needs to be more directive.

These verifications are non-blocking but should run before the
experiment is considered fully closed.

## Reproducing

The tools used here are reusable for any future audit of Nestor's
prompt overhead. Re-run them whenever new files are added to the
workspace bootstrap path or when behaviour suggests bloat:

```bash
TOOLS=~/REPOS/local-first-ai/tasks/chronos/tools
"$TOOLS/bootstrap_tokens.sh"
"$TOOLS/bootstrap_lint.sh"
```

The lint heuristics tend to fire on files that have been useful at
some point and grown by accretion. Periodic re-runs catch the
accretion before it consumes the cliff budget.

## Status

Closed (with pending behavioural verification — see above).
