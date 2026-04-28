# Incident 003-Alpha — Investigation

Companion repository to the `scientific_log.md` entry for Incident
003-Alpha (2026-04-26) and its 2026-04-27 update.

## What this is

A preregistered, script-driven investigation into the Nestor +
OpenClaw + Ollama + Gemma 4 26B-A4B runaway observed at ~40k tokens
of accumulated session context on miktam02 (Apple M4 Pro Mac Mini,
64 GB unified memory).

The 2026-04-26 entry framed the failure as an operating envelope of
the stack. The 2026-04-27 update revises that: the runaway is more
likely an interaction between recent Ollama bugs in the Gemma 4 path
and OpenClaw's context negotiation, not a property of miktam02's
hardware envelope. This repo tests four hypotheses under that revised
framing.

## Layout

```
incident-003-alpha/
├── README.md                    # this file
├── hypotheses/                  # one preregistered hypothesis per file
│   ├── H1-fa-cpu-fallback.md
│   ├── H2-moe-empty-content.md
│   ├── H3-num-ctx-negotiation.md
│   ├── H4-openclaw-token-drift.md
│   └── H5-thinking-regression.md
├── tests/                       # one script per hypothesis
│   ├── h1_fa_cpu_fallback.sh
│   ├── h2_moe_empty_content.sh
│   ├── h3_num_ctx_negotiation.sh
│   ├── h4_openclaw_token_drift.sh
│   └── h5_thinking_regression.sh
├── lib/                         # shared helpers
│   ├── ollama_probe.sh
│   └── powermetrics_probe.sh
├── evidence/                    # APPEND-ONLY, timestamped, never edited
└── results/                     # one summary per investigation run
```

## Test script contract

Every script in `tests/` follows the same contract:

- **Stdout**: a single JSON object on the last line:
  `{"hypothesis_id":"...", "status":"...", "evidence_dir":"...", "summary":"..."}`
- **Stderr**: human-readable progress.
- **Exit codes**:
  - `0` — hypothesis supported
  - `1` — hypothesis rejected
  - `2` — inconclusive (test ran cleanly but signal is mixed)
  - `3` — test could not run (environment problem, missing tool, etc.)
- **Evidence**: written to a fresh timestamped subdirectory of
  `evidence/`. Older directories are never modified.
- **Self-contained**: no test depends on another test having run.

## Prerequisites

- macOS on Apple Silicon (tested target: M4 Pro)
- `ollama` running on `http://127.0.0.1:11434`
- `gemma4-think:26b` pulled — local alias of `gemma4:26b` with explicit
  `num_ctx=131072`. Production target. Used by H1 and H5.
- `gemma4:26b` pulled — base model. Used by H2 and H3.
- `gemma4:31b` pulled (optional, used by H2 for cross-architecture comparison)
- `jq`, `curl`, `python3` on `$PATH`
- `sudo` access (H1 only — needed for `powermetrics`)
- An OpenClaw installation with at least one session under
  `~/.openclaw/agents/main/sessions/` (H4 only)

## Note on the gemma4-think alias

OpenClaw uses a name-based heuristic to decide whether a model
supports thinking mode: only models whose name contains "r1",
"reasoning", or "think" get `"think": false` sent on chat requests.
Gemma 4's renderer defaults thinking ON unless that flag is sent.

The local alias `gemma4-think:26b` (defined as `FROM gemma4:26b` plus
`PARAMETER num_ctx 131072`) was created on 2026-04-12 to satisfy the
heuristic. It shares weights with the base model — same Metal
kernels, same llama.cpp path, same KV cache layout — so the FA bug
H1 tests for is bug-equivalent against either. We test the alias
because that's what production uses.

H1 and H5 target the alias and pass `"think": false` to match
production. H2 and H3 target the base model because their claims are
about base-model behaviour.

## Running

Run a single hypothesis:

```
bash tests/h3_num_ctx_negotiation.sh
```

Run everything and write a session summary to `results/`:

```
bash run_all.sh
```

The runner does not stop on a failed hypothesis; every test runs and
its result is recorded.

## Reading evidence

Each evidence directory is timestamped in UTC:
`evidence/2026-04-28T09-12-44Z-H1/`. Directory contents vary by test
but always include at least:

- `run.json` — structured summary of the run
- `stderr.log` — full human-readable trace
- raw probe outputs (powermetrics, ollama ps, response bodies, etc.)

If you re-run the same test, a new timestamped directory is created.
The old one stays. This is deliberate: hypotheses can flip from
supported to rejected over time as upstream code changes, and both
runs need to remain on disk for the log to remain auditable.

## Conventions

- Hypothesis IDs are stable. `H1` always means FA CPU fallback.
- Hypothesis files (`hypotheses/H*.md`) are written before the test
  runs. Predictions are not edited after evidence is collected.
- Test scripts can be revised, but revisions are tracked in git.
- Results land in two places: a one-paragraph entry in the project's
  `scientific_log.md`, and a longer entry in `results/<date>-summary.md`
  citing the evidence directory.
