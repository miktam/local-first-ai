# exp_013 — Local Audit Loop (recall instrument)

A controlled instrument for measuring **where** a local model fails on cross-document
compliance/code auditing, and whether a given change moves the needle. Follow-up to
`exp_012_cost_capability`, where `gemma4:26b` scored 0/8 while every Claude tier scored
5/8.

This is deliberately **not** an agent framework. The model is the only moving part so
that results are attributable to the model, not to orchestration.

## Why a custom loop instead of installing an off-the-shelf agent

A general autonomous agent (e.g. OpenClaw) is built to *do work* on a heartbeat with a
large skill surface. For a benchmark that's a liability: skill selection, retries, web
access and hidden state all confound attribution, and you'd be measuring
framework-plus-model. Use the instrument to measure; package a validated loop as a
deployment skill afterwards if you want day-to-day use.

## Files

| file | purpose |
|------|---------|
| `exp_013_local_audit_loop.py` | the instrument (stdlib + `requests`, no frameworks) |
| `CLAUDE.md` | working rules for Claude Code in this dir — read first |
| `intervention_ledger.md` | log of every change, classed generic / fair-evidence / leakage |
| `rubric.template.md` | skeleton to copy to `rubric.md` and pre-register before any run |
| `requirements.txt` | `requests` |

Run artifacts (`gaps.json`, `trace.jsonl`) are produced per run; commit the trace with
each result so the diagnosis is reproducible after the fact.

## Quick start

```bash
pip install -r requirements.txt
python exp_013_local_audit_loop.py \
  --code inference_log.py mcp_server.py \
  --policy ropa.md retention_schedule.md \
  --model gemma4:26b --num-ctx 32768 \
  --out gaps.json --trace trace.jsonl
# re-label an old run, no model calls:
python exp_013_local_audit_loop.py --diagnose-trace trace.jsonl
```

## Pipeline

0. **Budget + canary guard.** Estimate tokens vs `num_ctx`; embed a canary at the top of
   the context and refuse to proceed unless the model echoes it (Ollama drops the oldest
   tokens silently — that's your cached prefix).
1. **Decomposed extraction** (fresh contexts): 1a atomic code facts incl. absences; 1b
   atomic policy rules with article refs.
2. **Bridge**: facts × rules → candidate gaps, biased for recall.
3. **Fresh-context verifier**: each candidate re-checked alone against the source; quote
   the supporting line or mark unsupported.
4. **Deterministic assembly**: survivors sorted by severity in Python — no final LLM pass.
5. **diagnose()**: rule-based label of where the run fell off (see CLAUDE.md table).

## Design rationale (corrections to the original "multi-agent loop" plan)

The starting point was a Gemini-authored plan to run a "multi-agent consensus loop" over
Ollama. It was competent on throughput but aimed at the wrong bottleneck and had two
concrete bugs. What changed:

- **Explicit `num_ctx` + canary guard.** The plan relied on prefix caching but never set
  `num_ctx`. Ollama defaults to 4096 and silently truncates the oldest tokens, i.e. the
  exact system prefix it wanted cached. Unset, the experiment audits a half-loaded context.
- **No `temperature=0` + `min_p`.** The plan set both on the final turn. Greedy decoding
  ignores `min_p`/`top_p`/`top_k`; the filter is dead code. Pick a regime.
- **Fresh-context verifier replaces single-context self-critique.** Self-critique in one
  growing context just re-reads its own tokens. Each claim is re-checked in a clean
  context with only the source + that claim — an external check, not self-soothing.
- **Recall-direction loop.** The original loop only pruned ("remove the gap if evidence
  missing"), optimizing precision. gemma's problem was recall (zero true positives), so
  generation is tuned for breadth and pruning is confined to the verifier.
- **Decomposed bridging.** The misses (A2, A5) required connecting a code observation to a
  GDPR article. That leap is split into extract-facts, extract-rules, then match.
- **Schema-constrained output** via Ollama `format`, not `[FINAL JSON]` token-tag slicing.
- **Coverage logging + deterministic diagnosis** so recall is read directly off the trace.

## The honesty boundary

Generic scaffolding (decomposition, absence checklists, article-matching, schemas) is fair
and is the publishable result. Supplying a missing source file is fair-evidence — note it.
A prompt that names a rubric answer is leakage and voids the score. Log every change in
`intervention_ledger.md`. The clean headline is: *generic scaffolding moved gemma 0 → N
without naming a single rubric item.*

## Honest expectation

Better scaffolding may move gemma off zero on findable items, but exp_012's curve suggests
a 26B open model won't reach 5/8 on cross-document legal auditing through loop engineering
alone. The useful measurement is *how much* of the gap is scaffolding-addressable — and the
trace tells you exactly where the remainder is lost.
