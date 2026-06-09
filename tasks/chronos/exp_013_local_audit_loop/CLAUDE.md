# CLAUDE.md — exp_013 local audit loop

Context and working rules for Claude Code when operating in this directory.
Read this before changing anything here.

## What this is

An **instrument**, not a product. exp_013 measures *where* a local model fails on a
cross-document compliance/code audit, and whether a given change moved the needle. It
is the controlled follow-up to exp_012 (`tasks/chronos/exp_012_cost_capability/`).

The whole value is that the model is the only moving part. Do not wrap this in an
agent framework (OpenClaw, LangChain, AutoGen, etc.) for benchmarking — that
reintroduces hidden state and confounds attribution. Frameworks are for *deploying* a
validated loop, never for *measuring* one.

## The measured starting point (from exp_012)

- `gemma4:26b` (local, Ollama, 64GB unified memory) scored **0/8**.
- The cheapest cloud model (Claude Haiku 4.5) scored **5/8**; Sonnet and Opus also 5/8.
- The gap is **recall-shaped**: gemma generated zero true positives. It is NOT a false-
  positive / precision problem and NOT a formatting or speed problem.
- Therefore: optimizing throughput, sampling, or output hygiene cannot move the score.
  Only interventions that improve *generation and bridging* can.

## First job is localization, not optimization

We do not yet know *why* gemma got 0/8. The first run's purpose is to localize the
drop-off, not to raise the score. The `diagnose()` stage labels every run:

| label              | meaning                                              | licensed fix |
|--------------------|------------------------------------------------------|--------------|
| `extraction_empty` | Stage 1 produced no facts/rules                      | present/absent checklist over property types; re-check the canary (could be num_ctx truncation) |
| `no_bridge`        | facts+rules existed, Stage 2 connected nothing       | make bridging a *matching* task over supplied rule text, not a recall task |
| `vague_candidates` | candidates generated, all too generic, verifier killed them | require a concrete anchor (file/line/article/identifier) per gap |
| `over_pruned`      | *specific* candidates all rejected                   | verifier too strict — run calibration check, loosen verifier before touching upstream |
| `produced_output`  | no structural zero                                   | remaining gap is rubric mapping (manual) |

`diagnose()` is deterministic and rule-based on the trace — **never make it call a
model**. Its reproducibility is the point.

## The fairness boundary (do not cross without flagging)

Every intervention goes in `intervention_ledger.md`, classed on this spectrum:

1. **generic-scaffolding** — decomposition, absence checklists, article-matching, output
   schemas. Fair and publishable. The interesting result is "generic scaffolding moved
   gemma 0 → N without naming a single rubric item."
2. **fair-evidence** — supplying a source file that was missing from the context bundle
   (e.g. the VLM file behind A2). Fair, but note it changed the inputs.
3. **rubric-leakage** — any prompt or checklist item that names or paraphrases a specific
   rubric answer (e.g. "check whether a model hash exists"). **Forbidden.** It voids the
   score; the number stops being interpretable.

If a proposed change is class 3, refuse it and say so. If unsure between 1 and 3, treat
it as 3 until proven otherwise.

## Invariants that must not regress

- Explicit `num_ctx` on every call **and** the canary head-retention guard. Ollama
  truncates the oldest tokens silently; the cached system prefix is what gets dropped.
- No `temperature=0` paired with `min_p`/`top_p`/`top_k` — greedy decoding ignores them.
- Verifier runs in a **fresh context per candidate**, seeing only the source + that one
  claim. Never let it see other candidates or prior turns.
- Stage 4 assembly is **deterministic Python** — no final LLM pass that could re-add noise.
- Scoring stays **manual** against a **pre-registered** `rubric.md`. Never tune prompts
  toward a target score.

## How to run

```bash
pip install -r requirements.txt   # just `requests`

# full run (prints diagnosis to stderr, writes gaps.json + trace.jsonl)
python exp_013_local_audit_loop.py \
  --code <code files...> --policy <policy/regulation files...> \
  --model gemma4:26b --num-ctx 32768 \
  --out gaps.json --trace trace.jsonl

# re-diagnose an existing trace with no model calls (reproducible)
python exp_013_local_audit_loop.py --diagnose-trace trace.jsonl
```

Confirm `ollama show <model>` reports the `num_ctx` you set, not the 4096 default.

## Pipeline at a glance

Stage 0 budget + canary guard → Stage 1a code facts / 1b policy rules (decomposed, fresh
contexts) → Stage 2 bridge facts×rules into candidate gaps (recall-biased) → Stage 3
fresh-context verifier per candidate → Stage 4 deterministic assembly → `diagnose()`.

## Provenance

Built from a review of a Gemini-authored "native Ollama multi-agent loop" plan. The
corrections that turned that plan into this instrument are documented in `README.md`
(silent num_ctx truncation, the temp0+min_p contradiction, single-context self-critique
replaced by an external fresh-context verifier, recall-direction loop, schema outputs,
coverage logging). Keep that rationale intact when editing.
