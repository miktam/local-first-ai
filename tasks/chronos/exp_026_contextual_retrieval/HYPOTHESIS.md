# Experiment 026 — Contextual Retrieval on the COAPI Corpus, Fully Local

*Pre-registered: 2026-08-30 · Status: pre-registered, execution pending*

**Inspired by:** Anthropic's "Contextual Retrieval" — before embedding a chunk, an LLM generates a short (~50-100 token) blurb situating that chunk within its source document, prepended to the chunk before embedding. Anthropic's published results show this meaningfully cuts retrieval-failure rates, because a naive chunk (e.g., "the deposit must be returned within 15 days") is retrievable on its own once it reads "In the context of earnest money deposits under Module 6 of the COAPI contract law course, the deposit must be returned within 15 days."

**Follows from:** exp_023 (generation efficiency across the local model family) — this experiment reuses those numbers directly to reason about whether local contextualization is *practical*, not just whether it *works*.

---

## What this experiment tests

CasaSol's `coapi_knowledge_*` collections (1307 chunks currently indexed for English alone) are built by `scripts/index_coapi.py` with **zero structural awareness**: cleaned PDF text is sliding-windowed at 500 words / 50-word overlap, with no chunk-level context of any kind. This is exactly the failure mode contextual retrieval targets — unlike `area_knowledge` (already naturally segmented by zone, each chunk self-identifying via its own `## Zone Name` header), COAPI chunks can start or end mid-clause, reference "the above" or "section 3" without the referent, or lose the module/topic they belong to entirely.

This experiment: (1) confirms contextual prefixing improves retrieval accuracy on this real corpus using local models only, and (2) prices out what it costs to generate those prefixes locally, using exp_023's speed numbers to judge whether it's practical to run at the scale of 1307+ chunks (and growing, since COAPI content isn't static).

---

## Hypotheses

**H1 — Contextual prefixing improves retrieval accuracy on this corpus:** on a fixed query set (one synthetic query per chunk, generated from the chunk itself), top-3 retrieval hit rate is higher for contextually-prefixed chunks than for the same chunks unprefixed, using the same embedding model CasaSol already uses in production (`paraphrase-multilingual-MiniLM-L12-v2`).

*Null:* top-3 hit rate is the same or worse with prefixing — the naive chunks already carry enough signal for this embedding model, and the technique doesn't transfer to a small local embedding model the way it does in Anthropic's own (much larger) embedding setup.

**H2 — A cheap local model can generate adequate prefixes:** `gemma4:e4b` (fastest in exp_023, ~55 tok/s) produces contextual prefixes that perform comparably to `gemma4:26b`-generated prefixes on the same H1 retrieval-accuracy measure — i.e., you don't need the expensive model for this specific sub-task.

*Null:* `gemma4:e4b`'s prefixes measurably underperform `gemma4:26b`'s on retrieval accuracy, meaning contextualization quality requires the larger model.

**H3 — Full-corpus contextualization is practical at local speeds:** extrapolating exp_023's `gemma4:e4b` generation numbers to all 1307+ currently-indexed COAPI-EN chunks (one contextualization call per chunk, each call re-reading the relevant source module as context) puts total one-time processing time in a range practical for an overnight/background job on this hardware (order of hours, not days).

*Null:* the extrapolated cost is impractical (multi-day) at current local speeds, meaning contextual retrieval would need either a faster model, a cheaper prompt-caching-equivalent (see Design note below), or cloud escalation for this specific step.

---

## Design

**Corpus sample:** one full module — `tasks/coapi/materials/text/en/10.modulo_6_el_contrato.txt` ("Module 6 — The Contract", ~5,088 words raw). Chunked with the *exact* production logic from `scripts/index_coapi.py` (`clean_text()` + `chunk_text(500, 50)`, imported directly, not reimplemented) — expected ~10-12 chunks.

**Ground truth queries:** for each chunk, `gemma4:26b` (the more capable model, used here only to produce reliable ground truth — not the thing under test) generates one realistic query a COAPI trainee might ask, for which that chunk is the best answer. Known confound, stated up front: adjacent chunks share a 50-word overlap and legal training text is naturally repetitive/cross-referential, so some queries may plausibly match a neighboring chunk too. Retrieval is scored as a hit if the source chunk **or its immediate sliding-window neighbor** appears in top-3 — a deliberately generous, documented criterion, not a way to inflate the result: both the prefixed and unprefixed conditions face the identical ambiguity, so the *comparison* between them stays fair even if the absolute hit-rate numbers aren't pristine.

**Contextualization (H1, H2):** for each chunk, both `gemma4:e4b` and `gemma4:26b` independently generate a contextual prefix, given the full cleaned module text plus the specific chunk, using a prompt adapted from Anthropic's published template:

> "Here is a chunk from Module 6 of a Spanish real estate professional certification course: <chunk>. Please give a short, succinct context (1-2 sentences) situating this chunk within the module, to improve search retrieval of the chunk. Answer with only the context, nothing else."

**Embedding + retrieval:** three variants per chunk — unprefixed, `gemma4:e4b`-prefixed, `gemma4:26b`-prefixed — each embedded with CasaSol's actual embedding function (`scripts/embedder.py::get_ef()`) into three separate throwaway in-memory/temp ChromaDB collections (never touching the real `coapi_knowledge_en` production collection). Each of the module's own queries run against all three collections; top-3 hit rate computed per variant.

**Cost accounting (H3):** timing/token counts captured per contextualization call (same fields as exp_023), extrapolated to the full current corpus size (1307 chunks, English only, checked fresh via `collection.count()` at run time since it grows).

---

## Evidence artefacts

```
results/
  run_YYYYMMDDTHHMMSSZ.json
```

Each result: per-chunk queries, per-variant top-3 hit/miss, contextualization timing/tokens per model, aggregate hit rates, and the H3 extrapolation.

---

*Experiment design: Andrei + Claude Sonnet 5 · 2026-08-30*
