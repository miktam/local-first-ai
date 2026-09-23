# Eval v2: construction record

How the fresh 30-row set for exp_035 was built (HYPOTHESIS.md, blocks dated 2026-09-23). The rows themselves are withheld until the set is sealed and the reference build is scored once.

| File | What |
|---|---|
| `grid_v2.jsonl` | the fixed design: 30 cells, module × type × language |
| `r1_v1_overlap.json` | round 1: 23 of 28 in-scope rows re-tested a v1 rule (18 as a v1 headline, 5 as a supporting sentence); wording similarity flagged 1 |
| `r2_v1_overlap_decisions.json` | rounds 2 and 2b: per-row decision, v1 golds read, reason |
| `dedupe_summary.json` | character-trigram Jaccard vs v1 questions, index questions and training questions (ids and scores only) |
| `gates_final.json` | schema/grid/citation gates, arithmetic recomputed in code, dedupe, on the set as reviewed |
| `blindness_summary.json` | which files the writer and reviewer agents opened (counts and paths) |
