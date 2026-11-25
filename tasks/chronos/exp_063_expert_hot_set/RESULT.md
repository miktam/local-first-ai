# exp_063 — result

**Run `runs/20261125T082826Z`, 2026-11-25, on the Mac mini (M4 Pro, 64 GB).** Sixteen text-by-model runs in one folder, two invocations (the futility pair, then the other fourteen), no failure; every block counted exactly 8 selections per token (24 in Inkling's looped blocks 6–17), 6,144 tokens per text. The futility stop did not fire (0.931 against 0.55). Scored by `kit/score63.py` as registered.

## Verdicts (half the experts kept, 64 of 128; uniform use would give 0.50)

| | gemma4 26b QAT | Inkling 2 Mini |
|---|---|---|
| **H1** hot set holds within a class (each ≥ 0.80) | **pass** 0.89–0.93 | **pass** 0.82–0.92 |
| **H2** it is class-specific (mean drop ≥ 0.10) | **pass** 0.282 | **pass** 0.313 |
| **H3** one mixed set serves everything (each ≥ 0.75) | **fail**: prose 0.86–0.87, code 0.69 | **fail**: prose 0.80–0.86, code 0.53 |

Share of selections on the *b* half (columns) when the hot set was ranked on the *a* half (rows):

| gemma, 50 % | English book | Russian lecture | Spanish site | C++ |
|---|---|---|---|---|
| English book | **0.93** | 0.85 | 0.73 | 0.34 |
| Russian lecture | 0.89 | **0.89** | 0.80 | 0.33 |
| Spanish site | 0.77 | 0.70 | **0.93** | 0.41 |
| C++ | 0.54 | 0.55 | 0.63 | **0.89** |

| Inkling, 50 % | English book | Russian lecture | Spanish site | C++ |
|---|---|---|---|---|
| English book | **0.92** | 0.79 | 0.69 | 0.34 |
| Russian lecture | 0.88 | **0.87** | 0.73 | 0.36 |
| Spanish site | 0.71 | 0.69 | **0.87** | 0.37 |
| C++ | 0.31 | 0.38 | 0.42 | **0.82** |

- A hot set ranked on prose serves code **worse than 64 experts chosen at random would** (0.33–0.41 against 0.50), in both models. Inkling's code set is also worse than random on prose (0.31–0.42); gemma's is slightly better (0.54–0.63).
- Keeping 75 %, the mixed set covers prose at 0.94–0.97 and code at 0.88 (gemma) and 0.79 (Inkling).
- The set ranked on a related text is 0.01–0.08 below the best fixed set for the scored text (the oracle, 0.88–0.95 at 50 %); the widest gap is Inkling on code.

Written after scoring, descriptive: leaving code out, the drop between the three prose classes is 0.128 (gemma) and 0.137 (Inkling), so H2 does not rest on code alone.

## Limits

- **"Same class" is near to "same source".** The halves are different files, but two chapters of one book, two sets from one lecture course, two sets of pages from one site and two files from one code base. H1 does not say a set holds for a new author or a new code base.
- **The mixed set is three parts prose to one part code.** H3's failure is for that mix. A set with more code in it would cover code better and prose worse.
- Counts over a text, not token by token: nothing here says how often a token finds all its experts present, whether the next token's can be predicted, or how fast anything runs. An adaptive cache would replace its set when the text changes; this measures a fixed one.
- Reading, not generating. Two models of 128 experts; Magnitude's own engine and models were not run.

## SO WHAT

- The premise of "keep the most used experts" holds on both models: half the experts, chosen on related text, took 82–93 % of the selections.
- The set belongs to the kind of text. A fixed set does not transfer between prose and code, and a personal agent moves between them. Whether that slows an engine depends on how fast it re-ranks, which is untested. The question to put to an expert-streaming benchmark is which prompt it used, and whether the cache was warm on it.
- What the miss rate costs is not measured here. That needs the engine, and the counts per token.
- Nothing changes in the stack. Every model in use on the mini fits in memory.

## Corrections, 2026-11-25 (same day, before any push)

- "One start" was wrong: two invocations.
- The SO WHAT said a single-prompt speed "says little about a mixed day" and that Magnitude's table gives no prompt. The first is not measured; the second rested on a page read by a subagent and not filed. Both removed. The same goes for the model's 512 experts, and for the 70.1 GB in HYPOTHESIS.md, which stays as registered.
- The registration's reason for not running Magnitude reads as if 70.1 GB did not fit in 85 GB. The reason is that it needs a self-updating engine installed and a download of that size, a decision of the owner's.
- Added on the same day after reading Magnitude's source at commit `78c92e4`: its cache is not a fixed set. It re-ranks by recent reads, so a fixed set's failure across kinds of text does not show that its cache fails.
