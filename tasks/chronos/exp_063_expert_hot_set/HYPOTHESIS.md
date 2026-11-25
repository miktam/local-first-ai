# exp_063 — is there a stable hot set of experts to cache?

**Registered 2026-11-25, before any of the eight texts was run.** Prompted by a post of 2026-11-24 on Magnitude, an inference engine that keeps a mixture-of-experts model's shared layers and "most used experts" in GPU memory, the next tier in RAM, and streams the rest from SSD. A cache of "most used experts" presumes that the experts used most on past text are used most on the next text. This tests that premise on two models already on the mini.

## Seen before registration

A smoke on one degenerate text (18 English words shuffled, 1,024 tokens), to check that the files load and the counts add up. Its counts were looked at: in gemma block 0 the most used expert fired on 755 of 1,024 tokens. No coverage figure was computed, and none of the eight texts below had been run. The smoke also showed that Inkling's blocks 6–17 count 24 per token, not 8; the file's metadata says why (loop.start 6, loop.end 18, loop.count 3). Each block is therefore normalised by its own total.

## Method

- **Tool:** `llama-imatrix`, llama.cpp `b12676` (d35dc4d), built separately in `~/models/exp063/build` (`-DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF`). It counts, per block, how often each expert is selected. Counts, not weights or activations, are used.
- **Models:** `gemma4:26b-a4b-it-qat` (Ollama blob sha256-4c856523d61d…, 30 blocks, 128 experts, 8 per token) and Inkling 2 Mini UD-Q4_K_M (24 blocks, 128 experts, 8 per pass). The first is the QAT file, not the production `gemma4:26b`, whose blob llama.cpp does not load.
- **Texts:** four classes, two halves each from different source files: English book, Russian lecture transcripts, Spanish website, C++ code (`kit/texts.json`). 12 chunks of 512 tokens each = 6,144 tokens. The texts stay on the mini; only their hashes and the counts are recorded.
- **Measure:** rank a block's experts on half *a*; the score is the top-k's share of selections on half *b*, averaged over blocks. k = 25, 50, 75 % of experts; the primary is **50 % (64 of 128)**. The null is uniform use, which gives 50 %.

## Hypotheses (per model, at 50 %)

- **H1, a hot set exists and holds within a class:** rank *a*, score *b* of the same class ≥ 0.80, in all four classes.
- **H2, it is specific to the class:** same-class coverage minus the mean coverage from the other three classes' rankings, averaged over the four, is ≥ 0.10.
- **H3, one mixed hot set serves everything:** the ranking from all four *a* halves together covers each *b* half ≥ 0.75.

Reported without a threshold: 25 % and 75 %, the leave-one-class-out ranking, and the oracle (rank and score on the same *b* text), an upper bound.

**Futility stop:** if gemma's English same-class coverage at 50 % is below 0.55, stop; nothing is worth caching.

## What this cannot show

- **Not Magnitude.** Its engine, its 180B 512-expert model at 2 bits and its tokens per second are not tested. The file is 70.1 GB (its model page) against 85 GB free here.
- **Marginal frequencies only.** The counts say nothing about the next token, so prediction and prefetch, the other half of the claim, are untested. Nor can they give how often a token finds all its experts present.
- **Reading text, not generating it.** Routing while the model writes may differ.
- Two models of similar size, both 128 experts; small texts; class and genre are confounded (the English is a book, the Spanish a website).
- **No timings.** Another session was using the CPU.
