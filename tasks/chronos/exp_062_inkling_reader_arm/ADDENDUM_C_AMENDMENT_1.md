# exp_062 Addendum C, Amendment 1: the same checks on the merged release

**Registered 2026-11-24, before any model is loaded on the new build.**
- **Go:** Andrei, 2026-11-24 ("Ok, let's run 1"), on the proposal made the same afternoon: build the first llama.cpp release that holds both fixes, run Addendum B's tool-call repro and Addendum C's speed check on it, and replace "provisional on a draft" with figures from a release.
- **Bars:** the bars below are Addendum C's, unchanged, plus one for the tool calls. They were set before the run and were not shown to him one by one.

## Why

- **Addendum C was provisional.** Its faster build was the head of a draft pull request: llama.cpp #32825 at `7b43c773`. Every published line about it says the pull request is not merged.
- **Both pull requests have since been merged** (read from GitHub's API on 2026-11-24, at about 16:30 UTC):
  - #32825, "metal : add NVFP4 support", merged 2026-11-23 at 19:02 UTC as `51831d47d0`. Its final head was `b284d58e`, not the `7b43c773` that Addendum C measured.
  - #32833, "chat : improve inkling parser/grammar", merged 2026-11-23 at 07:54 UTC as `dd53f70d08`. Its description lists "Fixes #32802", the issue Andrei filed on 2026-11-22 when llama-server dropped the research model's tool calls.
- **The release holds both.** Release tag `b12676` (published 2026-11-24 13:24 UTC, commit `d35dc4d9e6827db8e08426a2340d8a71ea863cd9`) is 4 commits past the first merge and 26 past the second, by GitHub's compare.

So, two questions:
1. **Does the release give what the draft gave?** That means the expert layers on the GPU, the same speed, the same model.
2. **Does llama-server on the release return the research model's tool calls?**

## Runtime, pinned

- **The new build:** `~/models/llama.cpp-b12676`, a shallow clone of tag `b12676` at the commit above. It is built exactly as Addendum C's three builds were: `cmake -B build -DCMAKE_BUILD_TYPE=Release`, target `llama-server`. The driver refuses to start if the commit differs.
- **The files:** Addendum C's two files, pinned by the same sizes and sha256: U is Unsloth's `UD-Q4_K_M` and N is ggml-org's `IQ4_NL`. For the tool calls, Addendum B's research file `TML-d24-varloop-research-IQ4_NL.gguf` is also used, at its default 3 passes.
- **Everything else:** Addendum C's text, prompts and jobs files are used, checked against the same hashes. Its seed, workloads, slot settings and thermal rule are unchanged.
- **Comparison figures** come from Addendum C's run folder, `runs_c/20261123T091929Z/`, and are not re-measured:
  - U0, which is `b12631`;
  - U2, which is the draft head `7b43c773`;
  - C-U0 and C-U2.

## Cells

| Name | Build | File | What |
|---|---|---|---|
| U3 | b12676 | U | Part 1 (two workloads, six requests each) and Part 3 (eight texts, token ids kept) |
| N3 | b12676 | N | the same, for the control file |
| P-U3 | b12676 | U | one short load with `GGML_SCHED_DEBUG=2`: where the expert products run |
| C-U3 | b12676 | U | Addendum A's Part 2: the fixed 12 jobs at 1, 2 and 4 watchers |
| T | b12676 | research, U, N | Addendum B's tool-call repro (`kit/repro62b_toolcall.py`), the same five made-up prompts, through `/v1/chat/completions` and raw |

There is no separate smoke test: the kit is Addendum C's, run unchanged except for the build, so cell U3 is the first check. Run folder: `runs_c2/<UTC stamp>/`. The repro's output goes to `dev_c2/toolcall-repro-b12676.txt`.

## Hypotheses

| | Claim | Passes if |
|---|---|---|
| **HD3** | the release computes the same model | with file U, the release's first token equals `b12631`'s (U0) on at least 7 of the 8 texts, **and** one watcher on the release is right on at least 11 of the 12 jobs |
| **HD1** | the fallback is gone on the release | file U generates at 22.8 tokens a second or more on the release (the Short workload, median of six) |
| **HD1b** | the gap is closed on the release | on both workloads, file U generates at 90 % or more of file N's speed on the release |
| **HD4** | the queue | with file U on the release, four watchers at once do more than 1.4 times the jobs a minute of one |
| **HD5** | the tool calls come back | for the research file, the chat endpoint returns a tool call on at least 4 of the 5 prompts, **and** both Mini files still return one on 5 of 5 |

- **Read first:** HD3 is read first, as HC3 was. If it fails, HD1, HD1b and HD4 are recorded as not readable.
- **Where the experts run:** P-U3 is reported as a count. Expected: no expert product on the CPU, as on the head.
- **Descriptive only:** the release against the draft head (U3 against U2: speed, and the agreed prefix in Part 3) and the same tool calls for one watcher (C-U0 against C-U3). The head and the release are different commits, so no bar is set between them.
- **HD4 failed on the head** (1.34). It is kept as registered, not re-set.

## Decision rule, declared now

- **Nothing changes in the stack, whatever the result.** No seat depends on this model's speed or on llama-server's parsing. Addendum B parsed the raw completions itself.
- **If HD3 and HD1 pass:**
  - a dated amendment note on `ADDENDUM_C_RESULT.md` says that the figures hold on a release, with the release's own numbers;
  - the blog post's update line ("the pull request is not merged") gets a proposed rewording for Andrei;
  - the post is not touched here.
- **If HD5 passes:** Andrei gets the counts and the raw tails for issue #32802, to confirm or close it in his own words. llama.cpp does not accept AI-written reports, so it gets a table from here and no prose.
- **If HD5 fails:** the same table goes to him, and the parse-it-yourself line stays.
- **If HD3 fails:** no speed from the release is quoted anywhere. The figures go to Andrei as a table.

## Stops

- **A pinned input that does not match** (the release's commit, a file's size or hash, a kit file's hash): the run stops before any model is loaded.
- **The release fails to load a file:** recorded. The bars that need that file fail as "does not run".
- **Thermal level:** Addendum C's rule.
- **Restarts:** each cell writes its file when it is complete, and a resume skips complete cells. The side server on port 18450 is stopped on any exit. The production Ollama is not called beyond `ollama ps`. The two small llama.cpp servers already running (ports 18442 and 18448) are noted and left alone.

## Time

From Addendum C's run, about 35 minutes:
- about a minute to hash the two files;
- 3 to 5 minutes for each cell;
- about 12 minutes for Part 2;
- about 10 minutes for the tool-call repro over three files.

## Limits

- **One machine, one chip** (M4 Pro, 64 GB), one run per cell.
- **The draft head is not re-run.** It is compared from Addendum C's record.
- **The repro is five made-up prompts.** It shows whether the parser returns the call. It does not show how often the model calls the right tool.
