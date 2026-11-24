# exp_062 Addendum C, Amendment 1, result: on llama.cpp's release b12676 the Unsloth file runs as fast as the draft made it, computes the same tokens as the draft did, and llama-server now returns the research model's tool calls; four watchers are still a queue

**Run:** `runs_c2/20261124T163640Z`, 2026-11-24 16:36Z to 16:53Z, then the tool-call repro until 17:00Z (24 minutes in all). Mac mini (M4 Pro, 64 GB), macOS 27.2.1, llama.cpp release `b12676` (`d35dc4d9`), built 2026-11-24 the same way as Addendum C's builds. Registration: [ADDENDUM_C_AMENDMENT_1.md](ADDENDUM_C_AMENDMENT_1.md) (`a93dfe0`), committed before any model was loaded. One start, no stop. The run script scored its own output in its last step (`kit/score62c2.py`, committed with the registration), so the raw files and the score are committed together.

**This replaces "provisional on a draft" for the speed figures.** Both pull requests are merged: #32825, NVFP4 on Metal, merged 2026-11-23 as `51831d47`; #32833, the Inkling parser, merged 2026-11-23 as `dd53f70d`. Both are in release `b12676`.

## Checks

- **Pinned inputs.** The release's commit, both files' sizes and sha256, and the kit files' hashes matched before the start (`meta.json`, `pinned.ok` true).
- **Thermal.** Level 0 at the start, before every cell and every level of Part 2, and at the end. No cell is flagged.
- **Network.** Under exp_054's audit hook, every connection from the three children went to `127.0.0.1:18450`, the side server (143 connections). The tool-call repro ran outside the hook, as it did on 2026-11-22, against the same side port. `ollama ps` was empty before and after. The two small llama.cpp servers already running (ports 18442 and 18448) were left alone.
- **One edit before commit.** In `meta.json`, `pgrep -fl llama-server` had also matched the launching shell's own command line. That line was removed and the edit is noted in the file. Nothing else was changed.

## Verdicts

| | Bar | Release b12676 | For comparison (Addendum C) | Verdict |
|---|---|---|---|---|
| **HD3**, same model, read first | first token as `b12631` on ≥ 7 of 8; one watcher right on ≥ 11 of 12 | **8 of 8; 12 of 12** | the draft head: 8 of 8; 12 of 12 | **pass** |
| **HD1**, fallback gone | Unsloth file, Short workload, ≥ 22.8 tok/s | **25.90** | `b12631` 15.86; draft head 25.84 | **pass** |
| **HD1b**, gap closed | Unsloth ≥ 90 % of ggml-org's, both workloads | **97.2 % (Short), 96.3 % (2,048-token)** | draft head 97 % and 97 % | **pass** |
| **HD4**, the queue | four watchers > 1.4 × the jobs a minute of one | **1.20** | draft head 1.34; `b12631` (experts on the CPU) 1.60 | **fail** |
| **HD5**, tool calls come back | research file ≥ 4 of 5 returned; both Mini files 5 of 5 | **research 5 of 5; Mini 5 of 5 and 5 of 5** | `b12631`: research **0 of 5**; Mini 5 and 5 | **pass** |

## What the release gives

**Speed.** Medians of six, tokens a second:

| Cell | Build | File | Prefill, 2,048 | Generation, 2,048 | Prefill, 128 | Generation, 128 → 512 |
|---|---|---|---|---|---|---|
| U0 | `b12631` | Unsloth | 70.48 | 15.29 | 69.41 | 15.86 |
| U2 | draft head `7b43c773` | Unsloth | 248.49 | 25.11 | 202.25 | 25.84 |
| **U3** | **release `b12676`** | **Unsloth** | **249.30** | **24.86** | **208.63** | **25.90** |
| N2 | draft head | ggml-org | 254.31 | 25.79 | 209.74 | 26.65 |
| **N3** | **release** | **ggml-org** | **253.42** | **25.82** | **211.54** | **26.65** |

**Where the experts run.** One load of the Unsloth file on the release, with `GGML_SCHED_DEBUG=2`:
- **all 96 expert products (`MUL_MAT_ID`) and all 977 other products on the GPU**;
- the graph is cut into 2 pieces (one CPU, one GPU), against 90 on `b12631`.

The raw log stays outside the repository (`~/models/exp062c/`).

**Same text.** The release and the draft head give **identical token ids to the end on all 8 texts** (256 tokens each). Against `b12631`, the release agrees on the first token on 8 of 8, and on a median of 16.5 tokens before the first difference: the same as the draft head against `b12631` in Addendum C. So everything Addendum C described about the GPU path's departure from the CPU path holds for the release, and it is still not explained. One watcher made the same tool calls on the release as on `b12631` on 12 of 12 jobs.

**Watchers at once** (Addendum A's 12 jobs, Unsloth file, one pass a level):

| | 1 | 2 | 4 | 4 ÷ 1 |
|---|---|---|---|---|
| Jobs a minute, release | 5.73 | 5.81 | 6.86 | **1.20** |
| Jobs a minute, draft head | 5.71 | 5.81 | 7.64 | 1.34 |
| Tokens a second, release | 21.2 | 26.7 | 31.6 | 1.49 |
| Tokens a second, draft head | 21.2 | 26.7 | 32.1 | 1.51 |
| Right, release | 12 | 12 | 12 | |

- **One and two watchers** are the same on the two builds to two decimals.
- **Four watchers** differ by 0.78 jobs a minute while the tokens a second differ by 1.5 %. So the difference is in how long the jobs ran at that level, not in the server's rate. With one pass a level that is within what one pass can show, and Addendum C already called it coarse.
- **The queue line stands:** four watchers on one mini do about 1.2 to 1.5 times the work of one, depending on the measure.

**The tool calls (issue #32802).** The same five made-up prompts as on 2026-11-22, through `/v1/chat/completions`. Full output: `dev_c2/toolcall-repro-b12676.txt`.

| File | `b12631` (2026-11-22) | release `b12676` (2026-11-24) | The model writes the tool's name before the call marker |
|---|---|---|---|
| research, ggml-org IQ4_NL, 3 passes | **0 of 5** returned, finish `stop` | **5 of 5** returned, finish `tool_calls` | 0 of 5 on both builds |
| Inkling 2 Mini, Unsloth UD-Q4_K_M | 5 of 5 | 5 of 5 | 5 of 5 |
| Inkling 2 Mini, ggml-org IQ4_NL | 5 of 5 | 5 of 5 | 5 of 5 |

- **The model still writes the call without the tool's name in front.** The raw tails are unchanged, for example `<|message_model|><|content_invoke_tool_json|>{"name":"read_file","args":{"path":"notes/todo.md"}}<|end_message|>`.
- **What changed is the server:** the merged parser accepts that form, as #32833's description says.

## Decision, by the rule declared before the run

- **Nothing changes in the stack.**
- **HD3 and HD1 pass**, so the speed findings of Addendum C hold on a release, not only on a draft:
  - the Unsloth file's expert layers ran on the CPU on `b12631` and `42fd370c`;
  - on `b12676` they run on the GPU at the ggml-org file's speed.
  A dated note goes on Addendum C's result.
- **For Andrei:** a proposed rewording of the blog post's update line ("the pull request is not merged"). The post is not touched here.
- **HD5 passes**, so Andrei gets the table above for issue #32802. llama.cpp does not accept AI-written reports: the comment, or the closing, is his, in his own words.
- **"Parse the calls yourself until the server does"** can now say that from release `b12676` the server does. That wording is his.
- **HD4 fails again.** "Several watchers on one mini are a queue" stands, now on a release.

## Limits

- One machine, one run per cell. Four watchers is one pass a level.
- The draft head was not re-run. Its figures are Addendum C's of the day before.
- The tool-call check is five made-up prompts. It shows that the server returns the call, not how well the model picks tools. Addendum B parsed the raw completions itself and is not affected.
- The release is newer than the merge by 4 and 26 commits. A change in those commits is part of what was measured.
