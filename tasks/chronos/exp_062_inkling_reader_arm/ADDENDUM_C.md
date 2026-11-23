# exp_062 Addendum C: did the slower Inkling file run its expert layers on the CPU?

**Registered 2026-11-23, before any kit and before any model is loaded on the two new builds.** Andrei approved the four bars on 2026-11-23 as they were proposed that morning ("addendum C - go") and asked for the run to be started for him while he was away. What this file adds to or changes in those four bars is listed under *Changed since the go*. Nothing here changes the stack, and no exp_055 message is sent to any model. The reader verdict of the main run stands.

## Why

Every timing of the main run and of Addendum A was taken with Unsloth's `UD-Q4_K_M` file, which generated at about 15.5 tokens a second on the mini. Addendum B switched to ggml-org's `IQ4_NL` file, which generated at 22.8 and 27.3 tokens a second in two probes on the same build. The record gives no cause: "The files differ in who quantised them and how" ([ADDENDUM_B.md](ADDENDUM_B.md)).

A cause was read on 2026-11-23, not measured:

- **The Unsloth file keeps 40 tensors in NVFP4**, 11.47 GB and 69 % of the file. By name they are `blk.N.ffn_down_exps.weight` and `blk.N.ffn_gate_up_exps.weight` for 20 of the 24 layers: the expert weights. The ggml-org file holds the same 40 as IQ4_NL and has no NVFP4 tensor. Read with the GGUF reader shipped in llama.cpp `b12631`.
- **llama.cpp's Metal backend does not multiply NVFP4 weights at the two builds used so far.** In `ggml/src/ggml-metal/ggml-metal-device.m` at `42fd370c` and at `b12631`, `ggml_metal_supports_mul_mat_op` returns false for an NVFP4 weight, and that function decides both `MUL_MAT` and `MUL_MAT_ID`.
- **A draft pull request adds the kernel:** llama.cpp #32825, "metal : add NVFP4 support", opened 2026-11-22. Its description: "The Metal backend does not support `GGML_TYPE_NVFP4`, so NVFP4 GGUF checkpoints fall back to the CPU." It was a draft, open and unmerged, when read again at 09:09 UTC on 2026-11-23, with its head still at `7b43c773`. It states no speed figure.

So the question: **on the mini, does the Unsloth file get as fast as the ggml-org file once its expert weights can run on the GPU, and does it still compute the same model?**

Three published lines rest on the slower file's speed: Addendum A's "half the time", the chatbot probe (119 s against 32 s), and "several watchers on one mini are a queue" (four slots, 1.4 times the work).

## Runtime, pinned

Three builds of llama.cpp, each in its own folder under `~/models`, each built the same way (`cmake -B build -DCMAKE_BUILD_TYPE=Release`, target `llama-server`):

| Name | Commit | What it is |
|---|---|---|
| **b12631** | `e300919cc704ddde3a8a8346a8714f57eeeee71b` | Addendum B's build, already on the mini. It is the draft's merge base. |
| **parent** | `1834cc254c77e536930d9bebaf7f85d701a034b6` | the commit under the draft's own commit: `b12631` plus nine commits of Metal work on BF16 inputs |
| **head** | `7b43c7730485cc77429485d8232c9643e8aab1f2` | the draft's one commit on top of the parent |

The parent is there because the Unsloth file also holds 32 small BF16 tensors, so the nine commits under the draft could move it too.

Two files of Inkling 2 Mini, both already on the mini:

| Name | File | sha256 |
|---|---|---|
| **U** | Unsloth `Inkling-2-Mini-UD-Q4_K_M.gguf`, 16,610,544,608 bytes | `78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae` |
| **N** | ggml-org `Inkling-2-Mini-IQ4_NL.gguf`, 17,627,940,768 bytes | `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` |

The mini: M4 Pro, 64 GB, macOS 27.2.1. One side server at a time on 127.0.0.1:18450. The production Ollama is not called; `ollama ps` is read before and after. Two small llama.cpp servers that were already running on the mini (ports 18442 and 18448) are left alone and listed in the run's record.

## Cells

One cell is one build with one file, in one server session with one slot of 8,192.

| Cell | Build | File |
|---|---|---|
| U0 | b12631 | U |
| N0 | b12631 | N |
| U1 | parent | U |
| U2 | head | U |
| N2 | head | N |
| U0r | b12631 | U, again at the end |

They run in that order. U0r is the noise floor and the check for drift.

## Part 1: speed

Raw completion (`/completion`), temperature 0, seed 62, `cache_prompt` off, `ignore_eos` on so that the count generated is fixed. Two workloads, six requests each in every cell, the median reported and the six values kept:

- **Long:** a made-up prose text cut to 2,048 tokens, 256 tokens generated. This is the shape Ollama's pull request #19843 used for its own table. It gives the prefill speed.
- **Short:** the first 128 tokens of the same text, 512 tokens generated. The probes behind 22.8 and 27.3 were short prompts.

Every request is checked: the whole prompt was evaluated (no token from a cache) and the full count was generated. A request that fails the check stops the run.

## Part 2: watchers at once

Addendum A's Part 2, with its driver and its 12 jobs unchanged (`kit/watch62a.py`, `kit/jobs62a.json`, sha256 `68b66f99…`): 1, 2 and 4 watchers at once on one server with four slots of 8,192. Three times:

| Run | Build | File |
|---|---|---|
| C-U0 | b12631 | U |
| C-N0 | b12631 | N |
| C-U2 | head | U |

Addendum A's published 1.4 was measured on `42fd370c`, so C-U0 is the baseline on the build the draft starts from. C-N0 shows what the same test gives when the expert weights already run on the GPU.

## Part 3: the same text

Eight short made-up prompts, raw completion, temperature 0, 256 tokens, token ids returned, and the five most likely first tokens with their probabilities. Run once in every cell. Compared by code:

- the first generated token, which is computed from an identical prompt in every cell;
- how many tokens agree before the first difference.

U0 against U0r is the noise floor. N0 against N2 is the control: the same file across the builds.

## Where the tensors ran, observed

One short load of U on `b12631` and on the head, and of N on `b12631`, with the scheduler's own debug output on (`GGML_SCHED_DEBUG=2`, verbose log), one eight-token request. Reported: how many pieces the graph is split into and which backend each piece is given. If the output cannot be read that way, that is stated and nothing is inferred from it.

## Hypotheses

| | Claim | Passes if |
|---|---|---|
| **HC3** | the head build computes the same model | with file U, the head's first token equals `b12631`'s on at least 7 of the 8 texts, **and** one watcher on the head is right on at least 11 of the 12 jobs |
| **HC1** | the fallback is the gap | file U generates at 22.8 tokens a second or more on the head (the Short workload, median of six) |
| **HC1b** | the gap is closed | on both workloads, file U on the head generates at 90 % or more of file N's speed on the head |
| **HC2** | the control does not move | file N's medians, prefill and generation, on both workloads, each change by less than 10 % between `b12631` and the head |
| **HC4** | the queue | with file U on the head, four watchers at once do more than 1.4 times the jobs a minute of one |

- **HC3 is read first.** If it fails, HC1, HC1b and HC4 are recorded as not readable: a kernel that computes something else can be fast.
- **HC1 is the bar as proposed.** HC1b is the direct answer to "is that the whole gap": the two files on one build and one workload.
- **The parent's figures are reported beside HC1,** so that the gain can be split between the nine commits under the draft and the draft itself.
- The agreed-prefix lengths of Part 3 are descriptive.

## Decision rule, declared now

- **Nothing changes in the stack, whatever the result.** No seat depends on this model's speed.
- **If HC3 and HC1 pass:** the cause is measured. A dated note goes on Addendum A's result and in this folder's README, saying that the main run's and Addendum A's timings are of a file whose expert layers ran on the CPU. The blog post's line "Check which file you pulled" gets a proposed rewording for Andrei; the post is not touched.
- **If HC3 passes and HC1 fails:** the record says the draft did not close the gap on this machine, with the figures.
- **If HC3 fails:** no speed from the head build is quoted anywhere. The figures go to Andrei as a table, for the pull request if he wants to post them in his own words.
- **Everything is provisional:** the pull request is a draft. A result holds for commit `7b43c773` only.

## Stops

- **A pinned input that does not match** (a build's commit, a file's size or hash, the jobs file's hash, the text file's hash) stops the run before any model is loaded.
- **The head build fails to load file U, or errors on it:** recorded; HC1, HC1b and HC4 fail as "does not run"; C-U2 is not run.
- **Thermal level:** read before every cell and every level of Part 2. If it is not 0 the driver waits up to ten minutes. If it is still not 0 the cell runs and is flagged, and flagged figures are reported apart.
- **A power cut or any other stop:** each cell writes its file when it is complete, and a resume skips complete cells and complete Part 2 runs. The side server is stopped on any exit.

## Time

From figures on record: about 3 to 5 minutes a cell for Parts 1 and 3, about 12 minutes for each run of Part 2, half a minute to hash each file. About 75 to 90 minutes in all. A smoke test goes first, on made-up inputs that are not in the test: the five smoke jobs of Addendum A (`kit/smoke62a.json`) on the head build with file U at one, two and four watchers, and one request of each workload on a different text.

## Limits

- **One machine, one chip.** The one comment on the draft measured an M5 Pro and found the choice of kernel for two-token batches depends on the chip.
- **A draft.** Nine commits sit under it, and it may change before it merges.
- **Raw completion with the end-of-stream token ignored** measures the speed of the arithmetic, not of a useful answer. Part 2 is the useful work.
- **Part 2 is 12 easy jobs,** the ones both models got right in Addendum A. It measures throughput, not quality.
- **Two different 4-bit recipes.** HC1b compares NVFP4 on its new kernel with IQ4_NL on an older one. It can fail with the fallback still being the cause.
- **Greedy text differs between a CPU and a GPU path** sooner or later, because the arithmetic is ordered differently. That is why HC3 reads the first token and the watcher jobs, and the prefix lengths are descriptive.
- **Simulated tools and a made-up world,** as in Addendum A.

## Changed since the go

The four bars Andrei approved were: HC1 (22.8 or more on the head, against about 15.5), HC2 (the ggml-org file moves by less than 10 %), HC3 (same text, described), HC4 (Addendum A's Part 2 again, more than 1.4). Added or changed here, before any run:

- **HC1's workload is named:** the Short one, because 22.8 came from short-prompt probes. The 2,048-token shape is reported beside it.
- **HC1b is new:** the two files against each other on the head.
- **HC3 has a bar and is read first,** with a noise floor (U0r) and a control (N2).
- **The scheduler's debug output is read,** to see where the tensors ran instead of inferring it from speed.
- **C-N0 is new:** Part 2 with the ggml-org file on `b12631`.
- **U0 is run twice,** first and last.
- **No separate tool-call or prose prompt in Part 1:** both workloads are prose continuations, and the tool-call turns are Part 2's jobs.
- **A thermal rule and the resume rule** are written down.

## Kit (2026-11-23, after the registration above was committed as `e24e4d0`, before the run)

**No request of the test has been sent to any model.** The smoke used another text, two other openings and Addendum A's five smoke jobs.

- `kit/build62c.py` writes the made-up texts, with no model call: `kit/text62c.txt` (seed 62, sha256 `3ffb77ad…c769df`; 8,959 tokens, of which the driver uses the first 2,048 and the first 128), `kit/prompts62c.json` (the eight openings of Part 3 and two for the smoke, sha256 `692326ba…e7f98c`) and `kit/smoke62c.txt` (seed 900). An invented harbour town, no person's name, no address.
- `kit/speed62c.py` is the driver. It checks the three builds' commits, both files' sizes and hashes and the kit files' hashes, then runs the six cells, the three placement loads and the three runs of Part 2, one side server at a time. Each child runs under exp_054's audit hook. Every timed request is checked (whole prompt evaluated, nothing from a cache, full count generated). A step's file is written when the step is complete, and a resume skips complete steps.
- `kit/score62c.py` computes HC1 to HC4 as registered, HC3 first. A missing cell makes a bar "not run", never a pass. Run on the smoke folder, where only one cell exists, it reports every bar as not run.

### What differs from the registration, and why

- **Part 2's loop is this kit's, around Addendum A's job runner.** The registration says "its driver … unchanged". `watch62a.run_job`, its tools, its system prompt and its jobs are used unchanged. The loop over 1, 2 and 4 watchers is rewritten so that the thermal rule applies before each level and each level is kept when complete; Addendum A's own loop deletes its files on entry and has no wait.
- **No text a model wrote is kept in the run folder,** so no private folder is needed. Parts 1 and 3 keep token ids. Part 2 keeps tool names, lookup arguments, counts and whether the job was right; the final sentence and the text of alerts and questions are replaced by their lengths.
- **The scheduler's raw output stays outside the repository** (`~/models/exp062c/`, about 33 MB a load). The run folder gets the counts: the pieces the last graph was cut into, by backend, and its `MUL_MAT` and `MUL_MAT_ID` nodes by backend.
- **Part 3 ignores the end-of-stream token too,** so that every text is 256 tokens in every cell.

### Smoke result (`dev_c/20261123T091411Z/`, 2026-11-23 09:14 to 09:18 UTC, smoke inputs only)

The head build with file U, one request of each workload, two openings, the five smoke jobs at 1, 2 and 4 watchers, and two placement loads.

- **It loads and runs.** Long workload: 240.9 tokens a second prefill, 25.2 generation. Short: 26.1 generation. All checks on the timed requests passed.
- **All five smoke jobs right at 1, 2 and 4 watchers** (15 of 15 job-runs). Five jobs cannot fill four slots, so the smoke says nothing about HC4.
- **The scheduler's output can be read.** File U on `b12631`: the last graph is cut into 90 pieces, 45 for the CPU and 45 for the GPU, and 88 of its 96 `MUL_MAT_ID` nodes are given to the CPU. File U on the head: 2 pieces, and all 96 on the GPU. All 977 `MUL_MAT` nodes are on the GPU in both.
- The audit logs show connections to the side port on the loopback address only. The thermal level read 0 throughout.

**Disclosed:** the smoke's speeds were seen before the run. The bars were committed before the smoke (`e24e4d0`), and the registered figures come from the run, on the test text.

**Time, from the smoke:** about 6 minutes a cell on the faster combinations and about 10 on the slowest, about 8 to 12 minutes for each run of Part 2. About 75 minutes in all.

## To run (on the mini)

Andrei asked on 2026-11-23 for the run to be started for him in tmux while he was away. `ollama ps` was empty and the thermal level read 0.

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_062_inkling_reader_arm && TS=$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p runs_c/$TS && tmux new-session -d -s exp062c -c "$PWD" "caffeinate -is /opt/homebrew/bin/python3.14 kit/speed62c.py --all --run runs_c/$TS 2>&1 | tee -a runs_c/$TS/run.log; echo \"[exp062c] exit \${pipestatus[1]} at \$(date)\" | tee -a runs_c/$TS/run.log; exec zsh" && echo "run folder: runs_c/$TS   attach: tmux attach -t exp062c"
```

A stopped run resumes with the same command and `TS=<the folder name it printed>`. To score: `python3.14 -I kit/score62c.py runs_c/<ts>`.
