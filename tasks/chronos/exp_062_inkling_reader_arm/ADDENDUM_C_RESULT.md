# exp_062 Addendum C result: the slower Inkling file ran its expert layers on the CPU; a draft kernel makes it as fast as the other file; four watchers are still a queue

**Run:** `runs_c/20261123T091929Z`, 2026-11-23 09:19Z to 10:28Z (69 minutes), on the Mac mini (M4 Pro, 64 GB), macOS 27.2.1. One start, no stop. Registration, kit note and smoke: [ADDENDUM_C.md](ADDENDUM_C.md). The raw files were committed before anything was scored (`f1cded0`).

**Everything here is provisional.** The faster build is the head of a draft pull request, llama.cpp #32825 at commit `7b43c773`, unmerged on 2026-11-23.

> **Note, 2026-11-24: the result holds on a release.** #32825 was merged on 2026-11-23 (`51831d47`). On llama.cpp release `b12676` the Unsloth file generates at 25.90 tokens a second (Short workload) and prefills at 249.3, with all 96 expert products on the GPU. Its tokens are identical to the draft head's on all 8 texts. HD3, HD1 and HD1b pass, and four watchers give 1.20 times one (HD4 fails, as HC4 did). The tool-call parser fix (#32833) is in the same release, and llama-server now returns the research model's calls on 5 of 5 prompts, against 0 of 5 on `b12631`. [ADDENDUM_C_AMENDMENT_1_RESULT.md](ADDENDUM_C_AMENDMENT_1_RESULT.md).

## Checks

- The three builds' commits, both files' sizes and hashes and the kit files' hashes matched before the start.
- exp_054's audit hook: every child connected to the side port on the loopback address only (431 connections).
- The thermal level read 0 before every cell, every level of Part 2 and every timed request. No cell is flagged.
- All 72 timed requests evaluated the whole prompt, took nothing from a cache and generated the full count.
- Inside every cell the six repeats of a workload produced the same tokens.

## Verdicts

| | Bar | Result | Verdict |
|---|---|---|---|
| **HC3** the head build computes the same model (read first) | first token equal on at least 7 of 8 texts, and one watcher right on at least 11 of 12 jobs | 8 of 8, and 12 of 12 | pass |
| **HC1** the fallback is the gap | file U generates at 22.8 tokens a second or more on the head, Short workload | 25.84 (15.86 on `b12631`, 15.88 on the parent) | pass |
| **HC1b** the gap is closed | file U at 90 % or more of file N's generation speed on the head, both workloads | 97.0 % and 97.4 % | pass |
| **HC2** the control does not move | file N changes by less than 10 % between `b12631` and the head, four medians | −0.5 % to +3.5 % | pass |
| **HC4** the queue | four watchers do more than 1.4 times the jobs a minute of one, file U on the head | 1.34 | **fail** |

**Decision, by the registered rule:** nothing changes in the stack. The cause is measured, so a dated note goes on Addendum A's result and in this folder's README. The blog post is not touched; a rewording of one line is proposed to Andrei below.

## Where the tensors ran, observed

The scheduler's own debug output, one short load each:

| File and build | Pieces the graph is cut into | Expert products (`MUL_MAT_ID`) on the CPU | on the GPU |
|---|---|---|---|
| U on `b12631` | 90 (45 CPU, 45 GPU) | 88 | 8 |
| U on the head | 2 | 0 | 96 |
| N on `b12631` | 2 | 0 | 96 |
| U on `42fd370c`, the main run's and Addendum A's build (added after the run, 10:34Z) | 90 (45 CPU, 45 GPU) | 88 | 8 |

- 96 is 48 layer applications (12 plain layers and 12 looped layers three times) with two expert tensors each. The 88 are the 20 layers whose expert weights are NVFP4. The 8 that stayed on the GPU are the last four layers, which this file holds as Q5_K and Q8_0.
- All 977 plain `MUL_MAT` nodes are on the GPU in all three.
- So on `b12631` the graph for one token is cut into 90 pieces that alternate between the CPU and the GPU. The registration did not include the main run's own build, `42fd370c`; it was loaded once after the run and shows the same split (`placement-42fd370c-added.json`).

## Part 1: speed

Median of six requests, tokens a second. Long is a 2,048-token prompt and 256 generated; Short is 128 and 512.

| Cell | Build | File | Long prefill | Long generation | Short generation |
|---|---|---|---|---|---|
| U0 | b12631 | U | 70.5 | 15.29 | 15.86 |
| U1 | parent | U | 70.2 | 15.38 | 15.88 |
| **U2** | **head** | **U** | **248.5** | **25.11** | **25.84** |
| N0 | b12631 | N | 254.5 | 25.92 | 26.75 |
| N2 | head | N | 254.3 | 25.79 | 26.65 |
| U0r | b12631 | U, repeated last | 70.6 | 15.29 | 15.87 |

- **Reading a prompt got 3.5 times faster and writing 1.6 times.** The fallback cost more on the way in than on the way out.
- **The draft's one commit does all of it.** The parent, with the nine commits under the draft, is where `b12631` is, in speed and in text.
- **The first cell repeated at the end agrees with itself** to within 0.1 % on every median. Inside a cell the six generation figures lie within 5 % of each other, and the slow cells' prefill figures within 10 %.
- File U on the head is 2 to 3 % slower than file N on the same build, in prefill and in generation.

## Part 2: watchers at once

Addendum A's 12 jobs, with 1, 2 and 4 watchers on one server with four slots. Every job-run was right in all nine levels (108 of 108).

| Run | 1 watcher | 2 | 4 | Four against one, jobs a minute | Four against one, tokens a second | Median job, 1 and 4 watchers | Server memory, 1 and 4 |
|---|---|---|---|---|---|---|---|
| C-U0: U on `b12631` | 2.83 jobs a minute | 3.22 | 4.52 | 1.60 | 1.38 | 19.6 s and 49.1 s | 20.2 and 25.8 GB |
| C-N0: N on `b12631` | 5.44 | 6.83 | 7.24 | 1.33 | 1.39 | 11.4 s and 27.9 s | 21.0 and 26.3 GB |
| **C-U2: U on the head** | **5.71** | **5.81** | **7.64** | **1.34** | **1.51** | 11.4 s and 27.8 s | 19.8 and 23.6 GB |

- **One watcher does twice the work on the head:** 5.71 jobs a minute against 2.83, and the median job goes from 19.6 s to 11.4 s.
- **Four watchers do not do more than 1.4 times the work of one where the experts run on the GPU:** 1.33 with file N and 1.34 with file U on the head. HC4 fails.
- **The baseline on `b12631` came out at 1.60, above Addendum A's 1.4** (measured on `42fd370c`: 2.94, 3.3 and 4.06). At four watchers this run wrote 2,658 tokens where Addendum A's wrote 2,984. Jobs a minute over 12 jobs moves with how much the models happen to write, so the tokens-a-second column was added after the run: 1.38, 1.39 and 1.51. It is descriptive and is not the registered measure.
- **With one watcher the head made the same tool calls in the same order as `b12631` on all 12 jobs.**
- Memory with four slots is about 2 GB lower on the head.

## Part 3: the same text

Eight openings, 256 tokens each, temperature 0. "Gap" is the difference in the log-probability of the first cell's most likely first token.

| Compared | First token equal | Identical to the end | Median tokens agreed | Gap, median | Gap, largest |
|---|---|---|---|---|---|
| U0 against U0r: the noise floor | 8 of 8 | 8 | 256 | 0 | 0 |
| U0 against U1: the parent | 8 of 8 | 8 | 256 | 0 | 0 |
| N0 against N2: the control | 8 of 8 | 8 | 256 | 0 | 0 |
| **U0 against U2: file U, CPU experts against GPU experts** | 8 of 8 | 1 | 16.5 | 0.015 | 0.096 |
| U0 against N0: the two files on one build (added after the run) | 8 of 8 | 1 | 13 | 0.016 | 0.161 |
| U2 against N2: the two files on the head | 8 of 8 | 1 | 12 | 0.028 | 0.136 |

- **The draft changes nothing it should not.** File N's text is identical across the builds, and so is file U's on the parent.
- **On file U the new GPU path and the old CPU path part ways after a median of 16.5 tokens.** A difference was expected: the arithmetic is ordered differently. Its size is that of the difference between the two 4-bit files on one build (0.015 against 0.016 at the median). Whether that is rounding or a difference in precision between the two paths was not examined, and nothing here says which path is nearer the full-precision model. There is also no baseline here for an ordinary CPU-against-GPU difference on a tensor type both paths support, so the size cannot be called large or small.
- The draft's one commenter reported "the same PPL/KLD as the CPU at every batch size" for a dense model on an M5 Pro. This is a mixture-of-experts model on an M4 Pro, and eight prompts.

## What this corrects in the record

- **[ADDENDUM_B.md](ADDENDUM_B.md)** says of the two files: "The files differ in who quantised them and how." The measured reason for the speed is narrower: Unsloth's file keeps the expert weights of 20 layers in NVFP4, and llama.cpp's Metal backend had no kernel for that type, so those products ran on the CPU.
- **Every timing of the main run and of Addendum A is of that configuration.** The counts and verdicts do not depend on it: the reader line stopped on accuracy.
- **[ADDENDUM_A_RESULT.md](ADDENDUM_A_RESULT.md): "gemma4:26b does it in half the time."** On the 12 jobs of Part 2, Addendum A's two arms took 250 s (Inkling) and 96 s (gemma4:26b, through Ollama). File U on the head took 126 s. gemma is still the faster on these jobs, by about a quarter of the time, not by more than half. gemma was not run again today.
- **Addendum A's chatbot probe (118.6 s on a 2,543-token prompt) was not repeated.** At 70 tokens a second the prompt alone took about 36 s; at 250 it would take about 10. That is arithmetic from Part 1, not a measurement.
- **"Several watchers on one mini are a queue" is not overturned.** With the experts on the GPU, four watchers gave 1.34 times the jobs a minute of one and 1.51 times the tokens a second, on llama.cpp with four slots, one pass a level. One pass is coarse: the same file with CPU experts gave 1.38 in Addendum A and 1.60 here.

## Limits

- **A draft, one commit of it, on one chip.** The pull request may change before it merges.
- **Raw completion with the end-of-stream token ignored** for Parts 1 and 3. Part 2 is the real work, and it is 12 easy jobs, one pass a level.
- **HC1b compares two 4-bit recipes on different kernels.** That they land within 3 % of each other says nothing about which answers better. Addendum B measured answers, with file N only.
- **Part 3 is eight prompts.** It can show that two paths differ, not which is right.
- **Three things were added after the run** and are marked: tokens a second in Part 2, U0 against N0 in Part 3, and the placement load on `42fd370c`.
- **What differs from the registration** is listed in the kit note: Part 2's loop is the kit's own around Addendum A's job runner, no model-written text is kept, and the scheduler's raw output stays outside the repository.

## SO WHAT

- **exp_062's slow Inkling was a file meeting a missing kernel, not the model and not the machine.** With the kernel, the same file reads a prompt 3.5 times as fast and writes 1.6 times as fast.
- **Until llama.cpp #32825 is merged and released, check the tensor types of a GGUF before timing it on a Mac.** A file labelled Q4_K_M can hold NVFP4. On this mini that cost 40 % of the writing speed and 70 % of the reading speed.
- **A slower-than-expected local model is a reason to look at where its graph runs before concluding anything about it.** `GGML_SCHED_DEBUG=2` showed the split in one short load.
- **Four watchers on one mini still gave 1.34 times the jobs of one (1.51 by tokens), under the 1.4 bar,** on this runtime, one pass a level. Moving the experts to the GPU doubled what one watcher does and did not make four watchers scale better.
- **The faster path does not compute the same numbers as the slower one.** Tool calls matched on 12 of 12 jobs; free text diverged within about 16 tokens, with no baseline to say whether that is ordinary. Anyone who pins a reference output for this model has to pin the build too.
- **Nothing changes in the stack.** No seat depends on this model's speed.

**Open, Andrei's:**

- the push of these commits;
- the blog post's line "Check which file you pulled. On this machine ggml-org's 4-bit file of Inkling 2 Mini generated at 26.6 tokens a second and another quantiser's at about 15." A possible replacement: "Check what is inside the file you pulled. One 4-bit file here kept the expert weights in NVFP4, which llama.cpp could not run on the Mac's GPU at the time: 15 tokens a second against 26. On a draft build with the missing kernel the same file does 25.";
- whether to post the M4 Pro figures on the pull request, in his own words. The figures are the tables above;
- whether Addendum A's chatbot probe is worth repeating.
