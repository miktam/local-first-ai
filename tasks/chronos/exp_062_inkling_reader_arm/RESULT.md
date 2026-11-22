# exp_062 result: Inkling 2 Mini flags almost every message it answers; gemma4:26b with the retry keeps the reader's seat

**Run:** `runs/20261121T131040Z`, 2026-11-21 13:10Z to 2026-11-22 01:39Z (12 h 28 min), on the Mac mini (M4 Pro, 64 GB), macOS 27.2.1. Inkling 2 Mini UD-Q4_K_M (sha256 `78fcedcf…e5834ae`) on a llama.cpp build of pull request #32704 at commit `42fd370c`, one slot, context 8,192, reasoning on (the model's default). One driver invocation, 220 calls, no failed call. The thermal pressure level read 0 before every call.

## Controls

| | Claim | Result | Verdict |
|---|---|---|---|
| **C1** | the inputs are exp_060's | the five frozen-input hashes match; the prompt was asserted byte for byte against exp_055's builder on every message; 220 answer rows, no error row | PASS |
| **C2** | the runner talks to the side server only | exp_054's audit hook on the runner: 220 connections, all to `127.0.0.1:18450`; 220 subprocesses, all the thermal reading. The server's own network behaviour is not measured, as registered | PASS |
| **C3** | the server ran as pinned | the model file's hash and the server's commit matched before the start; the server log shows one slot with a context of 8,192 | PASS |
| **C4** | the timing is the mini's own | thermal level 0 before all 220 calls. The `ollama ps` half is **not evidenced**: the listing shows gemma4:26b resident before and after and cannot show whether it generated in between; the bot's two side servers were not watched | PASS on the thermal half only |

## Verdicts

| | Bar | Result | Verdict |
|---|---|---|---|
| **H1** it answers | a valid band on ≥ 218 of 220 | **194** | **fail** |
| **H2** it reads at least as well as the seat's holder | balanced accuracy ≥ 0.914, and on the messages where exactly one of I and F is right, I right at least as often | **0.645**; only I right 11, only F right 70 | **fail** |
| **H3** it does not buy catches with false alarms | genuine flagged ≤ 11 of 110, and attribution ≥ 85 % | **74 of 110**; attribution 91.3 % | **fail** |

**Decision, by the registered rule: the line stops.** Balanced accuracy is below 0.882, the figure of gemma4:26b alone on the same call shape, so no further Inkling arm is registered for this seat. gemma4:26b with exp_060's fix F keeps it.

## The numbers

| Arm | Answered | Lures caught | Genuine flagged | Balanced accuracy | Item named | Median | p95 |
|---|---|---|---|---|---|---|---|
| **I, Inkling 2 Mini, reasoning on** | 194 | 106/110 (96 %; CI 91–99) | **74/110 (67 %; CI 58–75)** | **0.645** | 91.3 % | 152.3 s | 488.5 s |
| G, gemma4:26b alone, exp_055 | 198 | 92/110 (84 %) | 8/110 (7 %) | 0.882 | 85 % | 33.6 s | |
| F, gemma4:26b with the retry, exp_060 | 220 | 100/110 (91 %) | 9/110 (8 %) | 0.914 | 95.2 % | 36.9 s | 162.3 s |
| O, Opus, exp_055 | 220 | 105/110 (95 %) | 6/110 (5 %) | 0.950 | 100 % | | |

Intervals are Wilson 95 %. An empty answer counts as not flagged, as registered. The gap to the frontier model is 0.305 (F's is 0.036).

## It flags what it answers

**Of the 194 messages it answered, it flagged 180.** The descriptive figures in this and the following sections are in `describe.json`, written after the run by `kit/describe62.py`.

| Truth | `benign` | `suspicious` | `likely-scam` | no answer |
|---|---|---|---|---|
| 110 genuine, Inkling | 14 | 66 | 8 | 22 |
| 110 genuine, F | 101 | 8 | 1 | 0 |
| 110 lures, Inkling | 0 | 17 | 89 | 4 |
| 110 lures, F | 10 | 34 | 66 | 0 |

- **Every lure it answered, it flagged: 106 of 106.** Its four misses (X048, X095, X147, X193) are all answers that never came.
- **Of the 88 genuine messages it answered, it flagged 74** and called 14 benign.
- **The 0.645 is flattered.** 36 genuine messages count as not flagged, and 22 of them are empty answers. Had those 22 been answered at the rate of the other 88, the figure would be lower.
- **The catch rate is not a gain over F.** A reader that flags 93 % of what it answers catches lures by default. On the pairs where one arm is right and the other wrong, I is right on 10 lures and 1 genuine message, and F on 4 lures and 66 genuine messages. The one genuine message (X008) is an empty answer of I's that F flagged.

**By domain age** (exp_055's labels):

| | Inkling flagged | F flagged |
|---|---|---|
| genuine, established domain | 44/72 | 4/72 |
| genuine, young domain | 19/20 | 5/20 |
| genuine, no domain (SMS, WhatsApp) | 11/18 | 0/18 |
| lure, new domain | 71/72 | 70/72 |
| lure, old domain | 23/24 | 18/24 |
| lure, no domain | 12/14 | 12/14 |

The false alarms are not a young-domain effect: it flags six in ten genuine messages from domains registered years ago, and six in ten with no domain at all.

**What its one-sentence reasons say**, counted by phrase over the 74 flagged genuine messages (regex counts, chosen after reading eight of them; they describe the wording, not the cause):

- 65 restate the prompt's own definition of `suspicious` ("verify through a separate, known channel");
- 39 say the sender or a detail is absent from her public footprint;
- 28 say something cannot be verified;
- 19 cite a domain's date; 15 call the sender unknown or the message unsolicited.

A typical one, on a WhatsApp message from a club mate who moved her sail (X144, F: `benign`): "the specific details (mainsail, bag label 1842) are not verifiable from her footprint, so she should confirm through a separate known channel before acting". The model treats "I cannot confirm this from the footprint" as grounds to flag. gemma4:26b reads the same prompt and calls 101 of the 110 benign. Eight of F's nine false alarms are also Inkling's.

## The empty answers

- **26 answers never came:** 22 genuine, 4 lures. Each ended with `finish_reason` `length` at prompt + generation = 8,192 tokens (7,152 to 7,449 generated), an empty `content`, and a median of 488 s.
- **They are not gemma's.** gemma4:26b alone left 22 empty on this corpus; one message (X141) is on both lists. F is right on 25 of Inkling's 26.
- **They cost 3.5 of the run's 12.5 hours.**
- **They are not what sinks the arm.** A retry in glue could at best answer the 26; the 74 false alarms would remain.

## Time

- **Median 152 s a message** (F: 36.9 s), 95th percentile 488.5 s, 12.48 hours in all. Over the 194 answered: median 139.9 s, 95th percentile 349.5 s, longest 420.2 s.
- **15.3 tokens a second** (median), 667,481 tokens generated; the median prompt is 944 tokens.
- **Genuine messages take twice the reasoning of lures:** a median of 3,460 generated tokens against 1,698.5.
- **97 % of what it wrote is reasoning,** by characters (2.31M of reasoning against 67K of answers). The registration promised this share by tokens; the server reports one token count for both, so it is given by characters.

## Deviations from the registration

- **The partial counts were looked at during the run** (Amendment 2, at 133 of 220, at Andrei's request). The registration's limit, that per-message results are not opened before the run is complete, is therefore not met. The amendment was committed before the look, the run went to its end, and the arm has no setting the look could have tuned. The counts seen then (71/73 lures, 41/60 genuine flagged) agree with the final ones.
- **The order was by message id,** the same order as F's and G's runs, not "the manifest's order" as the registration says; the manifest's keys are not sorted. No verdict depends on the order.
- **The run took 12.5 hours, not the seven expected.** The dummy probe's 118 s was below the median of the answered messages (140 s), and the estimate did not allow for the 26 answers that ran to the context.
- **The pull request had already merged.** The registration calls #32704 open and unmerged, and says its head had moved. GitHub's API, read on 2026-11-22, shows it merged on 2026-11-21 at 11:10 UTC, 83 minutes before the registration was committed, with a single commit, `42fd370c`, the one this build uses. The registration's statement was out of date when written. The latest tagged release on that date (`v1.0.0`, 2026-11-13) predates the merge, so no tagged release carries the fix yet.
- **C4's `ollama ps` half** is recorded but shows nothing about generation (above). Time is descriptive here, so no verdict rests on it.

## Held back

The reasoning text is kept out of this repository, as exp_060 kept the thinking text: `I/reasoning.jsonl`, sha256 `f0eab5fc73e1e34c426f9fbad1cb7d942dd34597245b11ef4ac62362dcbfdf06`. Neither `score62.py` nor `describe62.py` reads it; the reasoning figures above are character counts recorded in `results.jsonl`.

## Limits, as registered and as met

- **The reader's seat only, with gemma's prompt.** exp_055's prompt was written for gemma4:26b, and F was chosen on this corpus. A prompt that tells the model what an unverifiable detail should count for was not tried, and by the stop rule is not registered. This says nothing about Inkling 2 Mini on translation, on code, or on the security benchmark that prompted the test.
- **Reasoning on only.** A zero reasoning budget on the server is a different reader and was not an arm.
- **One quantisation, one build, one machine, one pass at temperature 0.** The build is the merged fix's own commit, not a tagged release.
- **Synthetic messages from one family of writers,** used for the third time. Andrei's real incidents remain the regression floor for anything that reaches the guard.
- **The reason counts are post hoc** and count phrases, not causes.
- **The frontier figure is exp_055's,** from October, and is not re-measured.

## Open

**Whether the repeat on a tagged release is still owed.** The registration says two things: every figure is provisional until the run is repeated on a tagged llama.cpp release that carries #32704, and the line stops below 0.882. The stop rule fired, and the figures stay provisional: this build is the commit that was merged, and no tagged release carried it on 2026-11-22. The repeat would take about 12.5 hours. It is Andrei's decision; it is not scheduled.

## SO WHAT

- **Nothing changes in the guard.** gemma4:26b with the think-then-retry call keeps the local reader's seat, at 0.914 against Inkling's 0.645 on the same 220 messages.
- **Inkling 2 Mini, at this quantisation and with this prompt, is not usable as a message reader.** It flags two genuine messages in three, about one in eight answers never comes, and it takes four times as long. A guard that flags most of the real mail is switched off by the person it guards.
- **Its caught-lure figure should not be quoted alone.** 106 of 110 is the catch rate of a reader that says "verify first" to almost everything.
- **The leaderboard told us nothing about this seat,** as the registration expected. A model that a public leaderboard puts level with a frontier model at finding bugs in code read messages worse than the one already on the disk. A benchmark for one security task does not transfer to another; each seat is measured on its own task.
- **A reading, not a measurement: the prompt leaves room for this.** The band `suspicious` is defined as "she should verify it through a separate, known channel", and a cautious reader can attach that to any message it cannot confirm. gemma does not; another model may. Any future challenger for this seat runs on this corpus before anything else, and its genuine-message count is read first.
