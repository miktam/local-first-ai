# exp_060 result: think, and retry without thinking when the answer is empty; a larger context only moves the empties

**Run:** `runs/20261101T152955Z`, 2026-11-01 15:29Z to 2026-11-02 02:22Z, on the Mac mini (M4 Pro, 64 GB), macOS 27.2 (Amendment 1), Ollama 0.40.2, gemma4:26b `777b927e3b6b`, production settings, cloud features off. Arm T took 10 minutes, F 3 h 13 min, C 7 h 29 min. The thermal pressure level read 0 on all 653 one-minute samples, so H4 is separable for every arm. `ollama ps` showed only gemma4:26b before, during and after each arm, and the daemon's log holds 682 `/api/chat` requests in the window, all from loopback. That is exactly the run's own: 220 + 220 + 21 retries + 220 + the closing restore call. Scored by `kit/score60.py` (`score.json`). The descriptive figures that `score.json` does not carry are from `kit/describe60.py` (`describe.json`), written after scoring.

## Controls

| | Claim | Result | Verdict |
|---|---|---|---|
| **C1** | the baseline reproduces | F's pass 1 gives exp_055's G band on **193 of 193** unchanged answered messages, and **20 of 20** unchanged empties are empty again, across the macOS 26.5.1 → 27.2 upgrade | PASS |
| **C2** | the wall is the cause | all 21 empty pass-1 answers end with `done_reason` `length` at prompt + generation = 8,192 tokens (7,129 to 7,369 generated) | PASS |
| **C3** | nothing leaves the machine | exp_054's audit hook: T 223, F 244 and C 225 connections, all to `127.0.0.1:11434` | PASS |
| **C4** | arm C ran at 32K | `ollama ps` during C: CONTEXT 32768, 23 GB; after the restore call `/api/ps` reads 8,192 | PASS |

## Verdicts

| Arm | H1: ≥ 218 answered | H2: agrees with G on ≥ 95 % of the 193 | H3: balanced accuracy ≥ 0.882 and attribution ≥ 85 % | H4: median ≤ 45 s, p95 ≤ 180 s, no error |
|---|---|---|---|---|
| **T** thinking off | 220, pass | 81.3 % (descriptive for T) | 0.850 / 92.3 %, **fail** | 2.8 s / 3.4 s / 0, pass |
| **F** think, then retry without | 220, pass | 100 % (C1), pass | 0.914 / 95.2 %, pass | 36.9 s / 162.3 s / 0, pass |
| **C** 32K context | 198, **fail** | 88.6 %, **fail** | 0.905 / 88.5 %, pass | 36.8 s / 884.8 s / 1 error, **fail** |

**Decision, by the registered rule: F.** It is the only arm that passes H1, H3 and H4. A1's reader call inherits it: thinking on at the 8,192 context, and on an empty answer one retry of the same prompt with `think: false`. Thinking earned its place.

## The numbers

| Arm | Answered | Lures caught | Genuine flagged | Balanced accuracy | Item named | Median | p95 |
|---|---|---|---|---|---|---|---|
| G, exp_055 (8K, thinking on) | 198 | 92/110 (84 %) | 8/110 (7 %) | 0.882 | 85 % | 33.6 s | |
| T, thinking off | 220 | 101/110 (92 %; CI 85–96) | 24/110 (22 %; CI 15–30) | 0.850 | 92.3 % | 2.8 s | 3.4 s |
| **F, think, then retry without** | 220 | 100/110 (91 %; CI 84–95) | 9/110 (8 %; CI 4–15) | **0.914** | 95.2 % | 36.9 s | 162.3 s |
| C, 32K context | 198 | 95/110 (86 %; CI 79–92) | 6/110 (5 %; CI 3–11) | 0.905 | 88.5 % | 36.8 s | 884.8 s |
| O, Opus, exp_055 | 220 | 105/110 (95 %) | 6/110 (5 %) | 0.950 | 100 % | | |

Intervals are Wilson 95 %. Empty answers count as not flagged, as registered. F narrows the gap to Opus from 0.068 (G) to **0.036**. F and C each put one genuine message at likely-scam, and T none.

## F: what the retry costs

21 messages fell back: the 20 unchanged empties and X161, whose published copy changed (X011, also changed, answered on pass 1). Each fallback spent a median of 159.5 s thinking into the wall, then 2.9 s on the retry, 162.3 s in all. The 199 answered on pass 1 took a median of 35.3 s (p95 79.7 s), so the fallbacks alone make F's p95. The retry is T's call on the same message, and it gave T's band on all 21 (the same items on 18).

On the 22, F's answers are right on 16. It caught 8 of the 11 lures and flagged 3 of the 11 genuine messages (X041, X100, X103). The three lures it missed (X071, X088, X157) are thinking-off answers; X088 and X157 were caught in arm C, where thinking finished.

## T: fast, and it cries wolf

T flags **15 of the 20 genuine messages from young domains**, where F flags 5 and C 4. It also flags 8 of the 72 from established domains, against F's 4. T and F disagree on 38 messages:

- only F is right on 19: 17 genuine messages that T flagged and 2 lures that T missed;
- only T is right on 5: 3 lures and 2 genuine messages;
- on 14 they differ only in strength (suspicious against likely-scam).

Thinking buys precision, not catch: T catches one lure more than F.

## C: a larger context moves the empty answers; it does not close them

At 32K, **22 answers were still empty, as many as G's at 8K, but not the same messages.**

- **11 of the original 22 finished, all 11 right** (6 lures caught, 5 genuine messages cleared). None of them needed the room: they finished in 1,026 to 5,304 generated tokens, inside the 8K wall.
- **The other 11 thought to the 32K wall.**
- **11 messages that G and F's pass 1 answered at 8K now failed.** Ten thought to the 32K wall (5 lures, 5 genuine). The eleventh, X189, was stopped by Ollama after 96 s with `HTTP 500: prediction aborted, token repeat limit reached`. That is the runtime's own loop guard, and it is the one call error that fails H4.

All 21 that reached the 32K wall generated 31,705 to 31,850 tokens, at a median of 885 s each.

The prompt, temperature 0 and seed 55 were unchanged. Changing only `num_ctx` changed the generated text on **115 of the 188** messages that finished in both F's pass 1 and C; 73 were identical. The smoke test's identical output on the dummy did not generalise. Of C's 22 disagreements with G on the 193:

- 11 are the new empties;
- 7 are lures re-graded between suspicious and likely-scam;
- 2 are lures now caught (X133, X207);
- 1 is a genuine message newly flagged (X204), and 1 has its false alarm cleared (X211).

H2 was registered to catch exactly this: reloading at a larger context is a different reader, not a fix. gemma4:26b took 23.3 GB at 32K against 18.7 GB at 8K.

## What the empty answers are: loops

C2 holds: the wall is where the empties end. The thinking shows why they do not end sooner: they loop. Repetition is measured as the share of the last 200 non-empty lines of the thinking that repeat an earlier line.

- **At 32K, every one of C's 21 runaways ends in a loop.** In 19 the share is 92 % or more (median 99 %), and in X116 it is 85 %. X052 writes no line breaks: its last 20,000 characters hold 3 distinct 40-character chunks out of 500.
- **At 8K the same loops are younger.** In F's pass 1 the median is still 99 %, and 14 of 21 are at 92 % or more. X020, at 45 % when the 8K wall cut it, is at 99 % by the 32K wall.
- **The commonest loop is on the attribution step.** The model checks the footprint for an item id that does not exist, and checks again. The footprint lists `A-0a9fbf` (a notes page) and five contact items numbered `C-`; the model writes `C-0a9fbf`. Its most-repeated lines read "`C-0a9fbf` is not there" or "`C-0a9fbf` is not an email", and one answer writes the id 1,745 times. A second invented id, `C-0069db50`, is a garbled copy of the booking page's `C-69db50`. An invented id appears in the thinking of 11 of F's 21 runaways, against 14 of the 199 that finished, and in 14 of C's 21, against 15 of 198. It is the commonest loop, and the loops are not all on it.
- **The budget curve** (F's pass 1, the 199 that finished) is what a plain token budget would have cut. A budget at 2,000 generated tokens would have cut 69 of them (65.3 % finish within it), 4,000 would have cut 7 (96.5 %), 6,000 would have cut 3 (98.5 %) and 7,000 none. The longest finished answer used 6,505. A budget at 7,000 saves about 4 s per runaway. One at 4,000 saves about 65 s, but sends 7 answers that would have finished to the thinking-off retry.

## Held back

The thinking text of arms F and C quotes the corpus's sender addresses. In the long loops it also writes variants of them, and of the corpus's fictional UK number, that fall outside the reserved and drama ranges. They were never checked against anything real. `score60.py` never reads the thinking. Both files are kept out of this repository, as in the aborted run and as exp_057 kept its machine log:

- `F/thinking.jsonl` sha256 `9b03795bc9a4f62869c8e6f301dab3737859b7de024a3a5b7108b09800510642`
- `C/thinking.jsonl` sha256 `ae6bccf3b0f465a73f2986bade6fbbf44a713339888243260728ac9ac9667bf5`

`T/thinking.jsonl` is published: its 220 rows are all empty, because thinking was off. The loop figures above are counts computed from the held-back files by `kit/describe60.py`, which writes no thinking text. Anyone holding the files can re-run it; without them it writes the rest of `describe.json` and marks the loop section absent.

## Limits, as registered and as met

- **The reader-alone call, not A1's.** A1's prompt carries Parry's rules as evidence and is longer, and its empty answers are a declared measure of their own.
- **One pass at temperature 0.** 22 of 220 looped at 8K and 22 of 220 at 32K, but over different sets. So the rate of about 10 % held on two configurations, while which messages loop did not.
- **One model, one build, one runtime:** gemma4:26b Q4_K_M on Ollama 0.40.2. A thinking budget or loop guard inside the runtime would be a different fix.
- **Synthetic messages from one family of writers.** Andrei's 7 real incidents remain the regression floor.
- **macOS 27.2, not 26.5.1 as registered** (Amendment 1). C1 shows the 8K answers did not move across the upgrade.
- **The loop figures are descriptive and post hoc.** The repetition measure was chosen after reading the runaways, and it measures repetition, not intent.

## SO WHAT

- **A1's reader call is F**, and Exposure Review's local reader now answers every message. Against exp_055's G: answered 198 → 220, balanced accuracy 0.882 → 0.914, gap to Opus 0.068 → 0.036, lures caught 92 → 100, false alarms 8 → 9. The price: about one message in ten takes 2.7 minutes instead of 35 s, and the run's 95th percentile is 162 s.
- **Thinking stays on.** Without it the reader is thirteen times faster and flags one genuine message in five, three in four from a young domain. For a check whose clients act on its flags, that is crying wolf.
- **Do not raise the context to fix a loop**, here or in the bot (exp_052 and exp_053 lost answers the same way). At 32K the loops run four times as many tokens (about 15 minutes each), the model takes 4.6 GB more memory, as many messages loop, and verdicts change. The production 8,192 cap stays, now for a second reason beside exp_025's slowdown.
- **The context size belongs to the reader's identity.** With the same weights, seed, temperature and prompt, a different `num_ctx` changed the generated text on 115 of 188 messages. Across the OS upgrade, at a fixed context, nothing changed (193 of 193, 20 of 20). Every comparison between runs must fix the context and record it. exp_059 (same weights, three chips) has to pin it identically on every arm, or the chip is not the only thing that changed.
- **Seeds for the next fix**, each to be registered before it is used:
  - **a loop detector on the stream.** The runaways repeat themselves long before the wall. Stopping on repetition, then retrying without thinking, could cut the fallbacks' 160 s without cutting an answer that would have finished, which a plain token budget cannot do. Untested.
  - **an attribution input that is harder to garble.** The commonest loop is a mis-copied footprint id. A1 could hand the model ids it cannot confuse across item types. Untested.
- **Thinking text is private by default.** It carries what the model copied and garbled from its inputs, including strings that may be real addresses and numbers. Campaign A records it, hashes it and holds it back, and no scorer reads it.
