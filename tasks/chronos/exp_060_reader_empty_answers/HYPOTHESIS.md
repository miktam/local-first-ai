# exp_060: which fix closes the local reader's empty answers, and what does it do to its verdicts?

**Registered 2026-10-28, before any kit or run.** Campaign A's mechanical precursor (A2 in `bodyguard/ROADMAP.md`). It runs before A1, the reader test, so that A1's numbers are not understated by a runtime failure. The corpus, the prompt and the model call are exp_055's, unchanged except for the one thing each arm changes. The run is Andrei's to start: it drives the shared production Ollama for several hours, and one arm reloads the model with a larger context.

## Why

In exp_055, gemma4:26b alone (arm G), given the fictional subject's footprint, the registry facts and the message, scored a balanced accuracy of 0.882 against the engineered check's 0.814. But 22 of its 220 answers (10 %) came back empty. Under the registered rule they count as not flagged, so G's catch rate is understated and its false-alarm rate flattered. exp_055 diagnosed the cause from timing alone: every empty answer took 148–154 s, which is 8,192 tokens at the mini's generation speed, while the answered ones took a median of 33.6 s (p90 63 s, max 136 s). The model thought until the daemon's 8,192-token context was full, and no answer token was ever generated. The same failure took the bot's answers in exp_052 and exp_053.

Campaign A inverts the pipeline around this reader. Every reader number in A1, A3 and A4 would carry the same 10 % hole, so the hole is closed first, on its own, and measured, so that A1 is a clean test of the design and not of the fix.

## What is at stake

The 22 empty answers, from exp_055's records (`runs/20261025T095614Z/G/results.jsonl` against `data/MANIFEST.json`):

| | n | of which |
|---|---|---|
| lures | 11 | 6 on a new domain, 4 on an aged or compromised domain, 1 with no link; asks: click 3, call 2, reply 2, pay 2, credentials 1, attachment 1 |
| genuine | 11 | 9 from established domains, 2 from young domains |

Prompt length is not the cause: the empties' prompts have a median of 2,811 characters against 2,815 for the rest.

If every recovered lure is caught and every recovered genuine message is cleared, G's balanced accuracy rises from 0.882 to **0.932** (catch 103/110, false alarms 8/110). If every recovered answer is flagged, it stays at 0.882 (catch 103/110, false alarms 19/110). So a fix that answers well is worth up to +0.05, and one that answers badly costs nothing on this measure but shows in the false alarms. Either way, the fix must not disturb the 198 answers that already worked; that is a hypothesis of its own.

## What the runtime offers

Ollama 0.40.2 (the production daemon: `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=2`, flash attention off, `OLLAMA_KEEP_ALIVE=-1`) and gemma4:26b `777b927e3b6b` (Q4_K_M; context length 262,144 by its card). `ollama show gemma4:26b` lists thinking with levels `false, true`, default `true`. **There is no thinking budget.** The API's `think` takes a boolean or a level, and this model has only the two booleans. Probed on 2026-10-28 with a one-line dummy prompt (not a corpus message): `think: "low"` was accepted and ignored (107 generated tokens and 339 characters of thinking, exactly as `think: true`); `think: false` generated 6 tokens and no thinking. Ollama's release notes up to 0.40.12 (2026-10-27) name no thinking budget either. So "a thinking budget" has to be built in glue, not requested from the runtime, and the three fixes below are the three mechanisms this runtime offers.

A request-level `num_ctx` is honoured per request: the daemon reloads the model with the larger context for both parallel slots and keeps it until a smaller request or a restart.

## Arms

All three arms send exp_055's prompt (`kit/prompt55.md`, including its date, 25 October 2026) with the footprint, the registry facts and the message to `/api/chat` on `127.0.0.1:11434`, `temperature 0`, `seed 55`, one call per message, in the manifest's order, exactly as arm G did. Each changes one thing.

| Arm | The fix | What changes | The question it answers |
|---|---|---|---|
| **T** | thinking off | `think: false`; context stays 8,192 | does the reader need to think at all? (Parry's own model step already runs this way) |
| **F** | fall back when thinking fails | pass 1 is G's call unchanged (thinking on, 8,192); when it returns no parsable band, one retry of the same prompt with `think: false` | can the loss be closed in glue, leaving every answer that worked untouched? |
| **C** | more room | `options.num_ctx: 32768`, thinking on; nothing else | does the thinking finish when given room, and what does it then say? |

**Why 32,768.** The empties' thinking was cut at roughly 7,300 tokens (8,192 less a prompt of about 900). 16K would double the room; 32K quadruples it, and exp_051 found no prefill or generation cliff on this runtime up to 40K with flash attention off. A message that still fills 32K is looping, not budgeting; it counts as empty. The call timeout is 900 s for arm C (a 32K fill at the observed ~50 tokens/s takes about 640 s) and 600 s otherwise.

**Recorded, every call:** the message id, the parsed band and items, the full `content`, the full `thinking` text, `done_reason`, `prompt_eval_count`, `eval_count`, `total_duration` and wall-clock seconds. Arm F writes pass 1 and the retry separately (`pass1.jsonl`, `retry.jsonl`) and the final answer (`results.jsonl`). exp_055 recorded none of the counters, which is why its diagnosis rested on timing; control C2 closes that.

**Order and daemon hygiene.** T first (about 15 min), then F (about 3.3 h), then C (2 to 5.5 h, depending on how long the former empties think). `ollama ps` and `ollama --version` are recorded before and after each arm. Arm C ends with one dummy call at `num_ctx 8192` so the daemon is left as found. Nothing else may generate on the daemon during the run.

**Scoring** reuses exp_055's definitions (`kit/score55.py`): flagged = suspicious or likely-scam; balanced accuracy = (catch rate + 1 − false-alarm rate) / 2; attribution = a distinctive declared item named, with Amendment 1's booking-page rule; Wilson 95 % intervals; the same breakdowns by domain age, ask and channel.

## The published copy

The corpus on disk is exp_055's published copy, in which five invented domains that turned out to be registered were replaced by `.example` names. Seven messages differ from what G saw in exp_055, all genuine: X011, X067, X085, X114, X148, X161, X174 (X011 and X161 were among the empties). They are scored like every other message but excluded from the agreement tests (C1, H2), which therefore run on the **193** unchanged messages G answered and the **20** unchanged empties.

## Controls

| | Claim | Passes if |
|---|---|---|
| **C1** | the baseline reproduces | arm F's pass 1 agrees with exp_055's G on the band of ≥ 95 % of the 193, and ≥ 18 of the 20 return empty again |
| **C2** | the wall is the cause | every empty pass-1 answer in F has `done_reason` `length`, or `prompt_eval_count + eval_count ≥ 8,000` |
| **C3** | nothing leaves the machine | the runner runs under exp_054's audit hook (`AUDIT54_LOG`); its log shows calls to `127.0.0.1:11434` and nothing else |
| **C4** | arm C ran at 32K | `ollama ps` during arm C shows CONTEXT 32768 for gemma4:26b |

If C1 fails, the failure is not deterministic on this runtime, H2 cannot be scored, and the record says so; H1, H3 and H4 are measured against fixed numbers and stand. If C2 fails, exp_055's diagnosis was wrong, and the result says what the counters show instead. If C3 fails, the run is void. If C4 fails, arm C is void (the daemon ignored the request); T and F stand.

## Hypotheses

Each is scored per arm, over all 220 messages, with an empty or unparsable answer counted as not flagged, as in exp_055.

| | Claim | Passes if |
|---|---|---|
| **H1** | the loss is closed | the arm returns a valid band on ≥ 218 of 220 (≤ 1 % empty; exp_055's G: 198) |
| **H2** | the fix moves nothing it did not have to | arm C's band agrees with exp_055's G on ≥ 95 % of the 193 unchanged answered messages (F inherits this from C1; T is expected to differ and is read by H3) |
| **H3** | the reader is no worse than its understated number | balanced accuracy ≥ 0.882 **and** attribution ≥ 85 % |
| **H4** | the fix is affordable | median wall-clock per message ≤ 45 s, 95th percentile ≤ 180 s, no call error or timeout (exp_055's G: median 33.6 s; the 22 empties 148–154 s) |

**Decision rule, declared now.** A1's reader call inherits the arm that passes H1, H3 and H4 with the highest balanced accuracy; two arms within 0.01 go to the lower median wall-clock. Arm C must also pass H2 to be chosen: a context bump that changes verdicts it did not need to change is a different reader, not a fix. If no arm passes H1, A1 is registered with the loss declared and empties scored as not flagged, as exp_055 did. If T is chosen, thinking did not earn its place on this task; if F or C is chosen, it did.

**Descriptive.**
- The fate of the 22 in each arm: band, right or wrong, by kind and by domain-age class; the 11 lures recovered or not.
- `eval_count` per arm; for the former empties in arm C, how far past 8,192 the thinking went and what it was doing (the thinking text is kept).
- A budget curve from F's pass 1: the share of answered messages whose generation finished within 2,000, 4,000, 6,000 and 7,000 tokens, which is what a streaming thinking budget at each of those values would have cut.
- T against F and C: where thinking changes the verdict, and on what kind of message.
- Catch by domain age and by ask, false alarms by domain age, as in exp_055; the Opus gap, 0.950 minus each arm.
- `ollama ps` memory for gemma4:26b at 32K against 8K.

## Limits

- **This measures the fix on the reader-alone call, not on A1's.** A1's prompt will carry Parry's rules as evidence and be longer; thinking may behave differently on it. A2 chooses the mechanism; A1 re-counts its empties as a declared measure.
- **One model, one build, one runtime.** gemma4:26b Q4_K_M on Ollama 0.40.2. 0.40.12 is current (2026-10-27); the runtime is not upgraded mid-line. A thinking budget inside the runtime would be a different fix, when and if one exists for this model.
- **Synthetic, one writer family, one pass** at temperature 0, as in exp_055. Andrei's 7 real incidents remain the regression floor for anything that reaches the product.
- **Seven messages differ** from what G saw (above); the agreement tests exclude them.
- **Arm C changes the shared daemon** for the length of its run. The production bot has been out of use since 2026-10-10, and the kit restores the context at the end; a crash before that leaves the daemon at 32K until the next 8K request or a restart.
- **Wall-clock is the mini's** (M4 Pro, 64 GB) with nothing else generating. H4 is void if `ollama ps` shows another model generating during the run.

## Frozen inputs

exp_055's published copy, read in place from `../exp_055_exposure_review_lures/`; nothing is copied. sha256:

| Input | sha256 |
|---|---|
| `data/MANIFEST.json` | `20ec58300860097529e2330201c25ead470c233068d8a7e26d9bd4259223cfc5` |
| `data/cache/lookups.json` | `1208be020334af7e9faa649c9e21ae177a28d738f006fa18152b45fbc4dca113` |
| `data/FOOTPRINT.txt` | `2217e48c7971ad72bccd4a8210369db8c67cc666d4fe12553c4b10b43fe6ed02` |
| `kit/prompt55.md` | `512959511bca927bb0cab5f932f0ca2d8b95ed8ffffa0a752b2b895617af95c9` |
| `data/messages/*.eml` then `*.txt`, concatenated | `d7d605dfbbfeed24a934fd0b06584712ba19e0f83863a4c142e7e9906fbbda28` |

Baseline: exp_055's `runs/20261025T095614Z/G/results.jsonl` and `score.json` (G: caught 92/110, false alarms 8/110, balanced accuracy 0.882, attribution 85 %, 22 empty).

## Dev before registration

- The 22 empties were characterised from exp_055's records (the table above). No corpus message was sent to any model.
- The API shape was probed with a one-line dummy prompt, three calls: `think: true`, `"low"` and `false` (the numbers above). This established that the response carries `thinking`, `done_reason`, `prompt_eval_count` and `eval_count`, and that this model has no thinking levels.
- **No kit exists yet.** It is written after this registration: `kit/arms60.py` (the three arms, the recording, the daemon bookkeeping) and `kit/score60.py` (exp_055's metrics, the agreement tests, C1–C4, H1–H4, the decision rule). It will be smoke-tested on exp_055's dummy message, which is not in the corpus, and the run goes to Andrei as one tmux block.

## Numbering

exp_057 named exp_058 for the harness-layer egress test, and exp_059 is registered, so this is **060**. The roadmap's "A1 (the seeded exp_057)" predates exp_057's registration as the local-inference egress test; A1 takes the next free number when it is registered.

## Kit (2026-10-28, after registration, before any run)

The kit was written the same day, after the registration above was committed, and smoke-tested once on `kit/dummy60.eml` through the driver (`dev/20261028T064040Z/`). The dummy is a fictional speaker-portal lure on a 5-day-old domain, not a corpus message. **No corpus message was sent to any model.**

- `kit/arms60.py` builds exp_055's prompt byte for byte (asserted against exp_055's own builder for all 220 messages, without calling the model), runs one arm or all three in order under exp_054's audit hook, and records every counter and the daemon's state before, during and after each arm.
- `kit/score60.py` scores a run with exp_055's metric functions imported unchanged, then C1–C4, H1–H4 and the decision rule. `kit/test_score60.py` checks it on a synthesized run built from the real manifest and G baseline (a clean run, a run with an external connection, an ignored `num_ctx`, flipped verdicts and empties, and a tie).
- **Smoke result.** T answered in 2.9 s (76 tokens). F's pass 1 and C gave the same band, the same items and the same 2,781 generated tokens (9,043 characters of thinking) in 53.8 s and 58.8 s: an early sign that the context size does not change what the model generates, which is what H2 asks. The daemon honoured `num_ctx 32768` (`/api/ps`: 32,768; 23.3 GB against 18.2 GB at 8,192), and the closing call returned it to 8,192 (18.7 GB). The audit hook recorded only `127.0.0.1:11434`.
- **One addition to the recorded values:** for arm F, a message's `seconds` in `results.jsonl` is the total of pass 1 and the retry, which is what H4 measures; `pass1_seconds` keeps the first pass alone.
