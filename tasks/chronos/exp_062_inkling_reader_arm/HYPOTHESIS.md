# exp_062: Inkling 2 Mini in the local reader's seat, reasoning on

**Registered 2026-11-21, unsigned, before any kit and before any corpus message was sent to this model.** The bars below are proposed; they are Andrei's to change until he starts the run, and starting the run signs them. One arm, on exp_055's corpus, with exp_055's prompt and exp_060's scoring, read in place. The run is Andrei's to start: about seven hours on the mini.

## Why

The local reader's seat is held by gemma4:26b with exp_060's fix F (think, and retry without thinking when the answer is empty): balanced accuracy 0.914 on exp_055's 220 messages, against 0.950 for the frontier model alone. On 2026-11-20 Thinking Machines released Inkling 2 Mini (26.4B parameters, 3.4B active, a looped transformer, Apache 2.0), small enough for the mini. A public leaderboard puts it level with a frontier model on finding and patching security bugs in code. That benchmark is agentic bug-finding, not reading a message, so it predicts nothing here either way. This experiment asks the question on the task this line actually has: **given the same call, does Inkling 2 Mini read these messages better than the model in the seat?**

It is a model swap in exp_055's reader-alone call (arm G's call shape). It is not Campaign A's A1, which changes the design around the reader.

## The arm

| Arm | Model | Call |
|---|---|---|
| **I** | Inkling 2 Mini, UD-Q4_K_M | exp_055's prompt as the single user message, `temperature 0`, `seed 55`, **no reasoning parameter** (the model's default: reasoning on), no output cap, context 8,192, one call per message in the manifest's order, timeout 900 s |

**No fallback.** An empty or unparsable answer counts as not flagged, as in exp_055 and exp_060. A retry in glue would be a second arm and is not registered here.

**Compared against, from the published records (nothing is re-run):**

| | Run | Answered | Lures caught | Genuine flagged | Balanced accuracy | Item named |
|---|---|---|---|---|---|---|
| G, gemma4:26b, same call shape | exp_055 `runs/20261025T095614Z/G` | 198 | 92/110 | 8/110 | 0.882 | 85 % |
| F, gemma4:26b with the retry (the seat's holder) | exp_060 `runs/20261101T152955Z/F` | 220 | 100/110 | 9/110 | 0.914 | 95.2 % |
| O, the frontier model alone | exp_055 | 220 | 105/110 | 6/110 | 0.950 | 100 % |

## The runtime, pinned

- **llama.cpp built from pull request #32704 at commit `42fd370cf9bc758f02f39d5cdfa15822f86f23a3`**, in `~/models/llama.cpp-pr32704`. The pull request was open and unmerged at registration and its head has moved since that commit; the build is not updated for this run. Unsloth's note on these files says that without this fix the looped layers lose their scales and the output is wrong.
- **Model file:** `Inkling-2-Mini-UD-Q4_K_M` from Unsloth's `Mainline_Support` folder, 16,610,544,608 bytes, sha256 `78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae`. The runner checks the hash before the server starts.
- **Server:** `llama-server -m <file> -c 8192 -np 1 -ngl 99 --host 127.0.0.1 --port 18450`. One slot, so the 8,192 context is the slot's. No reasoning flag.
- **The mini** (M4 Pro, 64 GB), macOS as recorded by the run. The production Ollama is not used and not changed; gemma4:26b may stay resident in it.

**Every figure this run produces is provisional** until #32704 is merged and the run is repeated on a tagged llama.cpp release. That repeat is the registered follow-up.

## Recorded, every call

The message id, the parsed band and items, the full `content`, `finish_reason`, prompt and completion token counts, the server's timings and wall-clock seconds; the thermal pressure level before the call. The reasoning text is written to a private folder outside this repository; its sha256 is recorded in the run folder, as exp_060 did with the thinking text.

## Controls

| | Claim | Passes if |
|---|---|---|
| **C1** | the inputs are exp_060's | the five sha256 values of exp_060's *Frozen inputs* table match, and every prompt equals exp_055's builder byte for byte |
| **C2** | the runner talks to the side server only | exp_054's audit hook on the runner shows connections to `127.0.0.1:18450` and nothing else. **This covers the runner, not the server:** the server is a build from an unmerged pull request and its own network behaviour is not measured here |
| **C3** | the server ran as pinned | the model file's hash matches before the start; the server log shows one slot with a context of 8,192 |
| **C4** | the timing is the mini's own | the thermal level reads 0 before every call; `ollama ps` shows nothing generating at the start and the end |

If C1 fails the run is void. If C2 fails the run is void. If C3 fails the run is void. If C4 fails, the timing figures are void and the verdicts stand.

## Hypotheses (proposed bars)

Scored over all 220 messages with exp_055's metric functions, imported unchanged.

| | Claim | Passes if |
|---|---|---|
| **H1** | it answers | a valid band on ≥ 218 of 220 (the standard F met; G returned 198) |
| **H2** | it reads at least as well as the seat's holder | balanced accuracy ≥ 0.914 **and**, on the messages where exactly one of I and F is right, I is right at least as often as F |
| **H3** | it does not buy catches with false alarms | genuine messages flagged ≤ 11 of 110 (F: 9) **and** attribution ≥ 85 % |

**Time is descriptive, declared now.** exp_060's bar (median ≤ 45 s) was set for gemma at about 50 tokens a second and is not applied. The dev probe below took 118 s for one message, so about two minutes a message and about seven hours for the run are expected. Median, 95th percentile, total hours and tokens generated per message are reported.

**Decision rule, declared now.**
- **The seat changes** only if H1, H2 and H3 all pass here **and** again on a tagged llama.cpp release that carries #32704. One pass on a provisional runtime changes nothing in the guard.
- **The line stops** if balanced accuracy is below 0.882 (G's, the same call shape): no further Inkling arm is registered for this seat.
- **Otherwise** the result is recorded and gemma4:26b with F keeps the seat. If H2 and H3 pass and only H1 fails, the registered next step is a retry in glue, as F was for gemma.

With 110 messages per class the intervals are wide: F's 0.914 and the frontier's 0.950 are not separable by them, which is why H2 also asks for the paired count.

## Descriptive

- The paired table against F and against G: both right, both wrong, only I, only F; the same against G.
- Catch by domain age and by ask, false alarms by domain age, as in exp_055; the gap to the frontier model, 0.950 minus I.
- Empties: `finish_reason` and token counts, and whether each hit the context.
- Reasoning length per message, and the share of generated tokens that were reasoning.

## Limits

- **The corpus has been used twice** (exp_055, exp_060), and F was chosen on it. Nothing here is tuned on it: one arm, one pass, the prompt unchanged. Per-message results of this arm are not opened before the run is complete.
- **Synthetic, one writer family, one pass** at temperature 0, as in exp_055. Andrei's real incidents remain the regression floor for anything that reaches the guard.
- **One quantisation, one build, one machine.** A provisional runtime, as above.
- **Reasoning on only.** The model card says the minimum reasoning setting does not fully disable reasoning; a smoke test on 2026-11-21 found that only a zero reasoning budget on the server removed it. That condition is not an arm here.
- **The frontier figure is exp_055's**, from October, and is not re-measured.

## Frozen inputs

exp_055's published copy, read in place from `../exp_055_exposure_review_lures/`; the sha256 values are those of exp_060's *Frozen inputs* table and are checked by the runner (C1). Baselines: exp_055 `runs/20261025T095614Z/G/results.jsonl` and exp_060 `runs/20261101T152955Z/F/results.jsonl`.

## Dev before registration

- **One probe on exp_060's dummy message** (`kit/dummy60.eml`, a fictional speaker-portal lure on a five-day-old domain, not a corpus message), 2026-11-21, with the server and the request exactly as pinned above: 919 prompt tokens, 1,637 generated, 117.6 s, `finish_reason` `stop`, 5,387 characters of reasoning returned in `reasoning_content`, and a `content` that exp_055's parser reads (a band and two items). gemma4:26b generated 2,781 tokens on the same dummy in exp_060's smoke test.
- **A translation smoke test the same day** (one made-up Russian passage) established the speed, about 16 tokens a second, and that `--reasoning off` and a request-level setting did not remove the reasoning.
- **No corpus message has been sent to this model.**

## Numbering

exp_061 is the latest registered folder, so this is **062**. It takes this fortnight's slot for a new experiment folder.

## Kit

Written after this registration is committed: `kit/arm62.py` (the arm, the recording, the checks for C1 and C3) and `kit/score62.py` (exp_055's metrics, C1–C4, H1–H3, the paired tables, the decision rule). It is smoke-tested on the dummy message through the driver, and the run goes to Andrei as one tmux block.
