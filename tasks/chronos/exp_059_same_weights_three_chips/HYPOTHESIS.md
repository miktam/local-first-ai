# exp_059: same weights, three chips — does the triage decision change with the device?

**Registered 2026-10-28, before any measured run.** Kit: `kit/`. Inputs frozen by the hashes at the end. The runs are Andrei's to start: the phone arm needs the app built and signed on the MacBook, and the MacBook arm needs that machine at home and awake.

## Why

The lead triage adopted in exp_044 (rev 4, now rev 5.1) is answered by one local decision model, Kev-4B, served on the Mac mini. The lead itself arrives on a phone: a WhatsApp message read at a viewing, a voice note in the car. An iPhone 18 Pro joined the setup on 2026-10-27 as the pocket end of the same system. The obvious product question is whether the triage can run **on the phone**, with no server at all. The obvious scientific question comes first: **when the same weights run on a different chip, does the model give the same answer?**

We have measured this kind of drift before, always on the mini: a runtime change moved nothing of substance but shifted speed (exp_051, exp_052), and a precision change made the bot more wrong (exp_053). We have never run one model file on two different GPU families and compared the decisions.

## What makes it falsifiable

Everything except the chip is pinned:

| Pinned | Value |
|---|---|
| Weights | `Kev-4B-Q8_0.gguf`, sha256 `7c2ebed90560522c2801389db482ac1dc4c36d828f201f6074c1d60e433948da` (the file the live gate serves) |
| Engine | llama.cpp tag **b11711**, commit `9d1e036`, on every arm (the mini's gate, the MacBook's server, the phone app's framework) |
| Prompts | the sealed exp_044 test set, 331 enquiries in 7 languages, in its public copy (`test.jsonl`, sha256 `514b6386…`; the as-run file `bf54b4d0…` differs only in phone numbers moved to fiction ranges after the run, and every phone number is masked before the model sees it), masked and rendered by the frozen rev 4 code (`kev_knife_rev4.py`, public copy sha256 `2cdfbdf7…`, as-run `b1940235…`) |
| Readout | the model's own `systemone` template and its `/v1/systemone` contract in b11711 (`tools/server/server-decision.cpp`): no token is generated, so there is no sampling and no seed; each option's score is a dot product between two halves of a 512-wide head, read at the option's `<\|box_end\|>` token and at the last token, then a softmax at the model's calibrated temperature `2.406049966812134` (metadata `qwen35.decision.temperature.*`) |
| Rules | the same rev 4 policy turns the answers into a band and an action on every arm, in code |

So a difference between arms can only come from the device: its GPU kernels, its float accumulation order, its runtime flags. That is the thing under test.

**Kev is a decision model, not a text model.** This matters for the method. The phone cannot run `llama-server`; it runs the engine library through llama.cpp's own iOS example. The app therefore reproduces the server's readout with the public API: tokenize the rendered prompt with special tokens parsed, decode once with embeddings on and pooling off, read the 512-wide projection (`cls.output`, applied by the Qwen3.5 graph) at the marker and pointer positions, take the dot product over the two halves, scale by `1/sqrt(256)`, softmax at the model's temperature. A control (C2 below) checks that this readout equals the server's on the same chip before the phone's numbers mean anything.

## Devices

| Arm | Device | Chip | Memory | Runs |
|---|---|---|---|---|
| `mini-a` | Mac mini (`miktam-mini`) | M4 Pro, 20-core GPU | 64 GB | the live gate: `llama-server` b11711, `-c 16384 -b 8192 -ub 8192`, port 18442 |
| `mini-b` | the same | | | the same gate again, later the same day (control C1) |
| `mbp` | MacBook Pro (`miktam-mbp`) | M5 Max, 40-core GPU | 128 GB | `llama-server` b11711 built there, same flags, loopback |
| `iphone` | iPhone 18 Pro (`miktam-iphone`) | A20 Pro | ~12 GB (reported by the press, not by Apple; the app records `physicalMemory`) | the batch app, airplane mode, Wi-Fi off |
| `mbp-app` | MacBook Pro | M5 Max | | the same app binary run on the Mac as "Designed for iPad" (control C2) |

## Method

1. **Render on the Mac, decide on the device.** `kit/render59.py` takes the sealed test set through the frozen knife's `masked()` and `LEAD_SCHEMA` and writes `data/prompts59.jsonl`: for every enquiry, the 12 question prompts exactly as the server's template renders them (state and instructions escaped as the server escapes them, option order as the schema lists it, `noul` options as `no`/`yes` in the order false, true). The phone does only the model mathematics on these strings; it holds no regex and no business rule.
2. **Mac arms** call the gate the way the knife does (`kit/run59_mac.py`): masked state plus the schema to `/v1/systemone`, one request per enquiry, wall time recorded. Raw answers are kept with full precision. **Run condition:** no other inference job on the machine during a Mac arm (`ollama ps` idle, no experiment running), because H3 compares against `mini-a`'s time in the same session; the smoke test ran at 3.9–6.5 s per enquiry on a loaded mini against exp_044's 1.7 s.
3. **Phone arm** (`kit/ios/`): the app loads the GGUF from its Documents folder, hashes it, reads `prompts59.jsonl`, and for each enquiry decodes the shared prefix (`<\|fim_prefix\|>` state `<\|fim_middle\|>`) once and each question's remainder after it, as the server shares a prefix between the questions of one request. It writes `raw_iphone.jsonl` with per-enquiry answers, wall time, tokens, thermal state and battery level, and `run_iphone.json` with the device identity, iOS version, physical memory, the model hash and the metadata it read. The run happens in one sitting, screen on, airplane mode, Wi-Fi off, phone on a desk at room temperature. Nothing leaves the phone; the output comes back over USB or AirDrop.
4. **Finish** (`kit/finish59.py`): every arm's raw answers go through the frozen `kev_knife_rev4.py` unchanged (its `ask()` replaced by the recorded answers, its `code_facts()` by the facts exp_044 recorded for the same text), so the band and the action are produced by the same bytes on every arm.
5. **Score** (`kit/score59.py`): the hypotheses, the controls, the descriptive values.

Context settings on the phone and the MacBook server follow the mini's gate where they apply: flash attention left at the engine default (`auto`), f16 KV cache, all layers on Metal, batch large enough that one prompt is one batch. The app logs what the engine resolved.

## Controls

- **C1, the floor.** `mini-a` against `mini-b`: the same device twice. Reports the run-to-run agreement of the gate itself. If C1 fails H1's or H2's bar, that hypothesis is reported as *not separable from run-to-run noise* and gets no verdict.
- **C2, the readout.** `mbp-app` against `mbp`: the app's readout and the server's readout on the same M5 Max. Must reach H1's bar on bands and H2's bar on probabilities, or the phone arm is void and the app is fixed before any phone number is reported. This is what separates "the chip changed the answer" from "the port is wrong".
- **C3, continuity.** `mini-a` against exp_044's published `pred_rev4.jsonl`, the same gate nine days earlier (rounded answers only, so decisions, not probabilities). Descriptive.

## Hypotheses (331 enquiries, paired by id; the phone against `mini-a` unless stated)

| | Claim | Passes if |
|---|---|---|
| **H1** | The decision survives the chip | request type (`intent.choice` after rules) agrees on ≥ 0.98 of enquiries **and** the next action agrees on ≥ 0.98 |
| **H2** | The probabilities survive the chip | over all 331 × 12 (enquiry, question) pairs, the largest absolute difference across an answer's probabilities is ≤ 0.01 for ≥ 0.99 of pairs |
| **H3** | The phone is fast enough for a viewing | median wall time per enquiry on the phone ≤ 3 × `mini-a`'s median in the same session (exp_044 measured 1.67 s on the mini) |
| **H4** | The Pro cooling holds for a batch | in run order, the median per-enquiry time of the last 100 enquiries ≤ 1.2 × the median of the first 100 |

**Descriptive, no verdict.**
- D1 tokens per second, prefill, per chip (the mini, the MacBook, the phone), from the per-enquiry token counts and times.
- D2 energy per enquiry where measurable: `powermetrics` on the Macs (the MacBook's pinned sudo rule), battery level and `thermalState` on the phone.
- D3 memory: the phone's `physicalMemory`, whether the Q8_0 file loaded, and the resident size if the app can read it.
- D4 agreement per language, and the list of every enquiry that flipped with its probability margin.
- D5 `mbp` against `mini-a`: two Macs, one engine. Whether any drift is "phone" or "GPU family".

## Declared fallback

If iOS refuses the 4.48 GB Q8_0 file (the per-app memory limit is not published for this device; the app requests the increased-memory-limit entitlement), the phone arm runs a **Q4_K_M** build of the same model, and H1–H2 are then scored against a Q4_K_M run on the mini, not against the live Q8_0 gate. The Q4 file would be produced once, hashed, and used on both arms; that is a dated amendment, not a silent change.

## Verdict map

1. **C2 fails** → the phone arm is void. Fix the readout, re-run C2, then the phone. Nothing is claimed about the chip.
2. **H1 and H2 pass** → the decision survives the chip. SO WHAT: triage can run on an agent's phone with no server; the share-sheet path and the on-device demo are unblocked, and the rule for moving a scored model to a new device is "re-run the controls, not the whole study".
3. **H1 fails** → the chip changed the decision. SO WHAT: the flipped enquiries and their margins are published, and the rule becomes "a decision model is scored per device"; the phone is a capture surface until it is scored on its own.
4. **H2 fails, H1 passes** → the numbers moved but the decisions held. The margins are published; the calibration from exp_044 (H5 there) is marked as mini-only.
5. **H3 fails** → the phone is a demo surface, not a triage device, at this model size.
6. **H4 fails** → sustained batches need a smaller model or a pause; single enquiries are unaffected.

## Limits

- **Synthetic enquiries**, the same 331 as exp_044. They test agreement, not accuracy; accuracy on them was measured in exp_044.
- **One model, one quantisation.** Nothing here says what a Q4 build or a different model would do.
- **The phone path is new code.** C2 is the guard; without it the phone numbers are not reported.
- **Flash attention is resolved per device** by the engine. If the resolution differs between the phone and the Macs, that is a device property under test, not a confound to remove; the log records it.
- **The MacBook is not the phone.** C2 validates the readout on an M5 Max, not on an A20 Pro. A readout bug that only shows on one GPU family would look like chip drift. The margins in D4 are the defence: numerical drift is small and spread; a readout bug is large or structured.
- **Memory on the phone is a guess** until the file loads. The fallback is declared above.
- **Energy is descriptive.** `powermetrics` and a battery gauge are not the same instrument.

## Dev before registration

The Mac path was smoke-tested on three enquiries against the live gate (`kit/run59_mac.py --limit 3`, then `finish59.py`): the finished rows reproduce exp_044's published decisions for those ids (request, action, urgency), and the rounded spam probabilities match to three decimals (0.652, 0.242, 0.452). No phone run, no MacBook run, no full pass on any arm. The smoke output is not kept as a result.

## Frozen input (sha256, first 12)

| File | Hash |
|---|---|
| `exp_044 data/test.jsonl` (public copy, the input on every arm) | 514b6386dfba |
| `exp_044 data/test.jsonl` as run in exp_044 (phone numbers not yet moved) | bf54b4d0cb01 |
| `kit/kev_knife_rev4.py` (copied from exp_044's public kit) | 2cdfbdf794ff |
| `data/prompts59.jsonl` | filled in by `render59.py`, recorded in `data/prompts59.sha256` |
| `Kev-4B-Q8_0.gguf` | 7c2ebed90560 |

## SO WHAT (to be filled in with the result)

---

*The text above is frozen at registration. Any change is a dated amendment below.*
