# exp_040 pre-registration: COAPI Voice — make correctness checkable

*Written 2026-10-16 on the mini, before any training step and before any v3 answer exists. **Signed by Andrei: 2026-10-16 21:31 CEST** ("sign it"); unchanged from the draft committed in 4004235. It turns DESIGN.md §9 into fixed rules, and lists every place the build departed from DESIGN. On Andrei's go it becomes the public `HYPOTHESIS.md` in local-first-ai `tasks/chronos/exp_040_…/`.*

## Status at writing — read this first

Everything below exists and is committed (casasol and workspace). **Nothing has been trained, and no arm has answered a v3 row.**

| Item | State | Pin |
|---|---|---|
| Step 0 (quantiser) | done, imatrix Q4_K_M adopted | `qwen3-4b-b11577:q4km-imat` `92bfae6dd5dd` (STEP0_RESULT.md) |
| Training bank | frozen: 734 train + 29 valid items | `6de249dd` / `67934fb5` |
| Abstention variants | frozen: 163 rows | `53421523` |
| v3 test set | **sealed** in a separate session; this session never opened a row | `2c6cbad1` (casasol `c93b451`) |
| Reward | rev 3, 39 tests | `rl40/reward40.py` |
| Prompts → offline filter | 2,267 prompts → **622 train rows**, 79 valid | train `9fe67900` |
| Kit | launcher, selection, H5, fuse/quantise, harness, statistics | casasol `084c4a5` + later commits; mini smoke and preflight green 2026-10-16 |

## What departed from DESIGN.md (all decided before training, each recorded in BUILD.md)

1. **Options are shown** after categorical and yes/no questions ("Options: a / b / c."), in training and at scoring (Andrei, 2026-10-16).
   - Without them, 344 of 427 categorical questions named no candidate, so right answers in other words scored "missing".
   - Retrieval still runs on the bare question.
   - Abstention rows show no options (v3's carry none).
2. **The answer key is also read after a sentence end** (". Answer: …"). Qwen3-4B under the glue's "80–200 words" writes one paragraph and puts the answer last. A key inside a sentence does not count.
3. **On an abstention row, any non-abstaining final line is wrong (−2), not missing (−1).** Guessing costs what false abstention costs.
4. **The training set is 622 rows (78 steps), not ~1,700 (~212).**
   - With the source in context and options shown, the base is right 6/6 on 71 % of categorical rows. The filter drops those because they carry no gradient.
   - Kept rows by type: categorical 204, arith 144, numeric 127, abstention 79, yes/no 68.
5. **Abstention variants** follow v3's method: the withheld chunks are removed from the index, and for PL the whole PL section is withheld.
   - One blind Fable reader per row; only an "undecidable" without doubt became ABSTAIN.
   - Result: 163 of 330 confirmed, then 69 kept by the filter (plus 10 off-topic rows).
6. **Glue arms (A13, R3+G) are scored on the model's raw reply** to glue's answer call. `glue.clean()` drops a period-less final line as a cut-off sentence. The served text is kept for the judged prose. A glue decline (scope "out") counts as an abstention.

## Arms (all served by the Ollama 0.33.0 side server; exp_039's ruler check passes before any cell)

| Arm | Model | Mode |
|---|---|---|
| **B0i** | untrained 4B, imatrix Q4_K_M (`92bfae6dd5dd`) | prompt |
| **R3** | trained 4B, same imatrix recipe (`fuse_quant40.sh`) | prompt |
| B0q8 / R3q8 | the same two at Q8_0 (`3e8eb60ca5fa` / from `fuse_quant40.sh`) | prompt |
| A13 | Unsloth Q4_K_M (`66f6d52aa56a`) + glue + final-line sentence | glue |
| R3G | R3 + the same glue | glue |

- **Prompt mode is the training prompt** (`prompt40.render40`): glue's system prompt + advice suffix + final-line sentence; retrieval k 5, budget 1200, ES from the EN index; serve options (`run_eval.OPTIONS`, greedy); thinking off.
- Exact match is code: reward40's answer verdict = "right".

## Training (mbp; `rl40/mbp_run40.sh`)

- exp_039's configuration, launcher and guards unchanged. Four values differ: data `data/rl40/train`, reward `rl40/reward40.py` (8 components), seed 40, adapter path `runs_rl/r3`.
- Other values: β 0.02, group 6, ε 0.2, temperature 0.9, 512 tokens, LoRA r16 q/k/v/o (11,796,480 parameters), one pass of 78 steps, checkpoints every 25 steps plus the final one.
- **Pilot:** 10 steps. If the projected full run is over 12 h → **STOP** (no full run; record and re-plan).

## Checkpoint selection (mini; `rl40/select_ckpt40.py`)

- Highest exact match on the 79 validation prompts (M5 T42–43 items in EN/PL/ES, plus valid arithmetic), greedy, thinking off. Ties go to the earlier step.
- The base is scored as a reference, not a candidate.
- No eval set is read.

## H5 gate — before any v3 answer (`rl40/h5_rollouts40.py`)

- **Rollouts:** the selected checkpoint answers 200 training prompts (stratified by language × type, seed 40) at the training sampling.
- **Readers:** the 40 with the highest reward go to **two blind readers, Fable and Opus, each alone**.
- **What they see:** the question as shown, the material, the answer, and whether code paid the final line as right.
- **Gaming:** a right final line on an answer that does not explain it, or a guessed label.
- **Rule: more than 4 of 40 gamed by either reader → Stopped — verifier.** No v3 cell runs then, and no second run happens under this pre-registration.

## Measurement (mini; `rl40/fuse_quant40.sh`, then `rl40/measure40.sh`)

1. **Register R3:** fuse the selected checkpoint (mlx-lm 0.32), convert with llama.cpp b11577 to f16, quantise to Q4_K_M with step 0's importance matrix and to Q8_0. Register as `coapi-voice:r3` / `:r3-q8`; the digests go in `models/r3/MANIFEST.json`.
2. **One measure40 call**, after R3 is registered: `B0i:v3 R3:v3 B0q8:v3 R3q8:v3 A13:v3 R3G:v3`.
   - Each arm gets exactly one v3 pass (the runner refuses a second, and any v3 cell before R3 exists).
   - The runner refuses on any pin mismatch: code b1a648c, committed rl40, index, server, digests, v3 seal.
3. **Statistics:** `rl40/compare40.py` R3 vs B0i, R3q8 vs B0q8, R3G vs A13.
4. **This session reads no v3 row** until all six passes are written.

## Hypotheses (v3, 150 rows; thresholds fixed here)

- **H1 (primary), correctness gained.** Exact match R3 − B0i, paired over the 150 rows.
  - **Shown** iff the stratified paired bootstrap 95 % lower bound is > 0 (10,000 resamples, seed 40, strata = language × kind) **and** the exact one-sided McNemar p < 0.05.
  - **Robustness:** the same must hold for R3q8 − B0q8, else "not robust".
  - Registered target +0.10.
- **H2, no loss in judged substance (non-inferiority).**
  - **Rows:** 60 answerable v3 rows (pairs and singles), 20 per language, seeded selection (seed 40), drawn by code after the six passes exist.
  - **Judge:** Fable, fresh instances, blind to arm, the arms' answers to a row shown together and shuffled.
  - **Rubric:** exp_039's scoring prompt and `rubric_chain_A12.md` verbatim. `substance_in_context` = whether the row's source chunk is in that arm's material.
  - **Drift check first:** the 10 fixed A12 v1 rows.
  - **Rule:** R3 ≥ B0i − 0.10 on mean correctness, paired bootstrap 95 % lower bound of (R3 − B0i) > −0.10.
  - Glue arms are judged on their served text.
- **H3, inversions.** Share of contrast pairs (30) with both twins right: R3 − B0i ≥ +0.10, with the H1 tests run over pairs.
- **H4, honest abstention.** On v3's 38 abstention rows (23 withheld-source + 15 off-topic), R3's abstaining-final-line rate rises by ≥ 0.20 over B0i. R3's false-abstention rate on the 112 answerable rows is ≤ 0.05.
- **D1 (descriptive).** R3G vs A13 on exact match (`compare40.py`) and on the judged substance of the same 60 rows.

**Judge budget:** H2 60 rows × 4 arms (B0i, R3, A13, R3G) = 240, plus drift 10, plus H5 2 × 40 = 330 readings.

## Verdict map (the first that fires decides)

1. H5 fails → **Stopped — verifier.**
2. H1 shown at Q4 but not at Q8 → **Not robust.**
3. H1 not shown → **No gain** (interval and McNemar p reported).
4. H1 shown, H2 fails → **Right labels, worse prose.**
5. H1 and H2 pass → **Correctness from a checkable reward**, with H3, H4 and D1 reported.

Any interval straddling its margin is reported as **not shown** for that hypothesis, with the interval.

## Known limits, written before the result

- **Headroom.** With options shown and the source in context, the base already picks right on most categorical rows. A gain, if any, is most likely in PL, arithmetic and abstention, and v3's mix is 51 % numeric and 25 % abstention.
- **Train/test link.** v3 was built blind to everything here except the frozen bank, which it excluded by rule overlap (28 rows dropped) and trigram.
- **Single reader.** The abstention variants rest on one Fable reader per row. A wrong "undecidable" would teach giving up, which H4's false-abstention rule watches for.
- **Labels vs truth.** Labels are checked against the guide text, not against current law. The guide's 30 provisional RDL 29/2026 rules were excluded from the bank.

**Publication:** as exp_039 — weights to Hugging Face whichever way the result goes; training data not published. Andrei decides at the end.

---

**Signature:** Andrei, 2026-10-16 21:31 CEST — "sign it". The text above is frozen at 4004235; any change from here is a dated amendment below.
