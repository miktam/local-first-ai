# exp_039 pre-registration: COAPI Voice, on-policy (measurement)

*Written 2026-10-14 on the mini. **Signed by Andrei: 2026-10-14 09:28 CEST** ("I read it and understood, sign it off"); unchanged from the draft committed in 5389e46. It becomes the public `HYPOTHESIS.md` in local-first-ai `tasks/chronos/exp_039_coapi_voice_onpolicy/` on his go.*

## Status at writing — read this first

- **Training was not pre-registered.** R1 was trained on 2026-10-13, 16:05–22:16, on the mbp. Andrei (2026-10-14): "maybe there was no preregistration". What existed before the run, committed:
  - DESIGN.md (rev 1, 2026-10-12): hypotheses, thresholds, verdict map;
  - the reward block (casasol `7bba79a`, 10:50 on 10-13);
  - the kit and prompt set (`ac4a2be`, `0276f69`, 12:45 on 10-13).
  
  This document fixes how R1 is **measured**. It is signed before any R1 answer to a v1 or v2.1 question exists.
- **Known before signing:** checkpoint selection on the 51 validation prompts, reward only, by the DESIGN §7 rule. It selected **step 105** (mean reward 3.078; base 2.353; steps 25/50/75/100: 2.843/2.961/3.049/3.020). No v1 or v2.1 question has been put to R1. No judge has read any R1 output.
- **Deviations from DESIGN, all before measurement:**
  1. Platform: the mbp with mlx-lm-lora 5.8.6, not TRL (DESIGN §8 option c, decided 2026-10-13).
  2. Fable did not review the arithmetic templates or the 140 new probes before training (DESIGN §6). They were written and checked by Opus, plus a blind pass/fail v2.1 rule check.
  3. v2 was replaced by **v2.1** (sealed `b40a3c79`, casasol `5e45fb4`), after the guide audit.
  4. Five silent package behaviours were found and guarded before the run (PREP.md, 2026-10-13 late). The run logs show all guards passed.
  5. A reward weakness was found after training: the `term` glossary contains non-Spanish entries ("base", "lssi", "opt-in", "opt-out"). `term` was therefore looser than designed. H1's Spanish-term floor is scored by `run_eval.py`, not by the reward, and H3's term item by the judge.
  6. H5 is run on fresh rollouts, because the trainer saves none (amendment below).

## Arms

| Arm | Model | Post-processing |
|---|---|---|
| B0 | untrained Qwen3-4B Q4_K_M, Ollama digest `66f6d52aa56a` | none |
| A12 | same | glue `b1a648c` |
| R1 | Qwen3-4B + exp_039 LoRA at step 105 (`1fd2c86b…`), fused with mlx-lm 0.32.0, GGUF via llama.cpp b11577, Q4_K_M | none |
| R1+G | R1 | glue `b1a648c`, unchanged |

All arms are served by the Ollama 0.33.0 side server, with `run_eval.OPTIONS` (temperature 0, seed 42, thinking off, 512 cap). R1 and R1+G are served with the same Modelfile parameters as A12's base. Before any arm is scored, `scripts/a12_side.sh repro` must PASS (3/3 byte-identical).

**R1's prompt (decision 3, train = serve).** R1 is trained on the glue's language-rendered system prompt. R1 without glue is therefore served with that same prompt and retrieval, and nothing else: no scope gate, no citation stripping, no hand-off append, no scrub. **B0 is run the same way (B0-r)**, so that H2a and H1 compare weights, not prompts. exp_035's B0 numbers (English fixed prompt) are reported alongside, descriptively.

## Sets

- **v1:** 60 rows, `coapi_eval_v1.jsonl`, sha256 `efc16ab2…`. It carries the substance question (H2a, H4).
- **v2.1:** 30 rows, sealed `b40a3c79`. A test set: one scored pass per arm; no change to glue, reward or decoding may be motivated by a v2.1 row. It carries H1, H2b and H3.
- Neither set's questions entered the prompt set (trigram max 0.43 vs v1, 0.39 vs v2.1).

## Hypotheses (as in DESIGN §4; v2 → v2.1)

- **H2a (primary, v1, n = 60, paired):** two conditions.
  - R1's with-substance correctness ≥ B0-r's − 0.10, by stratified paired bootstrap (language × type strata, 10,000 resamples, 95% CI).
  - R1's harmful rows ≤ B0-r's harmful rows.
  
  *Null:* R1 < B0-r − 0.10 with the CI excluding 0. *Not shown:* the CI straddles the margin.
- **H2b (descriptive, v2.1):** the same comparison on v2.1, with its CI and no verdict.
- **H1 (v2.1, in-scope rows, R1 no post-processing):** at least 6 of 7 floors.
  - language 100%;
  - plain text ≥ 95%;
  - out of scope: all rows declined;
  - hand-off ≥ 90% of rows that require one;
  - citation-in-context ≥ 94%;
  - numeric preservation ≥ 98%;
  - Spanish term present ≥ 80%.
  
  All by `run_eval.py score` unchanged.
- **H3 (v2.1, R1+G vs A12):** at least one of:
  - Spanish term at first use (strict judge) ≥ 60%;
  - tail collapse ≤ 3/30;
  - harmful rows at least 2 fewer than A12's.
- **H4:** three conditions.
  - R1's v2.1 correctness by módulo within 0.2 across módulos;
  - R1's unsupported-number count on v1 ≤ B0-r's;
  - R1's fabricated-citation count on v1 ≤ B0-r's.
- **H5 (gate, before any merge is used for scoring):** amended as follows. The trainer saved no rollouts, so the selected checkpoint answers 200 training prompts, stratified by language × type, one each, at the training sampling (T 0.9, top_p 0.95, top_k 20, seed 39; casasol `7bb46e8`). The 40 with the highest total reward are read **blind by Fable**, in fresh instances, one question each: *a genuine attempt, or does it satisfy the checks without answering (context copied, numbers or citations stuffed, template with no content, same sentence repeated)?* More than 4 gamed → **STOP**: the verifier is unsound, nothing is claimed about the objective, no scoring follows.
  
  Disclosed in advance: on validation, step 105's copy penalty fired more often than the base's (−0.22 vs −0.16 mean). This is the expected failure shape.
- **D1 (descriptive):** R1 at Q8_0 vs Q4_K_M, field-level agreement on 10 pre-named v1 rows ≥ 9/10 (E35's H5 rows).

## Judge

- Fable, fresh instances through this session's subagents, blind to arm. The A12 rubric chain is copied verbatim from `designA_A12_judge_20260922.json`, with `substance_in_context` set per row.
- The arms' answers to one row are presented together, shuffled and unlabelled.
- **Drift check first:** 10 fixed A12 v1 rows are re-judged, and the delta against 1.217 (E35) and 1.067 (exp_038) is reported before any arm is read.
- Budget: v1 4 × 60 + v2.1 4 × 30 + H5 40 + drift 10 = 410 judgments. B0-r adds 90 more (v1 60 + v2.1 30).

## Verdict map (first condition that fires decides)

1. H5 fails → **Stopped — verifier.**
2. H2a null → **No gain**: at this scale, verifier-scored RL costs substance too. Published.
3. H2a pass, H1 null → **Substance kept, form not learned.** Published.
4. H2a, H1 pass, H3 null → **Form from the verifier, nothing over glue.** R1 kept as evidence. Published.
5. H2a, H1, H3 pass, D1 ≥ 9/10 → **Earns the phone trial.**
6. Any CI straddling its margin → **Not shown** for that hypothesis, with the interval.

R1 weights go to Hugging Face (`miktam/coapi-voice-r1-gguf`) whichever way the result goes. The training data is not published.

## Amendment 1 — the index (written 2026-10-14, before any measurement; **signed by Andrei: 2026-10-14 09:43 CEST**, unchanged from 6746d2f)

**Found while writing the runner.** The ruler check (`a12_side.sh repro`) pins the retrieval index of 22 September (`a1793b82` / `ce512559`). That index was rebuilt on 2026-10-13 after the guide audit; it is now `f1e28120` / `9a422b0e` (RIGHTS.md). The new one is also what R1's training prompts were retrieved from, and the index v2.1 was sealed against (`split_manifest_v2_1.json`). The 22 September files are kept in `index/archive/`.

**Fixed now, before any arm is measured:**
1. **Every arm retrieves from the post-audit index** `f1e28120` / `9a422b0e`, on v1 and on v2.1. This is the same index for all arms, the one R1 was trained and served with, and the one v2.1's golds follow.
2. **The ruler check runs on the archived 22 September index.** It shows that server, model, glue and runner are unchanged, byte for byte (A12, v1 rows 1–3). It PASSED through the new runner on 2026-10-14 (casasol `8284f1c`).
3. **A12 is re-run on v1** with the post-audit index. E35's A12 v1 numbers (old index) are reported descriptively, not compared.
4. **exp_035's B0 v1 numbers** (old index, English prompt) stay descriptive only. B0-r carries every B0 comparison.
5. **Known limitation:** v1's golds were never re-checked against current law (guide-accuracy point 5 is open). Some v1 golds may now disagree with the corrected guide text that every arm retrieves. This affects all arms equally, so the paired comparisons stand. The judge's per-row `substance_in_context` excludes rows whose gold rule is no longer in the retrieved text. The count of such rows is reported.

**Runner** (casasol `8284f1c`, `rl/run_arm.py` + `rl/measure.sh`):
- **R1 and B0-r** get exactly the training prompt. A test shows 60 training rows going out byte for byte through the measurement path.
- **A12 and R1+G** go through `glue.answer()` unchanged.
- **Refusal:** the runner refuses on any pin mismatch, and runs each v2.1 cell once.

## Amendment 2 — quantisation (written 2026-10-14, before any measurement; **signed by Andrei: 2026-10-14 09:49 CEST**, unchanged from ef2a2b1)

**Found while registering R1.** B0 and A12 run on Unsloth's Q4_K_M file of Qwen3-4B (digest `66f6d52aa56a`). R1 is quantised by us: mlx-lm 0.32 fuse, llama.cpp b11577 `convert_hf_to_gguf` f16, then `llama-quantize` Q4_K_M. Unsloth's recipe is not ours (it may use importance-matrix calibration). R1 against Unsloth's base would therefore differ in two things, the training and the quantisation, which is what B0-r exists to prevent.

**Fixed now, before any arm is measured:**
1. **B0-r runs on the untrained snapshot `1cfa9a72` put through R1's exact pipeline.** That is the same converter and quantiser (b11577) and the same Modelfile: `qwen3-4b-b11577:q4km`, digest `8fa738d6b7b9`, GGUF sha256 `5a3a3aab…`. R1 and B0-r now differ only in the LoRA trained by exp_039.
2. **A12 stays on Unsloth's file** (`66f6d52aa56a`), the reference and the ruler. H3 (R1+G vs A12) is a comparison of deployable builds, so its quantisation difference is part of what is compared. It is stated as such in the write-up.
3. **R1:** `coapi-voice:r1` (Q4_K_M, digest `08b5643e92c3`, GGUF `220a06ab…`). For D1: `coapi-voice:r1-q8` (Q8_0, `6d8fbf573900`, GGUF `2d1e1066…`). The Modelfile is the base's own (`ollama show --modelfile`), identical except for `FROM`.
4. Pinned in `rl/run_arm.py`; the runner refuses on any other digest.

## Amendment 3 — judge model, decided by calibration (written 2026-10-14, before any judge has read anything; approved by Andrei 09:50: "let's decide using small calibration")

The pre-registration names Fable for the H5 read and for scoring. Fable costs about 2.5× Opus 5.5 per token and hit its usage limit on 2026-10-13. A switch to Opus 5.5 for **scoring only** is decided by a calibration run before any arm is judged:

1. **Items.**
   - **H5:** the 40 blind rollouts (`audit_blind.jsonl`, sha256 `caae60f0…`), read with `judge/h5_reader_prompt.md`. Four fresh instances, 10 items each.
   - **Drift:** 10 A12 v1 answers from the 2026-09-22 run, judged with `judge/drift_prompt.md` (the A12 rubric chain, verbatim). Two fresh instances, 5 rows each.
   - **Drift rows**, fixed by rule (in-scope v1 rows not named in the rubric, `random.Random(39).sample(…, 10)`): M1-02, M1-09, M1-10, M2-09, M2-10, M3-03, M3-05, M4-06, M6-06, M6-08.
2. **Both models read every item, blind,** with identical prompts: Opus 5.5 and Fable 5.1. Fable's H5 verdicts are **the H5 result of record** (as signed), as are Fable's drift scores.
3. **Opus is used for all scoring judgments only if every one of these holds:**
   - H5 verdicts identical on ≥ 36 of 40;
   - both models on the same side of the H5 stop line (≤ 4 gamed or > 4);
   - drift correctness identical on ≥ 8 of 10 rows;
   - |mean drift correctness, Opus − Fable| ≤ 0.10;
   - harmful flags (usefulness ≤ 1) identical on all 10 rows.
   
   Otherwise Fable judges everything. The decision is computed mechanically once both sets exist, and recorded before any arm is judged.
4. **Whichever model judges,** it judges every arm, with fresh instances and the arms shuffled per row. Comparisons with exp_035 / exp_038 judge numbers (Fable) are descriptive, with the calibration deltas reported next to them.

## Amendment 4 — run 2 with reward rev 2 (written 2026-10-14, after run 1 stopped at H5; **signed by Andrei: 2026-10-14 13:43 CEST**, "amendment is right", unchanged from ec92ca5)

**Run 1** (R1, step 105) stopped at H5: Fable judged 5/40 top-reward rollouts gamed (RESULT_H5.md). Run 1 is recorded as **Stopped — verifier** and is not discarded; the write-up reports both runs.

**The one change: reward rev 2.** The 80-word floor of `length` is removed (casasol `3992722`, `reward.REWARD_REV = 2`). `length` now costs −0.5 only above 220 words (in-scope) and −1 at the cap. Nothing else in the reward changes.

Why only this, measured on the 40 audited rollouts before writing it:
- All 5 gamed answers ran 102–148 words, padded past the floor where the material was thin. The floor was a standing pressure to pad on every answer.
- Three other candidate rules were rejected because they cannot separate gamed from genuine answers by code:
  - citations restricted to the best-matching chunk: genuine answers cite lower-ranked chunks too;
  - sentence-level near-repetition: the padding is paraphrase, and its overlap of 0.05–0.18 sits within the genuine range;
  - no credit for "the material doesn't cover it": the system prompt asks for that sentence, and genuine answers use it.
- The glue's system prompt still asks for "80–200 words". It is unchanged (train = serve, glue frozen), so the model is still told; it is just no longer paid for it.

**Run 2 is otherwise identical:**
- the same prompt set (train/valid byte-identical, `ff21a5fd…` / `2e7081df…`; manifest rebuilt at `d802c96`);
- the same config, seed 39, base snapshot, adapter (q/k/v/o r16, 11,796,480), β 0.02, group 6, ε 0.2, T 0.9;
- the same mbp and pinned environment.

Adapters go to `runs_rl/r2`; the launcher refuses to overwrite a directory that holds adapters. No new pilot: run 1 measured 6 h 11 min for the same configuration on the same machine. The 12 h rule stands as a wall-clock stop: a run not finished in 12 h is stopped and recorded.

**Then, unchanged from the signed procedure:**
1. Checkpoint selection by validation reward. The validation reward is computed with rev 2.
2. H5 on the selected checkpoint: 200 fresh rollouts (seed 39), the top 40 by rev-2 reward, read blind by 4 fresh Fable instances with `judge/h5_reader_prompt.md`. More than 4 gamed → STOP.
3. If H5 passes, R2 replaces R1 in the arms R1 and R1+G: fused, converted and quantised through the same pipeline (amendment 2), digests pinned in `run_arm.py` before any scoring. B0-r and A12 are unchanged. The scoring judge is Fable (amendment 3 outcome).

**One more run only.** If run 2 also fails H5, the verdict for exp_039 is final: **Stopped — verifier (two reward revisions)**, published as such, with no third run under this pre-registration.

## Amendment 5 — the quantisation control (written 2026-10-15 05:18 CEST, before any Q8_0 answer exists; approved by Andrei: "ok, go")

**Why.** The adversarial review (REVIEW_2026-10-15.md) found that the same untrained weights score 0.225 higher in Unsloth's Q4_K_M behind the glue (A12) than in our b11577 Q4_K_M (B0-r). That is the size of R2's H2a gain. So the main result cannot separate "GRPO raised correctness" from "GRPO recovered what our quantisation cost the base". This control re-runs the H2a pair at Q8_0, where quantisation loss is small.

**Arms (prompt mode, exactly as B0-r and R1):**
- **B0-r-Q8:** the untrained snapshot 1cfa9a72 through the same pipeline (b11577 f16 → `llama-quantize Q8_0`), registered with the same Modelfile as `qwen3-4b-b11577:q8`.
- **R2-Q8:** `coapi-voice:r2-q8` (`8478de4545b7`), already registered for D1.

**Rows and judge:**
- The 40 v1 rows flagged substance_in_context = true in the main scoring (`judge_rows.jsonl`; flags fixed, not re-judged).
- Both arms run on all 60 v1 rows through `rl/measure.sh` (ruler check first); only the 40 are judged.
- Fable, fresh instances, `judge/scoring_prompt.md` unchanged, the two answers per row shuffled and unlabelled.

**Primary test (fixed now):** d = corr(R2-Q8) − corr(B0-r-Q8), paired over the 40 rows.
- **"Training gain shown at Q8_0"** if and only if both:
  - the stratified paired bootstrap (lang × type, 10,000 resamples, seed 39, as H2a) has its 95% lower bound > 0;
  - the exact one-sided sign-flip permutation test has p < 0.05.
- **"Not worse at Q8_0"** if the bootstrap lower bound ≥ −0.10 without the gain criteria.
- **Otherwise "not shown"**, with the interval.

**Descriptive only:** harmful counts for both arms; and each Q8 arm against its own Q4_K_M score from the main run (a different judging session, so judge drift applies).

**Reading:**
- If the gain is shown at Q8_0, RESULT.md may say GRPO improved substance on v1, with this interval.
- If not, the publishable claim stays "GRPO did not lose substance", and the +0.20 at Q4_K_M is recorded as confounded by quantisation.
