# exp_039: COAPI Voice, on-policy — form and grounding from a verifier, not from a teacher (design, rev 1)

*Written 2026-10-12 on the mini, after Andrei said: "I want to try to train again, using what Coinbase did and my previous experience." This is a design only. Nothing is built, nothing has run, no exp_039 value exists. It becomes `HYPOTHESIS.md` once Andrei has answered the decisions in §14 and sealed eval v2.*

**Path keys**
- **E35** = `~/REPOS/local-first-ai/tasks/chronos/exp_035_coapi_voice` (public record of the three SFT runs and the glue)
- **CV** = `~/REPOS/casasol/coapi_voice` (glue, scorer, index, data; private)
- **CB** = `~/repos/library/sources/coinbase/owning-intelligence-fraud-agent.md` (the source summary)
- **E39** = the proposed public directory `~/REPOS/local-first-ai/tasks/chronos/exp_039_coapi_voice_onpolicy`
- **P39** = the private home for sets, outputs and judge files (same convention as exp_038's P38)

**Exposure record (2026-10-13):** the session that prepared the v2 seal opened `REVIEW_v2.md` lines 1–60 (rows V2-M1-01..04). The kit (§5–§6) is therefore built in a fresh session; see PREP.md, notes.

**What the designer did not open (2026-10-12):** `logs/conversations.jsonl` (S4), `tasks/coapi/materials/` (S3), the v2 eval rows themselves (only the construction record in E35 and the file listing in `CV/data/eval_v2/`), any exp_035 judge file. The designer read E35's HYPOTHESIS in full, `CV/run_eval.py` (scorer and decoding options), `CV/glue.py` (function list), `CV/build_train.py` (split and prompt shape), `CV/train_pipeline.sh` (adapter config), `CV/RIGHTS.md`, and CB.

---

## 0. Why now, and what Coinbase changes

exp_035's conclusion (E35, 2026-09-22 12:20): three SFT runs on 363 Fable-written answers installed the shape and lowered the base's use of retrieved facts in every configuration; the damage was **not** a function of adapter strength (33 M → 1.3 M), epochs, truncation, quantisation or decoding. "The common factor is the data-and-objective: 363 answer-only sequences under prompt masking teach a *distribution over answers* that the base then follows into invented continuations." Three designs were listed. A (form in glue) was built and is the reference. **B — "change the objective, not the adapter" — was never run.** The published post adds one confound: "every training answer was written by a much stronger model. A 4B imitating that confident register may be learning to *sound* certain where the base would have hedged."

Coinbase (CB, Oct 2026) post-trained Qwen3.5-9B for a closed-output decision with GRPO against a deterministic reward computed from matured fraud outcomes, class-balanced, "not dependent on another LLM judging how persuasive the response sounds", training prompts in the same format as evaluation, facts still in the prompt. It reports gains over Opus 4.5 on all four of its metrics for under $100 of GPU.

What transfers: the objective (on-policy samples scored by code, no teacher text), the discipline (benchmark fixed before and apart from training, compare under the existing policy, class balance, format checks inside the reward, train format = serve format). What does not transfer: Coinbase's output is one label with a true outcome behind it. Ours is prose, and the only true outcomes code can compute are the format floors, grounding against the retrieved context, and arithmetic. Correctness stays with a judge outside the reward. **That gap is the experiment.**

Lessons from exp_035 that are honoured by construction: the device chose the model (step 0 reused; the 4B stays); the adapter's key set is printed and chosen, never defaulted; the glue is the control with a floor; regex for what regex can check and a stronger reader on the page for the rest; the eval deduplicated by rule, not wording (v2 was built that way); negative results published either way.

## 1. The one variable

| | exp_035 v2 (SFT) | exp_039 R1 (GRPO) |
|---|---|---|
| Base | Qwen/Qwen3-4B, HF snapshot `1cfa9a72…` | same snapshot |
| Retrieval | r2 index, k=5, budget 1200, citations-with-tema | same, index hashes pinned |
| Prompt | system prompt v2 + context inside the user turn | same shape; system prompt per decision 3 |
| Thinking | off (`think: false`, as `run_eval.OPTIONS`) | off at rollout and serve |
| Targets | 363 Fable-written answers, regex-gated | **none** — the model's own samples |
| Signal | token-level imitation under prompt masking | **deterministic reward per sample, group-relative** |
| Adapter | LoRA r16, all 7 projections incl. MLP (33 M) | LoRA r16, q/k/v/o only, all 36 layers (~12 M, printed before training) |
| Judge | Fable, A12 rubric, after the run | same, after the run, never in the reward |

Everything else that differs (platform, framework, learning rate, number of steps) is forced by the objective and is recorded, not varied.

## 2. Arms

| Arm | Model | Post-processing | Role |
|---|---|---|---|
| **B0** | untrained Qwen3-4B Q4_K_M (Ollama id `66f6d52aa56a`) | none | the floor for substance and form |
| **A12** | same | glue build `b1a648c` (CV commit) | the reference; must reproduce E35's 60 v1 answers byte for byte, as exp_038 did |
| **R1** | GRPO-trained 4B, merged, Q4_K_M | none | the test of the objective |
| **R1+G** | R1 | the same glue, unchanged | the deployable candidate |

No Gemma arm, no 1.7B arm, no 9B arm (rejected: `casasol/strategy/sources/coinbase/rejected.md`). One trained checkpoint is scored. Checkpoint selection is by reward on validation-tema prompts (§7), never by anything judged.

## 3. Sets and their roles

**v1** — `CV/data/eval/coapi_eval_v1.jsonl`, 60 rows, sha256 `efc16ab2…`. Burned for glue development (disclosed in the post). It is **clean for the model text**: the v1 questions were never training targets and are not in the exp_039 prompt set, and B0 and R1 run without glue. So v1 carries the substance question at n = 60, paired. A12 and R1+G are scored on it too, descriptively, with the known fit.

**v2** — `CV/data/eval_v2/eval_v2_reviewed.jsonl`, 30 rows on the pre-registered grid, rule-fresh against every v1 gold, gates all passed on 2026-09-23, **not yet sealed** (`seal.py` present, no sealed file). It carries form and the ceiling items on rules the glue has never met. Use policy stands as pre-registered in E35 (amendment 6): a test set, never a development set; scored once per cell; no glue change may be motivated by a v2 row. The four arms are four cells.

**Step 0, before anything else:** Andrei reads `REVIEW_v2.md` and runs `seal.py`. A12 is then scored on v2 once and judged. *(2026-10-13: the run goes through the Ollama 0.33.0 side server via `scripts/a12_side.sh`, because production is 0.40.2; reproduction checked 3/3 byte-identical. Amendment 2's pinned path `data/eval_v2/coapi_eval_v2.jsonl` is corrected to the sealed file `data/eval/coapi_eval_v2.jsonl`. Steps A1–A7 in PREP.md.)* That closes E35's E1–E3 and the post's open line ("the number will be added here"). It also fixes A12's v2 numbers before R1 exists, so nothing about R1 can influence them.

**Training-side validation set** — prompts from Módulo 5 temas 42–43 (E35's validation temas), reward only, never judged; used for checkpoint selection and the pilot projection.

## 4. Hypotheses, nulls, thresholds

Correctness is on E35's 0–2 scale; "with substance" means the judge found the gold's key rule and numbers in the retrieved context. Thresholds come from E35's numbers: B0 on v1 with substance 1.50 (38 rows), SFT v2 1.18, v3 1.21, A12 1.50 / overall 1.217 / 13 harmful / tail 7/60 / native ES 10/12, PL 11/17 / Spanish term at first use 43.5% strict.

**H2a — substance kept (primary, confirmatory, v1, n = 60).** R1's with-substance correctness ≥ B0's − 0.10, paired by row, stratified paired bootstrap as in exp_038 (95% CI); and R1's harmful rows ≤ B0's harmful rows. *Null:* R1 < B0 − 0.10 with the CI excluding 0 → substance is lost under an on-policy objective with no teacher text; the exp_035 claim widens from "answer-only SFT" to "SFT and verifier-scored RL at this scale". *Not shown:* CI straddles the margin.

**H2b — substance on fresh rules (descriptive, v2, n = 30).** Same comparison on v2; reported with its CI, no verdict, because n is small and a new set changes difficulty.

**H1 — form from the verifier, no post-processing (v2, 28 in-scope rows).** R1 meets at least 6 of the 7 floors: language 100%, plain text ≥ 95%, out-of-scope 2/2, hand-off ≥ 90% of rows that require one, scripted citation-in-context ≥ 94%, numeric preservation ≥ 98%, Spanish term present ≥ 80%. *Baseline:* B0 met 1/7 on v1; SFT v2 met 6/7. *Null:* fewer than 6 → the verifier-trained objective does not carry form at this size; glue stays.

**H3 — the ceiling moves (v2, R1+G vs A12).** At least one of: Spanish term at first use (strict judge) ≥ 60%; tail collapse ≤ 3/30; harmful rows at least 2 fewer than A12's on v2. These are the three things E35 recorded as "not glue-fixable; the model's ceiling". *Null:* none moves → the trained model adds nothing over glue; published, not deployed.

**H4 — no memorisation, no new fabrication.** R1's v2 correctness by módulo within 0.2 across módulos (Módulo 6 is never trained; v2 has 5 M6 rows); R1's unsupported-number count on v1 ≤ B0's; R1's fabricated-citation count on v1 (no glue) ≤ B0's. *Null:* any missed → leak or new fabrication, reported per row.

**H5 — the verifier is sound (a gate, not a hypothesis).** From the selected checkpoint's last 200 rollouts, the 40 with the highest total reward are read blind by a Fable instance with one question per rollout: *is this answer a genuine attempt, or does it satisfy the checks without answering (context copied, numbers or citations stuffed, template with no content, same sentence repeated)?* More than 4 gamed → **STOP**, record, fix the reward, re-register the reward block, re-run. Also logged every step: mean and p95 rollout length, fraction at the 512-token cap, 8-gram repetition rate, per-component reward means. A jump in total reward is a verifier question before it is a capability result (RIVER and MiMo, [[verifiable-rewards-territory]]).

**D1 — device fidelity (descriptive).** Merged R1 at Q4_K_M is in the base's size class (~2.5 GB). Q8_0 vs Q4_K_M field-level agreement on 10 pre-named v1 rows ≥ 9/10 (E35's H5 shape, on the mini; the phone port is FEATURE-010's).

**Power, stated.** With n = 60 and E35's row variance, H2a can detect a loss of exp_035's size (−0.3) and cannot rule out a loss of −0.1. The margin is the same tolerance E35 used, and "not shown" is a live outcome.

## 5. The reward

Deterministic, computed per rollout by `CV/rl/reward.py`, built from `run_eval.py`'s primitives (`detect_answer_lang`, `_cites`, `cite_ok`, `_nums`, `derivable`, `_HANDOFF`, `_MARKDOWN`, `_REFUSAL`, glossary from `guide_bm25.load_index("en")`). No model call of any kind inside the reward. Components are summed; GRPO's per-group normalisation then compares the six samples of one prompt against each other.

| Component | Check | Value |
|---|---|---|
| `lang` | answer language == question language | +1 / −1 |
| `plain` | no markdown | +0.5 / −0.5 |
| `cite` | in-scope row: ≥ 1 citation and every citation passes `cite_ok` against the context | +1; −1 per unsupported citation, floor −2; 0 if no citation |
| `num` | every number in context ∪ question, or `derivable` from them | +1; −1 per unsupported number, floor −2 |
| `arith` | arithmetic prompts only: the code-computed gold present within 0.5% | +2 / −1 |
| `term` | a glossary Spanish term (≥ 4 chars) present | +0.5 / 0 |
| `handoff` | advice prompts: a professional route present (lawyer, notary, appraiser, tax adviser, gestor) and no figure or drafted clause | +1 / −2 |
| `oos` | out-of-scope prompts: decline, no citation, < 80 words | +2 / −2 |
| `length` | 80–220 words 0; outside −0.5; hit the 512-token cap −1 |
| `repeat` | any 8-gram three or more times | −1 |
| `copy` | longest common substring with the context longer than T characters | −1 |

**T** is set before training at the 95th percentile of A12's 60 v1 answers (the glue's answers are the model's text, quoting the guide as a good answer legitimately does), and recorded. Weights are fixed now and not tuned; a change is a re-registration of this block.

**Class balance** (Coinbase's rule, applied to prompts): stratified sampling so that EN, PL and ES prompts are 1:1:1 (the exp_035 set was EN-heavy and PL was the weakest floor) and the five types (numeric, simple, multi, advice, oos) each get at least 10% of groups; and, because each GRPO group is one prompt, no prompt type can dominate the gradient by volume.

**What the reward cannot see — written down, as [[evals-first]] requires.** A rule inverted with the right numbers (exp_035's dominant failure; the regex scored 97.7% numeric preservation on B2m while it inverted their meaning). An invented statute name on a real article number. Confident filler that cites once and says nothing. Native fluency. The reward is blind to all four; H2a is the only guard for the first three and the judge the only reader of the fourth. The `copy` and `length` terms are there because the cheapest way to satisfy `cite` and `num` is to quote the context back.

## 6. The prompt set

No answers are needed, only prompts and, for arithmetic, a number computed in code.

- **Sources:** the 377 training questions already in `CV/data/train/prep_*.jsonl` (built from doc2query for train temas); further doc2query questions from the same temas (`CV/data/doc2query/M1–M5*.jsonl`) up to ~600 in-scope prompts; `prep_special.jsonl`'s out-of-scope and advice probes plus enough new ones, templated in code in three languages, to reach the 10% floors; **arithmetic prompts** (~120) generated in code from the guide's own formulas with the gold computed in code — the closest thing we have to Coinbase's matured label. **No template may test a v1 or v2 key rule** (correction 2026-10-13: the first draft of this list named v1 rules — usufruct 89 − age is v1 M6-03, early-repayment caps v1 M2-02, IBI proration v1 M4-05 — which would train the very rules H2a measures). Templates are filtered against the v1 key-rule avoid-list of exp_035 amendment 9, and against v2's key rules by a script whose output the builder sees only as pass/fail.
- **Split:** E35's `split_manifest_v1.json` stands. Train temas: Módulos 1–4 and M5 T39–41. Validation temas (M5 T42–43): reward-only validation. Módulo 6: never. No v1 or v2 question text enters; a trigram check against both at the 0.45 threshold of `dedupe.py` is run and its maxima recorded.
- **Context:** retrieved by the same chunker and budget the phone uses, inserted into the user turn exactly as `build_train.py prepare` does, so train format = serve format (Coinbase's rule; E35's shape).
- **Fable's role:** strongest model first, in its reviewing capacity only — it reviews the arithmetic templates and the new oos/advice prompts for correctness and naturalness in PL and ES, blind to v1 and v2. It writes no answer and sees no rollout except in the H5 audit.
- **Manifest:** counts per language and type, sha256 of the prompt file, of the index files, of the guide (RIGHTS.md hashes, re-verified), of `reward.py` and of the TRL config, all written before the pilot.

## 7. Training recipe

- **Base:** `Qwen/Qwen3-4B` bf16 safetensors, the E35 snapshot. Chat template with thinking disabled for every rollout; the same `think: false` at serve.
- **Adapter:** LoRA rank 16, alpha 32, dropout 0.05, targets `q_proj, k_proj, v_proj, o_proj` on all 36 layers. The harness prints the target set and the trainable parameter count before the first step and writes both to the manifest (E35's "print the adapter's key set" rule).
- **GRPO (TRL `GRPOTrainer`):** `num_generations` 6; `max_completion_length` 512 (matches `run_eval.OPTIONS["num_predict"]`); rollout temperature 0.9, top-p 0.95; 8 prompts per step (48 rollouts); learning rate 1e-5, cosine, warm-up 5 steps; `beta` per decision 4; `epsilon` 0.2; loss type TRL's default at the pinned version, recorded; one pass over the prompt set (~100 steps); seed 39.
- **Checkpoints:** every 25 steps, all kept. Selection: the checkpoint with the highest mean reward on the validation-tema prompts (one rollout per prompt at temperature 0). H5 is run on that checkpoint. If H5 fails there, no other checkpoint is tried; the run stops.
- **Pilot and frozen budget rule** (exp_037's lesson): 10 steps first, timed. On the mbp: projected full run > 12 h wall → **STOP** and re-plan (fallback (a): > 6 h or > $60). The projection and the rule are written before the pilot runs.
- **Nothing is swept.** One configuration, pre-registered. A second configuration is a second pre-registered block, after the first is judged.

## 8. Platform, environment, rights

| Option | For | Against |
|---|---|---|
| **(a) One rented NVIDIA GPU, 80–96 GB; TRL + vLLM colocate** | Coinbase's path at our scale; rollouts dominate and vLLM batches them; a 4B LoRA fits with room; ≤ 3 h, ≤ $60 | a new account and a dated rights line; environment must be pinned (CUDA, torch, transformers, trl, vllm) |
| (b) The mbp (M5 Max, 128 GB), TRL over MPS | no money, no account | no vLLM; HF generation on MPS is 5–10× slower, so 10–30 h; TRL on MPS is less travelled |
| (c) **mlx-lm-lora on the mbp** (M5 Max, 128 GB) | local, no account, no spend; GRPO / Dr. GRPO / DAPO / GSPO with a custom reward-functions file; built on mlx-lm (5.8.6 needs mlx ≥ 0.32.3, mlx-lm ≥ 0.32.0) | community package moving fast (1.0.5 in March → 5.8.6 in October): pin exactly, smoke-test on the mini first; Andrei runs the mbp by hand |

*Correction 2026-10-13:* rev 1 said nothing upstream supports GRPO on MLX, citing mlx-lm 0.31.3 as latest. mlx-lm 0.32.0 shipped on 1 October (no GRPO in it), and mlx-lm-lora carries GRPO on top of it. Recommendation changed from (a) to **(c)**, with (a) as the fallback if the mbp pilot projects past the time cap. Model unchanged: Qwen3-4B. Full step list in PREP.md, part C. Rights: only S1/S2 (Andrei's own study notes), Fable-written doc2query questions, code-generated prompts and the base model leave the machine. S3 and S4 never do. FEATURE-009's system boundary already permits "an optional Colab training run on non-sensitive data"; a rented GPU is the same case. Andrei writes one dated line in `RIGHTS.md` naming the platform. Cloud *LLM* calls stay in this session's subagents ([[feedback-cloud-via-session-not-api]]); a GPU host is not an LLM API.

## 9. Verifier audit and monitoring

Per step: mean total reward, each component's mean, rollout length mean and p95, cap-hit fraction, repetition rate, `copy` trigger rate. Plotted after the run, in the record. The H5 audit (§4) runs on the selected checkpoint before any merge. Its 40 rollouts and the reader's verdicts are kept in P39 and summarised in E39.

## 10. Artefacts

Merge the adapter (`mlx_lm fuse` on the mbp from the HF adapter, or `peft merge` on the GPU host; one path, recorded), convert to GGUF, quantise Q4_K_M and Q8_0 with the `train_pipeline.sh --no-train` steps, register `coapi-voice:r1` in Ollama with the same Modelfile as v2. D1 on the mini. The R1 weights go to Hugging Face as `miktam/coapi-voice-r1-gguf` **whichever way the result goes**, with the card written in the v1–v3 pattern ("evidence, not product").

## 11. Scoring and judging

- **Regex scoring:** `run_eval.py score` on every arm × set.
- **Judge:** fresh Fable instances through this session's subagents, blind to arm, A12 rubric chain copied verbatim from `designA_A12_judge_20260922.json`, `substance_in_context` set fresh per row. Arms' answers to the same row are presented together, shuffled and unlabeled, so the comparison is paired. **Drift check** (exp_038's addendum lesson, registered here): 10 fixed A12 v1 rows re-judged first; the delta against E35's 1.217 and exp_038's 1.067 is reported before any arm is read.
- **Statistics:** stratified paired bootstrap over rows (language × type strata), 10,000 resamples, as exp_038.
- **Budget:** v1 4 arms × 60 = 240; v2 4 × 30 = 120; H5 audit 40; drift 10 → about **410 judgments**.

## 12. Verdict map (ordered; the first condition that fires decides)

1. H5 fails → **Stopped — verifier.** Reward recorded as unsound; nothing claimed about the objective.
2. H2a null (substance lost, CI excluding 0) → **No gain.** The exp_035 claim widens: at this scale, with this data, both SFT on teacher answers and on-policy RL against a deterministic verifier cost substance. Glue stays. Published.
3. H2a pass, H1 null → **Substance kept, form not learned.** The objective is harmless and useless here; glue stays. Published.
4. H2a pass, H1 pass, H3 null → **Form from the verifier, nothing over glue.** The post-processing moves into the weights at no substance cost, but the ceiling is unmoved. Glue stays the reference; R1 kept as evidence. Published.
5. H2a, H1 and H3 pass, D1 ≥ 9/10 → **Earns the phone trial.** R1+G becomes the FEATURE-010 candidate; the swap bar written in E35 for Gemma applies to it.
6. Any CI straddling its margin → **Not shown** for that hypothesis, with the interval.

## 13. Sequence and budget

| Step | Who | Time |
|---|---|---|
| 0. Seal v2; score and judge A12 on v2; add the number to the post | Andrei (seal) + session | 1 h Andrei, ~1 h session |
| 1. `reward.py` + unit tests on A12's v1 answers (calibrates T; recorded) | session | 0.5 day |
| 2. Prompt builder, arithmetic generator, Fable review of templates, manifest | session | 0.5 day |
| 3. TRL harness, pinned env, pilot 10 steps, projection against the frozen rule | session + Andrei (account) | 0.5 day + GPU ~15 min |
| 4. Full run, checkpoints, selection, H5 audit | GPU + session | ≤ 3 h GPU, ≤ $60 |
| 5. Merge, GGUF, Ollama tag, D1 | session, mini | 1 h |
| 6. Score B0, R1, R1+G, A12 on v1 and v2; judge; bootstrap | mini + session | ~1 h mini, ~410 judgments |
| 7. Verdict, record in E39, HF upload, blog post | session + Andrei | 0.5 day |

Andrei's own time: about 3 h (seal review, decisions, rights line, GPU account, reading the result).

## 14. Decisions for Andrei (recommendation in bold)

1. Platform: **(c) the mbp with mlx-lm-lora (GRPO), smoke-tested on the mini** / (a) rented NVIDIA GPU with TRL + vLLM as fallback / (b) TRL over MPS. *(Changed 2026-10-13; see §8.)*
2. Visibility: **public pre-registration, reward code and prompt builder; sets, glue and guide private, bound by sha256 (exp_038 convention).**
3. Training system prompt: **the glue's language-rendered prompt (train = serve)** / exp_035's fixed English prompt.
4. KL leash: **β = 0.02** / 0.
5. Seal v2 and score A12 on it before the build: **yes.**
6. Scope-gate head in exp_039: **no** (exp_040 if the harness works).
7. Stop rules: **as written in §7 (mbp: 12 h projected) and §4 (H5).**

## 15. Bearing on the published exp_035 post

Five edits, all cheap, two of them on this experiment's critical path:

1. **Add the v2 number** the post promised ("It runs once, after the set is sealed, and the number will be added here"), in an `**Update, <date>.**` block — the site's own convention since the 5 October post — together with the judge-drift datapoint exp_038 already produced (the same A12 answers scored 1.067 under a fresh judge against 1.217). Step 0 produces it.
2. **State the claim boundary against Coinbase's result** in the same block: a 9B trained with GRPO against a deterministic outcome reward beat Opus 4.5 on a closed-output decision; the post's refutation covers answer-only SFT on a stronger model's prose; exp_039 tests whether the objective was the variable. This sharpens the title's claim without retracting it, and it names the cheap experiment the post asked someone to run.
3. **Add the harm row to the table** (answers that would mislead a buyer, for base, v2, v3 and glue, from the judge files). It is the consequence-weighted number, and it currently sits in prose after the mean.
4. **Correct the format-floor count (found 2026-10-13):** the post's table gives the glue 7 of 7 floors; A12's judge record shows 5 of 7 (answers citing 43/60 and artefact rows 15 both missed). The explainer page already shows 5.
5. **Add a reproduce footer:** base GGUF digest `66f6d52aa56a`, index hashes, glue commit `b1a648c`, Ollama 0.33.0, eval v1 sha256, and the sentence the HF cards already carry — the training data is not published, the study notes are Andrei's own.

## 16. Relation to patterns

[[facts-in-context-form-in-weights]] (the decision row; the variable is the objective) · [[verifiable-rewards-territory]] (a verifier can be gamed; audit it; class balance) · [[evals-first]] (what the reward cannot see; benchmark fixed before and apart from training) · [[fidelity-not-benchmark]] (same base, same glue, one variable) · [[deterministic-glue-pipeline]] (the control arm) · [[feedback-strongest-model-first]] (Fable reviews, never writes answers) · [[feedback-check-runtime-releases]] (re-check mlx-lm and TRL releases before pinning).
