# Experiment 038: Kolibri as a drop-in answering model in CasaSol's COAPI glue

*Pre-registered: 2026-10-10T08:29:29Z · Status: pre-registered; no scored row has run.*

**Design:** rev 2, with Andrei's decisions of 2026-10-06 and 2026-10-10.
**Builds on:**
- exp_035, whose A12 glue is the reference answering pipeline;
- exp_036 and exp_037, our MLX port of Aleph Alpha's Kolibri-1. exp_037's forced-routing gate PASSED on run 3.

**Private home:** a local-only repository on the mini (no remote) holds the glue, the prompts and strings, the sets, the raw outputs and the judge notes. This directory publishes the code that can be published, the rules, and the sha256 of every private file.

---

## Question

Untrained, behind CasaSol's A12 glue with everything but the model held fixed, does Kolibri-1 4-bit answer Spanish property-law questions more correctly than gemma4:26b, by enough to earn a live-bot trial? Is German where it wins?

"The glue" is exp_035's A12 pipeline. It is CasaSol's only answering path with a sealed eval, a frozen rubric and a judged reference:
- BM25 retrieval over the study guide;
- a one-call scope gate;
- one answer call;
- deterministic post-processing (citation stripping, hand-off).

It is not the live bot's production prompt or retrieval. Those are covered by competitor protection, and the retrieval there draws on material that is not cleared for cloud judging. **G26 is the incumbent model run inside the A12 glue, not the production responder.** Transfer to the live bot is untested.

## Arms

| Arm | Model and build | Host and runtime | Role |
|---|---|---|---|
| **K4** | Kolibri-1, our MLX affine 4-bit g64 build | M5 Max MacBook Pro; macOS 27.0 (26A428), MLX and mlx-metal 0.32.3, mlx-lm 0.32.0; through exp_037's gated runner, B = 1, `reasoning_effort="none"` | **Primary** |
| **G26** | `gemma4:26b` (Ollama id `5571076f3d70`; GGUF Q4_K_M; on Ollama 0.40.2 the llamacpp runner, recorded by sub-manifest and blob digests) | Mac mini M4 Pro, 64 GB; Ollama 0.40.2 | The incumbent model, in the A12 glue |
| **A12** | `hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M` (`66f6d52aa56a`) | mini; Ollama **0.33.0**, exp_035's version, run as a side server | Reproducibility anchor; descriptive comparator (D5) |
| **K8** | Kolibri-1, our MLX affine 8-bit g64 build | mbp; as K4 | Reference only: separates model from quantisation (D8). Does not fit 64 GB |
| **K4-MED** | K4, with the **answer call** at `reasoning_effort="medium"`; the scope-gate call stays at effort none | mbp; as K4 | Reference only (D9): what effort none costs Kolibri |

**Binding to exp_037 (code, not prose).** No Kolibri row is produced unless all of these hold:
- exp_037's newest real gate record is run 3's, `gate_20261008T180634Z.json`, sha256 `e6d80d7b4565fef08ba275b1c15f635880c3155441784c713ff4d8b296c62341`, with verdict K8 PASS and K4 PASS (exit 0);
- `require_environment` matches its pins;
- exp_037's runner tree hashes to `75a50caa…` (after its Amendments 1 and 2);
- the build's port file is exp_037's (`2c153357…`).

Every Kolibri row carries the gate record's name and sha256.

**What exp_037 taught, built in:**
- One fresh batch generator per model call.
- exp_037 Amendment 2's settle of the `BatchKVCache` offsets, once per decode step. mlx-lm 0.32.0 leaks one Metal buffer per NoPE layer per step without it.
- No tokenizer check cached by `id()`.
- B = 1 everywhere. exp_037's control C1 flipped 6 of 100 answers at batch 8.

Tests on tiny builds include:
- a control: with the settle disabled, a 4,000-step call raises the pilot's `Resource limit` error under a 15,000-buffer margin;
- a soak: one call of 33,796 decode steps passes with 484,000 of 499,000 buffers held.

**Effort none is outside Kolibri's RL training mix** (exp_037 E12; vendor report p. 183/189). H1 tests Kolibri in a mode its vendor did not RL-train. K4-MED measures what that costs.

## Call profiles

The glue makes two model calls per question. Both are greedy, at temperature 0 and seed 42.

| Call | Settings | Repeat penalty (as G26's engine applies it on Ollama 0.40.2, W0r) |
|---|---|---|
| Scope gate | num_predict 4, num_ctx 1024 | none: Ollama's default is 1.0, and gemma4:26b's Modelfile sets none |
| Answer | exp_035's options: num_predict 512, num_ctx 4096 | 1.05 over the last 256 tokens of prompt + completion: prompt tokens count (probed), and the penalty is applied (probed) |

The MLX arms apply the same effective values, using mlx-lm's `make_repetition_penalty`. Its rule equals llama.cpp's penalties sampler at presence 0 and frequency 0 (test B4, 20 random trials).

**K4-MED reasoning cap**, set by a rule fixed here. Before any scored K4-MED row, K4-MED answers the DE dev set and the 10 `step0` questions; neither is an eval set. The cap is the smallest power of two that is at least twice the longest reasoning segment seen, clamped to [2,048, 16,384] tokens. The visible answer keeps the 512 cap. A row whose reasoning reaches the cap is scored as it stands (no answer), and truncations are counted.

## Sets (sealed 2026-10-10T08:28:27Z)

| Set | n | Languages | sha256 (private file) | Use |
|---|---|---|---|---|
| exp_035 v1 | 60 (57 in scope / 3 oos) | EN 31, PL 17, ES 12 | `efc16ab2…` (exp_035) | **P**, the primary set |
| German-buyer slice | 36 (33 / 3) | DE | `4b5c36f5767e…` | H3, D4, D4b |
| its English twins | 36 | EN | `f34afe8de2e0…` | D4, D4b |
| Beta-pattern probes | 10 | EN 7, ES 3 | `c2c664b7b677…` | Deterministic checks only (D6) |
| DE dev set | 12 | DE | `f1e6fef4aa12…` | Smoke test and K4-MED pilot; never scored or published |

**v2 is NOT RUN.** Andrei, 2026-10-10: P = v1. Declared before any arm runs.

**The German slice** was written, reviewed, flagged and language-checked by separate Fable instances. Each ran as a subagent of the main Claude Code session, never through an API, and each one's file opens are audited from its own tool calls:
- the grid: 6 rows per module; numeric 11, simple 10, multi 9, advice 3, oos 3;
- the writer, working from the English guide only. Lines that M0 flagged were withheld from it;
- code gates:
  - evidence verbatim;
  - citations present in the guide;
  - arithmetic recomputed;
  - character-trigram Jaccard < 0.45 against v1, v2, the 1,652 doc2query questions and the 377 training questions;
- an adversarial review: 29 rows ok, 7 fixed, 0 replaced;
- a key-rule gate against v1: 2 rows failed (each repeated a v1 headline rule) and were replaced by a fresh writer. Their own review fixed 1, and they passed flags, the language check and the gate. Final: 36/36 pass; 21 fresh even by the stricter amendment-10 criterion;
- substance flags, from gold plus retrieved context only: 21 of 33 in-scope rows have the key rule in context, 12 are retrieval misses (rule X1);
- an LLM German language check: 3 rewrites. **This is a limitation: no human native reader.**
- **Sealed by Andrei, 2026-10-10T08:28:27Z,** with every agent's blindness audit CLEAN or reviewed (15 roles).

Each German row's twin is a faithful English translation. The German row is retrieved and routed on its twin (`route_q`), while the German question is what the model and its scope gate receive. **exp_038 measures a German answerer, not German retrieval.**

**Probes, in abstract form:** 10 deterministic probes in four abstract categories, shaped by Andrei's beta-lesson summaries. The pattern sheet and the gazetteer are private; their sha256 are public.

## The German path (A12-DE)

The glue is vendored byte-identical from casasol `b1a648c` and never edited. The German path is installed by the harness and is additive:
- German strings are added under a `"de"` key, which the en/pl/es paths never read.
- For the duration of each German row only, the harness swaps in:
  - German alternatives in the answer-side regexes;
  - German citation folding;
  - German citation-stripping filler, in the case its preposition governs;
  - question-side regexes and retrieval that read the twin.
- Every original object is restored afterwards.

**R5:** with the German path installed and exercised, 1,060 EN, PL and ES items (A12's 60 v1 answers and 1,000 seeded synthetic strings) give output identical to a pristine import.

The German strings were drafted and then edited by two separate Fable instances (an LLM-only language check). They were frozen in R4 (answer-path tree `fd6b407c…`) before the slice writer started.

## Checks before any scored row

| Check | Result or rule |
|---|---|
| R1 | A12 through the vendored glue on the 0.33.0 side server reproduces exp_035's A12 v1 run **byte for byte, 60/60**, under Python 3.14.8 and 3.12.13. PASS 2026-10-10 |
| R2 | The vendored scorer re-scores that run to exp_035's published summary and every row's scores exactly. PASS |
| R3 | With the penalty processor at 1.0, the MLX driver reproduces exp_037's runner's greedy token ids on 5 answer and 5 gate prompts per MLX arm (the first 5 `step0` questions) |
| R4 | Answer-path tree frozen, `fd6b407c…` |
| R5 | EN/PL/ES unchanged with the German path installed. PASS |
| R6 | Each arm re-runs 5 pre-named v1 rows after its scored pass. Byte identity is reported per arm; a mismatch is recorded with its rate and does not void the run |
| R7 | On the mini, retrieval and post-processing are replayed on every MLX row's raw text. They must equal the mbp's final answer byte for byte |

## Scoring

**Stage 1, deterministic (code only, recorded before any judge call).** Per arm and per language:
- language compliance;
- plain text;
- out-of-scope declines;
- hand-off;
- numeric preservation;
- scripted citation-in-context;
- scope-gate decisions;
- failed calls;
- truncations;
- word counts.

The probes are checked by code: BP1 the scope gate says IN; BP2 no unsupported number, a reply within 120 s, and a hand-off; BP3 no unsupported euro figure; BP4 no near-miss place name against a private gazetteer.

**Stage 2, the judge.**
- **Who:** blind Fable instances (`claude-fable-5-1`), each a subagent of the main session.
- **The rubric:** exp_035's rubric chain (rules 1–18, A–E, (a)–(o)), mechanically redacted. Every row-tied example and quoted answer fragment is withheld, and build and model names are neutralised. sha256 before `64da983f…`, after `ceb80d5b…`.
- **Additions written before any output:**
  - **X1, retrieval misses:** a gold rule stated from the model's own weights on a row whose substance is not in context scores at most 1 and is flagged `from_weights`.
  - **X2, blind:** answers carry per-row random codes in a shuffled order.
  - **X3, bundles only:** each judge gets only its own block.
  - **X4:** German follows the PL/ES native-language convention.
  - **X5:** glue templates are judged for shape only.
- **Blocks:** rows are dealt, seeded, into blocks of about 9. A DE row and its twin never share a block. One judge instance scores every arm's answer for the rows in its block, on the absolute scale.
- **The audit:** a judge blindness audit lists the files each judge opened.
- **Fields:**
  - correctness 0/1/2;
  - usefulness 1–10;
  - unsupported claims;
  - fabricated citations kept;
  - language_ok and language_native;
  - tail degradation;
  - from_weights.
- **Harmful** means usefulness ≤ 1.

**Reliability (pre-registered).** A seeded 5 of the blocks are re-dealt to fresh judges. We report exact agreement and quadratic-weighted κ on correctness, and the H1 statistic on that subset under each judgment. If agreement is below 0.80, or the H1 verdict changes when the second judgment replaces the first, every row is judged twice and the mean is the scored value. That is a typed amendment, "judge protocol".

**Drift check.** A12's v1 answers are byte-identical to exp_035's (R1), so they are scored under both protocols, and exact agreement is reported. Arm comparisons are always within exp_038's protocol.

## Hypotheses

The statistic is d_i = correctness(K4) − correctness(G26) on identical rows. D̄ is its mean.

- **H1, substance against the incumbent (confirmatory).** K4 − G26 on P (v1, n = 60).
  - *Null:* D̄ ≤ 0.
  - *Test:* a paired bootstrap stratified by language, 10,000 replicates, seed sha256("exp038|H1"); p = (1 + #{mean ≤ 0})/(B + 1), p_rev = (1 + #{mean ≥ 0.20})/(B + 1).
  - *States:* CONFIRMED if p ≤ 0.05; REFUTED if not CONFIRMED and p_rev ≤ 0.05; INCONCLUSIVE otherwise.
  - Stated with C13: Gemma 4 rephrased part of Kolibri's English pre-training data, so the arms' outputs may be correlated.
- **H3, German-speaking buyers.** K4 − G26 on the 33 in-scope DE rows, by an exact paired sign-flip test (with the bootstrap beside it).
  - Tested at 0.05 only if H1 is CONFIRMED (fixed sequence). Otherwise it is an estimate with a 95 % CI, "not tested".
  - **H3 not CONFIRMED is not evidence against a German edge below about +0.4.**
- **H4, format floors** per arm and language (deterministic):
  - language 100 %;
  - plain text 100 %;
  - every out-of-scope row declined;
  - hand-off on 100 % of in-scope rows;
  - numeric preservation ≥ 98 %;
  - citation-in-context ≥ 94 %;
  - 0 fabricated citations kept;
  - 0 scope-gate false-OUT;
  - 0 failed calls.

  **Floor scope** (Andrei, 2026-10-10, "go" on the recommendation): the categorical floors (language, plain text, declines, hand-off, scope-gate false-OUT, failed calls) apply **per language**. The two ratio floors (numeric preservation ≥ 98 %, citation-in-context ≥ 94 %) apply **pooled per arm** across its languages, as exp_035 reported them. A12's per-language PL citation rate (5/6) was known when this was decided. A K4 miss of any floor in any language means "not drop-in in this glue" for that language.

**Margin δ = +0.20** (about 2.2 SE at n = 60).

**Headroom rule.** Under X1, the attainable ceiling C is computed from the sealed flags. If C − mean(G26) < 0.30, every H1 verdict carries "; ceiling-limited", and REFUTED reads "a +0.20 gain is not attainable on this set".

**Descriptive (no verdicts):**

| Item | What |
|---|---|
| D1 | H1 by language |
| D2 | H1 on substance rows only; correctness on miss rows; `from_weights` per arm |
| D3 | Harmful rows (h+/h−), native quality, truncations, word counts |
| D4 | The German penalty per arm: twin − DE |
| D4b | The interaction (K4_DE − G26_DE) − (K4_EN − G26_EN): mean, 95 % CI, sign-flip p |
| D5 | K4 − A12 on v1 |
| D6 | Probes; gate decisions; wall time against 120 s; R6 rate |
| D8 | K8 − K4 and K8 − G26 (model against 4-bit) |
| D9 | K4-MED − K4 and K4-MED − G26, plus wall time per call against 120 s |

**Power** (n = 60, σ_d 0.70): P(CONFIRMED) is 0.72 at +0.20 and 0.87 at +0.25. P(REFUTED) at a true 0 is 0.72.

## Verdict map (ordered; first match wins)

| # | Verdict | When |
|---|---|---|
| 0 | NOT RUN | A stop rule fired |
| 1 | "No gain of +0.20 (in the A12 glue)" | H1 REFUTED |
| 2 | "Earns a live-bot trial (evidence from the A12 glue only)" | H1 CONFIRMED, point ≥ +0.20, harm guard holds, every K4 floor met in EN, PL, ES and DE, and **E38-D1** CONFIRMED |
| 3 | "Better, not drop-in (<failing conditions>)" | H1 CONFIRMED and a floor, E38-D1 or the harm guard fails |
| 4 | "Better by less than the margin" | H1 CONFIRMED, point < +0.20, everything else met |
| 5 | "Not shown" | H1 INCONCLUSIVE, with the point estimate, its 95 % CI, and "a gain of +0.20 is neither shown nor ruled out at n = 60" |

**Harm guard:** h+ − h− ≤ 0, where h+ = rows harmful for K4 but not for G26. Every verdict line ends "; German: <H3 state>, interaction <D4b>". It also ends "; ceiling-limited" when the headroom rule fires.

**E38-D1, fit on a 64 GB node, measured on the mini.** This uses exp_037's own D1 bench, unchanged: K4 alone, 32k and 64k tokens of `pad_120k.txt` plus 512 greedy tokens, 3 reps, peak MLX memory. Ollama is stopped during the run. The K4 copy is checked against exp_037's fingerprint, and the host limit is asserted at 51.84 GiB.
- CONFIRMED if the median peak at 64k is ≤ 46.66 GiB;
- REFUTED if it is > 51.84 GiB;
- INCONCLUSIVE otherwise.

It is memory only. No K4 quality row runs on the mini, because the gate certified the M5 Max only.

## Stop rules and order

**Stop rules.** exp_038 is NOT RUN, recorded before any K4 row, if exp_037's gate record is not exit 0 (it is), or if E38-D1 is REFUTED. E38-D1 INCONCLUSIVE caps the verdict at 3.

**Order of events after this push:**
1. E38-D1.
2. R3, then the K4-MED pilot on the dev set, which fixes the cap.
3. The scored arms, in the order probes, v1, DE, twins. R6 runs after each arm.
4. R7.
5. Stage 1, recorded.
6. Stage 2, the judge.
7. The analysis.

Every run starts on Andrei's command. Aborted runs go to `aborted/` and are never deleted. After this push, the only changes allowed are typed amendments, `## Amendment k — <plan | set correction | adapter fix | judge protocol | diagnostics> (<UTC>)`. A diagnostics amendment commits its outcome-to-action table before anything runs.

## Runtime pins (release check 2026-10-10)

- **MLX, mlx-metal and mlx-lm:** 0.32.3, 0.32.3 and 0.32.0, the latest on PyPI. The offset hazard is still on mlx-lm main.
- **Ollama:** 0.40.2 is production. 0.33.0 runs as the A12 side server.
- **Holds:** macOS, MLX and mlx-lm updates on the mbp are held until the last mbp cell.

## Rights

- **S1 and S2** (Andrei's study guides) are used for writing and judging only through Fable subagents of the main Claude Code session, never through the Anthropic API (Andrei, 2026-10-10).
- **M0:** a local script compared 12-word shingles of S1/S2 with the course materials (S3). It found 41 hits in 13 line ranges. Those lines never reach a writer, and judge contexts show "[withheld]" in their place.
- **S3** (course materials) and **S4** (bot conversation logs) are never opened by any agent and never sent to any model.

## Published and private

- **Public here:**
  - this file;
  - the kit (MLX backend, harness skeleton, R checks, analysis and verdict code, tests that need no private text);
  - per-row scores by id;
  - per-arm summaries;
  - a structured judge export (scores, flags, the code-to-arm map after unblinding);
  - the blindness audits (file lists);
  - the sha256 of every private file.
- **Private:**
  - the glue's prompts and strings, including every German string;
  - the sets and golds;
  - raw outputs and contexts;
  - judge notes;
  - the rights record.

A leak check (exp_037's pattern) runs before every push, with every private string as forbidden content. Only the owner can re-run exp_038. The public can re-derive every verdict from the per-row scores.

## Known-values ledger (everything seen before this push)

**exp_035:**
- A12 v1 correctness mean 1.217; gemma-3-4b 1.233;
- σ_d 0.68–0.70 between 4B glue arms;
- A12 6.8 s per question;
- 38 of 60 v1 rows have substance in context.

**exp_036 and exp_037:**
- the vendor's MMLU-ProX-Lite gaps;
- exp_037 H1 0.8455 (K4 against G4 decode);
- exp_037 D1 43.37 GiB on the mbp;
- gate run 3: K4 mean KL to the fp32 reference 0.131 (T1–8) against K8's 0.041;
- C1: 6/100 answers flipped at batch 8.

**This build:**
- R1 and R2 PASS;
- W0r: G26 averages 11.4 s and at most 13.3 s per `step0` question;
- stage 1 on A12's v1 answers (exp_035's run): every floor met in EN and ES; **PL citation-in-context 5/6 = 0.833 per language**; pooled 100/103 = 0.971;
- the A12 DE dev-set smoke: 12/12 German, no DE-path defect;
- the mini's Metal limit 51.84 GiB.

**Disclosure.** While adapting gate scripts, the main session read exp_035's `eval_v2/arith_check.py`, which embeds v2 gold arithmetic. v2 is not run here, and the main session writes no items.

## Predictions (designer's priors)

| Item | Prediction |
|---|---|
| H1 | D̄ ≈ 0 (−0.10 to +0.15), from the vendor's MMLU-ProX-Lite gaps and C13; effort none may push it lower |
| H3, D4b | Small and uncertain; D4b ≈ 0. Any German edge would show in fluency (usefulness, native) rather than correctness |
| H4 | K4 misses the language floor on ES and/or PL (Kolibri is trained on EN and DE only). G26 holds every floor |
| D9 | K4-MED ≥ K4 on correctness; wall time per call above 120 s on some rows |
| E38-D1 | CONFIRMED, with a 64k peak ≈ 43.4 GiB, as on the mbp |

## Limitations

- **Effort none** is outside Kolibri's RL mix.
- **The German check** is LLM-only, with no human native reader.
- **Retrieval routes on the English twin,** which idealises the German front end.
- **The v1 glue was tuned on Qwen and Gemma-family outputs** (exp_035), which favours the incumbents.
- **Teacher confound C13:** Gemma 4 and Qwen3.8 are partly Kolibri's teachers.
- **One judge family,** blind and audited.
- **G26 is not the live responder.**
- **The fit is measured on the mini; quality only on the mbp.**
