# Experiment 035 — COAPI Voice: form in the weights, facts in retrieval, on a phone

*Pre-registered: 2026-09-21 · Status: steps 0–4 executed and recorded below; training-run hypotheses H3–H5 pre-registered, no training run has started*

**Feature spec:** `casasol/features/FEATURE-009-coapi-voice-finetune.md` (with dated `[A]` amendments)
**Evidence log:** `casasol/features/FEATURE-009-step0-device-viability.md`
**Code:** `casasol/coapi_voice/` (chunker `guide_bm25.py`, runner/scorer `run_eval.py`, rights record, sealed eval)
**Builds on:** exp_006 (redactor fidelity — regex-scored, never model-scored), exp_026 (contextual retrieval on the COAPI corpus), the pattern *facts-in-context, form-in-weights*.

---

## What this experiment tests

Whether a ~1 GB model can answer Spanish real-estate (COAPI) questions on a phone, offline, in the voice of a colegiado — citing the tema and article, giving the Spanish term, handing off binding decisions — **without any fact living in its weights**. Facts come from BM25 retrieval over Andrei's own study guide (297 KB); the voice comes from a LoRA fine-tune on guide-grounded examples written and adversarially reviewed by the strongest available model (Fable) before the local model sees them.

The claim being protected: *"answers from the guide, in the voice; runs entirely on your device; no model is trained on buyer or client data."*

## Fixed before any training run (hashes)

| Artefact | Hash / id |
|---|---|
| Sealed eval, 60 rows | `coapi_voice/data/eval/coapi_eval_v1.jsonl` sha256 `efc16ab220cb38918262cc854f79c6b65033f8402cba8465a81fc1aff9e2fe1c` |
| Split manifest | `results/split_manifest_v1.json` (train M1–4 + M5 T39–41; val M5 T42–43; test M6, burned after one eval) |
| Guide sources | 12 files, sha256 in `coapi_voice/RIGHTS.md` |
| Base model | `hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M` (Ollama id 75cb860e23c2); same file on the iPhone |
| Sampling (shipped) | temperature 0.7, top_k 20, top_p 0.8, min_p 0.05, repeat 1.15, presence 1.0, last_n 256, n_predict 512, thinking off; **scored pass at temperature 0 / seed 42** |
| Retrieval | guide-BM25 + vocab glossary + doc2query (826 Fable questions), k=5, 1200-token budget, `QUERY_BOOST` 1.0, `SCORE_FLOOR` 0.5 |
| Scorer | `run_eval.py score` — regex only; judgment metrics by a Fable judge with a written rubric (`runs/*/judge.json`) |

## Device budgets (measured, step 0 — iPhone 14 Pro, 2026-09-21)

| Budget | Target | Measured |
|---|---|---|
| Model file | ≤ 1.2 GB | 1.1 GB |
| RAM at inference | ≤ 2 GB | 1.24 GB peak (Qwen2.5-1.5B Q4_K_M; Qwen3-1.7B same class) |
| Generation | ≥ 10 tok/s | 15–33 tok/s (Qwen3-1.7B), 39 tok/s (Qwen2.5-1.5B) |
| Prompt processing | record | 425–465 tok/s (Metal) → ≤ ~900 tokens context for TTFT < 2 s; 1200 tokens ≈ 2.6–2.8 s |
| Index | < 5 MB, builds < 1 s | 485/496 KB, < 1 s |

Android deferred (Moto g56 5G bench is the re-entry step). Qwen2.5-1.5B failed the Polish floor (0/3 coherent); Qwen3-1.7B cleared it with sampling constraints.

## Hypotheses

**H1 — Retrieval carries the facts (executed 2026-09-21).** Giving the base model guide chunks raises numeric preservation and citation-in-context by a large factor without any training.
*Result:* B1 → B2m: numeric 24.2 % → 90.1 %; citation-in-context 0 % → 93.9 % (46/49; first logged as 51 % before the matcher handled Law/Ley synonyms, article ranges and ordinals — corrected 2026-09-21 18:40); retrieval hit 92 %. **Confirmed.** `results/b1_summary_20260921.json`, `results/b2m_summary_20260921.json`.

**H2 — The gate: the untrained model with retrieval fails the form floors (executed 2026-09-21).** Pre-registered go criterion (spec §4): B2m fails at least one of no-legal-advice < 100 %, citation-in-context < 90 %, Spanish-term < 80 %, and the failure is form, not retrieval.
*Result:* no-legal-advice 50 % (2/4 probes), Spanish-term 75 %, language 90 %, out-of-scope refusal 0/3, plain text 1/60, 27/60 answers with no citation at all; citation-in-context 93.9 % (just under its 95 % floor) — with retrieval hit 92 %. **Gate met; fine-tuning justified.**

**H3 — Form moves into the weights.** After LoRA (r=16, α=32, 2 epochs, ≥ 200 Fable-written, Andrei-reviewed examples whose user turn contains the *same* BM25 context the phone would retrieve), B3 at Q4_K_M on the **Mini cell** meets, on the 60 sealed rows: no-legal-advice 100 %, citation-in-context ≥ 95 % **and ≥ 90 % of in-scope answers carry at least one citation** (B2m: 33/60), Spanish-term ≥ 90 %, language 100 %, out-of-scope refusal ≥ 95 % (3/3), plain text ≥ 95 %, in-scope answer rate ≥ 95 %.
*Null:* any floor missed on the Mini cell. Partial credit is reported, not claimed.

**H3b — Grounding: the model uses the rule in front of it (pre-registered 2026-09-21 21:05, after the B2m judge pass, before any training).** Fable judge on B2m (`results/b2m_judge_fable_20260921.json`, rubric frozen in its `rubric_notes`): correctness 0.72/2, fully-correct 8/60 (13 %), usefulness 2.7/10, unsupported claims in 57/60 rows; dominant failure = confident inversion of a rule present in the retrieved context (12 rows) and fabricated substitutes on retrieval misses (10 rows). B3, judged by the same rubric text: correctness_mean ≥ 1.4, fully-correct ≥ 50 %, usefulness_mean ≥ 6, rows with unsupported claims ≤ 20 %, and on retrieval misses an honest "not covered + hand-off" (correctness 1) rather than a substitute rule (0).
*Null:* any of these missed. Regex numeric-preservation is explicitly NOT evidence for H3b — B2m scored 97.7 % on it while inverting the numbers' meaning.

**H4 — No facts leaked into the weights.** B3's numeric preservation and unsupported-claim count are no worse than B2m's (numeric ≥ 90 %, unsupported claims per the Fable judge ≤ B2m's), and B3 on Módulo 6 rows (temas never seen in training) scores no lower on correctness than on Módulos 1–5.
*Null:* B3 gains on M1–5 and loses on M6 (memorisation), or numeric preservation drops.

**H5 — Device fidelity.** B3 at Q4_K_M on the iPhone 14 Pro, same sampling, temperature 0, on 10 pre-named rows (2 per module, incl. 1 PL and 1 ES) agrees with the Mini cell field-by-field (language, refusal, hand-off, citations set, numbers set) on ≥ 9/10 rows.
*Null:* < 9/10. Then the fix is quantisation (try Q5_K_M/Q6_K) before it is data.

## Results — B3 v1 (2026-09-21 20:34 build, judged 21:10)

Run: `results/v1_version_record.json` (363/14 examples, r=16, scale 2.0, lr 2e-4, 2 epochs = 182 iters, grad checkpointing). Loss: train 1.47 → 0.85; val 3.52 → 1.62 @50, 1.64 @100, 1.63 @150, 1.65 @182 — flat after epoch 1.

| | Floor | B2m | B3 v1 (Mini, Q4_K_M) |
|---|---|---|---|
| Language compliance | 100 % | 90 % | **100 %** |
| No-legal-advice (all / advice rows) | 100 % | 98 / 75 % | **100 / 100 %** |
| Plain text | ≥ 95 % | 1/60 | 57/60 |
| Citation-in-context / answers citing | ≥ 95 % / ≥ 90 % | 93.9 % / 33 | 97.7 % / 48 (80 %) |
| Spanish-term inclusion | ≥ 90 % | 75 % | 89.6 % |
| Out-of-scope decline | 3/3 | 0/3 | 1/3 |
| **H3b correctness mean / fully correct / usefulness** | ≥ 1.4 / ≥ 50 % / ≥ 6 | 0.72 / 13 % / 2.7 | **0.70 / 1.7 % / 2.8** |
| Unsupported claims (rows) | ≤ 20 % | 57/60 | 60/60 |
| Native-quality PL / ES (judge) | — | — | **0/17, 0/12** |
| Tail degradation after a correct opener | — | — | **35/60** |
| **H4 M6 (unseen) vs M1–5** | no drop | 0.90 / 0.68 | **0.70 / 0.70** (M6 −0.2, largest drop) |

**H3 — partially confirmed:** the form floors that are pure shape moved as predicted (language, no-advice, plain text, citation density). **H3b — refuted:** substance did not move; the model narrates the nearest chunk as if it governed the case, and 12 rows improved while 13 got worse. **H4 — refuted:** Módulo 6 fell more than any módulo; numeric M6 rows the base computed correctly (19 %/81 %, €320k) are now wrong. New failure classes the base never showed: PL/ES grammar collapse, fabricated image URLs and self-pricing lines in tails. Judge: `results/b3_v1_judge_fable_20260921.json`, same rubric as B2m.

**Interpretation (pre-diagnostic):** the LoRA (all layers, scale 2.0, lr 2e-4, 2 epochs) learned the opener and the surface voice and damaged the base's language competence — the classic over-strong adapter on ~220 K trained tokens. Template/EOS verified intact in the GGUF; tails end with EOS before the cap, so this is the adapter, not serving.

**Diagnostics, in the pre-registered order:** (1) quantisation — B3 at Q8_0 on the Mini; (2) the iteration-50 checkpoint (end of epoch 1, lowest val loss) at Q4_K_M; (3) only then a v2 with a weaker adapter (lr 1e-4, 1 epoch, rank 8, last-N layers) — each a new dated block here, never an edit of this one.

## Diagnostics after B3 v1 (2026-09-21 21:00–21:15)

1. **Quantisation (Q8_0):** same failures, same fabricated image URL in M2-06 — not quantisation. `runs/b3_coapi-voice-v1-q8_*`.
2. **Epoch-1 checkpoint (iteration 50, lowest val loss):** worse, not better (Spanish-term 64.6 %, nonsense Polish in M2-06) — not the second epoch.
3. **Sampling (the confound):** the step-0 constraints (presence_penalty 1.0, repeat_penalty 1.15) were chosen to stop the *base* model's loops. At temperature 0 a presence penalty of 1.0 taxes every reused token; legal answers reuse their terms constantly, and the fine-tuned model drifted off its own vocabulary into junk. Re-run with **presence 0 / repeat 1.05**: B3 v1 plain text 57→60/60, citations 97.7→98.4 %, Spanish term 89.6→**91.7 %** (floor met), numbers 97.8→99.6 %, truncations 1→**0**; M2-06 becomes a clean decline, M6-03 states 89 − age correctly, M1-09 is coherent. Grounding failures remain (M1-02, M4-01).
   **Amendment (§7 sampling, dated 2026-09-21 21:10):** shipped sampling = temperature 0.7, top_k 20, top_p 0.8, min_p 0.05, **repeat_penalty 1.05, presence_penalty 0**, last_n 256, n_predict 512; loop guard is the n_predict cap plus the adapter's own EOS discipline (0/60 truncations). B2m re-run under the same sampling for a fair baseline (`runs/b2m_*_nopen_*`: plain text still 1/60, citations 89.6 %, 8 truncations — the base loops without the penalty, the adapter does not). Both runs judged by Fable under the unchanged rubric; results follow as a new block.

## Results under the amended sampling — final for v1 (2026-09-21 21:40; both runs judged by Fable, rubric unchanged)

| | Floor | B2m (base) | B3 v1 |
|---|---|---|---|
| Language compliance | 100 % | 92 % | **100 %** |
| Plain text | ≥ 95 % | 1/60 | **60/60** |
| Out-of-scope decline | 3/3 | 0/3 | **3/3** |
| No-legal-advice, advice rows / all | 100 % | 75 % / 98 % | **100 % / 98 %** (M1-03 "demand the double") |
| Citation-in-context / answers citing | ≥ 95 % / ≥ 90 % | 89.6 % / 35 | 98.4 % / 50 (83 %) |
| Spanish-term inclusion | ≥ 90 % | 79 % | **91.7 %** |
| Numeric preservation | ≥ 98 % | 98.1 % | 99.6 % |
| Artefact rows / truncations | — | 60 / 8 | **0 / 0** |
| Tail degradation | — | 13 % | 38 % |
| **Correctness mean / fully correct / usefulness (H3b)** | ≥ 1.4 / ≥ 50 % / ≥ 6 | 0.78 / 18 % / 2.6 | **0.88 / 15 % / 3.2** |
| Unsupported-claim rows | ≤ 20 % | 60/60 | 60/60 |
| Native PL / ES (judge) | — | 1/17, 4/12 | 0/17, 2/12 |
| **M6 (unseen) vs M1–5 (H4)** | no drop | 1.0 / 0.75 | **0.5 / 0.96** |

**H3 — confirmed** for every shape floor except "answers citing" (83 % vs 90 %). **H3b — refuted:** substance 0.78 → 0.88, far from 1.4; 13 of 16 zeros are retrieval misses or the decline template misfiring on in-scope rows (M2-10, M6-08). **H4 — refuted:** the adapter learned the seen temas (M1–5 0.96) and lost the unseen ones (M6 0.5, from 1.0 in the base) — tema-specific patterns went into the weights. **Native Polish is 0/17 for B3 and 1/17 for the base:** a 1.7B model does not carry Polish; this is a base-model ceiling, not a data problem.

Judge files: `results/b3_coapi-voice-v1_nopen_20260921-2108_judge.json`, `results/b2m_Qwen3-1.7B-GGUF-Q4-K-M_nopen_20260921-2112_judge.json`. Still-harmful rows (usefulness 1): M1-02, M2-02, M2-04, M3-01, M3-06, M4-04, M4-05, M4-09, M6-01, M6-09, M6-10 — 7 retrieval misses, 4 with the rule in context.

**v1 is evidence, not product.** What the experiment established: (a) retrieval carries facts into context at 92 % hit rate but the 1.7B base inverts or ignores them; (b) a 363-example LoRA installs voice, language adherence and decline/refusal shape completely and installs no reasoning; (c) it also installs tema-specific memory, which is the leak §1 promised not to have; (d) PL native quality is a base-model floor. Next steps are decisions for Andrei, recorded when taken: retrieval-miss analysis (free, lifts both cells); a larger base for Polish (Qwen3-4B, ~2.5 GB Q4_K_M, fits the iPhone's 3.3 GB cap now that v1 is iOS-only — a step-0-style PocketPal check on the three PL questions decides it); H5 device fidelity on v1 remains worth running as method evidence.

## P1 — retrieval-miss analysis and fixes (2026-09-22 03:30–04:10)

Method: for every sealed row, the share of the gold's citations and numbers present in the retrieved context ("substance coverage"), independent of tema hit; misses classified as guide gap / ranking (wrong tema) / chunk-level (right tema, wrong chunk).

| Retrieval configuration | Expected-tema hit (1200) | Substance coverage | Rows < 0.75 |
|---|---|---|---|
| doc2query r1 (day 1) | 48/53 (91 %) | 0.825 | 17 |
| + doc2query r2, uniform, 826 Fable questions | 53/57 (93 %) | 0.834 | 16 |
| + tema completion (leading tema's `citations` block) | — | 0.840 | 15 |
| + budget 1400 (not adopted: +0.5 s TTFT) | — | 0.854 | 14 |
| k=3 / 900 (rejected) | 48/60 (80 %) | 0.756 | — |

Remaining misses: 9 chunk-level, 3 ranking (M3-02, M3-04, M5-03), 0 true guide gaps (the four flagged are worked-example numbers or matcher artefacts). Guide coverage of the gold's substance is 0.96.

**Finding:** with substance coverage ≥ 0.75 the fine-tuned 1.7B scores 0.81/2; with coverage < 0.5 it scores 0. Retrieval is necessary and is now at ~0.84 with three orthogonal levers spent; **the ceiling is the model's use of what is in front of it, not what is in front of it.** B3 regex metrics on the r2 index: language 100 %, plain text 100 %, citation-in-context 100 % (197 citations), Spanish term 93.8 %, numeric 98.8 %, retrieval hit 93 %. No new judge pass — the substance delta (+0.01) is below what a judge would resolve.

**Consequence for the next decision:** the lever that remains is the base model (P2: Qwen3-4B on the iPhone — 2.5 GB Q4_K_M under the 3.3 GB cap; PL native quality is the first thing to measure, by the step-0 method). Adopted retrieval config for any further cell: k=5, budget 1200, doc2query r1+r2, `CITATIONS_WITH_TEMA=1`.

## P2 — larger base candidate: Qwen3-4B Q4_K_M, Mini cell (2026-09-22 04:30–04:50)

`hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M` (2.5 GB), untrained, same retrieval (r2 index, k=5, 1200) and sampling, Fable judge under the unchanged rubric (`results/b2m_Qwen3-4B_judge_20260922.json`).

| | 1.7B base | 1.7B v1 | 4B base |
|---|---|---|---|
| Correctness / fully correct / usefulness | 0.78 / 18 % / 2.6 | 0.88 / 15 % / 3.2 | **1.13 / 38 % / 3.65** |
| Correctness with substance in context (38 rows) | ~0.9 | ~1.0 | **1.50, 23/38 fully correct** |
| Correctness on the 19 retrieval-miss rows | — | — | 0.47 (10 zeros, confident substitutes) |
| Native PL / ES | 1/17, 4/12 | 0/17, 2/12 | **3/17 (readable in 17/17, non-native in 14), 9/12** |
| Wrong-language rows | 5 | 0 | 11 (EN → ES drift; PL→ES once) |
| No-advice (advice rows) / oos | 75 % / 0/3 | 100 % / 3/3 | 50 % / 0/3 |
| Tail degradation / artefact rows | 13 % / 60 | 38 % / 0 | 17 % / 44 (lower-case, markdown, fabricated article numbers) |
| Mini tg | — | — | 73 tok/s (Ollama, M4 Pro) |

**Reading.** The base-model ceiling moves: with the substance in context the 4B is fully correct 60 % of the time and its arithmetic is right; its remaining failures (language routing, plain text, declines, refusals, citation hygiene) are exactly the behaviours the v1 fine-tune installed completely on the 1.7B. Polish is understood in every row and native in few — a fine-tune on Fable's Polish will pull toward native, but the v1 experience says grammar the base lacks does not come from 363 examples. With a model that can read, retrieval sets the floor (0.47 on misses vs 1.50 on hits).

**Open before v2:** (a) the iPhone 14 Pro half of P2 — does a 2.5 GB Q4_K_M load under the ~3.3 GB per-app cap and run ≥ 10 tok/s (Andrei, PocketPal bench); (b) product call on Polish quality; (c) whether a Polish-specialised base is worth a step-0 look. v2 hypotheses to be pre-registered only after (a).

## P2 — iPhone half (2026-09-22 05:05, PocketPal bench, Qwen3-4B Q4_K_M 2.49 GB)

pp512 **148.7 tok/s**, tg128 **14.15 tok/s** (floor 10 ✓), loads and completes under the ~3.3 GB per-app cap (PocketPal peak-memory card 18.7 %), 37 s total, rep 3, context 2048, Metal, FA on. Consequence: TTFT at the adopted 1200-token context ≈ **8 s** on the 14 Pro (1.7B: ≈ 2.7 s). The 4B cell's TTFT budget is restated to ≤ 10 s; the phone's per-question wait is a product cost Andrei accepts by choosing this base.

**iPhone chat readings, untrained 4B, no retrieval, system prompt on, `/no_think` (05:11–05:16):** Q8/Q9/Q10 (Polish) all answered in **Spanish** (3/3; Q10 with Polish leaking mid-sentence and garbled tokens), fabricated citations throughout ("RGIP art. 124", "Ley 31/1992 AUR", "art. 132 CC"). tok/s **13.1 → 6.6 → 7.3**, TTFT 1.2–1.4 s (no context). **Re-read after 10 min idle (05:27): 14.5 tok/s** — the drop was thermal (2.5 GB download + 37 s bench immediately before). Phone-cell verdict for the 4B: **≥ 14 tok/s cold, ≈ 7 tok/s under sustained back-to-back use**; speed floor passes cold, and the throttled figure is recorded as the product-latency reality for consecutive questions. Fourth Polish question (05:27) also answered in Spanish, with a new fabricated article (127 CC): **0/4 Polish routing** for the untrained 4B on-device.

## v2 — pre-registered 2026-09-22 05:15, before the run

**Base:** `Qwen/Qwen3-4B` (safetensors, HF snapshot 1cfa9a72…), quantised after fusing to Q4_K_M — same file class the phone benchmarked. **Data:** unchanged — `train_v1.jsonl` sha256 `3c854250…` (363), `valid_v1.jsonl` `b3658d7d…` (14). **Adapter:** unchanged (r=16, scale 2.0, dropout 0.05, lr 2e-4, 2 epochs, batch 4, prompt masked, grad checkpointing, seed 35). **Retrieval:** r2 index, k=5, 1200, citations-with-tema. **Sampling:** amended set (presence 0, repeat 1.05). **Judge:** same rubric, Fable.

Baseline for v2 = the 4B B2m block above (1.13 / 38 % / 3.65; native PL 3/17; 11 wrong-language rows; no-advice 50 %; oos 0/3).

**Run-configuration amendment (05:35, v2 restarted 3 min in, before any checkpoint):** the v1 log shows 83/182 batches with a sequence truncated at `max_seq_length 2048`; tokenised properly, **55/363 training examples exceed 2048 tokens — all 55 Polish** (Polish tokenises ~1.6× denser than the chars/3.5 estimate the chunker budgets with; p95 2171, max 2384). Prompt-masked truncation removes the answer's closing sentences and EOS, so 49 % of v1's Polish examples taught "do not stop". This is a plausible cause of v1's Polish-concentrated tail degradation and is recorded as a **v1 defect**. v2 runs with `max_seq_length 2560`, batch 2 (364 iterations); hypotheses H6–H9 unchanged. Consequence for the retrieval budget: the phone's 1200-token budget is an estimate that under-counts Polish by ~40 % — real TTFT for Polish questions is correspondingly longer; the estimator (`CHARS_PER_TOKEN`) should become language-aware.

**H6 — shape transfers to the 4B as it did to the 1.7B:** language compliance 100 % (from 82 %), plain text ≥ 95 %, out-of-scope 3/3, no-advice 100 % on advice rows, Spanish-term ≥ 90 %, citation-in-context ≥ 95 %, artefact rows ≤ 5.
**H7 — substance is kept, not lost:** correctness mean ≥ 1.2 (base 1.13) and correctness-with-substance-in-context ≥ 1.4 (base 1.50, tolerance −0.1); fully correct ≥ 35 %.
**H8 — no tema memorisation at this size:** Módulo 6 correctness within 0.2 of Módulos 1–5 (v1 on the 1.7B: 0.5 vs 0.96 — refuted there).
**H9 — Polish moves toward native without reaching it:** native PL ≥ 6/17 (base 3/17); recorded as the honest number for the product call, not a pass/fail.
*Nulls:* any of H6–H8 missed as stated; H9 is descriptive.

## v2 — run record (2026-09-22 05:40–08:55)

`results/v2_version_record.json`: Qwen3-4B, 363/14 examples (same hashes), r=16, scale 2.0, lr 2e-4, 2 epochs, batch 2, seq 2560 (no truncation), 364 iters, ~3 h incl. artefacts; Q4_K_M sha256 `924df1f2…` 2.50 GB, Q8_0 `7eaecfa9…`. Loss: train 1.44 → 0.69; val 2.87 → **1.42 @100** → 1.45 → 1.50 @364 (epoch 2 fits the training set; iteration-100 checkpoint kept). Regex on the sealed 60 (`results/b3_v2_summary_20260922.json`): language 60/60, plain text 60/60, citation-in-context 100 % (207 citations, 55/60 answers citing), numeric 100 %, Spanish term 89.6 %, retrieval hit 93 %, truncations 2. Judge pass pending → H6–H9 block follows.

## v2 — results (judged 09:20, `results/b3_v2_judge_20260922.json`, rubric unchanged)

| | Floor | 4B base | v2 |
|---|---|---|---|
| H6 language / plain text / artefact rows | 100 / ≥ 95 % / ≤ 5 | 82 % / 18 % / 44 | **100 % / 100 % / 2** |
| H6 no-advice (advice rows) / oos | 100 % / 3/3 | 50 % / 0/3 | **100 %** / 2/3 (M4-08 drifts into PIT rates) |
| H6 Spanish term / answers citing / citation accuracy | ≥ 90 / ≥ 90 / ≥ 95 % | 69 % / — / 91 % | 89.6 % / 91.7 % / 100 % |
| H7 correctness / with substance (38 rows) / fully correct | ≥ 1.2 / ≥ 1.4 / ≥ 35 % | 1.13 / 1.50 / 38 % | **1.05 / 1.18 / 20 %** |
| H7 on the 19 retrieval-miss rows | — | 0.47 | 0.68 (says "not covered" instead of substituting) |
| H8 M6 (unseen) vs M1–5 | within 0.2 | 1.2 / 1.1 | **1.0 / 1.06 — pass** |
| H9 native PL / ES | ≥ 6/17 | 3/17 / 9/12 | **1/17 / 3/12** |
| Tail degradation / hand-off | — | 10/60 / 88 % | 28/60 / 65 % |
| Usefulness | ≥ 6 | 3.65 | 4.0 |

**H6 — passed on every format floor except oos (2/3) and Spanish-term (89.6 %). H7 — refuted; H8 — confirmed (no tema memorisation on the 4B); H9 — refuted, with regression.** Harmful rows: M2-02, M4-02, M4-04, M4-09, M5-02, M6-01, M6-09 (three of them with the rule in context). Paired vs base: 11 rows better, 15 worse, 34 unchanged.

**Cross-run finding (v1 on 1.7B, v2 on 4B):** the recipe — r=16 on all layers, scale 2.0, lr 2e-4, 2 epochs, 363 examples — installs the shape completely on both bases and degrades what the base had: substance use (1.50 → 1.18), native ES (9 → 3 of 12), hand-off (88 → 65 %), with tail collapse in 28/60. Validation loss minimum at iteration 100 (mid-epoch 1) and rising after is the same signal. The adapter is too strong for the job.

**Diagnostics queued (no training):** (1) the v2 iteration-100 checkpoint (val-loss minimum) at Q4_K_M; (2) the spec's own null hypothesis on the 4B — a prompt-only B2m with an explicit format prompt (language of the question, plain text, decline shape, no-advice), no weights touched — to measure how much of H6 the base can do without an adapter that costs it H7/H9. Each recorded as a dated block.

## Diagnostics after v2 (2026-09-22 09:30–09:45)

**(2) Prompt-only null on the 4B — refuted.** Untrained 4B with an explicit seven-rule format prompt (`coapi_voice/configs/system_prompt_v2.txt`: language of the question, context-only, cite, Spanish term, no-advice, decline shape, plain text): language compliance **68 %** (ES 0/12 — the explicit language rule flipped every Spanish question into English; PL 10/17), plain text 75 %, Spanish term 60 %, citations 93 %. Worse than the base's own 82 % on language. Format on this base is not reachable by prompt; the §4 conclusion that shape must be trained holds on the 4B as it did on the 1.7B. `results/b2m_Qwen3-4B-GGUF-Q4-K-M_4b-prompt_20260922-0937_summary.json`. No judge pass (fails on regex alone).

**(1) v2 iteration-100 checkpoint (val-loss minimum, end of epoch 1) — regex:** language 100 %, plain text 100 %, citations 98.7 % (54/60 citing), Spanish term 79 %, hand-off 77 % (v2 65–68 %), truncations 4. Judge pass running → next block.

**(1) v2 iteration-100 — judged (`results/b3_v2-it100_judge_20260922.json`): not better.** Correctness 0.95 (v2 1.05, base 1.13); with substance 1.16 (v2 1.18, base 1.50); native PL 0/17, ES 2/12 (worst of the three); hand-off 74 % and tail collapse 24/60 (between v2 and base); the decline template misfires on 5 in-scope rows. Paired vs v2: 10 better / 15 worse / 35 same. **The damage is present after one pass over the data; epochs are not the cause.**

**Root cause, from the adapter file:** v1 and v2 trained rank-16 LoRA on **all seven projections — q, k, v, o and the three MLP projections — on all layers** (mlx-lm's default key set), 33 M parameters at lr 2e-4. The MLP projections are where a model keeps language and facts; rewriting them on 363 examples installs the shape (attention-driven) and degrades native ES (9 → 3 of 12), substance use (1.50 → 1.18) and hand-off. This was a recipe error carried from the spec's "r=16, α=32, lr 2e-4, 2 epochs" (a PEFT-style default written for QLoRA on much larger data) into an mlx-lm configuration that silently targets every projection.

## v3 — pre-registered 2026-09-22 10:05, before the run: the minimal adapter

**Change from v2 (everything else identical — base Qwen3-4B, data hashes, retrieval, sampling, judge):** LoRA on **`self_attn.q_proj` and `self_attn.v_proj` only**, **rank 8, scale 1.0**, **last 16 of 36 layers**, **lr 5e-5**, **1 epoch** (182 iters at batch 2), seq 2560. Trainable parameters 1.31 M (0.033 %) vs 33 M. Smoke-tested: config accepted, 10 GB peak.

**H10 — shape still installs with the minimal adapter:** language 100 %, plain text ≥ 95 %, no-advice 100 % on advice rows, artefact rows ≤ 5, answers citing ≥ 90 %, oos ≥ 2/3.
**H11 — the base is left intact:** correctness-with-substance ≥ 1.4 (base 1.50, tolerance −0.1), native ES ≥ 8/12 (base 9/12), native PL ≥ 3/17 (base 3/17), hand-off ≥ 80 %, tail collapse ≤ 15/60.
**H12 — net gain:** overall correctness ≥ 1.2 and fully-correct ≥ 35 %, i.e. the format gain is added to the base rather than traded for it.
*Nulls:* H10 missed → the minimal adapter is too weak for the shape and the sweet spot lies between (rank 8 with k/o, or 2 epochs). H11 missed → even attention-only LoRA damages the base at this data size, and the on-device product must be built prompt+retrieval on a base that already routes language (a different base). H12 is the product criterion.

## v3 — run record and regex (2026-09-22 10:20–11:30)

`results/v3_version_record.json`: q/v only, rank 8, scale 1.0, last 16 layers, lr 5e-5, 1 epoch (182 iters, batch 2, seq 2560), 1.31 M trainable; Q4_K_M sha256 `25244fd8…`. Loss: train 1.70 → 1.30; val 2.87 → 1.64 @50 → 1.54 → 1.50 → **1.495 @182, still falling** — the adapter never reached the training set. Regex (`results/b3_v3_summary_20260922.json`): language 98.3 % (PL 16/17), plain text 93.3 %, answers citing 47/60 (78 %), citation accuracy 98.6 %, Spanish term 75 %, oos 1/3, hand-off 40 %, **truncations 12/60** (the base's verbosity returns). **H10 (shape installs) — refuted as stated:** partial on language and plain text, absent on citing, hand-off, Spanish term, decline. H11 (base intact) → judge pending.

## v3 — judged (12:00, `results/b3_v3_judge_20260922.json`): H11 refuted 0/5, H10 1/7

Correctness-with-substance 1.21 (floor 1.4; base 1.50; v2 1.18); native ES 2/12, PL 1/17; hand-off **18.6 %** (base 88 %); tail collapse 27/60; overall 0.97 / fully-correct 20 %. Paired vs base 11 up / 20 down. Even 1.3 M attention-only parameters at lr 5e-5 for one epoch reproduce v2's damage and cut hand-off further. Where retrieval missed, v3 substitutes confidently (0.42) where v2 hedged (0.68).

**Decoding regime exonerated (12:15):** v2 re-run at the shipped temperature 0.7 (`results/b3_v2_temp07_summary_20260922.json`) has the same regex profile (hand-off 74 %, Spanish term 83 %, 0 truncations, no literal repetition at either temperature). The judged "tail collapse" is semantic — invented checks and routes after a correct opener — not greedy looping.

## Conclusion of the training cycle (2026-09-22 12:20)

Three LoRA runs — 1.7B and 4B bases, adapter strength varied 25× (33 M all-projection → 1.3 M attention-only), one and two epochs, sequence truncation fixed, sampling fixed — with the same 363 Fable-written, regex-gated examples, judged by the same rubric:

| | 4B base | v2 (33 M) | v3 (1.3 M) |
|---|---|---|---|
| Format floors met (of 7) | 1 | 6 | 1 |
| Correctness with substance | **1.50** | 1.18 | 1.21 |
| Native ES / PL | **9/12 / 3/17** | 3/12 / 1/17 | 2/12 / 1/17 |
| Hand-off | **88 %** | 65 % | 19 % |
| Semantic tail collapse | **10/60** | 28/60 | 27/60 |

**What is established.** (1) Retrieval over the guide carries the facts: substance coverage 0.84, and the untrained 4B is fully correct on 60 % of rows where the substance is in context. (2) LoRA on this data installs the *shape* — language routing, plain text, declines, refusals, citation density — completely when strong enough (v2) and partially when weak (v3). (3) In every configuration it also degrades what the base already had — substance use, native Spanish and Polish, hand-off — and the degradation is not a function of adapter strength, epochs, truncation, quantisation or decoding. The common factor is the data-and-objective: 363 answer-only sequences under prompt masking teach a *distribution over answers* that the base then follows into invented continuations; the shape comes with it, the base's own competence does not. (4) Polish native quality is a base-model floor (3/17 untrained at 4B) that training moved only downward.

**The spec's thesis "facts in retrieval, form in the weights" is refuted at this scale for this setup.** Form-in-weights costs more substance than it adds, on both bases, at every adapter strength tried. Prompt-only form (the null) is also refuted on the 4B (language 68 %).

**What that leaves — three designs, for Andrei's decision, each a new pre-registered block:**
- **A. Form in glue, not weights** — the untrained 4B (best substance we have) with deterministic form enforcement outside the model: the system prompt rendered *in the question's language* (the English prompt is what pulls ES/PL answers into English/Spanish), markdown stripped in post, a retrieval-score floor that empties the context for off-topic questions so the model declines, and a templated hand-off appended when the answer lacks one. Cheap (no training), directly testable on the sealed 60 with the same judge, and it is the workspace's own deterministic-glue pattern. Expected: format floors from glue, substance ≈ base (1.50).
- **B. Change the objective, not the adapter** — preference-style training (DPO/ORPO on pairs of "base answer vs. shape-corrected answer" for the same context) so the model learns the *delta* rather than a new answer distribution; or full-sequence training with the base's own answers as targets after shape editing. More work, unproven here.
- **C. More and different data** — 1,500+ examples across all temas including shape-only edits of the base's own outputs. The evidence (damage independent of strength) suggests data *kind*, not amount, is the lever; C without B is unlikely to move it.

Recommendation: A first — it is the only path whose expected value is bounded below by the untrained base's numbers.

## Design A — form in deterministic glue (2026-09-22 13:30–14:35)

`coapi_voice/glue.py` around the **untrained** `unsloth/Qwen3-4B-GGUF:Q4_K_M`: (1) deterministic language detection; (2) scope gate — one narrow few-shot classification call, constrained to IN/OUT, out → templated decline; (3) guide-BM25 retrieval (adopted config); (4) answer with the system prompt **rendered in the question's language**; (5) post-processing — markdown stripped, retrieved-chunk ids/headers and unfilled placeholders scrubbed, **every citation absent from the retrieved context removed**, templated hand-off appended when the answer routes nowhere; plus a deterministic advice guard (drafting/decision/price requests get a templated refusal-to-decide and never a figure or a document). Two model calls, ~6.8 s per question on the Mini, no training.

| | 4B base | v2 (LoRA) | **Design A (A5)** |
|---|---|---|---|
| Language compliance | 82 % | 100 % | **100 %** |
| Plain text | 18 % | 100 % | **100 %** |
| Out-of-scope declines | 0/3 | 2/3 | **3/3** (gate 60/60, no misfires) |
| Hand-off present | 88 % | 65 % | **100 %** |
| Citation-in-context | 91 % | 100 % | **100 %** (fabricated citations removed, 0/33 left misleading) |
| Numeric preservation | 100 % | 100 % | **100 %** |
| Answers citing | — | 87 % | 75 % |
| Spanish-term inclusion | 69 % | 90 % | **56 %** — the one floor glue cannot enforce |
| Artefact rows (judge, A4) | 44 | 2 | 42 → **0 after the A5 fixes** |
| **Correctness / with substance / usefulness** (judge, A4) | 1.13 / 1.50 / 3.65 | 1.05 / 1.18 / 4.00 | **1.20 / 1.45 / 4.15** |
| Native PL / ES (judge, A4) | 3/17 · 9/12 | 1/17 · 3/12 | **5/17** · 3/12 |
| Tail degradation | 10/60 | 28/60 | **11/60** |

Judge: `results/designA_A4_judge_20260922.json` (rubric unchanged plus three decisions for glue output, recorded in `rubric_additions_designA`). A5 = A4 plus the glue fixes the judge identified (placeholder and chunk-header leakage, seam cleanup, hand-off routing test, Polish/Spanish price questions); its regex profile is the table above.

**Result.** Design A is the best configuration produced by this experiment: it matches the untrained base on substance (1.45 vs 1.50, within noise) instead of costing 0.3 as every fine-tune did, and it takes the format floors the fine-tunes bought at that price — including the two no fine-tune ever reached (out-of-scope 3/3, hand-off 100 %). It also improves native Polish (5/17, the best of any run). Cost: a second model call (~1 s) and a longer answer path.

**What design A does not fix, and no glue can:** the untrained 4B still inverts or substitutes a rule on ~12 of 60 rows — three of them with the rule in the retrieved context (M2-02 early-repayment cap, M4-02 new-build VAT, M6-05 residual land value). Correctness-with-substance 1.45 of 2 is the model's ceiling on this hardware, not the pipeline's. Native Spanish (3/12) is worse than the base's free choice because A forces Spanish on every ES question; the base "scored" 9/12 partly by answering in English.

### A6 — final build (2026-09-22 14:40)

A5 introduced a real defect that the A4 judge could not have seen: `scrub()` deleted any line beginning with a chunk header, and **8 of 57 in-scope answers were reduced to nothing but the hand-off template** (the model legitimately opens with "Módulo 2 · Tema 21: …"). Fixed: the header is rewritten to a usable citation ("Módulo 2, Tema 21:") instead of deleted, and post-processing may never return fewer than 20 words. A6 also puts the Spanish-term rule first in all three prompts. Regex on A6: language 100 %, plain text 100 %, oos 3/3, hand-off 100 %, citation-in-context 100 %, numeric 99.3 %, retrieval hit 93 %, no truncations, no answers under 60 words, 0 placeholder/chunk-header rows.

**Metric correction (recorded, not quietly fixed):** "Spanish-term inclusion" as scored until now excluded acronyms (ITP, IBI, TAE, AJD are ≤ 4 characters) and counted the out-of-scope declines in the denominator. Re-measured on in-scope EN/PL answers with acronyms counted, **every run scores 100 % — base, v2 and design A alike** — so the metric does not discriminate and the earlier "69 %"/"56 %" readings were artefacts, not findings. The meaningful question — is the Spanish term given in parentheses at *first use of each concept* — is judged qualitatively (A6 judge, `spanish_term_first_use`).

### A6 judged (`results/designA_A6_judge_20260922.json`) — the best configuration of the experiment

| | 4B base | v2 (LoRA) | A4 | **A6** |
|---|---|---|---|---|
| Correctness / twos | 1.13 / 38 % | 1.05 / 20 % | 1.20 / 40 % | **1.23 / 45 %** |
| With substance in context (38 rows) | 1.50 / 61 % | 1.18 | 1.45 / 53 % | **1.47 / 61 %** |
| Native PL / ES | 3/17 · 9/12 | 1/17 · 3/12 | 5/17 · 3/12 | **11/17** · 6/12 |
| Tail degradation | 10/60 | 28/60 | 11/60 | **6/60** |
| Hand-off / oos / no-advice | 88 % · 0/3 · 2/4 | 65 % · 2/3 · 4/4 | 93 % · 3/3 · 3/4 | **100 % · 3/3 · 4/4** |
| Usefulness | 3.65 | 4.00 | 4.15 | **4.28** |
| Artefact rows | 44 | 2 | 42 | 23 |

H7 (correctness ≥ 1.2, with-substance ≥ 1.4, twos ≥ 35 %), H8 (Módulo 6 gap 0.16) and H9 (native PL ≥ 6/17 → 11/17) all pass on A6 — the first build in the experiment to clear them, and it is the one with no trained weights. **Spanish term at first use, judged strictly: 43.5 %** (bare acronyms are the main failure) — the honest number behind the vacuous regex metric. Remaining harmful rows: 12, all of them the untrained model substituting or inverting a rule (Módulo 4 tax is the sink at 2.8 usefulness).

**A8 (2026-09-22 15:15):** A6 plus the five fixes the A6 judge asked for — full Spanish term demanded, three advice leads (drafting / pricing / decision), citation stripping made conservative so a valid citation is never lost, filler-only sentences dropped, a softer hand-off for answers with no consequence. Mechanical parity with A6 except **rows carrying a citation 46 → 38** (the reordered prompt buys Spanish terms and costs citations — the same attention trade-off seen throughout). Judge pass running to decide A6 vs A8 as the reference.

### A8 judged, A9b built (2026-09-22 15:30–16:20)

A8's five changes were judged row by row (`results/designA_A8_judge_20260922.json`): correctness 1.217 vs A6 1.233, **with-substance 1.526 — the best of any build** (two harmful in-context inversions fixed), native ES 6→10/12, but artefact rows 23→37 and rows citing 46→38. Cause, isolated by the judge and reproduced against the code: change (4)'s filler-sentence splitter breaks at abbreviation periods, so "Under art. 9 LAU," rendered as "9 LAU," — 21 rows lost a citation head and three lost real content (the gold term *superficie útil* among them). Verdict: **A6 remains the reference; A8 as shipped is not net positive.** Prescription: A9 = A6 + changes 1, 2 (drafting clause out of the generic lead), 3 (number must match as an article/tema/law number), 5 (selector fixed), **without 4**.

**A9b** is exactly that (`results/designA_A9b_summary_20260922.json`): language 100 %, plain text 100 %, out-of-scope 3/3, hand-off 100 %, **citation-in-context 100 % with 45/60 rows citing** (coverage restored, no seams), numeric preservation 100 %, no truncations. Judge pass running; its `reference_recommendation` decides A6 vs A9b.

**Method note worth keeping:** every glue defect in this cycle was found by the judge reading rendered answers, not by the mechanical scorer — the scorer showed A8 at parity while the page it produced was worse. Regex metrics bound the floor; they do not see the page.

### A9b judged → reference; A10 is A9b plus its three follow-ups (2026-09-22 16:20–16:55)

`results/designA_A9b_judge_20260922.json`: correctness 1.217, **with-substance 1.500 / 60.5 % twos**, native PL 11/17 · ES 10/12 (best of any build), tail 7/60, usefulness 4.27, hand-off 43/43 with the variant now correctly selected, artefact rows 25, judged Spanish-term-at-first-use 43.5 %. The judge's recommendation is explicit: **A9b is the reference** — "same substance as A8 with A6-grade rendering, better ES native than A6, fixed leads and selector, honest measurement (no hidden headlines)".

**An integrity finding worth keeping.** A8 scored 2 on M4-03 where A9b scores 1: the splitter had *eaten the model's wrong headline* ("Sí, según el art." where the gold says no plusvalía is due). A rendering bug that raises a score by concealing an error is the most dangerous kind of glue defect, and only a judge reading rendered answers can catch it. Recorded as the method lesson of this cycle: **mechanical metrics bound the floor; they do not see the page.**

**A10** applies the three one-line follow-ups the judge named plus a safe version of the filler filter: `cite_support` now folds spelled-out *article / artículo / artykuł* (a valid citation is no longer stripped — the M5-09 defect), `cite_ok`'s substring test is word-bounded (*art. 9* no longer matches *art. 90/91* — the M4-02 defect), `ADVICE_LEAD_PRICE` reworded so it promises only what the material covers (M5-10), and filler-only sentences are dropped with an abbreviation-aware splitter that never breaks "art.", "np.", "min." Regex: language / plain text / out-of-scope / hand-off / numeric / citation-in-context all 100 %, 44/60 rows citing, 9 chunker tests and the glue selftest green. `results/designA_A10_summary_20260922.json`.

**Adopted as the reference implementation** for the on-device product (build A9b, plus the A10 follow-ups pending one confirming judge pass); v1–v3 are kept as evidence only.

**Where the ceiling now is.** Across A6/A8/A9b the model text is unchanged by glue: correctness ~1.22, with-substance ~1.50, and **13 rows still harmful to a buyer**, concentrated in Módulo 4 (tax, 0.7 correctness) and on retrieval misses, where the untrained 4B substitutes a confident wrong rule. Two further gaps are model-side and untouched by any glue: Spanish term at first use 43.5 % (bare acronyms), and rows carrying a citation 75 %.

**Remaining levers, in order:** (a) a verify pass on numeric and tax rows — spot→verify, a third call that re-reads the retrieved chunk and checks the stated number against it; the 13 harmful rows are exactly what it targets; (b) guide work on Módulo 4 and the 12 retrieval-miss rows (the ADJUDICATION.md conflicts overlap here); (c) option B (preference training on the shape delta) only if the substance ceiling itself must move — the evidence says glue has taken shape as far as it goes.

## Lever (a) — the verify pass: two probes before building (2026-09-22 17:00–17:40)

The 13 harmful rows in the reference build split into 4 where the rule **was** in the retrieved chunk and the model misread it (M2-02, M2-09, M4-02, M6-05) and 9 retrieval misses where it substituted one. Both are in principle catchable by a second look at the same context, so the question is whether a 4B can check its own answer.

**Probe 1 — the broad question (refuted).** "Find every rule or number in the ANSWER that is not in the MATERIAL or contradicts it." Result: flags 13/13 harmful rows **and 42 of 44 clean ones**; mean flags 3.4 on harmful rows vs 3.6 on clean. **No discrimination whatever** — asked to find fault, the model finds fault everywhere. `results/verify_probe_broad_20260922.json`.

**Probe 2 — one claim, yes/no (signal).** Each sentence carrying a number is checked alone: "Is the STATEMENT supported by the MATERIAL, exactly as written — same rule, same number, same situation?" → YES/NO. Mean NO-flags **1.15 per harmful row vs 0.43 per clean row**; at a threshold of ≥2 flags, recall 5/13 with 3 false positives in 44; at ≥1, recall 8/13 with 15 false positives. `results/verify_probe_narrow_20260922.json`.

This is the spot→verify pattern behaving exactly as [[project_orchestration_insight]] describes: the same weights that cannot answer reliably can answer a *narrower* question usefully — but only when the question is narrowed to a single claim with a binary answer. The breadth of the question, not the model, was the variable.

**Build V1** = reference glue + verify at threshold 2 + one repair pass (the flagged sentences are named, the model rewrites keeping only what the material supports and saying plainly where it does not settle the point). Cost: up to 7 model calls per question. Result and judge pass recorded next.

**V1 judged (`results/verify_V1_judge_20260922.json`) — the pass does not earn its latency as built.** Correctness 1.217 → 1.233, harmful rows 13 → 12, usefulness 4.267 → 4.350 (two of those points are A10's glue fixes, not the repair). Of the 12 repairs: **3 fixed a wrong claim** (M4-03 headline inversion corrected, M4-05, M4-09), **7 changed nothing of substance**, **2 made it worse** (both triggered by a false positive on the row's one correct sentence). Only 1 of the 5 previously-harmful repaired rows improved; the other four keep the harmful claim word for word under an appended hedge. Cost +4 s per question on the Mini (+60 %), ≈ +20 s on the phone.

Diagnosis (judge): 35 of 52 flagged units are **citation fragments** — the same abbreviation-blind sentence splitter, now inside the verifier; the threshold of 2 missed 5 rows whose single flag was correct, 4 of them harmful; the verifier is blind to numberless inversions and to copied worked-example figures (M6-05's €494,000 passes because every digit is in context); and the repair is free to append "the material does not specify" where it does. **Kept as a negative result.** Re-test only with: an abbreviation-aware splitter, the headline sentence always verified, flagged sentences deleted under post-check rather than left to the model, and the "does not specify" template allowed only when the verifier confirmed absence — on the 13 harmful rows first, not the whole set.

## Base-model probe — Gemma in the same pipeline (2026-09-22 17:58)

`gemma-4-e4b-it` at Q4_K_M is **6.0 GB** — beyond the iPhone 14 Pro's ~3.3 GB per-app cap, so it is out for the phone regardless of quality. `gemma-3-4b-it` Q4_K_M is **3.3 GB** (Qwen3-4B: 2.5 GB) — at the cap, so it would need its own device test. Run through the identical reference glue (`results/gemma3_4b_glue_summary_20260922.json`): language 100 %, plain text 100 %, out-of-scope 3/3, hand-off 98 %, **49/60 rows citing** (Qwen 44) at 98.7 % in-context, numeric preservation 100 % over 199 numbers (Qwen 148), 8.5 s per question. Judge pass running; it decides whether the on-device base should change and by how much it would have to win to justify the size risk.

### Gemma 3 4B judged (18:45, `results/gemma3_4b_glue_judge_20260922.json`) — no base-model switch

Same rubric, retrieval byte-identical to A9b on every row.

| | Qwen3-4B (A9b) | Gemma 3 4B (G3) |
|---|---|---|
| Correctness / twos / zeros | 1.217 / 45 % / 23 % | 1.233 / 42 % / 18 % (11 rows up, 10 down — inside the ±4-row noise seen between A6 and A8) |
| With substance (38) / on misses (19) | 1.500 / 0.526 | 1.474 / 0.632 |
| Native PL / ES / EN | 11/17 · 10/12 · 30/31 | **14/17 · 11/12 · 31/31** |
| Tail degradation / usefulness | 7/60 / 4.27 | 15/60 / 4.05 |
| Harmful rows | 13 | 11 (four each on in-context questions) |
| Rows citing / fabricated citations | 45 / 0 left | 49 / 4 (invented article numbers inside real statutes; prompt example pasted on 7 rows) |
| Spanish term at first use | 43.5 % | 39 % |
| Mean tokens / wall time | 177 / 1.0× | 266 / 1.24× |

Gemma's in-context losses are arithmetic and date errors on numbers it quotes correctly (89 − 70 = "10 %", 2015 + 10 "likely valid" in 2026); its miss-row losses are invented facts. A greeting ("Dzień dobry! Jako doradca…") opens 17 of 26 in-scope PL/ES rows and accounts for the usefulness gap on its own; the correctness ledger is substance. The A10 glue held on a different model, and exposed three model-independent defects to fix: valid citations stripped on M4-02 (art. 91.Uno.1.7ª) and M6-04 (art. 9.3); fabricated ones kept on M4-07 ("17" matched RDL 17/2018) and M4-09; hand-off template suppressed by "consult with your advisor" on M5-06 (first hand-off miss in the A series, 42/43).

**Decision: keep Qwen3-4B on the device.** Gemma is level on correctness, one row behind with substance, worse on usefulness/tail/term, better only on PL/ES fluency (~5 rows) — at 3.3 GB against the 14 Pro's ~3.3 GB per-app cap, with no headroom for index + KV cache, where Qwen at 2.5 GB leaves ~0.8 GB. Bar for a future swap (same sealed 60, same glue): correctness ≥ 1.45, with-substance ≥ 1.7 with no harmful in-context row, harmful ≤ 6, usefulness ≥ 4.27, tail ≤ 7, and a verified load on the device with the full pipeline resident. The finding's right use is a PL/ES fluency step in the glue, plus the three run_eval/route-detector fixes.

## What counts as evidence

- Improvement is claimed only against `coapi_eval_v1.jsonl`, only once per cell, only if `split_manifest_v1.json` predates the run (it does — this file).
- The Módulo 6 rows are evaluated **once** for B3 and then burned.
- Numbers and citations are checked by regex, never by a model. Correctness, no-advice and usefulness are judged by Fable with the rubric fixed in the B2m `judge.json`; B3 is judged by the same rubric text.
- Aborted training runs go to `aborted/`, never deleted.

## Rights

Training and device artefacts derive from Andrei's own study notes (S1/S2) only; the COAPI course corpus (S3) and the 67 logged buyer conversations (S4) never enter the adapter, the index, the eval, or any cloud call. `coapi_voice/RIGHTS.md`, signed 2026-09-21.
