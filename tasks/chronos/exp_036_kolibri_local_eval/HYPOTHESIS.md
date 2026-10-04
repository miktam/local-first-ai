# Experiment 036 — Kolibri on a MacBook: does it fit, how fast, and does its scorecard hold on public rows?

*Pre-registered: 2026-10-03T22:33:25Z · Status: pre-registered, awaiting Andrei's sign-off; no scored run*

**Builds on:**
- exp_007: the padding fixtures, loaded here by their real file names.
- exp_008 / exp_010: the prefill-cliff rule.
- exp_011: the MLX runtime. Its fixture-name defect is not repeated here.
- exp_012 / exp_012-Alpha: state total and active parameters for every model.
- exp_014: plan for variance, and score blind.
- exp_015: pin by revision, never by label.
- exp_023: result and verdict shape.
- exp_035: pre-registration shape, version record and withheld rows. exp_035 pushed its full pre-registration only together with its results; exp_036 pushes this file and the kit before any scored run.
- incident_003_alpha: the test-script contract and thermal logging.

**Directory:** `tasks/chronos/exp_036_kolibri_local_eval/`. Assets are pinned in [`assets.json`](./assets.json) and [`ASSETS.md`](./ASSETS.md) (commit 8078b69; New assets appended before the pre-registration push). The kit is described in [`BUILD_SPEC.md`](./BUILD_SPEC.md), and the run steps are in [`RUNBOOK.md`](./RUNBOOK.md).

**Which document wins.** `scientific_log.md` > this file > `BUILD_SPEC.md` > `RUNBOOK.md`. The scientific_log entry holds pointers and verdict words only, no figures, so the two cannot disagree on numbers. If a lower document disagrees with a higher one, the lower one is wrong and is fixed by a dated amendment.

**What is edited in place.** Only the Status line, the sign-off fields (by Andrei) and the placeholders filled at pre-registration. Everything else is appended: dated run-record blocks, numbered amendments and results.

**Amendments.** Numbered in the order they are appended, each with a typed title: `## Amendment k — <gate fix | threshold change | peer drop | extractor fix | plan | re-pilot | Tier-2 analysis> (<UTC>)`. Amendment 0 is reserved for the sign-off choices. Tools look an amendment up by its type, not its number. From the pre-registration push until the mbp hands back (RUNBOOK step 19), only the mbp session appends to this file; the mini pushes its amendments as files under `amendments/`, and the mbp appends them verbatim at its next commit.

---

## What this experiment tests

Andrei's question, in his words: *"create experiment 36 to see how good is this model, end result — published on github and localfirstai.eu".*

Kolibri-1 is Aleph Alpha's English–German MoE: 78.1B total parameters, 3.46B active, Apache-2.0 weights, released 2026-10-03. As of 2026-10-03 no local runtime supports it and nobody has evaluated it independently. We write an MLX port, check it against a separately written numpy reference (Phase 0), and run it on Andrei's MacBook Pro (Apple M5 Max, 40-core GPU, 128 GB unified memory). It runs next to the models we already use, on the same runtime and the same host. The publication must give a plain answer to five questions; the wording is pre-registered under "Plain answers".

| # | Question | Answered by |
|---|---|---|
| 1 | Does it run on our kind of hardware, and how fast? | D1 (fits a 64 GB node at 4-bit), H1 (speed class against Gemma 4), E6, E7 |
| 2 | Does it live up to its own scorecard on what can be checked? | H2 (public deterministic rows against the vendor's values), E8 (peer positive control) |
| 3 | How does it compare with the models we already run, in English and German? | H3 (knowledge gap EN/DE), H4 (instruction following), H5 (German tokenizer), E1 (public-only composite) |
| 4 | Where is it weak, and what are the trade-offs? | H6 (closed-book correct-answer rate), E2 (does it say so? does it know when forced?), H3, E3, E5, E12 |
| 5 | What does squeezing it onto a 64 GB node (4-bit) cost? | H7 (task cost K4 vs K8), H8 (logit fidelity against the peers), D1 |

**Arms:**
- **K8, K4:** Kolibri, converted by us to MLX affine 8-bit and 4-bit (group 64).
- **G8, G4:** Gemma 4 26B-A4B-it, 8-bit and 4-bit. Its family is our production Reducer.
- **Q36-8, Q36-4:** Qwen3.6 35B-A3B, 8-bit and 4-bit.
- **Q38-8, Q38-4:** Qwen3.8 27B dense, 8-bit and 4-bit.

All peer builds are mlx-community conversions pinned in assets.json. One host (the M5 Max MacBook Pro) and one runtime (mlx 0.31.2 / mlx-lm 0.31.3, in process) serve every arm. Nothing runs on miktam-mini except building the kit, weight-free unit tests and the analysis afterwards.

## The claim being protected

The claim being protected is the localfirstai.eu verifiability contract, applied to a model nobody has checked yet. Every number we publish about Kolibri must meet three conditions:
1. It comes from a port that matches a separately written numpy fp32 re-implementation of the vendor's source (Phase 0, blocking). The two share no code but were written from the same specification. The vendor's own runtime is not run unless the optional cloud FP8 anchor is approved.
2. It is compared under settings fixed and pushed publicly before any output was scored.
3. Wherever a vendor setting is public, it is honoured: effort high, generation-config sampling, the eval-framework prompts and extractors.

A gate failure, an INCONCLUSIVE verdict or a NOT RUN cell is published as such.

---

## Sources & Rights

- **Vendor technical report.**
  - Source: <https://aleph-alpha.com/downloads/tech-report.pdf>, read as retrieved 2026-10-03.
  - Pin: 3,465,511 bytes, sha256 `01520e07506e67d53aebfb16b5298384940870d652409ea3943fe990f7a68b0e`, Last-Modified Sat, 03 Oct 2026 08:53:19 GMT, ETag `"34e127-65cebc7cd3d0c"`.
  - **Not committed** anywhere, including the workspace submodule. Aleph Alpha retains all rights to it, and Apache-2.0 covers only the weights and config files.
  - Every vendor number in this experiment is cited as "p. N/189".
- **Model card.** Aleph-Alpha/Kolibri-1-BF16 README at revision `7a8f290e`. Its licence section says the licence "does not extend to … model architecture", and also names "underlying code" and "parameter settings". Whether the port and the reference may go into the public repo is decided by Andrei **before** the pre-registration push (see "Pre-registration decisions"). This is not legal advice.
- **Code derived from aleph-alpha-inference.** The source is aleph-alpha-inference v1.0.0 (commit `049a6a7bd2`, Apache-2.0, © 2026 Aleph Alpha GmbH).
  - Derived files: `port/kolibri1.py`, `reference/kolibri_ref.py`, `reference/mutants.py`, `gate/port_mutants.py`, `tests/tiny_checkpoint.py`, `tests/test_routing.py` and the vendor chat template fixtures.
  - Obligations: the SPDX headers are kept, each derived file carries a "Modified by Miktam for Chronos exp_036" line, the Apache-2.0 text is `LICENSE-APACHE-2.0` in this directory, and every Apache-derived or copied file is named in `NOTICE` (checked by `tests/test_notice.py`).
- **eval-framework v0.14.2** (Aleph-Alpha-Research, Apache-2.0). The prompt templates, option-shuffle code and answer extractors are vendored under `tasks/vendored_evalfw/` byte for byte, with one exception: `gpqa.py` contains the full text of one GPQA question (`_OVERLONG_QUESTION`), which is replaced by its sha256 before the file enters the repo. That modification is recorded as an Apache §4(b) change (a "Modified by Miktam" line, both the upstream and modified sha256 in `vendored_evalfw/MANIFEST.json`, a note in `SOURCE.md`). `scorers/mc.py`, `scorers/aime.py` and `tasks/prompts/aime_de.txt` (a German translation of the NeMo-Skills wrapper) are Apache-derived and listed in NOTICE. Any upstream NOTICE file of eval-framework is reproduced.
- **Peer chat templates.** The Gemma 4 and Qwen templates the runner uses are committed under `runner/templates/` (Apache-2.0 per their repositories' metadata) and listed in NOTICE.
- **IFBench checkers.** allenai/IFBench @ `1c40f0c1` (Apache-2.0), used in place from `$EXP036_DATA/IFBench-src` and never vendored.
- **RGB.** chen700564/RGB. The data is CC BY-NC-SA 4.0 (stated in the repository README at the pinned commit); `evalue.py` carries no licence header. We therefore do **not** port RGB's code. `scorers/rgb.py` is our own implementation of the rule RGB describes (lower-cased alias containment for every answer slot), with RGB cited as the source of the rule. A test checks it against RGB's own function, run in place from `$EXP036_DATA/RGB-src`. RGB prompts are rendered at run time from the local RGB files; no RGB text, not even the instruction strings, is committed. Using RGB under its NonCommercial terms for localfirstai.eu is a pre-registration decision of Andrei's.
- **Datasets:**

| Dataset | Revision | Licence / terms | What is published |
|---|---|---|---|
| Idavidrein/gpqa | `83022cef` | CC BY 4.0 plus gate terms: do not reveal examples online | item ids, item sha256, extracted letters, scores, token counts. **Never** item text, gold or raw outputs (`$EXP036_PRIVATE`, hash-listed in `evidence/withheld_manifest.jsonl`) |
| ellamind/gpqa-multilingual | `bc70ca17` | gated, handled exactly like GPQA | as GPQA |
| math-ai/aime26 | `79037aeb` | Apache-2.0 as labelled by the uploader; the problems are © MAA | ids, hashes, gold integers, raw outputs, scores |
| ellamind/aime26-multilingual | `3c8bc18f` | no licence stated | ids, hashes, scores; raw outputs withheld |
| allenai/IFBench_test | `2e8a48de` | ODC-BY 1.0 (attributed in NOTICE) | ids, hashes, raw outputs, scores |
| li-lab/MMLU-ProX(-Lite) | `8e6106a6` / `e82aafb9` | MIT, © li-lab | ids, hashes, category, gold letters, raw outputs, scores |
| chen700564/RGB (new) | pin in assets.json | data CC BY-NC-SA 4.0 | ids, document indices, hashes, category scores; raw outputs withheld |
| HuggingFaceFW/fineweb-2 deu_Latn test (new) | pin in assets.json | ODC-By 1.0; subject to Common Crawl terms | derived counts; for gate texts T5–T6 only parquet row indices, token counts and sha256. **No web text is committed.** |

Item manifests in the repo carry ids, sha256 and, for MMLU-ProX and AIME EN only, the gold label and category. No manifest carries item text.

- **Gate texts.**
  - T1 and T2: our own posts (2026-09-22, exp_035; 2026-09-30, the Malaga-AI study-group post), written after Kolibri's 2026-06-18 cutoff.
  - T3: German prose written by Claude for exp_036 (post-cutoff). LLM-written text has unusually low perplexity, so G3 reports NLL per text and T3 is read on its own. Andrei may substitute his own text only before the pre-registration push.
  - T4: Grundgesetz Art. 1–19, an amtliches Werk under § 5 UrhG, in the public domain; the source URL is recorded in `gate/texts/MANIFEST.json`.
  - T5–T6: FineWeb-2 documents, used locally only (see above).
- **Withheld text is never published anywhere.** `tools/leak_check.py` runs before every commit and, as a git pre-push hook, before every push. It checks every file changed since the upstream commit plus the index, not only the last staged files. The rules are in BUILD_SPEC §5.9.
- **Disclosure.** We have no relationship with Aleph Alpha. Aprimerose builds local-first deployments and would use Kolibri if it held up.

### New assets needed

The main session pins these and adds them to assets.json and ASSETS.md **before** the pre-registration push; this file and RUNBOOK then reference assets.json only. Only the first two are downloads for Andrei on the mbp.

| Asset | Exact source | Licence | Size | Why |
|---|---|---|---|---|
| RGB | github.com/chen700564/RGB, commit pinned by the main session (candidate `65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615`): `data/en.json`, `data/en_fact.json`, `data/en_int.json`, `config/instruction.yaml`, `config/instruction_fact.yaml`, `evalue.py`, `README.md` → `$EXP036_DATA/RGB-src` | data CC BY-NC-SA 4.0; code unlicensed | ≈ 25 MB | H6, H7, E2, E3, pilot. These are the vendor's non-selection rows where Kolibri scores worst (p. 101/189). |
| FineWeb-2 deu_Latn | HuggingFaceFW/fineweb-2 @ `af9c13333eb981300149d5ca60a8e9d659b276b9`, `data/deu_Latn/test/000_00000.parquet` → `$EXP036_DATA/fineweb-2` | ODC-By 1.0 | 104 MB | H5 (the vendor measured on FineWeb-2, p. 10/189); gate texts T5–T6 |
| aleph-alpha-inference v1.0.0 | @ `049a6a7bd2`: LICENSE, `kolibri1.py`, `tests/checkpoints.py`, `tests/kolibri1_chat_template.jinja`, `tests/test_kolibri1.py` | Apache-2.0 | < 1 MB | Already on the mini. Vendored into the kit; no mbp download. |
| eval-framework v0.14.2 | `benchmarks/{cot,gpqa,gpqa_ellamind,mmlu_pro,math_reasoning}.py`, `answer.py`, the helpers they import (`task_style.shuffle_correct_with_distractors`, `utils.get_n_letters`), and its NOTICE if any | Apache-2.0 | < 1 MB | Partly on the mini; the main session fetches the rest at the same tag. Vendored; no mbp download. |
| Upstream peer tokenizer and template files | google/gemma-4-26B-A4B-it @ `20da991a`; the Qwen3.6-35B-A3B and Qwen3.8-27B base repos at revisions pinned by the main session: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json` | Apache-2.0 | < 30 MB | **Default: fetch on the mini at build time.** Byte parity of the mlx-community peers' tokenizer and template with upstream (the vendor ran upstream). Any difference is recorded, and the runner renders with the upstream template (committed under `runner/templates/`). |
| *Optional:* FineWeb EN | HuggingFaceFW/fineweb @ `9bb295dd`, `sample/10BT/000_00000.parquet` | ODC-By 1.0 | 2.1 GB | Descriptive EN tokenizer ratio only. **Default: skip.** EN is then reported on our own EN gate texts plus MMLU-ProX-Lite EN question text, labelled "not FineWeb". |

Build-time fetches on the mini (main session, with Andrei's OK, before the pre-registration push; none of them is an mbp download) are listed in BUILD_SPEC §8. They are: the FineWeb-2 parquet and RGB at the pins above, the Kolibri and mlx-community peer tokenizer and template files, the upstream peer files, the eval-framework helpers, the Grundgesetz text, allenai/IFBench_test at its assets.json pin, and the IFBench checker environment.

---

## Fixed before any run (hashes)

The placeholder values in this table are filled by `tools/hash_tree.py --fill` in the pre-registration commit and pushed before any scored run. After the push, one line `Pre-registration commit: <sha>` is appended at the end of this file. Every built item manifest (public and withheld) is created on the mbp in S1 by the frozen rule below; its per-file sha256 goes into the plan amendment, which is also pushed before any scored run.

| Item | Value |
|---|---|
| Vendor report | sha256 `01520e07506e67d53aebfb16b5298384940870d652409ea3943fe990f7a68b0e`, 3,465,511 bytes (not committed) |
| Kolibri source weights | Aleph-Alpha/Kolibri-1-BF16 @ `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, 156,228,414,296 bytes, 32 shards. Every shard's sha256 is checked against its LFS oid at preflight. |
| Kolibri FP8 repo | Aleph-Alpha/Kolibri-1 @ `e52eb4627d11516b0c01de49210ab5a4e4061444`. Not downloaded; referenced only for the optional cloud anchor. |
| Peers | gemma-4-26b-a4b-it-8bit `33c6d237` / -4bit `0d77464e`; Qwen3.6-35B-A3B-8bit `e06a74e6` / -4bit `38740b84`; Qwen3.8-27B-8bit `815b83c0` / -4bit `10c35caa` (full SHAs in assets.json) |
| Asset manifest | `assets.json` sha256 `42803f55e14c90c3bf7b229010b7aad76ba2135d51e4b0e83bd526fda47ff5a3` (after the New assets are appended) |
| Vendor plugin | aleph-alpha-inference v1.0.0 @ `049a6a7bd2`. The vLLM semantics are resolved against tag v0.29.0; files are listed in `reference/DERIVATION.md`. |
| eval-framework | v0.14.2; vendored files sha256 `e7ff393392d7e60d8e2ba81e76a70c12233da01511d8da2bb7de6ea751bde1bb` |
| Runtime | Python 3.12, mlx 0.31.2, mlx-metal 0.31.2, mlx-lm 0.31.3, plus the pins in `env/requirements-mbp.txt` (sha256 `66ad262cdef58c5a5793f70f145322e255a260de6c56d8df3794ac9281fbe73b`) |
| Port | `port/kolibri1.py` sha256 `cd6153b8b00b81f7058e786f33672a07b46c494be0203ad75b04372179dc5584`; `port/convert.py` sha256 `6c16fc4bd66db3efa4b470ba2707d7eee9907cd83b515de90c79b2d2a44c29a2` |
| Reference | `reference/` tree sha256 `85337ed7efeef6b0463f3560fff6da6496661a2fa7a14de5e50a1bce9e251bbd`. Frozen before the port meets real weights. |
| Gate code | `gate/` tree minus `thresholds.json` and `texts/`, sha256 `d48f01b85485413e792d4101e1d480b20b21966cac285ab244a4695fc316c012` (includes `behaviour_prompts.json`) |
| Gate thresholds | `gate/thresholds.json` sha256 `9ff12f6561924e46fa44aa80566c338d0eaf856c048fb35769fae272f7030852` |
| Gate text | `gate/texts/MANIFEST.json` sha256 `1a295dacf20d285512dcf6db755623e89053ec704abef2bd151682178d91b9e8`: ids, decoded text and sha256 of T1–T4 and T7–T8; for T5–T6 the selection rule and, if the parquet was on the mini at build time, their row indices and sha256 (otherwise those go into the gate record) |
| Runner | `runner/` tree sha256 `c79556cd7d815992497aeb87c9b0a29789f219bc15eaaa9517b8de28fd00b334` (sampler, chat kwargs and templates, generate, memory, plan rules) |
| Tasks code | `tasks/` tree minus `manifests/`, sha256 `14c301535eee68c60e6e5d265d902465a0b1cfc2aa58bbda14d9efe0de1d8865` (loaders, renderers, `prompts/aime_de.txt`, `prompts/rgb_forced_en.txt`, vendored eval-framework) |
| Manifest rule | `tasks/selection_rules.json` + `tasks/build_manifests.py` sha256 `cadee104542128f05a106d1f348c3fa3e579c1681ce1d01220971a04138a6955` |
| Plan rules | `runner/plan_rules.json` sha256 `2707ea2571cc122b7388f9c9794e3e1130aefdf13c6f844b7de449485bce878c` (budget, ladder P0–P10, Tier B B1–B10, cap rule, memory rule, crash fallback) |
| Scorers | `scorers/` tree sha256 `771920143434427e901ced7bfe280f3c5518feeea8642c45f9b25e45506cd8d3`, incl. `abstain_lexicon.json`, the IFBench adapter and golden fixtures (no withheld text) |
| Analysis / verdict code (Tier 1) | `analysis/` tree minus the Tier-2 files (`exploratory.py`, `tables.py`), sha256 `3b4c976b437c17d33d6b99ba01128c61f13c7b854ce0c5e684b7699c64d596ed`, incl. `stats.py`, `verdicts.py`, `power.py`, `margins.json` and `vendor_values.json`. Exploratory analysis code (Tier 2) is frozen by its own amendment before any scoring (see "What counts as evidence"). |
| Bench | `bench/` tree minus `ladder.py` (Tier 2), sha256 `82f4629f5623b6bf3ce8e01042ad9496be3809af053b73936b2e8aeddf26713a` |
| Tools | `tools/` tree minus `withheld_shingles.sha256` (built in S1), sha256 `e979f786a2f35ea61c217c212830ed8c6e8fe2ae08410367ed809e8992571f48` (preflight, leak check and its hook, hash tree, peer check, status, redaction) |
| Quantisation | Affine, group 64, 8 and 4 bits (the mlx-community peers' recipe). Kolibri's router weight is stored as an exact fp32 upcast of the bf16 checkpoint and gives fp32 logits; `expert_bias` (fp32) and all RMSNorm weights are unquantised. Default head policy (subject to the sign-off): `embed_tokens` and `lm_head` quantised at the arm's bits, as in the peers, with fp32 activations into the head, so the logits are fp32. KV cache is bf16 for every model; no `--kv-bits`. |
| Reasoning | Kolibri `reasoning_effort="high"`, passed explicitly. Gemma 4 `enable_thinking=True`. Qwen3.6 / Qwen3.8 `enable_thinking=True` (their template default). This is the vendor protocol, Table 48, p. 164/189. |
| Sampling | Each model's pinned generation_config: Kolibri T 1.0 / top-p 0.97 / top-k 128; Gemma 4 1.0 / 0.95 / 64; Qwen 1.0 / 0.95 / 20. Applied by the kit's vLLM-order sampler (temperature, then top-k, renormalise, then top-p) for **every** model, on fp32 logits; no min-p and no penalties. The seed per cell is `H("exp036", arm, task, effort, pass)`. |
| Output caps | Initial: GPQA and MMLU 32,768; AIME 65,536; IFBench and RGB 16,384 completion tokens. Ceilings: GPQA and MMLU 65,536; IFBench and RGB 32,768; AIME 98,304. At most one raise per task, straight to the ceiling, by the pilot rule below. Truncated = wrong (our rule; see C7). |
| Statistics | Bootstrap B = 10,000 (100,000 near a threshold, below), numpy PCG64 with a per-hypothesis seed = the first 8 bytes of sha256(`"exp036|" + H_id`), percentile intervals; H1 uses a t-test. Holm at family-wise α = 0.05, separately for CONFIRMED and for REFUTED. |

---

## Pre-registration decisions (Andrei, before the push)

*Recorded by the main session from Andrei's answers, with the date and the words he used. Claude never chooses these.*

Answered by Andrei on 2026-10-03 at about 18:10 UTC, in the mini session, choosing from the options put to him. His selections are quoted.

- Publish `port/` and `reference/` in the public repo with this push, given the model card's licence section (default recommended: yes; if no, the push carries only their sha256 and the code follows after his decision): **yes** — "Yes, publish (Recommended)".
- RGB under CC BY-NC-SA 4.0 for evaluation and publication of scores only, no redistribution of data or raw outputs, on localfirstai.eu (default recommended: yes; if no, H6, E2 and E3 and the RGB rows of H7 are removed before the push and m = 7): **yes** — "Yes, scores only (Recommended)".
- New assets (RGB, FineWeb-2 deu_Latn) approved for the mbp, and the build-time fetches on the mini (BUILD_SPEC §8) approved (default recommended: yes): **yes** — "Approve all (Recommended)". The mini fetches were made on 2026-10-03 into `~/models/exp036-mini/` at the pins recorded in assets.json.
- T3 German prose: Claude's text as committed, or Andrei's own (default: Claude's, labelled so): **Claude's text, labelled so** — "Accept all three (Recommended)".
- Overrun ceiling for the automatic Tier-A continuation S3b: 44 h of run machine time (default) or 40 h: **44 h** — same answer.
- README rows "Pre-registered" in both index tables with this push (default: yes): **yes** — same answer.

---

## Confounds (pre-registered)

| # | Confound | Control / statement |
|---|---|---|
| C1 | **Precision grid.** The vendor evaluated FP8 e4m3 128×128 weights with dynamic FP8 activations and an FP8 KV cache, after FP8 quantisation-aware RL (p. 74, 180/189). We run MLX int8 / int4 affine g64 weights with bf16 activations and a bf16 KV cache. The vendor targets datacentre FP8 serving (model card: minimum 2× A100 80 GB or 1× H200/B200); we test outside that envelope. | "MLX int8 g64 is not the vendor's FP8-QAT grid. The gate bounds the port against the reference, not the grid." The gate reports K8 and K4 KL to fp32 descriptively. E11 reports what the quantised head and embeddings contribute. E8 bounds the joint grid and protocol effect through the peers. An optional cloud FP8 anchor runs only if Andrei asks. |
| C2 | **Runtime.** vLLM 0.29 plus the vendor plugin, against our MLX port. | Phase 0 gate (blocking, hash-bound). |
| C3 | **Peers run on mlx-lm's own modules**, from mlx-vlm conversions that we did not validate. | The peer check (S1): strict text-only load, parameter count within 2 % of the config arithmetic, NLL(8-bit) ≤ NLL(4-bit) + 0.02 nats/token on the gate text, KL(8‖4) < 0.2, template and tokenizer byte parity with upstream, and batched-path parity through the same BatchGenerator code the scored runs use. E8 uses the peers as a positive control. |
| C4 | **Router precision asymmetry.** The peers' routers are 8-bit (Gemma `router.proj`; Qwen3.6 `mlp.gate`, `shared_expert_gate`). Kolibri's router gives fp32 logits, as the vendor's does. | Stated. E11 measures Kolibri's routing agreement and KL with an 8-bit router. |
| C5 | **Sampler order.** Stock mlx-lm (`sample_utils.make_sampler`) applies top-p before top-k on the full distribution. vLLM applies top-k, renormalises, then applies top-p. | The kit's vLLM-order sampler is used for every model and unit-tested against a numpy transcription of vLLM v0.29 `apply_top_k_top_p`. |
| C6 | **Logit dtype.** Kolibri computes fp32 logits (vendor `head_dtype`). The peers' mlx-lm modules return bf16 logits, which is also vLLM's default for them (head dtype = model dtype). | The kit upcasts every model's logits to fp32 before normalisation and sampling. H8 compares like with like, using bf16-rounded logits for every model; Kolibri's fp32-logit KL is descriptive. |
| C7 | **Output caps vs the vendor's 256k window.** The vendor ran a 256k window with no completion cap and scored overflow as wrong only on Tau3 and BFCL (Table 48, p. 98, 164/189). | Truncated = wrong is **our** rule, at caps far below the vendor's window, so it penalises long reasoners more than the vendor's protocol does. Controls: initial caps of 16–32k with one raise straight to the ceiling (65k for GPQA/MMLU) on a sensitive pilot trigger; a paired truncation sensitivity for H3, H4, H6 and H7 that decides the headline (below); and, for every H2 row, the per-model truncation rate and the result excluding truncated items shown in the key-numbers table next to the confirmatory number. The reasoning caps above 10k tokens are a declared exception to the deterministic-glue per-stage budget, because the vendor protocol requires long reasoning. |
| C8 | **avg@N vs k.** The vendor ran avg@8 (GPQA), avg@16 (AIME) and avg@5 (IFBench), p. 96–97/189. We run k = 1. | H2's variance includes the vendor's avg@N term. A second K8 GPQA pass is Tier-B item B10; if it runs, the GPQA rows use the two-pass mean and the within-item variance estimator (below). |
| C9 | **Templates and thinking modes differ per model.** | Every kwarg is passed explicitly. The rendered prompt's sha256 and its token ids are stored per item. Kolibri template parity is a gate check; peer template and tokenizer byte parity with upstream is checked at build time. Per-model reasoning splitters are tested on fixtures. Gemma's top_k = 64 assumes vLLM `--generation-config auto` applied the checkpoint default; the vendor lists only "1.0 / 0.95" for Gemma (Table 48, p. 164/189). |
| C10 | **Harness: V vs R rows.** | V = eval-framework v0.14.2 prompt and extractor, verbatim (GPQA EN/DE, AIME EN). R = reconstructed (MMLU EN on ProX-Lite text, MMLU-ProX DE, AIME DE wrapper, IFBench checker, RGB). Every row carries its label. V-row extractors are never amended; if one had to be, the row would be relabelled R. |
| C11 | **Selection overlap and in-distribution training.** GPQA, AIME, IFBench and MMLU-Pro chose the SFT mix and soup; the vendor itself calls those scores optimistic (pp. 67, 70/189). IFBench is also the target of an RL instruction-following environment with constraint verifiers (88,761 rows, 10k budget, p. 172/189), so H4 measures that training as well as generalisation. RGB was not a selection benchmark, but RGB Negative matches an RL environment that "poses unanswerable questions with their full context and rewards only abstention" (p. 174/189). | E9 reports the selection rows, RGB CB/FC and RGB Negative as three separate groups. |
| C12 | **Contamination.** Pre-training was not decontaminated (p. 35–36/189). 39.1 % of MMLU-Pro test questions were found in the pool (p. 157/189). AIME 2026 predates the 2026-06-18 cutoff. | Not probed; stated as a limit on H2, H3 and E1. The peers' contamination is unknown and may differ. Gate texts T1–T3 are post-cutoff. |
| C13 | **Teacher confound.** Gemma-4-26B-A4B rephrased Kolibri's English pre-training data (p. 25/189). Qwen3.8-27B generated SFT completions (p. 53/189). | Stated with every comparison. |
| C14 | **Item subsampling and composition.** MMLU-ProX-Lite has 588 test items per language, 42 in each of 14 categories; the vendor's MMLU rows are on the full, category-imbalanced sets (MMLU-Pro 12,032; MMLU-ProX DE). GPQA EN: 197 items (eval-framework) vs 198. RGB Negative: 300 vs the vendor's inferred 299. RGB Closed-Book: every vendor value is an integer, so the vendor's n is inferred to be 100, on an unknown subset; our 400 items differ. | H2's MMLU rows are post-stratified to the full set's category shares (below); the n_M ladder keeps every category equal. 197 is primary and 198 a sensitivity check. H6's power is stated as conditional on the vendor gap replicating on our items. |
| C15 | **Batching numerics.** Continuous batching with left padding and `BatchRotatingKVCache`. | Gate G5 batch parity with mid-run admission (K8), the peer check's batched-path parity (G8, Q36-8), and Control C1 (greedy answer-flip rate at the memory-rule B vs B = 1). |
| C16 | **Active vs total parameters.** | Stated for every arm, from the configs at preflight (exp_012-Alpha lesson). |
| C17 | **Laptop conditions.** Thermals, power and background load on a 14" MacBook Pro. | High Power energy mode on AC; `pmset -g` power mode, the power source and macOS `thermalState` are logged per bench block and every 10 minutes in sessions. Speed cells start after a 10-minute idle. The plan projection uses decode steps from the last third of each arm's pilot. Quality verdicts never depend on timing. |
| C18 | **Hardware transfer** to the 64 GB M4 Pro node. | D1 assumes allocations depend on MLX version and tensor shapes, not on the chip. Absolute tok/s is labelled "M5 Max, MLX 0.31.2". The mini is never used as a run host, and its speed is not measured. |
| C19 | **Pilot leakage.** | Pilot items are disjoint for GPQA (main set, filtered by Diamond Record ID), MMLU (ProX full, non-Lite) and RGB (`en_int`). IFBench and AIME pilot outputs are discarded: they feed only length and rate statistics, never amendments or tuning. |
| C20 | **Comparator set** differs from the vendor's: no Qwen3.5, GPT-OSS or Nemotron. | Stated in the post. |
| C21 | **RGB protocol.** The vendor's exact prompts are not public. | We follow RGB's own code paths: closed-book is `passage_num = 0` (instruction verbatim, empty documents, no system prompt), and negative and fact-check use RGB's system prompts. Labelled R. |
| C22 | **Kolibri's RL abstention environment** is in-distribution for RGB Negative and plausibly for closed-book questions asked with an empty "Document:" frame. H6 may therefore measure abstention policy as well as knowledge. | The forced-answer condition (Tier A from P0 to P7) separates them; without it, H6 is worded as a correct-answer rate only (see H6 and "Plain answers"). |

---

## Phase 0 — port-correctness gate (blocking)

**Reference.**
- `reference/kolibri_ref.py` is numpy only. It imports neither mlx nor the port. It was written separately from the port, from the same numbered specification (BUILD_SPEC §4), so a misreading common to both would pass every comparison between them. G3 and the vendor's routing test are the only checks independent of that specification.
- It reads the BF16 shards with its own parser in `reference/safetensors_np.py`: memory-mapped uint16 widened by `<< 16`, which is exact.
- It streams one layer at a time (≈ 3.1 GB bf16, 6.2 GB fp32) and never holds the model.
- Each block cites the vendor `kolibri1.py` line and the vLLM v0.29.0 line it implements (`reference/DERIVATION.md`).
- Modes: fp32 (the reference proper); **bf16 emulation**, which rounds to bf16 wherever vLLM stores bf16 (norm outputs, q/k/v, RoPE output, attention output, o_proj, each expert output, the residual after the fp32 add) and keeps router logits, routing weights and the head in fp32; and **dequantised**, which runs a layer on weights unpacked from a converted MLX directory by its own numpy unpacker for MLX affine packing (written from the format, not imported from mlx).
- It dumps per-layer `h_in`, `h_mid` (after the attention add) and `h_out`, the branch outputs `r_attn` and `r_moe`, router logits, biased scores and top-6 sets, and final fp32 log-probs, to `$EXP036_WORK` (not committed; sha256 and summary statistics are committed).
- It is hash-frozen in the pre-registration commit, before the port meets real weights.

**Gate text** (`gate/texts/`; T1–T8 are exactly 1,536 Kolibri tokens each = 12,288 tokens, committed as token ids with sha256 and the decoded text alongside; the gate consumes the ids):

| Sequence | Content |
|---|---|
| T1 | EN, our exp_035 post (2026-09-22), first 1,536 tokens of the body |
| T2 | EN, our Malaga-AI post (2026-09-30), first 1,536 tokens of the body |
| T3 | DE, German prose written by Claude for exp_036 (≥ 1,600 tokens before the cut) |
| T4 | DE, Grundgesetz Art. 1–19 |
| T5–T6 | DE, the first two FineWeb-2 deu_Latn test documents with ≥ 1,536 Kolibri tokens at file index ≥ 5,000 (disjoint from H5). Built by rule into `$EXP036_WORK/gate_texts/`; only indices, token counts and sha256 are committed. |
| T7 | EN chat render at effort high, with a closed think block |
| T8 | DE chat render at effort none |
| T9 | Long sequence, 16,384 tokens: T1, T4 and T3 in full, then T5, T6 and further FineWeb-2 deu_Latn test documents from index ≥ 5,000 in file order, concatenated by rule and cut at 16,384. Not committed (it contains web text); its rule, indices and sha256 are. |

T1–T8 each cross the 513-token window; T9 exercises the full-attention layers over long key ranges, cache growth and RoPE at large positions.

**Checks.** Thresholds are frozen in `gate/thresholds.json`, and the verdict is computed in code (`gate/run_gate.py`). "emu" below means the bf16-emulated reference compared with the fp32 reference on the same input; it calibrates what a correct bf16 implementation achieves on these weights. "Forced" means the port is forced onto the reference's expert ids, with its weights computed from its own fp32 logits (spec item 12). Blocking checks are sharp and self-calibrating; absolute fidelity numbers are reported but do not block.

| Check | What | Pass threshold (blocking unless marked descriptive) |
|---|---|---|
| G0 static | Census: tensor count; parameter count; every shard's sha256 = LFS oid; strict load consumes every tensor exactly once and the port declares no parameter absent from the checkpoint (this catches a retained afmoe attention gate); full-attention layers = {4, 9, …, 49} | 58,353 tensors; 78,103,074,560 parameters; 0 mismatches |
| | Template parity: 12 conversations × 10 settings (no kwarg, `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `enable_thinking=false`, effort `high` overriding `enable_thinking=false`, system + tools) against the vendor jinja | byte-identical, 120/120 |
| | Tokenizer: harness ids vs raw `tokenizers` ids on ≥ 2,000 lines (lines of T1–T4, identifiers from the kit's own code and seeded synthetic digit strings, by a committed rule; plus MMLU-ProX EN/DE question lines read at gate time and never committed) and the gate text; decode round-trip; `fix_mistral_regex` never True | 0 mismatches |
| | Converted configs: quantization `{group_size 64, bits 8 \| 4, mode affine}`; tensor policy equal to the one recorded in `exp036_convert_record.json`, which must equal the signed-off head policy; no `.scales` for the router, `expert_bias` or norms; bf16 scales and biases on every quantised tensor (head and embedding included); router weight dtype float32 | exact |
| | Value-level dtype checks on a K8 and K4 forward: router-logit and head-output dtype float32, and the fraction of router logits and of head logits exactly representable in bf16 (`x == float32(bfloat16(x))`) | dtype float32; each fraction ≤ 1 % (fp32 compute gives ≈ 2⁻¹⁶, bf16 compute 100 %) |
| G1 weight-free (also pytest on the mini) | The existing calibrated suite: port in fp32 vs reference on the tiny checkpoints; vendor `test_routing_semantics`; window and NoPE through `RotatingKVCache` and `BatchRotatingKVCache`; chunked prefill; cache decode; refill with mid-run admission on rotated caches; router logits fp32 of bf16 operands; quantised port vs reference on dequantised weights; sampler parity; 15 port mutants | as in `tests/INTEGRATION_LOG.md` and BUILD_SPEC §6 (fp32 ≤ 1e-4 logits / ≤ 1e-5 caches; forced-routing (b1) and natural-routing (b2) bf16 bounds) |
| G2 per layer, real weights | Port layer *l* fed the reference input *h_l*, all 50 layers. Errors are measured on the branch outputs before the residual add, each relative to its own reference norm: e_attn = ‖r_attn,port − r_attn,ref‖ / ‖r_attn,ref‖, likewise e_moe. | see rows below |
| | **fp32 mode**, forced | median e ≤ 1e-4 and max e ≤ 1e-3 per layer and branch; embedding exact; final norm + head max \|Δlogit\| ≤ 1e-3 |
| | **fp32 mode**, natural selection | top-6 sets identical for every (token, layer) except where the reference's 6th-vs-7th biased-score gap < 1e-4 (exempt pairs reported with their gaps) |
| | **bf16 mode** (production dtype, unquantised), forced | per layer and branch: median e ≤ 3 × emu's median, and p99 e ≤ max(5e-2, 3 × emu's p99) |
| | **bf16 mode**, natural selection | fraction of (token, layer) top-6 disagreements ≤ max(0.2 %, 2 × emu's fraction), and every disagreement has a reference 6th–7th biased gap < 6·σ_l, where σ_l is the measured RMS of (port − reference) router logits in layer *l* |
| | bf16-router mutant 10 vs port, paired token bootstrap of agree_port − agree_mutant | descriptive; blocks only if its 99th percentile < 0 (the port agrees with the reference *less* than the mutant). Mutant 10 itself is caught by the value-level check above and by `test_router_logits_are_fp32_of_bf16_operands`. |
| | **G2q, converted weights** (K8 and K4): the port's quantised layer, loaded from the converted directory, vs the reference run on that layer's dequantised weights, same *h_l*, forced; plus the embedding and the head (both sides fed the reference's normed hidden state) | per layer and branch as the bf16-mode forced row; per-expert-group maximum error reported; embedding max relative error ≤ 4e-3 (one bf16 rounding); head relative L2 error ≤ 1e-3 per position |
| | **T9 long sequence**, bf16 mode, forced, bucketed by position (0–2k, 2–8k, 8–16k) | each bucket meets the bf16-mode forced row, against emu's statistics from T1–T8 for the same layer and branch |
| | Real-weight mutants 1–8 and 12–15, through the same per-layer harness on layers {0, 3, 4, 49} | each fails some G2 fp32 check (branch error, selection or embedding) on ≥ 1 layer |
| G3 reference oracle | Reference bits per UTF-8 byte on the plain texts T1–T6, per text | each text ≤ 1.2 bpb, and the mean ≤ 1.25 × the best peer (G8, Q36-8) on the same bytes, from the peer check |
| | Seven reference mutants (sigmoid+bias selection, RoPE on full layers, (1+w) RMSNorm, renormalised top-k, swapped sandwich norms, traditional RoPE, q/k-norm after RoPE), as one layer-streamed pass carrying parallel hidden streams | for each mutant, the 1st percentile of a 48-block bootstrap of (NLL_mutant − NLL_ref) on T3 is > 0. A mutant whose difference is not resolvable is reported as "rests on vendor source" and does not fail the gate. |
| G4 end to end | **K8** (converted, normal load) vs the fp32 reference over all 12,288 positions and over T9 | gross-bug backstop: mean KL(ref‖K8) ≤ 0.10 nats/token and top-1 agreement ≥ 99 % at positions where the reference leads by ≥ 2 nats. Everything else descriptive: mean KL, top-1, ΔNLL, KL by window position and by T9 bucket, EN and DE separately. |
| | **K4** vs the fp32 reference | descriptive only (feeds H8 and E11). K4's bug checks are G0, G2q and G5 behaviour. |
| G5 generation path | K8 greedy 256-token continuations of 8 prompts (4 EN, 4 DE; efforts none and high; prompts ≥ 600 tokens), teacher-forced through the reference | at positions where the reference leads by ≥ 2 nats, generated token = reference top-1 at ≥ 99.5 %; within the reference top-5 at ≥ 99 % of all positions |
| | Noise floor: K8 prefill with `prefill_step_size` 2048 vs 64 on the same positions | measured; defines floor_KL and floor_dis |
| | K8 decode vs prefill at positions 520–1,100 of T1 and T3, and 15,000–15,300 of T9 | mean KL ≤ max(1e-4, 3 × floor_KL); top-1 disagreement ≤ 3 × floor_dis + 0.2 pp |
| | Batch parity: B = 8 through `runner.generate`, left-padded, lengths {37, 300, 700, 1,100, …}, staggered `max_tokens` so that sequences are admitted mid-run, vs single, teacher-forced | K8: the decode-vs-prefill bound. K4: reported. |
| | Behaviour on 20 frozen short factual or one-step prompts (`gate/behaviour_prompts.json`, 10 EN, 10 DE), K8 and K4, effort high capped at 8,192 tokens, and effort none | blocks only if, for either arm, ≥ 2/20 loop (a 32-token span repeated ≥ 4 times consecutively), ≥ 2/20 end without either EOS id (127906, 127901), or ≥ 2/20 get `lang_tag` "unknown". Think-format and language-match rates are descriptive (E4 measures language). |

**Port mutants.**
1. Selection on sigmoid + bias (the afmoe and llama.cpp trap).
2. Window 512 instead of 513.
3. RoPE on the full-attention layers.
4. (1+w) RMSNorm.
5. `post_attn_norm` ↔ `post_attention_layernorm` swapped.
6. Renormalised top-k.
7. afmoe `route_scale` 2.826.
8. muP √d embedding scaling.
9. A retained afmoe attention output gate.
10. bf16 router logits.
11. bf16 lm_head logits.
12. Traditional (interleaved) RoPE.
13. q/k-norm applied after RoPE.
14. Routing weights from the biased score, sigmoid(logits + bias).
15. SwiGLU with silu on `up` instead of `gate`.

On real weights, mutant 9 must fail G0 (strict load), and mutants 10 and 11 must fail the value-level bf16-exactness check. On the tiny checkpoint, mutants 10 and 11 use fixtures built so that bf16 rounding changes the outcome: a near-tie in the biased logits, and logits of magnitude ≥ 16.

**Verdict and refusal.**
- **K8 PASS** iff every blocking check for K8 passes (G0–G5). **K4 PASS** iff G0 for K4, the same port sha256 as K8, G2q at 4-bit and the K4 behaviour check pass. A K4 quality outcome (its KL or top-1 against the reference) is a result for H8 and E11, never a gate failure.
- `run_gate.py` exits 0 (both PASS), 4 (K8 PASS, K4 FAIL: H1, H7, H8 and D1 become NOT RUN unless a fix cycle repairs K4), 1 (K8 FAIL: stop) or 3 (could not run).
- The runner refuses every Kolibri pilot, bench or scored run unless the latest `results/gate/gate_<UTC>.json` says PASS for that arm, and its recorded sha256 of `port/kolibri1.py`, of the converted-directory manifest, of `gate/thresholds.json` and of `reference/` equal the current ones.
- On FAIL, the record is committed and pushed, partial outputs go to `aborted/` with a note, and the fix becomes a numbered amendment followed by a **whole** gate re-run. Reference dumps whose reference tree sha, checkpoint fingerprint and gate-text sha all match are reused; every port-side check reruns in full.
- At most two fix cycles. A fix cycle normally moves the gate to a later day. A fix to the reference must cite a vendor or vLLM line and may not touch `gate/thresholds.json` in the same amendment. A threshold may change only through a numbered amendment typed by Andrei that states the original value, the observed value and the reason; it counts as a fix cycle, and the post reports the original thresholds and every relaxation in its gate section.
- After a third failure, the gate failure (which check, which layer) is published as the exp_036 result. No speed, memory or quality number from an ungated port is ever reported.
- **Optional, only on Andrei's explicit ask:** one cloud GPU hour running vLLM 0.29 with the vendor plugin on the FP8 repo, with `prompt_logprobs` on the committed public gate texts T1–T4 and T7–T8. This anchors the reference to the vendor's runtime.

**Peers are verified, not gated.** The peer check (C3) runs in S1, before the gate (G3 uses its NLL). Its batched-path check runs G8 and Q36-8 through `runner.generate` with mixed prompt lengths 37–1,100 and mid-run admission: B = 8 vs B = 1 teacher-forced under the G5 noise-floor rule, and a greedy answer-flip rate ≤ 2 % on 30 MMLU-ProX full (non-Lite) EN items with thinking off. A peer that fails the batched-path check runs at B = 1 (re-projected by the plan rule); a peer that fails anything else is dropped by amendment before any scored run. H3 and H6 then use the remaining MoE peer instead of the mean, and H4 is NOT RUN if Qwen3.6 is the peer dropped.

---

## Pilot and the rule that fixes n and max_tokens

**Pilot** (S1, after the gate and the bench cells; its outputs are never scored for accuracy):

| Arm | Items |
|---|---|
| K8 | GPQA main (non-Diamond, filtered by Diamond Record ID) EN 8 + DE 8 (ellamind `is_diamond = false`); MMLU-ProX full (non-Lite) EN 8 + DE 8, parallel; IFBench 8 (discarded); RGB `en_int` 16 closed-book + 8 forced-answer; AIME26 EN 4 (discarded) |
| G8, Q36-8 | the same minus AIME |
| K4 | the same minus AIME and forced-answer |

Each arm runs at the memory-rule B at the initial caps, task-grouped. Every decode step is logged (n_live, padded length, step seconds).

**Measured:** completion-length distribution per (arm, task) (mean, SE, max), truncation count (a completion that ends at the cap, including one still inside its reasoning segment), reasoning-segment status counts, parse failures, prefill rate, and the per-step decode log.

**Rules** (`runner/plan_fix.py`, deterministic, constants in `runner/plan_rules.json`):

1. **Memory rule for B.**
   - B per cell = the largest of {16, 8, 4, 2, 1} with need(B) ≤ 0.9 × L, where need(B) = weight bytes (summed safetensors sizes of the arm's directory) + 2 × B × (max prompt + cap) × KV bytes per token + B × fixed window bytes + 4 GiB. The factor 2 covers the transient copy when the batch KV grows or admits a sequence; every row is padded to the longest live row. AIME cells are capped at B ≤ 8. Tasks are never mixed in a batch. The runner calls `mx.set_cache_limit(4 GiB)`.
   - KV bytes per token from the config: Kolibri, Gemma 4 and Qwen3.6 20,480; Qwen3.8 65,536. Fixed window bytes per sequence: Kolibri ≈ 42 MB, Gemma 4 ≈ 210 MB, Qwen3.6 its DeltaNet state; computed by `tools/kv_bytes.py`.
   - L = max(MLX `max_recommended_working_set_size`, `iogpu.wired_limit_mb` × 2²⁰), both recorded. Measured on the mbp on 2026-10-03 (`host/mbp_preflight_20261003T170152Z.json`): 115,448,725,504 B = 107.5 GiB (84 % of 128 GiB), `iogpu.wired_limit_mb` = 0 (default). At that L, K8 runs GPQA and MMLU at B = 8 (32k caps; 4 after a raise to 65k) and IFBench and RGB at B = 16.
2. **Cap raise.**
   - Trigger, per task: in any model's pilot cell, ≥ 1 truncation, or any completion longer than 0.5 × the cap (0.4 × the cap for GPQA, whose pilot uses easier non-Diamond items).
   - Effect: that task's cap goes straight to its ceiling for **all** models (GPQA and MMLU 65,536; IFBench and RGB 32,768; AIME 98,304), and the memory rule is re-evaluated. At most one raise per task.
3. **Parse and reasoning gate.**
   - Parse failure = no answer extractable after the reasoning segment (GPQA, MMLU, AIME), or an empty post-reasoning answer (IFBench, RGB), among non-truncated items. A completion that ends unclosed at the cap counts as a truncation, never as a reasoning-gate failure.
   - A cell is BLOCKED only if it has ≥ 2 parse failures **and** the pre-written diagnostic (`runner/plan_fix.diagnose`) shows a delimiter or extractor mismatch, e.g. an answer letter present after the reasoning but unmatched by the vendored regex, or a reasoning delimiter the splitter does not know. Otherwise the rates are descriptive.
   - A BLOCKED cell is fixed only by a numbered extractor or delimiter amendment before any scored run, and is then re-piloted. V-row extractors are never amended (C10). IFBench and AIME pilot outputs never drive an amendment.
   - For Gemma 4, a completion that never opens a thought channel (status "none") is scored on the whole completion and does not count against the reasoning gate. For Kolibri and Qwen at effort high, status "none" is impossible (the think block is prefilled) and is recorded as a defect.
4. **n.**
   - Each cell is projected by a deterministic continuous-batching simulation: B slots; completion lengths resampled (seed 36) from the pilot distribution of that (arm, task) and scaled by (mean + 1 SE)/mean; GPQA-D cells scaled by a further × 1.25 (main → Diamond); pilot items truncated at the initial cap counted at the post-raise cap; step time = a + b·n_live + c·n_live·padded_len, fitted per arm on the decode steps of the last third of that arm's pilot; prefill at the measured rate; the cell's tail included. Tier-B cells with no pilot cell of their own use fixed ratios: peer AIME = the arm's GPQA pilot lengths × K8's AIME/GPQA length ratio; K8 at effort none, low and medium = 0.08 ×, 0.4 × and 0.7 × K8's effort-high lengths in the same task; Q38-8 (B9) = Q36-8's GPQA pilot lengths, with a step model fitted on Q38-8's descriptive speed cells; B4 = a fixed 1.4 h.
   - Projected hours T(plan) = Σ simulated cell hours × 1.15.
   - S1 hours = the sum of the machine intervals recorded by the S1 blocks (preflight `--deep`, convert, peer check, gate, bench, pilot), not wall-clock time.
   - Main budget B_main = min(31 h, 40 h − S1 hours).
   - The ordered queue (below) is split at cell boundaries into S2 and S3, each ≤ 16 h projected. Take the first plan in P0 … P10 whose total is ≤ B_main and whose two-session split succeeds. Then add Tier-B items B1 … B10 in order; each item is added whole if the total stays ≤ B_main and the split still succeeds, otherwise it is skipped and the next one is considered.
   - If even P10 does not fit: **STOP**. No scored run until Andrei decides on a 4th session by amendment.

| Plan | Tier A |
|---|---|
| P0 | n_M = 588 per language, forced-answer CB, RGB Negative and Fact-Check |
| P1 | P0 minus all RGB Negative and Fact-Check (K8, G8, Q36-8) |
| P2 | P1 with n_M = 504 |
| P3 | n_M = 406 |
| P4 | n_M = 350 |
| P5 | n_M = 294 |
| P6 | n_M = 252 |
| P7 | n_M = 196 |
| P8 | P7 minus the forced-answer CB cells (H6 then uses the correct-answer-rate wording) |
| P9 | P8 with n_M = 154 |
| P10 | P9 minus K8 GPQA-D DE (H2 then runs on 4 rows) |

n_M is always a multiple of 14 (42, 36, 29, 25, 21, 18, 14 and 11 items per category), and each smaller set is a per-category prefix of one seed-36 order, so every set is category-balanced and nested.

**Tier A** cells, in queue order:
1. K8: GPQA-D EN 198, GPQA-D DE 198, MMLU-ProX-Lite EN n_M, DE n_M (parallel ids), IFBench 300, RGB closed-book 400.
2. G8 and Q36-8: MMLU EN and DE n_M (H3 complete); Q36-8 IFBench 300 (H4 complete), G8 IFBench 300; G8 and Q36-8 RGB closed-book 400 (H6 complete).
3. K4: MMLU EN and DE n_M, IFBench 300, RGB closed-book 400 (H7 complete).
4. P0–P7: forced-answer closed-book 400 for K8, G8, Q36-8.
5. P0 only: RGB Negative 300 and Fact-Check 100 for K8, G8, Q36-8.

**Tier B**, in order (the vendor's claimed strength rows for the peers come first):

| Item | Cells | Feeds |
|---|---|---|
| B1 | K8 AIME26 EN and DE | H2 secondary, E4 |
| B2 | G8 and Q36-8 on AIME26 EN and DE | E1, E8 |
| B3 | G8 and Q36-8 on GPQA-D EN | E1, E8 |
| B4 | context ladder for K4, K8, G4 | E6 |
| B5 | K8 at effort none on GPQA-D, MMLU, IFBench, RGB closed-book | E12 |
| B6 | G8 and Q36-8 on GPQA-D DE | E1, E8 |
| B7 | K8 at effort low and medium on GPQA-D EN | E5 |
| B8 | K4 on GPQA-D EN and DE | adds two rows to H7, fixed in the plan amendment |
| B9 | Q38-8 on GPQA-D DE | E10 |
| B10 | K8 GPQA-D pass 2 | H2 variance (C8) |

5. **Freeze.**
   - The output is `results/plan_fixed_<UTC>.json` plus `results/AMENDMENT_<k>_<UTC>.md` (type "plan"), where k is the next free amendment number in this file and `amendments/`. `plan_fix` never overwrites a file.
   - The plan amendment holds: plan, n_M, caps (any raise), B per cell, the L used, the ordered queue and its S2/S3 split, Tier-B items, the H2 and H7 row sets, the post-stratification category counts, the sha256 of every built manifest (public and withheld), gate and peer-check record sha256s, whether K8 runs, whether the forced-answer cells run, and power recomputed at the fixed n.
   - It is appended verbatim to this file, committed and pushed from the mbp before S2. The runner refuses S2 unless it is at HEAD and HEAD equals upstream.
   - A re-pilot after a BLOCKED cell produces a new plan amendment that names the one it supersedes; the earlier amendment stays in place.
6. **Cells and sessions.**
   - A cell's n is fixed before it starts, and every item in it runs to completion (resumable across sessions), so a time cutoff never truncates a cell towards short, easy items.
   - `run.py session` stops admitting new cells once the session's elapsed hours plus the next cell's projection exceed 16 h; started cells finish. S3 continues the same queue.
   - **Tier-A cells always run to completion.** If S3 ends with Tier-A cells outstanding, a continuation S3b runs automatically, with no discretion, counted in the budget. A Tier-A cell is admitted in S3b only if the projected run total stays ≤ the overrun ceiling (44 h by default); otherwise the hypotheses that need it are NOT RUN with p = 1 and m unchanged. Tier-B cells not started by the end of S3 are NOT RUN.
   - If K8 cannot run (below), K4 replaces K8 in H2, H3, H4 and H6 through the plan amendment, and H7 is NOT RUN. H2 then keeps its −4 pp margin but is titled "at MLX 4-bit", and the H8 KL and K4's KL to the reference are cited next to it.
   - **Crash fallback.** `run.py` writes a heartbeat (current cell, B). On a start that finds a stale heartbeat for an unfinished cell, that cell resumes at the next lower B in {16, 8, 4, 2, 1}, and its records carry `b_fallback_from`. After two fallbacks the cell moves to `aborted/` with a NOTE.md, its hypotheses are NOT RUN (p = 1), and the queue continues.

**K8 working-set rule.**
- Preflight computes K8's memory-rule B for every K8 cell at the current L and at L' = 112 GiB. It prints `sudo sysctl iogpu.wired_limit_mb=114688` (112 GiB, leaving ≈ 16 GiB for macOS) whenever L' would raise the B of some K8 cell, or whenever need(1) > 0.9 × L. Andrei runs it by hand and preflight is re-run; the value goes into the sign-off. The kit never calls sudo. At the measured default L (107.5 GiB) and the initial caps, no line is printed.
- If Andrei declines, B for K8 follows the memory rule. If need(1) still exceeds 0.9 × L (L < ≈ 90 GiB), K8 is **NOT RUN** and K4 becomes primary through the plan amendment, before any scored run.
- The sysctl resets on reboot. `run.py` refuses any K8 cell (exit 1, printing the sysctl line) if the current L is below the L the plan used.
- The 64 GB mini measured 51.84 GiB (81 %).

---

## Confirmatory hypotheses (H1–H8, Holm family)

**Common rules.**
- **Two Holm families over H1–H8.** For each H, p is the one-sided p-value for its null (CONFIRMED side) and p_rev the one-sided p-value for its reverse null (θ on the confirm side of its REFUTED threshold). Holm is run over the eight p, and separately over the eight p_rev, each at family-wise α = 0.05.
- **CONFIRMED** iff the step-down Holm-adjusted p ≤ 0.05. **REFUTED** iff not CONFIRMED and the Holm-adjusted p_rev ≤ 0.05. Otherwise **INCONCLUSIVE**; it is published as "INCONCLUSIVE (leans refuted, nominal 95 %)" if the unadjusted p_rev ≤ 0.025. Decisions use the adjusted p only; the bound at the adjusted level is reported for information.
- Bootstrap p-values: p = (1 + #{θ\*_b on the null side of θ₀}) / (B + 1), where θ\*_b = θ₀ counts on the null side. Every null below includes equality.
- B = 10,000 per hypothesis, from its own seed (the first 8 bytes of sha256(`"exp036|" + H_id`) into PCG64), so results do not depend on evaluation order or on which hypotheses ran. Any H whose raw p or p_rev lies within a factor of 2 of its Holm threshold is recomputed with B = 100,000 before Holm is final. The Monte Carlo SE of every p is reported.
- **m = 8 is frozen at the plan amendment.** An H that becomes NOT RUN later (crash, overrun, peer drop) keeps its slot with p = p_rev = 1.
- Truncation, parse failure and unclosed reasoning count as wrong. Only the text after the reasoning segment is scored.
- **Paired truncation sensitivity** for H3, H4, H6 and H7: each is recomputed after excluding every item on which any compared arm truncated. The per-arm truncation rate is reported in the same table row as the verdict. If the sensitivity changes the verdict, both are published and the headline uses a verdict only if the two agree; otherwise it says INCONCLUSIVE (truncation-sensitive).
- **Peer positive control (E8) flag.** If |D_peer| = |local − vendor| > 8 pp for a peer on a row an H uses (H3: MMLU EN/DE; H4: IFBench; H6: RGB closed-book), the H's verdict is still computed as specified but is published as "<verdict> — peer control failed (<peer>, <row>)" and may not appear in a headline. For H4 the post then leads with the Qwen3.6 local vs vendor discrepancy.
- Power is quoted at α/8 (worst case, primary) and at α/3 (if five others are confirmed first).

### H1 — Speed class (Q1)
Same host and runtime (M5 Max, MLX 0.31.2 / mlx-lm 0.31.3): Kolibri 4-bit decodes at batch 1 at least 0.75 × as fast as Gemma 4 26B-A4B 4-bit.
*Null:* K4/G4 batch-1 decode ratio ≤ 0.75.

**Measurement.**
- K4 and G4 are both resident (≈ 56 GiB), with one untimed warm-up each.
- 10 blocks. In each block, the order of K4 and G4 is randomised (seed 36).
- One run is the first 1,024 tokens of the model's own tokenisation of `exp_007_hardware_comparison/fixtures/padding/pad_4k.txt`, loaded by its real name (hard fail if missing), with a fresh cache, greedy decoding, EOS masked and exactly 256 generated tokens. We record mlx-lm `generation_tps`, which excludes prefill.
- 30 s idle between runs; power mode, power source and `thermalState` are logged per block.
- Statistic: the mean over blocks of ln(tps(K4)/tps(G4)), i.e. the geometric-mean ratio. One-sample t-test on the 10 block log-ratios (df = 9): p = 1 − F_t9((mean − ln 0.75)/(sd/√10)), p_rev = F_t9((mean − ln 0.75)/(sd/√10)); the 95 % interval uses t₉,₀.₉₇₅.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05. REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (ratio < 0.75). INCONCLUSIVE otherwise.

**Prediction.** ≈ 0.85 with the default head policy. K4 reads ≈ 2.2 GB/token at batch 1 (≈ 2.0 GB of weights plus 0.2 GB of fp32 router weights) vs ≈ 2.2 GB for G4, but Kolibri has 50 layers to Gemma's 30, so more kernel dispatches. With the vendor-faithful head (fp32 `lm_head`, +1.13 GB/token) the prediction is ≈ 0.62, and H1 would likely be REFUTED by the head policy alone; the sign-off records which build ran.

**Power** (per-block log-ratio SD 0.042, i.e. 3 % per run): 1.00 at a true 0.9; 0.93 / 0.98 (α/8 / α/3) at 0.80; 0.47 / 0.67 at 0.78. At SD 0.06: 0.62 / 0.80 at 0.80.

**Descriptive.** Prefill tok/s at 4,096 tokens (ratio); K8/G8; the Qwen arms; aggregate decode at B ∈ {1, 2, 4}.

### D1 — Fit on a 64 GB node (Q1, Q5; deterministic check, outside Holm)
Kolibri 4-bit, as the only resident model, peaks at ≤ 46.66 GiB of MLX memory with a 64k-token context: 0.9 × the 51.84 GiB Metal working-set limit measured on miktam-mini.
*Null:* peak at 64k > 46.66 GiB.

**Measurement.** K4 alone. Per cell: `mx.clear_cache()`, `mx.reset_peak_memory()`, then prefill of exactly N Kolibri tokens of an exact-length prefix of `pad_120k.txt` (real name), `prefill_step_size` 2048, then 512 greedy tokens, then `mx.get_peak_memory()`; prefill tok/s is recorded too. N ∈ {32,768; 65,536}, 3 reps each.

**Rule.** CONFIRMED if median peak(64k) ≤ 46.66 GiB. REFUTED if median peak(64k) > 51.84 GiB (does not fit at 64k even with zero headroom). INCONCLUSIVE ("fits at 64k only without the 10 % headroom") otherwise. peak(32k) is descriptive.

**Prediction.** Default head policy: 41.0 GiB of weights (incl. the fp32 router) + 1.25 GiB of KV + ≤ 1.5 GiB transient ≈ 43.7 GiB at 64k. Vendor-faithful head: ≈ 45.2 GiB.

**Transfer** to the M4 Pro is an assumption; the mini is not used. "Fits" means a dedicated node: the mini keeps Ollama models resident for the CasaSol bot, and they would have to be unloaded. K8 (77.4 GiB of weights) not fitting 64 GB is arithmetic, not a test.

### H2 — Scorecard (Q2)
Through the gated port at MLX 8-bit, Kolibri lives up to its post-training scorecard on the public, deterministically scored rows: its mean shortfall against the vendor's values is less than 4 pp.
*Null:* mean D ≤ −4 pp.

**Rows** (vendor values: Table 28, p. 100/189, and Table 29, p. 101/189; frozen with N_v, n_v and page in `analysis/vendor_values.json`):

| Row | Vendor | Label | Items | Our protocol |
|---|---|---|---|---|
| GPQA Diamond EN | 84.3 | V | 197 = eval-framework set; 198 as sensitivity | `GPQA_DIAMOND_COT_V2` prompt, option shuffle, `tulu_answer_v2(4)` |
| GPQA Diamond DE | 81.3 | V | 198 | `GPQA_ELLAMIND_DIAMOND_COT_DE`, `tulu_answer_de` |
| MMLU-Pro CoT EN | 80.0 | R | MMLU-ProX-Lite EN, n_M, post-stratified | `MMLU_PRO_COT_V2` prompt, `tulu_answer_v2(10)` |
| MMLU-ProX CoT DE | 75.5 | R | Lite DE, n_M, post-stratified | eval-framework German Tülu CoT prompt with A–J, German lenient extractor extended to A–J |
| IFBench loose-prompt | 78.1 | R | 300 | official allenai checker |

The row set is fixed at these 5 rows (4 under P10). AIME 2026 EN 96.0 (V) and DE 90.0 (R, our frozen German NeMo-Skills wrapper) form a secondary 7-row estimate if B1 runs; it is reported as an estimate and 95 % CI only, with no verdict, and its per-row D is descriptive.

**Measurement.**
- D_r = ours_r − vendor_r; D̄ = unweighted mean over rows.
- MMLU rows are **post-stratified**: each category's mean is weighted by that category's share of the full MMLU-ProX test split of that language, counted at manifest build by the frozen rule and recorded in the plan amendment. The unweighted Lite mean is descriptive. `vendor_values.json` also records the MMLU-Pro 12,032 category counts for comparison.
- Uncertainty: an item bootstrap within rows, stratified by category for the MMLU rows. For the full-set rows (GPQA EN/DE, IFBench) this is conservative, because it adds between-item variance for items the vendor also ran. If B10 runs, the GPQA rows use the two-pass mean, and the variance of each GPQA row is estimated from the passes as Σ(x₁ − x₂)²/(4n²) instead of by resampling items. The vendor's avg@N term is drawn per resample as N(0, p_v(1 − p_v)/(N_v · n_v)), with p_v the vendor value, N_v = 8 (GPQA), 5 (IFBench) and 16 (AIME secondary), and n_v the vendor's item count; it is 0 for MMLU, where the vendor ran the full sets once.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D̄ ≤ −4). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D̄ ≥ −4). INCONCLUSIVE otherwise.

**H2_detail (pre-registered).**
- Per-row D with CI, V/R label, the per-model truncation rate and the result excluding truncated items; every row with |D_r| > 5 pp named; the GPQA-198 sensitivity.
- **Protocol control (from E8):** the peers' mean D over the shared rows (MMLU EN, MMLU DE, IFBench; G8 and Q36-8) with 95 % CI, and the difference-in-differences DiD = D̄_K,shared − mean D_peer,shared with 95 % CI.

**Power** (k = 1; post-stratification design effect 1.11; α/8 / α/3):

| n_M | SE(D̄) | True D = 0 | D = −1 | D = −2 | D = −3 |
|---|---|---|---|---|---|
| 588 | 1.09 pp | 0.88 / 0.94 | 0.60 / 0.74 | 0.25 / 0.39 | 0.06 / 0.11 |
| 406 | 1.14 pp | 0.84 / 0.92 | 0.55 / 0.69 | 0.23 / 0.35 | 0.05 / 0.11 |
| 294 | 1.20 pp | 0.80 / 0.89 | 0.50 / 0.64 | 0.20 / 0.32 | 0.05 / 0.10 |
| 196 | 1.31 pp | 0.71 / 0.83 | 0.42 / 0.57 | 0.17 / 0.28 | 0.04 / 0.09 |
| 154 | 1.39 pp | 0.65 / 0.78 | 0.37 / 0.51 | 0.15 / 0.25 | 0.04 / 0.08 |
| 154, 4 rows (P10) | 1.57 pp | 0.52 / 0.66 | 0.28 / 0.41 | 0.11 / 0.20 | 0.03 / 0.07 |

**Under the drift we expect (1–2 pp, from the int8 grid vs FP8-QAT and three reconstructed rows), H2 is more likely INCONCLUSIVE than CONFIRMED.** REFUTED needs a true shortfall well beyond the margin: power 0.88 at D = −8 (n_M = 588), 0.71 at n_M = 196.

**Why −4 pp.** It is one GPQA item in 25, averaged over rows. Our 8-bit is a different grid from the vendor's FP8-QAT, and three rows are reconstructions, so a 1–2 pp drift is not a failure to live up to the scorecard. The margin is an open sign-off choice (−3 pp lowers power further; −5 pp raises power at D = −2 to ≈ 0.40 / 0.56 at n_M = 196 but weakens the claim).

### H3 — Knowledge gap, English and German (Q3, Q4)
Kolibri 8-bit scores below the mean of Gemma 4 8-bit and Qwen3.6 8-bit on MMLU-ProX-Lite, pooled over parallel English and German items. Vendor gaps: −4.4 EN, −6.0 DE, −5.2 pooled (p. 100/189).
*Null:* K8 − peer mean ≥ 0.

**Measurement.** The same item ids in EN and DE, category-balanced. Per item d = K − (G + Q)/2. Paired item bootstrap stratified by language and category.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D ≥ 0). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D ≤ −2 pp, i.e. REFUTED when the gap is clearly smaller than 2 pp). INCONCLUSIVE otherwise. By construction CONFIRMED and REFUTED could both hold only for SE < 0.45 pp, which the design cannot reach; REFUTED is evaluated only if not CONFIRMED.

**Descriptive.** Per-language gaps; the post-stratified gap next to the vendor's −5.2 (−4.4 EN / −6.0 DE); each model's EN→DE drop (vendor: Kolibri 4.5, Gemma 4 3.4, Qwen3.6 2.4).

**Power** (SD of d ≈ 0.40; α/8 / α/3):

| n_M | at the vendor's −5.2 | at −3 |
|---|---|---|
| 588 | 0.98 / 0.99 | 0.53 / 0.67 |
| 406 | 0.89 / 0.94 | 0.36 / 0.50 |
| 294 | 0.74 / 0.85 | 0.25 / 0.38 |
| 196 | 0.53 / 0.67 | 0.16 / 0.26 |
| 154 | 0.41 / 0.56 | 0.12 / 0.21 |

This is the hypothesis most sensitive to throughput, and that is stated now.

### H4 — Instruction following against Qwen3.6 (Q3)
Kolibri 8-bit is ahead of Qwen3.6 8-bit on IFBench loose-prompt accuracy. Vendor: 78.1 vs 66.1, +12.0 pp (p. 101/189).
*Null:* K8 − Q36-8 ≤ 0.

**Measurement.** 300 IFBench_test prompts, paired, k = 1. Official checkers (loose-prompt; strict also reported) on the post-reasoning answer, run in IFBench's own environment on the mini.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D ≤ 0). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D ≥ +4 pp). INCONCLUSIVE otherwise. REFUTED is evaluated only if not CONFIRMED.

**Power** (SD of the paired difference ≈ 0.55, SE 3.2 pp; α/8 / α/3): 0.90 / 0.95 at +12; 0.27 / 0.41 at +6.

**Framing (pre-registered).** H4 compares Kolibri with the peer that has the lowest vendor IFBench score (66.1). Every statement of H4 in the post reports Kolibri vs Gemma 4 on IFBench next to it (vendor: 78.1 vs 79.9, −1.8). IFBench is a selection row with an RL environment behind it (C11).

### H5 — German tokenizer (Q3)
On German web text, Kolibri's tokenizer packs at least 1.15 × the UTF-8 bytes per token of both the Gemma 4 tokenizer and the Qwen3.6 tokenizer. The report implies 4.90/4.13 = 1.186 and 4.90/4.17 = 1.175 (Fig. 5, p. 10–11/189). The threshold is set so that the test bears on the claimed ≈ 1.18: the report's own figures guarantee only ≥ 4.90/4.35 = 1.126 against any compared tokenizer.
*Null:* either ratio ≤ 1.15.

**Measurement.**
- Corpus: the first 5,000 documents of the FineWeb-2 deu_Latn test parquet in file order.
- Bytes/token = Σ UTF-8 bytes / Σ tokens, with `add_special_tokens = False`, no template and no normalisation.
- Tokenizers: Kolibri through raw `tokenizers` (id parity checked in G0); Gemma 4 and Qwen3.6 from the pinned peer folders (byte parity with upstream checked at build time). Qwen3.8 is hashed and, if identical to Qwen3.6, reported as one.
- Document bootstrap. Intersection–union test for the claim: p = max(p_Gemma, p_Qwen). For REFUTED (either ratio clearly below 1.15), Bonferroni over the two: p_rev = min(1, 2 × min(p_rev,Gemma, p_rev,Qwen)).

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: either ratio ≤ 1.15). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: both ratios ≥ 1.15). INCONCLUSIVE otherwise.

**Sub-verdict** on Kolibri's absolute German bytes/token, by point estimate with the CI shown: report-consistent if the estimate is in [4.80, 5.00]; card-consistent (≈ 4.7) if it is in [4.60, 4.80); neither otherwise.

**Power** (CI ≈ ±1 %): 0.96 (α/8) at the vendor-implied 1.175; 0.22 at 1.16. CPU only, minutes. The mini re-runs it as a check.

**Descriptive.**
- The EN ratio on our own EN gate texts plus MMLU-ProX-Lite EN question text, labelled "not FineWeb" (FineWeb EN is optional).
- A digit-dense German subset (≥ 5 % digits; Kolibri splits every digit).

**Disclosure: H5 is not blind.** H5 is deterministic: no model runs, only three tokenizers over a pinned file. The kit's tests and its end-to-end dry run (`tools/dry_run.py`) compute it on the mini with the real tokenizers and the pinned FineWeb-2 shard, and they did so during the build on 2026-10-03/04, after the 1.15 threshold above was written into the draft and before this file was pushed. The verdict those runs printed was CONFIRMED. The threshold was not changed after that. Read H5 as a check of a deterministic vendor claim, not as a blind prediction.

### H6 — Closed-book correct-answer deficit (Q4)
Without documents, Kolibri 8-bit answers RGB's questions correctly at least 10 pp less often than the mean of Gemma 4 8-bit and Qwen3.6 8-bit. Vendor RGB Closed-Book: 51.0 vs 79.0 / 79.0, −28 pp (p. 101/189).
*Null:* K8 − peer mean ≥ −10 pp.

**Measurement.**
- 400 questions: RGB `en.json` 300 + `en_fact.json` 100.
- RGB's own closed-book path (`passage_num = 0`): the instruction verbatim with empty documents and no system prompt. Effort high / thinking on, vendor sampling.
- **Correct** iff our checkanswer (the RGB rule) returns 1 for every answer slot on the post-reasoning answer **and** the case-sensitive RGB string "insufficient information" is absent. The frozen abstention lexicon only splits non-correct items into abstain and wrong (E2); it never changes "correct".
- Paired item bootstrap.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D ≥ −10). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D ≤ −10). INCONCLUSIVE otherwise.

**Wording (pre-registered, C22).** Without the forced-answer cells (P8–P10), H6 is published as a "closed-book correct-answer rate", and the words "recall", "knowledge" and "knows less" are not used. With them, "knows less" requires the forced-answer deficit's 95 % upper bound < −10 pp (E2).

**Descriptive.** The en_fact-100 subset next to the vendor's 51.0 / 79.0 / 79.0; the four-way split (E2).

**Power** (SE ≈ 2.25 pp; conditional on the vendor gap replicating on our items, C14): ≈ 1.0 at the vendor's −28; 0.39 / 0.54 (α/8 / α/3) at a true −15.

### H7 — 4-bit task cost (Q5)
Squeezing Kolibri to 4-bit costs less than 3 pp. K4 is non-inferior to K8, paired on identical items, averaged over the rows K4 runs: MMLU-ProX-Lite EN and DE (the same n_M items), IFBench 300 and RGB closed-book 400. GPQA-D EN and DE are added if B8 is in the plan amendment.
*Null:* mean (K4 − K8) ≤ −3 pp.

**Measurement.** Unweighted mean of the row differences; paired item bootstrap stratified by row (and by category within the MMLU rows).

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D̄ ≤ −3). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D̄ ≥ −3). INCONCLUSIVE otherwise.

**Power** at a true cost of 0 (sampling at T = 1 makes paired discordance large; α/8 / α/3):

| n_M | at 0 | at −1 |
|---|---|---|
| 588 | 0.70 / 0.81 | 0.31 / 0.45 |
| 406 | 0.61 / 0.75 | 0.26 / 0.39 |
| 294 | 0.53 / 0.67 | 0.22 / 0.34 |
| 196 | 0.41 / 0.56 | 0.16 / 0.27 |
| 154 | 0.34 / 0.49 | 0.14 / 0.23 |

H8 is the noise-free complement.

**Descriptive.** K4 vs G8 on the Tier-A task rows; K4 vs G4 on speed (H1), fit and logit fidelity (H8) only, since G4 runs no task cells. These confound model with bits and are labelled so.

### H8 — 4-bit logit fidelity against the peers, on the six gate texts (Q5)
On our six plain gate texts, 4-bit costs Kolibri no more fidelity than it costs the peers: Kolibri's KL(8-bit ‖ 4-bit) per UTF-8 byte is ≤ 1.5 × the median of Gemma 4's, Qwen3.6's and Qwen3.8's, all built with the same affine g64 recipe. The estimand is these six texts (2 EN, 4 DE; blog, legal and web prose), not English or German text in general.
*Null:* ratio ≥ 1.5.

**Measurement.**
- Every model teacher-forces the same bytes (the decoded text of T1–T6, ≈ 43 kB) with its own tokenizer.
- KL is computed from bf16-rounded logits for every model (like for like, C6), upcast to fp32.
- Normalisation: Σ KL / Σ bytes. Per-token normalisation is a pre-registered sensitivity, published next to it, because per-byte KL folds Kolibri's denser German tokenizer into the ratio.
- Blocks: 8 byte-aligned blocks per text, split at whitespace; each token belongs to the block holding its first byte. **Stratified block bootstrap**: 8 blocks are resampled within each text, text weights fixed at their byte shares, joint across models.
- Kolibri's K8 log-probs are dumped to `$EXP036_WORK` and then compared with K4. Each peer's 8-bit and 4-bit models are loaded together.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: ratio ≥ 1.5) **and** Kolibri's per-text ratio is ≤ 1.5 on ≥ 5 of the 6 texts. REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: ratio ≤ 1.5). INCONCLUSIVE otherwise. Per-text ratios are published. If H8 is CONFIRMED per byte but the per-token ratio exceeds 1.5, the post says so.

**Prediction.** ≈ 1.0. **Power** (within-block SD of the log ratio 0.3): 0.99 at a true 1.0 with a between-text SD of 0.15, 0.88 with 0.3; 0.85 / 0.52 at a true 1.2 (α/8).

**Descriptive.** KL with Kolibri's fp32 logits; top-1 agreement; ΔNLL per byte; K8 and K4 KL to the fp32 reference (from the gate).

### Holm family
H1, H2, H3, H4, H5, H6, H7, H8 (m = 8, frozen at the plan amendment). D1 is a deterministic check with its own rule, outside the family. The verdict file reports, for each H: raw p and p_rev, both Holm-adjusted values, the Holm step, the Monte Carlo SE, the bound at the adjusted level, the 95 % interval, the truncation sensitivity and any E8 flag.

---

## Exploratory and descriptive (predictions recorded now; no verdicts)

| ID | What | Prediction recorded 2026-10-03 | Source of the data |
|---|---|---|---|
| E1 | Public-only composite: the unweighted mean over the vendor rows that are public and deterministically scored, computed only over rows every compared model ran. A composite that does not include AIME and GPQA for the peers is labelled "excludes the rows where the vendor reports Kolibri ahead of these peers (AIME, GPQA)", and that label also appears in the post's Q3 answer. Includes the three-group split of E9, leave-one-row-out, and EN and DE sub-composites. | Vendor table: 10 public rows K 75.6 / G 80.8 / Q36 80.4 / Q38 81.8 — Kolibri last. Selection 7 rows: K 83.6 / G 83.2 / Q36 81.7. RGB 3 rows: K 56.9 / G 75.3 / Q36 77.5. Tier-A 6-row composite (MMLU EN/DE, IFBench, RGB × 3): K 67.4 / G 78.6 / Q36 77.5. We predict K8 is last on every composite that contains RGB CB and FC, and within ±1.5 pp of G8 on the selection rows. | Tier A + B2/B3/B6 |
| E2 | Says so? Knows when forced? H6 outputs split four ways (correct / abstain / wrong / truncated). Δwrong = K8 wrong-rate − peer mean; A = K8 abstentions / non-correct, non-truncated. The forced-answer cells ("Answer with a short phrase. Always give your best answer.") separate knowledge from abstention. Pre-registered reading: "says so" requires the 95 % upper bound of Δwrong ≤ +3 pp **and** the 95 % lower bound of A ≥ 0.40; "guesses" requires the 95 % lower bound of Δwrong > +3 pp; otherwise "unclear". "Knows less" requires the forced-answer deficit's 95 % upper bound < −10 pp. | Δwrong ≤ +3 pp; A ≥ 0.40 (vendor AA-Omniscience non-hallucination 44.0 vs Gemma 14.3 / Qwen3.6 56.7, p. 101/189). Under forced answering the K8 deficit shrinks by ≥ 5 pp relative to H6's. | Tier A (forced cells P0–P7) |
| E3 | RGB Negative (300; noise rate 1.0, 5 negative passages) and Fact-Check (en_fact 100, counterfactual documents) for K8, G8, Q36-8. Fact-Check is split into corrected / deferred / detected, with a knowledge-gated reading on the items each model answered correctly closed-book. | Negative: K8 within ±6 pp of G8 (vendor 85.6 vs 86.0) and above Q36-8 (79.6). Fact-Check: K8 lowest of the three (vendor 34 vs 61 / 74), with "deferred" its most common failure. | Tier A, P0 only |
| E4 | German reasoning consistency: the share of K8 outputs on DE items (GPQA-DE, MMLU-DE, AIME-DE if B1) whose reasoning segment a frozen deterministic stopword tagger labels German. | ≥ 95 % (vendor 99 % on AIME-DE after RL, Fig. 34, p. 79/189). | Tier A, B1 |
| E5 | Effort curve on the released model: K8 at none / low / medium on GPQA-D EN (high from the main run). | Tokens none < low < medium < high; acc(high) − acc(medium) ≤ 3 pp; acc(low) − acc(none) ≥ 10 pp. The SFT checkpoint showed +0.7 and a 26.5-pp none gap, p. 63–64/189; "none" is outside the RL effort mix, p. 183/189. | B7 (none from B5) |
| E6 | Context ladder and prefill cliff: K4, K8, G4 at 4k / 15k / 32k / 64k (+120k for K4), from `pad_4k.txt`, `pad_15k.txt` and exact-length prefixes of `pad_120k.txt`, by real name, 3 reps, fresh cache. | No cliff: prefill ms/token at 32k and 64k ≤ 2 × the 15k value (FLOP arithmetic: 1.27 × and 1.65 ×; 120k ≈ 2.4 ×). KV slope 20,480 B/token ± 15 %. | B4 (K4 32k/64k prefill also from D1) |
| E7 | Operator metrics: tokens and seconds per correct answer; correct answers per hour at batch 1; German chars/s = batch-1 decode × chars/token; aggregate tok/s at the chosen B; truncation and parse-failure rates. | Kolibri's German chars/s at batch 1 ≥ 1.0 × G4's (the tokenizer offsets the decode deficit). | all |
| E8 | Relative replication and peer positive control: each peer's local − vendor score per row (D_peer), computed exactly as H2's rows (post-stratified MMLU; vendor peer values in `vendor_values.json`); DiD = D_K − mean D_peer per row; the flag rule of the common rules; the H2 headline rule. | \|D_peer\| ≤ 5 pp on MMLU and IFBench. | Tier A (+ B2/B3/B6) |
| E9 | Three groups, reported separately: selection rows (GPQA, AIME, MMLU, IFBench), RGB closed-book and fact-check, RGB Negative (in-distribution with an RL abstention environment, C11). Mean D_K and Kolibri-vs-peer gaps per group. | Kolibri's standing against the peers is better on the selection rows than on RGB CB/FC. | all |
| E10 | Qwen3.8-27B 8-bit (dense) on GPQA-D DE. | Q38 > K8 (vendor +6.8). | B9 |
| E11 | Gate diagnostics: KL contributed by the quantised head and embeddings; routing agreement and KL with an 8-bit router; layer 0–1 routed-expert intervention rate; QK-norm bound max\|γ\| × √128; K8 and K4 KL to the fp32 reference. | Head + embedding KL < 0.005 nats/token at 8-bit; 8-bit router agreement ≥ 97 %; layer 0–1 intervention ≥ 95 % (vendor 98.5 % at pre-training, p. 137–140/189); bound < 448 (p. 12/189). | S1 |
| E12 | Workload mode: K8 at effort none on GPQA-D, MMLU, IFBench and RGB closed-book, labelled "effort none — outside Kolibri's RL effort mix (p. 183/189)". | ≥ 10 pp below effort high on GPQA-D and MMLU. | B5 |
| Control C1 | Greedy answer-flip rate from batching: K8 at effort none, greedy, 100 MMLU-ProX full (non-Lite) EN items at the memory-rule B for the MMLU cell at the initial cap (computed at preflight, recorded in the C1 file) vs B = 1. | ≤ 2 % of extracted answers flip. | S1 |

Exploratory results are labelled as such. Anything computed after the outputs were seen and not listed here is labelled post hoc.

---

## Arms

| Arm | Model | Total / active params | Bits (affine g64) | Revision | Runtime | Template / thinking | Sampling (T / top-p / top-k) | Max tokens |
|---|---|---|---|---|---|---|---|---|
| K8 | Kolibri-1 (our conversion) | 78.1B / 3.46B (card: 78,103,074,560 / 3,457,573,120) | 8; router (fp32 weight), expert_bias and norms unquantised; embed and head at 8 bits by default; fp32 router and head logits | BF16 `7a8f290e` → `Kolibri-1-MLX-8bit-g64` (manifest sha in the gate record) | mlx 0.31.2 / mlx-lm 0.31.3, `port/kolibri1.py` via `model_file` | vendor jinja; `reasoning_effort="high"` (E5, E12: none / low / medium) | 1.0 / 0.97 / 128 | caps above |
| K4 | Kolibri-1 (our conversion) | as K8 | 4 (same policy) | → `Kolibri-1-MLX-4bit-g64` | as K8 | as K8 | as K8 | as K8 |
| G8 | Gemma 4 26B-A4B-it | ≈ 25.2B / ≈ 3.8B (exact from config at preflight) | 8 (router.proj 8-bit) | mlx-community `33c6d237` | mlx-lm `gemma4`, text-only | upstream template; `enable_thinking=True`; thought channel `<\|channel>thought … <channel\|>` | 1.0 / 0.95 / 64 | as K8 |
| G4 | Gemma 4 26B-A4B-it | as G8 | 4 | `0d77464e` | as G8 | as G8 | greedy (speed cells), teacher-forced (H8) | H1, H8, E6 only |
| Q36-8 | Qwen3.6 35B-A3B | ≈ 34.7B / ≈ 3B | 8 (mlp.gate, shared_expert_gate 8-bit) | `e06a74e6` | mlx-lm `qwen3_5_moe`, text-only | upstream template; `enable_thinking=True` (`<think>…</think>`) | 1.0 / 0.95 / 20 | as K8 |
| Q36-4 | Qwen3.6 35B-A3B | as Q36-8 | 4 | `38740b84` | as Q36-8 | as Q36-8 | greedy / teacher-forced | speed (descriptive), H8 only |
| Q38-8 | Qwen3.8 27B dense | 27B / 27B | 8 | `815b83c0` | mlx-lm `qwen3_5`, text-only | upstream template; `enable_thinking=True`; template default effort | 1.0 / 0.95 / 20 | B9 only |
| Q38-4 | Qwen3.8 27B dense | as Q38-8 | 4 | `10c35caa` | as Q38-8 | as Q38-8 | greedy / teacher-forced | speed (descriptive), H8 only |

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; every path comes from assets.json under `$EXP036_MODELS`. One item per sequence, through mlx-lm BatchGenerator with one vLLM-order sampler per cell and per-sequence `max_tokens`. No Ollama, no GGUF and no server. Total and active parameters are recomputed from each config at preflight and written to the version record.

## Task sets

| Set | Source @ revision (assets.json) | n | Language | Scoring | Publishable? |
|---|---|---|---|---|---|
| GPQA Diamond EN | Idavidrein/gpqa @ `83022cef`, `gpqa_diamond.csv` | 198 (197 primary) | EN | V: eval-framework `GPQA_DIAMOND_COT_V2` prompt, option shuffle and `tulu_answer_v2(4)` on the post-reasoning text | ids, hashes, letters, scores |
| GPQA Diamond DE | ellamind/gpqa-multilingual @ `bc70ca17`, `deu`, `is_diamond` | 198 | DE | V: `GPQA_ELLAMIND_DIAMOND_COT_DE` + `tulu_answer_de` | ids, hashes, letters, scores |
| GPQA main (pilot) | the same two sources, rows whose Record ID is not in `gpqa_diamond.csv` | 8 + 8 per arm | EN, DE | pilot only | no |
| MMLU-ProX-Lite EN/DE | li-lab/MMLU-ProX-Lite @ `e82aafb9` | n_M per language, parallel ids, category-balanced | EN, DE | R: EN `MMLU_PRO_COT_V2` + `tulu_answer_v2(10)`; DE German Tülu CoT A–J + German lenient extractor (A–J) | ids, hashes, gold, raw outputs, scores |
| MMLU-ProX full (pilot, C1, weights) | li-lab/MMLU-ProX @ `8e6106a6`, non-Lite ids | 8 + 8 per arm; 100 for C1; full test split category counts for post-stratification | EN, DE | pilot / control / weights | ids, hashes, counts |
| AIME 2026 EN | math-ai/aime26 @ `79037aeb` | 30 | EN | V: eval-framework `AIME2026` NeMo-Skills prompt and boxed extractor | ids, hashes, gold, raw outputs, scores |
| AIME 2026 DE | ellamind/aime26-multilingual @ `3c8bc18f`, `deu` | 30 | DE | R: frozen German NeMo-Skills wrapper (`tasks/prompts/aime_de.txt`), same extractor | ids, hashes, scores |
| IFBench | allenai/IFBench_test @ `2e8a48de` + allenai/IFBench @ `1c40f0c1` | 300 | EN | R: official checkers, loose-prompt (strict reported); scored on the mini | ids, hashes, raw outputs, scores |
| RGB closed-book | chen700564/RGB @ pin, `en.json` + `en_fact.json` | 400 | EN | R: `passage_num = 0` path; our checkanswer; RGB rejection string; frozen lexicon for abstain vs wrong | ids and hashes only |
| RGB closed-book, forced | as above | 400 | EN | frozen forced-answer instruction (`tasks/prompts/rgb_forced_en.txt`); our checkanswer | ids and hashes only |
| RGB Negative | `en.json`; noise 1.0; 5 negative passages chosen by RGB's `processdata` rule with `random.seed(2333)`, indices frozen in the manifest | 300 | EN | R: RGB system and instruction verbatim (rendered from local files); rejection string plus lexicon | ids and hashes only |
| RGB Fact-Check | `en_fact.json`; counterfactual passages; `instruction_fact.yaml` | 100 | EN | R: corrected / deferred / detected rules | ids and hashes only |
| RGB pilot | `en_int.json` | 16 + 8 | EN | pilot only | no |
| FineWeb-2 deu_Latn | HuggingFaceFW/fineweb-2 @ pin | 5,000 docs | DE | bytes/token, chars/token | counts |
| Gate text | `gate/texts/` | T1–T8: 8 × 1,536 tokens; T9: 16,384 | EN, DE | gate | T1–T4, T7–T8 yes; T5–T6 and T9 indices and hashes only |
| Speed / context fixtures | `exp_007_hardware_comparison/fixtures/padding/pad_4k.txt`, `pad_15k.txt`, `pad_120k.txt` (exact-length prefixes for 32k, 64k and 120k) | — | EN | timing, peak memory | yes |

---

## What counts as evidence

- Only cells run under the frozen plan amendment count. Pilot outputs never do.
- Every verdict is computed by `analysis/verdicts.py` from the raw JSONL. The mini re-scores every set, public and withheld, from the raw JSONL and the private mirror, and its scores must match the mbp's `scores/*.jsonl` byte for byte; `verdicts.py` refuses to run otherwise. No result file is ever rebuilt from console output. If a run crashes after generating, the raw JSONL is kept and the verdicts are recomputed from it.
- Scoring is deterministic code: vendored regexes, exact match, our RGB rule and the official IFBench checkers. There is no LLM judge, and no cloud call unless Andrei explicitly asks.
- The abstention lexicon is frozen at the plan amendment at the latest. `run.py` refuses S2 unless the `scorers/` tree sha equals the value in the plan amendment. A later change produces a separately labelled post hoc score; the frozen-lexicon verdict stays primary.
- Confirmatory verdict code (Tier 1: H1–H8, D1, stats) is frozen in this pre-registration. Exploratory analysis code (Tier 2: E1–E12 analyses, tables) is frozen by its own numbered amendment, pushed from the mini before `scorers/score_all.py` first runs and before any B4 ladder cell starts; both refuse to run otherwise.
- Quality numbers come only from a gated port: a PASS record bound to the port's and the artefact's hashes.
- Truncated, unparseable or unclosed output is wrong and is never re-asked. Its rate is reported.
- Vendor numbers and our numbers stay in separate columns. Vendor numbers are cited "p. N/189".
- Every absolute speed or memory figure is labelled "M5 Max, MLX 0.31.2".
- INCONCLUSIVE and NOT RUN are published as such. `aborted/` is never deleted. Nothing from `$EXP036_PRIVATE` is ever moved or copied into the repo; a torn private line is kept in `$EXP036_PRIVATE/aborted/`, and the repo note records only its sha256 and byte count.

## Evidence layout

```
exp_036_kolibri_local_eval/
├── results/
│   ├── preflight_<UTC>.json          chip, hw.model, memory, macOS, mx.device_info, iogpu, power mode, disk, versions, git identity, asset revisions (+ sha256 in --deep); a fixed host label, never a hostname, serial or UUID
│   ├── version_record_<UTC>.json     exp_035 shape: OS, Python, mlx, mlx-metal, mlx-lm, numpy, tokenizers, safetensors, pip freeze sha, git commit + dirty flag, every tree sha of the hash table, converted manifests, asset revisions, Metal limits
│   ├── convert/convert_{8,4}bit_<UTC>.json
│   ├── gate/gate_<UTC>.json (+ gate_<UTC>_layers.csv, gate_<UTC>_mutants.json)
│   ├── peers_<UTC>.json
│   ├── bench/{fit,speed,speed_desc,c1,ladder}_<UTC>.jsonl
│   ├── tokenizer_<UTC>.json, kl_8v4_<UTC>.json
│   ├── manifests_<UTC>.json                            sha256 of every built manifest
│   ├── pilot/<UTC>/*.jsonl, pilot_summary_<UTC>.json   (never scored for accuracy)
│   ├── plan_fixed_<UTC>.json, AMENDMENT_<k>_<UTC>.md
│   ├── raw/<session>/<arm>/<task>_<effort>.jsonl       (withheld sets: text replaced by text_sha256; full text in $EXP036_PRIVATE)
│   ├── scores/<arm>/<task>_<effort>.jsonl              (item-hash keyed: extracted, category, correct, truncated; no free text for withheld sets)
│   ├── rescore_mini_<UTC>.json                         mini re-score vs mbp scores, per file
│   └── verdicts_<UTC>.json                             (exp_023 shape: ts, hardware, config, manifests, summary, verdicts{H*, H*_detail, D1, E*, C1})
├── tasks/manifests/                  ids, hashes, category and public gold; built on the mbp in S1
├── tools/withheld_shingles.sha256    hashed shingles of withheld text, built on the mbp in S1 (for the leak check)
├── evidence/withheld_manifest.jsonl   sha256 of every withheld raw file (GPQA EN/DE, RGB, AIME-DE)
├── aborted/<UTC>-<what>/ + NOTE.md   never deleted
└── version_record.json               the final run's record, copied from results/
```

---

## Sessions & budget

**Planning assumptions** (the pilot simulation replaces all of them):

| Model | Aggregate decode at chosen B (nominal / pessimistic / adverse) | Mean completion tokens at high effort (nominal / pessimistic / adverse) |
|---|---|---|
| Kolibri | K8 180 / 120 / 100 tok/s; K4 240 / 160 / 140 | GPQA 5k / 8k / 10k; MMLU 2k / 3k / 3.5k; IFBench 1.5k / 2.5k / 3k; RGB CB 0.6k / 1k / 1.2k; forced 0.4k / 0.8k / 1k; AIME 14k / 20k / 24k |
| Gemma 4 | G8 220 / 150 / 140 | GPQA 4k / 6k / 7k; MMLU 1.5k / 2.5k / 2.8k; IFBench 1.2k / 2k / 2.2k; RGB 0.6k / 1k / 1k; AIME 12k / 15k / 18k |
| Qwen3.6 | Q36-8 250 / 160 / 150 | GPQA 7k / 9k / 10k; MMLU 3k / 3.5k / 4k; IFBench 2k / 3k / 3.2k; RGB 0.9k / 1.2k / 1.3k; AIME 15k / 18k / 20k |
| Qwen3.8 | Q38-8 100 / 60 / 55 | GPQA 7k / 9k / 10k |

Prefill: 1,500 / 800 / 700 tok/s for the MoE arms. RGB prompts with documents are ≈ 1,500 tokens. The adverse column is "modestly longer reasoning and some thermal throttling", close to what the vendor's RL budgets imply as typical (reasoning budget 65,536 and math 131,072 tokens, p. 172, 183/189).

**S1** (machine time; the nominal and pessimistic estimates are deliberately padded where the arithmetic is short):

| Block | Nominal h | Pessimistic h | Arithmetic |
|---|---|---|---|
| Preflight `--deep` | 0.1 | 0.3 | sha256 of 303 GB ≈ 1–2 min at 2.5–5 GB/s; padded for a cold cache |
| Convert K8 + K4 | 0.5 | 1.5 | 2 × 156 GB read, 127 GB written; dominated by mlx_lm.convert's load–quantise–save, unmeasured |
| Peer check | 0.4 | 0.6 | 6 loads, NLL and KL on T1–T6, batched-path parity for G8 and Q36-8 |
| Phase 0 gate | 1.8 | 3.0 | fp32 reference pass on T1–T8 (12,288 tokens, 156 GB streamed, ≈ 1e14 FLOP) and T9 (16,384 tokens); bf16-emulation pass; seven reference mutants in one pass on T3; a reference pass on the G5 continuations; 50 × 3 per-layer checks; G2q for K8 and K4; K8/K4 end to end; generation, batch, noise-floor and behaviour checks |
| D1, H1, descriptive speed cells, C1 | 0.8 | 1.1 | D1: 2 sizes × 3 reps. H1: 10 blocks × 2 runs × (≈ 3 s + 30 s idle) + warm-up. Descriptive: 6 arms × 5 reps. C1: 2 × 100 short greedy answers. |
| H5 tokenizer | 0.1 | 0.1 | CPU, 5,000 documents × 3 tokenizers |
| H8 KL | 0.3 | 0.5 | K8 log-prob dump + K4; 3 peers × (8-bit + 4-bit) on ≈ 43 kB |
| Pilot | 1.3 | 1.9 | K8 ≈ 0.3M tokens + 3 arms × ≈ 0.12–0.17M tokens + loads |
| Plan-fix, amendment, commit | 0.1 | 0.1 | — |
| **S1 total** | **5.4** | **9.1** | |

**Ladder totals** (Tier A, projected hours × 1.15; nominal / pessimistic / adverse): P0 25.1 / 56.4 / 74.3 · P1 23.4 / 52.8 / 69.9 · P2 21.3 / 48.3 / 64.0 · P3 18.9 / 43.0 / 57.2 · P4 17.4 / 40.0 / 53.3 · P5 16.0 / 37.0 / 49.4 · P6 15.0 / 34.7 / 46.5 · P7 13.6 / 31.7 / 42.6 · P8 12.7 / 29.4 / 39.6 · P9 11.7 / 27.1 / 36.6 · P10 9.9 / 22.9 / 30.3.

**Tier B** (× 1.15; nominal / pessimistic): B1 1.5 / 3.2 · B2 2.2 / 4.1 · B3 2.9 / 6.1 · B4 1.2 / 1.4 · B5 0.9 / 2.4 · B6 2.9 / 6.1 · B7 2.0 / 4.3 · B8 2.7 / 6.4 · B9 4.5 / 9.6 · B10 3.5 / 8.5.

**Plan picked by the rule** (`analysis/power.py` and `runner/plan_fix.py` reproduce these from the assumptions above):

| Scenario | S1 | B_main | Plan | Tier B | S2 / S3 | Main | Run total |
|---|---|---|---|---|---|---|---|
| Nominal | 5.4 | 31.0 | P0 (n_M 588, forced, Negative and Fact-Check) | B1, B2, B5 | 14.0 / 15.7 | 29.7 | 35.1 h |
| Pessimistic | 9.1 | 30.9 | P8 (n_M 196, no forced cells) | none | 15.8 / 13.5 | 29.4 | 38.5 h |
| Adverse | 9.6 | 30.4 | P10 (n_M 154, no forced cells, H2 on 4 rows) | none | 15.3 / 15.0 | 30.3 | 39.9 h |
| Slightly worse than adverse | | | STOP: Andrei decides on a 4th session by amendment | | | | |

At P10, power is 0.41 / 0.56 for H3 at the vendor's gap and 0.34 / 0.49 for H7 at a true cost of 0 (α/8 / α/3); Andrei signs off knowing this.

**Sessions.**
- S1 is daytime, ≈ 6–9 h of machine time; a gate fix cycle moves the rest of S1 to a later day.
- S2 and S3 are overnight-plus-morning, ≤ 16 h each, over one ordered queue: K8 Tier A first, then the peers' and K4's Tier-A cells in hypothesis order, then the forced and RGB Negative/Fact-Check cells, then Tier B. The split point is fixed in the plan amendment.
- The queue is resumable. A started cell finishes, if necessary in the next session. Tier-A cells left at the end of S3 run in the automatic continuation S3b (overrun ceiling 44 h).

---

## Plain answers (pre-registered wording)

The post answers each question with one sentence chosen by the verdicts, then the numbers. `<…>` are filled from `verdicts_<UTC>.json`; a NOT RUN state always reads "<H> was not run (<reason>); we make no claim about it."

| Q | Verdict state | Sentence |
|---|---|---|
| 1 | always | "Kolibri runs on an Apple M5 Max (128 GB) through our gated MLX port: <K8 tok/s> tok/s at 8-bit and <K4 tok/s> at 4-bit, batch 1 (M5 Max, MLX 0.31.2)." |
| 1 | H1 CONFIRMED / REFUTED / INCONCLUSIVE | "At 4-bit it decodes at <r> × Gemma 4's speed [CI]: at least / below / we cannot tell whether it reaches three quarters of it." |
| 1 | D1 CONFIRMED / INCONCLUSIVE / REFUTED | "On a 64 GB M4 Pro node the 4-bit build fits a 64k context with 10 % headroom / fits only without headroom / does not fit, if the node runs nothing else. We did not measure speed there; the 4-bit-vs-Gemma ratio is the number that transfers." |
| 2 | H2 CONFIRMED, peers' mean D on the shared rows within ±2 pp | "On average across the five public rows we could check, at MLX 8-bit and k = 1, Kolibri scores no more than 4 pp below its own scorecard (D̄ = <x> [CI]); three of the five rows are our reconstructions." Any row whose 95 % upper bound is < −8 pp is named in the same sentence. |
| 2 | H2 REFUTED, peers within ±2 pp | "On the public rows we could check, at MLX 8-bit, Kolibri scores <x> pp below its own table while Gemma 4 and Qwen3.6 land within <y> pp of theirs. The port passed its gate; grid and protocol effects are bounded by the peers." |
| 2 | H2 any verdict, peers' mean D on the shared rows outside ±2 pp (takes precedence over the other Q2 rows) | "Our local protocol scores every model <lower/higher> than the vendor's table: Kolibri by <x> pp, the peers by <y> pp; Kolibri's gap relative to the peers is <DiD> [CI]." |
| 2 | H2 INCONCLUSIVE | "We can't tell at this sample size: Kolibri is <x> pp [CI] from its own table, on average across the public rows." |
| 2 | H2 any verdict, peer control (E8) NOT RUN (takes precedence over the other Q2 rows) | "Kolibri is <x> pp [CI] from its own table on the public rows (H2 <verdict>), but the peer control did not run (<reason>), so we cannot separate Kolibri from our local protocol; this is not a headline." |
| 3 | always | "In English and German on MMLU-ProX-Lite, Kolibri 8-bit scores <x> pp [CI] against the mean of Gemma 4 and Qwen3.6 (EN <a>, DE <b>) — <H3 verdict>. On IFBench it is <y> pp ahead of Qwen3.6 (<H4 verdict>) and <z> pp against Gemma 4. Its tokenizer packs German <r> × denser than both (<H5 verdict>). The German comparison rests on <MMLU-ProX DE / plus GPQA-D DE and AIME DE>." The E1 label applies when the peers ran no AIME/GPQA. |
| 4 | H6 CONFIRMED, forced cells ran, forced deficit upper bound < −10 pp | "Kolibri knows less without documents: it answers <x> pp fewer RGB questions correctly than the peers, and still <y> pp fewer when told to always answer. Of the questions it gets wrong, <A> are abstentions: it <says so / guesses / unclear>." |
| 4 | H6 CONFIRMED otherwise | "Without documents, Kolibri's closed-book correct-answer rate is <x> pp below the peers'; <A> of its non-correct answers are abstentions (<E2 reading>)." |
| 4 | H6 REFUTED / INCONCLUSIVE | "We did not reproduce the vendor's closed-book gap: <x> pp [CI]." / "We can't tell: <x> pp [CI]." |
| 5 | H7 and H8 | "Squeezing Kolibri to 4-bit costs <x> pp on the task rows (<H7 verdict>), and its logit fidelity loss is <r> × the peers' on our six gate texts (<H8 verdict>)." If H7 or H8 is NOT RUN: "We could not measure the task cost / fidelity loss of 4-bit (<reason>); D1 says only whether it fits." |

---

## Publication angle

**Working title.** "We ran Aleph Alpha's Kolibri on a MacBook: does it fit, how fast is it, and does its scorecard hold on public rows?"

This would be the first independent, locally run, port-validated numbers for a model released on 2026-10-03, which had no MLX or GGUF build and no Artificial Analysis listing that day. Re-check "first" on the publication day.

**One key-numbers table, ordered Q1→Q5:** vendor, ours, a V/R/untestable label, the truncation rate and the result excluding truncated items. The headline is the Q2 sentence chosen by the table above. Every headline names the precision (MLX 8-bit or 4-bit), states k = 1, and states that 3 of the 5 H2 rows are reconstructed. A verdict carrying an E8 flag or a truncation-sensitive verdict never appears in a headline.

**E1, told as it is in the vendor's own table.** The vendor's "best MoE" claim is on its Overall rows, which include internal and in-distribution rows (p. 98–101/189). On the 10 public deterministic rows of its own table Kolibri is last (75.6 vs 80.8 / 80.4); it leads only on the 7 rows its SFT mix was selected on, which the vendor itself calls optimistic (p. 70/189). The vendor disclosed the closed-book weakness itself (p. 99/189); we credit that and report whether it replicates locally. The comparator set is named as not the vendor's.

**The port and gate as a public artefact.** The per-layer error table, the self-calibrated bounds, the mutation tests and the routing trap: afmoe and llama.cpp select experts on sigmoid + bias, while Kolibri selects on logits + bias. Also the tokenizer warning that turns out to be a false positive.

**What we could not check, and why:**
- Internal rows: Industry RAG and Honeypot.
- Rows scored on test splits of Kolibri's own RL environments: MuSiQue and Agentic Wiki QA DE.
- SQuAD M/A (the vendor's own metric and RL reward).
- Benchmarks needing a user simulator or the web: Tau2/Tau3 and BFCL.
- Code and agentic rows.
- LLM-judged rows: AA-Omniscience, HLE, FRAMES, SealQA, AA-LCR.
- Long-context benchmarks; base-model claims (no base weights); AIME 2025 (no pinned asset).
- The vendor's serving-cost Pareto (8×B200, high-concurrency decode, p. 130–131/189): our batch-1 Mac numbers neither test nor refute it.

**What we owe the reader:**
1. Gemma 4 and Qwen3.8 are partly Kolibri's teachers.
2. Selection overlap, in-distribution RL environments and contamination.
3. The reference is ours. A specification error common to the port and the reference would show only through G3's NLL sanity and the vendor's routing test.
4. The vendor targets datacentre FP8 serving; we test outside that envelope. We have no relationship with Aleph Alpha; Aprimerose builds local-first deployments and would use Kolibri if it held up.

**Style.** English only, sections separated by `---`, first names only, quotes of model output only from public sets (MMLU-ProX, IFBench, AIME EN, gate behaviour prompts), no withheld text anywhere. The Cohere merger is mentioned only as sourced fact, if at all.

**Order.** Each outward step is marked; the pushes from the mbp in RUNBOOK steps 4–17 are the pre-approved hand-off.
1. Verdicts, then the HYPOTHESIS results block and the Status line (mini).
2. The scientific_log result block (pointer and verdict words) and the result in both README rows, in one commit (mini).
3. Push **(Andrei's go)**.
4. The blog post, committed locally in the blog repo after `tools/leak_check.py <post.md>` (`post: … — exp_036`, body "Committed only; deploy.sh not run.").
5. `deploy.sh` and the SFTP upload **(Andrei)**; then check that the post's URL returns 200.
6. The README blog-index bullet, then push **(Andrei's go)**.

**Gate failure or STOP.** A dated block in this file, a scientific_log block, README status "Gate failed — <check>" or "Stopped — budget", pushed with Andrei's go; a post only on his go.

**Optional, each needing Andrei's go:** HF upload of the conversions (Apache-2.0 + NOTICE, "as evidence, not product"), an mlx-lm PR, a Malaga-AI post, the cloud FP8 anchor.

---

## Edge cases settled at build time (2026-10-04, before the pre-registration push)

The build's review found nine situations the rules above did not settle. The main session settled them as below, choosing the conservative reading each time; the code implements exactly this, and Andrei's sign-off covers it. Changing any of them later is an amendment.

1. **Crash at B = 1.** A crash of any cause in a cell already running at B = 1 aborts that cell (moved to `aborted/` with a NOTE.md); there is no retry at B = 1. Its hypotheses are NOT RUN unless rule 2 applies.
2. **Aborted or unstarted B8 / B10 cell.** H2 then uses its single-pass estimate and the affected H7 row is dropped (named in the results); the hypotheses themselves still run.
3. **Plan P10.** B8 still runs its K4 GPQA-Diamond DE cell. No hypothesis uses it; it is reported as descriptive only.
4. **Q2 when the peer control did not run.** Answered by the added row in "Plain answers", which is never a headline.
5. **Plain-answer fill-ins.** The `<…>` slots, the precision named ("MLX 8-bit"), the row counts and the margin come from `verdicts_<UTC>.json` and the plan amendment; the wording around them does not change.
6. **Bench counts.** H1 needs exactly 10 blocks with both K4 and G4, and D1 exactly 3 reps. Any other count makes it NOT RUN. An incomplete bench cell may be re-run as a whole, with the incomplete run moved to `aborted/`.
7. **Peer-check drops and H1 / H8.** If G4 is dropped, H1 is NOT RUN. If either build of a peer family is dropped, that family leaves H8's peer median; H8 is NOT RUN if no peer family remains.
8. **Crash fallback and resume.** As written under "Crash fallback": two fallbacks move a cell to `aborted/`; a resumed cell carries `b_fallback_from` in its records.
9. **Peer-check failures outside H1, H3, H4, H6, H8.** Recorded and published with the peer-check record; they change no other verdict.

## Sign-off

*Typed by Andrei on the mbp before the first scored run; committed and pushed from the mbp. Claude never fills any field of this block. A choice that differs from the pre-registered default is recorded in the same commit as **Amendment 0**, which also gives the new `analysis/` tree sha256: `analysis/margins.json` for the H2 margin, the convert flags for the head policy.*

- Signed off by:
- Date and time (UTC, read from the clock):
- H2 margin (−4 pp as pre-registered, −3 pp or −5 pp):
- Kolibri `embed_tokens` / `lm_head` policy, applied to K8 and K4 (quantised at the arm's bits with fp32 logits as pre-registered, or vendor-faithful unquantised; the latter likely makes H1 REFUTED by itself):
- K8 Metal limit (sysctl value run by hand, "not needed" if preflight printed no line, or declined):
- Optional cloud FP8 anchor (no, unless explicitly asked):
- Commit:

The "Signed off by" line must read `Signed off by: Andrei (typed by Andrei on the mbp, <UTC>)`; the runner checks this form.

---

*Experiment design: Andrei + Claude Opus 5.5 · 2026-10-03*

Pre-registration commit: 725d628a322007761dde013bd807d312dfbad1ea

## Amendment 1 — gate fix (2026-10-04T05:18:45Z)

*Written by the main session on the mini before any scored run, pushed as `amendments/1_gatefix_20261004T051845Z.md`, and appended verbatim by the mbp session (one writer). It changes no hypothesis, threshold, margin, arm, task, n or verdict rule.*

**Why.** RUNBOOK step 3 failed on the mbp, as recorded in `aborted/20261004T044821Z-tests/NOTE.md` (commit 71bf9d7). There were two independent causes.

1. **TF32 on the M5 GPU.** mlx 0.31.2 turns TF32 on by default. The `MLX_ENABLE_TF32` variable is read once per process, at the first fp32 GPU kernel that asks for it (`enable_tf32()` in `mlx/utils.h`, a function-local static). On the M5 Max, fp32 `matmul` with M ≥ 2, sorted `gather_mm` and multi-query SDPA then truncate their operands to 10 mantissa bits. The relative L2 error against float64 is about 7.7e-4 instead of about 3e-7. The build host (M4 Pro) has no such kernels, so the suite passed there.
   - With TF32 on, prefill runs in TF32 and decode in exact fp32, and the gate's fp32 checks measure the GPU instead of the port: the tiny gate failed K8 on five checks.
   - With `MLX_ENABLE_TF32=0` all 41 failing tests pass (measured on the mbp).
2. **Leak check.** Six 3-word GPQA options match ordinary English in the kit's own code and documents: 30 places in 22 files, one of them inside the list of generic phrases.

**Change.**
1. **Exact fp32 in every process.**
   - `tools/precision.py` sets `MLX_ENABLE_TF32=0` unless it is already set, and refuses any other value. It then probes an fp32 GPU matmul and refuses if the relative L2 error exceeds 1e-5.
   - It is called as the first statement of `main()` in `gate/run_gate.py`, `runner/run.py`, `bench/run_bench.py`, `tools/peer_check.py`, `tools/preflight.py` and `tools/dry_run.py`.
   - `tests/conftest.py` sets the variable before any test and checks it at session start.
   - `tools/version_record.py` records it as `fp32_precision`.
   - **Why off everywhere.** This matches the vendor's arithmetic: vLLM on CUDA computes the router and the head as exact bf16 products with fp32 accumulation, and PyTorch leaves TF32 off for matmul. bf16 and quantised matmuls are not affected by the variable. Tests, gate, peer check, bench and scored runs now share one setting, so the gate checks what the runs execute, and every arm's speed (H1 and the descriptive cells) is measured under the same setting.
2. **Leak check.** Withheld options and short withheld questions are now hashed from 4 words up: `MIN_OPTION_WORDS` in `tools/shingles.py` goes from 3 to 4. A phrase of 1 to 3 words, without its question, reveals nothing withheld, and the RGB raw outputs stay withheld in any case.
   - The wording in `tasks/selection_rules.json` and the `tasks/build_manifests.py` docstring still says "≥ 3 words". It is left unchanged on purpose: every built manifest records the sha256 of `selection_rules.json`, so editing it would unfreeze them.
   - The bound in force is `tools.shingles.MIN_OPTION_WORDS`. The `tasks` and `manifest_rule` hashes are unchanged.
3. **Dry run.** `tools/dry_run.py` now puts the experiment root on `sys.path` at import time, so that its `main()` can import the guard when it runs as a script.

**New tree hashes.** The other 13 scopes keep their pre-registered values.

hash_tree: GATE_CODE_SHA256 = 0d280ed0136aa87701d6e598fe4a5b2a232976a772e46b85d1cb5af0523dc9b3
hash_tree: RUNNER_SHA256 = 3495b7bffa172699207d2ebcbbbf87da94146fe7b5b071f1ea6f172d796f32e4
hash_tree: BENCH_SHA256 = a69fd1281030c7037c0b21cbc8043a4d9589882564ed5228864991f8cdbfa2e1
hash_tree: TOOLS_SHA256 = 0eaa8b814ce3c7014b12f6e71a71dd7552fdd1918fe4ee116fc957221319b961

**Verified on the mini.** The M4 Pro has no TF32 kernels, so the hardware effect itself is verified by the mbp's re-run of step 3.
- 1,201 tests pass in both Python 3.12 and Python 3.14, with every test required.
- `tools/dry_run.py` passes all 17 stages.
- The leak check finds 0 findings.
- `hash_tree --check` matches all 17 scopes with this amendment appended.

**What the mbp does.**
1. Pull.
2. Run `"$PY" tools/status.py --sync-amendments`, which appends this file to HYPOTHESIS.md.
3. Re-run step 3. The tests turn TF32 off themselves through `tests/conftest.py`, so no RUNBOOK command changes.

## Amendment 2 — gate fix (2026-10-04T05:58:00Z)

*Written by the main session on the mini before any scored run, pushed as `amendments/2_gatefix_20261004T055800Z.md`, and appended verbatim by the mbp session (one writer). It changes only the leak check's withheld-text fallback; no hypothesis, threshold, margin, arm, task, n or verdict rule changes.*

**Why.** After Amendment 1, `tools/leak_check.py --all --no-gpqa-source` on the mbp reported 9 findings, all one 4-word option hash, at 9 kit lines (`aborted/20261004T053036Z-leakcheck/NOTE.md`, commit 83b2074). Its only source is `gpqa_extended.csv`, which the kit never uses, and the phrase also occurs in public MMLU-ProX text. Before step 7 builds `tools/withheld_shingles.sha256`, the check falls back to reading the withheld sources directly. That fallback read every GPQA EN file and subtracted no public text, while step 7 hashes only `gpqa_diamond.csv` and `gpqa_main.csv` and subtracts the kit's public text (the Apache licence and the gate texts). The same tree could therefore pass after step 7 and fail before it.

**Change.** `tools/shingles.py`:
- `collect_from_sources()` now reads only `GPQA_EN_FILES` = `gpqa_diamond.csv`, `gpqa_main.csv`, the files step 7 hashes. If neither is present, it reads every file and warns.
- It then subtracts the kit's public text, exactly as `write_shingle_file` does at step 7. The shared helpers are `public_set()`, `subtract_public()` and `public_file_texts()`. `write_shingle_file`'s output is unchanged; the byte-identity test still holds.
- New test: `tests/test_tools_leak_check.py::test_fallback_mirrors_step_7`.
- **Not changed.** The public filter remains the licence and the gate texts, as BUILD_SPEC §5.5 specifies. The public datasets (MMLU-ProX, IFBench, AIME EN) are not subtracted. A 4-word match in raw or pilot outputs is a warning, not a finding (`MIN_OPTION_WORDS_RAW` = 5). A finding in a later commit would stop the run for a decision.

**New tree hash.** Only `tools` changes; every other scope keeps its pre-registered or Amendment 1 value.

hash_tree: TOOLS_SHA256 = 68a29f18b2fee5baf2b36bfdcb722b9876a660b5653a20b0dac555bba31097af

**Verified on the mini.** 1,202 tests pass in Python 3.12 and 3.14 with every test required. `tools/dry_run.py` passes all 17 stages. The leak check finds 0 findings. `hash_tree --check` matches all 17 scopes with this amendment appended. The mbp holds the GPQA data, so only the mbp can confirm that the 9 findings are gone.

**What the mbp does.**
1. Pull.
2. Run `"$PY" tools/status.py --sync-amendments`.
3. Re-run step 3.
4. Run `"$PY" tools/leak_check.py --all --no-gpqa-source`; expect 0 findings.
5. Run `"$PY" tools/hash_tree.py --check HYPOTHESIS.md`; expect 17 matches.
6. Continue at step 4a.
