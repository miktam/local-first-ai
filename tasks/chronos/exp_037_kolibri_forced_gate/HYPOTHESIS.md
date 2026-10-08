# Experiment 037 — Kolibri through a forced gate: does it fit, how fast, and does its scorecard hold on public rows?

*Pre-registered: 2026-10-07T04:14:29Z · Status: gate run 1 (20261007T110355Z) K8 FAIL (g1), K4 PASS; Andrei's choice b · diagnostics (2026-10-07T14:49:52Z); no scored run*

**Builds on:**
- **exp_036** (`../exp_036_kolibri_local_eval/`, tree at `222c845`): the question, hypotheses, arms, task sets, plan rules and kit, re-registered here. exp_036's Phase 0 gate failed for K8 (G2, G3, G4 and G5; record `e3e01b5c…`) and it was published as a gate failure with no scored run. On 2026-10-05 Andrei chose to re-register the hypotheses with a gate built from that run's bug hunt (decision D8, below). Nothing from exp_036 is cited as evidence about Kolibri's quality.
- exp_007: the padding fixtures, loaded by their real file names.
- exp_008 / exp_010: the prefill-cliff rule.
- exp_011: the MLX runtime. Its fixture-name defect is not repeated here.
- exp_012 / exp_012-Alpha: state total and active parameters for every model.
- exp_014: plan for variance, and score blind.
- exp_015: pin by revision, never by label.
- exp_023: result and verdict shape.
- exp_035: pre-registration shape, version record and withheld rows.
- incident_003_alpha: the test-script contract and thermal logging.

**Directory:** `tasks/chronos/exp_037_kolibri_forced_gate/`, a sibling of exp_036's directory. Nothing in exp_036's directory is edited, moved or deleted. The exp_036 files that exp_037 reads in place are pinned by sha256 below. Assets are pinned in [`assets.json`](./assets.json) and [`ASSETS.md`](./ASSETS.md), copied unchanged from exp_036. The kit is described in [`BUILD_SPEC.md`](./BUILD_SPEC.md) (a delta over exp_036's BUILD_SPEC.md) and [`BUILD_LOG.md`](./BUILD_LOG.md); the run steps are in [`RUNBOOK.md`](./RUNBOOK.md).

**Which document wins.** `scientific_log.md` > this file > `BUILD_SPEC.md` (and, where it says "unchanged", exp_036's BUILD_SPEC.md) > `RUNBOOK.md`. The scientific_log entry holds pointers and verdict words only, no figures, so the two cannot disagree on numbers. If a lower document disagrees with a higher one, the lower one is wrong and is fixed: before the freeze as a draft change recorded in BUILD_LOG.md, after it by a dated amendment of a permitted type.

**What is edited in place.** Only the Status line, the sign-off fields (by Andrei) and the placeholders filled at the pre-registration. Everything else is appended: dated run-record blocks, numbered amendments and results.

**Amendments.** After the freeze an amendment is one of the types in "Fix-cycle, freeze and amendment rules" below, appended as a section with the heading `## Amendment k — <type> (<UTC>)`. Amendment 0 is reserved for the sign-off choices. Tools look an amendment up by its type, not its number. From the pre-registration push until the mbp hands back (RUNBOOK step 19), only the mbp session appends to this file; the mini pushes its amendments as files under `amendments/`, and the mbp appends them verbatim at its next commit. exp_036's Amendments 1–8 belong to exp_036. Where this file names "Amendment 3", "5" or "6" it means exp_036's, whose content is folded into the text below.

---

## What this experiment tests

Andrei's question, as asked for exp_036: *"create experiment 36 to see how good is this model, end result — published on github and localfirstai.eu".* exp_037 asks the same question again, through a gate that a correct port can pass and that shows its own power in the same run.

Kolibri-1 is Aleph Alpha's English–German MoE: 78.1B total parameters, 3.46B active, Apache-2.0 weights, released 2026-10-03. We run it through our MLX port (exp_036's port with the vLLM-angle RoPE and two attention hooks), check that port against the same separately written numpy reference with routing forced, per layer and end to end (Phase 0), and run it on Andrei's MacBook Pro (Apple M5 Max, 40-core GPU, 128 GB unified memory). It runs next to the models we already use, on the same runtime and the same host. The publication must give a plain answer to five questions; the wording is pre-registered under "Plain answers".

| # | Question | Answered by |
|---|---|---|
| 1 | Does it run on our kind of hardware, and how fast? | D1 (fits a 64 GB node at 4-bit), H1 (speed class against Gemma 4), E6, E7 |
| 2 | Does it live up to its own scorecard on what can be checked? | H2 (public deterministic rows against the vendor's values), E8 (peer positive control), the H2 tripwire T-G3 |
| 3 | How does it compare with the models we already run, in English and German? | H3 (knowledge gap EN/DE), H4 (instruction following), H5 (German tokenizer), E1 (public-only composite) |
| 4 | Where is it weak, and what are the trade-offs? | H6 (closed-book correct-answer rate), E2 (does it say so? does it know when forced?), H3, E3, E5, E12 |
| 5 | What does squeezing it onto a 64 GB node (4-bit) cost? | H7 (task cost K4 vs K8), H8 (logit fidelity against the peers), D1 |

**Arms:**
- **K8, K4:** Kolibri at MLX affine 8-bit and 4-bit (group 64). These are exp_036's conversions, bound by exp_036's convert records, cloned in S1 into `$EXP037_BUILDS` and given exp_037's port file (RUNBOOK step 8). No new conversion is made.
- **G8, G4:** Gemma 4 26B-A4B-it, 8-bit and 4-bit. Its family is our production Reducer.
- **Q36-8, Q36-4:** Qwen3.6 35B-A3B, 8-bit and 4-bit.
- **Q38-8, Q38-4:** Qwen3.8 27B dense, 8-bit and 4-bit.

All peer builds are mlx-community conversions pinned in assets.json. One host (the M5 Max MacBook Pro) and one runtime (mlx 0.32.3 / mlx-lm 0.32.0, in process; decision F2) serve every arm. Nothing runs on the mini except building the kit, weight-free unit tests on tiny checkpoints and the analysis afterwards.

## The claim being protected

The claim being protected is the localfirstai.eu verifiability contract, applied to a model nobody has checked yet. Every number we publish about Kolibri must meet three conditions:
1. It comes from a port that matches a separately written numpy fp32 reference with routing forced, per layer and end to end; a misreading shared by both is not excluded (see the blind spot below).
2. It is compared under settings fixed and pushed publicly before any output was scored, and before any gate value of exp_037 existed.
3. Wherever a vendor setting is public, it is honoured: effort high, generation-config sampling, the eval-framework prompts and extractors.

**The blind spot.**

> Port and reference were written separately from one specification, so a misreading they share passes every comparison between them. Only the vendor's own runtime could test that; Andrei chose on 2026-10-05 not to rent it (and had withdrawn the same anchor earlier that day, Amendment 8 D4, 08:37:33Z). exp_036 registered G3's NLL sanity, beside the vendor's routing test (routing only; it passed), as the local detectors of such a misreading (exp_036 HYPOTHESIS.md l.776), and G3 fired: our reference scores T1 at 1.31 bits per byte (the bound was 1.2) and 1.30 × Qwen3.6's bits per byte on our six texts, and its secondary signs (G3-D) remain unexplained. A uniform +9.2 % NLL error would explain the first excess — about 0.17 nats per token on T3, under a third of the subtlest registered reference mutant — and no local test excludes it. The only downstream check is a tripwire on the vendor's public scorecard. A misreading that moves Kolibri's measured shortfall there by up to about 12–15 pp is more likely missed than caught; only beyond about 15 pp is detection near-certain. Every quality result is conditional on its absence. A tiny-checkpoint comparison of the vendor's own model code, through vLLM 0.29.0, against our reference agreed to 1.1e-6 at every layer; it excludes a misreading in the wiring it exercises, not one in vLLM's kernels, in FP8 serving, or in behaviour tiny random weights cannot show.

The last sentence is the vendor-code spike's (decision E, below). Its scope and its literal exceedance on two real-layout runs are disclosed under "Disclosures".

A gate failure, an INCOMPLETE gate, an INCONCLUSIVE verdict or a NOT RUN cell is published as such.

---

## Sources & Rights

- **Vendor technical report.**
  - Source: <https://aleph-alpha.com/downloads/tech-report.pdf>, read as retrieved 2026-10-03.
  - Pin: 3,465,511 bytes, sha256 `01520e07506e67d53aebfb16b5298384940870d652409ea3943fe990f7a68b0e`, Last-Modified Sat, 03 Oct 2026 08:53:19 GMT, ETag `"34e127-65cebc7cd3d0c"`.
  - **Not committed** anywhere, including the workspace submodule. Aleph Alpha retains all rights to it, and Apache-2.0 covers only the weights and config files.
  - Every vendor number in this experiment is cited as "p. N/189".
- **Model card.** Aleph-Alpha/Kolibri-1-BF16 README at revision `7a8f290e`. Its licence section says the licence "does not extend to … model architecture", and also names "underlying code" and "parameter settings". Andrei decided for exp_036, before its pre-registration push, to publish the port and the reference in the public repo ("Yes, publish (Recommended)"); exp_037 publishes its port and reference under the same decision. This is not legal advice.
- **Code derived from aleph-alpha-inference.** The source is aleph-alpha-inference v1.0.0 (commit `049a6a7bd2`, Apache-2.0, © 2026 Aleph Alpha GmbH).
  - Derived files: `port/kolibri1.py`, `reference/kolibri_ref.py`, `reference/mutants.py`, `gate/port_mutants.py`, `tests/tiny_checkpoint.py`, `tests/test_routing.py` and the vendor chat template fixtures.
  - Obligations: the SPDX headers are kept; each derived file carries a "Modified by Miktam for Chronos exp_036" line, and a second line "Modified by Miktam for Chronos exp_037" where exp_037 changed it (`port/kolibri1.py`, `gate/port_mutants.py`); the Apache-2.0 text is `LICENSE-APACHE-2.0` in this directory; every Apache-derived or copied file is named in `NOTICE` (checked by `tests/test_notice.py`).
- **eval-framework v0.14.2** (Aleph-Alpha-Research, Apache-2.0). Vendored under `tasks/vendored_evalfw/` byte for byte, with one exception: `gpqa.py` contains the full text of one GPQA question (`_OVERLONG_QUESTION`), which is replaced by its sha256. That modification is recorded as an Apache §4(b) change (a "Modified by Miktam" line, both the upstream and modified sha256 in `vendored_evalfw/MANIFEST.json`, a note in `SOURCE.md`). `scorers/mc.py`, `scorers/aime.py` and `tasks/prompts/aime_de.txt` (a German translation of the NeMo-Skills wrapper) are Apache-derived and listed in NOTICE.
- **Peer chat templates.** The Gemma 4 and Qwen templates the runner uses are committed under `runner/templates/` (Apache-2.0 per their repositories' metadata) and listed in NOTICE.
- **IFBench checkers.** allenai/IFBench @ `1c40f0c1` (Apache-2.0), used in place from `$EXP036_DATA/IFBench-src` and never vendored.
- **RGB.** chen700564/RGB. The data is CC BY-NC-SA 4.0 (stated in the repository README at the pinned commit); `evalue.py` carries no licence header. We therefore do **not** port RGB's code. `scorers/rgb.py` is our own implementation of the rule RGB describes (lower-cased alias containment for every answer slot), with RGB cited as the source of the rule. RGB prompts are rendered at run time from the local RGB files; no RGB text, not even the instruction strings, is committed. Andrei decided for exp_036 to use RGB under its NonCommercial terms for evaluation and publication of scores only ("Yes, scores only (Recommended)"); exp_037 uses it under the same decision.
- **Datasets:**

| Dataset | Revision | Licence / terms | What is published |
|---|---|---|---|
| Idavidrein/gpqa | `83022cef` | CC BY 4.0 plus gate terms: do not reveal examples online | item ids, item sha256, extracted letters, scores, token counts. **Never** item text, gold or raw outputs (`$EXP036_PRIVATE`, hash-listed in `evidence/withheld_manifest.jsonl`) |
| ellamind/gpqa-multilingual | `bc70ca17` | gated, handled exactly like GPQA | as GPQA |
| math-ai/aime26 | `79037aeb` | Apache-2.0 as labelled by the uploader; the problems are © MAA | ids, hashes, gold integers, raw outputs, scores |
| ellamind/aime26-multilingual | `3c8bc18f` | no licence stated | ids, hashes, scores; raw outputs withheld |
| allenai/IFBench_test | `2e8a48de` | ODC-BY 1.0 (attributed in NOTICE) | ids, hashes, raw outputs, scores |
| li-lab/MMLU-ProX(-Lite) | `8e6106a6` / `e82aafb9` | MIT, © li-lab | ids, hashes, category, gold letters, raw outputs, scores |
| chen700564/RGB | `65ec39e4` (assets.json) | data CC BY-NC-SA 4.0 | ids, document indices, hashes, category scores; raw outputs withheld |
| HuggingFaceFW/fineweb-2 deu_Latn test | `af9c1333` (assets.json) | ODC-By 1.0; subject to Common Crawl terms | derived counts; for gate texts T5–T6 only parquet row indices, token counts and sha256. **No web text is committed.** |

Item manifests in the repo carry ids, sha256 and, for MMLU-ProX and AIME EN only, the gold label and category. No manifest carries item text.

- **Gate texts** (unchanged from exp_036, `gate/texts/` copied byte for byte):
  - T1 and T2: our own posts (2026-09-22, exp_035; 2026-09-30, the Malaga-AI study-group post), written after Kolibri's 2026-06-18 cutoff.
  - T3: German prose written by Claude for exp_036 (post-cutoff), labelled so (Andrei's exp_036 choice, "Accept all three (Recommended)"). LLM-written text has unusually low perplexity, so G3 reports NLL per text and T3 is read on its own.
  - T4: Grundgesetz Art. 1–19, an amtliches Werk under § 5 UrhG, in the public domain; the source URL is recorded in `gate/texts/MANIFEST.json`.
  - T5–T6: FineWeb-2 documents, used locally only (see above).
- **Withheld text is never published anywhere.** `tools/leak_check.py` runs before every commit and, as a git pre-push hook, before every push. It checks every file changed since the upstream commit plus the index, not only the last staged files.
- **Assets.** exp_037 needs no new asset. Every model, dataset and code pin is exp_036's (assets.json, ASSETS.md), downloaded for exp_036 and still on the mbp. The Gemma 4 bf16 folder that exp_036's Amendment 6 record used (`mlx-community/gemma-4-26b-a4b-it-bf16` @ `e13fae2a81ec07e3092a3ebb70c80970b640dc3c`; 20 files, of which 11 weight shards and 12 LFS files with `tokenizer.json`) is reused for the S1 fidelity record (RUNBOOK step 9a). If it is no longer on the mbp, a new download is an outward step that needs Andrei's go; without it, step 9a does not run.
- **The vendor-code spike's downloads** (decision E-downloads): aleph-alpha-inference @ `049a6a7`, a CPU torch wheel and vLLM 0.29.0, with vLLM's 148 declared dependency packages and build tools (about 2.7 GB in all), went into a scratch folder on the mini only. None of them entered an exp_037 venv or the repo; `diagnostics/vendor_spike/` holds only the spike's scripts, outputs and README.
- **Disclosure.** We have no relationship with Aleph Alpha. Aprimerose builds local-first deployments and would use Kolibri if it held up.

---

## Fixed before any run (hashes)

The placeholder values in this table are filled by `tools/hash_tree.py --fill` in the pre-registration commit and pushed before any gate run. That push is the freeze (see "Fix-cycle, freeze and amendment rules"). After the push, one line `Pre-registration commit: <sha>` is appended at the end of this file. exp_036's values are cited by prefix only, so that every row below holds no 64-hex value other than its own.

| Item | Value |
|---|---|
| Vendor report | sha256 `01520e07506e67d53aebfb16b5298384940870d652409ea3943fe990f7a68b0e`, 3,465,511 bytes (not committed) |
| Kolibri source weights | Aleph-Alpha/Kolibri-1-BF16 @ `7a8f290e7858825c3cf5e4c447ba68345de9f1d3`, 156,228,414,296 bytes, 32 shards. Every shard's sha256 is checked against its LFS oid at preflight. |
| Kolibri FP8 repo | Aleph-Alpha/Kolibri-1 @ `e52eb4627d11516b0c01de49210ab5a4e4061444`. Not downloaded; no cloud anchor runs (declined). |
| Peers | gemma-4-26b-a4b-it-8bit `33c6d237` / -4bit `0d77464e`; Qwen3.6-35B-A3B-8bit `e06a74e6` / -4bit `38740b84`; Qwen3.8-27B-8bit `815b83c0` / -4bit `10c35caa` (full SHAs in assets.json) |
| Asset manifest | `assets.json` sha256 `42803f55e14c90c3bf7b229010b7aad76ba2135d51e4b0e83bd526fda47ff5a3` (copied unchanged from exp_036, whose value was `42803f55…`) |
| Vendor plugin | aleph-alpha-inference v1.0.0 @ `049a6a7bd2`. The vLLM semantics are resolved against tag v0.29.0; files are listed in `reference/DERIVATION.md`. |
| eval-framework | v0.14.2; vendored files sha256 `e7ff393392d7e60d8e2ba81e76a70c12233da01511d8da2bb7de6ea751bde1bb` |
| Runtime | Python 3.12, mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 (decision F2), plus the pins in `env/requirements-mbp.txt` (sha256 `32488f23411026229ef831ef9ec475fb922d10fd24d6be062ab08f7d4df533e6`); every other pin is exp_036's. macOS 27.0 (26A428) on the mbp, unchanged since exp_036. No macOS, MLX or mlx-lm update is made from this push until S3 ends. |
| Converted builds | exp_036's K8 and K4 shards and `config.json`, bound by exp_036's convert records (8-bit `bd84bf3b…`, 4-bit `e1ec02c7…`), checked at gate start (P4). They are cloned with `cp -Rc` into `$EXP037_BUILDS` and given exp_037's port file in S1; each refresh writes `results/convert/refresh_{8,4}bit_<UTC>.json`. |
| Port | `port/kolibri1.py` sha256 `2c153357862f15b61f3cadaea4f436de2939567180bf5a01a304f3fa60e3f182` (SPEC_VERSION exp037-port-1; exp_036's gated port was `cd6153b8…`); `port/convert.py` sha256 `99dfad12a4dde03dfa0355f94183953cd52fd846ad123d9f4adeb0f0ea25ef68` |
| Reference | `reference/` tree sha256 `85337ed7efeef6b0463f3560fff6da6496661a2fa7a14de5e50a1bce9e251bbd`, copied unchanged from exp_036 (`85337ed7…`). The vendor-code spike forced no reference fix. |
| Gate code | `gate/` tree minus `thresholds.json` and `texts/`, sha256 `1d41328359b9c6af3c58b51894cfd1f38ac028f79c6866cfcf246d293ac8edbf` (includes `behaviour_prompts.json`, `controls.json`, `calibration.json`, the checks and the harness) |
| Gate thresholds | `gate/thresholds.json` (version exp037-gate-1) sha256 `e27bdc0650a8d072aa1709a583b0afcadb12172446aca312565be47253c1f4c4` |
| Gate rules (frozen) | `gate/` tree restricted to `calibration.json`, `controls.json`, `port_mutants.py`, `rules.py` and `thresholds.json`: every pass/fail decision, threshold, control and mutant of the gate. sha256 `5e9e1a530496e57657f1d1c2661d0992520a0f2b0717737e5b9cbb18c4a33cd0`. No amendment may change it. |
| Gate text | `gate/texts/MANIFEST.json` sha256 `1a295dacf20d285512dcf6db755623e89053ec704abef2bd151682178d91b9e8` (copied unchanged; exp_036's `1a295dac…`): ids, decoded text and sha256 of T1–T4 and T7–T8; for T5, T6 and T9 the selection rule, row indices and sha256 |
| Runner | `runner/` tree sha256 `439590de6a5c7971fc167e8918f9570f142446bd7046e49fac8c38cac8b45d74` (sampler, chat kwargs and templates, generate with the one BatchGenerator construction, memory, guard, plan rules) |
| Tasks code | `tasks/` tree minus `manifests/`, sha256 `306861c155653ddff747cc78c4ad5a49262c926b1d256890c9a7555a53d11334` (copied unchanged: loaders, renderers, prompts, vendored eval-framework) |
| Manifest rule | `tasks/selection_rules.json` + `tasks/build_manifests.py` sha256 `4c3fa6803e74b4a58f7282ed19965769f1de62d16d53979b7854a84b77f69857` (copied unchanged) |
| Plan rules | `runner/plan_rules.json` sha256 `2707ea2571cc122b7388f9c9794e3e1130aefdf13c6f844b7de449485bce878c` (copied unchanged: budget, ladder P0–P10, Tier B B1–B10, cap rule, memory rule, crash fallback) |
| Scorers | `scorers/` tree sha256 `7d90acdc0b8b7aa4b4dd1161d4e9290e6f31bdbfee5bee898898e1eff52cd7d6`, incl. `abstain_lexicon.json`, the IFBench adapter and golden fixtures (no withheld text). It differs from exp_036's only in `score_all.py`'s private layout (decision (c)). |
| Analysis / verdict code (Tier 1) | `analysis/` tree minus the Tier-2 files (`exploratory.py`, `tables.py`), sha256 `0a40ec19b38c43bed57df2e9c6d03fc1e9149159dbea172d921af4e7a6345187`, incl. `stats.py`, `verdicts.py` (with the H2 tripwire), `power.py`, `margins.json` and `vendor_values.json`. Exploratory analysis code (Tier 2) is frozen by its own amendment before any scoring. |
| Bench | `bench/` tree minus `ladder.py` (Tier 2), sha256 `2f923d0b80cb357173ed5cdfefa6d7ef5b56fb21485cc288a86046c51c3d6709` |
| Tools | `tools/` tree minus `withheld_shingles.sha256`, sha256 `d2e85aa8e3e6c62bcbd4207ce87d3a568a91ae69da8431eef1e7bf334dcc4f0e` (preflight, version record, leak check and its hook, hash tree, peer check, the fidelity pin, refresh_builds, status, redaction, the descriptive power log) |
| Tests | `tests/` tree sha256 `fee7c32c84fabe4d65346309c43b42f50a928e1c5dd8b4e73fa609e027b93d83`: G1's weight-free suite with its skip policy (`tests/conftest.py`), the heavy tiny tests and the fixtures (the tiny control table included). A gate fix carries its test here, so its amendment gives this row's `hash_tree:` line. |
| Environment scripts | `env/exp037.settings.sh`, `env/setup.sh` and `env/versions.json` (the tree rule over exactly these three files), sha256 `8d97b3df8207fb8a7621977a52693b436c6cb267cee37f9042b76248bb78e6a9`. `env/requirements-mbp.txt` is the Runtime row's; exp_036's env file is in neither. |
| Quantisation | exp_036's builds: affine, group 64, 8 and 4 bits (the mlx-community peers' recipe). Kolibri's router weight is stored as an exact fp32 upcast of the bf16 checkpoint and gives fp32 logits; `expert_bias` (fp32) and all RMSNorm weights are unquantised. Head policy (exp_036's sign-off, re-confirmed): `embed_tokens` and `lm_head` quantised at the arm's bits, with fp32 activations into the head, so the logits are fp32. KV cache is bf16 for every model; no `--kv-bits`. |
| Reasoning | Kolibri `reasoning_effort="high"`, passed explicitly. Gemma 4 `enable_thinking=True`. Qwen3.6 / Qwen3.8 `enable_thinking=True` (their template default). This is the vendor protocol, Table 48, p. 164/189. |
| Sampling | Each model's pinned generation_config: Kolibri T 1.0 / top-p 0.97 / top-k 128; Gemma 4 1.0 / 0.95 / 64; Qwen 1.0 / 0.95 / 20. Applied by the kit's vLLM-order sampler (temperature, then top-k, renormalise, then top-p) for **every** model, on fp32 logits; no min-p and no penalties. The seed per cell is `H("exp037", arm, task, effort, pass)`. |
| Output caps | Initial: GPQA and MMLU 32,768; AIME 65,536; IFBench and RGB 16,384 completion tokens. Ceilings: GPQA and MMLU 65,536; IFBench and RGB 32,768; AIME 98,304. At most one raise per task, straight to the ceiling, by the pilot rule below. Truncated = wrong (our rule; see C7). |
| Statistics | Bootstrap B = 10,000 (100,000 near a threshold, below), numpy PCG64 with a per-hypothesis seed = the first 8 bytes of sha256(`"exp037|" + H_id`), percentile intervals; H1 uses a t-test. Holm at family-wise α = 0.05, separately for CONFIRMED and for REFUTED. |

**exp_036 files read in place** (read-only, by relative path; never a gate input except where marked):

| exp_036 file | Used by | sha256 |
|---|---|---|
| exp_036's env file (`env/` in exp_036's directory) | sourced by every mbp block (RUNBOOK); never opened, hashed or copied | — |
| `results/convert/convert_8bit_20261004T133423Z.json` | P4 | `bd84bf3b7c4a3db891c02e41f86f618c1d93301581146fca8e598e04313e0e9d` |
| `results/convert/convert_4bit_20261004T133455Z.json` | P4 | `e1ec02c764667c7873f49dedd00c353db804409c3592f913444c8599b5d7f8a2` |
| `results/version_record_20261004T063157Z.json` | provenance of P2's OS pins (macOS 27.0, 26A428) | `27ca857435aca8bd8e7332dbc368e6da5ed107772f8fbb99f6934e154e4293e2` |
| `results/preflight_20261004T063132Z.json` | provenance of P2's architecture pin (`applegpu_g17s`) | `43ce35db4a0aa1d4dab721624a842d66600356a38ae59b5a43978bad5dbaf6c5` |
| `results/gate/gate_20261005T050112Z.json` | the known-values ledger only (no code reads it) | `e3e01b5cf9114afe4d39de7395a74d47fcb0565ca7d93343d29788d69bf00f0b` |
| `diagnostics/gate1/BUGHUNT.md` | the gate's design (BH §6.8) and the M5 defect's resolution (§6.9) | `5bbe997923bd3b26f0ef3b68a7fbc93b4092c14494adb5e0a8e57f75c0e3b7b2` |
| `diagnostics/gate1/FROZEN_RULES.md` | A1's pre-committed per-pair form (G2 leg (p)) | `bd1253a0c83540b3072a5901dbe28db292ace1fea21d5acf770e049392f2afb8` |
| `diagnostics/gate1/confirm/out/mlx_repro_m5max_mlx0323_20261006.json` | the evidence behind F2 and G0k's expected outcome | `da9923cbaebdd76d51337b10932e19454e6067e1c582a0f97f170e68d779a6f7` |
| `diagnostics/gate1/confirm/out/mlx_repro_min_m5max_mlx0323_20261006.json` | as above (the boundary sweep) | `81cc4f8ea91dd59019e4a29170aa42d79a011f31cebd56f3d06ce0c4d908f671` |
| R1, exp_036's fp32 reference dump on T1–T9 (`$EXP036_WORK`, key `85337ed7efeef6b0/370e372120385671/0c30e965690ba1d9/fp32`) | reused under its registered key (numpy, no MLX); recomputed under `$EXP036_WORK/exp037/` if the key does not match | key as given |

`COPY_RECORD.json` records the source and copy sha256 of every file copied from exp_036, with its status (unchanged, modified, new).

---

## Pre-registration decisions

*Recorded by the main session from Andrei's answers, with the UTC time and the words or option labels he used. Claude never chooses these. Decisions (a)–(e) further below are the main session's own, taken before the freeze; they are not Andrei's.*

**Carried from exp_036** (Andrei's answers of 2026-10-03, about 18:10 UTC, which exp_037 keeps): publish `port/` and `reference/` ("Yes, publish (Recommended)"); RGB for evaluation and scores only ("Yes, scores only (Recommended)"); the new assets and mini fetches ("Approve all (Recommended)"); T3 is Claude's text, labelled so ("Accept all three (Recommended)"); the overrun ceiling for S3b is 44 h.

**Decisions for exp_037:**

| ID | UTC | Words / label | Content |
|---|---|---|---|
| D8 (exp_036) | 2026-10-05T14:42:57Z | "Confirm, publish, then exp_037 (Recommended)" | exp_036 published as a gate failure; exp_037 re-registers the hypotheses with a gate built from the bug hunt (BUGHUNT §6.8) |
| G3 | 2026-10-05T18:08:20Z | "yes, option 2" | Stay local: no rented GPU, no FP8 vendor anchor. G3 bits per byte not gating; a declared blind spot plus an H2 tripwire |
| Push of 5c3d178 | go not recorded; given before the push, which the mini's reflog records at 2026-10-05T18:17:01Z | "go" | Andrei explicitly said go to pushing the README line 5c3d178 (committed 18:02:42Z) together with f740816 (committed 18:12:19Z, parent 5c3d178); both reached origin in that push |
| A | 2026-10-06T02:30:15Z | "as recommended" | Scope: the **lean** variant, not the full draft |
| B | same | "as recommended" | The four loosenings (G2 6σ, G4 99 %, G3 bpb, G5 batch parity) are approved, **conditional on** a blocking runner-at-B = 1 check (G5-R1) and the G2 per-pair leg (p), and on the disclosure sentence "every loosened rule failed in exp_036; none that passed was loosened" |
| C | same | "as recommended" | G3b (the seven reference mutants) stays **blocking** under its rule **as registered**. This narrows "G3 non-blocking". G3a bits per byte is descriptive |
| D | same | "as recommended" | Hugging Face upload criterion 2 becomes "G4-F32, G4-F16 and G4-N(i) pass in a gate run", set now, before any result |
| E | same | "as recommended" | Vendor-code layer spike: yes, before the freeze, at most 2 h, on tiny checkpoints. A found mismatch means a reference fix before the freeze, or stop |
| E-downloads | 2026-10-06T04:06:38Z | "yes, downloading is fine" | Confirms the spike's downloads: aleph-alpha-inference @049a6a7, a CPU torch wheel, vLLM 0.29.0. vLLM's install also pulled its 148 declared dependency packages and build tools, about 2.7 GB, all inside the scratch folder; disclosed |
| E-verdict | 2026-10-06T05:17:51Z | "ok, use your recommendatino" (to the recommendation "record as no wiring mismatch, with the disclosure") | The spike's result is classified as **no wiring mismatch**. The literal end-to-end reading of the spike's 1e-4 rule was exceeded on two real-layout 600-token runs (2.2e-2 seed 23, 1.7e-4 seed 29) by routing chaos. That reading was settled after the result was seen, and is disclosed |
| F | 2026-10-06T02:48:30Z | "let's update mlx on both machines" | **Runtime pin F2.** MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0; every other package keeps exp_036's pin. New venvs beside exp_036's: mbp `~/models/exp037/venv` (Python 3.12; checked as `mlx 0.32.3 \| mlx-lm 0.32.0 \| applegpu_g17s`), mini `~/models/exp037-mini/venv312` and `venv314`. exp_036's venvs keep 0.31.2 / 0.31.3, so exp_036 stays reproducible. Taken after the M5 defect was shown fixed in MLX 0.32.3 (BUGHUNT §6.9, `222c845`, 02:47:53Z), which arrived after decisions A–E. The alternative not chosen, F1, was to keep MLX 0.31.2 / mlx-lm 0.31.3 with a mandatory 32,768-row limit. The release-notes review behind it is Annex A |
| W15-04 | 2026-10-06T18:41:39Z | "as recommended" | Approves the final review's threshold change: G5-D32's mean and max factors go from 10 to 100 × Csort, which is G4-F32's τ factor. The floors and both Csort ceilings are unchanged. A loosening, decided after the tiny stress records; disclosed under "Values known when every rule was written" |
| W15-02/03 | same | "as recommended" | Approves the freeze-mechanics tightenings. The `tests` and `env` hash scopes join the table. Interrupted or crashed gate runs are recorded, and orphaned runs are refused by P1(d) |
| W15-R3 | same | "as recommended" | G4-F16's R3 ceiling stays at 1e-2. The final review estimates a real-weight R3 of about 1–4e-3, with a 5–10 % chance of a VOID on T9 that would cost a cycle. Andrei accepted that risk and declined the optional pre-freeze reference-only R3 pass on the mbp |
| go #1 | 2026-10-07T04:11:40Z | "go" (to the five parts as put to him) | (a) Fill the hashes and push this pre-registration. (b) The README index rows read "Pre-registered". (c) The same pre-approval of mbp pushes as exp_036's hand-off decision: sign-off, records, amendments, raw outputs and scores, each after the leak check. (d) Confirms the "power log" amendment type ("Fix-cycle, freeze and amendment rules": after S3 only, `tools/power_log.py` and its test only, with their `hash_tree:` lines; it decides nothing). (e) Decision E on the per-step decode logs: the pilot's are committed, since the plan rule fits its step model on a summarised pilot's; the sessions' are git-ignored working files, never committed or deleted, sha256-bound in the session run records and copied to the mini at RUNBOOK step 18 (BUILD_LOG "Pre-freeze: inherited-values sweep and fixes"). Parts (d) and (e) were added after W15-11's mbp check and the main session's inherited-values sweep (2026-10-06 19:08–23:05Z: 15 defects fixed, none in a gate rule, threshold, control or gate text) |

The quoted words of E-verdict are Andrei's as typed.

**Defaults accepted with A–E:**
- **RoPE:** the vLLM-angle RoPE in the port.
- **Residual add:** exp_036's bf16 add is kept; no fused add-then-norm, not even as a probe.
- **Strict freeze:** no threshold-change and no verdict-rule-change amendment; at most 2 fix cycles, counted by re-runs; INCOMPLETE counts as a cycle; crashes and failed preconditions do not.
- **Tripwire:** an H2 tripwire at −10 pp on both one-sided 95 % upper bounds (D̄_K and DiD), with the blind-spot wording as approved.
- **G5 greedy:** blocking at 99.5 % / 99 % as registered.
- **Peer check:** re-run in S1 under the exp_037 runner; flip rule unchanged.
- **H8:** confirmatory, with the non-blindness disclosure.
- **exp_036 sign-off items re-confirmed:** H2 margin −4 pp; quantised head at the arm's bits with fp32 logits; K8 Metal limit per preflight; 44 h overrun ceiling; cloud anchor declined. Seed prefix `exp037`.
- **No E13. No T10.**

**What the lean scope cut** (each with its reason; accepted with A):
- **T10:** the blind tests are blind on T1–T9 already. T10 costs a web-text pipeline, about 0.6 h per gate and a dump key.
- **G4-N (ii) and (iii), the 1,024 and 512 chaos draws, R4, the McNemar legs:** (ii) is dominated by (i) at exp_036's levels. The p-value legs are invalid under chaos.
- **A G2 tail leg, its binomial, a T9 natural-selection extension:** at n ≈ 8,448 the effect size binds, so the p-value adds machinery but no power.
- **G4-F16 p99, McNemar and shadow-rate legs; the fused-add probe; probes 10 and 11:** κ = 9 is the only derived constant. The rest are reported only.
- **F4 as a rule:** it is descriptive, because the window machinery is a gate-bug risk. G5-D32's chunk-64 leg covers edges sharply on its ranges.
- **Mutants 16, 17, 18, 21, 23, 24, 25 and a `_biased_weights` → `forced_route` patch:** forcing goes through `route()` only, where mutant 14 is visible.
- **G5-BP forced replay, its fp32 arm, McNemar, arm A′, per-row id recording:** they only decide throughput, and the id mapping inside BatchGenerator is the most bug-prone item.
- **The null simulation (≥ 20 draws), Bonferroni, a power script for it:** no p-value leg remains.
- **A G0k fallback search and a per-session G0k probe:** a G0k failure is exit 3 and Andrei decides. Sessions are bound by version (P2).
- **A legacy036 module:** the gate record already carries every old statistic, and the post prints the old thresholds beside them.
- **E13; a second review:** one combined review instead.

**The main session's decisions before the freeze** (2026-10-06; not Andrei's; no gate threshold changes; open to the final review):

| | When | Decision | Reason |
|---|---|---|---|
| (a) | after the build's stage 3 (the orchestration task was blocked) | The unsharpened seed-29 real-layout tiny build (q/k norms × 1) is the validation build of the whole tiny gate and of the drivers' port run; every check of the tiny gate is judged on it. Its free-routing checks (G4-N(i), the noise floor, decode-vs-prefill, G5-R1 with control 26, G5-BP-lean with control 27, behaviour) run there on fp32 activations, the tiny default, while the real gate runs them in bf16; one bf16 run of the whole tiny gate on this build is recorded beside it (final review W15-05). The sharpened seed-23 / 29 builds stay for F_tiny (calibration 23, validation 29) and for the fp32 forced checks' own tests. The sharpened failures are recorded as a stress finding. No threshold changes | The × 3 q/k sharpening makes 8-bit quantisation of random tiny weights unrepresentative: KL(R1‖R2F) 0.21 with routing held identical and top-1 agreement 12 %, against about 99 % for real K8, so those checks would measure fixture noise, not code |
| (b) | same | Mutant 27 (`batch_decode_pad_keys_visible`, an `attn_mask` hook override) replaces mutant 22 as the required control of G5-BP-lean (d). It was chosen and justified before any measurement of it. Mutant 22 stays as a reported probe, in tiny mode only. Decision 8's fallback (an uncaught required control sets allowed_B = {1}) is unchanged | Mutant 22 cannot reach its tiny margin on any build tried: 1.76 sharpened, 5.75 unsharpened, 19.6 on pattern5, against 30. It shifts relative RoPE positions, which flat tiny attention barely sees. Mutant 27 is the realistic gross defect, a padding mask not applied at decode, and only the batched path can see it |
| (c) | same | Integration fixes: three test literals that followed the SPEC_VERSION, private-path and hook-path renames; `scorers/score_all.py` on exp_037's private layout; the Gemma 4 fidelity reference points to a new record produced in S1 under MLX 0.32.3 by a byte-identical copy of exp_036's producer (RUNBOOK step 9a); `tests/test_gate_ftiny.py` runs on the mini only (`build-host only:` elsewhere) | The literals are mechanical consequences of the renames. The scorer could not score a withheld cell under the split private layout. exp_036's Gemma record was computed under MLX 0.31.2, and no MLX-computed exp_036 value is reused. F_tiny is a mini-produced constant whose validation margin is thin by construction |
| (d) | same | A flag for the final review, no change: G4-F16's R3 ceiling 1e-2 has no real-weight value behind it; the review estimates from exp_036's records whether real K8's forced bf16-emulation KL could approach it, and recommends | A VOID makes K8 INCOMPLETE with no remedy under the freeze if the breach is the emulation's honest bf16 level. The sharpened tiny build shows R3 can exceed the ceiling 13–16× on a chaotic fixture |
| (e) | after mutant 27 was measured (11.93 / 6.41) | G5-BP-lean's required control stays mutant 27. For this ratio leg only, its tiny acceptance replaces the 30× tiny margin with: (d) caught on BOTH subsets (first wave and mid-run) at the gate's own factor (ratio > 3), on the unsharpened seed-29 build, with the fixture ceiling recorded beside it (the ratio an unrelated-context swap reaches: measured 22.3 first wave / 19.1 mid-run). Every other control keeps its tiny margin | The tiny fixture's next-token distributions are near-uniform (entropy 6.61 of a possible 6.93 nats; KL between unrelated contexts 0.607 on average), so no batched-path defect that acts through attention can reach 30 on it. This is a pre-freeze code check, not a gate threshold. G5-BP-lean never fails the gate (decision B), and its control is re-tested on real weights in the same gate run: uncaught there gives allowed_B = {1}, and Decision 8 is unchanged |

Decision (e) was chosen after mutant 27 measured 11.93 / 6.41, which meets it; this is disclosed under "Disclosures". With (e) the main session also recorded a correction: under mlx-lm 0.32.0 a padded row's first generated token is affected by mutant 27, because `GenerationBatch` runs its first step at construction as an `L == 1` step over the batch caches. The earlier statement that "the first generated token comes from prefill and is unaffected" was wrong; the mutant's definition is unchanged, and the mutant tests pin the actual behaviour.

**At the freeze (go #1, Andrei's, each recorded here with its UTC before the push):** (a) the go to fill the hashes and push this pre-registration; (b) whether the README index rows read "Pre-registered" with this push (default yes, as for exp_036); (c) the same pre-approval of mbp pushes as exp_036's hand-off decision: sign-off, records, amendments, raw outputs and scores, each after the leak check. *Given 2026-10-07T04:11:40Z ("go") to (a)–(c) and to two parts added before it: (d) the "power log" amendment type and (e) decision E on the step logs (decision table, row go #1).*

---

## Confounds (pre-registered)

| # | Confound | Control / statement |
|---|---|---|
| C1 | **Precision grid.** The vendor evaluated FP8 e4m3 128×128 weights with dynamic FP8 activations and an FP8 KV cache, after FP8 quantisation-aware RL (p. 74, 180/189). We run MLX int8 / int4 affine g64 weights with bf16 activations and a bf16 KV cache. The vendor targets datacentre FP8 serving (model card: minimum 2× A100 80 GB or 1× H200/B200); we test outside that envelope. | "MLX int8 g64 is not the vendor's FP8-QAT grid. The gate bounds the port against the reference, not the grid." The gate reports K8 and K4 KL to fp32 descriptively. E11 reports what the quantised head and embeddings contribute. E8 bounds the joint grid and protocol effect through the peers. No cloud FP8 anchor runs: Andrei declined it twice (exp_036 Amendment 8 D4; decision G3). |
| C2 | **Runtime.** vLLM 0.29 plus the vendor plugin, against our MLX port. | Phase 0 gate (blocking, hash-bound): the port against the reference per layer and, with routing forced, end to end in fp32 (G4-F32) and in production bf16 (G4-F16), plus the production runner at B = 1 against the reference (G5-R1). |
| C3 | **Peers run on mlx-lm's own modules**, from mlx-vlm conversions that we did not validate. | The peer check (S1): strict text-only load, parameter count within 2 % of the config arithmetic, NLL(8-bit) ≤ NLL(4-bit) + 0.02 nats/token and KL(8‖4) < 0.2 on the gate texts scored chat-wrapped (each text as the assistant turn after "Write a text.", thinking off; exp_036 Amendment 5), template and tokenizer byte parity with upstream, and batched-path parity through the same BatchGenerator code the scored runs use. A failed family NLL / KL rule is attributed per build by the S1 bf16 fidelity record when it is usable (exp_036 Amendment 6's rule; "Peers are verified, not gated"). E8 uses the peers as a positive control. |
| C4 | **Router precision asymmetry.** The peers' routers are 8-bit (Gemma `router.proj`; Qwen3.6 `mlp.gate`, `shared_expert_gate`). Kolibri's router gives fp32 logits, as the vendor's does. | Stated. E11 measures Kolibri's routing agreement and KL with an 8-bit router. |
| C5 | **Sampler order.** Stock mlx-lm (`sample_utils.make_sampler`) applies top-p before top-k on the full distribution. vLLM applies top-k, renormalises, then applies top-p. | The kit's vLLM-order sampler is used for every model and unit-tested against a numpy transcription of vLLM v0.29 `apply_top_k_top_p`. It imports nothing from `mlx_lm.sample_utils`, so mlx-lm 0.32.0's sampler changes do not reach it. |
| C6 | **Logit dtype.** Kolibri computes fp32 logits (vendor `head_dtype`). The peers' mlx-lm modules return bf16 logits, which is also vLLM's default for them (head dtype = model dtype). | The kit upcasts every model's logits to fp32 before normalisation and sampling. H8 compares like with like, using bf16-rounded logits for every model; Kolibri's fp32-logit KL is descriptive. |
| C7 | **Output caps vs the vendor's 256k window.** The vendor ran a 256k window with no completion cap and scored overflow as wrong only on Tau3 and BFCL (Table 48, p. 98, 164/189). | Truncated = wrong is **our** rule, at caps far below the vendor's window, so it penalises long reasoners more than the vendor's protocol does. Controls: initial caps of 16–32k with one raise straight to the ceiling (65k for GPQA/MMLU) on a sensitive pilot trigger; a paired truncation sensitivity for H3, H4, H6 and H7 that decides the headline (below); and, for every H2 row, the per-model truncation rate and the result excluding truncated items shown in the key-numbers table next to the confirmatory number. The reasoning caps above 10k tokens are a declared exception to the deterministic-glue per-stage budget, because the vendor protocol requires long reasoning. |
| C8 | **avg@N vs k.** The vendor ran avg@8 (GPQA), avg@16 (AIME) and avg@5 (IFBench), p. 96–97/189. We run k = 1. | H2's variance includes the vendor's avg@N term. A second K8 GPQA pass is Tier-B item B10; if it runs, the GPQA rows use the two-pass mean and the within-item variance estimator (below). |
| C9 | **Templates and thinking modes differ per model.** | Every kwarg is passed explicitly. The rendered prompt's sha256 and its token ids are stored per item. Kolibri template parity is a gate check; peer template and tokenizer byte parity with upstream is checked at build time. Per-model reasoning splitters are tested on fixtures. Gemma's top_k = 64 assumes vLLM `--generation-config auto` applied the checkpoint default; the vendor lists only "1.0 / 0.95" for Gemma (Table 48, p. 164/189). |
| C10 | **Harness: V vs R rows.** | V = eval-framework v0.14.2 prompt and extractor, verbatim (GPQA EN/DE, AIME EN). R = reconstructed (MMLU EN on ProX-Lite text, MMLU-ProX DE, AIME DE wrapper, IFBench checker, RGB). Every row carries its label. V-row extractors are never amended; if one had to be, the row would be relabelled R. |
| C11 | **Selection overlap and in-distribution training.** GPQA, AIME, IFBench and MMLU-Pro chose the SFT mix and soup; the vendor itself calls those scores optimistic (pp. 67, 70/189). IFBench is also the target of an RL instruction-following environment with constraint verifiers (88,761 rows, 10k budget, p. 172/189), so H4 measures that training as well as generalisation. RGB was not a selection benchmark, but RGB Negative matches an RL environment that "poses unanswerable questions with their full context and rewards only abstention" (p. 174/189). | E9 reports the selection rows, RGB CB/FC and RGB Negative as three separate groups. |
| C12 | **Contamination.** Pre-training was not decontaminated (p. 35–36/189). 39.1 % of MMLU-Pro test questions were found in the pool (p. 157/189). AIME 2026 predates the 2026-06-18 cutoff. | Not probed; stated as a limit on H2, H3 and E1. The peers' contamination is unknown and may differ. Gate texts T1–T3 are post-cutoff. |
| C13 | **Teacher confound.** Gemma-4-26B-A4B rephrased Kolibri's English pre-training data (p. 25/189). Qwen3.8-27B generated SFT completions (p. 53/189). | Stated with every comparison. |
| C14 | **Item subsampling and composition.** MMLU-ProX-Lite has 588 test items per language in 14 categories of unequal size (exp_036 Amendment 4); the vendor's MMLU rows are on the full, category-imbalanced sets (MMLU-Pro 12,032; MMLU-ProX DE). GPQA EN: all 198 Diamond items (exp_036 Amendment 3: eval-framework's over-long filter removes none of them). RGB Negative: 300 vs the vendor's inferred 299. RGB Closed-Book: every vendor value is an integer, so the vendor's n is inferred to be 100, on an unknown subset; our 400 items differ. | H2's MMLU rows are post-stratified to the full set's category shares (below); every n_M set is a house-monotone Webster allocation proportional to Lite's own counts, nested along the ladder. H6's power is stated as conditional on the vendor gap replicating on our items. |
| C15 | **Batching numerics.** Continuous batching through mlx-lm's BatchGenerator: prompts admitted together are right-padded, then finalised to left padding (mlx-lm 0.31.3 and 0.32.0 alike), with `BatchRotatingKVCache` for the sliding layers. | exp_037 keeps the registered `prefill_batch_size = min(B, 8)` (decision F2) and decodes each Kolibri cell at a B within allowed_B. Controls: the gate's G5-R1 (the runner at B = 1 against the reference, blocking) and G5-BP-lean (batched against single, anchored to the reference, at B = 8 and 16; it sets allowed_B), the peer check's batched-path parity (G8, Q36-8), and Control C1 (greedy answer-flip rate at the clipped memory-rule B vs B = 1). |
| C16 | **Active vs total parameters.** | Stated for every arm, from the configs at preflight (exp_012-Alpha lesson). |
| C17 | **Laptop conditions.** Thermals, power and background load on a 14" MacBook Pro. | High Power energy mode on AC; `pmset -g` power mode, the power source and macOS `thermalState` are logged per bench block and every 10 minutes in sessions. Speed cells start after a 10-minute idle. The plan projection uses decode steps from the last third of each arm's pilot. Quality verdicts never depend on timing. |
| C18 | **Hardware transfer** to the 64 GB M4 Pro node. | D1 assumes allocations depend on MLX version and tensor shapes, not on the chip. Absolute tok/s is labelled "M5 Max, MLX 0.32.3". The mini is never used as a run host, and its speed is not measured. |
| C19 | **Pilot leakage.** | Pilot items are disjoint for GPQA (main set, filtered by Diamond Record ID), MMLU (ProX full, non-Lite, gold-consistent) and RGB (`en_int`). IFBench and AIME pilot outputs are discarded: they feed only length and rate statistics, never amendments or tuning. |
| C20 | **Comparator set** differs from the vendor's: no Qwen3.5, GPT-OSS or Nemotron. | Stated in the post. |
| C21 | **RGB protocol.** The vendor's exact prompts are not public. | We follow RGB's own code paths: closed-book is `passage_num = 0` (instruction verbatim, empty documents, no system prompt), and negative and fact-check use RGB's system prompts. Labelled R. |
| C22 | **Kolibri's RL abstention environment** is in-distribution for RGB Negative and plausibly for closed-book questions asked with an empty "Document:" frame. H6 may therefore measure abstention policy as well as knowledge. | The forced-answer condition (Tier A from P0 to P7) separates them; without it, H6 is worded as a correct-answer rate only (see H6 and "Plain answers"). |

---

## Phase 0 — the forced port-correctness gate (blocking)

exp_036's gate failed K8 on four checks that a correct implementation could not meet under free routing (BUGHUNT §1–§6). exp_037's gate keeps every registered check that exp_036's port passed, makes the four failed rules descriptive or replaces them as decided (B), and adds checks with routing **forced**, so that routing chaos cannot move them: forced end to end in fp32 (G4-F32) and in production bf16 (G4-F16), forced decode and chunking (G5-D32), and the production runner at B = 1 against the reference (G5-R1). Every new blocking check shows its power in the same run through a required control mutant. Batching sets allowed_B per arm (G5-BP-lean) and never fails the gate.

### Reference (unchanged from exp_036)

- `reference/kolibri_ref.py` is numpy only. It imports neither mlx nor the port. It was written separately from the port, from the same numbered specification (exp_036 BUILD_SPEC §4), so a misreading common to both would pass every comparison between them (the blind spot above).
- It reads the BF16 shards with its own parser in `reference/safetensors_np.py`: memory-mapped uint16 widened by `<< 16`, which is exact. It streams one layer at a time.
- Each block cites the vendor `kolibri1.py` line and the vLLM v0.29.0 line it implements (`reference/DERIVATION.md`).
- Modes: fp32 (the reference proper); **bf16 emulation**, which rounds to bf16 wherever vLLM stores bf16 (norm outputs, q/k/v, RoPE output, attention output, o_proj, each expert output, the residual after the fp32 add) and keeps router logits, routing weights and the head in fp32; and **dequantised**, which runs a layer on weights unpacked from a converted MLX directory by its own numpy unpacker. Its `layer_forward(l, h, segments, force_ids=…)` forces a layer onto given expert ids.
- The tree is copied byte for byte from exp_036 (sha256 `85337ed7…`); the vendor-code spike found no wiring mismatch, so no reference fix was made.

### Gate text (unchanged from exp_036)

`gate/texts/` is copied byte for byte. T1–T8 are exactly 1,536 Kolibri tokens each = 12,288 tokens, committed as token ids with sha256 and the decoded text alongside; the gate consumes the ids.

| Sequence | Content |
|---|---|
| T1 | EN, our exp_035 post (2026-09-22), first 1,536 tokens of the body |
| T2 | EN, our Malaga-AI post (2026-09-30), first 1,536 tokens of the body |
| T3 | DE, German prose written by Claude for exp_036 (≥ 1,600 tokens before the cut) |
| T4 | DE, Grundgesetz Art. 1–19 |
| T5–T6 | DE, the first two FineWeb-2 deu_Latn test documents with ≥ 1,536 Kolibri tokens at file index ≥ 5,000 (disjoint from H5). Read from `$EXP036_WORK/gate_texts/` and checked against MANIFEST.json; only indices, token counts and sha256 are committed. |
| T7 | EN chat render at effort high, with a closed think block |
| T8 | DE chat render at effort none |
| T9 | Long sequence, 16,384 tokens: T1, T4 and T3 in full, then T5, T6 and further FineWeb-2 deu_Latn test documents from index ≥ 5,000 in file order, cut at 16,384. Not committed (it contains web text); its rule, indices and sha256 are. |

The pack order is T1…T8 then T9, 28,672 rows. T1–8 are each teacher-forced from position 0 as one chunk (12,288 positions); T9 runs in 2,048-token chunks through one cache, with buckets [0, 2,048), [2,048, 8,192) and [8,192, 16,384).

### Common definitions

- **KL estimator.** Every KL in exp_037, F_tiny's included, is `gate.common.kl_rows(ref_logits, q_logits)`: a float64 log-softmax of fp32 logits over the full 128,000-token vocabulary, in nats per position. A KL computed from fp32 log-probabilities is never used.
- **Decisive.** At position t, the comparator's top-1 minus top-2 log-probability is ≥ 2 nats (registered).
- **R1.** The fp32 reference on the BF16 checkpoint, natural routing: exp_036's dump, reused under its registered key (reference tree, checkpoint fingerprint, text key, mode).
- **I.** For layer l and pack row n, I[l, n] = the ascending sort of R1's top-6 expert ids. Forcing ids never come from the port.
- **R2F.** The fp32 reference on the dequantised K8 weights, forced onto I (`KolibriReference(bf16_dir, dequant_dir=<K8 build>)`, layer by layer over the pack). It stores fp32 logits and, per layer, its **shadow**: the top-6 of the biased router scores, with the 6th–7th biased gap.
- **R3.** The same driver with `emulate_bf16=True`, forced onto I. It stores logits and shadows.
- **S(t)** = KL(R2F‖R3)(t), reference-only. It normalises content in F3 and is never a bound.
- **ForcedTrunk.** A route tap installed as the port module's `route` for one model. For each forward call it counts route calls (layer = call index) and raises unless the count equals the number of layers. In force mode it adds a boost to the forced ids, calls the original `route`, checks that the selected set equals the forced set per row, and returns the (weights, ids) pairs in the order of the forced ids. The weights still come from the port's own `route()` on its own fp32 logits, so weight-path defects (mutants 6, 7, 14) stay visible. It also records the shadow: the ids the port's own `route` selects from the unboosted logits plus bias.
- **Csort.** The port, forced by ForcedTrunk onto its own free-run ids sorted ascending, against that free run, with identical chunking. Forcing and free then differ only in the order of the six weighted expert terms, the same order the forced comparison with I imposes. P5b shows that forcing onto the free ids in their own order reproduces the free run bitwise, so Csort measures term order, not forcing. Csort cannot see a deterministic defect: the cache, mask, RoPE and weights are identical in both runs.
- **K8 fp32 port.** The K8 build loaded through the port, then `set_dtype(float32)`, then `fp32_on_dequantised_weights` (lifted verbatim from exp_036's `diagnostics/gate1/diaglib.py` l.247–279), so the port multiplies exactly the weights R2F reads.

### Preconditions (P1–P5)

A precondition failure gives exit 3: the run stops there, it is not a gate run and never a fix cycle, the record is written and partial outputs go to `aborted/<UTC>-gate/` with a NOTE.md. P1–P4 run at phase 0. P5 runs at phase 2b, after the R1 dump exists and before any real-weight port-against-reference value is computed. A code change after any exit 3 is made only as a typed gate fix.

- **P1, hashes and freeze** (real mode only). All of: (a) `tools/hash_tree.check` reports `match` for all 20 scopes of the table above; (b) the registered values of `GATE_RULES_SHA256`, `THRESHOLDS_SHA256` and `GATE_TEXT_SHA256` come from the pre-registration (no amendment line names them); (c) the commit named on this file's line `Pre-registration commit: <sha>` is an ancestor of HEAD and of the upstream tracking branch, which shows the freeze push happened; (d) re-run admissibility: fewer than 3 gate runs exist, and if the latest gate run did not exit 0, the code hashes it recorded differ from the current ones in at least one of port, reference tree, runner tree, gate code tree or builds manifest; no earlier run directory `results/gate/<UTC>/` lacks its record (a run killed before it could write one; `gate/run_gate.py --close-orphans` writes its record and moves an exit-3 run to `aborted/`; an exit-1 orphan stays in `results/gate/<UTC>/`); and every exit-3 run that had passed P5 and whose recorded code hashes differ from the current ones is named, by its UTC, in the first line of a numbered amendment; (e) the rows of the hash table in HEAD's HYPOTHESIS.md equal those of the pre-registration commit (only amendment lines may supersede them).
- **P2, environment binding** (decision F2). `platform.mac_ver()[0]` = 27.0; `sysctl kern.osversion` = 26A428; `mlx.__version__` = 0.32.3; mlx-metal = 0.32.3; `mlx_lm.__version__` = 0.32.0; `mx.device_info()["architecture"]` starts with `applegpu_g17`; `MLX_ENABLE_TF32` = 0. The package pins come from `env/requirements-mbp.txt`, the OS pins from exp_036's version record and the architecture from exp_036's preflight record, each cited by sha256 in `thresholds.json`. In real mode P2 also requires the newest exp_037 version record (`results/version_record_<UTC>.json`, schema `exp037 version record v1`, RUNBOOK step 1) and the newest exp_037 preflight record to carry these values; exp_036's records never satisfy P2. The gate record cites both sha256. Tiny mode records the values without judging them.
- **P3, G0k, the kernel probe at the run's own shapes** (lifted from exp_036's `mlx_repro.py` and `mlx_repro_min.py`). Seed 37. Kolibri's expert projections (gate/up 512 × 2,560; down 2,560 × 512; E 384; affine group 64; at 8 and 4 bits; bf16 activations). Each row count is judged at 8 and 4 bits for gate/up and for down: decode 6, 12, 24, 48, 96 (Kolibri at B = 1, 2, 4, 8, 16) and 8, 16, 32, 64, 128 (peers, top-8); one prefill chunk 12,288 (Kolibri, uniform routing), 2,048 rows on one expert (the hottest one expert can get from one chunk), 16,384 (peers); batched prefill under the registered `min(B, 8)`: Kolibri 24,576, 49,152 and 98,304 (the largest Kolibri call the run issues), exp_036's first wave 8 × 1,099 × 6 = 52,752 rows (not a multiple of 64), peers 32,768, 65,536 and 131,072; G2's registered T1–8 call 8 × 1,536 × 6 = 73,728; the B = 16 bound 16 × 2,048 × 6 = 196,608 (never issued under min(B, 8); judged as headroom); and the boundary set of `mlx_repro_min.py`: 16,384; 24,576; 30,000; 32,000; 32,704; 32,760; 32,767; 32,768; 32,769; 32,770; 32,776; 32,832; 33,000; 34,000; 40,000; 49,152; 65,536. Nothing is recorded only. **Pass, at every judged shape and bit width:** rel_f64(sorted) ≤ 3 × rel_f64(unsorted), each the relative L2 error against float64 on identical inputs; zero rows where ‖sorted − unsorted‖ > 0.05 × ‖unsorted‖; and a repeated sorted call is bitwise equal. The factor 3 is G2's registered bf16 factor; the control is the unsorted path on identical inputs, so the rule holds at 4 bits, where correct calls sit at 0.0071–0.0076. **A G0k failure is exit 3, and Andrei decides.** There is no fallback search, and the code never turns a failure into a row limit.
- **P4, build identity.** For each K8 and K4 build under `$EXP037_BUILDS`: the set of `model*.safetensors` names and each file's sha256 equal the entries of exp_036's convert record (whose own sha256 is checked first); `config.json` equals by sha256; and `convert.check_port_file(dir)` passes (the clone's `kolibri1.py` equals exp_037's port).
- **P5, harness integrity** (phase 2b, real weights, no port-against-reference value). **P5a, reference driver:** for layers 0 and 4 (one sliding, one full), `layer_forward(l, h_in_l, segments, force_ids=top6_l)` on the full 28,672-row pack from R1's dump reproduces the dump's `h_in_{l+1}` bitwise. **P5b, ForcedTrunk exactness:** the K8 fp32 port on T1 (one 1,536-token chunk), forced onto its own free-run ids in their own order, reproduces the free run's logits bitwise. **P5c, forcing active:** the same port forced onto the ids of position t − 1 (layer by layer; position 0 keeps its own) gives mean KL(free‖forced) ≥ 0.10 over T1 positions 1–1,535; 0.10 is the registered G4 gross-bug bound, and a silently inactive forcing stays at the rounding floor (≤ 1e-6).

### The checks

Thresholds are frozen in `gate/thresholds.json`; every pass/fail decision is a pure function in `gate/rules.py` (both in the `gate_rules` scope); the checks under `gate/checks/` only measure. "emu" is the bf16-emulated reference against the fp32 reference on the same input. The rows below are blocking for K8 unless marked.

| Check | What | Pass rule (blocking unless marked descriptive) |
|---|---|---|
| G0 static | Census: tensor count; parameter count; every shard's sha256 = LFS oid; strict load consumes every tensor exactly once and the port declares no parameter absent from the checkpoint; full-attention layers = {4, 9, …, 49} | 58,353 tensors; 78,103,074,560 parameters; 0 mismatches |
| | Template parity: 12 conversations × 10 settings against the vendor jinja | byte-identical, 120/120 |
| | Tokenizer: harness ids vs raw `tokenizers` ids on ≥ 2,000 lines (the registered fixture rule, seed `exp036`) and the gate text; decode round-trip; `fix_mistral_regex` never True | 0 mismatches |
| | Converted configs: quantization `{group_size 64, bits 8 \| 4, mode affine}`; tensor policy equal to the convert record's and to the signed-off head policy; no `.scales` for the router, `expert_bias` or norms; router weight dtype float32 | exact |
| | Value-level dtype checks on a K8 and K4 forward: router-logit and head-output dtype float32, and the fraction of each exactly representable in bf16 | dtype float32; each fraction ≤ 1 % |
| | Same port sha256 for K8 and K4 | equal |
| G1 weight-free | The kit's suite on the run host, with every skip a failure except `build-host only:` skips; the heavy tiny tests (`test_gate_drivers_tiny.py`, `test_gate_end_to_end_tiny.py`, `test_gate_ftiny.py`) are excluded and run on the mini before the freeze | 0 failures, 0 errors |
| G2 per layer, real weights | Port layer *l* fed the reference input *h_l*, all 50 layers, through the registered single call over all eight T1–8 sequences (73,728 expert rows, a G0k shape). e = ‖r_port − r_ref‖ / ‖r_ref‖ per branch (attention, MoE), before the residual add. | rows below |
| | fp32 mode, forced | median e ≤ 1e-4 and max e ≤ 1e-3 per layer and branch; embedding exact; final norm + head max \|Δlogit\| ≤ 1e-3 |
| | fp32 mode, natural selection | top-6 sets identical for every (token, layer) except where the reference's 6th-vs-7th biased-score gap < 1e-4 (exempt pairs reported) |
| | bf16 mode (unquantised), forced | per layer and branch: median e ≤ 3 × emu's median, and p99 e ≤ max(5e-2, 3 × emu's p99) |
| | bf16 mode, natural selection | fails if either (a) the disagreement fraction > max(0.2 %, 2 × emu's fraction), as registered, or (p) M_port > min(8, 1.2 × M_emu), with z = the reference's 6th–7th biased gap / σ_l per disagreeing (token, layer) and M the maximum z over layers and pairs (σ_port,l and σ_emu,l the RMS router-logit error over T1–8 in layer l). The count beyond 6σ is descriptive |
| | bf16-router mutant 10 vs port, paired token bootstrap of agree_port − agree_mutant | blocks only if its 99th percentile < 0 |
| | G2q, converted weights (K8 and K4): the port's quantised layer vs the reference on that layer's dequantised weights, same *h_l*, forced; plus embedding and head | as the bf16-mode forced row; embedding max relative error ≤ 4e-3 (one bf16 rounding); head relative L2 error ≤ 1e-3 per position |
| | T9 long sequence, bf16 mode, forced, by bucket (0–2k, 2–8k, 8–16k) | each bucket meets the bf16-mode forced row against emu's T1–8 statistics |
| | Real-weight mutants 1–8 and 12–15 on layers {0, 3, 4, 49} | each fails some G2 fp32 check (branch error, selection or embedding) on ≥ 1 layer |
| G3a reference bits per byte | Per-text bpb on T1–T6; pooled ratio to the best peer (from the S1 peer check); 128-byte window ratios; bpb from token 64 | descriptive; exp_036's limits (1.2; 1.25 ×) printed beside them as "not an exp_037 criterion" |
| G3b reference mutants | The seven reference mutants on T3: a 48-block bootstrap (B = 10,000) of NLL_mutant − NLL_ref, seeds `seed_from("G3", name)` with prefix `exp037` | fails only if some mutant's 99th percentile < 0 ("mutant_better") or the mutant count ≠ 7; a 1st percentile ≤ 0 is reported as "rests on vendor source" |
| G4-F32 forced, fp32 | The K8 fp32 port forced onto I by ForcedTrunk, against R2F, on T1–8, T9 and T9's buckets; per position KL, R2F lead, top-1, Csort, S and per-layer shadow | fails if any of: F1 a top-1 change where R2F leads by ≥ 2 nats; F2 the mean KL above τ_set = max(F_tiny, 100 × C̄sort(set)) in T1–8, T9 or any T9 bucket (F_tiny from `gate/calibration.json`, capped at F_tiny ≤ 1e-5; T9's buckets use T9's τ); F3 ρ(B2) > 10 × ρ(B0) and the port's B2 mean > 10 × Csort's B2 mean, with ρ = mean KL / mean S over a bucket; F5 any position with KL > 1e-2; F6 any (token, layer) whose port shadow set ≠ R2F's shadow set where R2F's 6th–7th biased gap ≥ 1e-2. C̄sort(set) > 1e-6 in either set is also a FAIL ("re-association control above its ceiling (cause not attributed)") |
| G4-F16 forced, bf16 | K8 as loaded (bf16), forced onto I, against R2F, calibrated by R3 | fails iff, in either set, mean KL(R2F‖port) > κ × mean KL(R2F‖R3), with κ = 9. R3's mean above 1e-2 in either set makes G4-F16 VOID, which sets K8 INCOMPLETE |
| G4-N(i) free routing | K8 bf16 as loaded, free routing, chunk 2,048, teacher-forced against R1 | mean KL ≤ 0.10 on T1–8 and on T9, each set pooled (the registered backstop's KL leg). Decisive top-1 (registered 99 %) descriptive |
| G4 descriptive | K4 end to end against R1; by window, bucket, language and text; F4 windows; Csort maxima and shares; shadow rates; the G4-D5 sparse-spike count | descriptive |
| G5 greedy | K8, the 8 G5 prompts, 256 greedy tokens through `generate_step`, teacher-forced through R1 | decisive top-1 ≥ 99.5 % and within R1's top-5 ≥ 99 % |
| G5 decode vs prefill | K8 noise floor (prefill step 2,048 vs 64) and decode vs prefill at positions 520–1,100 of T1 and T3, and 15,000–15,300 of T9 | mean KL ≤ max(1e-4, 3 × floor_KL) and top-1 disagreement ≤ 3 × floor_dis + 0.2 pp |
| G5-D32 forced decode and chunking, fp32 | The K8 fp32 port on T1 and T3 positions 520–1,100 and T9 positions 15,000–15,300: (i) free, chunk 2,048; (ii) forced onto (i)'s ids sorted ascending, chunk 2,048; (iii) forced decode (prefix in 2,048-token chunks, then one token at a time); (iv) chunk 64; (iii) and (iv) forced onto (i)'s sorted ids at every position from 0. Comparisons (iii)–(ii) and (iv)–(ii); Csort = KL((i)‖(ii)) | per range, for each of decode and chunk 64: 0 top-1 changes where (ii) leads by ≥ 2 nats; mean KL ≤ max(1e-8, 100 × Csort's mean over the range); max KL ≤ max(1e-6, 100 × Csort's max over the range). Csort's mean > 1e-6 or max > 1e-4 in a range is a G5-D32 FAIL |
| G5-R1 the runner at B = 1 | `runner.generate.make_batch_generator(model, 1, …)` with a greedy sampler, the 8 G5 prompts alone, 256 tokens; arms K8 and K4 as loaded. Comparators: "single" (prompt + runner tokens teacher-forced alone through `gate.harness.logits_at`) and R1 | R-anchor (both arms): mean KL(R1‖runner) ≤ 3 × mean KL(R1‖single). R-greedy (K8): the greedy rule above on the runner's tokens. R-parity (K8): mean KL(single‖runner) ≤ max(1e-4, 3 × floor_KL) and top-1 disagreement ≤ 3 × floor_dis + 0.2 pp |
| G5-BP-lean batched path | `make_batch_generator(model, B, …)` with `prefill_batch_size = min(B, 8)`, greedy, through the runner's admission loop, at B = 8 and 16, arms K8 and K4, on 26 prompts (below) | never fails the gate; it sets allowed_B per arm (below) |
| G5 behaviour | The 20 frozen prompts (10 EN, 10 DE), K8 and K4, effort high capped at 8,192 and effort none, through `run_cell` at the registered B = 8 (`prefill_batch_size = 8`), seeds prefix `exp037` | blocks if, for either arm at one effort, ≥ 2/20 loop (a 32-token span repeated ≥ 4 times consecutively), ≥ 2/20 end without either EOS id (127906, 127901), or ≥ 2/20 get `lang_tag` "unknown" |

**G2 bf16 natural selection.** Leg (p) is A1's pre-committed form (exp_036 FROZEN_RULES l.280–281); A1 was conditional on no G2-D sign, and G2-D1 and G2-D3 fired on the flagged pair and remain unexplained, so adopting it is a loosening of the registered 6σ rule (disclosed). The descriptive rows add M for mutant 10 and the flagged pair (layer 20, T4 position 19): whether port and emulation disagree there, z_port, z_emu and r_t = σ_t(port) / σ_t(emu) at that token. The emulation statistics are recomputed on T1–8 into `emu_stats_exp037.json`. No control is required; mutant 10 fails (a) at about 6.8 % against 2.47 % on exp_036's record (reported).

**G4-F32 constants.** 100 in KL is about 10 in logit amplitude (KL is quadratic in small logit perturbations): one decade of cross-implementation rounding over the single re-association Csort measures. F_tiny is the floor that keeps τ from collapsing when Csort is near 0, and τ never exceeds 1e-4 = max(F_tiny's cap 1e-5, 100 × Csort's ceiling 1e-6). F1 is structural: fp32 rounding on identical weights and experts cannot move a 2-nat lead. F3's 10 and 10: one decade of growth in sensitivity-normalised KL from the first bucket to the last, with the port's late mean a decade above the re-association floor. F5's 1e-2 = 100 × τ's cap. F6's 1e-2 = 100 × G2's registered near-tie gap (1e-4). It catches chunk-edge cache defects, rotating-cache wrap, NoPE growth to 16k, RoPE offset or angle drift, weight-path defects and small systematic deviations compounding over 50 layers; it cannot see selection defects inside the forced path (the shadow and G2 cover these), bf16-only defects (G4-F16), batched paths (G5-R1, G5-BP-lean) or misreadings shared with the reference.

**G4-F16 constants.** κ = 9 = 3²: G2's registered 3× on relative branch error, squared because KL is quadratic. R3's ceiling 1e-2: one bf16 rounding per stored tensor composed over 50 layers gives KL of order 1e-3, plus one decade. No real-weight forced-emulation value exists (exp_036's 0.038 / 0.077 are free-routing values, dominated by routing chaos), so a breach is possible with no permitted remedy if it is the emulation's honest bf16 level (see "Calibrators and ceilings"). Descriptive: the p99 ratio, decisive misses against R3, the bf16 shadow rate against R3's, and n(KL(R2F‖port) > 1 nat and KL(R2F‖R3) < 0.1) per set.

**G4-N(i).** The registered gross backstop's KL leg, on one free-routing draw. Its decisive top-1 is printed beside the registered 99 % and exp_036's emulation value (68 of 5,943 = 0.98856). The vLLM angle changes the free-routing draw, and so do MLX 0.32.3's quantised kernels; with exp_036's T9 margin of 1.2×, a false fail is possible (disclosed under "Statistical and model risk").

**G5-D32 constants.** 100 in KL, about 10 in logit amplitude, over the same-run re-association control: G4-F32's τ factor, for G4-F32's reason. Forced decode (one token per step: GEMV-shaped matmuls and MLX's single-query attention kernel) and chunk 64 run other kernel shapes than chunk 2,048, so they differ from it by rounding in every layer, not by the single re-association Csort measures. The factor was 10 (one decade) until the final pre-freeze review set it to 100 for the mean and the max legs (see "Values known when every rule was written"). The floors are the Csort mean ceiling / 100 and the Csort mean ceiling, unchanged.

**G5-R1.** allowed_B always contains 1, and exp_037 may score every Kolibri cell at B = 1. In exp_036 nothing compared the runner's own B = 1 path with the reference, so a runner defect present at every B would deny B = 8 and 16 and then score everything through itself. The generating phase writes the fp32 log-probs at the generated positions per sequence to `$EXP036_WORK/exp037/run/<UTC>/g5/<arm>/g5_r1/<seq>.{runner,single}.npy`, records their sha256 and the tokens in its phase file, and reduces R-parity there; phase 6 computes R1 at the generated positions only. R-parity is not applied to K4: K4 has no noise floor of its own, and R-anchor cancels its 4-bit error. Every constant was registered for exp_036 on 2026-10-03: the factor 3, 99.5 %, 99 %, the 2-nat lead, max(1e-4, 3 × floor) + 0.2 pp. G5-R1 exercises the greedy sampler, not the production `make_vllm_sampler`, and R-greedy re-applies the greedy rule on nearly the same continuations as G5 greedy (disclosed).

**G5-BP-lean and allowed_B.**
- **The 26 prompts and their queue order.** A_j (j = 0–11) = `ids[order[j % 8]][:L_j]` with `max_tokens` M_j: exp_036's 12 prompts, L = 37, 300, 700, 1,100, 64, 520, 900, 150, 37, 700, 300, 1,100 (8 distinct lengths) and M = 48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44. L1 = T9 ids [0, 2,100) with 24 tokens and L2 = T9 ids [4,096, 7,096) with 40 tokens; both cross a prefill chunk. B_j = `ids[order[(j + 4) % 8]][:L_j]` with M_j (a different text from A_j). Queue order A_0…A_11, L1, L2, B_0…B_11, `order` = T1…T8: 16,916 prompt tokens and 792 generated positions per configuration.
- **Subsets.** The first wave is the first B sequences inserted; mid-run is every sequence inserted after the first finish.
- **(d)** For the first wave and for mid-run separately: mean KL(R1‖batched) ≤ 3 × mean KL(R1‖single) over that subset's generated positions. **(e)** `admitted_mid_run` ≥ 1 and `max_live` ≤ B. pass(arm, B) iff (d) holds for both subsets and (e) holds.
- **Consequence (pre-registered; not an amendment):** allowed_B(arm) = {1} ∪ ({2, 4, 8} if pass(arm, 8)) ∪ ({16} if pass(arm, 8) and pass(arm, 16)). B = 2 and 4 inherit from 8: they run the same kernel classes (unsorted decode below 64 rows, sorted batched prefill of at most 4 × 2,048 × 6 rows, G0k-judged) and the same cache code. The gate record carries allowed_B, and `require_pass` returns it.
- **Power.** Mutant 27 at K8, B = 8, bound to (d). If (d) does not fire on it, allowed_B = {1} for both arms, recorded as "power not shown"; there is no INCOMPLETE and no cycle. Probe 22 runs in tiny mode only and is reported, never used.
- **No calibrator ceiling.** The denominator is the port's single path, gated by G4-N(i) and G4-F16 (K8) and G2q (K4). A ceiling on it would pin K4 to B = 1 whatever its batched path does (exp_036 recorded K4's KL to R1 at 0.128 / 0.228).
- **Publication.** Any (d) or (e) failure at (arm, B) is published as "batched-path discrepancy detected at (arm, B), leg X; production restricted to allowed_B", with the values. The post may not call that arm "passing batched parity".
- **Behaviour runs at B = 8 as registered** even when G5-BP-lean denies B = 8, because allowed_B is computed in phase 7, after behaviour has run. A batched-path defect can therefore fail K8 through behaviour although production would run at B = 1 (disclosed).

### Port mutants and required controls

**Registered mutants** (exp_036, unchanged): 1 selection on sigmoid + bias; 2 window 512 instead of 513; 3 RoPE on the full-attention layers; 4 (1+w) RMSNorm; 5 `post_attn_norm` ↔ `post_attention_layernorm` swapped; 6 renormalised top-k; 7 afmoe `route_scale` 2.826; 8 muP √d embedding scaling; 9 a retained afmoe attention output gate; 10 bf16 router logits; 11 bf16 lm_head logits; 12 traditional (interleaved) RoPE; 13 q/k-norm applied after RoPE; 14 routing weights from the biased score; 15 SwiGLU with silu on `up`. On real weights mutant 9 fails G0 (strict load), and mutants 10 and 11 fail the value-level bf16-exactness check.

**New mutants** (`gate/port_mutants.py`, frozen with GATE_RULES). Each overrides only the port's `rope_offset` or `attn_mask` hook of `Attention`, never `__call__`, so a post-freeze port fix to `__call__` reaches the mutants too.

| # | Name | Patch | Bound leg |
|---|---|---|---|
| 19 | `rope_restart_per_chunk` | `rope_offset(cache, L)` returns 0 when L > 1 (any prefill chunk), else the port's value | G4-F32 F2 on T9 or a T9 bucket |
| 20 | `decode_window_256` | at decode (L = 1) on a sliding layer's single-sequence rotating cache, a mask over the cache's key slots that keeps only the 256 most recent keys, the query's own included; otherwise the port's mask | G5-D32 decode leg |
| 22 | `batched_rope_positions_in_padded_frame` | on a batch cache, every row's offset is the longest row's (positions counted in the padded frame); bitwise the port at B = 1 | none: a reported probe, tiny mode only |
| 26 | `batch_decode_pos_frozen` | on a batch cache at decode, the offset stays at the value it had at the sequence's first decode step (a stored copy, re-frozen when the shape changes). mlx-lm 0.32.0 uses batch caches even for one sequence, so the defect is present at every B | G5-R1 R-parity at K8 |
| 27 | `batch_decode_pad_keys_visible` | at decode (L = 1) on a batch cache with an array mask, an all-True mask: every cached slot, padded slots included, is visible to the decode query, on every layer. Prefill and single-sequence caches are untouched | G5-BP-lean (d) at K8 B = 8 |

Mutant 27 was chosen before it was measured (decision (b)). At decode, the only False entries of a batch cache's mask are padded slots, so the all-True mask is exactly "batched rows ignore their own left padding at decode": the classic batched-generation defect, a padding mask not applied on the one-token step. Where no row is padded (B = 1, or rows of equal length) it is bitwise a no-op, so the single-sequence checks, G5-R1 and the "single" comparator cannot see it; only G5-BP-lean's batched path can. Every generated token of a padded row is affected, the first included, because under mlx-lm 0.32.0 the first token comes from an `L == 1` step over the batch caches; only an unpadded row keeps it.

**Required controls** (`gate/controls.json`, frozen). Each is bound to one leg. "Caught" means that leg fires on the mutant's series, judged with the unmutated run's bounds and calibrators.

| Check | Required control | Caught iff | If not caught |
|---|---|---|---|
| G2 real-weight mutants (registered) | 1–8, 12–15 | each fails some G2 fp32 check on ≥ 1 of layers {0, 3, 4, 49}, as registered | the registered check **FAILs**, as in exp_036 |
| G4-F32 | 1 | F6 fires (T1–8 or T9) | G4-F32 INCOMPLETE |
| G4-F32 | 6 | F2 fires in T1–8 or T9 | G4-F32 INCOMPLETE |
| G4-F32 | 19 | F2 fires on T9 or a T9 bucket | G4-F32 INCOMPLETE |
| G4-F16 | 6 | the κ rule fires in a set | G4-F16 INCOMPLETE |
| G5-D32 | 20 | a decode-leg rule fires in ≥ 1 range (with the unmutated Csort) | G5-D32 INCOMPLETE |
| G5-R1 | 26 | R-parity fires at K8 | G5-R1 INCOMPLETE (K8 and K4) |
| G5-BP-lean | 27 | (d) fires at K8 B = 8 (first wave or mid-run) | allowed_B = {1} for both arms ("power not shown"); no INCOMPLETE |

**Probes** (`controls.json` `probes`, frozen with the controls): G5-BP-lean's probe 22, run in tiny mode only, reports (d)'s ratio per subset at K8 B = 8 and whether (d) fires. A probe enters no verdict, no INCOMPLETE and no allowed_B.

**The tiny margin.** A weak control could make a correct port INCOMPLETE and cost a cycle, so each required control had to fail its bound leg on the tiny gate before the freeze by a margin per leg type: mean legs (F2, G5-D32 decode mean, R-parity's KL) at ≥ 10 × the bound; ratio legs (κ, (d), R-anchor) at ≥ 10 × the factor; count legs (F1, F6, G5-D32's top-1 count) at ≥ 10 qualifying (token, layer) pairs or positions; max legs (F5, G5-D32 max) at ≥ 10 × the threshold. The margin is measured on the unsharpened seed-29 tiny build (decision (a)); G4-F32's controls 1, 6 and 19 also meet it in F_tiny's sharpened validation. **G5BP/27 is the one exception (decision (e)):** caught on both subsets at the gate's own factor (ratio > 3), with the fixture ceiling recorded beside it. The gate record still writes G5BP/27's 10 × factor margin as `rules.py` computes it (required 30; not met on the tiny build). A control is never changed after the freeze; an uncaught control can be resolved only by a port or harness fix with an independent test. There is no negative control (a "Csort through F3 and F4" control could never fire).

### Calibrators and ceilings

- **F_tiny.** F_tiny = 10 × max_b mean_{t ∈ b} KL(R2F_tiny‖port_tiny)(t) over the 2,048-position blocks b of one 16,384-position sequence, produced by the exact G4-F32 code path on the tiny real-layout calibration build (seed 23, q/k norms × 3, 10 layers, 48 q / 4 kv heads, head_dim 128, window 513, E 64 top-6, hidden 256, 8-bit g64; token ids `np.random.default_rng(5).integers(0, 1008, 16384)`). **Cap: F_tiny ≤ 1e-5** (= 10 × the Csort mean ceiling); above it, the build stops for Andrei; the cap is not raised. **Out-of-sample validation** on seed 29 (token ids from `rng(6)`, laid out as T1–8 and T9) with the full G4-F32 code: F1, F3, F5 and F6 silent; every F2 set and bucket mean ≤ τ / 10; mutants 1, 6 and 19 fire their bound legs at the tiny margin. The result is recorded in `gate/calibration.json` (frozen): **F_tiny = 4.112e-7** (block means 2.06e-8 to 4.11e-8), 24× under the cap; the validation passes with τ = F_tiny and headroom 2.60× (T1–8) and 1.06× (bucket 8,192–16,384) under τ / 10. The margin is thin by construction: with τ = F_tiny, τ / 10 is the calibration's largest block mean. F_tiny is a constant produced on the mini (M4 Pro, MLX 0.32.3), so `tests/test_gate_ftiny.py` runs there only and skips as `build-host only:` elsewhere. exp_036's bug-hunt number 6.4e-7 (a block maximum with fp32 log-prob inputs) is context only, not F_tiny.
- **Csort.** Measured in every run through ForcedTrunk, over T1–8 and T9 for G4-F32 and over each G5-D32 range. Mean ceiling 1e-6 (each G4-F32 set; each G5-D32 range) and max ceiling 1e-4 (each G5-D32 range); a breach is a FAIL of that check, published as "re-association control above its ceiling (cause not attributed)". Derivation: a re-association whose logit change stays within G2's registered head tolerance (|Δlogit| ≤ 1e-3) gives KL ≤ ½ × (1e-3)² = 5e-7 at any position; the mean ceiling is 2× that, and the max ceiling 100 × it. FAIL rather than VOID because an over-ceiling Csort leaves τ meaningless and costs the same cycle.
- **R3.** Mean KL(R2F‖R3) ≤ 1e-2 per set; a breach makes G4-F16 VOID and K8 INCOMPLETE. If the breach is a reference defect, a reference fix is the remedy; if it is the emulation's honest bf16 level, there is none: the run ends INCOMPLETE and the check is published as invalid.
- **The same-run chaos floor** (floor_KL, floor_dis; registered) feeds decode-vs-prefill and G5-R1 R-parity, with no ceiling.
- **S** is a normaliser only (F3), never a bound.

### Verdicts, exits and the record

**Blocking checks.** K8: the shared G0 checks; K8 strict load, converted config, router and head bf16-exact; G1; G2 fp32 forced branch, fp32 natural selection, embedding, head, bf16 forced branch, bf16 natural selection, router margin, T9 buckets, G2q K8, real-weight mutants; G3b; G4-F32; G4-F16; G4-N(i); G5 greedy; G5 decode-vs-prefill; G5-D32; G5-R1 K8 (R-anchor, R-greedy, R-parity); G5 behaviour K8. K4: the shared G0 checks; K4 strict load, converted config, router and head bf16-exact; same port sha; G2q K4; G5 behaviour K4; G5-R1 K4 (R-anchor). **Not blocking:** G3a, the G4 descriptive rows, G5-BP-lean (it sets allowed_B), F4 and every descriptive item.

**Check states:** PASS; FAIL; INCOMPLETE ("power not shown: <mutant>" or "VOID: <calibrator> above ceiling"); missing (a crash, exit 3). **Arm verdict:** FAIL if any blocking check FAILs; otherwise INCOMPLETE if any is INCOMPLETE; otherwise PASS, if every blocking check has a value.

| Condition, first match wins | Exit | Gate run? |
|---|---|---|
| A precondition (P1–P5) failed (the run stops there) | 3 | no |
| K8 FAIL | 1 | yes |
| An exception, or any blocking value missing (and K8 not FAIL) | 3 | no |
| K8 INCOMPLETE, or K4 INCOMPLETE | 5 | yes |
| K8 PASS, K4 FAIL | 4 | yes |
| K8 PASS, K4 PASS | 0 | yes |

**Phases.** 0: P1–P4 (G0k included). 1: G0, G1. 2a: R1 reuse, emulation statistics, G3. 2b: P5. 2c: R2F, R3. 3: G2. 4: K8 bf16 first (G0 dtypes, G4-N(i), noise floor and decode-vs-prefill, greedy continuations, G4-F16, G5-R1, G5-BP-lean at B = 8 and 16, behaviour), then converted in place to fp32 (Csort, G4-F32, G5-D32), then the control mutants, each loaded alone with the cache cleared before every load. 5: K4 (G0 dtypes, G4 descriptive, G5-R1, G5-BP-lean, behaviour). 6: the R1 anchor pass over every sequence collected in phases 4 and 5, with `lm_head` applied only to the generated rows. 7: `rules.py` and the record.

**The gate record** (`results/gate/gate_<UTC>.json`) adds to exp_036's fields: `experiment` "exp_037"; `allowed_B`; `gate_rules_sha256`; `preconditions` (P1–P5 values); `run_counts` (gate runs so far, cycles used); `blind_phase_reached` (true once P5 has passed); `controls` (each required mutant, caught or not, with its bound leg and margin); `probes` (tiny mode only; empty in real mode); `calibrators` (Csort per set and range, R3, F_tiny, τ per set); `g5_logprob_files` (path and sha256 of every G5 log-prob file). `require_pass(arm)` refuses every Kolibri pilot, bench or scored run unless the newest gate record says PASS for that arm and its recorded hashes (port, converted manifest, thresholds, reference, `GATE_RULES_SHA256`, builds manifest) equal the current ones; it returns allowed_B.

**On FAIL or INCOMPLETE.** The record is committed and pushed, partial outputs go to `aborted/`, and Andrei chooses (see "Fix-cycle, freeze and amendment rules"). After the third failing run, or on stop, the gate failure (check, layer, set) is the exp_037 result, and no speed, memory or quality number from an ungated port is ever reported.

### Statistical and model risk (no p-value leg exists)

- **Deterministic or forced rules:** G0, G1, the G2 rows, G3b (given its seeds), G4-F32, G4-F16 and G5-D32. Their risk is model risk, a bound in the wrong place. Mitigations: the tiny validation (G4-F32 at ≥ 10× on the validation build), the required controls and the ceilings. Three bounds carry a stated false-fail risk for a correct port; each was estimated by the final pre-freeze review from exp_036's records, not simulated:
  - **R3's ceiling has no real-weight value behind it**: a breach makes K8 INCOMPLETE, and if the forced emulation's honest bf16 level is above 1e-2, no permitted remedy exists under the freeze. Estimated P(VOID) about 5–10 %, almost all of it on T9 (basis in the known-values ledger). The only remedy that keeps every port value blind is a reference-only R2F / R3 pass before the freeze (numpy, no port value; about 0.5 h on the mbp), which is Andrei's choice.
  - **G5-D32** (factor 100 since the final review): the remaining exposure is the max legs' floor and Csort's max ceiling. exp_036's T9 decode and chunk-64 maxima were 6.5e-7 and 6.4e-7 against the 1e-6 floor (through a different code path), and its one Csort-type maximum, 1.7e-5, sits 6× under the 1e-4 ceiling, whose breach is a FAIL. Estimated at a few per cent; at the factor 10 the review put it at 10–20 %.
  - **G2 bf16 natural selection, leg (p)**: M_port is a maximum statistic (the largest per-pair z), recomputed under MLX 0.32.3's quantised kernels and the vLLM-angle RoPE, with a 1.2× margin on exp_036's record (6.078 against min(8, 1.2 × 6.100) = 7.32). Estimated false-fail risk about 5–10 %.
- **Rules with sampling or chaos variance**, with their risk stated rather than simulated: G4-N(i) (one draw; T9 margin 1.2× on exp_036's draw); decode-vs-prefill and R-parity (calibrated by a same-run chaos floor); G5 greedy and R-greedy (the analytic table below; R-greedy's joint false-fail exposure is at most 1 − (1 − p)², ≤ 0.34 at the 0.32 % miss rate, and equals the single rule's when the runner's tokens coincide with `generate_step`'s); R-anchor (a ratio, same run); behaviour (seeded sampling, at B = 8 even if G5-BP-lean denies it); the router margin (bootstrap).
- **G5 greedy false fail** = P(≥ 5 decisive misses in about 941 decisive positions), exactly:

| Per-position miss rate | P(false fail) |
|---|---|
| 0.05 % | 0.0001 |
| 0.1 % | 0.003 |
| 0.32 % (95 % upper bound from exp_036's 0 / 941) | 0.19 |

The range is 0.01 %–19 %; the rule is kept as registered. At the exact upper bound 0.318 % the value is 0.183.

### Peers are verified, not gated

The peer check (`tools/peer_check.py`, C3) re-runs in S1 (RUNBOOK step 9) under the exp_037 runner (`prefill_batch_size = min(B, 8)`) and MLX 0.32.3 / mlx-lm 0.32.0. exp_036's peer records are not reused: no MLX-computed exp_036 value is. The peers load as built-in mlx-lm model types, with no `trust_remote_code`.
- **Checks:** strict text-only load; parameter count within 2 % of the config arithmetic; NLL(8) ≤ NLL(4) + 0.02 nats/token and KL(8‖4) < 0.2, measured chat-wrapped (each gate text T1–T6 as the assistant turn after the user message "Write a text.", in the model's own template with thinking off, only the text's tokens scored); template and tokenizer byte parity with upstream; EOS ids; and, for G8 and Q36-8, the batched-path check: B = 8 vs B = 1 teacher-forced on the wrapper followed by the stream with mixed prompt lengths 37–1,100 and mid-run admission, under the G5 noise-floor rule, plus a greedy answer-flip rate ≤ 2 % on 30 MMLU-ProX full (non-Lite, gold-consistent) EN items with thinking off.
- **Consequences:** a peer that fails only the batched-path check (its parity, its flip rate, or both) runs every cell at B = 1 (`B=1`; not a drop). A peer that fails anything else is dropped by amendment before any scored run; H3 and H6 then use the remaining MoE peer instead of the mean, and H4 is NOT RUN if Qwen3.6 is the peer dropped. Raw-text NLL (G3a's peer ratio) stays in `arms.<a>.nll`.
- **Fidelity attribution (exp_036 Amendment 6's rule, folded in; decision (c)).** When a family's NLL(8) / KL(8‖4) rule fails, each build of the family is judged by its own token-weighted KL(bf16‖arm) on T1–T4 against the registered KL_MAX 0.2: below it the build passes fidelity and keeps its other verdict; at or above it the build is `speed-only` (no quality cell; out of H8's peer median; H1, the speed cells and the B4 ladder keep it; exit 2). The rows must agree with this run's chat-wrapped NLL (token counts exact, NLL within 0.02 nats/token). The reference record is a new one, produced in S1 under MLX 0.32.3 by `diagnostics/gemma_quant_check.py`, a byte-identical copy of exp_036's producer (sha256 `9393765d…`), run with A–D and `--bf16` before the peer check (RUNBOOK step 9a). A record that does not exist at the freeze cannot be pinned by sha256, so `tools/fidelity_reference.json` (schema `exp037 fidelity reference v1`) names it by rule: it is usable iff exactly one file matches `diagnostics/gemma_quant_check_<YYYYMMDDTHHMMSSZ>.json`, its UTC is earlier than the peer check's `t_start`, the producer has the pinned sha256, the record's `mlx` is 0.32.3 and its `device` is `Apple M5 Max`, and its rows cover exactly the pinned texts for both builds. The peer record carries the chosen record's path and sha256. An unusable record leaves the registered rule unchanged: both builds `fail` if the family rule fails. A repeat of step 9a after a crash moves the earlier output to `aborted/<UTC>-gemmacheck/` first, so exactly one record remains.

### Runtime, port and runner rules

- **The M5 defect under decision F2.** MLX 0.31.2's sorted `gather_qmm` overflows above 32,768 rows on the M5's NAX path; it is fixed upstream in MLX 0.32.3 (mlx#3922, released 2026-09-29) and does not reproduce under 0.32.3 on the same M5 Max (BUGHUNT §6.9). exp_037 runs 0.32.3 with **no mandatory row limit**: no port guard, no runner assertion and no harness restriction (G2 keeps its registered single call). G0k at gate start is the insurance (P3). P2 runs at the gate and at every session start.
- **The port** (`port/kolibri1.py`, exp037-port-1). Sliding layers use `VllmRoPE`, verbatim from exp_036's diagnostic copy (`diagnostics/gate1/mini/bughunt/verify/kit_rope/port/kolibri1.py` l.249–266): angle(p, j) = fp32(p × inv_freq[j]) with inv_freq[j] = 1 / 10000^(2j / 128) in fp32, vLLM's `RotaryEmbedding._compute_inv_freq`, identical to the reference's `rope_inv_freq`; MLX's `base=` path differs by −2 to +8 fp32 ulps. `Attention` gets two hooks, `rope_offset(cache, L)` and `attn_mask(mask, cache, L)`, which change no output bitwise. vLLM stores cos and sin in bf16; neither side is bit-faithful to that, and the rounding is flat in position. The bf16 residual add is kept. Both port sha256 values (exp_036's `cd6153b8…`, exp_037's in the hash table) are named in the post.
- **Loads.** mlx-lm 0.32.0 executes a `model_file` only with `trust_remote_code=True` (mlx-lm #1385, CVE-2026-5843). Every kit load of our own `kolibri1.py` passes it through `port.convert.model_file_trust(dir)`, which grants it only after `check_port_file` has verified the directory's port file against the kit's, refuses a foreign or stale file before anything executes, and gives `{}` to a directory without `model_file`. The peers never get the flag. The conversion path also refuses a source that ships code of its own (`auto_map`).
- **One BatchGenerator construction** (`runner.generate.make_batch_generator`): `BatchGenerator(fp32_logits(model), stop_tokens=[[e] for e in eos], sampler=…, completion_batch_size=B, prefill_batch_size=min(B, 8), prefill_step_size=2048, max_kv_size=None)`, used by `run_cell`, the gate's G5 checks, the bench and the peer check's batched path. Under mlx-lm 0.32.0 prompts admitted together are right-padded and finalised to left padding, each sequence's cache comes from `make_prompt_cache` and the batch caches from each cache class's `merge`. The step log reads `BatchGenerator.stats()` windows.
- **allowed_B.** `runner/run.py` (pilot, session, cell) refuses a Kolibri cell whose B ∉ allowed_B(arm). The plan rule clips each Kolibri cell's memory-rule B to the largest allowed value ≤ it; the crash fallback steps down within allowed_B; C1 runs at the largest allowed B ≤ the memory-rule B and is NOT RUN ("no batching to control") if that is 1; the descriptive speed cells run Kolibri only at B ∈ allowed_B ∩ {1, 2, 4}.
- **Version binding.** At every session start, `runner/guard.require_environment` checks P2's pins against the newest real-mode gate record and refuses on a mismatch (a pre-registered consequence, not a failure).
- **Work and private files.** New work files go only under `$EXP036_WORK/exp037/`, except where the inherited bench cells keep exp_036's default paths: H8's K8 dump in `$EXP036_WORK/kl/<UTC>/` and C1's full records in `$EXP036_WORK/c1/` (new UTC-stamped names beside exp_036's; nothing of exp_036's is touched); exp_037's raw, pilot and aborted private records go under `$EXP036_PRIVATE/exp037/`, and the reused withheld manifests are read from `$EXP036_PRIVATE/manifests/`. `scorers/score_all.py` reads raw records from `<private>/exp037/raw/` and manifests from `<private>/manifests/`.

---

## Fix-cycle, freeze and amendment rules

These rules are part of the pre-registration and need no later reading. RUNBOOK and BUILD_SPEC defer to them.

**The freeze.**
- **When.** The freeze is the push to `origin` of the pre-registration commit whose hash table above carries the filled `GATE_RULES_SHA256`.
- **What it fixes.** Every run after it is post-freeze, exit 3 included. After it, nothing in the frozen scope changes until exp_037 closes.
- **Before the push.** Every document and file is a draft. Changes are free and recorded in BUILD_LOG.md; no amendment exists before the freeze.
- **Enforcement.** P1(c) and P1(e) check it in code.

**What is frozen, by hash.** The 20 scopes of the table above (`tools/hash_tree.py` `SCOPES`), among them `gate_rules`: the tree of `gate/` restricted to `calibration.json`, `controls.json`, `port_mutants.py`, `rules.py` and `thresholds.json`. `gate/rules.py` holds every pass/fail decision of the gate (G0, G2, G3b, G4-F32, G4-F16, G4-N(i), G5 greedy, decode-vs-prefill, G5-D32, G5-R1, behaviour and G5-BP-lean), BLOCKING, the controls evaluation, the arm verdict, the exit code and allowed_B; a G1 test asserts that no module under `gate/checks/` emits a `"pass"` key. The existing scopes also freeze the gate texts, the tripwire and caveat code and margins (analysis Tier 1), the batching code (runner) and the plan rules. `tests` freezes G1's weight-free suite with its skip policy (`tests/conftest.py`) and every other test, and `env` the three environment scripts (`exp037.settings.sh`, which sets `PY` and `$EXP037_BUILDS`, `setup.sh` and `versions.json`); so editing, skipping or removing a test, or changing `gate/checks/g1_synthetic.py`'s exclusions (which a G1 test asserts), needs an amendment line like any other code change. After the freeze these can change only by the amendment types below, each giving its new `hash_tree:` line. **No amendment may change `gate_rules`, `thresholds` or `gate_text`**: P1(b) refuses an amendment line for any of the three, and P1(e) refuses a direct edit of the table.

**No relaxation route.**
- exp_037 has no threshold-change amendment and no verdict-rule-change amendment.
- A check shown to be invalid keeps its failure. "Invalid" means a correct implementation evidently cannot meet it. The failure is published, and redesigning the check belongs to a later experiment.

**Gate runs, cycles and crashes.**
1. **Gate run.** A real-mode `gate/run_gate.py --all` that writes a record with exit 0, 1, 4 or 5.
2. **Fix cycle.** Each gate run after the first, if it follows a gate run whose K8 verdict was FAIL or INCOMPLETE or whose K4 verdict was FAIL or INCOMPLETE (any gate run that exited 1, 4 or 5). At most **two** cycles, so at most three gate runs. Cycles are counted by runs, never by amendments.
3. **Exit 3** (a precondition or crash) is not a gate run and never a cycle.
   - The partial output goes to `aborted/<UTC>-gate/` with a NOTE.md, and the run may be repeated unchanged. After two unchanged repeats, Andrei decides.
   - A code change after an exit 3 is made only as a typed gate fix (a test that fails before the fix, with an oracle independent of every gate value, and Andrei's go).
   - If the crashed run had passed P5 (`blind_phase_reached = true`), the amendment's first line names the crashed run (its UTC) and every port-against-reference value it computed, and those partial values are published from `aborted/`. The next run still does not count as a cycle. P1(d) refuses a changed re-run until such a first line exists.
   - **Interrupts count as crashes.** Ctrl-C, SIGTERM, SIGHUP and SystemExit during the phases are recorded like any crash: the record is written with `blind_phase_reached`, an exit-3 run moves to `aborted/`, and only then does the process end. A K8 check that had already failed makes the interrupted run exit 1, a gate run (the exit table: K8 FAIL wins over a crash). A run killed outright (SIGKILL, an out-of-memory kill, a power cut, a kernel panic) leaves `results/gate/<UTC>/` without a record; P1(d) refuses every gate run until `gate/run_gate.py --close-orphans` has judged its phase files as they stand, with the error "interrupted", written its record and moved it to `aborted/` (or, if a K8 check had failed, kept it as a gate run).
4. **P1(d) refuses a re-run** after a gate run that did not exit 0 unless at least one of the port, reference tree, runner tree, gate-code tree or builds manifest changed. It also refuses when two cycles are used, while a run directory has no record, and when a code change follows a blind crash that no amendment's first line names.
5. **Every re-run is the whole gate,** for both arms and every check. Dumps are reused only when their keys match.

**Amendments after the freeze.**
- **Gate fix** (type "gate fix", typed by the main session, on Andrei's go). It is the only amendment that may precede a re-run, or a repeat after an exit 3 with a code change. One amendment may list several fixes. Each fix:
  - is a code change in `port/`, in the generation path of `runner/`, or in gate measurement code (`gate/` outside the `gate_rules` scope);
  - comes with a test that fails before the fix and passes after it, whose oracle is independent of every gate value (a tiny checkpoint, the reference, or a symbolic check); the test changes the `tests` scope, so the amendment gives its `hash_tree: TESTS_SHA256` line;
  - **never changes a mutant, a control or a threshold.** An uncaught required control can be resolved only by a port or harness fix with such a test. Because the mutants override only the port's hooks, a port fix reaches them without changing them.
- **Reference fix** (type "gate fix", subtype reference). It must also cite a vendor or vLLM v0.29.0 line, and it invalidates every reference dump.
- **Other permitted types,** unchanged from exp_036 and touching no frozen scope: peer drop, extractor fix, plan, re-pilot, Tier-2 analysis, and process record (it records Andrei's choices and changes no rule). Amendment 0 records sign-off choices, as in exp_036.
- **Power log** (type "power log", after S3 only): when the frozen `tools/power_log.py` cannot read a post-freeze raw log, the fix to that file and its test, with their `hash_tree:` lines for TOOLS_SHA256 and TESTS_SHA256, and the power records it then writes. It changes no other file, reads no result and decides nothing ("Power and energy (descriptive)").
- **Writer.** The mbp session appends amendments verbatim (one writer). An amendment written after a gate result says so in its first line.

**Pre-registered consequences** (not failures; no amendment): allowed_B per arm; exit 4, after which H1, H7, H8 and D1 are NOT RUN if the cycles end with K4 failing; a session refused for version drift; the tripwire; the plan rule's outcome, STOP included.

**Diagnostics between runs.** They are allowed. Their outcome-to-action rules are committed before they run. They write no gate record, never replace a gate value, and are not cycles. Their only permitted consequences are a gate fix, a reference fix, or stop and publish.

**Who acts.**
- After a FAIL or INCOMPLETE, Andrei chooses one of: a gate fix plus a re-run (if cycles remain), diagnostics, or stop and publish. Each choice is recorded with its UTC time and label. Stop and publish is available at every point.
- The main session drafts fixes. Every re-run needs Andrei's go.
- **End of the gate.** After the third failing run, or on stop, the gate failure (check, layer, set) is the exp_037 result. No speed, memory or quality number from an ungated port is reported.

---

## Pilot and the rule that fixes n and max_tokens

Unchanged from exp_036, except that every Kolibri cell's B is clipped to allowed_B and the crash fallback steps within allowed_B. exp_036's Amendments 3 (GPQA-Diamond EN is all 198 items) and 4 (the MMLU allocation and the full-split pools) are folded in.

**Pilot** (S1, after the gate and the bench cells; its outputs are never scored for accuracy):

| Arm | Items |
|---|---|
| K8 | GPQA main (non-Diamond, filtered by Diamond Record ID) EN 8 + DE 8 (ellamind `is_diamond = false`); MMLU-ProX full (non-Lite, gold-consistent) EN 8 + DE 8, parallel; IFBench 8 (discarded); RGB `en_int` 16 closed-book + 8 forced-answer; AIME26 EN 4 (discarded) |
| G8, Q36-8 | the same minus AIME |
| K4 | the same minus AIME and forced-answer |

Each arm runs at the memory-rule B (clipped to allowed_B for K8 and K4) at the initial caps, task-grouped. Every decode step is logged (n_live, padded length, step seconds).

**Measured:** completion-length distribution per (arm, task) (mean, SE, max), truncation count (a completion that ends at the cap, including one still inside its reasoning segment), reasoning-segment status counts, parse failures, prefill rate, and the per-step decode log (`results/pilot/<UTC>/<arm>/*.steps.jsonl`, about 190 B per decode step). The pilot's step logs are committed, so that rule 4's step-model fit, and with it the plan, can be re-derived from the public repository; the plan's run record gives their sha256. One over 50 MB stays uncommitted and is copied to the mini instead (see "Evidence layout"). The sessions' step logs are git-ignored working files (same section).

**Rules** (`runner/plan_fix.py`, deterministic, constants in `runner/plan_rules.json`):

1. **Memory rule for B.**
   - B per cell = the largest of {16, 8, 4, 2, 1} with need(B) ≤ 0.9 × L, where need(B) = weight bytes (summed safetensors sizes of the arm's directory) + 2 × B × (max prompt + cap) × KV bytes per token + B × fixed window bytes + 4 GiB. The factor 2 covers the transient copy when the batch KV grows or admits a sequence; every row is padded to the longest live row. AIME cells are capped at B ≤ 8. Tasks are never mixed in a batch. The runner calls `mx.set_cache_limit(4 GiB)`.
   - **For K8 and K4, B is then clipped to the largest value of allowed_B(arm) ≤ it** (G5-BP-lean).
   - KV bytes per token from the config: Kolibri, Gemma 4 and Qwen3.6 20,480; Qwen3.8 65,536. Fixed window bytes per sequence: Kolibri ≈ 42 MB, Gemma 4 ≈ 210 MB, Qwen3.6 its DeltaNet state; computed by `tools/kv_bytes.py` (checked against mlx-lm 0.32.0's caches at build time).
   - L = max(MLX `max_recommended_working_set_size`, `iogpu.wired_limit_mb` × 2²⁰), both recorded. Measured on the mbp on 2026-10-03: 115,448,725,504 B = 107.5 GiB (84 % of 128 GiB), `iogpu.wired_limit_mb` = 0 (default). At that L, K8 runs GPQA and MMLU at B = 8 (32k caps; 4 after a raise to 65k) and IFBench and RGB at B = 16, before the clip.
2. **Cap raise.**
   - Trigger, per task: in any model's pilot cell, ≥ 1 truncation, or any completion longer than 0.5 × the cap (0.4 × the cap for GPQA, whose pilot uses easier non-Diamond items).
   - Effect: that task's cap goes straight to its ceiling for **all** models (GPQA and MMLU 65,536; IFBench and RGB 32,768; AIME 98,304), and the memory rule is re-evaluated. At most one raise per task.
3. **Parse and reasoning gate.**
   - Parse failure = no answer extractable after the reasoning segment (GPQA, MMLU, AIME), or an empty post-reasoning answer (IFBench, RGB), among non-truncated items. A completion that ends unclosed at the cap counts as a truncation, never as a reasoning-gate failure.
   - A cell is BLOCKED only if it has ≥ 2 parse failures **and** the pre-written diagnostic (`runner/plan_fix.diagnose`) shows a delimiter or extractor mismatch. Otherwise the rates are descriptive.
   - A BLOCKED cell is fixed only by a numbered extractor or delimiter amendment before any scored run, and is then re-piloted. V-row extractors are never amended (C10). IFBench and AIME pilot outputs never drive an amendment.
   - For Gemma 4, a completion that never opens a thought channel (status "none") is scored on the whole completion and does not count against the reasoning gate. For Kolibri and Qwen at effort high, status "none" is impossible (the think block is prefilled) and is recorded as a defect.
4. **n.**
   - Each cell is projected by a deterministic continuous-batching simulation: B slots; completion lengths resampled (seed 36) from the pilot distribution of that (arm, task) and scaled by (mean + 1 SE)/mean; GPQA-D cells scaled by a further × 1.25 (main → Diamond); pilot items truncated at the initial cap counted at the post-raise cap; step time = a + b·n_live + c·n_live·padded_len, fitted per arm (a, b, c ≥ 0) on the decode steps of the last third of that arm's pilot; prefill at the measured rate; the cell's tail included. Tier-B cells with no pilot cell of their own use fixed ratios: peer AIME = the arm's GPQA pilot lengths × K8's AIME/GPQA length ratio; K8 at effort none, low and medium = 0.08 ×, 0.4 × and 0.7 × K8's effort-high lengths in the same task; Q38-8 (B9) = Q36-8's GPQA pilot lengths, with a step model fitted on Q38-8's descriptive speed cells; B4 = a fixed 1.4 h.
   - Projected hours T(plan) = Σ simulated cell hours × 1.15.
   - S1 hours = the sum of the machine intervals recorded by the S1 blocks (preflight `--deep`, the port refresh, peer check, gate, bench, pilot), not wall-clock time.
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

Every n_M set is a house-monotone Webster (Sainte-Laguë) allocation proportional to MMLU-ProX-Lite's own category counts (exp_036 Amendment 4): smaller sets are subsets of larger ones, items within a category are a prefix of the registered seed-36 order, n_M = 588 is all of Lite, EN and DE get identical ids, and every category has at least 5 items at every rung.

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

5. **Freeze of the plan.**
   - The output is `results/plan_fixed_<UTC>.json` plus `results/AMENDMENT_<k>_<UTC>.md` (type "plan"), where k is the next free amendment number in this file and `amendments/`. `plan_fix` never overwrites a file.
   - The plan amendment holds: plan, n_M, caps (any raise), B per cell (with each Kolibri arm's allowed_B), the L used, the ordered queue and its S2/S3 split, Tier-B items, the H2 and H7 row sets, the post-stratification category counts, the per-category Webster allocation at the fixed n_M, the sha256 of every manifest (public and withheld), gate and peer-check record sha256s, whether K8 runs, whether the forced-answer cells run, and power recomputed at the fixed n.
   - It is appended verbatim to this file, committed and pushed from the mbp before S2. The runner refuses S2 unless it is at HEAD and HEAD equals upstream.
   - A re-pilot after a BLOCKED cell produces a new plan amendment that names the one it supersedes; the earlier amendment stays in place.
6. **Cells and sessions.**
   - A cell's n is fixed before it starts, and every item in it runs to completion (resumable across sessions), so a time cutoff never truncates a cell towards short, easy items.
   - `run.py session` stops admitting new cells once the session's elapsed hours plus the next cell's projection exceed 16 h; started cells finish. S3 continues the same queue.
   - **Tier-A cells always run to completion.** If S3 ends with Tier-A cells outstanding, a continuation S3b runs automatically, with no discretion, counted in the budget. A Tier-A cell is admitted in S3b only if the projected run total stays ≤ the overrun ceiling (44 h); otherwise the hypotheses that need it are NOT RUN with p = 1 and m unchanged. Tier-B cells not started by the end of S3 are NOT RUN.
   - If K8 cannot run (below), K4 replaces K8 in H2, H3, H4 and H6 through the plan amendment, and H7 is NOT RUN. H2 then keeps its −4 pp margin but is titled "at MLX 4-bit", and the H8 KL and K4's KL to the reference are cited next to it.
   - **Crash fallback.** `run.py` writes a heartbeat (current cell, B). On a start that finds a stale heartbeat for an unfinished cell, that cell resumes at the next lower B in {16, 8, 4, 2, 1} (for K8 and K4: the next lower B within allowed_B), and its records carry `b_fallback_from`. After two fallbacks, or when there is no lower B, the cell moves to `aborted/` with a NOTE.md, its hypotheses are NOT RUN (p = 1), and the queue continues.

**K8 working-set rule.**
- Preflight computes K8's memory-rule B for every K8 cell at the current L and at L' = 112 GiB. It prints `sudo sysctl iogpu.wired_limit_mb=114688` (112 GiB, leaving ≈ 16 GiB for macOS) whenever L' would raise the B of some K8 cell, or whenever need(1) > 0.9 × L. Andrei runs it by hand and preflight is re-run; the value goes into the sign-off. The kit never calls sudo. At the measured default L (107.5 GiB) and the initial caps, no line is printed.
- If Andrei declines, B for K8 follows the memory rule. If need(1) still exceeds 0.9 × L (L < ≈ 90 GiB), K8 is **NOT RUN** and K4 becomes primary through the plan amendment, before any scored run.
- The sysctl resets on reboot. `run.py` refuses any K8 cell (exit 1, printing the sysctl line) if the current L is below the L the plan used.
- The 64 GB mini measured 51.84 GiB (81 %).

---

## Confirmatory hypotheses (H1–H8, Holm family)

The hypotheses, nulls, margins, rules and power tables are exp_036's, unchanged in substance; the bootstrap seeds move to `exp037`, and exp_036's Amendments 3, 5 and 6 are folded in where they apply.

**Common rules.**
- **Two Holm families over H1–H8.** For each H, p is the one-sided p-value for its null (CONFIRMED side) and p_rev the one-sided p-value for its reverse null (θ on the confirm side of its REFUTED threshold). Holm is run over the eight p, and separately over the eight p_rev, each at family-wise α = 0.05.
- **CONFIRMED** iff the step-down Holm-adjusted p ≤ 0.05. **REFUTED** iff not CONFIRMED and the Holm-adjusted p_rev ≤ 0.05. Otherwise **INCONCLUSIVE**; it is published as "INCONCLUSIVE (leans refuted, nominal 95 %)" if the unadjusted p_rev ≤ 0.025. Decisions use the adjusted p only; the bound at the adjusted level is reported for information.
- Bootstrap p-values: p = (1 + #{θ\*_b on the null side of θ₀}) / (B + 1), where θ\*_b = θ₀ counts on the null side. Every null below includes equality.
- B = 10,000 per hypothesis, from its own seed (the first 8 bytes of sha256(`"exp037|" + H_id`) into PCG64), so results do not depend on evaluation order or on which hypotheses ran. Any H whose raw p or p_rev lies within a factor of 2 of its Holm threshold is recomputed with B = 100,000 before Holm is final. The Monte Carlo SE of every p is reported.
- **m = 8 is frozen at the plan amendment.** An H that becomes NOT RUN later (crash, overrun, peer drop) keeps its slot with p = p_rev = 1.
- Truncation, parse failure and unclosed reasoning count as wrong. Only the text after the reasoning segment is scored.
- **Paired truncation sensitivity** for H3, H4, H6 and H7: each is recomputed after excluding every item on which any compared arm truncated. The per-arm truncation rate is reported in the same table row as the verdict. If the sensitivity changes the verdict, both are published and the headline uses a verdict only if the two agree; otherwise it says INCONCLUSIVE (truncation-sensitive).
- **Peer positive control (E8) flag.** If |D_peer| = |local − vendor| > 8 pp for a peer on a row an H uses (H3: MMLU EN/DE; H4: IFBench; H6: RGB closed-book), the H's verdict is still computed as specified but is published as "<verdict> — peer control failed (<peer>, <row>)" and may not appear in a headline. For H4 the post then leads with the Qwen3.6 local vs vendor discrepancy.
- **The H2 tripwire** (T-G3, below) labels or caveats every Kolibri quality verdict; it changes no verdict word.
- Power is quoted at α/8 (worst case, primary) and at α/3 (if five others are confirmed first).

### H1 — Speed class (Q1)
Same host and runtime (M5 Max, MLX 0.32.3 / mlx-lm 0.32.0): Kolibri 4-bit decodes at batch 1 at least 0.75 × as fast as Gemma 4 26B-A4B 4-bit.
*Null:* K4/G4 batch-1 decode ratio ≤ 0.75.

**Measurement.**
- K4 and G4 are both resident (≈ 56 GiB), with one untimed warm-up each.
- 10 blocks. In each block, the order of K4 and G4 is randomised (seed 36).
- One run is the first 1,024 tokens of the model's own tokenisation of `exp_007_hardware_comparison/fixtures/padding/pad_4k.txt`, loaded by its real name (hard fail if missing), with a fresh cache, greedy decoding, EOS masked and exactly 256 generated tokens. We record mlx-lm `generation_tps`, which excludes prefill.
- 30 s idle between runs; power mode, power source and `thermalState` are logged per block.
- Statistic: the mean over blocks of ln(tps(K4)/tps(G4)), i.e. the geometric-mean ratio. One-sample t-test on the 10 block log-ratios (df = 9): p = 1 − F_t9((mean − ln 0.75)/(sd/√10)), p_rev = F_t9((mean − ln 0.75)/(sd/√10)); the 95 % interval uses t₉,₀.₉₇₅.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05. REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (ratio < 0.75). INCONCLUSIVE otherwise.

**Prediction.** ≈ 0.85 with the quantised head. K4 reads ≈ 2.2 GB/token at batch 1 (≈ 2.0 GB of weights plus 0.2 GB of fp32 router weights) vs ≈ 2.2 GB for G4, but Kolibri has 50 layers to Gemma's 30, so more kernel dispatches. MLX 0.32.2's quantised-MoE kernel changes may move decode speed either way; nothing is credited for them.

**Power** (per-block log-ratio SD 0.042, i.e. 3 % per run): 1.00 at a true 0.9; 0.93 / 0.98 (α/8 / α/3) at 0.80; 0.47 / 0.67 at 0.78. At SD 0.06: 0.62 / 0.80 at 0.80.

**Descriptive.** Prefill tok/s at 4,096 tokens (ratio); K8/G8; the Qwen arms; aggregate decode at B ∈ {1, 2, 4} (Kolibri only at B ∈ allowed_B).

### D1 — Fit on a 64 GB node (Q1, Q5; deterministic check, outside Holm)
Kolibri 4-bit, as the only resident model, peaks at ≤ 46.66 GiB of MLX memory with a 64k-token context: 0.9 × the 51.84 GiB Metal working-set limit measured on the mini.
*Null:* peak at 64k > 46.66 GiB.

**Measurement.** K4 alone. Per cell: `mx.clear_cache()`, `mx.reset_peak_memory()`, then prefill of exactly N Kolibri tokens of an exact-length prefix of `pad_120k.txt` (real name), `prefill_step_size` 2048, then 512 greedy tokens, then `mx.get_peak_memory()`; prefill tok/s is recorded too. N ∈ {32,768; 65,536}, 3 reps each.

**Rule.** CONFIRMED if median peak(64k) ≤ 46.66 GiB. REFUTED if median peak(64k) > 51.84 GiB (does not fit at 64k even with zero headroom). INCONCLUSIVE ("fits at 64k only without the 10 % headroom") otherwise. peak(32k) is descriptive.

**Prediction.** 41.0 GiB of weights (incl. the fp32 router) + 1.25 GiB of KV + ≤ 1.5 GiB transient ≈ 43.7 GiB at 64k.

**Transfer** to the M4 Pro is an assumption; the mini is not used. "Fits" means a dedicated node: the mini keeps Ollama models resident for the CasaSol bot, and they would have to be unloaded. K8 (77.4 GiB of weights) not fitting 64 GB is arithmetic, not a test.

### H2 — Scorecard (Q2)
Through the gated port at MLX 8-bit, Kolibri lives up to its post-training scorecard on the public, deterministically scored rows: its mean shortfall against the vendor's values is less than 4 pp.
*Null:* mean D ≤ −4 pp.

**Rows** (vendor values: Table 28, p. 100/189, and Table 29, p. 101/189; frozen with N_v, n_v and page in `analysis/vendor_values.json`):

| Row | Vendor | Label | Items | Our protocol |
|---|---|---|---|---|
| GPQA Diamond EN | 84.3 | V | 198 (all Diamond items; exp_036 Amendment 3) | `GPQA_DIAMOND_COT_V2` prompt, option shuffle, `tulu_answer_v2(4)` |
| GPQA Diamond DE | 81.3 | V | 198 | `GPQA_ELLAMIND_DIAMOND_COT_DE`, `tulu_answer_de` |
| MMLU-Pro CoT EN | 80.0 | R | MMLU-ProX-Lite EN, n_M, post-stratified | `MMLU_PRO_COT_V2` prompt, `tulu_answer_v2(10)` |
| MMLU-ProX CoT DE | 75.5 | R | Lite DE, n_M, post-stratified | eval-framework German Tülu CoT prompt with A–J, German lenient extractor extended to A–J |
| IFBench loose-prompt | 78.1 | R | 300 | official allenai checker |

The row set is fixed at these 5 rows (4 under P10). AIME 2026 EN 96.0 (V) and DE 90.0 (R, our frozen German NeMo-Skills wrapper) form a secondary 7-row estimate if B1 runs; it is reported as an estimate and 95 % CI only, with no verdict, and its per-row D is descriptive.

**Measurement.**
- D_r = ours_r − vendor_r; D̄ = unweighted mean over rows.
- MMLU rows are **post-stratified**: each category's mean is weighted by that category's share of the full MMLU-ProX test split of that language, counted at manifest build by the frozen rule (item 3787, whose two gold fields disagree, is still counted: its category is valid) and recorded in the plan amendment. The unweighted Lite mean is descriptive.
- Uncertainty: an item bootstrap within rows, stratified by category for the MMLU rows. If B10 runs, the GPQA rows use the two-pass mean, and the variance of each GPQA row is estimated from the passes as Σ(x₁ − x₂)²/(4n²) instead of by resampling items. The vendor's avg@N term is drawn per resample as N(0, p_v(1 − p_v)/(N_v · n_v)), with p_v the vendor value, N_v = 8 (GPQA), 5 (IFBench) and 16 (AIME secondary), and n_v the vendor's item count (198 for GPQA EN); it is 0 for MMLU, where the vendor ran the full sets once.
- Any recorded GPQA EN exclusion stops the analysis with an error (exp_036 Amendment 3).

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: D̄ ≤ −4). REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: D̄ ≥ −4). INCONCLUSIVE otherwise.

**H2_detail (pre-registered).**
- Per-row D with CI, V/R label, the per-model truncation rate and the result excluding truncated items; every row with |D_r| > 5 pp named.
- **Protocol control (from E8):** the peers' mean D over the shared rows (MMLU EN, MMLU DE, IFBench; G8 and Q36-8) with 95 % CI, and the difference-in-differences DiD = D̄_K,shared − mean D_peer,shared with 95 % CI.

**Power** (k = 1; post-stratification design effect 1.11, registered and conservative: under the proportional Lite allocation it is about 1.003; α/8 / α/3; computed at n = 197 for GPQA EN, and no figure moves by more than 0.00095 at 198):

| n_M | SE(D̄) | True D = 0 | D = −1 | D = −2 | D = −3 |
|---|---|---|---|---|---|
| 588 | 1.09 pp | 0.88 / 0.94 | 0.60 / 0.74 | 0.25 / 0.39 | 0.06 / 0.11 |
| 406 | 1.14 pp | 0.84 / 0.92 | 0.55 / 0.69 | 0.23 / 0.35 | 0.05 / 0.11 |
| 294 | 1.20 pp | 0.80 / 0.89 | 0.50 / 0.64 | 0.20 / 0.32 | 0.05 / 0.10 |
| 196 | 1.31 pp | 0.71 / 0.83 | 0.42 / 0.57 | 0.17 / 0.28 | 0.04 / 0.09 |
| 154 | 1.39 pp | 0.65 / 0.78 | 0.37 / 0.51 | 0.15 / 0.25 | 0.04 / 0.08 |
| 154, 4 rows (P10) | 1.57 pp | 0.52 / 0.66 | 0.28 / 0.41 | 0.11 / 0.20 | 0.03 / 0.07 |

**Under the drift we expect (1–2 pp, from the int8 grid vs FP8-QAT and three reconstructed rows), H2 is more likely INCONCLUSIVE than CONFIRMED.** REFUTED needs a true shortfall well beyond the margin: power 0.88 at D = −8 (n_M = 588), 0.71 at n_M = 196.

**Why −4 pp.** It is one GPQA item in 25, averaged over rows. Our 8-bit is a different grid from the vendor's FP8-QAT, and three rows are reconstructions, so a 1–2 pp drift is not a failure to live up to the scorecard. The margin was re-confirmed with decisions A–E.

### H2 tripwire T-G3 (Tier 1, frozen; no verdict changes)
- **When.** Evaluated once, after H2 and its protocol control. It draws no new random numbers.
- **Inputs.** The bootstrap replicates of D̄_K over H2's frozen rows (5, or 4 under P10), primary values, unadjusted, with B as used by the H2 verdict (10,000, or 100,000 if escalated), seed sha256("exp037|H2"); and the protocol control's DiD replicates (seed sha256("exp037|H2|protocol"); shared rows MMLU EN, MMLU DE, IFBench; peers G8 and Q36-8).
- **Bounds:** UB_K = the 0.95 quantile of the D̄_K replicates; UB_DiD = the 0.95 quantile of the DiD replicates.
- **Rule. TRIPPED iff UB_K < −0.10 and (the protocol control is NOT RUN or UB_DiD < −0.10).** `analysis/margins.json`: `H2_tripwire_ub` −0.10, `H2_tripwire_did_ub` −0.10, `H2_tripwire_q` 0.95.
- **Arm.** K is H2's arm: K8, or K4 if K8 is NOT RUN under the registered rule. If H2 is NOT RUN, the tripwire is NOT_RUN and no sentence is published for it.
- **The constant.** −10 pp is 2.5 × H2's −4 pp margin, and above E8's 8-pp protocol band. It is not calibrated against any measured misreading, and E13 was cut, so it has no calibration point. Both facts are disclosed.
- **Power** (D̄_K leg only; normal approximation; the DiD leg can only lower these values; P(UB_K < −10 pp) = Φ((−10 − 1.645·SE − D) / SE)):

| True D̄_K | SE 1.09 (n_M 588) | SE 1.31 (196, P7–P8) | SE 1.39 (154, P9) | SE 1.57 (P10) |
|---|---|---|---|---|
| −4 pp | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| −8 pp | 0.0003 | 0.0008 | 0.0010 | 0.0018 |
| −10 pp | 0.05 | 0.05 | 0.05 | 0.05 |
| −12.5 pp | 0.74 | 0.60 | 0.56 | 0.48 |
| −15 pp | 0.998 | 0.985 | 0.97 | 0.94 |
| 50 % point | −11.8 | −12.2 | −12.3 | −12.6 |

- **If TRIPPED:** H2 is computed as registered but published as "<verdict>, implementation-uncertain (G3 blind spot)", and never headlines. Q2's answer is the tripped row of "Plain answers", which takes precedence over every other Q2 row. H3, H4, H6, H7, H8, E1–E5, E9, E10 and E12 carry the same label and leave the headlines; H1, H5 and D1 are unaffected. There is no Hugging Face upload and no re-run inside exp_037; the post names the vendor-runtime anchor as the test that would separate the two readings. No gate verdict changes.
- **If not TRIPPED:** every Kolibri quality verdict carries the caveat sentence: "A misreading shared by port and reference that moves Kolibri's measured shortfall on these rows by up to about 12–15 pp is more likely missed than caught locally; only one beyond about 15 pp is reliably caught." An H2 REFUTED between −4 and −10 pp is worded so that it cannot distinguish the model from such a misreading.

### H3 — Knowledge gap, English and German (Q3, Q4)
Kolibri 8-bit scores below the mean of Gemma 4 8-bit and Qwen3.6 8-bit on MMLU-ProX-Lite, pooled over parallel English and German items. Vendor gaps: −4.4 EN, −6.0 DE, −5.2 pooled (p. 100/189).
*Null:* K8 − peer mean ≥ 0.

**Measurement.** The same item ids in EN and DE, under the Webster allocation (a plain item mean over the n_M set, its categories weighted close to the full split's shares). Per item d = K − (G + Q)/2. Paired item bootstrap stratified by language and category.

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

This is the hypothesis most sensitive to throughput, and that is stated now. If allowed_B(K8) = {1}, throughput may fall several rungs (see "Sessions & budget").

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
- The EN ratio on our own EN gate texts plus MMLU-ProX-Lite EN question text, labelled "not FineWeb".
- A digit-dense German subset (≥ 5 % digits; Kolibri splits every digit).

**Disclosure: H5 is not blind.** H5 is deterministic: no model runs, only three tokenizers over a pinned file. exp_036's tests and dry run computed it on the mini with the real tokenizers and the pinned FineWeb-2 shard during exp_036's build (2026-10-03/04), and the verdict they printed was CONFIRMED; exp_037's dry run computes it again. The threshold was not changed. Read H5 as a check of a deterministic vendor claim, not as a blind prediction.

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

**Descriptive.** K4 vs G8 on the Tier-A task rows; K4 vs G4 on speed (H1), fit and logit fidelity (H8) only, since G4 runs no task cells. If G4 is `speed-only` (peer check), its logit fidelity is the S1 fidelity record's KL(bf16‖G4) and its descriptive KL(8‖4). These confound model with bits and are labelled so.

### H8 — 4-bit logit fidelity against the peers, on the six gate texts (Q5)
On our six plain gate texts, 4-bit costs Kolibri no more fidelity than it costs the peers: Kolibri's KL(8-bit ‖ 4-bit) per UTF-8 byte is ≤ 1.5 × the median of Gemma 4's, Qwen3.6's and Qwen3.8's, all built with the same affine g64 recipe. A peer family with an excluded or `speed-only` build leaves that median (edge case 7); if no family remains, H8 is NOT RUN. The estimand is these six texts (2 EN, 4 DE; blog, legal and web prose), not English or German text in general.
*Null:* ratio ≥ 1.5.

**Measurement.**
- Every model scores each gate text T1–T6 as the assistant turn after the single user message "Write a text.", rendered through its own chat template with thinking off (Kolibri `reasoning_effort="none"`; Gemma 4, Qwen3.6 and Qwen3.8 `enable_thinking=False`); the text is tokenised alone and appended, and only its tokens are scored (exp_036 Amendment 5). Every output records the wrapper; the 8-bit and 4-bit builds of a family must render identical prompt ids.
- KL is computed from bf16-rounded logits for every model (like for like, C6), upcast to fp32.
- Normalisation: Σ KL / Σ bytes. Per-token normalisation is a pre-registered sensitivity, published next to it, because per-byte KL folds Kolibri's denser German tokenizer into the ratio.
- Blocks: 8 byte-aligned blocks per text, split at whitespace; each token belongs to the block holding its first byte. **Stratified block bootstrap**: 8 blocks are resampled within each text, text weights fixed at their byte shares, joint across models.
- Kolibri's K8 log-probs are dumped to `$EXP036_WORK/kl/<UTC>/` (the unchanged exp_036 cell's default; the record's `k8_dump` gives the folder and each file's sha256) and then compared with K4. Each peer's 8-bit and 4-bit models are loaded together.
- A known boundary effect: for Kolibri, Qwen3.6 and Qwen3.8 the first text token of T1, T2 and T6 (which start with a newline) follows the template's trailing "\n\n" as a separate token; each record carries `wrapper_boundary` and `first_rows`, and a sensitivity excluding the first two text rows is descriptive.

**Rule.** CONFIRMED if Holm-adjusted p ≤ 0.05 (H₀: ratio ≥ 1.5) **and** Kolibri's per-text ratio is ≤ 1.5 on ≥ 5 of the 6 texts. REFUTED if not CONFIRMED and Holm-adjusted p_rev ≤ 0.05 (reverse H₀: ratio ≤ 1.5). INCONCLUSIVE otherwise. Per-text ratios are published. If H8 is CONFIRMED per byte but the per-token ratio exceeds 1.5, the post says so.

**Prediction.** ≈ 1.0. **Power** (within-block SD of the log ratio 0.3): 0.99 at a true 1.0 with a between-text SD of 0.15, 0.88 with 0.3; 0.85 / 0.52 at a true 1.2 (α/8).

**Descriptive.** KL with Kolibri's fp32 logits; top-1 agreement; ΔNLL per byte; K8 and K4 KL to the fp32 reference (from the gate).

**H8 is confirmatory and not blind.** The 1.5× threshold is unchanged from 2026-10-03. Seen before this pre-registration: K8's and K4's KL to R1 in exp_036's gate record (`e3e01b5c…`) and the peers' KL(8‖4) in exp_036's `peers_20261005T045649Z.json` (prefix `a5aa5e0728f7a9ad`); their values are not restated. exp_037's port (RoPE angle) and runtime (MLX 0.32.3) differ, so its values, the peers' included, are new measurements.

### Holm family
H1, H2, H3, H4, H5, H6, H7, H8 (m = 8, frozen at the plan amendment). D1 is a deterministic check with its own rule, outside the family. The verdict file reports, for each H: raw p and p_rev, both Holm-adjusted values, the Holm step, the Monte Carlo SE, the bound at the adjusted level, the 95 % interval, the truncation sensitivity, any E8 flag and the tripwire label or caveat.

---

## Exploratory and descriptive (predictions recorded 2026-10-03; no verdicts)

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
| Control C1 | Greedy answer-flip rate from batching: K8 at effort none, greedy, 100 MMLU-ProX full (non-Lite, gold-consistent) EN items at the memory-rule B for the MMLU cell at the initial cap, clipped to allowed_B(K8) (computed at preflight, recorded in the C1 file) vs B = 1. NOT RUN ("no batching to control") if the clipped B is 1. | ≤ 2 % of extracted answers flip. | S1 |

Exploratory results are labelled as such. Anything computed after the outputs were seen and not listed here is labelled post hoc.

### Power and energy (descriptive)

Added 2026-10-06 at Andrei's request ("also add powermetrics"). It registers no hypothesis and no prediction, and it decides nothing.

- **What is measured:** CPU + GPU + ANE power on the mbp: `powermetrics`' estimates of the three and their sum, `combined_power` (the record's key `combined`). The `powermetrics` man page lists CPU, GPU and ANE among the "SoC subsystems" whose power it estimates; their sum is not the whole SoC. It is sampled every second by exactly `/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist`, the mbp's pinned sudoers rule; sudo refuses any other argument. The logger runs in its own terminal through the peer check, the gate, the bench cells, the pilot, S2 and S3 (RUNBOOK steps 9–12, 14 and 16). It is started before each step and stopped after it, never in between. Every number carries the label "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power".
- **What is not measured:**
  - the rest of the SoC (the memory controllers and fabric, the system-level cache, the media and display engines, I/O) and DRAM, which bandwidth-bound decode loads heavily;
  - wall (AC) power and the adapter's losses, and battery charging;
  - the display and the SSD;
  - fans and the rest of the board;
  - the mini.

  The measured energy is therefore a lower bound of the mbp's energy, and a loose one: it may be about half of it. For scale only, from a different machine: on the mini (an M4 Pro), a `macmon` spot reading on 2026-10-06 under an MLX GPU load gave CPU + GPU power of 28.6 W against a system total of 64.6 W, about 44 % (exp_036 `compute/COMPUTE.md` gives the system figure). Nothing here converts it to the whole SoC or to wall energy. The man page also says that powermetrics' average power values "are estimated and may be inaccurate" and "should not be used for any comparison between devices"; nothing here compares machines.
- **Why:** exp_036's energy could only be estimated, as measured time × an assumed wall draw (exp_036 `compute/COMPUTE.md`). The logger adds a small background load (one root sampler at 1 Hz) to every logged step alike; it is not measured or corrected for. It runs through a whole step or not at all, so H1's interleaved blocks see it equally on both arms.
- **How it is reported:** `tools/power_log.py` joins the samples to the windows that the run records already carry and writes `results/power/power_<UTC>.json`. The windows are the gate and its phases, the peer check, each bench cell, the speed cells' cool-down idle (an idle baseline of the same CPU + GPU + ANE power), the pilot and its arms, and each S2/S3/S3b start/stop pair. Per window, the record gives:
  - energy in Wh (combined power × sampled time) and its CPU/GPU/ANE split;
  - mean and peak W, and the sample count;
  - coverage, every gap over 5 s, and the time that more than one sample counts (`overlap_s`: a few ms per sample from the whole-second timestamps).

  Nothing is extrapolated over a gap. Energy is reported per phase, in Wh, with its coverage beside it. The raw log stays on the mbp and is never committed, because it carries boot and host fields; the record keeps its sha256 and size.
- **No energy is counted twice:** the tool writes no record for a raw log whose documents' elapsed times sum to more than its span by over 2 % + 2 s. powermetrics' poweravg summaries (`-a`, by default every 10 samples, which the pinned command cannot turn off) are the known way this could happen; how they appear in `-f plist` is not yet known.
- **Before the freeze:** a 60 s log from the pinned command on the mbp, read with the tool's `--check` and `--layout` (key names and plist types only, no values), tests the parser against real output. BUILD_LOG.md records the layout it showed, or that the smoke did not run.
- **If the frozen tool cannot read a post-freeze log** (it refuses the log, or `--check` reports "key layout differs" or overlapping documents): the raw logs are kept on the mbp, `tools/` is not edited (a frozen scope; P1(a) refuses every gate run on a changed tools hash), and the steps go on. After S3, a "power log" amendment fixes `tools/power_log.py` and its test and gives their `hash_tree:` lines (TOOLS_SHA256, TESTS_SHA256); the records are written then. It touches no other file and decides nothing.
- **It decides nothing:** no hypothesis, verdict, gate rule, threshold, plan rule or stopping rule reads a power record, and none of them changed with it. A missing, late or partial log blocks nothing, is never repeated, and is reported with its coverage.

---

## Arms

| Arm | Model | Total / active params | Bits (affine g64) | Revision | Runtime | Template / thinking | Sampling (T / top-p / top-k) | Max tokens |
|---|---|---|---|---|---|---|---|---|
| K8 | Kolibri-1 (exp_036's conversion) | 78.1B / 3.46B (card: 78,103,074,560 / 3,457,573,120) | 8; router (fp32 weight), expert_bias and norms unquantised; embed and head at 8 bits; fp32 router and head logits | BF16 `7a8f290e` → `Kolibri-1-MLX-8bit-g64`, cloned into `$EXP037_BUILDS` with exp_037's port file (manifest sha in the gate record) | mlx 0.32.3 / mlx-lm 0.32.0, `port/kolibri1.py` via `model_file` with `trust_remote_code=True` after `check_port_file` | vendor jinja; `reasoning_effort="high"` (E5, E12: none / low / medium) | 1.0 / 0.97 / 128 | caps above |
| K4 | Kolibri-1 (exp_036's conversion) | as K8 | 4 (same policy) | → `Kolibri-1-MLX-4bit-g64`, cloned likewise | as K8 | as K8 | as K8 | as K8 |
| G8 | Gemma 4 26B-A4B-it | ≈ 25.2B / ≈ 3.8B (exact from config at preflight) | 8 (router.proj 8-bit) | mlx-community `33c6d237` | mlx-lm `gemma4`, text-only | upstream template; `enable_thinking=True`; thought channel `<\|channel>thought … <channel\|>` | 1.0 / 0.95 / 64 | as K8 |
| G4 | Gemma 4 26B-A4B-it | as G8 | 4 | `0d77464e` | as G8 | as G8 | greedy (speed cells), teacher-forced (H8) | H1, H8, E6 only (H1 and E6 only if `speed-only`) |
| Q36-8 | Qwen3.6 35B-A3B | ≈ 34.7B / ≈ 3B | 8 (mlp.gate, shared_expert_gate 8-bit) | `e06a74e6` | mlx-lm `qwen3_5_moe`, text-only | upstream template; `enable_thinking=True` (`<think>…</think>`) | 1.0 / 0.95 / 20 | as K8 |
| Q36-4 | Qwen3.6 35B-A3B | as Q36-8 | 4 | `38740b84` | as Q36-8 | as Q36-8 | greedy / teacher-forced | speed (descriptive), H8 only |
| Q38-8 | Qwen3.8 27B dense | 27B / 27B | 8 | `815b83c0` | mlx-lm `qwen3_5`, text-only | upstream template; `enable_thinking=True`; template default effort | 1.0 / 0.95 / 20 | B9 only |
| Q38-4 | Qwen3.8 27B dense | as Q38-8 | 4 | `10c35caa` | as Q38-8 | as Q38-8 | greedy / teacher-forced | speed (descriptive), H8 only |

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; every peer path comes from assets.json under `$EXP036_MODELS`, and K8 and K4 resolve under `$EXP037_BUILDS`. One item per sequence, through mlx-lm's BatchGenerator (the one construction) with one vLLM-order sampler per cell and per-sequence `max_tokens`. No Ollama, no GGUF and no server. Total and active parameters are recomputed from each config at preflight and written to the version record.

## Task sets

Unchanged from exp_036; the manifests built on the mbp on 2026-10-04 (`results/manifests_20261004T131126Z.json`, copied) are reused and verified at RUNBOOK step 7.

| Set | Source @ revision (assets.json) | n | Language | Scoring | Publishable? |
|---|---|---|---|---|---|
| GPQA Diamond EN | Idavidrein/gpqa @ `83022cef`, `gpqa_diamond.csv` | 198 | EN | V: eval-framework `GPQA_DIAMOND_COT_V2` prompt, option shuffle and `tulu_answer_v2(4)` on the post-reasoning text | ids, hashes, letters, scores |
| GPQA Diamond DE | ellamind/gpqa-multilingual @ `bc70ca17`, `deu`, `is_diamond` | 198 | DE | V: `GPQA_ELLAMIND_DIAMOND_COT_DE` + `tulu_answer_de` | ids, hashes, letters, scores |
| GPQA main (pilot) | the same two sources, rows whose Record ID is not in `gpqa_diamond.csv` | 8 + 8 per arm | EN, DE | pilot only | no |
| MMLU-ProX-Lite EN/DE | li-lab/MMLU-ProX-Lite @ `e82aafb9` | n_M per language, parallel ids, Webster allocation over Lite's counts | EN, DE | R: EN `MMLU_PRO_COT_V2` + `tulu_answer_v2(10)`; DE German Tülu CoT A–J + German lenient extractor (A–J) | ids, hashes, gold, raw outputs, scores |
| MMLU-ProX full (pilot, C1, peer check, weights) | li-lab/MMLU-ProX @ `8e6106a6`, non-Lite ids whose answer letter agrees with `answer_index` (3787 excluded in EN and DE) | 8 + 8 per arm; 100 for C1; 30 for the peer check; full test split category counts for post-stratification | EN, DE | pilot / control / weights | ids, hashes, counts |
| AIME 2026 EN | math-ai/aime26 @ `79037aeb` | 30 | EN | V: eval-framework `AIME2026` NeMo-Skills prompt and boxed extractor | ids, hashes, gold, raw outputs, scores |
| AIME 2026 DE | ellamind/aime26-multilingual @ `3c8bc18f`, `deu` | 30 | DE | R: frozen German NeMo-Skills wrapper (`tasks/prompts/aime_de.txt`), same extractor | ids, hashes, scores |
| IFBench | allenai/IFBench_test @ `2e8a48de` + allenai/IFBench @ `1c40f0c1` | 300 | EN | R: official checkers, loose-prompt (strict reported); scored on the mini | ids, hashes, raw outputs, scores |
| RGB closed-book | chen700564/RGB @ pin, `en.json` + `en_fact.json` | 400 | EN | R: `passage_num = 0` path; our checkanswer; RGB rejection string; frozen lexicon for abstain vs wrong | ids and hashes only |
| RGB closed-book, forced | as above | 400 | EN | frozen forced-answer instruction (`tasks/prompts/rgb_forced_en.txt`); our checkanswer | ids and hashes only |
| RGB Negative | `en.json`; noise 1.0; 5 negative passages chosen by RGB's `processdata` rule with `random.seed(2333)`, indices frozen in the manifest | 300 | EN | R: RGB system and instruction verbatim (rendered from local files); rejection string plus lexicon | ids and hashes only |
| RGB Fact-Check | `en_fact.json`; counterfactual passages; `config/instruction.yaml` with the `positive_wrong` documents | 100 | EN | R: corrected / deferred / detected rules | ids and hashes only |
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
- Confirmatory verdict code (Tier 1: H1–H8, D1, the H2 tripwire, stats) is frozen in this pre-registration. Exploratory analysis code (Tier 2: E1–E12 analyses, tables) is frozen by its own numbered amendment, pushed from the mini before `scorers/score_all.py` first runs and before any B4 ladder cell starts; both refuse to run otherwise.
- Quality numbers come only from a gated port: an exp_037 gate PASS record bound to the port's and the artefact's hashes.
- **No MLX-computed exp_036 value is reused as an exp_037 value** (MLX 0.32.2 changed the quantised MoE kernels on the M5): the peer records, the Gemma fidelity record, every port-side G2 row, G2q, the G4 and G5 values, the noise floors and every MLX-computed diagnostic are recomputed. They stay citable as exp_036's record. Reused are only numpy values under their own keys (R1, the reference's emulation dumps, G3's reference passes) and data (the K8/K4 shards and configs, the gate texts, the task and withheld manifests).
- Truncated, unparseable or unclosed output is wrong and is never re-asked. Its rate is reported.
- Vendor numbers and our numbers stay in separate columns. Vendor numbers are cited "p. N/189".
- Every absolute speed or memory figure is labelled "M5 Max, MLX 0.32.3".
- Every measurement made during the build on the mini is excluded as evidence about Kolibri; only committed records with sha256 are cited, and they concern the kit, not the model.
- INCONCLUSIVE and NOT RUN are published as such. `aborted/` is never deleted. Nothing from `$EXP036_PRIVATE` is ever moved or copied into the repo; a torn private line is kept in `$EXP036_PRIVATE/exp037/aborted/`, and the repo note records only its sha256 and byte count.

## Evidence layout

```
exp_037_kolibri_forced_gate/
├── results/
│   ├── version_record_<UTC>.json     exp037 version record v1: OS, Python, mlx, mlx-metal, mlx-lm, numpy, tokenizers, safetensors, pip freeze sha, git commit + dirty flag, every tree sha of the hash table, mx_device_info (P2's binding record), Metal limits
│   ├── preflight_<UTC>.json          exp037 preflight v1: chip, memory, macOS, mx.device_info, iogpu, power mode, disk, versions, git identity, asset revisions (+ sha256 in --deep); a fixed host label, never a hostname, serial or UUID
│   ├── manifests_<UTC>.json          sha256 of every manifest: the copied 20261004T131126Z record and step 7's verification build
│   ├── convert/refresh_{8,4}bit_<UTC>.json   the clone-and-refresh records (step 8)
│   ├── gate/gate_<UTC>.json (+ gate_<UTC>_layers.csv, gate_<UTC>_mutants.json, gate/<UTC>/ phase files)
│   ├── peers_<UTC>.json
│   ├── bench/{fit,speed,speed_desc,c1,ladder}_<UTC>.jsonl
│   ├── tokenizer_<UTC>.json, kl_8v4_<UTC>.json
│   ├── pilot/<UTC>/<arm>/*.jsonl, pilot_summary_<UTC>.json   (never scored for accuracy)
│   ├── pilot/<UTC>/<arm>/*.steps.jsonl                 the pilot's per-step decode logs (about 190 B per decode step: about 1 MB a cell at B = 8, at most about 50 MB at B = 1 with every item at the 32k cap), and an aborted pilot's copies under aborted/<UTC>-pilot/: committed (the aborted copies with the rest of that pilot), because the plan rule fits its step model on a summarised pilot's (each cell's steps_file; runner/plan_fix.py load_steps); sha256 in the plan run record; one over 50 MB stays uncommitted (the mbp's .git/info/exclude) and is copied to the mini at RUNBOOK step 18
│   ├── power/power_<UTC>.json                          CPU + GPU + ANE power per run window (powermetrics combined_power, an estimate; not the whole SoC, not wall power; descriptive; tools/power_log.py); the raw powermetrics log is never committed
│   ├── plan_fixed_<UTC>.json, AMENDMENT_<k>_<UTC>.md
│   ├── raw/<session>/<arm>/<task>_<effort>.jsonl       (withheld sets: text replaced by text_sha256; full text in $EXP036_PRIVATE/exp037/raw/)
│   ├── raw/<session>/<arm>/*.steps.jsonl               the sessions' per-step decode logs (about 190 MB for a K8 cell at B = 1, gigabytes over all sessions), and an aborted cell's copy aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl: git-ignored working files, never committed and never deleted; kept on the mbp and copied to the mini at RUNBOOK step 18; sha256 in the S2 and S3 run records; no score, verdict or registered rule reads them
│   ├── scores/<arm>/<task>_<effort>.jsonl              (item-hash keyed: extracted, category, correct, truncated; no free text for withheld sets)
│   ├── rescore_mini_<UTC>.json                         mini re-score vs mbp scores, per file
│   └── verdicts_<UTC>.json, .md                        (exp_023 shape, plus the H2 tripwire)
├── gate/calibration.json             F_tiny and its out-of-sample validation (frozen with GATE_RULES)
├── diagnostics/gemma_quant_check_<UTC>.json   the S1 Gemma fidelity record (RUNBOOK step 9a)
├── diagnostics/tiny_stress/          the two tiny stress records (pre-freeze; outside every hash scope)
├── diagnostics/vendor_spike/         the vendor-code spike's README, scripts and outputs (pre-freeze)
├── tasks/manifests/                  exp_036's manifests (ids, hashes, category and public gold), reused
├── tools/withheld_shingles.sha256    exp_036's hashed shingles of withheld text, reused (for the leak check)
├── evidence/withheld_manifest.jsonl  sha256 of every withheld raw file (GPQA EN/DE, RGB, AIME-DE)
├── aborted/<UTC>-<what>/ + NOTE.md   never deleted
└── version_record.json               the final run's record, copied from results/
```

---

## Sessions & budget

**Planning assumptions** (exp_036's, unchanged; the pilot simulation replaces all of them):

| Model | Aggregate decode at chosen B (nominal / pessimistic / adverse) | Mean completion tokens at high effort (nominal / pessimistic / adverse) |
|---|---|---|
| Kolibri | K8 180 / 120 / 100 tok/s; K4 240 / 160 / 140 | GPQA 5k / 8k / 10k; MMLU 2k / 3k / 3.5k; IFBench 1.5k / 2.5k / 3k; RGB CB 0.6k / 1k / 1.2k; forced 0.4k / 0.8k / 1k; AIME 14k / 20k / 24k |
| Gemma 4 | G8 220 / 150 / 140 | GPQA 4k / 6k / 7k; MMLU 1.5k / 2.5k / 2.8k; IFBench 1.2k / 2k / 2.2k; RGB 0.6k / 1k / 1k; AIME 12k / 15k / 18k |
| Qwen3.6 | Q36-8 250 / 160 / 150 | GPQA 7k / 9k / 10k; MMLU 3k / 3.5k / 4k; IFBench 2k / 3k / 3.2k; RGB 0.9k / 1.2k / 1.3k; AIME 15k / 18k / 20k |
| Qwen3.8 | Q38-8 100 / 60 / 55 | GPQA 7k / 9k / 10k |

Prefill: 1,500 / 800 / 700 tok/s for the MoE arms. RGB prompts with documents are ≈ 1,500 tokens. The adverse column is "modestly longer reasoning and some thermal throttling", close to what the vendor's RL budgets imply as typical (reasoning budget 65,536 and math 131,072 tokens, p. 172, 183/189).

**S1** (machine time, h, nominal / pessimistic):

| Block | Nominal | Pessimistic |
|---|---|---|
| Preflight `--deep` | 0.1 | 0.3 |
| Port refresh, clones (replaces convert) | 0.1 | 0.2 |
| Peer check, re-run | 0.4 | 0.6 |
| Gate (below) | 3.0 | 4.4 |
| D1, H1, speed, C1 | 0.8 | 1.1 |
| H5 | 0.1 | 0.1 |
| H8 KL | 0.3 | 0.5 |
| Pilot | 1.3 | 1.9 |
| Plan fix, amendment | 0.1 | 0.1 |
| **S1** | **6.2** | **9.2** |

Adverse: 9.7, exp_036's adverse uplift (+0.5) added to the pessimistic case.

**The gate's hours.** exp_036's gate took 0.93 h (phase 1 209 s; reference passes 1,549 s; G2 909 s; K8 329 s; the reference on 7,525 greedy tokens 198 s; K4 156 s). exp_037's blocks, nominal / pessimistic: P1–P4 and G0k 0.1 / 0.15; G0, G1 0.1 / 0.15; R1 reuse, emulation statistics and the G3 mutant pass 0.25 / 0.5; P5 0.05 / 0.1; R2F and R3 0.45 / 0.65; G2 0.25 / 0.35; K8 (bf16 checks, controls, fp32 checks, log-prob files) 0.6 / 0.9; K4 0.15 / 0.25; the R1 anchor (≈ 80k distinct tokens nominal, ≤ 111,115 pessimistic; at exp_036's 23–26 ms per token) 0.55 / 0.9; a fixed margin 0.45. **Gate: 3.0 / 4.4 h.** A re-run reuses R2F, R3, the emulation statistics and R1 when their keys match: about 2.4 / 3.3 h (3.2 / 4.4 h after a reference fix, which invalidates every dump).

**RUNBOOK step 9a** (the Gemma fidelity record) takes about 0.2 h of wall time before the peer check. Its record is in `diagnostics/` without `t_start` / `t_end`, so it is not an S1 block under rule 4; B_main is unchanged, and the rest of S1 starts about 0.2 h later in the day.

**Ladder totals** (Tier A, projected hours × 1.15; nominal / pessimistic / adverse): P0 25.1 / 56.4 / 74.3 · P1 23.4 / 52.8 / 69.9 · P2 21.3 / 48.3 / 64.0 · P3 18.9 / 43.0 / 57.2 · P4 17.4 / 40.0 / 53.3 · P5 16.0 / 37.0 / 49.4 · P6 15.0 / 34.7 / 46.5 · P7 13.6 / 31.7 / 42.6 · P8 12.7 / 29.4 / 39.6 · P9 11.7 / 27.1 / 36.6 · P10 9.9 / 22.9 / 30.3.

**Tier B** (× 1.15; nominal / pessimistic): B1 1.5 / 3.2 · B2 2.2 / 4.1 · B3 2.9 / 6.1 · B4 1.2 / 1.4 · B5 0.9 / 2.4 · B6 2.9 / 6.1 · B7 2.0 / 4.3 · B8 2.7 / 6.4 · B9 4.5 / 9.6 · B10 3.5 / 8.5.

**Plan picked by the registered rule** (B_main = min(31, 40 − S1); exp_036's ladder totals; a re-run costs 2.4 h nominal and 3.3 h pessimistic, also used for adverse):

| Scenario | Fix cycles | S1 | B_main | Plan | Main | Run total |
|---|---|---|---|---|---|---|
| Nominal | 0 | 6.2 | 31.0 | P0 + B1, B2, B5 (as exp_036) | 29.7 | 35.9 |
| Pessimistic | 0 | 9.2 | 30.8 | P8 | 29.4 | 38.6 |
| Adverse | 0 | 9.7 | 30.3 | P10, with no slack (S1 above 9.7 h gives STOP) | 30.3 | 40.0 |
| Nominal | 1 | 8.6 | 31.0 | P0 + Tier B as it fits | ≤ 31.0 | ≤ 39.6 |
| Nominal | 2 | 11.0 | 29.0 | P0 + Tier B as it fits | ≤ 29.0 | ≤ 40.0 |
| Pessimistic | 1 | 12.5 | 27.5 | P9 | 27.1 | 39.6 |
| Pessimistic | 2 | 15.8 | 24.2 | P10 | 22.9 | 38.7 |
| Adverse | 1 | 13.0 | 27.0 | **STOP** | — | — |
| Adverse | 2 | 16.3 | 23.7 | **STOP** | — | — |

- **Cycle cost.** A gate re-run moves the rest of S1 to a later day.
- **These projections assume K8 at its memory-rule B** (exp_036's planning assumptions, made for the registered batching that runs again under F2). If allowed_B(K8) = {1}, the rule re-simulates K8 cells at B = 1 from pilot step times measured under the exp_037 runner; K8's aggregate decode at B = 1 is unmeasured, and the plan is then likely to fall several rungs, possibly to STOP in the pessimistic or adverse case. Andrei accepted this with decision A.
- **The pilot measures throughput directly** under MLX 0.32.3 / mlx-lm 0.32.0; nothing is credited for the 0.32.2 kernel changes.
- At P10, power is 0.41 / 0.56 for H3 at the vendor's gap and 0.34 / 0.49 for H7 at a true cost of 0 (α/8 / α/3); Andrei signs off knowing this. The 44 h overrun ceiling is unchanged.

**Sessions.**
- S1 is daytime, ≈ 6–9 h of machine time; a gate fix cycle moves the rest of S1 to a later day.
- S2 and S3 are overnight-plus-morning, ≤ 16 h each, over one ordered queue: K8 Tier A first, then the peers' and K4's Tier-A cells in hypothesis order, then the forced and RGB Negative/Fact-Check cells, then Tier B. The split point is fixed in the plan amendment.
- The queue is resumable. A started cell finishes, if necessary in the next session. Tier-A cells left at the end of S3 run in the automatic continuation S3b (overrun ceiling 44 h).

---

## Plain answers (pre-registered wording)

The post answers each question with one sentence chosen by the verdicts, then the numbers. `<…>` are filled from `verdicts_<UTC>.json`; a NOT RUN state always reads "<H> was not run (<reason>); we make no claim about it."

| Q | Verdict state | Sentence |
|---|---|---|
| 1 | always | "Kolibri runs on an Apple M5 Max (128 GB) through our gated MLX port: <K8 tok/s> tok/s at 8-bit and <K4 tok/s> at 4-bit, batch 1 (M5 Max, MLX 0.32.3)." |
| 1 | H1 CONFIRMED / REFUTED / INCONCLUSIVE | "At 4-bit it decodes at <r> × Gemma 4's speed [CI]: at least / below / we cannot tell whether it reaches three quarters of it." |
| 1 | D1 CONFIRMED / INCONCLUSIVE / REFUTED | "On a 64 GB M4 Pro node the 4-bit build fits a 64k context with 10 % headroom / fits only without headroom / does not fit, if the node runs nothing else. We did not measure speed there; the 4-bit-vs-Gemma ratio is the number that transfers." |
| 2 | H2 tripwire TRIPPED, protocol control run (takes precedence over every other Q2 row) | "Through our port, Kolibri scores <x> pp [CI] below its own table on the public rows, more than 10 pp beyond what the peers show against theirs. A misreading of the architecture shared by our port and our reference would look exactly like this. No local test can rule it out, so we make no claim that Kolibri itself falls short." |
| 2 | H2 tripwire TRIPPED, protocol control NOT RUN (takes precedence over every other Q2 row) | "Through our port, Kolibri scores <x> pp [CI] below its own table on the public rows. A misreading of the architecture shared by our port and our reference would look exactly like this, and without the peer control we cannot separate it from our protocol either. No local test can rule it out, so we make no claim that Kolibri itself falls short." |
| 2 | H2 CONFIRMED, peers' mean D on the shared rows within ±2 pp | "On average across the five public rows we could check, at MLX 8-bit and k = 1, Kolibri scores no more than 4 pp below its own scorecard (D̄ = <x> [CI]); three of the five rows are our reconstructions." Any row whose 95 % upper bound is < −8 pp is named in the same sentence. |
| 2 | H2 REFUTED, peers within ±2 pp | "On the public rows we could check, at MLX 8-bit, Kolibri scores <x> pp below its own table while Gemma 4 and Qwen3.6 land within <y> pp of theirs. The port passed its gate; grid and protocol effects are bounded by the peers." |
| 2 | H2 any verdict, peers' mean D on the shared rows outside ±2 pp (takes precedence over the other untripped Q2 rows) | "Our local protocol scores every model <lower/higher> than the vendor's table: Kolibri by <x> pp, the peers by <y> pp; Kolibri's gap relative to the peers is <DiD> [CI]." |
| 2 | H2 INCONCLUSIVE | "We can't tell at this sample size: Kolibri is <x> pp [CI] from its own table, on average across the public rows." |
| 2 | H2 any verdict, peer control (E8) NOT RUN (takes precedence over the other untripped Q2 rows) | "Kolibri is <x> pp [CI] from its own table on the public rows (H2 <verdict>), but the peer control did not run (<reason>), so we cannot separate Kolibri from our local protocol; this is not a headline." |
| 3 | always | "In English and German on MMLU-ProX-Lite, Kolibri 8-bit scores <x> pp [CI] against the mean of Gemma 4 and Qwen3.6 (EN <a>, DE <b>) — <H3 verdict>. On IFBench it is <y> pp ahead of Qwen3.6 (<H4 verdict>) and <z> pp against Gemma 4. Its tokenizer packs German <r> × denser than both (<H5 verdict>). The German comparison rests on <MMLU-ProX DE / plus GPQA-D DE and AIME DE>." The E1 label applies when the peers ran no AIME/GPQA. |
| 4 | H6 CONFIRMED, forced cells ran, forced deficit upper bound < −10 pp | "Kolibri knows less without documents: it answers <x> pp fewer RGB questions correctly than the peers, and still <y> pp fewer when told to always answer. Of the questions it gets wrong, <A> are abstentions: it <says so / guesses / unclear>." |
| 4 | H6 CONFIRMED otherwise | "Without documents, Kolibri's closed-book correct-answer rate is <x> pp below the peers'; <A> of its non-correct answers are abstentions (<E2 reading>)." |
| 4 | H6 REFUTED / INCONCLUSIVE | "We did not reproduce the vendor's closed-book gap: <x> pp [CI]." / "We can't tell: <x> pp [CI]." |
| 5 | H7 and H8 | "Squeezing Kolibri to 4-bit costs <x> pp on the task rows (<H7 verdict>), and its logit fidelity loss is <r> × the peers' on our six gate texts (<H8 verdict>)." If H7 or H8 is NOT RUN: "We could not measure the task cost / fidelity loss of 4-bit (<reason>); D1 says only whether it fits." |

**The tripwire's labels and caveat.** If the tripwire is TRIPPED, the Q2–Q5 sentences that rest on a Kolibri quality verdict carry "Implementation-uncertain (G3 blind spot): <Hs>." and never headline. If it is not TRIPPED, every Kolibri quality verdict carries the caveat sentence: "A misreading shared by port and reference that moves Kolibri's measured shortfall on these rows by up to about 12–15 pp is more likely missed than caught locally; only one beyond about 15 pp is reliably caught." The words of both tripped rows and of the caveat are fixed in `analysis/verdicts.py` (`PLAIN_ANSWERS`, `TRIPWIRE_CAVEAT`).

---

## Hugging Face upload criteria (decision D; set before any result)

The K8 MLX build may be uploaded only if all of these hold:
1. **Gate.** K8 PASS in a gate record within the two fix cycles (≤ 3 gate runs).
2. **Fidelity.** G4-F32, G4-F16 and G4-N(i) pass in that gate run.
3. **Scorecard.** H2 is not REFUTED, and the H2 tripwire is not TRIPPED.
4. **Pilot and batching.** The pilot shows no unexplained truncation or parse-failure pattern. G5-BP-lean's allowed_B, and any detected batched-path discrepancy, are stated on the model card. The card also names the pinned runtime (MLX 0.32.3 / mlx-lm 0.32.0; decision F2) and states that M5-class GPUs need MLX ≥ 0.32.3 (mlx#3922).
5. **Clean venv.** In a fresh environment with the F2 pins, the published build loads through `model_file: kolibri1.py` (byte-identical) with `trust_remote_code=True` (mlx-lm ≥ 0.32.0; the card says so) and generates sensible EN and DE answers on the gate's behaviour prompts at efforts none and high. The check is recorded as a result file.
6. **Licence and labelling.** Apache-2.0 and NOTICE (Aleph Alpha's weights; aleph-alpha-inference). Marked unofficial. No vendor technical report.

Meeting them permits the upload but does not make it automatic: the upload stays an outward step needing Andrei's go. There is no K4 upload. If the criteria are not met, the results are still published in full on GitHub and localfirstai.eu; only the converted weights are withheld.

## NOT RUN wording

- The sentence is unchanged: "<H> was not run (<reason>); we make no claim about it."
- **Reasons used by exp_037:** "the exp_037 Phase 0 gate failed for K8 (<first failing check>)"; "the exp_037 Phase 0 gate ended INCOMPLETE for K8 (<reason>) after the last permitted run"; "K4 failed its gate checks (exit 4)" (H1, H7, H8, D1); "plan STOP: budget"; and the registered crash, overrun and peer-drop reasons.
- The tripwire is NOT_RUN whenever H2 is NOT RUN; no sentence is published for it.
- A gate failure is published under "Gate failure or STOP" (Publication angle), with exp_037's record.

---

## Publication angle

**Working title.** "We ran Aleph Alpha's Kolibri on a MacBook: does it fit, how fast is it, and does its scorecard hold on public rows?" The post also tells how the port got there: exp_036's gate failed, its bug hunt showed why free-routing bounds could not be met, and exp_037's gate forces the routing.

These would be the first independent, locally run, port-validated numbers for this model; re-check "first" on the publication day.

**One key-numbers table, ordered Q1→Q5:** vendor, ours, a V/R/untestable label, the truncation rate and the result excluding truncated items. The headline is the Q2 sentence chosen by "Plain answers". Every headline names the precision (MLX 8-bit or 4-bit), states k = 1, and states that 3 of the 5 H2 rows are reconstructed. A verdict carrying an E8 flag, a truncation-sensitive verdict or a tripwire label never appears in a headline.

**The gate section** prints, for each loosened exp_036 threshold, the old threshold beside exp_037's value taken from the gate record; it states allowed_B per arm and any detected batched-path discrepancy as such; and it carries the blind-spot paragraph verbatim.

**E1, told as it is in the vendor's own table.** The vendor's "best MoE" claim is on its Overall rows, which include internal and in-distribution rows (p. 98–101/189). On the 10 public deterministic rows of its own table Kolibri is last (75.6 vs 80.8 / 80.4); it leads only on the 7 rows its SFT mix was selected on, which the vendor itself calls optimistic (p. 70/189). The vendor disclosed the closed-book weakness itself (p. 99/189); we credit that and report whether it replicates locally. The comparator set is named as not the vendor's.

**The port and gate as a public artefact.** The per-layer error table, the forced end-to-end checks with their same-run controls, the mutation tests and the routing trap: afmoe and llama.cpp select experts on sigmoid + bias, while Kolibri selects on logits + bias.

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
3. The blind spot:

   > Port and reference were written separately from one specification, so a misreading they share passes every comparison between them. Only the vendor's own runtime could test that; Andrei chose on 2026-10-05 not to rent it (and had withdrawn the same anchor earlier that day, Amendment 8 D4, 08:37:33Z). exp_036 registered G3's NLL sanity, beside the vendor's routing test (routing only; it passed), as the local detectors of such a misreading (exp_036 HYPOTHESIS.md l.776), and G3 fired: our reference scores T1 at 1.31 bits per byte (the bound was 1.2) and 1.30 × Qwen3.6's bits per byte on our six texts, and its secondary signs (G3-D) remain unexplained. A uniform +9.2 % NLL error would explain the first excess — about 0.17 nats per token on T3, under a third of the subtlest registered reference mutant — and no local test excludes it. The only downstream check is a tripwire on the vendor's public scorecard. A misreading that moves Kolibri's measured shortfall there by up to about 12–15 pp is more likely missed than caught; only beyond about 15 pp is detection near-certain. Every quality result is conditional on its absence. A tiny-checkpoint comparison of the vendor's own model code, through vLLM 0.29.0, against our reference agreed to 1.1e-6 at every layer; it excludes a misreading in the wiring it exercises, not one in vLLM's kernels, in FP8 serving, or in behaviour tiny random weights cannot show.
4. The vendor targets datacentre FP8 serving; we test outside that envelope. We have no relationship with Aleph Alpha; Aprimerose builds local-first deployments and would use Kolibri if it held up.

**Style.** English only, sections separated by `---`, first names only, quotes of model output only from public sets (MMLU-ProX, IFBench, AIME EN, gate behaviour prompts), no withheld text anywhere.

**Order.** Each outward step is marked; the pushes from the mbp in RUNBOOK steps 4–17 are the pre-approved hand-off (go #1 (c)).
1. Verdicts with the tripwire, then the HYPOTHESIS results block and the Status line (mini).
2. The scientific_log result block (pointer and verdict words) and the result in both README rows, in one commit (mini).
3. Push **(Andrei's go)**.
4. The blog post, committed locally in the blog repo after `tools/leak_check.py <post.md>` (`post: … — exp_037`, body "Committed only; deploy.sh not run.").
5. `deploy.sh` and the SFTP upload **(Andrei)**; then check that the post's URL returns 200.
6. The README blog-index bullet, then push **(Andrei's go)**.

**Gate failure or STOP.** A dated block in this file, a scientific_log block, README status "Gate failed — <check>", "Gate incomplete — <reason>" or "Stopped — budget", pushed with Andrei's go; a post only on his go.

**Optional, each needing Andrei's go:** the Hugging Face upload (only if the criteria above hold), an mlx-lm pull request, a Malaga-AI post.

---

## Edge cases settled at build time

exp_036's nine, unchanged except where marked; the code implements exactly this, and Andrei's sign-off covers it. Changing any of them later is an amendment.

1. **Crash at B = 1.** A crash of any cause in a cell already running at B = 1 aborts that cell (moved to `aborted/` with a NOTE.md); there is no retry at B = 1. Its hypotheses are NOT RUN unless rule 2 applies.
2. **Aborted or unstarted B8 / B10 cell.** H2 then uses its single-pass estimate and the affected H7 row is dropped (named in the results); the hypotheses themselves still run.
3. **Plan P10.** B8 still runs its K4 GPQA-Diamond DE cell. No hypothesis uses it; it is reported as descriptive only.
4. **Q2 when the peer control did not run.** Answered by the added row in "Plain answers", which is never a headline.
5. **Plain-answer fill-ins.** The `<…>` slots, the precision named ("MLX 8-bit"), the row counts and the margin come from `verdicts_<UTC>.json` and the plan amendment; the wording around them does not change.
6. **Bench counts.** H1 needs exactly 10 blocks with both K4 and G4, and D1 exactly 3 reps. Any other count makes it NOT RUN. An incomplete bench cell may be re-run as a whole, with the incomplete run moved to `aborted/`.
7. **Peer-check drops and H1 / H8.** If G4 is dropped (verdict `fail`), H1 is NOT RUN; a `speed-only` G4 stays in H1. If either build of a peer family is dropped or `speed-only`, that family leaves H8's peer median; H8 is NOT RUN if no peer family remains (extended by exp_036 Amendment 6).
8. **Crash fallback and resume.** As written under "Crash fallback": two fallbacks move a cell to `aborted/`; a resumed cell carries `b_fallback_from` in its records.
9. **Peer-check failures outside H1, H3, H4, H6, H8.** Recorded and published with the peer-check record; they change no other verdict.
10. **The crash fallback steps down within allowed_B** for K8 and K4 (exp_037).

---

## Disclosures

### Chronology (UTC)

| Time | Event |
|---|---|
| 2026-10-05 05:01:12–05:57:03 | exp_036 gate run (record `e3e01b5c…`) |
| 08:37:33 | exp_036 Amendment 8 D4: anchor withdrawn ("No need to rent anything") |
| 13:06:16 | exp_036 stage 5 (the frozen rules applied) |
| 14:42:57 | D8: "Confirm, publish, then exp_037 (Recommended)" |
| 14:51:22 | Confirmation runs start (outputs in 2530ad4, 14:59:33) |
| 17:07:30 | exp_036 closing block (block stamp; commit 9451f47 at 17:15:10, pushed 18:01:11) |
| 18:02:42 | 5c3d178 (README line), committed |
| 18:08:20 | "yes, option 2" (G3 local; no anchor) |
| 18:12:19 | f740816 (M4 `mlx_repro` baseline; parent 5c3d178), committed |
| 18:17:01 | 5c3d178 and f740816 pushed together, on Andrei's explicit go (the go's own time is not recorded) |
| 18:17–18:21 | M5 runs; bcff589 (M4 boundary sweep) committed 18:19:47 and pushed 18:20:13; 6b38136 committed on the mbp at 18:20:56 (parent bcff589) |
| 18:22:46 | a2dbcc4: BUGHUNT §6.9 (pushed 18:29:09) |
| 18:39:38 | dc74aff: ml-explore/mlx#4632 (pushed 18:40:16) |
| after | draft design, three reviews, the decision brief |
| 2026-10-06 02:30:15 | Decisions A–E, "as recommended" |
| 02:47:03 | b6c591b (committed on the mbp): `mlx_repro` and `mlx_repro_min` on the M5 Max under MLX 0.32.3, clean |
| 02:47:53 | 222c845: BUGHUNT §6.9 resolved: fixed in MLX 0.32.3 (mlx#3922) before exp_036's run; #4632 closed as a duplicate (pushed 03:00:56) |
| 02:48:30 | Decision F: F2, "let's update mlx on both machines" |
| after | the exp_037 venvs on both machines; a scratch compatibility probe of mlx-lm 0.32.0 on the mini (not evidence); the release-notes review (Annex A); the build specification, its review and its F2 revision; the build (stages 0–3) |
| 04:06:38 | E-downloads: "yes, downloading is fine" |
| 04:07:20–04:18:22 | the vendor-code spike on the mini (README about 04:20) |
| 05:17:51 | E-verdict: "ok, use your recommendatino" |
| later that day | decisions (a)–(d) after the build's stage 3; the stage-3b fixes; mutant 27 measured; decision (e) |
| 12:33:45 | `gate/calibration.json` written by an F_tiny re-run on the gate code of that time |
| 12:43–12:50 | the two tiny stress records (`diagnostics/tiny_stress/`) |
| afternoon | the final pre-freeze review (W15) and its fixes: the document switches, the `tests` and `env` scopes, the interrupt and crash rules, G5-D32's factor 10 → 100, the provenance and risk corrections, the R3 estimate |
| 13:51:07 | "also add powermetrics": the descriptive power log ("Power and energy (descriptive)"; `tools/power_log.py`, built after this) |
| 14:43:27 | the mbp's pinned NOPASSWD sudoers rule for exactly the logger's command line (Andrei) |
| 15:10:32 | `gate/calibration.json` re-written by the final F_tiny run on the final gate code (F_tiny 4.1119e-7, unchanged) |
| 15:10–15:16 | the bf16 tiny stress record (`diagnostics/tiny_stress/unsharpened_seed29_bf16/`) |
| after 13:51 | the power log (`tools/power_log.py`), its review and fixes |
| 18:41:39 | W15-04, W15-02/03 and W15-R3: "as recommended" |
| ~19:02 | W15-11's read-only mbp check: the Gemma 4 bf16 folder holds 11 shards, 12 LFS files, 20 files; step 9a's count corrected from 12 shards |
| 19:08–23:05 | the inherited-values sweep (547 values traced) and its fixes: 15 defects, 1 blocker (`runner/plan_fix.py` read the gate's `_mutants.json` sibling), 3 major, 11 minor; no gate rule, threshold, control or gate text changed (BUILD_LOG) |
| 2026-10-07 04:11:40 | go #1 (a)–(e): "go" |
| at the freeze | the hash fill and stamp, the pre-registration push |

BUGHUNT's sha256 is `5bbe9979…` (at `222c845`). It was `1affc2fd…` at `dc74aff`; exp_036's closing block's `d3ea24fd…` predates §6.9. This pre-registration cites the current one.

### Values known when every rule was written

Every exp_036 gate value; exp_036's diagnostic stages 0–5; BUGHUNT §1–6.9 including its resolution; the confirmation runs; `mlx_repro` on both machines and on the M5 Max under MLX 0.32.3 (sha256 `da9923cb…` and `81cc4f8e…`); the peer records; the tiny bug-hunt series; the release-notes review (Annex A) and the scratch compatibility probe under 0.32.3 / 0.32.0 (not evidence). Every exp_037 constant was set after all of these, G0k's F2 shape set included.

Decisions (a)–(e) were taken after the build's tiny measurements: F_tiny and its validation, the whole tiny gate on the sharpened, unsharpened and pattern5 builds, and mutant 27's measurement. They changed which tiny build validates which check, which mutant is G5-BP-lean's control and how that control's tiny acceptance is judged. No gate threshold, rule or control margin in `gate/` changed, and no real-weight value of exp_037 exists.

The final pre-freeze review (W15, 2026-10-06) then changed one threshold and tightened the freeze mechanics, again with no exp_037 real-weight value in existence. **G5-D32's mean and max factors went from 10 to 100** (G4-F32's τ factor; the floors and both Csort ceilings unchanged). It was decided after the committed tiny stress records, in which the sharpened build's correct port failed the factor 10 at 11–16 × Csort on five of six legs (`diagnostics/tiny_stress/sharpened_seed29/`, record `e53158e9…`), and after re-reading BUGHUNT §6.2, where Csort-type statistics swing 26× between equivalent runs (8.1e-8 against 3.1e-9) and each mean is carried by one position (max 1.7e-5). Mutant 20, G5-D32's control, fires at 2.3e7 × its bound on the tiny build, so the change costs no shown power. The freeze mechanics: the `tests` and `env` scopes joined the hash table, and P1(d) gained the crash rules (see "Fix-cycle, freeze and amendment rules").

### Known-values ledger

| Rule | Outcome on exp_036's record |
|---|---|
| Precondition P2 (environment binding) | no exp_036 value. P2 refuses exp_036's 0.31.2 / 0.31.3; the mbp's exp_037 venv matches its pins |
| Precondition P3 (G0k) | near-predictable. Under 0.32.3 on the M5 Max, every count of both repro scripts passed (to 65,536 rows; the 52,752-row first wave at a 1.41 sorted / unsorted ratio against float64). 73,728, 98,304, 131,072 and 196,608 rows have no M5 value; on the mini (M4, no NAX path) every judged shape passed in the kit's tests |
| every MLX-computed value (G2 port rows and G2q, G4-N(i), G5 greedy, decode-vs-prefill, behaviour, peers) | recomputed under 0.32.3. The 0.32.2 quantised-kernel changes may move K8 and K4 bitwise, so every "predictable" below is a prediction from exp_036's 0.31.2 values, not a reuse |
| G0; G1 inherited part; G2 kept rows; behaviour | pass, predictable |
| G2 bf16 natural (a) and (p) | pass, near-predictable (1.375 % against 2.47 %; 6.078 ≤ min(8, 1.2 × 6.100 = 7.32)). Leg (p) is a maximum recomputed under the new kernels and RoPE with a 1.2× margin (see "Statistical and model risk") |
| G3a | fails its old bounds; now descriptive; values reproduce (T1 1.3106 bpb; numpy) |
| G3b | pass, predictable (deterministic NLL; reference unchanged; only the bootstrap seed moves; minimum p01 0.546 in exp_036) |
| G4-N(i) | pass on one draw (0.041 / 0.083); the new port's draw is unmeasured; T9 margin 1.2× |
| G5 greedy; decode-vs-prefill | pass, predictable (941 / 941; 0.038 against 0.151) |
| G5-D32 | T9 and T1 decode near-predictable (T9 forced decode 3.7e-9 and chunk 64 3.1e-9 against a control of 8.1e-8, through a different code path; T1 free decode mean 2.8e-11); T3 forced unknown |
| **G4-F32, G4-F16, F6 shadow, G5-R1, G5-BP-lean at the new configuration, P4, P5** | **unknown: never measured.** G4-F16 is predicted to pass with about 5–8× margin: exp_036's per-layer bf16 forced branch ratios, port to emulation, were median 1.05 and max 1.23, which predicts an end-to-end KL ratio of about 1.1–1.9 |
| R3 ceiling (G4-F16 VOID) | **unknown: no real-weight forced-emulation value exists.** A breach makes K8 INCOMPLETE with no remedy if it is the emulation's honest bf16 level. The final pre-freeze review's estimate (decision (d)), from exp_036's records only: mean KL(R2F‖R3) about 1e-3 to 3e-3 on T1–8 and 2e-3 to 4e-3 on T9, rough 2σ upper values 8e-3 and 1e-2; P(VOID) about 5–10 %, almost all of it on T9. Basis: exp_036's G2 bf16 forced branch errors (`gate_20261005T050112Z_layers.csv`: emulation medians 0.26–0.86 %, mean 0.51 %, flat across T9's three buckets; port to emulation median ratios 1.04 (attention) / 1.11 (MoE), p99 ratios up to 1.38); the free-routing totals of exp_036's stage-3 comparison (`diagnostics/gate1/out/stage3_g4_20261005T124342Z.json` `summaries`), which cap R3 from above: emulation against R1 0.0380 / 0.0772, the 8-bit fp32 port alone 0.0357 / 0.0797, the bf16 port 0.0411 / 0.0834, so adding bf16 moves the free total by +0.0023 / −0.0025 (emulation) and +0.0054 / +0.0037 (port), inside the review's estimate of run-to-run noise (about ±0.003–0.004); and the residual-stream rounding G2 does not measure (perhaps 1.5–3× in amplitude), which takes the upper range to 1e-2 on T9. The tiny builds bracket R3 (5.5e-5, linear; 0.13–0.16, chaotic) but do not predict it; the chaotic regime would put the free emulation total at ≥ 0.1 and is therefore ruled out for real weights. The ceiling is kept. G4-F16's κ ratio is predicted at 1.1–1.9 against 9, and control 6 (renorm_topk) keeps its power at R3 = 1e-2: the same defect in the reference shifted its NLL by 0.64 nats per token in exp_036's G3b |
| Peer check: the Gemma fidelity attribution | near-predictable. exp_036's record gave KL(bf16‖G8) 0.0167 and KL(bf16‖G4) 0.3778 under 0.31.2 (G8 passes fidelity, G4 speed-only). The S1 record under 0.32.3 is new |
| G5-BP-lean at exp_036's configuration | under MLX 0.31.2 it would fail (d): 0.299 against 3 × 0.069, so K8 would get B = 1. The cause is attributed to 0.31.2's sorted `gather_qmm` defect at the 52,752-row first-wave call, which 0.32.3 fixes. exp_037's value is unknown |

**Optics, stated plainly.** Wherever a lean rule can be computed on exp_036's record, it passes exp_036's port. Only the forced checks, shadow selection, G5-R1 and G5-BP-lean at the new configuration are blind. Without T10, every distributional outcome is predictable.

### Loosenings against exp_036, each with its counterfactual

**Every loosened rule failed in exp_036; none that passed was loosened.**

| Rule | exp_036 (registered) | exp_036 outcome | exp_037 | In-run calibrator fails the old rule? |
|---|---|---|---|---|
| G2 bf16 selection | every disagreement < 6σ_l | FAIL (6.08σ) | registered fraction rule + per-pair leg (p) min(8, 1.2 × M_emu); the 6σ count descriptive | yes: the reference's bf16 emulation reaches M_emu 6.10 |
| G4 decisive top-1 | ≥ 99 %, blocking | FAIL (0.98889; 66 / 5,943, where ≤ 59 passes) | descriptive | yes: the reference's bf16 emulation misses 68 / 5,943 = 0.98856 |
| G3 bpb | ≤ 1.2 per text; ≤ 1.25 × best peer | FAIL (1.311; 1.296) | descriptive | no calibrator; Andrei's option 2 |
| G5 batch parity | blocking at B = 8 | FAIL (0.307 > 0.151) | sets allowed_B (G5-BP-lean); blocking G5-R1 at B = 1 | the cause was MLX 0.31.2's kernel defect at > 32,768 rows, not chaos. exp_037 runs 0.32.3, which fixes it (decision F2), with the registered batching, and G0k checks the fix at the run's shapes |

- **A1's precondition failed.** It was pre-committed (FROZEN_RULES l.280–281) and conditional on no G2-D sign. G2-D1 and G2-D3 fired on the flagged pair and remain unexplained. Adopting A1's form is therefore a loosening relative to the registered rule, made on the principle in the last column.
- **"Vendor-faithful" wording** is replaced by "the reference's bf16 emulation": it shares any misreading.
- **Sharper in exchange:** G4-F32 and G4-F16 (forced, end to end); shadow selection; G5-D32; G5-R1; required controls on every new blocking check.
- **Not loosened:** G5 greedy and decode-vs-prefill.
- **Process tightened:** no threshold or verdict amendment at all; INCOMPLETE counts as a cycle; a code change after a crash is a typed gate fix, whose first line names the crashed run and any blind value it computed (P1(d) checks the run is named); interrupts are recorded as crashes, and a run killed without a record blocks every gate run until it is closed; G1's tests and the environment scripts are hashed.
- **What the lean scope gave up** (accepted with decision A): T10's fresh data; G4-N (iii)'s power against natural-routing defects that double decisive misses while staying under 0.10; bf16 tail-only and sparse-spike defects (descriptive only); F4 as a rule; row-level verification of the batched path; the resolution reports of seven extra mutants; the null simulation; E13 (so the tripwire has no calibration point); the G0k fallback.
- **Further disclosures:** G5-R1 exercises the greedy sampler, not the production sampler; R-greedy re-applies the greedy rule on nearly the same continuations; behaviour runs at B = 8 even when G5-BP-lean denies B = 8.

### Constant provenance

**Inherited** (registered 2026-10-03, before any real-weight data): the factors 3 and 2; 0.2 %, 0.10 and the 2-nat lead; 1e-4 and 1e-3; 5e-2; 99.5 % and 99 %; 2 / 20; 48 blocks at q = 0.01; G5's 1e-4, 3× and +0.2 pp. A1's 1.2 and 8 are not among them: they were written on 2026-10-05 with exp_036's port value in view (the "New" table below).

**New:**

| Constant | Where | Derivation | Known value in view |
|---|---|---|---|
| κ = 9 | G4-F16 | 3² | per-layer ratios 1.05 / 1.23 |
| A1's 1.2 and 8: leg (p), M_port ≤ min(8, 1.2 × M_emu) | G2 bf16 natural selection | A1's pre-committed form (exp_036 `diagnostics/gate1/FROZEN_RULES.md` l.281, committed 2026-10-05, after exp_036's gate run and before its diagnostics ran) | **port M 6.0777** (exp_036's gate record; FROZEN_RULES l.282 reads "original 6.0, observed 6.0777", and its withdrawn 5.1 switch point was "6.08 ÷ 1.2", l.488); M_emu not yet measured (6.100 afterwards) |
| 100 | τ; G5-D32's mean and max legs (since the final review) | about 10 in amplitude over the re-association scale: cross-implementation (τ) or cross-kernel (G5-D32) rounding in every layer against one re-association | one Csort window (exp_036's branches path, mean 8.1e-8, max 1.7e-5); G2 fp32 1.7e-6 worst median; for G5-D32 also exp_036's T9 values (3.1e-9 / 3.7e-9, a different path) and the sharpened tiny build's 11–16 × Csort |
| F_tiny = 10 × max block mean; cap 1e-5 | τ floor | tiny cross-implementation level; cap = 10 × Csort ceiling | none (measured in the build: 4.112e-7) |
| Csort mean ceiling 1e-6; max ceiling 1e-4 | G4-F32, G5-D32 | 2 × the head-tolerance KL; 100 × the mean ceiling | 8.1e-8 / 1.7e-5 on one T9 window |
| 1e-2 | F5; F6 gap; R3 ceiling | 100 × τ cap; 100 × G2 near-tie; bf16 composition + 1 decade | none (per-layer bf16 forced errors 0.5–1 %) |
| 10 | F3 | one decade | — |
| floors 1e-8, 1e-6 | G5-D32 | Csort mean ceiling / 100, and itself | — |
| G0k's judged row counts (to 196,608) | P3 | the run's shapes under `min(B, 8)` batching, the B = 16 bound, and `mlx_repro_min.py`'s boundary set | the M5 results under 0.31.2 and 0.32.3 |
| 3 (G0k); 5 % bad rows | P3 | G2's bf16 factor; the minimal script's statistic | the M4 and M5 repro files |
| 0.10 (P5c) | P5 | the registered gross-bug bound | none |
| 10× tiny margin per leg type | controls | one decade, the gross-control principle | none |
| decision (e)'s ratio > 3 on both subsets (G5BP/27's tiny acceptance only) | the pre-freeze tiny test | the gate's own factor; the fixture ceiling recorded beside it | mutant 27's 11.93 / 6.41, measured before the decision (disclosed below) |
| −10 pp, 0.95 | tripwire | 2.5 × H2's margin; above E8's band | no task score exists |
| 26 prompts; 2,100 and 3,000 tokens | G5-BP-lean | mid-run admission at B = 16; chunk crossing | — |

**No constant is justified only by "it passes exp_036's values".** One caveat: A1's 1.2 and 8 were written with exp_036's port value 6.0777 in view. They rest on the emulation calibrator (the per-pair ratio to M_emu), not on that value, but they are not blind to it.

### Port changes after exp_036's diagnostics

- **The vLLM-angle RoPE.** Decided on the specification (exp_036 BUILD_SPEC item 8; BUGHUNT §2) before any forced real-weight value existed. Its free-routing effect was mixed (BUGHUNT §6.3); that is not the ground.
- **The two attention hooks** change no output. There is no rows guard (decision F2). mlx-lm 0.32.0's `trust_remote_code` is a call-site change outside the port file; inside it, only the docstring changes.
- **The bf16 residual add is kept,** so no outcome-motivated port change exists. Both port sha256 values are named.

### G3

The anchor was declined twice: exp_036 Amendment 8 D4 at 08:37:33Z, and option 2 at 18:08:20Z. G3b stays blocking under its registered rule (decision C). The blind-spot paragraph and the tripwire are verbatim above. The tripwire's constants were set with no task score in existence.

### Hypotheses that are not blind

H5 (CONFIRMED at exp_036's build) and H8 (its seen inputs are listed under H8).

### The M5 defect and the runtime pin

The defect is MLX 0.31.2's sorted `gather_qmm` row overflow on the M5's NAX path. It was fixed upstream in MLX 0.32.3 (mlx#3922), released 2026-09-29, six days before exp_036's gate.
- **The runtime.** exp_037 runs MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0, which contains the fix (decision F2, 2026-10-06T02:48:30Z, "let's update mlx on both machines").
- **No row limit.** exp_037 uses exp_036's registered batching (`prefill_batch_size = min(B, 8)`).
- **G0k checks the fix** at gate start at the run's shapes, up to 196,608 rows, and on `mlx_repro_min.py`'s boundary set. A failure is exit 3, and Andrei decides.
- **What else the new runtime changes.** The quantised MoE kernels on the M5 changed (0.32.2), so no MLX-computed exp_036 value is reused. mlx-lm 0.32.0 requires `trust_remote_code=True` for our `model_file` loads.
- **The release-notes review** that preceded the pin is reproduced in full as Annex A.
- The versions are frozen until S3 ends.

### Long context

The gate covers positions 0–16,384, while scored completions reach 32–65k. The angle is unit-tested to 262,143 (maximum relative L2 1.056e-7 against the reference's rotation, bound 1e-6). The cache and NoPE paths beyond 16k are extrapolated (exp_036 BUILD_SPEC item 25(c)–(d)).

### Corrections carried from the reviews

- The draft's build total was 23.5–24.5 h, not 18–20.
- The draft's cut-offs 170 / 61 should have been 169 / 59; they belonged to legs exp_037 does not have (the draft's tail leg, McNemar).
- exp_036 BUILD_SPEC item 27 is at l.318–322; item 8's "In MLX:" line is l.245.
- The MLX angle offsets are −2 to +8 ulps (recorded), not −1 to +8 (printed).
- "Not tested" (G4 forced on real weights) is BUGHUNT §6.1, l.78.
- FROZEN_RULES l.304 reads "where one code path computes the same thing two ways".
- The vLLM function names come from BUGHUNT §2 l.31 and `reference/kolibri_ref.py` l.236.
- exp_036's G5 set is 12 prompts with 8 distinct lengths.
- The 32,769-row defect on the M5 is rel 0.031–0.076 (down_proj 0.0757), not 0.031–0.044.
- The +9.2 % NLL excess is about 0.17 nats per token on T3 (0.092 × 1.793), not 0.15.
- The lean gate estimate of 2.8 / 4.1 h is 3.0 / 4.4 h once P5, G5-R1 for both arms and the R1 anchor on every collected sequence are counted.
- Every scratch measurement is excluded as evidence. Only committed files with sha256 are cited.
- Under mlx-lm 0.32.0 a padded row's first generated token is affected by mutant 27 (see decision (e)); the earlier statement that it comes from prefill and is unaffected was wrong.

### Blindness of exp_037 values

No exp_037 value is seen before the freeze. No forced real-weight series, G5-R1 or G5-BP-lean value is computed before the hashes are set and pushed.

### The tiny-fixture stress finding (decision (a))

The whole tiny gate fails on the sharpened seed-29 build and passes on its unsharpened twin: K8 PASS, K4 PASS, exit 0, every required control caught. Every control but 27 meets its tiny margin there; 27 meets decision (e)'s acceptance instead, and probe 22's (d) fires. The sharpened failures come from the fixture, not from the code, by two routes: G4-N(i), the G4-F16 VOID and the two uncaught controls from 8-bit quantisation of the sharpened random weights; G5-D32, which is fp32 and port against port, from kernel rounding that the sharpening amplifies past one decade over Csort (the final review then set G5-D32's factor to 100; see "Values known when every rule was written"). Decisions (a)–(e) changed no threshold, and the sharpened builds keep their roles (F_tiny; the fp32 forced checks' own tests).

**Production precision (final review W15-05).** The tiny gate runs its free-routing checks on fp32 activations by default; the real gate runs them in bf16. A third run of the whole tiny gate, on the unsharpened build with `--tiny-precision bf16`, also passes (K8 PASS, K4 PASS, exit 0) with every required control caught: control 27 at 10.50 / 7.01 (decision (e) met; fixture ceiling 19.42 / 20.20) and control 26 at **4.63 × its bound, under the 10 × tiny margin**. Mutant 26 moves the tiny model as much in bf16 as in fp32 (mean KL 0.148 against 0.141), but R-parity's bound rises from the 1e-4 floor to 3 × the bf16 decode-vs-prefill chaos floor (0.032). The tiny margins are judged on the fp32 run; this run shows the power left to 26 at production precision on a near-uniform fixture. On real weights the same rule is calibrated by the real chaos floor (exp_036's bound was 0.151).

**Sources.** Both builds were re-run on the mini on 2026-10-06 (12:43–12:50 UTC; Python 3.12, MLX 0.32.3 / mlx-lm 0.32.0) with `gate/run_gate.py --tiny DIR --head-policy quantised_head`, on the gate code of that time (`gate_rules` `488dfa7d…`, port `2c153357…`, reference `85337ed7…`); the bf16 run followed at 15:10–15:16 UTC with `--tiny-precision bf16` added, on the final review's code (`gate_rules` `5e9e1a53…`). Their records are in `diagnostics/tiny_stress/` (README there):
- `sharpened_seed29/gate_20261006T124326Z.json`, sha256 `e53158e91e76821ee7460633342655ed15b92d2745eb179339cf3577a6e50548` (exit 1);
- `unsharpened_seed29/gate_20261006T124326Z.json`, sha256 `9547697b15ec127bffb0787eca5e54836e827197fc92378eda3b85d61546201c` (exit 0);
- `unsharpened_seed29_bf16/gate_20261006T151049Z.json`, sha256 `67ce52f050dd7c1f5beeddf3d3a2255d2a8d64c4cadb8ae4f3d6db4f9978d30b` (exit 0).

The two tiny columns are those records' values (the unsharpened column's bf16 values are the third record's), except the two cells marked "scratch", which are in no record and are not evidence. The real-weight column is exp_036's record; several cells are not the same computation, and each such cell says so.

| Check (bound) | Sharpened seed 29 (× 3) | Unsharpened seed 29 (× 1) | Real weights (exp_036) |
|---|---|---|---|
| G4-N(i) K8, mean KL(R1‖K8), T1–8 / T9 (≤ 0.10) | 0.378 / 0.393: FAIL | 0.023 / 0.020 | 0.041 / 0.083 (gate record `e3e01b5c…`) |
| 8-bit quantisation alone: KL(R1‖R2F), routing held identical; top-1 agreement | 0.214 / 0.242; 12 % (scratch) | not recorded | no forced value exists; free-routing decisive top-1 of K8 against R1 98.9 % (66 of 5,943 missed) |
| G4-F16's calibrator R3 = KL(R2F‖R3), T1–8 / T9 (ceiling 1e-2) | 0.127 / 0.158: VOID | 5.5e-5 / 5.1e-5 | no real-weight forced-emulation value (decision (d)) |
| G4-F16 control 6, κ ratio (margin 90) | 3.80: not caught | 2,992 | — |
| G5-D32, unmutated port: mean KL against max(1e-8, 10 × Csort mean), the factor in force when the records were made | over the bound by 1.14–1.63× (mean KL 1.8e-8 to 4.5e-8 against 1.6e-8 to 2.8e-8) on T1 and T3 decode and chunk 64, and on T9 chunk 64: FAIL. The max and top-1 legs were silent. Under the final review's factor 100 the same values sit at 0.11–0.16 of the bound: PASS | ≤ 1.63e-4 of the bound | T9 range: forced chunk 64 3.1e-9 and forced decode 3.7e-9, against a forced-self (sorted-order) control of 8.1e-8 (BUGHUNT §6.2; through `branches(force_ids)`, compared with the free run, so not this exact computation) |
| G5-R1 control 26, R-parity (margin 10 × the bound) | 14.7× (fp32 free-routing activations, the gate's tiny default); 0.52× (bf16; scratch) | 1,408× (fp32); 4.63× in bf16 (the third record): caught, under the margin | — |
| G5-BP-lean probe 22, (d) ratio, first wave / mid-run (a probe: no margin) | 1.76 / 1.70: (d) does not fire | 5.70 / 5.75: (d) fires | — |
| G5-BP-lean mutant 27, (d) ratio, first wave / mid-run (decision (e): > 3 on both) | 1.57 / 1.28: not caught, so allowed_B = {1} ("power not shown"); fixture ceiling 2.23 / 2.06, under the factor 3 | 11.93 / 6.41; fixture ceiling 22.27 / 19.12 (fp32); 10.50 / 7.01, ceiling 19.42 / 20.20 in bf16 | — |

- **pattern5** (exp_036's tiny build; the drivers test's former port run): G2 bf16 natural selection leg (p) fires (M_port 4.50 > min(8, 1.2 × M_emu 3.26 = 3.91)); G5-D32 is INCOMPLETE, because mutant 20 cannot act with a window of 17 below 256 keys; mutant 22's (d) ratio is 19.6. The port run therefore moved to the unsharpened seed-29 build.
- **Reading the G5-D32 row.** On the sharpened build, rounding alone takes the forced decode and chunk-64 paths past one decade over the same-run re-association control, by up to 1.63×. That check is fp32 and port against port (forced decode and chunk 64 against forced chunk 2,048, all on the same dequantised K8 weights), so 8-bit quantisation is not its cause: the sharpening amplifies kernel rounding, which a single re-association does not show. The final review therefore set the factor to G4-F32's 100 before the freeze. On exp_036's real T9 range, the analogous values sat 22–26× below that control itself. No forced real-weight value exists for T1 or T3. Whether a real range comes as close as the sharpened build is unknown.
- **Reading the G4-F16 row.** The sharpened R3 lies 13–16× above the ceiling. The 1e-2 ceiling has no real-weight value behind it. The final review estimates whether real K8 could approach it (decision (d)).

### The vendor-code spike (decision E)

- **What ran** (2026-10-06T04:07–04:20Z, on the mini, scratch only): Route A, vLLM 0.29.0 built from source for the mini's CPU, with the vendor's unmodified `Kolibri1ForCausalLM` from aleph-alpha-inference @049a6a7; torch 2.13.0 (CPU), fp32, one prefill per sequence; two runs bitwise identical. Compared, against `reference/kolibri_ref.py` in fp32: each layer's output and the logits, as relative L2 error. The pre-registered action rule: a mismatch is any layer or logit relative L2 error > 1e-4 (G1's fp32 logits bound), and means a reference fix before the freeze or stop.
- **The vendor preset** (seeds 0 and 1, window 65, 64 and 600 tokens) and the w513 preset (600 tokens): maximum relative L2 over every layer's output and the logits 1.1e-6, argmax agreement 100 %. Clean.
- **Real-layout checkpoints** (seeds 23 and 29, q/k norms × 3; their shards are byte-identical to the kit's `tests/tiny_real_layout.py` builds at those seeds, checked by sha256): clean at 64 tokens (4.9e-5 and 4.4e-5); at 600 tokens, end to end, **2.2e-2 (seed 23, argmax 99.17 %) and 1.7e-4 (seed 29), over the literal bound.**
- **Localisation.** Fed the vendor's own input, our reference layer reproduces the vendor's layer output to ≤ 1.94e-6 at every layer of every checkpoint; the expert choices are identical and the router logits agree to ≤ 1.7e-6; the embedding and expert bias are bitwise equal, the final norm agrees to 5e-8 and the head exactly. End to end, the error grows about 1.5× per layer; on seed 23 one expert choice flips at layer 7, token 244, at a near-tie (gap 5.6e-5). **Chaos control:** our reference against itself, perturbed by 1.5e-6 after layer 0, diverges the same way or more: 2.7e-1 (seed 23, 600 tokens), 4.8e-3 (seed 29, 600 tokens).
- **Discrimination.** exp_036's seven reference mutants on the vendor preset (600 tokens) give 0.15–0.98, and a window shifted by ±1 gives 0.20–0.22; on real-layout seed 29 at 64 tokens every variant is ≥ 0.71.
- **Classification: no wiring mismatch** (E-verdict), and the sentence appended to the blind spot. **Disclosed:** the literal end-to-end exceedance on the two real-layout 600-token runs; its cause, routing chaos, shown by the reference-against-itself control; and that the classification was decided after the result was seen. The spike changed nothing in the kit. Its README, scripts and outputs are in `diagnostics/vendor_spike/`; the downloaded vendor and vLLM sources and the venv are not committed.

### Decision (e)

Decision (e) was chosen after mutant 27 measured 11.93 (first wave) / 6.41 (mid-run) on the unsharpened seed-29 build, which meets it. A rule chosen after its value was seen is weaker evidence of the control's power on the tiny build. Two things limit the cost: the fixture ceiling (22.3 / 19.1, the ratio a full unrelated-context swap reaches) shows that the replaced 30× margin could not be met by any control of this kind on this fixture; and the gate itself re-tests mutant 27 on real weights, where an uncaught control can only take batching away (allowed_B = {1}). `gate/controls.json` and `gate/rules.py` are unchanged by (e); the acceptance lives in the tiny end-to-end test and in the tiny control table (`tests/fixtures/exp037_tiny_controls.json`), whose G5BP/27 row carries the acceptance and the ceiling beside `rules.py`'s margin (required 30, not met).

---

## Annex A. W0r: runtime release-notes review (exp_037), 2026-10-06

*Reproduced verbatim from the main session's review, except that its headings are one level lower (source sha256 `624767f55dbed165e26a99e046210456539c7f938c77879073db1e83f275ea16`). It is the record of the runtime pin (decision F) and is reproduced again in BUILD_LOG.md.*

*This review was done by the main session on the mini before the runtime pin, under the rule "check the latest release and its notes before pinning". Sources: the GitHub releases of ml-explore/mlx (v0.31.2 to v0.32.3), the compare view of ml-explore/mlx-lm v0.31.3...v0.32.0 (133 commits; mlx-lm 0.32.0 has no GitHub release page), and PyPI.*

### Decision F: the runtime pin

Andrei upgraded, at 2026-10-06T02:48:30Z ("let's update mlx on both machines"). This came after the M5 defect was shown fixed in MLX 0.32.3 (BUGHUNT §6.9, `222c845`).

**Pins:** MLX 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0. Every other package keeps exp_036's pin.

**Environments,** created beside exp_036's, which keep 0.31.2 / 0.31.3 so that exp_036 stays reproducible:
- mbp: `~/models/exp037/venv` (Python 3.12), checked as `mlx 0.32.3 | mlx-lm 0.32.0 | applegpu_g17s`;
- mini: `~/models/exp037-mini/venv312` and `venv314`.

### MLX 0.31.2 → 0.32.3: items that touch this kit
- **0.32.3, #3922: "Fix sorted gather_qmm NAX row overflow above 32K".** This is exp_036's G5 defect (BUGHUNT §6.9).
  - Under 0.32.3 it does not reproduce on the M5 Max: `confirm/out/mlx_repro*_mlx0323_20261006.json`.
  - exp_037 therefore keeps no mandatory 32,768-row limit. The G0k probe still runs at gate start, as insurance (below).
- **0.32.2, #4352 ("skip unnecessary simdgroup computations for quantised MoE matmuls on NAX") and #4171 ("32-row block in qmm_t_nax").** Quantised matmul kernels on the M5 changed.
  - K8 and K4 numerics may differ bitwise from exp_036's.
  - Consequence: no MLX-computed exp_036 value is reused as an exp_037 value. This covers the peer parity records, the G2q, G4 and G5 values, and the stage outputs.
  - The numpy reference dumps (R1, the emulation, G3) stay reusable, because they do not run on MLX.
- **0.32.3: #4392 (non-transposed affine qmm dispatch), #4009 (sorted gather_qmm on ragged K), #4458 / #4483 (global scales in qmm).** The port calls transposed qmm with K = 2,560 / 512, multiples of 64, and affine mode without global scales. No expected effect; G1 and G0k cover them.
- **NAX attention changes: 0.32.1 #3843, 0.32.2 #3842, 0.32.3 #4416 and #4455.** They concern head_dim 256 and D72/D80. Kolibri has head_dim 128, so no expected effect; G1, G2 and G4 cover them.
- **0.32.0, #3524 / #3425.** int32 shape-product overflow is now detected, which gives clearer errors.

### mlx-lm 0.31.3 → 0.32.0: items that touch this kit
1. **#1385, CVE-2026-5843: `model_file` execution now requires `trust_remote_code=True`.** exp_037 passes it explicitly in every kit load of our own `kolibri1.py`. The peers load built-in model types and need no flag. *(compat break 1)*
2. **`generate.py` and `cache.py` rewritten,** +480/−442 and +156/−178:
   - `BatchGenerator.stats` is now a window over monotonic counters (#1829);
   - `BatchKVCache` rebinds its offset instead of mutating it (#1848), and reads left padding without a reduction (#1824);
   - float32 promotion in `BatchKVCache` and `BatchRotatingKVCache` `extend()` is fixed (#1491);
   - cache `state` returns the full state (#1778);
   - `_make_cache` is removed (use `models.cache.make_prompt_cache`);
   - the `BatchGenerator` signature is unchanged in the arguments we pass (`prefill_batch_size`, `prefill_step_size`, `completion_batch_size`, `sampler`, `stop_tokens`, `max_tokens`), but the private `_prompt_tokens_counter` is gone.

   *(compat breaks 2 and 3)* The runner and the port's `make_cache` mapping must be re-validated against the new cache semantics. G5-R1, G5-BP-lean and the G1 batching tests are the checks.
3. **#1467:** fixes a broadcast crash in quantized SDPA with GQA and a batched padding mask. It applies to the quantized KV cache only, which the kit does not use.
4. **#1777:** prefilled prompt tokens now enter the logits-processor history. The kit's sampler is its own (`runner/sampler.py`) and it uses no logits processors, so there is no effect. To be confirmed in W8c.
5. **`switch_layers.py`:** `stop_gradient` on the indices only; inference is unchanged. The `do_sort` threshold is still `indices.size >= 64`.
6. **Samplers:** the xtc default changed (#1372) and top_p and min_p changed (#1825, #1912). The kit uses its own greedy and seeded samplers. To be confirmed in W8c.

### Measured compatibility on the mini (probe, not evidence)
exp_036's test suite was run under 0.32.3 / 0.32.0, with no code change:
- unmodified: 101 failed, 991 passed and 148 errors;
- with `trust_remote_code` defaulted by a probe hook: 28 failed and 1,212 passed. Two of those failures are hook artefacts, and 26 are in the runner and batching, from `_prompt_tokens_counter` and `_make_cache`.

These go to build task W8c.

### Consequences for the specification
- §6.4 M5-ROWS is replaced:
  - no mandatory row limit and no `prefill_batch_size = 1` rule;
  - the registered batching (`prefill_batch_size = min(B, 8)`) returns, subject to G5-BP-lean's allowed_B;
  - G0k at gate start probes sorted `gather_qmm` at the run's shapes, including > 32,768 rows (B × 2,048 × 6 up to 98,304) and the boundary set of `mlx_repro_min.py`, with the fix-5 rule.
- A G0k failure gives exit 3, and Andrei decides.
- P2 binds to MLX 0.32.3 / mlx-lm 0.32.0 and the `applegpu_g17*` architecture. It needs a new exp_037 version record from the mbp (RUNBOOK step 1).
- No MLX-computed exp_036 artefact is reused. The peer check re-runs in S1, as the default already said. The numpy reference dumps are reused.

---

## Sign-off

*Typed by Andrei on the mbp before the first gate run (RUNBOOK step 6); committed and pushed from the mbp. Claude never fills any field of this block. It re-confirms the items decided with A–E. A choice that differs from the pre-registered value is recorded in the same commit as **Amendment 0**: only the H2 margin can differ (`analysis/margins.json`, with the line `hash_tree --amend-line analysis` prints); the head policy cannot, because P4 binds exp_036's quantised-head builds.*

- Signed off by: Andrei (typed by Andrei on the mbp, 2026-10-07T05:28:24Z)
- Date and time (UTC, read from the clock): 2026-10-07T05:28:24Z
- H2 margin (−4 pp as pre-registered and re-confirmed with A–E, −3 pp or −5 pp): -4 pp
- Kolibri `embed_tokens` / `lm_head` policy, applied to K8 and K4 (re-confirm: quantised at the arm's bits with fp32 logits, as in exp_036's builds that exp_037 reuses; vendor-faithful would need a new conversion, which this pre-registration does not provide): quantised at the arm's bits with fp32 logits
- K8 Metal limit (sysctl value run by hand, "not needed" if preflight printed no line, or declined): not needed
- Overrun ceiling for S3b (44 h as pre-registered): 44h
- Cloud FP8 anchor (declined, as decided with G3 and A–E): declined
- Commit: 5fb17d0

The "Signed off by" line must read `Signed off by: Andrei (typed by Andrei on the mbp, <UTC>)`; the runner checks this form.

---

*Experiment design: Andrei + Claude Opus 5.5 · 2026-10-06*

Pre-registration commit: 5dfcfecf6070b3649f0d9f48cbd4ad7e0a5a3af4

## sign-off — run record (2026-10-07T05:30:56Z)

- Phase: sign-off
- UTC: 2026-10-07T05:30:56Z (from the clock)
- Commit: 5fb17d07cf9d11fffb1a293d32f82ed903262850 (uncommitted changes: yes)
- Result files: none

## manifests — run record (2026-10-07T05:31:30Z)

- Phase: manifests
- UTC: 2026-10-07T05:31:30Z (from the clock)
- Commit: b06995a38da6a4fd216879fce6c348050e7d3555 (uncommitted changes: yes)
- Result files (sha256):
  - `results/manifests_20261007T053114Z.json` `20b4252f7ef614cbbaf05c282c9864a79062846846d2df672662e519ef9fb1f1`

## refresh — run record (2026-10-07T05:37:02Z)

- Phase: refresh
- UTC: 2026-10-07T05:37:02Z (from the clock)
- Commit: 48b68e16295311dc6d99fc1f53f7a87d3f74666d (uncommitted changes: yes)
- Result files (sha256):
  - `results/convert/refresh_8bit_20261007T053702Z.json` `17943c5c039440706bc5c22291ecaf39067dde5b9b9dd44f26b41f6189f69ba0`
  - `results/convert/refresh_4bit_20261007T053702Z.json` `2a9871566855f04d8a6478b7d5b4891092898e41c0cf2e19e6c940101ba19820`

## fidelity record — run record (2026-10-07T05:53:55Z)

- Phase: fidelity record
- UTC: 2026-10-07T05:53:55Z (from the clock)
- Commit: 014373fb3e36898d64e07a2bab7a8ca5f6200165 (uncommitted changes: yes)
- Result files (sha256):
  - `diagnostics/gemma_quant_check_20261007T054737Z.json` `7e3d75ade841d21946cfd6f6e95c8bacbc6049c900cea183686ff4cc49fcfd16`

## peers — run record (2026-10-07T06:07:29Z)

- Phase: peers
- UTC: 2026-10-07T06:07:29Z (from the clock)
- Commit: 44b73940ce2163d3d262949f67568c5532e01cb4 (uncommitted changes: yes)
- Result files (sha256):
  - `results/peers_20261007T060615Z.json` `a071bc76e5a6d86d011a35c99f866ede3655c8939268d6327278a402cefe6fa7`

## gate — run record (2026-10-07T13:52:06Z)

- Phase: gate
- UTC: 2026-10-07T13:52:06Z (from the clock)
- Commit: 8dea1dc934e9c8cf1796e2e59a2b1f16213100ca (uncommitted changes: yes)
- Result files (sha256):
  - `results/gate/gate_20261007T110355Z.json` `cb21c2dfa461c0cd17b68f47e9577cca248062bf3a7c6becbbfbd90dc9bdd809`
- Gate verdict: K4 PASS, K8 FAIL

## choice after gate run 1 (20261007T110355Z): b · diagnostics, by Andrei at 2026-10-07T14:49:52Z — run record (2026-10-07T15:40:20Z)

- Phase: choice after gate run 1 (20261007T110355Z): b · diagnostics, by Andrei at 2026-10-07T14:49:52Z
- UTC: 2026-10-07T15:40:20Z (from the clock)
- Commit: 1efbadca1c4eaf70f0399950a570069149287e22 (uncommitted changes: no)
- Result files: none

## Amendment 1 — gate fix (2026-10-07T18:45:20Z)

*Written after a gate result: gate run 1 (`20261007T110355Z`, exit 1: K8 FAIL on check G1 only, K4 PASS). Typed by the main session on the mini on Andrei's go, pushed as `amendments/1_gatefix_20261007T184520Z.md`, and appended verbatim by the mbp session with `tools/status.py --sync-amendments` (one writer). It changes no hypothesis, threshold, control, mutant, gate text, gate rule, arm, task, n or verdict rule.*

**Andrei's go.** Two answers, both recorded here.
- **Before the classification: a conditional go.** "ok, done, go" (relayed at 2026-10-07T18:31:35Z) answered the main session's request to commit the mini outputs, run the classification and, *if* it named gate fix 6.6, write it. That time lies 9 s after commit `f5b28a6` (18:31:26Z), 6 s after the classification file's stamp (`20261007T183129Z`) and 4 s before commit `c4e3189` (18:31:39Z). So it was given before Andrei had read the result, and it is recorded as conditional only.
- **After the classification: the go for this fix.** The main session reported the classification (O7, fix 6.6) to Andrei, together with the code change, the before/after test, the full-suite results and the residual risk below. Andrei answered "Go: push the fix" at 2026-10-08T04:35:08Z. That is the go for writing and pushing this amendment and its code (PLAN.md section 5 and 9.4).
- **The re-run.** In the same answer, at 2026-10-08T04:35:08Z, Andrei gave the separate go for the re-run ("Go: re-run"): fix cycle 1 of 2, the whole gate, after `--sync-amendments` and RUNBOOK step 3.

**Why.**
1. **Gate run 1.**
   - Run `20261007T110355Z` on the mbp. Record `results/gate/gate_20261007T110355Z.json` (sha256 `cb21c2df…`), committed in `1efbadc`; the gate ran code `8dea1dc`.
   - G1 reported 1 failed, 1,900 passed and 8 skipped (all `build-host only:`).
   - The failing test was `tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records`, at line 222: `assert content(records(p)) == content(records(ref))`.
   - Its junit text (`diagnostics/g1_resume_20261007/gate1_junit_failure.txt`, commit `9bdbb5e`) says "Omitting 3 identical items". q003 and q004 are shown differing with equal `answer_text`; pytest's `-q` truncation cut off the field that differed.
2. **Andrei's choice.** (b) diagnostics, at 2026-10-07T14:49:52Z. It is recorded in this file's run-record block, commit `d60f3b6`.
3. **The diagnostics.**
   - The package `diagnostics/g1_resume_20261007/` (`PLAN.md`, `README.md`, `diag_resume.py`) was committed in `6ba420e`, before any run.
   - The outputs were committed in `a2cdf8d` (mbp) and `f5b28a6` (mini).
   - The classification, `out/classification_20261007T183129Z.json` (sha256 `0c6215d4…`), was committed in `c4e3189`. Outcome **O7**, by rule 7 only. Decision: gate fix, fix **6.6**. The mbp is the governing host; rule 8 (the mini fallback) did not apply.
   - No pytest arm reproduced the failure, and neither did (iii).
     - mbp: (i) 50 of 50 runs equal, (i′) 30 of 30, (i″) 30 of 30, (ii) 4 of 4. In (iii), all 50 A–C pairs and all 50 uninterrupted pairs were equal.
     - mini: (i) 50 of 50, (ii) 1 of 1, (iii) 50 and 50 equal. Its (iv) outcome was "stale_only".
     - These counts are disclosed with the result. Under PLAN.md section 5 they select no rule.
4. **The mechanism.** (iv) on the mbp showed it: `out/mbp_cachestate_20261007T171033Z.json` (sha256 `bc904886…`), outcome "hit".
   - `scorers.reasoning.check_tokenizer` caches every passed check in the module-global set `_CHECKED`, keyed by `(family, id(tokenizer))`. It never removes a key.
   - Earlier in G1's order, `tests/test_bench_batch_flip.py::test_default_prepare_and_extract_through_the_sibling_modules` checks a real Kolibri tokenizer (`mlx_lm.tokenizer_utils.TokenizerWrapper`) and then frees it. The check runs through `bench.batch_flip.default_extract` and `scorers.reasoning.split_reasoning`. The probe recorded:
     - one key added, the tokenizer's;
     - the tokenizer freed (checked through a weak reference).
   - `TokenizerWrapper` and the test's `StubTok` have the same object layout: `__basicsize__` 16, no `__slots__`. A later `StubTok()` can therefore be given the freed address.
     - In the probe, no `StubTok` hit the freed address in phase A's 20,000 one-at-a-time allocations.
     - In phase B, the second held `StubTok` did.
   - For that stub, `check_tokenizer` returned early, so `runner.generate._scorer_split` returned the scorers' split. Every record of that `run_cell` call then got `split_by = "scorers.reasoning.split_ids"` instead of `"runner.generate.split_ids"`.
   - The probe then ran the failing test's scenario with that stub as the resumed run's tokenizer. The result was a field-only difference:
     - q003, q004 and q005 differ in `split_by` only; `completion_ids` and `answer_text` are equal;
     - q000–q002 are equal.
   - Two comparison runs:
     - the control, with stubs outside the cache, was equal;
     - the seeded run, with a stub put into `_CHECKED`, gave the same difference as the hit.
   - This matches the gate's visible output: three differing items, with equal `answer_text`.
   - The probe shows what the kit does with this allocation history. It cannot show what G1's own process did (PLAN.md 3.6 and section 8).
   - **Scope class B** (PLAN.md section 4). The state is written by `scorers/` (reached through `bench/`) and read by the generation path. The fix therefore goes at the reading site.

**Change.** One function changes: `_scorer_split` in `runner/generate.py`. Its signature and return contract are unchanged. Nothing else in the file changes, and nothing in `scorers/` does.
- **Before.** It called `scorers.reasoning.check_tokenizer(family, tokenizer)` and returned the scorers' split unless that raised. A tokenizer whose id was in `_CHECKED` passed without its ids being looked at.
- **After.** On every call it compares the tokenizer's ids with the family table itself, using the scorers' public helpers:
  - `family_spec(family)` gives the turn-start, open, close and EOS tokens with their ids;
  - `token_id(tokenizer, token)` must equal each of those ids;
  - any mismatch, or any exception, returns None (the runner's split).

  It neither reads nor writes `_CHECKED`, and it keeps no state keyed by `id()`.
- **Why real tokenizers are unaffected.** This is the same comparison `check_tokenizer` makes, without the cache. For every tokenizer, the decision now equals what an uncached `check_tokenizer` decides.
  - A tokenizer whose ids match its family table gets the same split as before: `scorers.reasoning.split_ids` on `prompt_state_ids`. Every record field stays as it was. This includes the pinned tokenizers that `runner/run.py` and `gate/checks/g5_generation.py` pass to `run_cell`.
  - A tokenizer whose ids do not match got None before, unless its id happened to be a stale cache key. It now gets None in every case. That stale-id case is the only change in what `_scorer_split` returns.
  - **One side effect goes.** Through `check_tokenizer`, the old code added `(family, id(tokenizer))` to `_CHECKED` for every tokenizer that passed. `run_cell` no longer does. A later scorers check on the same tokenizer object therefore runs the comparison again instead of skipping it, with the same result. No record field changes, and the runner no longer leaves keys of its own in `_CHECKED`, stale ones included.
- **Cost.** At most five `token_id` calls per `run_cell` call for Kolibri, Qwen3.6 and Qwen3.8, and six for Gemma 4 (three EOS tokens). The comparison stops at the first mismatch. `run_cell` calls `_scorer_split` once, not per record.

**Test.** One new file, `tests/test_runner_scorer_split_cache.py`. Nothing else under `tests/` changes.
- **`test_resume_with_the_stub_in_the_check_cache_gives_identical_records`** (the 6.6 test).
  - **Setup.** It is the failing test's scenario, step by step. It uses `tests/test_runner_tiny.py`'s own `items`, `run`, `records`, `content` and `StubTok`, on that module's fixture `model`: the tiny vendor checkpoint, seed 0, fp32.
  - **Steps.** Reference run; a stopped run after three records; a torn fourth line; `completed_keys`; the resumed run.
  - **The seeded stub.** The resumed run's `StubTok` is created first and held for the whole test. It is made the only key of `_CHECKED`, as `("kolibri", id(stub))`, the key form `check_tokenizer` writes (it keys by `family_spec(family).name`). `run()` receives that stub where it calls `StubTok()`, through a `monkeypatch` context around the resumed call only.
  - **What it asserts.**
    - every assertion of the original test;
    - `content(records(p)) == content(records(ref))`, with an itemised field diff as the failure message;
    - every record carries `split_by = "runner.generate.split_ids"`;
    - `_scorer_split("kolibri", stub)` is None, both with the key seeded and with the cache empty.
- **`test_scorer_split_follows_the_family_table_whatever_the_cache_holds`**, for each of gemma4, kolibri, qwen3_6 and qwen3_8.
  - **Three tokenizer stubs per family:**
    - one carrying the table's ids;
    - `StubTok`;
    - one carrying the table's ids except its EOS ids, shifted by 1.
  - **Three cache states:** `_CHECKED` empty; holding the three stubs' keys; holding their keys for every family.
  - **What it asserts in every state.**
    - The first stub gets the scorers' split, checked against `scorers.reasoning.split_ids` on seven prompt/completion cases. The cases cover plain, closed and open prompts; closed, unclosed and none splits; a leading blank; and a trailing EOS.
    - The other two stubs get None.
  - **Agreement with the uncached check.** On an empty cache, the uncached `check_tokenizer` passes for the first stub and raises for the other two.
- **`test_scorer_split_unchanged_for_the_real_kolibri_tokenizer`.**
  - **Setup.** The real Kolibri tokenizer comes from conftest's `tokenizer_dir` fixture, under its skip policy: the first of `EXP036_TOK`, `EXP036_TOKENIZER_DIR` and `$EXP036_MODELS/Kolibri-1-BF16` that holds `tokenizer.json` and `tokenizer_config.json`, else skip (a failure under `EXP036_REQUIRE_ALL`). It is loaded by `mlx_lm.tokenizer_utils.load` as a `TokenizerWrapper`, the class that `mlx_lm.load` gives `run_cell`.
  - **What it asserts.** With the cache empty or holding its key, the tokenizer gets the scorers' split, checked as above with `run_cell`'s `is_blank`.
- Every test snapshots `_CHECKED` and restores it in a `finally` block, so none leaves a key behind.
- **Oracle.** Symbolic: the family table's turn-start, delimiter and EOS ids (`scorers.reasoning.FAMILIES`) against the tokenizer's, plus the scorers' own `split_ids`. It is independent of every gate value. The tiny checkpoint serves only as the scenario's generator.
- **Trigger.** Deterministic: a seeded cache state (PLAN.md 6.8). So there is one invocation before the fix and one after.
  - **Setting.** Both ran on the mini, in venv312 (Python 3.12.13, pytest 9.1.1), with `EXP036_REQUIRE_ALL=1`, `EXP036_TOK` set, `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider`.
  - **Before** (2026-10-07T18:40:41Z).
    - **How the old code was loaded.** The pre-fix `runner/generate.py` was backed up before the edit; its sha256 `e1cce3fb…` is that of the blob at `c4e3189`. A scratch pytest plugin, kept outside the repository, loaded it as `runner.generate` before any test module imported it. The kit's file was not edited back.
    - **Result.** 5 failed, 1 passed.
    - **The 6.6 test** failed at the content comparison; every earlier assertion of the scenario passed. For q003, q004 and q005, `split_by` was `'scorers.reasoning.split_ids'` in the resumed file and `'runner.generate.split_ids'` in the reference. No other field differed.
    - **The four family cases** failed at `_scorer_split(family, StubTok) is None` with the stub's key seeded.
    - **The real-tokenizer test** passed. Its behaviour is unchanged by the fix.
  - **After** (2026-10-07T18:47:07Z). The kit's fixed `runner/generate.py`, final version (sha256 `ef85ee00…`): 6 passed.
  - **Where the evidence is.** The two pytest logs and the scratch plugin stayed in the session's scratch directory and are not committed. The claim rests on the reproduction below, which anyone can re-run from the repository.
  - **To re-check.**
    1. From the kit directory, write the pre-fix file: `git show c4e3189:./runner/generate.py` (sha256 `e1cce3fb…`).
    2. Before any test module is imported, load that file as `runner.generate`: `importlib.util.spec_from_file_location("runner.generate", <file>)`, put the module in `sys.modules["runner.generate"]`, execute it, and set it as the `runner` package's `generate` attribute. A pytest plugin passed with `-p` does this early enough.
    3. Run `pytest -p no:cacheprovider tests/test_runner_scorer_split_cache.py` with the settings above: 5 failed, 1 passed, as listed.
    4. Run it again without the plugin, on this amendment's `runner/generate.py`: 6 passed.

    These steps were re-run as written on the mini, with a new plugin, in venv312 and venv314 (2026-10-07T19:50:56Z–19:51:00Z). Both venvs gave 5 failed and 1 passed before, with the same three `split_by` differences, and 6 passed after.

**Tests run on the mini after the fix.**
- **Setting.** Both venvs, `EXP036_REQUIRE_ALL=1`, all files in one process.
- **Files**, with `tests/test_bench_batch_flip.py` first, as in G1's order:
  - the new file, `tests/test_runner_tiny.py` and `tests/test_bench_batch_flip.py`;
  - every test file that imports `runner.generate` or `scorers.reasoning`: `test_bench_common.py`, `test_gate_checks.py`, `test_gate_g5_runner.py`, `test_port_ref_cache_batching.py`, `test_runner_allowed_b.py`, `test_runner_mlxlm032.py`, `test_runner_session.py` and `test_scorers_reasoning.py`.
- **Results.** 220 passed in venv312 (Python 3.12.13). 220 passed in venv314 (Python 3.14.7).
- **Full suite.** `pytest -q -p no:cacheprovider -rfEs tests/` on the mini, with the same settings, on the final `runner/generate.py` (sha256 `ef85ee00…`) and the final test file:
  - venv312 (Python 3.12.13): 1,968 passed, 0 failed, 0 skipped, 2026-10-07T18:53:38Z to 19:18:25Z;
  - venv314 (Python 3.14.7): 1,968 passed, 0 failed, 0 skipped, 19:18:25Z to 19:43:10Z.

  The 1,968 include the 6 new tests. RUNBOOK step 3 runs the full suite on the mbp before the re-run.

**Scopes changed, with their new values.** Two of the 20 scopes change. The other 18 keep their registered values. On the mini, `tools/hash_tree.py --check HYPOTHESIS.md` reports 18 matches; `runner` and `tests` show MISMATCH until this amendment is appended.

hash_tree: RUNNER_SHA256 = 772d08e760d9ba0d04f6939e31c7082644b27eefaa968773659aa122da1b3dee
hash_tree: TESTS_SHA256 = 5adae3c86ae59b4cb655ac91dfe5b620028ebd52b4b61e8fcf54adea09a71fc7

- **What changed.** In `runner`, only `runner/generate.py`. In `tests`, only the new test file.
- **What did not change, by hash.** `port`, `convert`, `reference`, `gate` (gate code), `gate_rules`, `thresholds`, `gate_text`, `scorers`, `bench`, `tools`, `env` and every other scope.
  - No mutant, control, threshold, gate text, `gate_rules` file or `scorers/` file changes.
  - HYPOTHESIS.md and RUNBOOK.md are not edited; the mbp appends this file.
- **P1(d).** The runner tree is one of the code hashes P1(d) compares (port, reference tree, runner tree, gate-code tree, builds manifest). P1(d) therefore admits a re-run.

**The re-run.** It is fix cycle 1 of 2. Gate runs so far: 1 of 3.
- **What it is.** The whole gate (PLAN.md 6.8; RUNBOOK "After exit 1, 4 or 5", item 4):
  1. step 1 (pull);
  2. `"$PY" tools/status.py --sync-amendments`;
  3. step 3;
  4. step 8, only if the port changed (it did not);
  5. step 10 in full.

  Matching dumps are reused.
- **Andrei's go.** Given at 2026-10-08T04:35:08Z ("Go: re-run"), separately from the go for this fix (above).
- **Stop and publish** stays available.
- **Residual risk, outside 6.6.** 6.6 changes nothing in `scorers/`, so `_CHECKED` still serves the scorers' own callers (`split_reasoning`, `split_reasoning_ids`, `split_token_counts`).
  - One G1 test relies on that check to refuse a tokenizer: `tests/test_scorers_reasoning.py::test_wrong_tokenizer_is_refused`. It expects `split_reasoning("gemma4", …)` to raise for a fresh Kolibri `tokenizers.Tokenizer`.
  - A scratch probe on the mini ran `tests/test_bench_batch_flip.py` and `tests/test_scorers_reasoning.py` in 10 processes (2026-10-07T19:47:04Z to 19:48:34Z). Each time, when this test ran, `_CHECKED` held four gemma4 keys, left by the two gemma4 template tests before it (each checks a `tokenizers.Tokenizer` and a transformers tokenizer). The fresh tokenizer never had one of those addresses.
  - If it did, the check would pass and the test would fail.
  - Such a failure would be state written and read outside the generation path (PLAN.md section 4, class C). No code is changed for it.

**The diagnostics package.** `diag_resume.py` pins `runner/generate.py` at sha256 `e1cce3fb…` as a precondition, so it now refuses to run (exit 3). Its outputs are complete and classified, and nothing in PLAN.md calls for running it again.
