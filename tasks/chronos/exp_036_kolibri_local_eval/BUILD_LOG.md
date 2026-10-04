# Experiment 036 — build log of the kit

*Integration of the eight build areas (port-ref, gate, runner, tasks, scorers, bench, analysis, tools), 2026-10-03, on the build host (the mini). Nothing here is a result and nothing is committed by the build. [`HYPOTHESIS.md`](./HYPOTHESIS.md) is the pre-registration and wins over this file; [`BUILD_SPEC.md`](./BUILD_SPEC.md) is the kit's contract; [`RUNBOOK.md`](./RUNBOOK.md) is how the mbp runs it.*

---

## What was built

| Area | Contents | Its own tests |
|---|---|---|
| port, reference | `port/kolibri1.py` (exp036-port-2: fp32 activations into the quantised head, fp32 router weight, embedding and head quantised by default, inspection hooks), `port/convert.py` (+ memory guard, results copy, port-file freshness), `port/convert_streaming.py`; `reference/kolibri_ref.py` (exp036-ref-2: bf16 emulation, dequantised mode, streamed passes, dumps), `reference/mlx_affine_np.py`, `reference/mutants.py` (7 G3 mutants), `reference/DERIVATION.md` | `test_port_ref_*`, the binding pre-existing port/reference tests |
| gate | `gate/run_gate.py` (6 phases, exit 0/4/1/3, `require_pass`), `gate/checks/g0…g5` + `ref_pass.py`, `gate/port_mutants.py` (15), `gate/build_gate_text.py`, `gate/texts/` (T1–T4, T7, T8, G5 prompts, MANIFEST), `gate/thresholds.json` (§7.1 verbatim), `gate/behaviour_prompts.json` | `test_gate_*` |
| runner | `runner/run.py` (pilot, session, cell, status), `generate.py` (BatchGenerator, item 27), `sampler.py` (vLLM order), `chat.py` + committed peer templates, `memory.py`, `jsonl.py`, `guard.py`, `seeds.py`, `simulate.py`, `plan_fix.py`, `plan_rules.json` | `test_runner_*` |
| tasks | loaders and renderers for GPQA EN/DE, MMLU-ProX(-Lite), AIME26 EN/DE, IFBench, RGB, FineWeb-2; `build_manifests.py`; `selection_rules.json`; vendored eval-framework v0.14.2 (gpqa.py patched by hash) with `shim.py` and `vendor.py`; `tasks/manifests/ifbench*.json` | `test_tasks_*` |
| scorers | `reasoning.py` (id-level split, 4 families), `mc.py`, `aime.py`, `rgb.py`, `lang_tag.py`, `abstain_lexicon.json`, IFBench adapter + driver, `score_all.py` (+ the mini re-score), golden fixtures in `scorers/golden/` | `test_scorers_*` |
| bench | H1 and descriptive speed, D1 fit, H8 KL 8-vs-4, H5 tokenizer ratio, C1 batch flip, E6 ladder (Tier 2), `run_bench.py` | `test_bench_*` |
| analysis | `stats.py`, `verdicts.py` (H1–H8, D1, Holm ×2), `power.py`, `margins.json`, `vendor_values.json`; Tier 2 `exploratory.py`, `tables.py` | `test_analysis_*` |
| tools, env | preflight, version record, leak check + pre-push hook, hash tree, peer check, status, kv bytes, params, redaction, shingles; `env/` pins, `versions.json`, `exp036.env`, `setup.sh` | `test_tools_*` |

## What the integration added or changed

1. **NOTICE.** The eight `NOTICE.addendum.*.md` files are merged into `NOTICE` (sections 1–9: aleph-alpha-inference derivatives incl. `reference/mutants.py` and `gate/port_mutants.py`; Kolibri template fixtures; eval-framework copies, the patched `gpqa.py`, `scorers/mc.py`, `scorers/aime.py`, `tasks/prompts/aime_de.txt`; vLLM-derived `runner/sampler.py`; the three peer templates; code and data used in place (IFBench, RGB incl. `en_refine.json` as a known limitation, FineWeb-2, MMLU-ProX, AIME, GPQA); recorded facts and gate texts) and deleted. **`tests/test_notice.py`** (new) checks that every file with an Apache SPDX header, every file under `tasks/vendored_evalfw/` and `runner/templates/`, every `tests/fixtures/*.jinja` and `tasks/prompts/aime_de.txt` is named in NOTICE, that SPDX files carry a "Modified by" line, that no addendum is left and that NOTICE names no missing file.
2. **Contract fixes found by the integration** (each pinned by `tests/test_integration_contracts.py`, new):
   - `tools/peer_check.py` called `g5_generation.noise_floor()` (absent) and `batch_parity(model, prompts, B=8)` (missing `max_tokens`). On the mbp every G8/Q36-8 batched-path check would have raised, which the peer check records as a problem, so both peers would have been **dropped**. It now calls `floor_and_decode`, `parity_bound` and `batch_parity` with the gate's layout: 12 prompts of 37–1,100 tokens with staggered `max_tokens` (8 prompts at B = 8 could never admit mid-run) and the peer's own EOS ids.
   - `analysis/verdicts.py` read `plan["post_strat_counts"]`, `runner/plan_fix.py` writes `plan["category_counts"]`: the plan's frozen counts now feed the post-stratification.
   - `analysis/verdicts.py` run as a script (the RUNBOOK way) crashed whenever an exploratory analysis was NOT RUN: `exploratory.py` caught `analysis.verdicts.NotRun`, a different class from `__main__.NotRun`. The script now dispatches to the package module (the same double-import trap as the pipeline/llm.py `think` flag).
   - `runner/simulate.py`: the step-model fit is constrained to a, b, c ≥ 0 (exact active-set search; unchanged whenever the plain least-squares fit is non-negative). The dry run's pilot fitted c < 0, every cell became "unprojectable" and the plan rule returned **STOP**; a noisy last third on the mbp could do the same.
   - Tests on the run host: `EXP036_REQUIRE_ALL=1` would fail 30 tests on the mbp, which lacks the mini's build-time files. New mode `EXP036_REQUIRE_ALL=run-host` (conftest): every skip fails except `build-host only:` skips (upstream peer files, the eval-framework checkout: 8 tests). Tests now find peer folders, peer configs, the parquet and the tokenizer in the run host's layout (`$EXP036_MODELS/<folder>`, `$EXP036_DATA`, `$EXP036_MODELS/Kolibri-1-BF16`) and accept BUILD_SPEC's `EXP036_TOK`; `test_tools_preflight` no longer depends on an unset `EXP036_DATA`.
   - The gate's **G1** ran the suite with `EXP036_REQUIRE_ALL=1` on the mbp and required 0 skips: it would have failed K8 on any real run. G1 now uses `run-host` and counts `build-host only:` skips separately.
   - `gate/behaviour_prompts.json` (version 2): every prompt asks for two complete sentences. `scorers/lang_tag.py` needs two function words; "Lisbon.", "51", "Drei Brote kosten 9,60 Euro." all tag "unknown", and ≥ 2/20 "unknown" blocks K8 and K4. Two-sentence answers tag reliably (checked on 8 plausible answers).
   - Pilot definitions: `runner/plan_rules.json` (cells, n), `tasks/selection_rules.json` (manifest per task), `runner/run.py` (MANIFEST_SET, PILOT_TASK) and `tasks/build_manifests.py` (set sizes) agree; every queued task has a manifest set that `scorers/score_all.py` finds by name, with the same withheld flag.
3. **`tools/dry_run.py`** (new): the whole pipeline end to end on tiny checkpoints in a temporary copy (below). RUNBOOK step 3b.
4. **RUNBOOK** reconciled: step 0b names the GPQA gate acceptance; step 0c holds the four "Added 2026-10-03" fetches of ASSETS.md (FineWeb-2 named positionally, RGB clone at the pin, `uv sync --frozen` of the IFBench venv, the four NLTK packages) and no longer lists `config/instruction_fact.yaml`; step 3 uses `bash env/setup.sh` (the uv-built venv has no pip) and `EXP036_REQUIRE_ALL=run-host`; step 3b the dry run; step 3c an optional manifest build on the real data into `$EXP036_WORK`; step 19 names the mini's commands (`score_all.py --ifbench`, `--rescore-compare`, `verdicts.py`). `test_integration_contracts.py` parses every `"$PY" <script>` command of the RUNBOOK (43) and checks that the script exists and every flag is in its `--help`, and that the `run_bench.py --cells` list parses.
5. **BUILD_SPEC** corrected where it named things differently from the code (never HYPOTHESIS semantics): the §1 tree (helper modules, `env/setup.sh`, `scorers/golden/`, `tools/dry_run.py`, no `tests_ifbench/`), environment variables (`EXP036_TOK` path on the mini, the `run-host` mode, test-only roots, `EXP036_IFBENCH_PY`, `EXP036_NLTK_DATA`), the resolved pins (pyarrow 25.0.1, pytest 9.1.1), G5 function names, the RGB Fact-Check prompt source, the non-negative step fit, convert's two config keys, G1's skip rule, the behaviour prompts, a map from the planned test file names to the built ones, and the pre-push checklist.
6. `results/README.md` (layout) and `.gitkeep` in `results/`, `amendments/`, `evidence/`, `aborted/` (BUILD_SPEC §1).

## Deviations from BUILD_SPEC, with reasons

The areas' own deviations are in their reports and module docstrings; the ones a reader of BUILD_SPEC needs:

| Where | Deviation | Reason |
|---|---|---|
| tests | `test_<area>_*.py` names, not BUILD_SPEC's | one owner per file during the parallel build; §6 now maps them |
| tests | no `tests_ifbench/`; `tests/test_scorers_ifbench.py` | the adapter starts the IFBench venv itself |
| scorers | golden fixtures in `scorers/golden/` | HYPOTHESIS puts golden fixtures in the hashed `scorers/` tree |
| tests, gate G1 | `EXP036_REQUIRE_ALL=run-host` on the mbp | the mbp cannot hold the upstream peer files or the eval-framework checkout |
| runner | non-negative step fit | see above; a physical fit is unchanged |
| gate | behaviour prompts ask for two sentences | see above |
| gate | G2 functions fused (`g2_layers.run`), `floor_and_decode`; forced routing by a +1e6 boost inside the port's `route()` | one weight load per layer; `forced_route` would hide mutants 6, 7, 14 |
| port | config keys `exp036_quantize_embeddings` / `_lm_head`; bf16 emulation follows vLLM's fused add-and-norm | sanitize must know the head policy at load; HYPOTHESIS defines the mode "wherever vLLM stores bf16" |
| tasks | manifests and private manifests carry `gold`, not `gold_letter`; RGB Fact-Check prompt = `config/instruction.yaml` + `positive_wrong` documents | RGB at the pin has no `instruction_fact.yaml` |
| tasks | `build_manifests` counts are enforced in the CLI; the dry run calls `build_all(enforce_counts=False)` | the GPQA over-long question cannot be synthesised (its text never enters the kit) |
| bench | H1 4,096-token prefill and batch rows from `pad_120k.txt`; E6's 120k rung is the whole file (117,610 tokens) | `pad_4k.txt` and `pad_120k.txt` are shorter than their names in Kolibri tokens |
| runner | RUNBOOK's `pip install` → `env/setup.sh` | uv venv, no pip |

## Test counts

| Run | Command | Result |
|---|---|---|
| Before the integration | the prescribed command (`EXP036_TOKENIZER_DIR=…`, no REQUIRE_ALL) | 1130 passed, 3 skipped (186 s) |
| After, prescribed command | same | **1147 passed, 3 skipped** (192 s; the 3 skips are the gate's FineWeb tests, which need `EXP036_DATA`) |
| After, BUILD_SPEC §6 mode on the mini | `EXP036_TOK=~/models/exp036-mini/kolibri/Kolibri-1-BF16 EXP036_DATA=~/models/exp036-mini/data EXP036_REQUIRE_ALL=1` | **1150 passed, 0 skipped** (193 s) |
| After, simulated mbp | `HOME` without the mini's files, `$EXP036_MODELS` holding the Kolibri tokenizer and the six peer folders' files, `$EXP036_DATA` with IFBench and FineWeb, `EXP036_REQUIRE_ALL=run-host` | **1142 passed, 8 skipped** (189 s; all 8 `build-host only:`) |

New tests: `test_notice.py` (6), `test_integration_contracts.py` (12). Only venv312 was tested; `~/models/exp036-mini/venv314`, which BUILD_SPEC §6 also names, does not exist on the mini.

## The dry run (`tools/dry_run.py`)

A temporary directory holds a git repository with a copy of the kit (identity Miktam, a bare remote, the kit's pre-push hook installed), `$EXP036_MODELS` with a tiny Kolibri BF16 checkpoint (the gate's pattern5 layout, the real 128,000 vocabulary and the real Kolibri tokenizer), its K8/K4 conversions by `port/convert.py`, and stand-ins for the six peers (Kolibri-architecture weights at each peer's vocabulary under that peer's real tokenizer, template and generation config, converted to 8 and 4 bits); `$EXP036_DATA` with synthetic GPQA/MMLU-ProX/AIME/RGB (reworded so the kit's own test literals never match the dry run's "withheld" shingles), revision metadata at the pinned revisions, and the public IFBench and FineWeb-2 files when present. Only `runner/plan_rules.json` is scaled in the copy (cell sizes 3–6, MMLU n_M 28/14, caps 24 → ceilings 32, B4 never queued); everything that runs is the kit's code. Three things are dry-run shims, labelled in their records: the placeholder sign-off line, the tiny gate's PASS re-labelled as a real-mode record for the guards, and a peers record marking the stand-ins ok (their config arithmetic cannot match the real peers by construction; `tools/peer_check.py`'s weight checks run on them first).

**Result on the mini (final run): ok, 17 of 17 stages, 3.2 min, exit 0.** Stages, all passing:

| Stage | What ran | Outcome |
|---|---|---|
| preregister | `hash_tree --fill`, `--stamp`, commit, push through the pre-push hook | 0 placeholders left; hook clean |
| preflight | `preflight.py --quick --not-run-host --out` | exit 2 (warnings; mini mode) |
| signoff | placeholder line, `status.py --record-block` | committed, pushed |
| manifests | `build_all` on synthetic data (21 sets, shingles, results record); `build_gate_text.py --work-only` on the real FineWeb shard | T5/T6/T9 match `MANIFEST.json` |
| convert | `port/convert.py` K8, K4 with the results copy | 2 records |
| peers | `peer_check.py --checks load,nll,kl,batch` | NLL/KL rule ran per family; batched path ok for G8 and Q36-8 (mid-run admission 4, max live 8) |
| gate | `run_gate.py --tiny` on `$EXP036_MODELS` | exit 0, tiny outcome K8 PASS, K4 PASS; `--require-pass K8/K4` accept the re-labelled record |
| tier2 | `amendments/1_tier2_<UTC>.md`, `status.py --sync-amendments` | Amendment 1 |
| bench | H1, descriptive speed, D1, H8 (T1–T6), H5 (60 docs), C1 (100 items, B 4 vs 1), E6 ladder | 7 complete files |
| pilot | `run.py pilot` | 28 cells, 252 items (all truncated: random weights) |
| plan | `plan_fix.py`, amendment appended, pushed | FIXED, P0, n_M 28, Tier B B1–B3, B5–B10 |
| sessions | `run.py session --name S2`, `S3`, `status` | 50 cells; withheld records hashed in the repo, text in `$EXP036_PRIVATE` |
| score | `score_all.py`; `version_record.py`; `score_all.py --ifbench` (real IFBench checkers in the IFBench venv); `--rescore-compare --ifbench` | 50 score files; re-score byte-identical |
| verdicts | `analysis/verdicts.py` | H1 INCONCLUSIVE, H2 REFUTED, H3/H4/H6 REFUTED (peer control failed), H5/H7/H8 CONFIRMED, D1 NOT RUN (fit run at 512/1,024 tokens) — numbers from random weights, meaningful only as a pipeline check |
| hash_check | `hash_tree.py --check HYPOTHESIS.md` | 17 of 17 scopes match |
| leak | `leak_check.py --all` (with the shingle file) and `--range @{u}..HEAD --staged` in the copy; `--all --no-gpqa-source` over this kit | 0 findings (warnings: the host role name in docs) |

It needs ≈ 15 GB of temporary disk (the tiny model's H8 dump has the real vocabulary) and deletes it at the end.

## Open risks

**For the main session and Andrei before the pre-registration push (HYPOTHESIS text; not changed here):**
1. **Kolibri's template does not prefill `<think>`** at effort low, medium or high; the generation prompt ends at `<|im_start|>assistant\n`. HYPOTHESIS pilot rule 3 ("the think block is prefilled") and BUILD_SPEC item 28 ("none is impossible") are wrong. The code records "none" at Kolibri effort high as a defect and scores the whole completion; the wording needs a decision.
2. **RGB facts:** HYPOTHESIS "New assets" lists `config/instruction_fact.yaml` and calls RGB's code unlicensed; at `65ec39e4` there is no such file and the README licenses code and data under CC BY-NC-SA 4.0. The Fact-Check prompt actually used (`instruction.yaml` + `positive_wrong` documents) belongs in the Task-sets row.
3. **Disclosure:** while probing, the bench builder tokenised H5 corpus rows 0–199 with all four tokenizers and saw the totals (ratios ≈ 1.19 and 1.18). Nothing was tuned on it; it should be disclosed before the push.
4. **Plan rule vs the scenario table:** the frozen rule picks P8 + B5 for the pessimistic assumptions where HYPOTHESIS's table says P8, no Tier B; B4 is priced 1.2 h with the margin in the table but a fixed 1.4 h in `plan_rules.json`.
5. **Definitions to settle before the freeze:** RGB Fact-Check `correct` (scorer: detected; E1/E9: corrected); AIME EN as a V row with an integer compare; m at the plan amendment when a hypothesis is NOT RUN from the start; dropped peers are inferred from the queue, not recorded by `plan_fix`; the G3 "resolvable" reading, the T9 reading, the 8,192 effort-none cap of the behaviour check and whether a crashed gate phase stops later ones; the 4,096-token caps of the peer flip rate and of C1 (not pre-registered); the abstention lexicon needs a review before it freezes.
6. **E11** has no producer: the gate record carries no E11 section, so E11 will report "not computed" unless the Tier-2 amendment adds one.

**Unverified until the mbp:**
7. Loaders for GPQA-DE, MMLU-ProX(-Lite) and AIME26 rest on inferred schemas; `primary_197` has been checked only against upstream's literal hash. Step 3c (new, optional) or step 7 is their first real run.
8. Real-weight paths: 156 GB shard hashing, the real tokenizer through `run_cell`, mlx_lm loading the real Gemma 4 and Qwen builds (text-only), convert's peak memory on 128 GB, the gate's run time and ≈ 61–100 GB of reference dumps, the real peers record feeding G3.
9. `runner/sampler.py` follows BUILD_SPEC item 26 but was not diffed against the tagged vLLM v0.29.0 file (not on the mini).
10. The behaviour check's "unknown" rule is mitigated by the two-sentence prompts, not removed: a model that ignores the instruction and answers tersely in ≥ 2 of 20 prompts still blocks.

**Housekeeping:**
11. `__pycache__/` directories exist under `analysis/`, `gate/`, `gate/checks/`, `port/`, `reference/`, `runner/`, `scorers/`, `tasks/`, `tasks/vendored_evalfw/`, `tests/`, `tools/` (git-ignored and excluded from every tree hash; a hook blocked their removal during the build).
12. The session scratchpad holds RGB-rendered dry-run manifests (`rgb_dry/`, from the tasks build) and integration scratch; it is outside the repository.
13. `runner/run.py status` writes `results/status_<UTC>.json` on any host; run it only on the mbp.
14. The pre-push hook prints the host role name in docs as warnings (not findings).

---

## Review fixes (2026-10-03, after the three-lens review)

*Findings of the runbook-walk, verdict-correctness and leaks-rights-determinism reviews, each verified here before it was fixed (re-running the reviewers' scenarios where they left scripts). HYPOTHESIS.md is untouched: no hypothesis, threshold or rule changed, and none of these fixes needed a factual correction of its prose. Items whose fix would change a pre-registered rule or wording are under "Needs Andrei" below.*

### Runbook walk

| # | Finding | Verdict | Change |
|---|---|---|---|
| 1 | Two tests pass only before the pre-registration fill; RUNBOOK step 3 and gate G1 fail after it | **Applied** (blocker confirmed) | `tools/hash_tree.py unfill()` (the template of a filled, stamped, amended text; an exact inverse of fill + stamp on the real file); the round trip rebuilds its template with it and never skips; the Status test reads the copy's current Status. Running the whole suite in `run-host` mode on a late-run copy (the dry run's kept repository: filled, signed off, amended, manifests and shingles built, records present; the real plan_rules restored) found a **third** state-dependent test, `test_h2_gpqa_primary_excludes_the_overlong_item` (it expected no GPQA manifest; after step 7 there is one): it now points at an empty manifests dir. That late-state run: 1178 passed, 8 skipped (all `build-host only:`), 1 failed before this fix; the affected files then passed |
| 2 | Step 2 copies the hook to a relative `.git/hooks` path | **Applied** (confirmed: `rev-parse --git-path hooks` prints `.git/hooks` on git 2.50) | Step 2 uses `--path-format=absolute`, checks `test -x` and prints "pre-push hook installed"; the same in `tools/hooks/pre-push`'s usage comment and in BUILD_SPEC |
| 3 | A missing Tier-2 amendment ends the session at B4 (rc 1, later cells unrun, S3 bookkeeping skipped); "the pull fetches it" was false; step 17 scored before committing | **Applied** | `runner/guard.py tier2_at_head()`: the amendment counts at HEAD in HYPOTHESIS.md **or** as the mini's committed `amendments/*_tier2_*.md` (HYPOTHESIS: "pushed from the mini"). `run.py session` skips B4 without it and runs the rest of the queue; at the end of S3 an unstarted B4 is NOT RUN with that reason. A pre-existing bug fixed on the way: NOT RUN records of the bench cell were keyed `bench:ladder:na` and never matched the queue's `B4:ladder` (they now carry the queue's id). New `tools/status.py --tier2`; RUNBOOK 14 (check before S2), 16, 17 (`--tier2` in the chain; a raw-outputs-only commit block; the mbp, never the mini, writes the mbp scores, so steps 18–19 wait for them) |
| 4 | Exit 4 (K4 FAIL) and a peer drop had no implementation: pilot, bench and plan_fix refused or STOPped | **Applied** | `guard.excluded_arms()` (K4 after a real gate record with K8 PASS and K4 FAIL; any arm whose peer-check verdict is `fail`); `run.py pilot --without ARM` (refused unless the records exclude the arm); `plan_fix` queues no cell of an excluded arm (B4 needs K4, K8 and G4) and writes `excluded_arms`, `peers` and `not_run` (K4: H1, H7, H8, D1; Qwen3.6: H4; no MoE peer: H3, H6) into the plan and the amendment; `verdicts.py` reads them. RUNBOOK 9–12 name the commands (`run_bench.py --cells speed_desc,tokenizer,c1`; `pilot --without K4`) |
| 5 | An interrupted pilot writes a normal summary; plan_fix STOPs on it | **Applied** | No summary unless the pilot ran to the end (`"complete": true`, plus `requested`, `skipped`, `without`); after Ctrl-C it exits 1 and prints its stamp; `run.py abort-pilot --stamp <UTC>` moves the partial pilot to `aborted/<UTC>-pilot/` (private files to `$EXP036_PRIVATE/aborted/`, sha256 only, a `private_moved` line in the withheld manifest); `plan_fix` refuses (exit 1, nothing written) an incomplete summary, a missing pilot cell, or cells of an excluded arm. RUNBOOK 12 |
| 6 | The re-pilot command and the summary order fail as written | **Applied** | `pilot --cells` accepts the `ARM:TASK:EFFORT` ids plan_fix prints and refuses an unknown cell; RUNBOOK 13 gives the re-pilot command, the two-summary `plan_fix` command (the full pilot first), and the STOP / BLOCKED amendment block with its own Status and commit message (`amendment <k> — plan <STATUS> by rule`) |
| 7 | Step 19 `--rescore-compare` without `--ifbench` cannot match | **Applied** | RUNBOOK uses `--rescore-compare --ifbench`; `score_all.py` implies `--ifbench` for `--rescore-compare` whenever IFBench score files exist |
| 8 | The mini's hand-back environment does not exist | **Applied in RUNBOOK; venv not created** | Step 19: the mini's git-ignored local env override (`EXP036_MODELS=$HOME/models/exp036-mini`, `PY=…/venv312/bin/python`; checked by sourcing a copy) and the `env/setup.sh` command for `venv312`; step 18 rsyncs `"$EXP036_PRIVATE/"`. Creating `~/models/exp036-mini/venv312` downloads from PyPI outside the kit: left to the main session with Andrei's OK (below) |
| 9 | The dry run fails `hash_check` on the mbp (kit already filled) and collides with the step-7 manifests | **Applied** | `tools/dry_run.py reset_run_state()`: the copy's HYPOTHESIS goes back to placeholders (`unfill`), the real step-7 manifests and the shingle file are removed; RUNBOOK 3b says it is valid at any step. While re-running it: the synthetic RGB instruction literal (`STUB …`) is now reworded in the dry run's data, since RGB instruction strings are withheld options (fix 28) |
| 10 | Step 18's file-count check cannot hold | **Applied** | `tools/status.py --verify-private` (on the mini): every listed private file at its latest sha256, following moves; `manifests/` and `aborted/` extras reported as expected; RUNBOOK 18 |
| 11 | "After two fallbacks" is wrong for cells at B ≤ 2 | **Applied (text); code needs Andrei** | RUNBOOK 14 says that B = 1 has no fallback and B = 2 one, that any stale heartbeat (power cut, update reboot, panic) counts, and how to avoid it. The code change (boot-time-aware resume, or retries at B = 1) would change the crash rule: below |
| 12 | Amendment 0 told to paste a bare hex | **Applied** | Step 6 uses `--amend-line analysis`, pasted verbatim |
| 13 | Steps 8 and 11 lacked resume guidance | **Applied** | Step 8: rerun only the missing line, never `--force` a finished folder; step 11: exit codes 0 / 1 / 3 and the rerun behaviour |
| 14 | A dirty checkout first shows up as a REFUSED at S2 | **Applied** | Step 1 runs `git -C "$LFA" status --porcelain` (must print nothing) and says why |
| 15 | Step 15 did not stage `aborted/` | **Applied** | Added to the `git add` |
| 16 | Blocks that source the env file and print with `cat` / `grep -n` are refused by the secret-path hook | **Applied (blocks rewritten); rename left to Andrei** | Step 1 uses `grep -c`; step 13 uses the new `tools/status.py --append-amendment` instead of appending with a shell redirect; a scan of every env-sourcing block finds no printing command left. The file was not renamed: the hook classifies it, and changing that is Andrei's call |

### Verdict correctness

| # | Finding | Verdict | Change |
|---|---|---|---|
| 17 | H2 NOT RUN when B10 is planned but pass 1 is missing | **Applied** | Two passes per GPQA row only where the pass-1 cell is complete; otherwise single pass with a `two_pass_note`; with two passes, the truncation rate and the excluding-truncated result use both passes |
| 18 | H7 NOT RUN when a B8 row is missing, and always under P10 | **Applied** | `plan_fix` adds GPQA-D DE to H7 only where K8 runs it (not under P10); `verdicts.h7` keeps a B8 row only where K4 and K8 both completed it and records dropped rows; a missing Tier-A row is still NOT RUN. Skipping the K4 GPQA-D DE cell of B8 under P10 would change B8's cells: below |
| 19 | Peer drop not implemented end to end | **Applied** | With fix 4: `plan_fix` writes `peers`; verdicts uses it (queue inference only for older plans); an integration test feeds a real plan_fix plan with Qwen3.6 dropped into the verdicts: H3 and H6 on Gemma 4, H4 NOT RUN |
| 20 | A truncation sensitivity that cannot be computed passed the headline guard | **Applied** | Treated as disagreement: `headline_eligible` false, headline "INCONCLUSIVE (truncation-sensitive)" (HYPOTHESIS: "uses a verdict only if the two agree") |
| 21 | All items of one weighted MMLU category truncated crashed the whole run | **Applied** | `D_excluding_truncated` is None with the reason; the verdicts are unaffected |
| 22 | Exact ties at 8 pp decided by floating-point error | **Applied** | `EPS = 1e-9` in the direction of each rule's wording (E8 "> 8 pp", the ±2 pp band, the 5 pp and −8 pp naming rules); tests at exactly ±8.0 pp and one item beyond |
| 23 | Protocol control with no peer gave NaN and "peers outside" | **Applied** | NOT RUN, "no MoE peer left" |
| 24 | Q2 sentence chosen without its peer-band precondition | **Applied (code); wording needs Andrei** | State `Q2_peer_control_not_run` (H2 CONFIRMED or REFUTED, band unknown): no pre-registered sentence, never a headline; tables.py prints a bracketed plain statement marked as not pre-registered |
| 25 | Q3 German basis and E1 label from the plan, not the cells | **Applied** | From `ctx.has_cell` for K and every peer (the same rule as E1's composite label) |
| 26 | Fixed words in the filled sentences (8-bit, five rows, three, 4 pp) | **Applied (code); HYPOTHESIS note needs Andrei** | `tables._adapt`: the precision from the primary arm, the row and reconstruction counts from H2_detail, the margin from margins.json; the defaults give the pre-registered text verbatim (tested) |
| 27 | H1 blocks and D1 reps not enforced | **Applied** | `margins.json` `H1_blocks` 10 and `D1_reps` 3; another count is NOT RUN (the dry run's 2-block H1 is now NOT RUN, as it should be) |

### Leaks, rights, determinism

| # | Finding | Verdict | Change |
|---|---|---|---|
| 28 | RGB documents, FineWeb text and RGB instruction strings were not leak-check sources | **Applied, by a different route** | Measured: the RGB documents and the FineWeb rows the kit reads hold ≈ 1.64 M + 1.99 M distinct 8-word shingles, ≈ 47 MB of hashes, too large to commit. Instead a **local-corpus pass** (`tools/shingles.py local_corpus_texts`, `leak_check.local_corpus_pass`) streams them from `$EXP036_DATA` on every run where the data is present (both hosts; ≈ 3 s) and matches them against the scanned text's own shingles. The kit's public text (the Apache licence, the gate texts T1–T4 and their sources, which quote the Grundgesetz that FineWeb also quotes) and all-digit shingles are never hits; public raw outputs are not matched, withheld-set raw files are. RGB's `instruction.yaml` strings are now in the on-the-fly sources (they were already in the shingle file), and the shingle file subtracts the kit's public text. Every planted case of the review (positive and negative passages, a FineWeb excerpt, the system prompt, an RGB-shaped record in `results/`) is a finding now; the current kit scans clean with all corpora loaded |
| 29 | Short RGB queries and 8-word options were invisible after step 7 | **Applied** | Option windows of 3–8 words; every RGB query, GPQA EN/DE question and AIME-DE problem of 3–8 words is hashed as an option (shingle file and on-the-fly). Re-measured on the 500 RGB queries quoted alone: 500 findings (before: 139 no hit, 94 warnings) |
| 30 | NUL / non-UTF-8 files skipped every rule; archives not refused | **Applied** | Inside the kit a NUL byte or invalid UTF-8 is a finding and archive and binary suffixes are refused; outside the kit both are warnings (other experiments in the repository keep their binaries); `.gitignore` lists the suffixes |
| 31 | `tools/leak_check.py` exempt from all rules; fixtures exempt from the type, size and token rules | **Applied** | No whole-file exemption (the checker passes its own rules; tested); `tests/fixtures/leak/` is exempt only from the personal-pattern rules |
| 32 | Tailscale IPv6 and JSON-escaped home paths missed | **Applied** | Rules for `fd7a:115c:a1e0::/48` and `\/Users\/<name>`. Seeding `EXP036_PRIVATE_NAMES` with both account names was not done: it would commit the very names it is meant to keep out |
| 33 | NOTICE attributed G5_prompts.json to Claude; T7 and T8 were not listed | **Applied** | NOTICE §8: T7, T8 and G5_prompts.json are listed under `gate/texts/` as excerpts of the T1–T4 sources with a Claude-written instruction, rendered with the vendor template; only `behaviour_prompts.json` and `template_cases.json` remain "written by Claude" |

BUILD_SPEC is updated wherever these fixes changed what a file does (§1 Tier 2; §5.4 guard, plan_fix, run.py; §5.5 shingle file; §5.8 verdicts; §5.9 leak check, hook, hash_tree, status, dry run). Open risk 5 above ("dropped peers are inferred from the queue") is closed by fixes 4 and 19.

### Tests and checks after the fixes

| Run | Result |
|---|---|
| Prescribed command (`EXP036_TOKENIZER_DIR=…`, venv312) | **1184 passed, 3 skipped** (194 s; the 3 skips are the gate's FineWeb tests, which need `EXP036_DATA`; before the fixes 1147 passed, 3 skipped) |
| BUILD_SPEC §6 mode on the mini (`EXP036_TOK`, `EXP036_DATA`, `EXP036_REQUIRE_ALL=1`) | **1187 passed, 0 skipped** (204 s) |
| Late-run copy in `run-host` mode, fake mbp home (fix 1) | 1178 passed, 8 skipped (`build-host only:`), 1 failed before the third test fix; afterwards the affected files pass |
| `tools/dry_run.py` on the mini | **ok, 17 of 17 stages, 3.3 min, exit 0** (H1 is now NOT RUN at the dry run's 2 blocks, by fix 27; hash_check 17 of 17 scopes; leak stage clean) |
| `tools/leak_check.py --all` with `EXP036_DATA` (RGB queries, instruction strings and every local corpus loaded) | **0 findings**, 11 warnings (10 host role names in docs, 1 `example.com` address in BUILD_SPEC), 253 files; `--all --no-gpqa-source` likewise 0 findings |

New or extended tests: `test_tools_dry_run.py` (new), and review-fix tests in `test_tools_hash_tree.py`, `test_tools_status.py`, `test_runner_guard.py`, `test_runner_session.py`, `test_runner_plan_fix.py`, `test_analysis_verdicts.py`, `test_scorers_score_all.py`, `test_tools_leak_check.py`.

### Needs Andrei (pre-registration semantics; nothing changed)

1. **Crash rule at B ≤ 2** (fix 11): at B = 1 one crash of any cause aborts the cell, and its hypotheses are NOT RUN. A boot-time-aware heartbeat (a reboot resumes at the same B) or retries at B = 1 up to `max_fallbacks` would change HYPOTHESIS rule 6, "Crash fallback".
2. **Aborted or unstarted Tier-B cells that feed Tier-A hypotheses** (fixes 17, 18): the code reads "If B10 runs" and "averaged over the rows K4 runs" literally, so an unstarted *or aborted* B10 or B8 cell leaves H2 single-pass and drops the H7 row. The crash rule's "its hypotheses are NOT RUN" could instead be read as making H2 or H7 NOT RUN when a B10 or B8 cell aborts. Confirm the first reading, or amend.
3. **B8 under P10** (fix 18): the K4 GPQA-D DE cell of B8 still runs under P10, although no hypothesis can use it (there is no K8 GPQA-D DE cell to pair it with). Dropping it would change B8's cell list.
4. **Q2 when the peer control is NOT RUN** (fix 24): no pre-registered sentence applies; the code prints a bracketed plain statement and makes Q2 ineligible for a headline. A sentence for this state should be pre-registered before the push.
5. **Filled slots of the plain answers** (fix 26): a dated note in HYPOTHESIS that the precision, the row counts and the margin are filled from the run (K4 primary, P10, Amendment 0), as the pre-registration's own conditional rules imply.
6. **H1 and D1 counts** (fix 27): any count other than 10 blocks or 3 reps is now NOT RUN; "a recorded deviation, still computed" is the other reading.
7. **Peer-check fails of G4, Q36-4, Q38-x** (fix 4): what they do to H1 (which needs G4) and H8 (every peer family at 8 and 4 bits) is not pre-registered; `plan_fix` records the exclusion and makes nothing else NOT RUN.
8. **The mini's `venv312`** (fix 8): create it before the pre-registration push (`EXP036_MODELS=~/models/exp036-mini EXP036_VENV=~/models/exp036-mini/venv312 bash env/setup.sh`, a PyPI download) and run BUILD_SPEC §6 in it. Only the session scratchpad venv was used for this review, as before.
9. **The env file's name** (fix 16): the RUNBOOK blocks no longer trip the secret-path hook; renaming the file and its local override to a name without `.env` would remove the class of problem, and would change how the hook treats it.

## Amendment 1 — mbp step 3 fixes (2026-10-04)

The mbp's first run of RUNBOOK step 3 failed (`aborted/20261004T044821Z-tests/NOTE.md`): 41 fp32-parity tests from TF32 on the M5 GPU, and one leak-check finding from 3-word GPQA options matching the kit's own text. Fixed as `amendments/1_gatefix_20261004T051845Z.md` describes: `tools/precision.py` (exact fp32 in every process), the guard as the first statement of six entry points, `tests/conftest.py` and `tools/version_record.py`; `tools/shingles.py` `MIN_OPTION_WORDS` 3 → 4; `tools/dry_run.py` sys.path at import. New tests: `tests/test_tools_precision.py` (10); two leak tests updated to the 4-word rule.

The fp32 headroom claim in `tests/INTEGRATION_LOG.md` ("≥ 50x") held only on the M4 Pro; on the M5 it holds with TF32 off. A first attempt also edited the prose in `tasks/selection_rules.json` and the `tasks/build_manifests.py` docstring; the dry run caught that every built manifest records the selection rules' sha256 (the frozen `ifbench.json` no longer matched), so both edits were undone and the amendment records the stale "≥ 3 words" wording instead.

Mini verification: 1,201 tests pass (Python 3.12 and 3.14, every test required); dry run 17/17; leak check 0 findings; `hash_tree --check` 17/17 with the amendment appended (checked on a copy; HYPOTHESIS.md itself is the mbp's to write).

## Amendment 2 — leak-check fallback mirrors step 7 (2026-10-04)

After Amendment 1 the mbp's `leak_check --all` found one 4-word option from `gpqa_extended.csv` (unused by the kit) at 9 kit lines (`aborted/20261004T053036Z-leakcheck/NOTE.md`). The pre-step-7 fallback read every GPQA EN file and subtracted no public text; step 7 reads Diamond and main and subtracts the licence and gate texts. `tools/shingles.py collect_from_sources()` now does the same through shared helpers; `write_shingle_file` output unchanged. New test `test_fallback_mirrors_step_7`. Mini: 1,202 tests pass (py3.12 and 3.14), dry run 17/17, leak check 0 findings, hash check 17/17 with the amendment.


## Amendment 3 — the GPQA Diamond EN primary set is all 198 items (2026-10-04)

RUNBOOK step 7 on the mbp stopped with `manifest gpqa_diamond_en: ValueError: expected 1 over-long item(s) excluded, found 0` (`aborted/20261004T082043Z-manifests/NOTE.md`; nothing was written). The mbp checked by hash only: eval-framework's `_OVERLONG_QUESTION_SHA256` (`04e898b3dfc3…`) matches 1 row of `gpqa_extended.csv` and none of the 198 Diamond or 448 main rows, with raw, stripped or CRLF text, and its Record ID is in neither file. All four GPQA CSVs are byte-identical at eval-framework's pinned revision `633f5ee8` and ours `83022cef`, so the vendor's `GPQA_DIAMOND_COT` scored 198 items too. Andrei's decision: the primary set is all 198 Diamond items, and the 197-vs-198 sensitivity check (HYPOTHESIS C14, H2) is dropped as meaningless. Margins, thresholds, rules, the German set, the pilot pools and the budgets are unchanged.

What changed (nothing committed; HYPOTHESIS.md and the vendored eval-framework files untouched):
- `tasks/gpqa.py`: `primary_197(expect_excluded=1)` becomes `primary_en`, which expects `DIAMOND_OVERLONG_EXCLUDED = 0`. The vendored filter stays in force. Any other count is an error, and so is a match that appears only after stripping.
- `tasks/selection_rules.json`: `diamond_en` now has `primary_n` 198, a new `overlong_excluded: 0` and a reworded `primary_rule`. `pilot_en` is unchanged.
- `tasks/build_manifests.py`: the 0-exclusion check now runs on every build, whether or not counts are enforced (dry run, `items_for` on the run host). `primary_n` is checked when counts are enforced. The GPQA EN manifest records `primary_n` and `overlong_excluded`, and the stdout JSON and `results/manifests_<UTC>.json` gain `"checks": {"gpqa_diamond_en_overlong_excluded": 0}`. This supersedes the reason given under "Deviations" above ("the GPQA over-long question cannot be synthesised"). The dry run still does not enforce counts, because its sets are small. The over-long assertion now runs there as it does on the real data.
- `tasks/manifests/ifbench.json` and `ifbench_pilot.json`: rebuilt with the kit's builder from the public IFBench data, because every manifest records the sha256 of `selection_rules.json` (the Amendment 1 trap). Only `selection_rules_sha256` changed (`8414586b…` to `87f9247c…`). The items are identical, checked against HEAD.
- `analysis/verdicts.py`: the primary GPQA EN set is every item. Any recorded exclusion, whether from the plan or from a manifest flag, is a hard error that names Amendment 3. The 198 sensitivity is removed, and the H2 detail key `gpqa198_sensitivity` is replaced by `gpqa_en_primary` (n 198, an empty excluded list, the rule).
- `analysis/vendor_values.json`: GPQA EN `n_v` goes from 197 to 198, because `n_v` is the vendor's item count (HYPOTHESIS l.422) and the vendor ran 198. This enters H2's vendor-noise term sqrt(p_v(1−p_v)/(N_v·n_v)). Kolibri's vendor sd goes from 0.009164 to 0.009141, and the largest change on any arm is 0.000025.
- `analysis/power.py`: the H2 power arithmetic stays at the registered GPQA EN n = 197 (`DEFAULTS["H2"]["rows_full"]`), so the registered tables still reproduce (SE at n_M = 154: 1.3854 pp, registered 1.39; at 198 it would be 1.3849 and round to 1.38). The largest power change from 197 to 198 is 0.00095 over every registered H2 design point (n_M ladder, 5 and 4 rows, D = 0, −1, −2, −3, α/8 and α/3). At the plan amendment's design points (D = 0, −2) it is 0.00078. `power_at_plan()`'s `assumptions` string now says "H2 GPQA EN n = 197 as registered, although Amendment 3 runs all 198 items", so the plan amendment's descriptive power record states the n it was computed at.
- `analysis/exploratory.py`, `tools/dry_run.py`: docstrings, plus the manifests stage now returns the `checks`. `RUNBOOK.md` step 3c expects `checks` 0 and all 198 items primary, and step 7 has a check line. `BUILD_SPEC.md`: the `tasks/gpqa.py` and `build_manifests` sections are updated.
- Tests: `tests/tasks_synthetic.py` adds a synthetic `gpqa_extended.csv` holding a stand-in over-long row, as on the real data. `test_tasks_gpqa.py` is rewritten for 0 exclusions: a planted Diamond match fails, and the strip-only refusal stays. `test_tasks_build_manifests.py` checks the recorded count and adds 2 tests (an extended-only question excludes nothing; a Diamond match stops the build and `items_for`). `test_analysis_world.py` sets `GPQA_EN_EXCLUDED = []`. `test_analysis_verdicts.py` covers n 198, no sensitivity, refused exclusions and the manifest flags. `test_analysis_exploratory.py` and `test_runner_plan_fix.py` are adjusted. `test_analysis_power.py` gets an `assumptions` check and the new `test_h2_power_gpqa_en_197_vs_198_under_0001`. The test named in the review fixes above, `test_h2_gpqa_primary_excludes_the_overlong_item`, is now `test_h2_gpqa_primary_is_every_diamond_item`. Open risk 7's `primary_197` is now `primary_en`, and step 7 has run it against the real data.

Hash scopes changed: `tasks`, `manifest_rule`, `analysis`, `tools`. The other 13 match (gate, runner and bench via Amendment 1).

Mini verification: 1,206 tests pass (Python 3.12 and 3.14, every test required; 1,202 before, plus 4); dry run 17/17 (3.2 min), with the manifests stage showing `"checks": {"gpqa_diamond_en_overlong_excluded": 0}`; leak check `--all --no-gpqa-source` 0 findings (11 host-name warnings); `hash_tree --check` 13 match and 4 MISMATCH (`tasks`, `manifest_rule`, `analysis`, `tools`) until the amendment is appended.


## Amendment 4 — MMLU-ProX-Lite composition and item 3787 (2026-10-04)

RUNBOOK step 7 on the mbp stopped at the MMLU-ProX sets (`aborted/20261004T095549Z-manifests-partial/NOTE.md`, commit 63e4593; 15 of 21 sets built). Two data facts at the pinned revisions: MMLU-ProX-Lite `e82aafb9` is not 42 per category (EN = DE: biology 36, business 40, chemistry 56, computer science 20, economics 42, engineering 48, health 35, history 19, law 48, math 68, other 46, philosophy 25, physics 65, psychology 40; 588), and in the full MMLU-ProX test split `8e6106a6` `question_id` 3787 has `answer` 'C' but `answer_index` 1, in EN and DE, the only such row of 11,759 per language (none in Lite). Andrei's decisions: "Proportional (Recommended)" and "Exclude, record id (Recommended)". Margins, thresholds, verdict rules, budgets, the n_M ladder, the post-stratification weights and the power tables are unchanged.

What changed (nothing committed; HYPOTHESIS.md and the vendored eval-framework files untouched):
- `tasks/mmlu_prox.py`: `webster_order(counts)` (the seat order of a house-monotone Webster / Sainte-Laguë allocation, built seat by seat with exact integer comparisons, ties to the earlier subject), `allocation(counts, n)`, and `select_ids` now takes the first n_M seats over the parallel-id counts, each seat the next id of its category's seed-36 order. On equal counts this is exactly the old rank interleave (a test pins it). `normalise_row` / `rows_by_id` / `load_items` take `keep_gold_mismatch` (full split only): such a row is flagged, counted in `category_counts`, left out of `full_pool_ids` (EN or DE) and never rendered; `gold_mismatch_ids` lists it per language; `load_full` is the full-split loader. Lite stays strict.
- `tasks/selection_rules.json`: `per_category_lite: 42` becomes `lite_category_counts` (the 14 counts above) with a rule note; new `nM_rule` (Webster), `gold_rule`, `gold_inconsistent_expected: {"en": ["3787"], "de": ["3787"]}`; `options_rule`, `full_pool_rule` and `category_counts` reworded.
- `tasks/build_manifests.py`: the Lite sets list all 588 in seat order and record `nM_allocation` (items per category at every ladder n_M); with counts enforced the parallel ids per category must equal `lite_category_counts`; every build checks that each category has an item at every ladder n_M. The full-pool sets require, on every build (enforced or not, `items_for` included), the gold-mismatch ids to equal `gold_inconsistent_expected`, record them as `gold_inconsistent_excluded`, and the `checks` gain `mmlu_prox_full_gold_inconsistent_excluded`; `mmlu_prox_category_counts.json` `_meta` records `gold_inconsistent_counted`.
- `tasks/manifests/ifbench.json`, `ifbench_pilot.json`: rebuilt with the kit's builder from the public IFBench data (the frozen-artefact trap); only `selection_rules_sha256` changed (`87f9247c…` → `7fc43f2d…`), items byte-identical to HEAD.
- `analysis/power.py`: `mmlu_deff(counts, sample=None)` generalises the design effect to any allocation (sum w_c²/s_c); the registered 1.114 (balanced) still drives every power figure; at the proportional allocation it is 1.003 at every ladder n_M, so the H2 tables are conservative. `power_at_plan` says so in `assumptions` and records `mmlu_deff`. `analysis/stats.py`: docstring only (unequal n per category).
- `runner/plan_fix.py`: the plan records `mmlu_allocation` (the first n_M categories of the Lite manifest) and the amendment prints it and the conservative-power note. `runner/run.py`: docstring only.
- `tools/dry_run.py`: the synthetic Lite has the real composition, the synthetic full split carries a gold-mismatch row 3787; the scaled n_M is 28 / 16 (16 is the smallest n at which every category gets an item; at 14 two have none).
- Tests: `tests/tasks_synthetic.py` (`MMLU_LITE_COUNTS`, the planted row, `write_mmlu(lite_counts=…, gold_mismatch=…)`); `test_tasks_mmlu.py` (allocation table, house-monotone and Webster at every n 1–588, within one item of quota, all 588, EN = DE, ties, gold mismatch excluded and listed, Lite strict); `test_tasks_build_manifests.py` (listing, `nM_allocation`, checks, 3 new tests); `test_analysis_world.py` (the world's MMLU cells have the Webster n per category); `test_analysis_verdicts.py` and `test_analysis_stats.py` (post-stratified mean with unequal n against a hand computation); `test_analysis_power.py`; `test_runner_plan_fix.py`; `test_integration_contracts.py`.
- Docs: `RUNBOOK.md` steps 3c and 7 expect the new `checks` line, the 588-item Lite manifests with `nM_allocation` and the listed 3787; `BUILD_SPEC.md` describes the new rules. Review fix: RUNBOOK 3c now builds into a new dated folder, `$EXP036_WORK/manifest_check_<UTC>/`, on every run. The mbp built 15 sets into `$EXP036_WORK/manifest_check/` under the Amendment 3 rules, and every manifest records `selection_rules_sha256`, so a re-run into that folder would stop at its first set with `FileExistsError` (reproduced on the mini with `ifbench`). Step 7 is unaffected: its partial outputs went to `aborted/`, and the committed IFBench manifests carry the new sha, so they come back `unchanged`.

The estimand of H3 and H7 on MMLU changes, although the code does not. Their point estimates are plain item means over the n_M set (`h3`: the mean over all paired differences; `h7`: the mean per row), and the stratified bootstrap uses its default weights n_s/N, the matching resample. With 42 items in every category, that was an equal-weight mean over categories. Under the proportional allocation, each category is weighted by its Webster allocation instead, which is close to its Lite share (at n_M 154, computer science and history have 5 items each and math 18). H2's MMLU rows are post-stratified to the full split's category shares, as before. The verdict rules, margins and thresholds are unchanged.

Ties in the allocation go to the earlier category in `categories_source` order: the vendored `MMLU_PRO_SUBJECTS` order, which is engineering, physics, psychology, chemistry, biology, law, philosophy, computer science, other, economics, business, history, math, health. The old interleave used the same order. Ties matter at three ladder points: at 406, psychology gets 28 and business 27 (both have 40 in Lite); at 294, physics and philosophy get the half-item seats before history and health; and at 252, engineering gets 21 and law 20 (both have 48 in Lite). Alphabetical order would change those rows.

The pre-registered lines that the HYPOTHESIS amendment text needs to supersede:
- Allocation and estimand: l.181 (C14, "42 in each of 14 categories" and "the n_M ladder keeps every category equal"), l.330 ("n_M is always a multiple of 14 … category-balanced"), l.464 (H3, "category-balanced") and l.622 (MMLU-ProX-Lite, "category-balanced").
- The full-pool rule, stated once: "non-Lite, with an answer letter that agrees with answer_index in EN and DE; 3787 excluded and listed". It supersedes l.276 (peer check, 30 items), l.286 (K8 pilot, 8 + 8), l.594 (Control C1, 100 items) and l.623 (MMLU-ProX full), and l.186 (C19) by reference. The full-split category counts for post-stratification still include 3787.
- Power: a note at l.445 ("post-stratification design effect 1.11") that the registered 1.11 is kept, so the tables are conservative under the proportional allocation. Its design effect is between 1.002 and 1.004 at every ladder n_M, computed with the MMLU-Pro category shares of `analysis/vendor_values.json`.

Hash scopes changed: `runner`, `tasks`, `manifest_rule`, `analysis`, `tools`. The other 12 match.

Mini verification, after the review fixes (which touched only `RUNBOOK.md` and this log, neither in a hash scope): 1,225 tests pass (Python 3.12 and 3.14, every test required; 1,206 before, plus 19; on the mbp, 1,217 passed and 8 `build-host only:` skips); dry run 17/17 (3.2 min), with the plan stage FIXED at P0 with n_M 28, and the manifests stage showing `mmlu_prox_lite_en/de` 588 and `"checks": {"gpqa_diamond_en_overlong_excluded": 0, "mmlu_prox_full_gold_inconsistent_excluded": {"de": ["3787"], "en": ["3787"]}}`; leak check `--all --no-gpqa-source` 0 findings (11 host-name warnings); `hash_tree --check` 12 match and 5 MISMATCH until the amendment is appended, 17/17 on a copy with the five `hash_tree:` lines appended.


## Amendment 5 — chat-wrapped peer fidelity and H8; a greedy-flip failure is B=1 (2026-10-04)

RUNBOOK step 9 on the mbp failed G8, G4 and Q36-8; Q36-4, Q38-8 and Q38-4 passed (`aborted/20261004T143051Z-peercheck/NOTE.md`, commit 381b1bd). There were two separate causes.

1. **Gemma 4.** The mbp measured NLL of 10.21 / 10.66 nats/token at 8 / 4 bits, KL(8‖4) of 4.57 and a G8 floor_kl of 0.276. Reproduced on the mini with `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit`: the instruction-tuned model, teacher-forced on raw text after `<bos>`, falls into its chat/thinking-channel format. Greedy decoding from "`<bos>`The capital of France is" emits "thought\n<channel|>…", and NLL is about 11 nats/token with top-1 0.12 on T1. Scored as the assistant turn after a user message, with thinking off, it behaves as a normal language model. None of the NOTE's suspects is the cause: the cached forward equals an uncached one exactly past the 1,024-token sliding window (T1, T4); bf16 rounding changes nothing; raw `tokenizers`, HF and joint encoding give the same ids.
2. **Q36-8** failed only the greedy flip (1/30 > `FLIP_MAX` 0.02). HYPOTHESIS counts the flip as part of the batched-path check, but `_verdict()` returned `fail` because the flip failure went under `problems`.

Andrei's decisions (2026-10-04): (1) "Chat-wrapped": every model scores each gate text as the assistant turn after the single user message "Write a text.", rendered with its own chat template, `add_generation_prompt=True` and thinking off. The text is tokenised alone and only its tokens are scored. This applies to the peer-check NLL / KL rule and to H8. G3 and every Kolibri gate check stay on raw text. Thresholds are unchanged. (2) "Keep as registered": `FLIP_MAX` 0.02 and `FLIP_ITEMS` 30 stay; only the misclassification is fixed.

What changed (nothing committed; HYPOTHESIS.md untouched):
- `bench/kl_8v4.py` (BENCH):
  - `chat_wrapper(family, dir)` renders the user message through `runner.chat.render(..., "none")`, the scored runs' own rendering, with the thinking-off kwargs passed explicitly. Kolibri uses `reasoning_effort="none"` with its tokenizer_config template (41 tokens). Gemma 4, Qwen3.6 and Qwen3.8 use `enable_thinking=False` with `runner/templates/` (17 / 16 / 16 tokens).
  - The input is `prompt_ids + ids[:-1]`, and `scored_chunks` keeps rows from `score_from = len(prompt) - 1` on. The 8-bit and 4-bit folders must render identical prompt ids, or the run stops.
  - Each family records its wrapper: messages, kwargs, template and its sha256, rendered sha256, prompt ids and prefill. This replaces `prefix`.
  - Blocks, bytes and the keys `analysis/verdicts.py` reads are unchanged. The K8 dump now has one row per text token.
  - After the review (below), each text also records `wrapper_boundary` and `first_rows`.
- `tools/peer_check.py` (TOOLS):
  - The NLL(8) / KL(8‖4) rule reads the chat-wrapped stats: `families.<f>.fidelity`, `arms.<a>.nll_chat` and `families.<f>.wrapper`.
  - The raw-text NLL and bpb are still computed and stay in `arms.<a>.nll`, where G3 reads them. `fidelity_raw_text` is recorded with `used_for_verdict: false`.
  - A parity or flip failure goes under `batched_path_problems` and gives `B=1`. Anything else, including a check that could not run, is still `fail`.
- `runner/plan_fix.py` (RUNNER): the plan records `peer_b1`, and the amendment gets the line "Peers at B = 1 …". The existing B rule already sets every Tier A and Tier B cell of such an arm to B = 1.
- `tools/dry_run.py` (TOOLS): the peers and bench stages fail if the wrapper, `nll` or `nll_chat` is missing.
- `RUNBOOK.md` step 9 and `BUILD_SPEC.md` (`kl_8v4.py`, `peer_check.py`): short Amendment 5 notes.
- Tests: 17 new across `tests/test_bench_kl.py` and `tests/test_tools_peer_check.py` (16 from the implementation, 1 from the review). `tests/test_integration_contracts.py` is updated for `wrapper_ids`.

The batched-path parity (the open point): it teacher-forced the raw T1–T4 token stream, with no BOS and no template, for both the noise floor (positions 520–1,100) and the B = 8 / B = 1 prompts. It therefore had the same Gemma degeneration, so the same wrapper now applies:
- The base is the wrapper followed by the stream.
- Prompt i is the wrapper followed by `stream[50i : 50i + n_i − len(wrapper)]`, which keeps the registered lengths 37–1,100.
- Every floor position falls inside the text.

The greedy flip already rendered its MMLU items through `runner.chat` with thinking off, so it is unchanged. Measured on the Gemma stand-in, re-run after the review:

| | raw (old) | chat-wrapped | Q36-8 on the mbp |
|---|---|---|---|
| floor_kl | 0.2305 | 0.0114 | 0.009 |
| KL bound | 0.6914 | 0.0343 | |
| parity mean KL | 0.1778 | 0.0115 | |
| ok | yes | yes | |

Downstream handling of `B=1`:
- `guard.require_peers` lets the arm run.
- `guard.excluded_arms` does not drop it.
- `pilot --without` refuses it.
- `plan_fix.build_context` reads it into `peer_b1`, so every cell of the arm gets B = 1 and the amendment names it.
- The pilot itself still runs such an arm at the memory-rule B (see "Needs Andrei").

### Review of the amendment (2 findings, both verified and applied)

1. **The wrapper/text boundary differs by family.** The implementer report and the test docstring said that encoding the prompt and text together gives the same ids "for every family". That holds for texts that start with a letter, not for the real T1, T2 and T6.
   - T1 and T2 start with "\n" and T6 with "\n\n". The Kolibri, Qwen3.6 and Qwen3.8 wrappers end on the token "\n\n" (Kolibri 263, Qwen 271). The scored sequence therefore has a separate newline token where the joint encoding has one merged token: Kolibri [263, 10] vs [120038] on T1 and T2, [263, 263] vs [120724] on T6; Qwen [271, 198] vs [1358] and [271, 271] vs [987]. All other ids are the same.
   - Gemma 4's wrapper ends on the special token `<channel|>`, so its joint and separate encodings are identical on all six texts. T3–T5 are identical for every family.
   - Verified with the real tokenizers on the mini. T5 and T6 were rebuilt from the pinned parquet and match MANIFEST.json.
   - Size, measured on a proxy (Qwen3-1.7B at 8 and 4 bits, affine g64, with the Qwen wrapper, through `compare_loaded`):
     - The row after the split newline has KL 1.59 / 1.59 / 2.26 nats on T1 / T2 / T6, against 0.09 / 0.12 / 0.005 for the same row on T3 / T4 / T5.
     - The excess is 0.48–0.53 % of each affected text's KL sum.
     - The first two text rows hold 0.70 % of the pooled KL chat-wrapped, against 1.58 % in the old raw measurement.
     - The first scored row predicts from the wrapper alone, so its KL is the same on every text of a family.
   - The scored ids stay as decided ("tokenised alone").
   - Applied:
     - The test docstring is corrected.
     - The new `test_wrapper_text_boundary_on_the_real_gate_texts` pins the split ids above, the merged joint ids, identity everywhere else, and Gemma's `<channel|>`.
     - `kl_8v4.py` documents the boundary.
     - Each text's H8 record now carries `wrapper_boundary` (`kl_8v4.wrapper_boundary()`: the differing stretch, or `joint_equals_scored: true`) and `first_rows`. `first_rows` holds the per-position kl, nll8, nll4 and agree of the first two scored rows (plus kl_fp32 for Kolibri) and the first bytes of tokens 0–2. A descriptive H8 sensitivity without those rows can therefore be computed from the record without a re-run. No analysis code changed; whether to publish that sensitivity is Andrei's call.
2. **No test covered the step from the peers record into the plan.** The downstream test set `peer_b1` by hand.
   - `test_a_b1_record_downstream_runs_every_cell_at_b1_and_is_never_a_drop` now calls `plan_fix.build_context` on a scratch experiment directory that holds the `run()` record. It asserts:
     - `peer_b1` is `["G8", "Q36-8"]`, `excluded_arms` is empty, and the record path is right;
     - every Tier A and Tier B cell of both arms gets B = 1, while K8 keeps its memory-rule B;
     - a newer record that fails G8 moves G8 from `peer_b1` to `excluded_arms`.
   - A mutation of the reader that ignores `verdict` makes the test fail.
   - This is test-only, and no tree hash changes from it.

### Real-Gemma smoke (re-run after the review)

`gemma-4-26B-A4B-it-OptiQ-4bit` (snapshot `dbfd2a77…`, the only Gemma build on the mini) stood in for both 8 and 4 bits. Its chat template has the same sha256 as `runner/templates/gemma4.jinja` (`36e3a42e…`). Wrapper: 17 tokens, prefill `empty_channel`.

| | T1 | T3 | T4 |
|---|---|---|---|
| Chat-wrapped NLL/token, full text | 4.013 | 2.107 | 2.238 |
| Top-1, full text | 0.396 | 0.633 | 0.723 |
| Chat-wrapped NLL/token, first 1,000 text tokens | 4.064 | 2.029 | 1.898 |
| Raw NLL/token (`<bos>` + text) | 10.76 | 7.61 | 10.37 |

- KL(stand-in‖itself) is 0 on every row, nll8 equals nll4 bit for bit, and agreement is 1.0.
- Through `pc.text_stats`, the fidelity rule gives NLL 2.706 / 2.706 and KL 0.0.
- `wrapper_boundary` is `joint_equals_scored: true` on T1, T3 and T4.
- The diagnosis probe's 4.10 / 2.03 / 1.84 is near the first-1,000-token row (and the first-1,024: 4.09 / 2.05 / 1.94), not the full text. The probe's exact scope is not recorded, so it most likely scored a prefix of about that length. T4's full-text figure is higher because its tokens after position 1,024 average 2.56.

### Verification on the mini

| Run | Result |
|---|---|
| Full suite, `EXP036_REQUIRE_ALL=1`, venv312 | **1,242 passed** (4:06; 1,225 before the amendment) |
| The same, venv314 | **1,242 passed** (4:17) |
| `tools/dry_run.py` | **ok, 17 of 17 stages, 3.6 min**. Peers stage: wrapper tokens Gemma 4 17, Qwen3.6 16, Qwen3.8 16; batched path ok for G8 and Q36-8. Bench stage: the H8 record carries the wrapper, `wrapper_boundary` and `first_rows`. Leak stage clean |
| `tools/leak_check.py --all --no-gpqa-source` with `EXP036_DATA` | **0 findings**, 11 host-name warnings |
| `tools/hash_tree.py --check HYPOTHESIS.md` | 14 match, 3 MISMATCH (`runner`, `bench`, `tools`) until the amendment is appended; 17 of 17 on a copy with the three lines below |

```
hash_tree: RUNNER_SHA256 = 1fc95a1accb7e9b8062360916724e07cf577a6c3f422aed602dcb8929e0b7b70
hash_tree: BENCH_SHA256 = 7f024661e956d7b7d5bc9384cae4201e63cb9b6f17527bc70efdbf12e166ba73
hash_tree: TOOLS_SHA256 = 2858d1e690c20a4457e57272048d215d865cb236c3a269934d27e8c4d7359b2c
```

`tasks/selection_rules.json`, the gate, analysis and thresholds are untouched, so the IFBench manifests need no rebuild. `ensure_exact_fp32()` is still the first call of every entry point.

### HYPOTHESIS passages the amendment text needs to supersede

- **C3** (assumptions table, "Peers run on mlx-lm's own modules"):
  - Registered text: "NLL(8-bit) ≤ NLL(4-bit) + 0.02 nats/token on the gate text, KL(8‖4) < 0.2".
  - Amended: the rule is measured on the gate texts, each scored as the assistant turn after "Write a text." in the model's own chat template with thinking off, the text tokenised alone and only its tokens scored. The thresholds are unchanged.
  - Its "batched-path parity" is teacher-forced on that wrapper followed by the T1–T4 token stream.
- **"Peers are verified, not gated."**
  - "(G3 uses its NLL)" means the raw-text NLL, which the peer check still records.
  - "B = 8 vs B = 1 teacher-forced under the G5 noise-floor rule" now runs on the wrapper plus the T1–T4 stream. Every prompt starts with the wrapper, and the registered lengths 37–1,100 are kept.
  - Confirm that the greedy flip is part of "the batched-path check", so that a flip-only failure is `B=1` and every Tier A and Tier B cell of the arm runs at B = 1. This is a reading of the registered sentence, not a change to it.
- **H8 "Measurement"**:
  - Registered text: "Every model teacher-forces the same bytes (the decoded text of T1–T6, ≈ 43 kB) with its own tokenizer."
  - Amended: each text is the assistant turn after the same fixed user message, rendered by the model's own chat template with thinking off (Kolibri `reasoning_effort` none; Gemma 4, Qwen3.6 and Qwen3.8 `enable_thinking` false). Only the text's tokens are scored. Bytes, blocks and normalisation are unchanged.
  - Add the boundary sentence: for Kolibri, Qwen3.6 and Qwen3.8, the first text token of T1, T2 and T6 follows the template's trailing "\n\n" as a separate token, where their tokenizers would merge the two. Gemma 4 is unaffected. On a proxy this adds about 0.5 % to each affected text's KL sum, in block 0. Each text's record carries `wrapper_boundary` and `first_rows`.
- **G3 row** ("from the peer check"): not superseded. Add a clarifying note that this is the peer check's raw-text bpb (`arms.<a>.nll`), so G3 stays raw-text on both sides.
- **Budget, "Peer check" row** ("NLL and KL on T1–T6"): unchanged at 0.4 / 0.6 h. The peer check now makes two passes per family (raw for G3, chat-wrapped for the rule); the Gemma passes above took minutes.

### Needs Andrei

1. **Pilot B for a `B=1` arm.** The pilot runs every arm at the memory-rule B, including an arm the peer check marked `B=1`. The plan then projects and queues it at B = 1, which is how the kit reads "re-projected by the plan rule". Running that arm's pilot cells at B = 1 as well would be a change to `runner/run.py`.
2. **The optional H8 sensitivity** without the first two text rows (review finding 1). It is computable from `first_rows`, but it is not pre-registered and is not in `analysis/`. Publishing it would need a line in the amendment.
3. **H1 and H8 consequences.** The step-9 failures of G4 and G8 came from the raw-text measurement. The mbp re-runs step 9 after this amendment is pushed, and the new record decides.

## Amendment 6 — a failed family fidelity rule attributed per build; G4 speed-only (2026-10-04)

The step-9 re-run after Amendment 5 failed G8 and G4 on the family rule (chat-wrapped NLL(8) 2.938 > NLL(4) 2.910 + 0.02; KL(8‖4) 0.323), with G8's greedy flip 1/30 on top (`results/peers_20261004T175538Z.json`, 3461c8c). The Gemma diagnostic on the mbp (6d70e09; records e6e8d34, 70fedbf) put the loss on G4: kernels exact at 4 and 8 bits, G4 the standard 4-bit quantisation of G8's weights (327 tensors at d84/e4 = 1.003), unquantised tensors bit-identical, and against bf16 @ `e13fae2a` on T1–T4: KL(bf16‖G8) 0.0167, KL(bf16‖G4) 0.3778.

Decision: **pending Andrei's confirmation.** The kit implements option 2 of the three put to him (1 keep G8, drop G4; 2 keep G8, drop G4 from quality, keep G4 as H1's speed reference; 3 drop both): G8 stays (B = 1 by its own flip); G4 leaves every quality use and H8's peer median but stays H1's speed reference. The relayed mbp message still held the three options as a placeholder, so nothing in the kit names his choice: the amendment's decision line is `{{ANDREI_DECISION}}`, and `tools/status.py` refuses to append it until the main session fills it in.

What changed (nothing committed; HYPOTHESIS.md untouched; the amendment is `amendments/6_peerfidelity_20261004T200930Z.md`):
- `tools/fidelity_reference.json` (new, TOOLS): pins `diagnostics/gemma_quant_check_20261004T192910Z_D.json` by sha256 `9334e973…`, its rows' field names, texts T1–T4, the bf16 source and the producer. `diagnostics/` stays outside every scope.
- `tools/peer_check.py` (TOOLS):
  - `load_fidelity_reference()` checks the pin, the record's sha256 and its rows (every arm of the family, exactly the pinned texts, no repeats); `{}` without a pin.
  - `attribute_fidelity()` (pure) requires each row to agree with this run's chat-wrapped NLL of that arm (tokens exact, NLL within `REFERENCE_NLL_TOL` = `NLL_MARGIN` = 0.02), then judges each arm by its token-weighted KL(bf16‖arm): < `KL_MAX` passes, ≥ `KL_MAX` is `speed-only`.
  - `apply_family_rule_failure()` replaces the two-line registered consequence: without a pin for the family, or with an unusable or non-matching one, both arms get the registered problem, and `families.<f>.fidelity_attribution` says why (`used: false` with a reason, or the mismatches); otherwise a passing arm gets no problem and a speed-only arm gets `speed_only_problems`.
  - `_verdict()`: `fail` > `speed-only` > `B=1` > `ok`; `exit_code()` gives 2 for `speed-only`; `arm_line()` prints the attribution. `run(…, reference=None)` loads the pin; any error reading it is recorded and leaves the registered rule.
  - `reference_classes()` for the dry run.
- `runner/guard.py` (RUNNER): `require_peers` refuses a `speed-only` arm with its own message; `speed_only_arms()`; `excluded_arms()` shares `_peer_verdict_arms()` (same output as before).
- `runner/run.py` (RUNNER): `pilot --without` also accepts a `speed-only` arm.
- `runner/plan_fix.py` (RUNNER): `speed_only_arms` from the record; no Tier-A or Tier-B task cell for them (`tier_b_cells(…, speed_only=…)` keeps the B4 ladder); a MoE peer among them counts as dropped for H3, H4, H6; `h8_peers` and `h8_families_left` from every excluded or speed-only build (`H8_PEER_FAMILIES` derived from `ARMS`), and H8 under `not_run` when no family remains; amendment lines "Speed-only arms …" and "H8 peer families …"; `check_pilot_inputs` treats speed-only arms as having no pilot cell.
- `tools/status.py` (TOOLS): `--sync-amendments` and `--append-amendment` refuse, appending nothing, an amendment that still holds an upper-case double-brace placeholder.
- `analysis/verdicts.py` (ANALYSIS): H8 drops a family with an arm in the plan's `excluded_arms` or `speed_only_arms` and records `families_left`. H1 is untouched.
- `tools/dry_run.py` (TOOLS): the dry-run peers record marks the pin's speed-only arms (G4) `speed-only`; the plan stage checks they have no task cell and that their family left H8; the verdicts stage checks H8's models.
- `bench/` unchanged (it reads no peer record; H1 and the ladder keep G4; `kl_8v4.py` still measures Gemma's KL, now descriptive).
- `RUNBOOK.md` steps 9 and 12 and the one-writer rule, `BUILD_SPEC.md` (`guard`, `plan_fix`, `run.py pilot`, `peer_check.py`, `status.py`): Amendment 6 notes.
- Tests: `tests/test_tools_peer_check_reference.py` (new) plus one each in `test_runner_guard.py`, `test_analysis_verdicts.py` and `test_tools_status.py`, and three in `test_runner_plan_fix.py`. Ten mutations of the new logic (speed-only as a problem, Tier A ignoring speed-only, analysis ignoring it, no agreement check, the ladder dropped, the 0.2 edge, H8 families ignoring it, no reason without a pin, no plan NOT RUN for H8, no placeholder refusal) each fail a test.

Re-attributing the committed step-9 record with the pin gives G8 `B=1` (fidelity passes at 0.0167; its flip), G4 `speed-only` (0.3778), Q36-8 `B=1`, Q36-4, Q38-8 and Q38-4 `ok`, exit 2; the record's NLL rows agree with that run to about 1e-7.

Review fixes (2026-10-04, before the push):
1. **Decision not on record.** The decision text came from the workflow's task text, not from Andrei; the relayed mbp message still held the options as a placeholder. The pin's `amendment` field, `guard.speed_only_arms` and the `peer_check` docstring now name Amendment 6 only; the amendment's decision line is `{{ANDREI_DECISION}}`, and `status.py` refuses to append it unfilled. Filling it changes no tree hash (`amendments/` is in no scope). If Andrei picks option 1 or 3, the code changes (see the hand-back).
2. **Opening sentence.** It said no verdict rule of H1–H8 changes; H8's peer set does (edge case 7 extended to a `speed-only` build). Reworded.
3. **No-pin reason.** `apply_family_rule_failure` wrote no reason for a family without a pin; it now writes `fidelity_attribution = {used: false, amendment, reason}`. A failed Qwen family on the step-9 re-run will say why no reference was used.
4. **H8 NOT RUN in the plan.** With every peer family gone, `plan_fix` wrote `h8_peers: []` but no `not_run` entry; it now adds H8 (a K4-exclusion reason is kept). Analysis already made H8 NOT RUN.
5. **RUNBOOK step 9** said `--without` is accepted only for a `fail` arm; now "`fail` or `speed-only`".

Pin sha256 after fix 1: `64e4e5c7fec06d6d98349bde81875d1d23e4e31596203d1fef11e66d0017f29e`. Tests: 1,274 pass in Python 3.12 (4:07) and 3.14 (4:18), every test required. Dry run 17 of 17 (3.6 min; G4 `speed-only`, `h8_peers` qwen3_6, qwen3_8; H8 models kolibri, qwen3_6, qwen3_8). Leak check: 310 file versions, 0 findings, 11 warnings (all pre-existing). `hash_tree --check`: 14 match, 3 MISMATCH (runner, analysis, tools) now; 17 of 17 on a copy with the amendment appended (decision line filled with test text).

hash_tree: RUNNER_SHA256 = d3d228a6395c976232d730363e70f166bfe0607766b5ae48f4cbddc33790690f
hash_tree: ANALYSIS_SHA256 = 83cedd792b8d77b3749be8e76abeb9db70cd92da0e33272e88f5abd92ded78a2
hash_tree: TOOLS_SHA256 = 690de58ba70e6346cdca38eae8446f9fcf226f7e1c0b308b9bd6d0942ad64dc9
