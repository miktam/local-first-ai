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
