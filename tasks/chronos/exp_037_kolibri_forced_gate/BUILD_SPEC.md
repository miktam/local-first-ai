# Experiment 037 — build specification of the kit (delta over exp_036's)

*This file is a delta. exp_036's [`BUILD_SPEC.md`](../exp_036_kolibri_local_eval/BUILD_SPEC.md) (sha256 `33464389ca7cd34a2beadc858f55a174ceba21fd78fcf198964e981b5fa5d2ed`, at `222c845`) stays the specification of every part of the kit this file does not change; its section numbers are used here. [`HYPOTHESIS.md`](./HYPOTHESIS.md) is the pre-registration; where the two disagree, HYPOTHESIS.md wins and this file is fixed. [`RUNBOOK.md`](./RUNBOOK.md) is how Andrei and the mbp session run it; [`BUILD_LOG.md`](./BUILD_LOG.md) records how it was built. Precedence: `scientific_log.md` > HYPOTHESIS.md > BUILD_SPEC.md (this delta, then exp_036's) > RUNBOOK.md.*

**Hosts**

| Host | Role |
|---|---|
| the mini | Builds the kit. Runs only weight-free unit tests and the whole gate on tiny checkpoints, and later the analysis. Environments (decision F2; never exp_036's): `~/models/exp037-mini/venv312` (Python 3.12) and `~/models/exp037-mini/venv314` (Python 3.14), both MLX 0.32.3 / mlx-lm 0.32.0. Tiny checkpoints, tokenizers and build-time data stay under `~/models/exp036-mini/`. |
| mbp | Runs everything that touches real weights, from `$EXP037_VENV` (default `~/models/exp037/venv`, Python 3.12), which `env/exp037.settings.sh` exports as `$PY`. |

**What "copied" means.** The kit was created from exp_036's tree at `222c845` (clean): 187 files copied byte for byte, 82 copied as pre-modification originals and then changed by their owner task, and new files added. `COPY_RECORD.json` records every file's source and copy sha256 and its status. exp_036's directory is read-only: nothing in it is edited, moved or deleted, and its env file is never opened, copied or hashed (it is sourced in place).

---

## 1. Directory tree (delta)

New in exp_037, beside exp_036's tree:

```
exp_037_kolibri_forced_gate/
├── HYPOTHESIS.md  BUILD_SPEC.md (this delta)  RUNBOOK.md  README.md  BUILD_LOG.md  NOTICE  COPY_RECORD.json
├── env/exp037.settings.sh            exports EXP036_DIR, EXP, EXP037_BUILDS, EXP037_VENV, PY (≤ 25 lines, idempotent)
├── port/kolibri1.py                  exp037-port-1: VllmRoPE, rope_offset / attn_mask hooks
├── gate/rules.py                     every pass/fail decision (pure functions of measured dicts)
├── gate/controls.json                required controls per check, bound legs, tiny-margin units; probes
├── gate/calibration.json             F_tiny and its validation (written by gate/checks/ftiny.py)
├── gate/preconditions.py             P1, P2, P3 (wrapper), P4
├── gate/checks/g0k_kernels.py        G0k probe (P3)
├── gate/checks/ref_drivers.py        R2F, R3, the R1 rows pass, P5a
├── gate/checks/g4_forced.py          G4-F32, G4-F16, Csort, S, the control mutant loop
├── gate/checks/g5_runner.py          G5-D32, G5-R1, G5-BP-lean, prompts26, the phase-6 anchor and reductions
├── gate/checks/ftiny.py              F_tiny calibration and validation
├── tools/refresh_builds.py           clone exp_036's builds and refresh their port file (RUNBOOK step 8)
├── tools/power_log.py                descriptive power log: powermetrics' CPU + GPU + ANE power per run window →
│                                     results/power/power_<UTC>.json (RUNBOOK "Power log"); decides nothing
├── diagnostics/gemma_quant_check.py  byte-identical copy of exp_036's producer (S1 fidelity record)
├── diagnostics/tiny_stress/          the tiny stress records: sharpened, unsharpened, unsharpened in bf16 (pre-freeze)
├── diagnostics/vendor_spike/         the vendor-code spike: README, scripts, outputs (pre-freeze)
├── tests/tiny_real_layout.py         the tiny real-layout builder (seeds 23, 29; sharpen 3 or 1)
├── tests/fixtures/rope_angle_262143.json   the RoPE angle fixture (mini, MLX 0.32.3)
├── tests/fixtures/exp037_tiny_controls.json   the tiny control table
└── tests/test_*.py                   the new test files of §6
```

Not built under decision F2: `runner/rows.py`, a port rows guard, `tests/test_port_rows_guard.py`. exp_036's `HYPOTHESIS.md`, `RUNBOOK.md`, `BUILD_SPEC.md`, `BUILD_LOG.md`, `README.md`, `results/` (except `results/manifests_20261004T131126Z.json`), `aborted/`, `amendments/`, `evidence/` and `diagnostics/` (except the Gemma producer) are not copied.

## 2. Invariants (delta)

Unchanged from exp_036 §2, plus:
- **Writers.** New work files go only under `$EXP036_WORK/exp037/` (`gate.common.work37_dir()`); a test asserts that every exp_037 writer path is under `exp037/` or is a new UTC-stamped name. R1 is read from exp_036's dump under its registered key and recomputed only under `work37_dir()` (no writer, `fp32_dump`'s `rmtree` included, targets a path outside it). Exceptions, by inherited bench code: `bench/kl_8v4.py` (unchanged) dumps H8's K8 log-probs to its default `$EXP036_WORK/kl/<UTC>/`, and `bench/batch_flip.py` writes C1's full records to `$EXP036_WORK/c1/c1_<UTC>.full.jsonl`; both are new UTC-stamped names beside exp_036's, and nothing of exp_036's is touched.
- **Step logs** (decided 2026-10-06, before the freeze). The per-step decode logs `*.steps.jsonl` (`runner/generate.StepLog`: one JSON line of about 190 B per decode step) follow two rules. The pilot's (`results/pilot/<UTC>/<arm>/`, and the copies `runner/run.py abort-pilot` moves under `aborted/<UTC>-pilot/`) are committed (an aborted pilot's copies with the rest of that pilot): the registered plan rule fits its step model on a summarised pilot's (each cell's `steps_file`; `runner/plan_fix.py` `load_steps`; HYPOTHESIS "Pilot and the rule that fixes n and max_tokens", rule 4), and `runner/run.py` sums the pilot's prefill from them, so that the plan can be re-derived from the public repository; they are small (about 1 MB a cell at B = 8, at most about 50 MB at B = 1 with every item at the 32k cap), and one still over `tools/leak_check.py`'s 50 MB limit stays uncommitted through the mbp's `.git/info/exclude` (RUNBOOK step 13). The sessions' (`results/raw/<session>/<arm>/`, and the copies `abort_cell` moves to `aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl`) are git-ignored working files (`.gitignore`: `results/raw/**/*.steps.jsonl`, `aborted/*/S*-*.steps.jsonl`): about 190 MB for a K8 cell at B = 1 and gigabytes over all sessions, over the 50 MB limit and GitHub's 100 MB file limit. They are never committed and never deleted, stay on the mbp and are copied to the mini at RUNBOOK step 18; no score, verdict or registered rule reads them (`scorers/score_all.py` and `analysis/exploratory.py` skip them; on the mbp `runner/run.py` moves them on a cell abort). Both are bound by sha256 in the run-record blocks (`tools/status.py` hashes files on disk, git-ignored or not): the plan block lists the pilot's, the S2 and S3 blocks the sessions'.
- **Private files.** `runner.common.private_dir()` = `$EXP036_PRIVATE/exp037/` for exp_037's raw, pilot and aborted private records; the reused withheld manifests are read from `$EXP036_PRIVATE/manifests/`.
- **Builds.** Every Kolibri arm directory resolves under `builds_dir()` = `$EXP037_BUILDS` if set, else `models_dir()` (`gate/common.py`, `runner/common.py`, `bench/common.py`, `tools/common.py`); peers stay under `$EXP036_MODELS`, read-only.
- **No MLX-computed exp_036 value is reused** as an exp_037 value (HYPOTHESIS, "What counts as evidence").
- **Identifiers.** Seeds, schemas, versions and messages that identify the experiment say exp_037 (`runner/seeds.py` `SEED_PREFIX = "exp037"`; `analysis/stats.py` `b"exp037|"`; `gate.common.seed_from(..., prefix="exp037")`; gate record `experiment` "exp_037"; `thresholds.json` `exp037-gate-1`; `analysis/verdicts.py` `exp037-verdicts-1`; `margins.json` `exp037-margins-1`; port `SPEC_VERSION` `exp037-port-1`; module prefix `exp037_gate_port_`; the version, preflight, peer-check and fidelity-reference schemas `exp037 …`; commit prefix `chronos/exp_037:`). Kept, because they are bound to reused artefacts or registered fixtures: every `EXP036_*` variable name; `exp036_convert_record.json`; the config keys `exp036_quantize_embeddings` / `exp036_quantize_lm_head`; the default paths `~/models/exp036` and `~/models/exp036-mini`; the gate-text strings; `tasks/manifests/*`; `tasks/common.py`'s `exp036|` item order; `tools/peer_check.py` `PARITY_SCHEMA`; the three registered fixture seeds (`gate/tokenizer_lines.py`, `gate/textset.py`, `gate/run_gate.py`), which pass `prefix="exp036"` with the comment "registered fixture seed, not an exp_037 draw"; the internal attribute `EXP036_PORT_MUTANT`; exp_036's "Modified by Miktam for Chronos exp_036" header lines (a second line is added where exp_037 changes a file).

## 3. Environment (decision F2)

`env/requirements-mbp.txt` is the reviewed pin file with MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0; every other pin is exp_036's. Its line 2 names `$EXP037_VENV` as the install target (exp_036's named exp_036's venv):

```
mlx==0.32.3
mlx-metal==0.32.3
mlx-lm==0.32.0
numpy==2.5.3
safetensors==0.8.0
tokenizers==0.23.2
transformers==5.18.0
jinja2==3.1.6
pyyaml==6.0.3
pyarrow==25.0.1
pytest==9.1.1
```

- `env/versions.json` mirrors it (core and pins; exp_036's structure and note).
- `env/setup.sh` creates or syncs `$EXP037_VENV` (default `exp037/venv` beside `$EXP036_MODELS`), never exp_036's venv; Python 3.12 only; idempotent; never calls sudo; its messages say exp_037.
- `env/exp037.settings.sh` is sourced after exp_036's env file, from this directory. It refuses (return 1, with a message) outside a directory named `exp_037_kolibri_forced_gate` with `HYPOTHESIS.md`, without a sibling `exp_036_kolibri_local_eval` holding `HYPOTHESIS.md`, or without `EXP036_MODELS`. It exports `EXP036_DIR`, `EXP` (this directory), `EXP037_BUILDS` (default `$EXP036_MODELS/exp037-builds`), `EXP037_VENV` and `PY="$EXP037_VENV/bin/python"`, keeps a preset `EXP037_BUILDS` or `EXP037_VENV`, and changes no `EXP036_*` value. Sourced twice, it gives the same values.
- `tools/preflight.py` and `tools/version_record.py` read the core pins from `env/versions.json` and default to `$EXP037_VENV`; the version record carries `mx_device_info` (with `architecture`) for P2.
- **Exact fp32** (exp_036 Amendment 1) is unchanged: `MLX_ENABLE_TF32=0` in every process (`tools/precision.py`); libmlx 0.32.3 still reads the variable.
- **mlx-lm 0.32.0** (decision F2; Annex A of HYPOTHESIS.md): `model_file` loads need `trust_remote_code=True` (#1385, CVE-2026-5843); `BatchGenerator.stats()` is a window over monotonic counters (#1829); `_make_cache` is removed (`models.cache.make_prompt_cache` plus each cache class's `merge`); batch caches rebind their offset (#1848), read left padding without a reduction (#1824), promote correctly in `extend()` (#1491) and return the full `state` (#1778).
- **Test environment on the mini** (no env file is sourced there):
  ```
  EXP036_MODELS=~/models/exp036-mini EXP036_DATA=~/models/exp036-mini/data \
  EXP036_TOK=~/models/exp036-mini/kolibri/Kolibri-1-BF16 EXP036_REQUIRE_ALL=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/models/exp037-mini/venv312/bin/python -m pytest -q -p no:cacheprovider tests
  ```
  and the same with `venv314`.

## 4. Forward-pass specification (delta)

Items 1–29 are unchanged except:

8. **RoPE** (amended). NeoX half-rotation, rotary_dim 128, absolute 0-based positions, keys cached after RoPE, as before. **In MLX** (replaces exp_036's "In MLX:" line, l.245): `VllmRoPE(128, 1e4)`, verbatim from exp_036's `diagnostics/gate1/mini/bughunt/verify/kit_rope/port/kolibri1.py` l.249–266: `exps = arange(0, dims, 2, float32) / float32(dims)`; `inv = float32(1) / (float32(base) ** exps)`; `_freqs = mx.array(float32(1) / inv)`; `__call__(x, offset)` = `mx.fast.rope(x, dims, traditional=False, base=None, scale=1.0, offset=offset, freqs=_freqs)`. angle(p, j) = fp32(p × inv_freq[j]) with inv_freq[j] = 1 / 10000^(2j / 128) in fp32: vLLM's `RotaryEmbedding._compute_inv_freq`, equal to the reference's `rope_inv_freq` (`float32(1 / _freqs)` equals it bitwise). MLX's `base=` path differs by −2 to +8 fp32 ulps. `_freqs` is private, so the strict load is unaffected. vLLM stores cos/sin in bf16; neither side is bit-faithful to that, and the rounding is flat in position.
10. **MLX window mapping** under mlx_lm 0.32.0: unchanged in substance; the cache citations are 0.32.0's (`models/cache.py` batch caches l.880–1461; `base.create_attention_mask` returns no mask for a one-token query on a single-sequence cache, l.51).
27. **Batching** (restated for mlx_lm 0.32.0; the registered arguments are unchanged). One construction, `runner.generate.make_batch_generator(model, B, eos, max_tokens, sampler, prefill_step_size=2048)` = `BatchGenerator(fp32_logits(model), max_tokens=…, stop_tokens=[[e] for e in eos], sampler=…, completion_batch_size=B, prefill_batch_size=min(B, 8), prefill_step_size=2048, max_kv_size=None)`; mlx-lm 0.32.0 still sets `completion_batch_size = max(completion_batch_size, prefill_batch_size)` (`generate.py` l.1544). Prompts admitted together are right-padded and then finalised to left padding (`PromptProcessingBatch.prompt`, l.1163–1209), as under 0.31.3. `_make_cache` (0.31.3 l.838–866) no longer exists: each sequence's cache comes from `models.cache.make_prompt_cache` (the port's `make_cache`), and the batch caches are built by each cache class's `merge` (`_merge_caches`, l.815–828; one sequence's caches are merged too, so the runner at B = 1 uses batch caches). `GenerationBatch` runs its first step at construction as an `L == 1` step over the group's batch caches. The private attributes the kit reads still exist and are pinned by tests: `_generation_batch`, `_prompt_batch`, `_unprocessed_sequences`, `GenerationBatch.prompt_cache`, `BatchKVCache._idx`, `BatchRotatingKVCache._offset`, both caches' `left_padding`. Under F2 there is no `prefill_batch_size = 1` rule, no row assertion and no rows guard; G0k checks the kernels at the run's shapes.

## 5. File-by-file specification (delta)

### 5.1 port/

**`port/kolibri1.py`** (exp037-port-1)
- `VllmRoPE` (item 8) before `Attention`; `self.rope = VllmRoPE(self.head_dim, args.rope_theta) if use_rope else None`.
- Hooks on `Attention`: `rope_offset(self, cache, L)` returns `cache.offset` (0 without a cache) and is called on RoPE layers before the cache update; `attn_mask(self, mask, cache, L)` returns `mask`, is called on every layer after `update_and_fetch`, and its result goes to `scaled_dot_product_attention`. Output-neutral bitwise (tested with no cache, single caches and batch caches on two tiny checkpoints). The mutants override only these hooks.
- `SPEC_VERSION = "exp037-port-1"`; a second header line "Modified by Miktam for Chronos exp_037, 2026-10-06 (vLLM-angle RoPE, attention hooks)"; the docstring names mlx_lm 0.32.0 and states that it executes the file through `model_file` only with `trust_remote_code=True`, which the kit passes after `check_port_file`.
- No rows guard (decision F2). The bf16 residual add is unchanged.

**`port/convert.py`**
- `model_file_trust(model_dir, port_file=PORT_FILE) -> dict`: `{"trust_remote_code": True}` iff `config.json` names `model_file: kolibri1.py` and `check_port_file` passes; `{}` without `model_file`; `UntrustedModelFile` (a `ValueError`) for any other `model_file`; `StalePortFile` for a stale, edited or unrecorded port copy. Nothing executes before the check.
- `trust_remote_code=True` at the two `model_file` loads of a directory convert itself just wrote: `mlx_convert` and `refresh_port_file`'s verification load.
- `check_source()` refuses a source whose `config.json` or `tokenizer_config.json` declares an `auto_map` (mlx-lm 0.32.0 passes the same flag to transformers' tokenizer loading, so it could reach code shipped with the source). The real Kolibri-1-BF16 has none.
- A second modified-by line. `port/convert_streaming.py` is unchanged (it builds the model from the port module itself).

**`port/README.md`**: RoPE, the hooks, loading under mlx-lm 0.32.0, and that exp_037 makes no conversion (the builds are clones with a refreshed port file).

### 5.2 reference/

Unchanged, byte for byte (tree `85337ed7…`). The vendor-code spike forced no reference fix.

### 5.3 gate/

**`gate/thresholds.json`**: frozen; content in §7.1 (version `exp037-gate-1`). Registered blocks are inherited value for value; G2 keeps `bf16_selection_gap_sigma` 6.0 under its exp_036 name, listed in G2's `descriptive`; G3's bpb bounds and G4's decisive top-1 are listed as descriptive. New blocks: P2 (the F2 pins and their sources by sha256), P3_G0k (the judged row counts by routing group), P4 (exp_036's convert records by sha256), P5, G4N, G4F32, G4F16, G5D32, G5R1, G5BP (with the 26-prompt layout and the tiny layout); `K4_blocking` = G0, same_port_sha, G2q, G5_behaviour, G5_R1_anchor; `fix_cycles_max` 2.

**`gate/rules.py`** (new, frozen in `gate_rules`): every decision as a pure function of measured dicts and thresholds: G0/G1, G2 (with leg (p) and the per-row verdicts for the layers CSV), G3b's mutant classes, G4-N(i) and G4 K4, G4-F32 (F1–F6, τ, the F_tiny cap, the Csort ceiling as a FAIL), G4-F16 (κ, R3 VOID), G5 greedy, decode-vs-prefill (`parity_bound`), G5-D32, G5-R1, behaviour, G5-BP-lean and `allowed_B`, P3 (`p3_g0k`) and P5; BLOCKING and the K4 blocking groups; `control_result` and `controls_outcome` (probes reported apart; a probe changes no verdict, INCOMPLETE or allowed_B); `arm_verdict` (FAIL > MISSING > INCOMPLETE > PASS); `exit_code_for` (HYPOTHESIS's exit table); `is_gate_run` and `run_counts`; `tripwire_constants` and a reference `tripwire_rule`. Comparisons are written so that NaN never passes. It imports no mlx, port or file writer. No module under `gate/checks/` emits a `"pass"` key (a G1 test).

**`gate/controls.json`** (new, frozen): G4F32/1 (F6), G4F32/6 (F2), G4F32/19 (F2 on T9 or a bucket), G4F16/6 (κ), G5D32/20 (decode leg), G5R1/26 (R-parity), G5BP/27 ((d) at K8 B = 8), each with its leg type for the tiny margin; the registered G2 mutant rule; `probes`: G5BP/22 (tiny mode only).

**`gate/calibration.json`** (new, frozen): F_tiny, its block means, both checkpoints' config and shard sha256, the token rng seeds and validation layout, the estimator, the validation results, the cap, and the provenance (code sha256 of the measurement and rule files, thresholds and controls sha256, host label, package versions). Written by `python -m gate.checks.ftiny --root <dir> --write`.

**`gate/preconditions.py`** (new): `p1` (a)–(e) through `tools/hash_tree` (20 scopes, `NO_AMENDMENT_SCOPES`, the pre-registration commit as an ancestor of HEAD and upstream, re-run admissibility with the crash rules of W15-03 (`orphaned_runs`: no run directory without its record; `blind_crashes` and `amendment_first_lines`: a code change after an exit 3 past P5 needs an amendment whose first line names the run's UTC), the table rows of the pre-registration commit against HEAD's and the working copy's), `p2` (pins, and in real mode the newest exp037 version and preflight records), `p3` (a wrapper of `rules.p3_g0k`), `p4` (shard and config sha256 against exp_036's convert records, `check_port_file`), `run_counts`, `code_hashes`; each returns `{ok, values, reason}`.

**`gate/checks/g0k_kernels.py`** (new): `probe(...)` measures `rel_f64_sorted`, `rel_f64_unsorted`, `bad_rows` and `repeat_bitwise` per (projection, bits, rows, routing), lifted from exp_036's `mlx_repro.py` and `mlx_repro_min.py`; row counts read from `thresholds.json` P3_G0k.

**`gate/harness.py`**: `PortLayers.run` and `forcing_crosscheck` unchanged (the registered single call over T1–8). Added: `ForcedTrunk(RouteTap)` (modes `free`, `force` with the forced ids handed ascending, `force_own_order`; order-preserving outputs; shadow recording; HarnessError on a route-call count, row count, id range or selected-set mismatch), `forced_stream` / `forced_logits`, `records_table`, `decode_spans`, `p5b_exactness`, `p5c_forcing_active`.

**`gate/checks/ref_drivers.py`** (new): `forced_ref_pass` (R2F in fp32, R3 in emulation; keyed by the reference tree sha, the K8 shards, the text key, the R1 top-6 file shas and the mode; written only under `work37_dir()/ref_forced/`), `i_from_r1`, `forced_dump_key`, `ForcedDump`, `r1_rows_pass` (the head only on the wanted rows), `p5a`, `require_under_work37`.

**`gate/checks/g2_layers.py`, `ref_pass.py`, `g3_oracle.py`**: measurement only (their `evaluate` functions moved to `rules.py`); per-pair z through one function (`ref_pass.natural_vs_r1`, float64) for the port and the emulation; `emu_stats_exp037.json` with σ_emu, z_emu, M_emu and the flagged-pair flags; the flagged-pair report (a stand-in at the last layer on tiny checkpoints); `layer_csv_rows(res, emu, row_verdicts)`.

**`gate/checks/g4_e2e.py`**: measurement only. **`gate/checks/g4_forced.py`** (new): `fp32_on_dequantised_weights` (lifted verbatim from exp_036's `diagnostics/gate1/diaglib.py` l.247–279), `make_fp32_port`, `reference_profile` (S), `forced_series`, `forced_texts`, `csort_series`, `csort_sequence`, the G4-F32 and G4-F16 measured dicts, F4 windows (descriptive), and the control mutant loop over `controls.json`.

**`gate/checks/g5_generation.py`**: measurement only; its BatchGenerator comes from `runner.generate.make_batch_generator`; `run_admission` is the shared admission loop. **`gate/checks/g5_runner.py`** (new): `g5_d32`, `g5_r1`, `g5_bp`, `prompts26`, `anchor_sequences`, `reduce_with_r1`, `greedy_block`, and the log-prob file helpers (`<seq>.{runner|batched,single}.npy` under `work37_dir()/run/<UTC>/g5/<arm>/<check>/`, sha256-checked on every load, never overwritten).

**`gate/checks/ftiny.py`** (new): the F_tiny calibration (seed 23) and out-of-sample validation (seed 29) through the exact G4-F32 code path, and the writer of `calibration.json`.

**`gate/port_mutants.py`**: mutants 19 `rope_restart_per_chunk`, 20 `decode_window_256`, 22 `batched_rope_positions_in_padded_frame`, 26 `batch_decode_pos_frozen` and 27 `batch_decode_pad_keys_visible`, each a subclass of the port's `Attention` overriding only `rope_offset` or `attn_mask`; added to `MUTANT_IDS`, `TARGETS`, `PATCHES` and `HOOK_MUTANTS`; batch caches recognised by the presence of `left_padding`, whose value is never read; mutant 26 freezes a copy of the offset. `REAL_WEIGHT_MUTANTS` unchanged. A second modified-by line.

**`gate/run_gate.py`**: the phases 0, 1, 2a, 2b, 2c, 3, 4, 5, 6 (each a subprocess in real runs) and 7 in process; preconditions give exit 3; the exit table and record fields of HYPOTHESIS; `require_pass` also requires `GATE_RULES_SHA256` and the builds manifest to match and returns allowed_B; every exit-3 run moves its phase directory to `aborted/<UTC>-gate/` with a NOTE.md. Interrupts (final review W15-03): KeyboardInterrupt, SIGTERM and SIGHUP (`GateInterrupted`) and SystemExit during the phases are recorded like a crash, the record is written (signals during the write deferred) and the interrupt is then re-raised; the CLI prints the verdict line with `"interrupted"` and exits with the record's code. Each run writes `results/gate/<UTC>/run.json` (mode, arms, checks, start); `--close-orphans` judges a run directory left without a record by its phase files and writes its record (`close_orphans`). Phase 4 runs the unmutated K8's bf16 checks, converts it in place to fp32 for its fp32 checks, deletes it, and then runs every control mutant (gc and `mx.clear_cache()` before each load); the bf16 and fp32 K8 never coexist. Probes run in tiny mode only, in a second mutant loop whose crash cannot change the exit code. Tiny-mode options (refused in real mode): `--tiny DIR`, `--results-dir`, `--port-mutant`, `--head-policy`, `--g1-select`, `--g0k-rows`, `--no-controls`, `--tiny-precision` (default fp32 activations for the free-routing checks, as exp_036's tiny run), `--subprocess-phases`. `TINY_NOT_APPLICABLE` keeps exp_036's tiny exemptions (G3, greedy, behaviour) and adds G5-R1 K8's R-greedy leg. Phase files store non-finite floats as `{"$nonfinite": "nan"}`, so a NaN reaches `rules.py` and fails there.

**`gate/checks/g1_synthetic.py`**: `EXCLUDE = ("test_gate_drivers_tiny.py", "test_gate_end_to_end_tiny.py", "test_gate_ftiny.py")`.

### 5.4 runner/

- `runner/generate.py`: `make_batch_generator` (item 27); the step log reads `with gen.stats() as st` around each `next()`; `run_cell(..., allowed_B=None)` refuses a B outside allowed_B before anything runs; the docstring describes mlx-lm 0.32.0.
- `runner/run.py`: loads through `convert.model_file_trust`; pilot, session and cell call `guard.require_environment(guard.newest_gate_record(results))`; a Kolibri cell is refused unless its B is in allowed_B; the pilot clips the memory-rule B; the crash fallback steps within allowed_B; private records under `$EXP036_PRIVATE/exp037/`.
- `runner/plan_fix.py`: `B_for` clips each Kolibri cell to the largest allowed B, on the same path as `peer_b1`; `build_context` reads allowed_B from the newest gate record (`gate_<UTC>.json` by `guard.GATE_RECORD_RE`, never its `_mutants.json` sibling); `plan_fixed` and the amendment record it; the refresh records count in S1 hours.
- `runner/guard.py`: `require_gate` returns the sorted allowed_B (a record without one gives [1]; a malformed set is refused); `allowed_B_of`, `newest_gate_record`, `observe_environment`, `require_environment` (judges P2's pins only for a real-mode record without `dry_run_promoted_from`), `P2_PINS`.
- `runner/common.py`: `builds_dir()`, `private_base_dir()`, `private_dir()` (= base/exp037; redaction label `$EXP036_PRIVATE/exp037`), `clip_B`. `runner/seeds.py`: `SEED_PREFIX = "exp037"`.

### 5.5 tasks/

Unchanged, byte for byte, including the built `tasks/manifests/*.json` (reused; verified at RUNBOOK step 7).

### 5.6 scorers/

Unchanged except `scorers/score_all.py`: `--private` (default `$EXP036_PRIVATE`) is the private base; raw records are read from `<base>/exp037/raw/<rel>` (a constant `PRIVATE_SUBDIR`, tested equal to `runner.common.private_dir()` relative to the base) and withheld manifests from `<base>/manifests/`. No scoring logic changed; a modified-by line.

### 5.7 bench/

- `bench/genutil.py`: `load_arm` through `model_file_trust`; `_batch_generator` → `runner.generate.make_batch_generator`; `timed_generate` passes `EosMask` as a logits processor, whose output is independent of the token history mlx-lm 0.32.0 now passes it (#1777; tested); `batch_aggregate` reads `gen.stats()` (#1829; `generation_tps` = generation_tokens / (wall_time − prompt_time); tested) and records `wall_time_s` and `prompt_time_s`.
- `bench/common.py`: `builds_dir()`, `allowed_B`, `clip_B`, `require_environment`. `bench/batch_flip.py`: C1 at the largest allowed B ≤ the memory-rule B, NOT RUN ("no batching to control") at 1, still writing a complete file. `bench/speed.py`: Kolibri descriptive cells only at B ∈ allowed_B ∩ {1, 2, 4}. `bench/run_bench.py`: the version binding right after the identity guard. `bench/kl_8v4.py`, `fit.py`, `ladder.py`, `tokenizer_ratio.py` unchanged.

### 5.8 analysis/

- `analysis/stats.py`: `b"exp037|"`.
- `analysis/verdicts.py` (`exp037-verdicts-1`): the H2 tripwire T-G3 (`tripwire(h2_result, margins)`, evaluated once after H2 and its protocol control, no new random numbers; `protocol_control` returns `_did_samples`, stripped before writing); the tripped Q2 rows and the caveat (`PLAIN_ANSWERS`, `TRIPWIRE_CAVEAT`, verbatim as in HYPOTHESIS); the labels on H2–H4, H6–H8 and the computed E1–E5, E9, E10, E12 rows when TRIPPED; the labels "M5 Max, MLX 0.32.3".
- `analysis/margins.json` (`exp037-margins-1`): `H2_tripwire_ub` −0.10, `H2_tripwire_did_ub` −0.10, `H2_tripwire_q` 0.95, equal to `gate/rules.tripwire_constants()`.
- `analysis/power.py`: the tripwire power table and the greedy false-fail table. `power.budget()` keeps exp_036's S1 hours (5.4 / 9.1 / 9.6), which `tests/test_analysis_power.py` pins; HYPOTHESIS's S1 projections (6.2 / 9.2 / 9.7, "Sessions & budget") are the pre-registered planning numbers, and the plan rule uses the measured S1 hours in either case.
- Tier 2 (`exploratory.py`, `tables.py`): the 0.32.3 labels; `tables.py` renders the tripped Q2 rows and the caveat.

### 5.9 tools/

- `tools/hash_tree.py`: the `gate_rules` scope, and the final review's `tests` (the `tests/` tree) and `env` (`env/exp037.settings.sh`, `setup.sh`, `versions.json`) scopes, both amendable (20 scopes); `NO_AMENDMENT_SCOPES = ("gate_rules", "thresholds", "gate_text")`; `--amend-line` refuses those three; `check` reports a missing HYPOTHESIS.md as every scope unfilled; experiment-relative messages.
- `tools/refresh_builds.py` (new): RUNBOOK step 8 (HYPOTHESIS "Fixed before any run", converted builds). It refuses when `EXP037_BUILDS` is unset, equals `EXP036_MODELS` or lies inside a source; refuses symbolic links; checks that every clone file has its own inode; compares a stat fingerprint of every source entry before and after; writes `results/convert/refresh_{8,4}bit_<UTC>.json`. `--expect-source-port` is for tests and dry runs only.
- `tools/preflight.py`, `tools/version_record.py`: schemas `exp037 preflight v1` / `exp037 version record v1`; core pins from `env/versions.json`; `$EXP037_VENV` by default (`tools/common.exp037_venv()`); preflight's phase plan carries the S1 numbers and the exp_037 next steps.
- `tools/peer_check.py`: `SCHEMA` `exp037 peer check v1`; `_load` through `model_file_trust` (real peers get `{}`); the fidelity reference by rule (`load_fidelity_reference(before=…)`, `_reference_for_run(t_start=…)`; schema `exp037 fidelity reference v1`; the peer record carries the record's path, sha256 and UTC).
- `tools/fidelity_reference.json`: the Gemma 4 pin names the S1 record by rule (HYPOTHESIS, "Peers are verified, not gated").
- `tools/dry_run.py`: runs the exp_037 gate `--tiny` (every phase, every control, probe 22) on the unsharpened seed-29 real-layout build (`tests/tiny_real_layout.build_gate_validation_set`), the clone-and-refresh stage instead of convert, `peer_check.SCHEMA`; `REL_EXP` and the commit subjects say exp_037.
- `tools/status.py`: a `refresh` record-block phase; the S3 block (`S3`, `S3, mbp scores`) hashes every `.jsonl` under `results/raw/S3/`, `results/raw/S3b/` and `results/raw/S2/` in its final state (an S2 cell finished in S3 keeps writing under S2; `SESSION_COVERS`), step logs included; every session block also hashes its aborted cells' step logs `aborted/*/<session>-*.steps.jsonl` (S3's for S2, S3 and S3b); the `plan` block, and a standalone `pilot` phase, hash `PILOT_FILES`: every `results/pilot_summary_*.json`, every `results/pilot/**/*.jsonl` (step logs included) and `aborted/*-pilot/**/*.steps.jsonl`; a `verdicts` phase hashes `results/verdicts_<UTC>.json` and `.md`, `results/rescore_mini_<UTC>.json` and the mini's `results/scores/*/ifbench_*.jsonl`; `--verify-private` treats files outside `$EXP036_PRIVATE/exp037/` as expected extras.
- `tools/leak_check.py`, `tools/common.py`, `tools/hooks/pre-push`: exp_037's directory and messages; the hook checks exp_037 only and falls back to `$EXP037_VENV`; the leak check's 50 MB limit (`MAX_BYTES_RAW`) covers `results/raw/`, `results/pilot/` and `aborted/` (which holds moved raw outputs and pilot files), 5 MB elsewhere; the raw outputs' relaxed text rules (an address a model wrote is a warning) also cover the copies an abort moves to `aborted/<tag>/S*-*.jsonl` and `aborted/<UTC>-pilot/pilot/` (`RAW_ABORTED_RE`).
- `tools/power_log.py` (new; descriptive only, HYPOTHESIS "Power and energy (descriptive)"; Andrei's request 2026-10-06). Stdlib only. It reads the NUL-separated `-f plist` stream that Andrei's second terminal writes with exactly the mbp's pinned sudoers command (`/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist`, RUNBOOK "Power log", steps 9–12, 14 and 16), keeps only `is_delta`, `elapsed_ns`, `timestamp` and the CPU, GPU, ANE and combined power (or energy) fields, joins the samples to the windows the run records already carry (gate and phases, peer check, bench cells and the cool-down idle, pilot and arms, session start/stop pairs; `--window` adds one) and writes `results/power/power_<UTC>.json` (schema `exp037 power v1`, label "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power"). It refuses a raw path inside the repository, a raw log already recorded (unless `--again`), overlapping raw logs, a log with no readable sample and documents that count the same time twice (summed elapsed over the span by more than 2 % + 2 s). `--check` (readable and current; one JSON line) and `--layout` (key names and plist types only, for the pre-freeze smoke) write nothing. No host field (`hw_model`, `kern_*`) is written; the raw `.plist` is never committed (`.gitignore`, the leak check's NUL-byte rule). No gate, runner, bench, analysis, scorer, port, reference or tasks code reads a power record.
- Checked, unchanged: `tools/kv_bytes.py` agrees with mlx-lm 0.32.0's `make_prompt_cache` (byte for byte on three tiny Kolibri layouts); `tools/precision.py`'s probe passes under 0.32.3.

## 6. Unit tests (delta)

The commands are §3's, in both mini venvs, with every skip a failure. On the mbp (RUNBOOK step 3) `EXP036_REQUIRE_ALL=run-host` accepts `build-host only:` skips, among them every test of `tests/test_gate_ftiny.py`.

New test files: `test_gate_common_exp037.py`, `test_port_rope_vllm.py`, `test_gate_port_mutants_exp037.py`, `test_gate_forced_harness.py`, `test_gate_ref_drivers.py`, `test_gate_g0k.py`, `test_gate_preconditions.py`, `test_gate_rules.py`, `test_gate_g2_pairwise.py`, `test_gate_g4_forced.py`, `test_gate_g5_runner.py`, `test_gate_ftiny.py` (heavy; mini only), `test_gate_end_to_end_tiny.py` (heavy), `test_runner_allowed_b.py`, `test_runner_mlxlm032.py`, `test_analysis_tripwire.py`, `test_tools_refresh_builds.py`, `test_tools_power_log.py`, `test_tiny_real_layout.py`. Adapted: the owners' files listed in BUILD_LOG.md.

What the new tests pin (HYPOTHESIS Phase 0 and the G1 list):
1. **RoPE:** `_freqs` and `rope_inv_freq` reciprocal bitwise; rotation against the reference's `rope_neox` within 1e-6 relative L2 at every position to 20,479 and every 97th to 262,143 (per-octave maxima in `tests/fixtures/rope_angle_262143.json`); array offsets bitwise equal to integer offsets; the hook refactor output-neutral.
2. **Batching and mlx-lm 0.32.0:** `make_batch_generator`'s arguments at B = 1, 2, 4, 8, 16; `model_file_trust` on a tiny conversion, a directory without `model_file` and a stale port file; the step log against `stats()` totals; the private attributes the kit reads.
3. **ForcedTrunk** on the tiny real-layout build: forced onto its own ids in their own order bitwise equal to the free run at chunks 64, 513 and 2,048; sorted order within 1e-6 mean KL; misaligned ids and a wrong route-call count raise; ids of t − 1 give KL ≥ 0.10; mutants 6, 7 and 14 fail G4-F32.
4. **Reference drivers:** R2F-tiny forced onto its own natural ids bitwise equal to the natural run; the dump keys; writer paths under `exp037/`.
5. **G0k rule** on synthetic arrays, and the probe at every judged shape on the mini (no NAX path there: it tests the code, not the M5 kernel).
6. **New mutants:** each changes the tiny model only in its stated regime and overrides no `__call__`; mutant 27's decode mask is False exactly at padded slots on both batch cache kinds, before and after rotation, and a padded row's first generated token changes.
7. **Tiny G4-F32** on the sharpened validation build at ≥ 10× margin (`test_gate_ftiny.py`; mini only).
8. **G5-R1 and G5-BP-lean mechanics** on tiny: first wave and mid-run labels, `max_live`, the one construction, phase 6 sees both arms' sequences, every log-prob file exists with its sha256.
9. **`gate/rules.py`:** every rule at its boundary; the exit table; allowed_B; the controls and probes semantics; the tripwire.
10. **Freeze mechanics:** P1 refuses an unpushed pre-registration, an amendment line for a frozen scope and a direct table edit; cycle counting. Final review (W15-02, W15-03): an edited, skipped or added test, a changed `conftest.py` or an edited env script changes the `tests` or `env` scope, and the env scope opens only its three files; P1(d) refuses while a run directory has no record and when a changed re-run follows a blind crash no amendment's first line names; Ctrl-C, SystemExit, SIGTERM and SIGHUP during the phases write the record (exit 3, or exit 1 after a K8 FAIL), move an exit-3 run to `aborted/` and are re-raised; a signal during the record write is deferred; `close_orphans` records a killed run (`tests/test_gate_refusal.py`, `tests/test_gate_preconditions.py`, `tests/test_tools_hash_tree.py`).
11. **`tools/refresh_builds.py`** on a tiny build; **`env/exp037.settings.sh`** on a stub exp_036 tree; the seed sites; the version binding.
12. **The whole tiny gate** (`test_gate_end_to_end_tiny.py`, on the unsharpened seed-29 build): K8 PASS, K4 PASS, exit 0; every required control caught at its tiny margin, G5BP/27 by decision (e)'s acceptance with the fixture ceiling recorded; probe 22 reported. It writes the tiny control table when `EXP037_CONTROL_TABLE` names a path.
13. **The descriptive power log** (`test_tools_power_log.py`, on synthetic plist streams; powermetrics never runs on the mini): parsing (both NUL placements, skipped text, a truncated tail, unreadable documents, integer or real `elapsed_ns`, the energy and `gpu`-dict fallbacks); energy, split, mean and peak W, coverage, overlap and gaps at window edges; refusal of documents that count the same time twice, without false alarms from whole-second timestamps; the windows found in the run records; no host field written, `--layout` printing names and types only; `--check`; the CLI's refusals and its default raw log; the RUNBOOK's logger lines equal the pinned sudoers command; the label agrees with HYPOTHESIS and RUNBOOK; no deciding code names a power record.

## 7. Frozen configuration files

### 7.1 `gate/thresholds.json`

The file, verbatim (`tests/test_gate_thresholds.py` checks that the two are equal):

```json
{
  "version": "exp037-gate-1",
  "G0": {"tensor_count": 58353, "param_count": 78103074560,
         "full_attention_layers": [4,9,14,19,24,29,34,39,44,49],
         "template_conversations": 12, "template_settings": 10,
         "tokenizer_lines_min": 2000, "tokenizer_mismatches_max": 0,
         "router_logits_dtype": "float32", "head_output_dtype": "float32", "router_weight_dtype": "float32",
         "router_bf16_exact_frac_max": 0.01, "head_bf16_exact_frac_max": 0.01,
         "scales_biases_dtype": "bfloat16",
         "tensor_policy": {"quantised_head": {"embed_tokens": "arm_bits", "lm_head": "arm_bits"},
                           "vendor_faithful": {"embed_tokens": "bf16", "lm_head": "fp32"}}},
  "G1": {"source": "tests/INTEGRATION_LOG.md bounds (a), (b1), (b2) and cache entries",
         "mutant_fail_factor": 100, "sampler_vectors": 1000, "port_mutants": 15},
  "G2": {"fp32_forced_rel_err_median_max": 1e-4, "fp32_forced_rel_err_max_max": 1e-3,
         "fp32_selection_near_tie_gap": 1e-4, "head_fp32_logit_maxabs": 1e-3,
         "bf16_forced_median_factor_vs_emu": 3.0,
         "bf16_forced_p99_floor": 5e-2, "bf16_forced_p99_factor_vs_emu": 3.0,
         "bf16_selection_disagree_floor": 0.002, "bf16_selection_disagree_factor_vs_emu": 2.0,
         "bf16_selection_gap_sigma": 6.0,
         "pairwise_z_cap": 8.0, "pairwise_z_factor_vs_emu": 1.2,
         "router_mutant_block_if_diff_p99_below": 0.0,
         "g2q_embedding_rel_err_max": 4e-3, "g2q_head_rel_l2_max": 1e-3,
         "t9_buckets": [[0, 2048], [2048, 8192], [8192, 16384]],
         "real_weight_mutant_layers": [0, 3, 4, 49], "real_weight_mutants": [1,2,3,4,5,6,7,8,12,13,14,15],
         "descriptive": ["bf16_selection_gap_sigma"]},
  "G3": {"ref_bpb_per_text_max": 1.2, "ref_bpb_mean_vs_best_peer_max": 1.25,
         "ref_mutant_text": "T3", "ref_mutant_blocks": 48, "ref_mutant_dnll_quantile": 0.01,
         "ref_mutants": 7,
         "descriptive": ["ref_bpb_per_text_max", "ref_bpb_mean_vs_best_peer_max"]},
  "G4": {"K8_backstop_mean_kl_max": 0.10, "K8_backstop_decisive_lead_nats": 2.0,
         "K8_backstop_decisive_top1_min": 0.99, "K4": "descriptive",
         "descriptive": ["K8_backstop_decisive_top1_min"]},
  "G5": {"greedy_tokens": 256, "greedy_prompts": 8, "greedy_prompt_min_tokens": 600,
         "greedy_decisive_lead_nats": 2.0, "greedy_decisive_top1_min": 0.995, "greedy_ref_top5_min": 0.99,
         "noise_floor_prefill_steps": [2048, 64],
         "parity_kl_floor": 1e-4, "parity_kl_factor": 3.0,
         "parity_dis_factor": 3.0, "parity_dis_add": 0.002,
         "decode_positions": [[520, 1100], [15000, 15300]],
         "batch_B": 8, "batch_staggered_max_tokens": true,
         "behaviour_prompts": 20, "behaviour_effort_high_cap": 8192,
         "behaviour_block_count": 2, "loop_span_tokens": 32, "loop_repeats": 4,
         "eos_ids": [127906, 127901]},
  "P2": {"macos": "27.0", "macos_build": "26A428",
         "packages": {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"},
         "architecture_prefix": "applegpu_g17", "MLX_ENABLE_TF32": "0",
         "version_record_schema": "exp037 version record v1", "preflight_schema": "exp037 preflight v1",
         "sources": {"packages": {"path": "env/requirements-mbp.txt",
                                  "sha256": "32488f23411026229ef831ef9ec475fb922d10fd24d6be062ab08f7d4df533e6"},
                     "os": {"path": "../exp_036_kolibri_local_eval/results/version_record_20261004T063157Z.json",
                            "sha256": "27ca857435aca8bd8e7332dbc368e6da5ed107772f8fbb99f6934e154e4293e2"},
                     "architecture": {"path": "../exp_036_kolibri_local_eval/results/preflight_20261004T063132Z.json",
                                      "sha256": "43ce35db4a0aa1d4dab721624a842d66600356a38ae59b5a43978bad5dbaf6c5"}}},
  "P3_G0k": {"seed": 37, "experts": 384, "group_size": 64, "mode": "affine", "bits": [8, 4],
             "activation_dtype": "bfloat16",
             "projections": {"gate_up": {"n_out": 512, "d_in": 2560}, "down": {"n_out": 2560, "d_in": 512}},
             "judged_rows": {"decode_kolibri": [6, 12, 24, 48, 96],
                             "decode_peers": [8, 16, 32, 64, 128],
                             "prefill_chunk_kolibri": [12288],
                             "prefill_chunk_peers": [16384],
                             "batched_prefill_kolibri": [24576, 49152, 98304],
                             "first_wave_exp036": [52752],
                             "batched_prefill_peers": [32768, 65536, 131072],
                             "g2_single_call": [73728],
                             "b16_bound": [196608],
                             "boundary_mlx_repro_min": [16384, 24576, 30000, 32000, 32704, 32760, 32767, 32768, 32769,
                                                        32770, 32776, 32832, 33000, 34000, 40000, 49152, 65536]},
             "judged_rows_one_expert": [2048],
             "recorded_only": [],
             "rel_sorted_max_factor_vs_unsorted": 3.0, "bad_row_rel": 0.05, "bad_rows_max": 0,
             "repeat_bitwise": true,
             "lifted_from": {"mlx_repro.py": "9c98cbc4c6f971e0", "mlx_repro_min.py": "7004dcf44754c598"}},
  "P4": {"builds": {"K8": {"dir": "Kolibri-1-MLX-8bit-g64",
                           "convert_record": "../exp_036_kolibri_local_eval/results/convert/convert_8bit_20261004T133423Z.json",
                           "convert_record_sha256": "bd84bf3b7c4a3db891c02e41f86f618c1d93301581146fca8e598e04313e0e9d"},
                    "K4": {"dir": "Kolibri-1-MLX-4bit-g64",
                           "convert_record": "../exp_036_kolibri_local_eval/results/convert/convert_4bit_20261004T133455Z.json",
                           "convert_record_sha256": "e1ec02c764667c7873f49dedd00c353db804409c3592f913444c8599b5d7f8a2"}},
         "shards": "model*.safetensors", "config": "config.json", "port_check": "port.convert.check_port_file"},
  "P5": {"p5a_layers": [0, 4], "p5a_bitwise": true,
         "p5b_text": "T1", "p5b_bitwise": true,
         "p5c_text": "T1", "p5c_positions": [1, 1535], "p5c_id_shift": 1, "p5c_mean_kl_min": 0.10},
  "G4N": {"sets": ["T1-8", "T9"], "routing": "free", "prefill_chunk": 2048, "comparator": "R1",
          "kl_leg": "G4.K8_backstop_mean_kl_max",
          "descriptive_emu_decisive_exp036": {"set": "T9", "misses": 68, "n": 5943,
                                              "source": "exp_036 diagnostics/gate1/out/stage3_g4_20261005T124342Z.json rule_inputs"}},
  "G4F32": {"sets": ["T1-8", "T9"], "tau_csort_factor": 100.0, "f_tiny_cap": 1e-5, "tau_cap": 1e-4,
            "csort_mean_ceiling": 1e-6,
            "f1_decisive_lead_nats": 2.0, "f3_rho_factor": 10.0, "f3_csort_factor": 10.0,
            "f5_position_kl_max": 1e-2, "f6_shadow_gap_min": 1e-2,
            "descriptive_f4_window": 32},
  "G4F16": {"sets": ["T1-8", "T9"], "kappa": 9.0, "r3_mean_kl_ceiling": 1e-2,
            "descriptive_decisive_lead_nats": 2.0,
            "descriptive_spike_port_kl_min": 1.0, "descriptive_spike_r3_kl_max": 0.1},
  "G5D32": {"ranges": ["T1", "T3", "T9"], "chunk": 2048, "chunk_small": 64, "decisive_lead_nats": 2.0,
            "mean_factor_vs_csort": 100.0, "mean_floor": 1e-8,
            "max_factor_vs_csort": 100.0, "max_floor": 1e-6,
            "csort_mean_ceiling": 1e-6, "csort_max_ceiling": 1e-4},
  "G5R1": {"B": 1, "max_tokens": 256, "prompts": 8, "prefill_step_size": 2048, "anchor_factor": 3.0,
           "legs": {"K8": ["R-anchor", "R-greedy", "R-parity"], "K4": ["R-anchor"]}},
  "G5BP": {"B": [8, 16], "prefill_batch_size_max": 8, "anchor_factor": 3.0,
           "subsets": ["first_wave", "mid_run"], "admitted_mid_run_min": 1,
           "allowed_B_always": [1], "allowed_B_if_pass_8": [2, 4, 8], "allowed_B_if_pass_8_and_16": [16],
           "layout": {"text_order": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
                      "lengths": [37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100],
                      "max_tokens": [48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44],
                      "b_text_offset": 4,
                      "L1": {"t9_ids": [0, 2100], "max_tokens": 24},
                      "L2": {"t9_ids": [4096, 7096], "max_tokens": 40},
                      "queue": ["A0-A11", "L1", "L2", "B0-B11"],
                      "prompt_tokens": 16916, "generated_positions": 792},
           "tiny_layout": {"L1": {"t9_ids": [0, 100], "max_tokens": 6},
                           "L2": {"t9_ids": [128, 300], "max_tokens": 10}}},
  "K4_blocking": ["G0", "same_port_sha", "G2q", "G5_behaviour", "G5_R1_anchor"],
  "fix_cycles_max": 2
}
```

### 7.2 `runner/plan_rules.json`

Unchanged from exp_036 (version `exp036-plan-2`), byte for byte.

## 8. Assets and build-time fetches

None new. The mbp holds exp_036's downloads (assets.json, ASSETS.md) and the Gemma 4 bf16 folder of exp_036's Amendment 6 record, which RUNBOOK step 9a reuses. The mini's build-time data are exp_036's, under `~/models/exp036-mini/`. The vendor-code spike's downloads (aleph-alpha-inference @ `049a6a7`, a CPU torch wheel, vLLM 0.29.0 and its dependencies; decision E-downloads) stayed in a scratch folder on the mini and are in no venv of the kit and not in the repo.
