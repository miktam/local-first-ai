# Experiment 037 — build log of the kit

*The kit was built on the mini on 2026-10-06 by build tasks W0–W15 and the power-log tasks (PowerLog), then tidied by the main session before the freeze, under the main session's build specification (`DESIGN.md`; a scratch document, not part of the kit, whose content is carried by [`HYPOTHESIS.md`](./HYPOTHESIS.md) and [`BUILD_SPEC.md`](./BUILD_SPEC.md)). Nothing here is a result and nothing is committed by the build. Every value measured during the build is a tiny-checkpoint or kit value, scratch unless it sits in a committed record with its sha256, and is never evidence about Kolibri. HYPOTHESIS.md wins over this file; BUILD_SPEC.md is the kit's contract; RUNBOOK.md is how the mbp runs it.*

**Environment of every run below:** the exp_037 venvs on the mini, `~/models/exp037-mini/venv312` (Python 3.12.13) and `venv314` (Python 3.14.7), MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0 (decision F2); `EXP036_MODELS=~/models/exp036-mini`, `EXP036_DATA=~/models/exp036-mini/data`, `EXP036_TOK=~/models/exp036-mini/kolibri/Kolibri-1-BF16`, `EXP036_REQUIRE_ALL=1` (every skip a failure), `PYTHONDONTWRITEBYTECODE=1`, `pytest -p no:cacheprovider`. exp_036's directory was never written to, and its env file was never opened.

---

## State before the freeze

| Stage | Task | Status | Outcome |
|---|---|---|---|
| before 0 | W0r release-notes review (main session) | done | decision F = F2 (Annex A) |
| spec | SpecF2 | done | the specification revised for F2 |
| 0 | W0 directory and copy | done | 187 files copied unchanged, 82 as originals to modify; F2 baseline 279 failing ids |
| 0 | W0b tiny real-layout fixtures, mlx-lm 0.32.0 load path | done | 201 baseline ids green |
| 0 | W0c gate common helpers | done | seeds, work37_dir, builds_dir |
| 1 | W1 port | partial → resolved by FixC | VllmRoPE, hooks, exp037-port-1 |
| 1 | W2 mutants | done (27 added by FixB) | 19, 20, 22, 26 (+ 27) |
| 1 | W3 forced harness, reference drivers | done | ForcedTrunk, R2F/R3 |
| 1 | W4 G0k, preconditions | done | P1–P4 |
| 1 | W6 rules, thresholds, controls | done | `gate/rules.py`, `controls.json`, `exp037-gate-1` |
| 1 | W8a runner | partial → resolved by FixC / Integrate | make_batch_generator, allowed_B, version binding |
| 1 | W8b bench | done | shared construction, allowed_B, C1 NOT RUN |
| 1 | W9 analysis | done | H2 tripwire, seeds, labels |
| 1 | W10 tools | partial → resolved by FixC | gate_rules scope, refresh_builds, schemas |
| — | W13 vendor-code spike (scratch) | done | no wiring mismatch (E-verdict) |
| 2 | W5a G2, G3, emulation statistics | done | per-pair z, leg (p) inputs |
| 2 | W5b G4-N, G4-F32, G4-F16 | done | forced series, control loop |
| 2 | W5c G5 family | done | G5-D32, G5-R1, G5-BP-lean |
| 2 | W5d F_tiny | done | F_tiny 4.112e-7, cap met, validation met |
| 3 | W7 orchestration | blocked → decisions (a)–(d) | the whole tiny gate on the sharpened build failed |
| 3b | SpecFix, FixA, FixB, FixC | done / partial / blocked / partial | FixB → decision (e) |
| 3b | Integrate (decision (e)) | partial → completed in W14 | e2e passes in both Pythons |
| 4 | W14 both Pythons, end to end, dry run | done (second attempt) | 1,894 passed; the 5 failures wait on W12 |
| 5 | W12 documents | done | see "Stage 5" |
| 6 | W15 final review; W15-fix | done; W15-04, W15-02/03 and W15-R3 approved by Andrei at 2026-10-06T18:41:39Z | 1,914 passed in both Pythons; see "Stage 6" |
| 7 | PowerLog build, review, fixes | done; the mbp smoke run (2 samples), layout as the fixture | 1,955 passed in both Pythons; see "Stage 7" |
| — | pre-freeze housekeeping (main session) | done | COPY_RECORD.json complete; see "Pre-freeze housekeeping" |

## The specification and its revisions

- 2026-10-05: a draft design after D8, checked by three reviews, then a decision brief. 2026-10-06T02:30:15Z: decisions A–E, "as recommended" (the lean scope). 02:48:30Z: decision F = F2, after BUGHUNT §6.9's resolution (02:47:53Z). The four reviews' issues were all confirmed and resolved in the specification (none rejected outright).
- **SpecF2** (sha256 after: `aa330057…`, 1,931 lines): the specification revised for F2 from the release-notes review. Judgement calls: G2 returns to its registered single call (73,728 rows, judged by G0k); 196,608 rows is judged as headroom; line 2 of the requirements names `$EXP037_VENV`. Found during verification: `port/convert.py`'s two `model_file` loads need `trust_remote_code` (hence `model_file_trust`); `tools/peer_check._load` needs it for the dry run's Kolibri-architecture stand-ins; the bench does use a logits processor (`EosMask`), which ignores the token history; `_generation_batch`, `_prompt_batch`, `_unprocessed_sequences`, `BatchKVCache._idx` and `BatchRotatingKVCache._offset` still exist in 0.32.0 (only the two counters are gone).
- **SpecFix** (`36ad3fdd…` → `076ab088…`, 1,951 → 2,143 lines): decisions (a)–(d) recorded, with a §14.2 entry and every section they change; the probe-22-in-tiny-mode-only and Gemma-record-by-rule judgement calls; W14's stress records added to its task list.
- **Integrate** (`076ab088…` → `2f8e9ab4…`, 2,209 lines): decision (e) and FixB's first-token correction recorded.
- **W14** updated §11.13 with the committed stress records. At W12 the specification's sha256 is `039d436d458a7514dfbc8e67fa6c5766302696c26a3150653c8cf8a930c829f2` (2,213 lines).

## W0r — the release-notes review and the runtime pin (main session)

Done before stage 0, under the rule "check the latest release and its notes before pinning" (BUGHUNT §6.9's lesson). The review is reproduced verbatim as Annex A below and in HYPOTHESIS.md. Decision F = F2 at 2026-10-06T02:48:30Z ("let's update mlx on both machines"). The exp_037 venvs were created beside exp_036's on both machines (mbp `~/models/exp037/venv`, checked as `mlx 0.32.3 | mlx-lm 0.32.0 | applegpu_g17s`; mini `venv312`, `venv314`). A scratch probe of exp_036's suite under 0.32.3 / 0.32.0 (not evidence): unmodified 101 failed, 991 passed, 148 errors; with `trust_remote_code` defaulted by a probe hook, 28 failed and 1,212 passed (2 hook artefacts, 26 in the runner and batching).

---

## Stage 0

### W0 — directory setup and copy (done)
- E37 created from exp_036 at `222c8450f5c1b87663ee0386c61a6eb86c353543` (clean tree) by `cp -p` of the git-tracked files: 187 unchanged (§2.2 of the specification), 82 pre-modification originals (§2.3), 325 exp_036 files not copied. exp_036's env file was never opened, hashed or copied.
- W0 wrote `.gitignore` (`env/exp036.local.env` → `env/*.local.env`), `env/requirements-mbp.txt` (the reviewed scratch file with only line 2 changed to `$EXP037_VENV`; sha256 `32488f23411026229ef831ef9ec475fb922d10fd24d6be062ab08f7d4df533e6`), `env/versions.json` (mirror; it keeps the schema string "exp036 versions v1", which no code checks), `env/setup.sh` (`$EXP037_VENV` the only venv override; exp_036's venv cannot be selected), `env/exp037.settings.sh` (21 lines), `results/README.md`, the `.gitkeep` files and `COPY_RECORD.json` (per-file owner and status; top-level counts, not-copied list, the referenced-in-place sha256 values, E36 HEAD, the scratch requirements sha256 `f5aa6fd4…`).
- Script check (both venvs): 25/25 ok; `hash_tree --tree reference` = `85337ed7…`, `gate_text` = `1a295dac…`, `requirements` = `32488f23…`; the requirements' package lines equal the scratch file's; `bash -n env/setup.sh` passes. The settings file on a stub tree, bash and zsh: 14/14 ok.
- **The F2 baseline:** the copied suite gives 123 failed, 995 passed, 156 errors of 1,274 in both venvs, the same 279 failing ids. Causes: 249 from mlx-lm #1385 (`model_file` without `trust_remote_code`); 20 from the exp_037 documents not existing yet; 4 because exp_036's env file is absent by design; 3 from the F2 pins and the `.gitignore` change; 3 in `test_tools_peer_check_reference.py` (the Amendment 6 record not copied). The probe's two "does not import mlx" failures pass here (hook artefacts).
- Raised for the main session: the Amendment 6 fidelity pin points at a record not copied (resolved by decision (c)); six unowned failing test files load through W0b's files (all turned green by W0b).

### W0b — tiny real-layout fixtures and the mlx-lm 0.32.0 load path (done)
- `tests/tiny_real_layout.py`: `write_real_layout(out, seed, sharpen=3.0)`, lifted from exp_036's bug-hunt `build_ck.py`. At seed 23 it reproduces exp_036's `ckv_s3` byte for byte (config `3ee351c7`, shards `52b676b4`, `ed2f74e5`; pinned and cross-checked against exp_036's mini MANIFEST). Session fixtures `tiny_real_cal` (seed 23) and `tiny_real_val` (seed 29), each BF16 plus 8-bit and 4-bit conversions, about 1 s each.
- `port/convert.py`: `model_file_trust(dir)`; `trust_remote_code=True` at `mlx_convert` and `refresh_port_file`'s load; `check_source()` refuses a source with an `auto_map` (the flag also reaches transformers' tokenizer loading); a second modified-by line. `tests/port_harness.py` and `tests/test_tiny_checkpoint.py` pass the flag.
- Tests: 148 passed in both venvs (test_convert 120, test_tiny_checkpoint 12, test_tiny_real_layout 10, test_notice 6). Full suite 70 failed, 1,240 passed, 8 errors; 201 baseline ids green, none new.

### W0c — gate common helpers (done)
- `gate/common.py`: `seed_from(*parts, prefix="exp037")`; `work37_dir()`; `builds_dir()`; `PORT_MODULE_PREFIX = "exp037_gate_port_"`; `load_port` passes `trust_remote_code=True` (inert: `model_config={"model_file": None}` builds from the kit's module). The three registered fixture seeds pass `prefix="exp036"` with the comment "registered fixture seed, not an exp_037 draw"; the tiny gate texts' keys and the G0 digits digest equal exp_036's.
- Tests: `tests/test_gate_common_exp037.py` 9 passed in both venvs; 11 mutations each caught. Full suite: the same 78 failing ids as W0b's run.

## Stage 1

### W1 — port (partial; the open item was resolved by FixC)
- `port/kolibri1.py` (sha256 at the time `2c153357862f15b61f3cadaea4f436de2939567180bf5a01a304f3fa60e3f182`): VllmRoPE verbatim from exp_036's `kit_rope` copy l.249–266 (checked by diff); `rope_offset` and `attn_mask` hooks; `SPEC_VERSION` `exp037-port-1`; the header line (dated 2026-10-06) and the 0.32.0 docstring; `port/README.md`.
- RoPE acceptance met without changing `_freqs`: `float32(1/_freqs)` equals `rope_inv_freq` bitwise (head_dim 128 and 32); maximum relative L2 against `rope_neox` 1.056e-7 over 22,972 positions to 262,143 (worst octave [4,096, 8,192)), 9.5× inside 1e-6. Power check (scratch): exp_036's `nn.RoPE(base=)` gives 3.06e-3 at 262,143, so the test catches the old angle.
- `tests/test_port_rope_vllm.py` 12 passed in both venvs (also with `-W error`); `tests/fixtures/rope_angle_262143.json` written once on venv312 with `EXP037_WRITE_ROPE_FIXTURE=1`.
- Open: `tests/test_port_ref_hooks.py:35` still asserted `exp036-port-2` (not W1's file). Fixed by FixC.

### W2 — mutants (done)
- Mutants 19, 20, 22 and 26 as subclasses of the port's `Attention` overriding only `rope_offset` (19, 22, 26) or `attn_mask` (20); `MUTANT_IDS`, `TARGETS` (`g4_f32_F2_T9`, `g5_d32_decode`, `g5_bp_lean_d_K8_B8`, `g5_r1_R_parity_K8`), `PATCHES`, `HOOK_MUTANTS`; batch caches recognised by `left_padding`'s presence; mutant 26 freezes a copy of the offset (a test shows it survives an in-place `+=`); mutant 20 acts only on single-sequence rotating caches, and a step with ≤ 256 keys is bitwise the port's.
- Regimes verified bitwise on the tiny w513 checkpoint, by hand-driven batch caches and through BatchGenerator: 19 only prefill chunks after the first; 20 only decode steps with ≥ 257 keys; 22 a no-op at B = 1 (600 and 1,100 tokens) and changes only the shorter row at B = 2; 26 batch-cache decode from the second step on, B = 1 included. Mutants 12 and 13 still run with VllmRoPE.
- Tests: 68 passed in both venvs (31 adapted, 37 new).

### W3 — forced harness and reference drivers (done)
- `gate/harness.py`: `PortLayers.run`, `forcing_crosscheck`, `stream_logits`, `logits_at` byte-for-byte unchanged; added `ForcedTrunk` (modes free / force / force_own_order; "force" hands the ids ascending), `forced_stream`, `forced_logits`, `records_table`, `decode_spans`, `p5b_exactness`, `p5c_forcing_active`. `gate/checks/ref_drivers.py`: `forced_ref_pass`, `i_from_r1`, the forced-dump key over five components, `ForcedDump`, `r1_rows_pass`, `p5a`, `require_under_work37`.
- Tiny measurements (seed 23; scratch): P5b bitwise at chunks 64, 513 and 2,048 (fp32) and 64 and 2,048 (bf16); Csort mean 1.49e-9, max 5.79e-9; P5c 0.531 against 0.10; R2F-tiny and R3-tiny forced onto their own natural ids bitwise equal to the natural runs; P5a bitwise, with a one-ulp change caught.
- Tests: 34 passed in both venvs.

### W4 — G0k and preconditions (done)
- `gate/checks/g0k_kernels.py` and `gate/preconditions.py` (P1 (a)–(e), P2 with the newest exp037 version and preflight records, P3 as a wrapper of `rules.p3_g0k`, P4, `run_counts`, `code_hashes`).
- G0k on the M4 at every judged shape (35 row counts to 196,608; gate/up and down; 8 and 4 bits; 148 measurements over 140 judged keys): 0 bad rows, every repeat bitwise, worst sorted/unsorted ratio 1.408 (the same 1.41 the M5 showed for the 52,752-row first wave under 0.32.3); about 94 s, 13 GB. The M4 has no NAX path: this tests the code, not the M5 kernel.
- Tests: 57 passed in both venvs; 6 mutations of `preconditions.py` each caught (one equivalent mutation survived).

### W6 — rules, thresholds, controls (done)
- `gate/rules.py` (every decision; NaN never passes; imports no mlx, port or writer), `gate/controls.json`, `gate/thresholds.json` → `exp037-gate-1` (registered blocks value for value; `bf16_selection_gap_sigma` kept under its exp_036 name and listed as descriptive; the G4 block kept; new P2, P3_G0k, P4, P5, G4N, G4F32, G4F16, G5D32, G5R1, G5BP blocks; P2's sources by sha256).
- Check states add MISSING (a crash or a missing blocking value; exit 3); the arm order FAIL > MISSING > INCOMPLETE > PASS; G4-F16's VOID takes precedence over κ. New check ids `g4_n_K8`, `g4_f32_K8`, `g4_f16_K8`, `g5_d32_K8`, `g5_r1_K8`, `g5_r1_K4`, `g5_bp_<arm>_B<8|16>` (non-blocking, role allowed_B); `g4_K8_backstop` and `g5_batch_parity_K8` retired.
- The document assertions of `tests/test_gate_thresholds.py` warn instead of failing while the documents are missing (`DOC_ASSERTIONS_ON = False`); they run as soon as HYPOTHESIS.md's gate table and BUILD_SPEC §7.1 exist.
- Tests: 74 passed in both venvs; 8 baseline ids green.

### W8a — runner (partial; the open items were resolved by FixC and decision (c))
- `make_batch_generator`; the step log through `gen.stats()` windows; `run_cell(..., allowed_B=)`; loads through `model_file_trust`; `require_environment` at pilot, session and cell; private records under `$EXP036_PRIVATE/exp037/`; `plan_fix.B_for` clipping; `guard.require_gate` returns allowed_B (none in the record gives [1]; a malformed set is refused); `P2_PINS`; `SEED_PREFIX = "exp037"`; `runner/common.builds_dir()`.
- W0r items 4 and 6 confirmed: `runner/` imports nothing from `mlx_lm.sample_utils`, and `run_cell` passes no logits processor.
- Tests: owned 179 passed and 2 failed in both venvs (the two plan-table tests that read HYPOTHESIS.md, which W12 writes); 31 of W8a's 33 baseline ids green; the drivers test's two `_prompt_tokens_counter` failures green as a side effect.
- Open, both resolved: `tests/test_runner_jsonl.py:107`'s private path (FixC), and the scorer's single `--private` (decision (c); FixC and Integrate).

### W8b — bench (done)
- `load_arm` through `model_file_trust`; `_batch_generator` → `make_batch_generator`; `EosMask` independent of the token history (#1777, tested); `batch_aggregate` on `stats()` (#1829, tested; it records `wall_time_s`, `prompt_time_s`); `builds_dir`, `allowed_B`, `clip_B`, `require_environment`; C1 at the largest allowed B or NOT RUN (a complete file); speed_desc Kolibri only at allowed B; `run_bench.run` returns the binding too.
- Tests: 143 passed in both venvs; all 20 bench baseline ids green.

### W9 — analysis (done)
- `stats.py` `b"exp037|"`; `margins.json` `exp037-margins-1` with the tripwire keys (equal to `rules.tripwire_constants()`); `verdicts.py` `exp037-verdicts-1`, the tripwire, the tripped Q2 rows, the caveat, the labels, "M5 Max, MLX 0.32.3"; `power.py`'s tripwire and greedy tables; `tables.py` renders the tripped rows (beyond "labels only", so the tripped states do not fall through to the INCONCLUSIVE sentence).
- Tests: 223 passed in both venvs (105 verdicts, 26 stats, 45 tripwire, 33 power, 11 exploratory, 3 world).
- Notes: the greedy table's 0.19 is computed at the rounded 0.32 %; at the exact bound 0.318 % it is 0.183 (both reported). `power.budget()` keeps exp_036's S1 hours (5.4 / 9.1 / 9.6), which `tests/test_analysis_power.py` pins. The tripwire predicate exists twice: `verdicts.py` compares with its 1e-9 tolerance, `rules.py` with a bare `<`; they agree except for a bound within 1e-9 of −0.10 (tested on a grid); status words differ by spelling only.

### W10 — tools (partial; the open items were resolved by FixC and decision (c))
- `hash_tree.py`: the `gate_rules` scope (18 scopes), `NO_AMENDMENT_SCOPES`, `--amend-line` refusal, a missing HYPOTHESIS.md reported as unfilled. `refresh_builds.py` (new; refusals on unsafe targets, inode check, stat fingerprints). `preflight.py`, `version_record.py` (schemas exp037, `$EXP037_VENV`, `mx_device_info`). `peer_check._load` through `model_file_trust`. `status.py` (`refresh` phase; private extras). `dry_run.py` (the new gate `--tiny`). The pre-push hook. `tools/common.exp037_venv()`.
- Tests: 174 passed (+3 W12-pending) in both venvs; 19 baseline ids green.
- **Checks only, no code change:** `tools/kv_bytes.py` agrees with mlx-lm 0.32.0's `make_prompt_cache` (0.31.3 and 0.32.0 identical there; growing and fixed bytes equal exactly on three tiny Kolibri layouts, e.g. w513 4,096 B/token and 8,404,992 B); `tools/precision.py`'s probe passes under 0.32.3 (rel L2 2.9e-7 on the M4), and libmlx 0.32.3 still reads `MLX_ENABLE_TF32`.
- Open, both resolved: `tests/test_tools_leak_check.py:305`'s hook fixture path (FixC) and the Amendment 6 pin (decision (c)). Observation: mlx-lm 0.32.0's qwen3_5 sanitize shifts norm weights only for unsanitized conv1d (0.31.3 also when `mtp.*` keys were present); the S1 peer check re-measures the peers either way.
- Disclosed: once, read-only, exp_036's mini venv was asked for its mlx-lm version (0.31.3).

## W13 — the vendor-code spike (scratch; decision E)

Run 2026-10-06T04:07:20–04:18:22Z on the mini, after E-downloads (04:06:38Z), about 13 minutes of the 2 h box. Route A (vLLM 0.29.0 built from source for the CPU, the vendor's `Kolibri1ForCausalLM` unmodified, torch 2.13.0 CPU, fp32) installed and imported in 3 min 13 s. Results: vendor and w513 presets ≤ 1.1e-6 relative L2 at every layer and the logits; real-layout seeds 23 and 29 clean at 64 tokens (4.9e-5, 4.4e-5) and over the literal 1e-4 at 600 tokens (2.2e-2, 1.7e-4), localised to routing chaos (teacher-forced layers ≤ 1.94e-6; the reference against itself diverges the same way); the discrimination control sees every reference mutant and a ±1 window. E-verdict (05:17:51Z): no wiring mismatch, with the disclosure. The README, scripts and small outputs are in `diagnostics/vendor_spike/` (copied by W12, below). An observation of SpecF2's: the spike's scratch folder appeared at 06:19 local (04:19Z), after E-downloads.

## Stage 2

### W5a — G2, G3, emulation statistics (done)
- `ref_pass.natural_vs_r1`: the one per-pair z computation (float64), for the port and the emulation; `emu_stats_exp037.json` with σ_emu, z_emu, M_emu and the flagged-pair flags; the flagged-pair report (a stand-in at the last layer on tiny checkpoints); real-weight mutant records with fp32 branch statistics; `evaluate` functions removed; `layer_csv_rows(res, emu, row_verdicts)`; `fp32_dump` writes only under `work37_dir()`.
- Tests: `tests/test_gate_g2_pairwise.py` 14 passed in both venvs.
- Finding: leg (p) fails the unmutated port on exp_036's pattern5 tiny build (M_port 4.496 > 1.2 × 3.255 = 3.906); on the real-layout builds it passes (seed 23 4.468 ≤ 5.621; seed 29 5.404 ≤ 5.869; scratch).

### W5b — G4-N, G4-F32, G4-F16 (done)
- `g4_e2e.py` measurement only; `g4_forced.py` (new): `fp32_on_dequantised_weights` (verbatim; a test compares the source), `make_fp32_port`, `reference_profile`, `forced_series`, `forced_texts`, `csort_series`, `csort_sequence`, the measured dicts, F4 windows, the control mutant loop.
- Tests: 21 passed in both venvs.
- Tiny (seed 23; scratch): G4-F32 PASS with every control at the 10× margin (mutant 1 F6 count 64; mutant 6 F2 about 4.9e4 × τ; mutant 19 F2 on T9 about 5.8e4 × τ; mutants 7 and 14 also fire); G4-F16 VOID on that sharpened build (R3 0.044 / 0.120), with control 6 at κ ratio 10.2 against a margin of 90. Raised for the main session; the build question became decision (a).

### W5c — the G5 family (done)
- `g5_generation.py` measurement only on the one construction, with `run_admission`; `g5_runner.py` (new): `g5_d32`, `g5_r1`, `g5_bp`, `prompts26`, `anchor_sequences`, `reduce_with_r1`, `greedy_block`, the log-prob file helpers. `g5_generation` still re-exports `parity_bound` and `behaviour_verdict` from `rules.py` for callers that have not switched.
- Tests: 18 passed in both venvs; 2 baseline ids green.
- Tiny findings (scratch): G5-D32's mean legs fire on the correct port on both sharpened builds (decode and chunk-64 means 11–17× Csort's); mutant 22 not caught at B = 8 ((d) 1.76 / 1.70); mutant 26 caught at margin 5,476.

### W5d — F_tiny calibration and validation (done)
- `gate/checks/ftiny.py`, `gate/calibration.json`, `tests/test_gate_ftiny.py` (heavy; excluded from the gate's G1).
- **Calibration** (seed 23, token ids `rng(5)`, 16,384 positions, 2,048-token blocks): block means 2.06e-8 to 4.11e-8 (largest: positions [10,240, 12,288)); **F_tiny = 4.1119e-7**, 24× under the 1e-5 cap.
- **Validation** (seed 29, `rng(6)`, T1–8 8 × 1,536 and T9 16,384): Csort means 1.42e-9 (T1–8) and 2.81e-9 (T9), so τ = F_tiny in both sets; `rules.g4_f32` PASS; F1, F3, F5, F6 silent; F2 headroom under τ / 10: T1–8 2.60×, T9 1.18×, buckets 2.15×, 1.18×, 1.06×. Controls: mutant 1 F6 count 64; mutant 6 F2 1.19e6 × τ; mutant 19 F2 on T9 9.06e5 × τ.
- Tests: 14 passed in both venvs (Python 3.14 reproduced the validation values bitwise).
- Flags: the validation margin is thin by construction (bucket 8,192–16,384 6 % under τ / 10), so the test belongs on the mini (decision (c)); `calibration.json` records code hashes and must be re-written after any change to the files it hashes (done in W14).

## Stage 3: W7 — orchestration (blocked → decisions (a)–(d))

- `gate/run_gate.py` runs phases 0, 1, 2a, 2b, 2c, 3, 4, 5, 6 (subprocesses) and 7 (in process); preconditions give exit 3; the exit table and record fields; `require_pass` returns allowed_B and refuses a record of another experiment; every exit-3 run moves its phase directory to `aborted/<UTC>-gate/` with a NOTE.md; `g1_synthetic.EXCLUDE` with a G1 test; non-finite floats stored as `{"$nonfinite": "nan"}`; tiny-only options.
- **Tiny-mode choices** (open to the final review): the free-routing checks use fp32 activations in tiny mode, as exp_036's tiny run did (in bf16 the sharpened build is pure noise: noise-floor KL 0.36, top-1 disagreement 0.97); `TINY_NOT_APPLICABLE` keeps exp_036's exemptions (G3, greedy, behaviour) and adds G5-R1 K8's R-greedy leg (in_top5 0.05 sharpened, 0.93 unsharpened); phase 4 runs the unmutated K8 first and all control mutants after it; P5 always runs on the unmutated port.
- Tests: full suite 11 failed, 1,845 passed, 3 errors in both venvs (14 of the 279 baseline ids still failing). W7's own files: 78 passed, 3 failed, all three from the tiny data.
- **The blocker.** On the sharpened seed-29 validation build the whole tiny gate ran cleanly and failed: G4-N(i) 0.378 / 0.393 against 0.10; G4-F16 VOID (R3 0.127 / 0.158), control 6 at κ 3.80; G5-D32's mean rule on the unmutated port at 1.14–1.63× its bound; control 22 at (d) 1.76. The drivers' port run on pattern5 failed G2 leg (p) and left G5-D32 INCOMPLETE (mutant 20 cannot act at window 17). The same seed-29 build unsharpened passed everything except mutant 22's margin (5.75 against 30).

## Decisions (a)–(d) (main session, after stage 3) and SpecFix

(a) the unsharpened seed-29 build validates the whole tiny gate; (b) mutant 27 replaces mutant 22 as G5-BP-lean's required control, and 22 becomes a tiny-mode probe; (c) the integration fixes (three test literals, the scorer's private layout, the Gemma fidelity record produced in S1, `test_gate_ftiny.py` on the mini only); (d) a flag for the final review on G4-F16's R3 ceiling. Recorded by SpecFix in the specification and in HYPOTHESIS.md's decisions table.

## Stage 3b

### FixA — decision (a) (partial; completed by Integrate)
- `tests/tiny_real_layout.py`: `SHARPEN_GATE = 1.0`, `build_gate_validation_set(root)`; `tests/conftest.py`: `tiny_real_val_unsharp`; the e2e test, the drivers' port run and the dry run's gate stage on it (the dry run keeps the directory name `tiny_real_val`).
- A new drivers fixture `pattern5_baseline`: each of the 15 port-mutant tests first asserts that its target check passes on the unmutated port, and `test_pattern5_baseline_fails_only_where_documented` pins the documented leg-(p) failure, so the mutant tests cannot pass with no mutant at all.
- Tests: e2e 4 passed, 1 failed (G5BP/27 not yet run by `run_gate.py`) in both venvs; drivers 32 passed in both venvs. The unsharpened build's shards are byte-identical to W7's measured build.

### FixB — decision (b) (blocked → decision (e))
- Mutant 27 `batch_decode_pad_keys_visible` in `gate/port_mutants.py` (it calls `super().attn_mask()` first, so a port fix reaches it), `controls.json` (G5BP/27 required; `probes` G5BP/22, tiny only), `tests/test_gate_port_mutants_exp037.py` (52 passed in both venvs): bitwise a no-op at B = 1, with equal lengths, on single caches and in prefill; at B = 2 the padded row moves at every decode step, its first token included; a structure pin through mlx-lm 0.32.0's own BatchGenerator (133 padded decode steps, 16 extends, 32 filters, 100 rotated steps) shows the mask False exactly at padded slots.
- **Measured on the unsharpened seed-29 build** (identical in 3.12.13 and 3.14.7): G5BP/27 caught on both subsets, (d) 11.93 (first wave, n 65) and 6.41 (mid-run, n 155), against the 30× margin; the unmutated K8 at B = 8 1.0000001; probe 22 5.701 / 5.746. Cause: the build's near-uniform next-token distributions (entropy 6.611 of 6.931 nats; KL between unrelated contexts mean 0.607, median 0.620, p90 0.699, max 0.851 over 4,000 pairs), so a full context swap would reach only about 22.3 / 19.1.
- **Correction found:** under mlx-lm 0.32.0 `GenerationBatch` runs its first step at construction as an `L == 1` step over the batch caches, so a padded row's first generated token is affected; the specification's "first token from prefill, unaffected" was wrong.

### FixC — decision (c) (partial; the two pending patches were applied by Integrate)
- `tests/test_port_ref_hooks.py` (`exp037-port-1`), `tests/test_runner_jsonl.py` (`exp037/raw/…`; torn lines under `exp037/aborted`), `tests/test_tools_leak_check.py` (the hook fixture path), `scorers/score_all.py` (the split private layout; `PRIVATE_SUBDIR`; a modified-by line), `diagnostics/gemma_quant_check.py` (byte-identical, `9393765d…`), `tools/fidelity_reference.json` (schema `exp037 fidelity reference v1`, the record by rule, `record_rule` {mlx 0.32.3, device Apple M5 Max}), `tools/peer_check.py` (`load_fidelity_reference(before=)`, `_reference_for_run(t_start=)`, the record's UTC in the peer record), `tests/test_tools_peer_check_reference.py` (the step-9 re-attribution test removed: it read an MLX 0.31.2 record), `tests/test_gate_ftiny.py` (per-test `build-host only:` skips; a module-level skip under `EXP036_REQUIRE_ALL=1` would abort the session).
- Tests: the owned heavy set 101 passed in both venvs. The copied producer's part A (`--skip-weights`) runs on the mini in about 2 s (worst rel_err 1.65e-6, Apple M4 Pro, mlx 0.32.3).

## Decision (e) and the integration (Integrate, 10:39–11:28Z)

- Decision (e) (main session, after FixB): G5BP/27 stays the required control; its tiny acceptance replaces the 30× margin with (d) caught on both subsets at ratio > 3 on the unsharpened seed-29 build, with the fixture ceiling recorded beside it. `gate/controls.json` and `gate/rules.py` are unchanged by it.
- `gate/run_gate.py`: phase 4 runs mutant 27 at K8 B = 8 in both modes and probe 22 in tiny mode only (a second mutant loop whose crash is recorded as `probes_error` and cannot change the exit code); the G5-R1 and G5-BP measures are tagged by the mutant actually loaded (which also fixed G5-R1's hard-coded 26); phase 6 anchors both; the record writes `probes` (empty in real mode). `gate/checks/g5_runner.py` docstrings. `tests/test_gate_end_to_end_tiny.py`: `DECISION_E`, `decision_e_acceptance`, `fixture_ceiling` (FixB's estimator: 4,000 pairs, seed 0, the unmutated K8 B8 `single` files in path order, `common.kl_rows`, each file sha256-checked). FixC's two patches applied (`tests/test_scorers_score_all.py`, 15 passed; `tests/test_tools_peer_check.py`'s schema test).
- Tests: e2e 6 passed in both venvs (K8 PASS, K4 PASS, exit 0; G5BP/27 11.932 / 6.414 with the ceiling 22.27 / 19.12, mean pair KL 0.6066, entropy 6.611 / 6.931; every other control at its margin; probe 22 5.701 / 5.746). Full suite 11 failed, 1,884 passed, 3 errors in both venvs (the drivers' record keys, five `G5BP/22` literals in other owners' tests, and the five W12-dependent ids).

## Stage 4: W14 — both Pythons, end to end, dry run

The first attempt (11:28–12:20Z) applied the cross-owner follow-ups and ran the suite, then ended on an API error; the second attempt finished the task.
- **Cross-owner follow-ups** (the edits of decision (b) in files outside Integrate's list): `gate/rules.py` (W6): the G5-BP-lean "power not shown" note and two docstrings name the required controls of `controls.json` (mutant 27) instead of mutant 22; `tests/test_gate_rules.py` (M27 fixtures), `tests/test_gate_g5_runner.py` (`g5_bp_m27` and the probe), `tests/test_integration_contracts.py` (the phase-6 test with the control and the probe block), `tests/test_gate_drivers_tiny.py` (`probes` in `RECORD_KEYS`, `K8/g5_bp_m27` in the block set).
- **Item 6, the calibration record.** `rules.py` changed, so F_tiny was re-run on the final code (`python -m gate.checks.ftiny --root <dir> --write`, 12:33:45Z): F_tiny 4.1119e-7 unchanged, cap and validation met; every code hash recorded in `gate/calibration.json` (ftiny, g4_forced, ref_drivers, ref_pass, common, harness, port_mutants, rules), `controls.json`, `thresholds.json`, the port (`2c153357…`) and the reference tree (`85337ed7…`) equal the final files (checked again by W12).
- **Item 3, the stress records** (12:43–12:50Z, Python 3.12): the whole gate `--tiny` with `--head-policy quantised_head` on the sharpened and on the unsharpened seed-29 build, committed to `diagnostics/tiny_stress/` with a README (records `e53158e9…`, exit 1, and `9547697b…`, exit 0). They reproduce W7's and FixB's scratch values; new are the sharpened build's mutant-27 values (1.57 / 1.28, not caught) and its fixture ceiling (2.23 / 2.06). The specification's §11.13 took the records' values.
- **Item 2, the tiny control table** (`tests/fixtures/exp037_tiny_controls.json`, from the unsharpened record): every control caught; G4F16/6 κ 2,992 (needs 90); G4F32/1 count 64; G4F32/6 400,642; G4F32/19 309,954; G5D32/20 2.30e7 (T9); G5R1/26 1,408; G5BP/27 by decision (e) (11.93 / 6.41, ratio > 3 on both; `rules.py`'s margin 30 not met, recorded beside it) with the ceiling 22.27 / 19.12; G2 registered mutants caught, none undetected; probe 22 5.70 / 5.75, no margin. The e2e test's tables in 3.12 and 3.14 equal the fixture apart from the record name.
- **Item 1, the full suite**, both venvs: **2 failed, 1,894 passed, 3 errors of 1,899** (about 30 min each). The five are the document-dependent ids W12 owns: `test_runner_plan_fix.py::test_plan_defs_match_the_hypothesis_ladder_table`, `::test_tier_b_cells_match_the_hypothesis_table`, and the three W12-pending switches (`test_tools_env.py::test_readme_links_resolve`, `test_tools_hash_tree.py::test_scopes_match_the_real_hypothesis_table`, `::test_round_trip_on_a_copy_of_the_real_hypothesis`). Of W0's 279 baseline ids: 266 pass, 8 no longer exist (renamed or dropped by their owners, as recorded above), and 5 are those five.
- **Item 4, the dry run.** Without HYPOTHESIS.md it stopped at its first stage ("world"); with a stub HYPOTHESIS.md written into its temporary copy only (a scratch wrapper) all 17 stages passed in 11.3 min (gate stage 396 s, verdict TINY, exit 0; plan P0; verdicts computed).
- **Item 5, the leak check** (`--all`): 332 file versions, 0 findings, 4 warnings, all the mini's host name in files copied byte for byte from exp_036 (`ASSETS.md:5`, `assets.json:187`, `reference/README.md:225`, `tests/INTEGRATION_LOG.md:3`). `assets.json` and `reference/` are pinned scopes (their values equal exp_036's), so they were left unchanged.
- **Hashes** (`hash_tree --list`, both venvs, same values): reference `85337ed7…`, gate_text `1a295dac…`, requirements `32488f23…`, port `2c153357…`, gate_rules `488dfa7d…`, thresholds `58f55f2c…`.

## Stage 5: W12 — documents

**Files written** (W12 owns them; no other file of the kit was changed):
- `HYPOTHESIS.md` (1,394 lines): exp_036's pre-registration restructured as the specification asks: exp_036's Amendments 3–6 folded in (GPQA-Diamond EN all 198 items; the Webster allocation and the gold-consistent full-split pools; chat-wrapped fidelity and flip → `B=1`; per-build fidelity attribution as a rule, with the S1 record named by rule, not exp_036's outcome); the runtime sentence for F2; the Phase 0 section in pre-registration prose (preconditions, the check table, controls, calibrators, verdicts and exits, risk, peers, runtime rules); the freeze rules; the plan rule (P0–P10, Tier A, Tier B unchanged); H1–H8 and D1 with the tripwire; the decisions record (exp_036's carried answers, D8, G3, the push of 5c3d178, A–F, E-downloads, E-verdict, the defaults, the lean cuts, the main session's (a)–(e), go #1's three answers to be recorded at the freeze); the hash table with 18 rows and placeholders (exp_036's values by prefix only, so no row holds a stray 64-hex value) and a table of the exp_036 files read in place; Sessions & budget; the plain answers with the tripped Q2 rows and the caveat verbatim from `analysis/verdicts.py`; the upload criteria; the NOT RUN wording; the publication angle with the blind-spot paragraph; the edge cases; the disclosures (chronology, known values, the known-values ledger, the loosenings with counterfactuals and the sentence "Every loosened rule failed in exp_036; none that passed was loosened.", constant provenance, the M5 defect, the tiny stress finding from the committed records, the vendor spike's literal exceedance, decision (e)); Annex A (W0r verbatim, inserted by script from its source, sha256 `624767f5…`, headings one level lower); and the Sign-off block with every field empty. Status: "Pre-registered; awaiting Andrei's sign-off; no gate run, no scored run".
- `RUNBOOK.md`: the mbp steps 0–19 under the block header (exp_036's env file sourced from exp_036's directory, then `env/exp037.settings.sh`, which sets `PY` to `~/models/exp037/venv`); step 1 writes the exp_037 version record; step 2 installs exp_037's hook; step 3 syncs `$EXP037_VENV` and runs the suite with `run-host` (`test_gate_ftiny.py` skips as `build-host only:`); step 7 verifies the inherited manifests, shingles and T5/T6/T9; step 8 runs `tools/refresh_builds.py`; step 9a the Gemma fidelity record (shards checked first); step 10 the gate with its exit table and the post-gate choices; the commit prefix `chronos/exp_037:`; the mini's steps after the hand-back with `EXP037_VENV` preset to `venv312`. Written with the Write tool, because it names exp_036's env file.
- `BUILD_SPEC.md`: the delta over exp_036's (items 8, 10 and 27; §3 F2 pins; §5.1 port; §5.3 gate; §5.4 runner; §5.6 scorers; §5.7 bench; §5.8 analysis; §5.9 tools; §6 tests; §7.1 `gate/thresholds.json` verbatim, inserted by script; §7.2; §8).
- `README.md`, this `BUILD_LOG.md` (Annex A inserted by script), `NOTICE` (exp_037 heading and provenance; the exp_037 changes to `port/kolibri1.py` and `gate/port_mutants.py` with their mutants; VllmRoPE's source and vLLM attribution; exp_037's own files in section 9; the diagnostic RoPE copy now named in exp_036's directory).
- `diagnostics/vendor_spike/`: the spike's README (plus a section "Copied into exp_037"), `scripts/` (8 files) and `out/` (`compare_vllmA.json`, `localise.json`, `discrimination.json`, `inputs_sha256.txt` byte for byte, plus a new `arrays_sha256.txt`). In the copy, the README's host name became "the mini", four Python scripts derive exp_036's directory from their own location instead of an absolute home path, and two shell scripts read the scratch folder from `$SPIKE_DIR`; each changed line says "exp_037 copy", and the README lists the scratch originals' sha256. Not copied: the downloaded sources, the venv, the uv cache, the logs, the checkpoints and the 30 `.npz` arrays (listed by sha256 and size instead).

**The spike's inputs against the kit's builds** (the specification's request): rebuilt with `tests/tiny_real_layout.write_real_layout(out, seed, sharpen=3.0)` at seeds 23 and 29, both shards and the config equal the spike's `rl_s23` / `rl_s29` sha256 (`52b676b4…`, `ed2f74e5…`; `057123e6…`, `be34b2e0…`; config `3ee351c7…`). No difference.

**Readings and choices** (open to W15):
1. **RUNBOOK step 7** has no dedicated tool in the kit: it re-runs `tasks/build_manifests.py` (write-once: every manifest, private manifest and the shingle file must come out `"unchanged"`; a `"written"` entry or a `FileExistsError` stops the run) and loads T5/T6/T9 read-only through `gate.textset.load_real()`, which checks their ids against MANIFEST.json. `gate/build_gate_text.py --work-only` is not used, because it would write into `$EXP036_WORK/gate_texts/`, outside `exp037/`.
2. **Step 1's version record** refuses without the git identity; on a fresh clone step 2 comes first. Step 4b's deep preflight writes a newer record, which is the one P2 reads.
3. **Sign-off and Amendment 0.** The block re-confirms the items decided with A–E. Amendment 0 can change only the H2 margin; the head policy cannot differ, because P4 binds exp_036's quantised-head builds and G0's converted-config check compares the build's policy with the signed-off one. The head-policy line keeps the form `run_gate.signed_off_policy` and `dry_run.fill_signoff` read.
4. **Step 9a's shard check** uses `tools/assetcheck.py`'s metadata and deep hash inline (as written by W12: 12 shards, every sha256 equal to its LFS oid, every file at the pinned commit; corrected before the freeze to 11 shards and all 20 files, see "Pre-freeze: step 9a's shard count"); exp_036 recorded only that the shards were sha256-checked.
5. **H8's peer median** is registered as Gemma 4's, Qwen3.6's and Qwen3.8's, with exp_036 Amendment 6's rule that a family with an excluded or `speed-only` build leaves it; exp_036's outcome (G4 `speed-only`) is not presupposed, because the S1 peer check re-measures it.
6. **Section references.** The freeze rules and the other verbatim parts of the specification keep their wording; references to the specification's section numbers are replaced by HYPOTHESIS.md's own section names, so the pre-registration is self-contained.
7. **Parsers the documents must satisfy:** no row of HYPOTHESIS.md other than the plan ladder starts with `| P<n> |` (the first draft's known-values ledger had `| P2 |` and `| P3 |` rows, which the plan-table test caught; they are now "Precondition P2/P3"), none other than Tier B starts with `| B<n> |`, no line starts with `## Amendment <n>`, no line starts with `- Signed off by: Andrei (`, and the only double-brace placeholders are the 18 scopes and the stamp.
8. **The spike's `.npz` arrays** (130 MB) are not copied, although the task named `out/`: the specification's list names only the four small files, the kit's `.gitignore` excludes `*.npz` and the leak check refuses them.

**Checks** (both Pythons; the test environment above):
- **The document-dependent tests, live** (`test_gate_thresholds`, `test_runner_plan_fix`, `test_gate_refusal`, `test_integration_contracts`, `test_tools_env`, `test_tools_hash_tree`, `test_tools_status`, `test_notice`, `test_tools_leak_check`, `test_tools_dry_run`, `test_tools_preflight`, `test_tools_version_record`, `test_gate_preconditions`, `test_runner_guard`, `test_tools_peer_check_reference`; 411 tests): **408 passed, 3 errors** in venv312 and in venv314. The three are the tests behind W10's `W12_HANDOFF = False` (`test_tools_env.py::test_readme_links_resolve`, `test_tools_hash_tree.py::test_scopes_match_the_real_hypothesis_table`, `::test_round_trip_on_a_copy_of_the_real_hypothesis`), skipped and therefore counted as errors under `EXP036_REQUIRE_ALL=1`. No "not written yet" warning remains: the gate table, BUILD_SPEC §7.1 and the RUNBOOK assertions now run on the real documents. Among them, the two plan-table tests that failed in W14's suite now pass.
- **With the five switches on** (a scratch copy of the kit laid out as the repository, git-initialised, exp_036 and exp_007 linked beside it; `W12_HANDOFF = True` in the three tools tests, `DOC_ASSERTIONS_ON = True` in `test_gate_thresholds.py` and `test_integration_contracts.py`): **411 passed, 0 failed** in venv312 and in venv314.
- **The hash table:** `hash_tree --list` gives the same 18 values in both Pythons, equal to W14's list (reference `85337ed7…`, gate_text `1a295dac…`, requirements `32488f23…`, port `2c153357…`, gate_rules `488dfa7d…`, thresholds `58f55f2c…`, …). `hash_tree --check HYPOTHESIS.md` reports all 18 scopes `unfilled` (exit 1), as it must before the freeze. On a copy, `--fill` and `--stamp` leave no placeholder and `--check` then gives 18 `match` (exit 0) in both Pythons; `hash_tree.unfill()` turns the filled copy back into the template exactly (the dry run's reset).
- **The dry run's HYPOTHESIS-dependent steps** on a copy: `dry_run.fill_signoff` fills the empty sign-off line and the policy line, after which `runner/guard.SIGNOFF_RE` matches (it does not on the unsigned file) and `run_gate.signed_off_policy` reads `quantised_head`; the next amendment number is 1; the Status line matches `tools/status.STATUS_RE`. The whole dry run (about 11 min) was not re-run here; RUNBOOK step 3b's command runs it.
- `tests/test_notice.py` 6 passed. The leak check on the 20 new or changed files: 0 findings, 0 warnings; `--all`: 351 file versions, 0 findings, the same 4 inherited host-name warnings as W14's.
- No `__pycache__` or `.pytest_cache` was left in the kit; exp_036's directory and `git status` outside the kit are unchanged.

**Hand-off note** (to W6, W7/W15 and W10): the documents exist. Set `DOC_ASSERTIONS_ON = True` in `tests/test_gate_thresholds.py` (W6) and `tests/test_integration_contracts.py` (W7's file), and `W12_HANDOFF = True` in `tests/test_tools_hash_tree.py`, `tests/test_tools_env.py` and `tests/test_tools_status.py` (W10). With them on, the suite has no document-dependent failure left (verified on the copy above).

## Stage 6: W15 — the final review and its fixes (W15-fix)

**The review** (W15, 14:17–14:44Z; verdict "ready after fixes") raised fifteen issues and answered decision (d) with an R3 estimate:
- **Blocker:** W15-01, the five document switches still off. Measured by the reviewer on the mini (venv312, `EXP036_REQUIRE_ALL=run-host`), the five switch files gave 93 passed and 3 errors (three skips whose reason does not start with `build-host only:`). Frozen like that, RUNBOOK step 3 would stop, and the real gate's G1 would fail, so K8 would FAIL (exit 1) and use a cycle on a correct port. On a scratch copy with the switches on: 96 passed, 0 errors.
- **Major:** W15-02, no hash scope covered `tests/` or the env scripts, so G1 could be relaxed after the freeze; W15-03, an interrupt, kill or power cut wrote no gate record, so P1(d) could not see the run and the two-cycle limit could be bypassed; W15-04, G5-D32's 10 × Csort contradicted G4-F32's own 100 × derivation (the sharpened build's correct port failed at 11–16 × Csort; estimated false-fail risk 10–20 %); W15-05, the tiny free-routing checks and the controls' power had run only in fp32, never at production bf16; W15-06, A1's 1.2 and 8 were written with exp_036's port M (6.0777) in view, and leg (p)'s 1.2 × margin was missing from the risk list.
- **Minor:** W15-07 (R-greedy blocking but exempt in tiny mode), W15-08 (P1(c)/(e) trust whichever commit the Pre-registration line names), W15-09 (`COPY_RECORD.json` stale), W15-10 (`power.py`'s S1 hours; the tripwire predicate twice), W15-11 (the mbp checks before go #1), W15-12 (F3's 10 × Csort clause always met by cross-implementation rounding), W15-13 (decision (e)'s "can reach 30" stated as a fact), W15-14 (the risks never combined; estimated 25–40 % that a correct port does not PASS), W15-15 (exp_036 cited at `222c845` while its HEAD is `3c99003`; the Pre-registration line against the one-writer rule; the leak check found nothing else, and the 4 inherited host names stay, as the review advised).

**W15-fix** (14:44–16:02Z) resolved W15-01 to W15-06; none was rejected. The minor issues were outside its task: W15-09 is resolved by the pre-freeze housekeeping below, and the others are open items.

**Approved by Andrei at 2026-10-06T18:41:39Z ("as recommended"; HYPOTHESIS.md's decisions table, rows W15-04, W15-02/03 and W15-R3, and its chronology):**
- W15-04: G5-D32's mean and max factors go from 10 to 100 × Csort. The floors and the Csort ceilings are unchanged. The review's optional 1e-3 max ceiling, or VOID for a Csort-max breach, was not taken.
- W15-02/03: the `tests` and `env` hash scopes, and the rules for interrupted, crashed and orphaned gate runs.
- W15-R3: G4-F16's R3 ceiling stays at 1e-2, with the estimated 5–10 % VOID risk accepted; the optional pre-freeze reference-only R2F / R3 pass on the mbp is declined.

The specification's §14.4 records the changes. Not part of the approvals: control 26 at 4.63 × its bound in bf16 (W15-05, below; open item 2).

- **W15-01 (blocker): the document switches.** `DOC_ASSERTIONS_ON = True` in `tests/test_gate_thresholds.py` and `tests/test_integration_contracts.py`; `W12_HANDOFF = True` in `tests/test_tools_env.py`, `tests/test_tools_hash_tree.py` and `tests/test_tools_status.py`. The five files: 97 passed under `EXP036_REQUIRE_ALL=1` and under `run-host`, in both Pythons (96 before W15-02's new test).
- **W15-02: two new hash scopes, both amendable.** `tests` (the `tests/` tree) and `env` (exactly `env/exp037.settings.sh`, `setup.sh`, `versions.json`; nothing else under `env/` is opened). `tools/hash_tree.py` `SCOPES` has 20 rows, `gate/preconditions.py` `REQUIRED_SCOPES = 20`; HYPOTHESIS.md's table gains the rows "Tests" and "Environment scripts", and its freeze rules say that editing, skipping or removing a test, or changing `g1_synthetic.EXCLUDE` (asserted by a G1 test), needs an amendment line, and that a gate fix's test gives a `TESTS_SHA256` line. Tests: `test_tests_and_env_scopes_see_every_test_and_only_their_env_files`; the stub trees and the scope-count assertions of `test_tools_hash_tree.py` and `test_gate_preconditions.py`.
- **W15-03: interrupts and killed runs.** `gate/run_gate.py`: Ctrl-C, SIGTERM and SIGHUP (`GateInterrupted`) and SystemExit during the phases are recorded like a crash ("interrupted in phase N: …"; exit 3, or exit 1 if a K8 check had already failed, by `rules.exit_code_for` unchanged), the record is written with `blind_phase_reached`, an exit-3 run moves to `aborted/<UTC>-gate/`, and the interrupt is then re-raised with the result attached; signals during the record write are deferred; the CLI prints the verdict line with `"interrupted"` and exits with the record's code. Every run writes `results/gate/<UTC>/run.json`; `--close-orphans` (`close_orphans`) judges a run directory left without a record by its phase files and writes its record. `gate/preconditions.py` P1(d): refuses while a run directory other than the running gate's has no record (`orphaned_runs`), and when a blind exit-3 run (`blind_crashes`) whose code hashes differ from the current ones is not named by its UTC in the first line of a numbered amendment (`amendment_first_lines`). Tests: 10 in `test_gate_refusal.py` (Ctrl-C and SystemExit after P5, an interrupt before P5, an interrupt after a K8 FAIL, real SIGTERM and SIGHUP, a deferred SIGINT during the record write, the CLI, a killed run closed as exit 3 and as exit 1) and 4 in `test_gate_preconditions.py`. RUNBOOK step 10, BUILD_SPEC §5.3 and §6 item 10, HYPOTHESIS's P1 and freeze rules.
- **W15-04: G5-D32's factor 10 → 100** (`mean_factor_vs_csort`, `max_factor_vs_csort`; floors and Csort ceilings unchanged; `gate/rules.py` docstring). The max-ceiling options (1e-3, or VOID for a breach) were not taken; the residual risk is stated in HYPOTHESIS's risk section. Because `rules.py` and `thresholds.json` changed, F_tiny was re-written (`python -m gate.checks.ftiny --root <dir> --write`, 15:10:32Z): F_tiny 4.1119e-7 unchanged, cap and validation met; `calibration.json` now records `rules.py` `fe2c99f0…` and `thresholds.json` `e27bdc06…`. BUILD_SPEC §7.1's block and HYPOTHESIS's gate table, constants, provenance, values-known paragraph and stress row follow. Tests: `test_gate_rules.py`'s G5-D32 boundaries and control 20 (with the sharpened build's values now passing), `test_gate_thresholds.py` (100 = G4-F32's τ factor). The tiny control table is unchanged (the unsharpened build's bound is the 1e-8 floor at either factor).
- **W15-05: the whole tiny gate in bf16** on the unsharpened seed-29 build (`--tiny-precision bf16`, 15:10–15:16Z, Python 3.12), committed as `diagnostics/tiny_stress/unsharpened_seed29_bf16/` (record sha256 `67ce52f0…`, exit 0): K8 PASS, K4 PASS, every required control caught; 27 at 10.50 / 7.01 (decision (e) met; fixture ceiling 19.42 / 20.20); **26 at 4.63 × its bound, under the 10 × tiny margin** (caught; its effect is unchanged, mean KL 0.148 against 0.141 in fp32, but R-parity's bound rises from the 1e-4 floor to 3 × the bf16 chaos floor, 0.032); probe 22 5.41 / 5.35; G4-N(i) 0.0245 / 0.0224. No rule changed. The two overstated sentences (decision (a)'s "serves every check with a quantised or bf16 arm", and "the failures come from the fixture under 8-bit quantisation", which does not hold for G5-D32) are corrected in HYPOTHESIS.md, the stress README and the specification.
- **W15-06: A1's provenance.** A1's 1.2 and 8 moved from "Inherited" to the "New" table with "port M 6.0777 in view" (exp_036 `diagnostics/gate1/FROZEN_RULES.md` l.281–282 and l.488, committed 2026-10-05T10:11:44Z, after exp_036's gate run), with a caveat under "No constant is justified only by …"; leg (p) is in the risk list (about 5–10 %).
- **Decision (d), R3:** the ceiling is kept; the review's estimate (P(VOID) about 5–10 %, almost all on T9) and its basis, re-checked against exp_036's `stage3_g4` summaries, layers CSV and G3b record, are in HYPOTHESIS's ledger and risk section.

**Checks after the fixes** (the environment above):
- **Full suite, both Pythons:** **1,914 passed, 0 failed, 0 errors, 0 skipped** in venv312 (28 min 49 s) and venv314 (29 min 7 s), started 15:22:10Z, `EXP036_REQUIRE_ALL=1`. The 15 more than W14's 1,899 are the new tests above. The end-to-end tiny gate (`test_gate_end_to_end_tiny.py`, 6 tests), the drivers (32) and F_tiny (15) pass in both; the control tables they write equal `tests/fixtures/exp037_tiny_controls.json` apart from the record name.
- **G1 as the gate runs it** (`gate/checks/g1_synthetic.run`: the suite minus `EXCLUDE`, `EXP036_REQUIRE_ALL=run-host`): **1,861 passed, 0 failures, 0 errors, 0 skips, return code 0** in venv312 (7 min 58 s) and venv314 (8 min 10 s): the 1,914 minus the 53 tests of the three excluded files. Before W15-01 this was 3 errors (the W15 review's measurement), which would have failed the real gate's G1.
- **The dry run** (`tools/dry_run.py`, venv312): all 17 stages ok in 10.5 min; the gate stage exit 0 (verdict TINY), `hash_check` "20 scopes match the filled table", plan P0.
- **The hash table** (`hash_tree --list`, identical in both Pythons): reference `85337ed7…`, port `2c153357…`, gate `1d413283…`, thresholds `e27bdc06…`, gate_rules `5e9e1a53…`, gate_text `1a295dac…`, requirements `32488f23…`, tools `24b0e22d…`, tests `15f72af3…`, env `8d97b3df…`. `--check HYPOTHESIS.md`: all 20 `unfilled` (exit 1), as before the freeze. On a copy: `--fill` 20 placeholders, `--stamp`, `--check` 20 `match`; `unfill()` gives the template back.
- **The leak check** (`--all`, both Pythons): 364 file versions, 0 findings, the same 4 inherited host-name warnings.
- No `__pycache__` or `.pytest_cache` in the kit; exp_036's directory unchanged; nothing committed.
- **File sha256 after W15-fix:** HYPOTHESIS.md `3ea1824b…` (1,409 lines; PowerLog and the main session's decision rows have edited it since), `gate/thresholds.json` `e27bdc06…`, `gate/calibration.json` `234a9745…`, the bf16 stress record `67ce52f0…`. The pre-edit copies and the scratch evidence stay in the session's scratch folder, outside the kit. That covers the suite logs and junit files, the G1 records, the dry-run log, the hash lists and round trips, the leak-check output and the bf16 run.

## Stage 7: the descriptive power log (PowerLog: build, review, fixes)

Andrei asked for it at 2026-10-06T13:51:07Z ("also add powermetrics"), so that exp_037's machine energy on the mbp is measured, where exp_036's was only estimated. The mbp's pinned NOPASSWD sudoers rule for exactly the logger's command line followed at 14:43:27Z (HYPOTHESIS chronology). It is descriptive only. None of the three tasks changed a GATE_RULES file, a threshold, a rule or gate code: gate `1d413283…`, thresholds `e27bdc06…`, gate_rules `5e9e1a53…` and env `8d97b3df…` are unchanged, and so are runner, analysis and bench. Only the `tools` and `tests` scopes moved. powermetrics was never run on the mini, exp_036's env file was never opened, and nothing was committed.

### Build (16:03–17:01Z)
- **`tools/power_log.py`** (new, stdlib only):
  - It reads the NUL-separated `-f plist` stream as it goes. A document cut short by Ctrl-C is reported as `truncated_tail`; a broken complete one counts as `unreadable`.
  - Per sample it keeps only `is_delta`, `elapsed_ns`, `timestamp` and `processor.{cpu,gpu,ane,combined}_power` (mW). `gpu.gpu_power` is the fallback, and combined is taken from the components when absent. A sample covers [timestamp − elapsed, timestamp].
  - It finds the windows in the run records: the gate record and its phases (or the phase files of a run with no record), the peer check, preflight `--deep`, refresh, the bench cells and the speed cells' cool-down idle, tokenizer and kl_8v4, the pilot and its arms, and the session start/stop pairs. `--window` adds one, and the whole log is always a window.
  - It writes `results/power/power_<UTC>.json` (schema `exp037 power v1`). Per window: energy, mean and peak W with the CPU/GPU/ANE split, samples, coverage and gaps over 5 s. Per raw file: name, redacted dir, sha256, size, counts and interval statistics. Also not_measured, caveats, method and the tool's sha256.
  - The only host field it keeps is `model_family`; `hw_model` and `kern_*` are never written, and a test enforces it.
  - It refuses a raw path inside the repository, a raw log already recorded (unless `--again`), overlapping raw logs, a log with no readable sample and a missing git identity. `--check` prints one JSON line and writes nothing.
- **`tests/test_tools_power_log.py`** (new): 28 tests on synthetic streams. They include that the RUNBOOK's logger lines are exactly the pinned sudoers command, and that no code in gate/, runner/, bench/, analysis/, scorers/, port/, reference/ or tasks/ names `power_log`, `results/power` or `powermetrics`.
- **`RUNBOOK.md`:** a "Power log" section in the preamble, and a second-terminal block in steps 9, 10, 11, 12, 14 and 16. The logger starts before the step and stops after it, never in between; Claude checks it, writes the record and commits it with the step's records. The raw `.plist` is never committed, and a missing log never blocks a step.
- **`HYPOTHESIS.md`:** the section "Power and energy (descriptive)", a `results/power/` line in the evidence layout, the Tools row's text and two chronology rows.
- **Checks** (both Pythons, `EXP036_REQUIRE_ALL=1`):
  - the new file: 28 passed;
  - full suite: **1,942 passed, 0 failed, 0 errors, 0 skipped** (1,914 + 28; 27 min 25 s and 27 min 43 s);
  - leak check `--all`: 366 file versions, 0 findings, the same 4 inherited warnings;
  - `hash_tree --check HYPOTHESIS.md`: 20 unfilled (exit 1);
  - `--list`, identical in both: tools `24b0e22d…` → `ea1bb859…`, tests `15f72af3…` → `f03a5da2…`;
  - file sha256: `tools/power_log.py` `814651e9…`, `tests/test_tools_power_log.py` `898ed78d…`.
- A scratch sanity run on exp_036's committed records with a synthetic 2 h log found the peer check, the gate and gate phases 1–6.

### Review (17:01–17:12Z)
Eleven issues, three of them major:
- **P1, double counting (major).** Documents that overlap in one log add their energy twice. powermetrics' poweravg summaries are the known source: `-a` defaults to every 10 samples, and the pinned command cannot turn it off. A probe gave 2.0 Wh where the truth was 1.0 Wh, and `--check` said ok.
- **P2, untested layout in a frozen scope (major).** No real powermetrics output had been read. The key names and types, the NUL placement, any header text and the poweravg documents were assumptions, and a parser fix after the freeze changes the `tools` scope. The review proposed a 60 s smoke on the mbp before the freeze and a hardened parser.
- **P3, the "SoC" label (major).** CPU + GPU + ANE leaves out the rest of the SoC and DRAM, and "costs no machine time" was wrong.
- **Minor:**
  - P4: `--check` diagnostics, and detecting a logger that is still running.
  - P5: the window of a crashed-then-resumed session, and the logger after a reboot.
  - P6: a closed orphan gate run's window ends at the close time.
  - P7: the no-deciding-code scan leaves out `tools/`.
  - P8: defence in depth for the raw `.plist`, in `.gitignore` and the leak check's refused suffixes.
  - P9: unlogged mbp steps (9a, 8, 4 `--deep`, 3, 17) and the mini, against the claim.
  - P10: nested windows must not be summed, and the per-component means.
  - P11: the BUILD_LOG, BUILD_SPEC, README and status plumbing, and the clash with "power" in the statistical sense.

### Fixes (17:12–18:07Z)
- **P1:**
  - Each raw file records `elapsed_sum_s`, `covered_s`, `overlap_s`, `excess_s` and `long_samples`, and each window `overlap_s`.
  - `build_record` refuses a log whose summed elapsed time exceeds its span by more than 2 % + 2 s. The trigger is the excess, not the overlap: whole-second plist dates make the overlap about elapsed − 1 s per sample. A parametrised test at 1,000.4 to 1,050 ms intervals shows no false alarm.
  - A long document (over 3 × the median) fails `--check` but stays in the record.
- **P2:**
  - `elapsed_ns` may be an integer or a real. `<c>_energy` (mJ) divided by elapsed is the fallback (`from_energy`), for the `gpu` dict too.
  - Text before or between documents is skipped and counted.
  - `--check` reports the skipped and parsed counts and gives specific problems ("key layout differs …", "no plist document").
  - The new `--layout` prints key names and plist types per layout, never a value (tested).
  - The default raw log matches only `power_<stamp>.plist`, never a smoke file.
  - RUNBOOK gains the smoke block and "If the tool cannot read a log after the freeze". HYPOTHESIS registers the same, plus a new "power log" amendment type: after S3 only, for `tools/power_log.py` and its test only, with their `hash_tree:` lines.
- **P3:**
  - The label is "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power". The key `soc` is renamed `combined`; the schema stays `exp037 power v1`, since no record exists yet.
  - What is not measured now includes the rest of the SoC and DRAM. The mini's spot reading gives the scale: 28.6 W CPU + GPU against 64.6 W for the system, about 44 %, on a different machine.
  - The 1 Hz background-load sentence replaces "costs no machine time", the man page is quoted, and a test forbids the old wording.
- **Not taken by the fix task:** P4–P7, P9 and P10 (open items below). The housekeeping below did P8's `.gitignore` half and P11's BUILD_LOG, BUILD_SPEC and README half.
- **Checks** (both Pythons, `EXP036_REQUIRE_ALL=1`):
  - power-log tests: **41 passed** (28 + 13).
  - Full suite: **1,955 passed, 0 failed, 0 errors, 0 skipped** in each. It ran in five foreground chunks of whole files, because a single foreground call was limited to 10 minutes. The chunks' junit reports cover the same 1,955 node ids, with no duplicate and none missing.
  - One first-attempt failure in both venvs, `test_gate_refusal.py::test_a_signal_while_the_record_is_written_is_deferred`, came from the launcher: a job started with `&` in a non-interactive shell inherits SIGINT ignored. The file passes in a true foreground run (41/41). The group re-ran green once the chunk runner reset SIGINT (1,039/1,039).
  - The review's probe: the poweravg-like log is refused. A real `elapsed_ns`, header text and an energy-only log are read. A sample spanning a sleep is flagged by `--check` and kept.
  - Sanity run on exp_036's records with the synthetic 2 h log: 9 windows (log, peer check, gate, phases 1–6), coverage 1.0, overlap 0. Its record is leak-checked clean.
  - Leak check `--all`: 366 file versions, 0 findings, the 4 inherited warnings. `hash_tree --check`: 20 unfilled.
  - `--list`, identical in both: tools `ea1bb859…` → `77a077f530ded752102c6f97d876cb83908c9268e9aec45549dbb0b65c1b63f7`, tests `f03a5da2…` → `ca1f1b427a99666bbf5baf14c8e85c3952f0e8f531743122a9da3dc4bc17a6f0`. Nothing else moved.
  - File sha256: `tools/power_log.py` `349657c8d32395e53b8cd1578263b87dd309193f3127155aaf4ea91bed66b3c4`, `tests/test_tools_power_log.py` `44f7c61f93d944fc6f75cda860ab630addf70762cac71706542979ebd79db02e`.

### The mbp smoke (the layout record HYPOTHESIS asks for)
**Run on 2026-10-06, before the freeze, for about 4 s rather than 60 s.** Andrei ran RUNBOOK's pinned command on the mbp (macOS 27, model family Mac17) and stopped it after a few seconds, which the main session accepted for the layout. The raw file was copied to the mini and read there with `tools/power_log.py` (sha256 `349657c8…`, the Stage 7 file) under venv312. It is never committed (`*.plist` in `.gitignore`), and neither is the scratch record, `power_20261006T184323Z.json`, which carries no host field.
- **The file:** 115,002 bytes, two documents, both `is_delta`; `elapsed_ns` 1,020 and 1,032 ms; samples from 18:42:51Z to 18:42:55Z; samplers seen `cpu_power`, `gpu_power`, `ane_power`. Each document ends with a NUL, as in the fixture's default stream; no text bytes, no truncated tail, nothing unreadable, no poweravg document, nothing skipped.
- **`--layout`** (key names and plist types only):
  - top: `elapsed_ns` integer, `gpu` dict, `hw_model` string, `is_delta` bool, `kern_bootargs` string, `kern_boottime` integer, `kern_osversion` string, `processor` dict, `timestamp` date;
  - `processor`: `ane_energy` integer, `ane_power` real, `clusters` array, `combined_power` real, `cpu_energy` integer, `cpu_power` real, `gpu_energy` integer, `gpu_power` real;
  - `gpu`: `dvfm_states` array, `freq_hz` real, `gpu_energy` integer, `idle_ns` integer, `idle_ratio` real, `sw_requested_state` array.
- **Against the fixture:** every key the tool reads is present with the fixture's type. The real documents carry `<c>_energy` beside `<c>_power`; the tool reads `<c>_power` first (`from_energy` 0). The fixture is therefore kept as built and not rebuilt.
- **Whole-second timestamps confirmed.** The two samples, about 1.03 s apart, carry the plist dates 18:42:53Z and 18:42:55Z. The span is therefore 3.02 s against 2.05 s of elapsed time (`excess_s` −0.968, coverage 0.68 on a 3 s log). The tool's 2 s whole-second slack covers this. `test_whole_second_timestamps_are_not_taken_for_double_counting` models it. On a run of minutes the placement error stays under 1 s at each window edge.
- **`--check`:** `ok` true, no problems (run with a large `--max-age`, since the copy was 15 min old).
- **Not exercised by two samples:** buffering over a long run, poweravg documents and a restarted logger. These stay covered by `--check` at each step and by the "power log" amendment after S3.

## Pre-freeze housekeeping (main session)

After the approvals of 18:41:39Z; only `BUILD_LOG.md`, `COPY_RECORD.json`, `.gitignore`, `README.md` and `BUILD_SPEC.md` were edited. None of them is in a hash scope.
- **`COPY_RECORD.json`** (W15-09; open item 4) now lists every file of the kit except itself: 365 entries, 95 `new`, 181 `unchanged`, 89 `to modify`.
  - 90 files created after W0's copy are added as `new`, each with its owner task and its sha256 at this update: the five documents (W12), the tiny stress records (W14; the bf16 record W15-fix), the vendor spike (W12, from W13's scratch spike), the exp_037 gate files, tools, tests and fixtures (each with its stage 0–4 owner task) and the power log and its test (PowerLog).
  - `diagnostics/gemma_quant_check.py` is added as `unchanged` (FixC; byte-identical to exp_036's, `9393765d…`).
  - Seven entries recorded `unchanged` that FixC or Integrate changed move to `to modify`, owner FixC: `scorers/score_all.py`, `tools/fidelity_reference.json`, `tests/test_port_ref_hooks.py`, `tests/test_runner_jsonl.py`, `tests/test_scorers_score_all.py` (applied by Integrate), `tests/test_tools_leak_check.py` and `tests/test_tools_peer_check_reference.py`.
  - The `.gitignore` entry notes the `*.plist` line.
  - The schema is kept; a top-level `updates` list records this update.
  - Every `unchanged` entry now hashes equal to its copy sha256, and every file in the tree has an entry.
  - A new file's `copy_sha256` is its sha256 at this update. The documents change again at the freeze (hash fill, stamp, sign-off).
- **`.gitignore`:** `*.plist`, so raw powermetrics logs are never committed (review P8, first half).
- **`README.md`:** the `tools/` row names the descriptive power log and RUNBOOK's "Power log" steps. The COPY_RECORD row now says that new files are listed too.
- **`BUILD_SPEC.md`:**
  - §1's tree gains `tools/power_log.py`, and the tiny stress records are now three.
  - §5.9 gains `tools/power_log.py`.
  - §6 lists `test_tools_power_log.py` and gains item 13.
- **This file:** Stage 6's review summary and approvals, Stage 7, this section and the open items.
- **Checks** (both Pythons, the environment above):
  - The document-dependent tests: **199 passed, 0 failed, 0 errors, 0 skipped** in venv312 and in venv314. By file: `test_tools_env` 25, `test_tools_hash_tree` 29, `test_runner_plan_fix` 43, `test_integration_contracts` 14, `test_gate_refusal` 41, `test_notice` 6, `test_tools_power_log` 41.
  - The other readers of the edited files (`test_gate_thresholds` for BUILD_SPEC §7.1, `test_tools_leak_check`, `test_tools_status`): 55 passed in each.
  - `hash_tree --list`: identical in both and equal to Stage 7's list (tools `77a077f5…`, tests `ca1f1b42…`, gate `1d413283…`, thresholds `e27bdc06…`, gate_rules `5e9e1a53…`, env `8d97b3df…`).
  - Leak check `--all`, run after the last edit to this file and to `COPY_RECORD.json`: 366 file versions, 0 findings, exit 0. Its 4 warnings are the same inherited host names (`ASSETS.md:5`, `assets.json:187`, `reference/README.md:225`, `tests/INTEGRATION_LOG.md:3`), and the withheld-text check was on.

## Pre-freeze: step 9a's shard count (main session)

**Found by W15-11's read-only mbp check, 2026-10-06 ~19:02Z, before go #1.** Andrei ran it on the mbp. Both K8 / K4 convert records are present, `data` is present and the exp_037 venv reports `0.32.3 0.32.0`. The Gemma 4 bf16 folder holds **11** `*.safetensors` and **20** download `.metadata` files.

**Cause.** exp_036 recorded the bf16 source as "12 shards, sha256-verified" (Amendment 6, D) and "12 LFS shards" (`tools/fidelity_reference.json`'s note). W12 copied that number into step 9a's inline check as `len(.safetensors) == 12`. The repository's file list at the pin was read from the Hugging Face API (`https://huggingface.co/api/models/mlx-community/gemma-4-26b-a4b-it-bf16/tree/e13fae2a81ec07e3092a3ebb70c80970b640dc3c?recursive=true`, 2026-10-06T19:05Z, response sha256 `58eb14f9…`). It has 20 files, of which 12 are LFS files: the **11** weight shards `model-000NN-of-00011.safetensors` and `tokenizer.json` (32,169,626 bytes). exp_036's "12" counted LFS files, not shards. On the intact folder, step 9a would have printed `11 bf16 shards: MISMATCH` and stopped.

**Correction** (no hash scope moves; RUNBOOK, HYPOTHESIS and this file are outside every scope):
- The RUNBOOK step 9a check now requires all of these: 20 files with metadata, of which 11 `.safetensors` and 12 sha256 (LFS) etags; every file's hash equal to its download metadata, with git blob sha1 for the 8 non-LFS files; every metadata commit equal to the pin. Its expected line is `11 bf16 shards, 12 LFS files, 20 files: every hash equal to its download metadata at the pin`. The step text says the same.
- HYPOTHESIS "Assets" now says "20 files, of which 11 weight shards and 12 LFS files with `tokenizer.json`". Reading 4 above names the correction.
- `tools/fidelity_reference.json`'s note ("12 LFS shards") is in the `tools` scope and is left as it is: as a count of LFS files it is true.
- **Tested** on the mini (venv312), with the command text extracted from RUNBOOK.md and run on synthetic folders built from the pinned file list:
  - the full layout: exit 0;
  - one shard missing: `10 … 11 … 19 …: MISMATCH`, exit 1;
  - a wrong commit: MISMATCH, exit 1;
  - one shard's bytes changed: MISMATCH, exit 1.
- **exp_036 is not changed.** Its "12 shards" (HYPOTHESIS D, Amendment 6) means 12 LFS files. Any erratum there is Andrei's call.

## Pre-freeze: inherited-values sweep and fixes (main session)

**Why.** Step 9a's count (above) was a value copied from prose into a check, and the W15 review did not catch it. The main session therefore swept the kit for the same class of defect before go #1 (2026-10-06, 19:08–19:22Z). Five read-only readers covered RUNBOOK steps 0–5, 6–9, 10–13 and 14–19 plus HYPOTHESIS's inherited facts. They traced 547 expected values (counts, paths, flags, printed lines, exit codes, hashes) to the code or a committed artefact. They reported 16 defects, and one adversarial verifier per defect tried to refute each. **15 were confirmed** (1 blocker, 3 major, 11 minor). One was refuted: K8's card parameter count, correctly quoted.

**The findings and what was done** (fix round 19:26–20:42Z, reconciliation 20:44–22:03Z, final items by the main session until 23:05Z, all 2026-10-06; times are those of the workflow records; every code change has a test that fails on the code before it):

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 6 | blocker | `runner/plan_fix.py` took the newest `gate/gate_*.json`, which on every real run is `gate_<UTC>_mutants.json` ('_' sorts after '.'), so allowed_B fell back to {1} for both arms and the plan named the wrong record | `build_context` filters with `runner/guard.GATE_RECORD_RE`, the same record the session side reads; regression test with a `_mutants.json` sibling |
| 1 | major | Step 4b's check said exit 0, but the step-1 and 4a records sit uncommitted, so preflight exits 2 (as in exp_036, `aborted/20261004T062910Z-preflight-deep/`) | RUNBOOK commits the step-1 and 4a records before 4b; 4b's check is `problems = []`, exit 0, or 2 only for the sysctl advice |
| 12 | major | The S2 / S3 blocks chain `preflight --quick && session`; a resume finds the uncommitted session tree, preflight exits 2 and the session never restarts | The blocks accept exit 0 or 2 and stop on 1; a bullet lists the expected warnings and says an AC / power / thermal warning means Ctrl-C |
| 13 | major | Per-step decode logs (about 190 B a step) reach about 190 MB for a K8 cell at B = 1: over the leak check's 50 MB and GitHub's 100 MB limits, gigabytes over all sessions | **Decision E** (below; Andrei's go (e)) |
| 2 | minor | Step 0c asked for ≥ 150 GB free; preflight requires 260 GB until step 8's clones exist | 260 GB, 120 GB after step 8 |
| 3 | minor | Step 8 checked `previous_port_sha256` `cd6153b8…`, true only on a first clone | Checks `source_port_sha256` |
| 4 | minor | Step 8 stated the trust step and `check_port_file` in the wrong order | Order corrected |
| 5 | minor | Step 7 said a `FileExistsError` names a differing file; two files fail without that name | Wording follows `main()`'s error |
| 7 | minor | Step 10 read allowed_B from the gate's last stdout line, which has no such key | Read from the record with a one-liner through `guard.newest_gate_record` |
| 8 | minor | The H8 K8 dump goes to `$EXP036_WORK/kl/<UTC>/`, not under `exp037/` | Docs; BUILD_SPEC §2 lists it (and `bench/batch_flip.py`'s C1 dump) as an exception for inherited bench code |
| 9 | minor | Pilot output paths lacked the `<arm>/` level | Corrected |
| 10 | minor | A speed-only G8 makes the pilot refuse unless `--without G8` | Step 12 covers it |
| 11 | minor | `--close-orphans` moves only an exit-3 run to `aborted/` | RUNBOOK corrected; HYPOTHESIS P1(d)'s parenthetical corrected to the same fact (the precondition is unchanged) |
| 14 | minor | The S3 record block missed `results/raw/S3b/` and S2 cells that S3 appended to | `tools/status.py` `SESSION_COVERS`: the S3 blocks hash S2 (final state), S3 and S3b, and each session block its aborted cells' step logs |
| 15 | minor | `--record-block "verdicts"` hashed nothing | A `verdicts` phase: `verdicts_<UTC>.json` / `.md`, `rescore_mini_<UTC>.json`, the mini's `scores/*/ifbench_*.jsonl` |

**Reconciliation and final items** (from the fix rounds' reviewers):
- The pilot is bound once, by the plan block (`tools/status.py` `PILOT_FILES`); RUNBOOK step 13's explicit pilot block was removed.
- Step 18:
  - every variable is guarded (`${EXP:?}`, `${EXP036_PRIVATE:?}`, `${MINI:?}`), since an empty `$EXP` would have made rsync's source `/`;
  - an oversized pilot log is copied with `cd "$EXP" && rsync -a --relative <path>`, which recreates its folders. openrsync ignores the `/./` marker; this was checked locally on the mini.
  - The step-log check drops a missing path only when an `aborted/<tag>/NOTE.md` names its move and a block lists the moved copy, which it then checks. It was tested on five synthetic cases (the exception; an unlisted copy; a missing copy; a changed log; no listing), run verbatim under zsh.
- `tools/leak_check.py`:
  - the 50 MB limit covers `results/raw/`, `results/pilot/` and `aborted/`;
  - a raw or pilot output moved by an abort (`aborted/<tag>/S*-*.jsonl`, `aborted/<UTC>-pilot/pilot/**`) keeps the raw text rules (`RAW_ABORTED_RE`), so an address a model wrote is a warning there too, as in `results/raw/`; other `aborted/` files stay strict. The test fails on the earlier module.
- `results/README.md`, BUILD_SPEC §2 / §5.9 and the RUNBOOK checks of steps 15 and 17 describe the coverage above.

**Decision E: step logs** (proposed by the main session 2026-10-06; **approved by Andrei's go (e), 2026-10-07T04:11:40Z**).
- **The pilot's** `results/pilot/<UTC>/<arm>/*.steps.jsonl` are committed, with an aborted pilot's copies alongside that pilot. The registered plan rule fits its step model on a summarised pilot's step logs (`runner/plan_fix.py` `load_steps`), so the plan stays re-derivable from the public repository.
  - Size: about 1 MB a cell at B = 8, at most about 50 MB at B = 1 with every item at the 32k cap.
  - One still over 50 MB stays uncommitted through the mbp's `.git/info/exclude` (step 13) and is copied to the mini at step 18.
- **The sessions'** (`results/raw/**/*.steps.jsonl`, and abort_cell's `aborted/*/S*-*.steps.jsonl`) are git-ignored working files:
  - never committed and never deleted;
  - kept on the mbp and copied to the mini at step 18;
  - no score, verdict or registered rule reads them.
- **Both kinds** are bound by sha256 in the run-record blocks (`tools/status.py` reads the filesystem, git-ignored or not).

**Checks:** see "Final pre-freeze check" below.

## Final pre-freeze check (main session)

- **Final review** (one adversarial reviewer on the main session's last edits, 2026-10-06 ~22:10–23:00Z). It confirmed `RAW_ABORTED_RE` against every writer to `aborted/` and confirmed that withheld-text and canary protection are unchanged. It raised six issues, all resolved before this entry:
  - the minute-level times of this section, first written without the records, corrected from the workflow records;
  - this section, which was missing;
  - the narrowed reason for committing an aborted pilot's step logs, carried into `.gitignore` and a test docstring;
  - step 13's oversized-pilot bullet, which now names step 18's own rsync line;
  - `COPY_RECORD.json`: a clock-read time for its update entry, and the `.gitignore` note.
  - A torn line's repo copy, `aborted/<UTC>-torn-<arm>-<stem>/torn.part`, can come from any appended JSONL, bench records included. It keeps the strict rules (fail-closed), and RUNBOOK step 15 gives the manual route for a finding there (leave it uncommitted; its committed NOTE.md keeps the sha256).
- **Full suite** on the tree after the main session's final items, before the review's six fixes: **1,962 passed**, 0 failed, 0 errors, 0 skipped, in each of venv312 (24 min 35 s) and venv314 (24 min 44 s), with `EXP036_REQUIRE_ALL=1`.
- **After the review's fixes** (comments, a test docstring, documents): the document-dependent tests plus `test_tools_leak_check`, `test_tools_status` and `test_tools_dry_run`, **290 passed** in each venv. Leak check `--all`: 366 file versions, **0 findings**, the 4 inherited host-name warnings. Only `tests` moved since the full run (`8ea52b9b…` → `fee7c32c…`, the docstring); `tools` is unchanged (`d2e85aa8…`).
- **`hash_tree --list`, final tree** (2026-10-06T23:03Z; these are the values the freeze fills in):
```
assets         42803f55e14c90c3bf7b229010b7aad76ba2135d51e4b0e83bd526fda47ff5a3
evalfw         e7ff393392d7e60d8e2ba81e76a70c12233da01511d8da2bb7de6ea751bde1bb
requirements   32488f23411026229ef831ef9ec475fb922d10fd24d6be062ab08f7d4df533e6
port           2c153357862f15b61f3cadaea4f436de2939567180bf5a01a304f3fa60e3f182
convert        99dfad12a4dde03dfa0355f94183953cd52fd846ad123d9f4adeb0f0ea25ef68
reference      85337ed7efeef6b0463f3560fff6da6496661a2fa7a14de5e50a1bce9e251bbd
gate           1d41328359b9c6af3c58b51894cfd1f38ac028f79c6866cfcf246d293ac8edbf
thresholds     e27bdc0650a8d072aa1709a583b0afcadb12172446aca312565be47253c1f4c4
gate_rules     5e9e1a530496e57657f1d1c2661d0992520a0f2b0717737e5b9cbb18c4a33cd0
gate_text      1a295dacf20d285512dcf6db755623e89053ec704abef2bd151682178d91b9e8
runner         439590de6a5c7971fc167e8918f9570f142446bd7046e49fac8c38cac8b45d74
tasks          306861c155653ddff747cc78c4ad5a49262c926b1d256890c9a7555a53d11334
manifest_rule  4c3fa6803e74b4a58f7282ed19965769f1de62d16d53979b7854a84b77f69857
plan_rules     2707ea2571cc122b7388f9c9794e3e1130aefdf13c6f844b7de449485bce878c
scorers        7d90acdc0b8b7aa4b4dd1161d4e9290e6f31bdbfee5bee898898e1eff52cd7d6
analysis       0a40ec19b38c43bed57df2e9c6d03fc1e9149159dbea172d921af4e7a6345187
bench          2f923d0b80cb357173ed5cdfefa6d7ef5b56fb21485cc288a86046c51c3d6709
tools          d2e85aa8e3e6c62bcbd4207ce87d3a568a91ae69da8431eef1e7bf334dcc4f0e
tests          fee7c32c84fabe4d65346309c43b42f50a928e1c5dd8b4e73fa609e027b93d83
env            8d97b3df8207fb8a7621977a52693b436c6cb267cee37f9042b76248bb78e6a9
```
- `gate`, `gate_rules`, `thresholds` and `gate_text` are unchanged since the W15 approvals. The sweep and its fixes moved `runner`, `tools` and `tests` only.

## Open items (main session and Andrei)

1. **Decision (d):** G4-F16's R3 ceiling (1e-2) has no real-weight value behind it; W15 estimates the risk from exp_036's records and recommends. *Done: kept, with the estimate disclosed; approved by Andrei at 18:41:39Z (W15-R3), who declined the reference-only R2F / R3 pass before the freeze.*
2. **W7's tiny-mode choices:** fp32 activations for the free-routing checks in tiny mode, and `TINY_NOT_APPLICABLE` including G5-R1 K8's R-greedy leg. *Stage 6: the bf16 run is recorded (W15-05); control 26 sits at 4.63 × its bound there (tiny margin 10 ×). Still for the main session: it was not among the decisions approved at 18:41:39Z.* *Resolved by the main session before the freeze: no change. The registered tiny margin is judged on the fp32 run, where 26 clears it at 1,408 ×. The bf16 value is disclosed in HYPOTHESIS ("Production precision"). Changing mutant 26 after seeing its value would be post hoc. The risk stated to Andrei: if 26 is uncaught on real weights, G5-R1 is INCOMPLETE and costs a cycle.*
3. **The switches** (*done in Stage 6, W15-01*). After W12's hand-off note, W6 sets `DOC_ASSERTIONS_ON = True` in `tests/test_gate_thresholds.py`, W7 (or W15) in `tests/test_integration_contracts.py`, and W10 sets `W12_HANDOFF = True` in `tests/test_tools_hash_tree.py`, `tests/test_tools_env.py` and `tests/test_tools_status.py`. W12 verified on a copy that all five pass with the switches on (below).
4. *Done in the pre-freeze housekeeping (also W15-09).* **`COPY_RECORD.json`** (the main session's, W0 being closed): add HYPOTHESIS.md, RUNBOOK.md, BUILD_SPEC.md, README.md and BUILD_LOG.md as new; the `diagnostics/vendor_spike/` and `diagnostics/tiny_stress/` files and `tests/fixtures/exp037_tiny_controls.json` as new; NOTICE's copy sha256 now differs from its source (status "to modify", owner W12); FixC's and Integrate's status moves (`scorers/score_all.py`, `tools/fidelity_reference.json`, the five decision-(c) tests, `tests/test_scorers_score_all.py`, `tests/test_tools_peer_check.py`) and `diagnostics/gemma_quant_check.py` as unchanged; the stage-1 new files (W3, W4, W8a) where not yet listed.
5. **`power.budget()`** reproduces exp_036's S1 hours (5.4 / 9.1 / 9.6) while HYPOTHESIS.md's "Sessions & budget" registers 6.2 / 9.2 / 9.7; the plan rule uses the measured S1 hours either way, but the projection tables differ. (Also W15-10.)
6. **The tripwire predicate** exists twice (`analysis/verdicts.py` with its 1e-9 tolerance; `gate/rules.py` with a bare `<`). (Also W15-10.)
7. **Small leftovers:** `analysis/__init__.py`'s docstring says "exp_036 analysis"; `env/versions.json` keeps the schema string "exp036 versions v1"; `g5_generation.py` still re-exports two rules functions; `analysis/exploratory.c1` reads a complete C1 NOT RUN file with a misleading reason text; four inherited files carry the mini's host name (item 5 of W14).
8. **The heavy tiny tests on the mbp.** RUNBOOK step 3 runs `test_gate_drivers_tiny.py` and `test_gate_end_to_end_tiny.py` on M5 numerics for the first time (only `test_gate_ftiny.py` is mini-only). Their margins are wide (the narrowest asserted one is decision (e)'s ratio > 3 at 6.41 and 11.93), but they have not run on an M5. W15-11 adds read-only checks on the mbp before go #1: the presence of exp_036's Gemma 4 bf16 folder (record D) and its download metadata, which RUNBOOK steps 0b and 9a assume.
9. **The W15 review's other minor issues**, which had no fix task:
   - W15-07: R-greedy is blocking but exempt in tiny mode. The review offers making it descriptive or pooling it with G5 greedy; it is kept and disclosed for now.
   - W15-08: P1(c) and P1(e) trust whichever commit the "Pre-registration commit:" line names. The review proposes requiring the named commit to be the one that removed the placeholder, with exactly one commit touching that line.
   - W15-12: G4-F32's F3 clause of 10 × Csort is always met by cross-implementation rounding (11–30 × Csort on tiny), so F3 rests on its ρ-growth clause. `f3_csort_factor` is still 10.
   - W15-13: decision (e)'s row says "no batched-path defect … can reach 30". The proposed wording is "is unlikely to reach 30 (the unrelated-context ceiling is 22.3 / 19.1)".
   - W15-14: one sentence on the combined exposure (estimated 25–40 % that a correct port does not PASS), adding that fix cycles help only real defects.
   - W15-15: cite exp_036's tree at the freeze (`3c99003`) next to the kit code's `222c845`, and exempt the Pre-registration line from the one-writer rule.
10. **The power log** (Stage 7):
    - *Done.* **The mbp smoke.** Its layout is in Stage 7 and matches the fixture's key names and types, so the fixture is not rebuilt; no poweravg document appeared.
    - *Done.* **The "power log" amendment type** (HYPOTHESIS, amendment rules: after S3 only, `tools/power_log.py` and its test only) was confirmed by Andrei's go (d), 2026-10-07T04:11:40Z.
    - **The review's minor items:**
      - P4: list `skipped` in `--check`; detect a running logger by file growth instead of the 30 s sample age; have Claude pass each step's raw file explicitly.
      - P5: a realistic crashed-then-resumed fixture; RUNBOOK steps 14 and 16 on restarting the logger after a reboot and passing every raw file of the session.
      - P6: end a closed orphan gate window at its last phase.
      - P7: extend the no-deciding-code scan to `tools/`.
      - P8, second half: `.plist` in `leak_check.REFUSED_SUFFIXES`.
      - P9: log step 9a, or list the unlogged steps and the mini as not measured and narrow the claim.
      - P10: a METHOD line saying windows nest and are never summed; per-component sampled time.
      - P11, rest: `status.py`'s record blocks and an explicit `git add` of `results/power/power_<UTC>.json` in steps 9–11; consider renaming the directory `results/energy/`.
    - The mini's 28.6 W CPU + GPU spot reading, which HYPOTHESIS quotes, is only in a scratch file. exp_036's committed `compute/COMPUTE.md` has only the 64.6 W system figure.
    - A raw log outside `$EXP036_WORK` and the home directory keeps its absolute `dir` in the record. The commit leak check catches it, and the RUNBOOK always writes under `$EXP036_WORK`.
11. *Done.* **`REVIEW.md`.** The name is dropped from BUILD_SPEC §1's tree and HYPOTHESIS's layout. As in exp_036, which has no REVIEW.md, the final review is recorded in this file's Stage 6.

---

## Annex A. W0r: runtime release-notes review (exp_037), 2026-10-06

*Reproduced verbatim from the main session's review, except that its headings are one level lower (source sha256 `624767f55dbed165e26a99e046210456539c7f938c77879073db1e83f275ea16`). It is the record of the runtime pin (decision F); HYPOTHESIS.md reproduces it as its Annex A.*

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
