# RUNBOOK step 3 — unit tests failed on the mbp (2026-10-04)

Host: MacBook Pro, Apple M5 Max (`applegpu_g17s`), 128 GB, macOS 27.0, Python 3.12.13. `env/setup.sh` installed the pins (mlx 0.31.2, mlx-metal 0.31.2, mlx-lm 0.31.3). Kit at `ec03dcf`.

Command, as RUNBOOK step 3 gives it:

```
EXP036_TOK="$EXP036_MODELS/Kolibri-1-BF16" EXP036_REQUIRE_ALL=run-host "$PY" -m pytest -q -rs tests
```

Result: **42 failed, 1139 passed, 8 skipped** in 207 s. All 8 skips are `build-host only:`. The runbook stopped at step 3; nothing was converted, gated, run or scored.

There are two independent causes:
- **A.** 41 fp32-parity failures, caused by TF32 on the M5 GPU.
- **B.** One leak-check finding that only appears where the withheld data is present.

Everything below was measured read-only on the mbp, with no model weights loaded.

## A. 41 fp32-parity failures: MLX uses TF32 for fp32 on the M5 GPU

### Cause (confirmed)

mlx 0.31.2 has an environment variable `MLX_ENABLE_TF32`, read as `get_var("MLX_ENABLE_TF32", 1)` in `include/mlx/utils.h`. It is **on by default**. On the M5 GPU, the neural-accelerator kernels then compute several fp32 operations by truncating the operands to 10 mantissa bits (round toward zero) and accumulating in fp32.

**Affected:**
- fp32 `matmul`/`addmm` with M ≥ 2;
- fp32 sorted `gather_mm` (the SwitchGLU path when T·k ≥ 64);
- fp32 SDPA with q_len > 1.

The relative L2 error against float64 is about 7.7e-4, against about 3e-7 on the CPU stream. It agrees with a TF32-truncation emulation to 1.5e-7.

**Not affected:** M = 1 GEMV (decode), single-query SDPA, unsorted `gather_mm`, `quantized_matmul` (including with fp32 activations), `softmax`, `logsumexp`, `sum`, `fast.rms_norm`, `layer_norm`, and the CPU stream.

The build host (M4 Pro, same mlx, per `tests/INTEGRATION_LOG.md:3`) has no such kernels, which is why the suite passed there. As a result, prefill runs in TF32 and decode in exact fp32 on this host, so port-vs-port tests fail as well as port-vs-reference tests. The tests affected that way are cached vs full, B = 4 vs B = 1, and chunked vs one-shot.

### Check

`MLX_ENABLE_TF32=0` makes **all 41 pass**, and all 261 tests in the 10 affected files. The variable is read once per process, before the first GPU matmul, so setting it later in the process has no effect. There is no Python API for it. With TF32 off, large fp32 GEMMs match the CPU (Accelerate) result exactly, and they run slower.

### Observed vs bound, per file

| Test file | Bound | Observed on the mbp |
|---|---|---|
| `test_cache_decode.py::test_fp32_cached_matches_full_forward` (16) | rel ≤ 1e-4 and the same argmax | 14 cases gross (7e-2 to 2.5e-1; routing flips amplify the ~1e-3 arithmetic error; with forced experts it is 1.2–1.6e-3), 2 marginal (7.3e-4, 8.9e-4) |
| `test_convert.py::test_g2q_head_and_embedding_on_dequantised_weights[8bit,4bit-vendor]` | rel L2 ≤ 1e-5 | 4.65e-4 and 4.92e-4 (G2's own `g2q_head_rel_l2_max` is 1e-3); the quantised-head variants pass |
| `test_gate_checks.py::test_batch_parity_on_the_port` | mean_kl < 1e-8 | 3.55e-7 (top1_dis 0) |
| `test_gate_drivers_tiny.py::test_tiny_gate_passes_on_the_port` | gate PASS | **K8 FAIL**, K4 PASS — see below |
| `test_gate_port_mutants.py::test_port_passes_the_fp32_bounds` | `fp32_forced_rel_err_max_max` 1e-3 | vendor 1.43e-3, pattern5 1.24e-3 (pattern5 dis_frac 6.9e-4) |
| `test_gate_port_mutants.py::test_selection_mutant_fails_natural_selection` | precondition: mutant worst ≤ 1e-3 | fails on the port's own error (1.43e-3 / 1.24e-3); the mutant's target check would still catch it (dis_frac 0.545 / 0.907 vs 0.2) |
| `test_integration_contracts.py::test_peer_batched_path_runs_through_the_gate_g5_functions` | mean_kl ≤ 1e-4 | 1.11e-4 |
| `test_port_ref_cache_batching.py` (3) | 1e-4 absolute on log-probs | 3.67e-3 and 4.36e-3; the argmax agrees |
| `test_port_ref_hooks.py::test_port_branches_match_reference_branches_fp32` (2) | per-token rel L2 ≤ 1e-4 | 1.0e-3 to 1.4e-3 (r_attn, r_moe) |
| `test_port_vs_ref.py` (8) | rel ≤ 1e-4 and the same argmax | vendor 7.5e-4 to 1.4e-3; pattern5 cascades from routing flips (up to 4.6e-1) |
| `test_runner_tiny.py` (3) | exact tokens; atol 2e-5 | B4 vs B1 diverges at token 21/28, rotated-cache admission at 32/40; teacher-force max abs 2.8e-3 |

The **tiny gate fails K8** on five checks, all marginal to moderate:
- `g2_fp32_forced_branch`: median 8.4e-4 vs 1e-4, max 1.30e-3 vs 1e-3.
- `g2_fp32_natural_selection`: 10 violations of 12,800 pairs.
- `g2_head`: 1.80e-3 vs 1e-3.
- `g5_decode_vs_prefill`: KL 6.0e-4 vs 1e-4, dis 0.034 vs 0.002. The floor is tiny because both of its prefills run in TF32.
- `g5_batch_parity_K8`: top1_dis 0.0098 vs 0.002.

This suggests step 10 on the real model would hit the same checks. `tests/INTEGRATION_LOG.md:181` claims "≥ 50x headroom" for the fp32 assertions; that does not hold on this GPU.

How to handle TF32 for the gate, the reference, the tests and the scored runs is the main session's decision. This note only records the facts.

## B. One leak-check failure: `tests/test_tools_leak_check.py::test_the_tools_area_is_clean`

```
FINDING [withheld_option] tools/shingles.py:64 — matches a withheld 3-word option, hash 303c5e7ff1a6 (text not shown)
```

**Mechanism.**
- `tools/withheld_shingles.sha256` does not exist yet; step 7 builds it. So `load_withheld()` falls back to `collect_from_sources($EXP036_DATA)`, which on this host reads the real GPQA, gpqa-multilingual `deu`, AIME-DE and RGB data. The check reports `withheld_check='on'`.
- Line 64 lies inside the `GENERIC_OPTIONS` constant (lines 62–68). A 3-word window on that line equals a withheld option.
- The exemption skips a window only when it equals a `GENERIC_OPTIONS` entry exactly, so the generic-phrase list trips the check on itself.

**Why the build host passed.** It has no withheld data and no shingle file, so the check is skipped there. Pointing `EXP036_DATA` at an empty directory reproduces that pass here.

**It persists after step 7.** `build_manifests.withheld_texts()` hashes the same option, and the step-7 public-text filter (LICENSE and the gate texts) does not contain it.

**What it blocks.**
- **Today:** only step 3. The RUNBOOK's commit check (`--range @{u}..HEAD --staged`) and the installed pre-push hook scan only new lines, and they pass.
- **More widely:** `leak_check.py --all --no-gpqa-source` on this host exits 1 with 30 `withheld_option` findings in 22 files, from 6 hashes: `7c8f50592903` (17 locations), `134b4e8753a4` (4), `639a58cf46f2` (3), `745a17b337b4` (2), `c761a01453b0` (1) and `303c5e7ff1a6` (1). All are 3-word GPQA options.
- **After step 7:** `134b4e8753a4` and `639a58cf46f2` drop out, because they occur in the public gate texts. The other four stay. A later commit that adds a line containing one of those phrases would then be blocked by both the commit check and the hook. That would hit the sign-off, amendments, records or notes.

**Allowlists.** The existing mechanisms do not cover withheld rules. `ALLOWLIST_FILES` is empty, `ALLOWLIST_DIRS` covers only the personal-pattern rules, and `BINARY_ALLOWLIST` is empty. None was used.

**Not in this note, on purpose.** The mbp can map each hash to its source rows. That mapping is not written here: together with the public kit line, it would reveal benchmark answers. Andrei can pass it on privately if the main session needs it.

## Failed tests (pytest `-rf`)

```
tests/test_cache_decode.py::test_fp32_cached_matches_full_forward  ×16  [vendor|pattern5]-[seed0|seed1]-[decode-from-short|decode-from-long|chunks-below-window|chunks-above-window]
tests/test_convert.py::test_g2q_head_and_embedding_on_dequantised_weights[8bit-g64-vendor]
tests/test_convert.py::test_g2q_head_and_embedding_on_dequantised_weights[4bit-g64-vendor]
tests/test_gate_checks.py::test_batch_parity_on_the_port
tests/test_gate_drivers_tiny.py::test_tiny_gate_passes_on_the_port
tests/test_gate_port_mutants.py::test_port_passes_the_fp32_bounds[vendor]
tests/test_gate_port_mutants.py::test_port_passes_the_fp32_bounds[pattern5]
tests/test_gate_port_mutants.py::test_selection_mutant_fails_natural_selection[vendor]
tests/test_gate_port_mutants.py::test_selection_mutant_fails_natural_selection[pattern5]
tests/test_integration_contracts.py::test_peer_batched_path_runs_through_the_gate_g5_functions
tests/test_port_ref_cache_batching.py::test_left_padded_batch_of_four_equals_single
tests/test_port_ref_cache_batching.py::test_refill_rotated[65]
tests/test_port_ref_cache_batching.py::test_refill_rotated[513]
tests/test_port_ref_hooks.py::test_port_branches_match_reference_branches_fp32[vendor]
tests/test_port_ref_hooks.py::test_port_branches_match_reference_branches_fp32[pattern5]
tests/test_port_vs_ref.py::test_fp32_logits_match_reference  ×4  [vendor|pattern5]-[seed0|seed1]
tests/test_port_vs_ref.py::test_fp32_hidden_states_and_routing_per_layer  ×4  [vendor|pattern5]-[seed0|seed1]
tests/test_runner_tiny.py::test_B4_equals_B1_greedy_in_fp32
tests/test_runner_tiny.py::test_rotated_sliding_caches_with_mid_run_admission
tests/test_runner_tiny.py::test_teacher_force_logprobs_chunked_equals_one_forward
tests/test_tools_leak_check.py::test_the_tools_area_is_clean
```

Skipped (all `build-host only:`): `test_bench_support.py:201` ×4 (mini upstream gemma/Qwen files), `test_tasks_no_withheld_text.py:122` and `test_tasks_vendored_evalfw.py:146` (no eval-framework v0.14.2 checkout), `test_tools_peer_check.py:55` and `:71` (build-time peer files).

## State of the mbp

- Steps 0–2 are done: downloads complete, 0d hygiene done (High Power on AC), git identity set, pre-push hook installed.
- The runbook is stopped at step 3; nothing was written to `results/`.
- The mbp session pulls and reruns step 3 once the main session pushes a fix.
