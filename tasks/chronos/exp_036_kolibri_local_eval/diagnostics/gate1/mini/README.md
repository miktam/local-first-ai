# exp_036 gate 1: the mini's diagnosis and bug-hunt evidence

This folder holds the scripts and result files that the main session produced on the Mac mini (Apple M4 Pro, MLX 0.31.2, mlx_lm 0.31.3, `MLX_ENABLE_TF32=0`) on 2026-10-05 for gate 1 of exp_036 (gate 20261005T050112Z, K8 FAIL). They were made in the session's scratchpad. They are published here because DIAGNOSIS revision 2 (§5 D8, §6 item 9) and `../FROZEN_RULES.md` §9 item 12 promise "this diagnosis, and the mini nulls" (rev 2 §5 D8), "the mini nulls (§A) and their scripts" (rev 2 §6 item 9) and "the mini nulls (Qwen3-4B, Qwen3-1.7B, Gemma-4 26B-A4B) and their scripts" (`../FROZEN_RULES.md` §9 item 12), and because `../BUGHUNT.md` cites results that existed only in the scratchpad.

**Nothing here decides any verdict.** The gate verdict is the gate record's: K8 FAIL. Stage 5 applied the frozen mapping in `../FROZEN_RULES.md` to the mbp's outputs in `../out/`. Its outcomes were G5-D, G2-D, G4-D and G3-D, and K8 stays FAIL (`../out/stage5_report_20261005T130616Z.json`). The files below are reference results on other models, experiments on tiny random checkpoints, and the record of the reasoning. None of them is a gate record or a diagnostic-stage output.

The work falls into two phases (file times, UTC, 2026-10-05):

- **Before stage 0** (06:22–08:17Z). This is after the gate failed (05:57Z) and before Andrei's answers (08:35Z) and the stage runs (from 11:48Z). It covers revision 1 of the diagnosis and its probes (`g5/`, `g3/`, `g2g4/`, `vendor_noise/`), the probes for the skeptic's review (`skeptic/`), the mini nulls (`nulls/`), and revision 2 (`diagnosis/`).
- **After stage 5** (13:12–14:25Z). This is the bug hunt behind `../BUGHUNT.md` (`bughunt/`).

## Layout

| folder | scratchpad source | phase | contents |
|---|---|---|---|
| `diagnosis/` | `gatediag/DIAGNOSIS.md` → `DIAGNOSIS_rev2.md`; `gatediag_rev/DIAGNOSIS_v1_backup.md` → `DIAGNOSIS_rev1_withdrawn.md` | before stage 0 | the diagnosis, revision 2 (frozen rules) and revision 1 (**withdrawn**, see below) |
| `nulls/` | `gatediag_rev/` | before stage 0 | the mini nulls of DIAGNOSIS rev 2 §A: e1 precision nulls (Qwen3-4B, Qwen3-1.7B: bf16 / q8 / q4 against fp32) with `e1_analyse_q4b_q17.txt` (the output of `e1_analyse.py`, written at staging), e1b τ\*, e2 real-weight batch-parity nulls (Qwen3-4B q8, Gemma-4 26B-A4B MoE), e2b localiser, e3 Gemma MoE activation null, e4 rounding points, kprobe2 on the M4 Pro |
| `g5/` | `gatediag_g5/` | before stage 0 (rev 1) | g5diag.py and the tiny-port G5 sweep (15 run files, plus `sweep63_ratios.json` for all 63 runs, written at staging), and the smoke tests of the rev-1 mbp script `g5_parity_diag_mbp.py` (superseded by `../stage1_g5.py`) |
| `g3/` | `gatediag_g3/` | before stage 0 (rev 1) | the prefix-rule proxy on Qwen3-4B (`proxy_qwen3_4b.json`); `g3_diag_mbp.py` and its stand-in smoke `standin.json`, which holds **no Kolibri numbers** (see "Fields named after Kolibri" below) |
| `g2g4/` | `gatediag_g2g4/` | before stage 0 (rev 1) | the rev-1 mbp script `diag_g2g4.py` (superseded by `../stage2_g2.py`, `../stage3_g4.py`), the noise-model fit `g2_tail*.py` (rev 1's P(trip) ≈ 0.2–0.96, withdrawn in rev 2 §1.1), T9 construction checks, and a tiny smoke output (synthetic tiny texts) |
| `skeptic/` | `gatediag_skeptic/` | before stage 0 | the probes run for the skeptic's review of rev 1. `kprobe.py`'s bf16-scale reference floored every qmm comparison at ≈ 1.6e-3 and was superseded by `nulls/kprobe2.py` |
| `vendor_noise/` | `gatediag_vendor/` | before stage 0 (rev 1) | the rev-1 mbp script "V" `diag_vendor_noise.py`, its tiny smoke `tiny_diag.json`, MLX variance probes, and `bias/fetch.py` (its fetched outputs are excluded) |
| `bughunt/cache/` | `bughunt_cache/` | after stage 5 | the first bug-hunt pass (exp1–7, symbolic cache check, RoPE and SDPA probes). It printed only; its arrays are excluded |
| `bughunt/g5bf16/` | `bughunt_g5bf16/` | after stage 5 | G5 bf16 first-wave probes (p1–p5, e1–e6), the real-dimension reproduction harness `rep.py` / `mk.py` with results, `kprobe3` (prefill shapes) on the M4 Pro, and `g5_ab_mbp.py` (scratch version) |
| `bughunt/reference/` | `bughunt_reference/` | after stage 5 | the reference and the port against a float64 dense implementation at 16k (`f64_dense.py`, `longctx_truth.py`, `forced.py`, `angle_attrib.py`; the forced-routing and angle reports), RoPE precision probes, and the Gemma-4 chunking/perturbation chaos run `gemma_chaos.py`, whose output is published only as the position-free `gemma_chaos_16384_summary.json` (written at staging) |
| `bughunt/rope/` | `bughunt_rope/` | after stage 5 | RoPE micro-probes (r1, r4–r6), tiny 16k runs (r2), and the routing-chaos sweep on sharpened tiny models (r3, `sweep/`) |
| `bughunt/verify/` | `bughunt_verify/` | after stage 5 | the clean re-runs behind each BUGHUNT claim (v1–v12), the full kit test runs with and without the vLLM-angle RoPE, and the scratch RoPE port `kit_rope/port/kolibri1.py`. It is the kit's port with only the RoPE angle changed; the rest of the two kit copies is excluded |

File names inside each folder are unchanged. `README.md` and `MANIFEST.json` were written when the folder was staged. So were three summary files, each marked `"derived": true` in MANIFEST with what it was derived from:

- `nulls/e1_analyse_q4b_q17.txt`;
- `g5/sweep63_ratios.json`;
- `bughunt/reference/gemma_chaos_16384_summary.json`.

## Revision 1 is withdrawn

`diagnosis/DIAGNOSIS_rev1_withdrawn.md` is published for transparency only. Its outcome-shaped rules were withdrawn after the integrity review and are not used anywhere (`../FROZEN_RULES.md` §9 item 11):

- A2 and B2;
- C1 (E_T9 − 0.005);
- G5-M / W / F;
- G2's 5.1 switch point;
- a G3 rule with no branch in which the failure stands.

Revision 2 is the diagnosis from which the frozen rules came. The withdrawn file opens with a two-line withdrawal notice, which was added at staging (MANIFEST `staging_edits`, B1). Below the notice, the text is unchanged apart from one path substitution (K2).

Both revisions refer to scratchpad paths: `scratchpad/gatediag_rev/` is `nulls/` here, and `scratchpad/gatediag/` is `diagnosis/`. Revision 1 cites the vendor's technical report as "report p.N" (p. N/189 in the kit's convention), with one-line paraphrases. It does not quote the report.

## Path placeholders

Absolute paths in scripts, JSON and markdown were replaced. Apart from these substitutions and the rev-1 notice above, no file was changed. `MANIFEST.json` gives every included file's `source_sha256` (the scratchpad original) and `sha256` (this copy). `sanitised: true` marks the 74 files that differ. The rules were applied in this order:

| id | replaced | with | occurrences (files) |
|---|---|---|---|
| S1 | the session scratchpad root (`/private/tmp/…/scratchpad`) | `$SCRATCH` | 21 (21) |
| S2 | the same root in its `/tmp/…` form | `$SCRATCH` | 0 |
| K1 | `<home>/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval` (the kit) | `$KIT` | 47 (46) |
| K2 | `~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval` | `$KIT` | 2 (2) |
| M1 | `<home>/models/exp036-mini` (the mini's models directory) | `$EXP036_MODELS` | 9 (6) |
| M2 | `~/models/exp036-mini` | `$EXP036_MODELS` | 0 |
| M3 | `<home>/models/exp036` (the mbp's models directory) | `$EXP036_MODELS` | 0 |
| M4 | `~/models/exp036` | `$EXP036_MODELS` | 0 |
| H1 | any other `<home>/` prefix (here the Hugging Face cache) | `$HOME/` | 6 (6) |
| H2 | `~/.cache/` | `$HOME/.cache/` | 2 (1) |

`<home>` stands for the mini user's home directory.

**`$EXP036_MODELS` has two meanings in this folder.**

- **The mini's models directory (`exp036-mini`).** This applies to the nine M1 substitutions, which are in six files (MANIFEST `substitutions_applied.M1`): `g2g4/t9_build.py`, `g2g4/t9_match.py`, `nulls/texts.py`, `g5/smoke_mbp.py`, `bughunt/reference/gemma_chaos.py` and `bughunt/verify/gemma_chaos_copy.py`. It also applies to this README's own use in `$EXP036_MODELS/venv312`.
- **The kit's own environment variable.** The other occurrences were already in the sources before staging. There, `$EXP036_MODELS` is the kit's variable from RUNBOOK, which on the mbp points to its models directory. They are `vendor_noise/diag_vendor_noise.py` line 7 and the rev-1 mbp commands in `diagnosis/DIAGNOSIS_rev1_withdrawn.md` (lines 410–414, counted with the notice).
- **`g5/smoke_mbp.py`.** This script also sets the environment variable `EXP036_MODELS` to its own `fake_models/` directory for the script it runs.

Several scripts carry a placeholder inside a string literal (for example `K = "$KIT"`). Before such a script runs, the placeholder must be replaced, or the script edited to read the environment. The script hashes that the DIAGNOSIS revisions cite are of the originals, which MANIFEST gives as `source_sha256`. For example, `e2_batch_null.py` is cited as e6afc2d5, but the published copy hashes to edc99020 because of K1.

## How the scripts were run

- **Working directory.** Each script ran from its own scratchpad folder, and many read their siblings by relative path.
- **Python.** The mini venv `$EXP036_MODELS/venv312` (mlx 0.31.2, mlx_lm 0.31.3) with `MLX_ENABLE_TF32=0`. The kit was imported from `$KIT` at a666f3e, plus `diagnostics/` for the bug hunt.
- **Models.**
  - Qwen3-4B, Qwen3-1.7B and Gemma-4 26B-A4B (mlx-community OptiQ 4-bit) came from the Hugging Face cache. Each result's `model` field gives the snapshot revision.
  - The mini has no Kolibri weights. Every Kolibri-shaped run uses a tiny random checkpoint.
- **Tiny checkpoints.** The weights, shard indexes, tokenizer copies and port copies are not published. Each checkpoint's `config.json` is included as its record. Builders:
  - `bughunt/cache/build.py` → `ck_real_s1`, `ck_real_s3`;
  - `bughunt/verify/build_ck.py` → `ckv_s1`, `ckv_s3`;
  - `bughunt/reference/build_tiny.py` → `ckpt_k16_L10`;
  - `bughunt/rope/r3_chaos.py` → `ckpt_chaos_*`; `bughunt/rope/r2_tiny16k.py` → `ckpt_h128_w513_L10`;
  - `g2g4/build_tiny.py` → `tiny/Kolibri-1-{BF16,MLX-8bit-g64,MLX-4bit-g64}`;
  - `vendor_noise/tiny_setup.py` → `tinyroot/` (through the kit's `tests/test_gate_drivers_tiny.build_tiny_models`);
  - `g5/smoke_mbp.py` → `fake_models/Kolibri-1-MLX-8bit-g64` (the real tokenizer was copied in; that copy is not published);
  - `g5/ckpt_{vendor,w513,w513h128,pattern5,p5h128}` were built in the session with the kit's `tests/tiny_checkpoint.write_tiny_checkpoint`. No build script survives, so `config.json` is the only record of them.
- **Inputs that are not published and never will be.**
  - `texts.json`: decoded T1–T9 including the FineWeb-2 web text. `nulls/texts.py` builds it, and e1, e1b, e2 and e3 read it.
  - The real T9 ids, `$SCRATCH/work/gate_texts/T9.ids.json`. They are read by `bughunt/reference/{longctx_truth,forced,angle_attrib,gemma_chaos}.py` and `bughunt/verify/gemma_chaos_copy.py`.
  - `g2g4/t9.json`.

  Runs that read these inputs reproduce only after the texts are rebuilt from the kit (`gate/build_gate_text.py` with the pinned FineWeb-2 parquet).
- **Outputs that are not published.** Four outputs of those runs list document-start positions that were derived from the real T9 ids:
  - `bughunt/reference/out_ckpt_k16_L10_16384/report.json`, the free-routing report of `longctx_truth.py`;
  - `run16k.log`, that run's console log;
  - both copies of `gemma_chaos_16384.json`.

  They are excluded, and MANIFEST lists their sha256. The Gemma result is published as `gemma_chaos_16384_summary.json`. It keeps every value except the position lists and keys and the window position counts. Its per-window maxima are kept as sorted lists.
- **Print-only scripts.** Several scripts printed their results to the session and wrote no file. Their numbers are marked **none** below.

## Fields named after Kolibri that are not Kolibri numbers

- **`g3/standin.json`.** This is `g3_diag_mbp.py` in stand-in mode, with `k_dir` = Qwen3-1.7B and `q_dir` = Qwen3-4B.
  - Every `kolibri*` field (`kolibri_pooled_bpb`, `texts.*.kolibri.*`) and every `k_*` field of `t4_recall_probe` is **Qwen3-1.7B**.
  - Every `q36_8*` field and every `q_*` field is **Qwen3-4B**.
  - Its T5 and T6 are stand-ins: the files it read were sha256-identical copies of the public T3 and T2. That is why its T5 values equal its T3 values and its T6 values equal its T2 values.
  - None of these is a Kolibri or Q36-8 bpb or NLL. While the gate fails, DIAGNOSIS rev 2 §4 keeps every K8 bpb and NLL out of the record, and nothing here breaks that.
- **`g5/smoke_out/` and `g5/smoke_out2/`.** Every `K8*` key and file name refers to the tiny random arm `g5/fake_models/Kolibri-1-MLX-8bit-g64`. That arm has random weights with the real vocabulary size and tokenizer, and its T5, T6 and T9 were replaced by seeded random ids. None of it is a Kolibri number.
- **Every other Kolibri-shaped run.** Each one uses a tiny random checkpoint, including the "K8 build" tiny of `bughunt/verify/v12_k8_forced.py`.

## What backs `../BUGHUNT.md`

Status: **here** = a file in this folder; **out/** = the committed mbp outputs in `../out/`; **none** = printed to the session only, with no file anywhere.

### §1 Port against itself (G4-D2): routing chaos

| claim | status | file → field |
|---|---|---|
| mbp: prefill 64 vs 2048 mean KL 0.052, max 12.45; prefill 2048 vs decode 1.6e-9 | out/ | `stage3_g4_20261005T124342Z.json` → `chunk64.ranges.T9.pairs.prefill2048_vs_prefill64`, `…prefill2048_vs_decode` |
| `mx.fast.rope` bitwise identical under every chunking and with array offsets | here | `bughunt/rope/r1_out.json` → `mlx_{scalar,transposed,arrayoffset}_chunk*.bitwise_equal_to_mlx_scalar_chunk2048` (all true), `bf16_input_chunk*_bitwise_eq_chunk2048`. The same check printed by `bughunt/verify/v1_rope.py` (a) and `bughunt/rope/r6_batch_offsets.py` has no file |
| symbolic cache/mask check: 0 wrong visible key sets over 32,842 queries (chunks 64, 513, 514, 2048, up to 16,421 tokens) | **none** | `bughunt/verify/v6_symbolic.py` prints only (32,842 = 2 × 16,421 queries per schedule). The first pass, `bughunt/cache/symcheck*.py`, also printed only |
| fp32 SDPA at the real shape within 5e-6 of float64 | **none** | `bughunt/verify/v5_sdpa.py` prints only |
| quantised matmul rows bitwise equal at M = 64 and M = 2048 | **none** | `bughunt/verify/v8_qmm.py` and the GEMM line of `v5_sdpa.py` print only |
| forced routing: prefill 64 vs 2048 ≤ 5.9e-8 mean KL (max ≤ 1.2e-6), on the tiny model and the K8 build with fp32 activations | partly here | tiny model: `bughunt/verify/v3_ckv_s3_16384.json` → `"base_p2048\|base_p64".block_mean` ≤ 3.4e-8, `block_max` ≤ 1.2e-6. The K8-build run (`v12_k8_forced.py`) printed only, so **5.9e-8 is in no file** |
| free routing on the same model: mean KL 0.044 (0.061 on the K8-build tiny), growing from 3.9e-5 in the first 2048 block | partly here | `bughunt/verify/v4_ckv_s3_16384.json` → `"base_p2048\|base_p64".range_mean` 0.0437 (positions 15,000–15,300) and `block_mean[0]` 3.9e-5. **0.061** (`v12_k8_forced.py`) is in no file. Control: in `v4_ckv_s1_16384.json` (q/k norms not sharpened) the port against itself, `"base_p2048\|base_p64"`, has 0 router flips and block-mean KL ≤ 3.3e-9. The port against the reference there, `"ref\|base_p2048"`, still has 582 flips and a block max of 0.09 |
| mature MoE: Gemma 4, chunk 64 vs 2048 mean KL 0.0095 at 8k–16k with 4 decisive top-1 changes; one weight × (1 + 2^-22) gives 0.0032 | here, with a mismatch | `bughunt/reference/gemma_chaos_16384_summary.json` → `"8192-16384".A_vs_B_prefill64` (0.00954, 4). **0.0032 is the 2,048–8,192 bucket's** `A_vs_C_perturbed` (0.00317, 2 decisive). At 8,192–16,384 the perturbation gives 0.00064 (1 decisive) |
| consequence: the frozen G4-D1/D2 constants | — | `../FROZEN_RULES.md` |

### §2 Port against reference at long context (G4-D1)

| claim | status | file → field |
|---|---|---|
| mbp: bug test mean KL 0.002 (T1–8) and 0.015 (T9); reference vs port decode 0.076 at 15,000–15,300; window max 5.89 at 3,911 | out/ | `stage3_g4_20261005T124342Z.json` → `dnll_over_kl.bug.*`, `chunk64.ranges.T9.ref_deqK8_fp32_vs_decode.mean_kl`, `bug_test_boundaries` |
| the reference's query chunking, sliding mask at large offsets, NoPE layers and RoPE positions match a dense float64 implementation at fp32 round-off over all 16,384 positions | here | `bughunt/reference/out_ckpt_k16_L10_16384/report_forced.json`: with routing forced to the float64 selection, the reference against float64 has mean KL 4.4e-10 and max 3.7e-9. The reference with `attn_chunk` 64 agrees with the default to 5e-13. `report_angles_shapes.json`: float64 with vLLM's fp32 angles against the forced reference is ≤ 4.8e-12 in every bucket. The free-routing `report.json` of the same run is not published (see "Outputs that are not published"). The reference's `attn_chunk` against dense float64 (`bughunt/verify/v7_sdpa_ref.py`) printed only |
| only 2 of the 15 document windows stand out (3,911 and 14,239) | out/ | `stage3_g4` → `bug_test_boundaries`. BUGHUNT §6.6 corrects the count against this same record |
| Gemma 4 shows the same concentration at document windows under a pure rounding change | here, with a mismatch | `gemma_chaos_16384_summary.json`. The concentration appears only under arm B, chunk 64 against 2048, which changes only the summation order. There, `doc_windows.A_vs_B_prefill64.mean_kl` is 0.0288 against `non_doc_ge2048.A_vs_B_prefill64.mean_kl` 0.0061, and five windows in `doc_start_window_max_kl_AB_sorted_desc` lie at 0.14–2.49. Arm C is the weight perturbation that `gemma_chaos.py` itself calls "a pure rounding-size perturbation". Under C the document windows are *quieter* than the rest: `doc_windows.A_vs_C_perturbed.mean_kl` 4.2e-4 against 1.77e-3. C's one elevated window (0.21) follows a `"\n\n"` inside T3, not a document start (the script marks every Kolibri `"\n\n"` id, paragraph breaks included) |
| MLX's exp2-form inv_freq differs from `1/θ^(2j/D)` by −1 to +8 fp32 ulps | **none** (related file here) | `bughunt/rope/r5_invfreq.py` printed it. The nearest file, `r1_out.json` → `inv_freq.exp2_form_vs_ref_ulps`, gives −2…+8 for the numpy exp2 candidate, not for MLX's recovered values |
| the rotated vectors differ by up to 3.3e-3 at position 15,000 | **none** | `bughunt/verify/v1_rope.py` (b) printed it. Files with related statistics: `r1_out.json` (unit probe, 9.8e-4 at 15,000–15,301) and `r4_out.json` |
| against float64, MLX and vLLM about equally accurate (9.2e-4 against 4–7e-4 rad) | **none** | `v1_rope.py` (c) printed it. `r1_out.json` gives a different statistic, the maximum over positions ≥ 12,288 (MLX 1.33e-3, vLLM fp32 9.7e-4 rad) |
| forced routing: port vs reference grows from 9.8e-7 to 3.4e-4 per block, max 1.6e-3; with vLLM-angle RoPE ≤ 6.4e-7 and flat | here | `bughunt/verify/v3_ckv_s3_16384.json` → `"ref\|base_p2048"` and `"ref\|rope_p2048"` (`block_mean`, `block_max`) |
| free routing: RoPE is one seed of the chaos | here | `v4_ckv_s3_16384.json` → `"ref\|base_p2048"`, `"ref\|rope_p2048"`, `"rope_p2048\|rope_p64"` |
| candidate change: unit difference at 15,000+ from 2.7e-3 to 4.8e-7 | **none** | `bughunt/verify/v2_rope_freqs.py` printed it. `r4_out.json` → `fix_a_fast_rope_freqs` is a related probe (≤ 1.2e-7 against the reference) |
| no test regressions | here | `bughunt/verify/baseline_full_tests.txt` and `rope_full_tests.txt` both show 20 failed, 1,219 passed and 35 skipped, with the same 20 failures (environment: git identity, fixtures, preflight). The changed port is `bughunt/verify/kit_rope/port/kolibri1.py` |

### §3 G5, bf16 batched vs single

| claim | status | file → field |
|---|---|---|
| mbp: fp32 C 1.5e-5; KL(ref‖batched) 0.299 vs KL(ref‖single) 0.069; first wave 0.469, mid-run 0.009; the 1,100-token row 0.41 vs 0.0095 | out/ | `stage1_g5_20261005T114806Z.json` → `C_decomposition_fp32act`, `R_reference_teacher_forcing` (0.2985 / 0.0686), `B_decomposition_bf16.by_class`, `B_per_sequence` |
| symbolic check of the gate's batch generator and admission loop: 0 wrong visible sets over 12,544 queries | **none** | `bughunt/verify/v9_g5_symbolic.py` prints only |
| bf16 SDPA bitwise unchanged when masked pad content is scaled ×1e4 | **none** | `bughunt/verify/v10_masked.py` and `bughunt/g5bf16/p5_masked_garbage.py` print only |
| M4 Pro: every part of the first-wave path bitwise row-independent in bf16 and q8 | here, with a nuance | `bughunt/g5bf16/kprobe3_m4pro.json` → the `neighbour_*` probes (a valid row is unchanged when only the pad rows' input changes) and the `sdpa_prefill_*` / `rope_*` probes are all bitwise (`any_not_bitwise: false`). p1–p4 printed only. The same file shows that the B = 8 call is not bitwise equal to the B = 1 call at short rows:<br>• q8 k / o / shared projections, at 0.50–1.0× the B = 1 error;<br>• the fp32 router, at 1.5–3.4× (flagged);<br>• the sorted SwitchGLU, up to 1.24× (0.0054 against 0.0044; flagged).<br>So "row-independent" holds, but "identical to B = 1" does not. BUGHUNT §6.6 now records this for the projections, the router and the SwitchGLU |
| real-dimension analogue (E 384, q8): batched/single 1.08–1.14, against 4.35 on the mbp | here | `classes.all.kl_tb / kl_ts` in `bughunt/g5bf16/r_e384_l5_q8.json` (1.085), `r_var.json` (1.081), and `bughunt/verify/rep_e384_q8_b2.0.json` (1.085) and `rep_e384_q8_b20.0.json` (1.138). 4.35 is 0.2985 / 0.0686 from `../out/stage1` R |
| remaining candidate and the confirmation scripts | committed | `../confirm/`. `kprobe3_prefill.py` and `kprobe3_m4pro.json` are identical to `bughunt/g5bf16/`. `g5_ab_mbp.py` and `mbp_verify_rope_chaos.py` were revised before commit, so the scratch versions here differ |

### §4–§5

The G2-D and G3-D numbers come from `../out/stage2_g2_*.json` and `../out/stage4_g3_*.json` (for example `mutants.T5.min_p01` 0.087). §5 adds no new numbers.

### §6 Confirmation on the real K8 (added to BUGHUNT after this folder's runs)

- **Where the numbers come from.** Every M5 Max number in §6 is from the committed mbp outputs `../confirm/out/{rope_chaos,kprobe3_m5,g5_ab}_20261005T145122Z.json`, whose sha256 are listed in BUGHUNT §6.
- **What this folder backs.** Only the M4 Pro column of §6.4 is the mini's. It comes from `bughunt/g5bf16/kprobe3_m4pro.json` (identical to `../confirm/kprobe3_m4pro.json`):
  - B = 8 relative error 0.0054;
  - B = 1 relative error 0.0044–0.0054;
  - the `neighbour_switchglu_q8_sorted` rows are bitwise unchanged.
- **Not in any file.** The M4 Pro's macOS version (macOS 26) is not recorded in the probe file, as §6.4 itself says.
- **What §6.6 corrects.** It corrects §1 (decode's shared history; call-shape invariance of the q8 matmul; the Gemma "0.0032" and "(fp32, the same T9 text)"), §2 (the document-window count; the Gemma concentration; the RoPE accuracy comparison), §3 (fp32 "exact"; "not reproduced on the M4" rests on random or tiny weights; row-independence against B = 1), §4 (G4-D5 and G4-D6 not examined) and §5 (G5 is not chaos; the forcing caveat). The corrections that rest on files here agree with them, including the three points under "Discrepancies with `../BUGHUNT.md`" below.

## What backs DIAGNOSIS revision 2 (the mini's numbers)

| claim | status | file → field |
|---|---|---|
| §A1: KL ranges, point dNLL and point dNLL/KL per text (Qwen3-4B, Qwen3-1.7B; bf16, q8, q4) | here | `nulls/e1_q4b.json`, `nulls/e1_q17.json` → `texts.*.{bf16,q8,q4}.{kl,dnll}` |
| §A1 and §1.3: the 95 % CIs on dNLL/KL, the T9 buckets (0–2k / 2–8k / 8–16k), decisive top-1 (0 / 7,998; 0.9787, 170 / 7,998; 8–16k 0.9751), and the Qwen rows of the §1.3 table | here (output re-run at staging) | `nulls/e1_analyse_q4b_q17.txt`, the output of `nulls/e1_analyse.py q4b q17` (see "About `e1_analyse_q4b_q17.txt`" below). It reproduces every figure that rev 2 cites:<br>• Qwen3-4B q8: T1 −3.70 [−6.03, −1.82], T2 +2.86 [+0.83, +5.70], T9 −1.14 [−1.92, −0.38];<br>• Qwen3-4B q4: T9 0.34 [0.26, 0.42] and 8–16k 0.25 [0.14, 0.35];<br>• decisive top-1: bf16 and q8 0 / 7,998, q4 170 / 7,998 (0.9787), q4 8–16k 0.9751;<br>• the §1.3 rows: q8 KL 0.0017 / 0.0014 / 0.0018, q4 KL 0.144 / 0.138 / 0.173;<br>• Qwen3-1.7B, per text: q4 0.49–0.67, bf16 +2.4 to +4.5 |
| §A2: τ\* | here | `nulls/e1b_tau_q4b.json`, `nulls/e1b_tau_q17.json` |
| §A3 table and the first two §1.4 rows (0.98 / 1.18 / 0.85, 0 / 146; 1.01 / 1.40 / 0.84, 0 / 216); identical B8 rows; B = 1 BatchGenerator exact | here | `nulls/e2_q4b_q8.json`, `nulls/e2_gemma_moe.json` → `mean.*`, `flips_where_single_leads_2` / `n_single_leads_2`, `identical_B8_rows_bitwise_equal`, `B1_batchgen_vs_decode_tf_mean_kl`. e2's `identical_B8_vs_B1_mean_kl` (5.55 for Gemma) compares beyond the first greedy divergence and is superseded by e2b |
| §A3: Gemma pre-divergence KL 1e-7 to 0.09 per step | **none** | `nulls/e2b_localiser.py` prints only |
| §1.3 Gemma row (0.0132 / 0.0125 / 0.0083, 3 / 8,669); dNLL/KL on T3 −2.55 and T6 +1.38 with CIs | here | `nulls/e3_gemma_moe.json` |
| §A4: rounding points | here | `nulls/e4.json` |
| §A5 and §1.4: kernels at decode shapes, router 1.15e-7 vs 8.3e-8, SDPA ×1.42 | here | `nulls/kprobe2_m4pro.json` (identical to the committed `../kprobe2_m4pro.json`). The 2-pass SDPA curve is in `skeptic/sdpa2_m4pro.json` |
| §1.4 tiny-port row: 63 runs, median 1.11 (0.23–2.38) | here | `g5/sweep63_ratios.json` → `summary.ratio_gate_kl_over_floor`, median 1.108, range 0.231–2.383, over all 63 runs. Rev 1's position-matched figure, 0.38–3.01, is `summary.ratio_gate_kl_over_matched_floor`. Each run's value is listed with its file's sha256. 15 of the run files are in `g5/` (`r_*_{bf16,q8,q4}_real.json`: median 1.10, range 0.67–1.91). The 48 seed replicates s1–s4 (`gatediag_g5/s/`, about 5.0 MB) are excluded for size only. The 63 runs are bf16, q8 and q4 (rev 1: "Tiny bf16/q8/q4 sweep"), although the rev-2 table row says "bf16" |
| §1.4: "batched further from truth in 42–48 %" | **none** | `skeptic/t0probe2.py` and `skeptic/t0sweep.py` print only |
| §1.1, §1.2 and §A7 facts; §A6 tails; G8 0.61; K8 6.1 | committed records | gate record 5c6a6e8 (`phase2.json`, `_layers.csv`) and the peers record, not this folder |
| §1.3: vLLM line citations (`ir_layernorm.py` l.53–58, `layernorm.py`) | — | upstream vLLM v0.29.0. The scratchpad's vendored copies (`gatediag_vendor/v0290/`) are excluded |

### About `e1_analyse_q4b_q17.txt`

- **How it was made.** The file is the standard output of `python3 e1_analyse.py q4b q17`. It was run in `$SCRATCH/gatediag_rev` with the mini venv (numpy 2.5.3) when this folder was staged. The session's own printout had not been saved.
- **Its inputs.** The script reads the per-position arrays `e1_{q4b,q17}_T*.npz`. These are excluded: the kit's leak check refuses `*.npz`, and DIAGNOSIS rev 2 §4 lists `.npz` as never committed. They hold float32 and bool arrays only, with no ids and no text, and MANIFEST gives their sha256.
- **Why it is deterministic.** The bootstrap is seeded (`default_rng(36)`), so the output is fixed for this argument order.

## Discrepancies with `../BUGHUNT.md`

Three points where a BUGHUNT §1–§3 sentence and the files here disagreed. BUGHUNT §6.6 now corrects all three; they are kept here as the record of what the files show.

1. **§1, the Gemma "0.0032".** It is the 2,048–8,192 bucket under the weight perturbation (arm C), not 8k–16k. At 8,192–16,384 the perturbation gives 0.00064 (`gemma_chaos_16384_summary.json`).
2. **§2, "Gemma 4 shows the same concentration at document windows under a pure rounding change".** The concentration shows under chunking (arm B), not under the weight perturbation that `gemma_chaos.py` calls a pure rounding-size change (arm C). Under arm C the document windows are quieter than the rest. See the §2 table above.
3. **§3, "every part of the first-wave path bitwise row-independent".** This holds, but the B = 8 call is not bitwise equal to B = 1 at short rows (`kprobe3_m4pro.json`). §6.6 corrects this for the q8 projections, the fp32 router (1.5–3.4×) and the sorted SwitchGLU (up to 1.24×).

Separately, the numbers marked **none** above were printed by their scripts and never written to a file. Their scripts are in this folder.

## Excluded

`MANIFEST.json` lists every file of the source folders: 229 included as files, plus the three summaries written at staging (232 in all), and 1,120 excluded. Each excluded file has its sha256, byte size and reason. The 23 `__pycache__`/`.pyc` files are given only as a count. The exclusions are:

- **Kit copies (701 files).** These are two verbatim copies of the kit tree (`bughunt_verify/kit_base`, `kit_rope`). Each file is sha256-identical to the same path under `$KIT`. `kit_rope` differs only in `port/kolibri1.py`, which is included.
- **Environment files (2).** These are the `env/exp036.env` files. They were never opened, hashed or copied, so their sha256 is null.
- **Tiny-checkpoint files (140).** These are 46 weight files, 71 metadata files and 23 port copies from the checkpoint directories. Each checkpoint's `config.json` is included.
- **Arrays (202).** These are `.npy` / `.npz` arrays, which the kit's `tools/leak_check.py` refuses.
  - Several of them also hold id arrays: random ids, or T9-derived positions in the Gemma chaos files.
  - The arrays of the 16k reference runs and of the Gemma chaos runs can be regenerated only with the real T9 ids or the T9 web text.
- **G5 seed replicates (48).** These are the JSON files in `gatediag_g5/s/`, about 5.0 MB. They are excluded for size only, and their content is safe to publish. `g5/sweep63_ratios.json` carries their values and sha256.
- **Rule-1 material (6).** This is `texts.json` (web text), `g2g4/t9.json` (real T9 ids), and the four stand-in T5/T6 `.txt` files.
- **T9-derived position lists (4).** These are `bughunt/reference/out_ckpt_k16_L10_16384/report.json`, `bughunt/reference/run16k.log`, and both copies of `gemma_chaos_16384.json` (see "Outputs that are not published").
- **Third-party material (7).** This is the vLLM v0.29.0 copies; `bias/index.json`, `bias_shards.json` and `bias_values.json`, fetched from the model repo; and a copy of the Kolibri tokenizer.
- **Others (10).** These are:
  - a tiny-gate record used as a smoke fixture, kept out so that no gate-record lookalike sits beside the real records;
  - tiny reference dumps;
  - the tiny smoke run's `ids.json`.
