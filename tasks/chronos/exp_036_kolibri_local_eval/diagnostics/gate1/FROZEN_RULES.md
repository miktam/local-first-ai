# Gate 1 diagnostics: frozen decision rules

*exp_036, gate run `20261005T050112Z` (K4 PASS, K8 FAIL). Written by the main session on the mini on 2026-10-05, before any stage of the diagnostic suite has run. It is committed in the same push as Amendment 8 (`amendments/8_process_20261005T084959Z.md`), and that commit's message records the sha256 of this file and of every stage script. Nothing here is a gate check, a threshold or a verdict. These rules map each diagnostic outcome to the action that may follow, and they fix that mapping before any outcome exists.*

**Source.** The rules come from §3–§4 of the diagnosis of the gate failure, in its revision 2. Revision 2 was written after a skeptic's review and an integrity review of revision 1. Revision 1's rules were withdrawn as outcome-shaped (§9 item 11), and none of them is used here. The rules are restated in full, so this file stands alone.

**Changes from the diagnosis text.** All of these were made before any stage runs.
1. **No external anchor.** Andrei approved the FP8 vendor anchor (diagnosis D4), then withdrew the approval the same day (Amendment 8). Every anchor branch is removed:
   - G3-D now leads only to "failure stands";
   - G3-1's anchor override and G3-2's "anchor 3–5 % lower" are gone;
   - the conditional reference fix "G3-D and the anchor reads ≥ 5 % lower" no longer exists.
2. **Completion rules** cover a missing, invalid or non-reproducing measurement (§1.4–§1.6). The diagnosis leaves these implicit.
   - A missing or invalid input leads only to "failure stands".
   - A non-reproducing measurement abandons a threshold-only option unless the new threshold clears the recorded value by more than the spread (§1.4, the diagnosis's own rule).
   - G5-N has no threshold to clear, so it requires stage 1 to reproduce the gate's G5 value exactly, on the same run that parts R and the decomposition examine. Otherwise G5's failure stands (§3).
   - G3 also requires the Q36-8 pass to reproduce the peers record the gate cites (§6).
3. **Numbers for terms the diagnosis leaves without one:**
   - "a step" at a boundary (G4-D1);
   - the form of the McNemar test (G4-D3);
   - the sign convention for an "excess" (G4-D6);
   - the 128-byte window attribution (G3-D1);
   - the Q36-8 reproduction tolerance (G3 validity).

   Five readings are also fixed:
   - G4-D1's buckets include T1-8;
   - in G4-D2, the 1e-6 bound applies to the port against itself, and chunk 64 against the reference takes the bug test's bounds (mean KL ≤ 1e-3, decisive top-1 ≥ 0.999). Read literally, the diagnosis's "KL > 1e-6 at any position" would also apply to chunk 64 against the reference. This reading is **looser** than that literal one;
   - a C1' value that would not change the recorded G4 verdict is outcome G4-2;
   - G3-D1 must hold on each of its four texts;
   - G5-D1's "2/364" is read as "more than 2 positions", so it holds at any position count.
4. **Types (stricter than the diagnosis).** The diagnosis types A3 and the M5-kernel workaround "K8 and K4 at B = 1" as gate fixes. By §1.10's definitions they are verdict-rule changes:
   - A3 fixes no defect, so it cannot have a test that reproduces one. It changes what failing means for G2.
   - At B = 1, `g5_batch_parity_K8` is still measured at B = 8 and still blocking. Removing it needs the same `run_gate.BLOCKING`, `plan_fix` and guard change as G5-B1.

   Each is its own amendment and needs Andrei's go. Only "a path that avoids the kernel", with G5 re-measured at B = 8, stays a gate fix.
5. **The fp32 alignment.** The bug test, the chunk-64 test and the descriptive `g4 fp32` pass run K8 with fp32 activations on exactly the dequantised K8 weights the reference reads (`diaglib.fp32_on_dequantised_weights`, §2 stage 3). The diagnosis writes `model.set_dtype(float32)` for chunk 64. On its own that cast changes the embedding's dequantisation:
   - on the tiny K8 build it gives a bug-test mean KL of 1.8e-3, over G4-D1's 1e-3 bound with no defect anywhere;
   - with the alignment it gives 8e-13;
   - `tests/test_convert.py::test_dequantised_mode_reads_the_mx_dequantize_values` confirms what the reference reads.

   Stage 1's part C keeps a plain `set_dtype`: it compares the port with itself.
6. **Locations.** Everything is under `diagnostics/gate1/`, with one script per stage, instead of `diagnostics/`.
7. **Labels.** In the diagnosis, "D5" in G5-N, in §0 item 7 and in D3 (b) means "stop and publish", which its §5 lists as **D8**. This file writes "stop and publish (D8)". The diagnosis's real D5 is copying K4 to the mini as a G5 discriminator. It was not decided and is not part of this suite.

---

## Summary

| check (gate record) | measured | limit | stage | outcomes, in evaluation order |
|---|---|---|---|---|
| `g5_batch_parity_K8` | mean KL 0.3073773544 (364 positions) | ≤ 0.1512924584 | 0, 1 | **G5-D** defect sign: failure stands; a gate fix or (M5 kernel, B = 1) a verdict-rule change only if localised. **G5-N** no sign, with exact reproduction: G5-B1, a verdict-rule change (D3). Otherwise failure stands |
| `g2_bf16_natural_selection` | 1 of 8,448 disagreements at 6.0777 σ_l (layer 20) | every disagreement < 6 σ_l | 2 (+3) | **G2-D**: failure stands; a gate fix only if localised. **G2-1**: A1, a threshold change. **G2-2**: A3, a verdict-rule change on Andrei's go. **G2-3**: failure stands |
| `g4_K8_backstop` | T9 `top1_decisive` 0.98889 (66 of 5,943 missed) | ≥ 0.99 (≤ 59 missed) | 3 | **G4-D**: failure stands; a gate fix only if localised. **G4-1**: C1', a threshold change. **G4-2**: failure stands |
| `g3_bpb_per_text` | T1 1.3106 | ≤ 1.2 | 4 | **invalid**: failure stands. **G3-D**: failure stands. **G3-1**: B1 permitted, a threshold change typed by Andrei. **G3-2**: Andrei declines B1, so the failure stands |
| `g3_bpb_vs_peers` | 1.2961 (0.91427 ÷ Q36-8 0.70540) | ≤ 1.25 | 4 | as `g3_bpb_per_text`; one outcome covers both keys |

Two corrections to `aborted/20261005T050112Z-gate/NOTE.md`:
- **G4.** The 8,192–16,384 bucket's mean KL (0.108) is descriptive and did not fail. `gate/checks/g4_e2e.py` blocks only on the T1-8 and T9 aggregates (mean KL 0.041 and 0.083, both passing).
- **G3.** The Kolibri side has no start token. The recorded rule scores NLL(ids[t+1] | ids[:t+1]) for t = 0 … T−2.

---

## 1. Ground rules

1. **No gate record.** No stage invokes `gate/run_gate.py` or writes under `results/gate/`.
2. **Not a fix cycle.** Andrei accepted the diagnostic route (D2, Amendment 8). It uses no fix cycle, no S1 hours and no gate record.
3. **Gate values stand.** A diagnostic value never replaces a gate value, and a favourable draw is never cited. Every rule below is applied to the gate record's failing values. The diagnostics decide only *why* a check failed and which action is permitted.
4. **Reproduction.** Four stages first reproduce the gate's own value:
   - stage 1 part B: G5 mean KL 0.3073773544462904;
   - stage 2: G2 `max_gap_over_sigma` 6.077697277069092 and the disagreement count;
   - stage 3, `g4 --arm K8 --act bf16`: T9 `top1_decisive` 0.98889449772842 (66 of 5,943);
   - stage 4: the reference bpb of T1–T6 from the reused dump.

   If a reproduction differs, the spread s = |diagnostic − gate| is recorded. A threshold-only option (A1, C1' or B1) is then permitted only if the recorded value clears the new threshold by more than s; otherwise it is abandoned, and the check's outcome becomes its "failure stands" branch.

   G5-N has no threshold, so the spread cannot protect it. It requires both:
   - `B_mean_kl` equals the record's 0.3073773544462904 exactly;
   - `B_instrumented_run_equals_gate_run` is true. The decomposition, the localisers and part R then examine the same batched draw that failed.

   Otherwise G5's failure stands (§3).

   Stage 4 also re-measures Q36-8, whose NLL enters G3-D1's window ratios. That pass must reproduce the peers record the gate cites (`results/peers_20261005T045649Z.json`): every T1–T6 bpb within 1e-4 of the record (`peer_bpb_reproduces_record`). Otherwise stage 4 is invalid (§6).
   - The two computations score the same per-token log-probs (`bench.kl_8v4.scored_chunks`, bf16-rounded logits). They differ only in summation precision: float32 sums in the record, float64 here. With a Qwen3-1.7B stand-in on the real T1–T4, stage 4 and `tools/peer_check.text_stats` differed by at most 2e-4 nats per text, under 1e-7 bpb.
   - 1e-4 bpb is about 0.1 % of Q36-8's smallest per-text bpb (T4, 0.086).
5. **Missing or invalid inputs.** A rule input can be missing, non-finite, or produced by a stage that did not complete or was declared invalid. In that case the check cannot reach any outcome other than "failure stands". No input is filled in by assumption.
6. **Re-running a stage.** A stage may be re-run only in two cases:
   - it crashed without writing its output;
   - a script defect is named and fixed. The defect must be shown independently of the values the stage produced (for example, a wrong index, or a crash in a later stage that reads its output).

   Both outputs, the fix and its reason go into the stage-5 report. A stage that completed is never re-run to replace its output.
7. **Evaluation.**
   - Rules are evaluated only at stage 5, after stages 0–4 have all completed. Andrei decides nothing between stages, because D3 was recorded before stage 1.
   - Defect signs are evaluated before any option. Branches are taken in the order written, and the first that applies is the outcome.
   - Signs are evaluated on all of their inputs. A localiser is never a defect test.
8. **What may be committed.** The restricted copies under `diagnostics/gate1/out/` may hold:
   - port-vs-reference and port-vs-emulation comparisons;
   - reference-only quantities;
   - aggregate gap statistics, peer numbers and kernel probes.

   These stay in `$EXP036_WORK/diag_gate1/` unless the gate passes: absolute K8 NLL or bpb, K8 recall counts, K8's own entropy or top-1 lead, and any other K8 quality number.

   K8 chat-wrapped NLL is never computed. The `.npz` files, `ids.json` and any token ids or text are never committed.

   **A disclosed limit.** A committed difference plus a committed reference value gives back the K8 value:
   - Per text, K8's bpb is the reference bpb plus `d1_k8_minus_ref_bpb`, to d1's float precision.
   - Adding the per-text `<|endoftext|>` Δbpb gives K8's prefixed bpb.

   This is accepted on the precedent of the gate record, whose `g4_K8_backstop` per-text `dnll_mean` (K8 minus reference) sits next to G3's reference bpb. The pooled relative `<|endoftext|>` effect would give the pooled K8 bpb directly, so it stays in the work file.
9. **Scripts and this file.** If a stage script and this file disagree on what a rule input means, this file wins, and the disagreement is a script defect. Found before any run, it is fixed before the commit. Found after a run, §1.6 applies.
10. **Who acts.**
    - A **threshold change** is typed by Andrei. It gives, for each key, the original value, the observed value, the diagnostic record (file and sha256), the frozen-rule outcome, the new value as the rule computes it, and the reason. Any number of keys go in one amendment. The companion files change with it:
      - `gate/thresholds.json`;
      - the BUILD_SPEC §7.1 block;
      - the `tests/test_gate_thresholds.py` pins;
      - `THRESHOLDS_SHA256`, and `GATE_CODE_SHA256` if `gate/` changes.
    - A **gate fix** is a code change in `gate/`, `runner/`, `port/` or `reference/` that fixes a localised defect. The main session pushes it as `amendments/<k>_gatefix_<UTC>.md`, with the test that reproduces the defect before the fix and passes after it. A change with no such test is not a gate fix. A reference fix must cite a vendor or vLLM line and may not touch `gate/thresholds.json` in the same amendment (HYPOTHESIS l.272).
    - A **verdict-rule change** changes what blocks or what failing means. There are three:
      - G5-B1 (§3);
      - A3 (§4, G2-2);
      - the M5-kernel workaround "K8 and K4 at B = 1" (§3).

      Each is drafted by the main session as **its own amendment**, so that no relaxation of what failing means rides inside another amendment. Each needs Andrei's go at the time; if he declines, that check's failure stands.
      - G5-B1 also carries his D3 choice. A3 and the B = 1 workaround carry no earlier choice of his.
      - Each is disclosed as a rule change made after the result.

---

## 2. Rule inputs

Each stage writes `$EXP036_WORK/diag_gate1/<stage>_<UTC>.json` and a restricted copy, `diagnostics/gate1/out/<stage>_<UTC>.json`. Every output holds a top-level object `rule_inputs` with the keys below. `stage5_report.py` reads only `rule_inputs` from the restricted copies, plus the committed gate record. Everything else in an output is descriptive.

**Shared definitions.**
- **Reference.** "ref" is the gate's fp32 reference, `reference.kolibri_ref` on the BF16 checkpoint, read from the gate's dump where one exists.
- **Decisive position.** One where the reference's top-1 log-prob exceeds its top-2 by ≥ 2.0 nats (`K8_backstop_decisive_lead_nats`).
- **KL(a‖b)** is in nats, per position, computed from fp32 log-softmax.
- **Emulation.** "emu" is the reference in its vendor-faithful bf16 mode (`--emulate-bf16`).
- **Buckets.** The T9 buckets are the gate's `t9_buckets`: 0–2,048, 2,048–8,192 and 8,192–16,384.

### Stage 0: `stage0_kernels.py`

kprobe2 logic, unchanged from sha256 `88c1ce6b…` (`88c1ce6bb2fa1b819c5790e5476f9a1cab37cd4b8410fa1ad0206d51c8c61d43`). The baseline is `kprobe2_m4pro.json` (`86bf6d7af34f81c019bc99a500a5278aad3da2cc86ed8afe4772eca57aee7a58`).

| key | meaning |
|---|---|
| `kprobe_verdict` | `baseline.verdict`: `"DEFECT"` or `"no chip-specific B=8 precision loss"` |
| `kprobe_defects` | `baseline.chip_specific_defects` (used for localisation only) |

### Stage 1: `stage1_g5.py` (K8; gate prompts, lengths, max_tokens and B = 8 as registered)

| key | meaning |
|---|---|
| `B_mean_kl`, `B_top1_dis` | the gate's `batch_parity`, bf16 as loaded (reproduction) |
| `B_instrumented_run_equals_gate_run` | true iff the instrumented copy of the loop (whose tokens and log-probs the decomposition, L3 and part R use) gives exactly `B_mean_kl` |
| `R_mean_kl_ref_single`, `R_mean_kl_ref_batched`, `R_mean_kl_ref_decode_tf` | the 12 sequences (prompt + batched tokens) teacher-forced through ref. Mean over all positions of KL(ref‖path), where path is the B = 1 teacher-forced port (prefill chunk 2,048, the gate's comparator), the B = 8 log-probs as generated, or the port's single-sequence teacher-forced decode |
| `R_n_decisive` | ref-decisive positions among them |
| `R_top1_decisive_single`, `R_top1_decisive_batched` | on ref-decisive positions, the fraction where the path's argmax equals ref's |
| `R_n_batched_wrong_single_right`, `R_n_single_wrong_batched_right` | on ref-decisive positions, the discordant counts |
| `C_mean_kl`, `C_top1_dis`, `C_n_positions` | the gate's `batch_parity` with fp32 activations on the real K8 weights (`model.set_dtype(float32)`, `MLX_ENABLE_TF32=0`) |
| `L1_rows_bitwise_equal`, `L1_max_pre_divergence_kl`, `L1_first_divergence_step` | 8 copies of T3[:700] at B = 8 against B = 1 through the same BatchGenerator, 24 tokens |
| `L2_max_kl` | B = 1 BatchGenerator against `decode_tf` on the same tokens |
| `L3_max_abs_dlogprob_seq0_seq8`, `L3_max_abs_dlogprob_seq3_seq11` | batched log-probs of sequences 0 and 8 (both T1[:37]), and of 3 and 11 (both T4[:1100]), over their common token prefix |
| `B_share_kl_top5`, `B_mean_kl_first_wave`, `B_mean_kl_mid_run`, `B_mean_kl_t0`, `B_mean_kl_t1_2`, `B_mean_kl_t3plus`, `B_mean_kl_prompt_ge_513` | concentration and classes, from the per-position decomposition |

### Stage 2: `stage2_g2.py` (T1–T8 rows as in the gate, plus T9 where stated)

| key | meaning |
|---|---|
| `reproduces_gate`, `port_max_gap_over_sigma`, `port_n_disagree_layer` | the gate's layer-20 statistic and disagreement count recomputed (reproduction). `reproduces_gate` is true iff the count, σ_l and the maximum all equal the record |
| `flip_text`, `flip_position`, `flip_position_class` | the layer-20 disagreement at 6.08 σ |
| `flip_e_attn_max_pct_16_20` | the flip token's e_attn percentile among all T1–T8 tokens, taken in each of layers 16–20; the maximum of the five |
| `flip_sigma_t_port`, `flip_sigma_t_emu`, `flip_r_t` | σ_t: the RMS over the 384 experts of the router-logit error at that token. r_t = σ_t(port) / σ_t(emu) |
| `r_p50_layer20`, `r_p99_layer20` | r over all T1–T8 tokens of layer 20 |
| `flip_gap`, `flip_gap_over_sigma_t_emu`, `flip_emu_flips_same` | the gate's gap for that disagreement, its ratio to σ_t(emu), and whether emu disagrees on the same (token, layer) |
| `M_emu`, `emu_n_beyond_6sigma` | emu's own gate statistic on T1–T8. M_emu is the maximum over the 50 layers of emu's max gap / σ_emu,l; the second key counts emu's disagreements ≥ 6 σ_emu,l |
| `tails.{att8,att29,moe13,moe30}.port_max_over_emu_max` | per-layer maximum of the branch error on the same rows (T1–T8; T9 for MoE 30), port ÷ emu |
| `tails.{…}.port_max_text`, `tails.{…}.port_max_position` | where the port's maximum is |
| `t9_disagree_frac_port.{bucket}`, `t9_disagree_frac_emu.{bucket}` | natural bf16 selection disagreement fraction on T9, by bucket (emu fed the fp32 h_l) |

### Stage 3: `stage3_g4.py`

| key | meaning |
|---|---|
| `bf16_T9_top1_decisive`, `bf16_T9_decisive_miss`, `bf16_T9_n_decisive` | `g4 --arm K8 --act bf16` against the fp32 dump (reproduction) |
| `top10_kl_positions` | the 10 positions of highest KL(ref‖K8 bf16) over T1–T8 and T9, as (text, position) |
| `bug_mean_kl.{T1-8,T9}` | the bug test: K8 with fp32 activations on exactly the dequantised K8 weights (see "fp32 on the dequantised weights" below) against `ref_deqK8_fp32` (the fp32 reference on the dequantised K8 weights) |
| `bug_top1_decisive.{T1-8,0-2048,2048-8192,8192-16384}` | the bug test, decisive top-1 per set |
| `bug_boundary_window_max` | the maximum, over T9 boundaries b, of the bug test's mean KL over positions b−32 … b+31. Boundaries: 2,048·k for k = 1 … 7 (8,192 included) and every document boundary inside T9 |
| `chunk64_max_kl`, `chunk64_n_decisive_top1_changes` | K8 fp32 on T1 and T3 at 520–1,100 and on T9 at 15,000–15,300: prefill chunk 2,048 against chunk 64, against decode, and chunk 64 against decode. The first key is the maximum per-position KL; the second counts top-1 changes where the chunk-2,048 lead is ≥ 2 nats |
| `chunk64_vs_ref_mean_kl`, `chunk64_vs_ref_top1_decisive` | chunk 64 against `ref_deqK8_fp32` at T9 15,000–15,300 |
| `E_T9`, `emu_T9_decisive_miss`, `emu_T9_n_decisive`, `emu_decisive_set_equals_gate` | `ref_deqK8_emu` against the fp32 dump on T9: E_T9 is its decisive top-1. The last key is true iff its decisive set is the gate's 5,943 positions |
| `mcnemar.{T9,0-2048,2048-8192,8192-16384}.{n10,n01,p_one_sided,port_miss,emu_miss}` | on ref-decisive positions: n10 counts port bf16 misses where emu hits, n01 the reverse. p is the exact one-sided binomial P(X ≥ n10 \| n10 + n01, ½), and 1 when n10 + n01 = 0 |
| `paired_dnll_T9.{mean,ci99_lo,ci99_hi}` | per position, NLL(port bf16) − NLL(emu) on T9. The CI uses `gate.checks.g3_oracle.block_bootstrap` with 64 blocks, B = 10,000, q = 0.005, and a fixed seed recorded in the output |
| `n_kl_port_gt5_emu_lt1` | positions over T1–T8 and T9 with KL(ref‖port bf16) > 5 nats and KL(ref‖emu) < 1 nat |
| `excess.{T5,T6}.{kl_port,kl_emu,miss_port,miss_emu}` | inside T9 minus standalone, on the same tokens: the mean KL, and the decisive miss rate |

**fp32 on the dequantised weights.** This applies to every `bug_*` and `chunk64_*` key and to the descriptive `g4 fp32` pass. K8 is loaded as the gate loads it, then `model.set_dtype(float32)` with `MLX_ENABLE_TF32=0`, then `diaglib.fp32_on_dequantised_weights`. That function keeps the quantised embedding's scales and biases at their stored bf16 (an exact round trip) and casts the embedding's output to fp32.
- **Why.** `QuantizedEmbedding` dequantises in its scales' dtype. A plain `set_dtype` turns the stored bf16 dequantisation, which is the value the reference's dequantised mode reads (`tests/test_convert.py::test_dequantised_mode_reads_the_mx_dequantize_values`), into an fp32 one, which is a different weight.
- **Size.** On the tiny K8 build, a plain `set_dtype` gives a bug-test mean KL of 1.8e-3 with no defect present. With the alignment it is 8e-13.
- **Linear layers** need nothing: `quantized_matmul` already dequantises in fp32.

Descriptive only: dNLL/KL with 95 % CIs, the entropy shift, K8-fp32 against the fp32 dump by bucket, and the web tail 13,399–16,384.

### Stage 4: `stage4_g3.py`

| key | meaning |
|---|---|
| `ref_bpb_reproduces_gate` | the reference bpb of T1–T6 recomputed from the dump equals the gate's (reproduction) |
| `d1_k8_minus_ref_bpb.{T1..T6}` | K8 bpb minus reference bpb, a difference (validity) |
| `mutant_dnll_p01.{mutant}.{T1,T5}`, `mutant_dnll_mean.{mutant}.{T1,T5}` | `gate.checks.ref_pass.mutant_nll` on T1 and T5 for the 7 registered mutants, scored with `gate.checks.g3_oracle.ref_mutant_nll` (48 blocks, B = 10,000, q = 0.01, the gate's seeds). The mutants are `one_plus_w_norm`, `qknorm_after_rope`, `renorm_topk`, `rope_on_full`, `rope_traditional`, `sigmoid_bias_select` and `swap_sandwich_norms`. Any further text is descriptive |
| `window_ratio.{T1,T2,T5,T6}.{p10,p50,p90,frac_kolibri_better}` | per 128-byte window of the text's UTF-8 bytes: the reference's bits ÷ Q36-8's bits. A token's NLL is attributed to the window holding its last byte, and windows where either model scores no token are dropped |
| `ref_bpb_from64_rel_diff.{T1..T6}` | (reference bpb from token 64 − full-text reference bpb) ÷ full-text reference bpb |
| `peer_bpb_reproduces_record`, `peer_bpb_max_abs_diff` | the Q36-8 pass's per-text bpb against the peers record the gate cites (`results/peers_20261005T045649Z.json`); true iff every T1–T6 difference is ≤ 1e-4 bpb (§1.4) |

Descriptive only:
- the reference's NLL on T4 inside Q36-8's recall windows;
- the `<|endoftext|>` effect as a Δbpb;
- τ\* of the reference and of Q36-8.

---

## 3. G5 batch parity (stages 0 and 1)

**G5-D: a defect sign.** The failure stands for this cycle. G5-D holds if any one of these is true:
1. `C_mean_kl` > 1e-4, or `C_top1_dis` × `C_n_positions` > 2 (that is, > 2/364): a mechanics defect.
2. `R_mean_kl_ref_batched` > 2 × `R_mean_kl_ref_single`: the batched path is worse, not just different.
3. `R_top1_decisive_batched` < 0.995 while `R_top1_decisive_single` ≥ 0.995.
4. `R_n_batched_wrong_single_right` − `R_n_single_wrong_batched_right` ≥ 3.
5. `kprobe_verdict` = `"DEFECT"`: an M5 kernel at Kolibri's shapes loses precision at B = 8.

After G5-D, the localisers decide *where* the defect is. They are L1, L2, L3, the classes, and concentration (the top-5 positions carrying ≥ 50 % of the KL sum). None of them is a defect test.
- **L1:** a pre-divergence KL well above 0 points to the M = 8 kernels; bitwise-equal rows point to padding, mixing or admission. A different but equally precise kernel can give an L1 up to about 5 × decode-vs-prefill, as on Gemma.
- **L2:** a value > 1e-6 means the batch cache classes compute differently even alone.
- **L3:** whether the first-wave and mid-run copies of the same prompt agree.

A localisation counts only when a test reproduces it:
- **An M5 kernel:** stage 0 lists the kernel in `kprobe_defects`. The stage-0 JSON goes into an upstream MLX report. There are two workarounds, each citing the kernel, with different types:
  - **a path that avoids the kernel,** with G5 re-measured at B = 8 on the next gate run: a **gate fix**. Its test runs the stage-0 probe's comparison on the kernel's path, where it fails, and on the path the runner then uses, where it passes;
  - **K8 and K4 at B = 1:** a **verdict-rule change**, its own amendment, on Andrei's go. `g5_batch_parity_K8` would still be measured at B = 8 and block, so this needs the same `run_gate.BLOCKING`, `plan_fix` and guard change as G5-B1.

  Stage 5 reports both, with their cycle costs, and chooses neither.
- **The port or the runner:** a head_dim-128 tiny parity test reproduces the class in bf16 and in fp32. It fails before the fix and passes after it. The places to look:
  - `runner/generate.py` batch construction;
  - the port's `make_cache` and masks under `BatchKVCache` / `BatchRotatingKVCache`;
  - mlx_lm `BatchGenerator`, citing the line.

  The consequence is a **gate fix** by the main session. K4's descriptive parity and behaviour change with it. A step-9 re-run is advisable after a runner change.
- **Not localised:** G5's failure stands. No amendment can relieve it. The D5 discriminator (K4 on the mini) would need Andrei's separate decision.

**Reproduction (§1.4).** G5-N can be reached only if stage 1 reproduced the gate's draw:
- `B_mean_kl` equals 0.3073773544462904 exactly;
- `B_instrumented_run_equals_gate_run` is true.

If either fails, or is missing, and no sign fired, G5's failure stands. D3 was agreed on the premise that the diagnostics examine the batched path that failed. A different draw cannot show that the failing draw was "different but not worse": a race-type kernel defect, one of the remaining candidates, is exactly the case that would not reproduce.

**G5-N: none of signs 1–5, and stage 1 reproduced the gate's draw.** The gate's 0.307 > 0.151 stands. The batched path is different but not worse. The consequence is the one Andrei chose before stage 1 (D3, "Run Kolibri at B = 1 (Recommended)"): **G5-B1**.
- It is a **verdict-rule change**: K8 and K4 run at B = 1, the consequence the peers' registered batched-path rule gives.
- Its gate code covers `run_gate.BLOCKING`, `plan_fix` and the guard.
- It is its own amendment, and a fix cycle under D1.
- The plan ladder drops to P2–P6 nominal, with STOP in the pessimistic case at assumed B = 1 rates of 45–90 tok/s (unmeasured).
- Amendment 7 criterion 4 then needs the cause stated on the model card.

**Failure stands** in every other case: G5-D not localised; no sign but no exact reproduction; Andrei declining a verdict-rule change; or a G5 consequence whose amendment does not fit the cycle budget (§8).

**What G5-B1 can still detect.** Nothing about batching: Kolibri never batches in a run. A B = 8 defect would still go to the model card and to an upstream report.

Optional and descriptive: Q36-8's batched check with fp32 activations. It decides nothing for Kolibri.

---

## 4. G2 bf16 natural selection (stage 2; sign 4 also reads stage 3)

**G2-D: failure stands; inspect.** G2-D holds if any one of these is true:
1. `flip_r_t` > `r_p99_layer20`.
2. `flip_position` is 0–1, 511–515 or 2,047–2,049 in its text, or `flip_e_attn_max_pct_16_20` ≥ 99.
3. `flip_emu_flips_same` is false and `flip_gap_over_sigma_t_emu` > 6.
4. For attention layers 8 or 29, or MoE layers 13 or 30: `port_max_over_emu_max` > 3, and the port's maximum is within ±2 tokens of one of `top10_kl_positions` in the same text.
5. For any bucket: `t9_disagree_frac_port` > 2 × `t9_disagree_frac_emu`.

If a G2-D sign is localised to a position class (0–1, 511–515 or a chunk edge) by a tiny G1 boundary test that fails before the fix, the consequence is a **gate fix** in the port's mask or cache at that boundary. Every check is then re-measured for both arms. Otherwise G2's failure stands.

**G2-1: A1, threshold change.** No G2-D sign, and `M_emu` ≥ 6.0: a correct bf16 implementation fails the registered rule itself.
- The new `bf16_selection_gap_sigma` = min(8.0, 1.2 × `M_emu`).
- Andrei types it: original 6.0, observed 6.0777.

**G2-2: A3, verdict-rule change.** No G2-D sign, `M_emu` < 6.0, and `flip_emu_flips_same` is true.
- A port disagreement that emu shares at the same (token, layer) is exempt from the σ rule.
- Every other disagreement keeps < 6 σ_l.
- **Type.** The diagnosis typed A3 a gate fix. It fixes no defect, so no test can reproduce one, and it changes what failing means for G2. It is therefore a verdict-rule change (§1.10): its own amendment, drafted by the main session, on Andrei's go. If he declines, G2's failure stands.

**G2-3: failure stands.** Every other case. In particular, if 5.07 ≤ `M_emu` < 6.0 and emu does not flip the token, the failure stands.

**What the relaxations can still detect.**
- **A1:** a single flip up to 1.2 × M_emu σ (≤ 8 σ) passes.
- **A3:** a flip the emulation shares at the same (token, layer) passes at any σ.

---

## 5. G4 K8 backstop (stage 3)

**G4-D: failure stands; a code question.** G4-D holds if any one of these is true:
1. **The bug test:**
   - `bug_mean_kl` > 1e-3 on T1-8 or on T9;
   - or `bug_top1_decisive` < 0.999 in any of the four sets;
   - or `bug_boundary_window_max` > 1e-3: a step within ±32 tokens of a 2,048·k chunk boundary, of 8,192, or of a document boundary.
2. **Chunk-64, port against itself:** `chunk64_max_kl` > 1e-6, or `chunk64_n_decisive_top1_changes` ≥ 1. Chunk 64 against the reference is judged by sign 1's bounds instead: `chunk64_vs_ref_mean_kl` > 1e-3, or `chunk64_vs_ref_top1_decisive` < 0.999. The 1e-6 bound applies where one code path computes the same thing two ways. The bug-test bounds apply where two implementations are compared.
3. **McNemar:** for T9 or any bucket, `p_one_sided` < 0.05 and `port_miss` > 1.25 × `emu_miss` in the same set.
4. **Paired dNLL:** the interval [`ci99_lo`, `ci99_hi`] excludes 0, and |`mean`| > 0.02 nats/token.
5. **Large-KL positions:** `n_kl_port_gt5_emu_lt1` ≥ 3.
6. **Excess inside T9:** for T5 or T6, and for the KL or the miss rate, the port's excess is > 0 and > 2 × max(emu's excess, 0).

A G4-D sign localised by a tiny G1 test at the boundary, with the bug test or chunk-64 test as evidence, leads to a **gate fix** in `port/kolibri1.py`'s long-context path:
- NoPE layers;
- cache growth;
- chunk edges.

Every check is then re-measured for both arms. Otherwise G4's failure stands.

**G4-1: C1', threshold change.** All of these must hold:
- no G4-D sign;
- `emu_decisive_set_equals_gate` is true, so `emu_T9_n_decisive` = 5,943;
- the new value clears the recorded 0.98889 by more than the reproduction spread (§1.4).

The new `K8_backstop_decisive_top1_min` = max(0.985, min(0.99, 1 − 1.25 × (1 − `E_T9`))).
- **When K8 passes.** The recorded K8 value passes iff emu misses ≥ 53 of the 5,943 positions (66 ÷ 1.25 = 52.8). If the computed value would not change the recorded verdict, the outcome is G4-2.
- **Andrei types it:** original 0.99, observed 0.98889 (66 / 5,943).
- **The key is shared.** It also applies to T1-8, where the recorded value is 0.99652; this is disclosed.
- **Where 1.25 comes from:** G2's port/emu per-layer error ratio (median 1.09, max 1.40) and the op-level rounding differences measured on the mini (≤ 1.4×).

**G4-2: failure stands.** Every other case.

**What C1' can still detect.** A port bf16 path up to 25 % worse than vendor-faithful bf16 in decisive misses passes, never below 0.985. Amendment 7 criterion 2 keeps G4's original bounds for the Hugging Face upload, so C1' does not unblock the upload.

---

## 6. G3 reference plausibility (stage 4), decided without an anchor

**Invalid: failure stands.** The stage is invalid if:
- |`d1_k8_minus_ref_bpb`| > 0.005 on any of T1–T6;
- `ref_bpb_reproduces_gate` is false; or
- `peer_bpb_reproduces_record` is false. G3-D1's window ratios divide by the re-measured Q36-8 bits, so a Q36-8 pass that reads higher than its record would lower them and make "no sign" more likely.

An invalid stage cannot show "no sign", so G3's failure stands.

**G3-D: misreading signs.** G3's failure stands, with no G3 amendment and no anchor route. G3-D holds if any one of these is true:
1. For **each** of T1, T2, T5 and T6: `p10` ≥ 1.1 and `frac_kolibri_better` < 0.10.
2. Any registered mutant has `mutant_dnll_p01` < 0.24 nats/token on T1 or on T5. B1 could not catch such a mutant there.
3. |`ref_bpb_from64_rel_diff`| > 0.05 on any of T1–T6.

**G3-1: B1 permitted.** The stage is valid and shows no G3-D sign. Andrei may type a threshold change, labelled a post-hoc gross-sanity bound:
- `ref_bpb_per_text_max` 1.2 → 1.5 (observed T1 1.3106);
- `ref_bpb_mean_vs_best_peer_max` 1.25 → 1.40 (observed 1.2961).

The amendment must state the blind spot in §9 item 7.

**G3-2: failure stands.** Andrei declines B1.

**What B1 can still detect.** A further uniform misreading of up to +0.24 nats/token, or +8.1 % relative, beyond the observed values passes. No local test excludes a misreading shared by port and reference of the size that would explain the failure: +9.2 % NLL takes T1 exactly to 1.2.

---

## 7. Stage 5: mapping and report (mini, minutes)

`stage5_report.py` reads the committed restricted copies and the gate record. It applies §3–§6 mechanically and writes `diagnostics/gate1/out/stage5_report_<UTC>.json` (JSON only). The report holds:
- **For each check:** every sign with its inputs, its value and whether it fired; the reproduction spread; the outcome (D / N / 1 / 2 / 3 / invalid); the frozen consequence and its type; and who acts.
- **The gate's own evaluation, re-applied.** The gate's own `evaluate` code is applied to the recorded measured values with only the permitted new thresholds, and the result is reported. A FAIL stays a FAIL. G5-B1 is reported as "non-blocking under a verdict-rule change", never as a pass.
- **The package and its cycle cost** under §8.

It searches for no passing rule and chooses nothing. Where a localisation leaves two workarounds of different types (§3, M5 kernel), it reports both with their cycle costs. If its reading of an input is ambiguous, the check's outcome is "failure stands" and the ambiguity is reported.

**Integrity.** Stage 5 refuses to run, and writes nothing, if any of these holds:
- its own sha256, or that of any other file in `diagnostics/gate1/*.py` and `run_stages.sh`, differs from `scripts_sha256` in the header of any stage output;
- this file's sha256 differs from `frozen_rules.sha256` in any stage header, or the gate record's from `gate_record.sha256`;
- the stage outputs mix modes;
- the kit's working tree is dirty;
- a stage has more than one output in `out/`, unless that stage's file is named explicitly together with the defect and the reason for the re-run (§1.6). The report then lists every output of that stage with its sha256, the file used, and the reason.

A smoke-test switch (`--smoke`) relaxes one check only, the dirty tree. It refuses to write into `out/` and marks its report as not for the record.

---

## 8. Cycle accounting under D1 (ii)

**The reading.** Andrei chose "(ii) Strict (Recommended)". The question put (ii) as: "Each typed amendment counts as a cycle (HYPOTHESIS: a threshold amendment "counts as a fix cycle"; the pre-registration outranks the RUNBOOK) — the next re-run is then the last."

This file applies it as follows. The qualifier in the first sentence is **the session's reading, not part of the option Andrei chose**: one fix cycle is one typed amendment that changes what the gate measures or how it judges. There are three such types:
- a threshold change, with any number of keys inside one amendment;
- a gate fix;
- a verdict-rule change.

Without the qualifier, Amendments 3–8 would each count, and no cycle would remain.

Each amendment has one type (HYPOTHESIS l.22). A threshold change is typed by Andrei and a gate fix is pushed by the main session, so the two never share an amendment. A reference fix may not touch thresholds in the same amendment (l.272).
- One gate-fix amendment may carry several independent fixes, as Amendment 1 did.
- Each verdict-rule change is its own amendment (§1.10).

At most two fix cycles are allowed (HYPOTHESIS l.272; RUNBOOK l.293).

**Cycles used so far: 0 of 2.**
- **Amendments 1–2 are not counted (D6).** D6 is the session's premise in the D1 question, not offered to Andrei as a choice. The question stated: "Either way, Amendments 1–2 (fixes before the first gate run) are not counted." He answered D1 with it stated.
  - Both are typed "gate fix". Counted under (ii), they would use both cycles and leave none.
  - The case for not counting them: they fixed step-3 tiny-gate test failures before any real gate ran, and the RUNBOOK counts fix cycles from the step-10 FAIL.
- Amendments 3–6 precede the first gate run and change no gate code, threshold or reference.
- Amendment 7 and Amendment 8 change nothing the gate measures or how it judges.

**Reading (iii)**, one cycle per threshold changed, was not offered to Andrei. The diagnosis lists it, notes that the l.797 precedent chose the conservative reading each time, and notes that the gate cannot pass under it: B1 alone changes two thresholds.

**Why every passable package needs exactly two amendments.**
- G3 is deterministic on the reused dump. It fails on any re-run unless B1, a threshold change, is typed. This suite produces no reference fix.
- G5 is expected identical on a re-run. It fails unless a gate fix or a verdict-rule change (G5-B1, or the M5-kernel B = 1 workaround) is made. A re-run that relies on a different G5 draw is excluded (§1.3).
- These are two amendments of different types.

So under (ii), with D6 and the qualifier, every passable package uses both cycles. The next gate re-run is the last, and any failure on it is the exp_036 result (RUNBOOK l.293; HYPOTHESIS "Gate failure or STOP"). D1's own option text says the same: "the next gate re-run is the final one, so it only happens once the diagnostics have settled every failing check."

| stage-5 outcome | amendments | cycles | next gate run |
|---|---|---|---|
| any check at "failure stands" with no localised, test-backed fix | none can make the gate pass | — | not useful; the route is stop and publish (D8), on Andrei's go |
| no defect anywhere: G2-1, G3-1, G4-1, G5-N | threshold change (A1, B1 ×2, C1'), plus verdict-rule change (G5-B1) | 2 | the last |
| as above, but G2-2 instead of G2-1 | threshold change, plus verdict-rule change (A3), plus verdict-rule change (G5-B1) | 3 > 2 | the gate cannot pass: stop and publish (D8) |
| G5-D localised and fixed by a gate fix (port or runner fix, or a path that avoids an M5 kernel, with G5 re-measured at B = 8), plus G3-1, with G2-1 / G4-1 where they apply | gate fix, plus threshold change | 2 | the last |
| G5-D localised to an M5 kernel, worked around by K8 and K4 at B = 1, plus G3-1, with G2-1 / G4-1 where they apply | verdict-rule change, plus threshold change | 2 | the last |
| either G5-D row above, with G2-2 instead of G2-1 | adds a verdict-rule change (A3) | 3 > 2 | the gate cannot pass: stop and publish (D8) |
| G2-D or G4-D fixed in the port, with G5-D fixed in the same gate-fix amendment, plus G3-1 | gate fix, plus threshold change | 2 | the last; every check re-measured for both arms |
| G2-D or G4-D fixed in the port, with G5-N or the B = 1 kernel workaround | gate fix, plus threshold change, plus verdict-rule change | 3 > 2 | the gate cannot pass: stop and publish (D8) |

Stop and publish (D8) is available at every point.

---

## 9. Disclosure block (the post's gate section, and every amendment that follows)

1. **Thresholds.** The original thresholds, the observed values and every relaxation (HYPOTHESIS l.272), each with the smallest defect the relaxed check still detects (§3–§6).
2. **Checked before commit.** Every amended threshold was checked offline against the recorded values before its amendment was committed. G3 is deterministic, so it passes by construction. G2, G4 and G5 pass by construction if they reproduce. Any non-reproduction and its spread are reported.
3. **Decisions made after the failure.** Every decision in Amendment 8 was made after the gate failure was known (gate end 05:57:03Z; answers 08:35:13Z) and before any diagnostic ran:
   - the diagnostic route (D2);
   - the cycle reading (D1, (ii));
   - the G5 consequence (D3).

   Three parts of the cycle accounting were not Andrei's choices:
   - **D6, not counting Amendments 1–2,** was the premise of the D1 question, not an option. Counted under (ii), they would leave no cycle.
   - **The qualifier** "that changes what the gate measures or how it judges" is the session's reading of (ii), not part of the option he chose. Without it, Amendments 3–8 would count and no cycle would remain.
   - **Reading (iii),** one cycle per threshold, was not offered. It is the conservative reading the l.797 precedent points to. Under it, the gate cannot pass.
4. **Constants set with the gate values known.** They were frozen before the diagnostic ran:
   - G2's 1.2 and 8.0;
   - G4's 1.25 and 0.985;
   - B1's 1.5 and 1.40;
   - the G5 defect-sign constants (1e-4, 2 of 364, 2×, 0.995, 3), taken from the mini nulls;
   - **the G2-D sign constants:**
     - r_t above layer 20's own p99;
     - the position classes 0–1, 511–515 and 2,047–2,049;
     - the 99th e_attn percentile;
     - gap/σ_t(emu) > 6;
     - the tail sign's 3× and ±2 tokens of G4's top 10;
     - 2× emu's T9 disagreement fraction;
   - **the G4-D sign constants:**
     - the bug test's 1e-3 and 0.999;
     - the ±32-token step window at 1e-3;
     - chunk 64's 1e-6;
     - McNemar's p < 0.05 and 1.25×;
     - the paired dNLL's 99 % interval and 0.02 nats/token;
     - ≥ 3 positions at KL > 5 with emu < 1;
     - 2× emu's excess;
   - **G2-D4's 3× in particular.** It is new in revision 2. It was chosen with the record's port/emu tail ratios known: attention 8 at 1.72×, attention 29 at 2.24×, MoE 13 at 1.13×, all below 3×. Stage 2 recomputes these on the same rows from deterministic inputs. So for those three tails the sign's ratio condition is known to be false before any run, and the sign can fire only through MoE 30 on T9, where emu has never been measured. It is in effect a one-tail test;
   - this file's completions: the exact McNemar test, the excess sign, the window attribution, the Q36-8 reproduction tolerance of 1e-4 bpb, and the five readings listed at its head (one of which, G4-D2's chunk-64-against-reference bound, is looser than the diagnosis's literal text);
   - the fp32 alignment of the bug and chunk-64 tests. It was chosen after a tiny-build smoke measurement showed that a plain `set_dtype` would fire G4-D1 with no defect (1.8e-3 against 8e-13; §2 stage 3).

   This file's sha256 is in the commit that adds it.
5. **Amendment 7's timing (D7).** Amendment 7 (06:40:12Z, commit a666f3e) postdates the gate result (05:57:03Z, record 5c6a6e8). Its criterion 4 clause "G5 batch parity is resolved or documented" was written with G5's failure known. Its "before any result" means before any scored result.
6. **The Hugging Face upload.** Amendment 7 criterion 2 keeps G4's original bounds for the upload, so no G4 relaxation unblocks it.
7. **G3 without an anchor.**
   - (a) No local test excludes a uniform misreading shared by port and reference of the size that would explain the failure: +9.2 % NLL takes T1 to exactly 1.2 (the ratio would then be 1.187). On T3 that is about 0.15 nats/token, under a third of the subtlest registered mutant.
   - (b) B1 passes a further uniform misreading of up to +0.24 nats/token (+8.1 % relative) beyond the observed values.
   - (c) The FP8 vendor anchor is the only test with power against this class. Andrei approved it and withdrew the approval the same day ("No need to rent anything"), and it was not run.
   - (d) The only downstream check is H2, the comparison with the vendor's public scorecard. A gross shared misreading would be expected to show there as a large shortfall; H2 cannot show a small one.

   G3 is one of only two gate checks independent of the shared specification, and the post gives it low confidence.
8. **Verdict-rule changes, if used.** Each is a rule change made after the result, in its own amendment, on Andrei's go.
   - **G5-B1, or the M5-kernel B = 1 workaround:**
     - K8 and K4 run at B = 1, as the peers whose batched path failed do;
     - Kolibri never batches in a scored run;
     - the B = 8 cause goes on the model card (Amendment 7 criterion 4) and, if a kernel is at fault, into an upstream report.
   - **A3:** a G2 disagreement that the emulation shares at the same (token, layer) passes at any σ.
9. **Beyond the per-layer table.** The diagnostics went beyond the RUNBOOK's "diagnoses from the per-layer table" to new real-weight measurements on the same texts. The records are listed with their sha256.
10. **The diagnostic route.** It is a diagnostic, not a fix cycle, accepted by Andrei after the failure. The precedent, Amendment 6's diagnostic, concerned a peer.
11. **Withdrawn rules.** Revision 1's outcome-shaped rules were withdrawn after an integrity review and are not used:
    - A2;
    - B2;
    - C1 (E_T9 − 0.005);
    - G5-M / W / F;
    - G2's 5.1 switch point (which was 6.08 ÷ 1.2);
    - a G3 rule with no branch in which the failure stands.
12. **Reference results on other models.** The mini nulls (Qwen3-4B, Qwen3-1.7B, Gemma-4 26B-A4B) and their scripts.
