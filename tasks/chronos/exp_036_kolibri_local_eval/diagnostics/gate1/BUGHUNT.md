# Gate-1 bug hunt: findings (2026-10-05)

*Main session, on the mini (M4 Pro), after stage 5. Stage 5 applied `FROZEN_RULES.md` to the mbp diagnostics (`out/stage0–4_*.json`), and the outcomes were G5-D, G2-D, G4-D and G3-D. No amendment package exists, and K8 stays FAIL (`out/stage5_report_20261005T130616Z.json`). Nothing below changes that verdict. This file records what the defect signs turned out to be, so that the exp_036 write-up and the exp_037 gate rest on evidence. The hunt ran on tiny checkpoints with Kolibri's real attention layout (48 q / 4 kv heads, head_dim 128, window 513, full attention at i % 5 == 4), on float64 dense implementations, and on Gemma 4 26B-A4B as a mature, correct MLX MoE. Kolibri's weights are only on the mbp; the confirmation runs in `confirm/` test the main claims there.*

## 1. Port against itself: chunk 64 vs chunk 2048 at T9 15,000–15,300 (G4-D2). No defect: routing chaos

- **Measured on the mbp** (fp32, dequantised K8): prefill 64 against prefill 2048 gives mean KL 0.052 and max 12.45, while prefill 2048 against decode gives 1.6e-9. In the test, decode prefills ids[:15000] in the same 2048 chunks, so it shares that whole history with the prefill-2048 run.
- **Primitives are exact.**
  - `mx.fast.rope` gives bitwise-identical output under every chunking and with array offsets.
  - A symbolic check of the rotating and full caches and masks found 0 wrong visible key sets over 32,842 queries, for chunk schedules of 64, 513, 514 and 2048 up to 16,421 tokens.
  - fp32 SDPA at the real shape is within 5e-6 of float64.
  - The quantised matmul rows are bitwise equal at M = 64 and M = 2048.
- **Forcing the expert choices removes the effect.** With each path's top-6 forced to the same ids, prefill 64 against prefill 2048 is flat at ≤ 5.9e-8 mean KL (max ≤ 1.2e-6). That holds on the tiny model and on the K8 build with fp32 activations.
- **Free routing on the same defect-free model reproduces the real signature.** Mean KL is 0.044 (0.061 on the K8-build tiny), growing from 3.9e-5 in the first 2048 block.
- **The mechanism.** A different chunking changes fp32 summation order at the 1e-7 level. Top-6-of-384 sigmoid routing with large expert biases has near-ties, and the rounding flips some of them. A flipped token's hidden state then reaches every later position through the 10 NoPE full-attention layers.
- **A mature MoE shows the same.** On Gemma 4 26B-A4B (fp32, the same T9 text), chunk 64 against chunk 2048 gives mean KL 0.0095 at 8k–16k, with 4 decisive top-1 changes. Changing one weight by 2^-22 relative gives 0.0032.
- **Consequence.** The frozen G4-D1 and G4-D2 constants (a port against itself ≤ 1e-6; bug-test mean KL ≤ 1e-3) cannot be met by a correct fp32 implementation of Kolibri at 16k context with free routing.

## 2. Port against reference at long context (G4-D1), and the document-start spikes

- **Measured on the mbp** (fp32, same dequantised weights): bug-test mean KL 0.002 on T1–8 and 0.015 on T9. At T9 15,000–15,300 the reference against port decode gives mean KL 0.076, with spikes at T9 document starts (3,911: window max 5.89).
- **No defect on either side.**
  - The reference's query chunking, sliding mask at large offsets, NoPE layers and RoPE positions match a dense float64 implementation at fp32 round-off over all 16,384 positions.
  - Neither side has any document-aware code path. T9's documents are joined by the ordinary "\n\n" ids, there is no BOS, and positions run on.
  - Only 2 of the 15 document windows stand out (3,911 and 14,239). The others are no larger than chunk windows at a similar depth. Gemma 4 shows the same concentration at document windows under a pure rounding change.
- **A systematic RoPE difference (not a defect).**
  - MLX's `fast.rope` computes inv_freq in an exp2 form, which differs from vLLM's and the reference's `1/θ^(2j/D)` by −1 to +8 fp32 ulps.
  - The rotated vectors therefore differ by up to 3.3e-3 at position 15,000. Against float64, MLX and vLLM are about equally accurate (9.2e-4 against 4–7e-4 rad).
  - Under forced routing it is the only port-vs-reference difference that grows with position: from 9.8e-7 to 3.4e-4 per block, max 1.6e-3. With a vLLM-angle RoPE in the port, this falls to ≤ 6.4e-7 and stays flat.
  - Under free routing, it is one seed of the chaotic divergence; the rest comes from the port's own chunking chaos (§1).
- **Candidate change for exp_037 (not applied here).** Use vLLM's angle computation in the port's RoPE: `mx.fast.rope(..., base=None, freqs=1/inv_freq_vLLM)`, citing vLLM `rotary_embedding/base.py` `_compute_inv_freq` and `_compute_cos_sin_cache`. In a scratch copy it brings the unit difference at 15,000+ from 2.7e-3 to 4.8e-7, with no test regressions. It is not a correctness fix, and it would not bring the free-routing bug test within the frozen bounds.

## 3. G5: batched vs single in bf16, concentrated in the first wave

- **Measured on the mbp.** In fp32 the batched path is exact (C, KL 1.5e-5). In bf16, KL(ref ‖ batched) is 0.299 against KL(ref ‖ single) 0.069. The first wave, eight prompts prefilled together right-padded to 1,099, carries KL 0.469, while sequences admitted mid-run carry 0.009. The batched run is also worse on the unpadded 1,100-token row (0.41, against 0.0095 for the same prompt admitted mid-run).
- **The mechanics are correct.**
  - A symbolic check of the gate's own batch generator and admission loop, with the port's caches and masks, found 0 wrong visible sets over 12,544 queries.
  - bf16 SDPA output is bitwise unchanged when masked pad content is scaled ×1e4.
  - On the M4 Pro, every part of the first-wave path is bitwise row-independent in bf16 and q8.
- **Not reproduced on the M4.** A real-dimension analogue (E 384, q8) gives a batched/single ratio of 1.08–1.14, against 4.35 on the mbp.
- **Remaining candidate.** An M5-specific (applegpu_g17) kernel path in the B=8 right-padded prefill. Stage 0's probe covered decode shapes only. `confirm/kprobe3_prefill.py` probes the prefill shapes, and `confirm/g5_ab_mbp.py` arm C (each prompt prefilled alone) tests the workaround.

## 4. Not covered by the hunt

- **G2-D** (layer 20, T4 position 19: r_t 3.19 against the layer's p99 2.37; the emulation does not flip that token).
- **G3-D** (the renormalised-top-k mutant is separable on T5 at only p01 0.087 < 0.24; bits per byte from token 64 differ by > 5 % on T3 and T4). G3-D stands under the frozen rules, since the anchor route was withdrawn.

## 5. What this means

- **For exp_036.** The gate failure stands as registered. The diagnostics show that two of its fidelity checks (G4's free-routing bounds, and G5's comparison with a chunking floor) measure Kolibri's routing chaos rather than implementation error. They also show that G3 cannot be settled without an external anchor.
- **For the post.** The finding is that near-exact cross-implementation agreement is impossible for Kolibri at long context, even between two fp32 runs of the same code. A port therefore has to be validated with routing forced, plus a distributional check, not by logit agreement.
- **For the exp_037 gate.**
  - Fidelity checks with forced routing (where a correct port reaches about 1e-6 and real defects survive).
  - vLLM-angle RoPE.
  - G5 decided by the A/B arms (an M5 workaround, or B = 1).
  - An explicit G3 decision: the anchor, or a declared blind spot.

## 6. Confirmation on the real K8 (M5 Max, started 2026-10-05T14:51:22Z)

*Added by the main session after the confirmation runs. Sections 1–5 above are left as first written; the corrections are in 6.6.*

- **Timing.** The predictions are in `confirm/README.md`, committed in 03f5f71 at 14:46:52Z. The runs started at 14:51:22Z (`U = 20261005T145122Z`), and the mbp committed their outputs in 2530ad4 at 14:59:33Z.
  - The run files record no git HEAD, so the claim that the mbp ran the 03f5f71 scripts unmodified is inferred, not recorded.
- **Hardware and software.** M5 Max (`applegpu_g17s`), macOS 27.0 (26A428), MLX 0.31.2, mlx-lm 0.31.3, `MLX_ENABLE_TF32=0`.
- **Status.** These are investigation runs. They wrote no gate record and used no fix cycle. A diagnostic value never replaces a gate value (FROZEN_RULES §1.3), and K8 FAIL stands.

| Output | sha256 |
|---|---|
| `confirm/out/rope_chaos_20261005T145122Z.json` | `c06186d40802a580b1a497c1216de5f97456e0e51babf888266f9eb94347a45b` |
| `confirm/out/kprobe3_m5_20261005T145122Z.json` | `1f434ca47b226dc4af5f078364cbd11a5da45f1195c7e183dd14d68d48126211` |
| `confirm/out/g5_ab_20261005T145122Z.json` | `01f2092c79aee738d9040632e03bb607ed0bfd8d94b237bc4042a2cdde0cfcff` |

### 6.1 Predictions against outcomes

| Run | Prediction (`confirm/README.md`) | Outcome |
|---|---|---|
| 1, forced chunk 64 | ≤ 1e-6, with 0 top-1 changes once routing is forced; the free run reproduces 0.0523 | **Met**, with the caveat on the forcing control below |
| 1, vLLM-angle RoPE | forced KL(reference ‖ port) at T9 drops to about 1e-6 and stays flat | **Not tested.** Part 2 ran the free-routing bug test instead; see 6.3 |
| 2, kprobe3 | a B = 8 right-padded prefill op loses precision on the M5 but not on the M4 Pro | **Met, and stronger than predicted:** wrong output, not lost precision; see 6.4 |
| 3, G5 A/B | arm A reproduces 0.3073773544; arm C brings the first wave to the mid-run level, about 0.01 | **Partly met:** A reproduces exactly; C's first wave is 0.0355, 13× lower but not 0.01; see 6.5 |

### 6.2 Run 1, part 1: the port against itself, routing forced

What was measured:
- the real K8 weights, dequantised, with fp32 activations;
- T9 positions 15,000–15,300 (301 positions);
- KL against the free chunk-2048 run.

"Forced" means the chunk-2048 run's top-6 expert ids are imposed on every layer at every position of the other run's history. The routing weights still come from that run's own fp32 router logits.

| Comparison | Mean KL | Max KL | Positions > 1e-6 | Top-1 changes |
|---|---|---|---|---|
| chunk 64, free | 0.05232 | 12.45 | 297 | 12 (none where chunk 2048 leads by ≥ 2 nats) |
| chunk 64, forced | 3.1e-9 | 6.4e-7 | 0 | 0 |
| decode, forced | 3.7e-9 | 6.5e-7 | 0 | 0 |
| chunk 2048 forced to its own ids (control) | 8.1e-8 | 1.7e-5 | 4 | 0 |

- **Reproduction.** The free row reproduces stage 3 in all four statistics.
- **What is shown.** On Kolibri's real weights, the chunk-64 divergence comes entirely from which experts are selected. Nothing else on the chunk-64 path contributes: the cache, the masks, RoPE and attention.
- **Inference only.** That the selection flips are rounding-triggered near-ties, and that they spread through the NoPE layers, still rests on the tiny checkpoints and Gemma 4.
- **Scope.** Only T9 was re-run forced. Stage 3's free chunk-64 differences at short context were not re-run: T1 positions 520–1,100 (max 1.1e-3) and T3 (max 7.7e-5).
- **Caveat: the control.** The control exceeds 1e-6 at 4 positions. Forcing passes the ids in sorted order, where free routing uses argpartition order, so forcing also changes the fp32 summation order of the six expert outputs. That is the code difference we know of. As the cause of the control's residual it is a hypothesis, not a measurement, and it does not explain why the control (mean 8.1e-8) is noisier than the forced chunk-64 run (3.1e-9), which carries the same reordering. With routing forced, the agreement level is about 1e-7 in the mean and up to about 1e-5 per position, not a fixed 1e-6 floor.

### 6.3 Run 1, part 2: the bug test with a vLLM-angle RoPE

- **What part 2 actually ran.** It is stage 3's free-routing bug test, re-run once with a vLLM-angle RoPE in the port. The test compares the reference with the port on the same dequantised K8 weights in fp32. Its "kit port" side re-summarises stage 3's saved file.
- **What it does not contain.** Part 2 has no forced series. The README's run comment and its prediction row described two different tests. A forced reference-vs-port series on real weights would need a new reference driver (`forward_packed` takes no forced ids) and a fresh multi-hour CPU reference pass.

| Bug test, free routing | Kit port (stage 3) | vLLM-angle RoPE | Frozen bound |
|---|---|---|---|
| T1–8 mean KL | 0.00197 | 0.00276 | ≤ 1e-3 |
| T1–8 decisive top-1 (misses of 4,632) | 0.99978 (1) | 0.99957 (2) | ≥ 0.999 |
| T9 mean KL | 0.0153 | 0.00288 | ≤ 1e-3 |
| T9 decisive top-1 (misses of 5,954) | 0.99765 (14) | 0.99950 (3) | ≥ 0.999 |
| T9 8,192–16,384 mean KL | 0.0240 | 0.00542 | — |
| Worst boundary window, mean KL | 0.141 (document start 3,911) | 0.0661 (document start 14,239) | ≤ 1e-3 |

- **Effect at long context.** The vLLM angle lowers T9.
- **Effect at short positions.** It raises T1–8. Two document windows also get worse: 14,239 (0.0315 → 0.0661) and 16,287 (0.0033 → 0.0095). That fits a change in which near-ties flip rather than a uniform move toward the reference; with one draw per arm, it is not shown.
- **Variance.** There is one draw per arm, so the T9 improvement cannot be separated from chance.
- **The frozen bounds.** They still fail on mean KL and on the window, and on decisive top-1 in the 8–16k bucket (0.99892).
- **Not cited.** The mbp relayed per-text values for T4 and T6 with the vLLM angle. They are not in the committed JSON and are not cited.
- **No bearing on the gate's G4.** Part 2's decisive set is not the population of the gate's G4 check, which compared K8 at its normal bf16 load with the reference on the BF16 checkpoint and missed 66 of 5,943. Nothing about G4 follows from part 2.

### 6.4 Run 2: kprobe3 at the first wave's prefill shapes

The probe used random weights and no model. It ran mlx-lm's `SwitchGLU` with the sorted `gather_qmm` path:
- 8-bit weights, group size 64;
- bf16 activations;
- E 384, top-6, H 2,560, I 512.

It ran at the first wave's prefill shape: 8 × 1,099 tokens, right-padded, which is 52,752 (token, expert) rows.

| | M5 Max, macOS 27.0 | M4 Pro, macOS 26 |
|---|---|---|
| Relative error, B = 8 call | 0.456–0.488 on every valid row | about 0.0054 |
| Relative error, same rows alone (B = 1) | 0.0044–0.0055 | 0.0044–0.0054 |
| Valid rows when only the pad rows' input changes | change (0.458–0.61) | bitwise unchanged |

- **The reference.** Errors are measured against MLX's own fp32 `SwitchGLU` on the same chip at the B = 1 shape, not against float64.
- **Wrong output, not lost precision.** An error of about 0.47 is about 100 times the bf16-level error of the same rows computed alone. The valid rows also depend on the content of the pad rows.
- **The other probed ops behave the same on both chips.** These are:
  - the q8 attention and shared-expert projections;
  - the fp32 router, which both chips flag with identical values (accumulation order, benign);
  - SDPA, causal and with window 513;
  - RoPE with array offsets.
- **Not separable from these files.** The two machines also run different macOS versions, and neither probe file records the OS. Three causes cannot be told apart:
  - MLX's kernel selection on the M5;
  - Apple's Metal compiler or driver on macOS 27.0;
  - the hardware itself.
- **No size threshold identified.** The M5 path is:
  - correct (against MLX's fp32) at 66 rows (stage 0) and at the B = 1 sizes up to 6,594 rows;
  - not bitwise equal to B = 1 at 33,552 rows of 8 identical unpadded copies (magnitude not recorded);
  - wrong at 52,752 rows.
- **The peers do not narrow it.** On the same mbp, G8 (Gemma 4) passes batched parity with 70,336 sorted rows, so row count alone is not the trigger. Q36-8 fails batched parity also with fp32 activations (0.0723), so it has at least one further cause.
- **Not yet a minimal upstream reproduction.** Four things are missing:
  - **Seeded weights.** The inputs and the projection and router weights come from a seeded numpy generator, and they match across the two chips digit for digit. The `SwitchGLU` weights come from MLX's own initialiser, which is not seeded, so they differ between the runs.
  - **Repeat runs.** Each chip ran once.
  - **The OS.** Neither probe file records it. The M5 Max's macOS 27.0 comes from the mbp's preflight and version records of 2026-10-03 and 2026-10-04; the M4 Pro's macOS 26 is in no file.
  - **A single-op script.** There is none for `gather_qmm`. Building one, and filing it upstream, is an outward step that needs Andrei's go.

### 6.5 Run 3: G5 A/B on the real K8

- **The quantity.** The KL is KL(single ‖ batched), the gate's own measure: the port alone at B = 1, teacher-forced, against the batched run. It is not measured against the fp32 reference.
- **Arm A** reproduces the gate value 0.3073773544462904 bitwise, with all 12 per-sequence values equal to stage 1. That is three identical runs: gate, stage 1 and this one.
- **Arm C** sets `prefill_batch_size = 1`, with decode still at B = 8. Under mlx-lm 0.31.3 this also changes the decode schedule: the batch ramps from 1 to 8, and the admission steps differ.

| | Arm A | Arm C |
|---|---|---|
| Mean KL (364 positions) | 0.3074 | 0.0298 |
| Top-1 disagreement | 52 (0.143) | 8 (0.022) |
| First wave (236) | 0.469 | 0.0355 |
| Mid-run (128) | 0.0090 | 0.0194 |
| First-wave t0 | 0.134 | 0.0118 |

- **Mid-run in arm C.** The rise comes almost entirely from sequence j9. Its tokens diverge from arm A at step 5, and one position at KL 1.24 carries most of it; without that position the mid-run mean is 0.0098.
- **Attribution.**
  - At t0, arm C equals the single-sequence decode path to 6e-11. Arm A's first-wave t0 excess therefore comes from processing the first-wave prompts as one B = 8 right-padded prompt batch.
  - Against arm A's first-wave j0 and j3, arm C's j0 and j3 differ by the same maximum |Δ log-prob| (5.6969 and 12.0953, at the same divergence steps) as arm A's mid-run copies j8 and j11 do (stage 1 L3), to every recorded digit. The common prefixes are 2 and 3 steps, and arm C was not compared with j8 and j11 directly. The match is therefore inferred from equal summary statistics; it is what a first-wave prompt following the mid-run path would give.
  - One first-wave prompt got worse under arm C: j1 (300 tokens) rose from 0.166 to 0.497.
  - The unpadded 1,100-token row falls from 0.4105 to 0.0024, so the excess is not that row's own padding.
- **What stays open.** Arms B (pad id 1 instead of 0) and D (no padding) were not run. On real weights, a size trigger and a padding trigger are not separated.
- **Arm C and the gate.** Arm C is one diagnostic draw, and a favourable draw is never set against the gate's bounds (FROZEN_RULES §1.3). No frozen branch opens: stage 5's G5-D needed a localisation by the frozen suite, and kprobe3 ran after stage 5.

### 6.6 Corrections to sections 1–5

- **§1, "shares that whole history".** Not exact. Decode prefills 14,336–14,999 as one 664-row chunk, where prefill 2048 runs 14,336–15,300 as one 965-row chunk. Decode stays at a max KL of 1.6e-7 free and 6.5e-7 forced.
- **§1, "the quantised matmul rows are bitwise equal at M = 64 and M = 2048".** This holds at the shapes tested on the mini. kprobe3 shows that the 8-bit `quantized_matmul` rows for k_proj, o_proj and the shared expert are not bitwise invariant to call shape (B = 8 × 1,099 against B = 1). This holds on both chips, with equal or better accuracy at B = 8.
- **§2, "only 2 of the 15 document windows stand out … the others are no larger than chunk windows at a similar depth".** Wrong against stage 3's record. Ranked by window mean KL:
  1. 3,911 (0.141);
  2. 15,123 (0.092);
  3. 14,239 (0.031);
  4. chunk 10,240 (0.020).

  Document windows 13,399, 14,390, 14,076 and 14,637 (0.012–0.016) exceed the nearest chunk window, 14,336 (0.0055), by 2–3×. The direction stands: document starts carry most of the free-routing divergence. The count does not.
- **§3, "in fp32 the batched path is exact".** Nearly exact. The whole fp32 residual sits on one row, the unpadded 1,100-token first-wave prompt (KL 3.4e-4). The other 11 rows are at ≤ 1.2e-10.
- **§3, "not reproduced on the M4".** This rests on random or tiny weights on the M4 Pro. Kolibri's real weights have run on the M5 Max only.
- **§3, "on the M4 Pro, every part of the first-wave path is bitwise row-independent".** This holds for the neighbour probes: the valid rows do not change when the pad rows change. It does not hold for the B = 8 and B = 1 comparisons on the M4 Pro. There, the fp32 router (1.5–3.4×) and the sorted `SwitchGLU` (up to 1.24×) are not bitwise equal; both are benign and at bf16 level. So are the q8 projections, as above (`kprobe3_m4pro.json`).
- **§1, Gemma 4 "changing one weight by 2^-22 relative gives 0.0032".** That value is the 2,048–8,192 bucket. At 8k–16k, the bucket the sentence compares with, the perturbation gives 0.00064.
- **§1, Gemma 4 "(fp32, the same T9 text)".** Only the activations were fp32: the weights are the mlx-community OptiQ 4-bit build, run on the M4 Pro. Gemma 4 tokenised the same character sequence, and its 16,384 tokens end near Kolibri position 14,100. Its "8k–16k" bucket is therefore not the same span of text as Kolibri's. At its own positions 15,000–15,300, chunk 64 against 2048 gives 0.0116.
- **§2, "Gemma 4 shows the same concentration at document windows under a pure rounding change".** Wrong.
  - Under the weight perturbation, Gemma's document windows are *lower* than elsewhere: 4.2e-4 against 1.77e-3.
  - The concentration appears only under chunking (chunk 64 against 2048): 0.0288 at document windows against 0.0061 elsewhere.
- **§2, "against float64, MLX and vLLM are about equally accurate (9.2e-4 against 4–7e-4 rad)".** Those figures were printed, not recorded. The recorded probe (`mini/bughunt/rope/r1_out.json`) has MLX somewhat less accurate beyond position 12,288: worst angle error 1.33e-3 rad (mean 8.1e-5), against 9.7e-4 (mean 4.4e-5) for the vLLM form, which equals float32's own rounding of the angle. The same order, not equal.
- **§4, the list is incomplete.** Stage 5 also fired G4-D5 (20 positions where the port's KL from the reference exceeds 5 nats while the emulation's is below 1) and G4-D6 (on T6, the port's KL rises from the standalone text to the same tokens inside T9 by more than twice the emulation's rise). §1–§2 address G4-D1 and G4-D2 only; these two signs were not examined.
- **§5, "two of its fidelity checks (G4's free-routing bounds, and G5's comparison with a chunking floor) measure Kolibri's routing chaos rather than implementation error".** Too broad for G5. G5's bound is set from the chunking floor, which is itself routing chaos, but G5's excess sits in the batched first-wave prefill (6.4–6.5), not in chaos. For G4, the free-routing diagnostic bounds are the ones §1 argues against; the gate's own G4 threshold (99 % decisive top-1) is also missed by a bf16 emulation of the reference on the same 8-bit weights (68 of 5,943, against the port's 66; stage 5), and two of G4's defect signs remain unexamined (above).
- **§5, "where a correct port reaches about 1e-6".** Under forcing this needs the control's caveat (6.2): about 1e-7 in the mean, up to about 1e-5 per position.

### 6.7 Where the evidence for §1–§3 is

The tiny-checkpoint, float64, symbolic, M4 Pro and Gemma 4 results are in [`mini/`](./mini/README.md). Its `MANIFEST.json` lists every file with its sha256, including the files withheld for size or content; the two env files, never opened, have a null sha256, and the 23 `__pycache__` files are given as a count. Two kinds of file are not there:
- **Arrays.** No `.npz` files are committed, and the large arrays are listed by sha256 in the manifest.
- **Files listing T9 positions.** They were computed on the mini from the real T9 ids, which are withheld with T9's web text. The Kolibri document-start positions cited in §2 and §6 are the ones already committed in `out/stage3_g4_20261005T124342Z.json` (`bug_test_boundaries`). A position-free summary replaces the Gemma 4 run.

Some numbers in §1–§3 were printed by their scripts and never written to a file. The scripts are in `mini/`, so these numbers can be re-run but are not recorded:
- the 32,842 and 12,544 symbolic queries;
- SDPA within 5e-6;
- the M = 64 and M = 2048 matmul check;
- the K8-build tiny model's forced 5.9e-8 and free 0.061;
- the −1 to +8 ulp range and the 3.3e-3 vector difference;
- the 9.2e-4 against 4–7e-4 rad comparison and the 2.7e-3 → 4.8e-7 drop;
- the pad-content ×1e4 check.

`mini/README.md` maps every other number to its file and field.

### 6.8 What changes for exp_037

1. **Forced-routing fidelity.** The bound is calibrated by a forced self-control, not by a fixed 1e-6. Add a forced reference-vs-port series on real weights; this needs a reference driver that takes forced ids.
2. **The vLLM-angle RoPE.** Adopt it, or not, on that forced series. Under free routing its effect is mixed.
3. **Batching on the M5.** Prefill one prompt at a time, or run at B = 1, with arms B and D run once to separate padding from size. Report the `SwitchGLU` / `gather_qmm` result upstream with a seeded single-op reproduction that records the OS, on Andrei's go.
4. **G3.** Decide it explicitly: run the anchor, or declare the blind spot.
