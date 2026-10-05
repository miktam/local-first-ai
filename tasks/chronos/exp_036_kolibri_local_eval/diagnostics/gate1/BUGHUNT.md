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
