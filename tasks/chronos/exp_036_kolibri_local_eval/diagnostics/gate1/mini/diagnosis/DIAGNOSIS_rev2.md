# exp_036 gate 20261005T050112Z: diagnosis of the K8 FAIL, and plan (revision 2)

Revision 2 answers two reviews of revision 1: a skeptic's review (9 objections) and an integrity review (12 objections). Six cheap separating experiments were run on the mini (§A). Every conclusion was re-graded against them. The decision rules in §4 are now frozen, with no free parameter left, and every check has an outcome in which its failure stands. The kit is at a666f3e (gate record 5c6a6e8, Amendment 7 appended since). Nothing in the kit was edited or committed. Kolibri weights are on the mbp only, so every real-weight claim comes from the committed records or is marked as needing the mbp. Revision 1 is kept at `scratchpad/gatediag_rev/DIAGNOSIS_v1_backup.md`.

---

## 0. Summary

1. **No code fix is ready.** No defect has been localised. One ready-to-run tool exists, `kprobe2.py`, which runs unchanged on the mbp in seconds.

2. **Verdicts, revised:**

   | check | verdict now | confidence | change from rev. 1 |
   |---|---|---|---|
   | G5 batch parity | defect suspected, not localised | medium that this is *not* generic MLX batching numerics | was "numerics leading (low)" |
   | G4 T9 decisive | unresolved | low | was "numerics vs uncalibrated bound (medium)" |
   | G2 6.08σ | leaning miscalibration, not shown | low | was medium |
   | G3 per text, G3 ratio | undetermined: a ≈ 9 % shared misreading is not excluded | low | was medium |

3. **The G5 change is the main result of this revision.**
   - Two correct implementations were run on the mini: dense Qwen3-4B q8 and the Gemma-4 26B-A4B MoE. On real weights, through the gate's own batch loop, they give batch parity ÷ chunking floor of 0.98 and 1.01.
   - In both, the batched path is no less accurate than the single path against an fp32 truth (ratio 0.85 and 0.84), with 0 decisive flips.
   - G8 on the M5 Max reads 0.61 on the same ratio. Kolibri K8 reads 6.1.
   - At Kolibri's real decode shapes, with an exact reference, every quantised matmul, the head and SwitchGLU (48 and 66 rows) are bitwise equal at B = 8 and B = 1 on the M4 Pro.
   - So the leading candidates are now an M5-only kernel path at Kolibri's shapes, or a Kolibri-specific real-weight path. "bf16 MoE batching noise" is no longer one of them.

4. **The skeptic's G4 noise argument does not survive the nulls, but G4 still lost confidence.**
   - In the nulls, the difference between a build's mean NLL and the reference's (dNLL) varies in sign from text to text, and its ratio to KL ranges from −3.7 to +8.5, with confidence intervals excluding zero on several texts. So negative dNLL alone is not a defect sign.
   - What the nulls do not show is Kolibri's growth with context: KL doubles at 8–16k and decisive misses triple. The Gemma MoE (bf16 vs fp32 activations) is flat across T9 buckets (0.013 / 0.0125 / 0.0083) with 3 misses in 8,669. Qwen3-4B q8 has 0 misses in 7,998.

5. **Integrity findings accepted:**
   - Revision 1's rules were outcome-shaped:
     - G2's 5.1 switch point was 6.08 ÷ 1.2.
     - A2's floor of 2 passed the one outlier for any emulation value.
     - G5 took the first M / W / F option that passed.
     - G3 had no branch in which the failure stands.
   - Package N mislabelled gate-code changes as threshold changes.
   - All of this is withdrawn. The rules in §4 are fixed now, before any diagnostic runs, and are disclosed as set after the gate values were seen.

6. **Recommended order:**
   - (a) Before anything runs, Andrei records three things (§5): the cycle reading (D1), the consequence for G5 if no defect is found (D3), and whether to run the FP8 anchor (D4).
   - (b) The main session adds the §4 measurements to the four diagnostic scripts. It commits them, together with `diagnostics/FROZEN_RULES.md` (the §4 rules verbatim), before any mbp run.
   - (c) Andrei runs stages 0–4 on the mbp: about 2.5–3 h of machine time, no gate record, no cycle, no S1 hours.
   - (d) The main session applies the frozen mapping mechanically. Stage 5 reports what the mapping gives. It does not search for a passing rule.

7. **Cycle consequence (§5 D1):**
   - Under reading (iii), one cycle per threshold changed, the gate cannot pass. The minimum passable package changes at least four thresholds plus a G5 rule, so D5 follows.
   - Under (ii), one cycle per typed amendment, a passable package costs 2 cycles, so the next re-run is the last.
   - Under (i), one cycle per re-run, one cycle stays in reserve.

8. **Corrections to `aborted/20261005T050112Z-gate/NOTE.md`** (unchanged from rev. 1):
   - The 8,192–16,384 bucket KL is descriptive and did not fail. The only blocking G4 element is T9 `top1_decisive`.
   - The Kolibri side has no start token.

9. **Amendment 7:**
   - Criterion 2 keeps G4's original bf16 bounds for the Hugging Face upload, so no G4 amendment unblocks the upload.
   - Criterion 4's G5 clause was written at 06:40Z, after the gate ended (05:57Z), with G5's failure known. The commit title "(Andrei, before any result)" should read "before any scored result". The amendment text itself says the gate stood at K8 FAIL.

---

## 1. Verdict per failing check

| check | measured | limit | verdict | confidence | same on re-run? |
|---|---|---|---|---|---|
| g2_bf16_natural_selection | 1 of 8,448 disagreements at 6.08 σ_l (layer 20) | every disagreement < 6 σ_l | leaning miscalibration; a token-local defect not excluded | low | expected identical |
| g3_bpb_per_text | T1 1.311 | ≤ 1.2 | undetermined: pre-data threshold vs ≈ 9 % shared misreading | low | identical (reused dump) |
| g3_bpb_vs_peers | 1.296 | ≤ 1.25 | undetermined (as above; T4 memorisation by Q36-8 explains 38 % of the gap) | low | identical |
| g4_K8_backstop | T9 top1_decisive 0.98889 (66 / 5,943 missed) | ≥ 0.99 (≤ 59) | unresolved: the count is within noise, but long-context growth is unexplained | low | expected identical |
| g5_batch_parity_K8 | mean KL 0.307 | ≤ 0.151 | defect suspected (Kolibri- or M5-specific), not localised | medium (not generic numerics) | expected identical |

### 1.1 G2 bf16 natural selection: leaning miscalibration, not shown (low)

**Facts (unchanged).**
- Layer 20 is a sliding layer: 225 disagreements, σ_20 = 1.16e-3, one disagreement at 6.08 σ. T1–T8 only.
- Disagreement fraction 0.01375 against a limit of 0.0247 (emu 0.0124).
- Port/emu per-layer error ratio: median 1.09, range 0.94–1.40.
- The fp32 natural selection is exact.
- Sorted layer maxima: …4.93, 5.08, 5.08, 6.08.

**Against "miscalibration" (accepted from the reviews).**
- Two of revision 1's five miscalibration triggers are the very signature of a token-local defect:
  - "σ_t in the layer's top 5 %";
  - "gap/σ_t ≤ 4.5".

  Both are removed (§4).
- The quoted P(trip) ≈ 0.2–0.96 came from fits to the port's own data, outlier included. It is withdrawn as evidence.

**New from the record (§A6).** `phase2.json` keeps the emulation's per-layer max as well as p99, so port and emu tails can be compared on the same T1–T8 rows:
- **MoE layer 13** (port max/p99 8.07): emu shows the same tail (7.90). Port/emu 1.02, so this is a property of the layer and its tokens, not of the port.
- **Attention layers 8 and 29:** port max/p99 5.66 and 5.25 against emu's 3.43 and 2.50. The port's maximum is 1.72× and 2.24× emu's maximum on the same rows, while the p99 ratios are about 1.05.
  - That is below the 3× trigger, but it is a localised excess.
  - Its token positions are not in the record.
- **MoE layer 30 at T9 8–16k** (11.99): emu was never run on T9, so its T1–T8 value (5.38) is not a like-for-like control. Unresolved.

**What decides it.** Stage 2's per-token test, port σ_t against emu σ_t at the flip token (§4).

### 1.2 G3: undetermined (low)

**Facts (unchanged).**
- Thresholds frozen in 725d628, before any peer or Kolibri number. Deterministic.
- Kolibri ÷ Q36-8 by text: T1 1.16, T2 1.32, T3 0.99, T4 5.75, T5 1.15, T6 1.27.
- T4 holds 38 % of the bit gap. The ratio is 1.19 without T4.
- All 7 reference mutants are resolvable on T3, with p01 ≥ 0.546 nats/token.

**Why confidence is now low (integrity objection 3, accepted).**
- A uniform shared misreading of +9.2 % NLL takes T1 exactly to 1.2. The ratio would then be 1.187.
- On T3, 9.2 % is ≈ 0.15 nats/token, under a third of the subtlest mutant.
- It would put Kolibri's true T3 at 0.451 bpb: 0.91× Q36-8 and 0.97× Q38-8. That is unremarkable for a German-focused model, so T3 parity excludes nothing.
- Other spec items have no mutant at all: attention scale, eps, how the shared expert is combined, finer norm placement.
- G3 is one of only two checks independent of the shared specification (HYPOTHESIS Phase 0).

**Tests withdrawn as unable to separate the two readings.**
- **The K8 recall probe.** K8 shares the specification with the reference, so a shared misreading would erase recall in both.
- **The τ\* (best-fit temperature) comparison.** It is too weak:
  - Two correct models of one family differ in NLL-optimal τ\* by up to 0.08 on T1 (Qwen3-4B 1.26, Qwen3-1.7B 1.18; §A2).
  - Chat-native models are overconfident on these raw texts, with τ\* 1.18–1.30 and a gain of 0.09–0.20 nats/token.

  It stays descriptive.
- **D3' as written in revision 1.** It required p10 ≥ 1.1 on T3, the one text with ratio 0.99, so it could essentially never fire. It is rewritten in §4.

**What the relaxed bounds can still detect** (computed from the record).
- B1 (per text 1.5, ratio 1.40) lets through a further uniform misreading of up to +0.24 nats/token additive, or +8.1 % relative.
- B2 lets through +0.11 nats/token, or +3.1 %. B2 is withdrawn: it sits 0.04 and 0.05 above the observed values.

**The only test with power against this class is the FP8 vLLM anchor** (§5 D4).

### 1.3 G4 K8 backstop: unresolved (low)

**Facts (unchanged).**
- 66 of 5,943 decisive positions missed, against ≤ 59 allowed. Wilson 95 % interval 0.87–1.41 %. P(≥ 66 | 1.00 %) ≈ 0.21.
- Misses by bucket: 3/606, 19/2,556, 44/2,781.
- G2's T9 rows show no per-layer growth with position. But those rows use forced routing (integrity objection 5).
- Natural routing at long context is checked only end to end, by G4 itself.

**The skeptic's noise-model argument, tested on the mini (§A1–A3).**
- The claim: zero-mean noise predicts dNLL ≈ +KL, so a negative dNLL is a defect signature.
- On correct MLX implementations, compared with their own fp32 versions on the gate texts, that does not hold:
  - **Qwen3-4B q4** (KL 0.11–0.17): dNLL/KL is 0.25–0.53 on every text, all confidence intervals positive except T2. At T9 8–16k it is 0.25 [0.14, 0.35].
  - **At small KL** (Qwen q8, Gemma MoE activations), dNLL/KL varies in sign by text, and individual texts are significant:
    - Qwen3-4B q8, T1: −3.70 [−6.03, −1.82];
    - Gemma T3: −2.55 [−4.39, −0.91];
    - Gemma T6: +1.38 [+0.37, +2.29].
  - The cause: a fixed weight or activation perturbation is one draw per text, not independent noise per position.
- So a negative dNLL is not by itself a defect sign. Kolibri K8's T1-8 −0.02 and T9 −0.10 are inside the range the nulls span.
- K4 at 8–16k (−0.20) and on T6 standalone (−0.41) is outside the dense q4 null. No MoE null exists at that KL. The emulation decides (§4).

**What the nulls do not reproduce: growth with context.**

| | T9 0–2k | 2–8k | 8–16k | decisive misses at 16k |
|---|---|---|---|---|
| Kolibri K8 vs fp32 ref | KL 0.079 | 0.052 | **0.108** | 66 / 5,943 (0.9889) |
| Gemma-4 MoE, bf16 vs fp32 activations, same weights | 0.0132 | 0.0125 | 0.0083 | 3 / 8,669 (0.99965) |
| Qwen3-4B q8 vs fp32 | 0.0017 | 0.0014 | 0.0018 | 0 / 7,998 |
| Qwen3-4B q4 vs fp32 | 0.144 | 0.138 | 0.173 | 170 / 7,998 (0.9787) |

- The 8–16k bucket is the T3 tail, T5, T6, then 10 short web documents.
- On the Gemma MoE, the same text sequence makes 8–16k the *quietest* bucket. So revision 1's "content, not position" explanation has no support from a correct MoE.
- Long-range NoPE attention and natural routing at long context are Kolibri-specific and untested.

**What decides it** (frozen in §4):
- the bug test (K8-fp32 against the dequantised reference);
- chunk-64 exactness in fp32 on the real weights;
- the emulation on the dequantised K8 weights (E_T9), with a paired port-vs-emu McNemar test;
- natural selection by T9 bucket, port vs emu.

**The yardstick.** The skeptic's point is correct that the emulation is not identical to the vendor in either direction (§A4):
- **Norm.** It normalises the unrounded sum, which is vLLM 0.29's IR op definition: `ir_layernorm.py` l.53–58, dispatched from `layernorm.py` `forward_cuda` → `forward_native`. The port normalises the bf16-rounded sum. On synthetic data with massive-activation channels, the port's order is *more* precise: 1.92e-3 against IR 2.40e-3, with one rounding at 1.69e-3.
- **Attention.** The emulation skips FlashAttention's bf16 rounding of P before P·V: 1.71e-3 against 1.89e-3, so emu is +10 % more precise there.
- Each difference is at most about one bf16 rounding at one op. Together with G2's measured port/emu per-layer ratio (median 1.09, max 1.40), this calibrates the 1.25 factor in §4's G4 rule.
- Which IR provider runs on CUDA by default (native or a vllm_c kernel) is not in the local sources and stays open. It does not change the rule.

### 1.4 G5 batch parity K8: defect suspected, not localised (medium that it is not generic numerics)

**Measured:** 0.3074 against 0.1513. `top1_dis` 0.143 passes. K4 is 0.223. The record keeps aggregates only.

**Every control puts batching noise at or below chunking noise:**

| | parity ÷ registered-style floor | parity ÷ decode-vs-prefill | KL(truth‖batched) ÷ KL(truth‖single) | decisive flips |
|---|---|---|---|---|
| Qwen3-4B q8, M4 Pro, real weights, gate loop (§A3) | 0.98 | 1.18 | 0.85 | 0 / 146 |
| Gemma-4 26B-A4B MoE, M4 Pro, real weights, gate loop | 1.01 | 1.40 | 0.84 | 0 / 216 |
| G8 (Gemma-4 8-bit), M5 Max, peers record | 0.61 | — | — | — |
| tiny Kolibri port, bf16, 63 runs (rev. 1) | median 1.11 (0.23–2.38) | — | batched further from truth in 42–48 % | — |
| **Kolibri K8, M5 Max, gate** | **6.1** (≈ 13 vs matched T1/T3) | **8.1** | not measured | not recorded |

**Kernels at Kolibri's real decode shapes, exact float64 reference** (§A5, `kprobe2.py`; this fixes `kprobe.py`'s bf16-scale reference, which floored every qmm comparison at ≈ 1.6e-3). On the M4 Pro:
- **Bitwise equal at B = 8 and B = 1:** q / k / o / shared q8 projections in bf16, the fp32-input q8 head at V = 128,000, and SwitchGLU q8 at 48 rows (B = 8 × top-6, the unsorted gather path, < 64) and at 66 rows (sorted path).
- **The fp32 router GEMM** differs: 1.15e-7 against 8.3e-8, which is fp32 accumulation order and immaterial.
- **bf16 decode SDPA** pays the 2-pass extra rounding for a short sequence inside a ≥ 1,024-key buffer: ×1.42, a known MLX property. The correct-implementation nulls above pass with it.

So on the M4 Pro, Kolibri's batched decode is numerically the B = 1 decode path, apart from SDPA's 2-pass rounding.

**Why that points away from generic noise.**
- Gemma's B = 8 decode uses the sorted gather path (8 × 8 = 64 rows) and is *not* bitwise equal to B = 1. Its KL before greedy divergence is 1e-7 to 0.09 per step (§A3). Its parity is still at the floor.
- Kolibri's B = 8 (48 rows) uses the unsorted path, which differs from G8's path on the M5.

**The remaining candidates:**
- (b) an M5-only kernel at Kolibri's shapes (the NAX variants: `gather_qmm_t_nax`, `qmm_t_nax`, `steel_gemm_*_nax`), which M4 probes cannot see;
- (d) a Kolibri-specific real-weight path in the batched caches (NoPE `BatchKVCache` and the sliding `BatchRotatingKVCache` under real attention patterns), which fp32-on-tiny cannot see;
- (c) a few catastrophic positions carrying the mean.

Revision 1's candidate (a), "bf16 MoE routing chaos", has no supporting null and is demoted.

**Part C cannot clear the bf16 path.** Fp32 runs separate kernels from the bf16 path, so a pass in part C is necessary but not sufficient (skeptic objection 2, accepted).

**Peers.**
- G8 is the matched control: the same machine, MLX, runner path, lengths, sliding caches and 8-bit MoE. It passes.
- Q36-8's failure is uninvestigated. Its batched path is mechanically different (`ArraysCache` for Gated DeltaNet, with `create_ssm_mask`). It is no evidence for Kolibri either way.

**What decides it:** stage 1 (§4).
- **Defect tests:**
  - the fp32-reference teacher forcing of the 12 batched sequences (worse, or only different?);
  - `kprobe2.py` on the M5;
  - part C (fp32 mechanics).
- **Localisers:**
  - the identical-prompt run (M = 8 kernels, or padding / admission?);
  - the B = 1 BatchGenerator control;
  - a free comparison between waves: sequences 0 and 8 are both T1[:37], and sequences 3 and 11 are both T4[:1100]. The record's construction gives first-wave and mid-run copies of the same prompts.

---

## 2. Code fixes

**None ready.** No defect is localised, and no vendor or vLLM line gives a citable deviation. On the yardstick question in §1.3:
- the emulation's norm order is the vLLM 0.29 IR definition, and citable;
- the omitted P rounding is a 10 % op-level difference, not a reference defect.

**Conditional fixes,** triggered only by the frozen rules in §4:

| trigger | where to look | test to add | consequence |
|---|---|---|---|
| G5-D (any G5 defect sign) localised to the port or runner | `runner/generate.py` batch construction; the port's `make_cache` and masks under `BatchKVCache` / `BatchRotatingKVCache`; mlx_lm `BatchGenerator` (cite the line) | head_dim-128 tiny parity test reproducing the class, in bf16 *and* fp32 | gate-fix amendment; K4's descriptive parity and behaviour change; step-9 re-run advisable after a runner change |
| G5-D localised to an M5 kernel (kprobe2 DEFECT, or identical-prompt localiser fires) | MLX upstream report with the kprobe2 JSON | kprobe2 kept as a host check | runner workaround, typed as a gate fix citing the kernel: Kolibri (and K4) at B = 1, or a path that avoids the kernel |
| G4-D bug test or chunk-64 test fails | `port/kolibri1.py` long-context path (NoPE layers, cache growth, chunk edges) | tiny G1 test at that boundary | port change; every check is re-measured for both arms |
| G2-D at a position class (0/1, 511–515, chunk edges) | port mask / cache at that boundary | tiny G1 boundary test | port change |
| G3-D and the FP8 anchor reads ≥ 5 % lower on T1, T2 or T4 | `reference/` and the mirrored port, citing a vendor or vLLM line | reference mutant test | separate amendment, no threshold change in it; dumps rebuilt (gate ≈ 56 min) |

**Not a fix, and outside every hash scope:** the diagnostic scripts and `diagnostics/FROZEN_RULES.md`.

---

## 3. Amendment options, typed honestly

**Rules that apply** (unchanged): a threshold changes only by an Andrei-typed amendment that gives the original value, the observed value and the reason. It counts as a fix cycle. A reference fix cannot change thresholds in the same amendment. The companions are:
- `thresholds.json`;
- the BUILD_SPEC §7.1 block;
- the `tests/test_gate_thresholds.py` pins;
- `THRESHOLDS_SHA256`;
- `GATE_CODE_SHA256` if `gate/` changes.

**Types (integrity objection 2, accepted).** Each option below carries its true type:
- **threshold change:** a `thresholds.json` value only;
- **gate fix:** a code change in `gate/`, `runner/`, `port/` or `reference/`;
- **verdict-rule change:** a change to what blocks or what failing means. This type is not registered. It is used only if Andrei accepts it, and is disclosed as a post-result rule change.

| option | check | type | frozen form | still kept? |
|---|---|---|---|---|
| A1 | G2 | threshold change | `bf16_selection_gap_sigma` = min(8.0, 1.2 × M_emu), used only if M_emu ≥ 6.0 (emu fails the registered rule itself) | yes, gated by §4 |
| A3 | G2 | gate fix | a port disagreement that emu shares, same (token, layer), is exempt from the σ rule; every other disagreement keeps < 6 σ_l | yes, gated by §4 |
| A2 (count ≤ max(2, 2× emu)) | G2 | gate fix | — | **withdrawn**: no power against sparse large flips (two 20σ flips pass) |
| B1 | G3 | threshold change ×2 keys | per text 1.5, ratio 1.40, labelled as a post-hoc gross-sanity bound | yes, gated by §4 |
| B2 | G3 | — | — | **withdrawn**: the observed values plus 0.04 / 0.05 |
| C1' | G4 | threshold change | `K8_backstop_decisive_top1_min` = max(0.985, min(0.99, 1 − 1.25 × (1 − E_T9))); the shared key also applies to T1-8 (disclosed) | yes, gated by §4 |
| C1 (E_T9 − 0.005), C2 (fp32 backstop), C3 | G4 | — | — | **withdrawn**: C1 was a post-hoc cushion; C2 removes the only bf16 long-context check and changes the blocking set; C3 is new gate code with no added power |
| G5-B1 | G5 | verdict-rule change (gate code: `run_gate.BLOCKING`, `plan_fix`, guard) | K8 and K4 at B = 1, consistent with the peers' registered consequence | yes, only on G5-N, and only if chosen in D3 before stage 1 |
| G5-M / G5-W / G5-F | G5 | — | — | **withdrawn**: selected by outcome; F changes the blocking set; W contradicts §1.2's own Amendment-5 reasoning |

**How much each kept relaxation can still detect** (this goes into the amendment):
- **A1:** a single flip up to 1.2 × M_emu σ (≤ 8σ).
- **B1:** a further uniform misreading up to +0.24 nats/token, or +8.1 %.
- **C1':** a port bf16 path up to 25 % worse than vendor-faithful bf16 in decisive misses, and never below 0.985.
- **G5-B1:** batching defects become irrelevant to the runs, because Kolibri never batches. A defect in the B = 8 path would still go to the model card (Amendment 7 criterion 4) and to an upstream report.

**Skeleton** (facts only; Andrei types the rest):
```
## Amendment <k> — threshold change (<UTC>)   [typed by Andrei]
| check | original | observed (gate_20261005T050112Z) | diagnostic (diagnostics/<file>, sha256) | frozen rule outcome | new | reason |
| g2_bf16_natural_selection | < 6·σ_l | 1 of 8,448 at 6.08σ (layer 20) | M_emu {{}}; emu n≥6σ {{}}; r_t {{}} (layer p99 {{}}); emu flips same {{}} | G2-{{}} | {{}} | {{}} |
| g3_bpb_per_text | ≤ 1.2 | T1 1.311 | D3'' {{}}; mutants on T1/T5 min ΔNLL {{}}; anchor {{}} | G3-{{}} | {{}} | {{}} |
| g3_bpb_vs_peers | ≤ 1.25× | 1.296 | as above | G3-{{}} | {{}} | {{}} |
| g4_K8_backstop | ≥ 0.99 | 0.98889 (66/5,943) | bug test {{}}; chunk-64 {{}}; E_T9 {{}} (misses {{}}); McNemar p {{}}; port/emu miss ratio {{}} | G4-{{}} | {{}} | {{}} |
Cycle reading: {{ANDREI_D1}} (chosen after the failure). Amendments 1–2: {{ANDREI_D6}}.
Disclosure: §6 items 1–9.
## Amendment <k+1> — <gate fix | verdict-rule change> (<UTC>)   [G5, if any]
```

---

## 4. mbp confirmation plan

**Ground rules:**
- **No gate record.** Never invoke `gate/run_gate.py`.
- **Diagnostics are not cycles** (precedent: Amendment 6 / `diagnostics/gemma_quant_check.py`). Andrei confirms the route (D2).
- **Diagnostic values never replace gate values.** The gate record's failing values stand.
  - If a reproduction check fails (MLX not deterministic on the M5), record the spread.
  - Every threshold-only option must then clear its margin by more than that spread, or it is abandoned.
  - A favourable draw is never cited.
- **Frozen before any run.** The main session commits `diagnostics/FROZEN_RULES.md` (this section's rules, verbatim) together with the script changes, and records its sha256 in the commit message. Any later change to a rule is itself disclosed.
- **What may be committed:**
  - **Allowed:** port-vs-reference and port-vs-emulation comparisons; aggregate gap statistics; peer numbers; kernel probes.
  - **Kept in `$EXP036_WORK` unless the gate passes:** absolute K8 NLL or bpb, K8 recall counts, and any K8 quality number. This keeps faith with "no quality number from an ungated port".
  - **Never computed:** K8 chat-wrapped NLL. That is half of H8's confirmatory regime.
  - **Never committed:** the `.npz` files and `ids.json`, which hold token ids and web text.
- **Preconditions:**
  - the tree is at a666f3e plus `diagnostics/`;
  - `$EXP036_WORK/ref` is present;
  - ≥ 45 GB free;
  - one model-loading process at a time.

**Setup:**
```
cd $KIT
source env/exp036.env && export MLX_ENABLE_TF32=0
D="$EXP036_WORK/diag_20261005"; mkdir -p "$D"
```

### Stage 0: kernels (≈ 1 min, no weights)
```
"$PY" diagnostics/kprobe2.py diagnostics/kprobe2_m4pro.json > "$D/kprobe2_m5max.json"
```
Copy `scratchpad/gatediag_rev/kprobe2.py` (sha256 `88c1ce6b…`) and its M4 Pro baseline `kprobe2_m4pro.json` into `diagnostics/`. The output's `baseline.verdict` is used directly.

### Stage 1: G5 (≈ 35–45 min)

**Additions to `g5_parity_diag.py`** (a reference implementation of the per-position decomposition, for mlx_lm models, is `scratchpad/gatediag_rev/e2_batch_null.py`):
- **B (bf16, real K8).** Reproduce 0.3073773544. Per position, record:
  - KL(single‖batched);
  - KL(single‖decode_tf);
  - KL(decode_tf‖batched);
  - the matched floor (single 2048 vs 64);
  - the lead of single;
  - flips where single leads by ≥ 2 nats.
- **R (new, decisive).** Teacher-force the 12 batched sequences (prompt + batched tokens) through the fp32 reference (`reference.kolibri_ref`, BF16 weights; the phase-5 machinery, ≈ 6.3k tokens, ≈ 4 min). Per position, compute KL(ref‖single), KL(ref‖batched) and KL(ref‖decode_tf), plus decisive (ref lead ≥ 2) top-1 for each path.
- **L1 (identical-prompt localiser).** B = 8 copies of T3[:700] against B = 1 through the same BatchGenerator, 24 tokens. Report the KL of each step up to the first greedy divergence, and whether the 8 rows are bitwise equal.
- **L2 (B = 1 control).** B = 1 BatchGenerator against `decode_tf` on the same tokens. Report max KL.
- **L3 (free, between waves).** Compare the batched log-probs of sequences 0 and 8 (both T1[:37]) and of 3 and 11 (both T4[:1100]) over their common prefix of tokens.
- **C (fp32 activations, real K8).** As in revision 1.
- **Removed:** A0 (replaced by stage 0) and D (chat-wrapped, which only served the withdrawn G5-W).
- **Optional, descriptive:** Q36-8's batched check with fp32 activations. A failure there makes Q36-8's own failure an mlx_lm `ArraysCache` question.

**Frozen G5 rule.** Thresholds come from the nulls in §A3, which give 0.85, 0.84 and 0 flips. The defect signs decide *whether* there is a defect; the localisers decide *where*.
- **G5-D, defect.** The failure stands for this cycle; localise, then §2. Any one of:
  1. C mean KL > 1e-4, or C top1_dis > 2/364 (a mechanics defect);
  2. mean KL(ref‖batched) > 2 × mean KL(ref‖single) (the batched path is *worse*, not just different);
  3. batched decisive top-1 against the ref < 0.995 while single ≥ 0.995;
  4. on ref-decisive positions, (batched wrong, single right) − (single wrong, batched right) ≥ 3;
  5. stage 0 verdict DEFECT (an M5 kernel at Kolibri's shapes loses precision at B = 8).
- **Localisers** (used only after G5-D; none of them is a defect test by itself):
  - **L1:** pre-divergence KL ≫ 0 means the M = 8 kernels. ≈ 0 (bitwise) means padding, mixing or admission. A different but equally precise kernel can give L1 up to ≈ 5 × decode-vs-prefill, as on Gemma (§A3), so L1 is not a defect test.
  - **L2:** max KL > 1e-6 means the batch cache classes compute differently even alone. Both nulls are exact.
  - **L3:** whether a first-wave and a mid-run copy of the same prompt agree, which isolates admission.
  - **Classes:** first wave / mid-run, t = 0 / 1–2 / 3+, prompt ≥ 513.
  - **Concentration:** whether the top-5 positions carry ≥ 50 % of the KL sum.
- **G5-N, none of signs 1–5, and parity still above the registered bound.** The batched path is different but not worse. The consequence is the one Andrei chose in D3 *before* stage 1: G5-B1 (a verdict-rule change) or the failure stands (D5).

### Stage 2: G2 (≈ 30 min)

**Additions to `diag_g2g4.py g2`:**
- **Layer 20 flip token:**
  - text, position, position class;
  - e_attn rank in layers 16–20;
  - σ_t(port) and σ_t(emu) (RMS over 384 experts of the router-logit error at that token);
  - r_t = σ_t(port)/σ_t(emu), with layer 20's p50 / p99 of r over all T1–T8 tokens;
  - gap, gap/σ_t(emu), and whether emu flips the same (token, layer).
- **All 50 layers:** σ_emu,l, emu's max gap/σ_emu,l (M_emu is the max over layers), and emu's count ≥ 6σ_emu,l.
- **Tails:** for attention layers 8 and 29 and MoE layer 13, the (text, position) of the port's max and of emu's max, and emu's per-token error at the port's max token.
- **T9 (new):**
  - natural bf16 selection disagreement fraction by bucket, port and emu (emu fed the fp32 h_l on T9; ≈ 12 min);
  - emu's forced per-layer errors on T9 for MoE layer 30 (the 11.99 tail), with the (text, position) of both maxima.

**Frozen G2 rule:**
- **G2-D, failure stands, inspect.** Any one of:
  1. r_t > layer 20's p99 of r;
  2. the token is at position 0–1, 511–515 or 2047–2049, or its e_attn in layers 16–20 is in the top 1 %;
  3. emu does not flip that (token, layer) and gap/σ_t(emu) > 6;
  4. a tail in attention 8/29 or MoE 13/30 where the port's max is > 3× emu's max on the same rows, at a position that is also among G4's 10 highest-KL positions (±2 tokens);
  5. T9 natural disagreement in any bucket > 2 × emu's in the same bucket.
- **G2-1, rule miscalibrated: A1.** No G2-D sign, and M_emu ≥ 6.0 (a correct bf16 implementation fails the registered rule itself). New value = min(8.0, 1.2 × M_emu).
- **G2-2, A3** (gate fix). No G2-D sign, M_emu < 6.0, and emu flips the same (token, layer).
- **G2-3, failure stands.** Otherwise. In particular, if 5.07 ≤ M_emu < 6.0 and emu does not flip the token, the failure stands. Revision 1's engineered 5.1 switch is gone.

### Stage 3: G4 (≈ 60–80 min)

Revision 1's stage 3 commands stand: g4 bf16, g4 fp32, t9ids, ref_deqK8_fp32, the bug test, ref_deqK8_emu, refcmp and analyse. Additions:
- **Chunk-64 exactness, fp32.** K8 with `model.set_dtype(float32)`, on T1 and T3 at 520–1,100 and T9 at 15,000–15,300:
  - prefill 2048 against prefill 64 against decode;
  - also chunk 64 against `ref_deqK8_fp32` at T9 15,000–15,300.
- **`analyse` additions:**
  - **McNemar.** One-sided, on decisive positions, port bf16 against emu (`refcmp_ref_deqK8_emu`), over T9 and per bucket: n10 (port misses, emu hits) against n01.
  - **Paired dNLL.** Port − emu per position, T9 and per bucket, 99 % 64-block bootstrap CI.
  - **dNLL/KL with 95 % CIs** for port bf16, port fp32 against deq ref, and emu.
  - **Entropy shift** H(port) − H(ref), and the same for emu.
  - **max_kl.** Positions with KL(ref‖port) > 5 nats, and emu's KL at those positions.
  - **Content-matched controls.** For both port and emu: T1, T4, T3, T5 and T6 standalone (gate record) against inside T9.
  - **The web tail** 13,399–16,384 as its own segment.

**Frozen G4 rule:**
- **G4-D, failure stands; a code question.** Any one of:
  1. bug test: T1-8 or T9 mean KL > 1e-3, or `top1_decisive` < 0.999 in any bucket, or a step within ±32 tokens of a 2,048·k chunk boundary, of 8,192, or of a document boundary;
  2. chunk-64 fp32: KL > 1e-6 at any position, or any change of top-1 where the lead is ≥ 2 nats;
  3. on T9 or any bucket, McNemar one-sided p < 0.05 *and* port misses > 1.25 × emu misses;
  4. paired dNLL port − emu on T9: the 99 % CI excludes 0 and |mean| > 0.02 nats/token;
  5. ≥ 3 positions with KL(ref‖port) > 5 nats where KL(ref‖emu) < 1 nat;
  6. the port's in-T9 excess over standalone (KL or miss rate), on T5 or T6, is > 2 × emu's excess.
- **G4-1, C1'.** No G4-D sign. New value = max(0.985, min(0.99, 1 − 1.25 × (1 − E_T9))).
  - K8 passes C1' iff emu misses ≥ 53 of 5,943 decisive positions on T9 (66 / 1.25), and only if E_T9 is measured on the same 5,943 decisive positions as the gate.
  - The factor 1.25 comes from G2's port/emu per-layer ratio (median 1.09, max 1.40) and §A4's op-level differences (≤ 1.4×).
- **G4-2, failure stands.** Otherwise.
- **Descriptive:** K8-fp32 against the fp32 reference (quantisation only) by bucket, the dNLL/KL table, and the entropy shift.

### Stage 4: G3 (≈ 30 min)

**Changes to `g3_diag.py`:**
- **Mutants on the gap texts (new, ≈ 15–20 min):** `gate.checks.ref_pass.mutant_nll(bf16_dir, ids)` on T1 and T5, with g3_oracle's 48-block bootstrap. Report each mutant's ΔNLL mean and p01 per text.
- **D3'' (rewritten).** On T1, T2, T5 and T6, report the per-128-byte-window ratio Kolibri-ref ÷ Q36-8 at p10, p50 and p90, and the fraction of windows where Kolibri is better.
- **Reference-only quantities:** the reference's bpb from token 64 against the full text, and the reference's NLL on T4 tokens inside Q36-8's recall windows (from the dump). These replace the K8 recall probe.
- **K8:** only D1 validity (K8 − ref bpb per text, a difference) and the `<|endoftext|>` effect as Δbpb. No chat-wrapped K8 and no absolute K8 values are committed.
- **Descriptive:** τ\* (NLL-optimal temperature) of the reference (from the dump) and of Q36-8.

**Frozen G3 rule:**
- **D1 validity.** |K8 − ref| ≤ 0.005 bpb on every text, else the diagnostic is invalid.
- **G3-D, misreading signs; no G3 amendment.** The route is the anchor (D4) or D5. Any one of:
  1. D3'': p10 window ratio ≥ 1.1 on all four of T1, T2, T5 and T6, with < 10 % of windows where Kolibri is better;
  2. any registered mutant with ΔNLL p01 < 0.24 nats/token on T1 or T5 (B1 could not catch it there);
  3. the reference bpb from token 64 differs from the full-text value by > 5 % on any text.
- **G3-1.** No G3-D sign: B1 is permitted. It is labelled a post-hoc gross-sanity bound, with the 0.24 nats/token (8.1 %) blind spot stated.
  - If Andrei ran the anchor (D4), its rule overrides: within 3 % on every text, amend; ≥ 5 % lower on T1, T2 or T4, reference fix (separate amendment).
- **G3-2, failure stands.** If Andrei declines B1, or the anchor reads 3–5 % lower.

### Stage 5: reporting (mini, minutes)

The main session applies the frozen mapping to the stage outputs and writes one table: check, signs found, the outcome (D / 1 / 2 / N), the frozen consequence, and its type.

It then re-applies the gate's own `evaluate` code to the recorded values with any permitted new thresholds, and *reports* the result. A FAIL stays a FAIL. The amendment states that the new values were checked against the recorded values before it was committed (§6 item 1).

**Package outcomes and cycle cost:**

| outcome | changes | (i) per re-run | (ii) per amendment | (iii) per threshold |
|---|---|---|---|---|
| any check at "failure stands", no fix | — | the gate cannot pass: D5, or a fix attempt that spends a cycle | same | same |
| a defect localised and fixed (gate fix) + G3-1 (+ G2-1, G4-1) | gate fix + threshold change | 1 | 2 (last) | ≥ 3: impossible |
| no defect anywhere; G2-1/2, G3-1, G4-1, G5-N with B1 | threshold change (+ A3 gate fix) + verdict-rule change | 1 | 2 or 3 (3 = impossible if A3 is needed) | ≥ 5: impossible |

---

## 5. Decisions for Andrei

Record these in writing before stage 1 runs, so none is chosen after the diagnostic. Each is disclosed as chosen after the gate failure.

**D1. Cycle reading.**
- (i) One cycle per whole re-run (RUNBOOK l.293).
- (ii) One cycle per typed amendment, with any number of keys inside one threshold amendment. HYPOTHESIS l.272: "…a numbered amendment … ; it counts as a fix cycle". Gate fixes and verdict-rule changes are separate amendments (l.22, l.272).
- (iii) One cycle per threshold changed (the singular "the original value, the observed value"). The l.797 precedent chose the conservative reading each time.

**Recommended: (ii).**
- HYPOTHESIS outranks RUNBOOK (l.18), and its "it" is most naturally the amendment.
- (ii) is stricter than (i) without reading a one-threshold limit into the text.

**Consequence:** the next re-run, carrying a threshold amendment plus a fix or rule change, is the last. Any failure on it is the result. Choosing (iii) means D5 now.

**D2. Diagnostic route for the gate** (the Amendment 6 precedent was a peer check).
- **Recommended: accept, with `FROZEN_RULES.md` committed before any run.**
- **Consequence:** ≈ 2.5–3 h of mbp machine time. No cycle, no S1 hours, no gate record. Outputs are restricted as in §4.

**D3. G5 consequence if stage 1 finds no defect sign (G5-N).**
- (a) **G5-B1:** K8 and K4 at B = 1, the peers' registered consequence. This is a verdict-rule change.
- (b) The G5 failure stands, which leads to D5.

**Recommended: (a).**
- It is the only non-fix option consistent with Q36-8's treatment.
- **Consequence:**
  - the plan ladder drops to P2–P6 nominal, with STOP in the pessimistic case at assumed B = 1 rates of 45–90 tok/s (unmeasured);
  - one more amendment (a cycle under (ii));
  - Amendment 7 criterion 4 then needs the cause stated on the model card.
- If a defect *is* found, the conditional fixes in §2 apply instead.

**D4. FP8 vLLM anchor.** One cloud GPU hour: vLLM 0.29 with the vendor plugin, `prompt_logprobs` on T1–T4 and T7–T8. It is pre-registered "only on Andrei's explicit ask".
- **Recommended: yes, before any G3 amendment.** It is the only test with power against a ≈ 8–9 % shared misreading. G3 is one of only two checks independent of the specification.
- **Consequence:**
  - a new script, since the kit has no tooling for it;
  - one cloud call, on Andrei's ask;
  - not a cycle.
  - Without it, a G3 amendment must state the 0.24 nats/token blind spot, and G3 stays low confidence in the post.

**D5. K4 on the mini (optional G5 discriminator).** Copy K4 (≈ 44 GB) to the mini and run the gate's `batch_parity` there.
- **Result:** ≈ 0.22 means not M5-specific; ≈ 0.02 means an M5 kernel path.
- **Recommended: only if stage 0 and L1 / L2 do not localise.**
- **Consequence:**
  - ≈ 44 GB transfer;
  - Ollama models unloaded during the run, which pauses the CasaSol bot;
  - ≈ 15 min.

**D6. Amendments 1–2 counting.**
- **Recommended:** not counted. They are typed "gate fix", but were step-3 tiny-gate fixes before any real gate. RUNBOOK counts from the step-10 FAIL.
- **Consequence:** stated in the amendment as Andrei's reading, not as a fact.

**D7. Amendment 7 wording.**
- **Recommended:** the next amendment corrects the commit title's "before any result" to "before any scored result; after the gate result", and states that criterion 4's G5 clause was written with G5's failure known.
- **Consequence:** none on the verdict; it removes a misleading line.

**D8. Stop now and publish the gate failure (always available).**
- **Consequence:** the post gives which checks failed and where, this diagnosis, and the mini nulls.
- No speed, memory or quality number for the Kolibri port is reported, and the K8 build is not uploaded.
- The peer findings (Amendments 5–6, the Gemma 4-bit fidelity result) remain publishable.

**Not an option:**
- a re-run unchanged, or with a code-only fix and no G3 decision (G3 is deterministic and fails);
- amending on current evidence (rev. 1 D0b), because every frozen rule needs a stage number.

---

## 6. Disclosure block (the amendment and the post's gate section)

1. Every amended threshold was checked offline against the recorded values before the amendment was committed. G3 is deterministic, so it passes by construction. G2, G4 and G5 pass by construction if they reproduce.
2. The cycle reading (D1) and the Amendments 1–2 reading (D6) were chosen after the failure.
3. The constants in the frozen rules were set with the gate values known: 1.2 and 8.0 (G2), 1.25 and 0.985 (G4), B1's 1.5 / 1.40 (G3), and the G5 null multipliers. They were frozen before the diagnostic ran.
4. Amendment 7 (06:40Z) postdates the gate result (05:57Z). Its G5 clause was written with the G5 failure known (D7).
5. The smallest defect each relaxed check still detects (§3).
6. G5-B1, if used: a verdict-rule change, the B = 1 / B = 8 parity with the peers, and the C15 control kept blocking in substance (Kolibri never batches).
7. The diagnostics went beyond the RUNBOOK's "diagnoses from the per-layer table" to new real-weight measurements on the same texts. The records are listed with their sha256.
8. Revision 1's outcome-shaped rules were withdrawn after review: A2, B2, C1 −0.005, G5-M / W / F, and the 5.1 switch.
9. The mini nulls (§A) and their scripts, as reference results on other models.

---

## A. Experiments run on the mini for this revision (2026-10-05)

Scripts and outputs are in `scratchpad/gatediag_rev/`. Each run took minutes. Ollama was idle (96 % memory free). Python is `exp036-mini/venv312` (mlx 0.31.2, mlx_lm 0.31.3), with `MLX_ENABLE_TF32=0`. `texts.json` (decoded T1–T9, including web text) is scratch only and must never be committed.

### A1. Precision null on a correct dense implementation

- **Script:** `e1_precision_null.py` (d3ceacfc), with `e1_analyse.py`.
- **Setup:** Qwen3-4B and Qwen3-1.7B in fp32 against bf16, q8 g64 and q4 g64 (bf16 activations), on T1–T6 and T9 (16,384 tokens).

| Qwen3-4B | KL | dNLL/KL (95 % CI) | decisive top-1 at T9 |
|---|---|---|---|
| bf16 | 0.0005–0.0007 | −2.3 … +8.5 (noise-dominated) | 1.0000 (0 / 7,998) |
| q8 | 0.0015–0.0018 | T1 −3.70 [−6.03, −1.82]; T2 +2.86 [+0.83, +5.70]; T9 −1.14 [−1.92, −0.38] | 1.0000 (0 / 7,998) |
| q4 | 0.11–0.17 | 0.25–0.53 on every text; T9 0.34 [0.26, 0.42]; 8–16k 0.25 [0.14, 0.35] | 0.9787; 8–16k 0.9751 |

- Qwen3-1.7B q4: 0.49–0.67. Its bf16 is +2.4 to +4.5, positive and significant.
- **Reading:**
  - dNLL is not ≈ +KL for real perturbations. It can be significantly negative on a given text, so the skeptic's G4 signature is withdrawn.
  - Kolibri K4's 8–16k (−0.20) and T6 (−0.41) lie outside the dense q4 range.

### A2. NLL-optimal temperature, fp32

- **Script:** `e1b_tau.py` (08386928).
- **Result:** τ\* is 1.20–1.30 for Qwen3-4B and 1.18–1.30 for Qwen3-1.7B, with a gain of 0.09–0.20 nats/token. The largest same-family difference is 0.08, on T1.
- **Reading:** the τ\* comparison cannot separate a sharpening misreading from chat-native overconfidence, so it is descriptive only.

### A3. Real-weight batch-parity nulls through the gate's own loop

- **Scripts:** `e2_batch_null.py` (e6afc2d5) and `e2b_localiser.py` (d5165cd8).
- **Setup:** registered lengths and max_tokens, B = 8, mid-run admission. Truth is the same weights with fp32 activations.

| | KL(s‖b) | floor (same positions) | KL(s‖decode) | KL(t‖b) / KL(t‖s) | decisive flips | identical B8 rows | B1 BatchGen vs decode |
|---|---|---|---|---|---|---|---|
| Qwen3-4B q8, raw text | 0.00096 | 0.00098 | 0.00082 | 0.00058 / 0.00068 | 0 / 146 | bitwise equal to B = 1 | exact |
| Gemma-4 26B-A4B MoE 4-bit, chat-wrapped | 0.0084 | 0.0083 | 0.0060 | 0.0091 / 0.0107 | 0 / 216 | rows equal; vs B = 1 pre-divergence KL 1e-7 to 0.09 per step (sorted 64-row gather path) | exact |

Batched decisive top-1 against truth was 1.000 in both.

### A4. Rounding-point sizes

- **Script:** `e4_rounding_points.py` (720d8e65).
- **Norm** (relative error against float64, synthetic data with massive-activation channels):
  - one rounding: 1.69e-3;
  - the port's order (bf16 sum, single rounding): 1.92e-3;
  - vLLM IR order (fp32 sum, round, × w, round): 2.40e-3.
- **Attention, 1,100 keys:**
  - P in fp32 (the emulation): 1.71e-3;
  - P rounded to bf16 (FlashAttention): 1.89e-3.

### A5. Kernel probe with an exact reference

- **Script:** `kprobe2.py` (88c1ce6b), output `kprobe2_m4pro.json`.
- **Bitwise equal at B = 8 and B = 1:** all q8 projections, the head and SwitchGLU at 48 / 66 rows.
- **Router fp32 GEMM:** 1.15e-7 against 8.3e-8.
- **SDPA bf16** with 300 or 700 valid keys in an 1,100-key buffer: ×1.42 against alone.
- Run against itself as a baseline: "no chip-specific B=8 precision loss".

### A6. Port vs emulation tails, from the committed record

From `phase2.json` emu max/p99 and `_layers.csv`:

| layer | port max/p99 | emu max/p99 |
|---|---|---|
| MoE 13 | 8.07 | 7.90 |
| attention 8 | 5.66 | 3.43 (port max 1.72× emu max) |
| attention 29 | 5.25 | 2.50 (port max 2.24× emu max) |

- Medians of port/emu tail ratio: attention 1.03, MoE 0.99.
- T9 rows have no emu, so MoE layer 30 at 8–16k (11.99) has no like-for-like control.

### A7. G3 arithmetic, from the record

- Bytes scored: T1 6,086, T2 6,560, T3 8,058, T4 8,424, T5 7,251, T6 7,006.
- A uniform ×1.092 NLL takes T1 to 1.200, the ratio to 1.187 and T3 to 0.451.
- B1 blind spot: +0.24 nats/token, or +8.1 %. B2: +0.11, or +3.1 %. The original bounds have no blind spot at the observed values.

---

## Open questions

- **G5:** an M5 kernel at Kolibri's shapes, or a Kolibri-specific batched-cache path. Stage 0 and L1–L3 separate them; D5 is the fallback discriminator.
- **G4:** long-context growth (KL ×2, misses ×3 at 8–16k), absent in both correct-implementation nulls. Stage 3's emu and natural-routing-by-bucket tests decide.
- **G3:** a shared misreading of ≤ 9 % is not excluded. Only the FP8 anchor tests it.
- **Which `fused_add_rms_norm` provider** vLLM 0.29 runs by default on CUDA. It does not affect any rule.
- **Q36-8's parity failure:** uninvestigated. The optional fp32-activation check in stage 1 settles whether it is an mlx_lm `ArraysCache` issue.
- **RUNBOOK l.79 against `plan_fix`** on S1 hours: harmless while S1 ≤ 9 h (unchanged).
