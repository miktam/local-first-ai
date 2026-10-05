# Gate 1 diagnostic suite (exp_036, gate run `20261005T050112Z`)

**What this is.** Gate run `20261005T050112Z` passed K4 and failed K8 on five checks: G2 bf16 natural selection, the two G3 checks, the G4 backstop and G5 batch parity. These scripts measure *why*, on the mbp, without spending a gate run.
- Andrei chose this route (D2, Amendment 8).
- The rules that turn the measurements into actions are in [`FROZEN_RULES.md`](FROZEN_RULES.md). They were committed before any stage ran.
- The plan they implement is §4 of the diagnosis (revision 2), with the FP8 vendor anchor removed: D4 was approved, then withdrawn with "No need to rent anything".

**What this is not.**
- It is not a gate run. No script invokes `gate/run_gate.py` or writes under `results/gate/`.
- It is not a fix cycle and uses no S1 hours.
- A diagnostic value never replaces a gate value.

Nothing here edits a hashed tree. All files live under `diagnostics/gate1/`, which no hash scope covers.

## Run it (mbp, Andrei's terminal)

Pull the commit that adds this folder and Amendment 8, close other apps, and keep the laptop on power. Then:

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
bash diagnostics/gate1/run_stages.sh
```

The script does the following:
- **Checks before starting,** and stops with a reason if any check fails:
  - all 17 hash scopes match (`tools/hash_tree.py --check HYPOTHESIS.md`);
  - HEAD contains `a666f3e`, and nothing outside `diagnostics/gate1/`, `amendments/` and `HYPOTHESIS.md` has changed since;
  - `FROZEN_RULES.md` is committed;
  - `diagnostics/gate1/` (apart from `out/`) equals the committed files, so the scripts that run are the frozen ones;
  - `$EXP036_WORK/ref` (the gate's reference dumps) is present;
  - at least 45 GB are free on `$EXP036_WORK`;
  - the K8, BF16 and Q36-8 directories exist.
- **Runs the stages in a fixed order:** 0, 1, 2, 3 (seven steps), 4.
  - Each stage, and each step of stage 3, runs under `caffeinate -i` in its own process. Only one process loads a model at a time.
  - It sets `MLX_ENABLE_TF32=0`, and every script calls `tools.precision.ensure_exact_fp32()` first. This only affects fp32 operands. The bf16 measurements (stage 1 part B, the L1–L3 localisers, stage 2's port side, stage 3's `g4 bf16`, stage 4's K8 pass) run with the production settings, as the gate did.
- **Resumes after a stop.** If it stops or crashes, fix the cause and run the same command again.
  - A finished stage leaves `$EXP036_WORK/diag_gate1/<stage>.done.json` and is skipped.
  - Stage 3's finished steps are skipped the same way.
  - A stage that completed is never re-run to replace its output (`FROZEN_RULES.md` §1.6).
- **Optional:** `DIAG_WITH_Q36=1 bash diagnostics/gate1/run_stages.sh` adds stage 1's descriptive Q36-8 check (fp32 activations, peer batched path). It decides nothing for Kolibri.
  - Set it on the **first** invocation. Once stage 1 has finished it cannot be added: the stage is skipped as done and is never re-run (`FROZEN_RULES.md` §1.6). The script prints a warning if the flag arrives too late.

### Stages, durations and memory

Durations are estimates for the M5 Max, scaled from the gate run's phase timings.

| stage | script | what it measures | time | peak memory |
|---|---|---|---|---|
| 0 | `stage0_kernels.py` | runs `kprobe2.py` unchanged (sha256 `88c1ce6b…`) against its M4 Pro baseline `kprobe2_m4pro.json`: Kolibri's decode-shape kernels at B = 8 against B = 1, against float64. No weights | ≈ 1 min | < 2 GB |
| 1 | `stage1_g5.py` | G5. Part B: the gate's `batch_parity` (must reproduce 0.3073773544) and the per-position decomposition. L1–L3 localisers. Part C: fp32 activations. Part R: the fp32 reference on the 12 batched sequences | ≈ 25–45 min | ≈ 90 GB (K8, then K8 with fp32 activations), then ≈ 15 GB (reference) |
| 2 | `stage2_g2.py` | G2, using the gate's harness one layer at a time: the layer-20 flip token (σ_t, r_t, e_attn ranks, whether emu flips it), emu's own statistic on all 50 layers (M_emu), the tails of attention 8/29 and MoE 13/30, and T9 natural selection by bucket for port and emu | ≈ 20–30 min | ≈ 10 GB |
| 3 | `stage3_g4.py` (7 steps) | G4. `g4` (K8 bf16, then fp32 activations, against the dump; the bf16 pass must reproduce 0.98889). `t9ids`. `ref_fp32` and `ref_emu` (the reference on the dequantised K8 weights). `bugtest` (bug test and chunk-64 exactness). `refcmp` (E_T9). `report` (McNemar, paired dNLL, controls, top-10 KL positions) | ≈ 55–80 min | ≈ 92 GB (K8 steps); ≈ 30–40 GB (reference steps); disk ≈ 31 GB in `$EXP036_WORK/diag_gate1/stage3_g4/` |
| 4 | `stage4_g3.py` | G3 without an anchor. The 7 mutants on T1 and T5. 128-byte window ratios against Q36-8, whose pass must reproduce the peers record the gate cites within 1e-4 bpb per text. Reference bpb from token 64. D1 validity and the `<|endoftext|>` effect (differences only). τ* (descriptive) | ≈ 20–30 min | ≈ 85 GB (K8), then ≈ 40 GB (Q36-8) |
| total | `run_stages.sh` | stages 0–4 | ≈ 2–3 h | |
| 5 | `stage5_report.py` | **on the mini, later:** applies `FROZEN_RULES.md` mechanically to the committed outputs and re-applies the gate's own `evaluate` code with only the permitted thresholds | minutes | < 2 GB |

## Outputs, and what is committed

Each stage writes two files:
- **`$EXP036_WORK/diag_gate1/<stage>_<UTC>.json`**, the full record. Its `work_only` part holds:
  - per-position rows;
  - K8's absolute bpb and entropy (`k8abs_*`), and the pooled relative `<|endoftext|>` effect, which would give the pooled K8 bpb back;
  - K8's own top-1 lead (stage 1's `*_k8_own_lead`);
  - the flip token's id.
- **`diagnostics/gate1/out/<stage>_<UTC>.json`**, the restricted copy. Its top-level `rule_inputs` holds exactly the keys of `FROZEN_RULES.md` §2, followed by descriptive comparisons. Before writing, `diaglib.leak_scan` refuses:
  - token ids, text and host paths;
  - any `k8abs_*` key.

**Committed** (by the mbp session, after the run, as below): `diagnostics/gate1/out/stage*_*.json` only, plus HYPOTHESIS.md if `--sync-amendments` appended anything.

**Never committed:**
- anything in `$EXP036_WORK/diag_gate1/`: the `.npz` and `.npy` files (per-position arrays and logits), `ids.json` (T9's web-text ids), the batched tokens, the logs, and the full JSONs;
- absolute K8 NLL, bpb or entropy, and K8 recall counts. They stay in the work file unless the gate passes.

K8 chat-wrapped NLL is never computed, and no K8 recall probe is run.

**A disclosed limit** (`FROZEN_RULES.md` §1.8). A committed difference plus a committed reference value gives back the K8 value:
- per text, K8's bpb is the reference bpb plus `d1_k8_minus_ref_bpb`;
- adding the per-text `<|endoftext|>` Δbpb gives K8's prefixed bpb.

This is accepted on the precedent of the gate record, whose per-text `dnll_mean` (K8 minus reference) already sits next to the reference bpb.

## After the run

The restricted copies are written into the mbp's checkout, so the mbp session commits them. The main session on the mini cannot see them until they are pushed. Uncommitted files there would also block the runner's later steps (RUNBOOK step 1: no untracked path outside `results/`, `aborted/` and `evidence/`).

1. **mbp session (Claude may run; seconds).** After "stages 0-4 complete":
   ```bash
   cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
   git -C "$LFA" pull --ff-only && \
   "$PY" tools/status.py --sync-amendments && \
   git add HYPOTHESIS.md diagnostics/gate1/out/stage*_*.json && \
   "$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
   git -C "$LFA" commit -m "chronos/exp_036: gate-1 diagnostics (stages 0-4)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
   git -C "$LFA" status --porcelain
   ```
   - Add nothing else. In particular, add nothing from `$EXP036_WORK`.
   - The last command must print nothing outside `results/`, `aborted/` and `evidence/`.
   - **Push only after Andrei's go.** His pre-approval (RUNBOOK, "Pushes") names the sign-off, manifests, records, amendments and S2/S3 outputs. It does not name diagnostic outputs.
2. **Main session on the mini,** after `git -C ~/REPOS/local-first-ai pull --ff-only`, runs stage 5 with the mini's own interpreter. `env/exp036.env` points `PY` at the mbp's venv, which does not exist on the mini, and stage 5 needs no `EXP036_*` variable:
   ```bash
   cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && \
   ~/models/exp036-mini/venv312/bin/python diagnostics/gate1/stage5_report.py
   ```
   **Integrity.** It refuses to run (`FROZEN_RULES.md` §7) if:
   - its own sha256, another script's here, `FROZEN_RULES.md`'s or the gate record's differs from what the stage headers recorded;
   - the tree is dirty;
   - a stage has more than one output, unless the file is named (`--stage1_g5 FILE`) with the defect and reason (`--rerun-reason stage1_g5 "…"`).

   It reads only `rule_inputs` from `out/stage{0..4}_*.json`, plus the committed gate record. It writes `out/stage5_report_<UTC>.json`, which holds:
   - every sign, with its inputs, its value, and whether it fired;
   - the reproduction spread;
   - each check's outcome, its consequence and type, and who acts;
   - the gate's own `evaluate` code (`g2_layers`, `g3_oracle`, `g4_e2e`, `run_gate.arm_verdict`) re-applied to the recorded values with the permitted thresholds only. G5-B1 shows as "non-blocking under a verdict-rule change", never as a pass;
   - every amendment package the permitted consequences allow, and its cycle cost under D1 (ii). Threshold changes share one amendment and gate fixes share one; each verdict-rule change is its own.

   It chooses nothing:
   - Andrei types any threshold change;
   - the main session drafts any gate fix or verdict-rule change;
   - each verdict-rule change (G5-B1, A3, the M5-kernel B = 1 workaround) needs Andrei's go.

## One measurement detail fixed before the run

**The bug test compares K8 with fp32 activations against the reference on the dequantised K8 weights.** `model.set_dtype(float32)` alone does not give the port those weights:
- `QuantizedEmbedding` dequantises in its scales' dtype, so the cast turns the stored bf16 dequantisation into an fp32 one;
- the reference's dequantised mode reads the stored bf16 dequantisation (`tests/test_convert.py`, `test_dequantised_mode_reads_the_mx_dequantize_values`).

On the tiny K8 build this alone gives a mean KL of 1.8e-3, over the bug test's 1e-3 bound, with no defect anywhere.

**The fix.** `diaglib.fp32_on_dequantised_weights` keeps the embedding at its stored bf16 scales and biases (an exact round trip) and casts its output to fp32. The bug test then measures 8e-13 on the tiny build (1.5e-10 on a peaked tiny build). The alignment is part of the frozen rules (`FROZEN_RULES.md` §0 item 5, §2 stage 3) and disclosed as chosen after this smoke measurement (§9 item 4).

**Where it applies.** Stage 3's `bugtest` step (bug test and chunk 64) and its descriptive `g4 fp32` pass. Stage 1's part C keeps a plain `model.set_dtype(float32)`, as `FROZEN_RULES.md` §2 states; it compares the port with itself.

## Files

| file | role |
|---|---|
| `FROZEN_RULES.md` | the decision rules, frozen before any stage ran |
| `diaglib.py` | shared plumbing: context (`--tiny` for smoke tests), the two outputs, `leak_scan`, the numerics helpers, the fp32 dequantised-weights alignment |
| `kprobe2.py`, `kprobe2_m4pro.json` | the kernel probe and its M4 Pro baseline, unchanged copies (sha256 checked by stage 0) |
| `stage0_kernels.py` … `stage4_g3.py` | the five mbp stages |
| `stage5_report.py` | the frozen mapping (mini) |
| `run_stages.sh` | stages 0–4 in order, with the preconditions, `caffeinate`, logs and resume |
| `out/` | restricted copies (committed) |

Every script takes `--tiny ROOT`: the same code on a tiny checkpoint root, with outputs kept under ROOT. That is how the suite was smoke-tested on the mini.
