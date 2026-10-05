# Gate-1 confirmation runs (after stage 5)

Stage 5 (`out/stage5_report_20261005T130616Z.json`) applied `FROZEN_RULES.md` to the mbp diagnostics:
- G2, G3, G4 and G5 all read `-D`, so their failures stand;
- no amendment package exists, and K8 is re-evaluated as FAIL.

Andrei's decision (2026-10-05) was **"Confirm, publish, then exp_037"**: run these confirmations, publish exp_036 as a gate failure, then pre-register a follow-up.

These runs **decide nothing about exp_036's verdict**, which is fixed. They test, on Kolibri's real weights, the explanations the mini's bug hunt found on tiny models and on Gemma 4 (`../BUGHUNT.md`), so that the write-up rests on the real model. They feed the exp_037 gate design. They write no gate record and use no fix cycle.

Run from the kit on the mbp, with `source env/exp036.env` done (it sets `$PY`, `$EXP036_WORK` and the model paths), in this order:

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env
mkdir -p "$EXP036_WORK/confirm"
U=$(date -u +%Y%m%dT%H%M%SZ)
# 1. Forced routing on the real K8, fp32 on the dequantised weights (needs stage 3's reference logits in $EXP036_WORK/diag_gate1/):
#    the chunk-64 test with expert choices forced, and the bug test with a vLLM-angle RoPE in the port.
PYTHONDONTWRITEBYTECODE=1 caffeinate -i "$PY" diagnostics/gate1/confirm/mbp_verify_rope_chaos.py --out "$EXP036_WORK/confirm/rope_chaos_$U.json"
# 2. G5: the M5 kernel probe at the first wave's prefill shapes, against the M4 Pro baseline (random weights, minutes).
"$PY" diagnostics/gate1/confirm/kprobe3_prefill.py diagnostics/gate1/confirm/kprobe3_m4pro.json > "$EXP036_WORK/confirm/kprobe3_m5_$U.json"
# 3. G5 A/B on the real K8. Arm A must reproduce 0.3073773544462904 exactly; arm C prefills each first-wave prompt alone (prefill_batch_size 1).
caffeinate -i "$PY" diagnostics/gate1/confirm/g5_ab_mbp.py --arms AC --out "$EXP036_WORK/confirm/g5_ab_$U.json"
```

## What each run settles

| Run | Prediction if the bug hunt is right | What would contradict it |
|---|---|---|
| 1, forced chunk 64 | KL(prefill 2048 ‖ prefill 64) ≤ 1e-6 with 0 top-1 changes once routing is forced; the free run reproduces mean KL 0.0523 | forced KL stays well above 1e-6: a real chunking defect |
| 1, vLLM-angle RoPE | forced KL(reference ‖ port) at T9 drops from about 1e-3 to about 1e-6 and stays flat with position | no drop: the long-context gap is not RoPE |
| 2, kprobe3 on the M5 | a B=8 right-padded prefill op loses precision against B=1 on the M5 but not on the M4 Pro: an MLX kernel path specific to the M5 | no flag: G5's first-wave excess remains unexplained |
| 3, G5 A/B | arm A reproduces 0.3073773544; arm C (each prompt prefilled alone) brings the first wave down to the mid-run level, about 0.01 | arm C stays high: the cause is not the batched prefill |

## Committing

These outputs hold only KL values, counts and kernel errors, no text and no K8 quality figures. Copy the JSON files into `diagnostics/gate1/confirm/out/` and commit them with the usual leak check:

```bash
mkdir -p diagnostics/gate1/confirm/out && cp "$EXP036_WORK"/confirm/*_"$U".json diagnostics/gate1/confirm/out/
"$PY" tools/status.py --sync-amendments
git add diagnostics/gate1/confirm/out/ HYPOTHESIS.md
"$PY" tools/leak_check.py --range @{u}..HEAD --staged
git commit -m "chronos/exp_036: gate-1 confirmation runs (forced routing, RoPE, G5 A/B)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Push with Andrei's go.

## Follow-up: the seeded single-op reproduction (`mlx_repro.py`)

This ran after exp_036 closed. It is investigation only and decides nothing.

`mlx_repro.py` calls the single op behind mlx-lm's `SwitchGLU` projections, `mx.gather_qmm`.
- **Inputs.** Weights and inputs come from a seeded numpy generator, so they are bitwise identical on every machine.
- **Reference.** A float64 computation from the dequantised weights.
- **Recorded.** The OS, the MLX and mlx-lm versions, and the GPU architecture.
- **Tests A–H** separate the sorted path, dependence on pad content, padding against size, a hot expert, total size, float32 and the sort itself (see the docstring).

The M4 Pro baseline is `out/mlx_repro_m4pro_*.json`:
- every bf16 case is at 0.0017–0.0023, which is bf16 output rounding;
- float32 is at 1e-6;
- valid rows do not depend on pad content;
- repeat runs are bitwise identical.

On the mbp it takes about a minute, and no model is loaded. Run it from the kit with the usual environment loaded (it sets `$PY`):

```bash
cd ~/REPOS/local-first-ai && git pull --ff-only && cd tasks/chronos/exp_036_kolibri_local_eval
U=$(date -u +%Y%m%dT%H%M%SZ); "$PY" diagnostics/gate1/confirm/mlx_repro.py --bits4 --out diagnostics/gate1/confirm/out/mlx_repro_m5max_$U.json
git add diagnostics/gate1/confirm/out/mlx_repro_m5max_$U.json && "$PY" tools/leak_check.py --range @{u}..HEAD --staged
git commit -m "chronos/exp_036: mlx_repro on the M5 Max" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Push with Andrei's go. The `flags` list in the printed summary says whether the M5 reproduces the defect at the level of the op.
