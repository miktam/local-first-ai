# Exp 015 — Active-Parameter Ablation: Dense vs MoE on the Audit Rubric

*Pre-registered: 2026-06-09*  
*Status: PRE-REGISTERED — awaiting execution*  
*Full log entry: [`scientific_log.md` § Exp 015](../scientific_log.md)*

---

## Motivation

Exp 012 found `gemma4:26b` (MoE A4B, ~4B active per forward pass) scored 0/8 on
cross-document auditing tasks. Exp 012-Alpha narrows the scope: Exp 012 tested one point
in the local model space — a large-total-parameter MoE with small active compute per token.

Open question: is the bottleneck *total parameters* (knowledge breadth — MoE provides
cheaply) or *active compute per token* (per-token reasoning depth — dense models provide)?
Exp 009 and 012 are consistent with the second reading: gemma4:26b matched the compliance
knowledge layer (recall-shaped, benefits from breadth) and failed the cross-document
reasoning layer (computation-shaped, benefits from depth).

---

## Hypotheses

**H1 (bottleneck is active compute):** A dense local model with ≥12B active parameters
per forward pass scores ≥2/8 on the frozen Exp 012 audit rubric without scaffolding.
Falsified if no tested dense model scores above 1/8.

**H2 (model selection principle):** If H1 confirmed, per-token active compute predicts
audit rubric performance better than total parameter count. MoE models suit transduction
and recall stages; dense models suit adjudication stages. Falsified if MoE and dense at
matched active-parameter counts perform equivalently.

---

## Design

1. Select candidate dense models runnable on 64 GB unified memory at Q4: target 12–32B
   active params (e.g. `qwen2.5:32b`, `mistral-small3.2`, or similar via Ollama). Confirm
   active-parameter count from model card; cross-check via observed decode throughput.
2. Run 3 reps per model on the frozen Exp 012 rubric (Task A + Task B, identical context
   bundles, identical scoring). Rubric committed before first run.
3. Score blind. Compare net scores against gemma4:26b baseline (0/8) and Haiku baseline
   (5/8 mean, variance documented in Exp 014).

---

## Pass criteria

| Criterion | Threshold |
|---|---|
| H1 confirmed | ≥1 dense model scores ≥2/8 net |
| H1 strong | ≥1 dense model scores ≥4/8 net (within frontier range) |
| H1 falsified | No tested dense model scores >1/8 |
| H2 confirmed | MoE/dense scores separate cleanly across ≥2 dense models |

---

## Artifacts structure

```
exp_015_active_param_ablation/
├── README.md          ← this file
└── measurements/      ← created at execution time
    ├── rubric.md      ← frozen before first run
    └── YYYYMMDDTHHMMSSZ-<model>-rep<N>.json
```
