# RUNBOOK step 10: port gate, K8 FAIL and K4 PASS (2026-10-05)

- **Run:** Andrei ran `caffeinate -i "$PY" gate/run_gate.py --all`, 05:01:12–05:57:03Z, in phases 1–6. Peak RSS in phase 3 was 39 GB. Reference dumps total 57 GB in `$EXP036_WORK/ref/`.
- **Result:** exit 1, verdict `{"K4": "PASS", "K8": "FAIL"}`.
- **Record:** `results/gate/gate_20261005T050112Z.json` (+ `_layers.csv`, `_mutants.json`, `20261005T050112Z/phase1–6.json`, `g5_continuations.json`), committed with this note.
- **Context:** peer record `peers_20261005T045649Z.json` (Amendment 6).

## The five failing K8 checks (`failing.K8`, in record order)

| check | measured | limit | note |
|---|---|---|---|
| `g2_bf16_natural_selection` | `max_gap_over_sigma` 6.08, `n_beyond_6sigma` 1 | 6σ | `disagree_frac` 0.01375 is within its limit 0.0247 (emulation 0.0124); a single outlier |
| `g3_bpb_per_text` | T1 **1.311** (T2 1.145, T3 0.493, T4 0.497, T5 1.188, T6 1.057; mean 0.914) | per text ≤ 1.2 | raw text after the context-start token |
| `g3_bpb_vs_peers` | K8 mean 0.914 vs best peer Q36-8 0.705, ratio **1.30** | ≤ 1.25 | the peer values are the raw-text `nll` from the peers record; G8's raw value there is 3.57 |
| `g4_K8_backstop` | T9 `top1_decisive` **0.9889**; bucket 8192–16384 `mean_kl` **0.108** | ≥ 0.99; mean KL ≤ 0.1 | T1-8: `top1_decisive` 0.9965, `mean_kl` 0.041; T9 `mean_kl` 0.083; `max_kl` 14.4 (T2), 14.8 (T9) |
| `g5_batch_parity_K8` | `mean_kl` **0.307**, `top1_dis` 0.143 | KL ≤ 0.151; dis ≤ 0.199 | B = 8 with 4 admitted mid-run, 12 sequences, 364 positions |

## Points for the diagnosis

- **K4 does not pass because K4 is better.** K4's G4 and G5 checks are descriptive (non-blocking), and its values are worse than K8's:
  - `g5_batch_parity_K4` `mean_kl` 0.223, above the same 0.151 bound.
  - `g4_K4` T1-8 `mean_kl` 0.128, `top1_decisive` 0.988.
  - The batch-parity gap shows in both arms, and K8 is the arm whose checks block.
- **Passing K8 checks:**
  - `g2q_K8`: embedding and head error 0.0.
  - `g2_bf16_forced_branch`.
  - The fp32 checks; Amendment 1's exact-fp32 setting is in effect.
- **Two checks are marginal:** G2 at 6.08σ against 6σ, and G4 at 0.9889 against 0.99. The G5 failure is 2× its bound. The G3 failures are a raw-text bits-per-byte comparison with the Qwen3.6 peer.

Per the RUNBOOK, the run stops here: no bench, pilot or scored run. The main session diagnoses from `gate_20261005T050112Z_layers.csv` and pushes a fix with `amendments/<k>_gatefix_<UTC>.md`. This is fix cycle 1 of at most 2. A threshold change needs Andrei's own typed amendment.
