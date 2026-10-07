# Tiny stress records (DESIGN §11.13, decision (a))

Two runs of the whole exp_037 gate `--tiny`, with the same code and the same command, on two builds of the tiny real-layout checkpoint that differ only in the q/k-norm factor, and a third run, added by the final review (W15-05), on the unsharpened build with the free-routing checks on bf16 activations (`--tiny-precision bf16`) instead of the tiny default fp32. They are the committed source of DESIGN §11.13's stress finding, so the finding cites committed files rather than scratch logs (§11.11). They are pre-freeze diagnostics, not exp_037 results, and `diagnostics/` is outside every hash scope.

| Folder | Build | Gate record | Exit | Outcome |
|---|---|---|---|---|
| `sharpened_seed29/` | `tests/tiny_real_layout.py` `build_real_layout_set(DIR, 29)`: seed 29, q/k norms × 3 (conftest `tiny_real_val`, F_tiny's validation build) | `gate_20261006T124326Z.json`, sha256 `e53158e91e76821ee7460633342655ed15b92d2745eb179339cf3577a6e50548` | 1 | K8 FAIL (g4_n_K8, g5_d32_K8), K4 PASS. Expected; this run is not a test |
| `unsharpened_seed29/` | `build_gate_validation_set(DIR)`: seed 29, q/k norms × 1 (conftest `tiny_real_val_unsharp`, the validation build of the whole tiny gate) | `gate_20261006T124326Z.json`, sha256 `9547697b15ec127bffb0787eca5e54836e827197fc92378eda3b85d61546201c` | 0 | K8 PASS, K4 PASS |
| `unsharpened_seed29_bf16/` | the same build, run with `--tiny-precision bf16`: G4-N(i), the noise floor, decode-vs-prefill, G5-R1 with control 26, G5-BP-lean with control 27 and probe 22, and behaviour on bf16 activations, as the real gate runs them | `gate_20261006T151049Z.json`, sha256 `67ce52f050dd7c1f5beeddf3d3a2255d2a8d64c4cadb8ae4f3d6db4f9978d30b` | 0 | K8 PASS, K4 PASS; every required control caught |

Each folder holds the gate's own output, unchanged: the record, its `_layers.csv` and `_mutants.json`, and the phase JSONs in `<UTC>/` (the record's `phase_dir`, `results/gate/<UTC>`, sits next to the record here; the bf16 run's also holds the `run.json` that gate runs write since W15-03). Every phase file matches the sha256 in the record's `phase_files`.

## How they were run

On the mini (M4 Pro), 2026-10-06, 12:43–12:50 UTC, the two runs side by side. Python 3.12 in the exp_037 venv, MLX 0.32.3, mlx-lm 0.32.0, macOS 26.5.1 (the record's P2 `observed`). Both records carry `gate_rules_sha256` `488dfa7d…` and the same `code_hashes` (reference tree `85337ed7…`, port `2c153357…`). The kit was not yet committed, so the records' `git` field says dirty.

```
"$PY" gate/run_gate.py --tiny DIR --tok "$EXP036_TOK" --results-dir DIR/results --head-policy quantised_head
```

This is the gate stage of `tools/dry_run.py`, with every phase (G0k at the P3 shapes, G1, the control mutants of phase 4 and the tiny-mode probe 22) and F_tiny from `gate/calibration.json` (written 2026-10-06T12:33:45Z).

The bf16 run (final review W15-05) was made on the mini on 2026-10-06, 15:10–15:16 UTC, Python 3.12, by the same command with `--tiny-precision bf16` added, on a fresh `build_gate_validation_set(DIR)` build. Its phase 4 carries no `tiny_precision` note, unlike the fp32 records' ("free-routing checks with fp32 activations"): the free-routing checks ran on the model as loaded, as in a real run. It ran on the final review's code: `gate_rules` `5e9e1a53…` (G5-D32's factor 100), gate code `1d413283…` (the interrupt handling), port `2c153357…`, reference `85337ed7…`, F_tiny from `gate/calibration.json` written 2026-10-06T15:10:32Z (sha256 `234a9745…`, F_tiny 4.1119e-7 unchanged).

Not committed: the builds, and the G5 log-prob arrays under `$EXP036_WORK/exp037` (`*.npy`, which the leak check refuses). The fixture ceiling below was computed from those arrays, each checked against the sha256 in the record's `g5_logprob_files`, with `tests/test_gate_end_to_end_tiny.py`'s `fixture_ceiling`.

## What they show

| Check (bound) | Sharpened seed 29 (× 3) | Unsharpened seed 29 (× 1) | Unsharpened, bf16 free routing |
|---|---|---|---|
| G4-N(i) K8, mean KL(R1‖K8), T1–8 / T9 (≤ 0.10) | 0.378 / 0.393: FAIL | 0.0229 / 0.0195: PASS | 0.0245 / 0.0224: PASS |
| G4-F16's calibrator R3 = KL(R2F‖R3), T1–8 / T9 (ceiling 1e-2) | 0.127 / 0.158: G4-F16 INCOMPLETE ("VOID: R3 above ceiling") | 5.5e-5 / 5.1e-5: PASS | the same (G4-F16 is bf16 in every run) |
| G4-F16 control 6, κ ratio (margin 90) | 3.80: not caught | 2,992: caught, margin met | 2,992: caught, margin met |
| G5-D32, unmutated port: mean KL against its bound (factor 10 in these two records; 100 since the final review, W15-04) | over the bound by 1.14–1.58× on T1 and T3 decode and chunk 64, and by 1.63× on T9 chunk 64 (T9 decode 0.96×): FAIL. The max and top-1 legs are silent. At factor 100: 0.11–0.16 of the bound | at most 1.63e-4 of the bound: PASS | factor 100: PASS (fp32, as in every run) |
| G5-R1 control 26, R-parity (margin 10 × the bound) | 14.7×: caught, margin met | 1,408×: caught, margin met (bound the 1e-4 floor; mutant mean KL 0.141) | **4.63×: caught, margin not met** (bound 3 × the bf16 chaos floor = 0.032; mutant mean KL 0.148, top-1 disagreement 0.76) |
| G5-BP-lean control 27, (d) ratio, first wave / mid-run (decision (e): > 3 on both) | 1.57 / 1.28: not caught, so allowed_B = {1} for both arms ("power not shown") | 11.93 / 6.41: caught on both subsets; the §4.2 margin of 30 is not met, decision (e) is | 10.50 / 7.01: caught on both; decision (e) met |
| Fixture ceiling for (d): the ratio of an unrelated-context swap, first wave / mid-run | 2.23 / 2.06 | 22.27 / 19.12 | 19.42 / 20.20 |
| Probe 22, (d) ratio, first wave / mid-run (no margin) | 1.76 / 1.70: (d) does not fire | 5.70 / 5.75: (d) fires | 5.41 / 5.35: (d) fires |
| G4-F32 controls 1, 6, 19 (count / mean margins) | 64 / 1.20e6 / 1.41e6: caught, met | 64 / 4.01e5 / 3.10e5: caught, met | 64 / 4.01e5 / 3.10e5: caught, met |
| G5-D32 control 20 (mean margin) | 3.41e7: caught, met | 2.30e7: caught, met | 2.30e7: caught, met |
| G2 registered mutants | caught, none undetected | caught, none undetected | caught, none undetected |

- **The sharpened build.** Its failures come from the fixture, not from the code, by two routes (DESIGN §5.1). The G4-N(i) failure, the G4-F16 VOID and the two uncaught controls come from 8-bit quantisation of the sharpened random weights, which the × 3 q/k sharpening makes unrepresentative. The G5-D32 failure does not: that check is fp32 and port against port (forced decode and chunk 64 against forced chunk 2,048 on the same dequantised weights), and the sharpening amplifies the kernel rounding between those paths past one decade over Csort's single re-association; the final review set G5-D32's factor to 100, G4-F32's, under which these values pass (W15-04). Two of its controls are not caught for the quantisation reason. κ's control 6 reaches only 3.8 × R3, and R3 is itself VOID, and on G5-BP-lean even a full context swap reaches only about 2.2, under the gate's factor of 3, because KL(R1‖single) is already 0.28–0.31 there (0.027–0.032 on the unsharpened build). No threshold changed for any of this.
- **The unsharpened build in bf16** (W15-05). The tiny gate's free-routing checks run on fp32 activations by default; the real gate runs them in bf16. On bf16 activations the unsharpened build still passes with every required control caught, and 27 meets decision (e)'s acceptance. Control 26's tiny margin is not met there: its effect is the same as in fp32 (mean KL 0.148 against 0.141), but R-parity's bound rises from the 1e-4 floor to 3 × the bf16 decode-vs-prefill chaos floor (0.032), so 26 sits at 4.63 × its bound, under the 10 × margin. The §4.2 margins are judged on the default fp32 run; this record shows what is left of 26's power at production precision on a near-uniform fixture.
- **The unsharpened build** passes with every required control caught. Every control except 27 meets its §4.2 margin; 27 meets decision (e)'s acceptance, with the ceiling beside it. Its values are identical to those of the end-to-end test (`tests/test_gate_end_to_end_tiny.py`) in Python 3.12 and 3.14. The tiny control table, `tests/fixtures/exp037_tiny_controls.json`, is built from this record.
- **Not applicable on tiny weights** (`run_gate.TINY_NOT_APPLICABLE`), on both builds: G5-R1 K8's R-greedy leg fires (R-anchor and R-parity hold), and g5_greedy and the behaviour cells fail. G3's rows (bpb per text, bpb against peers, the reference mutants) are not applicable either.
- **Entropy.** On both builds the generated positions' next-token entropy averages 6.61 nats against log V = 6.93, and the mean KL between unrelated positions is 0.61 (unsharpened) and 0.63 (sharpened).
