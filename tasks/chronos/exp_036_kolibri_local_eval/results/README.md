# results/ — layout

Written by the kit on the run host (the mbp) and, after the hand-back, by the mini. Every file is UTF-8 JSON or JSONL with sorted keys; nothing here is edited after it is written (HYPOTHESIS "Evidence layout"; BUILD_SPEC §2 Append-only). Withheld sets (GPQA EN/DE, RGB, AIME-DE) carry hashes only; their text is in `$EXP036_PRIVATE`.

| Path | Written by | RUNBOOK step |
|---|---|---|
| `preflight_<UTC>.json`, `version_record_<UTC>.json` | `tools/preflight.py`, `tools/version_record.py` | 4, 17 |
| `manifests_<UTC>.json` | `tasks/build_manifests.py` (sha256 of every manifest) | 7 |
| `gate_texts_<UTC>.json` | `gate/build_gate_text.py --work-only` (only if MANIFEST.json lacks the web texts) | 7 |
| `convert/convert_{8,4}bit_<UTC>.json` | `port/convert.py` | 8 |
| `peers_<UTC>.json` | `tools/peer_check.py` | 9 |
| `gate/gate_<UTC>.json` (+ `_layers.csv`, `_mutants.json`, `<UTC>/phase*.json`) | `gate/run_gate.py` | 10 |
| `bench/{speed,speed_desc,fit,c1,ladder}_<UTC>.jsonl`, `tokenizer_<UTC>.json`, `kl_8v4_<UTC>.json` | `bench/` | 11 (ladder: B4) |
| `pilot/<UTC>/<arm>/*.jsonl` (+ `.steps.jsonl`), `pilot_summary_<UTC>.json` | `runner/run.py pilot` (never scored) | 12 |
| `plan_fixed_<UTC>.json`, `AMENDMENT_<k>_<UTC>.md` | `runner/plan_fix.py` | 13 |
| `raw/<session>/<arm>/<task>_<effort>.jsonl` (+ `.steps.jsonl`, `.heartbeat`, `session.jsonl`, `not_run.jsonl`) | `runner/run.py session` | 14, 16 |
| `status_<UTC>.json` | `runner/run.py status` | any |
| `scores/<arm>/<task>_<effort>.jsonl` | `scorers/score_all.py` | 17 (IFBench: the mini) |
| `rescore_mini_<UTC>.json` | `scorers/score_all.py --rescore-compare` (the mini) | after 19 |
| `verdicts_<UTC>.json`, `verdicts_<UTC>.md` | `analysis/verdicts.py` (the mini) | after 19 |

Next to `results/`: `evidence/withheld_manifest.jsonl` (sha256 of every private file), `aborted/<UTC>-<what>/NOTE.md` (never deleted), and `amendments/` (the mini's amendment inbox during the run; the mbp appends each file to HYPOTHESIS.md with `tools/status.py --sync-amendments`).
