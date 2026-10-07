# results/ — exp_037

Written by the kit on the run host (the mbp) and, after the hand-back, by the mini. As in exp_036, every file is UTF-8 JSON or JSONL with sorted keys, and nothing here is edited after it is written. The file layout is exp_036's (`../exp_036_kolibri_local_eval/results/README.md`), with two changes:
- The S1 convert step is replaced by `tools/refresh_builds.py`, which writes `convert/refresh_{8,4}bit_<UTC>.json`.
- The sessions' per-step decode logs (`raw/<session>/<arm>/*.steps.jsonl`, and an aborted cell's copy `../aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl`) are git-ignored working files: about 190 MB for a K8 cell at B = 1, they are never committed and never deleted, stay on the mbp, are copied to the mini at RUNBOOK step 18 and are listed with their sha256 in the S2 and S3 run records. The pilot's step logs (`pilot/<UTC>/<arm>/*.steps.jsonl`) stay committed, because the plan rule fits its step model on them; one over 50 MB stays uncommitted (RUNBOOK step 13).

RUNBOOK.md names each file with the step that writes it.

## The one inherited file

| File | sha256 | Origin |
|---|---|---|
| `manifests_20261004T131126Z.json` | `feebccff446535d3eb55b1390a3a59c34b6bfc5709a5f4cf43d354c7c9085664` | exp_036's `results/manifests_20261004T131126Z.json`, copied byte-identical at exp_036 commit `222c8450f5c1b87663ee0386c61a6eb86c353543` (recorded in `../COPY_RECORD.json`) |

- **Why it is here.** exp_037 reuses exp_036's task manifests (`tasks/manifests/*.json`, built and never scored) and the withheld manifests. This record holds the sha256 of every one of them. `runner/plan_fix.py` reads the newest `results/manifests_*.json` as an input and records its path and sha256.
- **It is exp_036's record.** exp_036's `tasks/build_manifests.py` wrote it on 2026-10-04 at 13:11:26 UTC. exp_037 does not rebuild the manifests. Instead, RUNBOOK's manifest step verifies the inherited manifests, shingles and T5/T6/T9 against their records.

## Not inherited
Every other result is exp_037's own and is measured again: the preflight, version, refresh, peer and gate records, the bench, pilot and session records, the scores and the verdicts. Under decision F2 (MLX 0.32.3 / mlx-lm 0.32.0), no value MLX computed for exp_036 is an exp_037 input. Where exp_037 reads one of exp_036's records (its convert records for P4, its version and preflight records as the provenance of P2's OS and architecture pins), it reads the file in `../exp_036_kolibri_local_eval/results/` by relative path and checks it against a pinned sha256. It never copies it here.
