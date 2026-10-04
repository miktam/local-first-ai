# RUNBOOK step 4b: deep preflight, exit 2 (warning only); moved here and re-run

- **Run:** `caffeinate -i "$PY" tools/preflight.py --deep`, 2026-10-04 06:29:10–06:29:35Z, on the mbp. The mbp Claude session ran it at Andrei's request.
- **Result:** exit 2 (`return 2 if record["warnings"]`). `problems = []`.
  - The single warning was "working tree has 2 changed paths in the experiment dir". Those were the step-4a quick records, `results/preflight_20261004T061350Z.json` and `results/version_record_20261004T061350Z.json`, which were still uncommitted because the RUNBOOK commits step 4 after 4b.
- **Checks:** every check passed otherwise.
  - Every asset was at its pin.
  - 291 files were hashed with 0 mismatches. Kolibri shard 1's recorded sha256 equals an independent `shasum -a 256`.
  - High Power on AC; thermal state nominal.
  - Free disk OK.
  - L = 107.52 GiB; "K8 working set: no sysctl needed".
- **What happened next:** the two records of this run were moved here unchanged. The 4a records were committed with this note, and 4b was re-run on a clean tree.
- **For the main session:** RUNBOOK step 4 runs 4a, which writes into `results/`, then 4b, which warns on any changed path. So 4b cannot exit 0 unless the 4a records are committed first, or unless preflight ignores its own untracked records.
