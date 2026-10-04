# Amendment 1 checks on the mbp: leak check `--all` is not 0 (2026-10-04)

These are the four checks the main session asked for after `1603cd2`, run on the mbp at `698b895`. Three pass. One does not, so the mbp has stopped before RUNBOOK step 4a.

| Check | Result |
|---|---|
| 1. `git pull --ff-only`, then `tools/status.py --sync-amendments`, then commit | Amendment 1 was appended to HYPOTHESIS.md. The diff is append-only (+41 lines) and equals `amendments/1_gatefix_20261004T051845Z.md`. The leak check was clean. Pushed as `698b895`. |
| 2. RUNBOOK step 3, exactly as written | **Pass:** 1193 passed, 0 failed, 8 skipped (all `build-host only:`), 201 s. |
| 3. `tools/leak_check.py --all --no-gpqa-source` | **Exit 1:** 9 findings and 2 warnings (below). |
| 4. `tools/hash_tree.py --check HYPOTHESIS.md` | **Pass:** 17 of 17 match. Four match via Amendment 1: gate `0d280ed0136aa877`, runner `3495b7bffa172699`, bench `a69fd1281030c703` and tools `0eaa8b814ce3c701`. |

## The 9 findings: one 4-word hash in 9 places

```text
FINDING [withheld_option] gate/port_mutants.py:84
FINDING [withheld_option] reference/kolibri_ref.py:954
FINDING [withheld_option] tests/test_analysis_stats.py:118
FINDING [withheld_option] tests/test_bench_tokenizer_ratio.py:36
FINDING [withheld_option] tests/test_bench_tokenizer_ratio.py:59
FINDING [withheld_option] tests/test_gate_drivers_tiny.py:135
FINDING [withheld_option] tests/test_gate_port_mutants.py:48
FINDING [withheld_option] tests/test_gate_thresholds.py:48
FINDING [withheld_option] tests/test_scorers_score_all.py:152
  each: "matches a withheld 4-word option, hash e1c6f278f4f3 (text not shown)"
warning [email] BUILD_SPEC.md:703 (an example.com placeholder)
warning [hostname] assets.json:4 (the run_host field)
```

The six 3-word hashes from the first note are gone, as Amendment 1 intended.

### Source of `e1c6f278f4f3`

The source was traced by hash only, by reproducing `collect_from_sources` one source at a time:
- It occurs **only in `gpqa/gpqa_extended.csv`**, in one option string.
- It is **not** in `gpqa_main.csv`, `gpqa_diamond.csv`, `gpqa_experts.csv`, the `deu` parquet, AIME-DE, RGB queries or answers, or RGB `instruction.yaml`.
- The same 4-word phrase **also occurs in the public MMLU-ProX and MMLU-ProX-Lite text** (en/de). It is ordinary language, not a leak.

### Why it appears now and how step 7 should change it

- `tools/withheld_shingles.sha256` does not exist yet, so `load_withheld()` falls back to `collect_from_sources($EXP036_DATA)`. The fallback reads **all four** English GPQA CSVs, including extended, plus every answer-like column, and applies **no public-text filter**.
- Step 7 (`tasks/build_manifests.py` `withheld_texts()`) takes options from `gpqa_diamond.csv` and `gpqa_main.csv` only. So this hash should not be in the shingle file, and `--all` should then stop reporting it.
- This is expected from the code. It is not verified, because step 7 comes after the sign-off.
- Small mismatch: the `write_shingle_file` docstring recommends the public sets (MMLU-ProX, IFBench, AIME EN) as `public_texts`, but step 7 passes only `LICENSE-APACHE-2.0` and the gate texts.

### What it blocks

- **Nothing in the RUNBOOK flow today.** Step 3 passes. The commit check (`--range @{u}..HEAD --staged`) and the pre-push hook scan only new lines, and these 9 lines are already upstream.
- **Before step 7,** a commit made on the mbp that adds or edits one of these 9 lines would be flagged. The mini cannot see this, because it has no withheld data.

The main session decides whether to:
- accept this (expected 0 after step 7),
- align the fallback with the step-7 sets and filter, or
- treat it another way.

The mbp continues with step 4a when told to.
