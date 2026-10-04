# RUNBOOK step 7 stopped again: two MMLU-ProX data assumptions (2026-10-04)

## The Amendment 3 checks pass (mbp)

- Amendment 3 was synced. The append-only diff (+45) equals `amendments/3_itemset_20261004T093556Z.md`. Pushed as `6bb80a7`.
- Step 3 passes: 1198 passed, 0 failed, and 8 `build-host only:` skips, which is the main session's 1206.
- `hash_tree --check`: 17 of 17 match. tasks, manifest_rule, analysis and tools match via Amendment 3; gate, runner and bench via Amendment 1.

## Step 7 (`"$PY" tasks/build_manifests.py`) at 09:55:49Z

GPQA now passes, with `gpqa_diamond_en_overlong_excluded: 0` and primary n 198. The build then stopped at:

```text
{"ok": false, "error": "manifest mmlu_prox_lite_en: ValueError: MMLU-ProX-Lite: categories without 42 parallel ids: {...}"}
```

To get the whole picture in one pass, RUNBOOK 3c was then run one set at a time, into `$EXP036_WORK/manifest_check/` only. **15 of 21 sets build** with the expected counts: `gpqa_*` (198, 198, 8, 8), `aime_en` 30, `aime_pilot_en` 4, `aime_de` 30, `ifbench` 300, `ifbench_pilot` 8, and `rgb_cb` 400, `rgb_forced` 400, `rgb_negative` 300, `rgb_fact` 100, `rgb_pilot_cb` 16, `rgb_pilot_forced` 8. **The six MMLU-ProX sets fail, for two independent reasons.** Both datasets are public, so ids and counts are given here.

### 1. MMLU-ProX-Lite is not 42 per category (`mmlu_prox_lite_en`, `mmlu_prox_lite_de`)

At the pinned revision `e82aafb9`, the test split has 588 rows per language in 14 categories, but the categories are uneven. EN and DE are identical:

| category | n | category | n |
|---|---|---|---|
| biology | 36 | history | 19 |
| business | 40 | law | 48 |
| chemistry | 56 | math | 68 |
| computer science | 20 | other | 46 |
| economics | 42 | philosophy | 25 |
| engineering | 48 | physics | 65 |
| health | 35 | psychology | 40 |

HYPOTHESIS C14 (line 181, "588 test items per language, 42 in each of 14 categories") and the loader's check assume a balanced design. The dataset's README does not claim one. The total of 588 and the EN/DE parallel ids are as expected.

### 2. MMLU-ProX full: one row's `answer` disagrees with `answer_index` (`mmlu_prox_full_pilot_en`, `mmlu_prox_full_pilot_de`, `mmlu_prox_c1_en`, `mmlu_prox_peercheck_en`)

```text
ValueError: MMLU-ProX 3787: answer 'C' disagrees with answer_index 1
```

At the pinned revision `8e6106a6`, `question_id` 3787 is the **only** such row among 11,759 test rows, in both EN and DE. The validation splits and MMLU-ProX-Lite have no such row.

## Partial outputs of the failed step-7 run, moved here

Before it stopped, `build_manifests.py` had written the four GPQA sets. They were moved, not deleted, so the tree is clean for the re-run.

- **Public (hash-only, in this folder):**
  - `gpqa_diamond_de.json`: 40633 bytes, sha256 `b495bd829569ef88de9bd456041411c98902fb33f3229a988d0068b34d13964f`
  - `gpqa_diamond_en.json`: 47209 bytes, sha256 `79e524c448903aeb9f6cdf1a9a4765e6f56b20053b2613fce64d73da1abd6382`
  - `gpqa_pilot_de.json`: 2263 bytes, sha256 `16f5cbb8127860a009a13f3396a5fb8891b4c19d63de02a2b0a71c39ebe424a7`
  - `gpqa_pilot_en.json`: 2166 bytes, sha256 `d8c22373f37749773f52fba36a3a32eb8c7c0642fbf0d12e15575b85d406fc48`
- **Private,** moved to `$EXP036_PRIVATE/aborted/20261004T095549Z-manifests-partial/`:
  - `gpqa_diamond_de.jsonl`: 298360 bytes, sha256 `35fe5bb3f8d297e5e85d0b5b99c145b70d18d60296a982a581fccf3184184f73`
  - `gpqa_diamond_en.jsonl`: 284571 bytes, sha256 `0271f914cbfdcef322391d3b877ed6b3a34b4373db53ed6bd28860a60675befe`
  - `gpqa_pilot_de.jsonl`: 11152 bytes, sha256 `a5bc22861ba60fac92ce4a79448690199043c0670d0b5684adb0f51a2ef7d3cc`
  - `gpqa_pilot_en.jsonl`: 12391 bytes, sha256 `46ce42a8e1290223919539edd39ab80d1a57c38c5b51aeaae68b578a5c1b5c9e`

`$EXP036_PRIVATE/manifests/` is empty again. No shingle file was written, and `gate/build_gate_text.py` was not run.

Both MMLU-ProX points change pre-registered item-set assumptions, so they need the main session's fix and Andrei's decision. The mbp waits at step 7.
