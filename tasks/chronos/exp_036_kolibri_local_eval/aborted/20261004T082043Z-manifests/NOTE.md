# RUNBOOK step 7 stopped: no over-long item in GPQA Diamond (2026-10-04)

The sign-off is committed (`8781087`, all pre-registered defaults). Step 7's first command then failed at once, on the mbp at `8781087`:

```text
"$PY" tasks/build_manifests.py
{"ok": false, "error": "manifest gpqa_diamond_en: ValueError: expected 1 over-long item(s) excluded, found 0"}
exit 1
```

It wrote nothing. The tree is clean, `$EXP036_PRIVATE/manifests/` is empty, there is no `tools/withheld_shingles.sha256`, and `gate/build_gate_text.py` was not run.

## What the data shows

The checks below used hashes, Record IDs and lengths only. No item text was read out.

- The vendored eval-framework constant `_OVERLONG_QUESTION_SHA256` (prefix `04e898b3dfc3`) matches **exactly one row in `gpqa_extended.csv`**. It matches **no row** in `gpqa_diamond.csv` (198 rows) or `gpqa_main.csv` (448 rows).
- Normalising the text does not change this. Raw, `strip()`, CRLF→LF and LF→CRLF all give the same counts, and no question contains a CR.
- That row's Record ID is in **neither** Diamond nor Main. It is not a revision of a Diamond question; it is an Extended-only record.
- By characters it is not long: 182 of the 198 Diamond questions are at least as long.
- The GPQA data is at the pinned revision `83022cef`, and step 4b hashed every file against its etag.

## Consequence

`tasks/gpqa.py::primary_197` (with `expect_excluded=1`) and `selection_rules.json` (`gpqa.diamond_en.primary_rule`: "drop the row … (exactly one)") expect the over-long question inside Diamond. With this data, the eval-framework filter removes nothing from Diamond, so its GPQA-Diamond primary set would be **all 198**, not 197.

The pre-registered figures assume 197: HYPOTHESIS.md and RUNBOOK 3c ("GPQA-D EN 198, one over-long item flagged out of the primary 197"). The mini built and tested this without GPQA data, so the rule has never run on the real files. The optional RUNBOOK step 3c would have shown it before the sign-off; it was not run on the mbp.

This changes a pre-registered n and the H2 primary row set, so it needs a decision and an amendment before step 7 can run. The mbp waits at step 7. Nothing else was run.
