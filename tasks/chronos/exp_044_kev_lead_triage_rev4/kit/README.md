# exp_044 kit

| File | What it is |
|---|---|
| `kev_knife_rev31.py` | The baseline as frozen: sha256 `bc2ae147…`, the decisions exp_043 measured. |
| `kev_knife_rev4.py` | rev 4 as frozen after dev tuning: casasol `scripts/kev_knife.py` at `abcfacf`, sha256 `b1940235…`. Adopted. |
| `budget_words.py` | Amounts written as words in seven languages (rev 4). |
| `leads44.py` | The pipeline: validate, checker batches, agreement, split, one-pass runs per version, offline tuning on dev, paired comparison H1–H5. |
| `test_budget_words.py`, `test_kev_knife.py` | Unit tests for the word reader, the rules and masking. |

**To reproduce the comparison** from the published predictions (no model needed), lay the files out in any folder `X` and run `python3 X/scripts/leads44.py compare test`:

```
X/scripts/leads44.py                       ← kit/leads44.py
X/data/leads44/test.jsonl                  ← data/test.jsonl
X/runs/exp044_test_rev31/pred.jsonl        ← results/test/pred_rev31.jsonl
X/runs/exp044_test_rev4/pred.jsonl         ← results/test/pred_rev4.jsonl
```

The output equals `results/test/compare.json` (checked 2026-10-19, on the public copy of the data).

**To re-run Kev** you need the CasaSol repo (the knife imports the bot's `detect_language`) and Kev-4B served on llama.cpp `/v1/systemone` at `127.0.0.1:18442`.
