# exp_043 kit

| File | What it is |
|---|---|
| `kev_knife_rev3_measured.py` | The code the test pass ran: casasol `scripts/kev_knife.py` at `d581267`, sha256 `98f30090…07fb83`. `leads43.py run test` refused any other version. |
| `kev_knife_current.py` | The same file after the result: spam's action is `review` (never discard), and budgets read units (млн, тыс, Mio). It was not measured on the test set. |
| `leads43.py` | The pipeline: validate, checker batches, writer × checker agreement, seeded split, run (test once), score. |
| `test_kev_knife.py` | Unit tests for the policy rules and contact masking. |

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X` and run `python3 X/scripts/leads43.py score test`:

```
X/scripts/leads43.py               ← kit/leads43.py
X/data/leads43/test.jsonl          ← data/test.jsonl
X/runs/exp043_test/pred.jsonl      ← results/test/pred.jsonl
```

The output equals `results/test/score.json` (checked 2026-10-18).

**To re-run Kev** you need the CasaSol repo (`kev_knife` imports the bot's `detect_language`) and Kev-4B served on llama.cpp `/v1/systemone` at `127.0.0.1:18442`.
