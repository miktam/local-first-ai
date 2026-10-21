# exp_047 kit

`clef47.py` is the runner and scorer, as run (casasol `d2c4bfe`, sha256 `2f9ef115…`). Both arms use the same triage file, rev 4.1. The only change is which decision server it calls: Kev-4B, or Clef 27B (`Cloudflare_clef-Q8_0.gguf`). Both are served by llama.cpp b11711's `/v1/systemone`.

Other pieces live in earlier kits:
- rev 4.1 (`kev_knife.py`, sha256 `2acc7a1d…`) is casasol's live triage. Its code-fact changes from rev 4 are described in exp_045's RESULT, and rev 4 itself is in [exp_044's kit](../../exp_044_kev_lead_triage_rev4/kit/);
- `leads44.py` is in exp_044's kit;
- `build_leads46.py` (the sets) and `translate46.py` (holding the sets' full hashes) are in [exp_046's kit](../../exp_046_translate_first/kit/).

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X`:

```
X/scripts/clef47.py                      ← kit/clef47.py
X/scripts/leads44.py                     ← ../exp_044_kev_lead_triage_rev4/kit/leads44.py
X/scripts/build_leads46.py               ← ../exp_046_translate_first/kit/build_leads46.py
X/runs/exp047_test_kev/pred.jsonl        ← results/test/pred_kev.jsonl
X/runs/exp047_test_clef/pred.jsonl       ← results/test/pred_clef.jsonl
```

Then run `python3 X/scripts/build_leads46.py <this repo>/tasks/chronos` and `python3 X/scripts/clef47.py score test`. The output equals `results/test/score.json` (checked 2026-10-21). The scorer reads exp_043's spam label "discard" as "review", the same decision under its current name.

**The run** is in `results/test/`: both tmux scripts and their logs, the first Clef server log (with the out-of-memory error on enquiry 36), and an excerpt of the resumed server's log.
