# exp_048 kit

- `flash48.py`: the runner and scorer, as run (casasol `3f695d1`, sha256 `77d18ef2…`). It covers the Flash arm, the hybrid's `recombine()`, the self-check that recombining Kev's own answers reproduces Kev's decisions, and the hypotheses.
- `kev_knife_rev41.py`: the triage, rev 4.1 (sha256 `2acc7a1d…`). Every arm here uses it: Kev-4B, Clef 27B and Clef-Flash 9B each answer its questions, and its rules turn the answers into decisions. Its code-fact changes from rev 4 are described in exp_045's RESULT.
- `leads44_2026-10-21.py`: `leads44.py` as of 2026-10-21. Its `VERSIONS` table names rev 4.1, and `flash48.py` loads that version through it. Rename it to `scripts/leads44.py`.

The comparison arms, Kev-4B and Clef 27B, are exp_047's recorded test passes ([exp_047](../../exp_047_clef_triage/)). `clef47.py` is in exp_047's kit, and `build_leads46.py` is in [exp_046's kit](../../exp_046_translate_first/kit/).

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X`:

```
X/scripts/flash48.py                     ← kit/flash48.py
X/scripts/leads44.py                     ← kit/leads44_2026-10-21.py
X/scripts/kev_knife_rev41.py             ← kit/kev_knife_rev41.py
X/scripts/clef47.py                      ← ../exp_047_clef_triage/kit/clef47.py
X/scripts/build_leads46.py               ← ../exp_046_translate_first/kit/build_leads46.py
X/runs/exp048_test_flash/pred.jsonl      ← results/test/pred_flash.jsonl
X/runs/exp047_test_kev/pred.jsonl        ← ../exp_047_clef_triage/results/test/pred_kev.jsonl
X/runs/exp047_test_clef/pred.jsonl       ← ../exp_047_clef_triage/results/test/pred_clef.jsonl
```

Then run `python3 X/scripts/build_leads46.py <this repo>/tasks/chronos` and `python3 X/scripts/flash48.py score test`. The output equals `results/test/score.json` (checked 2026-10-21).

**The run:** `results/test/` holds the tmux script and its log, and an excerpt of the Flash server's log. There were no server errors.

**Public copy (2026-10-21):** the one phone number in `DEMO`, the built-in example list, was a valid-format Swedish mobile (+46 70 …). It is now in Ofcom's fiction range, as in exp_043's kit. Nothing else changed, and `DEMO` is not on the triage path. The as-run hashes above are of the files the runs used. This copy's hashes: `kev_knife_rev41.py` `f2fd92fa03e1731c…`.
