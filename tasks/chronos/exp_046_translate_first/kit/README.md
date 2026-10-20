# exp_046 kit

- `translate46.py`: the runner and scorer, as run (casasol `5f0a1c4`). It covers the translation call and its checks, the triage re-assembled so Kev reads the translation, the guards, and the score.
- `build_leads46.py`: rebuilds the registered sets from the public data of exp_043 and exp_044. It concatenates each set's rows and tags every row with its source. Run it from a folder whose parent holds `data/`, and it prints `OK` when the sha256 matches.

The triage code, rev 4, and `leads44.py` are in [exp_044's kit](../../exp_044_kev_lead_triage_rev4/kit/).

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X`:

```
X/scripts/translate46.py                        ← kit/translate46.py
X/scripts/build_leads46.py                      ← kit/build_leads46.py
X/scripts/leads44.py                            ← ../exp_044_kev_lead_triage_rev4/kit/leads44.py
X/runs/exp046_test_native/pred.jsonl            ← results/test/pred_native.jsonl
X/runs/exp046_test_translated/pred.jsonl        ← results/test/pred_translated.jsonl
X/runs/exp046_test_translated/translations.jsonl ← results/test/translations.jsonl
```

Then run `python3 X/scripts/build_leads46.py <this repo>/tasks/chronos` and `python3 X/scripts/translate46.py score test`. The output equals `results/test/score.json` (checked 2026-10-20).

**To re-run** you need gemma4:26b on Ollama (digest `777b927e3b6b…`; options in `translate46.py`) and the Kev-4B decision server that exp_044's kit calls.

**Contact details:** every phone number in these enquiries is in Ofcom's drama range, and contacts are masked before the translator sees the text. We scanned every source, translation and rejected translation in `results/` for anything else. Nothing turned up.
