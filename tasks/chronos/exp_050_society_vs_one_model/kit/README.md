# exp_050 kit

- **`desk50.py`:** both front desks, as run on test (casasol `0364491`).
  - **G, the society:** triage rev 5.1 → the agency's rules → Flash's listing questions → the fit-rule join → the writer → code-rendered cards and closing lines → the audit.
  - **M:** one call with the rules and the whole inventory.
  - It also holds `InfraError`, which separates serving failures from the model's own output, and the prompt log.
- **`leads50.py`:** the inventory generator, the data pipeline (writer validation, checker batches, agreement, split, extra listings), the runner and `compare`, which has the registered bars. It includes the dated scorer correction (casasol `7612456`), off by default; with `EXP50_SCORER=corrected` it applies.
- **Helpers published with earlier experiments:**
  - triage rev 5.1: [`kev_knife_rev51.py`](../../exp_049_kev_rev5/kit/kev_knife_rev51.py) in exp_049's kit. It is the public copy: its demo list's one phone number was replaced, which changes the hash, not the logic.
  - [`leads44_2026-10-21.py`](../../exp_048_flash_hybrid/kit/leads44_2026-10-21.py) in exp_048's kit.
  - [`budget_words.py`](../../exp_044_kev_lead_triage_rev4/kit/budget_words.py) in exp_044's kit, unchanged since.

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X`:

```
X/scripts/leads50.py              ← kit/leads50.py
X/scripts/desk50.py               ← kit/desk50.py
X/scripts/kev_knife.py            ← ../exp_049_kev_rev5/kit/kev_knife_rev51.py
X/scripts/leads44.py              ← ../exp_048_flash_hybrid/kit/leads44_2026-10-21.py
X/scripts/budget_words.py         ← ../exp_044_kev_lead_triage_rev4/kit/budget_words.py
X/data/leads50/test.jsonl         ← data/test.jsonl
X/data/leads50/inventory.jsonl    ← data/inventory.jsonl
X/data/leads50/extra.jsonl        ← data/extra.jsonl
X/runs/exp050_test_<ARM>_r<K>/pred.jsonl     ← test_run/pred_<ARM>_r<K>.jsonl     (G_r1–3, M_r1–3, Gs_r1, Ms_r1)
X/runs/exp050_test_<ARM>_r<K>/prompts.jsonl  ← test_run/prompts_<ARM>_r<K>.jsonl
```

Then run:
- `python3 X/scripts/leads50.py compare test`. The output equals `test_run/compare_registered.json`.
- `EXP50_SCORER=corrected python3 X/scripts/leads50.py compare test`. The output equals `test_run/compare_corrected.json`. Both checked 2026-10-24 in a clean folder.

**Data:**
- `data/raw/` holds the eight Fable writers' files, with labels, contacts and a "why" for each lead.
- `data/chk/` holds the checker batches (`c_*`, id and text only) and Opus's blind labels (`out_*`).
- Then come `agreed`, `dev` and `test` with their hashes, the invented inventory, run 3's two extra listings, and the 30 leads Andrei read before signing.

**Runs:**
- `dev_runs/` holds every dev variant: M's prompt revisions, temperature 0, thinking on 30 leads, and G's unsure-cut re-run.
- `test_run/` holds every test prediction, every prompt sent to a model (H5), the run's own script and log, both scorings, and the blind judge's batches, unblinding key and verdicts.
- `aborted/` holds the first test pass, stopped after 12 leads, and its incident record.

**Contact details:**
- every phone number is in Ofcom's drama range (never allocated), in digits or spelled out in words, and every e-mail is `@example.com`;
- the names in the messages are invented, and so are the inventory's listings, refs and prices.
