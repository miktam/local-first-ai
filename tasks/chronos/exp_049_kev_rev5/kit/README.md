# exp_049 kit

- `leads49.py`: the pipeline and scorer, as run (casasol `bfcd0d6`). It covers writer validation, the checker batches, the agreement rule, gemma4:26b's third reading, the split, the runs, `compare`, and `tune` with `recombine()`.
- `kev_knife_rev42.py`: rev 4.2 (sha256 `5b79950f…`). Kev-4B reads the request, Clef-Flash 9B the facts (exp_048).
- `kev_knife_rev5.py`: rev 5 as run (`8d268660…`). Rev 4.2 plus F1 (on behalf), F2 (price only), F3 (dated timeline question) and F4 (financing "use"), with cuts tuned on dev.
- `kev_knife_rev51.py`: the adopted rev 5.1 (`897c0ac2…`), rev 5 with F2 off. It passed the regression gate on exp_044's sealed test and is live in CasaSol Guide since 2026-10-21. Its test numbers in RESULT.md are exact: they are recomputed from rev 5's recorded answers.
- `leads44.py` (its helpers) is in [exp_048's kit](../../exp_048_flash_hybrid/kit/leads44_2026-10-21.py).

**To re-score** from the published predictions (no model needed), lay the files out in any folder `X`:

```
X/scripts/leads49.py                     ← kit/leads49.py
X/scripts/leads44.py                     ← ../exp_048_flash_hybrid/kit/leads44_2026-10-21.py
X/data/leads49/test.jsonl                ← data/test.jsonl
X/runs/exp049_test_rev42/pred.jsonl      ← results/test/pred_rev42.jsonl
X/runs/exp049_test_rev5/pred.jsonl       ← results/test/pred_rev5.jsonl
```

Then run `python3 X/scripts/leads49.py compare test`. The output equals `results/test/compare.json` (checked 2026-10-21).

**Data:** `data/raw/` holds the eight Fable writers' files, with labels and a "why" per lead. `data/chk/` holds the checker batches (`c_*`, id and text only) and Opus's blind labels (`out_*`). Then come `agreed`, `dev` and `test` with their hashes, the 30 leads Andrei read and his notes, and gemma4:26b's labels for every kept lead (`third_gemma4.jsonl`).

**Contact details:**
- every phone number is in Ofcom's drama range (never allocated), in digits or spelled out in words, and every e-mail is `@example.com`;
- before publishing, we scanned all files for realistic numbers in digits and converted every spelled-out run of 7+ digits back to digits. Only Ofcom numbers and prices were found;
- the names in the messages are invented.

**Public copy (2026-10-21):** the one phone number in `DEMO`, the built-in example list, was a valid-format Swedish mobile (+46 70 …). It is now in Ofcom's fiction range, as in exp_043's kit. Nothing else changed, and `DEMO` is not on the triage path. The as-run hashes above are of the files the runs used. This copy's hashes: `kev_knife_rev42.py` `029f44095cbde819…`; `kev_knife_rev5.py` `d9b46aaf82366f06…`; `kev_knife_rev51.py` `b9d1fac6e4067cd5…`.
