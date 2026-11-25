# exp_063 — is there a stable hot set of experts to cache?

Engines that stream a mixture-of-experts model from SSD keep the "most used experts" in memory. This checks on two local models whether the experts used most on one text are the ones used most on the next. Counts only; no engine, no timings. **Run 2026-11-25: a hot set exists and is class-specific (H1, H2 pass on both models); one mixed set fails on code (H3).**

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration, written before the texts were run |
| [RESULT.md](RESULT.md) | verdicts, the two coverage tables, limits, SO WHAT |
| `kit/` | text builder, runner, count extractor, scorer, text hashes |
| `runs/` | per-text counts, logs, `score.json` |
