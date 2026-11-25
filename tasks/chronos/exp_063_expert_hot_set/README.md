# exp_063 — is there a stable hot set of experts to cache?

Engines that stream a mixture-of-experts model from SSD keep the "most used experts" in memory. This checks on two local models whether the experts used most on one text are the ones used most on the next. Counts only; no engine, no timings.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration, written before the texts were run |
| `kit/` | text builder, runner, count extractor, scorer, text hashes |
| `runs/` | per-text counts, logs, `score.json` |
