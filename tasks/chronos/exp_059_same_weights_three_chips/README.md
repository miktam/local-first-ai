# exp_059 — same weights, three chips

One model file (Kev-4B, Q8_0), one engine commit (llama.cpp b11711), the sealed exp_044 test set (331 enquiries), run on
three Apple chips: the Mac mini's M4 Pro, the MacBook Pro's M5 Max and the iPhone 18 Pro's A20 Pro. The question is
whether the triage **decision** changes with the device. Registered 2026-10-28, before any run.

Kev is a decision model: it generates no token. The server scores each option by a dot product on a 512-wide head and a
softmax at a calibrated temperature. The phone cannot run the server, so the app in `kit/ios/` reproduces that readout
with the engine's public API, and a control (C2) checks the app against the server on the same Mac before any phone
number counts.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: pinned inputs, arms, method, controls C1–C3, H1–H4, descriptive values, fallback, verdict map, limits |
| `kit/render59.py` | renders the 12 question prompts per enquiry exactly as the server's `systemone` template does → `data/prompts59.jsonl` |
| `kit/run59_mac.py` | a Mac arm: the masked state and the schema to `/v1/systemone`, one request per enquiry, raw answers and wall time |
| `kit/finish59.py` | raw answers → band and action through the frozen `kev_knife_rev4.py`, unchanged, on every arm |
| `kit/score59.py` | H1–H4, C1–C3, D1–D5 |
| `kit/kev_knife_rev4.py` | the frozen rev 4 code, copied from exp_044's public kit (sha256 `2cdfbdf7…`) |
| `kit/ios/` | the phone arm: `KevBatch.swift` (the readout and the batch loop), `Exp059View.swift` (the screen), `README.md` (build on the MacBook, sign, copy files, run, copy out) |
| `data/prompts59.jsonl` | the rendered prompts, hashed in `data/prompts59.sha256` |
| `raw/`, `pred/`, `results/` | per arm: raw answers, finished rows, the score (written by the runs) |

**Status: registered, not yet run.** Numbering: exp_057 names exp_058 for the harness-layer egress test, so this is 059.

**To reproduce the comparison** from published raw files (no model needed): lay out `raw/<arm>.jsonl` for the arms you have,
run `python3 kit/finish59.py <arm>` for each, then `python3 kit/score59.py`. `finish59.py` needs exp_044's
`data/test.jsonl` and `results/test/pred_rev4.jsonl` (for the enquiry text and the recorded facts); the default paths point
at the sibling folder in this repository.
