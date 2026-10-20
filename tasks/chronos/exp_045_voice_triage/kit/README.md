# exp_045 kit

- `voice45.py`: the runner and scorer, as run (casasol `ed02fbb`).
- `notetaker_voice_path.py`: the voice path it calls, verbatim from casasol `scripts/notetaker.py` at `ed02fbb`. That is `to_wav` and `transcribe` (mlx-whisper large-v3-turbo) plus `join_spoken_numbers`, the phone-number joiner built on dev.
- `test_voice45.py`: tests for the emoji filter, the joiner and the noise.

The triage code, rev 4, and `leads44.py` are in [exp_044's kit](../../exp_044_kev_lead_triage_rev4/kit/). The messages and labels are exp_044's public test set, [`data/test.jsonl`](../../exp_044_kev_lead_triage_rev4/data/test.jsonl) (sha256 `514b6386…`).

**To re-score** from the published predictions (no model, no audio), lay the files out in any folder `X` and run `python3 X/scripts/voice45.py score test`:

```
X/scripts/voice45.py                     ← kit/voice45.py
X/scripts/leads44.py                     ← ../exp_044_kev_lead_triage_rev4/kit/leads44.py
X/data/leads45/test.jsonl                ← ../exp_044_kev_lead_triage_rev4/data/test.jsonl
X/runs/exp045_test_text/pred.jsonl       ← results/test/pred_text.jsonl
X/runs/exp045_test_whatsapp/pred.jsonl   ← results/test/pred_whatsapp.jsonl
X/runs/exp045_test_noisy/pred.jsonl      ← results/test/pred_noisy.jsonl
X/runs/exp044_test_rev4/pred.jsonl       ← ../exp_044_kev_lead_triage_rev4/results/test/pred_rev4.jsonl
```

The output equals `results/test/score.json` (checked 2026-10-20). Its H4 flag includes a condition the signed hypothesis dropped; see RESULT.md.

**To re-run** you need:
- macOS with the voices named in `VOICES`;
- ffmpeg with libopus;
- mlx-whisper with `mlx-community/whisper-large-v3-turbo`;
- the Kev-4B decision server that exp_044's kit calls.

Rename `notetaker_voice_path.py` to `scripts/notetaker.py`. The run also imports the bot's `_TRIAGE_LANG`, the seven language codes en, es, de, fr, sv, ru and pl. The audio (26 MB, Opus) is not published: the voices and the seeded noise regenerate it, though `say` may differ across macOS versions.

**Transcripts and phone numbers.** Every phone number in the messages is in Ofcom's drama range, which is never allocated. Whisper sometimes garbles one: an extra 0 or 4, a dropped digit, or "900" repeated in a loop (X2-38, X2-72). We checked all 23 garbled forms in the transcripts. None is a valid number, since a UK mobile has exactly 10 digits after +44.
