# exp_045: lead triage from voice notes

*Written 2026-10-19, night, before any call on the test set. **Signed by Andrei 2026-10-20 06:08 CEST** ("sign"); text as committed in `1cfc602`, unchanged. Andrei had said "go" for option A ("triage from voice notes"). He runs the test pass himself (about 50 min). Code: casasol `ed02fbb` (`scripts/voice45.py` sha256 `77c60411…`, `scripts/notetaker.py` `3a479493…`); triage = rev 4 as frozen in exp_044 (`kev_knife.py` `b1940235…`).*

## Question

Agencies get many enquiries as WhatsApp voice notes, and the bot already triages voice notes. **Does an enquiry keep its triage when it arrives as a voice note instead of text?** Same messages, same frozen code, two paths:

- **text:** the written enquiry → rev 4.
- **voice:** the same enquiry read aloud → a WhatsApp-quality voice note → transcribed on the Mac Mini → rev 4, exactly as the bot does it.

No cloud model is involved anywhere: no writers, no checker, no judge.

## Design

- **Messages:** exp_044's sealed test set, 331 synthetic enquiries in seven languages, with their labels. We use the **public copy** (`514b6386…`, local-first-ai `9d87aa7`): its contact numbers are in Ofcom's reserved range, so the audio and transcripts can be published. Kev's masked input and the code facts are identical to the sealed originals, except for 3 test enquiries (X1-48, X1-60, X6-04) that spell their phone number out in words.
  - exp_044 used this set for its verdict. It is reused here for a different question, and nothing is tuned on it.
- **Voices:** one macOS voice per writer batch: Samantha (English, X1, the messy batch), Daniel (English, X2), Zosia (Polish), Mónica (Spanish), Anna (German), Alva (Swedish), Thomas (French), Milena (Russian). Emoji are dropped before reading, since a voice note has none. Nothing else is rewritten.
- **Voice note:** mono, 16 kHz, Opus at 16 kbit/s (`-application voip`), the format and bitrate of a WhatsApp voice note.
- **Transcription:** the bot's own code, `notetaker.to_wav` + `notetaker.transcribe` (mlx-whisper large-v3-turbo, cached, offline). Whisper picks the language.
- **Then, as in the bot:**
  - phone numbers read out in groups are joined (`notetaker.join_spoken_numbers`, built on dev, below);
  - rev 4 triages the transcript;
  - Whisper's language replaces the text detector's when it is one of the seven. Language is display-only: no rule reads it.
- **Arms:**
  - **text:** rev 4 on the public-copy text. It is re-run so that both arms read identical words. Kev is deterministic, so it should repeat exp_044's recorded test decisions, except perhaps on those 3 enquiries. Reported.
  - **whatsapp:** the voice path above. This is the registered arm.
  - **noisy:** the same, with seeded pink noise mixed in at 10 dB SNR before encoding, like a street or café. Descriptive only.
- **Guards** (`voice45.py run test`) refuse:
  - a set that isn't the registered public copy;
  - a `kev_knife.py` that isn't the frozen rev 4;
  - an uncommitted `voice45.py` or `notetaker.py`;
  - a second pass of any arm.

## As built: the dev shakedown (155 dev enquiries, 2026-10-19)

- **v0, without the phone joiner:** the voices read numbers in pairs, and Whisper writes them as "plus 44, 77, 00, 90, 01, 28". The commas hid them from the phone check, so phones were seen in 15 enquiries by voice against 24 by text, and the next step was −0.013. **Fix:** `join_spoken_numbers` joins a run that starts with a spoken plus or a 0 and holds 9–15 digits. Prices, dates and "12, 00, 15, 00 par mois" are left alone (tests in `tests/test_voice45.py`). The bot's voice triage uses it too (`ed02fbb`; live after the next bot deploy).
- **Final dev, whatsapp arm** (text → voice):

  | | text | voice |
  |---|---|---|
  | request | 0.923 | 0.903 (−0.020) |
  | next step | 0.916 | 0.923 (+0.007) |
  | genuine enquiries flagged as spam (131) | 1 | 2 |
  | spam caught (24) | 23 | 22 |
  | budget, where stated (92) | 0.783 | 0.728 |
  | budget written in words (16) | 0.938 | 0.812 |
  | language shown right | 0.994 | 1.00 |
  | phones seen | 24 | 24 |

- **Budget losses on dev have two causes** (8 enquiries, split by code):
  - **Whisper writes spoken amounts as bare digits** (3): "eight hundred and fifty thousand" → "850,000", with no "euros" and no "thousand", and rev 4's reader needs one of those. This is a real voice-path problem. Fixing it is a reader change, so a rev 5 seed, not a dev tweak: rev 4 is frozen.
  - **The computer voice misreads abbreviations** (5): "2,5 M€" → "mœurs", "250k" → "K2". A person would say "two and a half million", so this side of the result is pessimistic.
- **The text arm on dev differs from exp_044's recorded dev run on 2 of 155.** It isn't nondeterminism: Kev's probabilities are identical. exp_044's dev run used rev 4 before the final cut was set (`OWN_CUT` 0.70 then, 0.65 frozen), and both enquiries sit at 0.67–0.69. exp_044's recorded test run used the frozen code.
- **Found on the way:** 14 phone numbers in exp_043/044 were spelled out in words. No digit scan saw them, and neither does the masker. Whisper turned one back into digits. They are now in Ofcom's range too (local-first-ai `9d87aa7`, not pushed yet).
- **Speed:** a median voice note is 20 s of speech. Transcription takes about 1.1 s and Kev about 1.7 s.

Dev files: [results/dev/](results/dev/).

## Hypotheses (test set, 331, whatsapp arm against text arm, paired)

| | Measure | Bar |
|---|---|---|
| **H1 request** | request accuracy, voice − text | ≥ −0.03 |
| **H2 next step** | next-step accuracy, voice − text | ≥ −0.03 |
| **H3 spam** | genuine enquiries flagged as spam (282); spam caught (49) | voice ≤ text + 1; voice caught ≥ 0.85 |
| **H4 budgets in words** | budget within 1 % on the 29 enquiries whose budget is written in words | voice ≥ 0.90 |
| **H5 language** | the language shown is the language written | voice ≥ 0.95 |

Reported, with no bar:
- all stated budgets, with the losses split by cause (abbreviation in the text or not);
- request by language and stratum;
- every enquiry whose request flips between arms;
- transcript similarity by language;
- phones seen;
- speed;
- McNemar p for H1 and H2;
- the noisy arm on every measure;
- the text arm against exp_044's recorded test decisions.

## Verdict map

1. **H1, H2 and H3 pass → "Voice notes keep the triage"** (computer voices, read-aloud). The bot's voice triage can be shown to agencies as equal to typed text.
   - H4 failing adds: "but spoken budgets need a reader fix" (rev 5 seed).
   - H5 failing adds: "but the language label needs Whisper's help".
2. **H1 or H2 fails → "The voice path costs accuracy."** Voice triage stays a demo. The flips show where it fails: transcription, language or numbers.
3. **H3 fails alone → "Voice adds false spam."** These go to review, never the bin, but it is noted before showing voice to agencies.

## What this cannot show

- **Computer voices reading written messages are not people speaking.** No accents, hesitations, background voices or real phones. The result is a best case for the voice path, except for abbreviations, where it is a worst case.
- **One voice per language, one pass per arm.**
- **The messages are exp_044's synthetic ones.** Labels are two Fable readers' agreement, and the arguable fifth was dropped. Everything tomorrow's post says about synthetic data applies here.
- **Real voice notes from agencies are the test that matters,** as for text.

## To run (after signing)

```bash
cd ~/REPOS/casasol
for a in text whatsapp noisy; do .venv/bin/python3 scripts/voice45.py run test --arm $a || break; done
.venv/bin/python3 scripts/voice45.py score test > runs/exp045_test_score.json
```

About 10 + 20 + 20 minutes. It prints progress every 25 enquiries and resumes if interrupted (a finished arm is refused, an unfinished one continues). Then Claude writes RESULT.md.
