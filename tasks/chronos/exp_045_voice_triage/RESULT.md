# exp_045 result: lead triage from voice notes

*Scored 2026-10-20 on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-20 06:08 CEST, `6c916ef`; text as in `1cfc602`). Andrei ran the test pass himself, 06:10–06:58, one pass per arm. Set: exp_044's test set, public copy (`514b6386…`, 331 synthetic enquiries, seven languages). Code: casasol `ed02fbb` (`voice45.py` `77c60411…`); triage = rev 4 as frozen (`b1940235…`); Whisper large-v3-turbo; macOS 26.5.1; ffmpeg 8.1.1. Files: [results/test/](results/test/) (predictions with transcripts, `done` records with every hash, `score.json`), [results/dev/](results/dev/). No cloud model was used anywhere.*

## Verdict: **Voice notes keep the triage** (H1–H5 pass; computer voices reading written enquiries)

| Hypothesis | text | voice (WhatsApp quality) | Bar | |
|---|---|---|---|---|
| **H1** request (331) | 0.912 | **0.900** (−0.012; 9 vs 5 discordant, p 0.42) | ≥ −0.03 | pass |
| **H2** next step (331) | 0.931 | **0.921** (−0.010; 9 vs 6, p 0.61) | ≥ −0.03 | pass |
| **H3** genuine enquiries flagged as spam (282) · spam caught (49) | 2 · 46 | **3 · 48** | ≤ text + 1 · ≥ 0.85 | pass |
| **H4** budgets written in words, within 1 % (29) | 1.00 | **0.966** (28) | ≥ 0.90 | pass |
| **H5** language shown = language written | 0.976 | **1.00** | ≥ 0.95 | pass |

**A departure in the scorer, recorded before this write-up.** `voice45.py score` prints H4 as `"pass": false`. Its H4 check still has a second condition from the dev draft: all stated budgets within 0.05 of text. The signed text moved that measure to "reported, with no bar", and the verdict follows the signed text. That measure is the weakest number in this experiment, so it comes first below.

**Kev is deterministic:** the text arm repeats exp_044's recorded test decisions on all 331 enquiries.

## What changed between text and voice (descriptive)

- **Budgets are where voice loses.** Of the 211 enquiries that state a budget, voice read it right 0.701 of the time, against 0.763 for text. 15 were lost and 2 gained. Of the 15:
  - **10 come from the computer voice, not the voice path:**
    - 7 abbreviations: "2,5 M€" and "250k" are read as letters;
    - 3 European amounts read by an English voice: "250.000 €" spoken as "250 euros".
  - **5 come from Whisper and the reader:**
    - 3 amounts written back as bare digits with no currency ("siebenhunderttausend" → "700.000", "до двух тысяч" → "до 2000", "до 2 тыс." → "до 2000"), where rev 4's reader needs a currency or a multiplier word;
    - 1 "1 400 €/mån" heard as "1400 euromån";
    - 1 **"€350,000"**, with the currency symbol first, which rev 4's reader misses. It misses it in typed text too ("up to €2,000 a month" → none), while "€350k" and "€1.5M" work. Only 2 of the 486 synthetic enquiries use that format, so the writers hid this gap, and Whisper's formatting exposed it.
- **Amounts taken for phone numbers.** Rev 4's phone detector reads any 7+ digits joined by spaces or dots as a phone. Two consequences: the masker hides the amount from Kev, and "has a phone" can change the next step.
  - By voice: "1 600 000 euro" (X3-54) and "1 200 000 евро" (X8-67).
  - In typed text: "3.500.000 €" (X4-39, X4-58).
- **Phones in words.** X6-04 spells its number out in Swedish. The text masker misses it, so Kev read it. By voice, Whisper writes digits and the phone is seen.
- **Request flips (16 of 331):**
  - **Voice moved 7 enquiries to holiday rent,** 5 of which typed text had right (4 buyers, 1 investor). It is exp_043's buy-vs-holiday confusion, and transcription tips borderline cases.
  - **Voice caught 2 pitches text missed,** X5-18 and X4-15. X4-15 is exp_044's known miss, a pitch read as "sell".
  - **1 new false spam,** X7-08: a relocation firm renting four flats for a client. Its business-pitch score went from 0.55 typed to 0.60 transcribed, landing on the cut. It is the same class as exp_043's false spams, and it goes to review.
- **Language:** Whisper heard the right language on 331 of 331, while the text detector got 323 right. Its 8 misses were short or mixed messages ("Hi, nur mal aus Neugier…", Polish without diacritics, French with English brand names). Voice shows the right language more often than text.
- **Request by language, text → voice:** French 1.00 → 0.932, Swedish 0.951 → 0.902, Polish 0.857 → 0.833, German 0.897 → 0.923, Russian 0.902 → 0.927; English (0.877) and Spanish (0.930) unchanged. Groups are 39–81 enquiries.
- **Same decision in both arms:** request 0.952, next step 0.952.
- **Noise barely matters.** With pink noise at 10 dB SNR:
  - request 0.909 and next step 0.924;
  - spam caught 49 of 49, with 3 false spam;
  - stated budgets 0.682 and budgets in words 0.897.
- **Speed:** the median voice note is 20.3 s of speech. Whisper takes 1.1 s and Kev 1.7 s, so about 3 s from voice note to triage on the Mac Mini.
- **Dev, for comparison:** request −0.020, next step +0.007, budgets in words 0.812 (16). Test was kinder than dev on budgets in words.

## Known limits (as registered)

- **Computer voices reading written messages, not people speaking:** no accents, hesitations, background voices or real phones. A best case for the voice path, except for abbreviations and number formats, where it is a worst case (10 of the 15 budget losses).
- **One voice per language, one pass per arm.**
- **exp_044's synthetic messages,** with labels from two Fable readers' agreement. The arguable fifth was dropped.
- **Real voice notes from agencies remain the test that matters.**

## SO WHAT (2026-10-20)

**Adopted** (code path checked):
- **The phone joiner** (`notetaker.join_spoken_numbers`) is in the bot's voice triage, `telegram_bot.handle_triage_voice` (casasol `ed02fbb`). It goes live with the next `bash scripts/deploy.sh bot`.
- **Voice triage can be shown to agencies as equal to typed triage.** Claim wording: "on 331 synthetic enquiries read aloud by computer voices, triage from a WhatsApp-quality voice note matched typed text within about one point, with budgets the exception". Never claim it without "synthetic" and "computer voices".
- **For voice, Whisper's language stays the one shown,** as the bot already does: it was right on 331 of 331.

**Seeds** (exp_046, rev 5, with valuation vs sell and the minimum confidence for Kev's own spam):
1. **The budget reader:**
   - a currency symbol before full digits ("€350,000", "€2,000 a month"), which typed text has too;
   - bare amounts after a budget cue ("up to 2000", "Budget max 700.000");
   - "euromån".
2. **The phone detector and masker:**
   - skip amounts grouped in thousands ("1 600 000", "3.500.000 €");
   - mask phone numbers spelled out in words (Kev reads them now).
3. **The text language detector** missed 8 of 331 on short and mixed messages.
4. **The business-pitch cut** sits where transcription noise can push a genuine firm over (X7-08 at 0.60). The review folder holds it.
5. **If voice is re-measured:** expand abbreviations and number formats per language before the voice reads them, so budgets are measured on speech a person would produce.
6. **Real voice notes:** people, accents, background noise. Ask for them with the free lead assessment.

**Published (2026-10-20):** local-first-ai `tasks/chronos/exp_045_voice_triage/`: both records, the kit, and every prediction with its transcript (test and dev). The audio (26 MB) is not published; the kit regenerates it. Re-scoring the published files reproduces `score.json` exactly. No blog post (Andrei).
