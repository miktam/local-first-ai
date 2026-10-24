# Kev-4B — model notes

*An open decision model: it picks among given options and returns a probability for each, and it never writes text. It is one of the open models that followed TypeSafe's Jev ([jev.md](./jev.md)). We run it as Q8_0 on the Mac mini, through llama.cpp's `/v1/systemone` endpoint. One fact per line, dated and sourced; newest at the bottom of each section.*

---

## What it is good at

- 2026-10-18 · As an answerability gate in front of an untrained 4B voice, it nearly matched Clef 27B. It had 1 false abstention in 101 against 0, and lifted exact answers by +0.067 against +0.080, at about 2 s per question instead of 12 s and 4.5 GB. [exp_042](../../tasks/chronos/exp_042_answerability_gate_retrieval/RESULT.md)
- 2026-10-18 · Lead triage, on 235 synthetic leads in seven languages: request 0.894, next step 0.898, spam caught 29 of 32 with 3 of 203 genuine leads flagged. [exp_043](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-19 · Rev 4, on 331 fresh synthetic leads: request 0.909 and next step 0.938 (standard stratum); 2 of 282 genuine leads flagged as spam. [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
- 2026-10-20 · From WhatsApp-quality voice notes, transcribed by Whisper large-v3-turbo, triage stayed within about a point of typed text: request 0.900 against 0.912, next step 0.921 against 0.931, on 331 synthetic enquiries read by computer voices. [exp_045](../../tasks/chronos/exp_045_voice_triage/RESULT.md)
- 2026-10-20 · Its model card says English only, yet on 425 non-English synthetic enquiries in six languages it read the request right 0.927 of the time, and translating them into English first (gemma4:26b) gave 0.925. [exp_046](../../tasks/chronos/exp_046_translate_first/RESULT.md)

## Speed and cost

- 2026-10-18 · One call answers every question about a lead (11 in rev 3): median 1.6 s on the mini. [exp_043](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-19 · With one more question (12, rev 4): median 1.67 s. [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
- 2026-10-20 · A 20 s voice note goes from audio to triage in about 3 s on the mini: Whisper 1.1 s, Kev 1.7 s. [exp_045](../../tasks/chronos/exp_045_voice_triage/RESULT.md)

## Where it is weak

- 2026-10-17 · Its model card says English only. We use it in Polish and Spanish too, and report those results separately. [exp_041](../../tasks/chronos/exp_041_answerability_gate/HYPOTHESIS.md)
- 2026-10-18 · Polish is its weakest language for triage intent: 0.77, against 0.87–1.00 for the other six. [exp_043](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-18 · It reads buyers of holiday homes as holiday renters: 10 of its 25 intent errors. [exp_043](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-18 · Its "qualified" (0.78) and timeline (0.73) answers are too weak to show to an agent. [exp_043](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-19 · It gives about 0.6 to facts a message never states (financing, for example). The demo shows yes/no facts only from 0.7. [exp_043, note of 2026-10-19](../../tasks/chronos/exp_043_kev_lead_triage/RESULT.md)
- 2026-10-19 · Its raw confidence is ordered but not calibrated (ECE 0.262): Sure 1.00 right, Fairly sure 0.889, Unsure 0.798. Show the bands, never the number. [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
- 2026-10-19 · New top confusion: an owner who asks for a price and mentions selling is read as a seller (9 times). [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
- 2026-10-20 · Transcription tips borderline buyers toward holiday rent: voice moved 7 enquiries there, 5 of which it had right when typed. [exp_045](../../tasks/chronos/exp_045_voice_triage/RESULT.md)
- 2026-10-20 · Polish stays its weakest language (0.853), and translation into English does not fix it (0.868, +1 net of 68). [exp_046](../../tasks/chronos/exp_046_translate_first/RESULT.md)
- 2026-10-21 · Behind the same rules, Clef 27B ([clef-27b.md](./clef-27b.md)) read the request right 0.991 of the time against Kev's 0.919, and removed most of Kev's known confusions (valuation read as sell, 10 → 1). Kev was the ceiling, not the rules. [exp_047](../../tasks/chronos/exp_047_clef_triage/RESULT.md)
- 2026-10-24 · Its intent confidence is underconfident (ECE 0.268): below 0.5 it read the request right 83 % of the time (53 leads), and above 0.8 on 95 of 95. A hand-off at 0.5 sent most of those to a person. [exp_050](../../tasks/chronos/exp_050_society_vs_one_model/RESULT.md)

## What helped

- 2026-10-19 · A separate yes/no question ("wants to own?") fixed more of the holiday confusion than rewording the options did: buy→holiday-rent errors fell from 11 to 4 across 331 leads. [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
- 2026-10-19 · A rule from code overrides Kev only when Kev's own probability agrees. That removed 3 of the 5 false spams at the cost of one more missed spam. [exp_044](../../tasks/chronos/exp_044_kev_lead_triage_rev4/RESULT.md)
