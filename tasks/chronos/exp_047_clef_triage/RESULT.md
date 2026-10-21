# exp_047 result: a bigger decision model behind the same rules

*Scored 2026-10-21, 00:05, on the mini, under HYPOTHESIS.md (signed by Andrei 2026-10-20 21:22 CEST, `8a856ce`; text as in `ee2cb65`). Andrei started the test pass in tmux at 21:23; Claude finished the write-up while he slept, as he asked. Set: all 566 enquiries of exp_043's and exp_044's test sets, public copies (`a5f823d9…`). Code: casasol `d2c4bfe` (`clef47.py` `2f9ef115…`); triage = rev 4.1 as frozen (`2acc7a1d…`) in both arms. Models: Kev-4B Q8_0 and Clef 27B Q8_0 (`9fe953f08485…`, matching exp_041's file), both on llama.cpp b11711 (`b1-9d1e036`). Files: [results/test/](results/test/) (predictions, `done` records with every hash, `score.json`, both run scripts and logs, the server logs), [results/dev/](results/dev/). No cloud model was used anywhere.*

## Verdict: **The bigger model lifts the weak fields** (H1–H4 pass)

| Hypothesis | Kev-4B | Clef 27B | Bar | |
|---|---|---|---|---|
| **H1** qualified (565) | 0.821 | **0.936** (+0.115; 76 vs 11) | ≥ 0.80 and ≥ Kev + 0.05 | pass |
| **H2** timeline (532) | 0.712 | **0.833** (+0.121; 99 vs 35) | ≥ 0.75 and ≥ Kev + 0.05 | pass |
| **H3** request (566) · next step (565) | 0.919 · 0.924 | **0.991 · 0.973** (+0.072 · +0.049) | ≥ −0.02 each | pass |
| **H4** genuine flagged as spam (485) · spam caught (81) | 2 · 75 | **0 · 81** | ≤ Kev + 1 · ≥ 0.85 | pass |

All four discordant splits favour Clef with one-sided McNemar p < 0.001. The verdict map's rule 1 applies to both weak fields. Adopting it is Andrei's decision.

## What changed (descriptive)

- **Clef is better on almost every field:**

  | | Kev-4B | Clef 27B |
  |---|---|---|
  | property type | 0.887 | 0.977 |
  | from abroad, where stated | 0.876 | 0.968 |
  | urgency within one step | 0.929 | 0.979 |
  | request, the model's own reading before any rule | 0.883 | 0.991 |

  With Kev, the code rules lift request from 0.883 to 0.919. With Clef they change nothing: its own reading is already 0.991.
- **One field got worse: financing**, where stated, went from 1.00 to 0.923, with 10 wrong only for Clef. All 10 are enquiries whose mortgage is *already approved* ("mortgage pre-approved", "hipoteca approved with Sabadell", "die Finanzierung steht bereits"). The question asks "Will the sender need a mortgage?". Clef reads it literally: the mortgage is already arranged. The labels mean "uses a mortgage". This is a question-wording problem, not a misreading of the enquiry.
- **Kev's known confusions largely disappear:**
  - valuation read as sell, 10 → 1;
  - buyers read as holiday renters, 9 → 1;
  - questions read as transactions, 11 → 0;
  - pitches read as sellers, 4 → 0.

  Clef's remaining 5 request errors: buy → invest 2, buy → info 1, buy → holiday rent 1, valuation → sell 1.
- **Polish was Kev's weakness, not the language's.** Polish request goes from 0.853 to 0.985. Every language is at 0.985–1.00 with Clef, against 0.853–0.986 with Kev. Request is +0.059 on exp_043's set and +0.082 on exp_044's.
- **Clef's yes/no answers are decisive.** 58 of its 566 "qualified" scores fall between 0.3 and 0.7, against 191 of Kev's. Agents would see fewer "unsure" answers.
- **The cost:**
  - **Speed:** median 13.5 s per enquiry against 1.66 s, 8.1× slower.
  - **Memory:** about 29 GB loaded, against Kev's 4.5 GB.

## Incident (recorded during the run)

- **The first attempt stopped on test enquiry 36 of Clef's arm** (21:47, `exp047_clef_server_test.log`): the GPU ran out of memory.
  - The script had unloaded Ollama's pinned models before starting Clef. Within minutes something requested gemma4:26b again, and Ollama's keep-alive holds it permanently.
  - The server returned HTTP 500 and the runner stopped. The failed enquiry was never written, and Kev's arm had finished.
- **The resume (22:00–00:05) unloaded the models again and continued from enquiry 36,** as the signed plan allows.
  - Its watcher logged gemma4:26b loaded again from 22:22 onward, 103 warnings, yet no error occurred.
  - So the first failure was most likely the moment gemma4:26b was loading while Clef was mid-computation, not the two models side by side.
- **What reloads gemma4:26b is not identified.** Nothing was connected to Ollama when checked, and the bot's logs show nothing. Two other Claude sessions in this workspace were busy at the time.

## Known limits (as registered)

- **Synthetic enquiries, labelled by two Fable readers who agreed.** The ambiguous fifth was dropped, so this test is easier than an inbox. A larger model may also simply read more like Fable does. Real enquiries remain unmeasured.
- **The cuts were tuned for Kev.** Clef did better anyway, but its own cuts were not explored.
- **One pass per arm.** These sets were used before for other verdicts. Nothing was tuned on them for Clef.

## SO WHAT (2026-10-21)

**Nothing changes on its own.** The bot still runs rev 4.1 on Kev-4B. The verdict makes Clef an adoption candidate, and that decision is Andrei's:
- **Clef for all triage:** 13.5 s per enquiry, which fits the queue design in `SPECIALISTS.md` (an agency inbox is dozens of enquiries a day, not hundreds a minute). Qualified and timeline become showable to agents. But it needs 29 GB on the mini, which competes with whatever keeps reloading gemma4:26b.
- **A hybrid:** Kev answers first, and Clef handles the weak fields or Kev's unsure answers. Cheaper, but untested; it would need its own measurement.
- **Claim wording:** "On 566 synthetic enquiries in seven languages, a 27B local decision model behind the same code rules read the request right 99.1 % of the time against 91.9 % for the 4B, and made 'qualified' (0.94) and timeline (0.83) good enough to show."

**Seeds:**
1. **Clef-Flash 9B** (on disk since exp_041): does a middle-sized model keep most of the gain at a fraction of the cost? Fully local, ready to run.
2. **The financing question's wording** ("will they need" against "will they use a mortgage, approved or not"), for rev 5.
3. **Rev 5's plan changes.**
   - Valuation vs sell, Kev's top confusion (10 errors), is 1 error with Clef. Kev's spam picks and pitches read as sellers vanish too.
   - Rev 5 should be measured against Clef as well as Kev before adding questions that only Kev needs.
4. **Find what reloads gemma4:26b** before running Clef in production, and decide what the mini keeps loaded.
5. **Labels from a different model family**, so a "99 %" doesn't partly measure agreement with Fable's reading. Real enquiries remain the test that matters.

**Published (2026-10-21):** local-first-ai `tasks/chronos/exp_047_clef_triage/`: both records, the kit, every prediction, both run scripts and logs, and the server logs. Re-scoring the published files reproduces `score.json` exactly. Pushed together with the exp_046 next-step correction. No blog post.
