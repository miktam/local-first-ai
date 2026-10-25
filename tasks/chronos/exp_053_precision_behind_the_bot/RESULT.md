# exp_053 result: the live Q4_K_M stays; 8-bit writes more and is preferred, but is slower and more often wrong

**Run:** `runs/20261024T174338Z`, 2026-10-24, 17:43–21:33Z, on the Mac mini.
- 118 questions per build. No outside request touched any answer.
- **Drift check:** the first 10 G4 questions took a median 28.6 s at the start and 24.6 s at the end. That is sampling spread on 10 thinking answers, not a slowdown.
- **Judging:** 8 blind Opus 5.5 judges covered all 236 pairs. Scores are in `judge/score.json` and `score.json`.

## Verdicts

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | 8-bit writes better answers | judges prefer **G8 41**, G4 21, ties 56 (sign p 0.015); but answers wrong **G8 9**, G4 5 | **fail**: preferred more often, but wrong more often too |
| H2 | QAT writes better answers | judges prefer GQ 27, **G4 36**, ties 55 (p 0.31); answers wrong GQ 10, G4 5 | **fail**: no better, if anything worse |
| H3 | Speed ≤ 1.10 × G4 | GQ 0.896 (29.2 s against 32.6 s); G8 1.167 (38.1 s) | GQ pass, G8 fail |
| H4 | Flags ≤ G4 + 3, as registered | G4 4, GQ 10, G8 8 | both fail (see the correction) |

**Decision (registered): keep `gemma4:26b` Q4_K_M.** Neither candidate passes its quality hypothesis, so the decision does not depend on the checker correction below.

## Where the 8-bit build wins and loses

| Routed intent | prefers G8 | prefers G4 | ties |
|---|---|---|---|
| legal and tax | **21** | 4 | 7 |
| process | **12** | 2 | 3 |
| area | 5 | 4 | 12 |
| market | 2 | 5 | 12 |
| property search | 1 | 6 | 22 |

- **Where 8-bit gains.** It is preferred on legal and process questions, mostly for giving the relevant general rule (the new fallback rule) where G4 gave less.
- **Where it loses.** More content means more errors: 9 wrong against 5. One of them claimed a viewing ("In meinen Besichtigungen"), against fix 1.
- **On area, market and property questions** it brings nothing, and G4 is slightly preferred.
- **Its costs:** 17 % slower median (p90 66 s against 59 s) and 28 GB of memory.

**QAT** is 10 % faster, but its answers are not better (27 against 36), and twice as often wrong (10 against 5). Twice it thought until the 8,192-token context ran out and returned an empty answer, one more than G4. It sometimes thinks for 5,000+ tokens before a one-line decline.

## The code checks, corrected after the run (descriptive)

The registered checker had two bugs. They showed up here, and both sit in exp_052's checker too:
- **V3, places.** It matched place names inside other words: "Coín" in Spanish "coincida", "Istán" in "distancia".
- **V1, language.** It read short one-line declines in French or German as Spanish or English, because the answers named Spanish places (Salobreña, Bahía de Marbella).

The correction:
- V3 now matches whole words only.
- V1 now ignores capitalised words; on the 916 labelled enquiries it scores 914, against 913 before.
- `checks_corrected.json` holds the result: G4 1, GQ 3, G8 0. After reading the 3 (QAT's remaining "language" flag is correct French), the real failures are G4 1, GQ 2, G8 0: all empty answers where thinking used up the context.

The registered H4 verdicts stand as computed. The decision does not change under the correction.

## The three bot fixes (deployed 2026-10-24, CasaSol f37b8a7)

1. **"Community report, not my visit."** No answer in any build said "I saw" or "my visit". Two answers still slipped into first-person viewing in German ("In meinen Besichtigungen", "Ich habe … nichts gesehen"). Partly fixed.
2. **The IRNR and deposit notes.** They were retrieved for 29 of 118 questions. One QAT answer misread them (the 19 % rate charged on gross income). Several judged errors were on facts the notes do not cover (below).
3. **The general rule when retrieval has nothing.** It works where the model knows the rule: that is what won the 8-bit build its legal and process pairs. The judges list legal questions where no build gave the rule: LAU art. 25 first refusal, Ley de Costas, the Beckham regime, delayed delivery. And as the rule says, there is no general fallback for area and market questions, which now get one-line declines.

## SO WHAT

- **The bot keeps `gemma4:26b` Q4_K_M on Ollama.** Neither QAT nor Q8_0 is the better trade: QAT is faster but worse, and Q8_0 is preferred but slower and more often wrong.
  - The exp_052 hint that higher precision explains MLX's quality win is only half supported. Q8_0 is preferred about as strongly as MLX was (41 against 21; MLX 39 against 22), but unlike MLX it made more errors.
  - Different questions and a different prompt separate the two runs, so they are not directly comparable.
- **The remaining errors are content, not weights, and every build shares them.** These are the next bot fixes:
  1. **Contributor names leak.** An answer attributed a note to "a local resident (Andrei)", because a contributor name is in the retrieved community notes. Strip names when the bot builds its context, or forbid naming contributors.
  2. **The course's hypothetical sales-pitch figures are presented as market data** ("prices rise 5–6 % a year", "build costs up 30 %"), in several answers on every build. The prompt or the index should mark course examples as examples.
  3. **Wrong legal facts every build repeats.** Each needs a fact-checked note:
     - The 7-year minimum lease applies when the *landlord* is a legal person, not the tenant.
     - Stamp duty (AJD) in Andalusia is 1.2 %, not 1.5 %.
     - No stressed-market zones are declared in Andalusia, so the 70 % and 90 % landlord reductions do not apply there.
     - The bank pays the AJD on a mortgage.
  4. **First person in German.** The fix-1 wording should also forbid "I have (not) seen".
- **The checker** needs the two corrections above before it is reused (applied in `checks53.check_v2` and `langcheck.detect_v2`).
