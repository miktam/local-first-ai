# exp_053: higher-precision gemma builds behind the live bot

After exp_052, the question was whether the quality gain came from the weights rather than the runtime. Three Ollama builds of gemma 4 26B-A4B answered as the CasaSol bot's answering model:
- the live Q4_K_M;
- Google's quantisation-aware-trained 4-bit (QAT);
- Q8_0.

They answered 120 fresh questions, behind the bot as fixed on 2026-10-24. Registered and run that day.

**Result.** The bot keeps the live Q4_K_M.
- **Q8_0.** Blind Opus judges preferred it 41 to 21 (p 0.015), but marked more of its answers wrong (9 against 5), and it is 17 % slower.
- **QAT.** It is 10 % faster, but less often preferred (27 to 36) and wrong twice as often (10 against 5).

The run also found two bugs in the code checker shared with exp_052, corrected and reported beside the registered counts. Details: [RESULT.md](RESULT.md).

**What is here:**

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration |
| [RESULT.md](RESULT.md) | verdicts, the 8-bit build by intent, the checker correction, the bot fixes and what they still miss |
| `data/questions.jsonl` | the 120 fresh messages (written by Fable 5.1, none close to exp_052's, no personal data) |
| `kit/` | freeze, arms, checks (registered `check` and corrected `check_v2`), scorer, judge packets, run script, as run |
| `results/` | aggregates: `score.json` (waits, checks, the registered decision), `judge_score.json` (unblinded tallies for both comparisons), `checks_corrected.json` (counts) |
| [PRIVATE_SHA256.json](PRIVATE_SHA256.json) | sha256 of every file that stays private |

**What is not here, and why.** As in exp_052, the retrieved text, the answers built on it, the judge batches and the judges' notes fall under CasaSol's competitor protection and stay on the mini. They are covered by the hashes.
