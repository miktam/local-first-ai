# exp_052: the live bot's answering model on MLX against Ollama

The CasaSol Telegram bot's answering model (gemma 4 26B-A4B, thinking on) was served two ways on the Mac mini: Ollama, as it runs live, and an mlx-lm server. Both got the same 120 questions, with the bot's exact requests frozen through its own code. Registered and run on 2026-10-24.

**Result.** MLX made the user wait 10 % longer (median 30.0 s against 27.3 s). By the code checks it broke the bot's rules about as often. Blind Opus judges preferred its answers 39 to 22 and marked fewer wrong (7 against 13). By the registered rule the bot stayed on Ollama. Details: [RESULT.md](RESULT.md).

**What is here:**

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration, and Amendment 1 (blind Opus judges on all 120 pairs, in place of a 30-pair read by Andrei) |
| [RESULT.md](RESULT.md) | verdicts, numbers, what they change |
| `data/questions.jsonl` | the 120 messages (written by Fable 5.1, no personal data) |
| `kit/` | freeze, arms, checks, scorer, the judge-packet builder and the run script, as run |
| `results/` | aggregates: wait and check counts per arm (`score.json`), the unblinded judge tallies (`judge_score.json`) |
| [PRIVATE_SHA256.json](PRIVATE_SHA256.json) | sha256 of every file that stays private |

**What is not here, and why.** The frozen requests hold text the bot retrieved from its own collections: Home History, its area notes, and course material. The answers and the judges' notes are built on that text. All of it falls under CasaSol's competitor protection and is kept on the mini. The hashes let anyone confirm later that a file shown to them is the one we scored.

`kit/freeze52.py` imports the bot's code, which is not public. The other scripts run on the published files plus a local Ollama or mlx-lm.
