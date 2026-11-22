# exp_062 — Inkling 2 Mini in the local reader's seat, reasoning on

**Complete, 2026-11-22: [RESULT.md](RESULT.md).**
The line stops. Inkling 2 Mini answered 194 of 220 messages and flagged 180 of them: 106 of 110 lures and 74 of 110 genuine messages, a balanced accuracy of 0.645 against 0.914 for gemma4:26b with exp_060's retry. H1, H2 and H3 all fail. gemma4:26b keeps the seat.

Thinking Machines released Inkling 2 Mini on 2026-11-20 (26.4B parameters, 3.4B active, a looped transformer, Apache 2.0), small enough for the Mac mini, and a public leaderboard put it level with a frontier model on finding and patching security bugs in code. This experiment asks the question on the task this line has: given exp_055's reader-alone call, does it read exp_055's 220 messages better than the model in the seat? One arm, one pass, the prompt unchanged, scored with exp_055's metric functions against the published runs of G (gemma4:26b alone, 0.882) and F (with the retry, 0.914).

Registered 2026-11-21 before any kit and before any corpus message was sent to the model: [HYPOTHESIS.md](HYPOTHESIS.md).

| File | What it is |
|---|---|
| [RESULT.md](RESULT.md) | controls, verdicts, the decision, the numbers, the bands by class, the empty answers, time, the deviations from the registration, what is held back, limits, the open question, SO WHAT |
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: the arm, the pinned runtime, controls C1–C4, H1–H3, the decision rule, limits, frozen inputs; the kit note; Amendment 1 (an error is not an answer, before the run); Amendment 2 (a look at the partial counts during the run) |
| `kit/arm62.py` | checks the frozen inputs, the server's commit and the model file's hash; starts the side server; runs the arm under exp_054's audit hook; a failed call stops the arm and is asked again on resume; the reasoning text goes to a private folder and is hashed |
| `kit/score62.py` | exp_055's metrics (imported unchanged), C1–C4, H1–H3, the paired tables against F and G, the decision rule; voids an incomplete run |
| `kit/describe62.py` | written after the run, descriptive only: bands by class, answered-only counts, domain age, the empty answers, the paired table by kind, phrase counts in the stated reasons, time, the reasoning share by characters; writes `runs/<ts>/describe.json` |
| `dev/20261121T123544Z/` | the smoke test on exp_060's dummy message (not a corpus message) |
| `runs/20261121T131040Z/` | the run: `I/results.jsonl`, `I/audit.jsonl`, `I/server.log`, `I/reasoning.sha256`, `meta.json`, `run.log`, `score.json`, `describe.json`. The reasoning text is held back and hashed |

**Inputs** are exp_055's published corpus, prompt and footprint, read in place from `../exp_055_exposure_review_lures/`, with the hashes of exp_060's *Frozen inputs* table.

## Run (Andrei, on the mini)

The run needs a llama.cpp build that carries pull request #32704 at `~/models/llama.cpp-pr32704` and the model file at `~/models/inkling2/`; the driver checks both. It took 12.5 hours.

```bash
tmux new -s exp062 'cd ~/REPOS/local-first-ai/tasks/chronos/exp_062_inkling_reader_arm && TS=$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p runs/$TS && echo "run folder: runs/$TS" && caffeinate -i /opt/homebrew/bin/python3.14 kit/arm62.py --all --run runs/$TS --private ~/exp062-private/runs/$TS 2>&1 | tee -a runs/$TS/run.log; exec zsh'
```

A run resumes with the same run folder. To score: `python3.14 -I kit/score62.py runs/<ts>`, then `python3.14 -I kit/describe62.py runs/<ts>`.

**Status: complete.** `runs/20261121T131040Z/` went from 2026-11-21 13:10Z to 2026-11-22 01:39Z, at thermal level 0 before every call. Every figure is from one build of the merged fix's commit, before any tagged release carried it; the repeat on a tagged release has not been run.

**Everything here is fictional:** the person, every organisation and domain. No message is operational.
