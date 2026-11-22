# exp_062 — Inkling 2 Mini in the local reader's seat, reasoning on

**Complete, 2026-11-22: [RESULT.md](RESULT.md).**
The line stops. Inkling 2 Mini answered 194 of 220 messages and flagged 180 of them: 106 of 110 lures and 74 of 110 genuine messages, a balanced accuracy of 0.645 against 0.914 for gemma4:26b with exp_060's retry. H1, H2 and H3 all fail. gemma4:26b keeps the seat.

Thinking Machines released Inkling 2 Mini on 2026-11-20 (26.4B parameters, 3.4B active, a looped transformer, Apache 2.0), small enough for the Mac mini, and a public leaderboard put it level with a frontier model on finding and patching security bugs in code. This experiment asks the question on the task this line has: given exp_055's reader-alone call, does it read exp_055's 220 messages better than the model in the seat? One arm, one pass, the prompt unchanged, scored with exp_055's metric functions against the published runs of G (gemma4:26b alone, 0.882) and F (with the retry, 0.914).

Registered 2026-11-21 before any kit and before any corpus message was sent to the model: [HYPOTHESIS.md](HYPOTHESIS.md).

| File | What it is |
|---|---|
| [RESULT.md](RESULT.md) | controls, verdicts, the decision, the numbers, the bands by class, the empty answers, time, the deviations from the registration, what is held back, limits, the open question, SO WHAT |
| [ADDENDUM_B.md](ADDENDUM_B.md) | registered 2026-11-22, unsigned; kit built, tuned in three rounds and frozen, the test drawn, not run: harder watcher jobs from a generator, Inkling 2 Mini against gemma4:26b, and the maker's variable-loop research model at one to five passes on the same jobs; llama.cpp at tag `b12631`; the dev probes of the day; a correction to Addendum A |
| [ADDENDUM_A_RESULT.md](ADDENDUM_A_RESULT.md) | Addendum A's result, 2026-11-22: both models right on all 40 watcher jobs as scored, neither acted irreversibly without a yes; Inkling median 19.8 s a job against gemma4:26b's 9.3 s, with 12 lookups and 3 alerts nobody asked for; four watchers at once give 1.4 times the throughput. A tie at the ceiling: the jobs were too easy to separate the models |
| [ADDENDUM_A.md](ADDENDUM_A.md) | registered 2026-11-22 and run the same day: a screen of the model as a persistent watcher with tools, reasoning on, against gemma4:26b on 40 made-up jobs; how many watchers the machine carries; whether a monitor can read its reasoning. Sends no exp_055 message; the reader verdict stands |
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: the arm, the pinned runtime, controls C1–C4, H1–H3, the decision rule, limits, frozen inputs; the kit note; Amendment 1 (an error is not an answer, before the run); Amendment 2 (a look at the partial counts during the run) |
| `kit/arm62.py` | checks the frozen inputs, the server's commit and the model file's hash; starts the side server; runs the arm under exp_054's audit hook; a failed call stops the arm and is asked again on resume; the reasoning text goes to a private folder and is hashed |
| `kit/score62.py` | exp_055's metrics (imported unchanged), C1–C4, H1–H3, the paired tables against F and G, the decision rule; voids an incomplete run |
| `kit/describe62.py` | written after the run, descriptive only: bands by class, answered-only counts, domain age, the empty answers, the paired table by kind, phrase counts in the stated reasons, time, the reasoning share by characters; writes `runs/<ts>/describe.json` |
| `kit/build62a.py`, `kit/jobs62a.json`, `kit/smoke62a.json` | Addendum A: the made-up world, the 40 frozen watcher jobs and the five smoke jobs |
| `kit/watch62a.py`, `kit/score62a.py`, `kit/monitor62a.py` | Addendum A: the driver (G through Ollama, I on the side server, the futility rule, the four-slot part), the scorer, and the two rules that read the reasoning |
| `kit/gen62b.py`, `kit/test62b.json` | Addendum B: the generator of hard watcher jobs with the rules and the reference that computes every expected outcome; the frozen test |
| `kit/watch62b.py`, `kit/score62b.py`, `kit/monitor62b.py` | Addendum B: the driver for the seven arms, the scorer, and the script that reads the reasoning for HB3 and HB4 |
| `kit/smoke62b.json`, `kit/dev62b-r1.json` to `-r3.json`, `dev_b/` | Addendum B: the smoke and the three tuning rounds, on other seeds than the test |
| `dev_a/20261122T075548Z/` | Addendum A's smoke test on the five smoke jobs |
| `runs_a/20261122T081037Z/` | Addendum A's run: `G/` and `I/` results, the audit logs, `capacity/`, `score.json`, `monitor.json`, `describe.json`. The reasoning text of both arms is held back and hashed |
| `kit/describe62a.py` | Addendum A, written after the run, descriptive only: the result by job kind, unrequested alerts, tokens |
| `dev/20261121T123544Z/` | the smoke test on exp_060's dummy message (not a corpus message) |
| `runs/20261121T131040Z/` | the run: `I/results.jsonl`, `I/audit.jsonl`, `I/server.log`, `I/reasoning.sha256`, `meta.json`, `run.log`, `score.json`, `describe.json`. The reasoning text is held back and hashed |

**Inputs** are exp_055's published corpus, prompt and footprint, read in place from `../exp_055_exposure_review_lures/`, with the hashes of exp_060's *Frozen inputs* table.

## Run (Andrei, on the mini)

The run needs a llama.cpp build that carries pull request #32704 at `~/models/llama.cpp-pr32704` and the model file at `~/models/inkling2/`; the driver checks both. It took 12.5 hours.

```bash
tmux new -s exp062 'cd ~/REPOS/local-first-ai/tasks/chronos/exp_062_inkling_reader_arm && TS=$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p runs/$TS && echo "run folder: runs/$TS" && caffeinate -i /opt/homebrew/bin/python3.14 kit/arm62.py --all --run runs/$TS --private ~/exp062-private/runs/$TS 2>&1 | tee -a runs/$TS/run.log; exec zsh'
```

A run resumes with the same run folder. To score: `python3.14 -I kit/score62.py runs/<ts>`, then `python3.14 -I kit/describe62.py runs/<ts>`.

**Status: the main run and Addendum A are complete; Addendum B is registered, unsigned and not run.** `runs/20261121T131040Z/` went from 2026-11-21 13:10Z to 2026-11-22 01:39Z, at thermal level 0 before every call. Every figure is from one build of the merged fix's commit; the repeat on a tagged llama.cpp build has not been run.

**Everything here is fictional:** the person, every organisation and domain. No message is operational.
