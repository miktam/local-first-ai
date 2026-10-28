# exp_060 — which fix closes the local reader's empty answers?

In exp_055, gemma4:26b alone read 220 messages against a fictional person's public footprint and beat the engineered check (balanced accuracy 0.882 against 0.814), but 22 of its answers (10 %) were empty: it thought until the daemon's 8,192-token context was full and never answered. Every number for the local reader is understated by that loss. Before the reader test (Campaign A's A1), this experiment measures three fixes on the same corpus, with the same prompt and the same model call:

- **T** — thinking off;
- **F** — think as before, and on an empty answer retry once without thinking (a fix in glue);
- **C** — the same call with a 32K context.

Ollama 0.40.2 offers no thinking budget for this model (`think` levels are `false` and `true` only; a `"low"` request is accepted and ignored), so these are the three mechanisms there are.

Registered 2026-10-28, before any kit or run. Kit written and smoke-tested the same day on a dummy message; **not run.**

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: what is at stake (the 22 by kind and class), the runtime's options, arms T/F/C, controls C1–C4, H1–H4, the decision rule for A1's reader call, limits, frozen inputs; the kit note at the end |
| `kit/arms60.py` | the three arms over exp_055's corpus, read in place; every counter and the thinking text recorded; the daemon's state before, during and after each arm; arm C restores the 8K context; `--all` runs T, F, C in order under exp_054's audit hook |
| `kit/score60.py` | exp_055's metrics (imported unchanged) per arm, C1–C4, H1–H4, the decision rule, the fate of the 22, the budget curve, disagreements, breakdowns |
| `kit/test_score60.py` | the scorer on a synthesized run built from the real manifest and G baseline; `python3.14 -I kit/test_score60.py` |
| `kit/dummy60.eml` | the fictional smoke-test message (not in the corpus) |
| `dev/20261028T064040Z/` | the smoke test: all three arms on the dummy, the daemon at 8K → 32K → 8K, the audit logs |
| `runs/<ts>/` | written by the run: per arm `results.jsonl`, `thinking.jsonl`, `audit.jsonl`, `daemon_*.json`; arm F also `pass1.jsonl`, `retry.jsonl`; arm C also `restore_call.jsonl`; then `score.json` |

**Inputs** are exp_055's published corpus, prompt and footprint, read in place from `../exp_055_exposure_review_lures/` and hashed in the registration. Seven genuine messages differ from what the exp_055 run saw (five invented domains were replaced by `.example` names before publishing); they are scored but excluded from the agreement tests.

## Run (Andrei, on the mini)

Check first that nothing else is generating on the daemon (`ollama ps`, running experiments). Arm C reloads gemma4:26b at a 32K context for its duration and restores 8K at the end. Expect T in about 15 minutes, F about 3.3 hours, C 2 to 5.5 hours.

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_060_reader_empty_answers && TS=$(date -u +%Y%m%dT%H%M%SZ) && mkdir -p runs/$TS && echo $TS > runs/CURRENT && tmux new-session -d -s exp060 "caffeinate -i zsh -e -o pipefail -c 'python3.14 kit/arms60.py --all --run runs/$TS 2>&1 | tee runs/$TS/run.log && python3.14 kit/score60.py runs/$TS 2>&1 | tee -a runs/$TS/run.log'" && tmux attach -t exp060
```

A run resumes: an arm skips the ids already in its `results.jsonl`. To score again: `python3.14 kit/score60.py runs/<ts>`.

**Status: registered; kit smoke-tested; not run.** Numbering: exp_057 names exp_058 for the harness-layer egress test, exp_059 is registered, so this is 060.

**Everything here is fictional:** the person, every organisation and domain. No message is operational.
