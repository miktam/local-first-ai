# exp_060 — which fix closes the local reader's empty answers?

In exp_055, gemma4:26b alone read 220 messages against a fictional person's public footprint and beat the engineered check (balanced accuracy 0.882 against 0.814), but 22 of its answers (10 %) were empty: it thought until the daemon's 8,192-token context was full and never answered. Every number for the local reader is understated by that loss. Before the reader test (Campaign A's A1), this experiment measures three fixes on the same corpus, with the same prompt and the same model call:

- **T** — thinking off;
- **F** — think as before, and on an empty answer retry once without thinking (a fix in glue);
- **C** — the same call with a 32K context.

Ollama 0.40.2 offers no thinking budget for this model (`think` levels are `false` and `true` only; a `"low"` request is accepted and ignored), so these are the three mechanisms there are.

Registered 2026-10-28, before any kit or run.

| File | What it is |
|---|---|
| [HYPOTHESIS.md](HYPOTHESIS.md) | the registration: what is at stake (the 22 by kind and class), the runtime's options, arms T/F/C, controls C1–C4, H1–H4, the decision rule for A1's reader call, limits, frozen inputs |
| `kit/` | to be written: `arms60.py` (the three arms, every counter recorded) and `score60.py` (exp_055's metrics, agreement tests, controls, hypotheses, decision) |
| `runs/<ts>/<arm>/` | written by the run: `results.jsonl`, `thinking.jsonl`, the daemon's state before and after; arm F also `pass1.jsonl` and `retry.jsonl` |

**Inputs** are exp_055's published corpus, prompt and footprint, read in place from `../exp_055_exposure_review_lures/` and hashed in the registration. Seven genuine messages differ from what the exp_055 run saw (five invented domains were replaced by `.example` names before publishing); they are scored but excluded from the agreement tests.

**Status: registered; no kit; not run.** Numbering: exp_057 names exp_058 for the harness-layer egress test, exp_059 is registered, so this is 060.

**Everything here is fictional:** the person, every organisation and domain. No message is operational.
