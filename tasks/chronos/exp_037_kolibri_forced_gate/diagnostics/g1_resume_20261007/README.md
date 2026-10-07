# G1 resume-test diagnostics (gate run 20261007T110355Z)

Gate run `20261007T110355Z` (mbp, 2026-10-07) exited 1. K8 failed on G1 only and K4 passed. Exactly one test failed:

`tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records`, at line 222.

At least records q003 and q004 of the resumed file differed from the uninterrupted reference; `answer_text` was equal and the differing field was cut off by pytest's `-q` truncation. The same test had passed on the same commit in RUNBOOK step 3 on the mbp, and in every run on the mini. After the failure, Andrei chose **(b) diagnostics** (HYPOTHESIS, "Who acts"; the mbp session records the choice in HYPOTHESIS.md).

This folder is that diagnostic, pre-registered. `PLAN.md` fixes everything before any run:
- the question, including one path found by reading the code: a tokenizer-check cache keyed by `id()` that kit code fills earlier in G1's process and the runner reads (PLAN.md section 2, Q6);
- six experiments with their exact commands and N;
- the outcome-to-action table;
- the rules of every follow-up the table can call for, and of every gate fix it can lead to.

Each outcome maps to exactly one consequence: a gate fix, stop and publish, or a named follow-up.

## Files

| File | What it is |
|---|---|
| `PLAN.md` | The pre-registration. It is binding for everything in this folder |
| `diag_resume.py` | The script that runs the experiments: `--experiment cachestate \| isolated \| module \| pair \| g1ctx \| inproc`, plus the follow-up `fresh` (L2). `--classify` applies the table to `out/*.json`. `--selftest` checks the script on stub models only |
| `out/` | The outputs, `<host>_<experiment>_<UTC>.json` and `classification_<UTC>.json`. Paths and names in them are redacted. They are committed after the leak check, partial ones included |

## What this folder is not

- **No kit changes.** It changes no test, runner, port, gate or environment file, and it does not edit HYPOTHESIS.md or RUNBOOK.md.
- **No gate record.** It writes nothing under `results/`, is not a gate run and is not a fix cycle.
- **No gate values touched.** It replaces no gate value and never changes a mutant, a control, a threshold or the gate text.
- **Its only permitted consequences** are a gate fix, a reference fix, or stop and publish (HYPOTHESIS, "Diagnostics between runs").

## Hash scope

`diagnostics/` is outside every hash scope. `tools/hash_tree.py` `SCOPES` holds 20 scopes, and none is rooted at `diagnostics/`:

| Scope | Root |
|---|---|
| `assets` | `assets.json` |
| `evalfw` | `tasks/vendored_evalfw/MANIFEST.json` |
| `requirements` | `env/requirements-mbp.txt` |
| `port`, `convert` | `port/kolibri1.py`, `port/convert.py` |
| `reference` | `reference/` |
| `gate`, `thresholds`, `gate_rules`, `gate_text` | `gate/` and files under it |
| `runner`, `plan_rules` | `runner/` and `runner/plan_rules.json` |
| `tasks`, `manifest_rule` | `tasks/` |
| `scorers` | `scorers/` |
| `analysis` | `analysis/` |
| `bench` | `bench/` |
| `tools` | `tools/` |
| `tests` | `tests/` |
| `env` | three files under `env/` |

This has three consequences:
- committing this folder or its outputs changes no hash;
- P1(a) and P1(e) do not see it;
- it needs no amendment line.

The leak check and the pre-push hook still scan it, like every file of the experiment directory. Before every run the script checks the other way round: `tools/hash_tree.py --check HYPOTHESIS.md` must pass and the gate record's code hashes must recompute exactly.

## Order

1. **Commit and push this package** (`PLAN.md`, `README.md`, `diag_resume.py`). The main session does it on Andrei's go, before any diagnostic runs, after:
   - pulling the mbp session's commit that records Andrei's choice in HYPOTHESIS.md, checking its UTC and label against `PLAN.md`'s header, and setting `CHOICE_COMMIT` in `diag_resume.py` (the script refuses every run until then);
   - pasting the junit failure text, supplied by the mbp session, into `PLAN.md` section 1;
   - the selftest, `py_compile`, the leak check and the hash check.
2. **Run on the mbp.** Pull, then the selftest. Andrei runs (iv) cachestate, (i) isolated, (i′) module, (i″) pair, (ii) G1 context and (iii) in-process, in that order, with the commands in `PLAN.md` 3.8.
3. **Run on the mini.** Pull, then the selftest. Andrei runs (iv), (i), (ii) and (iii).
4. **Commit the outputs.** Each host's `out/*.json` is committed after the leak check: by the mbp session for the mbp, by the main session for the mini.
5. **Apply the table.** The main session runs `diag_resume.py --classify --write` and commits the classification. The consequence follows mechanically:
   - a follow-up runs only after its own code is committed and pushed, in a new file (`PLAN.md` and `diag_resume.py` do not change after the pre-registration commit);
   - a gate fix and any re-run need Andrei's go.

## Selftest

The selftest uses stub models only. It imports nothing from the kit, no MLX and no port, and needs no tiny checkpoint. It runs pytest on small synthetic stub tests, to check the junit, `--basetemp`, invalid-run and C-assertion handling of the pytest arms.

```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate && \
PYTHONDONTWRITEBYTECODE=1 "$PY" diagnostics/g1_resume_20261007/diag_resume.py --selftest --work <a directory outside the repository>
```

It must print `"selftest": "ok"`. Among its checks, the stub outcomes:

| Stub | Outcome |
|---|---|
| deterministic | O5 (no difference): stop and publish |
| injected non-determinism | O1 (L1 pending) |
| resume path alone differs | O2 (L1 pending) |
| field-only difference, no cache state | O4-trace (L1f pending) |
| field-only `split_by` difference with the cache state of (iv) | O4-split: gate fix 6.6 |
| (iv) on a stub world with the same id-keyed cache | "hit" or "stale_only", never "none" |
