# exp_037: diagnostics of the G1 failure in gate run 20261007T110355Z (pre-registered)

| | |
|---|---|
| Gate run | `20261007T110355Z` on the mbp, 11:03:55–12:55:09Z. Record `results/gate/gate_20261007T110355Z.json` (sha256 `cb21c2dfa461c0cd17b68f47e9577cca248062bf3a7c6becbbfbd90dc9bdd809`), committed in `1efbadc` with its phase files, its gate block and the power record `results/power/power_20261007T135204Z.json` |
| Verdict | Exit 1: K8 FAIL, K4 PASS. The only failing K8 check is `g1`. Every other check passed on both arms; allowed_B is {1, 2, 4, 8, 16} for K8 and for K4. run_counts: gate_runs 1/3, cycles 0/2, may_run_again true; blind_phase_reached true |
| Code the gate ran | `8dea1dc` (the record's `git.head`). Between `8dea1dc` and `1efbadc` only `HYPOTHESIS.md` and `results/` changed. The record has `git.dirty = true`, so before every run the script also recomputes the record's code hashes (port, runner tree, gate code, reference tree) and checks all 20 frozen scopes (3.0) |
| Andrei's choice | (b) diagnostics, 2026-10-07T14:49:52Z, label "b · diagnostics". The mbp session recorded it in HYPOTHESIS.md (the run-record block "choice after gate run 1 (20261007T110355Z): b · diagnostics, by Andrei at 2026-10-07T14:49:52Z", written 2026-10-07T15:40:20Z) in commit `d60f3b6ec857b021a540d847a6594ef14a8d554d` (`CHOICE_COMMIT`). The main session checked the UTC and the label against that entry before committing this package; they match |
| Written against | Drafted at `1efbadca1c4eaf70f0399950a570069149287e22` (`WRITTEN_AGAINST`); committed on top of `CHOICE_COMMIT` (set at commit, 9.1). Between the two only HYPOTHESIS.md changes. The main session wrote this on the mini before any diagnostic ran. The only thing run here so far is `diag_resume.py --selftest`, which uses stub models and imports nothing from the kit (no port, no tiny checkpoint) |
| Binding rules | HYPOTHESIS.md, "Fix-cycle, freeze and amendment rules": "No relaxation route", "Gate runs, cycles and crashes" (P1(d) included), "Gate fix" and "Diagnostics between runs". This file applies those rules and changes none of them |

Files in this folder: `PLAN.md` (this file), `README.md`, `diag_resume.py` (the experiments and the table, in code), and `out/` (the outputs, committed after they are produced).

---

## 1. What failed

G1 runs the kit's weight-free suite on the run host. It runs `tests/` minus `test_gate_drivers_tiny.py`, `test_gate_end_to_end_tiny.py` and `test_gate_ftiny.py`, through `gate/checks/g1_synthetic.py`. Its pass rule is 0 failures and 0 errors.

In gate run `20261007T110355Z` it reported 1 failed, 1,900 passed and 8 skipped (all 8 marked `build-host only:`), in 384.76 s, between 11:05:53Z and 11:12:18Z. The junit is a work file on the mbp and is not committed: `$EXP036_WORK/exp037/gate_g1/junit_20261007T110355Z.xml`, sha256 `5395d8aabeace5c95011d86b8b6ec0749984abc25aca14e9e942d6983f7e41be`.

**The failing test:** `tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records`. It failed at the assertion on line 222:

```
assert content(records(p)) == content(records(ref))
```

Records **at least** q003 and q004 differed (see below).

**The failure text from the junit** (the target test's `<failure>` message and text, home path redacted, verbatim). It was not available on the mini when this package was committed. The mbp session commits it as `gate1_junit_failure.txt` in this folder, before M0 and before any run (9.2). Reading an existing gate artefact runs nothing, and no rule in this plan depends on its content.

**What the visible output does and does not show.**
- G1 runs pytest with `-q`. Without `-vv`, pytest truncates the whole assertion explanation to 8 lines or 640 characters, whichever comes first, and ends it with "Full output truncated … use '-vv' to show".
- For two unequal dicts the explanation first says "Omitting N identical items", then "Differing items:", then one line per differing item. The differing keys are iterated from a set, so their order is arbitrary; the 640-character limit can cut the list after one or two items. The visible q003 and q004 are therefore a subset: all three resumed items (q003–q005) may have differed. The "Omitting N identical items" line in the quoted text gives the exact count (N = 3 means three items differed).
- Each differing item is printed through saferepr, capped at 240 characters per side, with the record's keys in sorted order. So `answer_text`, the first key, was shown and was equal. The field that actually differs is in the cut-off part and is unknown.
- `StubTok` decodes id i as `"<i>"`, so equal `answer_text` means equal answer ids. A difference in `completion_ids` could hide only in a trailing EOS or in a reasoning segment. A field-only difference is therefore likely. This is descriptive and is not an input to any rule.

**Where the same test passed:**
- on the mbp in RUNBOOK step 3 (04:28–04:44Z), on the same commit. That run was the full suite, with the heavy tiny tests included;
- in every run on the mini (1,962 passed, in venv312 and venv314).

**What the test does.** It loads the model by the module fixture `model`, which calls `port_harness.load_port(tiny_vendor_dir, float32=True)`. That loads the exp_037 port from the tiny vendor checkpoint: seed 0, written by `tests/conftest.py` `_write`. It then runs these steps:

1. **Items:** `items(6, max_tokens = 8 + 2k)` gives q000–q005, with random prompts of 12 + 9k tokens.
2. **Reference run:** `run(model, ref, its, B=1, cap=20)` writes `ref.jsonl`. It calls `runner/generate.run_cell`, greedy, with a new `StubTok()`, which decodes id i as `"<i>"`.
3. **Stopped run:** the same run, with another new `StubTok()`, into `raw/S2/K8/mmlu_en_high.jsonl`, stopped after three records.
4. **Torn line:** a torn fourth line is appended.
5. **Completed keys:** `jsonl.completed_keys` collects the items already done.
6. **Resumed run:** the run resumes with `done`, with a third new `StubTok()`, and generates the last three items.
7. **Comparison:** the records of both files are compared through `content()`, which drops the TIMING fields: `t_submit`, `t_first_token`, `t_done`, `wall_s` and `batch_id`.

## 2. The question

Why did `content(records(p))` differ from `content(records(ref))` in the gate's G1 on the mbp? The experiments answer six sub-questions:

- **Q1.** Is greedy generation on the tiny checkpoint non-deterministic within one process, so that two uninterrupted runs differ?
- **Q2.** Does the resume path alone change records while uninterrupted runs agree?
- **Q3.** Does the failure need the state that other tests leave in G1's process?
- **Q4.** Do only non-generation fields differ, with completion_ids identical?
- **Q5.** Does the failure reproduce at all?
- **Q6.** Does kit code that ran earlier in G1's process leave state that the runner reads when it writes a record?

**A path for Q6, found by reading the code (nothing was run).**
- `runner/generate._scorer_split` (lines 181–198) calls `scorers.reasoning.check_tokenizer(family, tokenizer)`. If the check passes, every record of that `run_cell` call is split by the scorers and gets `split_by = "scorers.reasoning.split_ids"`; if it raises, the runner's own split is used and `split_by = "runner.generate.split_ids"` (lines 464–469).
- `check_tokenizer` (scorers/reasoning.py lines 355–370) caches each passed check in the module-global set `_CHECKED`, keyed by `(family, id(tokenizer))`, and never removes a key.
- In G1's order, `tests/test_bench_batch_flip.py::test_default_prepare_and_extract_through_the_sibling_modules` (lines 221–244) runs before `tests/test_runner_tiny.py`. It loads the real Kolibri tokenizer and extracts answers through `bench/batch_flip.default_extract` → `scorers.reasoning.split_reasoning`, which puts `("kolibri", id(tok))` into `_CHECKED`. The tokenizer object is freed after the test; its id stays in the set.
- `StubTok` and mlx-lm's `TokenizerWrapper` are both plain classes without `__slots__`, so their instances have the same allocation size. CPython can give a new `StubTok()` the freed tokenizer's address. The check then returns early for the stub, and that run's records get the scorers' split. With the tiny vocabulary both splits give status "none" and the same answer ids, so `answer_text` stays equal and `split_by` differs. That matches the visible output.
- Whether an address is reused depends on the allocation history, which differed in RUNBOOK step 3 (heavy tests included) and on the mini. (i) and (iii) never fill `_CHECKED`, so they cannot see this path; (i″), (ii) and (iv) can.

This is a hypothesis. The experiments below test it alongside every other answer, and the table in section 5 decides.

Each possible answer maps to exactly one permitted consequence (section 5): a gate fix, stop and publish, or a named follow-up diagnostic whose own rules are fixed here. None leads to a reference fix, because the failing test does not involve the reference.

## 3. The experiments

### 3.0 Common to all

- **No changes to the kit.** `diag_resume.py` runs the experiments. It changes no kit file, writes no gate record and writes nothing under `results/`.
- **Preconditions.** If any fails, the script exits with code 3 and the run is not data.
  - **Kit binding:**
    - the gate record has the sha256 above;
    - these files have the sha256 pinned in the script, each unchanged since `8dea1dc`: the failing test, `tests/conftest.py`, `tests/tiny_checkpoint.py`, `tests/port_harness.py`, `tests/test_bench_batch_flip.py`, `gate/checks/g1_synthetic.py`, `runner/generate.py`, `runner/jsonl.py`, `scorers/reasoning.py`, `bench/batch_flip.py` and `port/kolibri1.py`;
    - since `8dea1dc`, committed or not, only `results/`, `diagnostics/`, `aborted/`, `evidence/`, `amendments/`, `HYPOTHESIS.md` and `BUILD_LOG.md` changed;
    - `tools/hash_tree.py --check HYPOTHESIS.md` passes (all 20 scopes match the frozen table), and `tools/hash_tree.py --tree` recomputes the gate record's `code_hashes` for `port`, `runner_tree`, `gate_code` and `reference_tree` exactly;
    - `g1_synthetic.EXCLUDE` and the test's `TIMING` are as read here;
    - HYPOTHESIS.md at `CHOICE_COMMIT` records Andrei's choice (header).
  - **Package:** `PLAN.md` and `diag_resume.py` are committed, unchanged in the working tree, and contained in the upstream branch (pushed), read from the local ref without network access. Each output records their sha256 (`package_sha256`).
  - **Runtime:**
    - both hosts: mlx 0.32.3, mlx-metal 0.32.3 and mlx-lm 0.32.0;
    - the mbp: macOS 27.0 (26A428) and an `applegpu_g17*` GPU (P2's pins), and not the mini's model identifier;
    - the mini: model identifier `Mac16,11` and not an `applegpu_g17*` GPU. Neither host's data can be filed under the other's label;
    - `MLX_ENABLE_TF32=0`, and `EXP036_PORT_FILE` unset.
- **Hosts.** The mbp is the run host, where G1 failed, and its results govern. The mini's results are descriptive, except through rule 8 (section 5).
- **Registered N.** No run stops early, and the script refuses any other N into `out/`.

  | | mbp | mini |
  |---|---|---|
  | (iv) cache-state probe | 1 | 1 |
  | (i) isolated | 50 | 50 |
  | (i′) module | 30 | — |
  | (i″) batch-flip + module | 30 | — |
  | (ii) G1 context | 4 | 1 |
  | (iii) in-process | 50 | 50 |

  (i′) and (i″) run on the mbp only: the mini's results could select no rule through them.
- **One official output per host and experiment.**
  - The script refuses an official run when `out/` already holds a complete output for that host and experiment.
  - After a crash (an incomplete output), one unchanged repeat is allowed. A further repeat needs Andrei's decision, recorded with its UTC and label and passed as `--repeat-approved '<UTC> <label>'`; the output records it and lists the earlier outputs.
  - Every file written to `out/` is committed, partial ones included.
  - `--classify` refuses more than one complete output per host and experiment, and refuses outputs whose `package_sha256` differ from each other or from the committed package files.
- **Invalid runs.** A run whose junit lacks the target test, in which the target errors during setup or is skipped, or whose files differ although the test passed, is not data. It is replaced within the same output until N valid runs exist, with at most N + max(3, ⌊N/5⌋) attempts in all (for N = 50: 60; 30: 36; 4: 7; 1: 4; L2's 100: 120). If the cap is reached first, the output is incomplete and counts as a crashed run. Invalid runs are reported with their reason.
- **Outputs.**
  - Each run writes `out/<host>_<experiment>_<UTC>.json`. The file is rewritten after every run or iteration, with `"complete": false` until the last.
  - Paths are written as `$EXP036_*`, `<kit>`, `<work>`, `<basetemp>`, `<tmpdir>` or `~`; the user and host names as `<name>`. The script refuses to write a file that still contains either.
  - A full copy always goes to the work directory. The `out/` copy is compacted only if it would exceed 4.5 MB (the leak check allows 5 MB). Compaction keeps every value for the first 10 differing runs, and for the later ones the field names, `completion_ids_equal` and the first divergence. The compacted copy records the full copy's sha256.
  - `--out` is this folder's `out/` or a directory outside the repository; any other path inside the repository is refused, for the experiments and for the classification.
- **Work files.** They go to `--work`, outside the repository, and are never committed or deleted:
  - junit XML, pytest logs, the full outputs, and a copy of each pytest run's two JSONL files (`records_<k>/ref.jsonl`, `records_<k>/resumed.jsonl`);
  - for (iii), (iv) and L2: the tiny checkpoint and every JSONL file.
  - mbp default: `$EXP036_WORK/exp037/diag_g1_resume_20261007/`; mini default: `~/models/exp037-mini/diag_g1_resume_20261007/`.

  The pytest arms' `tmp_path` trees, with their tiny checkpoints, stay under pytest's `--basetemp` in the system temp directory, pytest's own default location, which macOS may purge. That keeps every test's `tmp_path` out of `$HOME` and every `$EXP036_*` / `$EXP037_*` path; only the JSONL copies above are kept.
- **Crashes.** The partial output of a crashed run stays, is committed, and its runs count. The experiment is then repeated unchanged once. If it crashes again, Andrei decides between another repeat and stop and publish.
- **Order.**
  - Across hosts: the mbp first, then the mini.
  - On each host: (iv), (i), (i′), (i″), (ii), (iii).
  - Andrei runs each command in his terminal under `caffeinate -i`, on AC power, with no other heavy job running.
  - No power log is kept: these are not registered steps.

### 3.1 (i) Isolated: the single test in N = 50 separate pytest processes

Each run is:

```
$PY -m pytest -vv -rs -p no:cacheprovider --junitxml=<work>/isolated_<UTC>/junit_<k>.xml \
    --basetemp=<tmpdir>/exp037_g1diag_<UTC>/isolated_<k> \
    <kit>/tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records
```

It runs with cwd the kit and G1's environment (`g1_synthetic.run`, lines 39–45). That is the caller's environment plus `EXP036_GATE_G1=1`, `PYTHONDONTWRITEBYTECODE=1`, `EXP036_REQUIRE_ALL=run-host`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Per run the script records:
- the test's outcome (passed, failed in the call phase, setup error, skipped) and failing line;
- the full `-vv` assertion text, from the junit;
- an itemised diff of the test's own `tmp_path` files (`ref.jsonl` and `raw/S2/K8/mmlu_en_high.jsonl`), read through a transcription of the test's `records()` and `content()` and of `runner/jsonl.py`'s `scan` and `first_records`; the test file and `runner/jsonl.py` are pinned by sha256. For every differing item it gives every differing field with both values, whether completion_ids are equal, and the index of the first divergence.

### 3.2 (i′) Module: the whole test module in N = 30 separate pytest processes (mbp)

```
$PY -m pytest -q -rs -p no:cacheprovider --junitxml=<work>/module_<UTC>/junit_<k>.xml \
    -o verbosity_assertions=2 --basetemp=<tmpdir>/exp037_g1diag_<UTC>/module_<k> \
    <kit>/tests/test_runner_tiny.py
```

G1's flags and environment, with (ii)'s two reporting additions. In G1, nine tests run on the module-scoped `model` object before the target: B = 4, 2, 1 and 8 runs, temperature-1 sampling and `on_step` probes. This arm replays that history in a fresh process. Per run: as (i) for the target, plus every other failing test (descriptive).

### 3.3 (i″) Batch-flip + module: N = 30 separate pytest processes (mbp)

```
$PY -m pytest -q -rs -p no:cacheprovider --junitxml=<work>/pair_<UTC>/junit_<k>.xml \
    -o verbosity_assertions=2 --basetemp=<tmpdir>/exp037_g1diag_<UTC>/pair_<k> \
    <kit>/tests/test_bench_batch_flip.py <kit>/tests/test_runner_tiny.py
```

The two files in G1's order. `tests/test_bench_batch_flip.py` checks a real Kolibri tokenizer through scorers (section 2, Q6). Per run: as (i′).

### 3.4 (ii) G1 context: G1's own selection and invocation, K = 4 runs on the mbp and K = 1 on the mini

This is the command of `gate/checks/g1_synthetic.run` (lines 36–38) with its environment (lines 39–45, `require_all=True`), exactly, under the same interpreter and cwd. Two arguments are added after `--junitxml`; both change only reporting and paths:

```
$PY -m pytest -q -rs -p no:cacheprovider --junitxml=<work>/g1ctx_<UTC>/junit_<k>.xml \
    -o verbosity_assertions=2 --basetemp=<tmpdir>/exp037_g1diag_<UTC>/g1ctx_<k> \
    --ignore=<kit>/tests/test_gate_drivers_tiny.py --ignore=<kit>/tests/test_gate_end_to_end_tiny.py \
    --ignore=<kit>/tests/test_gate_ftiny.py <kit>/tests
```

- **`-o verbosity_assertions=2`** gives `-vv`'s untruncated assertion diff to every test while keeping G1's `-q` reporting. pytest cannot apply `-vv` to a single test.
- **`--basetemp`** keeps the `tmp_path` files where the script can read them.

Per run the script records the same as (i) for the target test, plus G1's counts and every other failing test (descriptive).

**Known differences from the gate's G1:**
- the gate ran G1 right after phase 0 (G0k on real shapes, a warm GPU) with the power logger running;
- in the gate, G1's parent was the gate's phase-1 process, which had just run G0 static in-process (`gate/run_gate.py` lines 608–654: the census, lazy strict loads of the BF16 source and both arms, template and tokenizer parity);
- the `tmp_path` root differs: `--basetemp` here, `pytest-of-<user>/pytest-<N>` in the gate;
- the two reporting arguments above.

The command, the environment (`EXP036_GATE_G1`, `PYTHONDONTWRITEBYTECODE`, `EXP036_REQUIRE_ALL=run-host`, the Hub offline, `MLX_ENABLE_TF32=0` through `ensure_exact_fp32`), the cwd, the interpreter, output capture and the timeout match.

### 3.5 (iii) In-process A/C/B: one process, N = 50 iterations, on both hosts

**Setup, in conftest's order:**
1. `MLX_ENABLE_TF32=0`, `sys.path` (`tests/` first), the Hub offline, and the TF32 probe (`tools/precision.ensure_exact_fp32`, as `pytest_sessionstart` runs it).
2. The test module, imported unchanged, so `items`, `run`, `records`, `content`, `StubTok`, `GREEDY` and `EOS` are the test's own.
3. The tiny vendor checkpoint, written as the session fixture writes it: `tiny_checkpoint.write_tiny_checkpoint(dir, seed=0, preset="vendor", copy_port=PORT_MODEL_FILE.exists())`, as `tests/conftest.py` `_write` calls it.
4. The model, loaded as the module fixture loads it: `port_harness.load_port(dir, float32=True)`.

**Each iteration** runs in a fresh directory, in this order:
- **A**, uninterrupted: `run(model, ref, its, B=1, cap=20)`, the test's line 204.
- **C**: the test's lines 206–221, step by step: stop after three records, the torn fourth line, `completed_keys`, the resume with `done`. Each assertion (lines 213, 217, 219, 221, 223) is recorded, by line, instead of raised; line 221 indexes `r["type"]` as the test does.
- **B**, a second uninterrupted run, like A.

`its = items(6, max_tokens = 8 + 2k)` is created once per iteration and shared by A, C and B, as the test shares it between ref and p. After each iteration the size of `scorers.reasoning._CHECKED` is recorded (descriptive; it stays 0 unless something passes the check).

**Comparisons**, each made exactly as line 222 makes it, through `content()`:
- the uninterrupted pairs: A vs B, A vs A of iteration 0, and B vs A of iteration 0;
- A vs C, the test's own pair;
- B vs C, descriptive only.

For every differing item the script records every differing field with both values (completion_ids, text, reasoning_text, answer_text, finish_reason, stop_token, reasoning_status, truncated, the token counts, split_by and so on), whether completion_ids are equal, and the first divergence.

### 3.6 (iv) The tokenizer-check cache probe: one process, on both hosts

It tests the path of section 2, Q6, directly. Setup as (iii) steps 1–2; then, in this order:

1. **The batch-flip test's calls.** Load the Kolibri tokenizer as `tests/test_bench_batch_flip.py` does (`mlx_lm.tokenizer_utils.load(test_bench_support.kolibri_tok_dir())`), then make the test's two extractions through `bench.batch_flip.default_extract(tok)`. Record the keys this adds to `scorers.reasoning._CHECKED`, the two classes' layouts (`__basicsize__`, `__dictoffset__`, `__slots__`), then drop the tokenizer, run `gc.collect()` and record through a weak reference that it was freed.
2. **The model.** Write the tiny checkpoint and load the port as (iii) steps 3–4, as G1 loads the module fixture after the batch-flip test.
3. **Phase A.** Create up to 20,000 `StubTok()` objects one at a time, each dropped before the next (as the test's `run()` creates them). Stop at the first whose id makes `("kolibri", id(stub))` a key of `_CHECKED`.
4. **Phase B** (only if A found none). Create up to 1,000,000 `StubTok()` objects and hold them, so that every free block of that size is handed out; stop at the first hit. Then release them.
5. **Control.** The failing test's scenario (reference run, stopped run, torn tail, resumed run, line-222 comparison) with three distinct `StubTok` objects that are not in the cache. Expected: equal.
6. **Hit** (if phase A or B found one). Record whether `runner.generate._scorer_split("kolibri", stub)` returns a split for it, then run the scenario with that stub as the resumed run's tokenizer (the reference and stopped runs keep the control's stubs).
7. **Seeded, always** (symbolic). Add `("kolibri", id(s))` to `_CHECKED` for the control's third stub, record `_scorer_split`'s result, run the scenario with `s` as the resumed run's tokenizer, then remove the key.

The scenario runs through the test's own `run()`, which receives the given object where it would call `StubTok()` (line 79); nothing else differs.

**Outcome (mechanical):**
- **hit:** step 1 added a key and freed the tokenizer, the control is equal, the seeded run is a field-only difference with `split_by` among its fields, and a stub from phase A or B got the scorers' split and gave a field-only difference with `split_by` among its fields.
- **stale_only:** as hit, except that no stub landed on a stale id.
- **none:** anything else (no key added, the tokenizer not freed, the control differs, or the seeded run does not change `split_by`).

The probe shows what the kit does in a process with the batch-flip test's history. Whether a given process reuses the address depends on its allocation history, so "stale_only" does not show that G1 did not.

### 3.7 Top-2 margins (descriptive)

The decode path's exact log-probabilities cannot be had without instrumenting the generation loop inside `run_cell`. A hook there would keep arrays alive inside the loop and change MLX's buffer reuse, which could create or hide the effect. No kit code is changed to get them.

Instead, after the 50 iterations, (iii) records teacher-forced margins: `runner.generate.teacher_force_logprobs`, unchanged, run as one forward pass over prompt + completion. For each generated step it records the top-1 and top-2 ids and the top-1 − top-2 log-prob margin:
- (a) for q003 and q004 along A of iteration 0, always;
- (b) for the first item whose completion_ids differ, along both variants.

They show how close the path runs to a tie. They are not the decode path's exact values and they decide nothing. The output is written complete before they are computed; an error while computing them is recorded as `margins_error` and leaves the output complete.

### 3.8 Commands and expected durations

**On the mbp**, each block is preceded by RUNBOOK's header. Claude may run M0 (seconds); Andrei runs M1–M6 in his terminal.

```bash
# M0  pull, then the selftest (stub models only)
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && \
"$PY" diagnostics/g1_resume_20261007/diag_resume.py --selftest --work "$EXP036_WORK/exp037/diag_g1_resume_20261007/selftest"

# M1-M6  one block per experiment, in this order: cachestate, isolated, module, pair, g1ctx, inproc
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" diagnostics/g1_resume_20261007/diag_resume.py --host mbp --experiment cachestate
#   then the same with --experiment isolated, module, pair, g1ctx, inproc (N is the registered one by default)
```

As in the gate, `EXP036_TOK` is not set here; the tests and (iv) fall back to `$EXP036_MODELS/Kolibri-1-BF16`.

**On the mini**, over SSH from the mbp, in a fresh terminal. Andrei runs N1–N4; Claude may run N0.

```bash
# N0  pull, then the selftest
cd ~/REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate && git -C ~/REPOS/local-first-ai pull --ff-only && \
EXP036_MODELS="$HOME/models/exp036-mini" EXP036_DATA="$HOME/models/exp036-mini/data" \
EXP036_TOK="$HOME/models/exp036-mini/kolibri/Kolibri-1-BF16" EXP036_REQUIRE_ALL=1 PYTHONDONTWRITEBYTECODE=1 \
"$HOME/models/exp037-mini/venv312/bin/python" diagnostics/g1_resume_20261007/diag_resume.py --selftest \
  --work "$HOME/models/exp037-mini/diag_g1_resume_20261007/selftest"

# N1-N4  one block per experiment, in this order: cachestate, isolated, g1ctx, inproc
cd ~/REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate && \
EXP036_MODELS="$HOME/models/exp036-mini" EXP036_DATA="$HOME/models/exp036-mini/data" \
EXP036_TOK="$HOME/models/exp036-mini/kolibri/Kolibri-1-BF16" EXP036_REQUIRE_ALL=1 PYTHONDONTWRITEBYTECODE=1 \
caffeinate -i "$HOME/models/exp037-mini/venv312/bin/python" diagnostics/g1_resume_20261007/diag_resume.py \
  --host mini --experiment cachestate
#   then the same with --experiment isolated, g1ctx, inproc
```

On the mini, the pytest arms run under `EXP036_REQUIRE_ALL=run-host`, because G1 sets it. The 8 tests that are build-host only can run there, since the mini holds their files.

**Durations.** These are estimates, not measurements; the script records the actual wall time.

| Step | mbp | mini |
|---|---|---|
| selftest | < 1 min | < 1 min |
| (iv) | 1–3 min (tokenizer, model, up to 10⁶ allocations, three scenario runs) | 1–3 min |
| (i), 50 processes | 5–15 min (each loads MLX, writes the tiny checkpoint, loads the port and runs 12 tiny generations) | 6–20 min |
| (i′), 30 processes | 10–30 min | — |
| (i″), 30 processes | 20–60 min (adds the batch-flip file's tiny bench cells and the real tokenizer) | — |
| (ii) | ≈ 26 min for 4 runs (G1 took 385 s in the gate) | 10–20 min for 1 run |
| (iii), 50 iterations | 3–10 min, plus the model load and margins (seconds) | 3–12 min |
| Total | ≈ 1.1–2.5 h | ≈ 20–55 min |
| L2, only if rule 5 applies | 15–30 min for 100 processes | — |

Disk: the pytest arms keep their tiny checkpoints (about 25 MB each) in the temp directory: about 1.3 GB for (i), 0.75 GB each for (i′) and (i″). (ii) keeps whatever G1's own tests write to their `tmp_path` (not measured; the same amount pytest's default location would hold). (iii) and (iv) add one checkpoint each and a few MB of JSONL.

## 4. Definitions

- **Differing pair.** Two record sets that the test's own `content()` comparison finds unequal:
  - in (iii): an uninterrupted pair or A–C;
  - in the pytest arms: the test's ref file against its resumed file;
  - in (iv): the scenario's reference file against its resumed file.
- **Generation difference.** A differing pair in which completion_ids differ for at least one item, or an item is missing on one side.
- **Field-only difference.** A differing pair in which completion_ids are identical for every item and some other non-TIMING field differs.
- **C-assertion failure.** One of the test's other assertions on the stop / torn tail / resume path is false: lines 213, 217, 219, 221 or 223. In (iii) it is recorded by line; in a pytest arm it is a failure of the target at one of those lines.
- **Other failure.** In a pytest arm, the target fails in the call phase at line 222 while its files cannot be read or compare equal, or at a line not listed above (for example an exception inside `run_cell`). It routes like a generation difference.
- **Invalid run.** Not data (3.0): the junit lacks the target, the target errors during setup or is skipped, or its files differ although it passed. A teardown error after a passing call is recorded and the run keeps the kind its files give.
- **Reproduces.** Any single one of these is a reproduction; rates are descriptive only, except where a rule below uses one:
  - a differing pair;
  - a C-assertion failure;
  - a failing run of the target test in a pytest arm.
- **Fields that depend on process state.** The record schema defines every field outside TIMING from the item, the cell, completion_ids and the tokenizer object passed to `run_cell`. Two groups also depend on process state:
  - **split fields:** `split_by`, `reasoning_tokens`, `answer_tokens`, `reasoning_text`, `answer_text`, `reasoning_status`. `_scorer_split` chooses the scorers' split or the runner's from `check_tokenizer`'s result, which reads `scorers.reasoning._CHECKED` (section 2, Q6);
  - **mlx-lm fields:** `finish_reason`, `truncated`, `stop_token`, from the finish reason that mlx-lm's `BatchGenerator` reports.

  So a field-only difference does not by itself show a runner defect. It routes by where the state that produced it lives (the scope classes below). The TIMING fields are excluded by the test, so no field-only difference can be a legitimate time or scheduling measurement.
- **The generation path of `runner/`** (HYPOTHESIS "Gate fix"), decided now. It is the runner code that `run_cell` and the failing test's scenario execute:
  - `runner/generate.py`: the whole file (`run_cell`, `make_batch_generator`, `_make_record`, `_scorer_split`, `split_ids`, `special_ids`, `_decode`, `_hashed`, `_padded_len`, `StepLog`, the heartbeat functions);
  - `runner/sampler.py`: the whole file (`make_vllm_sampler`, `fp32_logits`);
  - `runner/chat.py`: `render`, `prompt_prefill`, `sampling_for`, `hf_tokenizer` and the helpers they call;
  - `runner/common.py`: the helpers those functions call (`sha256_text`, `ids_sha256`, `cell_id`, `utc_iso`, `power_mode`, `thermal_state`);
  - `runner/jsonl.py`: `Writer`, `open_run` and its torn-tail helpers, `scan`, `read_jsonl`, `record_key`, `is_complete`, `first_records`, `cell_files` and `completed_keys`. They count because they decide which items a resumed cell generates and they write every record.

  The rest of `runner/` (`run.py`, `guard.py`, `memory.py`, `plan_fix.py`, `plan_rules.json`, `seeds.py`, `simulate.py`, `templates/`) is not in the generation path.
- **Scope classes** (where the producing code or state lives), and their consequence; every follow-up ends in one of them:
  - **A. Port, generation path or gate measurement code** (`port/`, the list above, `gate/` outside `gate_rules`): **gate fix there** (6.8).
  - **B. State written by kit code outside that scope** (`scorers/`, `bench/`, `tools/`, `analysis/`, `tasks/`, `reference/`) **and read by a generation-path function:** **gate fix at the reading site**, so that the generation path no longer depends on that state (6.8). Decided now: such a runner-side guard is a change in the generation path.
  - **C. Kit code outside that scope, not read through the generation path:** stop and publish.
  - **D. Inside MLX or mlx-lm:** class M; a gate fix only through 6.5's listed remedies, else stop and publish (G1 keeps its failure; the facts go to Andrei for an upstream report that he writes himself).
  - **E. Test code only** (fixtures, conftest, test-module globals, monkeypatch residues): stop and publish. No permitted fix exists: a tests-only change does not satisfy P1(d), and G1 keeps its failure.
  - **F. Not localised:** stop and publish, because no failing test can be written.
- **Governing host.** The mbp. The mini's results are descriptive and published beside the mbp's; they select a rule only through rule 8.

## 5. The outcome-to-action table (decided now, applied mechanically)

`diag_resume.py --classify` applies the table once each host has exactly one complete output with the registered N for each of its experiments (3.0). Every recorded valid run and iteration counts, the partial outputs of crashed runs included.

**Every condition stands alone and is read on the mbp's data.** Every rule that matches is reported and pursued. Field-only and C-assertion reproductions go to rules 2 and 4 in whatever arm they occur; rules 5 and 6 take only generation differences and other failures.

| Rule | Outcome | Condition (mbp) | Consequence |
|---|---|---|---|
| 1 | **O1** | In (iii), an uninterrupted pair is a generation difference | Generation is non-deterministic within one process. **L1** (6.1) on that pair; its class decides through the scope classes |
| 2 | **O4** | Some reproduction, in any arm, is a field-only difference | Let F be the union of the differing fields and A the arms where they occurred.<br>• **O4-split:** F contains only split fields, `split_by` is among them, and (iv) is "hit" or "stale_only": the state of section 2, Q6 exists on the run host and produces exactly this change. **Gate fix 6.6** (class B).<br>• **O4-trace:** otherwise, if A includes (i) or (iii): **L1f** (6.4) in that setting (in-process if (iii), fresh processes if only (i)).<br>• **O4-order:** otherwise (context arms only): **L3** (6.3) on the smallest selection in A: (i′), then (i″), then (ii) |
| 3 | **O2** | In (iii), an A–C pair is a generation difference | The resume path changes generation while uninterrupted runs agree. **L1** on the A–C pair; its class decides |
| 4 | **O2c** | A C-assertion failure, in any arm | • **O2c-line:** if it occurred in (iii) or (i): **gate fix 6.7** in the producing code of that line (all in the generation path), with the failing test's scenario as its test.<br>• **O2c-order:** otherwise: **L3** on the smallest selection where it occurred |
| 5 | **O6** | In (i), a run reproduces as a generation difference or another failure | **L2** (6.2), then its result:<br>• a generation difference in an uninterrupted pair: L1 in fresh processes (O6-O1);<br>• an A–C generation difference: L1 in fresh processes (O6-O2);<br>• a C-assertion failure: gate fix 6.7 (O6-O2c);<br>• a field-only difference: O4-split's consequence if its conditions hold (O6-O4-split), else L1f in fresh processes (O6-O4);<br>• nothing in 100 processes: **stop and publish** (O6-stop), because the trigger is in the pytest harness or test code, not in kit runtime code |
| 6 | **O3** | In (i′), (i″) or (ii), a run reproduces as a generation difference or another failure | The failure needs state left by other tests. **L3** on the smallest selection that reproduced |
| 7 | **O7** | (iv) is "hit" | A StubTok created after the batch-flip test's calls lands on a stale cache id and gets the scorers' split in the failing test's own scenario: state written by `scorers/` (reached through `bench/`) and read by the generation path. **Gate fix 6.6** (class B). The gate's visible output (equal `answer_text`, another field differing) is consistent with it |
| 8 | mini fallback | No rule 1–7 matches on the mbp; on the mini, (iii) shows an uninterrupted generation difference (**O1-mini**) or (iv) is "hit" (**O7-mini**) | As O1 (L1 on the mini) or O7. Two uninterrupted greedy runs that differ in one process, or the cache state of 3.6, show a defect in the code G1 runs on both hosts. Every fix criterion must then also hold on the mbp (6.5, 6.8) |
| 9 | **O5** | Nothing above matches | **Stop and publish.** G1 keeps its failure. The flakiness is disclosed with the observed counts on both hosts and the detection power (section 8). No code change exists, so P1(d) refuses a re-run |

**Combining matched rules.**
- If any pursued rule ends in stop and publish, now or after its follow-up, the result is **stop and publish**, and every defect found is published with it. A re-run is justified only when every reproduced failure mode has a qualifying fix; otherwise G1 would be re-run on a known unfixed failure.
- Otherwise, once every follow-up has ended, the result is **one gate-fix amendment** that lists every fix (HYPOTHESIS: "One amendment may list several fixes"). A fix named by two rules (6.6, by rules 2 and 7) is listed once.
- While a follow-up is pending, `--classify` reports "follow-up pending" with each follow-up named; each runs only after its code is committed and pushed, then the table is applied again.
- `--classify` also reports the mini's matching rules (descriptive) and `mini_reproduced` with every outcome.

- **Stop and publish** follows HYPOTHESIS "End of the gate". The gate failure is exp_037's result: K8, check G1, this test, line 222. The diagnostics are published with it. No speed, memory or quality number from an ungated port is reported.
- **A gate fix** follows 6.8. Writing it, and any re-run, needs Andrei's go. Stop and publish stays available to Andrei at every point.

## 6. Follow-up diagnostics (rules fixed now)

Each follow-up runs only when the table sends to it. Its code implements the procedure below and adds no decision. The code is committed and pushed before it runs, in new files in this folder: `PLAN.md` and `diag_resume.py` are not changed after the pre-registration commit (L2's code is already in `diag_resume.py`). Like the experiments in section 3, a follow-up writes no gate record, is not a cycle and replaces no gate value. Its outputs go to `out/` as `<host>_<follow-up>_<UTC>.json`, under 3.0's rules for outputs, invalid runs and crashes.

Every capture below runs in the diagnostic process only; no kit file changes. Every follow-up that captures first measures its trigger's rate **without** capture in the same setting, because holding references can change MLX's buffer reuse (3.7).

### 6.1 L1: localisation of a generation difference

L1 runs on the host and setting named by its rule, on the triggering pair type: uninterrupted for O1, A–C for O2.

**L1a phase 0, uncaptured.** Repeat (iii)'s iteration (A, C, B, same layout) up to 200 times in one process, or in up to 200 fresh processes (one iteration each) when the trigger came through L2. Stop after 5 differing pairs of the triggering type. The rate is p̂ = differing pairs / repetitions. If none in 200: class **N**.

**L1a phase 1, captured.** The same, up to 200 times, stopping after 5 differing pairs, with two passive captures:
1. **The decode path's log-probabilities.** The diagnostic process patches `runner.sampler.make_vllm_sampler` to wrap the sampler that `run_cell` builds. The wrapper keeps a reference to the lazy log-prob array it receives and returns the original sampler's output; it adds no operation and forces no evaluation.
2. **The runner's inputs.** The prompt ids and `max_tokens` that the runner passes to `BatchGenerator.insert`.

For each differing pair it records the first item; the first sampler call whose log-probs differ bitwise, with its token index and both variants' top-2 values; and whether `insert`'s inputs differed. **If phase 0 found a difference and phase 1 finds none in 200, the capture suppresses the effect: stop and publish, disclosed as such.**

**L1b, trace.** Repeat the same pair up to 200 times, stopping after 3 differing captures. At the call found in L1a, keep references to every tensor the port computes, then hash each one after the run:
- per decoder layer: the input, the attention branch output, the cache keys/values and offsets the call reads, the router logits, logits + expert_bias, the selected ids, the routed experts' output, the shared expert's output and the layer output;
- then the final norm, the head's logits, BatchGenerator's log-softmax and the sampled id.

The wrappers go on the loaded model's classes in the diagnostic process, as `tests/port_harness.py`'s `record_*` helpers do. The locus is the first tensor, in execution order, whose sha256 differs between the two variants. **If L1b finds no differing capture in 200, the trace suppresses the effect: stop and publish, disclosed.**

**Classes (mechanical), mapped to the scope classes of section 4:**
- **K-in.** The first difference is in what enters the call (the token id, the cache contents or offsets read, the mask), and every tensor of the previous call agreed. State carried between calls differs:
  - written by kit code in port/ or the generation path (`runner/generate.py`, the port's `make_cache` / `SlidingKVCache`): class A, **gate fix there**;
  - written inside mlx-lm (BatchGenerator; cache merge, extend or filter) by the same kit calls in both variants: class D (**M**).
- **K-sel.** The first difference is a selection made by kit code from bitwise-identical inputs (a tie or an unstable selection): the port's route ids (`mx.argpartition`) or the runner's greedy `mx.argmax`. Class A, **gate fix in that call:** a deterministic selection, with the reference's rule as the test's oracle (`reference/` route_ref's selection for the router; the first index of the maximum, as numpy's argmax, for the sampler).
- **M.** The first difference is the output of an MLX or mlx-lm op whose inputs hash equal: a matmul, gather_mm in SwitchGLU, scaled_dot_product_attention, rope, rms_norm, softmax or logsumexp, or a reduction. **Replay:** the op is re-run 1,000 times on the saved inputs in a fresh process, recording whether two or more distinct outputs appear. The class is M whether or not the replay confirms it, since a race may show only in context. **Next:** 6.5; if no listed remedy qualifies, stop and publish.
- **N.** No differing capture: **stop and publish** (class F).

### 6.2 L2: fresh processes (for O6)

Command: `--experiment fresh --n 100` on the mbp, under the RUNBOOK header and `caffeinate -i`. It runs (iii) with one iteration per process, in 100 processes, each loading the model as the fixture does. Its outcome rule is rule 5's (section 5); several results can hold at once, and each is pursued. N = 100 rather than 50 because a rate that (i) found low must not be missed by chance (section 8).

### 6.3 L3: order bisection (for O2c-order, O3 and O4-order)

L3 runs on the mbp, with G1's flags and environment, on the smallest selection that reproduced: (i′), else (i″), else (ii)'s full G1 selection. Each run keeps the whole selection collected, so every test module is imported as in G1; tests are dropped with `--deselect`, never by listing node ids. It can take hours; Andrei runs it.

1. **Collect.** List the selection's collection order with `pytest --collect-only -q`. P is the node ids collected before the target.
2. **Reproduce.** Run the selection unchanged up to 20 times, stopping after 3 target failures of the triggering kind. The rate is p̂ = failures / runs. If none in 20: **stop and publish**, because it cannot be reproduced in a testable way. R is the smallest integer with (1 − p̂)^R ≤ 0.05, capped at 60 (with p̂ ≥ 1/20, R ≤ 59).
3. **Bisect.**
   - Split P in halves and try first the half nearer the target, deselecting the other half. A candidate reproduces if the target fails at least once in R runs.
   - Recurse into the half that reproduces. If neither half reproduces alone, keep both and stop at that granularity.
   - Files first, then the test functions within them, the same way. The result is the minimal preceding set S.
4. **Name the state.** Compare the process after S with a process that ran only the target's module setup, on:
   - the module globals of every kit package (`runner/`, `port/`, `tools/`, `gate/`, `scorers/`, `bench/`, `analysis/`, `tasks/`, `reference/`), by a hash of each global's repr (sets and dicts by their sorted items);
   - kit functions replaced and not restored;
   - `os.environ`;
   - the MLX settings its API exposes: cache limit, wired limit, default device and stream, random state;
   - the module-scoped `model` fixture object: its non-parameter attributes and any cached arrays, hashed, against a model freshly loaded from the same checkpoint.

   For each difference, run S + target with that state reset just before the target, R times. A reset after which the target never fails names the state.
5. **Classify** each named state by the scope classes: who writes it and who reads it.
   - Written by port/, the generation path or gate measurement code: class A, **gate fix there.** The kit must neither leak that state nor depend on it. The test: S's kit calls, then the target scenario, with records equal (6.8).
   - Written by kit code outside that scope and read by a generation-path function (for example `scorers.reasoning._CHECKED`, written through `bench/` and read by `runner.generate._scorer_split`): class B, **gate fix at the reading site** (6.6 if it is that cache).
   - Written by kit code outside the scope and not read through the generation path: class C, stop and publish.
   - Set only by test code: class E, stop and publish.
   - No reset names the state (MLX-internal state such as the buffer or kernel caches): class D. 6.5 needs a localised kit call, which L3 does not give, so **stop and publish**.
   - **Several named states:** each is classified. If every one is in class A or B, one amendment fixes them all. If any is in class C, D or E, the result is stop and publish (section 5, combining).

### 6.4 L1f: field trace (for O4-trace and O6-O4)

L1f localises a field-only difference without holding any MLX array. It runs in the setting its rule names: in one process as (iii) if the difference appeared in (iii), else in fresh processes, one iteration each, as L2.

1. **Phase 0, uncaptured.** Repeat the setting up to 200 times, stopping after 3 field-only pairs of the triggering type. The rate is p̂. If none in 200: **stop and publish** (class F).
2. **Phase 1, captured.** The same, up to 200 times, stopping after 3, with `runner.generate._make_record` and `runner.generate._scorer_split` wrapped in the diagnostic process. Per record the wrapper keeps Python values only: whether the run's scorer split was None, whether `(family, id(tokenizer))` was in `scorers.reasoning._CHECKED`, `open_id`, `close_id`, the prompt prefill, `finish_reason`, the completion ids and the EOS ids. If phase 0 found a difference and phase 1 finds none, the capture suppresses the effect: stop and publish, disclosed.
3. **Locus.** The first captured input, in the order listed, that differs between the two variants of a differing item:
   - the scorer split's choice: the state it read. If that is the `_CHECKED` membership, class B, **gate fix 6.6**; otherwise L3's step 4 inventory, run in this setting, names the state, and its class decides;
   - `open_id` / `close_id`: `runner.generate.special_ids` or `runner.chat.hf_tokenizer`, class A;
   - the prompt prefill: `runner.chat.prompt_prefill`, class A;
   - `finish_reason` with equal completion ids: mlx-lm's BatchGenerator, class D; 6.5 lists no remedy for a finish reason, so **stop and publish**;
   - no captured input differs: the field is produced inside `_make_record` from equal inputs, class A, a gate fix in `_make_record`.

### 6.5 What counts as a kit-side remedy for class M

The remedy is a change confined to port/, the generation path, or gate measurement code, at the localised kit call and nowhere else. Only the candidates listed here are tried, in this order, and the first that meets all four criteria below is taken; nothing else is searched.

| Op at the localised call | Candidates, in order |
|---|---|
| matmul (`@`, `mx.matmul`, a Linear) | 1. (r2) `mx.contiguous` copies of both operands. 2. (r1) the same contraction through `mx.einsum` |
| `gather_mm` in SwitchGLU | 1. (r2) contiguous copies of the input and index arrays, the sorted order and `sorted_indices` unchanged (G0k judges the sorted kernel class) |
| `scaled_dot_product_attention` | 1. (r2) contiguous q, k, v. 2. (r1) explicit fp32 attention: `softmax(q @ kᵀ · scale + mask, precise=True) @ v` |
| `rope` | 1. (r2) a contiguous input. 2. (r1) the explicit rotation with the same base, scale, dims and layout |
| `rms_norm` | 1. (r2) a contiguous input. 2. (r1) explicit fp32 `x · rsqrt(mean(x², −1) + eps) · w` |
| `softmax` / `logsumexp` | 1. (r2) a contiguous input. 2. (r1) `softmax(…, precise=True)`, or `m + log(sum(exp(x − m)))` with m the maximum |
| a reduction (sum, mean, max) | 1. (r2) a contiguous input |
| inside mlx-lm (BatchGenerator, its caches, its finish reasons) | none: stop and publish |

(r1) replaces the call by another MLX call that computes the same mathematical function; (r2) is a layout change at that call that preserves the dtype.

A candidate qualifies only if all four hold:
1. With the change, L1a's phase-1 capture shows 0 differing pairs in R_fix repetitions on the mbp, where R_fix is the smallest integer with (1 − p̂)^R_fix ≤ 0.01, p̂ is L1a phase 0's rate, and R_fix ≥ 200; and (iii) on the mbp shows 0 differences in 50 iterations. This holds on the mbp also when L1 ran on the mini.
2. It changes no registered parameter or kernel class: B, prefill_batch_size = min(B, 8), prefill_step_size 2,048, max_kv_size None, the fp32 logits, the sampler's definition, the EOS ids, the routing rule, the runtime pins, MLX_ENABLE_TF32 = 0, or the kernel classes G0k judges (sorted gather_qmm at the registered shapes).
3. The whole suite passes on the mini with `EXP036_REQUIRE_ALL=1`, and the port-vs-reference tests are unchanged.
4. It comes with a test that meets 6.8.

If no candidate qualifies, **stop and publish**.

### 6.6 The split-cache fix (O4-split, O6-O4-split, O7, O7-mini, and class B for that cache)

- **Location.** `runner/generate.py` `_scorer_split`, the reading site, in the generation path. Nothing in `scorers/` changes (it is outside the gate-fix scope).
- **Content.** Whether a run uses the scorers' split must not depend on `scorers.reasoning._CHECKED` or on any other state keyed by `id()`. For example, `_scorer_split` compares the tokenizer's delimiter and EOS ids with the family table itself (`scorers.reasoning.token_id` against `family_spec`) on every call. For a tokenizer whose ids match the table, which is every real run, every record field stays as it is.
- **Test** (symbolic oracle, deterministic). In the failing test's own scenario, the resumed run's StubTok is seeded into `_CHECKED` as `("kolibri", id(stub))`. Before the fix, `content(records(p)) != content(records(ref))`, with `split_by` differing on q003–q005; after it, they are equal, and `_scorer_split("kolibri", stub)` is None whatever the cache holds. The oracle, the family table's delimiter ids against the tokenizer's, is independent of every gate value.

### 6.7 C-assertion fixes by line (O2c-line, O6-O2c)

| Failing line | What it checks | Producing code (all in the generation path) |
|---|---|---|
| 213 | the stopped run stopped after three records | `run_cell`'s stop handling; `runner/jsonl.Writer.append` |
| 217 | three completed keys after the torn line | `runner/jsonl.completed_keys`, `cell_files`, `scan`, `first_records` |
| 219 | the resumed run's `n_todo` and `n_done` | `run_cell`'s `done` filtering |
| 221 | one unparsable line, then header and resume | `runner/jsonl.open_run` (torn tail set aside, newline, resume header) and `scan` |
| 223 | all six keys completed after the resume | `runner/jsonl.completed_keys` |

The gate fix is in that code (class A). Its test is the failing test's scenario under 6.8, with p̂ the rate of the arm where the failure occurred.

### 6.8 Every gate fix (HYPOTHESIS "Gate fix", restated, not changed)

- **Form.** The main session types it as `amendments/<k>_gatefix_<UTC>.md` plus code, on Andrei's go. The mbp session appends it to HYPOTHESIS.md with `tools/status.py --sync-amendments`. One amendment lists every fix the table named.
- **Location.** It is a code change in `port/`, in the generation path of `runner/` (section 4), or in gate measurement code (`gate/` outside `gate_rules`).
- **Test.** Its test fails before the fix and passes after it. The oracle is independent of every gate value: a tiny checkpoint, the reference, or a symbolic check. Fixed now, so that nothing is tuned after the results:
  - **In the failing test's own scenario.** Before the fix the test shows `content(records)` unequal in that scenario (the reference run against the resumed run, or an uninterrupted pair), not only on an intermediate tensor; after it, equal.
  - **Flaky trigger.** R_test is the smallest integer with (1 − p̂)^R_test ≤ 0.01, where p̂ is the rate the localising diagnostic measured (L1a phase 0, L1f phase 0, L3 step 2, or for 6.7 the rate of the reproducing arm). Before the fix the test's loop of R_test repetitions finds at least one difference in each of 3 invocations; after it, none in each of 3 invocations, on the mbp.
  - **Deterministic trigger** (a seeded state, as 6.6): one invocation before and one after.
  - **Evidence from the mini** (rule 8, or L1 run on the mini): all of the above also on the mbp, and (iii) on the mbp shows 0 differences in 50 iterations after the fix.
  - The test changes the `tests` scope, so the amendment gives its `hash_tree: TESTS_SHA256` line.
- **Limits.** It never changes a mutant, a control or a threshold.
- **Re-run.** P1(d) admits a re-run only because the port, runner or gate-code hash changed; a tests-only change does not qualify. The re-run:
  - is fix cycle 1 of 2;
  - is the whole gate: RUNBOOK step 1, `--sync-amendments`, step 3, step 8 if the port changed, then step 10 in full;
  - needs Andrei's go.

## 7. What no outcome can do

- **Gate values and records.** No outcome changes a mutant, a control, a threshold, the gate text, `gate_rules` or any gate value. None writes or replaces a gate record. Gate run `20261007T110355Z` and its G1 value stand as recorded.
- **Status.** These diagnostics are not a gate run and not a cycle. Their only permitted consequences are a gate fix, a reference fix (impossible here, see section 2), or stop and publish.
- **Invalid checks.** "A check shown to be invalid keeps its failure." Where a branch finds the test invalid as written, G1's failure is published. A redesigned test belongs to a later experiment.
- **The package.** `PLAN.md` and `diag_resume.py` do not change after the pre-registration commit; `--classify` refuses outputs from any other version.
- **Where outputs go.** Never under `results/`. `diagnostics/` lies outside every hash scope (README.md).

## 8. Detection power and disclosures

**Detection power.** If a reproduction has probability p per run, independently, then N runs miss it with probability (1 − p)^N:

| N | p = 0.01 | p = 0.02 | p = 0.05 | p = 0.10 | p = 0.5 |
|---|---|---|---|---|---|
| 50, (i) and (iii) | 0.61 | 0.36 | 0.077 | 0.005 | ≈ 0 |
| 30, (i′) and (i″) | 0.74 | 0.55 | 0.21 | 0.042 | ≈ 0 |
| 100, L2 | 0.37 | 0.13 | 0.006 | ≈ 0 | ≈ 0 |
| 4, (ii) on the mbp | 0.96 | 0.92 | 0.82 | 0.66 | 0.063 |

(i) and (iii) have no power against the path of section 2, Q6: they never fill the cache. (iv) tests that path directly; its outcome does not depend on N.

O5 publishes this table with the counts. A rate that was low in the gate can pass every experiment by chance, and O5 says so.

**What the gate itself showed.** In G1's context the test failed in 1 run of 1. The same test passed in RUNBOOK step 3 on the mbp, in a different preceding context (the full suite with the heavy tiny tests), and in every run on the mini.

**Disclosures:**
- (iii) and (iv) reproduce the fixture's setup in a plain Python process, not under pytest.
- (ii) differs from the gate's G1 as listed in 3.4.
- (iv) shows what the kit does with a given allocation history; it cannot show what G1's process did.
- The margins are teacher-forced proxies (3.7).

## 9. Order of work

1. **Commit the package.** On Andrei's go, the main session:
   - pulls the mbp session's commit that records Andrei's choice in HYPOTHESIS.md, checks its UTC and label against the header, corrects the header and `CHOICE_UTC` / `CHOICE_LABEL` if they differ, and sets `CHOICE_COMMIT` to that commit;
   - runs the selftest, `py_compile`, the leak check and `tools/hash_tree.py --check HYPOTHESIS.md`;
   - commits this folder (`PLAN.md`, `README.md`, `diag_resume.py`) as `chronos/exp_037: G1 diagnostics package g1_resume_20261007 (pre-registered, before any run)`;
   - pushes it.

   No diagnostic runs before this push; the script refuses to run until the package is committed, clean and in the upstream branch.
2. **The mbp.** Pull. First, before M0 or any run: the mbp session prints the target test's `<failure>` message and text from `$EXP036_WORK/exp037/gate_g1/junit_20261007T110355Z.xml` (home path replaced by `~`), writes it verbatim to `gate1_junit_failure.txt` in this folder, runs the leak check, and commits and pushes it alone as `chronos/exp_037: gate run 1 junit failure text (G1 target test)`. Then M0 (the selftest). Andrei runs M1–M6, in order. The mbp session runs the leak check, commits every `out/mbp_*.json` as `chronos/exp_037: G1 diagnostics outputs (mbp)`, and pushes on Andrei's go.
3. **The mini.** Pull, then N0. Andrei runs N1–N4. The main session runs the leak check, commits every `out/mini_*.json`, and pushes on Andrei's go.
4. **Apply the table.** The main session runs `diag_resume.py --classify --write`, commits `out/classification_<UTC>.json`, and reports the matched rules, the decision and its consequence to Andrei. Then:
   - a follow-up (L1, L1f, L2 or L3) runs only after its code is committed and pushed, and the table is applied again with its output;
   - a gate fix follows 6.8;
   - stop and publish follows HYPOTHESIS "End of the gate".

   Each of Andrei's choices is recorded with its UTC and label, as before.
