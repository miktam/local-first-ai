# Experiment 037 — runbook (MacBook Pro M5 Max)

*For Andrei and the Claude session on the mbp. The steps run in order; none is skipped or reordered. [`HYPOTHESIS.md`](./HYPOTHESIS.md) is the pre-registration and wins over this file; [`BUILD_SPEC.md`](./BUILD_SPEC.md) describes the kit (a delta over exp_036's). Nothing in this runbook runs on the mini except step 18's target and the work after the hand-back. Hand-off is through GitHub (`miktam/local-first-ai`).*

**Who runs what:**
- **Andrei runs in his terminal under `caffeinate -i`** — any job longer than a few minutes, or anything that loads model weights. The mbp Claude session prepares the exact command; Andrei pastes it and keeps the laptop on AC power, lid open, in High Power mode.
- **Claude may run** — short, weight-free steps: git, venv, tests, quick preflight, the manifest verification, plan-fix, scoring, leak check, commit and push.
- **Andrei by hand** — `sudo` lines, the sign-off block (Claude never fills any sign-off field), and every go the pre-registration names.

**Pushes.** If Andrei gave go #1 (c) at the freeze (HYPOTHESIS, "At the freeze"), pushes from the mbp of the sign-off, records, amendments and S2/S3 raw outputs and scores are pre-approved, each after the leak check (steps 4–17). Otherwise every push asks him first. Always Andrei's go: the mini's verdict and results push, any README or scientific_log result push, `deploy.sh`, SFTP, the Hugging Face upload, PRs, social posts, and every gate re-run.

**Every Claude-run block** starts with the header below, because shell variables do not persist between Claude's commands. exp_036's env file refuses to be sourced outside exp_036's directory, hence the `cd` order:
```
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && <command>
```
- exp_036's env file sets `EXP036_MODELS` (default `~/models/exp036`), `EXP036_DATA`, `EXP036_PRIVATE`, `EXP036_WORK`, `LFA`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and exp_036's `PY`. It is never opened, copied or edited.
- `env/exp037.settings.sh` refuses outside this directory, then sets `EXP036_DIR`, `EXP` (this directory), `EXP037_BUILDS` (default `$EXP036_MODELS/exp037-builds`), `EXP037_VENV` (default `~/models/exp037/venv`, beside `$EXP036_MODELS`) and **`PY="$EXP037_VENV/bin/python"`**: the exp_037 venv (MLX 0.32.3 / mlx-lm 0.32.0; decision F2), never exp_036's. It changes no `EXP036_*` value. exp_037's own work and private files go to `$EXP036_WORK/exp037/` and `$EXP036_PRIVATE/exp037/`.
- Andrei's terminal sources the same two files once, in the same order.

**Every commit**:
- uses the prefix `chronos/exp_037:` and identity `Miktam <hello@localfirstai.eu>`;
- is preceded by `"$PY" tools/leak_check.py --range @{u}..HEAD --staged` (a finding blocks it; raw-output warnings go into the commit body);
- ends with the trailer: `git -C "$LFA" commit -m "<subject>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`;
- is followed by `git -C "$LFA" status --porcelain`, which must print nothing outside this directory's `results/`, `aborted/`, `evidence/` and `diagnostics/`.

The pre-push hook (step 2) runs the leak check again on every push.

**HYPOTHESIS.md has one writer during the run.** From the pre-registration push to step 19, only this mbp session appends to HYPOTHESIS.md. The main session on the mini pushes any amendment it writes (gate fix, extractor fix, Tier-2 analysis, process record) as a file `amendments/<k>_<type>_<UTC>.md` plus code, and never edits HYPOTHESIS.md. The mbp session pulls (`git -C "$LFA" pull --ff-only`) before every commit, and `"$PY" tools/status.py --sync-amendments` appends any `amendments/` file not yet in HYPOTHESIS.md, verbatim, in that commit. It refuses, appending nothing, an amendment that still holds a `{{PLACEHOLDER}}`: stop and tell the main session, which fills it and pushes again.

**The freeze rules hold throughout** (HYPOTHESIS, "Fix-cycle, freeze and amendment rules"): no threshold, rule, control or gate-text amendment exists; at most two fix cycles, counted by gate runs; a gate exit 3 is never a cycle; every re-run needs Andrei's go.

**The version freeze.** No macOS, MLX or mlx-lm update from the pre-registration push until S3 ends. P2 binds the gate and every session to macOS 27.0 (26A428), mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 and an `applegpu_g17*` GPU; a session refused for version drift is a pre-registered consequence.

**Never** move or copy a file from `$EXP036_PRIVATE` into the repo, and never delete anything: broken files go to `aborted/<UTC>-<what>/` with a `NOTE.md` (private ones to `$EXP036_PRIVATE/exp037/aborted/`, with only their sha256 and byte count in the repo note).

**Step logs: the pilot's are committed, the sessions' are working files** (decided 2026-10-06, before the freeze). Every decode step adds one JSON line of about 190 B to the cell's per-step decode log `*.steps.jsonl`, beside its outputs. The leak check allows 50 MB per file in `results/raw/`, `results/pilot/` and `aborted/` (5 MB elsewhere); GitHub refuses files over 100 MB.
- **Pilot step logs** (`results/pilot/<UTC>/<arm>/*.steps.jsonl`, and the copies `abort-pilot` moves under `aborted/<UTC>-pilot/`) are committed with the pilot at step 13 (an aborted pilot's copies with the rest of that pilot). The registered plan rule fits its step model a + b·n_live + c·n_live·padded_len on a summarised pilot's (each cell's `steps_file`; HYPOTHESIS, "Pilot and the rule that fixes n and max_tokens", rule 4; `runner/plan_fix.py` `load_steps`), and `runner/run.py` sums the pilot's prefill from them into the summary, so that the plan can be re-derived from the public repository. They are small: about 1 MB a cell at B = 8, at most about 50 MB at B = 1 with every item at the 32k cap. One still over 50 MB stays uncommitted (step 13).
- **Session step logs** (`results/raw/<session>/<arm>/*.steps.jsonl`, and the copies a cell abort moves to `aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl`) are git-ignored working files (`.gitignore`), so no `git add` takes them: about 190 MB for a K8 cell at B = 1, gigabytes over all sessions. They are never committed and never deleted; they stay on the mbp and are copied to the mini at step 18. No score, verdict or registered rule reads them (`scorers/score_all.py` and `analysis/exploratory.py` skip them); on the mbp `runner/run.py` moves them on a cell abort.
- **Both are bound by sha256** in the run-record blocks, which `tools/status.py` builds from the files on disk, git-ignored or not: the plan block (step 13) lists every pilot step log, and the S2 (step 15) and S3 (step 17) blocks every session step log, an aborted cell's copy included.

**Power log (descriptive; Andrei's request, 2026-10-06).** Steps 9, 10, 11, 12, 14 and 16 each run a power logger in a second terminal, so that exp_037's machine energy is measured rather than estimated (HYPOTHESIS, "Power and energy (descriptive)"). Its label is "CPU + GPU + ANE power (powermetrics combined_power, an estimate); not the whole SoC, not wall power": the rest of the SoC, DRAM, the display, the SSD and the adapter are not in it, so it is a loose lower bound of the mbp's energy. The logger adds a small background load (one root sampler at 1 Hz) to every logged step alike; it is not measured or corrected for. It decides nothing. A missing, late or broken power log never blocks, delays or repeats a step.
- **The sudoers rule.** The mbp has a pinned NOPASSWD rule for exactly `/usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist`. Any other argument, `-o` and `-a` included, is refused by design. powermetrics writes to stdout, and Andrei's shell redirects it to a file under `$EXP036_WORK/exp037/power/`. `sudo -n` never asks for a password: without the rule it fails at once and the file stays empty.
- **The smoke, once, before the freeze** (Andrei and the mbp Claude session; not a numbered step; skip it when BUILD_LOG.md already records its layout). It tests the parser on real output while `tools/` can still change. Andrei first checks that the rule matches, without running anything: the line prints the command with its arguments and exits 0; anything else means the rule does not match.
```bash
sudo -n -l /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist
```
  Then he runs the logger for about 60 s and stops it with Ctrl-C:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_smoke_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
  Claude runs `"$PY" tools/power_log.py --check --max-age 0 <the smoke file>` and `"$PY" tools/power_log.py --layout <the smoke file>`. `--layout` prints, per key layout, the key names and plist types of the top level and of the `processor` and `gpu` dicts, with document and sample counts and elapsed times; it prints no other value. The main session records the layout in BUILD_LOG.md and builds the tests' fixture from it (the real key names and types, synthetic values); if poweravg summaries show up, the parser learns to skip them by their layout. All of this happens before the freeze. The smoke file stays on the mbp: no record is written for it, and the tool never takes a `power_smoke_*` file as the newest log.
- **Start it before the step's command** (Andrei, in a second terminal; the block in each step). Never start or stop it in the middle of a step. It then runs through every arm and block of the step alike, which keeps H1's interleaved blocks symmetric.
- **Check it** (Claude may run; seconds), about 30 s after the start: `"$PY" tools/power_log.py --check` prints one JSON line about the newest raw log, with `"ok": true` when it is readable and current. Output reaches the file in bursts, so `empty` or `at least 2` means wait 30 s and check again. `key layout differs` (documents parse but give no sample), `counted twice` (documents overlap) or `3 x the median interval` (a long document) mean the frozen tool does not read this log as intended: leave the logger running and see the last bullet. If a check still fails, run the step anyway and note it in the commit body.
- **Stop it after the step** with Ctrl-C **in the logger's own terminal**. It runs as root, so no script can stop it. Ctrl-C in the step's terminal would interrupt the run itself; for the gate that is a crash, exit 3, or exit 1, a counted gate run, if a K8 check had already failed (step 10).
- **Write the record** (Claude may run; seconds, a few minutes for a 16 h log): `"$PY" tools/power_log.py` reads the newest raw log, joins it to the windows that the run records carry, and writes `results/power/power_<UTC>.json`. Per window, the record gives the energy in Wh (key `combined`: CPU + GPU + ANE), the CPU/GPU/ANE split, mean and peak W, coverage, gaps over 5 s, the time counted by more than one sample (`overlap_s`), and the raw file's sha256 and size. A run that crashed without a closing time gets `--window <name> <start UTC> <end UTC>`. The tool refuses a raw log that already has a record (`--again` writes a second record that names the first), and a raw log whose documents overlap, so that no energy is counted twice.
- **Commit the record** with the step's own records (it lies under `results/`). **Never commit the raw `.plist`.** It stays in `$EXP036_WORK/exp037/power/` and is never copied into the repository or mirrored to the mini, because it carries the boot arguments, the boot time and the model id. The tool refuses a raw path inside the repository, and the leak check refuses any file with NUL bytes in this directory.
- A 16 h session's raw log is estimated at 0.6–1.7 GB (the per-sample size on the M5 Max is unmeasured). The free disk of step 0c covers it.
- **If the tool cannot read a log after the freeze** (it refuses the log as overlapping or as holding no readable sample, or `--check` keeps reporting `key layout differs`, `counted twice` or a long document): keep every raw `.plist` in `$EXP036_WORK/exp037/power/` and never edit or delete one. **Do not edit `tools/`**: it is a frozen scope, and P1(a) refuses every gate run on a changed tools hash. Note it in the step's commit body and go on, logging the later steps as usual. After S3, the main session writes a "power log" amendment (HYPOTHESIS, "Fix-cycle, freeze and amendment rules") that fixes `tools/power_log.py` and its test; Claude then writes one record per raw log.

---

## 0. One-off preparation (before Session 1)

**0a. Repo and push access.** *Claude may run; seconds.*
```bash
[ -d ~/REPOS/local-first-ai ] || git clone git@github.com:miktam/local-first-ai.git ~/REPOS/local-first-ai
git -C ~/REPOS/local-first-ai pull --ff-only && git -C ~/REPOS/local-first-ai push --dry-run
```
Check: the clone or pull and the dry run succeed.

**0b. Assets and venv present.** *Claude may run; seconds.* exp_037 needs no new download: every model, dataset and code pin is exp_036's, still on the mbp. The exp_037 venv was created on 2026-10-06 (checked as `mlx 0.32.3 | mlx-lm 0.32.0 | applegpu_g17s`); step 3 syncs it.
```bash
ls ~/models/exp036 ~/models/exp036/data && \
ls ~/models/exp036/Kolibri-1-MLX-8bit-g64/exp036_convert_record.json ~/models/exp036/Kolibri-1-MLX-4bit-g64/exp036_convert_record.json && \
ls ~/models/exp036/gemma-4-26b-a4b-it-bf16 && \
ls ~/models/exp037/venv/bin/python
```
Check: every folder of `ASSETS.md` is present; exp_036's K8 and K4 builds hold their convert records (step 8 clones them); the Gemma 4 bf16 folder of exp_036's Amendment 6 record is present (step 9a; if it is absent, its download is an outward step that needs Andrei's go, and without it step 9a does not run); the exp_037 venv exists (otherwise step 3 creates it). Paths elsewhere: set them in exp_036's git-ignored local override and use the same names.

**0c. Machine hygiene.** *Andrei by hand; once.*
- System Settings › Battery › Energy Mode (on power adapter): **High Power**. Low Power Mode off.
- Defer automatic macOS updates for the experiment window: an update reboots, resets any sysctl and breaks the version freeze.
- Disk: keep ≥ 260 GB free beyond the downloads. `tools/preflight.py` requires it at step 4 (below it, exit 1), because step 8's clones under `$EXP037_BUILDS` do not exist yet; after step 8 it requires 120 GB. The clones of step 8 are APFS clones that share blocks with exp_036's builds; the new work files under `$EXP036_WORK/exp037/` (R2F and R3 logits, about 15 GB each; the G5 log-prob files, about 8.3 GB) and the H8 K8 dump under `$EXP036_WORK/kl/<UTC>/` (step 11) take about 50 GB, and the git-ignored session step logs of steps 14 and 16 a few GB at most.

---

## Session 1 — daytime, ≈ 6.2 h nominal / 9.2 h pessimistic of machine time

S1 hours are the sum of the machine intervals recorded by its blocks, not wall-clock time; Andrei's reading time, step 9a and any gate fix cycle do not count. A gate fix cycle moves the rest of S1 to a later day.

### 1. Pull and the exp_037 version record
*Claude may run; seconds.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && \
git -C "$LFA" log --oneline -6 -- tasks/chronos/exp_037_kolibri_forced_gate && \
git -C "$LFA" status --porcelain && \
grep -c '{{' HYPOTHESIS.md; grep -c '^Pre-registration commit:' HYPOTHESIS.md; \
"$PY" tools/version_record.py
```
Check:
- HEAD contains the commit named on HYPOTHESIS.md's line `Pre-registration commit: <sha>`. Check it by sha, not by subject: `git merge-base --is-ancestor "$(sed -n 's/^Pre-registration commit: //p' HYPOTHESIS.md)" HEAD && echo ok`. The gate's P1(c) also requires it to be an ancestor of the upstream branch.
- `git status --porcelain` prints nothing: the **whole** checkout is clean.
- The first `grep -c` prints `0`, the second `1`.
- `tools/version_record.py` writes `results/version_record_<UTC>.json` (schema `exp037 version record v1`) under `$PY`: macOS 27.0 (26A428), mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 and `mx_device_info.architecture` `applegpu_g17s`. It is the binding record P2 checks; step 4b writes a newer one, and the newest counts. It refuses without the git identity: on a fresh clone, run step 2 first and repeat this line.

Outputs: `results/version_record_<UTC>.json`. Committed with step 4a's records, before 4b.

### 2. Git identity and the exp_037 pre-push hook
*Claude may run; seconds.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" config user.name  "Miktam" && \
git -C "$LFA" config user.email "hello@localfirstai.eu" && \
HOOKS="$(git -C "$LFA" rev-parse --path-format=absolute --git-path hooks)" && \
cp tools/hooks/pre-push "$HOOKS/pre-push" && chmod +x "$HOOKS/pre-push" && \
test -x "$HOOKS/pre-push" && echo "pre-push hook installed" && \
git -C "$LFA" config --get user.name && git -C "$LFA" config --get user.email
```
Check: it prints `pre-push hook installed`, `Miktam` and `hello@localfirstai.eu`. The installed hook now checks exp_037's directory only; it replaces exp_036's hook (exp_036 is closed). No commit.

### 3. venv sync and the unit tests on the mbp
*Claude may run; ≈ 20–30 min (the suite took 30 min on the mini).*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
bash env/setup.sh && \
EXP036_TOK="$EXP036_MODELS/Kolibri-1-BF16" EXP036_REQUIRE_ALL=run-host "$PY" -m pytest -q -rs tests
```
- `env/setup.sh` creates or syncs `$EXP037_VENV` (default `~/models/exp037/venv`) with the exact pins of `env/requirements-mbp.txt`, never exp_036's venv. It is idempotent and never calls sudo.
- `EXP036_REQUIRE_ALL=run-host` turns every skip into a failure, except a skip whose reason starts with `build-host only:`: the tests that compare with files only the mini holds, and `tests/test_gate_ftiny.py` (F_tiny is a constant produced on the mini; its validation margin is thin by construction, so it runs on the mini only). `-rs` lists them.
- The suite includes the heavy tiny tests (`test_gate_drivers_tiny.py`, `test_gate_end_to_end_tiny.py`), which run the whole tiny gate; the gate's own G1 at step 10 leaves them out.

Check:
- `env/setup.sh` prints mlx 0.32.3, mlx-metal 0.32.3 and mlx-lm 0.32.0.
- All tests pass; the only skips are `build-host only:`.
- The tests that read a copy of the live HYPOTHESIS.md (the hash-table round trip, the Status line) do not depend on its state: filled, signed off or amended, they pass, here and in the gate's G1.
- A failure stops the runbook. The test output goes to the main session through a pushed note (`aborted/<UTC>-tests/NOTE.md`, commit `chronos/exp_037: mbp unit tests failed — note`). Paths in the note are written as `~/…` or `$EXP036_MODELS/…`.

Outputs: none in `results/`. No commit when the tests pass.

**3b. Optional self-test: the dry run.** *Claude may run; ≈ 12 min (11.3 min on the mini); weight-free (tiny random checkpoints).* It runs the whole pipeline end to end in a temporary copy of the kit (its own git repository and bare remote, tiny Kolibri arms and tiny stand-ins for every peer, synthetic task data): preflight → manifests → refresh_builds → peer check → the exp_037 gate `--tiny` on the unsharpened seed-29 build, with every control and probe 22 → bench → pilot → plan fix → S2/S3 → scorers → verdicts → `hash_tree --check` → leak check. Nothing is written into this repository, `$EXP036_MODELS` or `$EXP036_PRIVATE`.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/dry_run.py --tok "$EXP036_MODELS/Kolibri-1-BF16" --peers "$EXP036_MODELS"
```
Check: the last stdout line is JSON with `"ok": true`; exit 0. A failure names the stage; send it to the main session like a test failure. The dry run is not evidence and is never committed. It is valid at any step: its copy of the kit is first reset to the pre-registration state.

**3c. Optional: the loaders on the real data.** *Claude may run; seconds.* It builds every manifest set from the real data into a new dated folder under `$EXP036_WORK/exp037/` only, with the counts enforced; nothing is written into the repository.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
CHK="$EXP036_WORK/exp037/manifest_check_$(date -u +%Y%m%dT%H%M%SZ)" && \
SETS=$("$PY" tasks/build_manifests.py --list | cut -f1 | paste -sd, -) && \
"$PY" tasks/build_manifests.py --sets "$SETS" --out "$CHK/public" --private "$CHK/private"
```
Check: the last line is JSON with `"ok": true`, `"checks": {"gpqa_diamond_en_overlong_excluded": 0, "mmlu_prox_full_gold_inconsistent_excluded": {"de": ["3787"], "en": ["3787"]}}` and the counts GPQA-D EN 198, GPQA-D DE 198, MMLU-ProX-Lite EN/DE 588 each, AIME EN/DE 30, IFBench 300, RGB closed-book 400, Negative 300, Fact-Check 100.

### 4. Preflight
**4a. Quick.** *Claude may run; < 1 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/preflight.py --quick
```
Check: `problems = []`, exit 0 or 2. Exit 2 is expected here: step 1's record is still uncommitted (`working tree has N changed paths in the experiment dir`), and the K8 sysctl advice, if printed, is a warning too. Exit 1 (a problem) stops the runbook.

Outputs: `results/preflight_<UTC>.json` and a newer `results/version_record_<UTC>.json`.

Commit (Claude, after the leak check): the records of steps 1 and 4a (`results/version_record_<UTC>.json` ×2, `results/preflight_<UTC>.json`), `chronos/exp_037: preflight <UTC> quick (M5 Max, L=<GiB>)`, then push. 4b warns on any changed path in the experiment directory, so it needs this clean tree.

**4b. Deep.** *Andrei runs under `caffeinate -i`; minutes (hashes the assets against their LFS oids).*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" tools/preflight.py --deep
```
Check:
- `problems = []`, and exit code 0, or 2 when the only warning is `K8 working-set rule: sysctl advice printed` (then step 5 applies). Any other warning fails the check. A changed-paths warning means the records of steps 1 and 4a were not committed first: move this run's two records to `aborted/<UTC>-preflight-deep/` with a NOTE.md, commit them with the step 1/4a records, and re-run 4b (as exp_036 did: its `aborted/20261004T062910Z-preflight-deep/`).
- Chip `Apple M5 Max`, memory 128 GB, power mode High Power on AC; `mx_device_info.architecture` `applegpu_g17s`; the venv's versions equal `env/versions.json` (mlx 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0).
- `iogpu.wired_limit_mb` and the Metal limit are recorded (107.5 GiB with the default 0 on 2026-10-03).
- Every asset's revision matches assets.json, and every shard's sha256 equals its etag.
- The exp_007 fixtures are found by their real names.
- Read the **K8 working-set line**: the memory-rule B per K8 cell and either "no sysctl needed" or a `sudo sysctl` line with the B it would gain. (allowed_B, set by the gate, may clip it later.)
- The printed next step.

Outputs: `results/preflight_<UTC>.json` (schema `exp037 preflight v1`) and a newer `results/version_record_<UTC>.json`.

Commit (Claude): the records of step 4b, `chronos/exp_037: preflight <UTC> (M5 Max, L=<GiB>)`, then push.

### 5. Metal working set for K8 (only if 4b printed a sysctl line)
*Andrei by hand; seconds.*
```bash
sudo sysctl iogpu.wired_limit_mb=114688      # 112 GiB, leaves ≈ 16 GiB for macOS; resets at reboot
```
Then Claude re-runs step 4a and commits its records (after the leak check; `chronos/exp_037: preflight <UTC> quick (M5 Max, L=<GiB>)`), then pushes, before any further preflight. Check: the quick preflight reports the new limit and K8's B per cell, with `problems = []`.
- If Andrei declines, the plan rule uses the default limit. If even B = 1 does not fit, K8 is NOT RUN and K4 becomes primary through the plan amendment (HYPOTHESIS, K8 working-set rule).
- Andrei writes the value, "not needed" or "declined" into the sign-off. **After any reboot, redo this line before steps 10–17;** `run.py` refuses K8 cells if the limit is below the one the plan used.

### 6. Sign-off
*Andrei types; Claude commits and pushes; ≈ 15 min of Andrei's reading.*

Andrei fills the **Sign-off** block at the end of `HYPOTHESIS.md` himself, in an editor on the mbp. It re-confirms the items decided with A–E:
- `- Signed off by: Andrei (typed by Andrei on the mbp, <UTC>)`;
- the UTC time from the clock;
- the H2 margin (−4 pp as pre-registered and re-confirmed; −3 or −5 only by Amendment 0);
- the embed/head policy: quantised at the arm's bits with fp32 logits (exp_036's builds, which P4 binds; vendor-faithful is not available without a new conversion);
- the K8 Metal value from step 5;
- the overrun ceiling (44 h) and the cloud anchor (declined).

**If the H2 margin differs from −4 pp** (Amendment 0, before the gate run): Claude edits `analysis/margins.json` (`"H2": -0.03` or `-0.05`) and appends `## Amendment 0 — sign-off choices (<UTC>)` to HYPOTHESIS.md, quoting Andrei's choice and giving the new `analysis/` tree value as the line `"$PY" tools/hash_tree.py --amend-line analysis` prints (`hash_tree: ANALYSIS_SHA256 = <hex>`), pasted verbatim.

Claude then runs:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/status.py --record-block "sign-off" && \
git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate/HYPOTHESIS.md tasks/chronos/exp_037_kolibri_forced_gate/analysis/margins.json && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_037: sign-off — Andrei, <UTC>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: the push succeeded, and the status prints nothing outside `results/`, `aborted/`, `evidence/` and `diagnostics/`. Every later runner command checks the "Signed off by" line's form at HEAD.

### 7. Verify the inherited manifests, withheld shingles and T5/T6/T9
*Claude may run; ≈ 2 min.* exp_037 reuses exp_036's item manifests (`tasks/manifests/`, copied), its withheld manifests (`$EXP036_PRIVATE/manifests/`) and its shingle file (`tools/withheld_shingles.sha256`, copied), and the T5, T6 and T9 ids in `$EXP036_WORK/gate_texts/`. The builder writes every manifest only once and refuses to replace a different one, so re-running it over the real data verifies them all; the second line loads T5, T6 and T9 read-only and checks their ids against `gate/texts/MANIFEST.json`.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tasks/build_manifests.py && \
"$PY" -c "from gate import textset; textset.load_real(); print('T5, T6, T9 present and equal to MANIFEST.json')"
```
Check:
- Both exit 0.
- The builder's stdout JSON shows `"status"` with **every** entry `"unchanged"` (public manifests, `private/…` manifests, `mmlu_prox_category_counts.json`, `withheld_shingles.sha256`) and the `checks` of step 3c. An entry `"written"` means a manifest or the shingle file was missing, and exit 1 with `"ok": false` and an `error` saying a file "exists with different content" (prefixed `manifest <name>: FileExistsError:` for a per-set manifest) means one differs: stop and tell the main session.
- `results/manifests_<UTC>.json` is written: exp_037's own record of the manifests' sha256, equal to the copied `results/manifests_20261004T131126Z.json`'s.
- The second line prints its message; a `FileNotFoundError` or `ValueError` means T5, T6 or T9 is missing or differs: stop and tell the main session (no web text is ever written into the repo).

Outputs: `results/manifests_<UTC>.json`.

Commit (Claude, after the leak check): `"$PY" tools/status.py --record-block "manifests"`, then `chronos/exp_037: inherited manifests verified (ids and hashes only)`. Push.

### 8. Clone and refresh the K8 and K4 builds
*Andrei runs under `caffeinate -i`; ≈ 0.1–0.2 h (hashes about 127 GB).*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" tools/refresh_builds.py && \
"$PY" tools/status.py --record-block "refresh"
```
`tools/refresh_builds.py` replaces exp_036's conversion. For each of `Kolibri-1-MLX-{8,4}bit-g64` under `$EXP036_MODELS` it checks the source (exp_036's port file `cd6153b8…` and its convert record), clones the directory with `cp -Rc` to `$EXP037_BUILDS` (an existing clone with identical shards and config is kept), gives the clone exp_037's port file through `port/convert.py`'s refresh (whose verification load passes `trust_remote_code=True` for the port file it has just copied; afterwards `check_port_file` must pass on the clone), re-checks the source and writes its record. It never writes into exp_036's builds, and a re-run after a port fix needs no manual delete.

Check:
- Exit 0; two records `results/convert/refresh_{8,4}bit_<UTC>.json`, each with the source record's sha256, `source_port_sha256` `cd6153b8…` (exp_036's port) and `port_sha256` equal to exp_037's port (HYPOTHESIS hash table). `previous_port_sha256` is also `cd6153b8…` on the first clone; on a re-run it is the port the clone already carried (equal to `port_sha256` if the port is unchanged).
- exp_036's builds are byte-identical to before (the tool checks it).
- The records count in S1 hours.

Commit (Claude): `chronos/exp_037: K8/K4 clone-and-refresh records`. Push.

### 9a. The Gemma 4 fidelity record under MLX 0.32.3
*Andrei runs under `caffeinate -i`; ≈ 0.2 h; before step 9.* The peer check uses this record only if the Gemma 4 family's NLL(8) / KL(8‖4) rule fails (HYPOTHESIS, "Peers are verified, not gated"). First the bf16 source's 20 files at the pin are checked against their download metadata (11 weight shards; 12 LFS files with `tokenizer.json`; the repository's file list at the pin, read from the Hugging Face API on 2026-10-06), then the byte-identical copy of exp_036's producer runs parts A–D under the exp_037 venv.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
G16="$EXP036_MODELS/gemma-4-26b-a4b-it-bf16" && \
"$PY" -c "import sys; from pathlib import Path; from tools import assetcheck as a; d = Path(sys.argv[1]); m = a.hf_metadata(d); r = a.deep_hash(d, m, 8); s = [k for k in r if k.endswith('.safetensors')]; lfs = [k for k, v in r.items() if v['algo'] == 'sha256']; ok = len(r) == 20 and len(s) == 11 and len(lfs) == 12 and all(v['ok'] for v in r.values()) and {x['commit'] for x in m.values()} == {sys.argv[2]}; print(len(s), 'bf16 shards,', len(lfs), 'LFS files,', len(r), 'files:', 'every hash equal to its download metadata at the pin' if ok else 'MISMATCH'); sys.exit(0 if ok else 1)" "$G16" e13fae2a81ec07e3092a3ebb70c80970b640dc3c && \
caffeinate -i "$PY" diagnostics/gemma_quant_check.py --g8 "$EXP036_MODELS/gemma-4-26b-a4b-it-8bit" --g4 "$EXP036_MODELS/gemma-4-26b-a4b-it-4bit" --bf16 "$G16" --out "diagnostics/gemma_quant_check_$(date -u +%Y%m%dT%H%M%SZ).json"
```
Check:
- The shard line prints `11 bf16 shards, 12 LFS files, 20 files: every hash equal to its download metadata at the pin`. `MISMATCH`, or a folder without download metadata: stop and tell the main session (a new download is an outward step that needs Andrei's go).
- Exactly one `diagnostics/gemma_quant_check_<UTC>.json` exists, with `"mlx": "0.32.3"`, `"device": "Apple M5 Max"` and `reference_kl` rows for G8 and G4 on T1–T4. The console prints each build's token-weighted KL(bf16‖build); exp_036's record under 0.31.2 gave 0.0167 (G8) and 0.3778 (G4), and the new values are expected near them, not bitwise.
- **After a crash:** move the earlier output to `aborted/<UTC>-gemmacheck/` with a NOTE.md first, then repeat, so that exactly one record remains.
- **If the bf16 folder is absent** and no download is made: skip this step; step 9 runs, and the registered rule applies to the Gemma family (both builds `fail` if its rule fails).

Commit (Claude, after the leak check): `"$PY" tools/status.py --record-block "fidelity record" --files diagnostics/gemma_quant_check_<UTC>.json`, then `chronos/exp_037: Gemma fidelity record <UTC> (step 9a)`. Push.

### 9. Peer check
*Andrei runs under `caffeinate -i`; ≈ 25–35 min.*

**Power log, second terminal** (Andrei, before the command below; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. After the peer check, Andrei presses Ctrl-C in the logger's terminal; then Claude runs `"$PY" tools/power_log.py`.

The peer check:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" tools/peer_check.py --arms G8,G4,Q36-8,Q36-4,Q38-8,Q38-4
```
It re-runs exp_036's peer check under the exp_037 runner (`prefill_batch_size = min(B, 8)`) and MLX 0.32.3 / mlx-lm 0.32.0; exp_036's peer records are not reused. The peers load as built-in mlx-lm model types, without `trust_remote_code`.

Check:
- Every arm says `ok`, `B=1` or `speed-only`: strict text-only load; parameter count within 2 %; NLL(8) ≤ NLL(4) + 0.02 and KL(8‖4) < 0.2 on the chat-wrapped texts (`families.<f>.fidelity`, `arms.<a>.nll_chat`; the raw-text NLL that G3a reads stays in `arms.<a>.nll`); template and tokenizer parity recorded; EOS ids recorded.
- G8 and Q36-8: the batched-path check (parity under the G5 noise-floor rule and a greedy flip rate ≤ 2 % of 30). A peer that fails only that check is `B=1`: every cell of it runs at B = 1, and it is not a drop.
- When a family's NLL / KL rule fails, each build is judged against the step-9a record by its own KL(bf16‖arm) on T1–T4: < 0.2 passes fidelity and keeps its other verdict; ≥ 0.2 gives `speed-only` (no quality cell; out of H8's peer median; H1, the speed cells and the B4 ladder keep it). The peer record names the record's path and sha256 under `fidelity_reference.families.gemma4`. An unusable record (none, two, a wrong producer, another MLX, a record newer than this run) leaves the registered rule: both builds `fail`. Exit code 2 is expected with `B=1` or `speed-only`.
- A peer that fails anything else (verdict `fail`) is dropped by amendment before any scored run. Stop and tell Andrei; the main session pushes `amendments/<k>_peerdrop_<UTC>.md`. From step 12 the pilot runs with `--without <arm>`, and the plan amendment records the drop: H3 and H6 use the remaining MoE peer, and H4 is NOT RUN if Qwen3.6 is the one dropped.

Outputs: `results/peers_<UTC>.json`, and the power record `results/power/power_<UTC>.json` (descriptive).

Commit (Claude): `"$PY" tools/status.py --record-block "peers"`, then `chronos/exp_037: peer check <UTC>`, with the power record if there is one (never the raw `.plist`). Push.

### 10. Phase 0 — the forced gate (blocking)
*Andrei runs under `caffeinate -i`; ≈ 3.0 h nominal / 4.4 h pessimistic.* If step 5 applied and the laptop rebooted, redo step 5 first. Close other apps.

**Power log, second terminal** (Andrei, before the gate; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. When the gate has printed its last line, Andrei presses Ctrl-C **in the logger's terminal only**: a Ctrl-C in the gate's terminal interrupts the gate, which then counts as a crash (exit 3, or exit 1 if a K8 check had already failed). Then Claude runs `"$PY" tools/power_log.py`. The record measures the gate and each of its phases. For a gate killed outright, it measures the phases that have records; add the missing span with `--window`.

The gate:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" gate/run_gate.py --all
```
It runs the phases of HYPOTHESIS Phase 0: preconditions P1–P4 (with G0k), G0 and G1, the R1 reuse, emulation statistics and G3, P5, R2F and R3, G2, K8 (bf16 checks, then fp32 checks, then the control mutants), K4, the R1 anchor pass, then the rules and the record. The last stdout line is JSON with `"verdict"`, `"exit_code"`, `"record"`, `"failing"` and `"error"` (plus `"interrupted"` after an interrupt); allowed_B is not on it.

Exit codes (HYPOTHESIS, "Verdicts, exits and the record"):
- **0** — K8 and K4 PASS. Continue. Read `allowed_B` for K8 and K4 from the gate record: it sets the B of every Kolibri cell from here on. Claude may run this read-only line (seconds); it prints the newest `results/gate/gate_<UTC>.json`'s name and its `allowed_B`, never a `_layers` or `_mutants` sibling:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
  "$PY" -c "from runner import guard; r = guard.newest_gate_record(); print(r['_file'], 'allowed_B', r.get('allowed_B'))"
  ```
- **1** — K8 FAIL. Stop.
- **4** — K8 PASS, K4 FAIL. Stop. If the fix cycles end with K4 still failing, the run continues without K4, and H1, H7, H8 and D1 are NOT RUN: step 11 runs `"$PY" bench/run_bench.py --cells speed_desc,tokenizer,c1`, step 12 runs `"$PY" runner/run.py pilot --without K4`, and `runner/plan_fix.py` queues no K4 cell (nor B4 or B8).
- **5** — K8 or K4 INCOMPLETE ("power not shown: <mutant>" or "VOID: <calibrator> above ceiling"). Stop. It counts as a gate run.
- **3** — a precondition failed or the run could not finish. It is not a gate run and never a cycle; the gate itself moves its partial output to `aborted/<UTC>-gate/` with a NOTE.md. Ctrl-C, SIGTERM and SIGHUP (a closed terminal) count as crashes: the gate writes the record first, then ends with the record's exit code; the last stdout line then also carries `"interrupted"`. A K8 check that had already failed makes such a run exit 1, a counted gate run.

Outputs:
- `results/gate/gate_<UTC>.json` (with `allowed_B`, `preconditions`, `controls`, `calibrators`, `run_counts`, `blind_phase_reached`), `gate_<UTC>_layers.csv`, `gate_<UTC>_mutants.json`, and the per-phase files under `results/gate/<UTC>/`.
- Work files under `$EXP036_WORK/exp037/` (forced dumps, the G5 log-prob files; not committed; their sha256 are in the record). R1 is read from exp_036's dump under its registered key.
- The power record `results/power/power_<UTC>.json` (descriptive; the raw `.plist` stays in `$EXP036_WORK/exp037/power/`).

Commit (Claude), after `"$PY" tools/status.py --sync-amendments && "$PY" tools/status.py --record-block "gate"`, with the power record if there is one:
- On exit 0: `chronos/exp_037: gate <UTC> — PASS (K8, K4; allowed_B K8 <set>, K4 <set>)`.
- On exit 1 or 4: `chronos/exp_037: gate <UTC> — FAIL (<first failing check>)`.
- On exit 5: `chronos/exp_037: gate <UTC> — INCOMPLETE (<reason>)`.
- On exit 3: `chronos/exp_037: gate <UTC> could not run — note`. Push.

**After exit 1, 4 or 5, stop here.**
1. No pilot, bench or scored run.
2. Andrei chooses, and the choice is recorded with its UTC time and label: a gate fix plus a re-run (if cycles remain), diagnostics (their outcome-to-action rules committed before they run), or stop and publish. Stop and publish is available at every point.
3. A gate fix comes from the main session as `amendments/<k>_gatefix_<UTC>.md` plus code, with a test that fails before the fix and an oracle independent of every gate value. It never changes a mutant, a control, a threshold or the gate text.
4. Re-run, on Andrei's go: step 1 (pull), `"$PY" tools/status.py --sync-amendments`, step 3, step 8 if the port changed, then step 10 **in full**. Matching dumps are reused; P1(d) refuses a re-run unless the port, reference, runner, gate code or builds changed.
5. At most two fix cycles (three gate runs). After the third failing run, or on stop, the gate failure is the exp_037 result; the main session follows HYPOTHESIS "Gate failure or STOP".

**After exit 3:** repeat step 10 unchanged once the cause is fixed outside the kit (disk, power), or wait for a typed gate fix if code must change. After two unchanged repeats, Andrei decides. If the record says `blind_phase_reached: true`, the gate fix's first line names the crashed run by its UTC and every port-against-reference value it computed (they are published from `aborted/<UTC>-gate/`); P1(d) refuses a re-run with changed code until that first line exists.

**If the run was killed outright** (power cut, kernel panic, an out-of-memory kill, `kill -9`): `results/gate/<UTC>/` exists without `results/gate/gate_<UTC>.json`, and P1(d) refuses every gate run until it is closed. Claude closes it first, on the mbp:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" gate/run_gate.py --close-orphans
```
It judges the run's phase files as they stand, with the error "interrupted", writes its record (`blind_phase_reached` from its own P5) and, when the record exits 3, moves the run to `aborted/<UTC>-gate/` with a NOTE.md. If a K8 check had already failed, the record exits 1 and counts as a gate run: its phase files stay in `results/gate/<UTC>/`, the CSV and mutants file go to `results/gate/`, and there is no aborted directory or NOTE.md. Then commit and push as for exit 3 (or exit 1), and continue with "After exit 3" (or "After exit 1, 4 or 5").

### 11. Bench cells: H1, descriptive speed, D1, H8, H5, C1
*Andrei runs under `caffeinate -i`; ≈ 1.2–1.7 h.* Close other apps first and leave the laptop idle for 10 minutes so it starts cool; the script checks `thermalState`.

**Power log, second terminal** (Andrei, before the bench command; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. The logger runs through every cell, so H1's blocks see it equally on both arms, and its record also measures the bench's 10-minute cool-down as an idle baseline of the same CPU + GPU + ANE power. After the bench, Andrei presses Ctrl-C in the logger's terminal; then Claude runs `"$PY" tools/power_log.py`.

The bench:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" bench/run_bench.py --cells speed,speed_desc,fit,kl,tokenizer,c1
```
Check:
- Every cell file exists and reports `complete`. The runtime binding (P2's pins) is checked first and recorded.
- H1: 10 blocks for each of K4 and G4. D1: 3 reps at 32k and at 64k. H8: 4 families × 6 texts. H5: 5,000 documents.
- C1 runs at the largest allowed B ≤ the memory-rule B; if allowed_B(K8) = {1} its file is complete with `"status": "NOT RUN"` ("no batching to control"). The descriptive speed cells run Kolibri only at B ∈ allowed_B ∩ {1, 2, 4}.
- Power mode and thermal state logged. Do **not** use the laptop during the speed cells.
- Exit code 0: every cell complete. **3**: a cell could not run (thermals): let the laptop cool and rerun the same command. **1**: a cell failed: send the stderr to the main session.

Outputs: `results/bench/{speed,speed_desc,fit,c1}_<UTC>.jsonl`, `results/tokenizer_<UTC>.json`, `results/kl_8v4_<UTC>.json`, and the power record `results/power/power_<UTC>.json` (descriptive). K8 log-probs go to `$EXP036_WORK/kl/<UTC>/` and are not committed: the unchanged exp_036 cell's default; the record's `k8_dump.dir` names the folder and `k8_dump.files` their sha256. C1's full records (completion ids included) go to `$EXP036_WORK/c1/c1_<UTC>.full.jsonl`, also not committed. These are the exceptions to "new work files only under `$EXP036_WORK/exp037/`" (BUILD_SPEC §2): inherited bench paths, new UTC-stamped names beside exp_036's, and nothing of exp_036's is touched.

Commit (Claude): `"$PY" tools/status.py --record-block "bench"`, then `chronos/exp_037: S1 bench cells — H1, D1, H5, H8 raw`, with the power record if there is one. Push.

### 12. Pilot
*Andrei runs under `caffeinate -i`; ≈ 1.3–1.9 h.*

**Power log, second terminal** (Andrei, before the pilot; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. After the pilot, Andrei presses Ctrl-C in the logger's terminal; then Claude runs `"$PY" tools/power_log.py`, which measures the pilot and each of its arms. An interrupted pilot has no summary and therefore no window: give its span with `--window`. A rerun pilot gets its own logger.

The pilot:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
caffeinate -i "$PY" runner/run.py pilot
```
Check:
- `pilot_summary_<UTC>.json` lists, per arm and task: mean, SE and max completion tokens, truncation count, reasoning-status counts, parse failures and diagnoses, prefill rate; the per-step decode logs exist (each cell's `steps_file`, `results/pilot/<UTC>/<arm>/*.steps.jsonl`); it records each Kolibri arm's allowed_B and the observed environment.
- The pilot outputs are **never** scored for accuracy.
- Without K4 (step 10, exit 4), or with a peer that step 9 dropped (`fail`: G8 or Q36-8) or made `speed-only` (only G8 can be; Qwen3.6 has no fidelity reference, so Q36-8 can only fail): `caffeinate -i "$PY" runner/run.py pilot --without K4` (or `--without G8`, `--without Q36-8`, or several, comma-separated). The pilot refuses a `speed-only` arm that `--without` does not name. G4 is not a pilot arm.
- **Interrupted (Ctrl-C) or crashed:** the pilot writes no summary. Move the partial pilot aside, then rerun this step in full:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
  "$PY" runner/run.py abort-pilot --stamp <UTC>
  ```
  It moves `results/pilot/<UTC>/` to `aborted/<UTC>-pilot/` with a NOTE.md (its step logs move with it; step 13 commits them with the rest), and the private files to `$EXP036_PRIVATE/exp037/aborted/<UTC>-pilot/`, with only their sha256 and byte counts in the note.

Outputs: `results/pilot/<UTC>/<arm>/*.jsonl` and the step logs `results/pilot/<UTC>/<arm>/*.steps.jsonl` (withheld sets: text in `$EXP036_PRIVATE/exp037/pilot/<UTC>/<arm>/`), `results/pilot_summary_<UTC>.json` and the power record `results/power/power_<UTC>.json` (descriptive). The pilot's step logs are committed ("Step logs"): `runner/plan_fix.py` fits the plan rule's step model on them, and step 13's plan record block lists their sha256.

No separate commit; it is committed with step 13 (whose `git add` of this directory takes the outputs and their step logs, the summary and the power record; never the raw `.plist`).

### 13. Fix n and max_tokens by rule (plan amendment)
*Claude may run; seconds, plus the commit.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && \
"$PY" runner/plan_fix.py --pilot results/pilot_summary_<UTC>.json
```
`plan_fix` reads S1 hours from the S1 records (the refresh records included), allowed_B from the newest gate record, the excluded and speed-only arms from the gate and peer records, and each cell's step log (`steps_file`) for the step-model fit. It prints the paths of the new `results/plan_fixed_<UTC>.json` and `results/AMENDMENT_<k>_<UTC>.md`, then a JSON line with the status. Exit 0: FIXED. Exit 2: STOP or BLOCKED (both files are still written). Exit 1: refused input; nothing is written. Then, for a FIXED plan:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --append-amendment results/AMENDMENT_<k>_<UTC>.md && \
"$PY" tools/status.py --record-block "plan" && \
git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_037: amendment <k> — plan fixed by rule (<plan>, n_M=<n>, Tier B <list>)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: `plan_fixed_<UTC>.json` gives the plan P0…P10 or STOP; n_M, the caps, B per cell (each Kolibri cell clipped to allowed_B), the L used, the ordered queue and its S2/S3 split, the Tier-B items; the row sets of H2 and H7, the manifest and gate sha256s, the `scorers/` tree sha; projected hours: total ≤ min(31, 40 − S1 hours), each session ≤ 16 h.
- The plan record block also hashes every pilot summary, every `results/pilot/**/*.jsonl` (the cell files and their step logs) and an aborted pilot's step logs (`tools/status.py` `PILOT_FILES`): the sha256 binding of the step logs on which `plan_fix` fitted the plan. After a re-pilot, the rerun of this step hashes both pilots.
- **A pilot step log over 50 MB** (beyond the estimate, whose ceiling is about 50 MB at B = 1 with every item at the 32k cap): the leak check refuses it (`too_large`) and the block stops before the commit. Leave that log uncommitted: unstage it (`git -C "$LFA" restore --staged -- tasks/chronos/exp_037_kolibri_forced_gate/<path>`), add its exact path, anchored at the repository root, to the mbp's `.git/info/exclude`, never to the kit's `.gitignore` (`printf '%s\n' '/tasks/chronos/exp_037_kolibri_forced_gate/<path>' >> "$(git -C "$LFA" rev-parse --path-format=absolute --git-path info/exclude)"`), and rerun the block from its `git add` line. Its sha256 stays in the plan record block, step 18 copies it to the mini with its own rsync line (`--relative`, by its kit-relative path), and a re-derivation of the plan needs that copy.

Outcomes:
- **STOP** or **BLOCKED**: the amendment is still appended and pushed, with its own Status line and commit message:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
  "$PY" tools/status.py --sync-amendments && \
  "$PY" tools/status.py --append-amendment results/AMENDMENT_<k>_<UTC>.md && \
  "$PY" tools/status.py --record-block "plan" --status "plan <STATUS> by rule; no scored run" && \
  git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate && \
  "$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
  git -C "$LFA" commit -m "chronos/exp_037: amendment <k> — plan <STATUS> by rule" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
  git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
  ```
- **STOP:** no scored run. Andrei decides on a 4th session by amendment.
- **BLOCKED cell:** after the commit above, the main session pushes an `amendments/` extractor or delimiter fix. Then pull, `"$PY" tools/status.py --sync-amendments`, and re-pilot only the BLOCKED cells, e.g. `caffeinate -i "$PY" runner/run.py pilot --cells K8:gpqa_main_en:high`. Then rerun this step with **both** summaries, the full pilot's first: `"$PY" runner/plan_fix.py --pilot results/pilot_summary_<first UTC>.json --pilot results/pilot_summary_<re-pilot UTC>.json`.
- The commit must be pushed and HEAD must equal upstream; `runner/run.py session` refuses otherwise.

**End of Session 1.** If the laptop must reboot, the sysctl from step 5 resets: redo step 5 before Session 2.

---

## Session 2 — overnight, ≤ 16 h: the first part of the queue (K8 Tier A first)

### 14. Run S2
*Andrei runs under `caffeinate -i`; ≤ 16 h; AC power, High Power mode, lid open, nothing else heavy running.*

**Power log, second terminal** (Andrei, before the session; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. The logger stays on overnight. When the session has ended, including any resume of it, Andrei presses Ctrl-C in the logger's terminal; then Claude runs `"$PY" tools/power_log.py` before step 15. The record measures each start/stop pair in `results/raw/S2/session.jsonl`.

The session:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && { "$PY" tools/preflight.py --quick; rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; } && \
caffeinate -i "$PY" runner/run.py session --name S2
```
- Preflight exit 0 or 2 (warnings only) starts the session; exit 1 (a problem) stops the block. Expected warnings: the uncommitted `results/` tree on a resume or after a `runner/run.py status` call (`working tree has N changed paths in the experiment dir`; the session's own guard allows `results/`, `aborted/` and `evidence/`), and the sysctl advice if step 5 was declined. Any power-source (not AC), Low Power or not-High-Power, or thermal warning means Ctrl-C and fix it before leaving the machine.
- Before Andrei starts it, Claude runs `"$PY" tools/status.py --tier2` (after the pull). `no` is not a stop: the B4 context ladder is the only cell that needs the Tier-2 amendment, and the session skips it.
- The session checks P2's pins against the newest gate record at its start and refuses on a mismatch (the version freeze).
- Every Kolibri cell runs at a B within allowed_B; the runner refuses any other.
- Progress at any time (Claude may run; seconds): `"$PY" runner/run.py status`.
- Ctrl-C stops cleanly after the current records. Rerunning the same command resumes from the missing keys, and a started cell always completes.
- After a crash, rerun the same command. `run.py` finds the stale heartbeat and resumes the cell at the next lower B (for K8 and K4 within allowed_B), at most twice; then, or when there is no lower B, it moves the cell to `aborted/` and continues the queue. **A cell running at B = 1 has no fallback:** any stale heartbeat (a power cut, a macOS update reboot, a kernel panic) aborts it, and its hypotheses are NOT RUN. Keep the laptop on AC with updates deferred (step 0c).

Check: `status` shows the S2 part of the queue complete, or the remaining cells and an ETA. Truncation and parse counters per cell are visible, but no accuracy is shown (scoring stays blind until after S3).

Outputs: `results/raw/S2/<arm>/*.jsonl` and the step logs `results/raw/S2/<arm>/*.steps.jsonl` (working files: git-ignored, never committed, kept on the mbp and copied to the mini at step 18; step 15's record block lists their sha256). Withheld sets carry `text_sha256` only; full text goes to `$EXP036_PRIVATE/exp037/raw/S2/`. The power record `results/power/power_<UTC>.json` (descriptive) goes into step 15's commit with the raw outputs (its `git add …/results` takes it; never the raw `.plist`).

### 15. Commit Session 2
*Claude may run; ≈ 2 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --record-block "S2" && \
git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate/results tasks/chronos/exp_037_kolibri_forced_gate/aborted tasks/chronos/exp_037_kolibri_forced_gate/evidence tasks/chronos/exp_037_kolibri_forced_gate/HYPOTHESIS.md && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_037: S2 — raw outputs (<n> records, <cells> cells)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: the push succeeded. `evidence/withheld_manifest.jsonl` is updated with the sha256 of every new private file. Cells the crash fallback moved to `aborted/` go into the same commit as their `not_run.jsonl` lines. The S2 record block lists the sha256 of every `.jsonl` file under `results/raw/S2/`, the git-ignored step logs included, and of the step logs of cells aborted in S2 (`aborted/*/S2-*.steps.jsonl`, also git-ignored); the commit holds no session step log.
- **A leak-check finding on `aborted/<UTC>-torn-<arm>-<stem>/torn.part`** (the torn tail of a line that a crash cut, copied aside by `runner/jsonl`; held to the strict rules, so an address a model wrote is a finding there, not a warning as in `results/raw/`): Claude reads the finding; if it is model text, not a leak, leave that `torn.part` uncommitted exactly as step 13 leaves an oversized pilot log (unstage it, add its root-anchored path to the mbp's `.git/info/exclude`, rerun from the `git add` line). Its NOTE.md, committed, keeps the torn bytes' sha256. Anything else stops the step and goes to Andrei. The same applies at step 17.

---

## Session 3 — overnight, ≤ 16 h: the rest of the queue, then Tier B

### 16. Run S3 (and S3b if needed)
*Andrei runs under `caffeinate -i`; ≤ 16 h.*

**Power log, second terminal** (Andrei, before the session; descriptive, see "Power log"):
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
mkdir -p "$EXP036_WORK/exp037/power" && \
sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power -i 1000 -f plist > "$EXP036_WORK/exp037/power/power_$(date -u +%Y%m%dT%H%M%SZ).plist"
```
Claude checks it with `"$PY" tools/power_log.py --check`. If S3b follows, the same logger keeps running through it. After the last session command, Andrei presses Ctrl-C in the logger's terminal; then Claude runs `"$PY" tools/power_log.py` before step 17. The record measures each start/stop pair of S3 and S3b, and the B4 ladder cell if it ran.

The session:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && { "$PY" tools/preflight.py --quick; rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ]; } && \
caffeinate -i "$PY" runner/run.py session --name S3
```
- Preflight exit 0 or 2 (warnings only) starts the session; exit 1 (a problem) stops the block. Expected warnings: the uncommitted `results/` tree on a resume or after a `runner/run.py status` call (`working tree has N changed paths in the experiment dir`), and the sysctl advice if step 5 was declined. Any power-source (not AC), Low Power or not-High-Power, or thermal warning means Ctrl-C and fix it before leaving the machine.
- A cell started in S2 finishes first, then the queue continues in `plan_fixed` order.
- Tier-B cells that have not started by the end of S3 are NOT RUN. B4 runs only if the Tier-2 amendment is at HEAD (`"$PY" tools/status.py --tier2` says which).
- **If S3 ends with Tier-A cells outstanding,** `run.py` prints `Tier-A outstanding — run S3b now`. Run it at once; there is no decision to make:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
  caffeinate -i "$PY" runner/run.py session --name S3b
  ```
  S3b admits a Tier-A cell only while the projected run total stays within the 44 h overrun ceiling; anything it cannot admit is recorded NOT RUN.

Check: `status` shows the queue complete, with any NOT RUN cell listed and its reason.

Outputs: `results/raw/S3/<arm>/*.jsonl` (and `S3b/`), with their step logs `*.steps.jsonl` beside them (working files, as in step 14) and private text as in step 14. A cell started in S2 keeps writing under `results/raw/S2/` when it finishes in S3. The power record `results/power/power_<UTC>.json` (descriptive) goes into step 17's commit (its `git add` takes it; never the raw `.plist`).

### 17. Score on the mbp and commit
*Claude may run; ≈ 5–10 min.* Scoring needs the Tier-2 amendment at HEAD. The block stops at `--tier2` when it is not there yet:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
git -C "$LFA" pull --ff-only && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --tier2 && \
"$PY" scorers/score_all.py && \
"$PY" tools/version_record.py && \
"$PY" tools/status.py --record-block "S3, mbp scores" && \
git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_037: S3 — raw outputs; mbp scores" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check:
- `results/scores/<arm>/*.jsonl` exist for every completed cell except IFBench (scored on the mini). `scorers/score_all.py` reads the withheld raw records from `$EXP036_PRIVATE/exp037/raw/` and the withheld manifests from `$EXP036_PRIVATE/manifests/`.
- The scores carry item hashes, extracted answers and categories, and no withheld text.
- The S3 record block (`"S3, mbp scores"` here, `"S3"` below) lists the sha256 of every `.jsonl` file under `results/raw/S3/`, `results/raw/S3b/` and `results/raw/S2/` in its final state (S2 cells finished in S3 included), the git-ignored step logs included; the step logs of cells aborted in any of the three (`aborted/*/S2-*.steps.jsonl`, `S3-*` and `S3b-*`; an S2 cell aborted in S3 keeps its `S2-` name); and with `mbp scores` every `.jsonl` file under `results/scores/`. The commit holds no session step log.
- The leak check is clean.

**If `--tier2` printed `no`:** commit the raw outputs without scores, tell Andrei, and keep this session open:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/status.py --record-block "S3" && \
git -C "$LFA" add tasks/chronos/exp_037_kolibri_forced_gate/results tasks/chronos/exp_037_kolibri_forced_gate/aborted tasks/chronos/exp_037_kolibri_forced_gate/evidence tasks/chronos/exp_037_kolibri_forced_gate/HYPOTHESIS.md && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_037: S3 — raw outputs (scores wait for the Tier-2 amendment)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
When the main session has pushed the amendment, this mbp session reruns the first block of this step.

### 18. Mirror the withheld outputs and the session step logs to the mini (required)
*Andrei runs; ≈ 1–5 min.* The mini re-scores every set, including GPQA, RGB and AIME-DE, byte for byte; `analysis/verdicts.py` refuses to run without it. `$MINI` is set in Andrei's shell and is never written into the repo. The second line copies only the git-ignored session step logs (`results/raw/<session>/<arm>/*.steps.jsonl` and an aborted cell's `aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl`) to the same relative paths in the mini's checkout, where their run-record sha256 can be checked; the committed pilot step logs reach the mini with git. Nothing else is copied, and nothing on either side is deleted. An unset or empty `$EXP036_PRIVATE`, `$EXP` or `$MINI` stops the block at its line (`${…:?}`); unguarded, an empty `$EXP` would make the source `/`.
```bash
rsync -a "${EXP036_PRIVATE:?}/" "${MINI:?}:models/exp036-mini/private/" && \
rsync -a --prune-empty-dirs --include='*/' --include='/results/raw/**/*.steps.jsonl' --include='/aborted/*/S*-*.steps.jsonl' --exclude='*' "${EXP:?}/" "${MINI:?}:REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate/"
```
- **If step 13 left a pilot step log uncommitted** (over 50 MB), copy each such file by its kit-relative path: `( cd "${EXP:?}" && rsync -a --relative "<path>" "${MINI:?}:REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate/" )`. `--relative` with a relative path recreates its folders on the mini, so the copy does not depend on the mini having pulled step 13's commit (openrsync ignores the `/./` marker; checked locally on the mini, 2026-10-06). The main session then adds the same anchored line to the mini's `.git/info/exclude`, so that no commit on the mini takes it.

Check: every rsync exits 0. Then, on the mini (the main session, after `git -C "$LFA" pull --ff-only`, with the block header of step 19):
- `"$PY" tools/status.py --verify-private` prints `"ok": true`: every private file `evidence/withheld_manifest.jsonl` lists is on the mini at its latest sha256. Files outside `exp037/` (exp_036's records and the reused manifests) are reported as expected extras, not a mismatch.
- The step-log check (read-only) exits 0: every `*.steps.jsonl` path the run-record blocks list is present at the sha256 of its newest listing. The ones that need it are the session step logs and any pilot step log step 13 left uncommitted, which arrive with the rsync above; the committed pilot step logs arrive with git. One exception is built in: a session step log that a cell abort moved after a block listed it (an S2 cell listed by step 15 and aborted in S3) is missing by design. The check drops such a missing path only when an `aborted/<tag>/NOTE.md` names its move (`runner/run.py` `abort_cell` writes `` - Moved `<path>` here as `<name>` ``) and a block lists the moved copy `aborted/<tag>/<name>`, which it then checks like any other (the S3 block lists it). Any other missing or changed path is named and gives exit 1, as does a record with no step log.
  ```bash
  "$PY" -c "import hashlib, pathlib, re; P = pathlib.Path; t = P('HYPOTHESIS.md').read_text(encoding='utf-8'); last = dict(re.findall(r'^  - \x60(\S+\.steps\.jsonl)\x60 \x60([0-9a-f]{64})\x60', t, re.M)); pairs = [(s, 'aborted/' + n.parent.name + '/' + d) for n in P('aborted').glob('*/NOTE.md') for s, d in re.findall(r'Moved \x60([^\x60]+)\x60 here as \x60([^\x60]+)\x60', n.read_text(encoding='utf-8'))]; moved = sorted(p for p in last if not P(p).exists() and any(s.endswith('/' + p) and d in last for s, d in pairs)); bad = sorted(p for p, h in last.items() if p not in moved and not (P(p).is_file() and hashlib.sha256(P(p).read_bytes()).hexdigest() == h)); print(len(last), 'step logs in the run records,', len(moved), 'moved by a cell abort (NOTE.md names the move; the copy is listed and checked):', ('missing or changed: ' + ', '.join(bad)) if bad else 'every other one present at its newest sha256'); raise SystemExit(1 if bad or not last else 0)"
  ```

### 19. Hand-back
*Claude may run; seconds.* The mbp session writes nothing further.
1. Confirm `git -C "$LFA" status` is clean and HEAD equals origin.
2. Tell Andrei that Session 3 is pushed and the private mirror is done.

From here the main session on the mini is the only writer of HYPOTHESIS.md. Its block header uses the mini's exp_037 venv; exp_036's git-ignored local override on the mini sets `EXP036_MODELS` to `~/models/exp036-mini`, so `$EXP036_PRIVATE` is the mirror of step 18:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
export EXP037_VENV="$HOME/models/exp037-mini/venv312" && \
cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh && \
"$PY" tools/status.py --verify-private && \
"$PY" scorers/score_all.py --ifbench && \
"$PY" scorers/score_all.py --rescore-compare --ifbench && \
"$PY" analysis/verdicts.py && \
"$PY" tools/status.py --record-block "verdicts"
```
- `--ifbench` scores IFBench in its own venv (`$EXP036_IFBENCH_PY`, else `<models>/ifbench-venv/bin/python`; NLTK data `$EXP036_NLTK_DATA`, else `<models>/nltk_data`).
- `--rescore-compare --ifbench` re-scores every set and writes `results/rescore_mini_<UTC>.json`, which must show byte-identical scores.
- `analysis/verdicts.py` refuses without a matching re-score. It computes H1–H8 and D1, the H2 tripwire after H2 and its protocol control, the labels or caveat it implies, and E1–E12.
- The `verdicts` record block hashes the mini's result files: `results/verdicts_<UTC>.json` and `.md`, `results/rescore_mini_<UTC>.json` and the IFBench scores `results/scores/<arm>/ifbench_*.jsonl`, which only the mini writes and no mbp record block lists.
- Then the HYPOTHESIS results block and the Status line, the scientific_log result block and the README results, and the post draft (HYPOTHESIS "Publication angle", Order).

Outward steps after this point (the results push, `deploy.sh`, SFTP, the Hugging Face upload) are Andrei's go.

---

## Quick reference

| Step | What | Who | Duration (nominal / pessimistic) | Commit |
|---|---|---|---|---|
| 0 | Repo access, assets and venv present, hygiene | Claude, Andrei | minutes | — |
| 1–3 | Pull and version record, identity and hook, venv + tests | Claude | ≈ 30 min | (records with 4) |
| 3b | Optional dry run on tiny checkpoints (`tools/dry_run.py`) | Claude | ≈ 12 min | — |
| 4 | Preflight quick (Claude) and deep (Andrei, `caffeinate -i`) | both | 0.1 / 0.3 h | `preflight <UTC> quick` (4a, before 4b), `preflight <UTC>` (4b) |
| 5 | `sudo sysctl iogpu.wired_limit_mb=114688`, only if printed | Andrei | seconds | — |
| 6 | Sign-off (+ Amendment 0 only for the H2 margin) | Andrei, then Claude | ≈ 15 min | `sign-off — Andrei` |
| 7 | Verify the inherited manifests, shingles, T5/T6/T9 | Claude | 2 min | `inherited manifests verified` |
| 8 | Clone and refresh K8, K4 (`tools/refresh_builds.py`) | Andrei, `caffeinate -i` | 0.1 / 0.2 h | `K8/K4 clone-and-refresh records` |
| 9a | Gemma fidelity record under MLX 0.32.3 (not an S1 block) | Andrei, `caffeinate -i` | ≈ 0.2 h | `Gemma fidelity record` |
| 9 | Peer check | Andrei, `caffeinate -i` | 0.4 / 0.6 h | `peer check` |
| 10 | Gate (STOP on exit 1, 4 or 5; repeat on 3) | Andrei, `caffeinate -i` | 3.0 / 4.4 h | `gate <UTC> — PASS\|FAIL\|INCOMPLETE` |
| 11 | Bench cells (H1, D1, H5, H8, C1) | Andrei, `caffeinate -i` | 1.2 / 1.7 h | `S1 bench cells` |
| 12 | Pilot (interrupted: `abort-pilot`, rerun in full) | Andrei, `caffeinate -i` | 1.3 / 1.9 h | (with 13) |
| 13 | Plan-fix, plan amendment | Claude | seconds | `amendment <k> — plan fixed by rule` |
| 14–15 | Session 2 | Andrei, `caffeinate -i`; then Claude | ≤ 16 h | `S2 — raw outputs` |
| 16–17 | Session 3 (+ S3b if printed), scoring | Andrei, `caffeinate -i`; then Claude | ≤ 16 h | `S3 — raw outputs; mbp scores` |
| 18 | Private mirror and session step logs to the mini (required) | Andrei | minutes | — |
| 19 | Hand-back | Claude | seconds | — |

Steps 9–12, 14 and 16 also run the power logger in a second terminal ("Power log"; descriptive, never blocking); its records `results/power/power_<UTC>.json` go into each step's commit, and the raw `.plist` logs never leave `$EXP036_WORK/exp037/power/`.

S1 is ≈ 6.2 h nominal, 9.2 h pessimistic and 9.7 h adverse of machine time; the run total ≈ 35.9 h nominal, 38.6 h pessimistic and 40.0 h adverse (HYPOTHESIS, "Sessions & budget"). The automatic S3b may extend it to the 44 h overrun ceiling, never beyond.
