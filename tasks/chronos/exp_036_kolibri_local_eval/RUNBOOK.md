# Experiment 036 — runbook (MacBook Pro M5 Max)

*For Andrei and the Claude session on the mbp. The steps run in order; none is skipped or reordered. [`HYPOTHESIS.md`](./HYPOTHESIS.md) is the pre-registration and wins over this file; [`BUILD_SPEC.md`](./BUILD_SPEC.md) describes the kit. Nothing in this runbook runs on miktam-mini except step 18's target and the hand-back. Hand-off is through GitHub (`miktam/local-first-ai`).*

**Who runs what:**
- **Andrei runs in his terminal under `caffeinate -i`** — any job longer than a few minutes, or anything that loads model weights. The mbp Claude session prepares the exact command; Andrei pastes it and keeps the laptop on AC power, lid open, in High Power mode.
- **Claude may run** — short, weight-free steps: git, venv, tests, quick preflight, manifests, plan-fix, scoring, leak check, commit and push.
- **Andrei by hand** — `sudo` lines, the sign-off block (Claude never fills any sign-off field), and any threshold amendment.

**Pushes.** Pre-approved by Andrei (hand-off decision, 2026-10-03): pushes from the mbp of the sign-off, manifests, records, amendments and S2/S3 raw outputs and scores, each after the leak check (steps 4–17). Needs Andrei's go: the mini's verdict and results push, any README or scientific_log result push, `deploy.sh`, SFTP, HF upload, PRs, social posts.

**Every Claude-run block** starts with `cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env &&`, because shell variables do not persist between Claude's commands. `env/exp036.env` sets `EXP036_MODELS` (default `~/models/exp036`), `EXP036_DATA`, `EXP036_PRIVATE`, `EXP036_WORK`, `LFA`, `EXP`, `PY="$EXP036_MODELS/venv/bin/python"`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Andrei's terminal sources the same file once.

**Every commit**:
- uses the prefix `chronos/exp_036:` and identity `Miktam <hello@localfirstai.eu>`;
- is preceded by `"$PY" tools/leak_check.py --range @{u}..HEAD --staged` (a finding blocks it; raw-output warnings go into the commit body);
- ends with the trailer: `git -C "$LFA" commit -m "<subject>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`;
- is followed by `git -C "$LFA" status --porcelain`, which must print nothing outside `results/`, `aborted/` and `evidence/`.

The pre-push hook (step 2) runs the leak check again on every push.

**HYPOTHESIS.md has one writer during the run.** From the pre-registration push to step 19, only this mbp session appends to HYPOTHESIS.md. The main session on the mini pushes any amendment it writes (gate fix, extractor fix, Tier-2 analysis) as a file `amendments/<k>_<type>_<UTC>.md` plus code, and never edits HYPOTHESIS.md. The mbp session pulls (`git -C "$LFA" pull --ff-only`) before every commit, and `"$PY" tools/status.py --sync-amendments` appends any `amendments/` file not yet in HYPOTHESIS.md, verbatim, in that commit.

**Never** move or copy a file from `$EXP036_PRIVATE` into the repo, and never delete anything: broken files go to `aborted/<UTC>-<what>/` with a `NOTE.md` (private ones to `$EXP036_PRIVATE/aborted/`, with only their sha256 and byte count in the repo note).

---

## 0. One-off preparation (before Session 1)

**0a. Repo and push access.** *Claude may run; seconds.*
```bash
[ -d ~/REPOS/local-first-ai ] || git clone git@github.com:miktam/local-first-ai.git ~/REPOS/local-first-ai
git -C ~/REPOS/local-first-ai pull --ff-only && git -C ~/REPOS/local-first-ai push --dry-run
```
Check: the clone or pull and the dry run succeed (SSH key and access). If the repo lives elsewhere, set `LFA` in the git-ignored local override `env/exp036.local.env` and use that path in every block.

**0b. Downloads finished.** *Andrei first accepts the two GPQA gates; then Claude may run; seconds.*

**GPQA gate acceptance (Andrei, once, before the GPQA downloads).** Both GPQA datasets are gated: on huggingface.co, logged in as the account the `hf` CLI uses, Andrei accepts the terms on the pages of `Idavidrein/gpqa` and `ellamind/gpqa-multilingual`. Without that, `hf download` writes only `README.md` and the build of the GPQA manifests at step 7 fails. GPQA's terms forbid revealing examples online; the kit commits only ids and hashes.
```bash
ls ~/models/exp036 ~/models/exp036/data
ls ~/models/exp036/data/gpqa ~/models/exp036/data/gpqa-multilingual
```
Check: every folder in `ASSETS.md` is present (≈ 303 GB). The GPQA folders contain data files (`gpqa_diamond.csv`, `gpqa_main.csv`; the `deu` config), not only `README.md`. The hashes are verified at step 4.

**0c. The assets added on 2026-10-03.** *Andrei runs; about 5 min.* These are the four entries of ASSETS.md "Added 2026-10-03" (pinned in `assets.json`; ASSETS.md shows the same commands). This is the only networked step, so it lifts the offline flags for itself. The FineWeb-2 file is named **positionally**: `hf download --include …` silently skipped files with hf 2.1.1 on the mini.
```bash
M=~/models/exp036; D=$M/data; A=~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval/assets.json
RGB_REV=$("$M/venv/bin/python" -c "import json,sys;print(next(c['commit'] for c in json.load(open(sys.argv[1]))['code'] if 'chen700564/RGB' in c['repo']))" "$A")
# FineWeb-2 deu_Latn test shard (104 MB): H5, gate texts T5/T6/T9
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE hf download HuggingFaceFW/fineweb-2 data/deu_Latn/test/000_00000.parquet \
  --repo-type dataset --revision af9c13333eb981300149d5ca60a8e9d659b276b9 --local-dir "$D/fineweb-2"
# RGB at its pin (code and data CC BY-NC-SA 4.0; scores only are published)
git clone https://github.com/chen700564/RGB "$D/RGB-src" && git -C "$D/RGB-src" checkout "$RGB_REV"
# IFBench checker environment, from IFBench's own lock file (Python 3.12)
(cd "$D/IFBench-src" && UV_PROJECT_ENVIRONMENT="$M/ifbench-venv" uv sync --frozen --python 3.12)
# NLTK data, so IFBench scoring never downloads at run time
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE "$M/ifbench-venv/bin/python" -c "import nltk,sys; [nltk.download(p, download_dir=sys.argv[1]) for p in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger_eng']]" "$M/nltk_data"
```
Check:
- `RGB-src/data/en.json`, `en_fact.json`, `en_int.json`, `config/instruction.yaml`, `evalue.py`, `readme.md` (lower-case at this commit). There is **no** `config/instruction_fact.yaml` at this commit; Fact-Check uses `config/instruction.yaml` with the `positive_wrong` documents (`tasks/selection_rules.json` "rgb" › "fact"). `data/en_refine.json` also exists and is not used (a recorded limitation).
- `fineweb-2/data/deu_Latn/test/000_00000.parquet`, 104,264,728 bytes.
- `ifbench-venv/bin/python` exists and `nltk_data/` holds the four packages.
- If `RGB_REV` comes out empty, take the RGB commit from ASSETS.md. Preflight (step 4) checks every revision and recorded sha256.

**0d. Machine hygiene.** *Andrei by hand; once.*
```bash
tmutil addexclusion ~/models/exp036                 # keep the model folder out of Time Machine
touch ~/models/exp036/.metadata_never_index         # keep Spotlight out
```
- System Settings › Battery › Energy Mode (on power adapter): **High Power**. Low Power Mode off.
- Defer automatic macOS updates for the experiment window (an update reboots and resets any sysctl).
- Disk: keep ≥ 600 GB free in total (downloads ≈ 303 GB, conversions ≈ 127 GB, gate and H8 work files ≈ 100 GB, margin). The mbp had 2.9 TB free on 2026-10-03.

---

## Session 1 — daytime, ≈ 5.4 h nominal / 9.1 h pessimistic of machine time

S1 hours are the sum of the machine intervals recorded by its blocks, not wall-clock time; Andrei's reading time and any gate fix cycle do not count. A gate fix cycle moves the rest of S1 to a later day.

### 1. Pull
*Claude may run; seconds.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && \
git -C "$LFA" log --oneline -6 -- tasks/chronos/exp_036_kolibri_local_eval && \
git -C "$LFA" status --porcelain && \
grep -c '{{' HYPOTHESIS.md; grep -c '^Pre-registration commit:' HYPOTHESIS.md
```
Check:
- HEAD contains the commit named on HYPOTHESIS.md's last line, `Pre-registration commit: 725d628a…` (actual subject: `chronos/exp_036: pre-registration, MLX port and kit, awaiting Andrei's sign-off`). Check it by sha, not by subject: `git merge-base --is-ancestor "$(sed -n 's/^Pre-registration commit: //p' HYPOTHESIS.md)" HEAD && echo ok`.
- `git status --porcelain` prints nothing: the **whole** checkout is clean. The runner refuses pilot, session and cell runs while any path outside the kit's `results/`, `aborted/` and `evidence/` is modified or untracked, anywhere in the repository; unrelated local work (blog, other experiments) is committed elsewhere or stashed before Session 1.
- The first `grep -c` prints `0`, the second `1` (the `Pre-registration commit:` line exists).

Outputs: none. No commit.

### 2. Git identity and pre-push hook
*Claude may run; seconds.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" config user.name  "Miktam" && \
git -C "$LFA" config user.email "hello@localfirstai.eu" && \
HOOKS="$(git -C "$LFA" rev-parse --path-format=absolute --git-path hooks)" && \
cp tools/hooks/pre-push "$HOOKS/pre-push" && chmod +x "$HOOKS/pre-push" && \
test -x "$HOOKS/pre-push" && echo "pre-push hook installed" && \
git -C "$LFA" config --get user.name && git -C "$LFA" config --get user.email
```
Check: it prints `pre-push hook installed`, `Miktam` and `hello@localfirstai.eu`. Every kit entry point refuses to write results otherwise. `--path-format=absolute` matters: a plain `--git-path hooks` is relative to the repository top (`.git/hooks`), not to this directory. No commit.

### 3. venv sync and the unit tests on the mbp
*Claude may run; ≈ 4–6 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
bash env/setup.sh && \
EXP036_TOK="$EXP036_MODELS/Kolibri-1-BF16" EXP036_REQUIRE_ALL=run-host "$PY" -m pytest -q -rs tests
```
- `env/setup.sh` installs the exact pins of `env/requirements-mbp.txt` into `$EXP036_MODELS/venv` with uv (the venv was built with uv and has no pip; `"$PY" -m pip` fails there). It is idempotent and never calls sudo. The package versions are recorded by preflight (step 4) against `env/versions.json`.
- `EXP036_REQUIRE_ALL=run-host` turns every skip into a failure, except a skip whose reason starts with `build-host only:`: the few tests that compare with files only the mini holds (the upstream peer tokenizer and template files, the eval-framework checkout). They passed on the mini before the push; `-rs` lists them.

Check:
- `env/setup.sh` prints mlx 0.31.2, mlx-metal 0.31.2 and mlx-lm 0.31.3.
- All tests pass; the only skips are `build-host only:`. These are the weight-free tests already passed on the mini, rerun here on Python 3.12 and the M5 Max.
- The tests that read a copy of the live HYPOTHESIS.md (the hash-table round trip, the Status line) do not depend on its state: filled, signed off or amended, they pass, here and in the gate's G1 at step 10.
- A failure stops the runbook. The test output goes to the main session through a pushed note (`aborted/<UTC>-tests/NOTE.md`, commit `chronos/exp_036: mbp unit tests failed — note`). Paths in the note are written as `~/…` or `$EXP036_MODELS/…`.

Outputs: none in `results/`. No commit when the tests pass.

**3b. Optional self-test: the dry run.** *Claude may run; ≈ 3–5 min (3.0 min on the mini); weight-free (tiny random checkpoints).* It runs the whole pipeline end to end in a temporary copy of the kit (its own git repository and bare remote, tiny Kolibri arms and tiny stand-ins for every peer, synthetic task data): preflight → convert K8/K4 → gate on tiny → bench cells at seconds scale → pilot → plan fix → a short S2/S3 queue → scorers → verdicts → `hash_tree --check` → leak check. Nothing is written into this repository, `$EXP036_MODELS` or `$EXP036_PRIVATE`. It needs ≈ 15 GB of free temporary disk while it runs (the H8 log-prob dump of the tiny model has the real vocabulary) and deletes it at the end (`--keep` keeps it for inspection).
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tools/dry_run.py --tok "$EXP036_MODELS/Kolibri-1-BF16" --peers "$EXP036_MODELS"
```
Check: the last stdout line is JSON with `"ok": true`; exit 0. A failure names the stage; send it to the main session like a test failure. The dry run is not part of the evidence and is never committed.

The dry run is valid at any step, including a rerun during a gate fix cycle: its copy of the kit is first reset to the pre-registration state (the hash table and the stamp back to their placeholders, `hash_tree:` amendment lines neutralised, the real item manifests of step 7 and `tools/withheld_shingles.sha256` removed), so it fills, scales and builds its own.

**3c. Optional: the loaders on the real data.** *Claude may run; seconds.* The GPQA-DE, MMLU-ProX(-Lite) and AIME26 loaders were built without those datasets on the mini (their file layouts are inferred and fail loudly when wrong). This builds every manifest set from the real data into `$EXP036_WORK` only, with the counts enforced; nothing is written into the repository (no shingle file, no results record).
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
SETS=$("$PY" tasks/build_manifests.py --list | cut -f1 | paste -sd, -) && \
"$PY" tasks/build_manifests.py --sets "$SETS" --out "$EXP036_WORK/manifest_check/public" --private "$EXP036_WORK/manifest_check/private"
```
Check: the last line is JSON with `"ok": true` and the counts GPQA-D EN 198 (one over-long item flagged out of the primary 197), GPQA-D DE 198, MMLU-ProX-Lite EN/DE 588 each, AIME EN/DE 30, IFBench 300, RGB closed-book 400, Negative 300, Fact-Check 100. A failure goes to the main session before step 6.

### 4. Preflight
**4a. Quick.** *Claude may run; < 1 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tools/preflight.py --quick
```

**4b. Deep.** *Andrei runs under `caffeinate -i`; minutes (hashes 303 GB against the LFS oids).*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" tools/preflight.py --deep
```
Check:
- Exit code 0.
- Chip `Apple M5 Max`, memory 128 GB, power mode High Power on AC.
- `mx.device_info()` and `iogpu.wired_limit_mb` are recorded. The value measured on 2026-10-03 was 107.5 GiB with `iogpu.wired_limit_mb` = 0.
- Every asset's revision matches assets.json, and every shard's sha256 equals its etag.
- The exp_007 fixtures are found by their real names.
- Free disk ≥ 260 GB beyond the downloads.
- Read the **K8 working-set line**. It shows the memory-rule B per K8 cell and either "no sysctl needed" or a `sudo sysctl` line with the B it would gain.

Outputs: `results/preflight_<UTC>.json` and `results/version_record_<UTC>.json`.

Commit (Claude): `chronos/exp_036: preflight <UTC> (M5 Max, L=<GiB>)`, then push.

### 5. Metal working set for K8 (only if 4b printed a sysctl line)
*Andrei by hand; seconds.*
```bash
sudo sysctl iogpu.wired_limit_mb=114688      # 112 GiB, leaves ≈ 16 GiB for macOS; resets at reboot
```
Then Claude re-runs step 4a. Check: the quick preflight reports the new limit and K8's B per cell.
- If Andrei declines, the plan rule uses the default limit. If even B = 1 does not fit, K8 is NOT RUN and K4 becomes primary through the plan amendment (HYPOTHESIS, K8 working-set rule).
- Andrei writes the value, "not needed" or "declined" into the sign-off. **After any reboot, redo this line before steps 10–17;** `run.py` refuses K8 cells if the limit is below the one the plan used.

### 6. Sign-off
*Andrei types; Claude commits and pushes; ≈ 15 min of Andrei's reading.*

Andrei fills the **Sign-off** block at the end of `HYPOTHESIS.md` himself, in an editor on the mbp:
- `- Signed off by: Andrei (typed by Andrei on the mbp, <UTC>)`;
- the UTC time from the clock;
- the H2 margin (−4 pp default, −3 or −5);
- the embed/head policy (default quantised at the arm's bits with fp32 logits; vendor-faithful likely makes H1 REFUTED by itself);
- the K8 Metal value from step 5;
- the cloud anchor (default no).

**If a choice differs from the pre-registered default** (Amendment 0, before any scored run):
- for the margin, Claude edits `analysis/margins.json` (`"H2": -0.03` or `-0.05`) and appends `## Amendment 0 — sign-off choices (<UTC>)` to HYPOTHESIS.md, quoting Andrei's choice and giving the new `analysis/` tree value as the line `"$PY" tools/hash_tree.py --amend-line analysis` prints (`hash_tree: ANALYSIS_SHA256 = <hex>`), pasted verbatim: `hash_tree --check` reads only that form inside an amendment;
- for the head, Amendment 0 records it and step 8 adds `--no-quantize-embeddings --no-quantize-lm-head` to **both** conversions.

Claude then runs:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tools/status.py --record-block "sign-off" && \
git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval/HYPOTHESIS.md tasks/chronos/exp_036_kolibri_local_eval/analysis/margins.json && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_036: sign-off — Andrei, <UTC>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: the push succeeded, and the status prints nothing outside `results/`, `aborted/` and `evidence/`. Every later runner command checks the "Signed off by" line's form at HEAD.

### 7. Item manifests, withheld shingles and the web gate texts
*Claude may run; ≈ 2 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tasks/build_manifests.py && \
"$PY" gate/build_gate_text.py --work-only --fineweb "$EXP036_DATA/fineweb-2/data/deu_Latn/test/000_00000.parquet"
```
Check:
- Both exit 0.
- `tasks/manifests/` holds hash-only manifests for GPQA EN/DE, RGB and AIME-DE (no `text`, no `gold`), and manifests with category and gold for MMLU-ProX and AIME EN; `mmlu_prox_category_counts.json` is present.
- `$EXP036_PRIVATE/manifests/` holds the text and gold of the withheld sets. The RGB document indices are recorded.
- `tools/withheld_shingles.sha256` exists.
- T5, T6 and T9 are in `$EXP036_WORK/gate_texts/`; their indices and sha256 match `gate/texts/MANIFEST.json` if the main session filled them, otherwise they are written to `results/gate_texts_<UTC>.json`. No web text is in the repo.

Outputs: `tasks/manifests/*.json`, `tools/withheld_shingles.sha256`, `results/manifests_<UTC>.json`, possibly `results/gate_texts_<UTC>.json`.

Commit (Claude, after the leak check, which from now on uses the shingle list): `chronos/exp_036: item manifests (ids and hashes only)`. Push.

### 8. Convert K8 and K4
*Andrei runs under `caffeinate -i`; ≈ 0.5–1.5 h for both.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" port/convert.py --src "$EXP036_MODELS/Kolibri-1-BF16" \
  --out "$EXP036_MODELS/Kolibri-1-MLX-8bit-g64" --bits 8 --group-size 64 && \
caffeinate -i "$PY" port/convert.py --src "$EXP036_MODELS/Kolibri-1-BF16" \
  --out "$EXP036_MODELS/Kolibri-1-MLX-4bit-g64" --bits 4 --group-size 64
```
- Add `--no-quantize-embeddings --no-quantize-lm-head` to both lines only if the sign-off chose the vendor-faithful head.
- `convert.py` watches its own memory: if swap grows by > 2 GB or its RSS passes 0.8 × RAM, it stops, cleans up and prints the `--streaming` command to run instead.
- After a crash or Ctrl-C, rerun **only** the line whose output folder has no `exp036_convert_record.json` yet (an interrupted run leaves only its `.partial` / staging folder, which the rerun cleans up). Rerunning the whole block stops at the finished K8 line ("exists and is not empty"). Never pass `--force` on a folder that has `exp036_convert_record.json`.

Check:
- Each output has `exp036_convert_record.json` with the tensor policy of the sign-off.
- K8 ≈ 83.1 GB, K4 ≈ 44.0 GB (default head policy).
- The quantization block is `{64, 8 | 4, affine}`; the router weight is fp32.
- `kolibri1.py` and the tokenizer files were copied byte for byte.

Outputs: `$EXP036_MODELS/Kolibri-1-MLX-{8,4}bit-g64/`, plus `results/convert/convert_{8,4}bit_<UTC>.json` (copies of the records).

Commit (Claude): `chronos/exp_036: K8/K4 conversion records`. Push.

### 9. Peer check
*Andrei runs under `caffeinate -i`; ≈ 25–35 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" tools/peer_check.py --arms G8,G4,Q36-8,Q36-4,Q38-8,Q38-4
```
Check:
- Every arm says `ok`: strict text-only load; parameter count within 2 %; NLL(8) ≤ NLL(4) + 0.02; KL(8‖4) < 0.2; template and tokenizer parity recorded; EOS ids recorded.
- G8 and Q36-8: batched-path parity `ok` and greedy flip rate ≤ 2 %. A peer that fails only the batched path is marked `B=1` (the plan rule re-projects it); that is not a drop.
- A peer that fails anything else (verdict `fail`) is dropped by amendment before any scored run (HYPOTHESIS, Phase 0, "Peers are verified, not gated"). Stop and tell Andrei; the main session pushes `amendments/<k>_peerdrop_<UTC>.md`. The run then goes on: from step 12 the pilot runs with `--without <arm>` (the runner accepts it only for an arm the newest peers record marks `fail`), and `runner/plan_fix.py` queues no cell of that arm and records the drop in the plan amendment: H3 and H6 use the remaining MoE peer, and H4 is NOT RUN if Qwen3.6 is the one dropped. What a failed G4, Q36-4 or Q38-x does to H1 or H8 is not pre-registered; Andrei decides it by amendment before step 13.

Outputs: `results/peers_<UTC>.json` (also read by the gate's G3).

Commit (Claude): `chronos/exp_036: peer check <UTC>`. Push.

### 10. Phase 0 — port gate (blocking)
*Andrei runs under `caffeinate -i`; ≈ 1.8–3 h.* If step 5 applied and the laptop rebooted, redo step 5 first.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" gate/run_gate.py --all
```
Check: the last stdout line is JSON with `"verdict"`. Exit codes:
- **0** — K8 and K4 PASS. Continue.
- **4** — K8 PASS, K4 FAIL. Stop as for 1: a K4 bug check (G0, G2q, behaviour) failing means a conversion or port defect. If the fix cycles end with K4 still failing, the run continues without K4, and H1, H7, H8 and D1 are NOT RUN: step 11 runs `"$PY" bench/run_bench.py --cells speed_desc,tokenizer,c1` (the H1 speed, D1 fit and H8 KL cells need K4), step 12 runs `"$PY" runner/run.py pilot --without K4`, and `runner/plan_fix.py` reads the gate record, queues no K4 cell (nor B4 or B8) and records the four NOT RUN hypotheses in the plan amendment.
- **1** — K8 FAIL. Stop.
- **3** — could not run (e.g. disk). Fix the cause and rerun this step.

Outputs:
- `results/gate/gate_<UTC>.json`, `gate_<UTC>_layers.csv`, `gate_<UTC>_mutants.json`, and the per-phase files under `results/gate/<UTC>/`.
- Reference dumps in `$EXP036_WORK/ref/` (≈ 100 GB on a first run; not committed; sha256 in the gate JSON).

Commit (Claude), after `"$PY" tools/status.py --sync-amendments && "$PY" tools/status.py --record-block "gate"`:
- On PASS: `chronos/exp_036: gate <UTC> — PASS (K8, K4)`.
- On FAIL: `chronos/exp_036: gate <UTC> — FAIL (<first failing check>)`. Push.

**On FAIL (exit 1 or 4), stop here.**
1. No pilot, bench or scored run.
2. The main session on the mini diagnoses from the per-layer table and pushes a fix plus `amendments/<k>_gatefix_<UTC>.md`. A threshold change needs Andrei's own typed amendment (HYPOTHESIS, Verdict and refusal).
3. Then: step 1 (pull), `"$PY" tools/status.py --sync-amendments` (appends the new `amendments/` file to HYPOTHESIS.md), re-run step 3, and refresh the converted port file. A forward-code-only fix uses `"$PY" port/convert.py --refresh-port-file --out "$EXP036_MODELS/Kolibri-1-MLX-8bit-g64"` and the same for 4-bit (seconds). A sanitize or quantisation fix means rerunning step 8.
4. Re-run step 10 **in full** (matching reference dumps are reused automatically; port-side checks always rerun).
5. At most two fix cycles. After a third FAIL, the failure is the exp_036 result; partial outputs move to `aborted/` with a note, and the main session follows HYPOTHESIS "Gate failure or STOP".

### 11. Bench cells: H1, descriptive speed, D1, H8, H5, C1
*Andrei runs under `caffeinate -i`; ≈ 1.2–1.7 h.* Close other apps first and leave the laptop idle for 10 minutes so it starts cool; the script checks `thermalState`.
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" bench/run_bench.py --cells speed,speed_desc,fit,kl,tokenizer,c1
```
Check:
- Every cell file exists and reports `complete`.
- H1: 10 blocks for each of K4 and G4. D1: 3 reps at 32k and at 64k. H8: 4 models × 6 texts. H5: 5,000 documents. C1: 100 items at the preflight B and at B = 1.
- Power mode and thermal state logged.
- Do **not** use the laptop during the speed cells.
- Exit code 0: every cell complete. **3**: a cell could not run (thermals): let the laptop cool and rerun the same command. **1**: a cell failed: send the stderr to the main session. On a rerun, complete cells are skipped and an incomplete cell's output moves to `aborted/` before it reruns.

Outputs: `results/bench/{speed,speed_desc,fit,c1}_<UTC>.jsonl`, `results/tokenizer_<UTC>.json`, `results/kl_8v4_<UTC>.json`. K8 log-probs go to `$EXP036_WORK/kl/` and are not committed.

Commit (Claude): `chronos/exp_036: S1 bench cells — H1, D1, H5, H8 raw`. Push.

### 12. Pilot
*Andrei runs under `caffeinate -i`; ≈ 1.3–1.9 h.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
caffeinate -i "$PY" runner/run.py pilot
```
Check:
- `pilot_summary_<UTC>.json` lists, per arm and task: mean, SE and max completion tokens, truncation count, reasoning-status counts, parse failures and diagnoses, prefill rate; the per-step decode logs exist.
- The pilot outputs are **never** scored for accuracy.
- Without K4 (step 10, exit 4) or with a dropped peer (step 9): `caffeinate -i "$PY" runner/run.py pilot --without K4` (or `--without Q36-8`, or both, comma-separated).
- **Interrupted (Ctrl-C) or crashed:** the pilot writes no summary (after Ctrl-C it exits 1 and prints its `<UTC>`). Move the partial pilot aside, then rerun this step in full:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  "$PY" runner/run.py abort-pilot --stamp <UTC>
  ```
  It moves `results/pilot/<UTC>/` to `aborted/<UTC>-pilot/` with a NOTE.md, and the private files to `$EXP036_PRIVATE/aborted/<UTC>-pilot/`, with only their sha256 and byte counts in the note. Step 13 takes only a complete summary (`"complete": true`); `plan_fix` refuses any other.

Outputs: `results/pilot/<UTC>/*.jsonl` and `*.steps.jsonl` (withheld sets: text in `$EXP036_PRIVATE/pilot/`) and `results/pilot_summary_<UTC>.json`.

No separate commit; it is committed with step 13.

### 13. Fix n and max_tokens by rule (plan amendment)
*Claude may run; seconds, plus the commit.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && \
"$PY" runner/plan_fix.py --pilot results/pilot_summary_<UTC>.json
```
`plan_fix` reads S1 hours from the S1 records and the excluded arms from the gate and peer records. It prints the paths of the new `results/plan_fixed_<UTC>.json` and `results/AMENDMENT_<k>_<UTC>.md`, then a JSON line with the status. Exit 0: FIXED. Exit 2: STOP or BLOCKED (both files are still written; see Outcomes). Exit 1: refused input (an incomplete summary, a missing pilot cell, cells of an excluded arm): nothing is written. Then, for a FIXED plan:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --append-amendment results/AMENDMENT_<k>_<UTC>.md && \
"$PY" tools/status.py --record-block "plan" && \
git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_036: amendment <k> — plan fixed by rule (<plan>, n_M=<n>, Tier B <list>)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: `plan_fixed_<UTC>.json` gives:
- the plan P0…P10 or STOP;
- n_M, the caps (any raise), B per cell, the L used, the ordered queue and its S2/S3 split, the Tier-B items;
- the row sets of H2 and H7, the manifest and gate sha256s, the `scorers/` tree sha;
- projected hours: total ≤ min(31, 40 − S1 hours), each session ≤ 16 h.

Outcomes:
- **STOP** or **BLOCKED**: the amendment is still appended and pushed, with its own Status line and commit message:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  "$PY" tools/status.py --sync-amendments && \
  "$PY" tools/status.py --append-amendment results/AMENDMENT_<k>_<UTC>.md && \
  "$PY" tools/status.py --record-block "plan" --status "plan <STATUS> by rule; no scored run" && \
  git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval && \
  "$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
  git -C "$LFA" commit -m "chronos/exp_036: amendment <k> — plan <STATUS> by rule" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
  git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
  ```
- **STOP:** no scored run. Andrei decides on a 4th session by amendment.
- **BLOCKED cell** (≥ 2 parse failures with a diagnosed extractor or delimiter mismatch): after the commit above, the main session pushes an `amendments/` extractor or delimiter fix. Then pull, `"$PY" tools/status.py --sync-amendments`, and re-pilot only the BLOCKED cells, named exactly as the amendment's `BLOCKED cells:` line prints them (comma-separated; `ARM:TASK` works too), e.g. `caffeinate -i "$PY" runner/run.py pilot --cells K8:gpqa_main_en:high`. Then rerun this step with **both** summaries, the full pilot's first: `"$PY" runner/plan_fix.py --pilot results/pilot_summary_<first UTC>.json --pilot results/pilot_summary_<re-pilot UTC>.json`. The new plan amendment names the one it supersedes; nothing is overwritten.
- The commit must be pushed and HEAD must equal upstream; `runner/run.py session` refuses otherwise.

**End of Session 1.** If the laptop must reboot, the sysctl from step 5 resets: redo step 5 before Session 2.

---

## Session 2 — overnight, ≤ 16 h: the first part of the queue (K8 Tier A first)

### 14. Run S2
*Andrei runs under `caffeinate -i`; ≤ 16 h; AC power, High Power mode, lid open, nothing else heavy running.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && "$PY" tools/preflight.py --quick && \
caffeinate -i "$PY" runner/run.py session --name S2
```
- Before Andrei starts it, Claude runs `"$PY" tools/status.py --tier2` (after the pull). `no` is not a stop: the B4 context ladder is the only cell that needs the Tier-2 amendment, and the session skips it and runs the rest of the queue. The amendment counts once the mini's `amendments/<k>_tier2_<UTC>.md` is committed upstream and pulled; it need not be in HYPOTHESIS.md yet.
- Progress at any time (Claude may run; seconds): `"$PY" runner/run.py status`.
- Ctrl-C stops cleanly after the current records. Rerunning the same command resumes from the missing keys, and a started cell always completes.
- After a crash, rerun the same command. `run.py` finds the stale heartbeat and resumes the cell at the next lower B in {16, 8, 4, 2, 1} (the crash fallback), at most twice; then, or when there is no lower B, it moves the cell to `aborted/` and continues the queue. **A cell running at B = 1 has no fallback, and one at B = 2 only one:** the peers marked `B=1` at step 9 and memory-bound K8 cells run there. Any stale heartbeat counts as a crash, including a power cut, a macOS update reboot or a kernel panic, so one such event during a B = 1 Tier-A cell aborts it and its hypotheses are NOT RUN. Keep the laptop on AC with updates deferred (step 0d).
- `run.py` stops admitting new cells when the next one would take S2 past 16 h; it prints where S3 will start.

Check:
- `status` shows the S2 part of the queue complete, or the remaining cells and an ETA.
- Truncation and parse counters per cell are visible in `status`, but no accuracy is shown (scoring stays blind until after S3).

Outputs: `results/raw/S2/<arm>/*.jsonl` and `*.steps.jsonl`. Withheld sets carry `text_sha256` only; full text goes to `$EXP036_PRIVATE/raw/S2/`.

### 15. Commit Session 2
*Claude may run; ≈ 2 min.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --record-block "S2" && \
git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval/results tasks/chronos/exp_036_kolibri_local_eval/aborted tasks/chronos/exp_036_kolibri_local_eval/evidence tasks/chronos/exp_036_kolibri_local_eval/HYPOTHESIS.md && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_036: S2 — raw outputs (<n> records, <cells> cells)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check: the push succeeded. `evidence/withheld_manifest.jsonl` is updated with the sha256 of every new private file; `run.py session` writes it. Cells the crash fallback moved to `aborted/` (their NOTE.md) go into the same commit as their `not_run.jsonl` lines.

---

## Session 3 — overnight, ≤ 16 h: the rest of the queue, then Tier B

### 16. Run S3 (and S3b if needed)
*Andrei runs under `caffeinate -i`; ≤ 16 h.*
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && "$PY" tools/preflight.py --quick && \
caffeinate -i "$PY" runner/run.py session --name S3
```
- A cell started in S2 finishes first, then the queue continues in `plan_fixed` order.
- Tier-B cells that have not started by the end of S3 are NOT RUN, and `status` records this. B4 (context ladder) runs only if the Tier-2 amendment is at HEAD: in HYPOTHESIS.md, or as the mini's committed `amendments/<k>_tier2_<UTC>.md`, which the pull above fetches (`"$PY" tools/status.py --tier2` says which). Without it the session skips B4 and runs the rest of the queue; at the end of S3 an unstarted B4 is NOT RUN with that reason.
- **If S3 ends with Tier-A cells outstanding,** `run.py` prints `Tier-A outstanding — run S3b now`. Run it at once; there is no decision to make:
  ```bash
  cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
  caffeinate -i "$PY" runner/run.py session --name S3b
  ```
  S3b admits a Tier-A cell only while the projected run total stays within the overrun ceiling (44 h by default); anything it cannot admit is recorded NOT RUN.

Check: `status` shows the queue complete, with any NOT RUN cell listed and its reason.

Outputs: `results/raw/S3/<arm>/*.jsonl` (and `S3b/`), with private text as in step 14.

### 17. Score on the mbp and commit
*Claude may run; ≈ 5–10 min.* Scoring needs the Tier-2 amendment at HEAD (in HYPOTHESIS.md or as the mini's committed `amendments/` file). The block stops at `--tier2` when it is not there yet:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
git -C "$LFA" pull --ff-only && \
"$PY" tools/status.py --sync-amendments && \
"$PY" tools/status.py --tier2 && \
"$PY" scorers/score_all.py && \
"$PY" tools/version_record.py && \
"$PY" tools/status.py --record-block "S3, mbp scores" && \
git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_036: S3 — raw outputs; mbp scores" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
Check:
- `results/scores/<arm>/*.jsonl` exist for every completed cell except IFBench (scored on the mini).
- The scores carry item hashes, extracted answers (letters, integers or instruction results; null for RGB) and categories, and no withheld text.
- The leak check is clean.

**If `--tier2` printed `no`:** commit the raw outputs without scores, tell Andrei, and keep this session open:
```bash
cd ~/REPOS/local-first-ai/tasks/chronos/exp_036_kolibri_local_eval && source env/exp036.env && \
"$PY" tools/status.py --record-block "S3" && \
git -C "$LFA" add tasks/chronos/exp_036_kolibri_local_eval/results tasks/chronos/exp_036_kolibri_local_eval/aborted tasks/chronos/exp_036_kolibri_local_eval/evidence tasks/chronos/exp_036_kolibri_local_eval/HYPOTHESIS.md && \
"$PY" tools/leak_check.py --range @{u}..HEAD --staged && \
git -C "$LFA" commit -m "chronos/exp_036: S3 — raw outputs (scores wait for the Tier-2 amendment)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && \
git -C "$LFA" push origin HEAD && git -C "$LFA" status --porcelain
```
When the main session has pushed the amendment, this mbp session reruns the first block of this step (it scores and commits). The mbp always writes the mbp scores; the mini only re-scores and compares, so steps 18 and 19 wait until the scores are pushed.

### 18. Mirror the withheld outputs to the mini (required)
*Andrei runs; ≈ 1–5 min.* The mini re-scores every set, including GPQA, RGB and AIME-DE, byte for byte; `analysis/verdicts.py` refuses to run without it. `$MINI` is set in Andrei's shell and is never written into the repo.
```bash
rsync -a "$EXP036_PRIVATE/" "$MINI:models/exp036-mini/private/"
```
Check: rsync exits 0. Then, on the mini (the main session, after `git -C "$LFA" pull --ff-only`): `"$PY" tools/status.py --verify-private` prints `"ok": true`: every private file `evidence/withheld_manifest.jsonl` lists is on the mini at its latest sha256 (a file the crash fallback moved, at its new path under `aborted/`). The list repeats a path each time the file grew, and it never names `manifests/` (step 7) or `aborted/` files it did not move; those are the `expected_extras`, not a mismatch.

### 19. Hand-back
*Claude may run; seconds.* The mbp session writes nothing further.
1. Confirm `git -C "$LFA" status` is clean and HEAD equals origin.
2. Tell Andrei that Session 3 is pushed and the private mirror is done.

From here the main session on the mini is the only writer of HYPOTHESIS.md. It:
- sets up the mini once, before its first command: the git-ignored `env/exp036.local.env` on the mini holds `export EXP036_MODELS="$HOME/models/exp036-mini"` and `export PY="$EXP036_MODELS/venv312/bin/python"`, so `source env/exp036.env` gives `$EXP036_PRIVATE` = `~/models/exp036-mini/private` (the mirror of step 18), `$EXP036_DATA`, the IFBench venv and NLTK data under `~/models/exp036-mini`; if `venv312` does not exist yet, `EXP036_MODELS=~/models/exp036-mini EXP036_VENV=~/models/exp036-mini/venv312 bash env/setup.sh` creates it (a download from PyPI: Andrei's OK);
- scores IFBench in its own venv: `"$PY" scorers/score_all.py --ifbench` (the IFBench python is `$EXP036_IFBENCH_PY`, else `<models>/ifbench-venv/bin/python`; NLTK data `$EXP036_NLTK_DATA`, else `<models>/nltk_data`);
- re-scores every set, IFBench included, and writes `results/rescore_mini_<UTC>.json`, which must show byte-identical scores: `"$PY" scorers/score_all.py --rescore-compare --ifbench` (without `--ifbench` the committed IFBench scores could not match; the script now implies it when they exist);
- computes the verdicts (`"$PY" analysis/verdicts.py`, which refuses without a matching re-score), appends the run-record and results blocks to HYPOTHESIS.md (`"$PY" tools/status.py --record-block "verdicts"`) and updates the Status line;
- writes the scientific_log result block and the README results, and drafts the post (HYPOTHESIS "Publication angle", Order).

Outward steps after this point (the results push, `deploy.sh`, SFTP) are Andrei's go.

---

## Quick reference

| Step | What | Who | Duration (nominal / pessimistic) | Commit |
|---|---|---|---|---|
| 0 | Repo access, downloads, new assets, hygiene | Claude, Andrei | minutes | — |
| 1–3 | Pull, identity and hook, venv + tests | Claude | ≈ 5 min | — |
| 3b | Optional dry run on tiny checkpoints (`tools/dry_run.py`) | Claude | ≈ 3–5 min | — |
| 4 | Preflight quick (Claude) and deep (Andrei, `caffeinate -i`) | both | 0.1 / 0.3 h | `preflight <UTC>` |
| 5 | `sudo sysctl iogpu.wired_limit_mb=114688`, only if printed | Andrei | seconds | — |
| 6 | Sign-off (+ Amendment 0 if any) | Andrei, then Claude | ≈ 15 min | `sign-off — Andrei` |
| 7 | Manifests, shingles, web gate texts | Claude | 2 min | `item manifests` |
| 8 | Convert K8, K4 | Andrei, `caffeinate -i` | 0.5 / 1.5 h | `K8/K4 conversion records` |
| 9 | Peer check | Andrei, `caffeinate -i` | 0.4 / 0.6 h | `peer check` |
| 10 | Gate (STOP on exit 1 or 4) | Andrei, `caffeinate -i` | 1.8 / 3.0 h | `gate <UTC> — PASS\|FAIL` |
| 11 | Bench cells (H1, D1, H5, H8, C1) | Andrei, `caffeinate -i` | 1.2 / 1.7 h | `S1 bench cells` |
| 12 | Pilot (interrupted: `abort-pilot`, rerun in full) | Andrei, `caffeinate -i` | 1.3 / 1.9 h | (with 13) |
| 13 | Plan-fix, plan amendment | Claude | seconds | `amendment <k> — plan fixed by rule` |
| 14–15 | Session 2 | Andrei, `caffeinate -i`; then Claude | ≤ 16 h | `S2 — raw outputs` |
| 16–17 | Session 3 (+ S3b if printed), scoring | Andrei, `caffeinate -i`; then Claude | ≤ 16 h | `S3 — raw outputs; mbp scores` |
| 18 | Private mirror to the mini (required) | Andrei | minutes | — |
| 19 | Hand-back | Claude | seconds | — |

Total machine time: ≈ 35.1 h nominal, ≈ 38.5 h pessimistic, ≈ 39.9 h adverse, over three sessions (HYPOTHESIS, "Sessions & budget"); the automatic S3b may extend it to the 44 h overrun ceiling, never beyond.
