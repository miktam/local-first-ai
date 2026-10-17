#!/usr/bin/env bash
# exp_040 R3 on the mbp (DESIGN §7; exp_039's mbp_run.sh with exp_040's data, reward and launcher). From the casasol checkout:
#
#   bash coapi_voice/rl40/mbp_run40.sh setup       # venv from rl/requirements.lock + Qwen3-4B at the pinned snapshot (as exp_039)
#   bash coapi_voice/rl40/mbp_run40.sh preflight   # machine, versions, hashes, reward self-check; refuses on mismatch
#   bash coapi_voice/rl40/mbp_run40.sh pilot       # 10 timed steps + the frozen projection rule (> 12 h → STOP)
#   bash coapi_voice/rl40/mbp_run40.sh full        # one pass over the filtered train set (78 steps)
#
# Long runs: start them yourself in a terminal you can watch. Logs go to coapi_voice/runs_rl/.
# Afterwards, copy runs_rl/r3 back to the mini (rsync into a named folder): selection, fuse and measurement run there.
set -euo pipefail

CV="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/models/exp039-venv"          # the same pinned environment as exp_039 (one variable: the reward)
PY="$VENV/bin/python"
REV="1cfa9a7208912126459214e8b04321603b3df60c"
CAP_HOURS=12            # frozen rule (DESIGN §12.5): projected full run > 12 h wall → STOP, re-plan
DATA="$CV/data/rl40/train"

say() { printf '\n== %s\n' "$*"; }

case "${1:-}" in
  setup)
    command -v python3.14 >/dev/null || { echo "python3.14 not found (brew install python@3.14)"; exit 2; }
    [ -d "$VENV" ] || python3.14 -m venv "$VENV"
    "$PY" -m pip install --quiet -r "$CV/rl/requirements.lock"
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-4B', revision='$REV'))"
    say "setup done; next: preflight"
    ;;

  preflight)
    say "machine"
    sysctl -n machdep.cpu.brand_string; echo "$(( $(sysctl -n hw.memsize) / 1073741824 )) GB"; sw_vers -productVersion
    say "packages vs rl/requirements.lock"
    diff <("$PY" -m pip freeze | sort) <(sort "$CV/rl/requirements.lock") && echo "identical" \
      || { echo "FAIL: installed packages differ from the lock"; exit 2; }
    say "base snapshot"
    "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-4B', revision='$REV', local_files_only=True))"
    say "filtered train set vs its manifests"
    "$PY" - "$CV" <<'EOF'
import hashlib, json, sys
from pathlib import Path
cv = Path(sys.argv[1]); h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
f = json.loads((cv / "data/rl40/train/filter_manifest.json").read_text())
for name in ("train.jsonl", "valid.jsonl"):
    ok = h(cv / "data/rl40/train" / name) == f["sha256"][name]
    print(("ok  " if ok else "FAIL"), "train/" + name); assert ok, name
p = json.loads((cv / "data/rl40/prompts/manifest.json").read_text())
ok = h(cv / "data/rl40/prompts/train.jsonl") == f["sha256"]["prompts/train.jsonl"] == p["sha256"]["out/train.jsonl"]
print(("ok  " if ok else "FAIL"), "prompts/train.jsonl = the filter's input"); assert ok
for name in ("reward40.py", "prompt40.py", "build_prompts40.py"):
    ok = h(cv / "rl40" / name) == p["sha256"]["code/" + name]
    print(("ok  " if ok else "FAIL"), "rl40/" + name); assert ok, name
ok = h(cv / "rl40/reward40.py") == f["sha256"]["code/reward40.py"]
print(("ok  " if ok else "FAIL"), "reward40.py = the filter's scorer"); assert ok
print("prompts built at casasol", p["casasol_commit"][:9], "· code dirty:", p["code_dirty"], "· train rows", f["kept"])
assert not p["code_dirty"], "prompt set was built from uncommitted code"
EOF
    say "reward self-check through the trainer's own loader"
    (cd "$CV" && "$PY" - <<'EOF'
import json
from mlx_lm_lora.train import load_reward_functions_from_file
from mlx_lm_lora.trainer.grpo_reward_functions import get_reward_function
assert load_reward_functions_from_file("rl40/reward40.py")
import custom_rewards as W
fns = [get_reward_function(n) for n in W.REWARD_FUNCTION_NAMES.split(",")]
rows = [json.loads(l) for l in open("data/rl40/train/valid.jsonl")]
row = next(r for r in rows if r["type"] == "numeric")
gold = json.loads(row["answer"])["value"]
for text, want in (("Test answer.", -1.0), (f"The material gives the figure.\n{W.ANSWER_KEY[row['lang']]}: {gold:g}", 3.0)):
    got = {f.__name__: f(prompts=[row["prompt"]], completions=[text], answer=[row["answer"]])[0] for f in fns}
    print(got)
    assert got["coapi40_answer"] == want, (text, got["coapi40_answer"])
print("answer term: missing −1, right +3, as registered")
EOF
    ) 2>&1 | grep -v PyTorch
    say "preflight green; next: pilot"
    ;;

  pilot)
    mkdir -p "$CV/runs_rl"; LOG="$CV/runs_rl/r3_pilot_$(date +%Y%m%d-%H%M%S).log"
    N=$(grep -c . "$DATA/train.jsonl"); FULL=$(( (N + 7) / 8 ))
    START=$(date +%s)
    (cd "$CV" && "$PY" -u rl40/train40.py --iters 10) 2>&1 | tee "$LOG"
    WALL=$(( $(date +%s) - START ))
    PROJ=$(( WALL * FULL / 10 ))     # conservative: wall / 10 includes model load and the step-1 validation batch
    say "pilot: 10 steps in ${WALL}s → projected full run ${FULL} steps ≈ $(( PROJ / 3600 ))h$(( PROJ % 3600 / 60 ))m (cap ${CAP_HOURS}h)"
    if [ "$PROJ" -gt $(( CAP_HOURS * 3600 )) ]; then
      echo "STOP: projection exceeds the frozen ${CAP_HOURS} h rule. Do not start the full run; record and re-plan." | tee -a "$LOG"
      exit 3
    fi
    echo "within the rule; next: full" | tee -a "$LOG"
    ;;

  full)
    mkdir -p "$CV/runs_rl"; LOG="$CV/runs_rl/r3_$(date +%Y%m%d-%H%M%S).log"
    (cd "$CV" && "$PY" -u rl40/train40.py) 2>&1 | tee "$LOG"
    say "done; adapters in coapi_voice/runs_rl/r3 (checkpoints every 25 steps). Copy runs_rl/r3 back to the mini."
    ;;

  *) sed -n 2,11p "$0"; exit 1 ;;
esac
