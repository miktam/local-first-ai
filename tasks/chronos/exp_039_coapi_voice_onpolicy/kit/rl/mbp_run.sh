#!/usr/bin/env bash
# exp_039 R1 on the mbp (DESIGN §7, PREP part C). Run from anywhere inside the casasol checkout:
#
#   bash coapi_voice/rl/mbp_run.sh setup       # venv from rl/requirements.lock + Qwen3-4B at the pinned snapshot
#   bash coapi_voice/rl/mbp_run.sh preflight   # machine, versions, hashes, reward self-check; refuses on mismatch
#   bash coapi_voice/rl/mbp_run.sh pilot       # 10 timed steps + the frozen projection rule (> 12 h → STOP)
#   bash coapi_voice/rl/mbp_run.sh full        # one pass over the train prompts (~105 steps)
#
# Long runs: start them yourself in a terminal you can watch. Logs go to coapi_voice/runs_rl/.
# Afterwards, copy runs_rl/ back to the mini (rsync into a named folder), where R1 is fused and measured.
set -euo pipefail

CV="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/models/exp039-venv"
PY="$VENV/bin/python"
REV="1cfa9a7208912126459214e8b04321603b3df60c"
CAP_HOURS=12            # frozen rule (DESIGN §7): projected full run > 12 h wall → STOP, re-plan
export EXP039_GLOSSARY="$CV/data/rl/glossary.json"

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
    say "prompt set vs its manifest"
    "$PY" - "$CV" <<'EOF'
import hashlib, json, sys
from pathlib import Path
cv = Path(sys.argv[1]); m = json.loads((cv / "data/rl/manifest.json").read_text())
for key, want in m["sha256"].items():
    if key.startswith("out/"):
        got = hashlib.sha256((cv / "data/rl" / key[4:]).read_bytes()).hexdigest()
        print(("ok  " if got == want else "FAIL"), key); assert got == want, key
for key in ("code/reward.py", "code/arith_templates.py", "code/probes.py", "code/build_prompts.py"):
    got = hashlib.sha256((cv / "rl" / key[5:]).read_bytes()).hexdigest()
    print(("ok  " if got == m["sha256"][key] else "FAIL"), key); assert got == m["sha256"][key], key
print("built at casasol", m["casasol_commit"][:9], "· code dirty:", m["code_dirty"])
assert not m["code_dirty"], "prompt set was built from uncommitted code"
EOF
    say "reward self-check through the trainer's own loader"
    (cd "$CV" && "$PY" - <<'EOF'
import json
from mlx_lm_lora.train import load_reward_functions_from_file
from mlx_lm_lora.trainer.grpo_reward_functions import get_reward_function
assert load_reward_functions_from_file("rl/reward.py")
import custom_rewards as R
row = json.loads(open("data/rl/valid.jsonl").readline())
fns = [get_reward_function(n) for n in R.REWARD_FUNCTION_NAMES.split(",")]
print({f.__name__: f(prompts=[row["prompt"]], completions=["Test answer."], answer=[row["answer"]])[0] for f in fns})
EOF
    ) 2>&1 | grep -v PyTorch
    say "preflight green; next: pilot"
    ;;

  pilot)
    mkdir -p "$CV/runs_rl"; LOG="$CV/runs_rl/pilot_$(date +%Y%m%d-%H%M%S).log"
    N=$(grep -c . "$CV/data/rl/train.jsonl"); FULL=$(( (N + 7) / 8 ))
    START=$(date +%s)
    (cd "$CV" && "$PY" -u rl/train_r1.py --iters 10) 2>&1 | tee "$LOG"
    WALL=$(( $(date +%s) - START ))
    # conservative: wall / 10 includes model load and the step-1 validation batch
    PROJ=$(( WALL * FULL / 10 ))
    say "pilot: 10 steps in ${WALL}s → projected full run ${FULL} steps ≈ $(( PROJ / 3600 ))h$(( PROJ % 3600 / 60 ))m (cap ${CAP_HOURS}h)"
    if [ "$PROJ" -gt $(( CAP_HOURS * 3600 )) ]; then
      echo "STOP: projection exceeds the frozen ${CAP_HOURS} h rule. Do not start the full run; record and re-plan." | tee -a "$LOG"
      exit 3
    fi
    echo "within the rule; next: full" | tee -a "$LOG"
    ;;

  full)
    mkdir -p "$CV/runs_rl"; LOG="$CV/runs_rl/r2_$(date +%Y%m%d-%H%M%S).log"
    (cd "$CV" && "$PY" -u rl/train_r1.py) 2>&1 | tee "$LOG"
    say "done; adapters in coapi_voice/runs_rl/r2 (checkpoints every 25 steps). Copy runs_rl/ back to the mini."
    ;;

  *) sed -n 2,12p "$0"; exit 1 ;;
esac
