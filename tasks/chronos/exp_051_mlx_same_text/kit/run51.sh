#!/bin/zsh -e
# exp_051: the whole registered run, in order. Run from tasks/exp051 inside tmux (session exp051-run).
cd "$(dirname "$0")/.."
TS=$(date -u +%Y%m%dT%H%M%SZ)
R=runs/$TS
mkdir -p $R
export HF_HUB_OFFLINE=1
B=(python3.14 -I kit/bench51.py)   # an array: zsh does not split a plain string into words
{
  echo "exp_051 run $TS  ollama $(curl -s localhost:11434/api/version)  macOS $(sw_vers -productVersion)"
  echo "loaded before: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
  $B --arm O  --out $R/O || { echo "FAILED: arm"; exit 1 }
  $B --arm ON --out $R/ON || { echo "FAILED: arm"; exit 1 }
  $B --arm X  --out $R/X || { echo "FAILED: arm"; exit 1 }
  $B --arm XN --out $R/XN || { echo "FAILED: arm"; exit 1 }
  $B --arm F  --out $R/F || { echo "FAILED: arm"; exit 1 }
  $B --arm O  --out $R/O_drift --sizes 15k --tag drift || { echo "FAILED: arm"; exit 1 }
  python3.14 -I kit/score51.py $R
  # restore the resident models as they were (default context, kept loaded)
  curl -s localhost:11434/api/generate -d '{"model":"gemma4:26b","keep_alive":-1}' >/dev/null
  curl -s localhost:11434/api/generate -d '{"model":"gemma4:e4b","keep_alive":-1}' >/dev/null
  echo "exp_051 run $TS finished $(date -u +%H:%M:%SZ); loaded after: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
} 2>&1 | tee $R/run.log
