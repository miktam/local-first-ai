#!/bin/zsh -e
# exp_053: the registered run, in order. From tasks/exp053, inside tmux (session exp053-run).
# The router (gemma4:e4b) stays loaded throughout; each arm loads only its own gemma4 26B build.
cd "$(dirname "$0")/.."
TS=$(date -u +%Y%m%dT%H%M%SZ)
R=runs/$TS
mkdir -p $R
A=(python3.14 -I kit/arms53.py --frozen data/frozen.jsonl)
{
  echo "exp_053 run $TS  ollama $(curl -s localhost:11434/api/version)  loaded before: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
  $A --arm G4 --out $R/G4 || { echo "FAILED: G4"; exit 1 }
  $A --arm GQ --out $R/GQ || { echo "FAILED: GQ"; exit 1 }
  $A --arm G8 --out $R/G8 || { echo "FAILED: G8"; exit 1 }
  $A --arm G4 --out $R/G4_drift --first 10 || { echo "FAILED: drift"; exit 1 }
  python3.14 -I kit/score53.py data/frozen.jsonl $R
  python3.14 -I kit/judge53.py build data/frozen.jsonl $R
  echo "exp_053 run $TS finished $(date -u +%H:%M:%SZ); loaded after: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
} 2>&1 | tee $R/run.log
