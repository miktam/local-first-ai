#!/bin/bash
# exp_042 runs on the mini. Each step owns its server and stops it; production Ollama on :11434 is never touched.
#
#   bash coapi_voice/rl42/run42.sh dev      # G4 + G2 on the dev set (llama.cpp b11711), then thresholds (select)
#   bash coapi_voice/rl42/run42.sh v3       # after thresholds.json is committed: S0 voice (0.33.0 side server) + G4 + G2 on v3, then score
set -euo pipefail
[ $# -eq 1 ] || { sed -n 2,6p "$0"; exit 1; }
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"; PY="$CS/.venv/bin/python3"
SRV="$HOME/models/llama.cpp-b11711/build/bin/llama-server"; OLD="$HOME/models/ollama-0.33.0/bin/ollama"
GPORT=18441; OPORT=18435; mkdir -p "$CV/runs"

wait_up() { for _ in $(seq 1 300); do curl -sf "$1" >/dev/null && return 0; sleep 1; done; echo "server did not come up: $1"; exit 2; }

gate() {   # gate GATE SET
  local GGUF; GGUF=$("$PY" -c "import sys; sys.path.insert(0,'$CV/rl41'); import gate41 as G; print(G.GATES['$1']['gguf'])")
  if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$GPORT/"; then echo "REFUSED: :$GPORT busy"; exit 2; fi
  "$SRV" -m "$GGUF" --host 127.0.0.1 --port $GPORT -c 16384 -b 8192 -ub 8192 >"$CV/runs/gate42_${2}_${1}_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
  local S=$!; trap "kill $S 2>/dev/null || true" EXIT
  wait_up "http://127.0.0.1:$GPORT/health"
  (cd "$CS" && "$PY" "$CV/rl42/run42.py" gate --host "http://127.0.0.1:$GPORT" "$1" "$2")
  kill $S; wait $S 2>/dev/null || true; trap - EXIT
}

case "$1" in
  dev)
    gate G4 dev; gate G2 dev
    (cd "$CS" && "$PY" "$CV/rl42/run42.py" select)
    echo "== dev done; commit data/gate42/thresholds.json before the v3 step" ;;
  v3)
    [ -z "$(git -C "$CS" status --porcelain -- coapi_voice/rl42 coapi_voice/data/gate42)" ] || { echo "REFUSED: uncommitted rl42/ or data/gate42/"; exit 2; }
    if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$OPORT/"; then echo "REFUSED: :$OPORT busy"; exit 2; fi
    OLLAMA_HOST=127.0.0.1:$OPORT OLLAMA_NOPRUNE=1 OLLAMA_FLASH_ATTENTION=0 OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=2 \
      OLLAMA_KEEP_ALIVE=10m OLLAMA_MODELS="$HOME/.ollama/models" "$OLD" serve >"$CV/runs/side_server_exp042_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
    O=$!; trap "kill $O 2>/dev/null || true" EXIT
    wait_up "http://127.0.0.1:$OPORT/api/version"
    (cd "$CS" && "$PY" "$CV/rl42/run42.py" voice --host "http://127.0.0.1:$OPORT")
    kill $O; wait $O 2>/dev/null || true; trap - EXIT
    gate G4 v3; gate G2 v3
    mkdir -p "$CV/runs/exp042_score"
    (cd "$CS" && "$PY" "$CV/rl42/run42.py" score --json "$CV/runs/exp042_score/score.json")
    echo "== v3 done" ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
