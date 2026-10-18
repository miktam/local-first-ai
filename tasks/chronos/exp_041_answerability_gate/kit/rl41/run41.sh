#!/bin/bash
# exp_041: serve one gate on llama.cpp b11711 (9d1e036) at :18441, run it on a set, stop the server. Gates run one at a
# time (G4 alone needs ~29 GB). Production Ollama and the exp_039/040 side server are never touched.
#
#   bash coapi_voice/rl41/run41.sh dev G1 G2 G3 G4     # the dev runs (resumable), then thresholds: rl41/select41.py
#   bash coapi_voice/rl41/run41.sh v3  G1 G2 G3 G4     # one v3 pass per gate, after thresholds.json is committed
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,7p "$0"; exit 1; }
SET="$1"; shift
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"; PY="$CS/.venv/bin/python3"
SRV="$HOME/models/llama.cpp-b11711/build/bin/llama-server"; PORT=18441; HOST="http://127.0.0.1:$PORT"
[ "$("$SRV" --version 2>&1 | grep -o 'commit [0-9a-f]*')" = "commit 9d1e036" ] || { echo "REFUSED: llama-server is not b11711 (9d1e036)"; exit 2; }
if [ "$SET" = v3 ] && [ -n "$(git -C "$CS" status --porcelain -- coapi_voice/rl41 coapi_voice/data/gate41)" ]; then
  echo "REFUSED: rl41/ or data/gate41/ has uncommitted changes; the v3 pass runs only from committed code and thresholds"; exit 2
fi
for GATE in "$@"; do
  GGUF=$("$PY" -c "import sys; sys.path.insert(0,'$CV/rl41'); import gate41 as G; print(G.GATES['$GATE']['gguf'])")
  KIND=$("$PY" -c "import sys; sys.path.insert(0,'$CV/rl41'); import gate41 as G; print(G.GATES['$GATE']['kind'])")
  if curl -s --max-time 2 -o /dev/null "$HOST/"; then echo "REFUSED: something already answers on :$PORT"; exit 2; fi
  LOG="$CV/runs/gate41_server_${SET}_${GATE}_$(date -u +%Y%m%dT%H%M%SZ).log"; mkdir -p "$CV/runs"
  if [ "$KIND" = text ]; then ARGS=(-c 8192 --jinja); else ARGS=(-c 16384 -b 8192 -ub 8192); fi
  "$SRV" -m "$GGUF" --host 127.0.0.1 --port $PORT "${ARGS[@]}" >"$LOG" 2>&1 &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
  for _ in $(seq 1 300); do
    curl -sf "$HOST/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server for $GATE exited:"; tail -3 "$LOG"; exit 2; }
    sleep 1
  done
  echo "== $GATE on $SET"
  (cd "$CS" && "$PY" "$CV/rl41/run41.py" --host "$HOST" "$GATE" "$SET")
  kill $SERVER; wait $SERVER 2>/dev/null || true; trap - EXIT
done
echo "== done"
