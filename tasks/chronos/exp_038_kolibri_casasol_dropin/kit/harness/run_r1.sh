#!/bin/bash
# R1 on the mini: start the Ollama 0.33.0 copy as a side server on :18435 (11435 is taken by the Caddy proxy) with exp_035's
# production env (unchanged 2026-08-30 to 2026-10-10, per the server log), run R1 under both
# interpreters, stop the server. Production (:11434, 0.40.2) and the bot are not touched.
set -euo pipefail
P38="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$P38/runs/r1_${STAMP}_side_server.log"
mkdir -p "$P38/runs"
PORT=18435
if curl -s --max-time 2 -o /dev/null "http://127.0.0.1:$PORT/"; then
  echo "REFUSED: something already answers on 127.0.0.1:$PORT"; exit 2
fi

OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_NOPRUNE=1 OLLAMA_FLASH_ATTENTION=0 OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=10m OLLAMA_MODELS="$HOME/.ollama/models" \
  "$HOME/models/ollama-0.33.0/bin/ollama" serve >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "side server exited:"; tail -3 "$LOG"; exit 2; }
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/api/version"; echo

rc=0
"$HOME/REPOS/casasol/.venv/bin/python3" "$P38/harness/r1.py" --base-url "http://127.0.0.1:$PORT" --out "$P38/runs/r1_${STAMP}_py314" || rc=1
"$HOME/models/exp037-mini/venv312/bin/python" "$P38/harness/r1.py" --base-url "http://127.0.0.1:$PORT" --out "$P38/runs/r1_${STAMP}_py312" || rc=1
echo "R1 done (exit $rc): $P38/runs/r1_${STAMP}_py314 and _py312"
exit $rc
