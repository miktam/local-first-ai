#!/bin/bash
# Run one command with the Ollama 0.33.0 side server up on :18435 (exp_035's env), then stop it.
# Usage: harness/with_side_server.sh <command> [args...]
set -euo pipefail
P38="$(cd "$(dirname "$0")/.." && pwd)"
PORT=18435
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$P38/runs/side_server_${STAMP}.log"
mkdir -p "$P38/runs"
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
"$@"
