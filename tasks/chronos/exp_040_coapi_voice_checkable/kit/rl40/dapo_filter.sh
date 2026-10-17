#!/bin/bash
# exp_040 offline filter (BUILD §5). Starts the kept Ollama 0.33.0 beside production on :18435 with the same env as
# rl/measure.sh, samples every train prompt (resumable: re-run after an interruption), selects, stops the server.
# Production Ollama on :11434 is never touched.
#
#   bash coapi_voice/rl40/dapo_filter.sh [--limit N]      # extra args go to `dapo_filter.py run`
set -euo pipefail
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"; PY="$CS/.venv/bin/python3"
OLD="$HOME/models/ollama-0.33.0/bin/ollama"; PORT=18435; HOST="http://127.0.0.1:$PORT"
mkdir -p "$CV/runs"
LOG="$CV/runs/side_server_exp040_dapo_$(date -u +%Y%m%dT%H%M%SZ).log"

if curl -s --max-time 2 -o /dev/null "$HOST/"; then echo "REFUSED: something already answers on :$PORT"; exit 2; fi
OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_NOPRUNE=1 OLLAMA_FLASH_ATTENTION=0 OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=10m OLLAMA_MODELS="$HOME/.ollama/models" \
  "$OLD" serve >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "$HOST/api/version" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "side server exited:"; tail -3 "$LOG"; exit 2; }
  sleep 1
done

cd "$CS"
"$PY" "$CV/rl40/dapo_filter.py" run --host "$HOST" --workers 2 "$@"
case " $* " in *" --limit "*) echo "== smoke run: not selecting"; exit 0;; esac
"$PY" "$CV/rl40/dapo_filter.py" select
echo "== done; side server stopped on exit. Log: $LOG"
