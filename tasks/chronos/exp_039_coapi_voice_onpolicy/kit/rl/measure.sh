#!/bin/bash
# exp_039 measurement (HYPOTHESIS.md + amendment 1). Starts the kept Ollama 0.33.0 beside production
# on :18435 with the production env of exp_035's day (as a12_side.sh), runs the ruler check, then the
# requested cells, then stops the server. Production Ollama on :11434 is never touched.
#
#   bash coapi_voice/rl/measure.sh repro                 # ruler check only
#   bash coapi_voice/rl/measure.sh B0r:v1 R1:v1 ...      # ruler check, then each ARM:SET cell in order
#
# ARM: B0r R1 A12 R1G · SET: v1 v21. A cell refuses on any pin mismatch; a v2.1 cell runs once.
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"; PY="$CS/.venv/bin/python3"
OLD="$HOME/models/ollama-0.33.0/bin/ollama"; PORT=18435; HOST="http://127.0.0.1:$PORT"
LOG="$CV/runs/side_server_exp039_$(date -u +%Y%m%dT%H%M%SZ).log"

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

arm() { (cd "$CS" && "$PY" "$CV/rl/run_arm.py" --host "$HOST" "$@"); }
echo "== ruler check: A12 on v1 rows 1–3, archived index, vs 2026-09-22"
arm repro
[ "$1" = repro ] && exit 0
for cell in "$@"; do
  echo "== cell $cell"
  arm "${cell%%:*}" "${cell##*:}"
done
echo "== done; side server stopped on exit. Log: $LOG"
