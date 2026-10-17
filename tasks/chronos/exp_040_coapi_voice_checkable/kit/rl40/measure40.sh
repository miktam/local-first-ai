#!/bin/bash
# exp_040 measurement (HYPOTHESIS.md). Starts the kept Ollama 0.33.0 beside production on :18435 with rl/measure.sh's env,
# runs exp_039's ruler check (A12 on v1 rows 1–3, archived index, byte-identical to 2026-09-22), then each ARM:SET cell
# through rl40/run_arm40.py, then stops the server. Production Ollama on :11434 is never touched.
#
#   bash coapi_voice/rl40/measure40.sh B0i:valid40:6 A13:valid40:6   # smoke on validation items (any time; :N = first N rows)
#   bash coapi_voice/rl40/measure40.sh B0i:v3 R3:v3 B0q8:v3 R3q8:v3 A13:v3 R3G:v3     # the scored pass, after R3 is registered
#
# ARM: B0i R3 B0q8 R3q8 A13 R3G · SET: v3 valid40. A cell refuses on any pin mismatch; a v3 cell runs once per arm.
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,9p "$0"; exit 1; }
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"; PY="$CS/.venv/bin/python3"
OLD="$HOME/models/ollama-0.33.0/bin/ollama"; PORT=18435; HOST="http://127.0.0.1:$PORT"
LOG="$CV/runs/side_server_exp040_$(date -u +%Y%m%dT%H%M%SZ).log"

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

echo "== ruler check: A12 on v1 rows 1–3, archived index, vs 2026-09-22"
(cd "$CS" && "$PY" "$CV/rl/run_arm.py" --host "$HOST" repro)
for cell in "$@"; do
  echo "== cell $cell"
  IFS=: read -r ARM SET LIMIT <<<"$cell"
  (cd "$CS" && "$PY" "$CV/rl40/run_arm40.py" --host "$HOST" "$ARM" "$SET" ${LIMIT:+--limit "$LIMIT"})
done
echo "== done; side server stopped on exit. Log: $LOG"
