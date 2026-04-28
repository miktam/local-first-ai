#!/usr/bin/env bash
# tests/h1_fa_cpu_fallback.sh
# H1: Flash Attention causes silent GPU→CPU fallback on Gemma 4 MoE.
#
# Requires sudo NOPASSWD on /usr/bin/powermetrics. Restarts Ollama
# twice via launchd. Plan for ~10–25 minutes wall time depending on
# prompt eval speed at incident scale.

set -euo pipefail

H_ID="H1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/ollama_probe.sh"
# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/powermetrics_probe.sh"

MODEL="${MODEL:-gemma4-think:26b}"
PROMPT_TOKENS="${PROMPT_TOKENS:-40000}"     # incident-scale by default
NUM_CTX="${NUM_CTX:-131072}"                # match the alias's PARAMETER
PM_DURATION="${PM_DURATION:-600}"           # 10 min cap per condition;
                                            # we window the mean to actual
                                            # request time, so trailing
                                            # idle samples are discarded.
GPU_LOW_THRESHOLD_MW="${GPU_LOW_THRESHOLD_MW:-1500}"
GPU_HIGH_THRESHOLD_MW="${GPU_HIGH_THRESHOLD_MW:-3000}"

TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
EVIDENCE_DIR="${ROOT_DIR}/evidence/${TS}-${H_ID}"
mkdir -p "$EVIDENCE_DIR"

emit_json() {
  jq -nc \
    --arg id "$H_ID" \
    --arg status "$1" \
    --arg dir "$EVIDENCE_DIR" \
    --arg summary "$2" \
    '{hypothesis_id:$id, status:$status, evidence_dir:$dir, summary:$summary}'
}

cleanup() {
  powermetrics_stop || true
}
trap cleanup EXIT

log "Hypothesis: $H_ID — FA CPU fallback on $MODEL"
log "Prompt size target: ~${PROMPT_TOKENS} tokens"
log "PM_DURATION cap: ${PM_DURATION}s per condition (mean is windowed)"
log "Evidence: $EVIDENCE_DIR"

# Preflight checks.
if ! ollama_is_up; then
  emit_json "could_not_run" "Ollama not reachable at $OLLAMA_HOST"
  exit 3
fi

if ! ollama_has_model "$MODEL"; then
  emit_json "could_not_run" "Model $MODEL not present locally"
  exit 3
fi

if ! powermetrics_check_sudo; then
  emit_json "could_not_run" "powermetrics requires sudo; see stderr for guidance"
  exit 3
fi

# Generate the prompt once. Same prompt used in both conditions for
# direct comparability; cache hits aren't a concern because Ollama
# is fully restarted between conditions.
PROMPT_FILE="$EVIDENCE_DIR/prompt.txt"
generate_filler_prompt "$PROMPT_TOKENS" "h1" > "$PROMPT_FILE"
PROMPT_BYTES=$(wc -c < "$PROMPT_FILE")
log "Generated prompt: ${PROMPT_BYTES} bytes"

run_one_condition() {
  local fa_value="$1"
  local label="$2"
  local outdir="$EVIDENCE_DIR/$label"
  mkdir -p "$outdir"

  log "── Condition: ${label} (OLLAMA_FLASH_ATTENTION=${fa_value}) ──"

  launchctl setenv OLLAMA_FLASH_ATTENTION "$fa_value"
  echo "$fa_value" > "$outdir/fa_value.txt"

  ollama_restart_app
  sleep 3

  log "Starting powermetrics (cap ${PM_DURATION}s)"
  powermetrics_start "$outdir/powermetrics.txt" "$PM_DURATION"

  # Build request body. think:false matches what OpenClaw sends.
  local body
  body=$(jq -nc \
    --arg m "$MODEL" \
    --rawfile p "$PROMPT_FILE" \
    --argjson nc "$NUM_CTX" \
    '{model:$m, prompt:$p, stream:false, think:false, options:{num_ctx:$nc}}')

  log "Sending /api/generate request"
  local start_ts
  start_ts=$(date +%s)

  curl -fsS --max-time "$PM_DURATION" "${OLLAMA_HOST}/api/generate" -d "$body" \
    > "$outdir/response.json" 2> "$outdir/curl.err" &
  local curl_pid=$!

  while kill -0 "$curl_pid" 2>/dev/null; do
    {
      printf '\n--- %s ---\n' "$(date -u +%H:%M:%S)"
      ollama_ps_json | jq .
    } >> "$outdir/ollama_ps.log" 2>/dev/null || true
    sleep 5
  done

  local curl_rc=0
  wait "$curl_pid" || curl_rc=$?
  local end_ts
  end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))

  log "curl exit: $curl_rc (elapsed ${elapsed}s)"

  # Wait for powermetrics to finish on its own. The windowed mean
  # below limits the average to the elapsed window, so trailing idle
  # samples don't dilute the signal.
  log "Waiting for powermetrics to finish"
  powermetrics_stop

  # Compute means: full window (legacy) and windowed.
  local gpu_full cpu_full gpu_window cpu_window
  gpu_full=$(powermetrics_mean_gpu_mw "$outdir/powermetrics.txt")
  cpu_full=$(powermetrics_mean_cpu_mw "$outdir/powermetrics.txt")
  local window=$((elapsed + 2))
  gpu_window=$(powermetrics_mean_gpu_mw_window "$outdir/powermetrics.txt" "$window")
  cpu_window=$(powermetrics_mean_cpu_mw_window "$outdir/powermetrics.txt" "$window")

  log "GPU power: full=${gpu_full}mW, windowed[0..${window}s]=${gpu_window}mW"
  log "CPU power: full=${cpu_full}mW, windowed[0..${window}s]=${cpu_window}mW"

  local prompt_eval_count eval_count total_duration_ms
  if [[ -s "$outdir/response.json" ]]; then
    prompt_eval_count=$(jq -r '.prompt_eval_count // 0' "$outdir/response.json")
    eval_count=$(jq -r '.eval_count // 0' "$outdir/response.json")
    total_duration_ms=$(jq -r '(.total_duration // 0) / 1000000' "$outdir/response.json")
    log "Tokens: prompt_eval=$prompt_eval_count, eval=$eval_count, total=${total_duration_ms}ms"
  else
    prompt_eval_count=0
    eval_count=0
    total_duration_ms=0
  fi

  jq -nc \
    --arg fa "$fa_value" \
    --argjson gpu_full    "$gpu_full"    --argjson gpu_window "$gpu_window" \
    --argjson cpu_full    "$cpu_full"    --argjson cpu_window "$cpu_window" \
    --argjson elapsed_s   "$elapsed"     --argjson curl_rc    "$curl_rc" \
    --argjson prompt_eval "$prompt_eval_count" \
    --argjson eval_count  "$eval_count" \
    --argjson total_ms    "$total_duration_ms" \
    '{fa_value:$fa,
      gpu_mean_mw_full:$gpu_full,
      gpu_mean_mw_window:$gpu_window,
      cpu_mean_mw_full:$cpu_full,
      cpu_mean_mw_window:$cpu_window,
      elapsed_s:$elapsed_s,
      curl_rc:$curl_rc,
      prompt_eval_count:$prompt_eval,
      eval_count:$eval_count,
      total_duration_ms:$total_ms}' \
    > "$outdir/condition.json"
}

run_one_condition 1 "fa_on"
run_one_condition 0 "fa_off"

GPU_ON=$(jq -r  '.gpu_mean_mw_window' "$EVIDENCE_DIR/fa_on/condition.json")
GPU_OFF=$(jq -r '.gpu_mean_mw_window' "$EVIDENCE_DIR/fa_off/condition.json")
CPU_ON=$(jq -r  '.cpu_mean_mw_window' "$EVIDENCE_DIR/fa_on/condition.json")
CPU_OFF=$(jq -r '.cpu_mean_mw_window' "$EVIDENCE_DIR/fa_off/condition.json")
ELAPSED_ON=$(jq -r  '.elapsed_s' "$EVIDENCE_DIR/fa_on/condition.json")
ELAPSED_OFF=$(jq -r '.elapsed_s' "$EVIDENCE_DIR/fa_off/condition.json")

jq -nc \
  --argjson gpu_on    "$GPU_ON"     --argjson gpu_off    "$GPU_OFF" \
  --argjson cpu_on    "$CPU_ON"     --argjson cpu_off    "$CPU_OFF" \
  --argjson elapsed_on "$ELAPSED_ON" --argjson elapsed_off "$ELAPSED_OFF" \
  '{windowed_gpu_on_mw:$gpu_on, windowed_gpu_off_mw:$gpu_off,
    windowed_cpu_on_mw:$cpu_on, windowed_cpu_off_mw:$cpu_off,
    elapsed_on_s:$elapsed_on,   elapsed_off_s:$elapsed_off}' \
  > "$EVIDENCE_DIR/run.json"

log "── Summary (windowed means) ──"
log "FA=1: GPU=${GPU_ON}mW CPU=${CPU_ON}mW elapsed=${ELAPSED_ON}s"
log "FA=0: GPU=${GPU_OFF}mW CPU=${CPU_OFF}mW elapsed=${ELAPSED_OFF}s"

GPU_ON_INT=${GPU_ON%.*}
GPU_OFF_INT=${GPU_OFF%.*}

if [[ "$GPU_ON_INT" -lt "$GPU_LOW_THRESHOLD_MW" \
   && "$GPU_OFF_INT" -gt "$GPU_HIGH_THRESHOLD_MW" ]]; then
  emit_json "supported" \
    "GPU idle with FA=1 (${GPU_ON}mW windowed) and active with FA=0 (${GPU_OFF}mW windowed)"
  exit 0
elif [[ "$GPU_ON_INT" -gt "$GPU_HIGH_THRESHOLD_MW" \
     && "$GPU_OFF_INT" -gt "$GPU_HIGH_THRESHOLD_MW" ]]; then
  emit_json "rejected" \
    "GPU active in both conditions (FA=1: ${GPU_ON}mW, FA=0: ${GPU_OFF}mW windowed)"
  exit 1
else
  emit_json "inconclusive" \
    "Mixed signal (FA=1 GPU=${GPU_ON}mW, FA=0 GPU=${GPU_OFF}mW windowed)"
  exit 2
fi
