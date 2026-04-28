#!/usr/bin/env bash
# tests/h6_prefill_scaling.sh
# H6: Prefill time scales as O(N²) in input length on gemma4-think:26b.
#
# Streamed sweep at {15k, 25k, 35k} tokens. Stops on first failure.
# Restarts Ollama between sizes for a fresh KV cache. Plan for ~45–60
# minutes wall time if all three sizes complete (15k ≈ 2 min, 25k ≈
# 5 min, 35k ≈ 10–14 min, plus model reload between sizes).

set -euo pipefail

H_ID="H6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/ollama_probe.sh"
# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/powermetrics_probe.sh"

MODEL="${MODEL:-gemma4-think:26b}"
NUM_CTX="${NUM_CTX:-131072}"
SIZES_DEFAULT=(15000 25000 35000)
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}"
PM_CAP="${PM_CAP:-1900}"  # > REQUEST_TIMEOUT to cover full request

# Allow override via SIZES env var (space-separated)
if [[ -n "${SIZES:-}" ]]; then
  read -r -a SIZE_LIST <<< "$SIZES"
else
  SIZE_LIST=("${SIZES_DEFAULT[@]}")
fi

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

log "Hypothesis: $H_ID — prefill scaling on $MODEL"
log "Sizes: ${SIZE_LIST[*]} tokens"
log "Per-request timeout: ${REQUEST_TIMEOUT}s"
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

PM_AVAILABLE="false"
if powermetrics_check_sudo; then
  PM_AVAILABLE="true"
  log "Powermetrics available: sampling per size"
else
  log "Powermetrics unavailable; timing data only"
fi

# Per-size results table.
SUMMARY_TSV="$EVIDENCE_DIR/sizes.tsv"
echo -e "size_tokens\tprompt_eval_count\tprompt_eval_ms\teval_count\teval_ms\tttfb_ms\tttft_ms\ttotal_ms\tms_per_prompt_token\tgpu_window_mw\tcpu_window_mw\toutcome" \
  > "$SUMMARY_TSV"

INSTRUCTION="\n\n=== END OF CONTEXT ===\n\nReply with the single word: ACK."

run_one_size() {
  local size="$1"
  local outdir="$EVIDENCE_DIR/size_${size}"
  mkdir -p "$outdir"

  log "── Size: ${size} tokens ──"

  # Fresh restart for clean KV cache.
  log "Restarting Ollama"
  if ! ollama_restart_app; then
    log "  Ollama restart failed; skipping size $size"
    printf '%d\t0\t0\t0\t0\t0\t0\t0\t0\t-1\t-1\trestart_failed\n' \
      "$size" >> "$SUMMARY_TSV"
    return 1
  fi
  sleep 3

  # Generate prompt for this size (calibrated 7 chars/token via lib).
  local variant="size${size}-${TS}"
  local prompt_file="$outdir/prompt.txt"
  generate_filler_prompt "$size" "$variant" > "$prompt_file"
  local prompt_bytes
  prompt_bytes=$(wc -c < "$prompt_file")
  log "  Prompt: ${prompt_bytes} bytes (seed=$variant)"

  # Start powermetrics.
  local pm_file="$outdir/powermetrics.txt"
  if [[ "$PM_AVAILABLE" == "true" ]]; then
    powermetrics_start "$pm_file" "$PM_CAP"
  fi

  # Build streaming request body.
  local body
  body=$(jq -nc \
    --arg m "$MODEL" \
    --rawfile p "$prompt_file" \
    --arg instr "$INSTRUCTION" \
    --argjson nc "$NUM_CTX" \
    '{model:$m,
      messages:[{role:"user", content:($p + $instr)}],
      stream:true,
      think:false,
      options:{num_ctx:$nc}}')

  # Capture stream as ndjson, with timing info.
  # We use curl --no-buffer to get bytes as they arrive, and tee
  # through a python wrapper that timestamps each line.
  local stream_file="$outdir/stream.ndjson"
  local timing_file="$outdir/timing.tsv"
  local err_file="$outdir/curl.err"
  echo -e "ts_ms\tpayload" > "$timing_file"

  log "  Streaming request (timeout ${REQUEST_TIMEOUT}s)"
  local start_ts
  start_ts=$(date +%s)
  local start_ms
  start_ms=$(python3 -c 'import time; print(int(time.time()*1000))')

  # Pipe curl through a python timestamper. Curl exits 0 on completion
  # or 28 on timeout. We capture exit code via PIPESTATUS.
  set +e
  curl -sS --no-buffer --max-time "$REQUEST_TIMEOUT" \
       "${OLLAMA_HOST}/api/chat" -d "$body" 2> "$err_file" \
    | python3 -u -c "
import sys, time
start_ms = $start_ms
for line in sys.stdin:
    line = line.rstrip('\n')
    if not line:
        continue
    ts = int(time.time() * 1000) - start_ms
    sys.stdout.write(line + '\n')
    sys.stderr.write(f'{ts}\t{line[:200]}\n')
    sys.stdout.flush()
    sys.stderr.flush()
" > "$stream_file" 2>> "$timing_file"
  local rcs=("${PIPESTATUS[@]}")
  set -e
  local curl_rc="${rcs[0]}"
  local end_ts
  end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))

  log "  curl exit: $curl_rc (elapsed ${elapsed}s)"

  # Wait for powermetrics to finish.
  if [[ "$PM_AVAILABLE" == "true" ]]; then
    log "  Waiting for powermetrics to finish"
    powermetrics_stop
  fi

  # Parse stream events to extract metrics.
  local event_count
  event_count=$(wc -l < "$stream_file" | tr -d ' ')
  log "  Stream events: $event_count"

  # Extract first-byte time and first-token time from timing file.
  local ttfb_ms ttft_ms
  ttfb_ms=$(awk -F'\t' 'NR==2 {print $1; exit}' "$timing_file")
  ttfb_ms="${ttfb_ms:-0}"
  # First token = first event whose payload has non-empty .message.content
  ttft_ms=$(python3 - "$timing_file" <<'EOF'
import sys, json
path = sys.argv[1]
with open(path) as fh:
    fh.readline()  # header
    for line in fh:
        parts = line.rstrip('\n').split('\t', 1)
        if len(parts) != 2: continue
        ts, payload = parts
        # payload is truncated at 200 chars; try to parse anyway
        try:
            obj = json.loads(payload)
        except Exception:
            # Truncated JSON — fall back to substring match
            if '"content":"' in payload and '"content":""' not in payload:
                print(ts); break
            continue
        msg = obj.get("message", {})
        if msg.get("content"):
            print(ts); break
    else:
        print(0)
EOF
  )

  # Final event metrics. Last line of stream should be done:true with metrics.
  local prompt_eval_count prompt_eval_ms eval_count eval_ms total_ms
  if [[ -s "$stream_file" ]]; then
    local last_event
    last_event=$(tail -n 1 "$stream_file")
    prompt_eval_count=$(echo "$last_event" | jq -r '.prompt_eval_count // 0' 2>/dev/null || echo 0)
    prompt_eval_ms=$(echo "$last_event" | jq -r '(.prompt_eval_duration // 0) / 1000000' 2>/dev/null || echo 0)
    eval_count=$(echo "$last_event" | jq -r '.eval_count // 0' 2>/dev/null || echo 0)
    eval_ms=$(echo "$last_event" | jq -r '(.eval_duration // 0) / 1000000' 2>/dev/null || echo 0)
    total_ms=$(echo "$last_event" | jq -r '(.total_duration // 0) / 1000000' 2>/dev/null || echo 0)
  else
    prompt_eval_count=0
    prompt_eval_ms=0
    eval_count=0
    eval_ms=0
    total_ms=0
  fi

  # ms per prompt token (the curve we care about).
  local ms_per_token
  if [[ "${prompt_eval_count%.*}" -gt 0 ]]; then
    ms_per_token=$(python3 -c "print(round($prompt_eval_ms / $prompt_eval_count, 3))")
  else
    ms_per_token=0
  fi

  # Windowed power means.
  local gpu_window cpu_window
  gpu_window="-1"
  cpu_window="-1"
  if [[ "$PM_AVAILABLE" == "true" ]]; then
    local window=$((elapsed + 2))
    gpu_window=$(powermetrics_mean_gpu_mw_window "$pm_file" "$window")
    cpu_window=$(powermetrics_mean_cpu_mw_window "$pm_file" "$window")
  fi

  # Outcome.
  local outcome
  if [[ "$curl_rc" -eq 0 && "${prompt_eval_count%.*}" -gt 0 ]]; then
    outcome="completed"
    log "  prompt_eval=${prompt_eval_count} tokens in ${prompt_eval_ms}ms (${ms_per_token} ms/tok)"
    log "  eval=${eval_count} tokens in ${eval_ms}ms"
    log "  ttfb=${ttfb_ms}ms ttft=${ttft_ms}ms total=${total_ms}ms"
    log "  GPU windowed=${gpu_window}mW, CPU=${cpu_window}mW"
  elif [[ "$curl_rc" -eq 28 ]]; then
    outcome="timeout"
    log "  TIMED OUT — no completion within ${REQUEST_TIMEOUT}s"
  else
    outcome="error_rc${curl_rc}"
    log "  ERROR rc=$curl_rc"
  fi

  printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$size" "$prompt_eval_count" "$prompt_eval_ms" \
    "$eval_count" "$eval_ms" \
    "$ttfb_ms" "$ttft_ms" "$total_ms" \
    "$ms_per_token" "$gpu_window" "$cpu_window" "$outcome" \
    >> "$SUMMARY_TSV"

  # Persist per-size summary too.
  jq -nc \
    --argjson size "$size" \
    --argjson prompt_eval_count "$prompt_eval_count" \
    --argjson prompt_eval_ms "$prompt_eval_ms" \
    --argjson eval_count "$eval_count" \
    --argjson eval_ms "$eval_ms" \
    --argjson ttfb_ms "$ttfb_ms" \
    --argjson ttft_ms "${ttft_ms:-0}" \
    --argjson total_ms "$total_ms" \
    --argjson ms_per_token "$ms_per_token" \
    --argjson gpu_window "$gpu_window" \
    --argjson cpu_window "$cpu_window" \
    --arg outcome "$outcome" \
    '{size_tokens:$size,
      prompt_eval_count:$prompt_eval_count,
      prompt_eval_ms:$prompt_eval_ms,
      eval_count:$eval_count,
      eval_ms:$eval_ms,
      ttfb_ms:$ttfb_ms,
      ttft_ms:$ttft_ms,
      total_ms:$total_ms,
      ms_per_prompt_token:$ms_per_token,
      gpu_window_mw:$gpu_window,
      cpu_window_mw:$cpu_window,
      outcome:$outcome}' \
    > "$outdir/summary.json"

  if [[ "$outcome" != "completed" ]]; then
    return 1
  fi
  return 0
}

# Run sizes in order, stop on first failure.
COMPLETED_SIZES=()
FAILED_SIZE=""
for size in "${SIZE_LIST[@]}"; do
  if run_one_size "$size"; then
    COMPLETED_SIZES+=("$size")
  else
    FAILED_SIZE="$size"
    log "Stopping sweep at size=$size (did not complete)"
    break
  fi
done

# ── Fit O(N²) curve to completed sizes ──────────────────────────────────────
log "── Curve fit ──"
log "Completed sizes: ${COMPLETED_SIZES[*]:-none}"
log "Failed at: ${FAILED_SIZE:-none}"

FIT_RESULT="not_enough_data"
WORST_RESIDUAL=""
if [[ ${#COMPLETED_SIZES[@]} -ge 2 ]]; then
  # Compute a = prompt_eval_ms / N² for each, check spread.
  FIT_RESULT=$(python3 - "$SUMMARY_TSV" <<'EOF'
import csv, sys
path = sys.argv[1]
rows = []
with open(path) as fh:
    r = csv.DictReader(fh, delimiter='\t')
    for row in r:
        try:
            n = int(row['size_tokens'])
            t = float(row['prompt_eval_ms'])
            outcome = row['outcome']
        except (ValueError, KeyError):
            continue
        if outcome == 'completed' and n > 0 and t > 0:
            rows.append((n, t))
if len(rows) < 2:
    print('not_enough_data')
    sys.exit(0)
# Fit a*N^2 (least squares, b=c=0): a = sum(t*N^2) / sum(N^4)
num = sum(t * n*n for n, t in rows)
den = sum(n**4 for n, t in rows)
a = num / den if den else 0
# Residuals as fraction of observed.
residuals = []
for n, t in rows:
    pred = a * n * n
    if t > 0:
        residuals.append(abs(pred - t) / t)
worst = max(residuals) if residuals else 1.0
print(f'a={a:.3e} worst_residual={worst:.3f} n_points={len(rows)}')
EOF
)
  log "Fit: $FIT_RESULT"
fi

# ── Persist final run.json ──────────────────────────────────────────────────
jq -nc \
  --argjson n_completed "${#COMPLETED_SIZES[@]}" \
  --arg failed_size "${FAILED_SIZE:-}" \
  --arg fit "$FIT_RESULT" \
  --arg sizes_attempted "${SIZE_LIST[*]}" \
  --arg sizes_completed "${COMPLETED_SIZES[*]:-}" \
  '{n_completed:$n_completed,
    failed_size:$failed_size,
    sizes_attempted:$sizes_attempted,
    sizes_completed:$sizes_completed,
    fit:$fit}' \
  > "$EVIDENCE_DIR/run.json"

# ── Decision rule ───────────────────────────────────────────────────────────
N_COMPLETED=${#COMPLETED_SIZES[@]}
N_TOTAL=${#SIZE_LIST[@]}

if [[ "$N_COMPLETED" -eq "$N_TOTAL" ]]; then
  # Check curve fit. Worst residual should be < 0.30 for "supported".
  WORST=$(echo "$FIT_RESULT" | grep -oE 'worst_residual=[0-9.]+' \
          | cut -d= -f2 || echo 1)
  if python3 -c "import sys; sys.exit(0 if float('$WORST') < 0.30 else 1)" 2>/dev/null; then
    emit_json "supported" \
      "All ${N_COMPLETED}/${N_TOTAL} sizes completed; O(N²) fit residual ${WORST} < 0.30"
    exit 0
  else
    emit_json "inconclusive" \
      "All sizes completed but O(N²) fit poor (worst residual ${WORST}); see fit details"
    exit 2
  fi
elif [[ "$N_COMPLETED" -ge 1 ]]; then
  emit_json "rejected" \
    "Cliff detected: ${N_COMPLETED}/${N_TOTAL} sizes completed; failure at size=${FAILED_SIZE}"
  exit 1
else
  emit_json "rejected" \
    "No sizes completed; smallest size (${SIZE_LIST[0]}) failed"
  exit 1
fi
