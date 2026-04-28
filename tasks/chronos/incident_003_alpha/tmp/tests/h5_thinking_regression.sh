#!/usr/bin/env bash
# tests/h5_thinking_regression.sh
# H5: Thinking mode re-engages at long context despite think:false.
#
# Tests whether gemma4-think:26b returns thinking content when called
# with "think": false at incident-scale prompts. Requires sudo
# NOPASSWD on /usr/bin/powermetrics. Plan for ~15–35 minutes wall
# time depending on whether the runaway manifests.
#
# Each repeat uses a unique seeded prompt so Ollama's prompt cache
# doesn't short-circuit subsequent requests (the lesson from the
# first H5 run: repeats 2 and 3 returned in 0–1s because the prompt
# was identical to repeat 1).
#
# Powermetrics samples GPU and CPU power per repeat so we can also
# observe whether long-context evaluation runs on GPU or falls back
# to CPU — orthogonal to thinking detection but cheap to capture.

set -euo pipefail

H_ID="H5"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/ollama_probe.sh"
# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/powermetrics_probe.sh"

MODEL="${MODEL:-gemma4-think:26b}"
INCIDENT_SCALE_TOKENS="${INCIDENT_SCALE_TOKENS:-40000}"
NUM_CTX="${NUM_CTX:-131072}"
REPEATS="${REPEATS:-3}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
PM_DURATION="${PM_DURATION:-700}"          # > REQUEST_TIMEOUT to cover full request

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

log "Hypothesis: $H_ID — thinking regression at long context"
log "Model: $MODEL"
log "Incident scale: ~${INCIDENT_SCALE_TOKENS} tokens"
log "Per-request timeout: ${REQUEST_TIMEOUT}s"
log "Repeats: $REPEATS (each with a unique seeded prompt)"
log "Evidence: $EVIDENCE_DIR"

if ! ollama_is_up; then
  emit_json "could_not_run" "Ollama not reachable at $OLLAMA_HOST"
  exit 3
fi

if ! ollama_has_model "$MODEL"; then
  emit_json "could_not_run" "Model $MODEL not present locally"
  exit 3
fi

# Powermetrics is optional for H5 — useful but not load-bearing for
# the decision rule. If unavailable, skip the sampling and warn.
PM_AVAILABLE="false"
if powermetrics_check_sudo; then
  PM_AVAILABLE="true"
  log "Powermetrics available: sampling per repeat"
else
  log "Powermetrics unavailable; thinking detection only, no power data"
fi

# ── Sanity check: think:false should suppress thinking at small scale ───────
log "── Sanity check: small prompt, think:false ──"

sanity_body=$(jq -nc --arg m "$MODEL" \
  '{model:$m,
    messages:[{role:"user", content:"Reply with the single word: OK."}],
    stream:false,
    think:false,
    options:{num_ctx:8192}}')

sanity_raw="$EVIDENCE_DIR/sanity_response.json"
if ! curl -fsS --max-time 60 "${OLLAMA_HOST}/api/chat" -d "$sanity_body" \
     > "$sanity_raw" 2> "$EVIDENCE_DIR/sanity_curl.err"; then
  log "Sanity request failed; cannot trust further results"
  emit_json "could_not_run" "Sanity request to $MODEL failed"
  exit 3
fi

sanity_content_len=$(jq -r '.message.content // "" | length' "$sanity_raw")
sanity_thinking_len=$(jq -r '.message.thinking // "" | length' "$sanity_raw")
sanity_inline_think=$(jq -r '.message.content // ""' "$sanity_raw" \
  | grep -c "<think>" || true)

log "Sanity: content_len=$sanity_content_len, thinking_len=$sanity_thinking_len, inline_think_tags=$sanity_inline_think"

SANITY_LEAKED="false"
if [[ "$sanity_thinking_len" -gt 0 || "$sanity_inline_think" -gt 0 ]]; then
  log "NOTE: sanity check shows thinking content despite think:false"
  log "  → regression may already be active at small scale"
  SANITY_LEAKED="true"
fi

if [[ "$sanity_content_len" -eq 0 ]]; then
  log "Sanity content is empty; environment may be broken"
  emit_json "could_not_run" "Sanity check returned empty content"
  exit 3
fi

# ── Three repeats at incident scale, each with a UNIQUE prompt ──────────────
THINKING_HITS=0
TIMEOUTS=0
COMPLETIONS=0
TOTAL_THINKING_CHARS=0
PER_REPEAT_SUMMARY="$EVIDENCE_DIR/per_repeat.tsv"
echo -e "repeat\telapsed_s\tprompt_eval_count\teval_count\tcontent_len\tthinking_len\tinline_think_tags\tgpu_mean_mw\tcpu_mean_mw\tdone_reason" \
  > "$PER_REPEAT_SUMMARY"

INSTRUCTION="\n\n=== END OF CONTEXT ===\n\nReply with the single word: ACK."

for r in $(seq 1 "$REPEATS"); do
  log "── Repeat $r/$REPEATS ──"

  # Generate a UNIQUE prompt for this repeat so Ollama's prompt cache
  # doesn't short-circuit. The seed is the timestamp + repeat index.
  variant="r${r}-${TS}"
  prompt_file="$EVIDENCE_DIR/prompt_${r}.txt"
  generate_filler_prompt "$INCIDENT_SCALE_TOKENS" "$variant" > "$prompt_file"
  prompt_bytes=$(wc -c < "$prompt_file")
  log "Prompt $r: ${prompt_bytes} bytes (seed=$variant)"

  # Start powermetrics if available.
  pm_file="$EVIDENCE_DIR/powermetrics_r${r}.txt"
  if [[ "$PM_AVAILABLE" == "true" ]]; then
    powermetrics_start "$pm_file" "$PM_DURATION"
  fi

  # Build request body.
  body=$(jq -nc \
    --arg m "$MODEL" \
    --rawfile p "$prompt_file" \
    --arg instr "$INSTRUCTION" \
    --argjson nc "$NUM_CTX" \
    '{model:$m,
      messages:[{role:"user", content:($p + $instr)}],
      stream:false,
      think:false,
      options:{num_ctx:$nc}}')

  raw="$EVIDENCE_DIR/repeat_${r}.json"
  err="$EVIDENCE_DIR/repeat_${r}.curl.err"

  start_time=$(date +%s)
  if curl -fsS --max-time "$REQUEST_TIMEOUT" \
       "${OLLAMA_HOST}/api/chat" -d "$body" \
       > "$raw" 2> "$err"; then
    elapsed=$(($(date +%s) - start_time))

    content_len=$(jq -r '.message.content // "" | length' "$raw")
    thinking_len=$(jq -r '.message.thinking // "" | length' "$raw")
    eval_count=$(jq -r '.eval_count // 0' "$raw")
    prompt_eval_count=$(jq -r '.prompt_eval_count // 0' "$raw")
    done_reason=$(jq -r '.done_reason // ""' "$raw")
    inline_think=$(jq -r '.message.content // ""' "$raw" \
      | grep -c "<think>" || true)

    log "  completed in ${elapsed}s"
    log "  prompt_eval=$prompt_eval_count eval=$eval_count content_len=$content_len thinking_len=$thinking_len inline_think=$inline_think done=$done_reason"

    COMPLETIONS=$((COMPLETIONS + 1))
    TOTAL_THINKING_CHARS=$((TOTAL_THINKING_CHARS + thinking_len))

    if [[ "$thinking_len" -gt 0 || "$inline_think" -gt 0 ]]; then
      THINKING_HITS=$((THINKING_HITS + 1))
      log "  → thinking content detected despite think:false"
    fi

    timed_out="no"
  else
    elapsed=$(($(date +%s) - start_time))
    log "  request failed/timed out after ${elapsed}s"
    cat "$err" >&2 || true
    TIMEOUTS=$((TIMEOUTS + 1))
    content_len=0
    thinking_len=0
    eval_count=0
    prompt_eval_count=0
    done_reason="TIMEOUT"
    inline_think=0
    timed_out="yes"
  fi

  # Stop powermetrics and compute windowed means.
  gpu_window="-1"
  cpu_window="-1"
  if [[ "$PM_AVAILABLE" == "true" ]]; then
    log "  Waiting for powermetrics to finish"
    powermetrics_stop
    window=$((elapsed + 2))
    gpu_window=$(powermetrics_mean_gpu_mw_window "$pm_file" "$window")
    cpu_window=$(powermetrics_mean_cpu_mw_window "$pm_file" "$window")
    log "  GPU windowed[0..${window}s]=${gpu_window}mW, CPU=${cpu_window}mW"
  fi

  printf '%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%s\t%s\n' \
    "$r" "$elapsed" "$prompt_eval_count" "$eval_count" \
    "$content_len" "$thinking_len" "$inline_think" \
    "$gpu_window" "$cpu_window" "$done_reason" \
    >> "$PER_REPEAT_SUMMARY"
done

# ── Persist run summary ─────────────────────────────────────────────────────
jq -nc \
  --argjson hits "$THINKING_HITS" \
  --argjson timeouts "$TIMEOUTS" \
  --argjson completions "$COMPLETIONS" \
  --argjson repeats "$REPEATS" \
  --argjson total_thinking_chars "$TOTAL_THINKING_CHARS" \
  --arg sanity_leaked "$SANITY_LEAKED" \
  --arg pm_available "$PM_AVAILABLE" \
  '{thinking_hits:$hits,
    timeouts:$timeouts,
    completions:$completions,
    repeats:$repeats,
    total_thinking_chars:$total_thinking_chars,
    sanity_leaked:($sanity_leaked == "true"),
    powermetrics_available:($pm_available == "true")}' \
  > "$EVIDENCE_DIR/run.json"

log "── Summary ──"
log "thinking_hits=$THINKING_HITS / $REPEATS"
log "timeouts=$TIMEOUTS / $REPEATS"
log "completions=$COMPLETIONS / $REPEATS"
log "total_thinking_chars=$TOTAL_THINKING_CHARS"
log "sanity_leaked=$SANITY_LEAKED"
log "Per-repeat detail: $PER_REPEAT_SUMMARY"

# ── Decision rule ───────────────────────────────────────────────────────────
if [[ "$THINKING_HITS" -ge 1 ]]; then
  emit_json "supported" \
    "Thinking content detected in $THINKING_HITS/$REPEATS repeats despite think:false (incident scale)"
  exit 0
elif [[ "$TIMEOUTS" -ge 2 && "$COMPLETIONS" -eq 0 ]]; then
  emit_json "inconclusive" \
    "Runaway-consistent: $TIMEOUTS/$REPEATS timed out at ${REQUEST_TIMEOUT}s, no completions; thinking content not directly observable"
  exit 2
elif [[ "$COMPLETIONS" -eq "$REPEATS" && "$THINKING_HITS" -eq 0 ]]; then
  emit_json "rejected" \
    "All $REPEATS repeats completed within timeout with no thinking content"
  exit 1
else
  emit_json "inconclusive" \
    "Mixed: hits=$THINKING_HITS timeouts=$TIMEOUTS completions=$COMPLETIONS"
  exit 2
fi
