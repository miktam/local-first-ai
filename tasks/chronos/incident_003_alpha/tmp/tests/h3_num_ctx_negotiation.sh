#!/usr/bin/env bash
# tests/h3_num_ctx_negotiation.sh
# H3: Ollama loads gemma4:26b with a smaller num_ctx than OpenClaw advertises.
#
# No sudo required. Sends three small generation requests to Ollama
# to compare reported vs. loaded context lengths.

set -euo pipefail

H_ID="H3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/ollama_probe.sh"

MODEL="${MODEL:-gemma4:26b}"
EXPLICIT_CTX="${EXPLICIT_CTX:-65536}"

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

log "Hypothesis: $H_ID — num_ctx negotiation drift"
log "Model: $MODEL"
log "Evidence: $EVIDENCE_DIR"

if ! ollama_is_up; then
  log "Ollama is not responding at $OLLAMA_HOST"
  emit_json "could_not_run" "Ollama not reachable at $OLLAMA_HOST"
  exit 3
fi

if ! ollama_has_model "$MODEL"; then
  log "Model $MODEL not pulled; \`ollama pull $MODEL\` first"
  emit_json "could_not_run" "Model $MODEL not present locally"
  exit 3
fi

# Step 1: what does /api/show advertise?
log "Step 1: querying /api/show"
SHOW_BODY="$EVIDENCE_DIR/api_show.json"
curl -fsS "${OLLAMA_HOST}/api/show" \
  -d "$(jq -nc --arg m "$MODEL" '{name:$m}')" \
  > "$SHOW_BODY"
SHOW_CTX=$(ollama_show_context_length "$MODEL")
log "Reported context_length: ${SHOW_CTX:-<unknown>}"

if [[ -z "$SHOW_CTX" ]]; then
  emit_json "could_not_run" "Could not read context_length from /api/show"
  exit 3
fi

# Step 2: load model with no num_ctx override, then check /api/ps.
log "Step 2: loading model with default num_ctx, then querying /api/ps"
ollama_load_model "$MODEL" >/dev/null
sleep 2
PS_DEFAULT="$EVIDENCE_DIR/api_ps_default.json"
ollama_ps_json > "$PS_DEFAULT"
DEFAULT_LOADED_CTX=$(ollama_ps_context_length "$MODEL")
log "Loaded context (default): ${DEFAULT_LOADED_CTX:-<unknown>}"

# Step 3: load model with explicit num_ctx, then check /api/ps.
log "Step 3: loading model with num_ctx=$EXPLICIT_CTX, then querying /api/ps"
ollama_load_model "$MODEL" "$EXPLICIT_CTX" >/dev/null
sleep 2
PS_EXPLICIT="$EVIDENCE_DIR/api_ps_explicit.json"
ollama_ps_json > "$PS_EXPLICIT"
EXPLICIT_LOADED_CTX=$(ollama_ps_context_length "$MODEL")
log "Loaded context (explicit ${EXPLICIT_CTX}): ${EXPLICIT_LOADED_CTX:-<unknown>}"

# Compute drift. Treat empty values as 0 for arithmetic safety.
SHOW_NUM=${SHOW_CTX:-0}
DEFAULT_NUM=${DEFAULT_LOADED_CTX:-0}
EXPLICIT_NUM=${EXPLICIT_LOADED_CTX:-0}
DRIFT=$((SHOW_NUM - DEFAULT_NUM))

# Persist a structured run summary.
jq -nc \
  --argjson show "$SHOW_NUM" \
  --argjson default_loaded "$DEFAULT_NUM" \
  --argjson explicit_loaded "$EXPLICIT_NUM" \
  --argjson drift "$DRIFT" \
  '{api_show_context_length:$show,
    default_loaded_context:$default_loaded,
    explicit_loaded_context:$explicit_loaded,
    drift:$drift}' \
  > "$EVIDENCE_DIR/run.json"

log "Drift (show − default_loaded): $DRIFT"

# Decision rule.
if [[ $DRIFT -gt 8000 ]]; then
  emit_json "supported" \
    "Drift=${DRIFT}: /api/show reports ${SHOW_NUM}, default load was ${DEFAULT_NUM}"
  exit 0
elif [[ $DEFAULT_NUM -ge 65536 ]]; then
  emit_json "rejected" \
    "Default-load context ${DEFAULT_NUM} matches advertised; no negotiation drift"
  exit 1
else
  emit_json "inconclusive" \
    "Drift=${DRIFT}: small but nonzero; manual inspection recommended"
  exit 2
fi
