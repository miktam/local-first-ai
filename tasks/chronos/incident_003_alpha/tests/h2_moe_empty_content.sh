#!/usr/bin/env bash
# tests/h2_moe_empty_content.sh
# H2: gemma4:26b MoE returns empty content with done_reason "stop" on long prompts.
#
# No sudo required. Sweeps system prompt size against gemma4:26b and
# (if available) gemma4:31b, three repeats per cell.

set -euo pipefail

H_ID="H2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "${ROOT_DIR}/lib/ollama_probe.sh"

MODELS_TO_TEST=("gemma4:26b" "gemma4:31b")
SIZES=(200 1000 2000 5000)
REPEATS=3
NUM_CTX="${NUM_CTX:-32768}"

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

log "Hypothesis: $H_ID — MoE empty content on long prompts"
log "Evidence: $EVIDENCE_DIR"

if ! ollama_is_up; then
  emit_json "could_not_run" "Ollama not reachable at $OLLAMA_HOST"
  exit 3
fi

# Filter to models that are actually pulled.
AVAILABLE_MODELS=()
for m in "${MODELS_TO_TEST[@]}"; do
  if ollama_has_model "$m"; then
    AVAILABLE_MODELS+=("$m")
    log "Model available: $m"
  else
    log "Model not pulled, skipping: $m"
  fi
done

if [[ ${#AVAILABLE_MODELS[@]} -eq 0 ]]; then
  emit_json "could_not_run" "Neither gemma4:26b nor gemma4:31b is pulled"
  exit 3
fi

# Generate filler text of N characters of deterministic content.
filler_chars() {
  local n="$1"
  python3 -c "
import sys
n=$n
phrase='Lorem ipsum dolor sit amet, consectetur adipiscing elit. '
out=(phrase * (n // len(phrase) + 1))[:n]
sys.stdout.write(out)
"
}

RESULTS_TSV="$EVIDENCE_DIR/results.tsv"
echo -e "model\tsize_chars\trepeat\tcontent_len\teval_count\tdone_reason\tprompt_eval_count" \
  > "$RESULTS_TSV"

USER_MSG="Please reply with the single word: ACK."

for model in "${AVAILABLE_MODELS[@]}"; do
  for size in "${SIZES[@]}"; do
    for r in $(seq 1 "$REPEATS"); do
      log "[$model] size=$size repeat=$r"
      sysp=$(filler_chars "$size")
      body=$(jq -nc \
        --arg m "$model" \
        --arg sysp "$sysp" \
        --arg userp "$USER_MSG" \
        --argjson nc "$NUM_CTX" \
        '{model:$m,
          messages:[
            {role:"system", content:$sysp},
            {role:"user",   content:$userp}
          ],
          stream:false,
          options:{num_ctx:$nc}}')

      raw="$EVIDENCE_DIR/raw_${model//[:\/]/_}_${size}_r${r}.json"
      if curl -fsS --max-time 600 "${OLLAMA_HOST}/api/chat" -d "$body" \
         > "$raw"; then
        content=$(jq -r '.message.content // ""' "$raw")
        eval_count=$(jq -r '.eval_count // 0' "$raw")
        done_reason=$(jq -r '.done_reason // ""' "$raw")
        prompt_eval=$(jq -r '.prompt_eval_count // 0' "$raw")
        clen=${#content}
      else
        log "  request failed"
        clen=-1
        eval_count=-1
        done_reason="ERROR"
        prompt_eval=-1
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model" "$size" "$r" "$clen" "$eval_count" "$done_reason" "$prompt_eval" \
        >> "$RESULTS_TSV"
    done
  done
done

log "Results written to $RESULTS_TSV"

# Decision: did gemma4:26b return empty content at any tested size in
# at least 2 of 3 repeats?
EMPTY_HITS_26B=0
NON_EMPTY_31B_AT_LARGEST=0

# 26b: count cells where >=2 of 3 repeats had content_len == 0 with done_reason stop
for size in "${SIZES[@]}"; do
  hits=$(awk -F'\t' -v sz="$size" \
    '$1=="gemma4:26b" && $2==sz && $4==0 && $6=="stop" { c++ } END { print c+0 }' \
    "$RESULTS_TSV")
  if [[ "$hits" -ge 2 ]]; then
    EMPTY_HITS_26B=$((EMPTY_HITS_26B + 1))
    log "  gemma4:26b returned empty on $hits/3 repeats at size=$size"
  fi
done

# 31b at largest size: count non-empty
LARGEST="${SIZES[$((${#SIZES[@]} - 1))]}"
NON_EMPTY_31B_AT_LARGEST=$(awk -F'\t' -v sz="$LARGEST" \
  '$1=="gemma4:31b" && $2==sz && $4>0 { c++ } END { print c+0 }' \
  "$RESULTS_TSV")

# Persist summary.
jq -nc \
  --argjson empty_hits_26b "$EMPTY_HITS_26B" \
  --argjson nonempty_31b_at_largest "$NON_EMPTY_31B_AT_LARGEST" \
  --arg results_file "$RESULTS_TSV" \
  '{empty_size_cells_26b:$empty_hits_26b,
    nonempty_31b_at_largest:$nonempty_31b_at_largest,
    results:$results_file}' \
  > "$EVIDENCE_DIR/run.json"

# Decision rule.
HAS_31B="false"
for m in "${AVAILABLE_MODELS[@]}"; do
  [[ "$m" == "gemma4:31b" ]] && HAS_31B="true"
done

if [[ $EMPTY_HITS_26B -ge 1 && "$HAS_31B" == "true" && $NON_EMPTY_31B_AT_LARGEST -ge 2 ]]; then
  emit_json "supported" \
    "26b empty at $EMPTY_HITS_26B size cell(s); 31b non-empty at largest size in $NON_EMPTY_31B_AT_LARGEST/3"
  exit 0
elif [[ $EMPTY_HITS_26B -ge 1 && "$HAS_31B" == "false" ]]; then
  emit_json "supported" \
    "26b empty at $EMPTY_HITS_26B size cell(s); 31b not pulled, cross-arch comparison absent"
  exit 0
elif [[ $EMPTY_HITS_26B -eq 0 ]]; then
  emit_json "rejected" \
    "26b returned non-empty content across all sizes and repeats"
  exit 1
else
  emit_json "inconclusive" \
    "26b empty at $EMPTY_HITS_26B cells; 31b non-empty count $NON_EMPTY_31B_AT_LARGEST"
  exit 2
fi
