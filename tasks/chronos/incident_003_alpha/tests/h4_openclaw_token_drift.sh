#!/usr/bin/env bash
# tests/h4_openclaw_token_drift.sh
# H4: OpenClaw caps input tokens to Ollama regardless of configured contextWindow.
#
# Purely passive analysis of existing session JSONL files. Safe to run
# at any time; sends no requests to Ollama or anywhere else.

set -euo pipefail

H_ID="H4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

SESSION_DIR="${SESSION_DIR:-$HOME/.openclaw/agents/main/sessions}"
CONFIG_CTX_WINDOW="${CONFIG_CTX_WINDOW:-131072}"

TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
EVIDENCE_DIR="${ROOT_DIR}/evidence/${TS}-${H_ID}"
mkdir -p "$EVIDENCE_DIR"

emit_json() {
  local status="$1" summary="$2"
  jq -nc \
    --arg id "$H_ID" \
    --arg status "$status" \
    --arg dir "$EVIDENCE_DIR" \
    --arg summary "$summary" \
    '{hypothesis_id:$id, status:$status, evidence_dir:$dir, summary:$summary}'
}

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }

log "Hypothesis: $H_ID — OpenClaw input token cap"
log "Session dir: $SESSION_DIR"
log "Configured contextWindow: $CONFIG_CTX_WINDOW"
log "Evidence: $EVIDENCE_DIR"

if [[ ! -d "$SESSION_DIR" ]]; then
  log "Session directory not found"
  emit_json "could_not_run" "Session directory $SESSION_DIR does not exist"
  exit 3
fi

# Find session files. Be permissive about what counts as a session log;
# OpenClaw has shipped slightly different layouts across versions.
SESSION_FILES=()
while IFS= read -r f; do
  SESSION_FILES+=("$f")
done < <(find "$SESSION_DIR" -type f -name '*.jsonl' 2>/dev/null)

if [[ ${#SESSION_FILES[@]} -eq 0 ]]; then
  log "No .jsonl session files found"
  emit_json "could_not_run" "No session files in $SESSION_DIR"
  exit 3
fi

log "Found ${#SESSION_FILES[@]} session file(s)"

# Extract every "input" token count we can find. OpenClaw versions vary;
# the field has shown up at top level and nested under usage/. Try both.
RAW_TOKENS="$EVIDENCE_DIR/raw_tokens.tsv"
echo -e "session_file\tinput_tokens" > "$RAW_TOKENS"

for f in "${SESSION_FILES[@]}"; do
  # Try several known shapes:
  #   {"input": 4096, ...}
  #   {"usage": {"input": 4096}, ...}
  #   {"usage": {"input_tokens": 4096}, ...}
  jq -r --arg fn "$f" '
    [
      .input,
      .usage.input,
      .usage.input_tokens,
      .tokens.input
    ] | map(select(type == "number")) | .[0] // empty
    | select(. != null) | "\($fn)\t\(.)"
  ' "$f" 2>/dev/null >> "$RAW_TOKENS" || true
done

COUNT=$(($(wc -l < "$RAW_TOKENS") - 1))
log "Extracted $COUNT input-token records"

if [[ $COUNT -eq 0 ]]; then
  log "Could not find any input token counts in session files"
  log "Inspect the session file format and update the jq filter above"
  cp "${SESSION_FILES[0]}" "$EVIDENCE_DIR/sample_session.jsonl"
  emit_json "could_not_run" \
    "No input-token fields recognised in session JSONL; sample saved to evidence dir"
  exit 3
fi

# Statistics via python (more readable than awk for this).
STATS_OUT="$EVIDENCE_DIR/stats.json"
python3 - <<EOF > "$STATS_OUT"
import csv, json, statistics
from collections import Counter
vals = []
with open("$RAW_TOKENS") as fh:
    r = csv.reader(fh, delimiter="\t")
    next(r, None)
    for row in r:
        if len(row) >= 2:
            try: vals.append(int(row[1]))
            except ValueError: pass
n = len(vals)
out = {"n": n}
if n:
    out["min"] = min(vals)
    out["max"] = max(vals)
    out["mean"] = statistics.mean(vals)
    out["median"] = statistics.median(vals)
    out["stdev"] = statistics.stdev(vals) if n > 1 else 0
    out["cv"] = (out["stdev"] / out["mean"]) if out["mean"] else 0
    counter = Counter(vals)
    out["mode_value"], out["mode_count"] = counter.most_common(1)[0]
    out["mode_share"] = out["mode_count"] / n
    out["histogram_top10"] = counter.most_common(10)
print(json.dumps(out, indent=2))
EOF

log "Statistics written to $STATS_OUT"
cat "$STATS_OUT" >&2

# Decision rule.
MODE_VALUE=$(jq -r '.mode_value' "$STATS_OUT")
MODE_SHARE=$(jq -r '.mode_share' "$STATS_OUT")
MEDIAN=$(jq -r '.median' "$STATS_OUT")
MAX=$(jq -r '.max' "$STATS_OUT")
QUARTER_WINDOW=$((CONFIG_CTX_WINDOW / 4))

# Strong form: mode is exactly 4096 and ≥80% of turns at the mode.
STRONG_HIT="false"
if [[ "$MODE_VALUE" == "4096" ]]; then
  if awk "BEGIN{exit !($MODE_SHARE >= 0.80)}"; then
    STRONG_HIT="true"
  fi
fi

# Weak form: median < contextWindow/4 across the dataset.
WEAK_HIT="false"
if awk "BEGIN{exit !($MEDIAN < $QUARTER_WINDOW)}"; then
  WEAK_HIT="true"
fi

# Falsification check: any single turn exceeded 8192?
EXCEEDED_8K="false"
if awk "BEGIN{exit !($MAX > 8192)}"; then
  EXCEEDED_8K="true"
fi

log "Strong-form (4096 cap, ≥80%): $STRONG_HIT"
log "Weak-form (median < ${QUARTER_WINDOW}): $WEAK_HIT"
log "Any turn > 8192: $EXCEEDED_8K"

if [[ "$STRONG_HIT" == "true" ]]; then
  emit_json "supported" \
    "Strong form: mode=$MODE_VALUE, mode_share=$MODE_SHARE, median=$MEDIAN, max=$MAX"
  exit 0
elif [[ "$WEAK_HIT" == "true" && "$EXCEEDED_8K" == "false" ]]; then
  emit_json "supported" \
    "Weak form: median=$MEDIAN < ${QUARTER_WINDOW}, max=$MAX, no turn exceeded 8192"
  exit 0
elif [[ "$EXCEEDED_8K" == "true" && "$STRONG_HIT" == "false" ]]; then
  emit_json "rejected" \
    "Strong form falsified: max=$MAX exceeded 8192; mode_share=$MODE_SHARE"
  exit 1
else
  emit_json "inconclusive" \
    "Mixed signal: mode=$MODE_VALUE share=$MODE_SHARE median=$MEDIAN max=$MAX"
  exit 2
fi
