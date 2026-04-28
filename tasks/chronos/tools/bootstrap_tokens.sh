#!/usr/bin/env bash
# bootstrap_tokens.sh
# Measures the per-turn fixed overhead Nestor's stack ships to Ollama
# before any user conversation begins.
#
# Two modes:
#
#   1. ESTIMATE mode (default): chars / chars-per-token rule of thumb.
#      Fast, no model required. Good for routine checks.
#
#   2. EXACT mode (--exact): asks Ollama to tokenize the assembled
#      prompt. Slower but uses the real tokenizer.
#
# Usage:
#   bootstrap_tokens.sh                  # estimate
#   bootstrap_tokens.sh --exact          # exact via /api/tokenize
#   bootstrap_tokens.sh --json           # machine-readable output
#   bootstrap_tokens.sh --workspace DIR  # override default workspace
#
# Requires: bash, jq, find, awk, wc. Optional: curl (for --exact).

set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-$HOME/.openclaw/workspace}"
CHARS_PER_TOKEN="${CHARS_PER_TOKEN:-7}"   # calibrated for gemma4 on Latin text
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
MODEL="${MODEL:-gemma4-think:26b}"
MODE="estimate"
JSON_OUT="false"

# Files Nestor's bootstrap path actually loads. Order matches the
# system prompt's "Project Context files have been loaded" block.
TRACKED_FILES=(
  "AGENTS.md"
  "SOUL.md"
  "IDENTITY.md"
  "USER.md"
  "TOOLS.md"
  "MEMORY.md"
  "HEARTBEAT.md"
)

# ── Args ────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exact)        MODE="exact"; shift ;;
    --json)         JSON_OUT="true"; shift ;;
    --workspace)    WORKSPACE="$2"; shift 2 ;;
    --chars-per-token) CHARS_PER_TOKEN="$2"; shift 2 ;;
    -h|--help)
      sed -n '/^# bootstrap_tokens.sh/,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Workspace not found: $WORKSPACE" >&2
  exit 3
fi

# ── Per-file measurement ────────────────────────────────────────────────────
RESULTS_TSV="$(mktemp)"
TABLE_TMP="$(mktemp)"
trap 'rm -f "$RESULTS_TSV" "$TABLE_TMP"' EXIT

echo -e "file\texists\tbytes\tchars\ttokens_est\ttokens_exact" > "$RESULTS_TSV"

ollama_tokenize_count() {
  # Returns the token count for the given content via Ollama's tokenize
  # endpoint, or "-1" if the call fails.
  local content="$1"
  local response
  response=$(jq -nc --arg m "$MODEL" --arg c "$content" \
    '{model:$m, content:$c}' \
    | curl -fsS --max-time 30 "${OLLAMA_HOST}/api/tokenize" -d @- 2>/dev/null) || {
    echo "-1"
    return
  }
  echo "$response" | jq -r '.tokens | length' 2>/dev/null || echo "-1"
}

total_bytes=0
total_chars=0
total_tokens_est=0
total_tokens_exact=0

for file in "${TRACKED_FILES[@]}"; do
  path="$WORKSPACE/$file"
  if [[ -f "$path" ]]; then
    bytes=$(wc -c < "$path" | tr -d ' ')
    chars=$(wc -m < "$path" | tr -d ' ')
    tokens_est=$(( chars / CHARS_PER_TOKEN ))
    tokens_exact="-"
    if [[ "$MODE" == "exact" ]]; then
      tokens_exact=$(ollama_tokenize_count "$(cat "$path")")
    fi
    total_bytes=$(( total_bytes + bytes ))
    total_chars=$(( total_chars + chars ))
    total_tokens_est=$(( total_tokens_est + tokens_est ))
    if [[ "$tokens_exact" =~ ^[0-9]+$ ]]; then
      total_tokens_exact=$(( total_tokens_exact + tokens_exact ))
    fi
    printf '%s\ttrue\t%d\t%d\t%d\t%s\n' \
      "$file" "$bytes" "$chars" "$tokens_est" "$tokens_exact" \
      >> "$RESULTS_TSV"
  else
    printf '%s\tfalse\t0\t0\t0\t-\n' "$file" >> "$RESULTS_TSV"
  fi
done

# ── System prompt overhead estimate ─────────────────────────────────────────
# OpenClaw assembles a system prompt at runtime (tool catalogue, skill
# list, runtime block) that is not on disk in the workspace. Default
# baseline of 11000 chars matches the 2026-04-28 trajectory inspection.
# Update by examining a captured prompt.submitted record:
#   jq 'select(.type == "prompt.submitted") | .data.systemPrompt | length' \
#     ~/.openclaw/agents/main/sessions/<session>.trajectory.jsonl
SYSTEM_PROMPT_CHARS_BASELINE="${SYSTEM_PROMPT_CHARS_BASELINE:-11000}"
SYSTEM_PROMPT_TOKENS_EST=$(( SYSTEM_PROMPT_CHARS_BASELINE / CHARS_PER_TOKEN ))

# ── Output ──────────────────────────────────────────────────────────────────
if [[ "$JSON_OUT" == "true" ]]; then
  jq -n \
    --arg workspace "$WORKSPACE" \
    --arg mode "$MODE" \
    --argjson cpt "$CHARS_PER_TOKEN" \
    --argjson syschars "$SYSTEM_PROMPT_CHARS_BASELINE" \
    --argjson systokens "$SYSTEM_PROMPT_TOKENS_EST" \
    --argjson totalbytes "$total_bytes" \
    --argjson totalchars "$total_chars" \
    --argjson totalest "$total_tokens_est" \
    --argjson totalexact "$total_tokens_exact" \
    --rawfile tsv "$RESULTS_TSV" \
    '{
      workspace:$workspace, mode:$mode, chars_per_token:$cpt,
      system_prompt_baseline_chars:$syschars,
      system_prompt_tokens_estimated:$systokens,
      tracked_files_total_bytes:$totalbytes,
      tracked_files_total_chars:$totalchars,
      tracked_files_total_tokens_estimated:$totalest,
      tracked_files_total_tokens_exact:(if $totalexact > 0 then $totalexact else null end),
      bootstrap_overhead_tokens_estimated:($systokens + $totalest),
      tsv_raw:$tsv
    }'
  exit 0
fi

# Pretty output for humans.
echo "Workspace: $WORKSPACE"
echo "Mode: $MODE${MODE:+ (chars/token=$CHARS_PER_TOKEN)}"
echo
echo "Per-file overhead (sorted by token cost):"
echo

# Two-pass awk formatter. Avoid `column` because it's not installed
# by default on macOS.
{
  printf "FILE\tBYTES\tCHARS\tTOK_EST\tTOK_EXACT\n"
  tail -n +2 "$RESULTS_TSV" \
    | awk -F'\t' '$2=="true"' \
    | sort -t$'\t' -k5 -nr \
    | awk -F'\t' '{printf "%s\t%d\t%d\t%d\t%s\n", $1, $3, $4, $5, $6}'
} > "$TABLE_TMP"

awk -F'\t' '
  NR==FNR { for (i=1;i<=NF;i++) if (length($i)>w[i]) w[i]=length($i); next }
  { for (i=1;i<=NF;i++) printf "%-*s%s", w[i]+2, $i, (i==NF?"\n":"") }
' "$TABLE_TMP" "$TABLE_TMP"

echo
echo "Tracked files total:"
echo "  bytes  : $total_bytes"
echo "  chars  : $total_chars"
echo "  tokens : ~$total_tokens_est (estimated)"
if [[ "$MODE" == "exact" && "$total_tokens_exact" -gt 0 ]]; then
  echo "  tokens : $total_tokens_exact (exact)"
fi
echo
echo "System prompt baseline (assembled by OpenClaw, not in workspace):"
echo "  chars  : $SYSTEM_PROMPT_CHARS_BASELINE"
echo "  tokens : ~$SYSTEM_PROMPT_TOKENS_EST"
echo
echo "Bootstrap overhead per turn (tracked files + system prompt):"
echo "  ~$(( total_tokens_est + SYSTEM_PROMPT_TOKENS_EST )) tokens"
echo
echo "  At the conservative cliff (25k tokens), this leaves room for"
echo "  ~$(( 25000 - total_tokens_est - SYSTEM_PROMPT_TOKENS_EST )) tokens of conversation history."
echo "  At the practical ceiling (35k tokens), ~$(( 35000 - total_tokens_est - SYSTEM_PROMPT_TOKENS_EST )) tokens."
echo "  (Cliff measured at 25-35k on-the-wire tokens in incident_003_alpha.)"
echo
echo "Files with no entries above are not loaded for this turn."
echo "Update SYSTEM_PROMPT_CHARS_BASELINE if you have a recent"
echo "prompt.submitted record showing a different size."
