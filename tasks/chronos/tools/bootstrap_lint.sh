#!/usr/bin/env bash
# bootstrap_lint.sh
# Heuristic linter for files in the OpenClaw bootstrap path. Surfaces
# patterns that bloat the prompt without proportional value: long
# lists, repeated boilerplate, near-duplicate sections across files,
# and content that has no business being in a system prompt at all.
#
# READ-ONLY: this script never modifies anything. It prints
# observations and suggested cuts. You decide what to keep.
#
# Usage:
#   bootstrap_lint.sh                 # lint default workspace
#   bootstrap_lint.sh --workspace DIR
#   bootstrap_lint.sh --json          # machine-readable
#   bootstrap_lint.sh --threshold 100 # min lines to flag a file as "long"

set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/.openclaw/workspace}"
LINE_THRESHOLD="${LINE_THRESHOLD:-100}"
JSON_OUT="false"

TRACKED_FILES=(
  "AGENTS.md"
  "SOUL.md"
  "IDENTITY.md"
  "USER.md"
  "TOOLS.md"
  "MEMORY.md"
  "HEARTBEAT.md"
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --threshold) LINE_THRESHOLD="$2"; shift 2 ;;
    --json)      JSON_OUT="true"; shift ;;
    -h|--help)
      sed -n '/^# bootstrap_lint.sh/,/^$/p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Workspace not found: $WORKSPACE" >&2
  exit 3
fi

# Each finding is a tab-separated row: file, severity, code, message.
FINDINGS=$(mktemp)
DUP_TMP=$(mktemp)
trap 'rm -f "$FINDINGS" "$DUP_TMP"' EXIT

flag() {
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$FINDINGS"
}

# ── Per-file checks ─────────────────────────────────────────────────────────
for file in "${TRACKED_FILES[@]}"; do
  path="$WORKSPACE/$file"
  [[ -f "$path" ]] || continue

  bytes=$(wc -c < "$path" | tr -d ' ')
  lines=$(wc -l < "$path" | tr -d ' ')
  chars=$(wc -m < "$path" | tr -d ' ')

  # Check 1: file is long enough to warrant scrutiny.
  if [[ "$lines" -gt "$LINE_THRESHOLD" ]]; then
    flag "$file" "warn" "long_file" \
      "$lines lines / $chars chars / ~$(( chars / 7 )) tokens — review for cuts"
  fi

  # Check 2: more than 12 markdown headers — likely a kitchen sink.
  headers=$(grep -cE '^#{1,6} ' "$path" || true)
  if [[ "$headers" -gt 12 ]]; then
    flag "$file" "warn" "many_sections" \
      "$headers markdown headers — file may be doing too many jobs; consider splitting or trimming"
  fi

  # Check 3: deeply nested headers.
  deep_headers=$(grep -cE '^#{5,6} ' "$path" || true)
  if [[ "$deep_headers" -gt 2 ]]; then
    flag "$file" "info" "deep_nesting" \
      "$deep_headers headers at h5+; consider flattening structure"
  fi

  # Check 4: bullet-heavy lists.
  bullets=$(grep -cE '^[[:space:]]*- ' "$path" || true)
  if [[ "$bullets" -gt 50 ]]; then
    flag "$file" "warn" "list_heavy" \
      "$bullets bullet items — long lists rarely earn their token cost"
  fi

  # Check 5: code fences. Often justified, but worth flagging volume.
  fence_lines=$(awk '/^```/ {f=!f; next} f' "$path" | wc -l | tr -d ' ')
  if [[ "$fence_lines" -gt 30 ]]; then
    flag "$file" "info" "code_block_volume" \
      "$fence_lines lines inside code fences — verify each block is needed in the system prompt"
  fi

  # Check 6: emoji density.
  emoji_count=$(python3 -c "
import sys, re
with open('$path', 'rb') as fh:
    text = fh.read().decode('utf-8', errors='ignore')
emoji_pattern = re.compile(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]')
print(len(emoji_pattern.findall(text)))
" 2>/dev/null || echo 0)
  if [[ "$emoji_count" -gt 5 ]]; then
    flag "$file" "info" "emoji_density" \
      "$emoji_count emoji — every one costs ~1 token of behaviourally-neutral fluff"
  fi

  # Check 7: example blocks.
  example_count=$(grep -cE -i '^(##|###).*example|^Examples:|For example' "$path" || true)
  if [[ "$example_count" -gt 3 ]]; then
    flag "$file" "info" "example_heavy" \
      "$example_count example sections — system prompts rarely need more than 1-2"
  fi

  # Check 8: meta-commentary about the file itself.
  meta_phrases='this file|this document|this readme|update this file|edit this'
  meta_count=$(grep -ciE "$meta_phrases" "$path" || true)
  if [[ "$meta_count" -gt 2 ]]; then
    flag "$file" "info" "self_referential" \
      "$meta_count self-referential phrases — meta-commentary is overhead the model doesn't need"
  fi

  # Check 9: HTML/script artifacts.
  if grep -qE '<details>|<summary>|<script' "$path"; then
    flag "$file" "warn" "html_artifact" \
      "HTML elements detected — markdown system prompts rarely benefit from these"
  fi

  # Check 10: passages copied from a Settings/UI panel.
  if grep -qE '^Settings$|^Claude Code$|^Tool access$' "$path"; then
    flag "$file" "warn" "ui_dump" \
      "Looks like a copy-paste from a UI/settings panel — this content has no business in a system prompt"
  fi

  # Check 11: unusually long single line.
  longest_line=$(awk '{ if (length > max) max = length } END { print max+0 }' "$path")
  if [[ "$longest_line" -gt 500 ]]; then
    flag "$file" "info" "long_line" \
      "Longest line is $longest_line chars — consider whether it's actually instructive content or accidental paste"
  fi
done

# ── Cross-file checks ───────────────────────────────────────────────────────
# Detect lines that appear (verbatim, ≥40 chars) in two or more tracked
# files. Catches copy-paste of identical instructions across files.

for file in "${TRACKED_FILES[@]}"; do
  path="$WORKSPACE/$file"
  [[ -f "$path" ]] || continue
  awk -v fn="$file" '
    {
      line = $0
      sub(/^[[:space:]]+/, "", line)
      if (length(line) < 40) next
      if (line ~ /^#+ /) next
      if (line ~ /^- ?$/) next
      if (line ~ /^[`*_-]+$/) next
      print fn "\t" line
    }
  ' "$path"
done | sort -t$'\t' -k2 | awk -F'\t' '
  {
    if ($2 == prev_line) {
      files[$1] = 1
      n_lines++
    } else {
      if (n_lines >= 1) {
        nf = 0; flist = ""
        for (f in files) { nf++; flist = flist (flist ? "," : "") f }
        if (nf >= 2) print flist "\t" prev_line
      }
      delete files
      files[$1] = 1
      n_lines = 0
      prev_line = $2
    }
  }
  END {
    nf = 0; flist = ""
    for (f in files) { nf++; flist = flist (flist ? "," : "") f }
    if (nf >= 2) print flist "\t" prev_line
  }
' | sort -u > "$DUP_TMP" || true

dup_count=$(wc -l < "$DUP_TMP" | tr -d ' ')
if [[ "$dup_count" -gt 0 ]]; then
  awk -F'\t' '{ pairs[$1]++ } END { for (p in pairs) print p "\t" pairs[p] }' \
    "$DUP_TMP" | sort -t$'\t' -k2 -nr | head -10 | \
  while IFS=$'\t' read -r pair count; do
    flag "(cross-file)" "warn" "duplicated_lines" \
      "$count duplicate lines (≥40 chars) shared between: $pair"
  done
fi

# ── Output ──────────────────────────────────────────────────────────────────
if [[ "$JSON_OUT" == "true" ]]; then
  jq -nR --slurpfile findings <(jq -R 'split("\t") | {file:.[0], severity:.[1], code:.[2], message:.[3]}' "$FINDINGS") \
    '{workspace:env.WORKSPACE // "", findings:$findings}'
  exit 0
fi

echo "Workspace: $WORKSPACE"
echo

if [[ ! -s "$FINDINGS" ]]; then
  echo "No findings. Bootstrap files look reasonable by all current heuristics."
  exit 0
fi

echo "Findings (read-only — nothing changed on disk):"
echo

current_file=""
sort -t$'\t' -k1,1 -k2,2 -k3,3 "$FINDINGS" | while IFS=$'\t' read -r file severity code message; do
  if [[ "$file" != "$current_file" ]]; then
    echo
    echo "── $file ──"
    current_file="$file"
  fi
  case "$severity" in
    warn) prefix="[WARN]" ;;
    info) prefix="[info]" ;;
    *)    prefix="[$severity]" ;;
  esac
  echo "  $prefix $code: $message"
done

echo
echo "──────"
echo "These are heuristics, not rules. Some files genuinely need length"
echo "(USER.md is allowed to be opinionated about its human). The signal"
echo "to act on is when several findings stack on the same file — that's"
echo "where the bloat-vs-value ratio is worst."
echo
echo "Pair this with bootstrap_tokens.sh to see actual token cost per file."
