#!/bin/bash
# ============================================================================
# nestor-compress-memory.sh — Compress daily notes via local Ollama
# ============================================================================
# Purpose:  Send each daily note to gemma4-think:26b (think=off) with a
#           tight extraction prompt. Save compressed versions alongside
#           originals. Everything runs locally — nothing leaves miktam02.
#
# Usage:    chmod +x nestor-compress-memory.sh && ./nestor-compress-memory.sh
# ============================================================================

set -euo pipefail

MEMORY_DIR="$HOME/.openclaw/workspace/memory"
PROJECT_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench"
OUTPUT_DIR="$PROJECT_DIR/benchmarks/compressed-notes"
RESULTS_FILE="$PROJECT_DIR/benchmarks/compression_$(date +%Y%m%d_%H%M%S).csv"
OLLAMA_API="http://localhost:11434"
MODEL="gemma4-think:26b"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  $1${RESET}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

info()  { echo -e "  ${DIM}→${RESET} $1"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $1"; }
fail()  { echo -e "  ${RED}✗${RESET} $1"; }

COMPRESS_PROMPT='You are a memory compactor. Given a daily note from an AI assistant session, extract ONLY:

1. DECISIONS made (what was chosen and why)
2. LESSONS learned (what worked, what failed, what surprised)
3. BLOCKERS or open issues
4. KEY FACTS discovered (configs, numbers, thresholds)

Rules:
- Output 5-8 lines maximum
- No metadata, no session IDs, no timestamps, no conversation markers
- No "user said" / "assistant said" — just the distilled knowledge
- Use present tense for facts, past tense for events
- If the note contains nothing worth remembering, output: NO_SIGNAL

Format:
- Decisions: one line each, starting with "DECIDED:"
- Lessons: one line each, starting with "LEARNED:"
- Blockers: one line each, starting with "BLOCKER:"
- Facts: one line each, starting with "FACT:"'

# --- Preflight ---
banner "MEMORY COMPRESSION"
echo -e "  ${DIM}Model:     $MODEL (think=off)${RESET}"
echo -e "  ${DIM}Source:     $MEMORY_DIR/2026-*.md${RESET}"
echo -e "  ${DIM}Output:    $OUTPUT_DIR/${RESET}"

if ! curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
    fail "Ollama not running"
    exit 1
fi
ok "Ollama running"

if ! command -v jq &> /dev/null; then
    fail "jq required. brew install jq"
    exit 1
fi
ok "jq available"

mkdir -p "$OUTPUT_DIR"
echo "date,original_lines,original_words,original_tokens_est,compressed_lines,compressed_words,compressed_tokens_est,compression_ratio,total_sec" > "$RESULTS_FILE"

NOTES=($(ls "$MEMORY_DIR"/2026-*.md 2>/dev/null))
TOTAL=${#NOTES[@]}

if [[ $TOTAL -eq 0 ]]; then
    fail "No daily notes found in $MEMORY_DIR"
    exit 1
fi

ok "Found $TOTAL notes to compress"
echo ""

# --- Process each note ---
echo -e "  ${DIM}  ┌────────────┬────────┬────────┬────────┬────────┬───────────┬──────────┐${RESET}"
echo -e "  ${DIM}  │ Date       │ Orig L │ Orig W │ Comp L │ Comp W │ Ratio     │ Time     │${RESET}"
echo -e "  ${DIM}  ├────────────┼────────┼────────┼────────┼────────┼───────────┼──────────┤${RESET}"

TOTAL_ORIG_WORDS=0
TOTAL_COMP_WORDS=0
SKIPPED=0

for note in "${NOTES[@]}"; do
    date=$(basename "$note" .md)
    orig_lines=$(wc -l < "$note" | tr -d ' ')
    orig_words=$(wc -w < "$note" | tr -d ' ')
    orig_tokens=$(echo "$orig_words" | awk '{printf "%.0f", $1 / 0.75}')

    # Skip very short notes (less than 5 lines — nothing to compress)
    if [[ $orig_lines -lt 5 ]]; then
        cp "$note" "$OUTPUT_DIR/$date.md"
        echo "$date,$orig_lines,$orig_words,$orig_tokens,$orig_lines,$orig_words,$orig_tokens,1.0,0" >> "$RESULTS_FILE"
        printf "  ${DIM}  │${RESET} %-10s ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %7s   ${DIM}│${RESET} skipped  ${DIM}│${RESET}\n" \
            "$date" "$orig_lines" "$orig_words" "-" "-" "1.0x"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    NOTE_CONTENT=$(cat "$note")

    # Send to Ollama — think=off, tight prompt
    RESPONSE=$(curl -sf "$OLLAMA_API/api/chat" \
        -d "$(jq -n \
            --arg model "$MODEL" \
            --arg system "$COMPRESS_PROMPT" \
            --arg content "$NOTE_CONTENT" \
            '{
                model: $model,
                messages: [
                    {role: "system", content: $system},
                    {role: "user", content: ("Compress this daily note:\n\n" + $content)}
                ],
                stream: false,
                think: false,
                options: {num_ctx: 16384}
            }'
        )" 2>/dev/null)

    if [[ -z "$RESPONSE" ]]; then
        fail "  Failed to compress $date"
        cp "$note" "$OUTPUT_DIR/$date.md"
        echo "$date,$orig_lines,$orig_words,$orig_tokens,$orig_lines,$orig_words,$orig_tokens,1.0,0" >> "$RESULTS_FILE"
        continue
    fi

    # Extract compressed text and timing
    COMPRESSED=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('message', {}).get('content', ''))
" 2>/dev/null)

    TOTAL_SEC=$(echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'{d.get(\"total_duration\", 0)/1e9:.2f}')
" 2>/dev/null)

    # Save compressed note with header
    {
        echo "# $date (compressed)"
        echo ""
        echo "$COMPRESSED"
    } > "$OUTPUT_DIR/$date.md"

    comp_lines=$(echo "$COMPRESSED" | wc -l | tr -d ' ')
    comp_words=$(echo "$COMPRESSED" | wc -w | tr -d ' ')
    comp_tokens=$(echo "$comp_words" | awk '{printf "%.0f", $1 / 0.75}')

    if [[ $comp_words -gt 0 ]]; then
        ratio=$(echo "$orig_words $comp_words" | awk '{printf "%.1f", $1/$2}')
    else
        ratio="N/A"
    fi

    echo "$date,$orig_lines,$orig_words,$orig_tokens,$comp_lines,$comp_words,$comp_tokens,$ratio,$TOTAL_SEC" >> "$RESULTS_FILE"

    TOTAL_ORIG_WORDS=$((TOTAL_ORIG_WORDS + orig_words))
    TOTAL_COMP_WORDS=$((TOTAL_COMP_WORDS + comp_words))

    printf "  ${DIM}  │${RESET} %-10s ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %7sx  ${DIM}│${RESET} %6ss  ${DIM}│${RESET}\n" \
        "$date" "$orig_lines" "$orig_words" "$comp_lines" "$comp_words" "$ratio" "$TOTAL_SEC"

    # Brief pause between notes
    sleep 1
done

echo -e "  ${DIM}  └────────────┴────────┴────────┴────────┴────────┴───────────┴──────────┘${RESET}"

# --- Summary ---
banner "COMPRESSION RESULTS"

if [[ $TOTAL_COMP_WORDS -gt 0 ]]; then
    OVERALL_RATIO=$(echo "$TOTAL_ORIG_WORDS $TOTAL_COMP_WORDS" | awk '{printf "%.1f", $1/$2}')
else
    OVERALL_RATIO="N/A"
fi

echo -e "  ${BOLD}Notes processed:${RESET}     $((TOTAL - SKIPPED)) / $TOTAL"
echo -e "  ${BOLD}Skipped (too short):${RESET} $SKIPPED"
echo -e "  ${BOLD}Original words:${RESET}      $TOTAL_ORIG_WORDS"
echo -e "  ${BOLD}Compressed words:${RESET}    $TOTAL_COMP_WORDS"
echo -e "  ${BOLD}Overall ratio:${RESET}       ${YELLOW}${OVERALL_RATIO}x${RESET}"
echo ""
echo -e "  ${DIM}Compressed notes saved to: $OUTPUT_DIR/${RESET}"
echo -e "  ${DIM}CSV results: $RESULTS_FILE${RESET}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo -e "  ${DIM}  1. Review a few compressed notes: ls $OUTPUT_DIR/${RESET}"
echo -e "  ${DIM}  2. Re-run Phase 2 benchmark against compressed notes${RESET}"
echo -e "  ${DIM}  3. Compare: baseline 91s vs compressed Xs${RESET}"
echo ""

ok "Done. All processing was local — nothing left miktam02."
