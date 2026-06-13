#!/bin/bash
# ============================================================================
# nestor-bench-phase2-compare.sh — Before/After Memory Timing
# ============================================================================
# Purpose:  Compare inference cost with raw notes vs compressed notes.
#           Same prompt, same model, same settings. Only the memory changes.
#
# Usage:    chmod +x nestor-bench-phase2-compare.sh && ./nestor-bench-phase2-compare.sh
# ============================================================================

set -euo pipefail

MEMORY_DIR="$HOME/.openclaw/workspace/memory"
PROJECT_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench"
COMPRESSED_DIR="$PROJECT_DIR/benchmarks/compressed-notes"
RESULTS_FILE="$PROJECT_DIR/benchmarks/phase2_compare_$(date +%Y%m%d_%H%M%S).csv"
OLLAMA_API="http://localhost:11434"
MODEL="gemma4-think:26b"
RUNS=3

TEST_PROMPTS=(
    "What decisions did I make last week?"
    "What are my current blockers?"
    "Summarise what I learned about Ollama."
)

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

# --- Preflight ---
banner "PHASE 2: Memory Timing Comparison"
echo -e "  ${DIM}Model:      $MODEL (think=off)${RESET}"
echo -e "  ${DIM}Runs:       $RUNS per prompt per memory type${RESET}"
echo -e "  ${DIM}Prompts:    ${#TEST_PROMPTS[@]}${RESET}"
echo -e "  ${DIM}Tests:      3 modes × ${#TEST_PROMPTS[@]} prompts × $RUNS runs = $((3 * ${#TEST_PROMPTS[@]} * RUNS)) total${RESET}"

if ! curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
    fail "Ollama not running"
    exit 1
fi
ok "Ollama running"

if ! command -v jq &> /dev/null; then
    fail "jq required"
    exit 1
fi
ok "jq available"

# Load memory contents
RAW_MEMORY=$(cat "$MEMORY_DIR"/2026-*.md 2>/dev/null)
COMPRESSED_MEMORY=$(cat "$COMPRESSED_DIR"/2026-*.md 2>/dev/null)

RAW_WORDS=$(echo "$RAW_MEMORY" | wc -w | tr -d ' ')
COMP_WORDS=$(echo "$COMPRESSED_MEMORY" | wc -w | tr -d ' ')

info "Raw memory:        $RAW_WORDS words"
info "Compressed memory: $COMP_WORDS words"

echo "test_id,prompt,memory_type,run,prompt_tokens,prompt_tps,gen_tokens,gen_tps,total_sec" > "$RESULTS_FILE"

# --- Helper ---
run_test() {
    local prompt="$1"
    local memory_type="$2"  # none, raw, compressed
    local memory_content="$3"

    if [[ "$memory_type" == "none" ]]; then
        curl -sf "$OLLAMA_API/api/chat" \
            -d "$(jq -n \
                --arg model "$MODEL" \
                --arg content "$prompt" \
                '{
                    model: $model,
                    messages: [{role: "user", content: $content}],
                    stream: false,
                    think: false,
                    options: {num_ctx: 130000}
                }'
            )" 2>/dev/null
    else
        curl -sf "$OLLAMA_API/api/chat" \
            -d "$(jq -n \
                --arg model "$MODEL" \
                --arg memory "$memory_content" \
                --arg content "$prompt" \
                '{
                    model: $model,
                    messages: [
                        {role: "system", content: ("You are Nestor. Here are your memory notes:\n\n" + $memory)},
                        {role: "user", content: $content}
                    ],
                    stream: false,
                    think: false,
                    options: {num_ctx: 130000}
                }'
            )" 2>/dev/null
    fi
}

parse_result() {
    python3 -c "
import sys, json
d = json.load(sys.stdin)
pt = d.get('prompt_eval_count', 0)
pd = d.get('prompt_eval_duration', 1)
gt = d.get('eval_count', 0)
gd = d.get('eval_duration', 1)
td = d.get('total_duration', 1)
print(f'{pt},{pt/pd*1e9:.1f},{gt},{gt/gd*1e9:.1f},{td/1e9:.2f}')
"
}

# --- Run tests ---
TEST_NUM=0
for prompt in "${TEST_PROMPTS[@]}"; do
    TEST_NUM=$((TEST_NUM + 1))
    PROMPT_SHORT="${prompt:0:40}"

    banner "PROMPT $TEST_NUM: \"$PROMPT_SHORT...\""

    for memory_type in "none" "raw" "compressed"; do
        case $memory_type in
            none)       LABEL="No memory" ; MEM_CONTENT="" ;;
            raw)        LABEL="Raw notes" ; MEM_CONTENT="$RAW_MEMORY" ;;
            compressed) LABEL="Compressed" ; MEM_CONTENT="$COMPRESSED_MEMORY" ;;
        esac

        echo ""
        echo -e "  ${BOLD}$LABEL${RESET}"
        echo -e "  ${DIM}  ┌──────┬────────────┬──────────┬───────────┐${RESET}"
        echo -e "  ${DIM}  │ Run  │ Prompt tok │ Gen t/s  │ Total sec │${RESET}"
        echo -e "  ${DIM}  ├──────┼────────────┼──────────┼───────────┤${RESET}"

        for run in $(seq 1 $RUNS); do
            RESPONSE=$(run_test "$prompt" "$memory_type" "$MEM_CONTENT")

            if [[ -z "$RESPONSE" ]]; then
                fail "  │  $run   │ FAILED     │ FAILED   │ FAILED    │"
                echo "$TEST_NUM,$PROMPT_SHORT,$memory_type,$run,0,0,0,0,0" >> "$RESULTS_FILE"
                continue
            fi

            PARSED=$(echo "$RESPONSE" | parse_result)
            IFS=',' read -r pt ptps gt gtps total <<< "$PARSED"

            echo "$TEST_NUM,$PROMPT_SHORT,$memory_type,$run,$pt,$ptps,$gt,$gtps,$total" >> "$RESULTS_FILE"

            printf "  ${DIM}  │${RESET}  %d   ${DIM}│${RESET} %8s   ${DIM}│${RESET} %6s   ${DIM}│${RESET} %7s   ${DIM}│${RESET}\n" \
                "$run" "$pt" "$gtps" "$total"

            sleep 2
        done

        echo -e "  ${DIM}  └──────┴────────────┴──────────┴───────────┘${RESET}"
    done
done

# --- Summary ---
banner "RESULTS SUMMARY"
ok "CSV: $RESULTS_FILE"
echo ""

python3 -c "
import csv
from collections import defaultdict

data = defaultdict(lambda: {'prompt_tokens': [], 'total': []})

with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = row['memory_type']
        pt = int(row['prompt_tokens'])
        total = float(row['total_sec'])
        if pt > 0:
            data[key]['prompt_tokens'].append(pt)
            data[key]['total'].append(total)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

print('  ┌──────────────┬──────────────┬────────────┬────────────┐')
print('  │ Memory Type  │ Avg Pmt Toks │ Avg Time   │ vs None    │')
print('  ├──────────────┼──────────────┼────────────┼────────────┤')

none_time = avg(data['none']['total'])

for mtype in ['none', 'raw', 'compressed']:
    d = data[mtype]
    at = avg(d['prompt_tokens'])
    tt = avg(d['total'])
    if mtype == 'none':
        delta = '-'
    else:
        delta = f'+{tt - none_time:.1f}s'
    print(f'  │ {mtype:>12} │ {at:>10.0f}   │ {tt:>8.2f}s  │ {delta:>10} │')

print('  └──────────────┴──────────────┴────────────┴────────────┘')

raw_time = avg(data['raw']['total'])
comp_time = avg(data['compressed']['total'])
if raw_time > 0 and comp_time > 0:
    savings = raw_time - comp_time
    pct = (savings / raw_time) * 100
    print(f'')
    print(f'  Compression saves {savings:.1f}s per request ({pct:.0f}% faster)')
    print(f'  Over 40 messages/day: {savings * 40 / 60:.0f} minutes saved')
"

echo ""
ok "Done. This is the before/after for blog post 2."
