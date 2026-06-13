#!/bin/bash
# ============================================================================
# nestor-bench-phase2-memory.sh — Memory Efficiency Analysis
# ============================================================================
# Purpose:  Measure the signal-to-noise ratio in Nestor's daily notes,
#           test memory search accuracy, and quantify the context cost
#           of uncompressed memory.
#
# Setup:    Mac Mini M4 Pro · 64GB · Ollama 0.20.2 · gemma4-think:26b
# Memory:   ~/.openclaw/workspace/memory/2026-*.md
#
# Usage:    chmod +x nestor-bench-phase2-memory.sh && ./nestor-bench-phase2-memory.sh
# ============================================================================

set -euo pipefail

MEMORY_DIR="$HOME/.openclaw/workspace/memory"
RESULTS_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/phase2_memory_$TIMESTAMP.csv"
ANALYSIS_FILE="$RESULTS_DIR/phase2_memory_analysis_$TIMESTAMP.md"

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

mkdir -p "$RESULTS_DIR"

# ============================================================================
# PART 1: Note inventory — what's actually in the memory folder?
# ============================================================================

banner "PART 1: Memory Inventory"

TOTAL_NOTES=$(ls "$MEMORY_DIR"/2026-*.md 2>/dev/null | wc -l | tr -d ' ')
TOTAL_LINES=$(cat "$MEMORY_DIR"/2026-*.md 2>/dev/null | wc -l | tr -d ' ')
TOTAL_BYTES=$(cat "$MEMORY_DIR"/2026-*.md 2>/dev/null | wc -c | tr -d ' ')
TOTAL_WORDS=$(cat "$MEMORY_DIR"/2026-*.md 2>/dev/null | wc -w | tr -d ' ')

# Estimate tokens (~0.75 words per token for English)
EST_TOKENS=$(echo "$TOTAL_WORDS" | awk '{printf "%.0f", $1 / 0.75}')

echo -e "  ${BOLD}Daily notes:${RESET}     $TOTAL_NOTES files"
echo -e "  ${BOLD}Total lines:${RESET}     $TOTAL_LINES"
echo -e "  ${BOLD}Total size:${RESET}      $(echo $TOTAL_BYTES | awk '{printf "%.1f KB", $1/1024}')"
echo -e "  ${BOLD}Total words:${RESET}     $TOTAL_WORDS"
echo -e "  ${BOLD}Est. tokens:${RESET}     $EST_TOKENS"

# ============================================================================
# PART 2: Noise analysis — how much is metadata vs content?
# ============================================================================

banner "PART 2: Noise Analysis"

# Count noise patterns
METADATA_LINES=$(grep -c "sender_id\|message_id\|session_id\|Session Key\|untrusted metadata\|sender_id\|timestamp\|label.*1808073" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
JSON_BLOCK_LINES=$(grep -c '```json\|```$\|"message_id"\|"sender"\|"sender_id"\|"timestamp"\|"username"\|"label"' "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
SESSION_HEADER_LINES=$(grep -c "Session Key\|Session ID\|Source: telegram\|# Session:" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
CONVERSATION_MARKERS=$(grep -c "^user:\|^assistant:" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')

TOTAL_NOISE=$((METADATA_LINES + JSON_BLOCK_LINES + SESSION_HEADER_LINES))
NOISE_PCT=$(echo "$TOTAL_NOISE $TOTAL_LINES" | awk '{if($2>0) printf "%.1f", ($1/$2)*100; else print "0"}')
SIGNAL_LINES=$((TOTAL_LINES - TOTAL_NOISE))

echo -e "  ${BOLD}Metadata lines:${RESET}          $METADATA_LINES"
echo -e "  ${BOLD}JSON block lines:${RESET}        $JSON_BLOCK_LINES"
echo -e "  ${BOLD}Session header lines:${RESET}    $SESSION_HEADER_LINES"
echo -e "  ${BOLD}Conversation markers:${RESET}    $CONVERSATION_MARKERS"
echo -e "  ${BOLD}Total noise lines:${RESET}       ${YELLOW}$TOTAL_NOISE${RESET}"
echo -e "  ${BOLD}Signal lines:${RESET}            ${GREEN}$SIGNAL_LINES${RESET}"
echo -e "  ${BOLD}Noise ratio:${RESET}             ${YELLOW}${NOISE_PCT}%${RESET}"

# ============================================================================
# PART 3: Per-note breakdown
# ============================================================================

banner "PART 3: Per-Note Breakdown"

echo "date,total_lines,noise_lines,signal_lines,noise_pct,bytes,est_tokens" > "$RESULTS_FILE"

echo -e "  ${DIM}  ┌────────────┬────────┬────────┬────────┬───────────┬──────────┐${RESET}"
echo -e "  ${DIM}  │ Date       │ Lines  │ Noise  │ Signal │ Noise %   │ Est Toks │${RESET}"
echo -e "  ${DIM}  ├────────────┼────────┼────────┼────────┼───────────┼──────────┤${RESET}"

for note in "$MEMORY_DIR"/2026-*.md; do
    date=$(basename "$note" .md)
    lines=$(wc -l < "$note" | tr -d ' ')
    bytes=$(wc -c < "$note" | tr -d ' ')
    words=$(wc -w < "$note" | tr -d ' ')
    tokens=$(echo "$words" | awk '{printf "%.0f", $1 / 0.75}')

    noise=$(grep -c "sender_id\|message_id\|session_id\|Session Key\|untrusted metadata\|label.*1808073\|\"timestamp\"\|\"username\"\|\"sender\"\|Session ID\|Source: telegram\|\`\`\`json\|\`\`\`$" "$note" 2>/dev/null || echo 0)
    signal=$((lines - noise))
    npct=$(echo "$noise $lines" | awk '{if($2>0) printf "%.0f", ($1/$2)*100; else print "0"}')

    echo "$date,$lines,$noise,$signal,$npct,$bytes,$tokens" >> "$RESULTS_FILE"

    printf "  ${DIM}  │${RESET} %-10s ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %5s  ${DIM}│${RESET} %7s%%  ${DIM}│${RESET} %7s  ${DIM}│${RESET}\n" \
        "$date" "$lines" "$noise" "$signal" "$npct" "$tokens"
done

echo -e "  ${DIM}  └────────────┴────────┴────────┴────────┴───────────┴──────────┘${RESET}"

# ============================================================================
# PART 4: Content type classification (sample)
# ============================================================================

banner "PART 4: Content Types (across all notes)"

DECISIONS=$(grep -ci "decided\|decision\|chose\|picked\|going with\|switched to\|will use\|resolved" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
LESSONS=$(grep -ci "learned\|lesson\|mistake\|insight\|reali[sz]ed\|turns out\|takeaway\|finding" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
TODOS=$(grep -ci "todo\|to-do\|need to\|should\|must\|next step\|action item\|follow up" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
COMMANDS=$(grep -ci "^\`\`\`\|ollama\|curl\|grep\|sed\|launchctl\|brew\|npm\|git " "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
CONFIGS=$(grep -ci "\.json\|\.toml\|\.plist\|config\|setting\|parameter\|env var" "$MEMORY_DIR"/2026-*.md 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')

echo -e "  ${BOLD}Decisions:${RESET}       $DECISIONS mentions"
echo -e "  ${BOLD}Lessons:${RESET}         $LESSONS mentions"
echo -e "  ${BOLD}Todos/Actions:${RESET}   $TODOS mentions"
echo -e "  ${BOLD}Commands:${RESET}        $COMMANDS mentions"
echo -e "  ${BOLD}Configs:${RESET}         $CONFIGS mentions"

# ============================================================================
# PART 5: What a compressed version would look like
# ============================================================================

banner "PART 5: Compression Potential"

# Calculate what memory SHOULD look like
# Rule of thumb: a good daily summary is 5-10 lines (decisions + lessons + blockers)
IDEAL_LINES=$((TOTAL_NOTES * 8))
IDEAL_TOKENS=$((IDEAL_LINES * 15))
COMPRESSION_RATIO=$(echo "$EST_TOKENS $IDEAL_TOKENS" | awk '{if($2>0) printf "%.1f", $1/$2; else print "0"}')

echo -e "  ${BOLD}Current total tokens:${RESET}     $EST_TOKENS"
echo -e "  ${BOLD}Ideal total tokens:${RESET}      $IDEAL_TOKENS (8 lines × $TOTAL_NOTES notes × ~15 tok/line)"
echo -e "  ${BOLD}Compression ratio:${RESET}       ${YELLOW}${COMPRESSION_RATIO}x${RESET}"
echo -e "  ${BOLD}Tokens recoverable:${RESET}      $((EST_TOKENS - IDEAL_TOKENS))"
echo ""
echo -e "  ${DIM}If Nestor loads all notes into context:${RESET}"
echo -e "  ${DIM}  Current:    ~${EST_TOKENS} tokens consumed before your question${RESET}"
echo -e "  ${DIM}  Compressed: ~${IDEAL_TOKENS} tokens consumed${RESET}"
echo -e "  ${DIM}  Savings:    $((EST_TOKENS - IDEAL_TOKENS)) tokens = $(echo "$EST_TOKENS $IDEAL_TOKENS" | awk '{printf "%.0f", (1-$2/$1)*100}')% less context burned${RESET}"

# ============================================================================
# PART 6: Inference cost — with memory vs without
# ============================================================================

banner "PART 6: Memory Context Cost (Live Benchmark)"

OLLAMA_API="http://localhost:11434"
MODEL="gemma4-think:26b"
TEST_PROMPT="What decisions did I make last week?"

# Check Ollama is running
if ! curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
    echo -e "  ${RED}✗ Ollama not running — skipping live benchmark${RESET}"
    BENCH_NO_MEM_PROMPT_TOKENS=0
    BENCH_NO_MEM_PROMPT_TPS=0
    BENCH_NO_MEM_TOTAL=0
    BENCH_MEM_PROMPT_TOKENS=0
    BENCH_MEM_PROMPT_TPS=0
    BENCH_MEM_TOTAL=0
else
    if ! command -v jq &> /dev/null; then
        echo -e "  ${RED}✗ jq not found — skipping live benchmark${RESET}"
        BENCH_NO_MEM_PROMPT_TOKENS=0
        BENCH_NO_MEM_PROMPT_TPS=0
        BENCH_NO_MEM_TOTAL=0
        BENCH_MEM_PROMPT_TOKENS=0
        BENCH_MEM_PROMPT_TPS=0
        BENCH_MEM_TOTAL=0
    else
        # Concatenate all memory notes into a single system prompt
        MEMORY_CONTENT=$(cat "$MEMORY_DIR"/2026-*.md 2>/dev/null)

        info "Test A: No memory context (baseline)"
        RESULT_A=$(curl -sf "$OLLAMA_API/api/chat" \
            -d "$(jq -n \
                --arg model "$MODEL" \
                --arg content "$TEST_PROMPT" \
                '{
                    model: $model,
                    messages: [{role: "user", content: $content}],
                    stream: false,
                    think: false,
                    options: {num_ctx: 130000}
                }'
            )" 2>/dev/null)

        BENCH_NO_MEM_PROMPT_TOKENS=$(echo "$RESULT_A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo 0)
        BENCH_NO_MEM_PROMPT_TPS=$(echo "$RESULT_A" | python3 -c "import sys,json; d=json.load(sys.stdin); pt=d.get('prompt_eval_count',0); pd=d.get('prompt_eval_duration',1); print(f'{pt/pd*1e9:.1f}')" 2>/dev/null || echo 0)
        BENCH_NO_MEM_TOTAL=$(echo "$RESULT_A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"total_duration\",0)/1e9:.2f}')" 2>/dev/null || echo 0)

        echo -e "    Prompt tokens: ${BENCH_NO_MEM_PROMPT_TOKENS}"
        echo -e "    Prompt eval:   ${BENCH_NO_MEM_PROMPT_TPS} t/s"
        echo -e "    Total time:    ${BENCH_NO_MEM_TOTAL}s"

        sleep 2

        info "Test B: Full memory loaded as system prompt"
        RESULT_B=$(curl -sf "$OLLAMA_API/api/chat" \
            -d "$(jq -n \
                --arg model "$MODEL" \
                --arg memory "$MEMORY_CONTENT" \
                --arg content "$TEST_PROMPT" \
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
            )" 2>/dev/null)

        BENCH_MEM_PROMPT_TOKENS=$(echo "$RESULT_B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt_eval_count',0))" 2>/dev/null || echo 0)
        BENCH_MEM_PROMPT_TPS=$(echo "$RESULT_B" | python3 -c "import sys,json; d=json.load(sys.stdin); pt=d.get('prompt_eval_count',0); pd=d.get('prompt_eval_duration',1); print(f'{pt/pd*1e9:.1f}')" 2>/dev/null || echo 0)
        BENCH_MEM_TOTAL=$(echo "$RESULT_B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d.get(\"total_duration\",0)/1e9:.2f}')" 2>/dev/null || echo 0)

        echo -e "    Prompt tokens: ${BENCH_MEM_PROMPT_TOKENS}"
        echo -e "    Prompt eval:   ${BENCH_MEM_PROMPT_TPS} t/s"
        echo -e "    Total time:    ${BENCH_MEM_TOTAL}s"

        echo ""
        DELTA_TOKENS=$((BENCH_MEM_PROMPT_TOKENS - BENCH_NO_MEM_PROMPT_TOKENS))
        echo -e "  ${BOLD}Memory overhead:${RESET}"
        echo -e "    Extra tokens:  ${YELLOW}${DELTA_TOKENS}${RESET}"
        echo -e "    Extra time:    ${YELLOW}$(echo "$BENCH_MEM_TOTAL $BENCH_NO_MEM_TOTAL" | awk '{printf "%.2f", $1-$2}')s${RESET}"
        echo -e "    Token ratio:   ${YELLOW}$(echo "$BENCH_MEM_PROMPT_TOKENS $BENCH_NO_MEM_PROMPT_TOKENS" | awk '{if($2>0) printf "%.1fx", $1/$2; else print "N/A"}')${RESET} more context to process"
    fi
fi

# ============================================================================
# PART 7: Write analysis report
# ============================================================================

cat > "$ANALYSIS_FILE" << EOF
# Phase 2: Memory Efficiency Analysis
**Date:** $(date +%Y-%m-%d)
**Memory dir:** $MEMORY_DIR

## Inventory
- Daily notes: $TOTAL_NOTES files
- Total lines: $TOTAL_LINES
- Total size: $(echo $TOTAL_BYTES | awk '{printf "%.1f KB", $1/1024}')
- Estimated tokens: $EST_TOKENS

## Noise Ratio
- Noise lines: $TOTAL_NOISE / $TOTAL_LINES (${NOISE_PCT}%)
- Categories: metadata ($METADATA_LINES), JSON blocks ($JSON_BLOCK_LINES), session headers ($SESSION_HEADER_LINES)

## Content Signal
- Decisions: $DECISIONS mentions
- Lessons: $LESSONS mentions
- Todos/Actions: $TODOS mentions
- Commands: $COMMANDS mentions
- Config references: $CONFIGS mentions

## Compression Potential
- Current: ~$EST_TOKENS tokens
- Ideal (8 lines/note): ~$IDEAL_TOKENS tokens
- Compression ratio: ${COMPRESSION_RATIO}x
- Recoverable: $((EST_TOKENS - IDEAL_TOKENS)) tokens

## Inference Cost (Memory Loaded vs Not)
- Without memory: ${BENCH_NO_MEM_PROMPT_TOKENS} prompt tokens, ${BENCH_NO_MEM_TOTAL}s total
- With memory: ${BENCH_MEM_PROMPT_TOKENS} prompt tokens, ${BENCH_MEM_TOTAL}s total
- Overhead: $((BENCH_MEM_PROMPT_TOKENS - BENCH_NO_MEM_PROMPT_TOKENS)) extra tokens, $(echo "$BENCH_MEM_TOTAL $BENCH_NO_MEM_TOTAL" | awk '{printf "%.2f", $1-$2}')s extra time

## Interpretation
At 41 t/s generation and ~388 t/s prompt eval (from Phase 1 at 130K):
- Current memory adds ~$(echo $EST_TOKENS | awk '{printf "%.1f", $1/388}')s to prompt evaluation per request
- Compressed memory would add ~$(echo $IDEAL_TOKENS | awk '{printf "%.1f", $1/388}')s
- Delta: ~$(echo "$EST_TOKENS $IDEAL_TOKENS" | awk '{printf "%.1f", ($1-$2)/388}')s saved per request

## Raw CSV
See: $(basename $RESULTS_FILE)
EOF

ok "Raw CSV: $RESULTS_FILE"
ok "Analysis: $ANALYSIS_FILE"

banner "SUMMARY"
echo ""
echo -e "  ${BOLD}Your memory in one line:${RESET}"
echo -e "  ${YELLOW}${TOTAL_NOTES} notes, ${NOISE_PCT}% noise, ${COMPRESSION_RATIO}x compressible${RESET}"
if [[ "$BENCH_MEM_TOTAL" != "0" ]]; then
    echo -e "  ${YELLOW}Memory adds $((BENCH_MEM_PROMPT_TOKENS - BENCH_NO_MEM_PROMPT_TOKENS)) tokens and $(echo "$BENCH_MEM_TOTAL $BENCH_NO_MEM_TOTAL" | awk '{printf "%.1f", $1-$2}')s to every request${RESET}"
fi
echo ""
echo -e "  ${DIM}Next: Run the compaction step — let Nestor summarise each note${RESET}"
echo -e "  ${DIM}into 5-8 lines of decisions/lessons/blockers, then re-benchmark.${RESET}"
echo ""
ok "Done. This data goes into blog post 2."
