#!/bin/bash
# ============================================================================
# nestor-bench-phase1.sh — Metal GPU Inference Baseline
# ============================================================================
# Purpose:  Measure how context window size affects inference speed on
#           Apple Silicon with Ollama. Proves the relationship between
#           num_ctx, KV cache allocation, and tokens/sec.
#
# Setup:    Mac Mini M4 Pro · 64GB Unified Memory · Ollama 0.20.2
# Model:    gemma4-think:26b (alias of gemma4:26b, renamed for OpenClaw reasoning detection)
#
# Usage:    chmod +x nestor-bench-phase1.sh && ./nestor-bench-phase1.sh
# ============================================================================

set -euo pipefail

MODEL="gemma4-think:26b"
OLLAMA_API="http://localhost:11434"
PROMPT="Explain what a KV cache is in one paragraph."
RESULTS_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/phase1_$TIMESTAMP.csv"
CONTEXT_SIZES=(4096 8192 16384 32768 65536 130000)
WARMUP_PROMPT="Say OK."
RUNS_PER_SIZE=3

# --- Colors for terminal output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

# --- Helper functions ---

banner() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${BOLD}  $1${RESET}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
}

info()  { echo -e "  ${DIM}→${RESET} $1"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET} $1"; }
fail()  { echo -e "  ${RED}✗${RESET} $1"; }

check_ollama() {
    if ! curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
        fail "Ollama is not running at $OLLAMA_API"
        echo "    Start it with: ollama serve"
        exit 1
    fi
}

check_model() {
    if ! curl -sf "$OLLAMA_API/api/show" -d "{\"name\":\"$MODEL\"}" > /dev/null 2>&1; then
        fail "Model $MODEL not found. Pull it with: ollama pull $MODEL"
        exit 1
    fi
}

# Extract inference metrics from Ollama's JSON response.
# Returns: prompt_tokens, prompt_tps, gen_tokens, gen_tps, total_seconds
parse_response() {
    python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pt = d.get('prompt_eval_count', 0)
    pd = d.get('prompt_eval_duration', 1)
    gt = d.get('eval_count', 0)
    gd = d.get('eval_duration', 1)
    td = d.get('total_duration', 1)
    print(f'{pt},{pt/pd*1e9:.1f},{gt},{gt/gd*1e9:.1f},{td/1e9:.2f}')
except Exception as e:
    print(f'0,0.0,0,0.0,0.00', file=sys.stderr)
    print(f'0,0.0,0,0.0,0.00')
"
}

# Capture system state before benchmarking
capture_system_state() {
    banner "SYSTEM STATE"

    info "Hardware"
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "  (unknown CPU)"
    echo -n "  RAM: "; sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0f GB\n", $1/1073741824}'

    info "Memory pressure"
    memory_pressure 2>/dev/null | grep "System-wide" | head -1 || echo "  (unavailable)"

    info "Swap usage"
    sysctl vm.swapusage 2>/dev/null | awk -F'  ' '{print "  "$2}' || echo "  (unavailable)"

    info "Ollama version"
    ollama --version 2>/dev/null || echo "  (unknown)"

    info "Model status"
    ollama ps 2>/dev/null

    info "Ollama env vars (from running process)"
    ps aux | grep "[o]llama serve" | head -1 | awk '{print "  PID:", $2, " CPU:", $3"%", " MEM:", $4"%"}'

    # Check for relevant env vars in the launchd plist
    local plist="$HOME/Library/LaunchAgents/com.ollama.serve.plist"
    if [[ -f "$plist" ]]; then
        info "Custom plist env vars:"
        grep -A1 "OLLAMA_FLASH_ATTENTION\|OLLAMA_KV_CACHE_TYPE\|OLLAMA_KEEP_ALIVE\|OLLAMA_NUM_PARALLEL" "$plist" 2>/dev/null | \
            grep -v "^--$" | sed 's/^/    /'
    fi

    echo ""
}

# Run a single inference and return metrics
run_inference() {
    local ctx_size=$1
    local prompt="$2"

    curl -sf "$OLLAMA_API/api/chat" \
        -d "$(jq -n \
            --arg model "$MODEL" \
            --arg content "$prompt" \
            --argjson num_ctx "$ctx_size" \
            '{
                model: $model,
                messages: [{role: "user", content: $content}],
                stream: false,
                options: {num_ctx: $num_ctx}
            }'
        )" 2>/dev/null | parse_response
}

# --- Main ---

banner "NESTOR BENCH — Phase 1: Metal GPU Inference Baseline"
echo -e "  ${DIM}Model:   $MODEL${RESET}"
echo -e "  ${DIM}Prompt:  \"$PROMPT\"${RESET}"
echo -e "  ${DIM}Sizes:   ${CONTEXT_SIZES[*]}${RESET}"
echo -e "  ${DIM}Runs:    $RUNS_PER_SIZE per context size${RESET}"

# Preflight checks
check_ollama
ok "Ollama running"
check_model
ok "Model $MODEL available"

# Check for jq
if ! command -v jq &> /dev/null; then
    fail "jq is required. Install with: brew install jq"
    exit 1
fi
ok "jq available"

# Capture system state
capture_system_state

# Create results directory
mkdir -p "$RESULTS_DIR"
echo "ctx_size,run,prompt_tokens,prompt_tps,gen_tokens,gen_tps,total_sec" > "$RESULTS_FILE"
ok "Results will be saved to: $RESULTS_FILE"

# Warmup — make sure model is loaded and hot
banner "WARMUP"
info "Loading model into GPU memory..."
curl -sf "$OLLAMA_API/api/chat" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$WARMUP_PROMPT\"}],\"stream\":false,\"options\":{\"num_ctx\":4096}}" \
    > /dev/null 2>&1
ok "Model warm"

# Check KV cache after warmup
info "KV cache after warmup:"
grep "kv cache" /opt/homebrew/var/log/ollama.log 2>/dev/null | tail -1 | sed 's/^/    /' || echo "    (log not found — check path)"

# Run benchmarks
banner "BENCHMARKING"

for ctx in "${CONTEXT_SIZES[@]}"; do
    echo ""
    echo -e "  ${BOLD}Context window: ${YELLOW}${ctx}${RESET} ${DIM}($(echo "$ctx" | awk '{printf "%.0fK", $1/1024}'))${RESET}"
    echo -e "  ${DIM}  ┌──────┬────────────┬────────────┬───────────┐${RESET}"
    echo -e "  ${DIM}  │ Run  │ Prompt t/s │ Gen t/s    │ Total sec │${RESET}"
    echo -e "  ${DIM}  ├──────┼────────────┼────────────┼───────────┤${RESET}"

    for run in $(seq 1 $RUNS_PER_SIZE); do
        result=$(run_inference "$ctx" "$PROMPT")

        if [[ -z "$result" || "$result" == "0,0.0,0,0.0,0.00" ]]; then
            fail "  │  $run   │ FAILED     │ FAILED     │ FAILED    │"
            echo "$ctx,$run,0,0.0,0,0.0,0.00" >> "$RESULTS_FILE"
            continue
        fi

        IFS=',' read -r pt ptps gt gtps total <<< "$result"
        echo "$ctx,$run,$pt,$ptps,$gt,$gtps,$total" >> "$RESULTS_FILE"

        printf "  ${DIM}  │${RESET}  %d   ${DIM}│${RESET} %8s   ${DIM}│${RESET} %8s   ${DIM}│${RESET} %7s   ${DIM}│${RESET}\n" \
            "$run" "${ptps}" "${gtps}" "${total}"

        # Brief pause between runs to let GPU settle
        sleep 2
    done

    echo -e "  ${DIM}  └──────┴────────────┴────────────┴───────────┘${RESET}"
done

# Capture KV cache at largest context for comparison
echo ""
info "KV cache at largest context:"
grep "kv cache" /opt/homebrew/var/log/ollama.log 2>/dev/null | tail -1 | sed 's/^/    /' || echo "    (log not found)"

# Summary
banner "RESULTS"
ok "Raw CSV saved to: $RESULTS_FILE"
echo ""

info "Averages by context size:"
echo ""
echo -e "  ${DIM}  ┌───────────┬────────────┬────────────┬───────────┐${RESET}"
echo -e "  ${DIM}  │ Context   │ Prompt t/s │ Gen t/s    │ Total sec │${RESET}"
echo -e "  ${DIM}  ├───────────┼────────────┼────────────┼───────────┤${RESET}"

python3 -c "
import csv, sys
from collections import defaultdict

data = defaultdict(lambda: {'ptps': [], 'gtps': [], 'total': []})

with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ctx = int(row['ctx_size'])
        ptps = float(row['prompt_tps'])
        gtps = float(row['gen_tps'])
        total = float(row['total_sec'])
        if ptps > 0:  # skip failed runs
            data[ctx]['ptps'].append(ptps)
            data[ctx]['gtps'].append(gtps)
            data[ctx]['total'].append(total)

for ctx in sorted(data.keys()):
    d = data[ctx]
    avg_p = sum(d['ptps']) / len(d['ptps'])
    avg_g = sum(d['gtps']) / len(d['gtps'])
    avg_t = sum(d['total']) / len(d['total'])
    label = f'{ctx//1024}K'
    print(f'    │ {label:>7}   │ {avg_p:>8.1f}   │ {avg_g:>8.1f}   │ {avg_t:>7.2f}   │')
"

echo -e "  ${DIM}  └───────────┴────────────┴────────────┴───────────┘${RESET}"

echo ""
echo -e "  ${DIM}What to look for:${RESET}"
echo -e "  ${DIM}  • Prompt t/s should decrease as context grows (more KV cache to allocate)${RESET}"
echo -e "  ${DIM}  • Gen t/s should stay roughly constant (MoE active params don't change)${RESET}"
echo -e "  ${DIM}  • If there's a cliff (sudden drop), that's your memory pressure threshold${RESET}"
echo -e "  ${DIM}  • Compare total_sec at 4K vs 130K — that ratio is your context tax${RESET}"
echo -e "  ${DIM}  • CRITICAL: eval_count (gen tokens) includes HIDDEN thinking tokens${RESET}"
echo -e "  ${DIM}    If eval_count is 500+ for a one-paragraph answer, thinking is the bottleneck${RESET}"
echo -e "  ${DIM}  • Next test: re-run with MODEL=gemma4-think:26b but add \"think\":false to isolate cost${RESET}"
echo ""

ok "Done. Run 'cat $RESULTS_FILE' to see raw data."
ok "Next: Phase 2 — KV cache lifecycle mapping."
