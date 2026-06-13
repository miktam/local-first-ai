#!/bin/bash
# ============================================================================
# nestor-bench-phase1b.sh — Thinking Token Cost Isolation
# ============================================================================
# Purpose:  Prove that runaway inference is caused by thinking token volume,
#           not context size. Phase 1 showed gen_tps is stable at ~41 t/s
#           across all context sizes. This test measures HOW MANY tokens
#           get generated for different prompt types, with think on vs off.
#
# Setup:    Mac Mini M4 Pro · 64GB Unified Memory · Ollama 0.20.2
# Model:    gemma4-think:26b (alias of gemma4:26b)
#
# Usage:    chmod +x nestor-bench-phase1b.sh && ./nestor-bench-phase1b.sh
# ============================================================================

set -euo pipefail

MODEL="gemma4-think:26b"
OLLAMA_API="http://localhost:11434"
RESULTS_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/phase1b_$TIMESTAMP.csv"
CONTEXT=130000
RUNS=3

# --- Prompt pairs: direct vs analytical ---
# Each pair tests the same topic but with different cognitive demand.

PROMPT_DIRECT_1="List the OWASP Top 10 vulnerabilities. One line each, no explanations."
PROMPT_ANALYTICAL_1="Analyse how the OWASP Top 10 vulnerabilities interact with each other. Which ones create cascading risks when combined? Provide a ranked assessment."

PROMPT_DIRECT_2="What is a KV cache in transformer inference? One paragraph."
PROMPT_ANALYTICAL_2="Compare the memory efficiency tradeoffs of FP16 vs quantized KV cache in MoE architectures on unified memory systems. Consider the interaction between expert routing overhead and cache compression."

PROMPT_DIRECT_3="Convert this JSON to a markdown table: {\"name\": \"Alice\", \"role\": \"engineer\", \"level\": \"senior\"}"
PROMPT_ANALYTICAL_3="Design a role-based access control system for a RAG pipeline where different user clearance levels should see different subsets of retrieved documents. Consider the security implications of vector similarity search crossing access boundaries."

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

check_ollama() {
    if ! curl -sf "$OLLAMA_API/api/tags" > /dev/null 2>&1; then
        fail "Ollama is not running at $OLLAMA_API"
        exit 1
    fi
}

check_model() {
    if ! curl -sf "$OLLAMA_API/api/show" -d "{\"name\":\"$MODEL\"}" > /dev/null 2>&1; then
        fail "Model $MODEL not found."
        exit 1
    fi
}

# Run inference and extract: prompt_tokens, prompt_tps, gen_tokens, gen_tps, total_sec, has_thinking
run_inference() {
    local prompt="$1"
    local think_mode="$2"  # "true" or "false"

    local response
    response=$(curl -sf "$OLLAMA_API/api/chat" \
        -d "$(jq -n \
            --arg model "$MODEL" \
            --arg content "$prompt" \
            --argjson num_ctx "$CONTEXT" \
            --argjson think "$think_mode" \
            '{
                model: $model,
                messages: [{role: "user", content: $content}],
                stream: false,
                think: $think,
                options: {num_ctx: $num_ctx}
            }'
        )" 2>/dev/null)

    if [[ -z "$response" ]]; then
        echo "0,0.0,0,0.0,0.00,0,error"
        return
    fi

    echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    pt = d.get('prompt_eval_count', 0)
    pd = d.get('prompt_eval_duration', 1)
    gt = d.get('eval_count', 0)
    gd = d.get('eval_duration', 1)
    td = d.get('total_duration', 1)

    # Count visible output length (chars in message content)
    msg = d.get('message', {})
    visible = len(msg.get('content', ''))
    thinking = len(msg.get('thinking', ''))

    print(f'{pt},{pt/pd*1e9:.1f},{gt},{gt/gd*1e9:.1f},{td/1e9:.2f},{visible},{thinking}')
except Exception as e:
    print(f'0,0.0,0,0.0,0.00,0,0')
"
}

# --- Preflight ---
banner "NESTOR BENCH — Phase 1b: Thinking Token Cost Isolation"
echo -e "  ${DIM}Model:     $MODEL${RESET}"
echo -e "  ${DIM}Context:   $CONTEXT${RESET}"
echo -e "  ${DIM}Runs:      $RUNS per prompt per mode${RESET}"
echo -e "  ${DIM}Test:      3 prompt pairs × 2 modes (think on/off) × $RUNS runs${RESET}"

check_ollama && ok "Ollama running"
check_model && ok "Model available"

if ! command -v jq &> /dev/null; then
    fail "jq required. brew install jq"
    exit 1
fi
ok "jq available"

mkdir -p "$RESULTS_DIR"
echo "test_id,prompt_type,think_mode,run,prompt_tokens,prompt_tps,gen_tokens,gen_tps,total_sec,visible_chars,thinking_chars" > "$RESULTS_FILE"
ok "Results: $RESULTS_FILE"

# Warmup
info "Warming up model..."
curl -sf "$OLLAMA_API/api/chat" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"stream\":false,\"think\":false,\"options\":{\"num_ctx\":4096}}" \
    > /dev/null 2>&1
ok "Model warm"

# --- Run tests ---

run_test() {
    local test_id="$1"
    local prompt_type="$2"
    local prompt="$3"
    local think_mode="$4"
    local think_label="$5"

    echo ""
    echo -e "  ${BOLD}Test $test_id: ${prompt_type} / think=${think_label}${RESET}"
    echo -e "  ${DIM}  Prompt: \"${prompt:0:70}...\"${RESET}"
    echo -e "  ${DIM}  ┌──────┬──────────┬──────────┬───────────┬─────────────┬──────────────┐${RESET}"
    echo -e "  ${DIM}  │ Run  │ Gen toks │ Gen t/s  │ Total sec │ Visible chr │ Thinking chr │${RESET}"
    echo -e "  ${DIM}  ├──────┼──────────┼──────────┼───────────┼─────────────┼──────────────┤${RESET}"

    for run in $(seq 1 $RUNS); do
        result=$(run_inference "$prompt" "$think_mode")

        if [[ -z "$result" || "$result" == "0,0.0,0,0.0,0.00,0,0" ]]; then
            fail "  │  $run   │ FAILED   │ FAILED   │ FAILED    │ FAILED      │ FAILED       │"
            echo "$test_id,$prompt_type,$think_label,$run,0,0.0,0,0.0,0.00,0,0" >> "$RESULTS_FILE"
            continue
        fi

        IFS=',' read -r pt ptps gt gtps total visible thinking <<< "$result"
        echo "$test_id,$prompt_type,$think_label,$run,$pt,$ptps,$gt,$gtps,$total,$visible,$thinking" >> "$RESULTS_FILE"

        printf "  ${DIM}  │${RESET}  %d   ${DIM}│${RESET} %6s   ${DIM}│${RESET} %6s   ${DIM}│${RESET} %7s   ${DIM}│${RESET} %9s   ${DIM}│${RESET} %10s   ${DIM}│${RESET}\n" \
            "$run" "$gt" "$gtps" "$total" "$visible" "$thinking"

        sleep 2
    done

    echo -e "  ${DIM}  └──────┴──────────┴──────────┴───────────┴─────────────┴──────────────┘${RESET}"
}

banner "TEST 1: OWASP (Direct vs Analytical)"
run_test "1a" "direct"     "$PROMPT_DIRECT_1"     "false" "off"
run_test "1b" "direct"     "$PROMPT_DIRECT_1"     "true"  "on"
run_test "1c" "analytical" "$PROMPT_ANALYTICAL_1" "false" "off"
run_test "1d" "analytical" "$PROMPT_ANALYTICAL_1" "true"  "on"

banner "TEST 2: KV Cache (Direct vs Analytical)"
run_test "2a" "direct"     "$PROMPT_DIRECT_2"     "false" "off"
run_test "2b" "direct"     "$PROMPT_DIRECT_2"     "true"  "on"
run_test "2c" "analytical" "$PROMPT_ANALYTICAL_2" "false" "off"
run_test "2d" "analytical" "$PROMPT_ANALYTICAL_2" "true"  "on"

banner "TEST 3: RBAC Design (Direct vs Analytical)"
run_test "3a" "direct"     "$PROMPT_DIRECT_3"     "false" "off"
run_test "3b" "direct"     "$PROMPT_DIRECT_3"     "true"  "on"
run_test "3c" "analytical" "$PROMPT_ANALYTICAL_3" "false" "off"
run_test "3d" "analytical" "$PROMPT_ANALYTICAL_3" "true"  "on"

# --- Summary ---
banner "RESULTS SUMMARY"
ok "Raw CSV: $RESULTS_FILE"
echo ""

python3 -c "
import csv
from collections import defaultdict

data = defaultdict(lambda: {'gen_tokens': [], 'gen_tps': [], 'total': [], 'visible': [], 'thinking': []})

with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['test_id'], row['prompt_type'], row['think_mode'])
        gt = int(row['gen_tokens'])
        if gt > 0:
            data[key]['gen_tokens'].append(gt)
            data[key]['gen_tps'].append(float(row['gen_tps']))
            data[key]['total'].append(float(row['total_sec']))
            data[key]['visible'].append(int(row['visible_chars']))
            data[key]['thinking'].append(int(row['thinking_chars']))

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

print('  ┌───────┬────────────┬────────┬───────────┬───────────┬─────────────┬──────────────┐')
print('  │ Test  │ Type       │ Think  │ Avg Toks  │ Avg Sec   │ Visible Chr │ Thinking Chr │')
print('  ├───────┼────────────┼────────┼───────────┼───────────┼─────────────┼──────────────┤')

for key in sorted(data.keys()):
    d = data[key]
    test_id, ptype, think = key
    print(f'  │ {test_id:>5} │ {ptype:>10} │ {think:>6} │ {avg(d[\"gen_tokens\"]):>7.0f}   │ {avg(d[\"total\"]):>7.2f}   │ {avg(d[\"visible\"]):>9.0f}   │ {avg(d[\"thinking\"]):>10.0f}   │')

print('  └───────┴────────────┴────────┴───────────┴───────────┴─────────────┴──────────────┘')
"

echo ""
echo -e "  ${DIM}What to look for:${RESET}"
echo -e "  ${DIM}  • Compare gen_tokens between think=on vs think=off for the SAME prompt${RESET}"
echo -e "  ${DIM}    The delta is pure thinking overhead${RESET}"
echo -e "  ${DIM}  • Compare gen_tokens between direct vs analytical with think=on${RESET}"
echo -e "  ${DIM}    This shows how prompt complexity amplifies thinking volume${RESET}"
echo -e "  ${DIM}  • The \"thinking_chars\" column shows the hidden reasoning output${RESET}"
echo -e "  ${DIM}    If this is 10-50x the visible chars, thinking mode is the runaway${RESET}"
echo -e "  ${DIM}  • Multiply gen_tokens by (1/41 t/s) to predict response time${RESET}"
echo -e "  ${DIM}    e.g. 5000 thinking tokens = ~122 seconds of wall time${RESET}"
echo ""

ok "Done. This data goes directly into the blog post."
ok "Next: Phase 2 — KV cache lifecycle mapping."
