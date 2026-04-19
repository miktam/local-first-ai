#!/bin/bash
# ============================================================================
# nestor-bench-phase2b-retrieval.sh — Selective Retrieval vs Load Everything
# ============================================================================
# Purpose:  Prove that loading only relevant notes (via embedding search)
#           matches or beats loading all notes, at a fraction of the cost.
#           Uses nomic-embed-text for embeddings, cosine similarity for
#           retrieval. Everything local.
#
# Usage:    chmod +x nestor-bench-phase2b-retrieval.sh && ./nestor-bench-phase2b-retrieval.sh
# ============================================================================

set -euo pipefail

PROJECT_DIR="$HOME/.openclaw/workspace/memory/projects/nestor-bench"
COMPRESSED_DIR="$PROJECT_DIR/benchmarks/compressed-notes"
RESULTS_DIR="$PROJECT_DIR/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/phase2b_retrieval_$TIMESTAMP.csv"
EMBEDDINGS_CACHE="$RESULTS_DIR/note_embeddings.json"
OLLAMA_API="http://localhost:11434"
MODEL="gemma4-think:26b"
EMBED_MODEL="nomic-embed-text"
TOP_K=3
RUNS=3

TEST_PROMPTS=(
    "What decisions did I make about Ollama configuration?"
    "What are my current blockers?"
    "What did I learn about thinking mode and performance?"
    "How did I fix the OpenClaw gateway crashes?"
    "What context window settings work best on Apple Silicon?"
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
banner "PHASE 2b: Selective Retrieval Benchmark"
echo -e "  ${DIM}Model:       $MODEL (think=off)${RESET}"
echo -e "  ${DIM}Embeddings:  $EMBED_MODEL${RESET}"
echo -e "  ${DIM}Top-K:       $TOP_K notes per query${RESET}"
echo -e "  ${DIM}Prompts:     ${#TEST_PROMPTS[@]}${RESET}"
echo -e "  ${DIM}Runs:        $RUNS per mode${RESET}"

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

# --- Step 1: Embed all compressed notes ---
banner "STEP 1: Embedding Compressed Notes"

NOTES=("$COMPRESSED_DIR"/2026-*.md)
TOTAL_NOTES=${#NOTES[@]}
info "Embedding $TOTAL_NOTES notes with $EMBED_MODEL..."

python3 << PYEMBED
import json, subprocess, os, sys

notes_dir = "$COMPRESSED_DIR"
cache_file = "$EMBEDDINGS_CACHE"
api = "$OLLAMA_API"
model = "$EMBED_MODEL"

# Collect all notes
notes = []
for f in sorted(os.listdir(notes_dir)):
    if f.startswith("2026-") and f.endswith(".md"):
        path = os.path.join(notes_dir, f)
        with open(path) as fh:
            content = fh.read().strip()
        notes.append({"date": f.replace(".md",""), "content": content, "path": path})

print(f"  Found {len(notes)} notes", flush=True)

# Embed each note
import urllib.request
for i, note in enumerate(notes):
    req = urllib.request.Request(
        f"{api}/api/embeddings",
        data=json.dumps({"model": model, "prompt": note["content"]}).encode(),
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    note["embedding"] = data["embedding"]
    print(f"  Embedded {i+1}/{len(notes)}: {note['date']}", flush=True)

# Save cache
with open(cache_file, "w") as f:
    json.dump(notes, f)

print(f"  Saved embeddings to {cache_file}", flush=True)
PYEMBED

ok "All notes embedded and cached"

# --- Step 2: Build retrieval function ---
# For each prompt: embed the prompt, find top-K similar notes, return their content

retrieve_notes() {
    local prompt="$1"
    local top_k="$2"

    python3 << PYRETRIEVE
import json, urllib.request, math

api = "$OLLAMA_API"
model = "$EMBED_MODEL"
cache_file = "$EMBEDDINGS_CACHE"
prompt = """$prompt"""
top_k = $top_k

# Embed the query
req = urllib.request.Request(
    f"{api}/api/embeddings",
    data=json.dumps({"model": model, "prompt": prompt}).encode(),
    headers={"Content-Type": "application/json"}
)
resp = urllib.request.urlopen(req)
query_emb = json.loads(resp.read())["embedding"]

# Load cached note embeddings
with open(cache_file) as f:
    notes = json.load(f)

# Cosine similarity
def cosine_sim(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    mag_a = math.sqrt(sum(x*x for x in a))
    mag_b = math.sqrt(sum(x*x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0
    return dot / (mag_a * mag_b)

# Rank
scored = [(cosine_sim(query_emb, n["embedding"]), n) for n in notes]
scored.sort(key=lambda x: -x[0])
top = scored[:top_k]

# Output: retrieved notes content + dates
result = {"dates": [], "content": ""}
parts = []
for score, note in top:
    result["dates"].append(f"{note['date']} (sim={score:.3f})")
    parts.append(note["content"])

result["content"] = "\n\n---\n\n".join(parts)
print(json.dumps(result))
PYRETRIEVE
}

# --- Step 3: Run comparison ---

ALL_MEMORY=$(cat "$COMPRESSED_DIR"/2026-*.md 2>/dev/null)
ALL_TOKENS_EST=$(echo "$ALL_MEMORY" | wc -w | awk '{printf "%.0f", $1/0.75}')

echo "test_id,prompt_short,memory_type,run,prompt_tokens,gen_tokens,gen_tps,total_sec,notes_loaded" > "$RESULTS_FILE"

run_inference() {
    local prompt="$1"
    local memory_type="$2"
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
                        {role: "system", content: ("You are Nestor. Here are your relevant memory notes:\n\n" + $memory)},
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
gt = d.get('eval_count', 0)
gd = d.get('eval_duration', 1)
td = d.get('total_duration', 1)
print(f'{pt},{gt},{gt/gd*1e9:.1f},{td/1e9:.2f}')
"
}

TEST_NUM=0
for prompt in "${TEST_PROMPTS[@]}"; do
    TEST_NUM=$((TEST_NUM + 1))
    PROMPT_SHORT="${prompt:0:45}"

    banner "PROMPT $TEST_NUM: \"$PROMPT_SHORT...\""

    # Retrieve relevant notes for this prompt
    RETRIEVAL=$(retrieve_notes "$prompt" "$TOP_K")
    RETRIEVED_CONTENT=$(echo "$RETRIEVAL" | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])")
    RETRIEVED_DATES=$(echo "$RETRIEVAL" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['dates']))")
    RETRIEVED_WORDS=$(echo "$RETRIEVED_CONTENT" | wc -w | tr -d ' ')

    info "Retrieved: $RETRIEVED_DATES"
    info "Retrieved size: $RETRIEVED_WORDS words"

    for memory_type in "none" "all_compressed" "top_${TOP_K}"; do
        case $memory_type in
            none)             LABEL="No memory" ; MEM="" ; NOTES_N=0 ;;
            all_compressed)   LABEL="All compressed ($TOTAL_NOTES notes)" ; MEM="$ALL_MEMORY" ; NOTES_N=$TOTAL_NOTES ;;
            top_*)            LABEL="Top-$TOP_K retrieved" ; MEM="$RETRIEVED_CONTENT" ; NOTES_N=$TOP_K ;;
        esac

        echo ""
        echo -e "  ${BOLD}$LABEL${RESET}"
        echo -e "  ${DIM}  ┌──────┬────────────┬──────────┬───────────┐${RESET}"
        echo -e "  ${DIM}  │ Run  │ Prompt tok │ Gen t/s  │ Total sec │${RESET}"
        echo -e "  ${DIM}  ├──────┼────────────┼──────────┼───────────┤${RESET}"

        for run in $(seq 1 $RUNS); do
            RESPONSE=$(run_inference "$prompt" "$memory_type" "$MEM")

            if [[ -z "$RESPONSE" ]]; then
                fail "  Failed"
                echo "$TEST_NUM,$PROMPT_SHORT,$memory_type,$run,0,0,0,0,$NOTES_N" >> "$RESULTS_FILE"
                continue
            fi

            PARSED=$(echo "$RESPONSE" | parse_result)
            IFS=',' read -r pt gt gtps total <<< "$PARSED"

            echo "$TEST_NUM,$PROMPT_SHORT,$memory_type,$run,$pt,$gt,$gtps,$total,$NOTES_N" >> "$RESULTS_FILE"

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
import csv, os
from collections import defaultdict

results_dir = '$RESULTS_DIR'
results_file = '$RESULTS_FILE'

data = defaultdict(lambda: {'prompt_tokens': [], 'total': [], 'gen_tps': []})

with open(results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = row['memory_type']
        pt = int(row['prompt_tokens'])
        total = float(row['total_sec'])
        gtps = float(row['gen_tps'])
        if pt > 0:
            data[key]['prompt_tokens'].append(pt)
            data[key]['total'].append(total)
            data[key]['gen_tps'].append(gtps)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0

print('  ┌──────────────────┬──────────────┬────────────┬──────────┐')
print('  │ Memory Type      │ Avg Pmt Toks │ Avg Time   │ Gen t/s  │')
print('  ├──────────────────┼──────────────┼────────────┼──────────┤')

for mtype in ['none', 'all_compressed', 'top_3']:
    d = data[mtype]
    at = avg(d['prompt_tokens'])
    tt = avg(d['total'])
    gt = avg(d['gen_tps'])
    label = {'none': 'No memory', 'all_compressed': 'All compressed', 'top_3': 'Top-3 retrieved'}.get(mtype, mtype)
    print(f'  │ {label:>16} │ {at:>10.0f}   │ {tt:>8.2f}s  │ {gt:>6.1f}   │')

print('  └──────────────────┴──────────────┴────────────┴──────────┘')

all_time = avg(data['all_compressed']['total'])
top3_time = avg(data['top_3']['total'])
none_time = avg(data['none']['total'])
all_tok = avg(data['all_compressed']['prompt_tokens'])
top3_tok = avg(data['top_3']['prompt_tokens'])

if all_time > 0 and top3_time > 0:
    print(f'')
    print(f'  Token reduction:  {all_tok:.0f} → {top3_tok:.0f} ({(1-top3_tok/all_tok)*100:.0f}% less context)')
    print(f'  Time reduction:   {all_time:.1f}s → {top3_time:.1f}s ({(1-top3_time/all_time)*100:.0f}% faster)')
    print(f'  vs no memory:     {top3_time:.1f}s vs {none_time:.1f}s (retrieval overhead: {top3_time-none_time:.1f}s)')
    print(f'')
    print(f'  The \"2KB that matters\" costs {top3_time-none_time:.1f}s.')
    print(f'  Loading everything costs {all_time-none_time:.1f}s.')
    print(f'  {(all_time-none_time)-(top3_time-none_time):.1f}s wasted on irrelevant memory per request.')
"

echo ""
ok "Done. This completes the memory benchmark trilogy."
ok "Blog post 2 data: Phase 2 (baseline) → compression → retrieval."
