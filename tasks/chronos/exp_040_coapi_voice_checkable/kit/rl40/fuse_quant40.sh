#!/bin/bash
# exp_040: the selected R3 checkpoint → fused model → GGUF → Q4_K_M with step 0's importance matrix (same recipe as B0-i)
# and Q8_0 (same recipe as B0-q8) → registered on the Ollama 0.33.0 side server as coapi-voice:r3 / coapi-voice:r3-q8.
# exp_039's path (HYPOTHESIS "Found while registering R1"): mlx-lm 0.32 fuse, llama.cpp b11577 convert_hf_to_gguf f16,
# llama-quantize. Production Ollama on :11434 is never touched. On the mini, after select_ckpt40.py:
#
#   bash coapi_voice/rl40/fuse_quant40.sh RUN_DIR STEP      # e.g. runs_rl/r3 50
set -euo pipefail
[ $# -eq 2 ] || { sed -n 2,7p "$0"; exit 1; }
CS="$HOME/REPOS/casasol"; CV="$CS/coapi_voice"
RUN="$(cd "$CV" && cd "$1" && pwd)"; STEP="$2"
VPY="$HOME/models/exp039-venv/bin/python"                 # mlx-lm 0.32, as exp_039's fuse
CPY="$CS/.venv/bin/python3"                                # converter interpreter, as exp_039
L="$HOME/models/llama.cpp-b11577"
SNAP="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
B="$CV/models/qwen3-4b-b11577"; IMAT="$B/imatrix_guide_en_pl.gguf"; M="$CV/models/r3"
OLD="$HOME/models/ollama-0.33.0/bin/ollama"; PORT=18435; HOST="http://127.0.0.1:$PORT"

ADP=$(printf '%s/%07d_adapters.safetensors' "$RUN" "$STEP")
[ -f "$ADP" ] || ADP="$RUN/adapters.safetensors"           # the final step is saved without a number
[ -f "$ADP" ] || { echo "no adapter for step $STEP in $RUN"; exit 2; }
[ -f "$IMAT" ] && [ "$(shasum -a 256 "$B/qwen3-4b-Q4_K_M-imat.gguf" | cut -c1-8)" = "38623d2a" ] \
  || { echo "REFUSED: step 0's imatrix or its B0-i GGUF is missing or changed"; exit 2; }
mkdir -p "$M"
TMP=$(mktemp -d); cp "$RUN/adapter_config.json" "$TMP/"; ln -s "$ADP" "$TMP/adapters.safetensors"

echo "== fuse step $STEP ($(basename "$ADP"))"
"$VPY" -m mlx_lm fuse --model "$SNAP" --adapter-path "$TMP" --save-path "$M/fused"
echo "== convert f16"
"$CPY" "$L/convert_hf_to_gguf.py" "$M/fused" --outtype f16 --outfile "$M/r3-f16.gguf"
echo "== quantise"
"$L/build/bin/llama-quantize" --imatrix "$IMAT" "$M/r3-f16.gguf" "$M/r3-Q4_K_M-imat.gguf" Q4_K_M
"$L/build/bin/llama-quantize" "$M/r3-f16.gguf" "$M/r3-Q8_0.gguf" Q8_0
sed "s#^FROM .*#FROM $M/r3-Q4_K_M-imat.gguf#" "$B/Modelfile.Q4_K_M-imat" > "$M/Modelfile.Q4_K_M-imat"
sed "s#^FROM .*#FROM $M/r3-Q8_0.gguf#" "$B/Modelfile.Q8_0" > "$M/Modelfile.Q8_0"

echo "== register on the side server"
if curl -s --max-time 2 -o /dev/null "$HOST/"; then echo "REFUSED: something already answers on :$PORT"; exit 2; fi
LOG="$M/register_$(date -u +%Y%m%dT%H%M%SZ).log"
OLLAMA_HOST=127.0.0.1:$PORT OLLAMA_NOPRUNE=1 OLLAMA_FLASH_ATTENTION=0 OLLAMA_CONTEXT_LENGTH=8192 \
OLLAMA_NUM_PARALLEL=2 OLLAMA_KEEP_ALIVE=10m OLLAMA_MODELS="$HOME/.ollama/models" "$OLD" serve >"$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true; rm -rf "$TMP"' EXIT
for _ in $(seq 1 60); do curl -sf "$HOST/api/version" >/dev/null && break; sleep 1; done
OLLAMA_HOST=127.0.0.1:$PORT "$OLD" create coapi-voice:r3 -f "$M/Modelfile.Q4_K_M-imat"
OLLAMA_HOST=127.0.0.1:$PORT "$OLD" create coapi-voice:r3-q8 -f "$M/Modelfile.Q8_0"

"$CPY" - "$HOST" "$M" "$ADP" "$STEP" <<'EOF'
import hashlib, json, sys, urllib.request
from pathlib import Path
host, m, adp, step = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
tags = {t["name"]: t["digest"] for t in json.load(urllib.request.urlopen(f"{host}/api/tags"))["models"]}
man = {"step": int(step), "adapter": str(adp), "adapter_sha256": h(adp),
       "gguf_sha256": {p.name: h(p) for p in sorted(m.glob("r3-*.gguf"))},
       "digest": {k: tags[k][:12] for k in ("coapi-voice:r3", "coapi-voice:r3-q8")},
       "recipe": "mlx-lm 0.32 fuse (exp039-venv) → llama.cpp b11577 convert_hf_to_gguf f16 → llama-quantize "
                 "Q4_K_M --imatrix imatrix_guide_en_pl.gguf (step 0) / Q8_0"}
(m / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(man["digest"]))
EOF
echo "== done; models/r3/MANIFEST.json pins the digests run_arm40.py checks"
