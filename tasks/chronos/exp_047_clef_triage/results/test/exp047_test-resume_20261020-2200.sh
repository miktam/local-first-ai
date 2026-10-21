for m in gemma4:26b gemma4:e4b hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M; do ollama stop "$m" 2>/dev/null || true; done
ollama ps
$HOME/models/llama.cpp-b11711/build/bin/llama-server -m $HOME/models/exp041/clef-27b/Cloudflare_clef-Q8_0.gguf --host 127.0.0.1 --port 18447 -c 16384 -b 8192 -ub 8192 > runs/exp047_clef_server_test_resume.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:18447/health > /dev/null; do kill -0 $SRV || { echo "clef server exited"; exit 2; }; sleep 2; done
( while kill -0 $SRV 2>/dev/null; do ollama ps | tail -n +2 | grep -q . && echo "WARNING $(date +%H:%M) Ollama reloaded: $(ollama ps | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"; sleep 60; done ) &
.venv/bin/python3 scripts/clef47.py run test --arm clef
.venv/bin/python3 scripts/clef47.py score test > runs/exp047_test_score.json
