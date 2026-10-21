$HOME/models/llama.cpp-b11711/build/bin/llama-server -m $HOME/models/exp041/clef-flash/Cloudflare_clef-flash-Q8_0.gguf --host 127.0.0.1 --port 18448 -c 16384 -b 8192 -ub 8192 > runs/exp048_flash_server_test.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:18448/health > /dev/null; do kill -0 $SRV || { echo "flash server exited"; exit 2; }; sleep 2; done
.venv/bin/python3 scripts/flash48.py run test
.venv/bin/python3 scripts/flash48.py score test > runs/exp048_test_score.json
