#!/usr/bin/env bash
# Exp 008 — Start Ollama with flash attention + q8_0 KV cache.
#
# Run this in a dedicated terminal before running bench_phase_a.py or bench_phase_b.py.
# Keep the terminal open — Ollama runs in the foreground so you can see its logs.
#
# If Ollama is running as a brew service (launchd), stop it first:
#   brew services stop ollama
#
# Usage:
#   chmod +x start_ollama_flags.sh
#   ./start_ollama_flags.sh

set -e

OLLAMA_BIN="/opt/homebrew/opt/ollama/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    # Fallback: try PATH
    OLLAMA_BIN="ollama"
fi

echo ""
echo "Exp 008 — Ollama with FA + q8_0 KV cache"
echo "========================================="
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_KV_CACHE_TYPE=q8_0"
echo ""
echo "Stopping any brew-managed Ollama service first..."
brew services stop ollama 2>/dev/null || true

# Give it a moment to release the port
sleep 3

echo "Starting Ollama with flags..."
echo "(keep this terminal open during benchmarks)"
echo ""

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

exec "$OLLAMA_BIN" serve
