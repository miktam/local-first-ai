#!/usr/bin/env bash
# Exp 010 — Condition B: FA=1, fp16 KV cache (FA isolated, no KV quantization).
# Stops the LaunchDaemon, starts Ollama with FA=1 but default fp16 KV cache.
# Keep this terminal open during Phase A + Phase B for this condition.
# Run restore_daemon.sh when done.

set -e
OLLAMA_BIN="/opt/homebrew/opt/ollama/bin/ollama"
DAEMON_PLIST="/Library/LaunchDaemons/com.ollama.serve.plist"

echo ""
echo "Exp 010 — Condition B: FA=1, fp16 KV (flash attention isolated)"
echo "================================================================="
echo "  OLLAMA_FLASH_ATTENTION=1"
echo "  OLLAMA_KV_CACHE_TYPE=fp16 (default, unset)"
echo ""

echo "Stopping LaunchDaemon..."
sudo launchctl bootout system "$DAEMON_PLIST" 2>/dev/null || true
sleep 3

echo "Starting Ollama (Condition B)..."
echo ""

export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export HOME=/Users/miktam02
unset OLLAMA_KV_CACHE_TYPE

exec "$OLLAMA_BIN" serve
