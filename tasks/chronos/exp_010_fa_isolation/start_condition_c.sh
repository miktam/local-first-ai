#!/usr/bin/env bash
# Exp 010 — Condition C: FA=0, q8_0 KV cache (KV quantization isolated, no FA).
# Stops the LaunchDaemon, starts Ollama with q8_0 KV but FA disabled.
# Keep this terminal open during Phase A + Phase B for this condition.
# Run restore_daemon.sh when done.

set -e
OLLAMA_BIN="/opt/homebrew/opt/ollama/bin/ollama"
DAEMON_PLIST="/Library/LaunchDaemons/com.ollama.serve.plist"

echo ""
echo "Exp 010 — Condition C: FA=0, q8_0 KV (KV quantization isolated)"
echo "================================================================="
echo "  OLLAMA_FLASH_ATTENTION=0"
echo "  OLLAMA_KV_CACHE_TYPE=q8_0"
echo ""

echo "Stopping LaunchDaemon..."
sudo launchctl bootout system "$DAEMON_PLIST" 2>/dev/null || true
sleep 3

echo "Starting Ollama (Condition C)..."
echo ""

export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export HOME=/Users/miktam02

exec "$OLLAMA_BIN" serve
