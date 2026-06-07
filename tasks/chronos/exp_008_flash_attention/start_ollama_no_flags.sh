#!/usr/bin/env bash
# Exp 008 — Start Ollama WITHOUT flash attention / KV quantization (FA=0 baseline run).
#
# Stops the LaunchDaemon, starts Ollama in the foreground with no optimization flags.
# Keep this terminal open during Phase A + Phase B.
# When benchmarks are done, run restore_ollama_daemon.sh to bring the daemon back.
#
# Usage:
#   chmod +x start_ollama_no_flags.sh
#   ./start_ollama_no_flags.sh

set -e

OLLAMA_BIN="/opt/homebrew/opt/ollama/bin/ollama"
DAEMON_PLIST="/Library/LaunchDaemons/com.ollama.serve.plist"

echo ""
echo "Exp 008 — Ollama FA=0 baseline (no optimization flags)"
echo "======================================================="
echo "  OLLAMA_FLASH_ATTENTION=0"
echo "  OLLAMA_KV_CACHE_TYPE=fp16 (default, unset)"
echo ""

echo "Stopping LaunchDaemon..."
sudo launchctl bootout system "$DAEMON_PLIST" 2>/dev/null || true
sleep 3

echo "Starting Ollama without optimization flags..."
echo "(keep this terminal open during benchmarks)"
echo ""

export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_KEEP_ALIVE=-1
export HOME=/Users/miktam02
unset OLLAMA_KV_CACHE_TYPE

exec "$OLLAMA_BIN" serve
