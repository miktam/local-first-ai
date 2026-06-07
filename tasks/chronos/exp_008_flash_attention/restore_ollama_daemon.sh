#!/usr/bin/env bash
# Exp 008 — Restore the LaunchDaemon after the FA=0 baseline run.
#
# Kills the manual Ollama process and re-bootstraps the daemon
# (which runs with FA=1 + q8_0 as configured in the plist).
#
# Usage:
#   chmod +x restore_ollama_daemon.sh
#   ./restore_ollama_daemon.sh

set -e

DAEMON_PLIST="/Library/LaunchDaemons/com.ollama.serve.plist"

echo ""
echo "Restoring Ollama LaunchDaemon (FA=1 + q8_0)..."
echo ""

echo "Stopping manual Ollama process..."
pkill -x ollama 2>/dev/null || true
sleep 3

echo "Bootstrapping daemon..."
sudo launchctl bootstrap system "$DAEMON_PLIST"

sleep 2
echo ""
echo "Verifying..."
curl -s http://localhost:11434/api/version && echo "" && echo "Daemon restored."
