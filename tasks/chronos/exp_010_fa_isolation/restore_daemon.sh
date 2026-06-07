#!/usr/bin/env bash
# Exp 010 — Restore production LaunchDaemon after any condition run.
# Kills the manual Ollama, re-bootstraps the daemon (FA=0, fp16).

set -e
DAEMON_PLIST="/Library/LaunchDaemons/com.ollama.serve.plist"

echo "Killing manual Ollama..."
pkill -x ollama 2>/dev/null || true
sleep 3

echo "Bootstrapping LaunchDaemon (FA=0, fp16)..."
sudo launchctl bootstrap system "$DAEMON_PLIST"
sleep 2

echo ""
curl -s http://localhost:11434/api/version && echo "" && echo "Daemon restored."
