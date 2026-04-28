#!/usr/bin/env bash
# lib/ollama_probe.sh
# Shared helpers for talking to a local Ollama instance on macOS.
# Source this from a test script; do not execute directly.

OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

# Print to stderr.
log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2
}

# Check that Ollama is responding. Returns 0 if up, 1 if not.
ollama_is_up() {
  curl -fsS --max-time 2 "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1
}

# Wait for Ollama to be up. Times out after $1 seconds (default 30).
ollama_wait_up() {
  local timeout="${1:-30}"
  local elapsed=0
  while ! ollama_is_up; do
    if [[ $elapsed -ge $timeout ]]; then
      log "Ollama did not come up within ${timeout}s"
      return 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 0
}

# Restart Ollama. Pulls in newly-set launchctl env vars.
#
# Three strategies, tried in order:
#   1. Custom launchd agent at OLLAMA_LAUNCHD_PLIST (e.g.
#      ~/Library/LaunchAgents/com.ollama.serve.plist) — bootout then
#      bootstrap. This is miktam02's setup.
#   2. Ollama.app — quit via osascript, re-launch via `open -a`.
#   3. `ollama serve` running standalone — kill the process; caller
#      must have a way to bring it back (we cannot restart something
#      we did not start).
#
# Returns 0 if Ollama responds within OLLAMA_RESTART_TIMEOUT seconds,
# 1 otherwise.
OLLAMA_LAUNCHD_PLIST="${OLLAMA_LAUNCHD_PLIST:-$HOME/Library/LaunchAgents/com.ollama.serve.plist}"
OLLAMA_RESTART_TIMEOUT="${OLLAMA_RESTART_TIMEOUT:-30}"

ollama_restart_app() {
  local uid
  uid="$(id -u)"

  if [[ -f "$OLLAMA_LAUNCHD_PLIST" ]]; then
    log "Restarting Ollama via launchctl ($OLLAMA_LAUNCHD_PLIST)"
    # bootout may fail if already booted out; that's fine.
    launchctl bootout "gui/${uid}" "$OLLAMA_LAUNCHD_PLIST" 2>/dev/null || true
    sleep 1
    if ! launchctl bootstrap "gui/${uid}" "$OLLAMA_LAUNCHD_PLIST"; then
      log "launchctl bootstrap failed"
      return 1
    fi
    ollama_wait_up "$OLLAMA_RESTART_TIMEOUT"
    return $?
  fi

  if [[ -d "/Applications/Ollama.app" ]]; then
    log "Restarting Ollama.app via osascript"
    osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
    pkill -f "Ollama.app/Contents" >/dev/null 2>&1 || true
    pkill -x ollama >/dev/null 2>&1 || true
    sleep 2
    open -a Ollama
    ollama_wait_up "$OLLAMA_RESTART_TIMEOUT"
    return $?
  fi

  log "Cannot restart Ollama: no launchd plist at $OLLAMA_LAUNCHD_PLIST"
  log "and /Applications/Ollama.app not present. If you run \`ollama serve\`"
  log "from a terminal, this test cannot restart it cleanly — set up a"
  log "launchd plist or move to Ollama.app, then re-run."
  return 1
}

# Get the model_info.context_length reported by /api/show.
# Args: model name. Returns numeric value or empty string.
ollama_show_context_length() {
  local model="$1"
  curl -fsS "${OLLAMA_HOST}/api/show" \
    -d "$(jq -nc --arg m "$model" '{name:$m}')" \
  | jq -r '.model_info | to_entries
           | map(select(.key | endswith(".context_length")))
           | .[0].value // empty'
}

# Get the loaded context length from /api/ps for a model.
# Args: model name. Returns numeric value or empty string.
ollama_ps_context_length() {
  local model="$1"
  curl -fsS "${OLLAMA_HOST}/api/ps" \
  | jq -r --arg m "$model" \
      '.models[] | select(.name == $m or .model == $m)
       | .context_length // empty' \
  | head -n1
}

# Get full /api/ps as JSON for evidence capture.
ollama_ps_json() {
  curl -fsS "${OLLAMA_HOST}/api/ps"
}

# Trigger a minimal generation to force a model load. Returns 0 on
# success regardless of generation content (we just want it loaded).
# Args: model name, optional num_ctx.
ollama_load_model() {
  local model="$1"
  local num_ctx="${2:-}"
  local body
  if [[ -n "$num_ctx" ]]; then
    body=$(jq -nc --arg m "$model" --argjson c "$num_ctx" \
      '{model:$m, prompt:"hi", stream:false, options:{num_ctx:$c}}')
  else
    body=$(jq -nc --arg m "$model" \
      '{model:$m, prompt:"hi", stream:false}')
  fi
  curl -fsS --max-time 120 "${OLLAMA_HOST}/api/generate" -d "$body" \
    >/dev/null
}

# Generate deterministic filler text of approximately N tokens.
#
# CHARS_PER_TOKEN defaults to 7, calibrated against gemma4 tokenizer
# on Latin filler (observed 6.87 chars/token in the H5 run; rounded up
# so we hit at-or-above target rather than below). Override via
# CHARS_PER_TOKEN env var if you measure differently for a different
# tokenizer or prompt style.
#
# Args: target token count, optional variant string. The variant is
# embedded as a unique header so consecutive prompts don't hit
# Ollama's prompt cache and short-circuit to zero-second responses
# (a problem we saw in the first H5 run).
#
# Writes filler text to stdout.
generate_filler_prompt() {
  local target_tokens="$1"
  local variant="${2:-default}"
  local chars_per_token="${CHARS_PER_TOKEN:-7}"
  local target_chars=$((target_tokens * chars_per_token))
  python3 -c "
import sys
target = $target_chars
variant = '$variant'
header = f'[seed: {variant}] '
phrase = 'lorem ipsum dolor sit amet consectetur adipiscing elit '
body_target = max(0, target - len(header))
body = (phrase * (body_target // len(phrase) + 1))[:body_target]
sys.stdout.write(header + body)
"
}

# Verify that a model is pulled locally. Returns 0 if present.
ollama_has_model() {
  local model="$1"
  curl -fsS "${OLLAMA_HOST}/api/tags" \
  | jq -e --arg m "$model" \
      '.models[] | select(.name == $m or .model == $m)' \
    >/dev/null 2>&1
}
