#!/usr/bin/env bash
# H2: is Ollama's API reachable and fully functional with zero credentials?
# Run with no argument for the pre-fix leg (expects success = confirmed
# exposure). Run with `post` as $1 after the auth layer is in place (expects
# the unauthenticated request to fail and an authenticated one to succeed).
set -uo pipefail

MODE="${1:-pre}"
TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/h2_ollama_exposure_${MODE}_${TS}.json"
mkdir -p "$RESULTS_DIR"

# Local, private config (real tailnet IP, never committed) — auto-loaded as a
# fallback so PROXY_URL doesn't need retyping every run. See
# ~/.config/ollama-proxy/config.env. Safe to be absent; nothing here is
# required for the pre-fix leg.
LOCAL_CONFIG="$HOME/.config/ollama-proxy/config.env"
if [ -z "${PROXY_URL:-}" ] && [ -f "$LOCAL_CONFIG" ]; then
  # shellcheck disable=SC1090
  source "$LOCAL_CONFIG"
fi

# Whatever ends up in $1 gets its Tailscale CGNAT address (100.64.0.0/10)
# stripped before it's ever written to a results file — those files are
# committed to a public repo, the real address never should be.
redact_tailnet_ip() {
  printf '%s' "$1" | sed -E 's/100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}/<tailnet-ip>/g'
}

PAYLOAD='{"model":"gemma4:26b","prompt":"reply with exactly one word: pong","stream":false}'

if [ "$MODE" = "pre" ]; then
  # Pre-fix: Ollama itself is the exposed target, reachable directly with no auth.
  TARGET="http://127.0.0.1:11434/api/generate"
  unauth_response=$(curl -s -m 20 -o /tmp/nestor-h2-unauth-$$.json -w "%{http_code}" \
    -X POST "$TARGET" -H "Content-Type: application/json" -d "$PAYLOAD" 2>&1)
  unauth_body=$(cat /tmp/nestor-h2-unauth-$$.json 2>/dev/null)
  rm -f /tmp/nestor-h2-unauth-$$.json

  if [ "$unauth_response" = "200" ] && printf '%s' "$unauth_body" | grep -qi "response"; then
    verdict="confirmed"
    code=0
    note="Unauthenticated POST to $TARGET returned HTTP 200 with a real model response — no credential of any kind was presented or required."
  else
    verdict="refuted"
    code=1
    note="Unauthenticated request did not succeed as expected (HTTP $unauth_response) — something already gates this endpoint."
  fi
else
  # Post-fix: the meaningful "unauthenticated" test is against the PROXY
  # (Ollama itself is correctly still reachable from localhost — that's by
  # design, Caddy forwards to it locally). PROXY_URL must be set explicitly
  # to the proxy's generate endpoint, e.g. http://<tailnet-ip>:11435/api/generate
  # — no default baked in here on purpose (avoid committing a real address).
  if [ -z "${PROXY_URL:-}" ]; then
    echo "Set PROXY_URL to the proxy's generate endpoint before running the post-fix leg." >&2
    exit 3
  fi
  TARGET="$PROXY_URL"
  unauth_response=$(curl -s -m 20 -o /dev/null -w "%{http_code}" \
    -X POST "$TARGET" -H "Content-Type: application/json" -d "$PAYLOAD" 2>&1)

  auth_header="${NESTOR_OLLAMA_PROXY_TOKEN:-}"
  auth_response="n/a — set NESTOR_OLLAMA_PROXY_TOKEN to run the authenticated leg"
  if [ -n "$auth_header" ]; then
    auth_response=$(curl -s -m 20 -o /dev/null -w "%{http_code}" \
      -X POST "$TARGET" -H "Content-Type: application/json" \
      -H "Authorization: Bearer $auth_header" -d "$PAYLOAD" 2>&1)
  fi
  if [ "$unauth_response" != "200" ] && [ "$auth_response" = "200" ]; then
    verdict="confirmed"
    code=0
    note="Unauthenticated request to the proxy now rejected (HTTP $unauth_response); authenticated request via the proxy succeeded (HTTP $auth_response)."
  else
    verdict="inconclusive"
    code=2
    note="Post-fix check incomplete — unauth HTTP $unauth_response, auth HTTP $auth_response. Re-run with NESTOR_OLLAMA_PROXY_TOKEN set once the proxy is live."
  fi
fi

TARGET_SAFE="$(redact_tailnet_ip "$TARGET")"
NOTE_SAFE="$(redact_tailnet_ip "$note")"

cat > "$OUT" <<EOF
{
  "hypothesis": "H2",
  "mode": "$MODE",
  "timestamp": "$TS",
  "target": "$TARGET_SAFE",
  "unauthenticated_http_status": "$unauth_response",
  "authenticated_http_status": "${auth_response:-n/a}",
  "verdict": "$verdict",
  "note": "$NOTE_SAFE"
}
EOF

echo "Result written to $OUT"
cat "$OUT"
exit "$code"
