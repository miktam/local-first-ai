#!/usr/bin/env bash
# H4: verify the two agentic-risk guardrails (block_secret_reveal.py, dcg)
# are actually registered and actually fire — not just present on disk.
# Re-run this any time to check the hooks haven't been silently dropped
# (dcg's own installer logged exactly that happening once mid-session on
# 2026-07-12: "DCG hook was missing from settings.json — re-registered
# automatically. This usually means Claude Code overwrote settings.json
# mid-session."). No privilege escalation, no secret values ever recorded.
set -uo pipefail

TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/h4_agentic_guardrails_${TS}.json"
mkdir -p "$RESULTS_DIR"
SETTINGS="$HOME/.claude/settings.json"
HOOK_SCRIPT="$HOME/.claude/hooks/block_secret_reveal.py"
DCG_BIN="$HOME/.local/bin/dcg"
DECOY="$(mktemp -t dcg_h4_decoy).env"
echo "DECOY_SECRET=not_real" > "$DECOY"

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'; }

# --- registration checks (jq -e, exit 0 = present) ---
block_secret_registered="false"
if jq -e '.hooks.PreToolUse[]?.hooks[]? | select(.command | test("block_secret_reveal"))' "$SETTINGS" >/dev/null 2>&1; then
  block_secret_registered="true"
fi

dcg_registered="false"
if jq -e '.hooks.PreToolUse[]?.hooks[]? | select(.command | test("dcg"))' "$SETTINGS" >/dev/null 2>&1; then
  dcg_registered="true"
fi

# --- block_secret_reveal.py: functional test via synthetic stdin, no real tool call ---
# The hook prints nothing at all on allow (by design) — empty stdout means allow,
# not "no decision"; jq's // default doesn't fire on truly empty input, so check first.
parse_hook_decision() {
  local raw="$1"
  if [ -z "$raw" ]; then
    echo "allow"
  else
    echo "$raw" | jq -r '.hookSpecificOutput.permissionDecision // "allow"' 2>/dev/null
  fi
}

should_block_raw=$(echo "{\"tool_name\":\"Bash\",\"tool_input\":{\"command\":\"cat ${DECOY}\"}}" | python3 "$HOOK_SCRIPT" 2>/dev/null)
should_block_decision=$(parse_hook_decision "$should_block_raw")
should_allow_raw=$(echo '{"tool_name":"Bash","tool_input":{"command":"ls -la /tmp"}}' | python3 "$HOOK_SCRIPT" 2>/dev/null)
should_allow_decision=$(parse_hook_decision "$should_allow_raw")

# --- dcg: functional test via its own explain subcommand, json format ---
dcg_version=$("$DCG_BIN" --version 2>/dev/null | head -1 | awk '{print $NF}')
dcg_destructive_json=$("$DCG_BIN" explain "rm -rf ./nonexistent_test_dir_h4" --format json 2>/dev/null)
dcg_destructive_decision=$(echo "$dcg_destructive_json" | jq -r '.decision // "unknown"' 2>/dev/null)
dcg_benign_json=$("$DCG_BIN" explain "ls -la /tmp" --format json 2>/dev/null)
dcg_benign_decision=$(echo "$dcg_benign_json" | jq -r '.decision // "unknown"' 2>/dev/null)

rm -f "$DECOY"

cat > "$OUT" <<EOF
{
  "hypothesis": "H4",
  "timestamp": "${TS}",
  "block_secret_reveal": {
    "registered_in_settings": ${block_secret_registered},
    "script_path": $(printf '%s' "$HOOK_SCRIPT" | json_escape),
    "decoy_cat_decision": $(printf '%s' "$should_block_decision" | json_escape),
    "decoy_cat_expected": "deny",
    "benign_ls_decision": $(printf '%s' "$should_allow_decision" | json_escape),
    "benign_ls_expected": "allow"
  },
  "dcg": {
    "registered_in_settings": ${dcg_registered},
    "version": $(printf '%s' "$dcg_version" | json_escape),
    "destructive_rm_decision": $(printf '%s' "$dcg_destructive_decision" | json_escape),
    "destructive_rm_expected": "BLOCK or similar non-allow decision",
    "benign_ls_decision": $(printf '%s' "$dcg_benign_decision" | json_escape),
    "benign_ls_expected": "ALLOW"
  },
  "note": "No secret values recorded. Decoy file used for block_secret_reveal test contained a synthetic placeholder only, deleted immediately after."
}
EOF

echo "Wrote $OUT"
cat "$OUT"
