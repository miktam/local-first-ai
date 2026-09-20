#!/usr/bin/env bash
# H3: what can a process already running as miktam02 read, with zero
# privilege escalation? No sudo anywhere in this script — the whole point is
# that none should be needed. Only existence/permissions/metadata are
# recorded; actual secret VALUES are never written to results/, matching the
# same discipline exp_020 applied to the real Tailscale IP.
set -uo pipefail

TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/h3_secret_exposure_${TS}.json"
mkdir -p "$RESULTS_DIR"

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'; }

# --- SSH private keys ---
ssh_keys_json="[]"
if [ -d "$HOME/.ssh" ]; then
  entries=()
  for key in "$HOME/.ssh"/id_rsa "$HOME/.ssh"/id_ed25519 "$HOME/.ssh"/id_ecdsa "$HOME/.ssh"/id_dsa; do
    [ -f "$key" ] || continue
    perm=$(stat -f "%OLp" "$key" 2>/dev/null || echo "unknown")
    # ssh-keygen -y just derives the public key from the private one; empty
    # passphrase (-P "") succeeds only if the key is NOT passphrase-protected.
    # Read-only, no modification, no network call.
    if ssh-keygen -y -f "$key" -P "" >/dev/null 2>&1; then
      passphrase_protected="false"
    else
      passphrase_protected="true"
    fi
    entries+=("{\"path\":\"$(basename "$key")\",\"permissions\":\"$perm\",\"passphrase_protected\":$passphrase_protected}")
  done
  if [ "${#entries[@]}" -gt 0 ]; then
    ssh_keys_json="[$(IFS=,; echo "${entries[*]}")]"
  fi
fi

# --- Ollama proxy token ---
token_path="$HOME/.config/ollama-proxy/token"
if [ -f "$token_path" ]; then
  token_perm=$(stat -f "%OLp" "$token_path" 2>/dev/null || echo "unknown")
  token_size=$(stat -f "%z" "$token_path" 2>/dev/null || echo "unknown")
  token_readable_by_this_account="true"
else
  token_perm="n/a"
  token_size="n/a"
  token_readable_by_this_account="n/a — file not found"
fi

# --- Shell startup files: variable NAMES only, never values ---
shell_files_json="[]"
entries=()
for rc in "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile"; do
  [ -f "$rc" ] || continue
  vars=$(grep -oE '\b[A-Z_]*(API_KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*\b' "$rc" 2>/dev/null | sort -u | tr '\n' ',' | sed 's/,$//')
  if [ -n "$vars" ]; then
    entries+=("{\"file\":\"$(basename "$rc")\",\"exported_secret_like_names\":\"$vars\"}")
  fi
done
if [ "${#entries[@]}" -gt 0 ]; then
  shell_files_json="[$(IFS=,; echo "${entries[*]}")]"
fi

# --- Current environment: variable NAMES only ---
env_secret_names=$(env | grep -oE '^[A-Z_]*(API_KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*=' | sed 's/=$//' | sort -u | tr '\n' ',' | sed 's/,$//')
[ -z "$env_secret_names" ] && env_secret_names="none found in current shell environment"

# --- Git credential storage ---
git_creds_plaintext="false"
[ -f "$HOME/.git-credentials" ] && git_creds_plaintext="true"
netrc_present="false"
[ -f "$HOME/.netrc" ] && netrc_present="true"
credential_helper=$(git config --global credential.helper 2>/dev/null || echo "none configured")

cat > "$OUT" <<EOF
{
  "hypothesis": "H3",
  "timestamp": "$TS",
  "account": "$(whoami)",
  "ssh_private_keys": $ssh_keys_json,
  "ollama_proxy_token": {
    "path": "~/.config/ollama-proxy/token",
    "exists": $([ -f "$token_path" ] && echo true || echo false),
    "permissions": "$token_perm",
    "size_bytes": "$token_size",
    "readable_by_this_account": "$token_readable_by_this_account"
  },
  "shell_startup_files_with_secret_like_var_names": $shell_files_json,
  "current_environment_secret_like_var_names": "$(printf '%s' "$env_secret_names" | json_escape | sed 's/^"//;s/"$//')",
  "git_credential_storage": {
    "plaintext_git_credentials_file": $git_creds_plaintext,
    "netrc_present": $netrc_present,
    "credential_helper": "$credential_helper"
  },
  "note": "No secret VALUES are recorded above, only existence/permissions/names, per the redaction discipline established in exp_020. See AUDIT_FINDINGS.md for the interpretation of these results."
}
EOF

echo "Result written to $OUT"
cat "$OUT"
