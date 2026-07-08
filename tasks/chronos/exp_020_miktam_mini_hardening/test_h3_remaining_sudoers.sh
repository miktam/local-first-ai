#!/usr/bin/env bash
# H3: for each remaining NOPASSWD-listed command, confirm empirically whether
# it's actually passwordless (not just apparently so in `sudo -l`), and whether
# it grants anything beyond its narrow intended scope. Nothing here shuts down,
# reboots, or restarts a real service — pharos-deploy is reviewed by source
# only (already read in the Phase 0 audit), not invoked, since running it for
# real would copy files and bounce a live daemon.
set -uo pipefail

TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/h3_remaining_sudoers_${TS}.json"
mkdir -p "$RESULTS_DIR"

check() {
  # NOTE (correction after first run): must resolve each binary's real path
  # with `which` before testing — an earlier run guessed /usr/sbin/pmset,
  # which doesn't exist, so sudo returned "command not found" instead of
  # "a password is required" and the string-match logic below misread that
  # as passwordless. Fixed by resolving the path first and treating "command
  # not found" as an error (exit 3), not a passwordless result.
  local name="$1"; shift
  local out exit_code
  out=$("$@" 2>&1)
  exit_code=$?
  if printf '%s' "$out" | grep -qi "command not found"; then
    passwordless="null"
    note="test error: wrong path, not a real result"
  elif printf '%s' "$out" | grep -qi "a password is required"; then
    passwordless="false"
    note=""
  else
    passwordless="true"
    note=""
  fi
  printf '{"command":"%s","passwordless":%s,"exit_code":%d,"note":"%s"}' "$name" "$passwordless" "$exit_code" "$note"
}

r_shutdown=$(check "shutdown (invalid flag, never actually shuts down)" sudo -n "$(which shutdown 2>/dev/null || echo /sbin/shutdown)" -zz9invalidflag)
r_lsof=$(check "lsof -p 1" sudo -n "$(which lsof)" -p 1)
r_asitop=$(check "asitop --help" sudo -n "$(which asitop)" --help)
r_powermetrics=$(check "powermetrics -n1 -i1 (single sample)" sudo -n "$(which powermetrics)" -n1 -i1 --samplers cpu_power)
r_killall_pm=$(check "killall powermetrics (no-op, none running)" sudo -n "$(which killall)" powermetrics)
r_pmset=$(check "pmset -g" sudo -n "$(which pmset)" -g)
r_cp=$(check "cp (control — expected password-gated per Phase 0 correction)" sudo -n "$(which cp)" /etc/hostname /tmp/nestor-h3-cp-recheck-$$.txt)

cat > "$OUT" <<EOF
{
  "hypothesis": "H3",
  "timestamp": "$TS",
  "note_on_scope": "pharos-deploy script reviewed by source only in Phase 0 (fixed hardcoded paths, no attacker-influenced input, bounded to one repo + one plist) — not invoked here, since a real invocation would copy files and bounce a live daemon.",
  "checks": [
    $r_shutdown,
    $r_lsof,
    $r_asitop,
    $r_powermetrics,
    $r_killall_pm,
    $r_pmset,
    $r_cp
  ]
}
EOF

echo "Result written to $OUT"
cat "$OUT"
rm -f /tmp/nestor-h3-cp-recheck-*.txt
