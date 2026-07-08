#!/usr/bin/env bash
# H1: does the unrestricted `tail` NOPASSWD grant allow reading a file miktam02
# cannot otherwise read? Target: /etc/sudoers itself (root:wheel, mode 440) —
# using the sudoers file as the target also makes the finding self-evidencing:
# if this succeeds, the same grant that's supposed to be narrow can read the
# full list of every other grant on the machine.
set -uo pipefail

TS="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
RESULTS_DIR="$(dirname "$0")/results"
OUT="$RESULTS_DIR/h1_tail_${TS}.json"
TARGET="/etc/sudoers"

direct_read_output=$(cat "$TARGET" 2>&1)
direct_read_exit=$?

sudo_tail_output=$(sudo -n /usr/bin/tail -n 5 "$TARGET" 2>&1)
sudo_tail_exit=$?

if [ "$direct_read_exit" -eq 0 ]; then
  verdict="inconclusive"
  code=2
  note="Direct read of $TARGET succeeded without sudo — not a useful escalation target on this run."
elif [ "$sudo_tail_exit" -eq 0 ]; then
  verdict="confirmed"
  code=0
  note="Direct read denied (exit $direct_read_exit); sudo -n tail succeeded (exit 0) with no password prompt — H1 confirmed."
else
  verdict="refuted"
  code=1
  note="Direct read denied and sudo -n tail also failed (exit $sudo_tail_exit) — tail grant did not bypass permissions on this target."
fi

mkdir -p "$RESULTS_DIR"
cat > "$OUT" <<EOF
{
  "hypothesis": "H1",
  "timestamp": "$TS",
  "target_file": "$TARGET",
  "direct_read_exit_code": $direct_read_exit,
  "direct_read_stderr": $(printf '%s' "$direct_read_output" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
  "sudo_tail_exit_code": $sudo_tail_exit,
  "sudo_tail_output_line_count": $(printf '%s' "$sudo_tail_output" | wc -l | tr -d ' '),
  "verdict": "$verdict",
  "note": "$note"
}
EOF

echo "Result written to $OUT"
cat "$OUT"
exit "$code"
