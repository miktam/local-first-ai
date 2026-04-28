#!/usr/bin/env bash
# run_all.sh
# Execute every script in tests/, capturing both the structured JSON
# result and the human-readable trace, and produce a summary in
# results/<date>-summary.md.

set -u  # NB: not -e; we want to continue past failed tests.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATE_UTC="$(date -u +%Y-%m-%d)"
TS_UTC="$(date -u +%FT%TZ)"
SUMMARY="${ROOT_DIR}/results/${DATE_UTC}-summary.md"

mkdir -p "${ROOT_DIR}/results"

{
  echo "# Investigation run ${TS_UTC}"
  echo
  echo "Host: $(hostname)"
  echo "OS:   $(uname -a)"
  echo "Ollama: $(ollama --version 2>/dev/null || echo 'unknown')"
  echo
} > "$SUMMARY"

declare -a TESTS
while IFS= read -r f; do
  TESTS+=("$f")
done < <(find "${ROOT_DIR}/tests" -type f -name '*.sh' | sort)

for t in "${TESTS[@]}"; do
  name="$(basename "$t")"
  echo "[runner] === $name ===" >&2

  stderr_log="${ROOT_DIR}/results/${DATE_UTC}-${name%.sh}.stderr.log"

  set +e
  out=$(bash "$t" 2> "$stderr_log")
  rc=$?
  set -e

  case "$rc" in
    0) status="✓ supported" ;;
    1) status="✗ rejected" ;;
    2) status="? inconclusive" ;;
    3) status="∅ could not run" ;;
    *) status="!! exit $rc" ;;
  esac

  {
    echo "## $name — $status"
    echo
    echo '```json'
    # Last line of stdout is the structured result; print all stdout
    # for completeness but expect the JSON at the bottom.
    echo "$out"
    echo '```'
    echo
    echo "Stderr trace: \`results/${DATE_UTC}-${name%.sh}.stderr.log\`"
    echo
  } >> "$SUMMARY"
done

echo "[runner] Summary written to $SUMMARY" >&2
