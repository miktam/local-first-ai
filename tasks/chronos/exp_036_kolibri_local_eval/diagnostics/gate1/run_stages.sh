#!/usr/bin/env bash
# exp_036 gate-1 diagnostic suite, stages 0-4 on the mbp (FROZEN_RULES.md; README.md here).
# Not a gate run: never invokes gate/run_gate.py, writes no gate record, uses no fix cycle.
#
#   bash diagnostics/gate1/run_stages.sh            # from the kit root, in Andrei's terminal
#
# Order: stage0 (kernels) -> stage1 (G5) -> stage2 (G2) -> stage3 (G4, 7 steps) -> stage4 (G3).
# Each stage runs under caffeinate -i in its own process (one model-loading process at a time).
# Resumable: a stage that finished left $EXP036_WORK/diag_gate1/<stage>.done.json and is skipped;
# stage 3's steps resume the same way. A finished stage is never re-run (FROZEN_RULES.md §1.6).
# Stage 5 (the frozen mapping) runs later on the mini over the committed restricted copies.
#
# Environment: DIAG_TINY=<root> runs the smoke test on a tiny checkpoint root (outputs stay under it);
# DIAG_WITH_Q36=1 adds stage 1's optional descriptive Q36-8 check (first invocation only: once stage 1
# has finished it cannot be added, and a warning says so); DIAG_SKIP_ENV=1 (smoke tests
# only) does not source env/exp036.env and takes PY / EXP036_* from the caller.
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$KIT"
if [[ "${DIAG_SKIP_ENV:-0}" != "1" ]]; then
  # shellcheck disable=SC1091
  source env/exp036.env
fi
export MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${PY:?PY is not set (source env/exp036.env)}"
: "${EXP036_WORK:?EXP036_WORK is not set (source env/exp036.env)}"

TINY="${DIAG_TINY:-}"
ARGS=()
if [[ -n "$TINY" ]]; then
  ARGS=(--tiny "$TINY")
  W="$TINY/work/diag_gate1"
else
  W="$EXP036_WORK/diag_gate1"
fi
mkdir -p "$W/logs"

say() { printf '[run_stages %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" >&2; }
die() { say "STOP: $*"; exit 1; }

# ---- preconditions (DIAGNOSIS §4; FROZEN_RULES.md §1) ----------------------------------
"$PY" tools/hash_tree.py --check HYPOTHESIS.md >/dev/null || die "a hashed tree differs from HYPOTHESIS.md (tools/hash_tree.py --check)"
if [[ -z "$TINY" ]]; then
  git merge-base --is-ancestor a666f3e HEAD 2>/dev/null || die "HEAD does not contain a666f3e"
  other="$(git diff --name-only a666f3e HEAD -- . ':(exclude)diagnostics/gate1' ':(exclude)amendments' ':(exclude)HYPOTHESIS.md')"
  [[ -z "$other" ]] || die "files outside diagnostics/gate1, amendments and HYPOTHESIS.md changed since a666f3e: $other"
  git ls-files --error-unmatch diagnostics/gate1/FROZEN_RULES.md >/dev/null 2>&1 || die "FROZEN_RULES.md is not committed"
  dirty="$(git status --porcelain -- diagnostics/gate1 ':(exclude)diagnostics/gate1/out')"
  [[ -z "$dirty" ]] || die "diagnostics/gate1 differs from the committed (frozen) files: $dirty"
  [[ -d "$EXP036_WORK/ref" ]] || die "\$EXP036_WORK/ref (the gate's reference dumps) is missing"
  free_kb="$(df -Pk "$EXP036_WORK" | awk 'NR==2 {print $4}')"
  [[ "$free_kb" -ge $((45 * 1000 * 1000)) ]] || die "less than 45 GB free on \$EXP036_WORK ($((free_kb / 1000000)) GB)"
  for d in Kolibri-1-BF16 Kolibri-1-MLX-8bit-g64 Qwen3.6-35B-A3B-8bit; do
    [[ -d "$EXP036_MODELS/$d" ]] || die "\$EXP036_MODELS/$d is missing"
  done
fi
say "preconditions ok; outputs: $W (full) and $( [[ -n "$TINY" ]] && echo "$TINY/repo_out" || echo diagnostics/gate1/out ) (restricted)"

run_stage() {   # run_stage <stage> <script args...>
  local stage="$1"; shift
  if [[ -f "$W/$stage.done.json" ]]; then
    say "$stage: done earlier, skipped"
    return 0
  fi
  local log="$W/logs/${stage}_$(date -u +%Y%m%dT%H%M%SZ).log"
  say "$stage: start (log $log)"
  local t0=$SECONDS
  if ! caffeinate -i "$PY" "diagnostics/gate1/$stage.py" "$@" 2> >(tee -a "$log" >&2) | tee -a "$log"; then
    die "$stage failed; fix the cause and run this script again: finished stages are kept"
  fi
  say "$stage: done in $(( (SECONDS - t0) / 60 )) min"
}

run_stage stage0_kernels ${ARGS[@]+"${ARGS[@]}"}
Q36=()
[[ "${DIAG_WITH_Q36:-0}" == "1" ]] && Q36=(--with-q36)
if [[ "${DIAG_WITH_Q36:-0}" == "1" && -f "$W/stage1_g5.done.json" ]]; then
  say "WARNING: DIAG_WITH_Q36=1 has no effect: stage 1 finished earlier without it, and a finished stage is never re-run (FROZEN_RULES.md §1.6). The optional Q36-8 check is not part of this run."
fi
run_stage stage1_g5 ${ARGS[@]+"${ARGS[@]}"} ${Q36[@]+"${Q36[@]}"}
run_stage stage2_g2 ${ARGS[@]+"${ARGS[@]}"}
if [[ ! -f "$W/stage3_g4.done.json" ]]; then
  for step in g4 t9ids ref_fp32 bugtest ref_emu refcmp; do
    log="$W/logs/stage3_${step}_$(date -u +%Y%m%dT%H%M%SZ).log"
    say "stage3_g4 $step (log $log)"
    caffeinate -i "$PY" diagnostics/gate1/stage3_g4.py "$step" ${ARGS[@]+"${ARGS[@]}"} 2> >(tee -a "$log" >&2) | tee -a "$log" \
      || die "stage3_g4 $step failed; run this script again to resume"
  done
fi
run_stage stage3_g4 report ${ARGS[@]+"${ARGS[@]}"}
run_stage stage4_g3 ${ARGS[@]+"${ARGS[@]}"}

say "stages 0-4 complete. Restricted copies to commit (JSON only, checked for ids/text/host paths):"
ls -1 "$( [[ -n "$TINY" ]] && echo "$TINY/repo_out" || echo diagnostics/gate1/out )" >&2
say "never commit anything from $W (work files: npz, ids.json, logs, full JSON)"
say "next: the mbp session commits diagnostics/gate1/out/stage*_*.json and pushes on Andrei's go (README.md, 'After the run')"
