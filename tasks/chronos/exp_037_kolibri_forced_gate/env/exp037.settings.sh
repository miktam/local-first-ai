# exp_037 settings (non-secret, idempotent). Source it from the exp_037 directory after exp_036's env file:
#   cd <exp_036 dir> && source <exp_036's env file> && cd ../exp_037_kolibri_forced_gate && source env/exp037.settings.sh
# Exports EXP036_DIR, EXP (exp_037 from here on), EXP037_BUILDS, EXP037_VENV and PY (decision F2: the exp_037
# venv, MLX 0.32.3 / mlx-lm 0.32.0). No EXP036_* value changes: data, peers, private and work paths stay exp_036's.
_e36="$(cd "$PWD/../exp_036_kolibri_local_eval" 2>/dev/null && pwd -P)"
if [ "${PWD##*/}" != "exp_037_kolibri_forced_gate" ] || [ ! -f "$PWD/HYPOTHESIS.md" ]; then
  echo "exp037.settings.sh: source this file from the exp_037 directory (HYPOTHESIS.md present)" >&2
  unset _e36; return 1 2>/dev/null || exit 1
elif [ -z "$_e36" ] || [ ! -f "$_e36/HYPOTHESIS.md" ]; then
  echo "exp037.settings.sh: no ../exp_036_kolibri_local_eval with HYPOTHESIS.md next to this directory" >&2
  unset _e36; return 1 2>/dev/null || exit 1
elif [ -z "${EXP036_MODELS:-}" ]; then
  echo "exp037.settings.sh: EXP036_MODELS is not set; source exp_036's env file first (RUNBOOK block header)" >&2
  unset _e36; return 1 2>/dev/null || exit 1
fi
export EXP036_DIR="$_e36"
unset _e36
export EXP="$PWD"
export EXP037_BUILDS="${EXP037_BUILDS:-$EXP036_MODELS/exp037-builds}"
export EXP037_VENV="${EXP037_VENV:-$(dirname "$EXP036_MODELS")/exp037/venv}"
export PY="$EXP037_VENV/bin/python"
