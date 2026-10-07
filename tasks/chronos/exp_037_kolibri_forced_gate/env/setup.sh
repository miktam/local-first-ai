#!/usr/bin/env bash
# exp_037: create or sync the run-host venv at $EXP037_VENV (BUILD_SPEC §3; RUNBOOK step 3).
# Default: exp037/venv beside $EXP036_MODELS (the mbp: ~/models/exp037/venv), as env/exp037.settings.sh
# sets it. exp_036's venv keeps MLX 0.31.2 / mlx-lm 0.31.3 and is never touched (decision F2).
#
#   cd ~/REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate && bash env/setup.sh [--exact]
#
# Idempotent: a second run changes nothing. Python 3.12 only. Uses uv when it is
# on PATH (the mbp venv was built with uv, which installs no pip), otherwise
# python -m pip (bootstrapped with ensurepip if missing).
#   default   install the exact pins of env/requirements-mbp.txt, keep other packages
#   --exact   make the venv hold exactly those pins (uv pip sync / pip-sync semantics)
# This is the one kit step that downloads (packages from PyPI); it never calls sudo.
# Afterwards: "$PY" tools/preflight.py --quick compares the venv with env/versions.json.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exp="$(dirname "$here")"
req="$here/requirements-mbp.txt"
models="${EXP036_MODELS:-$HOME/models/exp036}"
venv="${EXP037_VENV:-$(dirname "$models")/exp037/venv}"
exact=0
[ "${1:-}" = "--exact" ] && exact=1

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE   # not used by pip/uv; set again by exp_036's env file

if command -v uv >/dev/null 2>&1; then
  if [ ! -x "$venv/bin/python" ]; then
    echo "[setup] creating $venv (Python 3.12, uv)"
    uv venv --python 3.12 "$venv"
  fi
  if [ $exact -eq 1 ]; then
    uv pip sync --python "$venv/bin/python" "$req"
  else
    uv pip install --python "$venv/bin/python" -r "$req"
  fi
else
  if [ ! -x "$venv/bin/python" ]; then
    py312="$(command -v python3.12 || true)"
    [ -n "$py312" ] || { echo "[setup] need uv or python3.12 on PATH" >&2; exit 1; }
    echo "[setup] creating $venv (Python 3.12, venv)"
    "$py312" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip --version >/dev/null 2>&1 || "$venv/bin/python" -m ensurepip --upgrade
  "$venv/bin/python" -m pip install -r "$req"
  if [ $exact -eq 1 ]; then
    echo "[setup] --exact needs uv; extra packages were kept (preflight records them)" >&2
  fi
fi

"$venv/bin/python" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), f"venv is Python {sys.version.split()[0]}, exp_037 needs 3.12"
import importlib.metadata as m
for name in ("mlx", "mlx-metal", "mlx-lm", "numpy", "safetensors", "tokenizers", "transformers", "pyarrow", "pytest"):
    print(f"[setup] {name}=={m.version(name)}")
import mlx.core, mlx_lm  # noqa: F401  (imports only; no network)
print("[setup] mlx and mlx_lm import")
PY
echo "[setup] done: $venv (exp_037). Next: the RUNBOOK block header (exp_036's env file, then env/exp037.settings.sh) && \"\$PY\" tools/preflight.py --quick"
