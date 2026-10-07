"""tools/precision.py: exact fp32 on the GPU (Amendment 1, 2026-10-04).

The hardware effect (TF32 on the M5 GPU) cannot be reproduced on a host
without TF32 kernels, so these tests pin the logic: the variable is set to 0
by default, any other value is refused before the GPU is touched, the probe
is exact in this process, and every entry point calls the guard first.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools import precision

EXP_DIR = Path(__file__).resolve().parents[1]
ENTRY_POINTS = ["gate/run_gate.py", "runner/run.py", "bench/run_bench.py",
                "tools/peer_check.py", "tools/preflight.py", "tools/dry_run.py"]


def test_suite_runs_with_tf32_off():
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
    rec = precision.ensure_exact_fp32()
    assert rec["MLX_ENABLE_TF32"] == "0"
    assert rec["probe_rel_l2"] <= precision.PROBE_REL_L2_MAX


def test_probe_is_far_from_tf32_error():
    # Exact fp32 is ~3e-7; TF32 is ~7.7e-4 (NOTE.md of the aborted mbp run).
    assert precision.probe_rel_l2() < 1e-6


def _run_guard(env_value):
    env = dict(os.environ)
    if env_value is None:
        env.pop("MLX_ENABLE_TF32", None)
    else:
        env["MLX_ENABLE_TF32"] = env_value
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from tools.precision import ensure_exact_fp32, PrecisionError\n"
            "import os\n"
            "try:\n"
            "    r = ensure_exact_fp32(probe=False)\n"
            "    print('ok', os.environ['MLX_ENABLE_TF32'])\n"
            "except PrecisionError as e:\n"
            "    print('refused')\n") % str(EXP_DIR)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_unset_becomes_zero():
    assert _run_guard(None) == "ok 0"


@pytest.mark.parametrize("value", ["1", "true", ""])
def test_any_other_value_is_refused(value):
    assert _run_guard(value) == "refused"


@pytest.mark.parametrize("rel", ENTRY_POINTS)
def test_entry_point_calls_the_guard_first(rel):
    tree = ast.parse((EXP_DIR / rel).read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    first, second = main.body[0], main.body[1]
    assert isinstance(first, ast.ImportFrom) and first.module == "tools.precision"
    assert isinstance(second, ast.Expr) and isinstance(second.value, ast.Call)
    assert getattr(second.value.func, "id", None) == "ensure_exact_fp32"
