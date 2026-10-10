"""Entry point for the MLX arms: exact fp32 first, then harness/run_arm.py unchanged.

exp_037's entry points pin exact fp32 in-process before any GPU work (tools/precision.py ensure_exact_fp32:
MLX_ENABLE_TF32=0, then a GPU probe that an fp32 matmul is exact); exp_037's environment guard refuses a session
without it. This wrapper does the same and then calls run_arm.main with the same arguments, so the frozen R4 answer
path (harness/run_arm.py, kit/mlx_backend.py) is not edited.

    "$PY" harness/mlx_entry.py --arm K4 --sets probes,v1,de,twins --out runs/arms/K4
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
E37 = Path(os.environ.get("EXP038_E37", Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate"))


def exact_fp32() -> dict:
    for p in (str(E37), str(E37 / "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from tools.precision import ensure_exact_fp32  # exp_037's own

    rec = ensure_exact_fp32(probe=True)
    print(json.dumps({"exact_fp32": rec}), file=sys.stderr)
    return rec


if __name__ == "__main__":
    exact_fp32()
    sys.path.insert(0, str(P38 / "harness"))
    import run_arm  # noqa: E402

    sys.exit(run_arm.main(sys.argv[1:]))
