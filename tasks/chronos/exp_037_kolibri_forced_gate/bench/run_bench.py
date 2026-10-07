# SPDX-License-Identifier: MIT
"""Bench driver (BUILD_SPEC §5.7 run_bench.py; RUNBOOK step 11).

    run_bench.py --cells speed,speed_desc,fit,kl,tokenizer,c1[,ladder]

The cells always run in one fixed order, whatever order --cells lists them in:
speed cells first on a cool machine (10-minute idle, then thermalState
nominal), then fit (D1), kl (H8), tokenizer (H5), c1 (C1), and ladder (E6,
Tier 2) last. A cell whose newest output is complete is skipped; incomplete
outputs of a cell are moved to aborted/<UTC>-bench-<cell>/ with a NOTE.md
before it reruns. The identity guard runs first, then the version binding
(exp_037 DESIGN §6.3: runner.guard.require_environment with the newest gate
record; a real-mode record binds the run to P2's pins, MLX 0.32.3 /
mlx-metal 0.32.3 / mlx-lm 0.32.0, and a mismatch refuses the whole run), and
the gate of every Kolibri arm the requested cells use is checked before the
cool-down starts.

Exit codes: 0 every requested cell complete (run now or earlier); 1 a cell
failed; 3 a cell could not run (thermals, Tier-2 amendment missing, ...), no
failure. The last stdout line is JSON: {"cells": {cell: {...}}, "exit": code,
"environment": <what the version binding returned>}.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common  # noqa: E402

CELL_ORDER = ("speed", "speed_desc", "fit", "kl", "tokenizer", "c1", "ladder")
SPEED_CELLS = ("speed", "speed_desc")
# Kolibri arms each cell loads (gated before anything starts).
CELL_KOLIBRI_ARMS = {
    "speed": ("K4",),
    "speed_desc": ("K8",),
    "fit": ("K4",),
    "kl": ("K8", "K4"),
    "tokenizer": (),
    "c1": ("K8",),
    "ladder": ("K4", "K8"),
}


def runners() -> dict[str, Callable[..., Path]]:
    from bench import batch_flip, fit, kl_8v4, ladder, speed, tokenizer_ratio

    return {
        "speed": speed.run_h1,
        "speed_desc": speed.run_desc,
        "fit": fit.run_cell,
        "kl": kl_8v4.run_cell,
        "tokenizer": tokenizer_ratio.run_cell,
        "c1": batch_flip.run_cell,
        "ladder": ladder.run_cell,
    }


def parse_cells(spec: str) -> list[str]:
    names = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [c for c in names if c not in CELL_ORDER]
    if unknown or not names:
        raise ValueError(f"unknown or empty --cells {unknown or spec!r}; choose from {', '.join(CELL_ORDER)}")
    return [c for c in CELL_ORDER if c in names]


def _jsonable(obj: Any) -> Any:
    """What the version binding returned, as plain JSON (str() for anything else)."""
    try:
        return json.loads(json.dumps(obj, sort_keys=True, default=str, allow_nan=False))
    except ValueError:
        return str(obj)


def run(cells: list[str], results_dir: Path, run_fns: Optional[dict] = None) -> tuple[int, dict, Any]:
    """Run the requested cells; returns (exit code, {cell: status}, the version
    binding's record). A refused guard (identity, environment, gate) exits
    before any cell runs or the machine cools down."""
    run_fns = run_fns or runners()
    common.require_identity()
    binding = _jsonable(common.require_environment(results_dir))
    todo = [c for c in cells if common.complete_output(results_dir, c) is None]
    for c in todo:
        common.require_gates(CELL_KOLIBRI_ARMS[c])  # a refused gate stops the run

    status: dict[str, dict] = {}
    cooled: Optional[dict] = None
    cool_error: Optional[common.CouldNotRun] = None  # one failed cool-down covers both speed cells
    failed = could_not = False
    if "ladder" in todo:
        try:
            common.require_tier2()
        except SystemExit:  # the guard refuses by exiting; only the ladder is affected
            cells = [c for c in cells if c != "ladder"]
            could_not = True
            status["ladder"] = {"status": "could not run", "reason": "the Tier-2 analysis amendment is not at HEAD"}
    for cell in cells:
        done = common.complete_output(results_dir, cell)
        if done is not None:
            status[cell] = {"status": "skipped", "reason": "complete output exists", "file": done.name}
            continue
        moved = common.quarantine_incomplete(results_dir, cell)
        try:
            kwargs = {}
            if cell in SPEED_CELLS:
                if cool_error is not None:
                    raise cool_error
                if cooled is None:
                    try:
                        cooled = common.wait_cool()
                    except common.CouldNotRun as e:
                        cool_error = e
                        raise
                kwargs["cool"] = cooled
            print(f"[bench] {cell}: start", flush=True)
            path = run_fns[cell](results_dir, **kwargs)
            status[cell] = {"status": "complete", "file": Path(path).name}
        except common.CouldNotRun as e:
            could_not = True
            status[cell] = {"status": "could not run", "reason": str(e)}
        except Exception as e:  # one failing cell does not stop the others
            failed = True
            traceback.print_exc(file=sys.stderr)
            status[cell] = {"status": "failed", "error_type": type(e).__name__}
        finally:
            if "mlx" in sys.modules:
                from bench import genutil

                genutil.release()
        if moved:
            status[cell]["moved_to_aborted"] = [p.name for p in moved]
    code = 1 if failed else (3 if could_not else 0)
    return code, status, binding


def main(argv: Optional[list[str]] = None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_037 bench cells (BUILD_SPEC §5.7)")
    ap.add_argument("--cells", required=True, help=f"comma list from: {','.join(CELL_ORDER)}")
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        cells = parse_cells(args.cells)
    except ValueError as e:
        ap.error(str(e))
    results_dir = Path(args.results_dir or common.default_results_dir())
    code, status, binding = run(cells, results_dir)
    print(common.dumps({"cells": status, "exit": code, "environment": binding}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
