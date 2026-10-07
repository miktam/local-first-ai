# SPDX-License-Identifier: MIT
"""G1: the weight-free suite (HYPOTHESIS Phase 0 G1; BUILD_SPEC §5.3 g1_synthetic.py).

Runs the same pytest functions as the pre-push run, in a subprocess of the
current interpreter, and records the junit counts and sha256 in the gate
JSON. The gate's own end-to-end tests (tests/test_gate_drivers_tiny.py and
tests/test_gate_end_to_end_tiny.py) are left out so that G1 never recurses
into the gate.

exp_037 (DESIGN §3.3): EXCLUDE is the heavy tiny tests, which the mini's full
suite runs before the freeze (W14) and the gate's G1 does not: the two gate
end-to-end tests and the F_tiny calibration and validation
(tests/test_gate_ftiny.py). A G1 test asserts this tuple
(tests/test_integration_contracts.py).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from gate import common

TESTS_DIR = common.EXP_DIR / "tests"
EXCLUDE = ("test_gate_drivers_tiny.py", "test_gate_end_to_end_tiny.py", "test_gate_ftiny.py")


def run(junit_path: Path, selection: list[str] | None = None, require_all: bool = True,
        timeout_s: float = 3600.0) -> dict:
    junit_path = Path(junit_path)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    targets = [str(TESTS_DIR / s) for s in selection] if selection else [str(TESTS_DIR)]
    cmd = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={junit_path}"]
    cmd += [f"--ignore={TESTS_DIR / e}" for e in EXCLUDE] + targets
    env = dict(os.environ, EXP036_GATE_G1="1", PYTHONDONTWRITEBYTECODE="1")
    if require_all:
        # Every skip is a failure except a "build-host only:" skip: the tests that compare with files only the
        # mini holds (upstream peer files, the eval-framework checkout) passed there before the push and cannot
        # run on the run host (tests/conftest.py, RUNBOOK step 3).
        env["EXP036_REQUIRE_ALL"] = "run-host"
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    t0, utc0 = time.time(), common.utc_iso()
    p = subprocess.run(cmd, cwd=str(common.EXP_DIR), env=env, capture_output=True, text=True, timeout=timeout_s)
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    build_host_only = 0
    if junit_path.is_file():
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
        for s in suites:
            for k in counts:
                counts[k] += int(s.get(k, 0))
        for sk in root.iter("skipped"):
            if "build-host only:" in (sk.get("message") or "") + (sk.text or ""):
                build_host_only += 1
    tail = [ln for ln in p.stdout.splitlines() if ln.strip()][-1:] or [""]
    return {
        **counts,
        "skipped_build_host_only": build_host_only,
        "returncode": p.returncode,
        "selection": selection or ["tests/ (all)"],
        "excluded": list(EXCLUDE),
        "require_all": "run-host" if require_all else None,
        "summary": tail[0][:300],
        "junit_sha256": common.sha256_file(junit_path) if junit_path.is_file() else None,
        "t_start": utc0, "t_end": common.utc_iso(), "wall_s": time.time() - t0,
    }
