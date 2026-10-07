"""Shared pytest setup for the exp_036 tests.

Puts the experiment directory (port/, reference/) and this tests/ directory on
sys.path, provides the tiny checkpoints (and, for exp_037, the tiny real-layout
sets tiny_real_cal, tiny_real_val and tiny_real_val_unsharp; DESIGN §5.1), and
resolves the Kolibri tokenizer from the environment. Nothing here touches the network.

Environment (BUILD_SPEC 1 and 6):
  EXP036_TOK            the Kolibri tokenizer directory (tokenizer.json and
                        tokenizer_config.json); EXP036_TOKENIZER_DIR is the
                        older name and still accepted.
  EXP036_REQUIRE_ALL=1  every skip becomes a failure (the pre-push run on the
                        mini), so a missing fixture or sibling module can never
                        pass silently. EXP036_REQUIRE_ALL=run-host (the mbp,
                        RUNBOOK step 3) does the same except for skips marked
                        "build-host only:" (files only the mini holds).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Do not leave __pycache__ in port/ and reference/ (or here) after a test run.
sys.dont_write_bytecode = True

# Exact fp32 on the GPU (tools/precision.py; Amendment 1): mlx reads
# MLX_ENABLE_TF32 once, at the first GPU matmul, so it is set here before any
# test module is imported. pytest_sessionstart below checks that it took.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

TESTS_DIR = Path(__file__).resolve().parent
EXP_DIR = TESTS_DIR.parent

for _p in (str(EXP_DIR), str(TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Tokenizer tests must never reach the Hub (transformers' Mistral-regex check
# can call hf_api for non-local paths).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import tiny_checkpoint  # noqa: E402
import tiny_real_layout  # noqa: E402


def _write(tmp_path_factory, preset: str) -> Path:
    out = tmp_path_factory.mktemp(f"tiny_{preset}")
    return tiny_checkpoint.write_tiny_checkpoint(
        out,
        seed=0,
        preset=preset,
        # The port file may not exist yet; reference-only tests do not need it.
        copy_port=tiny_checkpoint.PORT_MODEL_FILE.exists(),
    )


@pytest.fixture(scope="session")
def tiny_vendor_dir(tmp_path_factory) -> Path:
    return _write(tmp_path_factory, "vendor")


@pytest.fixture(scope="session")
def tiny_pattern5_dir(tmp_path_factory) -> Path:
    return _write(tmp_path_factory, "pattern5")


@pytest.fixture(scope="session")
def tiny_checkpoint_factory(tmp_path_factory):
    """make(preset, seed) -> checkpoint dir, written once per (preset, seed)
    per session, for tests that run both presets at several weight seeds."""
    made: dict[tuple[str, int], Path] = {}

    def make(preset: str, seed: int) -> Path:
        key = (preset, seed)
        if key not in made:
            out = tmp_path_factory.mktemp(f"tiny_{preset}_s{seed}")
            made[key] = tiny_checkpoint.write_tiny_checkpoint(
                out,
                seed=seed,
                preset=preset,
                copy_port=tiny_checkpoint.PORT_MODEL_FILE.exists(),
            )
        return made[key]

    return make


# exp_037 (DESIGN §5.1; W0b): the tiny real-layout checkpoints, each a
# tiny_real_layout.RealLayoutSet (.bf16, .k8, .k4, .root as a models dir), written
# once per session when first requested. Shared: copy before modifying.
#   tiny_real_cal          seed 23, q/k norms x 3: F_tiny's calibration; the fp32 forced checks' own tests
#   tiny_real_val          seed 29, x 3: F_tiny's out-of-sample validation
#   tiny_real_val_unsharp  seed 29, x 1: the whole tiny gate's validation build (decision (a))
@pytest.fixture(scope="session")
def tiny_real_cal(tmp_path_factory) -> "tiny_real_layout.RealLayoutSet":
    """F_tiny's calibration checkpoint: seed 23, BF16, and its 8-bit and 4-bit
    group-64 conversions by port/convert.py."""
    root = tmp_path_factory.mktemp(f"tiny_real_s{tiny_real_layout.SEED_CAL}")
    return tiny_real_layout.build_real_layout_set(root, tiny_real_layout.SEED_CAL)


@pytest.fixture(scope="session")
def tiny_real_val(tmp_path_factory) -> "tiny_real_layout.RealLayoutSet":
    """The out-of-sample validation checkpoint: seed 29, BF16, and its 8-bit
    and 4-bit group-64 conversions by port/convert.py."""
    root = tmp_path_factory.mktemp(f"tiny_real_s{tiny_real_layout.SEED_VAL}")
    return tiny_real_layout.build_real_layout_set(root, tiny_real_layout.SEED_VAL)


@pytest.fixture(scope="session")
def tiny_real_val_unsharp(tmp_path_factory) -> "tiny_real_layout.RealLayoutSet":
    """The whole tiny gate's validation build (DESIGN §5.1; decision (a), 2026-10-06):
    seed 29 with the q/k norms unsharpened (sharpen = 1), BF16 and its 8-bit and 4-bit
    group-64 conversions by port/convert.py (tiny_real_layout.build_gate_validation_set).
    tests/test_gate_end_to_end_tiny.py and the port run of tests/test_gate_drivers_tiny.py
    run on it. F_tiny and the fp32 forced checks' own tests keep the sharpened builds above."""
    root = tmp_path_factory.mktemp(f"tiny_real_s{tiny_real_layout.SEED_VAL}_unsharp")
    return tiny_real_layout.build_gate_validation_set(root)


# EXP036_REQUIRE_ALL=1         every skip is a failure (the pre-push run on the build host, the mini).
# EXP036_REQUIRE_ALL=run-host  every skip is a failure except a "build-host only:" skip (a test that compares with
#                              files only the mini holds: upstream peer files, the eval-framework checkout); RUNBOOK
#                              step 3 on the mbp. Those tests already passed on the mini before the push.
REQUIRE_ALL = os.environ.get("EXP036_REQUIRE_ALL", "") in ("1", "run-host")
RUN_HOST = os.environ.get("EXP036_REQUIRE_ALL", "") == "run-host"
BUILD_HOST_ONLY = "build-host only:"


def _no_skips(report) -> None:
    if REQUIRE_ALL and report.skipped and not hasattr(report, "wasxfail"):
        reason = report.longrepr[-1] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        if RUN_HOST and BUILD_HOST_ONLY in str(reason):
            return
        report.outcome = "failed"
        report.longrepr = (f"EXP036_REQUIRE_ALL={os.environ.get('EXP036_REQUIRE_ALL')}: "
                           f"skipped tests count as failures ({reason})")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _no_skips(outcome.get_result())


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    outcome = yield
    _no_skips(outcome.get_result())


def pytest_sessionstart(session):
    """Refuse to run the suite with TF32 active (tools/precision.py; Amendment 1)."""
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()


@pytest.fixture(scope="session")
def tokenizer_dir() -> Path:
    """Directory holding tokenizer.json + tokenizer_config.json of Kolibri 1.

    $EXP036_TOK, else $EXP036_TOKENIZER_DIR, else $EXP036_MODELS/Kolibri-1-BF16,
    else skip.
    """
    candidates = []
    for var in ("EXP036_TOK", "EXP036_TOKENIZER_DIR"):
        if os.environ.get(var):
            candidates.append(Path(os.environ[var]).expanduser())
    if os.environ.get("EXP036_MODELS"):
        candidates.append(Path(os.environ["EXP036_MODELS"]).expanduser() / "Kolibri-1-BF16")
    for d in candidates:
        if (d / "tokenizer.json").is_file() and (d / "tokenizer_config.json").is_file():
            return d
    pytest.skip(
        "no Kolibri tokenizer: set EXP036_TOK (or EXP036_TOKENIZER_DIR) or EXP036_MODELS "
        f"(checked: {[str(c) for c in candidates] or 'nothing set'})"
    )
