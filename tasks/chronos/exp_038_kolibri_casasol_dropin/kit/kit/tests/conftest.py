"""Tests for exp_038's kit, run on the mini under exp_037's mini venv (MLX 0.32.3, mlx-lm 0.32.0).

Every model here is one of exp_037's tiny real-layout builds, written by exp_037's own test writers
(E37 tests/tiny_real_layout.py, port/convert.py), never real weights. E37 is imported read-only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
os.environ.setdefault("MLX_ENABLE_TF32", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

KIT = Path(__file__).resolve().parents[1]
E37 = Path(os.environ.get("EXP038_E37", Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate"))
for p in (str(KIT), str(E37), str(E37 / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

import tiny_real_layout  # noqa: E402  (E37)


@pytest.fixture(scope="session")
def tiny_set(tmp_path_factory):
    """exp_037's whole-tiny-gate validation build (seed 29, unsharpened): BF16 and its K8/K4 conversions."""
    return tiny_real_layout.build_gate_validation_set(tmp_path_factory.mktemp("tiny_real_s29_unsharp"))
