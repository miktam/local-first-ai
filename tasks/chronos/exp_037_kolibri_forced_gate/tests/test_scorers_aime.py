"""scorers/aime.py: vendored boxed extraction + strip, then an integer compare (BUILD_SPEC §5.6)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((EXP_DIR / "scorers" / "golden" / "aime_golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def aime():
    try:
        importlib.import_module("tasks.vendored_evalfw.shim")
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("tasks"):
            pytest.skip(f"tasks/vendored_evalfw/shim.py is not there yet ({e.name})")
        raise
    from scorers import aime as module

    return module


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["answer"][:40])
def test_golden(aime, case):
    got = aime.score(case["answer"], case["gold"])
    assert got == {"extracted": case["extracted"], "correct": case["correct"], "parse_status": case["parse_status"]}
    if case["extracted"] is not None:
        assert aime.correct(aime.extract(case["answer"]), case["gold"]) is case["correct"]


def test_extract_is_the_vendored_pipeline(aime):
    shim = importlib.import_module("tasks.vendored_evalfw.shim")
    for text in ("x \\boxed{12} y", "\\boxed{042}", "\\boxed{\\dfrac{3}{4}}", "none"):
        boxed = shim.extract_boxed(text)
        want = None if boxed is None else shim.strip_string_with_bug(boxed)
        assert aime.extract(text) == want


def test_to_int_is_strict(aime):
    assert aime.to_int("7") == 7
    assert aime.to_int(" 7 ") == 7
    assert aime.to_int("+7") == 7
    for s in ("7.0", "7.", "\\frac{14}{2}", "seven", "", None, "1 2"):
        assert aime.to_int(s) is None
