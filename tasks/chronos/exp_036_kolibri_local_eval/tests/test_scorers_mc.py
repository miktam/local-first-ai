"""scorers/mc.py: the vendored eval-framework extractors on golden synthetic answers (BUILD_SPEC §5.6).

Needs tasks/vendored_evalfw/shim.py (the tasks area). The test skips only while that module is absent.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((EXP_DIR / "scorers" / "golden" / "mc_golden.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mc():
    try:
        importlib.import_module("tasks.vendored_evalfw.shim")
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("tasks"):
            pytest.skip(f"tasks/vendored_evalfw/shim.py is not there yet ({e.name})")
        raise
    from scorers import mc as module

    return module


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["answer"][:40])
def test_golden(mc, case):
    for task in mc.TASKS:
        assert mc.EXTRACT[task](case["answer"]) == case[task], (task, case["answer"])


def test_v_rows_use_the_vendored_extractors_unchanged(mc):
    """GPQA EN/DE and MMLU EN are eval-framework's own extractor functions (C10: never amended)."""
    shim = importlib.import_module("tasks.vendored_evalfw.shim")
    probe = "Daher ist die Antwort (B). Also the answer is (C), then answer is: d"
    for task, policy in (("gpqa_en", shim.tulu_answer_v2(4)), ("mmlu_en", shim.tulu_answer_v2(10)),
                         ("gpqa_de", shim.tulu_answer_de())):
        upstream = policy.extract_answer(probe, context=None, ground_truth=None, messages=[])
        assert mc.extractor(task)(probe) == upstream
    assert mc.extractor("gpqa_en")("no letter here") == mc.INVALID


def test_mmlu_de_changes_only_the_letter_class(mc):
    de = mc._closure_pattern(mc._as_extractor(importlib.import_module("tasks.vendored_evalfw.shim").tulu_answer_de()))
    widened = mc.mmlu_de_pattern()
    assert widened.flags == de.flags
    assert widened.flags & re.IGNORECASE
    assert de.pattern.replace("([A-D])", "([A-J])") == widened.pattern
    # Hand-written expectation of the full pattern (eval-framework v0.14.2 gpqa_ellamind.py, A-J):
    assert widened.pattern == r"\b(?:ist\s+die\s+Antwort|Antwort\s+ist|Antwort:|answer\s+is)\s*\(?([A-J])\b\)?"


def test_letters_beyond_the_option_count_are_not_extracted(mc):
    for letter in "EFGHIJ":
        assert mc.extract_gpqa_en(f"Therefore, the answer is ({letter})") is None
        assert mc.extract_gpqa_de(f"Daher ist die Antwort ({letter})") is None
        assert mc.extract_mmlu_en(f"Therefore, the answer is ({letter})") == letter
        assert mc.extract_mmlu_de(f"Daher ist die Antwort ({letter})") == letter
    assert mc.extract_mmlu_en("Therefore, the answer is (K)") is None
    assert mc.extract_mmlu_de("Daher ist die Antwort (K)") is None


def test_score_statuses(mc):
    assert mc.score("gpqa_en", "Therefore, the answer is (B)", "B") == {"extracted": "B", "correct": True, "parse_status": "ok"}
    assert mc.score("gpqa_en", "Therefore, the answer is (B)", "c") == {"extracted": "B", "correct": False, "parse_status": "ok"}
    assert mc.score("mmlu_de", "keine Ahnung", "A") == {"extracted": None, "correct": False, "parse_status": "no_match"}
    assert mc.score("mmlu_en", "  \n", "A") == {"extracted": None, "correct": False, "parse_status": "empty_answer"}
    assert mc.score("mmlu_en", None, "A")["parse_status"] == "empty_answer"
    with pytest.raises(ValueError):
        mc.score("aime_en", "x", "A")


def test_last_match_wins_across_languages(mc):
    text = "Zuerst: Daher ist die Antwort (A). Korrektur: Die Antwort ist C."
    assert mc.extract_gpqa_de(text) == "C"
    assert mc.extract_mmlu_de(text) == "C"
    assert mc.extract_gpqa_en(text) is None
