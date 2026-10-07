"""scorers/lang_tag.py: the frozen stopword tagger (BUILD_SPEC §5.6; gate G5 behaviour, E4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scorers import lang_tag

EXP_DIR = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((EXP_DIR / "scorers" / "golden" / "lang_tag_golden.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["text"][:30] or "<empty>")
def test_golden(case):
    assert lang_tag.tag(case["text"]) == case["expected"]


def test_lists_are_frozen_and_disjoint():
    assert len(lang_tag.ENGLISH) == 108
    assert len(lang_tag.GERMAN) == 107
    assert not lang_tag.ENGLISH & lang_tag.GERMAN
    for ambiguous in ("in", "an", "so", "was", "will", "also", "man", "war", "her"):
        assert ambiguous not in lang_tag.ENGLISH and ambiguous not in lang_tag.GERMAN
    assert all(w == w.lower() and w.isalpha() for w in lang_tag.ENGLISH | lang_tag.GERMAN)
    assert (lang_tag.MIN_HITS, lang_tag.DE_SHARE, lang_tag.EN_SHARE) == (2, 0.8, 0.2)


def test_thresholds_at_the_boundaries():
    de, en = "und", "the"
    assert lang_tag.tag(f"{de}") == "unknown"  # one hit < MIN_HITS
    assert lang_tag.tag(f"{de} {de}") == "de"
    assert lang_tag.tag(" ".join([de] * 4 + [en])) == "de"  # share 0.8
    assert lang_tag.tag(" ".join([de] * 3 + [en])) == "mixed"  # share 0.75
    assert lang_tag.tag(" ".join([en] * 4 + [de])) == "en"  # share 0.2
    assert lang_tag.tag(" ".join([en] * 3 + [de])) == "mixed"
    assert lang_tag.tag(None) == "unknown"


def test_counts_ignore_digits_and_punctuation():
    assert lang_tag.counts("Der 1. und der 2. Teil; the end!") == (3, 1)
