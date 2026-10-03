"""RGB scoring (BUILD_SPEC §5.6; HYPOTHESIS H6, E2, E3, C21).

RGB (Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented Generation", AAAI 2024;
github.com/chen700564/RGB @ 65ec39e4) is the source of the answer rule implemented here. RGB's
repository, code and data alike, is licensed CC BY-NC-SA 4.0; we do not copy its code. This module is
our own implementation of the rule RGB describes, written for exp_036 and covered by the repository's
MIT licence. tests/test_scorers_rgb.py runs RGB's own `checkanswer` in place from the local RGB checkout
and checks that the two agree on 500 synthetic cases. No RGB text (data, instructions, prompts) is in
this file or in any fixture.

The rule. A gold answer is a list of slots. A slot is either one string or a list of aliases. A slot is
satisfied when its string (or any one of its aliases), lower-cased, occurs in the lower-cased prediction.
A gold that is not a list counts as a single slot. `checkanswer` returns one 0/1 label per slot.

Categories (all on the post-reasoning answer; truncation is decided upstream and wins):
- closed-book / forced:  correct iff every slot is satisfied (at least one slot) and the case-sensitive
  string "insufficient information" (RGB's rejection string) is absent. Otherwise abstain if that string
  or an "abstain" lexicon phrase is present, else wrong. The lexicon never changes "correct" (H6).
- negative:  rejected iff "insufficient information" (case-sensitive) or an "abstain" lexicon phrase.
- fact-check:  detected iff "factual errors" (case-sensitive, RGB's string) or a "fact_error" lexicon
  phrase. corrected = detected and every slot of the true answer satisfied; detected = detected but not
  corrected; deferred = not detected and the counterfactual (fake) answer given without the true one;
  other = everything else (including a silent true answer with no detection).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

REJECT_STRING = "insufficient information"  # RGB's English rejection string, matched case-sensitively
FACT_STRING = "factual errors"  # RGB's English fact-error string, matched case-sensitively

LEXICON_PATH = Path(__file__).resolve().parent / "abstain_lexicon.json"

CB_CATEGORIES = ("correct", "abstain", "wrong")
NEG_CATEGORIES = ("rejected", "not")
FACT_CATEGORIES = ("corrected", "deferred", "detected", "other")


# ---------------------------------------------------------------------------------------------------
# The answer rule


def checkanswer(pred: str, gold) -> list[int]:
    """One label per gold slot: 1 if the slot's string, or any of its aliases, occurs (lower-cased) in pred."""
    haystack = pred.lower()
    slots = gold if isinstance(gold, list) else [gold]
    labels: list[int] = []
    for slot in slots:
        if isinstance(slot, list):
            satisfied = any(alias.lower() in haystack for alias in slot)
        else:
            satisfied = slot.lower() in haystack
        labels.append(int(satisfied))
    return labels


def all_slots(pred: str, gold) -> bool:
    """Every slot satisfied, and there is at least one slot (RGB counts `0 not in label and 1 in label`)."""
    labels = checkanswer(pred, gold)
    return bool(labels) and all(labels)


# ---------------------------------------------------------------------------------------------------
# The frozen lexicon


@functools.lru_cache(maxsize=1)
def lexicon() -> dict:
    with open(LEXICON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for key in ("abstain", "fact_error"):
        phrases = data[key]
        if len(set(phrases)) != len(phrases):
            raise ValueError(f"duplicate phrase in abstain_lexicon.json[{key!r}]")
        if any(p != normalise(p) for p in phrases):
            raise ValueError(f"abstain_lexicon.json[{key!r}] holds a phrase that is not in normal form")
    return data


def lexicon_sha256() -> str:
    return hashlib.sha256(LEXICON_PATH.read_bytes()).hexdigest()


_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_QUOTES)).strip()


@functools.lru_cache(maxsize=2)
def _matcher(kind: str) -> re.Pattern:
    phrases = sorted(lexicon()[kind], key=lambda p: (-len(p), p))
    alternation = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<![a-z0-9])(?:{alternation})(?![a-z0-9])")


def lexicon_hits(answer: str, kind: str) -> list[str]:
    """The lexicon phrases of one kind ("abstain" or "fact_error") found in the answer, in text order."""
    if kind not in ("abstain", "fact_error"):
        raise ValueError(f"unknown lexicon kind {kind!r}")
    return [m.group(0) for m in _matcher(kind).finditer(normalise(answer))]


def has_lexicon_hit(answer: str, kind: str) -> bool:
    return _matcher(kind).search(normalise(answer)) is not None


# ---------------------------------------------------------------------------------------------------
# Categories


def is_correct(answer: str, gold) -> bool:
    """H6 / forced-answer correctness: every slot satisfied and RGB's rejection string absent."""
    return REJECT_STRING not in answer and all_slots(answer, gold)


def classify_cb(answer: str, gold) -> str:
    """correct | abstain | wrong for a closed-book or forced-answer item (truncated is set upstream)."""
    if is_correct(answer, gold):
        return "correct"
    if REJECT_STRING in answer or has_lexicon_hit(answer, "abstain"):
        return "abstain"
    return "wrong"


def classify_neg(answer: str) -> str:
    """rejected | not for an RGB Negative item (noise rate 1.0: the right behaviour is to reject)."""
    if REJECT_STRING in answer or has_lexicon_hit(answer, "abstain"):
        return "rejected"
    return "not"


def detected_fact_error(answer: str) -> bool:
    return FACT_STRING in answer or has_lexicon_hit(answer, "fact_error")


def classify_fact(answer: str, true, fake) -> str:
    """corrected | deferred | detected | other for an RGB Fact-Check item."""
    detected = detected_fact_error(answer)
    true_given = all_slots(answer, true)
    if detected and true_given:
        return "corrected"
    if detected:
        return "detected"
    if all_slots(answer, fake) and not true_given:
        return "deferred"
    return "other"
