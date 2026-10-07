# SPDX-License-Identifier: Apache-2.0
# Uses the answer extractors of eval-framework v0.14.2 (Copyright 2025 Aleph Alpha Research GmbH), vendored
# verbatim under tasks/vendored_evalfw/. Modified by Miktam for Chronos exp_036, 2026-10-03: the German
# extractor's letter class [A-D] is widened to [A-J] for MMLU-ProX DE; nothing else is changed.
"""Multiple-choice letter extraction for GPQA and MMLU (BUILD_SPEC §5.6, HYPOTHESIS H2 rows, C10).

| task     | extractor                                   | label |
|----------|---------------------------------------------|-------|
| gpqa_en  | `tulu_answer_v2(4)`                         | V     |
| gpqa_de  | `tulu_answer_de()`                          | V     |
| mmlu_en  | `tulu_answer_v2(10)`                        | R     |
| mmlu_de  | `tulu_answer_de()` with `[A-D]` -> `[A-J]`  | R     |

The extractors are the vendored eval-framework functions themselves, imported lazily from
`tasks/vendored_evalfw/shim.py`; nothing is re-implemented here. The last match wins (eval-framework's
`last_match`, which also upper-cases). eval-framework's "[invalid]" becomes None, which the caller
records as a parse failure. V-row extractors are never amended (C10).

The only change is for MMLU-ProX DE: the German pattern is taken from the vendored `tulu_answer_de()`
object, its letter class `([A-D])` is replaced by `([A-J])` (exactly one occurrence, same flags) and it
is wrapped with the vendored `last_match`. `mmlu_de_pattern()` exposes the result for the tests.
"""

from __future__ import annotations

import functools
import importlib
import re
from typing import Callable

INVALID = "[invalid]"  # eval-framework's no-match sentinel (answer.py first_match / last_match)

TASKS = ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de")
N_OPTIONS = {"gpqa_en": 4, "gpqa_de": 4, "mmlu_en": 10, "mmlu_de": 10}

_DE_CLASS_FROM = "([A-D])"
_DE_CLASS_TO = "([A-J])"


@functools.lru_cache(maxsize=1)
def _shim():
    """The vendored eval-framework shim (tasks/vendored_evalfw/shim.py), imported on first use."""
    return importlib.import_module("tasks.vendored_evalfw.shim")


def _as_extractor(obj) -> Callable[[str], str]:
    """A plain str -> str extractor from an eval-framework ExtractFromCompletion or a bare callable."""
    inner = getattr(obj, "_extract", None)
    if callable(inner):
        return inner
    if hasattr(obj, "extract_answer"):
        return lambda text: obj.extract_answer(text, context=None, ground_truth=None, messages=[])
    if callable(obj):
        return obj
    raise TypeError(f"not an eval-framework extractor: {obj!r}")


def _closure_pattern(fn: Callable) -> re.Pattern:
    """The compiled regex captured by eval-framework's last_match(answer_re) closure."""
    for cell in fn.__closure__ or ():
        value = cell.cell_contents
        if isinstance(value, re.Pattern):
            return value
    raise TypeError("the vendored extractor does not close over a compiled pattern")


def _last_match_factory(de_extract: Callable) -> Callable[[re.Pattern], Callable[[str], str]]:
    shim = _shim()
    lm = getattr(shim, "last_match", None)
    if lm is None and hasattr(shim, "module"):
        lm = getattr(shim.module("eval_framework.answer"), "last_match", None)
    if lm is None:
        # The closure was built by answer.py's last_match; take that function from the same module.
        lm = de_extract.__globals__.get("last_match")
    if lm is None:
        raise ImportError("eval-framework last_match is not reachable through the vendored shim")
    return lm


@functools.lru_cache(maxsize=1)
def mmlu_de_pattern() -> re.Pattern:
    """`tulu_answer_de()`'s pattern with the letter class widened from A-D to A-J (nothing else)."""
    de = _as_extractor(_shim().tulu_answer_de())
    base = _closure_pattern(de)
    if base.pattern.count(_DE_CLASS_FROM) != 1:
        raise ValueError(f"expected exactly one {_DE_CLASS_FROM} in the vendored German pattern: {base.pattern!r}")
    return re.compile(base.pattern.replace(_DE_CLASS_FROM, _DE_CLASS_TO), base.flags)


@functools.lru_cache(maxsize=None)
def extractor(task: str) -> Callable[[str], str]:
    """The raw eval-framework extractor for a task (returns a letter or "[invalid]")."""
    shim = _shim()
    if task == "gpqa_en":
        return _as_extractor(shim.tulu_answer_v2(4))
    if task == "mmlu_en":
        return _as_extractor(shim.tulu_answer_v2(10))
    if task == "gpqa_de":
        return _as_extractor(shim.tulu_answer_de())
    if task == "mmlu_de":
        de = _as_extractor(shim.tulu_answer_de())
        return _last_match_factory(de)(mmlu_de_pattern())
    raise ValueError(f"no multiple-choice extractor for task {task!r}; known: {TASKS}")


def _extract(task: str, answer: str | None) -> str | None:
    if answer is None:
        return None
    letter = extractor(task)(answer)
    if letter == INVALID or not letter:
        return None
    return letter


def extract_gpqa_en(answer: str | None) -> str | None:
    return _extract("gpqa_en", answer)


def extract_gpqa_de(answer: str | None) -> str | None:
    return _extract("gpqa_de", answer)


def extract_mmlu_en(answer: str | None) -> str | None:
    return _extract("mmlu_en", answer)


def extract_mmlu_de(answer: str | None) -> str | None:
    return _extract("mmlu_de", answer)


EXTRACT = {
    "gpqa_en": extract_gpqa_en,
    "gpqa_de": extract_gpqa_de,
    "mmlu_en": extract_mmlu_en,
    "mmlu_de": extract_mmlu_de,
}


def score(task: str, answer: str | None, gold_letter: str) -> dict:
    """{extracted, correct, parse_status} for one post-reasoning answer (truncation is handled upstream)."""
    if task not in EXTRACT:
        raise ValueError(f"unknown multiple-choice task {task!r}")
    if answer is None or not answer.strip():
        return {"extracted": None, "correct": False, "parse_status": "empty_answer"}
    letter = EXTRACT[task](answer)
    if letter is None:
        return {"extracted": None, "correct": False, "parse_status": "no_match"}
    return {"extracted": letter, "correct": letter == str(gold_letter).strip().upper(), "parse_status": "ok"}
