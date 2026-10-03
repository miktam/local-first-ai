# SPDX-License-Identifier: Apache-2.0
# Uses _extract_boxed and _strip_string_with_bug of eval-framework v0.14.2 (Copyright 2025 Aleph Alpha
# Research GmbH), vendored verbatim under tasks/vendored_evalfw/. Modified by Miktam for Chronos exp_036,
# 2026-10-03: the answer is compared as an integer instead of eval-framework's symbolic comparison.
"""AIME 2026 scoring (BUILD_SPEC §5.6; HYPOTHESIS Task sets: AIME EN is a V row, AIME DE an R row).

    extract(answer)  = _strip_string_with_bug(_extract_boxed(answer))   (vendored, unchanged)
    correct          = int(extract(answer)) == int(gold)

`_extract_boxed` takes the last `\\boxed{...}` (or `\\fbox`) in the post-reasoning answer, with braces
balanced; the vendored AIME benchmark has no "Answer:" fallback, and neither does this. The strip is the
MATH normalisation with its preserved bug, which eval-framework notes is inert on integer answers; it
also removes leading zeros ("042" -> "42").

The integer compare replaces eval-framework's MathReasoningCompletion (sympy) metric. For integer gold
in 0-999 the two agree on every plain integer; they differ only on forms such as "42.0", "\\frac{84}{2}"
or "6\\cdot7", which the sympy metric would accept and this does not. Those are recorded as
parse_status "non_integer" (an answer was given; it is wrong), not as a parse failure.
"""

from __future__ import annotations

import functools
import importlib
import re

_INT = re.compile(r"[+-]?\d+")


@functools.lru_cache(maxsize=1)
def _shim():
    return importlib.import_module("tasks.vendored_evalfw.shim")


def _vendored(*names: str):
    """The first shim attribute of these names (the shim exposes `extract_boxed`; BUILD_SPEC §5.5 lists
    the upstream names `_extract_boxed` / `_strip_string_with_bug`)."""
    shim = _shim()
    for n in names:
        fn = getattr(shim, n, None)
        if fn is not None:
            return fn
    raise ImportError(f"tasks/vendored_evalfw/shim.py exposes none of {names}")


def extract(answer: str | None) -> str | None:
    """The normalised content of the last boxed answer, or None if there is none."""
    if answer is None:
        return None
    boxed = _vendored("extract_boxed", "_extract_boxed")(answer)
    if boxed is None:
        return None
    return _vendored("strip_string_with_bug", "_strip_string_with_bug")(boxed)


def to_int(extracted: str | None) -> int | None:
    """An integer from the stripped boxed content, or None if it is not a plain integer."""
    if extracted is None:
        return None
    s = extracted.strip()
    if not _INT.fullmatch(s):
        return None
    return int(s)


def correct(extracted: str | None, gold) -> bool:
    value = to_int(extracted)
    return value is not None and value == int(str(gold).strip())


def score(answer: str | None, gold) -> dict:
    """{extracted (int or None), correct, parse_status} for one post-reasoning answer."""
    if answer is None or not answer.strip():
        return {"extracted": None, "correct": False, "parse_status": "empty_answer"}
    raw = extract(answer)
    if raw is None or not raw.strip():
        return {"extracted": None, "correct": False, "parse_status": "no_match"}
    value = to_int(raw)
    if value is None:
        return {"extracted": None, "correct": False, "parse_status": "non_integer"}
    return {"extracted": value, "correct": value == int(str(gold).strip()), "parse_status": "ok"}
