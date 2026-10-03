"""Redaction helpers (BUILD_SPEC §2 Privacy, §5.9 `tools/redact.py`).

- `HOST_LABEL` is the fixed label every run record carries instead of a
  hostname, computer name, serial number or hardware UUID.
- `redact_path(p)` rewrites a path so that it starts with `$EXP036_PRIVATE`,
  `$EXP036_WORK`, `$EXP036_DATA`, `$EXP036_MODELS`, `$EXP` (this experiment
  directory), `$LFA` (the repository) or `~` instead of the real directory.
  The longest matching prefix wins, so a file under $EXP036_DATA is written as
  `$EXP036_DATA/...` even when $EXP036_DATA lies inside $EXP036_MODELS.
- `redact_text(s)` applies the same rewriting to every occurrence inside free
  text (error messages, command lines).
- `forbidden_keys(obj)` lists dictionary keys that look like machine
  identifiers; records must have none (tested in test_tools_preflight.py).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common

# The run host (HYPOTHESIS "Arms"; BUILD_SPEC §2 Privacy).
HOST_LABEL = "mbp (M5 Max, 128 GB)"
# The build host, for records written there (analysis, re-scoring).
MINI_LABEL = "mini (M4 Pro, 64 GB)"

FORBIDDEN_KEY_RE = re.compile(r"serial|uuid|hostname|computername", re.IGNORECASE)


def host_label(chip: str | None = None, memory_bytes: int | None = None) -> str:
    """HOST_LABEL on the run host, MINI_LABEL on the build host, otherwise a
    generic label built only from the chip name and the memory size."""
    if chip == "Apple M5 Max" and memory_bytes == 128 * 2**30:
        return HOST_LABEL
    if chip == "Apple M4 Pro" and memory_bytes == 64 * 2**30:
        return MINI_LABEL
    gb = f"{round(memory_bytes / 2**30)} GB" if memory_bytes else "unknown memory"
    return f"other host ({chip or 'unknown chip'}, {gb})"


def _prefixes() -> list[tuple[str, str]]:
    """(real prefix, label) pairs, longest real prefix first."""
    pairs: list[tuple[Path, str]] = [
        (common.private_dir(), "$EXP036_PRIVATE"),
        (common.work_dir(), "$EXP036_WORK"),
        (common.data_dir(), "$EXP036_DATA"),
        (common.models_dir(), "$EXP036_MODELS"),
        (common.EXP_DIR, "$EXP"),
    ]
    lfa = os.environ.get("LFA")
    if lfa:
        pairs.append((Path(lfa).expanduser(), "$LFA"))
    else:
        # The repository root, derived from this file (tasks/chronos/<exp>/tools).
        pairs.append((common.EXP_DIR.parents[2], "$LFA"))
    pairs.append((Path.home(), "~"))
    out: list[tuple[str, str]] = []
    seen = set()
    for p, label in pairs:
        for variant in {str(p), str(p.resolve()) if p.exists() else str(p)}:
            v = variant.rstrip("/")
            if v and v != "/" and v not in seen:
                seen.add(v)
                out.append((v, label))
    out.sort(key=lambda t: len(t[0]), reverse=True)
    return out


def redact_path(p) -> str:
    """Path → string with the environment prefixes and $HOME replaced."""
    s = str(Path(p).expanduser()) if not isinstance(p, str) else os.path.expanduser(p)
    for real, label in _prefixes():
        if s == real:
            return label
        if s.startswith(real + "/"):
            return label + s[len(real):]
    return s


def redact_text(text: str) -> str:
    """Replace every occurrence of the prefixes inside free text."""
    for real, label in _prefixes():
        text = re.sub(re.escape(real) + r"(?=/|\b|$)", label, text)
    return text


def forbidden_keys(obj, path: str = "") -> list[str]:
    """Key paths in a JSON-like object whose key matches serial|uuid|hostname|
    computername (case-insensitive)."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = f"{path}.{k}" if path else str(k)
            if FORBIDDEN_KEY_RE.search(str(k)):
                found.append(kp)
            found.extend(forbidden_keys(v, kp))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found.extend(forbidden_keys(v, f"{path}[{i}]"))
    return found
