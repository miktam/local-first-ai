# SPDX-License-Identifier: MIT
"""The committed rule for the G0 tokenizer-parity lines (HYPOTHESIS Phase 0 G0;
BUILD_SPEC §5.3 tokenizer_lines.py).

  1. every non-empty line of the gate-text sources T1-T4 (gate/texts/src/);
  2. every distinct identifier ([A-Za-z_][A-Za-z0-9_]*) in the kit's own .py
     files under this experiment directory, sorted;
  3. 600 seeded synthetic digit strings (numbers, decimals in both notations,
     dates, times, thousands separators, ranges, units), numpy PCG64 seeded
     with the first 8 bytes of sha256("exp036|tokenizer_lines");
  4. at gate time only, never committed: the lines of the MMLU-ProX(-Lite)
     EN and DE test questions found under $EXP036_DATA.

The gate record holds only the counts and the sha256 of each group (the lines
joined with "\\n"), never the MMLU text.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from gate import common

N_DIGIT_STRINGS = 600
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SKIP_PARTS = {"__pycache__", ".pytest_cache", "results", "aborted", "evidence"}


def source_lines() -> list[str]:
    out = []
    for tid in ("T1", "T2", "T3", "T4"):
        text = (common.TEXTS_DIR / "src" / f"{tid}_source.txt").read_text(encoding="utf-8")
        out += [ln for ln in text.split("\n") if ln.strip()]
    return out


def identifier_lines(root: Path = common.EXP_DIR) -> list[str]:
    names = set()
    for p in sorted(Path(root).rglob("*.py")):
        if any(part in _SKIP_PARTS for part in p.relative_to(root).parts):
            continue
        names.update(_IDENT.findall(p.read_text(encoding="utf-8", errors="replace")))
    return sorted(names)


def digit_lines(n: int = N_DIGIT_STRINGS) -> list[str]:
    rng = np.random.Generator(np.random.PCG64(common.seed_from("tokenizer_lines", prefix="exp036")))  # registered fixture seed, not an exp_037 draw

    def digits(k):
        return "".join(str(int(d)) for d in rng.integers(0, 10, size=k))

    shapes = [
        lambda: digits(int(rng.integers(1, 41))),
        lambda: f"{digits(int(rng.integers(1, 5)))}.{digits(int(rng.integers(1, 7)))}",
        lambda: f"{digits(int(rng.integers(1, 5)))},{digits(int(rng.integers(1, 4)))} km",
        lambda: f"{int(rng.integers(1900, 2100))}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}",
        lambda: f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d} Uhr",
        lambda: f"{int(rng.integers(1, 1000)):,}.{digits(3)}".replace(",", "."),
        lambda: f"{int(rng.integers(1, 10**9)):,}",
        lambda: f"{digits(int(rng.integers(1, 4)))}–{digits(int(rng.integers(1, 4)))} %",
        lambda: f"1:{digits(int(rng.integers(2, 6)))}",
        lambda: f"€{digits(int(rng.integers(1, 6)))},{digits(2)}",
    ]
    return [shapes[i % len(shapes)]() for i in range(n)]


def mmlu_lines(data: Path | None = None) -> tuple[list[str], list[str]]:
    """(lines, files) of the MMLU-ProX(-Lite) EN/DE test questions under
    $EXP036_DATA (pyarrow imported lazily). Read at gate time; never written."""
    data = Path(data) if data is not None else common.data_dir()
    files = []
    for ds in ("MMLU-ProX-Lite", "MMLU-ProX"):
        root = data / ds
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.parquet")):
            rel = p.relative_to(root).as_posix().lower()
            parts = re.split(r"[/_.\-]", rel)
            if "test" in rel and ("en" in parts or "de" in parts):
                files.append(p)
    lines, used = [], []
    if files:
        import pyarrow.parquet as pq

        for p in files:
            try:
                table = pq.read_table(str(p), columns=["question"])
            except Exception:  # not a question file (another layout): skipped, not fatal
                continue
            used.append(p)
            for q in table.column("question").to_pylist():
                lines += [ln for ln in str(q).split("\n") if ln.strip()]
    return lines, [common.redact_path(p) for p in used]


def committed_rule_lines() -> dict[str, list[str]]:
    return {"sources": source_lines(), "identifiers": identifier_lines(), "digits": digit_lines()}


def group_digest(lines: list[str]) -> str:
    return common.sha256_bytes("\n".join(lines).encode("utf-8"))
