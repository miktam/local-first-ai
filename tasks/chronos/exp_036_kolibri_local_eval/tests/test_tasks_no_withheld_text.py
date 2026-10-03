"""tasks/ (and the tasks tests) contain no withheld text (BUILD_SPEC §2 Rights; HYPOTHESIS "Sources & Rights").

Checked against the withheld sources available on this host:
- RGB (from $EXP036_DATA/RGB-src, or the mini's build-time copy): 8-word shingles of every query, answer,
  document and instruction string; every query of >= 5 words and both instruction strings as whole
  normalised substrings.
- The GPQA question that upstream eval-framework embeds in gpqa.py (from $EXP036_EVALFW_SRC or the mini's
  checkout): 4-word shingles, i.e. no fragment of it at all.
GPQA, German GPQA and AIME-DE data are not on the mini; tools/leak_check.py covers them on the mbp with the
shingle list built in S1. Each check skips (with the reason) when its source is absent.
"""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path

import pytest

SHINGLE_WORDS = 8


def normalise_words(text: str) -> list[str]:
    """NFKC, lower-case, keep letters/digits/whitespace (the leak check's normalisation, tools/shingles.py)."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace()).split()

EXP = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent


def _our_files() -> list[Path]:
    files = [p for p in (EXP / "tasks").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += sorted(TESTS.glob("test_tasks_*.py")) + [TESTS / "tasks_synthetic.py"]
    return files


def _words(files: list[Path]) -> list[list[str]]:
    return [normalise_words(p.read_text(encoding="utf-8", errors="ignore")) for p in files]


def _grams(words: list[str], k: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _rgb_dir() -> Path | None:
    for c in (os.environ.get("EXP036_DATA"), str(Path.home() / "models" / "exp036-mini" / "data")):
        if c and (Path(c).expanduser() / "RGB-src" / "data" / "en.json").is_file():
            return Path(c).expanduser() / "RGB-src"
    return None


def _flatten(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten(y)


def _rgb_rows(rgb_dir: Path):
    for name in ("en.json", "en_fact.json", "en_int.json"):
        for line in (rgb_dir / "data" / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def rgb_hits(files: list[Path], rgb_dir: Path) -> int:
    """Number of RGB fragments (8-word shingles, whole queries, instruction strings) found in `files`."""
    import yaml

    words = _words(files)
    ours = set()
    for w in words:
        ours |= _grams(w, SHINGLE_WORDS)
    joined = " " + " ".join(" ".join(w) for w in words) + " "
    hits = 0
    for row in _rgb_rows(rgb_dir):
        texts = [row["query"], *_flatten(row.get("answer")), *_flatten(row.get("positive")),
                 *_flatten(row.get("negative")), *_flatten(row.get("positive_wrong")), *_flatten(row.get("fakeanswer"))]
        for t in texts:
            hits += len(_grams(normalise_words(t), SHINGLE_WORDS) & ours)
        q = normalise_words(row["query"])
        if len(q) >= 5 and f" {' '.join(q)} " in joined:
            hits += 1
    instr = yaml.safe_load((rgb_dir / "config" / "instruction.yaml").read_text(encoding="utf-8"))["en"]
    for s in (instr["system"], instr["instruction"]):
        w = normalise_words(s)
        hits += len(_grams(w, SHINGLE_WORDS) & ours)
        if len(w) >= 3 and f" {' '.join(w)} " in joined:
            hits += 1
    return hits


def test_no_rgb_text_in_tasks():
    rgb_dir = _rgb_dir()
    if rgb_dir is None:
        pytest.skip("no local RGB copy (set EXP036_DATA)")
    assert rgb_hits(_our_files(), rgb_dir) == 0, "RGB text fragments found in tasks/ or the tasks tests"


def test_rgb_detector_positive_control(tmp_path):
    """The detector finds a planted query (written to a pytest temp dir only, never to the repo)."""
    rgb_dir = _rgb_dir()
    if rgb_dir is None:
        pytest.skip("no local RGB copy (set EXP036_DATA)")
    row = next(r for r in _rgb_rows(rgb_dir) if len(normalise_words(r["query"])) >= 8)
    planted = tmp_path / "planted.py"
    planted.write_text(f"# {row['query'].upper()}\n", encoding="utf-8")
    assert rgb_hits([planted], rgb_dir) >= 1


def test_no_fragment_of_the_upstream_gpqa_question():
    up = None
    for c in (os.environ.get("EXP036_EVALFW_SRC"), str(Path.home() / "models" / "exp036-mini" / "evalfw")):
        if c and (Path(c).expanduser() / "src/eval_framework/benchmarks/gpqa.py").is_file():
            up = Path(c).expanduser()
            break
    if up is None:
        pytest.skip("build-host only: no eval-framework v0.14.2 checkout (set EXP036_EVALFW_SRC)")
    import sys

    sys.path.insert(0, str(EXP / "tasks" / "vendored_evalfw"))
    import vendor

    value, _, _ = vendor.overlong_literal((up / "src/eval_framework/benchmarks/gpqa.py").read_text(encoding="utf-8"))
    target = _grams(normalise_words(value), 4)
    hits = sum(len(_grams(w, 4) & target) for w in _words(_our_files()))
    assert hits == 0
    long_tokens = [t for t in value.split() if len(t) >= 16]
    raw = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in _our_files())
    assert not any(t in raw for t in long_tokens)
