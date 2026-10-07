"""NOTICE names every Apache-derived or copied file (BUILD_SPEC §6 test_notice.py;
HYPOTHESIS "Sources & Rights": every Apache-derived or copied file is named in NOTICE).

The rule, checked over the experiment directory:
- every file whose header (first five lines) carries "SPDX-License-Identifier: Apache-2.0";
- every file under tasks/vendored_evalfw/ and runner/templates/;
- every tests/fixtures/*.jinja;
- tasks/prompts/aime_de.txt (an Apache-derived translation that is sent verbatim, so it can carry no header);
is named in NOTICE by its path relative to the experiment directory. Files with an Apache SPDX header also
carry a "Modified by Miktam" / "modified for MLX/numpy by Miktam" line. LICENSE-APACHE-2.0 sits next to
NOTICE, no NOTICE.addendum.* file is left over, and NOTICE names no file that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
NOTICE = EXP / "NOTICE"
SKIP_PARTS = {"__pycache__", ".pytest_cache", "results", "aborted", "evidence", "amendments"}
SPDX_APACHE = re.compile(r"^\s*(#|//|\{#|<!--)?\s*SPDX-License-Identifier:\s*Apache-2\.0\b")
MODIFIED = re.compile(r"(Modified by Miktam|modified for MLX/numpy by Miktam|modified by Miktam)", re.I)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jinja", ".sh", ".yaml", ".yml", ".toml", ""}


def _files() -> list[Path]:
    out = []
    for p in sorted(EXP.rglob("*")):
        rel = p.relative_to(EXP)
        if not p.is_file() or SKIP_PARTS & set(rel.parts) or p.name == ".DS_Store":
            continue
        out.append(p)
    return out


def _header(p: Path, n: int = 5) -> list[str]:
    if p.suffix not in TEXT_SUFFIXES:
        return []
    try:
        with p.open(encoding="utf-8") as f:
            return [next(f, "") for _ in range(n)]
    except UnicodeDecodeError:
        return []


def apache_header_files() -> list[str]:
    return [p.relative_to(EXP).as_posix() for p in _files() if any(SPDX_APACHE.match(ln) for ln in _header(p))]


def required_paths() -> list[str]:
    req = set(apache_header_files())
    for p in _files():
        rel = p.relative_to(EXP).as_posix()
        if rel.startswith(("tasks/vendored_evalfw/", "runner/templates/")):
            req.add(rel)
        if rel.startswith("tests/fixtures/") and rel.endswith(".jinja"):
            req.add(rel)
    req.add("tasks/prompts/aime_de.txt")
    return sorted(req)


def notice_text() -> str:
    return NOTICE.read_text(encoding="utf-8")


def test_notice_and_licence_present():
    assert NOTICE.is_file()
    lic = EXP / "LICENSE-APACHE-2.0"
    assert lic.is_file() and "Apache License" in lic.read_text(encoding="utf-8")[:400]
    assert (EXP / "tasks" / "vendored_evalfw" / "LICENSE").is_file()


def test_no_addendum_left():
    left = sorted(p.name for p in EXP.glob("NOTICE.addendum*"))
    assert not left, f"merge these into NOTICE and delete them: {left}"


def test_every_apache_derived_or_copied_file_is_named():
    text = notice_text()
    missing = [rel for rel in required_paths() if rel not in text]
    assert not missing, f"not named in NOTICE: {missing}"


def test_apache_header_files_carry_a_modified_by_line():
    bad = []
    for rel in apache_header_files():
        head = "".join(_header(EXP / rel, 6))
        if not MODIFIED.search(head):
            bad.append(rel)
    assert not bad, f"Apache SPDX header without a 'Modified by Miktam' line: {bad}"


def test_known_derived_files_are_detected():
    # Guards the detector itself: these files are Apache-derived by HYPOTHESIS "Sources & Rights".
    found = set(apache_header_files())
    for rel in ("port/kolibri1.py", "reference/kolibri_ref.py", "reference/mutants.py", "gate/port_mutants.py",
                "tests/tiny_checkpoint.py", "tests/test_routing.py", "runner/sampler.py", "scorers/mc.py",
                "scorers/aime.py", "tasks/vendored_evalfw/src/eval_framework/benchmarks/gpqa.py"):
        assert rel in found, rel


def test_named_paths_exist():
    # Every kit path NOTICE names (a token with a "/" and a file suffix under a kit directory) exists.
    text = notice_text()
    roots = ("port/", "reference/", "gate/", "runner/", "tasks/", "scorers/", "bench/", "analysis/", "tools/",
             "tests/", "env/")
    cands = set(re.findall(r"(?<![\w/.$-])((?:%s)[\w./-]+\.(?:py|json|jinja|txt|md|sha256))" %
                           "|".join(re.escape(r) for r in roots), text))
    generated = {"tools/withheld_shingles.sha256"}  # written on the run host at RUNBOOK step 7
    upstream = {"tests/checkpoints.py", "tests/test_kolibri1.py", "tests/kolibri1_chat_template.jinja"}  # vendor paths
    missing = sorted(c for c in cands if c not in generated | upstream and not (EXP / c).exists())
    assert not missing, f"NOTICE names files that do not exist: {missing}"
