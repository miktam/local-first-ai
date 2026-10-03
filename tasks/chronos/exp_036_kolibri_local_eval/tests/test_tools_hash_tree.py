"""tools/hash_tree.py (BUILD_SPEC §5.9; HYPOTHESIS "Fixed before any run").

Fill / check round trip on a temporary copy, stamp format, idempotence, the
include / exclude sets of every scope, amendments, and that SCOPES matches the
real HYPOTHESIS.md table row by row.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import hash_tree as ht

EXP = Path(__file__).resolve().parents[1]


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hypothesis_template() -> str:
    """A small HYPOTHESIS.md with the real table's row labels and placeholders."""
    rows = {}
    for s in ht.SCOPES:
        rows.setdefault(s.row, []).append(s)
    lines = [
        "# Experiment 036 — test copy",
        "",
        "*Pre-registered: {{PREREG_UTC}} · Status: pre-registered, awaiting Andrei's sign-off; no scored run*",
        "",
        "## Fixed before any run (hashes)",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Vendor report | sha256 `" + "0" * 64 + "` (not committed) |",
    ]
    for row, scopes in rows.items():
        cell = "; ".join(f"`{s.path}` sha256 `{{{{{s.placeholder}}}}}`" for s in sorted(scopes, key=lambda x: x.ordinal))
        lines.append(f"| {row} | {cell} |")
    lines += ["", "## Pre-registration decisions", "", "- none", ""]
    return "\n".join(lines)


def _make_tree(root: Path) -> None:
    files = {
        "assets.json": "{}\n",
        "tasks/vendored_evalfw/MANIFEST.json": "{}\n",
        "tasks/vendored_evalfw/gpqa.py": "x = 1\n",
        "tasks/build_manifests.py": "# rule\n",
        "tasks/selection_rules.json": "{}\n",
        "tasks/gpqa.py": "# loader\n",
        "tasks/manifests/gpqa_en.json": "[]\n",
        "env/requirements-mbp.txt": "mlx==0.31.2\n",
        "port/kolibri1.py": "# port\n",
        "port/convert.py": "# convert\n",
        "reference/kolibri_ref.py": "# ref\n",
        "reference/README.md": "ref\n",
        "gate/run_gate.py": "# gate\n",
        "gate/thresholds.json": "{}\n",
        "gate/texts/MANIFEST.json": "{}\n",
        "gate/texts/T1_x.txt": "text\n",
        "runner/run.py": "# run\n",
        "runner/plan_rules.json": "{}\n",
        "scorers/mc.py": "# mc\n",
        "analysis/stats.py": "# stats\n",
        "analysis/exploratory.py": "# tier 2\n",
        "analysis/tables.py": "# tier 2\n",
        "bench/fit.py": "# fit\n",
        "bench/ladder.py": "# tier 2\n",
        "tools/hash_tree.py": "# tools\n",
        "tools/withheld_shingles.sha256": "abcdefabcdef\n",
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (root / "HYPOTHESIS.md").write_text(_hypothesis_template(), encoding="utf-8")


@pytest.fixture
def tree(tmp_path) -> Path:
    root = tmp_path / "exp"
    _make_tree(root)
    return root


def test_tree_rule_is_sorted_relpath_tab_sha256(tmp_path):
    d = tmp_path / "t"
    (d / "b").mkdir(parents=True)
    (d / "b" / "z.txt").write_bytes(b"z")
    (d / "a.txt").write_bytes(b"a")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.pyc").write_bytes(b"junk")
    (d / ".DS_Store").write_bytes(b"junk")
    expected = _sha(f"a.txt\t{_sha(b'a')}\nb/z.txt\t{_sha(b'z')}\n".encode())
    assert ht.tree_sha256(d, use_git=False) == expected
    assert ht.manifest_sha256({"b/z.txt": _sha(b"z"), "a.txt": _sha(b"a")}) == expected


def test_fill_check_round_trip(tree):
    hyp = tree / "HYPOTHESIS.md"
    filled = ht.fill(hyp, tree, use_git=False)
    assert set(filled) == {s.placeholder for s in ht.SCOPES}
    text = hyp.read_text()
    assert "{{PREREG_UTC}}" in text
    assert re.findall(r"\{\{([A-Z0-9_]+)\}\}", text) == ["PREREG_UTC"]
    res = ht.check(hyp, tree, use_git=False)
    assert res["ok"], {k: v["status"] for k, v in res["scopes"].items()}
    for s in ht.SCOPES:
        assert res["scopes"][s.name]["source"] == "pre-registration"


def test_fill_is_idempotent_and_stamp_format(tree):
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    once = hyp.read_bytes()
    assert ht.fill(hyp, tree, use_git=False) == {}
    assert hyp.read_bytes() == once
    value = ht.stamp(hyp)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value)
    assert "{{" not in hyp.read_text()
    assert ht.stamp(hyp) is None
    assert ht.check(hyp, tree, use_git=False)["ok"]


def test_fill_is_all_or_nothing(tree):
    hyp = tree / "HYPOTHESIS.md"
    before = hyp.read_bytes()
    (tree / "gate" / "thresholds.json").unlink()
    with pytest.raises(ht.MissingScope):
        ht.fill(hyp, tree, use_git=False)
    assert hyp.read_bytes() == before


def test_unknown_placeholder_is_refused(tree):
    hyp = tree / "HYPOTHESIS.md"
    hyp.write_text(hyp.read_text() + "\n{{SOMETHING_ELSE}}\n")
    with pytest.raises(ValueError):
        ht.fill(hyp, tree, use_git=False)


def test_check_reports_the_changed_scope_only(tree):
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    (tree / "scorers" / "mc.py").write_text("# changed\n")
    res = ht.check(hyp, tree, use_git=False)
    assert not res["ok"]
    bad = {k for k, v in res["scopes"].items() if v["status"] != "match"}
    assert bad == {"scorers"}


@pytest.mark.parametrize("rel", [
    "analysis/exploratory.py", "analysis/tables.py", "bench/ladder.py", "tools/withheld_shingles.sha256",
    "tasks/manifests/gpqa_en.json", "gate/texts/T1_x.txt", "tasks/manifests/new.json",
])
def test_excluded_files_do_not_change_any_scope(tree, rel):
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    p = tree / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("changed after pre-registration\n")
    assert ht.check(hyp, tree, use_git=False)["ok"]


def test_manifest_rule_covers_only_its_two_files(tree):
    v0 = ht.scope_value(ht.BY_NAME["manifest_rule"], tree, use_git=False)
    (tree / "tasks" / "gpqa.py").write_text("# other loader change\n")
    assert ht.scope_value(ht.BY_NAME["manifest_rule"], tree, use_git=False) == v0
    (tree / "tasks" / "selection_rules.json").write_text('{"seed": 36}\n')
    assert ht.scope_value(ht.BY_NAME["manifest_rule"], tree, use_git=False) != v0


def test_amendment_line_replaces_the_registered_value(tree):
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    (tree / "analysis" / "stats.py").write_text("# fixed by amendment\n")
    assert not ht.check(hyp, tree, use_git=False)["ok"]
    new = ht.scope_value(ht.BY_NAME["analysis"], tree, use_git=False)
    hyp.write_text(hyp.read_text() + f"\n## Amendment 0 — sign-off choices (2026-10-04T08:00:00Z)\n\n"
                   f"hash_tree: ANALYSIS_SHA256 = {new}\n")
    res = ht.check(hyp, tree, use_git=False)
    assert res["ok"]
    assert res["scopes"]["analysis"]["source"] == "amendment 0"


def test_hash_lines_outside_amendments_are_ignored(tree):
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    (tree / "analysis" / "stats.py").write_text("# changed\n")
    new = ht.scope_value(ht.BY_NAME["analysis"], tree, use_git=False)
    hyp.write_text(hyp.read_text() + f"\n## Notes\n\nhash_tree: ANALYSIS_SHA256 = {new}\n")
    assert not ht.check(hyp, tree, use_git=False)["ok"]


def test_git_ignored_files_are_left_out(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    (d / ".gitignore").write_text("*.npy\nwork/\n")
    sub = d / "tools"
    sub.mkdir()
    (sub / "a.py").write_text("a\n")
    v0 = ht.tree_sha256(sub)
    (sub / "big.npy").write_bytes(b"\x00" * 10)
    (sub / "work").mkdir()
    (sub / "work" / "x.txt").write_text("x\n")
    assert ht.tree_sha256(sub) == v0
    (sub / "b.py").write_text("b\n")   # untracked but not ignored: counts
    assert ht.tree_sha256(sub) != v0


def test_cli_tree_and_amend_line(tree, capsys):
    assert ht.main(["--tree", "tools", "--root", str(tree)]) == 0
    out = capsys.readouterr().out.strip()
    assert re.fullmatch(r"[0-9a-f]{64}", out)
    assert ht.main(["--amend-line", "analysis", "--root", str(tree)]) == 0
    assert re.fullmatch(r"hash_tree: ANALYSIS_SHA256 = [0-9a-f]{64}", capsys.readouterr().out.strip())


def test_scopes_match_the_real_hypothesis_table():
    """Every scope's row exists in HYPOTHESIS.md and holds its placeholder, or
    a 64-hex value at its ordinal once filled."""
    text = (EXP / "HYPOTHESIS.md").read_text(encoding="utf-8")
    rows = ht._table_rows(text)
    for s in ht.SCOPES:
        assert s.row in rows, f"row {s.row!r} not in the HYPOTHESIS table"
        line = rows[s.row]
        if "{{" + s.placeholder + "}}" in line:
            continue
        assert len(ht.HEX64.findall(line)) > s.ordinal, f"{s.name}: no value at position {s.ordinal}"
    placeholders = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    assert placeholders <= {s.placeholder for s in ht.SCOPES} | {ht.STAMP_PLACEHOLDER}


def test_round_trip_on_a_copy_of_the_real_hypothesis(tmp_path):
    """fill + stamp on a copy of the real HYPOTHESIS.md over a synthetic tree:
    no double-brace placeholder is left and check passes.

    Independent of where the run is (review fix 2026-10-03): once the real file
    is filled, stamped and amended, the copy is first turned back into its
    template by hash_tree.unfill(), so the test never skips (RUNBOOK step 3 and
    gate G1 run it with EXP036_REQUIRE_ALL=run-host after the fill)."""
    root = tmp_path / "exp"
    _make_tree(root)
    hyp = root / "HYPOTHESIS.md"
    hyp.write_text(ht.unfill((EXP / "HYPOTHESIS.md").read_text(encoding="utf-8")), encoding="utf-8")
    text = hyp.read_text()
    assert {s.placeholder for s in ht.SCOPES} <= set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    assert "{{PREREG_UTC}}" in text
    ht.fill(hyp, root, use_git=False)
    ht.stamp(hyp)
    assert "{{" not in hyp.read_text()
    assert ht.check(hyp, root, use_git=False)["ok"]


def test_unfill_inverts_fill_stamp_and_amendment_lines(tmp_path):
    """unfill() on a filled, stamped and amended copy gives back the template exactly."""
    root = tmp_path / "exp"
    _make_tree(root)
    hyp = root / "HYPOTHESIS.md"
    template = _hypothesis_template()
    hyp.write_text(template)
    ht.fill(hyp, root, use_git=False)
    ht.stamp(hyp)
    filled = hyp.read_text()
    assert "{{" not in filled
    assert ht.unfill(filled) == template
    amended = filled + "\n## Amendment 3 — gate fix (2026-10-05T10:00:00Z)\n\nhash_tree: ANALYSIS_SHA256 = " + "a" * 64 + "\n"
    back = ht.unfill(amended)
    assert back.startswith(template)
    assert not ht.AMEND_LINE.search(back)
