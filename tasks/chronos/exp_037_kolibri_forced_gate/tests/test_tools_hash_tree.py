"""tools/hash_tree.py (exp_036 BUILD_SPEC §5.9; HYPOTHESIS "Fixed before any run"; exp_037 DESIGN §9.2).

Fill / check round trip on a temporary copy, stamp format, idempotence, the
include / exclude sets of every scope, amendments, and that SCOPES matches the
real HYPOTHESIS.md table row by row. exp_037: the gate_rules scope (its exact
definition, its five files, the 17 inherited scopes unchanged), the scopes no
amendment may change, and a check without a HYPOTHESIS.md; the final review's
tests and env scopes (W15-02).
"""

from __future__ import annotations

import ast
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import hash_tree as ht

EXP = Path(__file__).resolve().parents[1]
E36 = EXP.parent / "exp_036_kolibri_local_eval"   # read-only

# DESIGN §12 W10: the assertions on the real HYPOTHESIS.md table are skipped until W12's hand-off note
# (HYPOTHESIS.md written with its hash table, 20 rows since W15-02); switched on by the final review (W15-01).
W12_HANDOFF = True
W12_PENDING = "exp_037 HYPOTHESIS.md table: switched on after W12's hand-off note (DESIGN §12 W10)"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hypothesis_template() -> str:
    """A small HYPOTHESIS.md with the real table's row labels and placeholders."""
    rows = {}
    for s in ht.SCOPES:
        rows.setdefault(s.row, []).append(s)
    lines = [
        "# Experiment 037 — test copy",
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
        "env/requirements-mbp.txt": "mlx==0.32.3\n",
        "port/kolibri1.py": "# port\n",
        "port/convert.py": "# convert\n",
        "reference/kolibri_ref.py": "# ref\n",
        "reference/README.md": "ref\n",
        "gate/run_gate.py": "# gate\n",
        "gate/checks/g2_layers.py": "# measures only\n",
        "gate/thresholds.json": "{}\n",
        "gate/rules.py": "# decisions\n",
        "gate/controls.json": "{}\n",
        "gate/calibration.json": "{}\n",
        "gate/port_mutants.py": "# mutants\n",
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
        "tests/conftest.py": "# skip policy\n",
        "tests/test_x.py": "def test_x():\n    pass\n",
        "env/exp037.settings.sh": "# settings\n",
        "env/setup.sh": "# setup\n",
        "env/versions.json": "{}\n",
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


@pytest.mark.skipif(not W12_HANDOFF, reason=W12_PENDING)
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


@pytest.mark.skipif(not W12_HANDOFF, reason=W12_PENDING)
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


# ---------------------------------------------------------------- exp_037: the gate_rules scope (DESIGN §9.2)

def _exp036_scopes() -> tuple:
    """exp_036's SCOPES, read from its tools/hash_tree.py source (never imported: E36 is read-only)."""
    src = (E36 / "tools" / "hash_tree.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "SCOPES":
            return eval(ast.get_source_segment(src, node.value), {"Scope": ht.Scope})  # noqa: S307 (own source)
    raise AssertionError("SCOPES not found in exp_036's hash_tree.py")


EXP037_SCOPES = ("gate_rules", "tests", "env")


def test_gate_rules_scope_is_the_specified_one_and_the_rest_are_exp036s():
    assert ht.BY_NAME["gate_rules"] == ht.Scope(
        "gate_rules", "GATE_RULES_SHA256", "Gate rules (frozen)", 0, "tree", "gate",
        include=("calibration.json", "controls.json", "port_mutants.py", "rules.py", "thresholds.json"))
    # Final review W15-02: G1's suite and the environment scripts are hashed too (amendable, like gate code).
    assert ht.BY_NAME["tests"] == ht.Scope("tests", "TESTS_SHA256", "Tests", 0, "tree", "tests")
    assert ht.BY_NAME["env"] == ht.Scope("env", "ENV_SHA256", "Environment scripts", 0, "tree", "env",
                                         include=("exp037.settings.sh", "setup.sh", "versions.json"))
    assert len(ht.SCOPES) == 20 and len(ht.BY_PLACEHOLDER) == 20
    inherited = _exp036_scopes()
    assert len(inherited) == 17
    assert [s for s in ht.SCOPES if s.name not in EXP037_SCOPES] == list(inherited)
    assert ht.NO_AMENDMENT_SCOPES == ("gate_rules", "thresholds", "gate_text")
    assert not set(EXP037_SCOPES[1:]) & set(ht.NO_AMENDMENT_SCOPES)   # a gate fix carries a test: amendable


def test_tests_and_env_scopes_see_every_test_and_only_their_env_files(tree):
    """W15-02: editing, skipping or adding a test, or editing conftest.py, changes TESTS_SHA256; the env
    scope reads exactly its three files (exp_036's env file, or a local override, is never read)."""
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)

    def bad():
        return {k for k, v in ht.check(hyp, tree, use_git=False)["scopes"].items() if v["status"] != "match"}

    (tree / "tests" / "test_x.py").write_text("import pytest\n\n\ndef test_x():\n    pytest.skip('build-host only: x')\n")
    assert bad() == {"tests"}
    (tree / "tests" / "test_x.py").write_text("def test_x():\n    pass\n")
    (tree / "tests" / "conftest.py").write_text("# skip policy relaxed\n")
    assert bad() == {"tests"}
    (tree / "tests" / "conftest.py").write_text("# skip policy\n")
    (tree / "tests" / "fixtures").mkdir()
    (tree / "tests" / "fixtures" / "new.json").write_text("{}\n")
    assert bad() == {"tests"}
    (tree / "tests" / "fixtures" / "new.json").unlink()
    assert bad() == set()
    (tree / "env" / "exp037.settings.sh").write_text('export PY="elsewhere"\n')
    assert bad() == {"env"}
    (tree / "env" / "exp037.settings.sh").write_text("# settings\n")
    (tree / "env" / "requirements-mbp.txt").write_text("mlx==0.32.4\n")
    assert bad() == {"requirements"}              # the Runtime row's file, not the env scope's
    (tree / "env" / "requirements-mbp.txt").write_text("mlx==0.32.3\n")
    secret = tree / "env" / "other.local.env"
    secret.write_text("TOKEN=x\n")
    secret.chmod(0)                                # unreadable: hashing the scope must not open it
    try:
        assert bad() == set()
    finally:
        secret.chmod(0o600)


def test_gate_rules_covers_exactly_its_five_files(tree):
    def values():
        return {n: ht.scope_value(ht.BY_NAME[n], tree, use_git=False) for n in ("gate_rules", "gate", "thresholds")}

    v0 = values()
    for rel in ("rules.py", "controls.json", "calibration.json", "port_mutants.py"):
        (tree / "gate" / rel).write_text(f"# changed {rel}\n")
        v1 = values()
        assert v1["gate_rules"] != v0["gate_rules"] and v1["gate"] != v0["gate"], rel
        assert v1["thresholds"] == v0["thresholds"], rel
        v0 = v1
    (tree / "gate" / "thresholds.json").write_text('{"version": "changed"}\n')
    v1 = values()
    assert v1["gate_rules"] != v0["gate_rules"] and v1["thresholds"] != v0["thresholds"]
    assert v1["gate"] == v0["gate"]                       # the gate code scope excludes thresholds.json
    v0 = v1
    (tree / "gate" / "checks" / "g2_layers.py").write_text("# measurement code changed\n")
    (tree / "gate" / "checks" / "new_check.py").write_text("# new\n")
    v1 = values()
    assert v1["gate"] != v0["gate"]
    assert v1["gate_rules"] == v0["gate_rules"] and v1["thresholds"] == v0["thresholds"]


def test_gate_rules_needs_all_five_files_and_fill_stays_all_or_nothing(tree):
    hyp = tree / "HYPOTHESIS.md"
    before = hyp.read_bytes()
    (tree / "gate" / "calibration.json").unlink()
    with pytest.raises(ht.MissingScope, match="calibration.json"):
        ht.scope_value(ht.BY_NAME["gate_rules"], tree, use_git=False)
    with pytest.raises(ht.MissingScope, match="GATE_RULES_SHA256"):
        ht.fill(hyp, tree, use_git=False)
    assert hyp.read_bytes() == before


def test_the_frozen_without_amendment_scopes_print_no_amend_line(tree, capsys):
    for name in ht.NO_AMENDMENT_SCOPES:
        assert ht.main(["--amend-line", name, "--root", str(tree)]) == 1
        out = capsys.readouterr()
        assert out.out == "" and "frozen without amendment" in out.err
    assert ht.main(["--tree", "gate_rules", "--root", str(tree)]) == 0      # its value is printed as usual
    assert re.fullmatch(r"[0-9a-f]{64}", capsys.readouterr().out.strip())


def test_an_amendment_line_naming_gate_rules_is_visible_to_p1b(tree):
    """hash_tree reports the source of each registered value; P1(b) (gate/preconditions.py) refuses
    GATE_RULES_SHA256, THRESHOLDS_SHA256 and GATE_TEXT_SHA256 unless the source is the pre-registration."""
    hyp = tree / "HYPOTHESIS.md"
    ht.fill(hyp, tree, use_git=False)
    reg = ht.registered_values(hyp)
    assert all(reg[n]["source"] == "pre-registration" for n in ht.NO_AMENDMENT_SCOPES)
    hyp.write_text(hyp.read_text() + "\n## Amendment 1 — gate fix (2026-10-08T10:00:00Z)\n\n"
                   f"hash_tree: GATE_RULES_SHA256 = {'b' * 64}\n")
    reg = ht.registered_values(hyp)
    assert reg["gate_rules"] == {"value": "b" * 64, "source": "amendment 1"}
    assert reg["thresholds"]["source"] == reg["gate_text"]["source"] == "pre-registration"


def test_check_without_a_hypothesis_is_unfilled_not_a_crash(tree, capsys):
    res = ht.check(tree / "NOPE.md", tree, use_git=False)
    assert not res["ok"]
    assert {v["status"] for v in res["scopes"].values()} == {"unfilled"}
    assert all(v["source"] == "NOPE.md not found" and v["current"] for v in res["scopes"].values())
    (tree / "HYPOTHESIS.md").unlink()
    assert ht.main(["--check", "--root", str(tree)]) == 1
    assert "HYPOTHESIS.md not found" in capsys.readouterr().out



def test_a_missing_scope_is_named_relative_to_the_experiment(tree):
    """A version record carries these messages (tools/version_record.py): no absolute path in them."""
    (tree / "gate" / "rules.py").unlink()
    with pytest.raises(ht.MissingScope) as e:
        ht.scope_value(ht.BY_NAME["gate_rules"], tree, use_git=False)
    assert str(e.value) == "gate: missing ['rules.py']"
    res = ht.check(tree / "HYPOTHESIS.md", tree, use_git=False)
    assert res["scopes"]["gate_rules"]["status"] == "missing" and str(tree) not in res["scopes"]["gate_rules"]["error"]
