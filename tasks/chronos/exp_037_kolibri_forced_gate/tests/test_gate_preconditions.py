# SPDX-License-Identifier: MIT
"""Gate preconditions P1, P2 and P4 (gate/preconditions.py; DESIGN §3.1, §9.1-§9.4;
§3.3 G1 item 10; decision F2).

P1, on a synthetic git repository with a bare remote and the 20 hash scopes:
  * a pre-registration that is filled, committed and pushed passes;
  * an unpushed pre-registration, a missing or unknown commit line and a
    branch without upstream are refused (c);
  * an amendment line for GATE_RULES_SHA256, THRESHOLDS_SHA256 or
    GATE_TEXT_SHA256 is refused (b), while one for another scope is not;
  * a direct edit of the hash table after the pre-registration is refused (e);
  * re-run admissibility and cycle counting (d): a run after exit 1, 4 or 5
    is a cycle, exit 3 never is; a re-run after a failed gate run needs a
    changed port, reference, runner, gate code or builds manifest.
P2 with monkeypatched versions (0.31.2 / 0.31.3 refused; 0.32.3 / 0.32.0
accepted) and with a missing or exp_036-schema version record (refused in
real mode; tiny mode records without judging).
P4 on the tiny real-layout build: a flipped shard byte, a wrong pinned record,
a missing or extra shard, an edited config.json and a stale port file are refused.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from gate import common
from gate import preconditions as pre
from tools import hash_tree as ht

EXP = Path(__file__).resolve().parents[1]
E36 = EXP.parent / "exp_036_kolibri_local_eval"
THRESHOLDS = json.loads((EXP / "gate" / "thresholds.json").read_text())
PINS = THRESHOLDS["P2"]
SIGN = "Miktam"
MAIL = "hello@localfirstai.eu"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ===================================================================== P1


def git(cwd, *args) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


STUB_FILES = {
    "assets.json": "{}\n",
    "tasks/vendored_evalfw/MANIFEST.json": "{}\n",
    "tasks/build_manifests.py": "# rule\n",
    "tasks/selection_rules.json": "{}\n",
    "tasks/gpqa.py": "# loader\n",
    "env/requirements-mbp.txt": "mlx==0.32.3\n",
    "port/kolibri1.py": "# port\n",
    "port/convert.py": "# convert\n",
    "reference/kolibri_ref.py": "# ref\n",
    "gate/run_gate.py": "# gate\n",
    "gate/calibration.json": "{}\n",
    "gate/controls.json": "{}\n",
    "gate/port_mutants.py": "# mutants\n",
    "gate/rules.py": "# rules\n",
    "gate/thresholds.json": '{"fix_cycles_max": 2}\n',
    "gate/texts/MANIFEST.json": "{}\n",
    "runner/run.py": "# run\n",
    "runner/plan_rules.json": "{}\n",
    "scorers/mc.py": "# mc\n",
    "analysis/stats.py": "# stats\n",
    "bench/fit.py": "# fit\n",
    "tools/hash_tree.py": "# tools\n",
    "tests/test_x.py": "def test_x():\n    pass\n",
    "env/exp037.settings.sh": "# settings\n",
    "env/setup.sh": "# setup\n",
    "env/versions.json": "{}\n",
}
# scope -> the stub file whose edit changes it (and every scope that file is in)
EDIT_FOR = {"gate_rules": "gate/rules.py", "thresholds": "gate/thresholds.json",
            "gate_text": "gate/texts/MANIFEST.json"}


def _template() -> str:
    rows: dict[str, list] = {}
    for s in ht.SCOPES:
        rows.setdefault(s.row, []).append(s)
    lines = ["# Experiment 037 (test copy)", "",
             "*Pre-registered: {{PREREG_UTC}} · Status: Pre-registered*", "",
             "## Fixed before any run (hashes)", "", "| Item | Value |", "|---|---|"]
    for row, scopes in rows.items():
        cell = "; ".join(f"`{s.path}` sha256 `{{{{{s.placeholder}}}}}`" for s in sorted(scopes, key=lambda x: x.ordinal))
        lines.append(f"| {row} | {cell} |")
    lines += ["", "## Sign-off", "", "- Signed off by: Andrei", ""]
    return "\n".join(lines)


def _write_tree(exp: Path) -> None:
    for rel, text in STUB_FILES.items():
        p = exp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    (exp / "HYPOTHESIS.md").write_text(_template(), encoding="utf-8")


def _commit(root, msg, push=True):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)
    if push:
        git(root, "push", "-q")


def _new_repo(tmp_path, with_remote=True):
    root = tmp_path / "lfa"
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", SIGN)
    git(root, "config", "user.email", MAIL)
    git(root, "config", "commit.gpgsign", "false")
    exp = root / "tasks" / "chronos" / "exp_037_kolibri_forced_gate"
    exp.mkdir(parents=True)
    (root / "README.md").write_text("lfa\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    if with_remote:
        remote = tmp_path / "remote.git"
        subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
        git(root, "remote", "add", "origin", str(remote))
        git(root, "push", "-q", "-u", "origin", "main")
    return root, exp


def _preregister(root, exp, push=True) -> str:
    """Fill the table, commit (the pre-registration), then append the commit
    line and commit again; push both when `push`."""
    _write_tree(exp)
    filled = ht.fill(exp / "HYPOTHESIS.md", exp)
    assert set(filled) == {s.placeholder for s in ht.SCOPES}
    _commit(root, "chronos/exp_037: pre-registration", push)
    sha = git(root, "rev-parse", "HEAD")
    with open(exp / "HYPOTHESIS.md", "a", encoding="utf-8") as f:
        f.write(f"\nPre-registration commit: {sha}\n")
    _commit(root, "chronos/exp_037: pre-registration commit line", push)
    return sha


@pytest.fixture
def repo(tmp_path):
    root, exp = _new_repo(tmp_path)
    sha = _preregister(root, exp)
    return root, exp, sha


def _p1(exp, tmp_path):
    return pre.p1(exp, builds=tmp_path / "no_builds")


def _amend(exp, number: int, lines: list[str]) -> None:
    with open(exp / "HYPOTHESIS.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Amendment {number} — gate fix (2026-10-10T00:00:0{number}Z)\n\n" + "\n".join(lines) + "\n")


def _amend_lines_for(exp, scopes) -> list[str]:
    return [f"hash_tree: {ht.BY_NAME[s].placeholder} = {ht.scope_value(ht.BY_NAME[s], exp)}" for s in scopes]


def test_hash_tree_has_the_20_scopes_and_the_frozen_three():
    assert len(ht.SCOPES) == pre.REQUIRED_SCOPES == 20
    assert {"tests", "env"} <= set(ht.BY_NAME)   # final review W15-02
    assert pre.FROZEN_SCOPES == ("gate_rules", "thresholds", "gate_text")
    s = ht.BY_NAME["gate_rules"]
    assert (s.placeholder, s.row, s.path) == ("GATE_RULES_SHA256", "Gate rules (frozen)", "gate")
    assert set(s.include) == {"calibration.json", "controls.json", "port_mutants.py", "rules.py", "thresholds.json"}


def test_p1_passes_on_a_pushed_pre_registration(repo, tmp_path):
    root, exp, sha = repo
    r = _p1(exp, tmp_path)
    assert r["ok"], r["reason"]
    v = r["values"]
    assert set(v) == {"a", "b", "c", "d", "e"} and all(p["ok"] for p in v.values())
    assert v["c"]["values"]["prereg_commit"] == sha and v["c"]["values"]["upstream"] == "origin/main"
    assert set(v["a"]["values"]["status"].values()) == {"match"} and len(v["a"]["values"]["status"]) == 20
    assert v["b"]["values"]["sources"] == {n: "pre-registration" for n in pre.FROZEN_SCOPES}
    assert v["d"]["values"]["run_counts"]["gate_runs"] == 0
    assert v["e"]["values"]["rows"] == len({s.row for s in ht.SCOPES}) + 2  # plus the header and rule rows
    assert pre.p1(exp, mode="tiny") == {"ok": True, "values": {"judged": False, "mode": "tiny"}, "reason": None}


def test_p1_refuses_an_unpushed_pre_registration(tmp_path):
    root, exp = _new_repo(tmp_path)
    sha = _preregister(root, exp, push=False)
    r = _p1(exp, tmp_path)
    assert not r["ok"]
    c = r["values"]["c"]
    assert not c["ok"] and c["values"]["ancestor_of_head"] and not c["values"]["ancestor_of_upstream"]
    assert "not pushed" in c["reason"] and "P1(c)" in r["reason"]
    assert r["values"]["a"]["ok"] and r["values"]["e"]["ok"]  # only the push is missing
    git(root, "push", "-q")
    assert _p1(exp, tmp_path)["ok"]
    assert sha in (exp / "HYPOTHESIS.md").read_text()


def test_p1_refuses_without_upstream_or_commit_line(tmp_path):
    root, exp = _new_repo(tmp_path, with_remote=False)
    _preregister(root, exp, push=False)
    r = _p1(exp, tmp_path)
    assert not r["values"]["c"]["ok"] and "no upstream" in r["values"]["c"]["reason"]

    root2, exp2 = _new_repo(tmp_path / "two")
    _write_tree(exp2)
    ht.fill(exp2 / "HYPOTHESIS.md", exp2)
    _commit(root2, "prereg without its commit line")
    r2 = _p1(exp2, tmp_path)
    assert not r2["values"]["c"]["ok"] and "no line 'Pre-registration commit" in r2["values"]["c"]["reason"]
    assert not r2["values"]["e"]["ok"]

    with open(exp2 / "HYPOTHESIS.md", "a", encoding="utf-8") as f:
        f.write("\nPre-registration commit: " + "0123456789abcdef" * 2 + "01234567\n")
    _commit(root2, "a commit line naming no commit")
    r3 = _p1(exp2, tmp_path)
    assert not r3["values"]["c"]["ok"] and "is not a commit" in r3["values"]["c"]["reason"]


@pytest.mark.parametrize("scope", ["gate_rules", "thresholds", "gate_text"])
@pytest.mark.parametrize("form", ["placeholder", "name"])
def test_p1_refuses_an_amendment_line_for_a_frozen_scope(repo, tmp_path, scope, form):
    root, exp, _ = repo
    f = exp / EDIT_FOR[scope]
    f.write_text(f.read_text() + " \n")
    changed = [s.name for s in ht.SCOPES if ht.scope_value(s, exp) != ht.registered_values(exp / "HYPOTHESIS.md")[s.name]["value"]]
    assert scope in changed
    lines = _amend_lines_for(exp, changed)
    if form == "name":
        lines = [ln.replace(ht.BY_NAME[s].placeholder, s) if s == scope else ln for ln, s in zip(lines, changed)]
    _amend(exp, 1, lines)
    _commit(root, f"amendment for {scope}")
    r = _p1(exp, tmp_path)
    v = r["values"]
    assert v["a"]["ok"], v["a"]["reason"]  # the amendment lines make every scope match ...
    assert not v["b"]["ok"] and not r["ok"]  # ... but no amendment may change a frozen scope
    assert any(a["scope"] == scope for a in v["b"]["values"]["frozen_amendment_lines"])
    assert v["b"]["values"]["sources"][scope] == "amendment 1"
    assert "frozen without amendment" in r["reason"]
    assert v["c"]["ok"] and v["d"]["ok"] and v["e"]["ok"]


def test_p1_accepts_an_amendment_for_another_scope_and_a_changed_rerun(repo, tmp_path):
    """A gate fix (runner change with its RUNNER_SHA256 line) after a failed gate run."""
    root, exp, _ = repo
    builds = tmp_path / "no_builds"
    rec = {"experiment": "exp_037", "mode": "real", "exit_code": 1, "utc": "20261010T000000Z",
           "code_hashes": pre.code_hashes(exp, builds)}
    common.write_new_json(exp / "results" / "gate" / "gate_20261010T000000Z.json", rec)
    r = _p1(exp, tmp_path)
    assert not r["ok"] and not r["values"]["d"]["ok"]
    assert "none of port, reference_tree, runner_tree, gate_code, builds_manifest has changed" in r["reason"]

    (exp / "runner" / "run.py").write_text("# run, fixed\n")
    _amend(exp, 1, _amend_lines_for(exp, ["runner"]))
    _commit(root, "gate fix: runner")
    r2 = _p1(exp, tmp_path)
    assert r2["ok"], r2["reason"]
    assert r2["values"]["d"]["values"]["changed"] == ["runner_tree"]
    assert r2["values"]["b"]["values"]["frozen_amendment_lines"] == []


def test_p1_refuses_a_direct_edit_of_the_hash_table(repo, tmp_path):
    root, exp, _ = repo
    hyp = exp / "HYPOTHESIS.md"
    old = ht.scope_value(ht.BY_NAME["runner"], exp)
    (exp / "runner" / "run.py").write_text("# run, edited\n")
    new = ht.scope_value(ht.BY_NAME["runner"], exp)
    hyp.write_text(hyp.read_text().replace(old, new))  # the table now "matches" the new runner
    r = _p1(exp, tmp_path)
    assert r["values"]["a"]["ok"]  # (a) alone would not see it
    e = r["values"]["e"]
    assert not e["ok"] and e["values"]["differs_worktree"] == ["Runner"] and e["values"]["differs_HEAD"] == []
    _commit(root, "edit the table")
    e2 = _p1(exp, tmp_path)["values"]["e"]
    assert not e2["ok"] and e2["values"]["differs_HEAD"] == ["Runner"]
    # a frozen row edited directly is refused the same way
    old_t = ht.scope_value(ht.BY_NAME["thresholds"], exp)
    (exp / "gate" / "thresholds.json").write_text('{"fix_cycles_max": 2, "x": 1}\n')
    hyp.write_text(hyp.read_text().replace(old_t, ht.scope_value(ht.BY_NAME["thresholds"], exp)))
    _commit(root, "edit a frozen row")
    e3 = _p1(exp, tmp_path)["values"]["e"]
    assert "Gate thresholds" in e3["values"]["differs_HEAD"]


# ------------------------------------------------------------------ P1(d): runs and cycles


def _rec(exit_code, mode="real", experiment="exp_037", **extra):
    return {"experiment": experiment, "mode": mode, "exit_code": exit_code, **extra}


@pytest.mark.parametrize("exits, runs, cycles, next_is_cycle", [
    ([], 0, 0, False),
    ([1], 1, 0, True),
    ([4], 1, 0, True),
    ([5], 1, 0, True),
    ([0], 1, 0, False),
    ([1, 4], 2, 1, True),
    ([5, 1, 4], 3, 2, True),
    ([1, 3, 4], 2, 1, True),        # exit 3 is not a gate run
    ([3, 3, 3], 0, 0, False),
    ([0, 1], 2, 0, True),           # a run after exit 0 is not a cycle
    ([1, 0], 2, 1, False),
])
def test_cycle_counting(exits, runs, cycles, next_is_cycle):
    c = pre.run_counts([_rec(e) for e in exits], fix_cycles_max=2)
    assert (c["gate_runs"], c["cycles_used"], c["next_run_is_cycle"]) == (runs, cycles, next_is_cycle)
    assert c["exit3_records"] == exits.count(3) and c["gate_runs_max"] == 3
    assert c["may_run_again"] == (runs < 3 and (not next_is_cycle or cycles < 2))


def test_tiny_dry_run_and_other_experiment_records_are_not_gate_runs():
    recs = [_rec(1, mode="tiny"), _rec(1, dry_run_promoted_from="x"), _rec(1, experiment="exp_036"),
            {"unreadable": True}, _rec(2)]
    assert pre.run_counts(recs)["gate_runs"] == 0
    assert pre.run_counts([("path", _rec(1)), ("path", _rec(4))])["cycles_used"] == 1  # (path, record) pairs


CODES = {"port": "p" * 64, "reference_tree": "r" * 64, "runner_tree": "u" * 64, "gate_code": "g" * 64,
         "builds_manifest": {"K8": "8" * 64, "K4": "4" * 64}}


def _write_runs(d: Path, specs) -> Path:
    for i, (exit_code, codes) in enumerate(specs):
        common.write_new_json(d / "gate" / f"gate_2026101{i}T000000Z.json", _rec(exit_code, code_hashes=codes))
    common.write_new_json(d / "gate" / "gate_20261019T000000Z_mutants.json", {"not": "a record"})
    return d


def test_p1d_rerun_admissibility(tmp_path):
    changed = {**CODES, "gate_code": "h" * 64}
    assert pre.p1d_rerun(tmp_path / "none", CODES)["ok"]
    assert pre.p1d_rerun(_write_runs(tmp_path / "a", [(0, CODES)]), CODES)["ok"]  # after exit 0
    same = pre.p1d_rerun(_write_runs(tmp_path / "b", [(1, CODES)]), CODES)
    assert not same["ok"] and "none of" in same["reason"]
    ok = pre.p1d_rerun(_write_runs(tmp_path / "c", [(5, CODES)]), changed)
    assert ok["ok"] and ok["values"]["changed"] == ["gate_code"]
    builds = pre.p1d_rerun(_write_runs(tmp_path / "d", [(4, CODES)]),
                           {**CODES, "builds_manifest": {"K8": "8" * 64, "K4": "5" * 64}})
    assert builds["ok"] and builds["values"]["changed"] == ["builds_manifest"]
    three = pre.p1d_rerun(_write_runs(tmp_path / "e", [(1, CODES), (1, changed), (4, CODES)]), changed)
    assert not three["ok"] and "3 gate runs exist (at most 3)" in three["reason"]
    assert three["values"]["run_counts"]["cycles_used"] == 2
    crashed = pre.p1d_rerun(_write_runs(tmp_path / "f", [(1, CODES), (3, CODES), (3, CODES)]), changed)
    assert crashed["ok"] and crashed["values"]["run_counts"]["exit3_records"] == 2
    one_cycle = pre.p1d_rerun(_write_runs(tmp_path / "g", [(1, CODES), (1, changed)]), CODES, fix_cycles_max=1)
    assert not one_cycle["ok"] and "1 fix cycles used (at most 1)" in one_cycle["reason"]
    # exp_036's record keys are read as aliases
    legacy = tmp_path / "h"
    common.write_new_json(legacy / "gate" / "gate_20261010T000000Z.json",
                          _rec(1, sha256={"port": CODES["port"], "reference_tree": CODES["reference_tree"],
                                          "gate_code": CODES["gate_code"],
                                          "converted_manifest": CODES["builds_manifest"]}))
    assert not pre.p1d_rerun(legacy, CODES)["ok"]
    assert pre.p1d_rerun(legacy, {**CODES, "port": "q" * 64})["ok"]
    nothing = tmp_path / "i"
    common.write_new_json(nothing / "gate" / "gate_20261010T000000Z.json", _rec(1))
    assert "recorded no code hashes" in pre.p1d_rerun(nothing, changed)["reason"]


def test_p1d_refuses_while_a_run_directory_has_no_record(tmp_path):
    """Final review W15-03: a run killed before phase 7 (SIGKILL, OOM, power cut) leaves
    results/gate/<UTC>/ without gate_<UTC>.json; P1(d) refuses until it is closed. The running
    gate's own directory (current_utc) is not an orphan."""
    d = tmp_path / "results"
    common.write_new_json(d / "gate" / "20261010T000000Z" / "phase0.json", {"phase": "0"})
    r = pre.p1d_rerun(d, CODES)
    assert not r["ok"] and r["values"]["orphaned_runs"] == ["20261010T000000Z"]
    assert "results/gate/20261010T000000Z" in r["reason"] and "--close-orphans" in r["reason"]
    assert pre.p1d_rerun(d, CODES, current_utc="20261010T000000Z")["ok"]
    (d / "gate" / "notes").mkdir()                                  # not a run directory
    common.write_new_json(d / "gate" / "gate_20261010T000000Z.json", _rec(3, code_hashes=CODES))
    ok = pre.p1d_rerun(d, CODES)
    assert ok["ok"] and ok["values"]["orphaned_runs"] == []


def test_p1_sees_an_orphan_in_the_real_results_directory(repo, tmp_path):
    root, exp, _ = repo
    (exp / "results" / "gate" / "20261011T000000Z").mkdir(parents=True)
    r = _p1(exp, tmp_path)
    assert not r["ok"] and not r["values"]["d"]["ok"] and "P1(d)" in r["reason"]
    assert pre.p1(exp, builds=tmp_path / "no_builds", current_utc="20261011T000000Z")["ok"]


def _blind(exit_code=3, codes=CODES, blind=True, utc="20261010T000000Z"):
    return _rec(exit_code, code_hashes=codes, blind_phase_reached=blind, utc=utc)


def test_p1d_needs_an_amendment_naming_a_blind_crash_before_a_changed_rerun(tmp_path):
    """§9.4 item 3 in code (W15-03): after an exit 3 that had passed P5, an unchanged repeat is
    free; a re-run with changed code needs a numbered amendment whose first line names the
    crashed run by its UTC (that line names the blind values; they are published from aborted/)."""
    changed = {**CODES, "port": "q" * 64}
    d = tmp_path / "results"
    common.write_new_json(d / "gate" / "gate_20261010T000000Z.json", _blind())
    hyp = tmp_path / "HYPOTHESIS.md"
    hyp.write_text("# H\n\n## Fixed before any run (hashes)\n\n| Item | Value |\n", encoding="utf-8")
    assert pre.p1d_rerun(d, CODES, hypothesis=hyp)["ok"]                    # an unchanged repeat
    r = pre.p1d_rerun(d, changed, hypothesis=hyp)
    assert not r["ok"] and "no amendment's first line names it" in r["reason"]
    assert r["values"]["blind_crashes"][0]["changed"] == ["port"]
    with open(hyp, "a", encoding="utf-8") as f:   # named, but not in the amendment's first line
        f.write("\n## Amendment 1 — gate fix (2026-10-10T08:00:00Z)\n\nA port fix.\n"
                "It follows the crashed run 20261010T000000Z.\n")
    assert not pre.p1d_rerun(d, changed, hypothesis=hyp)["ok"]
    with open(hyp, "a", encoding="utf-8") as f:
        f.write("\n## Amendment 2 — gate fix (2026-10-10T09:00:00Z)\n\n"
                "After the crashed run 20261010T000000Z (aborted/20261010T000000Z-gate), which computed "
                "G2 and G4-F32 T1-8 mean KL 1.2e-9: a port fix.\n\nhash_tree: PORT_SHA256 = " + "q" * 64 + "\n")
    ok = pre.p1d_rerun(d, changed, hypothesis=hyp)
    assert ok["ok"], ok["reason"]
    assert ok["values"]["blind_crashes"][0]["named_by"] == [2]
    # a crash before P5 needs no such line; a blind crash without code hashes always does
    e = tmp_path / "e"
    common.write_new_json(e / "gate" / "gate_20261010T000000Z.json", _blind(blind=False))
    assert pre.p1d_rerun(e, changed)["ok"]
    f_ = tmp_path / "f"
    common.write_new_json(f_ / "gate" / "gate_20261010T000000Z.json", _rec(3, blind_phase_reached=True,
                                                                            utc="20261010T000000Z"))
    assert "no code hashes recorded" in pre.p1d_rerun(f_, CODES)["reason"]


def test_amendment_first_lines():
    text = ("## Fixed before any run (hashes)\n\n| a |\n\n## Amendment 0 — sign-off (x)\n\n\nFirst line 0.\nSecond.\n"
            "\n## Amendment 3 — gate fix (y)\nFirst line 3.\n## Notes\n\nnot an amendment\n")
    assert pre.amendment_first_lines(text) == {0: "First line 0.", 3: "First line 3."}


def test_code_hashes_are_the_scopes_and_the_builds_manifests(tmp_path):
    c = pre.code_hashes(EXP, builds=tmp_path)
    assert set(c) == set(pre.CODE_HASH_KEYS)
    assert c["port"] == _sha(EXP / "port" / "kolibri1.py")
    assert c["runner_tree"] == ht.scope_value(ht.BY_NAME["runner"], EXP)
    assert c["builds_manifest"] == {"K8": None, "K4": None}


# ===================================================================== P2


def _good_observed():
    return {"macos": "27.0", "macos_build": "26A428",
            "packages": {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"},
            "architecture": "applegpu_g17s", "fp32": {"MLX_ENABLE_TF32": "0", "probe_rel_l2": 3e-7}}


def _write_records(d: Path, stamp="20261012T060000Z", version=None, preflight=None) -> Path:
    v = {"schema": "exp037 version record v1", "os": {"macos": "27.0", "build": "26A428"},
         "packages": {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0", "numpy": "2.5.3"},
         "mx_device_info": {"architecture": "applegpu_g17s"}}
    p = {"schema": "exp037 preflight v1", "mx_device_info": {"architecture": "applegpu_g17s"}}
    if version is not False:
        common.write_new_json(d / f"version_record_{stamp}.json", {**v, **(version or {})})
    if preflight is not False:
        common.write_new_json(d / f"preflight_{stamp}.json", {**p, **(preflight or {})})
    return d


def _patch_versions(monkeypatch, mlx_v, metal_v, mlx_lm_v):
    """The versions as the real modules report them (P2 reads mlx.__version__,
    the mlx-metal distribution and mlx_lm.__version__)."""
    import mlx.core as mx
    import mlx_lm

    real_version = importlib.metadata.version
    monkeypatch.setattr(mx, "__version__", mlx_v)
    monkeypatch.setattr(mlx_lm, "__version__", mlx_lm_v)
    monkeypatch.setattr(importlib.metadata, "version", lambda d: metal_v if d == "mlx-metal" else real_version(d))
    monkeypatch.setattr(pre, "_mac_ver", lambda: "27.0")
    monkeypatch.setattr(pre, "_os_build", lambda: "26A428")
    monkeypatch.setattr(pre, "_architecture", lambda: "applegpu_g17s")
    monkeypatch.setattr(pre, "_tf32", lambda: {"MLX_ENABLE_TF32": "0", "probe_rel_l2": 3e-7})


def test_p2_refuses_the_exp036_versions_and_accepts_the_f2_pins(monkeypatch, tmp_path):
    results = _write_records(tmp_path)
    _patch_versions(monkeypatch, "0.31.2", "0.31.2", "0.31.3")
    r = pre.p2(PINS, "real", results)
    assert not r["ok"]
    assert r["values"]["observed"]["packages"] == {"mlx": "0.31.2", "mlx-metal": "0.31.2", "mlx-lm": "0.31.3"}
    for s in ("mlx '0.31.2' != '0.32.3'", "mlx-metal '0.31.2' != '0.32.3'", "mlx-lm '0.31.3' != '0.32.0'"):
        assert s in r["reason"]
    _patch_versions(monkeypatch, "0.32.3", "0.32.3", "0.32.0")
    ok = pre.p2(PINS, "real", results)
    assert ok["ok"], ok["reason"]
    assert ok["values"]["version_record"]["file"] == "version_record_20261012T060000Z.json"
    assert ok["values"]["version_record"]["sha256"] == _sha(results / "version_record_20261012T060000Z.json")
    assert ok["values"]["preflight_record"]["sha256"] == _sha(results / "preflight_20261012T060000Z.json")
    _patch_versions(monkeypatch, "0.32.3", "0.32.3", "0.32.1")
    assert not pre.p2(PINS, "real", results)["ok"]


@pytest.mark.parametrize("change, needle", [
    ({"macos": "26.5.1"}, "macOS '26.5.1' != '27.0'"),
    ({"macos_build": "25F80"}, "macOS build '25F80' != '26A428'"),
    ({"architecture": "applegpu_g16s"}, "architecture 'applegpu_g16s' does not start with 'applegpu_g17'"),
    ({"fp32": {"MLX_ENABLE_TF32": "1", "error": "TF32 active"}}, "MLX_ENABLE_TF32 '1' != '0' (TF32 active)"),
])
def test_p2_refuses_each_process_pin(tmp_path, change, needle):
    r = pre.p2(PINS, "real", _write_records(tmp_path), observed={**_good_observed(), **change})
    assert not r["ok"] and needle in r["reason"]
    assert pre.p2(PINS, "real", _write_records(tmp_path / "ok"), observed=_good_observed())["ok"]


def test_p2_refuses_a_missing_or_exp036_version_record_in_real_mode(tmp_path):
    obs = _good_observed()
    missing = pre.p2(PINS, "real", _write_records(tmp_path / "a", version=False), obs)
    assert not missing["ok"] and "no results/version_record_<UTC>.json" in missing["reason"]
    old = _write_records(tmp_path / "b", version={"schema": "exp036 version record v1",
                                                  "packages": {"mlx": "0.31.2", "mlx-metal": "0.31.2",
                                                               "mlx-lm": "0.31.3"}})
    r = pre.p2(PINS, "real", old, obs)
    assert not r["ok"] and "schema 'exp036 version record v1'" in r["reason"]
    # the newest record binds: an exp_036 record after an exp_037 one is refused
    newer = _write_records(tmp_path / "c")
    common.write_new_json(newer / "version_record_20261013T000000Z.json", {"schema": "exp036 version record v1"})
    assert not pre.p2(PINS, "real", newer, obs)["ok"]
    pkg = pre.p2(PINS, "real", _write_records(tmp_path / "d", version={"packages": {
        "mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.31.3"}}), obs)
    assert not pkg["ok"] and "packages.mlx-lm '0.31.3' != '0.32.0'" in pkg["reason"]
    os_ = pre.p2(PINS, "real", _write_records(tmp_path / "e", version={"os": {"macos": "27.0", "build": "26A1"}}), obs)
    assert not os_["ok"] and "os.build '26A1'" in os_["reason"]
    nopf = pre.p2(PINS, "real", _write_records(tmp_path / "f", preflight=False), obs)
    assert not nopf["ok"] and "no results/preflight_<UTC>.json" in nopf["reason"]
    pf = pre.p2(PINS, "real", _write_records(tmp_path / "g", preflight={"mx_device_info": {"architecture": "applegpu_g16s"}}), obs)
    assert not pf["ok"] and "preflight record" in pf["reason"]
    pf36 = pre.p2(PINS, "real", _write_records(tmp_path / "h", preflight={"schema": "exp036 preflight v1"}), obs)
    assert not pf36["ok"]


def test_p2_tiny_mode_records_without_judging(monkeypatch, tmp_path):
    _patch_versions(monkeypatch, "0.31.2", "0.31.2", "0.31.3")
    r = pre.p2(PINS, "tiny", tmp_path / "nothing")
    assert r["ok"] and r["reason"] is None and r["values"]["judged"] is False
    assert r["values"]["observed"]["packages"]["mlx-lm"] == "0.31.3"


def test_p2_observes_this_process():
    import platform

    import mlx.core as mx
    import mlx_lm

    obs = pre.observe_environment()
    assert obs["packages"] == {"mlx": mx.__version__, "mlx-metal": importlib.metadata.version("mlx-metal"),
                               "mlx-lm": mlx_lm.__version__}
    assert obs["packages"] == PINS["packages"]  # the exp_037 venvs (decision F2)
    assert obs["macos"] == platform.mac_ver()[0]
    assert obs["architecture"] == mx.device_info()["architecture"]
    assert obs["macos_build"] == subprocess.run(["sysctl", "-n", "kern.osversion"], capture_output=True,
                                                text=True).stdout.strip()
    assert obs["fp32"]["MLX_ENABLE_TF32"] == "0" and obs["fp32"]["probe_rel_l2"] < 1e-5


def test_p2_pins_are_the_f2_pins():
    assert PINS["packages"] == {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}
    assert (PINS["macos"], PINS["macos_build"], PINS["architecture_prefix"], PINS["MLX_ENABLE_TF32"]) == \
        ("27.0", "26A428", "applegpu_g17", "0")
    req = (EXP / "env" / "requirements-mbp.txt").read_text().splitlines()
    for pkg, v in PINS["packages"].items():
        assert f"{pkg}=={v}" in req
    from tools import preflight, version_record

    assert PINS["version_record_schema"] == version_record.SCHEMA
    assert PINS["preflight_schema"] == preflight.SCHEMA


# ===================================================================== P4


@pytest.fixture
def tiny_builds(tmp_path, tiny_real_val):
    """Copies of the tiny validation build's K8 and K4 under builds/, their
    convert records copied out as the "exp_036 records" (pinned by sha256),
    and a snapshot of the port they were converted with."""
    exp = tmp_path / "exp"
    exp.mkdir()
    builds = tmp_path / "builds"
    block = {"builds": {}, "shards": "model*.safetensors", "config": "config.json"}
    for arm, bits in (("K8", 8), ("K4", 4)):
        src = tiny_real_val.build(bits)
        assert src.name == common.ARM_DIRS[arm]
        shutil.copytree(src, builds / src.name)
        rec = tmp_path / "records" / f"convert_{bits}bit.json"
        rec.parent.mkdir(exist_ok=True)
        shutil.copyfile(src / common.CONVERT_RECORD, rec)
        block["builds"][arm] = {"dir": src.name, "convert_record": f"../records/{rec.name}",
                                "convert_record_sha256": _sha(rec)}
    port = tmp_path / "kolibri1_snapshot.py"
    shutil.copyfile(builds / common.ARM_DIRS["K8"] / "kolibri1.py", port)
    return exp, builds, block, port


def _p4(t, **kw):
    exp, builds, block, port = t
    return pre.p4(kw.pop("block", block), builds=builds, exp_dir=exp, port_file=kw.pop("port_file", port), **kw)


def _shards(d: Path) -> list[Path]:
    return sorted(d.glob("model*.safetensors"))


def test_p4_passes_on_the_recorded_build(tiny_builds):
    r = _p4(tiny_builds)
    assert r["ok"], r["reason"]
    for arm in ("K8", "K4"):
        v = r["values"]["arms"][arm]["values"]
        assert v["shards"] >= 1 and v["missing"] == v["extra"] == v["shard_mismatches"] == []
        assert v["port_sha256"] == _sha(tiny_builds[3]) and v["record_sha256"] == v["record_sha256_pinned"]


def test_p4_refuses_one_flipped_shard_byte(tiny_builds):
    shard = _shards(tiny_builds[1] / common.ARM_DIRS["K4"])[-1]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0x01
    shard.write_bytes(bytes(data))
    r = _p4(tiny_builds)
    assert not r["ok"]
    assert r["values"]["arms"]["K8"]["ok"]
    k4 = r["values"]["arms"]["K4"]
    assert k4["values"]["shard_mismatches"] == [shard.name]
    assert "P4(K4)" in r["reason"] and shard.name in r["reason"]
    tiny = _p4(tiny_builds, mode="tiny")  # the build's own record lists the original bytes too
    assert not tiny["ok"] and tiny["values"]["mode"] == "tiny"


def test_p4_checks_the_pinned_record_first(tiny_builds):
    exp, builds, block, port = tiny_builds
    bad = json.loads(json.dumps(block))
    bad["builds"]["K8"]["convert_record_sha256"] = "0" * 64
    r = _p4(tiny_builds, block=bad)
    v = r["values"]["arms"]["K8"]["values"]
    assert not r["ok"] and "pinned" in r["reason"] and "shards" not in v  # nothing hashed after the refusal
    gone = json.loads(json.dumps(block))
    gone["builds"]["K4"]["convert_record"] = "../records/missing.json"
    assert "not found" in _p4(tiny_builds, block=gone)["reason"]


def test_p4_refuses_missing_and_extra_shards_and_an_edited_config(tiny_builds):
    exp, builds, block, port = tiny_builds
    k8 = builds / common.ARM_DIRS["K8"]
    first = _shards(k8)[0]
    shutil.copyfile(first, k8 / "model-09999-of-09999.safetensors")
    r = _p4(tiny_builds)
    assert not r["ok"] and r["values"]["arms"]["K8"]["values"]["extra"] == ["model-09999-of-09999.safetensors"]
    (k8 / "model-09999-of-09999.safetensors").unlink()
    moved = first.with_name(first.name + ".bak")
    first.rename(moved)
    r2 = _p4(tiny_builds)
    assert not r2["ok"] and r2["values"]["arms"]["K8"]["values"]["missing"] == [first.name]
    moved.rename(first)
    assert _p4(tiny_builds)["ok"]
    cfg = builds / common.ARM_DIRS["K4"] / "config.json"
    cfg.write_text(cfg.read_text().replace("{", "{ ", 1))
    r3 = _p4(tiny_builds)
    assert not r3["ok"] and "config.json sha256" in r3["reason"]


def test_p4_refuses_a_stale_port_file(tiny_builds, tmp_path):
    other = tmp_path / "kolibri1_other.py"
    other.write_text(tiny_builds[3].read_text() + "\n# a later port\n")
    r = _p4(tiny_builds, port_file=other)
    assert not r["ok"] and "stale port file" in r["reason"]
    assert all(r["values"]["arms"][a]["values"]["port_sha256"] is None for a in ("K8", "K4"))


def test_p4_tiny_mode_reads_the_builds_own_records(tiny_builds):
    r = _p4(tiny_builds, mode="tiny")
    assert r["ok"], r["reason"]
    assert all(r["values"]["arms"][a]["values"]["record_sha256_pinned"] is None for a in ("K8", "K4"))


def test_p4_real_block_pins_exp036s_convert_records():
    block = THRESHOLDS["P4"]
    assert {a: s["dir"] for a, s in block["builds"].items()} == common.ARM_DIRS
    pinned = {"K8": "bd84bf3b7c4a3db891c02e41f86f618c1d93301581146fca8e598e04313e0e9d",
              "K4": "e1ec02c764667c7873f49dedd00c353db804409c3592f913444c8599b5d7f8a2"}  # DESIGN §2.6
    for arm, spec in block["builds"].items():
        rec = (EXP / spec["convert_record"]).resolve()
        assert rec.parent == (E36 / "results" / "convert").resolve()
        assert spec["convert_record_sha256"] == pinned[arm] == _sha(rec)
        files = json.loads(rec.read_text())["files"]
        shards = {"K8": 17, "K4": 9}[arm]  # exp_036's 8-bit and 4-bit builds
        assert sum(1 for n in files if n.startswith("model-")) == shards and "config.json" in files
