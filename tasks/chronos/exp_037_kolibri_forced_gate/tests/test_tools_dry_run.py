"""tools/dry_run.py helpers that need no model (review fix 2026-10-03): a copy of a kit that is already in its run
is reset to its pre-registration state before the dry run fills, scales and builds its own.

exp_037 (DESIGN §12 W10): the copy sits at tasks/chronos/exp_037_kolibri_forced_gate and commits as
"chronos/exp_037:"; the world's Kolibri arms resolve under $EXP037_BUILDS; the stages follow exp_037's RUNBOOK
(refresh instead of convert); the peers record carries peer_check.SCHEMA; the gate's tiny PASS on the validation
set is promoted only with every control caught, rebound to the refreshed clones; and the S1-hours inputs include
the refresh records. The end-to-end dry run itself is W14's.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools import dry_run, hash_tree as ht

EXP = Path(__file__).resolve().parents[1]


def _template() -> str:
    """A HYPOTHESIS.md with every scope's placeholder in the "Fixed before any run" table (the real file's
    layout; exp_037's own is W12's)."""
    rows = {}
    for s in ht.SCOPES:
        rows.setdefault(s.row, []).append(s)
    lines = ["# Experiment 037 — dry-run test copy", "",
             "*Pre-registered: {{PREREG_UTC}} · Status: pre-registered, awaiting Andrei's sign-off; no scored run*", "",
             "## Fixed before any run (hashes)", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {row} | " + "; ".join(f"sha256 `{{{{{s.placeholder}}}}}`" for s in sorted(scopes, key=lambda x: x.ordinal))
              + " |" for row, scopes in rows.items()]
    return "\n".join(lines + ["", "## Pre-registration decisions", "", "- none", ""])


def test_reset_run_state_unfills_and_removes_real_manifests_and_shingles(tmp_path):
    exp = tmp_path / "exp"
    (exp / "tasks" / "manifests").mkdir(parents=True)
    (exp / "tools").mkdir()
    filled = _template()
    for i, s in enumerate(ht.SCOPES):                                   # a filled, stamped, amended copy
        filled = filled.replace("{{" + s.placeholder + "}}", f"{i:064x}", 1)
    filled = filled.replace("{{PREREG_UTC}}", "2026-10-07T06:00:00Z")
    filled += "\n## Amendment 0 — sign-off choices (2026-10-07T07:00:00Z)\n\nhash_tree: ANALYSIS_SHA256 = " + "e" * 64 + "\n"
    (exp / "HYPOTHESIS.md").write_text(filled, encoding="utf-8")
    for name in ("gpqa_diamond_en.json", "mmlu_prox_category_counts.json", "ifbench.json", "ifbench_pilot.json"):
        (exp / "tasks" / "manifests" / name).write_text("{}\n")
    (exp / "tools" / "withheld_shingles.sha256").write_text("# x\n")
    done = dry_run.reset_run_state(exp)
    text = (exp / "HYPOTHESIS.md").read_text(encoding="utf-8")
    assert {s.placeholder for s in ht.SCOPES} <= set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
    assert "{{GATE_RULES_SHA256}}" in text
    assert "{{PREREG_UTC}}" in text and not ht.AMEND_LINE.search(text)
    assert sorted(p.name for p in (exp / "tasks" / "manifests").iterdir()) == ["ifbench.json", "ifbench_pilot.json"]
    assert not (exp / "tools" / "withheld_shingles.sha256").exists()
    assert "tools/withheld_shingles.sha256" in done
    assert dry_run.reset_run_state(exp) == []                           # idempotent


# ---------------------------------------------------------------- exp_037 identifiers and the world

def test_the_copy_is_this_kit_under_its_exp037_path():
    assert dry_run.REL_EXP == "tasks/chronos/exp_037_kolibri_forced_gate"
    assert (EXP.parents[2] / dry_run.REL_EXP).resolve() == EXP.resolve()
    assert dry_run.EXP036_DIR == EXP.parent / "exp_036_kolibri_local_eval"


def test_commit_subjects_and_temp_prefix_are_exp037s():
    src = (EXP / "tools" / "dry_run.py").read_text(encoding="utf-8")
    assert dry_run.COMMIT_PREFIX == "chronos/exp_037:" and dry_run.TEMP_PREFIX == "exp037_dry_run_"
    assert "chronos/exp_036" not in src and "exp036_dry_run_" not in src
    subjects = re.findall(r'(?:commit\(|"-m", )f?"([^"]+)"', src)
    subjects = [s for s in subjects if s != "dry run"]
    assert subjects and all(s.startswith("{COMMIT_PREFIX} ") for s in subjects), subjects


def test_stage_order_follows_the_exp037_runbook():
    order = dry_run.STAGE_ORDER
    assert "convert" not in order
    assert order.index("manifests") < order.index("refresh") < order.index("peers") < order.index("gate")
    assert order[0] == "world" and order[-2:] == ("hash_check", "leak")


def test_world_environment(tmp_path, monkeypatch):
    for k, v in {"EXP036_MODELS": "/host/models", "EXP036_GATE_G1": "x", "EXP037_BUILDS": "/host/builds",
                 "EXP037_VENV": "/host/venv"}.items():
        monkeypatch.setenv(k, v)
    w = dry_run.World(tmp_path)
    e = w.env()
    assert e["EXP036_MODELS"] == str(tmp_path / "models")
    assert e["EXP037_BUILDS"] == str(tmp_path / "models" / "exp037-builds") == str(w.builds)
    assert e["EXP037_VENV"] == sys.prefix and e["PY"] == sys.executable
    assert "EXP036_GATE_G1" not in e
    assert not any(v.startswith("/host/") for k, v in e.items() if k.startswith(("EXP036_", "EXP037_")))
    g = w.gate_env()
    assert g["EXP036_MODELS"] == g["EXP037_BUILDS"] == str(tmp_path / "tiny_real_val") == str(w.val)
    assert g["EXP036_TOK"] == str(tmp_path / "models" / "Kolibri-1-BF16")


@pytest.mark.parametrize("rel,skipped", [
    ("env/exp037.local.env", True), ("env/exp036.local.env", True), ("results/README.md", True),
    ("tools/__pycache__/x.pyc", True), (".DS_Store", True), ("env/exp037.settings.sh", False),
    ("tools/refresh_builds.py", False), ("tests/fixtures/leak/x.txt", False),
])
def test_copy_skips_caches_run_dirs_and_local_env(rel, skipped):
    assert dry_run.copy_skipped(Path(rel)) is skipped


def test_scale_plan_rules_is_a_scaled_copy():
    rules = json.loads((EXP / "runner" / "plan_rules.json").read_text())
    before = json.dumps(rules, sort_keys=True)
    r = dry_run.scale_plan_rules(rules)
    assert json.dumps(rules, sort_keys=True) == before                   # the input is not touched
    assert r["nM_ladder"] == [dry_run.DRY_NM_LARGE] * 2 + [dry_run.DRY_NM_SMALL] * 6
    assert set(r["caps"].values()) == {24} and r["ladder_estimate_h"] == {"B4": 1000.0}
    assert "dry_run" in r and "dry_run" not in rules


def test_exp036_port_is_found_and_pinned():
    """The stand-in sources carry exp_036's port, as the refresh expects of the real ones."""
    from tools import refresh_builds

    p = dry_run.exp036_port_file()
    assert p == EXP.parent / "exp_036_kolibri_local_eval" / "port" / "kolibri1.py"
    assert dry_run.sha256_file(p) == refresh_builds.EXP036_PORT_SHA256


# ---------------------------------------------------------------- peers record, gate promotion, S1 inputs

def test_peers_record_uses_the_peer_check_schema_and_the_reference_classes(monkeypatch):
    from tools import peer_check as pc

    monkeypatch.setattr(pc, "reference_classes", lambda reference=None: {"G8": "pass", "G4": pc.SPEED_ONLY})
    rec = dry_run.peers_record(datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc))
    assert rec["schema"] == pc.SCHEMA == "exp037 peer check v1"
    assert sorted(rec["arms"]) == sorted(pc.ARM_DIRS)
    assert {a: v["verdict"] for a, v in rec["arms"].items()} == {
        "G8": "ok", "G4": pc.SPEED_ONLY, "Q36-8": "ok", "Q36-4": "ok", "Q38-8": "ok", "Q38-4": "ok"}
    assert all(v["dry_run"] for v in rec["arms"].values()) and rec["reference"] == {"status": "pinned"}
    assert rec["t_start"] == rec["t_end"] == "2026-10-07T09:00:00Z"


def test_peers_record_with_an_unusable_reference_marks_no_arm_speed_only(monkeypatch):
    """As peer_check.run() keeps an unusable pin: recorded, and the registered rule applies (no speed-only)."""
    from tools import peer_check as pc

    def broken(reference=None):
        raise pc.ReferenceError("gemma4: reference record diagnostics/x.json is missing")

    monkeypatch.setattr(pc, "reference_classes", broken)
    rec = dry_run.peers_record(datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc))
    assert {v["verdict"] for v in rec["arms"].values()} == {"ok"}
    assert rec["reference"]["status"] == "unusable" and "is missing" in rec["reference"]["error"]


def _tiny_record(**kw) -> dict:
    rec = {"mode": "tiny", "tiny_outcome": {"K8": "PASS", "K4": "PASS"}, "exit_code": 0,
           "utc": "20261007T100000Z", "verdict": {"K8": "TINY", "K4": "TINY"},
           "sha256": {"port": "p" * 64, "thresholds": "t" * 64, "reference_tree": "r" * 64,
                      "converted_manifest": {"K8": "a" * 64, "K4": "b" * 64}},
           "allowed_B": {"K8": [1, 2, 4, 8], "K4": [1]},
           "controls": {"G4-F32": {"1": {"caught": True, "leg": "F6"}, "6": {"caught": True, "leg": "F2"},
                                   "19": {"caught": True, "leg": "F2"}},
                        "G4-F16": {"6": {"caught": True, "leg": "kappa"}},
                        "G5-D32": {"20": {"caught": True, "leg": "decode"}},
                        "G5-R1": {"26": {"caught": True, "leg": "R-parity"}},
                        "G5-BP-lean": {"22": {"caught": True, "leg": "(d)"}}}}
    rec.update(kw)
    return rec


def test_promote_record_relabels_and_rebinds_to_the_refreshed_clones():
    rec = _tiny_record()
    now = datetime(2026, 10, 7, 10, 0, 0, tzinfo=timezone.utc)          # same second as the tiny record
    clones = {"K8": "c" * 64, "K4": "d" * 64}
    out = dry_run.promote_record(rec, "gate_20261007T100000Z.json", now, clones)
    assert out["mode"] == "real" and out["verdict"] == {"K8": "PASS", "K4": "PASS"}
    assert out["utc"] == "20261007T100001Z"                             # strictly after the tiny record
    assert out["dry_run_promoted_from"] == "gate_20261007T100000Z.json"
    assert out["sha256"]["converted_manifest"] == clones
    assert out["dry_run_gate_build"]["converted_manifest"] == {"K8": "a" * 64, "K4": "b" * 64}
    assert {k: out["sha256"][k] for k in ("port", "thresholds", "reference_tree")} == \
        {k: rec["sha256"][k] for k in ("port", "thresholds", "reference_tree")}
    assert out["allowed_B"] == rec["allowed_B"] and out["controls"] == rec["controls"]
    assert rec["mode"] == "tiny" and rec["sha256"]["converted_manifest"]["K8"] == "a" * 64   # input untouched


@pytest.mark.parametrize("bad", [
    {"mode": "real"}, {"tiny_outcome": {"K8": "PASS", "K4": "FAIL"}}, {"tiny_outcome": {"K8": "INCOMPLETE",
                                                                                       "K4": "PASS"}},
    {"exit_code": 5}, {"controls": None}, {"controls": {}},
    {"controls": {"G5-R1": {"26": {"caught": False, "leg": "R-parity"}}, "G4-F32": {"1": {"caught": True}}}},
    {"controls": [{"mutant": 22, "caught": "yes"}]},
])
def test_promote_record_refuses_anything_but_a_full_tiny_pass(bad):
    with pytest.raises(dry_run.StageError):
        dry_run.promote_record(_tiny_record(**bad), "gate_x.json", datetime.now(timezone.utc), {})


def test_control_outcomes_walks_any_nesting():
    found = dry_run.control_outcomes({"a": [{"caught": True, "leg": "F6"}, {"x": {"caught": False}}], "b": "text"})
    assert [(o["path"], o["caught"]) for o in found] == [("a[0]", True), ("a[1]/x", False)]
    assert dry_run.control_outcomes(None) == []


def test_the_s1_hours_inputs_include_the_refresh_records(tmp_path):
    """DESIGN §2.8: runner/plan_fix.py's S1_GLOBS ("convert/*.json") counts the refresh records, and the dry
    run's plan stage finds both through refresh_phases."""
    from runner import plan_fix

    conv = tmp_path / "results" / "convert"
    conv.mkdir(parents=True)
    for bits, t in ((8, "09"), (4, "10")):
        (conv / f"refresh_{bits}bit_20261007T{t}0000Z.json").write_text(json.dumps(
            {"t_start": f"2026-10-07T{t}:00:00Z", "t_end": f"2026-10-07T{t}:03:00Z", "bits": bits}))
    recs = plan_fix.load_s1_records(tmp_path / "results")
    got = dry_run.refresh_phases(recs)
    assert got == {8: ["convert/refresh_8bit_20261007T090000Z.json"], 4: ["convert/refresh_4bit_20261007T100000Z.json"]}
    assert dry_run.refresh_phases([{"phase": "convert/convert_8bit_x.json"}]) == {8: [], 4: []}


def test_cli_description_names_exp037(capsys, monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    with pytest.raises(SystemExit):
        dry_run.main(["--help"])
    assert "exp_037 dry run" in capsys.readouterr().out
    assert os.environ.get("MLX_ENABLE_TF32") == "0"
