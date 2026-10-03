# SPDX-License-Identifier: MIT
"""gate/run_gate.py require_pass (HYPOTHESIS Phase 0 "Verdict and refusal").

The runner refuses every Kolibri pilot, bench or scored run unless the latest
results/gate/gate_<UTC>.json says PASS for that arm, and its recorded sha256
of port/kolibri1.py, of the converted-directory manifest, of
gate/thresholds.json and of reference/ equal the current ones; the converted
directory's own kolibri1.py copy must be the current port as well.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _record_dir(models: Path, arm_dir: str) -> Path:
    from gate import common

    d = models / arm_dir
    d.mkdir(parents=True)
    port = common.PORT_FILE.read_bytes()
    (d / "kolibri1.py").write_bytes(port)
    (d / "model.safetensors").write_bytes(b"weights " + arm_dir.encode())
    files = {"kolibri1.py": {"sha256": common.sha256_file(d / "kolibri1.py"), "bytes": len(port)},
             "model.safetensors": {"sha256": common.sha256_file(d / "model.safetensors"), "bytes": 20}}
    rec = {"port_sha256": files["kolibri1.py"]["sha256"], "files": files}
    (d / common.CONVERT_RECORD).write_text(json.dumps(rec))
    return d


@pytest.fixture
def world(tmp_path):
    from gate import common

    models = tmp_path / "models"
    for arm in ("K8", "K4"):
        _record_dir(models, common.ARM_DIRS[arm])
    results = tmp_path / "results"
    (results / "gate").mkdir(parents=True)

    def sha(**over):
        s = {"port": common.port_sha256(), "thresholds": common.thresholds_sha256(),
             "reference_tree": common.reference_tree_sha256(),
             "converted_manifest": {a: common.converted_manifest_sha256(models / common.ARM_DIRS[a]) for a in ("K8", "K4")}}
        s.update(over)
        return s

    def write(utc, verdict, mode="real", **over):
        rec = {"mode": mode, "utc": utc, "verdict": verdict, "sha256": sha(**over)}
        (results / "gate" / f"gate_{utc}.json").write_text(json.dumps(rec))

    return {"models": models, "results": results, "write": write, "sha": sha}


def _req(world, arm):
    from gate import run_gate

    return run_gate.require_pass(arm, results_dir=world["results"], models_dir=world["models"])


def test_pass_allows_both_arms(world):
    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    for arm in ("K8", "K4"):
        rec = _req(world, arm)
        assert rec.verdict == "PASS" and rec.arm == arm and rec.path.name == "gate_20261004T100000Z.json"


def test_k4_fail_refuses_only_k4(world):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "FAIL"})
    assert _req(world, "K8").verdict == "PASS"
    with pytest.raises(run_gate.GateRefused, match="K4 verdict is 'FAIL'"):
        _req(world, "K4")


def test_latest_record_wins(world):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    world["write"]("20261004T120000Z", {"K8": "FAIL", "K4": "FAIL"})
    (world["results"] / "gate" / "gate_20261004T130000Z_mutants.json").write_text("{}")  # not a record
    with pytest.raises(run_gate.GateRefused, match="20261004T120000Z"):
        _req(world, "K8")


@pytest.mark.parametrize("key", ["port", "thresholds", "reference_tree"])
def test_changed_code_is_refused(world, key):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"}, **{key: "0" * 64})
    with pytest.raises(run_gate.GateRefused, match=key):
        _req(world, "K8")


def test_changed_manifest_is_refused(world):
    from gate import common, run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    d = world["models"] / common.ARM_DIRS["K8"]
    rec = json.loads((d / common.CONVERT_RECORD).read_text())
    rec["files"]["model.safetensors"]["sha256"] = "f" * 64  # a different conversion
    (d / common.CONVERT_RECORD).write_text(json.dumps(rec))
    with pytest.raises(run_gate.GateRefused, match="manifest"):
        _req(world, "K8")
    assert _req(world, "K4").verdict == "PASS"


def test_stale_port_copy_is_refused(world):
    from gate import common, run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    with open(world["models"] / common.ARM_DIRS["K4"] / "kolibri1.py", "a") as f:
        f.write("\n# edited\n")
    with pytest.raises(run_gate.GateRefused, match="refresh-port-file"):
        _req(world, "K4")


def test_tiny_records_and_missing_records_are_refused(world):
    from gate import run_gate

    with pytest.raises(run_gate.GateRefused, match="no gate record"):
        _req(world, "K8")
    world["write"]("20261004T100000Z", {"K8": "TINY", "K4": "TINY"}, mode="tiny")
    with pytest.raises(run_gate.GateRefused, match="tiny"):
        _req(world, "K8")
    with pytest.raises(run_gate.GateRefused, match="not a Kolibri arm"):
        _req(world, "G8")


def test_refusal_is_an_exit_1_system_exit():
    from gate import run_gate

    e = run_gate.GateRefused("reason")
    assert isinstance(e, SystemExit) and str(e) == "reason"


def test_cli_require_pass(monkeypatch, capsys):
    from gate import run_gate

    monkeypatch.setattr(run_gate, "require_pass", lambda arm: (_ for _ in ()).throw(run_gate.GateRefused("no PASS")))
    assert run_gate.main(["--require-pass", "K8"]) == 1
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {"require_pass": "K8", "allowed": False, "reason": "no PASS"}


def test_runner_guard_delegates_to_require_pass(world, monkeypatch):
    """runner/guard.py require_gate calls gate.run_gate.require_pass(arm)."""
    from exp036_helpers import import_sibling

    guard = import_sibling("runner.guard")
    from gate import run_gate

    real = run_gate.require_pass
    monkeypatch.setattr(run_gate, "require_pass", lambda arm, results_dir=None, models_dir=None:
                        real(arm, results_dir=results_dir, models_dir=world["models"]))
    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "FAIL"})
    assert guard.require_gate("K8", results=world["results"]) is not None
    with pytest.raises(SystemExit):
        guard.require_gate("K4", results=world["results"])


def test_signed_off_policy(tmp_path):
    from gate import run_gate

    line = ("- Kolibri `embed_tokens` / `lm_head` policy, applied to K8 and K4 (quantised at the arm's bits with fp32 "
            "logits as pre-registered, or vendor-faithful unquantised; the latter likely makes H1 REFUTED by itself):")
    h = tmp_path / "HYPOTHESIS.md"
    h.write_text(line + "\n")
    assert run_gate.signed_off_policy(h)[0] == "quantised_head"
    h.write_text(line + " vendor-faithful, unquantised\n")
    assert run_gate.signed_off_policy(h)[0] == "vendor_faithful"
    h.write_text(line + " quantised at the arm's bits (as pre-registered)\n")
    assert run_gate.signed_off_policy(h)[0] == "quantised_head"
    # The live HYPOTHESIS.md has the line (unsigned until Andrei types it).
    assert run_gate.signed_off_policy()[0] in ("quantised_head", "vendor_faithful")
