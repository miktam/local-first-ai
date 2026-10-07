# SPDX-License-Identifier: MIT
"""gate/run_gate.py require_pass (HYPOTHESIS Phase 0 "Verdict and refusal"; exp_037 DESIGN §3.16),
and the orchestration's exits and run counts (DESIGN §3.16, §9.4).

The runner refuses every Kolibri pilot, bench or scored run unless the latest
results/gate/gate_<UTC>.json is a real-mode exp_037 record that says PASS for that
arm, and its recorded sha256 of port/kolibri1.py, gate/thresholds.json, reference/,
the gate rules (GATE_RULES_SHA256) and the arm's converted manifest under builds_dir()
equal the current ones; the build's own kolibri1.py copy must be the current port as
well. require_pass returns the record with the arm's allowed_B.

The exit table and the run counts are checked on run_gate.run_all with its phases
replaced by fakes (no model is loaded): exit 0, 1, 4 and 5 are gate runs, exit 3 never
is; a run after exit 1, 4 or 5 is a fix cycle; G1 FAIL followed by a P5 failure is
exit 3 (the precondition wins). Interrupts (final review W15-03): Ctrl-C, SIGTERM,
SIGHUP and SystemExit during the phases write the record (exit 3, or exit 1 when a K8
check had already failed) and move an exit-3 run to aborted/ before they are
re-raised; a signal during the record write is deferred; a run killed outright is
closed afterwards by close_orphans, which P1(d) requires.
"""

from __future__ import annotations

import io
import json
import os
import signal
import time
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
    from gate import common, run_gate

    models = tmp_path / "builds"
    for arm in ("K8", "K4"):
        _record_dir(models, common.ARM_DIRS[arm])
    results = tmp_path / "results"
    (results / "gate").mkdir(parents=True)

    def sha(**over):
        s = {"port": common.port_sha256(), "thresholds": common.thresholds_sha256(),
             "reference_tree": common.reference_tree_sha256(), "gate_rules": run_gate.gate_rules_sha256(),
             "converted_manifest": {a: common.converted_manifest_sha256(models / common.ARM_DIRS[a]) for a in ("K8", "K4")}}
        s.update(over)
        return s

    def write(utc, verdict, mode="real", experiment="exp_037", gate_rules=None, allowed_B=None, **over):
        s = sha(**over)
        rec = {"mode": mode, "utc": utc, "verdict": verdict, "experiment": experiment, "sha256": s,
               "gate_rules_sha256": gate_rules or s["gate_rules"],
               "allowed_B": allowed_B if allowed_B is not None else {"K8": [1, 2, 4, 8], "K4": [1]}}
        (results / "gate" / f"gate_{utc}.json").write_text(json.dumps(rec))

    return {"models": models, "results": results, "write": write, "sha": sha}


def _req(world, arm):
    from gate import run_gate

    return run_gate.require_pass(arm, results_dir=world["results"], builds_dir=world["models"])


def test_pass_allows_both_arms_and_returns_allowed_B(world):
    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    for arm, allowed in (("K8", [1, 2, 4, 8]), ("K4", [1])):
        rec = _req(world, arm)
        assert rec.verdict == "PASS" and rec.arm == arm and rec.path.name == "gate_20261004T100000Z.json"
        assert rec.allowed_B == allowed
    world["write"]("20261004T110000Z", {"K8": "PASS", "K4": "PASS"}, allowed_B={})
    assert _req(world, "K8").allowed_B == [1]  # no allowed_B: B = 1 only


def test_k4_fail_refuses_only_k4(world):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "FAIL"})
    assert _req(world, "K8").verdict == "PASS"
    with pytest.raises(run_gate.GateRefused, match="K4 verdict is 'FAIL'"):
        _req(world, "K4")


def test_incomplete_and_missing_verdicts_are_refused(world):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "INCOMPLETE", "K4": "MISSING"})
    with pytest.raises(run_gate.GateRefused, match="'INCOMPLETE', not PASS"):
        _req(world, "K8")
    with pytest.raises(run_gate.GateRefused, match="'MISSING', not PASS"):
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


def test_changed_gate_rules_are_refused(world):
    """DESIGN §3.16: require_pass also binds GATE_RULES_SHA256."""
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"}, gate_rules="1" * 64)
    with pytest.raises(run_gate.GateRefused, match="GATE_RULES_SHA256"):
        _req(world, "K8")


def test_an_exp036_record_is_refused(world):
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"}, experiment="exp_036")
    with pytest.raises(run_gate.GateRefused, match="not an exp_037 gate record"):
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


def test_arms_resolve_under_builds_dir_by_default(world, monkeypatch):
    """DESIGN §2.8: without an explicit directory the arms resolve under $EXP037_BUILDS."""
    from gate import run_gate

    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "PASS"})
    monkeypatch.setenv("EXP037_BUILDS", str(world["models"]))
    assert run_gate.require_pass("K8", results_dir=world["results"]).verdict == "PASS"
    monkeypatch.setenv("EXP037_BUILDS", str(world["models"].parent / "elsewhere"))
    with pytest.raises(run_gate.GateRefused, match="no converted directory"):
        run_gate.require_pass("K8", results_dir=world["results"])


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
    """runner/guard.py require_gate calls gate.run_gate.require_pass(arm) and returns allowed_B."""
    from exp036_helpers import import_sibling

    guard = import_sibling("runner.guard")
    from gate import run_gate

    real = run_gate.require_pass
    monkeypatch.setattr(run_gate, "require_pass", lambda arm, results_dir=None, models_dir=None:
                        real(arm, results_dir=results_dir, builds_dir=world["models"]))
    world["write"]("20261004T100000Z", {"K8": "PASS", "K4": "FAIL"})
    assert guard.require_gate("K8", results=world["results"]) == [1, 2, 4, 8]
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
    assert run_gate.signed_off_policy(tmp_path / "absent.md") == ("quantised_head", "pre-registered default (no HYPOTHESIS.md)")
    # The live HYPOTHESIS.md (W12 writes it) has the line, unsigned until Andrei types it.
    assert run_gate.signed_off_policy()[0] in ("quantised_head", "vendor_faithful")


# ---------------------------------------------------------------------------
# Exits and run counts on run_all with fake phases (DESIGN §3.16, §9.4)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gate(tmp_path, monkeypatch):
    """run(scenario) -> GateResult: a real-mode run_all whose phases are fakes. A scenario
    gives each blocking check's state ("checks": {cid: state}, default PASS), the phase
    that raises ("crash"), and the preconditions that fail ({"P5": ...})."""
    from gate import common, rules, run_gate

    monkeypatch.setattr(common, "require_identity", lambda: None)
    monkeypatch.setattr(run_gate, "textset_for", lambda ctx: (_ for _ in ()).throw(RuntimeError("no texts")))
    builds = tmp_path / "builds"
    for arm in ("K8", "K4"):
        _record_dir(builds, common.ARM_DIRS[arm])
    state = {}
    ran = []

    def phase(n):
        def fn(ctx):
            ran.append(n)
            sc = state["scenario"]
            if sc.get("crash") == n:
                raise RuntimeError(f"crash in phase {n}")
            if (sc.get("interrupt") or (None,))[0] == n:   # W15-03: Ctrl-C, SystemExit, ... in phase n
                raise sc["interrupt"][1]
            if (sc.get("signal") or (None,))[0] == n:      # W15-03: a real signal to this process in phase n
                os.kill(os.getpid(), sc["signal"][1])
                for _ in range(200):
                    time.sleep(0.01)                       # the handler raises at the next bytecode boundary
                raise AssertionError("the signal handler did not run")
            out = {"data": {"fake": n}}
            if n in ("0", "2b"):
                names = ("P1", "P2", "P3", "P4") if n == "0" else ("P5",)
                pre = {k: {"ok": k not in sc.get("fail", ()), "values": {}, "reason": None if k not in sc.get("fail", ())
                           else f"{k} failed"} for k in names}
                failed = [k for k, v in pre.items() if not v["ok"]]
                out.update(preconditions=pre, precondition_failed=bool(failed), failed_preconditions=failed)
            return out
        return fn

    def evaluate(ctx, phases):
        sc = state["scenario"]
        checks = {}
        for cid in rules.BLOCKING["K8"] + rules.BLOCKING["K4"]:
            s = sc.get("checks", {}).get(cid, rules.PASS)
            if s is not None:
                checks[cid] = rules.check(s, {"fake": True}, None, reason=None if s == rules.PASS else f"{cid} {s}")
        return {"checks": checks, "layer_rows": [], "g5bp": rules.g5_bp(None, None, ctx.thresholds), "notes": {}}

    monkeypatch.setattr(run_gate, "PHASE_FN", {n: phase(n) for n in run_gate.PHASES if n != "7"})
    monkeypatch.setattr(run_gate, "evaluate", evaluate)
    times = iter(f"20261006T10{m:02d}00Z" for m in range(60))

    def run(scenario):
        state["scenario"] = scenario
        ran.clear()
        ctx = run_gate.GateContext(models_dir=tmp_path / "models", work_dir=tmp_path / "work",
                                   thresholds=common.load_thresholds(), results_dir=tmp_path / "results",
                                   builds_dir=builds, log_stream=io.StringIO(), utc=next(times))
        state["ctx"] = ctx
        res = run_gate.run_all(ctx)
        res.ran = list(ran)
        return res

    run.state = state
    run.ran = ran
    return run


@pytest.mark.parametrize("scenario, code, verdicts", [
    ({}, 0, {"K8": "PASS", "K4": "PASS"}),
    ({"checks": {"g4_f32_K8": "FAIL"}}, 1, {"K8": "FAIL", "K4": "PASS"}),
    ({"checks": {"g5_behaviour_K4": "FAIL"}}, 4, {"K8": "PASS", "K4": "FAIL"}),
    ({"checks": {"g4_f16_K8": "INCOMPLETE"}}, 5, {"K8": "INCOMPLETE", "K4": "PASS"}),
    ({"checks": {"g5_r1_K4": "INCOMPLETE"}}, 5, {"K8": "PASS", "K4": "INCOMPLETE"}),
    ({"checks": {"g4_f32_K8": "FAIL", "g5_r1_K4": "INCOMPLETE"}}, 1, {"K8": "FAIL", "K4": "INCOMPLETE"}),
    ({"checks": {"g5_d32_K8": None}}, 3, {"K8": "MISSING", "K4": "PASS"}),
    ({"checks": {"g5_d32_K8": "MISSING"}}, 3, {"K8": "MISSING", "K4": "PASS"}),
    ({"crash": "4"}, 3, {"K8": "MISSING", "K4": "MISSING"}),
    ({"crash": "6", "checks": {"g2_head": "FAIL"}}, 1, {"K8": "FAIL", "K4": "MISSING"}),
    ({"fail": ("P2",)}, 3, {"K8": "MISSING", "K4": "MISSING"}),
])
def test_exit_table(fake_gate, scenario, code, verdicts):
    res = fake_gate(scenario)
    rec = res.record
    assert (res.exit_code, rec["exit_code"]) == (code, code), (rec["error"], rec["failing"], rec["missing"])
    assert rec["verdict"] == verdicts
    assert rec["experiment"] == "exp_037" and rec["mode"] == "real"
    aborted = code == 3
    assert rec["phase_dir"] == (f"aborted/{rec['utc']}-gate" if aborted else f"results/gate/{rec['utc']}")


def test_g1_fail_then_a_p5_failure_is_exit_3(fake_gate):
    """DESIGN §3.16, G1 item 9: a precondition failure wins over an earlier FAIL; the run
    stops at P5 (no forced reference, no G2, no arm), it is not a gate run, and the
    partial output goes to aborted/<UTC>-gate/ with a NOTE.md."""
    res = fake_gate({"checks": {"g1": "FAIL"}, "fail": ("P5",)})
    rec = res.record
    assert res.exit_code == 3 and rec["precondition_failed"] and rec["failed_preconditions"] == ["P5"]
    assert res.ran == ["0", "1", "2a", "2b"]
    assert rec["blind_phase_reached"] is False
    assert rec["verdict"] == {"K8": "MISSING", "K4": "MISSING"}
    assert rec["run_counts"]["gate_runs"] == 0 and rec["run_counts"]["exit3_records"] == 1
    note = Path(res.path).parent.parent.parent / rec["phase_dir"] / "NOTE.md"
    assert note.is_file() and "P5" in note.read_text(encoding="utf-8")
    assert (note.parent / "phase2b.json").is_file() and not (note.parent / "phase2c.json").exists()


def test_a_crash_after_p5_records_the_blind_phase(fake_gate):
    """DESIGN §9.4 item 3: a crash after P5 passed is exit 3 with blind_phase_reached true;
    the NOTE names the phases that computed port-against-reference values."""
    res = fake_gate({"crash": "5"})
    rec = res.record
    assert res.exit_code == 3 and rec["blind_phase_reached"] is True and "phase 5" in rec["error"]
    note = (Path(res.path).parent.parent.parent / rec["phase_dir"] / "NOTE.md").read_text(encoding="utf-8")
    assert "blind_phase_reached: true" in note and "2c, 3, 4" in note


def test_cycle_counting(fake_gate):
    """DESIGN §9.4: exit 0, 1, 4 and 5 are gate runs and exit 3 never is; each gate run after
    one that exited 1, 4 or 5 is a fix cycle; at most two cycles, so three gate runs."""
    seq = [({"checks": {"g4_f32_K8": "FAIL"}}, 1), ({"crash": "3"}, 3),
           ({"checks": {"g5_behaviour_K4": "FAIL"}}, 4), ({"fail": ("P4",)}, 3),
           ({"checks": {"g4_f16_K8": "INCOMPLETE"}}, 5)]
    counts = []
    for scenario, code in seq:
        res = fake_gate(scenario)
        assert res.exit_code == code
        counts.append(res.record["run_counts"])
    assert [c["gate_runs"] for c in counts] == [1, 1, 2, 2, 3]
    assert [c["cycles_used"] for c in counts] == [0, 0, 1, 1, 2]
    assert [c["exit3_records"] for c in counts] == [0, 1, 1, 2, 2]
    assert counts[-1]["may_run_again"] is False and counts[-1]["gate_runs_max"] == 3
    assert counts[2]["next_run_is_cycle"] is True and counts[2]["may_run_again"] is True


# ---------------------------------------------------------------------------
# Interrupts and killed runs (final review W15-03; DESIGN §9.4 item 3)
# ---------------------------------------------------------------------------


def _record_on_disk(fake_gate) -> tuple[dict, Path]:
    ctx = fake_gate.state["ctx"]
    path = ctx.gate_dir / f"gate_{ctx.utc}.json"
    assert path.is_file(), "the interrupted run wrote no record"
    return json.loads(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(2)], ids=["ctrl-c", "system-exit"])
def test_an_interrupt_after_p5_is_recorded_moved_to_aborted_and_re_raised(fake_gate, exc):
    with pytest.raises(type(exc)) as e:
        fake_gate({"interrupt": ("4", exc)})
    rec, path = _record_on_disk(fake_gate)
    assert e.value.gate_result.path == path and e.value.gate_result.exit_code == 3
    assert rec["exit_code"] == 3 and rec["blind_phase_reached"] is True
    assert rec["error"] == f"interrupted in phase 4: {type(exc).__name__}"
    assert rec["verdict"] == {"K8": "MISSING", "K4": "MISSING"}
    ctx = fake_gate.state["ctx"]
    assert rec["phase_dir"] == f"aborted/{ctx.utc}-gate" and not ctx.run_dir.exists()
    note = (ctx.aborted_dir / "NOTE.md").read_text(encoding="utf-8")
    assert "blind_phase_reached: true" in note and "interrupted in phase 4" in note
    assert (ctx.aborted_dir / "phase2b.json").is_file() and (ctx.aborted_dir / "run.json").is_file()
    assert rec["run_counts"]["gate_runs"] == 0 and rec["run_counts"]["exit3_records"] == 1


def test_an_interrupt_before_p5_is_not_blind(fake_gate):
    with pytest.raises(KeyboardInterrupt):
        fake_gate({"interrupt": ("2a", KeyboardInterrupt())})
    rec, _ = _record_on_disk(fake_gate)
    assert rec["exit_code"] == 3 and rec["blind_phase_reached"] is False
    assert fake_gate.ran == ["0", "1", "2a"]


def test_an_interrupt_after_a_k8_fail_is_a_counted_gate_run(fake_gate):
    """A K8 check that already failed makes the interrupted run exit 1 (rules.exit_code_for:
    K8 FAIL wins over a crash): a gate run, so the next run is a fix cycle."""
    with pytest.raises(KeyboardInterrupt) as e:
        fake_gate({"interrupt": ("5", KeyboardInterrupt()), "checks": {"g4_f32_K8": "FAIL"}})
    rec, _ = _record_on_disk(fake_gate)
    assert rec["exit_code"] == 1 == e.value.gate_result.exit_code and rec["verdict"]["K8"] == "FAIL"
    ctx = fake_gate.state["ctx"]
    assert rec["phase_dir"] == f"results/gate/{ctx.utc}" and ctx.run_dir.is_dir()
    assert rec["run_counts"]["gate_runs"] == 1 and rec["run_counts"]["next_run_is_cycle"] is True


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_are_recorded_like_ctrl_c(fake_gate, signame):
    from gate import run_gate

    before = signal.getsignal(getattr(signal, signame))
    with pytest.raises(run_gate.GateInterrupted) as e:
        fake_gate({"signal": ("3", getattr(signal, signame))})
    rec, _ = _record_on_disk(fake_gate)
    assert e.value.signame == signame and rec["error"] == f"interrupted in phase 3: {signame}"
    assert rec["exit_code"] == 3 and rec["blind_phase_reached"] is True
    assert signal.getsignal(getattr(signal, signame)) is before        # the run's handlers are removed


def test_a_signal_while_the_record_is_written_is_deferred(fake_gate, monkeypatch):
    from gate import run_gate

    real = run_gate.aggregate

    def aggregate(ctx, phases, error, t0):
        os.kill(os.getpid(), signal.SIGINT)    # Ctrl-C during phase 7
        time.sleep(0.05)
        return real(ctx, phases, error, t0)

    monkeypatch.setattr(run_gate, "aggregate", aggregate)
    with pytest.raises(KeyboardInterrupt) as e:
        fake_gate({})
    rec, path = _record_on_disk(fake_gate)
    assert rec["exit_code"] == 0 and rec["error"] is None and e.value.gate_result.path == path
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_cli_reports_an_interrupted_run_with_the_records_exit_code(fake_gate, monkeypatch, capsys):
    from gate import run_gate

    monkeypatch.setattr(run_gate, "build_context", lambda args: fake_gate.state["ctx"])
    monkeypatch.setattr(run_gate.common, "set_offline_env", lambda: None)

    def run_all(ctx, argv=None):
        e = KeyboardInterrupt()
        e.gate_result = run_gate.GateResult({"verdict": {"K8": "MISSING", "K4": "MISSING"}, "failing": {},
                                             "error": "interrupted in phase 4: KeyboardInterrupt"}, None, 3)
        raise e

    fake_gate({})                                  # gives the fixture a ctx
    monkeypatch.setattr(run_gate, "run_all", run_all)
    assert run_gate.main(["--all"]) == 3
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["exit_code"] == 3 and out["interrupted"] == "KeyboardInterrupt"


class _Killed(BaseException):
    """Stands for SIGKILL: nothing in run_all can catch it."""


@pytest.mark.parametrize("scenario, code", [({}, 3), ({"checks": {"g2_head": "FAIL"}}, 1)])
def test_a_killed_run_is_an_orphan_until_close_orphans_records_it(fake_gate, monkeypatch, scenario, code):
    from gate import preconditions, run_gate

    real_write = run_gate._write_records

    def killed(*a, **k):
        raise _Killed()

    monkeypatch.setattr(run_gate, "_write_records", killed)
    with pytest.raises(_Killed):
        fake_gate({**scenario, "interrupt": ("5", _Killed())})
    ctx = fake_gate.state["ctx"]
    assert ctx.run_dir.is_dir() and not (ctx.gate_dir / f"gate_{ctx.utc}.json").exists()
    assert preconditions.orphaned_runs(ctx.results_dir) == [ctx.utc]
    d = preconditions.p1d_rerun(ctx.results_dir, {})
    assert not d["ok"] and "--close-orphans" in d["reason"]

    monkeypatch.setattr(run_gate, "_write_records", real_write)
    fake_gate.state["scenario"] = scenario
    closed = run_gate.close_orphans(run_gate.replace(ctx, utc="20261006T115900Z"))
    assert [r.record["utc"] for r in closed] == [ctx.utc]
    rec = closed[0].record
    assert rec["exit_code"] == code and rec["blind_phase_reached"] is True
    assert rec["error"].startswith("interrupted: the run wrote no record") and rec["t_start"]
    assert rec["arms"] == ["K8", "K4"] and rec["mode"] == "real"
    if code == 3:
        assert rec["phase_dir"] == f"aborted/{ctx.utc}-gate" and (ctx.aborted_dir / "NOTE.md").is_file()
    else:
        assert rec["phase_dir"] == f"results/gate/{ctx.utc}" and rec["run_counts"]["gate_runs"] == 1
    assert preconditions.orphaned_runs(ctx.results_dir) == []
    assert run_gate.close_orphans(ctx) == []      # idempotent
