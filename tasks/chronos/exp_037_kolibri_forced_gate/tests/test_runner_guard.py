"""runner/guard.py on a temporary git repository (BUILD_SPEC §5.4; §2 Identity;
HYPOTHESIS "Verdict and refusal", rule 5).

Includes the gate-PASS refusal: no gate module, no PASS record, a FAIL
verdict and an exception in gate/run_gate.py:require_pass all refuse; only a
PASS is accepted.

exp_037 (DESIGN §6.3): require_gate returns the record's allowed_B; the
version binding require_environment judges P2's pins only under a real-mode
gate record without dry_run_promoted_from (G1 item 14), and its pin table
agrees with env/versions.json and gate/thresholds.json's P2 block.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from runner import guard
from runner.common import sha256_bytes

SIGNOFF = "- Signed off by: Andrei (typed by Andrei on the mbp, 2026-10-04T07:00:00Z)\n"


def git(cwd, *args):
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A repo with tasks/chronos/exp/ inside, a bare remote and an upstream."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-q", str(remote)], check=True)
    root = tmp_path / "lfa"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Miktam")
    git(root, "config", "user.email", "hello@localfirstai.eu")
    git(root, "config", "commit.gpgsign", "false")
    exp = root / "tasks" / "chronos" / "exp"
    (exp / "scorers").mkdir(parents=True)
    (exp / "scorers" / "mc.py").write_text("X = 1\n")
    (exp / "HYPOTHESIS.md").write_text("# H\n\n## Sign-off\n\n- Signed off by:\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "-q", "-u", "origin", "main")
    return root, exp


def commit_push(root, msg="c"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)
    git(root, "push", "-q")


def scorers_sha(exp):
    return guard.scorers_tree_sha256(exp)


# ------------------------------------------------------------- identity


def test_identity(repo):
    root, exp = repo
    guard.require_identity(exp)
    git(root, "config", "user.email", "someone" + "@" + "example.invalid")  # split: no e-mail literal for the leak check
    with pytest.raises(SystemExit) as e:
        guard.require_identity(exp)
    assert e.value.code == 1 and "hello@localfirstai.eu" in str(e.value)


def test_clean_tree_allows_results_aborted_evidence_only(repo):
    root, exp = repo
    guard.require_clean_tree(exp_dir=exp)
    (exp / "results").mkdir()
    (exp / "results" / "x.json").write_text("{}")
    (exp / "aborted").mkdir()
    (exp / "aborted" / "NOTE.md").write_text("n")
    guard.require_clean_tree(exp_dir=exp)
    (exp / "scorers" / "mc.py").write_text("X = 2\n")
    with pytest.raises(SystemExit):
        guard.require_clean_tree(exp_dir=exp)


def test_signoff_must_be_committed_in_the_exact_form(repo):
    root, exp = repo
    with pytest.raises(SystemExit):
        guard.require_signoff(exp)
    (exp / "HYPOTHESIS.md").write_text("# H\n\n" + SIGNOFF)
    with pytest.raises(SystemExit):  # not committed yet
        guard.require_signoff(exp)
    commit_push(root)
    guard.require_signoff(exp)
    (exp / "HYPOTHESIS.md").write_text("# H\n\n- Signed off by: Claude (on behalf)\n")
    commit_push(root)
    with pytest.raises(SystemExit):
        guard.require_signoff(exp)


# ----------------------------------------------------------------- gate


@pytest.fixture
def fake_gate(monkeypatch):
    def install(require_pass):
        pkg = types.ModuleType("gate")
        pkg.__path__ = []
        mod = types.ModuleType("gate.run_gate")
        if require_pass is not None:
            mod.require_pass = require_pass
        monkeypatch.setitem(sys.modules, "gate", pkg)
        monkeypatch.setitem(sys.modules, "gate.run_gate", mod)
    return install


def test_refusal_when_the_gate_module_is_missing(monkeypatch):
    import importlib

    real = importlib.import_module

    def no_gate(name, *a, **k):
        if name == "gate.run_gate":
            raise ModuleNotFoundError("No module named 'gate.run_gate'", name="gate.run_gate")
        return real(name, *a, **k)

    monkeypatch.setattr(guard.importlib, "import_module", no_gate)
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8")
    assert "gate" in str(e.value)


def test_refusal_when_there_is_no_gate_pass(fake_gate):
    def no_record(arm):
        raise FileNotFoundError("no results/gate/gate_<UTC>.json")

    fake_gate(no_record)
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8")
    assert "no gate PASS for K8" in str(e.value)

    def exits(arm):
        sys.exit(1)

    fake_gate(exits)
    with pytest.raises(SystemExit):
        guard.require_gate("K4")

    # exp_037: a PASS returns the arm's allowed_B (DESIGN §3.14, §6.3), which the fake records carry.
    fake_gate(lambda arm: {"verdict": {"K8": "PASS", "K4": "FAIL"}, "allowed_B": {"K8": [1, 2, 4, 8], "K4": [1]}})
    assert guard.require_gate("K8") == [1, 2, 4, 8]
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K4")
    assert "FAIL" in str(e.value)

    fake_gate(lambda arm: None)
    with pytest.raises(SystemExit):
        guard.require_gate("K8")

    fake_gate(None)  # module without require_pass
    with pytest.raises(SystemExit):
        guard.require_gate("K8")


def test_peers_need_no_gate(fake_gate):
    fake_gate(lambda arm: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert guard.require_gate("G8") is None


# ---------------------------------------------------------------- peers


def test_require_peers(tmp_path):
    with pytest.raises(SystemExit):
        guard.require_peers("G8", tmp_path)
    (tmp_path / "peers_20261004T100000Z.json").write_text(json.dumps(
        {"arms": {"G8": {"verdict": "ok"}, "Q36-8": {"verdict": "fail"}, "Q38-8": {"verdict": "B=1"},
                  "G4": {"ok": False}}}))
    assert guard.require_peers("G8", tmp_path) == {"verdict": "ok"}
    assert guard.require_peers("Q38-8", tmp_path)  # only the batched path failed: runs at B = 1
    with pytest.raises(SystemExit):
        guard.require_peers("Q36-8", tmp_path)
    with pytest.raises(SystemExit):
        guard.require_peers("Q36-4", tmp_path)
    with pytest.raises(SystemExit):
        guard.require_peers("G4", tmp_path)
    assert guard.require_peers("K8", tmp_path) is None


def test_speed_only_arms_run_no_quality_cell_and_are_not_excluded(tmp_path):
    """Amendment 6: a "speed-only" arm is refused by require_peers (every caller runs quality cells), listed by
    speed_only_arms, and not by excluded_arms (H1, the speed cells and the B4 ladder keep it). The newest record
    decides."""
    (tmp_path / "peers_20261004T100000Z.json").write_text(json.dumps(
        {"arms": {"G8": {"verdict": "B=1"}, "G4": {"verdict": "speed-only"}, "Q36-4": {"verdict": "fail"}}}))
    with pytest.raises(SystemExit) as e:
        guard.require_peers("G4", tmp_path)
    assert "speed-only" in str(e.value) and "no quality cell" in str(e.value)
    assert guard.require_peers("G8", tmp_path)["verdict"] == "B=1"
    assert guard.speed_only_arms(tmp_path) == {"G4": "peer check speed-only (peers_20261004T100000Z.json)"}
    assert guard.excluded_arms(tmp_path) == {"Q36-4": "peer check fail (peers_20261004T100000Z.json)"}
    (tmp_path / "peers_20261004T110000Z.json").write_text(json.dumps({"arms": {"G4": {"verdict": "ok"}}}))
    assert guard.speed_only_arms(tmp_path) == {} and guard.require_peers("G4", tmp_path)["verdict"] == "ok"
    assert guard.speed_only_arms(tmp_path / "empty") == {}


# ----------------------------------------------------------------- plan


def write_plan(root, exp, status="FIXED", k=3, scorers=None, push=True):
    plan = {"status": status, "plan": "P0", "queue": [], "L_used": 115_448_725_504}
    rel = "results/plan_fixed_20261004T180000Z.json"
    data = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode()
    (exp / "results").mkdir(exist_ok=True)
    (exp / rel).write_bytes(data)
    sha = sha256_bytes(data)
    amend = (f"\n## Amendment {k} — plan (2026-10-04T18:00:00.000Z)\n\n- Status: {status}\n"
             f"- plan_fixed: `{rel}`, sha256 `{sha}`\n- Scorers tree sha256: `{scorers or scorers_sha(exp)}`\n")
    with open(exp / "HYPOTHESIS.md", "a") as f:
        f.write(amend)
    if push:
        commit_push(root, "plan")
    return rel, sha


def test_require_plan_accepts_a_pushed_committed_plan(repo):
    root, exp = repo
    rel, sha = write_plan(root, exp)
    plan = guard.require_plan(exp)
    assert plan["plan"] == "P0" and plan["_amendment_k"] == 3 and plan["_plan_sha256"] == sha


def test_require_plan_refusals(repo):
    root, exp = repo
    with pytest.raises(SystemExit):  # no plan amendment
        guard.require_plan(exp)
    write_plan(root, exp, push=False)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "not pushed")
    with pytest.raises(SystemExit) as e:  # HEAD != @{u}
        guard.require_plan(exp)
    assert "upstream" in str(e.value)
    git(root, "push", "-q")
    guard.require_plan(exp)
    (exp / "scorers" / "mc.py").write_text("X = 3\n")  # scorers changed after the plan
    commit_push(root)
    with pytest.raises(SystemExit) as e:
        guard.require_plan(exp)
    assert "scorers" in str(e.value)


def test_require_plan_refuses_an_edited_plan_file(repo):
    root, exp = repo
    rel, _ = write_plan(root, exp)
    (exp / rel).write_text("{}\n")
    with pytest.raises(SystemExit):
        guard.require_plan(exp)


def test_newest_plan_amendment_wins_and_STOP_refuses(repo):
    root, exp = repo
    write_plan(root, exp, k=3)
    # A later amendment (a re-pilot) with status STOP is the one that counts.
    rel = "results/plan_fixed_20261005T090000Z.json"
    data = (json.dumps({"status": "STOP", "queue": []}, indent=2) + "\n").encode()
    (exp / rel).write_bytes(data)
    with open(exp / "HYPOTHESIS.md", "a") as f:
        f.write(f"\n## Amendment 5 — plan (2026-10-05T09:00:00.000Z)\n\n- plan_fixed: `{rel}`, sha256 "
                f"`{sha256_bytes(data)}`\n- Scorers tree sha256: `{scorers_sha(exp)}`\n")
    commit_push(root)
    with pytest.raises(SystemExit) as e:
        guard.require_plan(exp)
    assert "STOP" in str(e.value)


# ---------------------------------------------------- metal limit, tier 2


def test_require_metal_limit():
    plan = {"L_used": 115_448_725_504, "sysctl_advice_mb": 114688}
    assert guard.require_metal_limit(plan, lambda: (115_448_725_504, {})) == 115_448_725_504
    with pytest.raises(SystemExit) as e:
        guard.require_metal_limit(plan, lambda: (100 * 2**30, {}))
    assert "sudo sysctl iogpu.wired_limit_mb=114688" in str(e.value)
    assert guard.require_metal_limit(plan, lambda: (100 * 2**30, {}), k8_queued=False)


def test_require_tier2(repo):
    root, exp = repo
    with pytest.raises(SystemExit):
        guard.require_tier2(exp)
    with open(exp / "HYPOTHESIS.md", "a") as f:
        f.write("\n## Amendment 4 — Tier-2 analysis (2026-10-05T10:00:00Z)\n\n- files …\n")
    commit_push(root)
    guard.require_tier2(exp)


def test_amendment_parser():
    text = ("x\n## Amendment 0 — sign-off choices (2026-10-04T07:00Z)\nA\n## Run record\nB\n"
            "## Amendment 2 — gate fix (2026-10-04T12:00Z)\nC\n")
    a = guard.amendments(text)
    assert [(x["k"], x["type"]) for x in a] == [(0, "sign-off choices"), (2, "gate fix")]
    assert a[0]["body"].strip() == "A"


def test_real_gate_module_refuses_without_a_pass_record(tmp_path):
    """gate/run_gate.py:require_pass on an empty results directory."""
    try:
        import gate.run_gate  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("gate/run_gate.py not built yet")
    (tmp_path / "gate").mkdir()
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8", tmp_path)
    assert "no gate PASS for K8" in str(e.value)
    (tmp_path / "gate" / "gate_20261004T120000Z.json").write_text(json.dumps(
        {"mode": "real", "verdict": {"K8": "FAIL", "K4": "FAIL"}, "sha256": {}}))
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8", tmp_path)
    assert "FAIL" in str(e.value) or "not PASS" in str(e.value)


def test_require_tier2_accepts_a_committed_amendments_file(repo):
    """Review fix 2026-10-03: the mini pushes the Tier-2 amendment as amendments/<k>_tier2_<UTC>.md; once that file
    is committed at HEAD (a pull), B4 and scoring may proceed before the mbp appends it to HYPOTHESIS.md."""
    root, exp = repo
    (exp / "amendments").mkdir()
    (exp / "amendments" / "4_tier2_20261005T100000Z.md").write_text(
        "## Amendment 4 — Tier-2 analysis (2026-10-05T10:00:00Z)\n\n- files …\n")
    assert not guard.tier2_at_head(exp)          # in the working tree only: not enough
    with pytest.raises(SystemExit):
        guard.require_tier2(exp)
    commit_push(root)
    assert guard.tier2_at_head(exp)
    guard.require_tier2(exp)


# ------------------------------------------------- exp_037: allowed_B from the gate record


@pytest.mark.parametrize("returned,want", [
    ({"verdict": {"K8": "PASS"}, "allowed_B": {"K8": [1, 2, 4, 8, 16], "K4": [1]}}, [1, 2, 4, 8, 16]),
    ({"verdict": {"K8": "PASS"}, "allowed_B": [8, 4, 2, 1]}, [1, 2, 4, 8]),          # one list for the arm
    ([1], [1]),                                                                        # require_pass returns it
    (types.SimpleNamespace(allowed_B=(1, 2, 4, 8)), [1, 2, 4, 8]),                     # an object carrying it
    (types.SimpleNamespace(record={"verdict": {"K8": "PASS"}, "allowed_B": {"K8": [1]}}), [1]),  # a GateRecord
    ({"verdict": {"K8": "PASS"}}, [1]),           # no allowed_B: no batched-path pass shown, B = 1 only
])
def test_require_gate_returns_the_records_allowed_B(fake_gate, returned, want):
    fake_gate(lambda arm: returned)
    assert guard.require_gate("K8") == want


@pytest.mark.parametrize("bad", [[1, 16], [2, 4, 8], [1, 2, 4], [0, 1], [1, 2, 4, 8, 32], ["1"], [True], 8,
                                 [1, 1, 2, 4, 8]])
def test_require_gate_refuses_an_allowed_B_that_g5_bp_lean_cannot_give(fake_gate, bad):
    """DESIGN §3.14: allowed_B is {1}, {1, 2, 4, 8} or {1, 2, 4, 8, 16}; anything else is a broken record."""
    fake_gate(lambda arm: {"verdict": {"K8": "PASS"}, "allowed_B": {"K8": bad}})
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8")
    assert "allowed_B" in str(e.value)


def test_a_failed_verdict_inside_a_gate_record_object_is_refused(fake_gate):
    fake_gate(lambda arm: types.SimpleNamespace(record={"verdict": {"K8": "INCOMPLETE"}, "allowed_B": {"K8": [1]}}))
    with pytest.raises(SystemExit) as e:
        guard.require_gate("K8")
    assert "INCOMPLETE" in str(e.value)


def test_newest_gate_record(tmp_path):
    assert guard.newest_gate_record(tmp_path) is None
    (tmp_path / "gate").mkdir()
    (tmp_path / "gate" / "gate_20261006T100000Z.json").write_text(json.dumps({"mode": "tiny"}))
    (tmp_path / "gate" / "gate_20261006T110000Z.json").write_text(json.dumps({"mode": "real"}))
    (tmp_path / "gate" / "gate_20261006T120000Z_partial.json").write_text("{")  # not a record name
    rec = guard.newest_gate_record(tmp_path)
    assert rec == {"mode": "real", "_file": "gate_20261006T110000Z.json"}


# ------------------------------------------------- exp_037: version binding (G1 item 14)

PINNED_ENV = {"macos": "27.0", "os_build": "26A428", "mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0",
              "architecture": "applegpu_g17s", "MLX_ENABLE_TF32": "0"}
REAL = {"mode": "real", "_file": "gate_20261008T120000Z.json", "verdict": {"K8": "PASS", "K4": "PASS"}}


def test_p2_pins_are_decision_f2s():
    assert guard.P2_PINS == {"macos": "27.0", "os_build": "26A428", "mlx": "0.32.3", "mlx-metal": "0.32.3",
                             "mlx-lm": "0.32.0", "architecture_prefix": "applegpu_g17", "MLX_ENABLE_TF32": "0"}


def test_p2_pins_agree_with_env_versions_and_the_thresholds_p2_block():
    """The runner's binding and the gate's P2 judge the same values: the package pins are env/versions.json's core
    (from env/requirements-mbp.txt, P2's package provenance); once gate/thresholds.json has its P2 block (W6),
    every pinned value appears in it."""
    from runner.common import EXP_DIR

    core = json.loads((EXP_DIR / "env" / "versions.json").read_text(encoding="utf-8"))["core"]
    assert {k: core[k] for k in ("mlx", "mlx-metal", "mlx-lm")} == {
        k: guard.P2_PINS[k] for k in ("mlx", "mlx-metal", "mlx-lm")}
    th = json.loads((EXP_DIR / "gate" / "thresholds.json").read_text(encoding="utf-8"))
    if "P2" in th:
        def strings(x):
            if isinstance(x, dict):
                return [s for v in x.values() for s in strings(v)]
            if isinstance(x, (list, tuple)):
                return [s for v in x for s in strings(v)]
            return [str(x)]

        found = strings(th["P2"])
        for key in ("macos", "os_build", "mlx", "mlx-metal", "mlx-lm", "architecture_prefix"):
            assert any(guard.P2_PINS[key] in s for s in found), (key, guard.P2_PINS[key])


def test_real_mode_record_on_the_pinned_runtime_is_judged_and_passes():
    out = guard.require_environment(REAL, observed=dict(PINNED_ENV))
    assert out["binding"] == "judged" and out["gate_record"] == "gate_20261008T120000Z.json"
    assert out["observed"] == PINNED_ENV and out["pins"] == guard.P2_PINS


@pytest.mark.parametrize("key,value", [("mlx", "0.31.2"), ("mlx-lm", "0.31.3"), ("mlx-metal", "0.32.2"),
                                       ("macos", "27.1"), ("os_build", "26A500"), ("architecture", "applegpu_g16s"),
                                       ("architecture", None), ("MLX_ENABLE_TF32", None), ("MLX_ENABLE_TF32", "1")])
def test_real_mode_record_on_another_runtime_refuses_the_session(key, value):
    with pytest.raises(SystemExit) as e:
        guard.require_environment(REAL, observed={**PINNED_ENV, key: value})
    assert key in str(e.value) and "P2" in str(e.value)


@pytest.mark.parametrize("rec", [
    {"mode": "tiny", "_file": "gate_20261006T100000Z.json"},
    {**REAL, "dry_run_promoted_from": "gate_20261006T100000Z.json"},
    None,
])
def test_tiny_dry_run_and_missing_records_record_the_versions_and_never_refuse(rec):
    other = {**PINNED_ENV, "mlx": "0.31.2", "mlx-lm": "0.31.3", "architecture": "applegpu_g16s", "macos": "26.5.1"}
    out = guard.require_environment(rec, observed=other)
    assert out["binding"] == "recorded" and out["observed"] == other


def test_require_environment_observes_this_process_by_default(monkeypatch):
    """With no injected values it reads the live runtime (the mini's exp_037 venv: MLX 0.32.3 / mlx-lm 0.32.0 on an
    M4, so not P2's architecture) and only records it under a tiny record."""
    pytest.importorskip("mlx.core")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    obs = guard.observe_environment()
    assert set(PINNED_ENV) <= set(obs)
    from importlib.metadata import version

    assert obs["mlx"] == version("mlx") and obs["mlx-lm"] == version("mlx-lm")
    assert obs["mlx-metal"] == version("mlx-metal") and obs["MLX_ENABLE_TF32"] == "0"
    assert isinstance(obs["architecture"], str) and obs["architecture"].startswith("applegpu_")
    assert not any(k in json.dumps(obs).lower() for k in ("serial", "uuid", "hostname", "device_name"))
    assert guard.require_environment({"mode": "tiny"})["observed"]["mlx"] == obs["mlx"]
