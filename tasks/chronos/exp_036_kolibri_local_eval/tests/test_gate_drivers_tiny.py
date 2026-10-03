# SPDX-License-Identifier: MIT
"""gate/run_gate.py end to end on a tiny checkpoint (BUILD_SPEC §6 test_gate_drivers_tiny.py).

A tiny Kolibri checkpoint (pattern5 preset: 10 layers, full attention at
i % 5 == 4, 16 experts top-6, window 17) is converted to 8 and 4 bits with
port/convert.py at the pre-registered policy. The gate then runs G0, G1 (a
subset), G2 (fp32, bf16 against the emulated reference, T9, router margin,
G2q, real-weight mutants), G3, G4 and G5 through the same drivers as a real
run and writes a schema-valid record with verdict "TINY".

* On the port the tiny outcome is PASS for K8 and K4 (exit 0).
* On each of the 15 port mutants the gate fails K8 (exit 1), and the failing
  checks include the mutant's target check.
* Exit codes 0 / 4 / 1 / 3 on constructed cases; a partial run is never
  PASS; a missing directory, a stale port copy or too little disk is
  "could not run" (exit 3); the last stdout line of the CLI is JSON.

Checks that tiny random weights cannot exercise are reported with
applicable=False (G3 bpb and mutants, the G5 greedy rate, behaviour); the
tiny run feeds G4/G5 with fp32 activations (tests/INTEGRATION_LOG.md 1-3).
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling

EXP = Path(__file__).resolve().parents[1]

RECORD_KEYS = {"experiment", "gate_version", "mode", "utc", "t_start", "t_end", "host", "git", "arms",
               "checks_requested", "verdict", "failing", "missing", "checks", "sha256", "tree_sha256_rule",
               "preconditions", "phases", "error", "exit_code", "files", "phase_files", "head_policy",
               "tiny_outcome", "mutant", "thresholds_version", "fix_cycles_max", "reference_passes",
               "reference_streamed_bytes"}


def _tok_dir():
    for name in ("EXP036_TOKENIZER_DIR", "EXP036_TOK"):
        v = os.environ.get(name)
        if v and (Path(v).expanduser() / "tokenizer.json").is_file():
            return Path(v).expanduser()
    return None


def _stub_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(tc.VOCAB_SIZE)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": f"t{tc.EOS_TOKEN_ID}", "pad_token": f"t{tc.PAD_TOKEN_ID}"}))


def build_tiny_models(root: Path, preset: str = "pattern5") -> Path:
    """root/Kolibri-1-BF16 (HF layout: no model_file) and its two conversions."""
    convert = import_sibling("port.convert")
    src = tc.write_tiny_checkpoint(root / "Kolibri-1-BF16", seed=0, preset=preset, copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _stub_tokenizer(src)
    kw = {"guard": False} if "guard" in inspect.signature(convert.convert).parameters else {}
    for bits in (8, 4):
        with contextlib.redirect_stdout(io.StringIO()):
            convert.convert(src, root / f"Kolibri-1-MLX-{bits}bit-g64", bits, 64, **kw)
    return root


@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory) -> Path:
    import_sibling("port.kolibri1")
    import_sibling("reference.kolibri_ref")
    return build_tiny_models(tmp_path_factory.mktemp("tiny_gate"))


def _ctx(root: Path, **kw):
    from gate import common, run_gate

    args = dict(models_dir=root, work_dir=root / "work", thresholds=common.load_thresholds(), tiny=True,
                results_dir=root / "results", tok_dir=_tok_dir(), head_policy="quantised_head",
                g1_select=["test_routing.py"], subprocess_phases=False, log_stream=io.StringIO())
    args.update(kw)
    return run_gate.GateContext(**args)


@pytest.fixture(scope="module")
def port_run(tiny_root):
    from gate import run_gate

    return run_gate.run_all(_ctx(tiny_root))


def test_tiny_gate_passes_on_the_port(port_run):
    from gate import run_gate

    rec = port_run.record
    assert rec["error"] is None, rec["error"]
    assert rec["tiny_outcome"] == {"K8": "PASS", "K4": "PASS"}, rec["failing"]
    assert port_run.exit_code == 0
    assert rec["verdict"] == {"K8": "TINY", "K4": "TINY"}
    for arm in ("K8", "K4"):
        for cid in run_gate.BLOCKING[arm]:
            c = rec["checks"][cid]
            assert c["pass"] is True or c.get("applicable") is False, (cid, c.get("pass"))


def test_tiny_record_is_schema_valid(port_run, tiny_root):
    from gate import common

    path = port_run.path
    text = path.read_text(encoding="utf-8")
    rec = json.loads(text)  # strict JSON: no NaN / Infinity
    assert set(rec) == RECORD_KEYS
    assert rec["mode"] == "tiny" and rec["gate_version"] == "exp036-gate-2"
    assert path.name == f"gate_{rec['utc']}.json" and path.parent == tiny_root / "results" / "gate"
    for cid, c in rec["checks"].items():
        assert {"pass", "measured", "threshold", "applicable", "arms", "blocking"} <= set(c), cid
    sha = rec["sha256"]
    assert sha["port"] == common.port_sha256() and sha["thresholds"] == common.thresholds_sha256()
    assert sha["reference_tree"] == common.reference_tree_sha256() and sha["gate_text"] == common.gate_text_sha256()
    assert set(sha["converted_manifest"]) == {"K8", "K4"} and len(sha["gate_code"]) == 64
    assert [p["phase"] for p in rec["phases"]] == [1, 2, 3, 4, 5, 6]
    assert all(p["t_start"] and p["t_end"] for p in rec["phases"])
    run_dir = path.parent / rec["utc"]
    for name, sha256 in rec["phase_files"].items():
        assert common.sha256_file(run_dir / f"{name}.json") == sha256
    assert (path.parent / rec["files"]["layers_csv"]).is_file() and (path.parent / rec["files"]["mutants_json"]).is_file()
    csv_text = (path.parent / rec["files"]["layers_csv"]).read_text()
    for mode in ("fp32_forced", "bf16_forced", "t9_bf16_forced", "g2q_K8", "g2q_K4"):
        assert f"\n{mode}," in csv_text, mode
    mutants = json.loads((path.parent / rec["files"]["mutants_json"]).read_text())
    assert set(mutants["real_weight_mutants"]["mutants"]) == {
        "sigmoid_bias_select", "window_512", "rope_on_full", "one_plus_w_norm", "swap_sandwich_norms", "renorm_topk",
        "route_scale_2826", "mup_embed_scale", "rope_traditional", "qknorm_after_rope", "biased_weights", "swiglu_swapped"}
    assert all(m["detected"] for m in mutants["real_weight_mutants"]["mutants"].values())
    assert mutants["forcing_crosscheck"]["max_rel"] < 1e-4  # route() boost == branches(force_ids) on the port
    assert len(mutants["g3_reference_mutants"]["mutants"]) == 7
    assert rec["head_policy"]["policy"] == "quantised_head"
    # One fp32 pass, one emulated pass, one mutant pass, two dequantised passes, the G5 pass.
    assert len(rec["reference_passes"]) == 6
    assert rec["reference_streamed_bytes"] == sum(p["bytes"] for p in rec["reference_passes"])


def test_g5_drivers_ran(port_run):
    c = port_run.record["checks"]
    assert c["g5_batch_parity_K8"]["measured"]["admitted_mid_run"] > 0
    assert c["g5_batch_parity_K8"]["measured"]["max_live"] <= 8
    assert c["g5_greedy"]["measured"]["n_positions"] > 0 and c["g5_greedy"]["applicable"] is False
    assert c["g5_behaviour_K8"]["measured"]["cells"]["high"]["n"] == 20


def test_reference_dumps_are_reused(port_run, tiny_root):
    from gate import common, run_gate

    res = run_gate.run_all(_ctx(tiny_root, checks=("g3",), arms=("K8", "K4")))
    p2 = common.read_json(res.path.parent / res.record["utc"] / "phase2.json")
    assert p2["data"]["dump"]["reused"] is True
    assert res.exit_code == 3  # partial: never PASS


def _target_id(target: str) -> list[str]:
    from gate import run_gate

    return [target] if target in run_gate.BLOCKING["K8"] else [f"{target}_K8", f"{target}_source"]


@pytest.mark.parametrize("mutant", [
    "sigmoid_bias_select", "window_512", "rope_on_full", "one_plus_w_norm", "swap_sandwich_norms", "renorm_topk",
    "route_scale_2826", "mup_embed_scale", "attn_output_gate", "bf16_router_logits", "bf16_head_logits",
    "rope_traditional", "qknorm_after_rope", "biased_weights", "swiglu_swapped"])
def test_tiny_gate_fails_on_each_port_mutant(port_run, tiny_root, mutant):
    from gate import port_mutants, run_gate

    res = run_gate.run_all(_ctx(tiny_root, mutant=mutant, checks=("g0", "g2"), arms=("K8",)))
    rec = res.record
    assert rec["tiny_outcome"]["K8"] == "FAIL", (mutant, rec["error"], rec["failing"])
    assert res.exit_code == 1
    assert any(t in rec["failing"]["K8"] for t in _target_id(port_mutants.TARGETS[mutant])), rec["failing"]["K8"]


def test_exit_codes_on_constructed_cases():
    from gate import run_gate

    assert run_gate.exit_code_for({"K8": "PASS", "K4": "PASS"}) == 0
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "FAIL"}) == 4
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "PASS"}) == 1
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "INCOMPLETE"}) == 1
    assert run_gate.exit_code_for({"K8": "INCOMPLETE", "K4": "PASS"}) == 3
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "INCOMPLETE"}) == 3
    ok = {"pass": True, "applicable": True}
    checks = {cid: dict(ok) for cid in run_gate.BLOCKING["K8"] + run_gate.BLOCKING["K4"]}
    assert run_gate.arm_verdict("K8", checks, False)[0] == "PASS"
    checks["g2q_K4"] = {"pass": False}
    assert run_gate.arm_verdict("K4", checks, False)[:2] == ("FAIL", ["g2q_K4"])
    assert run_gate.arm_verdict("K8", checks, False)[0] == "PASS"  # G2q at 4 bits is not a K8 check
    checks["g3_bpb_vs_peers"] = {"pass": None}
    assert run_gate.arm_verdict("K8", checks, False) == ("INCOMPLETE", [], ["g3_bpb_vs_peers"])
    checks["g3_bpb_vs_peers"] = {"pass": None, "applicable": False}
    assert run_gate.arm_verdict("K8", checks, True)[0] == "PASS"  # tiny: not applicable is skipped
    assert run_gate.arm_verdict("K8", checks, False)[0] == "INCOMPLETE"  # real: never skipped
    del checks["g5_behaviour_K8"]
    assert run_gate.arm_verdict("K8", checks, True) == ("INCOMPLETE", [], ["g5_behaviour_K8"])


def test_partial_run_is_never_pass(port_run, tiny_root):
    from gate import run_gate

    res = run_gate.run_all(_ctx(tiny_root, checks=("g0",)))
    assert res.record["tiny_outcome"] == {"K8": "INCOMPLETE", "K4": "INCOMPLETE"}
    assert res.exit_code == 3


def test_missing_directory_could_not_run(tiny_root, tmp_path):
    from gate import run_gate

    root = tmp_path / "models"
    root.mkdir()
    for name in ("Kolibri-1-BF16", "Kolibri-1-MLX-8bit-g64"):
        (root / name).symlink_to(tiny_root / name)
    res = run_gate.run_all(_ctx(root))
    assert res.exit_code == 3 and "Kolibri-1-MLX-4bit-g64" in res.record["error"]


def test_stale_port_copy_could_not_run(tiny_root, tmp_path):
    from gate import run_gate

    root = tmp_path / "models"
    root.mkdir()
    (root / "Kolibri-1-BF16").symlink_to(tiny_root / "Kolibri-1-BF16")
    (root / "Kolibri-1-MLX-4bit-g64").symlink_to(tiny_root / "Kolibri-1-MLX-4bit-g64")
    shutil.copytree(tiny_root / "Kolibri-1-MLX-8bit-g64", root / "Kolibri-1-MLX-8bit-g64")
    with open(root / "Kolibri-1-MLX-8bit-g64" / "kolibri1.py", "a") as f:
        f.write("\n# edited by hand\n")
    res = run_gate.run_all(_ctx(root))
    assert res.exit_code == 3 and "refresh-port-file" in res.record["error"]


def test_too_little_disk_could_not_run(tiny_root, tmp_path, monkeypatch):
    from gate import run_gate

    monkeypatch.setattr(run_gate.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10, 10, 0))
    res = run_gate.run_all(_ctx(tiny_root, work_dir=tmp_path / "work", checks=("g2",)))
    assert res.exit_code == 3 and "GB free" in res.record["error"]


def test_cli_last_line_is_json(tiny_root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, str(EXP / "gate" / "run_gate.py"), "--tiny", str(tiny_root), "--checks", "g0",
           "--arms", "K8", "--head-policy", "quantised_head", "--subprocess-phases"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(EXP), timeout=600)
    last = json.loads(p.stdout.strip().splitlines()[-1])
    assert p.returncode == last["exit_code"] == 3  # partial
    assert last["verdict"] == {"K8": "TINY", "K4": "TINY"}


def test_real_mode_options_refused():
    from gate import run_gate

    with pytest.raises(SystemExit, match="tiny"):
        run_gate.main(["--checks", "g0", "--port-mutant", "window_512"])


def test_real_mode_refuses_a_wrong_git_identity(tmp_path, monkeypatch):
    """BUILD_SPEC §2 Identity: nothing is written under results/ without it."""
    from gate import common, run_gate

    def refuse():
        raise RuntimeError("git identity is not Miktam <hello@localfirstai.eu>")

    monkeypatch.setattr(common, "require_identity", refuse)
    ctx = run_gate.GateContext(models_dir=tmp_path / "models", work_dir=tmp_path / "work",
                               thresholds=common.load_thresholds(), results_dir=tmp_path / "results")
    res = run_gate.run_all(ctx)
    assert res.exit_code == 3 and res.path is None and "identity" in res.record["error"]
    assert not (tmp_path / "results").exists()


def test_records_hold_no_host_name_or_absolute_path(port_run, tiny_root):
    """BUILD_SPEC §2 Privacy: no hostname, serial or home path in anything
    the gate writes under results/ (junit XML stays in the work dir)."""
    import re
    import socket

    gate_dir = tiny_root / "results" / "gate"
    host = socket.gethostname().split(".")[0]
    for p in gate_dir.rglob("*"):
        if p.is_file():
            text = p.read_text(encoding="utf-8")
            assert host not in text, p.name
            assert not re.search(r"/(Users|home|private|var/folders)/", text), p.name
            assert not re.search(r'"[^"]*(serial|uuid|hostname|computername)[^"]*"\s*:', text, re.I), p.name
    assert not list(gate_dir.rglob("*.xml"))
