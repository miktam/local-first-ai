# SPDX-License-Identifier: MIT
"""gate/run_gate.py end to end on a tiny checkpoint (BUILD_SPEC §6 test_gate_drivers_tiny.py;
exp_037 DESIGN §12 W7). Heavy: left out of the gate's G1 (gate/checks/g1_synthetic.EXCLUDE).

Two tiny builds, each laid out as a models dir with its 8- and 4-bit conversions by
port/convert.py at the pre-registered policy (DESIGN §5.1; decision (a), 2026-10-06):
  * the port run (port_run, on val_root): the whole tiny gate's validation build, conftest.py's
    tiny_real_val_unsharp (tests/tiny_real_layout.py: the released layer and head layout at a
    tiny width, window 513, seed 29, q/k norms unsharpened), copied. Every check with a
    quantised or bf16 arm is judged on it, as in tests/test_gate_end_to_end_tiny.py.
    pattern5 no longer serves the port run: under the exp_037 rules its G2 bf16 natural
    selection leg (p) fires, and mutant 20's decode window of 256 keys cannot act on
    window 17, which leaves G5-D32 INCOMPLETE (DESIGN §11.13);
  * every other test (tiny_root): pattern5 (tests/tiny_checkpoint.py, exp_036's: 10 layers,
    full attention at i % 5 == 4, 16 experts top-6, window 17), as before.
The gate runs its exp_037 phases (0 preconditions with G0k, 1, 2a, 2b P5, 2c R2F/R3, 3,
4 with the control mutants, 5, 6 the R1 anchor, 7 the rules) through the same drivers as
a real run and writes a schema-valid record with verdict "TINY".

* On the port the tiny outcome is PASS for K8 and K4 (exit 0); every control mutant
  runs (its catch and tiny margin are tests/test_gate_end_to_end_tiny.py's, on the
  same validation build).
* On each of the 15 registered port mutants the gate fails K8 (exit 1), and the
  failing checks include the mutant's target check, which passes on the unmutated
  port with the same checks on the same build (pattern5_baseline).
* Exit codes on constructed cases follow DESIGN §3.16; a partial run is never
  PASS; a missing directory, a stale port copy, too little disk or a failed
  precondition is exit 3, and the run's files then go to aborted/<UTC>-gate/;
  the last stdout line of the CLI is JSON.
* Phase 0 runs G0k at the §3.1 P3 shapes, and P2 binds the F2 pins (decision F2).

Checks that tiny random weights cannot exercise are reported with
applicable=False (run_gate.TINY_NOT_APPLICABLE: G3, the G5 greedy rule and
G5-R1 K8, whose R-greedy leg is that rule, behaviour); the tiny run feeds its
free-routing checks fp32 activations (tests/INTEGRATION_LOG.md 1-3). G5-R1 K8's
R-anchor and R-parity legs are asserted here directly.
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

RECORD_KEYS = {"experiment", "gate_version", "rules_version", "mode", "utc", "t_start", "t_end", "host", "git", "arms",
               "checks_requested", "verdict", "failing", "missing", "incomplete", "reasons", "checks", "allowed_B",
               "allowed_B_power_note", "sha256", "gate_rules_sha256", "code_hashes", "tree_sha256_rule",
               "preconditions", "precondition_failed", "failed_preconditions", "blind_phase_reached", "controls",
               "probes", "calibrators", "g5_logprob_files", "phases", "error", "exit_code", "files", "phase_files",
               "phase_dir", "head_policy", "tiny_outcome", "mutant", "thresholds_version", "fix_cycles_max", "notes",
               "reference_passes", "reference_streamed_bytes", "run_counts"}
ALL_PHASES = ["0", "1", "2a", "2b", "2c", "3", "4", "5", "6", "7"]


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
                g1_select=["test_routing.py"], subprocess_phases=False, log_stream=io.StringIO(), builds_dir=root,
                g0k_rows=())
    args.update(kw)
    return run_gate.GateContext(**args)


@pytest.fixture(scope="module")
def val_root(tiny_real_val_unsharp, tmp_path_factory) -> Path:
    """A copy of the whole tiny gate's validation build (conftest.py's tiny_real_val_unsharp: seed 29,
    q/k norms unsharpened; DESIGN §5.1, decision (a)), so the shared session fixture is never written to."""
    import tiny_real_layout as trl

    import_sibling("port.kolibri1")
    import_sibling("reference.kolibri_ref")
    s = tiny_real_val_unsharp
    assert (s.seed, s.sharpen) == (trl.SEED_VAL, trl.SHARPEN_GATE) == (29, 1.0), (s.seed, s.sharpen)
    root = tmp_path_factory.mktemp("tiny_gate_val")
    for name in (trl.BF16_NAME, trl.build_dir_name(8), trl.build_dir_name(4)):
        shutil.copytree(s.root / name, root / name)
    return root


@pytest.fixture(scope="module")
def port_run(val_root):
    """The whole tiny gate on the port, on the validation build: every phase, every control, G0k at the
    §3.1 P3 shapes."""
    from gate import run_gate

    return run_gate.run_all(_ctx(val_root, g0k_rows=None))


@pytest.fixture(scope="module")
def pattern5_baseline(tiny_root):
    """The unmutated port on pattern5 with the port-mutant runs' checks and arm (g0, g2; K8): each mutant's
    target check must pass here, so its failure under the mutant is the mutant's (the port run, which used
    to give this baseline, is on the validation build now)."""
    from gate import run_gate

    return run_gate.run_all(_ctx(tiny_root, checks=("g0", "g2"), arms=("K8",)))


def _phase_file(res, name: str) -> Path:
    rec = res.record
    return Path(res.path).parent.parent.parent / rec["phase_dir"] / f"phase{name}.json"


def test_tiny_gate_passes_on_the_port(port_run):
    from gate import run_gate

    rec = port_run.record
    assert rec["error"] is None, rec["error"]
    assert rec["tiny_outcome"] == {"K8": "PASS", "K4": "PASS"}, (rec["failing"], rec["incomplete"], rec["missing"],
                                                                 rec["reasons"])
    assert port_run.exit_code == 0
    assert rec["verdict"] == {"K8": "TINY", "K4": "TINY"}
    for arm in ("K8", "K4"):
        for cid in run_gate.BLOCKING[arm]:
            c = rec["checks"][cid]
            assert c["pass"] is True or c.get("applicable") is False, (cid, c.get("state"), c.get("reason"))
    # G5-R1 K8 is tiny-exempt for its R-greedy leg only: R-anchor and R-parity hold on the port.
    legs = rec["checks"]["g5_r1_K8"]["legs"]
    assert legs["R-anchor"]["ok"] and legs["R-parity"]["ok"], legs
    assert rec["checks"]["g5_r1_K4"]["legs"]["R-anchor"]["ok"]


def test_tiny_record_is_schema_valid(port_run, val_root):
    from gate import common, rules, run_gate

    path = port_run.path
    text = path.read_text(encoding="utf-8")
    rec = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert set(rec) == RECORD_KEYS, set(rec) ^ RECORD_KEYS
    assert rec["mode"] == "tiny" and rec["gate_version"] == "exp037-gate-1" and rec["experiment"] == "exp_037"
    assert path.name == f"gate_{rec['utc']}.json" and path.parent == val_root / "results" / "gate"
    for cid, c in rec["checks"].items():
        assert {"state", "pass", "measured", "threshold", "applicable", "arms", "blocking"} <= set(c), cid
        assert c["state"] in rules.STATES, cid
    sha = rec["sha256"]
    assert sha["port"] == common.port_sha256() and sha["thresholds"] == common.thresholds_sha256()
    assert sha["reference_tree"] == common.reference_tree_sha256() and sha["gate_text"] == common.gate_text_sha256()
    assert sha["gate_rules"] == rec["gate_rules_sha256"] == run_gate.gate_rules_sha256()
    assert set(sha["converted_manifest"]) == {"K8", "K4"} and len(sha["gate_code"]) == 64
    assert set(rec["code_hashes"]) == {"port", "reference_tree", "runner_tree", "gate_code", "builds_manifest"}
    assert [p["phase"] for p in rec["phases"]] == ALL_PHASES
    assert all(p["t_start"] and p["t_end"] for p in rec["phases"])
    assert rec["phase_dir"] == f"results/gate/{rec['utc']}"
    run_dir = path.parent / rec["utc"]
    assert sorted(rec["phase_files"]) == sorted(f"phase{n}" for n in ALL_PHASES if n != "7")
    for name, sha256 in rec["phase_files"].items():
        assert common.sha256_file(run_dir / f"{name}.json") == sha256
    assert (path.parent / rec["files"]["layers_csv"]).is_file() and (path.parent / rec["files"]["mutants_json"]).is_file()
    csv_text = (path.parent / rec["files"]["layers_csv"]).read_text()
    for mode in ("fp32_forced", "bf16_forced", "t9_bf16_forced", "g2q_K8", "g2q_K4"):
        assert f"\n{mode}," in csv_text, mode
    assert csv_text.splitlines()[0].endswith(",pass")  # the per-row verdicts are gate/rules.py's
    mutants = json.loads((path.parent / rec["files"]["mutants_json"]).read_text())
    assert set(mutants["real_weight_mutants"]["mutants"]) == {
        "sigmoid_bias_select", "window_512", "rope_on_full", "one_plus_w_norm", "swap_sandwich_norms", "renorm_topk",
        "route_scale_2826", "mup_embed_scale", "rope_traditional", "qknorm_after_rope", "biased_weights", "swiglu_swapped"}
    assert all(m["detected"] for m in mutants["real_weight_mutants"]["mutants"].values())
    assert mutants["forcing_crosscheck"]["max_rel"] < 1e-4  # route() boost == branches(force_ids) on the port
    assert len(mutants["g3_reference_mutants"]["mutants"]) == 7
    assert rec["head_policy"]["policy"] == "quantised_head"
    # R1, the emulation, the G3 mutant pass, P5a, R2F, R3, two dequantised passes (G2q), the R1 anchor.
    whats = [p["what"].split(":")[0] for p in rec["reference_passes"]]
    assert len(rec["reference_passes"]) == 9, whats
    assert rec["reference_streamed_bytes"] == sum(p["bytes"] for p in rec["reference_passes"])
    assert set(rec["preconditions"]) == {"P1", "P2", "P3", "P4", "P5"} and not rec["precondition_failed"]
    assert all(rec["preconditions"][k]["ok"] for k in rec["preconditions"])
    assert rec["blind_phase_reached"] is True
    assert rec["run_counts"]["gate_runs"] == 0  # a tiny run is never a gate run
    assert set(rec["allowed_B"]) == {"K8", "K4"} and all(1 in v for v in rec["allowed_B"].values())
    # DESIGN §3.16, §4.2 (decision (b)): probe 22 is reported apart from the controls (tiny mode only)
    assert set(rec["probes"]) == {"G5BP/22"} and "G5BP/22" not in rec["controls"] and "G5BP/27" in rec["controls"]
    cal = rec["calibrators"]
    assert set(cal["tau"]) == {"T1-8", "T9"} and cal["f_tiny"] > 0 and set(cal["r3"]) == {"T1-8", "T9"}
    assert set(cal["csort_ranges"]) == {"T1", "T3", "T9"}


def test_phase0_runs_g0k_at_the_p3_shapes_and_p2_binds_the_f2_pins(port_run):
    """DESIGN §3.1 P2-P3 (decision F2): G0k judged at every (projection, bits, rows) of
    thresholds.json P3_G0k; P2's pins are MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0
    (recorded, not judged, in tiny mode); a real-mode P2 refuses another runtime."""
    from gate import common, preconditions, rules

    th = common.load_thresholds()
    rec = port_run.record
    p3 = rec["preconditions"]["P3"]
    assert p3["ok"] and p3["values"]["absent"] == [] and p3["values"]["failing"] == []
    # 24,576 and 49,152 rows are probed with both routings (Kolibri calls and boundary counts): more measurements
    # than judged (projection, bits, rows) shapes, each judged on all of its measurements.
    assert p3["values"]["n_judged"] == len(rules.p3_judged_shapes(th)) <= p3["values"]["measured"]
    assert "tiny_subset" not in p3["values"] and p3["values"]["file"]["path"].startswith("$EXP036_WORK/exp037/")
    p2 = rec["preconditions"]["P2"]["values"]
    assert p2["judged"] is False and p2["pins"]["packages"] == {"mlx": "0.32.3", "mlx-metal": "0.32.3",
                                                                "mlx-lm": "0.32.0"}
    assert p2["observed"]["packages"] == p2["pins"]["packages"]  # the exp_037 venv runs the F2 pins
    problems = preconditions.pin_problems(p2["observed"], th["P2"])
    assert not [p for p in problems if p.split()[0] in ("mlx", "mlx-metal", "mlx-lm", "MLX_ENABLE_TF32")], problems
    old = json.loads(json.dumps(p2["observed"]))
    old["packages"].update({"mlx": "0.31.2", "mlx-lm": "0.31.3"})
    real = preconditions.p2(th["P2"], mode="real", results_dir=EXP / "no-such-results", observed=old)
    assert not real["ok"] and "mlx '0.31.2' != '0.32.3'" in real["reason"] and "mlx-lm '0.31.3'" in real["reason"]


def test_g5_drivers_ran(port_run):
    c = port_run.record["checks"]
    for arm in ("K8", "K4"):
        for B in (8, 16):
            m = c[f"g5_bp_{arm}_B{B}"]["measured"]
            assert m["admitted_mid_run"] > 0 and m["max_live"] <= B and m["first_wave"]["n"] > 0
            assert c[f"g5_bp_{arm}_B{B}"]["blocking"] is False
    assert c["g5_greedy"]["measured"]["n_positions"] > 0 and c["g5_greedy"]["applicable"] is False
    assert c["g5_behaviour_K8"]["measured"]["cells"]["high"]["n"] == 20
    assert c["g5_decode_vs_prefill"]["state"] == "PASS"
    files = port_run.record["g5_logprob_files"]
    assert files["root"] == "$EXP036_WORK/exp037" and files["files"]
    blocks = {f["block"] for f in files["files"]}
    assert {"K8/g5_r1", "K4/g5_r1", "K8/g5_bp_B8", "K8/g5_bp_B16", "K4/g5_bp_B8", "K4/g5_bp_B16", "K8/g5_r1_m26",
            "K8/g5_bp_m27", "K8/g5_bp_m22"} <= blocks  # 27: the required control; 22: the tiny-mode probe


def test_reference_dumps_are_reused(port_run, val_root):
    from gate import common, run_gate

    res = run_gate.run_all(_ctx(val_root, checks=("g3",), arms=("K8", "K4")))  # the port run's build and work dir
    assert res.exit_code == 3  # partial: never PASS
    p2 = common.read_json(_phase_file(res, "2a"))
    # No exp_036 dump exists for a tiny build: R1 was computed under $EXP036_WORK/exp037 and is reused there.
    assert p2["data"]["dump"]["reused"] is True and p2["data"]["dump"]["registered_key"] is False
    assert p2["data"]["dump"]["dir"].startswith("$EXP036_WORK/exp037/ref/")
    assert res.record["phase_dir"] == f"aborted/{res.record['utc']}-gate"
    assert (_phase_file(res, "2a").parent / "NOTE.md").is_file()


def _target_id(target: str) -> list[str]:
    from gate import run_gate

    return [target] if target in run_gate.BLOCKING["K8"] else [f"{target}_K8", f"{target}_source"]


def test_pattern5_baseline_fails_only_where_documented(pattern5_baseline):
    """The unmutated port on pattern5 (g0, g2; K8). Under the exp_037 rules only G2's bf16 natural-selection
    leg (p) may fire there (DESIGN §11.13; the reason the port run moved to the validation build). The K8
    FAIL and exit 1 of the mutant runs below therefore say nothing alone; their target check, which passes
    here, is what each mutant must fail."""
    rec = pattern5_baseline.record
    assert rec["error"] is None, rec["error"]
    assert set(rec["failing"]["K8"]) <= {"g2_bf16_natural_selection"}, rec["failing"]["K8"]
    if rec["failing"]["K8"]:
        legs = rec["checks"]["g2_bf16_natural_selection"]["legs"]
        assert legs["a"]["ok"] and not legs["p"]["ok"], legs


@pytest.mark.parametrize("mutant", [
    "sigmoid_bias_select", "window_512", "rope_on_full", "one_plus_w_norm", "swap_sandwich_norms", "renorm_topk",
    "route_scale_2826", "mup_embed_scale", "attn_output_gate", "bf16_router_logits", "bf16_head_logits",
    "rope_traditional", "qknorm_after_rope", "biased_weights", "swiglu_swapped"])
def test_tiny_gate_fails_on_each_port_mutant(pattern5_baseline, tiny_root, mutant):
    from gate import port_mutants, run_gate

    targets = _target_id(port_mutants.TARGETS[mutant])
    base = pattern5_baseline.record
    present = [t for t in targets if t in base["checks"]]
    assert present and all(base["checks"][t]["pass"] is True for t in present), (
        mutant, {t: base["checks"][t].get("state") for t in present})  # the unmutated port passes the target
    res = run_gate.run_all(_ctx(tiny_root, mutant=mutant, checks=("g0", "g2"), arms=("K8",)))
    rec = res.record
    assert rec["tiny_outcome"]["K8"] == "FAIL", (mutant, rec["error"], rec["failing"])
    assert res.exit_code == 1
    assert any(t in rec["failing"]["K8"] for t in targets), rec["failing"]["K8"]
    assert rec["preconditions"]["P5"]["ok"]  # P5 tests the harness on the unmutated port


def test_exit_codes_on_constructed_cases():
    """DESIGN §3.16 (gate/rules.py, re-exported by run_gate): first match wins."""
    from gate import run_gate

    assert run_gate.exit_code_for({"K8": "PASS", "K4": "PASS"}) == 0
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "FAIL"}) == 4
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "PASS"}) == 1
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "INCOMPLETE"}) == 1
    assert run_gate.exit_code_for({"K8": "INCOMPLETE", "K4": "PASS"}) == 5
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "INCOMPLETE"}) == 5
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "MISSING"}) == 3
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "PASS"}, precondition_failed=True) == 3
    assert run_gate.exit_code_for({"K8": "PASS", "K4": "PASS"}, error="phase 4: crash") == 3
    assert run_gate.exit_code_for({"K8": "FAIL", "K4": "PASS"}, error="phase 6: crash") == 1
    ok = {"state": "PASS", "pass": True, "applicable": True}
    checks = {cid: dict(ok) for cid in run_gate.BLOCKING["K8"] + run_gate.BLOCKING["K4"]}
    assert run_gate.arm_verdict("K8", checks, False)["verdict"] == "PASS"
    checks["g2q_K4"] = {"state": "FAIL", "pass": False}
    v = run_gate.arm_verdict("K4", checks, False)
    assert (v["verdict"], v["failing"]) == ("FAIL", ["g2q_K4"])
    assert run_gate.arm_verdict("K8", checks, False)["verdict"] == "PASS"  # G2q at 4 bits is not a K8 check
    checks["g4_f16_K8"] = {"state": "INCOMPLETE", "pass": None, "reason": "VOID: R3 above ceiling (T9)"}
    v = run_gate.arm_verdict("K8", checks, False)
    assert (v["verdict"], v["incomplete"]) == ("INCOMPLETE", ["g4_f16_K8"])
    checks["g3_ref_mutants"] = {"state": "MISSING", "pass": None}
    assert run_gate.arm_verdict("K8", checks, False)["verdict"] == "MISSING"  # a missing value outranks INCOMPLETE
    checks["g3_ref_mutants"] = {"state": "MISSING", "pass": None, "applicable": False}
    assert run_gate.arm_verdict("K8", checks, True)["verdict"] == "INCOMPLETE"  # tiny: not applicable is skipped
    assert run_gate.arm_verdict("K8", checks, False)["verdict"] == "MISSING"  # real: never skipped
    del checks["g5_behaviour_K8"]
    assert run_gate.arm_verdict("K8", checks, True)["missing"] == ["g5_behaviour_K8"]


def test_partial_run_is_never_pass(tiny_root):
    from gate import run_gate

    res = run_gate.run_all(_ctx(tiny_root, checks=("g0",)))
    assert res.record["tiny_outcome"] == {"K8": "MISSING", "K4": "MISSING"}
    assert res.exit_code == 3
    assert [p["phase"] for p in res.record["phases"]] == ["0", "1", "4", "5", "7"]


def test_missing_directory_could_not_run(tiny_root, tmp_path):
    from gate import run_gate

    root = tmp_path / "models"
    root.mkdir()
    for name in ("Kolibri-1-BF16", "Kolibri-1-MLX-8bit-g64"):
        (root / name).symlink_to(tiny_root / name)
    res = run_gate.run_all(_ctx(root, builds_dir=root))
    assert res.exit_code == 3 and "Kolibri-1-MLX-4bit-g64" in res.record["error"]


def test_arm_directories_resolve_under_the_builds_dir(tiny_root, tmp_path):
    """DESIGN §2.8: GateContext.arm_dir is builds_dir()/Kolibri-1-MLX-{8,4}bit-g64; the
    BF16 source stays under the models dir."""
    from gate import run_gate

    models, builds = tmp_path / "models", tmp_path / "builds"
    models.mkdir()
    builds.mkdir()
    (models / "Kolibri-1-BF16").symlink_to(tiny_root / "Kolibri-1-BF16")
    for name in ("Kolibri-1-MLX-8bit-g64", "Kolibri-1-MLX-4bit-g64"):
        (models / name).symlink_to(tiny_root / name)  # exp_036's place: must not be used
    ctx = _ctx(models, builds_dir=builds)
    assert ctx.arm_dir("K8") == builds / "Kolibri-1-MLX-8bit-g64" and ctx.bf16_dir == models / "Kolibri-1-BF16"
    res = run_gate.run_all(ctx)
    assert res.exit_code == 3 and "Kolibri-1-MLX-8bit-g64 is missing" in res.record["error"]


def test_stale_port_copy_could_not_run(tiny_root, tmp_path):
    from gate import run_gate

    root = tmp_path / "models"
    root.mkdir()
    (root / "Kolibri-1-BF16").symlink_to(tiny_root / "Kolibri-1-BF16")
    (root / "Kolibri-1-MLX-4bit-g64").symlink_to(tiny_root / "Kolibri-1-MLX-4bit-g64")
    shutil.copytree(tiny_root / "Kolibri-1-MLX-8bit-g64", root / "Kolibri-1-MLX-8bit-g64")
    with open(root / "Kolibri-1-MLX-8bit-g64" / "kolibri1.py", "a") as f:
        f.write("\n# edited by hand\n")
    res = run_gate.run_all(_ctx(root, builds_dir=root))
    assert res.exit_code == 3 and "refresh-port-file" in res.record["error"]


def test_too_little_disk_could_not_run(tiny_root, tmp_path, monkeypatch):
    from gate import run_gate

    monkeypatch.setattr(run_gate.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10, 10, 0))
    res = run_gate.run_all(_ctx(tiny_root, work_dir=tmp_path / "work", checks=("g2",)))
    assert res.exit_code == 3 and "GB free" in res.record["error"]


def test_a_failed_precondition_stops_the_run(tiny_root, tmp_path):
    """DESIGN §3.1: P4 refuses a build whose shard differs from its convert record; the run
    stops at phase 0 (exit 3, not a gate run), the record is written and the partial
    outputs go to aborted/<UTC>-gate/ with a NOTE.md."""
    from gate import run_gate

    root = tmp_path / "models"
    root.mkdir()
    (root / "Kolibri-1-BF16").symlink_to(tiny_root / "Kolibri-1-BF16")
    (root / "Kolibri-1-MLX-4bit-g64").symlink_to(tiny_root / "Kolibri-1-MLX-4bit-g64")
    shutil.copytree(tiny_root / "Kolibri-1-MLX-8bit-g64", root / "Kolibri-1-MLX-8bit-g64")
    shard = sorted((root / "Kolibri-1-MLX-8bit-g64").glob("model*.safetensors"))[0]
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0xFF
    shard.write_bytes(bytes(data))
    res = run_gate.run_all(_ctx(root, builds_dir=root))
    rec = res.record
    assert res.exit_code == 3 and rec["precondition_failed"] and rec["failed_preconditions"] == ["P4"]
    assert [p["phase"] for p in rec["phases"]] == ["0", "7"]
    assert rec["tiny_outcome"] == {"K8": "MISSING", "K4": "MISSING"} and rec["blind_phase_reached"] is False
    assert "shard sha256 differs" in rec["preconditions"]["P4"]["reason"]
    aborted = root / "aborted" / f"{rec['utc']}-gate"
    assert rec["phase_dir"] == f"aborted/{rec['utc']}-gate" and (aborted / "phase0.json").is_file()
    note = (aborted / "NOTE.md").read_text(encoding="utf-8")
    assert "P4" in note and "not a gate run" in note
    assert not (root / "results" / "gate" / rec["utc"]).exists() and res.path.is_file()


def test_cli_last_line_is_json(tiny_root):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [sys.executable, str(EXP / "gate" / "run_gate.py"), "--tiny", str(tiny_root), "--checks", "g0",
           "--arms", "K8", "--head-policy", "quantised_head", "--g0k-rows", "none", "--subprocess-phases"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(EXP), timeout=600)
    last = json.loads(p.stdout.strip().splitlines()[-1])
    assert p.returncode == last["exit_code"] == 3, p.stderr[-2000:]  # partial
    assert last["verdict"] == {"K8": "TINY", "K4": "TINY"} and last["error"] is None


def test_real_mode_options_refused():
    from gate import run_gate

    with pytest.raises(SystemExit, match="tiny"):
        run_gate.main(["--checks", "g0", "--port-mutant", "window_512"])
    with pytest.raises(SystemExit, match="tiny"):
        run_gate.main(["--checks", "g0", "--g0k-rows", "6"])
    with pytest.raises(SystemExit, match="tiny"):
        run_gate.main(["--checks", "g0", "--tiny-precision", "bf16"])


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


def test_records_hold_no_host_name_or_absolute_path(port_run, val_root, tiny_root):
    """BUILD_SPEC §2 Privacy: no hostname, serial or home path in anything the gate
    writes under results/ or aborted/ (junit XML and the G0k, G5 and R1 work files
    stay under $EXP036_WORK/exp037), on both builds this module runs the gate on."""
    import re
    import socket

    host = socket.gethostname().split(".")[0]
    dirs = [r / sub for r in (val_root, tiny_root) for sub in ("results/gate", "aborted") if (r / sub).is_dir()]
    assert val_root / "results" / "gate" in dirs  # the port run's record
    seen = 0
    for d in dirs:
        for p in d.rglob("*"):
            if p.is_file():
                seen += 1
                text = p.read_text(encoding="utf-8")
                assert host not in text, p.name
                assert not re.search(r"/(Users|home|private|var/folders)/", text), p.name
                assert not re.search(r'"[^"]*(serial|uuid|hostname|computername)[^"]*"\s*:', text, re.I), p.name
        assert not list(d.rglob("*.xml"))
    assert seen
