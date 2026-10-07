# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W7).
"""The whole exp_037 gate --tiny on the tiny real-layout validation build (DESIGN §12 W7,
§4.2, §5.1). Heavy (about 5 minutes on the mini): left out of the gate's G1
(gate/checks/g1_synthetic.EXCLUDE); the mini's full suite runs it before the freeze (W14),
and tools/dry_run.py's gate stage runs the same command.

The build is conftest.py's tiny_real_val_unsharp (tests/tiny_real_layout.py
build_gate_validation_set: seed 29 with the q/k norms unsharpened, sharpen = 1, converted to
8 and 4 bits by port/convert.py), copied, so the shared fixture is never written to. It is
the validation build of the whole tiny gate (DESIGN §5.1; decision (a), 2026-10-06): every
check with a quantised or bf16 arm (G4-N(i), G4-F16 with R3, G4 K4, G5-R1, G5-BP-lean,
behaviour, the noise floor, free decode-vs-prefill) is judged on it, and so are the same run's
fp32 forced checks (G4-F32, G5-D32) and every control margin. The sharpened seed-29 build
(tiny_real_val) stays F_tiny's validation (tests/test_gate_ftiny.py); on it the x 3 q/k
sharpening makes 8-bit quantisation of random tiny weights unrepresentative, and the whole
gate's failures there are DESIGN §11.13's stress finding, not this test's.
The gate runs as `gate/run_gate.py --tiny DIR --head-policy quantised_head` does: every phase
(0 with G0k at the §3.1 P3 shapes, 1, 2a, 2b, 2c, 3, 4 with the control mutants and the
tiny-mode probe, 5, 6, 7), the tiny profile of gate/textset.py (DESIGN §5.1), F_tiny from
gate/calibration.json.

It must:
  * PASS for K8 and K4 (exit 0), every blocking check passing or not applicable on tiny
    weights (run_gate.TINY_NOT_APPLICABLE), with G5-R1's R-anchor and R-parity holding;
  * catch every required control of gate/controls.json by its bound leg, with the tiny
    margin of DESIGN §4.2 in that leg's units (mean and max legs >= 10 x the bound, ratio
    legs >= 10 x the factor, count legs >= 10), and the registered G2 mutants. The one
    exception is G5-BP-lean's control G5BP/27, which meets decision (e)'s acceptance
    instead (DECISION_E below; DESIGN §4.2, §14.3): (d) caught on both subsets at the
    gate's own factor, with the fixture ceiling recorded beside it;
  * report probe 22 (controls.json "probes", tiny mode only) in the record's "probes",
    with (d)'s ratio per subset and no margin, and keep it out of allowed_B.
Decision (e)'s acceptance lives here, in this test, and in the tiny control table; it is
not a gate rule: gate/controls.json and gate/rules.py are unchanged by it, and the gate
record still writes G5BP/27's §4.2 margin (required 30) as rules.py computes it.
The control table (control_table) is what W14 records as
tests/fixtures/exp037_tiny_controls.json; set EXP037_CONTROL_TABLE=<path> to have the
test write it there.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1]
ALL_PHASES = ["0", "1", "2a", "2b", "2c", "3", "4", "5", "6", "7"]

# Decision (e) (DESIGN §4.2, §14.3; the main session, 2026-10-06, before the freeze, after
# mutant 27 was measured at 11.93 / 6.41): for G5-BP-lean's required control, and for that
# ratio leg only, the tiny acceptance is (d) caught on BOTH subsets (first wave and mid-run)
# at the gate's own factor (ratio > thresholds G5BP anchor_factor = 3) on this build, with
# the fixture ceiling recorded beside it. It replaces §4.2's 10 x factor (30) for this
# control alone; every other control keeps its §4.2 margin. Reason: the build's next-token
# distributions are near-uniform, so no batched-path defect acting through attention can
# reach 30 on it (the ceiling, below). It is a pre-freeze code check, not a gate threshold:
# G5-BP-lean never fails the gate, and the gate run re-tests 27 on real weights (uncaught
# there: allowed_B = {1}, Decision 8 unchanged).
DECISION_E = {"G5BP/27": ("decision (e), 2026-10-06 (DESIGN §4.2, §14.3): (d) caught on both subsets at the "
                          "gate's own factor (ratio > G5BP anchor_factor) on the unsharpened seed-29 build, "
                          "with the fixture ceiling recorded beside it; replaces the 10 x factor margin for "
                          "this control only")}
# The fixture ceiling (FixB's measure, 2026-10-06): CEILING_PAIRS random pairs of generated
# positions from different sequences of the unmutated K8 B = 8 G5-BP-lean block's single
# log-prob files (files in path order, np.random.default_rng(CEILING_SEED)).
CEILING_BLOCK, CEILING_KIND = "K8/g5_bp_B8", "single"
CEILING_PAIRS, CEILING_SEED = 4000, 0


def decision_e_acceptance(r: dict, control: dict, th: dict) -> dict:
    """Decision (e)'s tiny acceptance for a G5-BP-lean control result `r` (the gate
    record's controls entry, values = rules.bp_d's legs): on every subset of the
    control's "where", n > 0 and the ratio mean KL(R1||batched) / mean KL(R1||single)
    above the gate's own factor, where rules.py's (d) leg fires (ok False)."""
    factor = float(th["G5BP"]["anchor_factor"])
    vals = r.get("values") or {}
    legs = {}
    for sub in control["where"]:
        v = vals.get(sub) or {}
        n, ratio = int(v.get("n") or 0), v.get("ratio")
        unbounded = ratio is None and (v.get("mean_kl_r1_batched") or 0.0) > 0.0
        above = unbounded or (ratio is not None and ratio > factor)
        fires = v.get("ok") is False
        legs[sub] = {"n": n, "ratio": ratio, "fires": fires, "caught": bool(n > 0 and above and fires)}
    return {"rule": DECISION_E[control["id"]], "factor": factor, "legs": legs,
            "met": bool(r.get("caught") is True and legs and all(x["caught"] for x in legs.values()))}


def fixture_ceiling(rec: dict, work37: Path) -> dict:
    """The ratio an unrelated-context swap reaches on this build (decision (e)): the mean
    KL(p_i || p_j) (common.kl_rows) over CEILING_PAIRS pairs of generated positions i, j from
    different sequences of the unmutated K8 B = 8 block's single log-prob files, divided by
    G5BP/27's mean KL(R1 || single) per subset (its (d) denominator). A defect that replaced
    a row's context by an unrelated one would score about this. Every file is loaded after
    its sha256 matches the record's g5_logprob_files entry."""
    from gate import common

    files = sorted((f for f in rec["g5_logprob_files"]["files"]
                    if f["block"] == CEILING_BLOCK and f["kind"] == CEILING_KIND), key=lambda f: f["path"])
    if not files:
        raise AssertionError(f"no {CEILING_KIND} log-prob files of {CEILING_BLOCK} in the record")
    arrays = []
    for f in files:
        p = Path(work37) / f["path"]
        if common.sha256_file(p) != f["sha256"]:
            raise AssertionError(f"{f['path']}: sha256 differs from the record's")
        arrays.append(np.load(p).astype(np.float64))
    arrays = [a for a in arrays if a.shape[0]]
    rows = np.concatenate(arrays)
    seq_of = np.concatenate([np.full(a.shape[0], i) for i, a in enumerate(arrays)])
    rng = np.random.default_rng(CEILING_SEED)
    pi, pj = [], []
    while len(pi) < CEILING_PAIRS:
        i, j = rng.integers(0, len(rows), 2)
        if seq_of[i] != seq_of[j]:
            pi.append(int(i)), pj.append(int(j))
    kl = common.kl_rows(rows[pi], rows[pj])
    lp = common.log_softmax64(rows)
    entropy = -(np.exp(lp) * lp).sum(axis=-1)
    vals = (rec["controls"].get("G5BP/27") or {}).get("values") or {}
    den = {sub: (vals.get(sub) or {}).get("mean_kl_r1_single") for sub in ("first_wave", "mid_run")}
    mean = float(kl.mean())
    return {"what": ("the (d) ratio of an unrelated-context swap: mean KL(p_i || p_j) over random pairs of "
                     "generated positions from different sequences, divided by G5BP/27's mean KL(R1 || single)"),
            "block": CEILING_BLOCK, "kind": CEILING_KIND, "files": len(files), "positions": int(rows.shape[0]),
            "vocab": int(rows.shape[1]), "pairs": CEILING_PAIRS, "seed": CEILING_SEED,
            "entropy_nats": {"mean": float(entropy.mean()), "log_vocab": float(math.log(rows.shape[1]))},
            "pair_kl": {"mean": mean, "median": float(np.median(kl)), "p90": float(np.quantile(kl, 0.9)),
                        "max": float(kl.max())},
            "single_denominator": den,
            "ratio": {sub: (mean / d if d else None) for sub, d in den.items()}}


def control_table(rec: dict, ceiling: dict | None = None) -> dict:
    """The tiny control table of DESIGN §4.2 from a gate record: per required control its
    check, bound leg, caught, and its margin {unit, value, required, met} as gate/rules.py
    computes it; for G5BP/27 also decision (e)'s acceptance and the fixture ceiling; the
    probes with their reported values and no margin."""
    import tiny_real_layout as trl

    from gate import common, rules

    th = common.load_thresholds()
    build = (f"tests/tiny_real_layout.py seed {trl.SEED_VAL}, q/k norms x {trl.SHARPEN_GATE:g} (unsharpened; "
             f"DESIGN §5.1, decision (a)), K8 at 8 bits, group {trl.GROUP_SIZE}")
    ctl = rules.load_controls()
    out = {"schema": "exp037 tiny control table v1", "record": f"gate_{rec['utc']}.json", "build": build,
           "margin_factor": ctl["margin_factor"], "controls": {}, "probes": {}}
    for c in ctl["required"]:
        r = rec["controls"].get(c["id"]) or {}
        row = {"check": c["check"], "mutant": c["mutant"], "mutant_name": c["mutant_name"], "leg": c["leg"],
               "caught": r.get("caught"), "margin": r.get("margin")}
        if c["id"] in DECISION_E:
            row["tiny_acceptance"] = decision_e_acceptance(r, c, th)
            row["fixture_ceiling"] = ceiling
        else:
            row["tiny_acceptance"] = {"rule": "DESIGN §4.2 margin", "met": bool((r.get("margin") or {}).get("met"))}
        out["controls"][c["id"]] = row
    for pid, p in (rec.get("probes") or {}).items():
        out["probes"][pid] = {k: p.get(k) for k in ("check", "mutant", "mutant_name", "leg", "measured", "fires",
                                                    "ratio", "margin")}
        out["probes"][pid]["note"] = "probe, no margin (DESIGN §4.2)"
    reg = rec["controls"].get("G2/registered") or {}
    out["registered_g2"] = {"caught": reg.get("caught"), "undetected": reg.get("undetected")}
    return out


def _tok_dir():
    """The Kolibri tokenizer for G0's template and tokenizer parity when the environment names one
    ($EXP036_TOK, as tools/dry_run.py's --tok); without it those two checks are not applicable."""
    for name in ("EXP036_TOK", "EXP036_TOKENIZER_DIR"):
        v = os.environ.get(name)
        if v and (Path(v).expanduser() / "tokenizer.json").is_file():
            return Path(v).expanduser()
    return None


@pytest.fixture(scope="module")
def e2e(tiny_real_val_unsharp, tmp_path_factory):
    """(record, record path, models dir, control table): the whole tiny gate on a copy of the
    unsharpened seed-29 validation build (DESIGN §5.1, decision (a))."""
    import tiny_real_layout as trl

    from gate import common, run_gate

    s = tiny_real_val_unsharp
    assert (s.seed, s.sharpen) == (trl.SEED_VAL, trl.SHARPEN_GATE) == (29, 1.0), (s.seed, s.sharpen)
    root = tmp_path_factory.mktemp("e2e_val")
    for name in (trl.BF16_NAME, trl.build_dir_name(8), trl.build_dir_name(4)):
        shutil.copytree(s.root / name, root / name)
    ctx = run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=common.load_thresholds(), tiny=True,
                               results_dir=root / "results", tok_dir=_tok_dir(), head_policy="quantised_head",
                               g1_select=["test_routing.py"], subprocess_phases=False, log_stream=io.StringIO(),
                               builds_dir=root)
    res = run_gate.run_all(ctx)
    ceiling = None
    if res.record.get("error") is None and (res.record["controls"].get("G5BP/27") or {}).get("values"):
        ceiling = fixture_ceiling(res.record, root / "work" / "exp037")
    table = control_table(res.record, ceiling)
    out = os.environ.get("EXP037_CONTROL_TABLE")
    if out:
        Path(out).write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return res.record, res.path, root, table


def test_the_whole_tiny_gate_passes_on_the_validation_build(e2e):
    from gate import run_gate

    rec, path, _, _ = e2e
    assert rec["error"] is None, rec["error"]
    assert [p["phase"] for p in rec["phases"]] == ALL_PHASES
    assert rec["tiny_outcome"] == {"K8": "PASS", "K4": "PASS"}, {
        k: rec[k] for k in ("failing", "incomplete", "missing", "reasons")}
    assert rec["exit_code"] == 0
    for arm in ("K8", "K4"):
        for cid in run_gate.BLOCKING[arm]:
            c = rec["checks"][cid]
            assert c["pass"] is True or c.get("applicable") is False, (cid, c.get("state"), c.get("reason"))
    legs = rec["checks"]["g5_r1_K8"]["legs"]
    assert legs["R-anchor"]["ok"] and legs["R-parity"]["ok"], legs
    assert rec["checks"]["g5_r1_K4"]["legs"]["R-anchor"]["ok"]
    assert all(rec["preconditions"][k]["ok"] for k in ("P1", "P2", "P3", "P4", "P5"))
    assert rec["blind_phase_reached"] is True


def test_every_required_control_is_caught_with_its_tiny_margin(e2e):
    """DESIGN §4.2: each control fails its bound leg by the margin of its leg type, except
    G5BP/27, which meets decision (e)'s acceptance (§4.2, §14.3): (d) caught on both the
    first wave and mid-run at the gate's own factor (> 3), with the fixture ceiling (the
    ratio an unrelated-context swap reaches on this build) recorded beside it."""
    rec, _, _, table = e2e
    assert set(table["controls"]) >= set(DECISION_E)
    problems = {cid: (c["caught"], c["tiny_acceptance"], c["margin"]) for cid, c in table["controls"].items()
                if c["caught"] is not True or not c["tiny_acceptance"]["met"]}
    assert not problems, problems
    for cid, c in table["controls"].items():
        if cid not in DECISION_E:  # every other control keeps its §4.2 margin, as rules.py computes it
            assert c["margin"]["met"] is True, (cid, c["margin"])
    assert table["registered_g2"]["caught"] is True and table["registered_g2"]["undetected"] == []
    bp = table["controls"]["G5BP/27"]
    ceiling = bp["fixture_ceiling"]
    assert ceiling is not None and ceiling["pairs"] == CEILING_PAIRS
    assert all(math.isfinite(ceiling["ratio"][s]) and ceiling["ratio"][s] > 0 for s in ("first_wave", "mid_run"))
    summary = {"G5BP/27": {"ratio": {s: v["ratio"] for s, v in bp["tiny_acceptance"]["legs"].items()},
                           "factor": bp["tiny_acceptance"]["factor"], "acceptance_met": bp["tiny_acceptance"]["met"],
                           "margin_4_2": bp["margin"], "fixture_ceiling": ceiling["ratio"],
                           "fixture_pair_kl_mean": ceiling["pair_kl"]["mean"],
                           "fixture_entropy_mean": ceiling["entropy_nats"]["mean"],
                           "log_vocab": ceiling["entropy_nats"]["log_vocab"]}}
    print("\n[decision (e)] " + json.dumps(summary, sort_keys=True))


def test_probe_22_is_reported_without_a_margin_and_enters_nothing(e2e):
    """DESIGN §3.16, §4.2: probe 22 runs in tiny mode at K8 B = 8 and is reported under the
    record's "probes" with (d)'s ratio per subset, whether (d) fires, and no margin; it is
    not a control, and allowed_B is what rules.g5_bp gives without it."""
    from gate import common, rules

    rec, path, _, table = e2e
    p = rec["probes"]["G5BP/22"]
    assert p["measured"] is True and p["margin"] is None and "error" not in p, p
    assert (p["check"], p["mutant"], p["leg"], p["modes"]) == ("g5_bp_K8_B8", 22, "d", ["tiny"])
    assert set(p["ratio"]) == {"first_wave", "mid_run"} and all(p["ratio"][s] > 0 for s in p["ratio"])
    assert isinstance(p["fires"], bool)
    assert "G5BP/22" not in rec["controls"] and "G5BP/22" not in table["controls"]
    blocks = {f["block"] for f in rec["g5_logprob_files"]["files"]}
    assert {"K8/g5_bp_m27", "K8/g5_bp_m22"} <= blocks
    red = common.read_json(path.parent / rec["utc"] / "phase6.json")["data"]["reduced"]
    assert set(red["g5_bp_mutants"]) == {"22", "27"}
    without = rules.g5_bp(red["g5_bp"], {"27": red["g5_bp_mutants"]["27"]}, common.load_thresholds())
    assert rec["allowed_B"] == without["allowed_B"] and rec["allowed_B_power_note"] == without["power_note"]
    print("\n[probe 22] " + json.dumps({"ratio": p["ratio"], "fires": p["fires"]}, sort_keys=True))


def test_g4_f32_uses_f_tiny_and_the_same_run_csort(e2e):
    """tau = max(F_tiny, 100 x mean Csort) per set (DESIGN §3.7), F_tiny from gate/calibration.json."""
    from gate.checks import ftiny

    rec, _, _, _ = e2e
    cal = rec["calibrators"]
    f = ftiny.load_f_tiny()
    assert cal["f_tiny"] == f
    for s in ("T1-8", "T9"):
        assert cal["tau"][s] == max(f, 100.0 * cal["csort_sets"][s]["mean"])
        assert cal["csort_sets"][s]["mean"] <= 1e-6


def test_g0k_ran_at_every_judged_shape(e2e):
    from gate import common, rules

    rec, _, _, _ = e2e
    p3 = rec["preconditions"]["P3"]["values"]
    assert p3["n_judged"] == len(rules.p3_judged_shapes(common.load_thresholds())) <= p3["measured"]
    assert p3["absent"] == [] and p3["failing"] == []


def test_every_log_prob_file_exists_with_its_sha256(e2e):
    """DESIGN §3.13-§3.14: every file named in the record exists under $EXP036_WORK/exp037 with its sha256."""
    from gate import common

    rec, _, root, _ = e2e
    files = rec["g5_logprob_files"]["files"]
    assert files and rec["g5_logprob_files"]["root"] == "$EXP036_WORK/exp037"
    for f in files:
        p = root / "work" / "exp037" / f["path"]
        assert p.is_file() and common.sha256_file(p) == f["sha256"], f["path"]
