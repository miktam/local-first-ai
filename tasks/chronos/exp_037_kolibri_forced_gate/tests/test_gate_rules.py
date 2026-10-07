# SPDX-License-Identifier: MIT
"""gate/rules.py: every gate decision as a pure function (DESIGN §3, §4.2, §7.2, §9.2; G1 item 9; W6).

* every rule at its boundary: equality passes and one ulp beyond fails (np.nextafter).
  Definitions written with ">=" in DESIGN §3 (an R2F-decisive position, lead >= 2
  nats; F6's gap >= 1e-2; behaviour's ">= 2 / 20") qualify at equality, as stated;
* NaN never passes;
* the exit table of §3.16, first match wins, including G1 FAIL followed by a P5
  failure (exit 3), and the arm verdict's precedence FAIL > MISSING > INCOMPLETE;
* gate runs and fix cycles (§9.4);
* allowed_B (§3.14);
* the controls map (controls.json, §4.2): each required control is caught only by
  its own bound leg, judged with the unmutated run's calibration, and its tiny
  margin is measured in its leg type's unit;
* the tripwire constants and predicate (§7.2);
* rules.py is pure: no mlx, port, gate.common or seed draw.
"""

from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gate import rules as R

EXP = Path(__file__).resolve().parents[1]
UP, DOWN = math.inf, -math.inf


def up(x: float) -> float:
    return float(np.nextafter(x, UP))


def down(x: float) -> float:
    return float(np.nextafter(x, DOWN))


@pytest.fixture(scope="module")
def th() -> dict:
    return R.load_thresholds()


@pytest.fixture(scope="module")
def ctl() -> dict:
    return R.load_controls()


# ------------------------------------------------------------------------------- purity and ids


def test_rules_module_is_pure():
    src = (EXP / "gate" / "rules.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "json", "math", "numbers", "pathlib", "numpy"}, mods
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(tree) if isinstance(n, (ast.Attribute, ast.Name))}
    for bad in ("seed_from", "random", "default_rng", "write_text", "open", "mkdir", "unlink"):
        assert bad not in names, bad


def test_blocking_lists(th):
    k8, k4 = R.BLOCKING["K8"], R.BLOCKING["K4"]
    assert len(k8) == len(set(k8)) and len(k4) == len(set(k4))
    for cid in ("g4_f32_K8", "g4_f16_K8", "g4_n_K8", "g5_d32_K8", "g5_r1_K8", "g3_ref_mutants", "g2_bf16_natural_selection"):
        assert cid in k8
    assert "g5_r1_K4" in k4 and "same_port_sha" in k4 and "g2q_K4" in k4 and "g5_behaviour_K4" in k4
    assert not set(R.NOT_BLOCKING) & (set(k8) | set(k4))
    for cid in ("g3_bpb_per_text", "g3_bpb_vs_peers", "g4_K4", "g5_bp_K8_B8", "g5_bp_K4_B16"):
        assert cid in R.NOT_BLOCKING
    assert R.group_of("same_port_sha") == "g0" and R.group_of("g4_f32_K8") == "g4" and R.group_of("g2q_K4") == "g2"
    assert R.arms_of("g0_census") == ["K8", "K4"] and R.arms_of("g5_r1_K4") == ["K4"]


def test_check_entries_and_states():
    for s, p in ((R.PASS, True), (R.FAIL, False), (R.INCOMPLETE, None), (R.MISSING, None), (R.DESCRIPTIVE, None)):
        c = R.check(s, {"x": 1}, 2)
        assert c["state"] == s and c["pass"] is p and R.state_of(c) == s
    assert R.check(R.DESCRIPTIVE)["descriptive"] is True
    with pytest.raises(ValueError):
        R.check("OK")
    # exp_036-style entries
    assert R.state_of({"pass": True}) == R.PASS and R.state_of({"pass": False}) == R.FAIL
    assert R.state_of({"pass": None}) == R.MISSING and R.state_of(None) == R.MISSING
    assert R.state_of({"pass": None, "descriptive": True}) == R.DESCRIPTIVE


def test_tail():
    t = R.tail(np.array([0.5, 3.0, np.nan, 2.0, 1.0]), k=3)
    assert t["n"] == 5 and math.isnan(t["top"][0]) and t["top"][1:] == [3.0, 2.0]
    t = R.tail([1.0, 2.0])
    assert t == {"n": 2, "top": [2.0, 1.0]} and R.tail([]) == {"n": 0, "top": []}
    assert R.tail_count_at_least({"n": 3, "top": [2.0, 2.0, down(2.0)]}, 2.0) == 2
    assert R.tail_count_above({"n": 2, "top": [up(0.01), 0.01]}, 0.01) == 1
    with pytest.raises(R.RuleInputError):
        R._tail({"n": 2, "top": [1.0, 2.0]}, "x")  # not descending
    with pytest.raises(R.RuleInputError):
        R._tail({"n": 1, "top": [2.0, 1.0]}, "x")  # longer than its count


# ------------------------------------------------------------------------------- G0, G1


def test_g0_boundaries(th):
    th0 = th["G0"]
    dt = {"router_logits_dtype": "float32", "router_bf16_exact_frac": 0.01, "head_dtype": "float32",
          "head_bf16_exact_frac": 0.01}
    r, h = R.g0_dtypes(dt, th0)
    assert r["state"] == h["state"] == R.PASS
    r, h = R.g0_dtypes(dict(dt, router_bf16_exact_frac=up(0.01), head_bf16_exact_frac=math.nan), th0)
    assert r["state"] == h["state"] == R.FAIL
    assert R.g0_dtypes(dict(dt, head_dtype="bfloat16"), th0)[1]["state"] == R.FAIL
    tk = {"n_lines": 2000, "mismatches": 0, "roundtrip_fail": 0, "fix_mistral_regex": False}
    assert R.g0_tokenizer_parity(tk, th0)["state"] == R.PASS
    assert R.g0_tokenizer_parity(dict(tk, n_lines=1999), th0)["state"] == R.FAIL
    assert R.g0_tokenizer_parity(dict(tk, mismatches=1), th0)["state"] == R.FAIL
    assert R.g0_tokenizer_parity(dict(tk, fix_mistral_regex=True), th0)["state"] == R.FAIL
    tp = {"n": 120, "mismatches": [], "runner_chat": {"mismatches": []}}
    assert R.g0_template_parity(tp, th0)["state"] == R.PASS
    assert R.g0_template_parity(dict(tp, n=119), th0)["state"] == R.FAIL
    assert R.g0_template_parity(dict(tp, runner_chat={"mismatches": ["x"]}), th0)["state"] == R.FAIL
    cen = {"tensors": 58353, "params": 78103074560, "full_attention_layers": th0["full_attention_layers"],
           "shard_sha": {"status": "checked", "mismatches": []}}
    exp = {k: th0[k] for k in ("tensor_count", "param_count", "full_attention_layers")}
    assert R.g0_census(cen, exp)["state"] == R.PASS
    assert R.g0_census(dict(cen, params=78103074561), exp)["state"] == R.FAIL
    assert R.g0_census(dict(cen, shard_sha={"status": "unavailable"}), exp)["state"] == R.FAIL
    assert R.g0_census(dict(cen, shard_sha={"status": "unavailable"}), exp, tiny=True)["state"] == R.PASS
    sl = {"n_unused_ckpt_tensors": 0, "n_missing_params": 0, "shape_mismatch": [], "load_error": None,
          "numel_equal": False, "quantized": True}
    assert R.g0_strict_load(sl)["state"] == R.PASS
    assert R.g0_strict_load(dict(sl, n_unused_ckpt_tensors=1))["state"] == R.FAIL
    assert R.g0_strict_load(dict(sl, quantized=False))["state"] == R.FAIL
    assert R.g0_converted_config({"n_problems": 0}, "quantised_head")["state"] == R.PASS
    assert R.g0_converted_config({"n_problems": 1}, "quantised_head")["state"] == R.FAIL
    assert R.same_port_sha("a", {"K8": "a", "K4": "a"})["state"] == R.PASS
    assert R.same_port_sha("a", {"K8": "a", "K4": "b"})["state"] == R.FAIL


def test_g1_rule(th):
    g = {"tests": 10, "failures": 0, "errors": 0, "skipped": 2, "skipped_build_host_only": 2}
    assert R.g1(g, th["G1"])["state"] == R.PASS
    assert R.g1(dict(g, skipped=3), th["G1"])["state"] == R.FAIL
    assert R.g1(dict(g, skipped=3), th["G1"], tiny=True)["state"] == R.PASS
    assert R.g1(dict(g, failures=1), th["G1"])["state"] == R.FAIL
    assert R.g1(dict(g, errors=1), th["G1"])["state"] == R.FAIL
    assert R.g1(dict(g, tests=0), th["G1"])["state"] == R.FAIL


# ------------------------------------------------------------------------------- G2


def _emu(median=0.01, p99=0.02, frac=0.005, m_emu=5.0, layers=1):
    return {"layers": [{"attn": {"median": median, "p99": p99}, "moe": {"median": median, "p99": p99},
                        "natural_disagree_frac": frac} for _ in range(layers)], "M_emu": m_emu}


def _g2_res(**over):
    row = {"layer": 0, "branch": "attn", "median": 1e-6, "p99": 1e-5, "max": 1e-5}
    res = {
        "layers_fp32": [dict(row), dict(row, branch="moe")],
        "natural_fp32": [{"layer": 0, "n": 100, "n_disagree": 0, "n_exempt": 0, "n_violations": 0, "exempt_gaps": []}],
        "embedding": {"exact": True, "max_abs": 0.0},
        "head": {"max_abs_dlogit": 1e-5},
        "layers_bf16": [dict(row, median=0.01, p99=0.02), dict(row, branch="moe", median=0.01, p99=0.02)],
        "natural_bf16": [{"layer": 0, "n": 1000, "n_disagree": 10, "sigma": 0.1, "max_gap_over_sigma": 3.0,
                          "n_beyond_6sigma": 0}],
        "layers_t9": [dict(row, median=0.01, p99=0.02, bucket=[0, 2048])],
        "router_margin": {"diff_p99": 0.01},
        "layers_g2q": {"K8": {"layers": [dict(row, median=0.01, p99=0.02)], "embedding_max_rel": 1e-3,
                              "head_rel_l2_max": 1e-4},
                       "K4": {"capability_missing": "no dequantised reference"}},
        "real_weight_mutants": {"layers": [0], "mutants": {}},
    }
    res.update(over)
    return res


def test_g2_all_pass_and_ids(th):
    c = R.g2_checks(_g2_res(), _emu(), th["G2"])
    assert set(c) == {"g2_fp32_forced_branch", "g2_fp32_natural_selection", "g2_embedding", "g2_head",
                      "g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_t9_buckets", "g2_router_margin", "g2q_K8"}
    assert all(v["state"] == R.PASS for v in c.values()), {k: v["state"] for k, v in c.items()}
    # without emulation statistics the bf16 rows and G2q are not judged
    assert set(R.g2_checks(_g2_res(), None, th["G2"])) == {"g2_fp32_forced_branch", "g2_fp32_natural_selection",
                                                          "g2_embedding", "g2_head"}


def test_g2_fp32_boundaries(th):
    th2 = th["G2"]
    for key, lim in (("median", 1e-4), ("max", 1e-3)):
        rows = [{"layer": 0, "branch": "attn", "median": 1e-6, "p99": 1e-5, "max": 1e-5}]
        rows[0][key] = lim
        assert R.g2_checks(_g2_res(layers_fp32=rows), None, th2)["g2_fp32_forced_branch"]["state"] == R.PASS
        rows[0][key] = up(lim)
        assert R.g2_checks(_g2_res(layers_fp32=rows), None, th2)["g2_fp32_forced_branch"]["state"] == R.FAIL
    res = _g2_res(head={"max_abs_dlogit": 1e-3})
    assert R.g2_checks(res, None, th2)["g2_head"]["state"] == R.PASS
    res = _g2_res(head={"max_abs_dlogit": up(1e-3)})
    assert R.g2_checks(res, None, th2)["g2_head"]["state"] == R.FAIL
    nat = [{"layer": 0, "n": 100, "n_disagree": 1, "n_exempt": 0, "n_violations": 1, "exempt_gaps": []}]
    assert R.g2_checks(_g2_res(natural_fp32=nat), None, th2)["g2_fp32_natural_selection"]["state"] == R.FAIL
    assert R.g2_checks(_g2_res(embedding={"exact": False, "max_abs": 1e-9}), None, th2)["g2_embedding"]["state"] == R.FAIL


def test_g2_bf16_forced_boundaries(th):
    th2 = th["G2"]
    emu = _emu(median=0.01, p99=0.01)
    lim_med = th2["bf16_forced_median_factor_vs_emu"] * 0.01
    lim_p99 = max(th2["bf16_forced_p99_floor"], th2["bf16_forced_p99_factor_vs_emu"] * 0.01)  # the 5e-2 floor
    assert lim_p99 == 0.05
    rows = [{"layer": 0, "branch": "moe", "median": lim_med, "p99": lim_p99, "max": 0.1}]
    ok, bad, v = R.bf16_rule(rows, emu["layers"], th2)
    assert ok and not bad and v[0]["pass"] and v[0]["limit_median"] == lim_med and v[0]["limit_p99"] == lim_p99
    for key, lim in (("median", lim_med), ("p99", lim_p99)):
        r = [dict(rows[0], **{key: up(lim)})]
        ok, bad, v = R.bf16_rule(r, emu["layers"], th2)
        assert not ok and bad == [{"layer": 0, "branch": "moe", "median": r[0]["median"], "p99": r[0]["p99"]}]
    emu_big = _emu(median=0.01, p99=0.03)  # 3 x 0.03 above the floor
    lim = th2["bf16_forced_p99_factor_vs_emu"] * 0.03
    assert R.bf16_rule([dict(rows[0], p99=lim)], emu_big["layers"], th2)[0]
    assert not R.bf16_rule([dict(rows[0], p99=up(lim))], emu_big["layers"], th2)[0]
    t9 = [dict(rows[0], bucket=[8192, 16384], median=up(lim_med))]
    c = R.g2_checks(_g2_res(layers_t9=t9), emu, th2)["g2_t9_buckets"]
    assert c["state"] == R.FAIL and c["measured"]["failing"][0]["bucket"] == [8192, 16384]


def test_g2_bf16_natural_selection_legs(th):
    th2 = th["G2"]
    # (a): the fraction against max(0.2 %, 2 x emu's fraction); emu 0.005 -> limit 0.01 = 10 / 1000
    nat = [{"layer": 0, "n": 1000, "n_disagree": 10, "sigma": 0.1, "max_gap_over_sigma": 1.0, "n_beyond_6sigma": 0}]
    c = R.g2_bf16_natural_selection({"natural_bf16": nat}, _emu(frac=0.005), th2)
    assert c["state"] == R.PASS and c["measured"]["limit"] == 10 / 1000
    c = R.g2_bf16_natural_selection({"natural_bf16": nat}, _emu(frac=down(0.005)), th2)
    assert c["state"] == R.FAIL and c["reason"] == "leg a" and not c["legs"]["a"]["ok"]
    # the floor: 2 / 1000 = 0.2 % with emu at 0
    nat0 = [dict(nat[0], n_disagree=2)]
    assert R.g2_bf16_natural_selection({"natural_bf16": nat0}, _emu(frac=0.0), th2)["state"] == R.PASS
    nat0 = [dict(nat[0], n_disagree=3)]
    assert R.g2_bf16_natural_selection({"natural_bf16": nat0}, _emu(frac=0.0), th2)["state"] == R.FAIL
    # (p): M_port <= min(8, 1.2 x M_emu)
    for m_emu in (5.0, 6.1, 10.0):
        bound = min(th2["pairwise_z_cap"], th2["pairwise_z_factor_vs_emu"] * m_emu)
        rows = [dict(nat[0], max_gap_over_sigma=bound), dict(nat[0], layer=1, max_gap_over_sigma=0.0)]
        c = R.g2_bf16_natural_selection({"natural_bf16": rows}, _emu(frac=0.005, m_emu=m_emu), th2)
        assert c["state"] == R.PASS and c["measured"]["M_port"] == bound and c["measured"]["z_limit"] == bound
        rows[0]["max_gap_over_sigma"] = up(bound)
        c = R.g2_bf16_natural_selection({"natural_bf16": rows}, _emu(frac=0.005, m_emu=m_emu), th2)
        assert c["state"] == R.FAIL and c["reason"] == "leg p"
    assert min(8.0, 1.2 * 10.0) == 8.0  # the cap binds at M_emu 10
    # exp_036's values (§3.5): 6.078 <= min(8, 1.2 x 6.100 = 7.32)
    rows = [dict(nat[0], max_gap_over_sigma=6.078)]
    assert R.g2_bf16_natural_selection({"natural_bf16": rows}, _emu(frac=0.0247 / 2, m_emu=6.100), th2)["legs"]["p"]["ok"]
    # NaN never passes; the 6-sigma count is descriptive only
    rows = [dict(nat[0], max_gap_over_sigma=math.nan)]
    assert R.g2_bf16_natural_selection({"natural_bf16": rows}, _emu(frac=0.005), th2)["state"] == R.FAIL
    rows = [dict(nat[0], n_beyond_6sigma=40)]
    c = R.g2_bf16_natural_selection({"natural_bf16": rows}, _emu(frac=0.005), th2)
    assert c["state"] == R.PASS and c["measured"]["descriptive"]["n_beyond_6sigma_port"] == 40
    with pytest.raises(R.RuleInputError):
        R.g2_bf16_natural_selection({"natural_bf16": nat}, {"layers": _emu()["layers"]}, th2)  # no M_emu


def test_g2_router_margin_and_g2q(th):
    th2 = th["G2"]
    assert R.g2_checks(_g2_res(router_margin={"diff_p99": 0.0}), _emu(), th2)["g2_router_margin"]["state"] == R.PASS
    assert R.g2_checks(_g2_res(router_margin={"diff_p99": down(0.0)}), _emu(), th2)["g2_router_margin"]["state"] == R.FAIL
    assert R.g2_checks(_g2_res(router_margin={"diff_p99": math.nan}), _emu(), th2)["g2_router_margin"]["state"] == R.FAIL
    g = _g2_res()["layers_g2q"]["K8"]
    for key, lim in (("embedding_max_rel", 4e-3), ("head_rel_l2_max", 1e-3)):
        q = {"K8": dict(g, **{key: lim})}
        assert R.g2_checks(_g2_res(layers_g2q=q), _emu(), th2)["g2q_K8"]["state"] == R.PASS
        q = {"K8": dict(g, **{key: up(lim)})}
        assert R.g2_checks(_g2_res(layers_g2q=q), _emu(), th2)["g2q_K8"]["state"] == R.FAIL


def test_g2_real_weight_mutants(th):
    th2 = th["G2"]
    clean = {"attn": {"median": 1e-6, "max": 1e-5}, "moe": {"median": 1e-6, "max": 1e-5}, "selection_violations": 0}
    at_edge = {"attn": {"median": 1e-4, "max": 1e-3}, "moe": dict(clean["moe"]), "selection_violations": 0}
    assert not R.g2_mutant_detected({"embedding_exact": True, "layers": {0: clean, 3: at_edge}}, th2)
    beyond = copy.deepcopy(at_edge)
    beyond["attn"]["median"] = up(1e-4)
    assert R.g2_mutant_detected({"embedding_exact": True, "layers": {0: clean, 3: beyond}}, th2)
    beyond = copy.deepcopy(at_edge)
    beyond["moe"]["max"] = up(1e-3)
    assert R.g2_mutant_detected({"embedding_exact": True, "layers": {"0": beyond}}, th2)
    assert R.g2_mutant_detected({"embedding_exact": True, "layers": {0: dict(clean, selection_violations=1)}}, th2)
    assert R.g2_mutant_detected({"embedding_exact": False, "layers": {0: clean}}, th2)
    muts = {"sigmoid_bias_select": {"embedding_exact": True, "layers": {0: dict(clean, selection_violations=3)}},
            "mup_embed_scale": {"embedding_exact": False, "layers": {0: clean}},
            "route_scale_2826": {"embedding_exact": True, "layers": {0: clean}}}
    c = R.g2_checks(_g2_res(real_weight_mutants={"layers": [0], "mutants": muts}), None, th2)["g2_real_weight_mutants"]
    assert c["state"] == R.FAIL and c["measured"]["undetected"] == ["route_scale_2826"]
    with pytest.raises(R.RuleInputError, match="branch stats"):
        R.g2_mutant_detected({"embedding_exact": True, "layers": {0: {"forced_fail": True, "selection_violations": 0}}}, th2)


def test_g2_row_verdicts_align_with_the_rows(th):
    th2 = th["G2"]
    res = _g2_res()
    res["layers_fp32"][1]["max"] = up(1e-3)
    v = R.g2_row_verdicts(res, _emu(), th2)
    assert set(v) == {"fp32_forced", "bf16_forced", "t9_bf16_forced", "g2q_K8", "g2q_K4"}
    assert [r["pass"] for r in v["fp32_forced"]] == [True, False]
    assert len(v["bf16_forced"]) == len(res["layers_bf16"]) and v["bf16_forced"][0]["limit_median"] == 3.0 * 0.01
    assert v["g2q_K4"] == []
    nv = R.g2_row_verdicts(res, None, th2)
    assert all(r["pass"] is None for r in nv["bf16_forced"] + nv["t9_bf16_forced"])


# ------------------------------------------------------------------------------- G3


def test_g3_mutant_class_boundaries():
    assert R.g3_mutant_class(up(0.0), 1.0) == R.G3_REFERENCE_BETTER
    assert R.g3_mutant_class(0.0, 1.0) == R.G3_NOT_RESOLVABLE
    assert R.g3_mutant_class(-1.0, 0.0) == R.G3_NOT_RESOLVABLE
    assert R.g3_mutant_class(-1.0, down(0.0)) == R.G3_MUTANT_BETTER
    with pytest.raises(R.RuleInputError):
        R.g3_mutant_class(math.nan, 1.0)


def test_g3_checks(th):
    th3 = th["G3"]
    muts = {f"m{i}": {"dnll_mean": 0.5, "dnll_p01": 0.3, "dnll_p99": 0.7} for i in range(7)}
    c = R.g3_checks(None, None, muts, th3)["g3_ref_mutants"]
    assert c["state"] == R.PASS and c["measured"]["mutant_better"] == []
    muts["m0"] = {"dnll_mean": 0.0, "dnll_p01": -0.1, "dnll_p99": 0.1}  # not resolvable: does not fail
    c = R.g3_checks(None, None, muts, th3)["g3_ref_mutants"]
    assert c["state"] == R.PASS and c["measured"]["rests_on_vendor_source"] == ["m0"]
    muts["m1"] = {"dnll_mean": -0.3, "dnll_p01": -0.5, "dnll_p99": down(0.0)}
    c = R.g3_checks(None, None, muts, th3)["g3_ref_mutants"]
    assert c["state"] == R.FAIL and c["measured"]["mutant_better"] == ["m1"]
    six = {k: v for k, v in muts.items() if k not in ("m1",)}
    six.pop("m6")
    c = R.g3_checks(None, None, six, th3)["g3_ref_mutants"]
    assert c["state"] == R.FAIL and "not 7" in c["reason"]
    # G3a: descriptive, exp_036's limits printed beside it
    bpb = {"per_text_bpb": {f"T{i}": 1.31 for i in range(1, 7)}, "mean_bpb": 1.04}
    peers = {"peers": {"G8": 0.80, "Q36-8": 0.90}, "source": "results/peers_x.json"}
    c = R.g3_checks(bpb, peers, None, th3)
    assert c["g3_bpb_per_text"]["state"] == c["g3_bpb_vs_peers"]["state"] == R.DESCRIPTIVE
    assert c["g3_bpb_per_text"]["within_exp036_limit"] is False and c["g3_bpb_vs_peers"]["within_exp036_limit"] is False
    assert c["g3_bpb_vs_peers"]["measured"]["ratio_to_best_peer"] == pytest.approx(1.3)
    assert R.g3_checks(bpb, None, None, th3)["g3_bpb_vs_peers"]["within_exp036_limit"] is None


# ------------------------------------------------------------------------------- G4-N(i)


def test_g4_n_boundaries(th):
    lim = th["G4"]["K8_backstop_mean_kl_max"]
    good = {"T1-8": {"mean_kl": lim, "top1_decisive": 0.95}, "T9": {"mean_kl": 0.05, "top1_decisive": 0.9}}
    c = R.g4_n(good, th)
    assert c["state"] == R.PASS  # decisive top-1 is descriptive now
    assert c["descriptive_rows"]["registered_decisive_top1_min"] == 0.99
    assert c["descriptive_rows"]["exp036_emulation_decisive_top1"]["value"] == pytest.approx(1 - 68 / 5943)
    for s in ("T1-8", "T9"):
        bad = copy.deepcopy(good)
        bad[s]["mean_kl"] = up(lim)
        c = R.g4_n(bad, th)
        assert c["state"] == R.FAIL and s in c["reason"]
    bad = copy.deepcopy(good)
    bad["T9"]["mean_kl"] = math.nan
    assert R.g4_n(bad, th)["state"] == R.FAIL
    assert R.g4_k4(good, th)["state"] == R.DESCRIPTIVE


# ------------------------------------------------------------------------------- G4-F32


F_TINY = 1e-6


def _f32_set(mean_kl=1e-7, csort=1e-9, leads=(), kls=(1e-6,), gaps=(), pairs=1000):
    return {"n": 1000, "mean_kl": mean_kl, "csort_mean": csort, "csort_max": csort * 10,
            "top1_change_leads": R.tail(leads), "kl": R.tail(kls), "shadow_gaps": R.tail(gaps), "shadow_pairs": pairs}


def _f32_buckets(means=(1e-7, 1e-7, 1e-7), S=(1e-3, 1e-3, 1e-3), csort=(1e-9, 1e-9, 1e-9)):
    edges = [(0, 2048), (2048, 8192), (8192, 16384)]
    return {f"{lo}-{hi}": {"lo": lo, "hi": hi, "n": hi - lo, "mean_kl": m, "mean_S": s, "csort_mean": c}
            for (lo, hi), m, s, c in zip(edges, means, S, csort)}


def _f32_series(t18=None, t9=None, buckets=None):
    return {"sets": {"T1-8": t18 or _f32_set(), "T9": t9 or _f32_set()}, "t9_buckets": buckets or _f32_buckets()}


def _f32_mutants():
    return {1: _f32_series(t18=_f32_set(gaps=[0.5] * 12)),
            6: _f32_series(t18=_f32_set(mean_kl=1e-4)),
            19: _f32_series(buckets=_f32_buckets(means=(1e-7, 1e-4, 1e-4)))}


def test_g4_f32_passes_with_every_control_caught(th, ctl):
    c = R.g4_f32(_f32_series(), _f32_mutants(), F_TINY, th, ctl)
    assert c["state"] == R.PASS, c["reason"]
    assert c["measured"]["tau"] == {"T1-8": F_TINY, "T9": F_TINY}
    assert {k: v["caught"] for k, v in c["controls"].items()} == {"G4F32/1": True, "G4F32/6": True, "G4F32/19": True}
    for v in c["controls"].values():
        assert v["margin"]["met"], v


def test_g4_f32_tau(th, ctl):
    # tau = max(F_tiny, 100 x mean Csort)
    base = _f32_series(t18=_f32_set(csort=1e-8))  # 100 x 1e-8 = 1e-6 ... equals F_tiny
    cal = R.g4_f32_calibration(base, F_TINY, th)
    assert cal["tau"]["T1-8"] == max(F_TINY, 100.0 * 1e-8)
    base = _f32_series(t18=_f32_set(csort=5e-7))
    assert R.g4_f32_calibration(base, F_TINY, th)["tau"]["T1-8"] == 100.0 * 5e-7
    with pytest.raises(R.RuleInputError, match="cap"):
        R.g4_f32_calibration(_f32_series(), up(th["G4F32"]["f_tiny_cap"]), th)
    R.g4_f32_calibration(_f32_series(), th["G4F32"]["f_tiny_cap"], th)  # equality is allowed
    for bad in (0.0, -1e-7, math.nan, None, True):
        with pytest.raises(R.RuleInputError):
            R.g4_f32_calibration(_f32_series(), bad, th)


def test_g4_f32_legs_at_their_boundaries(th, ctl):
    t = th["G4F32"]
    muts = _f32_mutants()

    def state(base):
        return R.g4_f32(base, muts, F_TINY, th, ctl)

    # F2: mean <= tau passes; one ulp above fires (T1-8, T9, a bucket at T9's tau)
    assert state(_f32_series(t18=_f32_set(mean_kl=F_TINY)))["state"] == R.PASS
    c = state(_f32_series(t18=_f32_set(mean_kl=up(F_TINY))))
    assert c["state"] == R.FAIL and c["legs"]["F2"]["per"]["T1-8"]["fired"]
    c = state(_f32_series(buckets=_f32_buckets(means=(1e-7, 1e-7, up(F_TINY)))))
    assert c["state"] == R.FAIL and c["legs"]["F2"]["per"]["T9:8192-16384"]["fired"]
    # F1: a top-1 change at lead >= 2 nats fires (decisive is inclusive); just below does not
    c = state(_f32_series(t9=_f32_set(leads=[2.0, 0.5])))
    assert c["state"] == R.FAIL and c["legs"]["F1"]["fired"] and c["legs"]["F1"]["count"] == 1
    assert state(_f32_series(t9=_f32_set(leads=[down(2.0), 0.5])))["state"] == R.PASS
    # F5: a position with KL > 1e-2 fires; equality does not
    assert state(_f32_series(t18=_f32_set(kls=[t["f5_position_kl_max"]])))["state"] == R.PASS
    c = state(_f32_series(t18=_f32_set(kls=[up(t["f5_position_kl_max"])])))
    assert c["state"] == R.FAIL and c["legs"]["F5"]["fired"]
    # F6: shadow sets differ where R2F's gap >= 1e-2 fires; just below does not
    c = state(_f32_series(t9=_f32_set(gaps=[t["f6_shadow_gap_min"], 1e-5])))
    assert c["state"] == R.FAIL and c["legs"]["F6"]["fired"] and c["legs"]["F6"]["per"]["T9"]["disagree_rate"] == 2 / 1000
    assert state(_f32_series(t9=_f32_set(gaps=[down(t["f6_shadow_gap_min"])])))["state"] == R.PASS
    # Csort ceiling: mean Csort <= 1e-6 passes; above is a FAIL, cause not attributed
    ceiling = t["csort_mean_ceiling"]
    base = _f32_series(t9=_f32_set(csort=ceiling, mean_kl=1e-7))
    assert state(base)["state"] == R.PASS
    c = state(_f32_series(t9=_f32_set(csort=up(ceiling), mean_kl=1e-7)))
    assert c["state"] == R.FAIL and R.CSORT_CEILING_REASON in c["reason"] and "T9" in c["reason"]


def test_g4_f32_f3_position_growth(th, ctl):
    """F3 fires iff rho(B2) > 10 x rho(B0) and mean KL(B2) > 10 x Csort(B2)."""
    muts = _f32_mutants()
    # T9 Csort 1e-6 gives a T9 tau of about 1e-4, high enough that F2 stays quiet
    t9 = _f32_set(csort=1e-6)
    cs2 = 1e-9
    m0, s0, s2 = 1e-7, 1e-3, 1e-3
    rho0 = m0 / s0
    m2 = 10.0 * rho0 * s2  # rho(B2) == 10 x rho(B0) exactly when s2 == s0
    assert m2 / s2 == 10.0 * rho0 and m2 > 10 * cs2

    def series(m2_):
        return _f32_series(t9=t9, buckets=_f32_buckets(means=(m0, 1e-7, m2_), S=(s0, 1e-3, s2), csort=(1e-9, 1e-9, cs2)))

    c = R.g4_f32(series(m2), muts, F_TINY, th, ctl)
    assert R.g4_f32_calibration(series(m2), F_TINY, th)["tau"]["T9"] == 100.0 * 1e-6  # about tau9
    assert not c["legs"]["F3"]["fired"] and c["state"] == R.PASS
    c = R.g4_f32(series(up(m2)), muts, F_TINY, th, ctl)
    assert c["legs"]["F3"]["fired"] and c["state"] == R.FAIL
    # the second clause: B2's mean must also exceed 10 x Csort(B2)
    big_cs = up(m2) / 10.0
    s = _f32_series(t9=t9, buckets=_f32_buckets(means=(m0, 1e-7, up(m2)), S=(s0, 1e-3, s2), csort=(1e-9, 1e-9, big_cs)))
    assert not R.g4_f32(s, muts, F_TINY, th, ctl)["legs"]["F3"]["fired"]


def test_g4_f32_controls_are_bound_to_their_legs(th, ctl):
    """Each control counts only through its own leg (§4.2)."""
    base = _f32_series()
    # mutant 1 fires F2 (not its leg F6): not caught -> INCOMPLETE
    muts = _f32_mutants()
    muts[1] = _f32_series(t18=_f32_set(mean_kl=1e-3))
    c = R.g4_f32(base, muts, F_TINY, th, ctl)
    assert c["state"] == R.INCOMPLETE and c["reason"] == "power not shown: mutant 1 (sigmoid_bias_select)"
    # mutant 19 fires F2 on T1-8 only (its leg is F2 on T9 or a T9 bucket): not caught
    muts = _f32_mutants()
    muts[19] = _f32_series(t18=_f32_set(mean_kl=1e-3))
    c = R.g4_f32(base, muts, F_TINY, th, ctl)
    assert c["state"] == R.INCOMPLETE and "mutant 19" in c["reason"]
    # mutant 6 fires F2 on T9: caught (T1-8 or T9)
    muts = _f32_mutants()
    muts[6] = _f32_series(t9=_f32_set(mean_kl=1e-3))
    assert R.g4_f32(base, muts, F_TINY, th, ctl)["state"] == R.PASS
    # a control's series absent: MISSING (a crash), never PASS
    muts = _f32_mutants()
    del muts[6]
    c = R.g4_f32(base, muts, F_TINY, th, ctl)
    assert c["state"] == R.MISSING and c["controls"]["G4F32/6"]["caught"] is None
    # a firing leg wins over an uncaught control
    muts = _f32_mutants()
    muts[1] = _f32_series()
    assert R.g4_f32(_f32_series(t9=_f32_set(leads=[5.0])), muts, F_TINY, th, ctl)["state"] == R.FAIL
    # a mutant is judged with the unmutated run's tau: a mutant series with a huge Csort of its own changes nothing
    muts = _f32_mutants()
    muts[6] = _f32_series(t18=_f32_set(mean_kl=1e-4, csort=1.0))
    assert R.g4_f32(base, muts, F_TINY, th, ctl)["controls"]["G4F32/6"]["caught"]
    assert R.g4_f32(None, muts, F_TINY, th, ctl)["state"] == R.MISSING


def test_g4_f32_control_margins(th, ctl):
    base = _f32_series()
    muts = _f32_mutants()
    muts[1] = _f32_series(t18=_f32_set(gaps=[0.5] * 9))  # 9 qualifying pairs: caught, margin not met
    muts[6] = _f32_series(t18=_f32_set(mean_kl=9.99e-6))  # 9.99 x tau
    muts[19] = _f32_series(buckets=_f32_buckets(means=(1e-7, 1e-7, 1e-5)))  # exactly 10 x tau
    c = R.g4_f32(base, muts, F_TINY, th, ctl)
    m = {k: v["margin"] for k, v in c["controls"].items()}
    assert c["state"] == R.PASS  # the margin is the tiny pre-freeze check, never a gate leg
    assert (m["G4F32/1"]["unit"], m["G4F32/1"]["value"], m["G4F32/1"]["met"]) == ("count", 9, False)
    assert m["G4F32/6"]["unit"] == "mean" and not m["G4F32/6"]["met"]
    assert m["G4F32/19"]["value"] == pytest.approx(10.0) and m["G4F32/19"]["met"] is (1e-5 / F_TINY >= 10.0)


# ------------------------------------------------------------------------------- G4-F16


def _f16(port=0.002, r3=0.001):
    return {"T1-8": {"mean_kl_port": port, "mean_kl_r3": r3, "p99_ratio": 1.2}, "T9": {"mean_kl_port": port, "mean_kl_r3": r3}}


def test_g4_f16_kappa_and_void(th, ctl):
    k = th["G4F16"]["kappa"]
    muts = {6: _f16(port=1.0)}
    r3 = 1e-3
    assert R.g4_f16(_f16(port=k * r3, r3=r3), muts, th, ctl)["state"] == R.PASS
    c = R.g4_f16(_f16(port=up(k * r3), r3=r3), muts, th, ctl)
    assert c["state"] == R.FAIL and c["legs"]["kappa"]["fired"]
    ceil = th["G4F16"]["r3_mean_kl_ceiling"]
    assert R.g4_f16(_f16(port=1e-3, r3=ceil), {6: _f16(port=1.0, r3=ceil)}, th, ctl)["state"] == R.PASS
    c = R.g4_f16(_f16(port=1e-3, r3=up(ceil)), muts, th, ctl)
    assert c["state"] == R.INCOMPLETE and c["reason"].startswith("VOID: R3 above ceiling")
    # VOID is not judged by kappa: a firing kappa under VOID is still VOID
    c = R.g4_f16(_f16(port=1.0, r3=up(ceil)), muts, th, ctl)
    assert c["state"] == R.INCOMPLETE and c["reason"].startswith("VOID")
    # mutant 6 must fire kappa (judged with the unmutated R3), with ratio >= 90 for the margin
    c = R.g4_f16(_f16(), {6: _f16(port=k * 1e-3)}, th, ctl)  # at the bound: not caught
    assert c["state"] == R.INCOMPLETE and "mutant 6" in c["reason"]
    c = R.g4_f16(_f16(), {6: {"T1-8": {"mean_kl_port": 0.05}, "T9": {"mean_kl_port": 1e-4}}}, th, ctl)
    mg = c["controls"]["G4F16/6"]["margin"]
    assert c["state"] == R.PASS and mg["unit"] == "ratio" and mg["required"] == 90.0 and mg["value"] == pytest.approx(50.0)
    assert not mg["met"]
    assert R.g4_f16(_f16(), {}, th, ctl)["state"] == R.MISSING


# ------------------------------------------------------------------------------- G5 greedy, decode vs prefill, behaviour


def test_g5_greedy_boundaries(th):
    th5 = th["G5"]
    res = {"decisive_top1": 0.995, "in_top5": 0.99, "n_positions": 1000, "n_decisive": 941}
    assert R.g5_greedy(res, th5)["state"] == R.PASS
    assert R.g5_greedy(dict(res, decisive_top1=down(0.995)), th5)["state"] == R.FAIL
    assert R.g5_greedy(dict(res, in_top5=down(0.99)), th5)["state"] == R.FAIL
    assert R.g5_greedy(dict(res, decisive_top1=None), th5)["state"] == R.PASS  # no decisive position
    assert R.g5_greedy(dict(res, in_top5=None), th5)["state"] == R.FAIL
    assert R.g5_greedy(dict(res, decisive_top1=math.nan), th5)["state"] == R.FAIL


def test_decode_vs_prefill_boundaries(th):
    th5 = th["G5"]
    fd = {"floor_kl": 0.05, "floor_dis": 0.06, "decode_kl": 0.0, "decode_dis": 0.0}
    b = R.parity_bound(fd, th5)
    assert b == {"kl_max": max(1e-4, 3.0 * 0.05), "dis_max": 3.0 * 0.06 + 0.002}
    assert R.parity_bound({"floor_kl": 0.0, "floor_dis": 0.0}, th5) == {"kl_max": 1e-4, "dis_max": 0.002}
    ok = dict(fd, decode_kl=b["kl_max"], decode_dis=b["dis_max"])
    assert R.g5_decode_vs_prefill(ok, th5)["state"] == R.PASS
    assert R.g5_decode_vs_prefill(dict(ok, decode_kl=up(b["kl_max"])), th5)["state"] == R.FAIL
    assert R.g5_decode_vs_prefill(dict(ok, decode_dis=up(b["dis_max"])), th5)["state"] == R.FAIL
    # exp_036's values (§3.11): 0.0379 against 0.1513
    assert R.g5_decode_vs_prefill({"floor_kl": 0.1513 / 3, "floor_dis": 0.0663, "decode_kl": 0.0379,
                                   "decode_dis": 0.0533}, th5)["state"] == R.PASS


def test_behaviour_rule(th):
    th5 = th["G5"]
    cell = {"n": 20, "n_loop": 1, "n_no_eos": 1, "n_unknown_lang": 1, "think_closed_rate": 1.0, "lang_match_rate": 1.0,
            "cap": 8192, "seed": 1}
    c = R.g5_behaviour({"high": cell, "none": cell}, th5)
    assert c["state"] == R.PASS and c["measured"]["blocked_efforts"] == []
    for k in ("n_loop", "n_no_eos", "n_unknown_lang"):
        c = R.g5_behaviour({"high": cell, "none": dict(cell, **{k: 2})}, th5)  # >= 2 / 20 blocks
        assert c["state"] == R.FAIL and c["measured"]["blocked_efforts"] == ["none"]


# ------------------------------------------------------------------------------- G5-D32


def _d32_comp(mean_kl=1e-10, max_kl=1e-8, leads=()):
    return {"n": 581, "mean_kl": mean_kl, "max_kl": max_kl, "top1_change_leads": R.tail(leads)}


def _d32(csort_mean=1e-9, csort_max=1e-7, **comp):
    return {r: {"csort": {"mean": csort_mean, "max": csort_max}, "decode": _d32_comp(**comp), "chunk64": _d32_comp()}
            for r in ("T1", "T3", "T9")}


def _d32_mut(r="T9", **comp):
    m = {x: {"decode": _d32_comp()} for x in ("T1", "T3", "T9")}
    m[r] = {"decode": _d32_comp(**comp)}
    return {20: m}


def test_g5_d32_boundaries(th, ctl):
    t = th["G5D32"]
    F = t["mean_factor_vs_csort"]
    assert F == t["max_factor_vs_csort"] == 100.0   # W15-04
    muts = _d32_mut(mean_kl=1e-5)
    # Csort 1e-9 / 1e-7: mean bound max(1e-8, 1e-7) = 1e-7, max bound max(1e-6, 1e-5) = 1e-5
    b = R.g5_d32_bounds(_d32(), th)["T1"]
    assert b["mean_bound"] == max(t["mean_floor"], F * 1e-9) and b["max_bound"] == max(t["max_floor"], F * 1e-7)
    # Csort 1e-11 / 1e-9: the floors set the bounds
    fl = R.g5_d32_bounds(_d32(csort_mean=1e-11, csort_max=1e-9), th)["T1"]
    assert (fl["mean_bound"], fl["max_bound"]) == (t["mean_floor"], t["max_floor"])
    assert R.g5_d32(_d32(mean_kl=b["mean_bound"], max_kl=b["max_bound"]), muts, th, ctl)["state"] == R.PASS
    assert R.g5_d32(_d32(mean_kl=up(b["mean_bound"])), muts, th, ctl)["state"] == R.FAIL
    assert R.g5_d32(_d32(max_kl=up(b["max_bound"])), muts, th, ctl)["state"] == R.FAIL
    # Csort above its floors sets the bound: 100 x Csort
    big = _d32(csort_mean=1e-7, csort_max=1e-5, mean_kl=F * 1e-7, max_kl=F * 1e-5)
    strong = _d32_mut(mean_kl=1e-3)  # mutant 20 judged against the larger bounds
    assert R.g5_d32(big, strong, th, ctl)["state"] == R.PASS
    assert R.g5_d32(_d32(csort_mean=1e-7, csort_max=1e-5, mean_kl=up(F * 1e-7)), strong, th, ctl)["state"] == R.FAIL
    assert R.g5_d32(_d32(csort_mean=1e-7, csort_max=1e-5, max_kl=up(F * 1e-5)), strong, th, ctl)["state"] == R.FAIL
    # the sharpened tiny build's unmutated values (diagnostics/tiny_stress, record e53158e9...): mean KL 11-16 x
    # Csort failed one decade and passes two
    sharp = _d32(csort_mean=1.6e-9, csort_max=4.8e-9, mean_kl=2.5e-8, max_kl=7.1e-8)
    assert R.g5_d32(sharp, strong, th, ctl)["state"] == R.PASS
    # the top-1 rule: a change where (ii) leads by >= 2 nats fires
    c = R.g5_d32(_d32(leads=[2.0]), muts, th, ctl)
    assert c["state"] == R.FAIL and c["legs"]["T1:decode"]["rules"]["top1"]["count"] == 1
    assert R.g5_d32(_d32(leads=[down(2.0)]), muts, th, ctl)["state"] == R.PASS
    chunk = _d32()
    chunk["T3"]["chunk64"] = _d32_comp(mean_kl=1e-5)
    c = R.g5_d32(chunk, muts, th, ctl)
    assert c["state"] == R.FAIL and c["legs"]["T3:chunk64"]["fired"]
    # ceilings: Csort mean <= 1e-6 and max <= 1e-4 per range; above is a FAIL
    at_ceiling = _d32(csort_mean=1e-6, csort_max=1e-4, mean_kl=F * 1e-6, max_kl=F * 1e-4)
    assert R.g5_d32(at_ceiling, _d32_mut(mean_kl=1.0), th, ctl)["state"] == R.PASS
    c = R.g5_d32(_d32(csort_mean=up(1e-6), csort_max=1e-4), muts, th, ctl)
    assert c["state"] == R.FAIL and R.CSORT_CEILING_REASON in c["reason"] and "T1 mean" in c["reason"]
    c = R.g5_d32(_d32(csort_max=up(1e-4)), muts, th, ctl)
    assert c["state"] == R.FAIL and "T9 max" in c["reason"]


def test_g5_d32_control_20(th, ctl):
    base = _d32()
    # mutant 20 fires the decode mean rule in T9 only: caught; margin mean 100 x (bound 1e-7 at Csort 1e-9)
    c = R.g5_d32(base, _d32_mut(mean_kl=1e-5), th, ctl)
    r = c["controls"]["G5D32/20"]
    assert c["state"] == R.PASS and r["caught"] and r["margin"]["met"] and r["margin"]["range"] == "T9"
    # a top-1 change alone (count 3): caught, margin (count >= 10) not met
    c = R.g5_d32(base, _d32_mut(leads=[3.0, 2.5, 2.0]), th, ctl)
    r = c["controls"]["G5D32/20"]
    assert r["caught"] and not r["margin"]["met"]
    # nothing fires: INCOMPLETE
    c = R.g5_d32(base, _d32_mut(), th, ctl)
    assert c["state"] == R.INCOMPLETE and "mutant 20 (decode_window_256)" in c["reason"]
    # mutant 20 is judged with the unmutated Csort of each range
    c = R.g5_d32(_d32(csort_mean=1e-6, csort_max=1e-4), _d32_mut(mean_kl=1e-5), th, ctl)
    assert c["state"] == R.INCOMPLETE  # 1e-5 <= 100 x 1e-6


# ------------------------------------------------------------------------------- G5-R1


def _r1_arms(runner=0.03, single=0.01, parity_kl=1e-5, dis=0.0):
    return {"K8": {"anchor": {"n": 2048, "mean_kl_r1_runner": runner, "mean_kl_r1_single": single},
                   "greedy": {"decisive_top1": 1.0, "in_top5": 1.0, "n_decisive": 941, "n": 2048},
                   "parity": {"mean_kl": parity_kl, "top1_dis": dis, "n": 2048}},
            "K4": {"anchor": {"n": 2048, "mean_kl_r1_runner": runner, "mean_kl_r1_single": single}}}


FLOOR = {"floor_kl": 0.01, "floor_dis": 0.01}
M26 = {26: {"parity": {"mean_kl": 1.0, "top1_dis": 0.5}}}


def test_g5_r1_legs(th, ctl):
    out = R.g5_r1(_r1_arms(), FLOOR, M26, th, ctl)
    assert out["g5_r1_K8"]["state"] == out["g5_r1_K4"]["state"] == R.PASS
    assert set(out["g5_r1_K8"]["legs"]) == {"R-anchor", "R-greedy", "R-parity"} and set(out["g5_r1_K4"]["legs"]) == {"R-anchor"}
    # R-anchor: runner <= 3 x single (both arms)
    out = R.g5_r1(_r1_arms(runner=up(0.03)), FLOOR, M26, th, ctl)
    assert out["g5_r1_K8"]["state"] == out["g5_r1_K4"]["state"] == R.FAIL
    # R-parity (K8 only): KL <= max(1e-4, 3 x floor_KL), top-1 dis <= 3 x floor_dis + 0.2 pp
    b = R.parity_bound(FLOOR, th["G5"])
    out = R.g5_r1(_r1_arms(parity_kl=b["kl_max"], dis=b["dis_max"]), FLOOR, M26, th, ctl)
    assert out["g5_r1_K8"]["state"] == R.PASS
    out = R.g5_r1(_r1_arms(parity_kl=up(b["kl_max"])), FLOOR, M26, th, ctl)
    assert out["g5_r1_K8"]["state"] == R.FAIL and out["g5_r1_K4"]["state"] == R.PASS
    out = R.g5_r1(_r1_arms(dis=up(b["dis_max"])), FLOOR, M26, th, ctl)
    assert out["g5_r1_K8"]["reason"] == "legs fired: R-parity"
    # R-greedy (K8): the registered greedy rule
    arms = _r1_arms()
    arms["K8"]["greedy"]["decisive_top1"] = down(0.995)
    assert R.g5_r1(arms, FLOOR, M26, th, ctl)["g5_r1_K8"]["reason"] == "legs fired: R-greedy"


def test_g5_r1_control_26_governs_both_arms(th, ctl):
    b = R.parity_bound(FLOOR, th["G5"])
    weak = {26: {"parity": {"mean_kl": b["kl_max"], "top1_dis": b["dis_max"]}}}  # at the bound: not caught
    out = R.g5_r1(_r1_arms(), FLOOR, weak, th, ctl)
    assert out["g5_r1_K8"]["state"] == out["g5_r1_K4"]["state"] == R.INCOMPLETE
    assert "mutant 26 (batch_decode_pos_frozen)" in out["g5_r1_K4"]["reason"]
    # caught by top-1 disagreement alone: caught, but the KL margin (mean, 10 x the bound) is not met
    dis_only = {26: {"parity": {"mean_kl": 0.0, "top1_dis": 0.5}}}
    out = R.g5_r1(_r1_arms(), FLOOR, dis_only, th, ctl)
    r = out["g5_r1_K8"]["controls"]["G5R1/26"]
    assert out["g5_r1_K8"]["state"] == R.PASS and r["caught"] and r["margin"]["unit"] == "mean" and not r["margin"]["met"]
    assert R.g5_r1(_r1_arms(), FLOOR, M26, th, ctl)["g5_r1_K8"]["controls"]["G5R1/26"]["margin"]["met"]
    # missing pieces
    out = R.g5_r1(_r1_arms(), None, M26, th, ctl)
    assert out["g5_r1_K8"]["state"] == R.MISSING and out["g5_r1_K4"]["state"] == R.MISSING
    out = R.g5_r1({"K8": _r1_arms()["K8"]}, FLOOR, M26, th, ctl)
    assert out["g5_r1_K4"]["state"] == R.MISSING and out["g5_r1_K8"]["state"] == R.PASS
    assert R.g5_r1(_r1_arms(), FLOOR, {}, th, ctl)["g5_r1_K8"]["state"] == R.MISSING


# ------------------------------------------------------------------------------- G5-BP-lean and allowed_B


def _bp(batched=0.03, single=0.01, adm=3, live=8, B=8, n=100):
    d = {"n": n, "mean_kl_r1_batched": batched, "mean_kl_r1_single": single}
    return {"B": B, "first_wave": dict(d), "mid_run": dict(d), "admitted_mid_run": adm, "max_live": live}


def _bp_results(over: dict | None = None):
    res = {arm: {8: _bp(), 16: _bp(B=16, live=16)} for arm in ("K8", "K4")}
    for (arm, B), p in (over or {}).items():
        res[arm][B] = p
    return res


# G5-BP-lean's required control is mutant 27 since decision (b) (DESIGN §4.1, §4.2);
# mutant 22 is a probe, reported in tiny mode only and never required.
M27 = {27: _bp(batched=2.0, single=0.05)}


def test_allowed_b(th):
    assert R.allowed_B(True, True, True, th) == [1, 2, 4, 8, 16]
    assert R.allowed_B(True, False, True, th) == [1, 2, 4, 8]
    assert R.allowed_B(False, True, True, th) == [1]  # 16 inherits from 8
    assert R.allowed_B(False, False, True, th) == [1]
    assert R.allowed_B(True, True, False, th) == [1]  # mutant 27 not caught: power not shown


def test_g5_bp_rules(th, ctl):
    out = R.g5_bp(_bp_results(), M27, th, ctl)
    assert out["allowed_B"] == {"K8": [1, 2, 4, 8, 16], "K4": [1, 2, 4, 8, 16]} and out["power_note"] is None
    assert all(c["state"] == R.PASS and c["blocking"] is False for c in out["checks"].values())
    # (d): batched <= 3 x single per subset; one ulp above fails at (arm, B)
    p = _bp(batched=up(0.03))
    out = R.g5_bp(_bp_results({("K4", 16): dict(p, B=16, max_live=16)}), M27, th, ctl)
    assert out["allowed_B"]["K4"] == [1, 2, 4, 8] and out["allowed_B"]["K8"] == [1, 2, 4, 8, 16]
    c = out["checks"]["g5_bp_K4_B16"]
    assert c["state"] == R.FAIL and "batched-path discrepancy detected at (K4, B=16)" in c["reason"]
    p = _bp()
    p["mid_run"]["mean_kl_r1_batched"] = up(0.03)
    out = R.g5_bp(_bp_results({("K8", 8): p}), M27, th, ctl)
    assert out["allowed_B"]["K8"] == [1] and "d:mid_run" in out["checks"]["g5_bp_K8_B8"]["reason"]
    # (e): admitted_mid_run >= 1 and max_live <= B
    out = R.g5_bp(_bp_results({("K8", 8): _bp(adm=1, live=8)}), M27, th, ctl)
    assert out["allowed_B"]["K8"] == [1, 2, 4, 8, 16]
    out = R.g5_bp(_bp_results({("K8", 8): _bp(adm=0)}), M27, th, ctl)
    assert out["allowed_B"]["K8"] == [1]
    out = R.g5_bp(_bp_results({("K8", 16): _bp(B=16, live=17)}), M27, th, ctl)
    assert out["allowed_B"]["K8"] == [1, 2, 4, 8]
    # a subset with no generated position cannot pass (d)
    p = _bp()
    p["first_wave"]["n"] = 0
    assert R.g5_bp(_bp_results({("K8", 8): p}), M27, th, ctl)["allowed_B"]["K8"] == [1]
    # an (arm, B) never measured is not allowed; JSON str keys are read
    res = _bp_results()
    del res["K4"][16]
    res["K8"] = {str(k): v for k, v in res["K8"].items()}
    out = R.g5_bp(res, {"27": M27[27]}, th, ctl)
    assert out["allowed_B"] == {"K8": [1, 2, 4, 8, 16], "K4": [1, 2, 4, 8]}
    assert out["checks"]["g5_bp_K4_B16"]["reason"] == "not measured"


def test_g5_bp_control_27(th, ctl):
    weak = {27: _bp(batched=0.03, single=0.01)}  # (d) does not fire: power not shown
    out = R.g5_bp(_bp_results(), weak, th, ctl)
    assert out["allowed_B"] == {"K8": [1], "K4": [1]}
    assert out["power_note"] == "power not shown: mutant 27 (batch_decode_pad_keys_visible); allowed_B = {1}"
    assert all(c["state"] == R.PASS for c in out["checks"].values())  # no INCOMPLETE, no cycle
    assert R.g5_bp(_bp_results(), {}, th, ctl)["allowed_B"] == {"K8": [1], "K4": [1]}
    r = R.g5_bp(_bp_results(), M27, th, ctl)["controls"]["G5BP/27"]
    assert r["caught"] and r["margin"] == {"unit": "ratio", "value": pytest.approx(40.0), "required": 30.0, "met": True,
                                           "unbounded": False}
    r = R.g5_bp(_bp_results(), {27: _bp(batched=0.2, single=0.01)}, th, ctl)["controls"]["G5BP/27"]
    assert r["caught"] and r["margin"]["value"] == pytest.approx(20.0) and not r["margin"]["met"]
    # decision (e)'s shape (DESIGN §4.2, §14.3): caught on both subsets at the gate's factor, the 30x margin
    # unmet; the record keeps rules.py's margin as computed (met false), the acceptance lives in the e2e test
    p = _bp(batched=0.03, single=0.01)
    p["first_wave"].update(mean_kl_r1_batched=0.325, mean_kl_r1_single=0.0272)
    p["mid_run"].update(mean_kl_r1_batched=0.203, mean_kl_r1_single=0.0317)
    r = R.g5_bp(_bp_results(), {27: p}, th, ctl)["controls"]["G5BP/27"]
    assert r["caught"] and not r["values"]["first_wave"]["ok"] and not r["values"]["mid_run"]["ok"]
    assert r["margin"]["value"] == pytest.approx(0.325 / 0.0272) and not r["margin"]["met"]


def test_probe_22_changes_no_verdict_incomplete_or_allowed_b(th, ctl):
    """controls.json's probes (DESIGN §4.2, decision (b)): G5BP/22 is reported in
    tiny mode only and enters no verdict, no INCOMPLETE and no allowed_B. rules.py
    never reads it: only the required control (27) governs G5-BP-lean."""
    assert [c["id"] for c in R.required_controls("g5_bp_K8_B8", ctl)] == ["G5BP/27"]
    assert [p["id"] for p in ctl["probes"]] == ["G5BP/22"] and ctl["probes"][0]["modes"] == ["tiny"]
    assert "G5BP/22" not in {c["id"] for c in ctl["required"]}
    strong22, weak22 = _bp(batched=2.0, single=0.05), _bp(batched=0.03, single=0.01)
    weak27 = {27: _bp(batched=0.03, single=0.01)}
    for res in (_bp_results(), _bp_results({("K8", 8): dict(_bp(adm=0))})):
        for base in (M27, weak27, {}):
            want = R.g5_bp(res, base, th, ctl)
            for m22 in (strong22, weak22):
                got = R.g5_bp(res, {**base, 22: m22}, th, ctl)
                assert got == want  # checks (verdicts), allowed_B, controls and power note all unchanged
                assert "G5BP/22" not in got["controls"]
    # a caught probe never rescues an uncaught control
    assert R.g5_bp(_bp_results(), {**weak27, 22: strong22}, th, ctl)["allowed_B"] == {"K8": [1], "K4": [1]}
    # the record's controls field lists required controls only; no G5-BP-lean check is blocking (no INCOMPLETE)
    out = R.controls_outcome({}, R.g5_bp(_bp_results(), {**M27, 22: strong22}, th, ctl), ctl)
    assert "G5BP/22" not in out and out["G5BP/27"]["caught"] is True
    assert not any(cid.startswith("g5_bp_") for arm in R.BLOCKING for cid in R.BLOCKING[arm])
    # the probe is judged by the same (d) leg (run_gate.py reports it without a margin)
    probe = ctl["probes"][0]
    assert R.control_result(probe, strong22, {}, th)["caught"] is True
    assert R.control_result(probe, weak22, {}, th)["caught"] is False


# ------------------------------------------------------------------------------- P3 (G0k), P5


def _g0k_entries(th, **over):
    out = []
    for proj, bits, rows in R.p3_judged_shapes(th):
        e = {"projection": proj, "bits": bits, "rows": rows, "rel_f64_sorted": 0.0025, "rel_f64_unsorted": 0.0025,
             "bad_rows": 0, "repeat_bitwise": True}
        if (proj, bits, rows) == over.get("at", (None,)):
            e.update(over["set"])
        out.append(e)
    return out


def test_p3_g0k(th):
    shapes = R.p3_judged_shapes(th)
    assert len(shapes) == 2 * 2 * len({n for v in th["P3_G0k"]["judged_rows"].values() for n in v} | {2048})
    assert ("gate_up", 4, 196608) in shapes and ("down", 8, 52752) in shapes and ("down", 4, 2048) in shapes
    assert R.p3_g0k(_g0k_entries(th), th)["ok"]
    at = ("down", 4, 32769)
    ok = R.p3_g0k(_g0k_entries(th, at=at, set={"rel_f64_sorted": 3.0 * 0.002, "rel_f64_unsorted": 0.002}), th)
    assert ok["ok"]
    res = R.p3_g0k(_g0k_entries(th, at=at, set={"rel_f64_sorted": up(3.0 * 0.002), "rel_f64_unsorted": 0.002}), th)
    assert not res["ok"] and res["failing"][0]["shape"] == list(at) and res["failing"][0]["failed"] == ["rel"]
    res = R.p3_g0k(_g0k_entries(th, at=at, set={"rel_f64_sorted": 3.01 * 0.002, "rel_f64_unsorted": 0.002}), th)
    assert not res["ok"]  # a ratio of 3.01 (G1 item 5)
    assert R.p3_g0k(_g0k_entries(th, at=at, set={"bad_rows": 1}), th)["failing"][0]["failed"] == ["bad_rows"]
    assert R.p3_g0k(_g0k_entries(th, at=at, set={"repeat_bitwise": False}), th)["failing"][0]["failed"] == ["repeat"]
    assert not R.p3_g0k(_g0k_entries(th, at=at, set={"rel_f64_sorted": math.nan}), th)["ok"]
    res = R.p3_g0k([e for e in _g0k_entries(th) if (e["projection"], e["bits"], e["rows"]) != at], th)
    assert not res["ok"] and res["absent"] == [list(at)]
    with pytest.raises(R.RuleInputError):
        R.p3_g0k([{"rows": 6, "rel_f64_sorted": 0.0, "rel_f64_unsorted": 0.0, "bad_rows": 0}], th)


def test_p5(th):
    m = {"p5a": {"bitwise": {"0": True, "4": True}}, "p5b": {"bitwise": True}, "p5c": {"mean_kl": 0.10}}
    assert R.p5(m, th)["ok"]
    assert not R.p5(dict(m, p5c={"mean_kl": down(0.10)}), th)["ok"]
    assert not R.p5(dict(m, p5c={"mean_kl": math.nan}), th)["ok"]
    assert not R.p5(dict(m, p5a={"bitwise": {0: True}}), th)["ok"]  # layer 4 not shown
    assert R.p5(dict(m, p5a={"bitwise": {0: True, 4: True}}), th)["ok"]
    assert not R.p5(dict(m, p5b={"bitwise": False}), th)["parts"]["P5b"]["ok"]


# ------------------------------------------------------------------------------- arm verdict, exit table, cycles


def _all_pass(arm: str) -> dict:
    return {cid: R.check(R.PASS) for cid in R.BLOCKING[arm]}


def test_arm_verdict_precedence():
    checks = _all_pass("K8")
    assert R.arm_verdict("K8", checks)["verdict"] == R.PASS
    checks["g4_f16_K8"] = R.check(R.INCOMPLETE, reason="VOID: R3 above ceiling (T9)")
    v = R.arm_verdict("K8", checks)
    assert v["verdict"] == R.INCOMPLETE and v["incomplete"] == ["g4_f16_K8"] and "VOID" in v["reasons"]["g4_f16_K8"]
    del checks["g5_d32_K8"]
    assert R.arm_verdict("K8", checks)["verdict"] == R.MISSING
    checks["g1"] = R.check(R.FAIL)
    v = R.arm_verdict("K8", checks)
    assert v["verdict"] == R.FAIL and v["failing"] == ["g1"] and v["missing"] == ["g5_d32_K8"]
    # a descriptive entry under a blocking id is a missing value, never a pass
    checks = _all_pass("K4")
    checks["g5_r1_K4"] = R.check(R.DESCRIPTIVE)
    assert R.arm_verdict("K4", checks)["verdict"] == R.MISSING
    # tiny runs skip checks marked not applicable
    checks = _all_pass("K8")
    checks["g5_greedy"] = dict(R.check(R.FAIL), applicable=False)
    assert R.arm_verdict("K8", checks, tiny=True)["verdict"] == R.PASS
    assert R.arm_verdict("K8", checks)["verdict"] == R.FAIL


def test_exit_table():
    P, F, I, M = R.PASS, R.FAIL, R.INCOMPLETE, R.MISSING
    cases = [
        ({"K8": P, "K4": P}, {}, 0),
        ({"K8": P, "K4": F}, {}, 4),
        ({"K8": F, "K4": P}, {}, 1),
        ({"K8": F, "K4": M}, {}, 1),
        ({"K8": F, "K4": P}, {"error": "phase 6: RuntimeError"}, 1),  # K8 FAIL first
        ({"K8": P, "K4": P}, {"error": "phase 6: RuntimeError"}, 3),
        ({"K8": P, "K4": M}, {}, 3),
        ({"K8": M, "K4": F}, {}, 3),
        ({"K8": M, "K4": I}, {}, 3),
        ({"K8": I, "K4": P}, {}, 5),
        ({"K8": P, "K4": I}, {}, 5),
        ({"K8": I, "K4": F}, {}, 5),
        ({"K8": P}, {}, 3),  # K4 never ran
        ({"K8": P, "K4": P}, {"precondition_failed": True}, 3),
        ({"K8": F, "K4": F}, {"precondition_failed": True}, 3),
    ]
    for v, kw, want in cases:
        assert R.exit_code_for(v, **kw) == want, (v, kw)
    assert [row[1] for row in R.EXIT_TABLE] == [3, 1, 3, 5, 4, 0]
    assert [row[2] for row in R.EXIT_TABLE] == [False, True, False, True, True, True]
    for code in (0, 1, 4, 5):
        assert R.is_gate_run(code)
    assert not R.is_gate_run(3)


def test_g1_fail_then_p5_failure_is_exit_3():
    """G1 runs in phase 1 and FAILs; P5 then fails at phase 2b: the run stops at
    the precondition, so it is exit 3 (not a gate run, never a cycle)."""
    checks = _all_pass("K8")
    checks["g1"] = R.g1({"tests": 5, "failures": 1, "errors": 0, "skipped": 0}, {"source": "x"})
    k8 = R.arm_verdict("K8", checks)["verdict"]
    assert k8 == R.FAIL and R.exit_code_for({"K8": k8, "K4": R.MISSING}) == 1
    p5 = R.p5({"p5a": {"bitwise": {0: True, 4: True}}, "p5b": {"bitwise": False}, "p5c": {"mean_kl": 0.5}}, R.load_thresholds())
    assert not p5["ok"]
    code = R.exit_code_for({"K8": k8, "K4": R.MISSING}, precondition_failed=not p5["ok"])
    assert code == 3 and not R.is_gate_run(code)


def test_run_counts(th):
    assert R.run_counts([], th) == {"gate_runs": 0, "cycles_used": 0, "cycles_max": 2, "gate_runs_max": 3,
                                    "next_run_is_cycle": False, "may_run_again": True}
    r = R.run_counts([3, 1, 3, 3], th)  # exit 3 never counts
    assert (r["gate_runs"], r["cycles_used"], r["next_run_is_cycle"], r["may_run_again"]) == (1, 0, True, True)
    r = R.run_counts([1, 5], th)
    assert (r["gate_runs"], r["cycles_used"], r["may_run_again"]) == (2, 1, True)
    r = R.run_counts([1, 3, 4, 3, 5], th)
    assert (r["gate_runs"], r["cycles_used"], r["may_run_again"]) == (3, 2, False)
    r = R.run_counts([0], th)
    assert (r["cycles_used"], r["next_run_is_cycle"]) == (0, False)
    r = R.run_counts([5, 0], th)  # INCOMPLETE counts as a cycle
    assert r["cycles_used"] == 1


# ------------------------------------------------------------------------------- controls map


def test_controls_json_semantics(th, ctl):
    assert ctl["version"] == "exp037-controls-1" and ctl["margin_factor"] == 10 and ctl["negative_controls"] == []
    assert set(ctl["margin_units"]) == {"mean", "ratio", "count", "max"}
    ids = [c["id"] for c in ctl["required"]]
    assert ids == ["G4F32/1", "G4F32/6", "G4F32/19", "G4F16/6", "G5D32/20", "G5R1/26", "G5BP/27"]
    names = {1: "sigmoid_bias_select", 6: "renorm_topk", 19: "rope_restart_per_chunk", 20: "decode_window_256",
             22: "batched_rope_positions_in_padded_frame", 26: "batch_decode_pos_frozen",
             27: "batch_decode_pad_keys_visible"}  # DESIGN §4.1, port_mutants
    bound = {"G4F32/1": ("F6", "count"), "G4F32/6": ("F2", "mean"), "G4F32/19": ("F2", "mean"),
             "G4F16/6": ("kappa", "ratio"), "G5R1/26": ("R-parity", "mean"), "G5BP/27": ("d", "ratio")}
    for c in ctl["required"]:
        assert c["mutant_name"] == names[c["mutant"]] and c["arm"] == "K8"
        assert c["check"] in R.BLOCKING["K8"] or c["check"] in R.NOT_BLOCKING
        assert c["id"].split("/")[1] == str(c["mutant"])
        if c["id"] in bound:
            assert (c["leg"], c["leg_type"]) == bound[c["id"]]
    d20 = next(c for c in ctl["required"] if c["id"] == "G5D32/20")
    assert d20["leg"] == "decode" and d20["leg_type"] == {"top1": "count", "mean": "mean", "max": "max"}
    assert next(c for c in ctl["required"] if c["id"] == "G5R1/26")["incomplete_checks"] == ["g5_r1_K8", "g5_r1_K4"]
    assert next(c for c in ctl["required"] if c["id"] == "G5BP/27")["incomplete_checks"] == []
    # probes (decision (b)): G5BP/22, tiny mode only, same check and leg as the required control
    (p22,) = ctl["probes"]
    assert (p22["id"], p22["mutant"], p22["mutant_name"], p22["check"], p22["leg"], p22["leg_type"], p22["modes"]) == (
        "G5BP/22", 22, names[22], "g5_bp_K8_B8", "d", "ratio", ["tiny"])
    reg = ctl["registered"]
    assert reg["mutants"] == th["G2"]["real_weight_mutants"] and reg["layers"] == th["G2"]["real_weight_mutant_layers"]
    assert reg["if_not_caught"] == "FAIL"
    for check_id in ("g4_f32_K8", "g4_f16_K8", "g5_d32_K8", "g5_r1_K8", "g5_bp_K8_B8"):
        assert R.required_controls(check_id, ctl)
    assert R.required_controls("g4_n_K8", ctl) == []  # G4-N(i) is the registered gross backstop: no control


def test_port_mutant_names_agree_where_present(ctl):
    """controls.json's mutant names equal gate/port_mutants.py's MUTANT_IDS for
    every control mutant that module defines (W2 adds 19, 20, 22, 26 and 27),
    probes included."""
    from gate import port_mutants

    for c in ctl["required"] + ctl["probes"]:
        if c["mutant"] in port_mutants.MUTANT_IDS:
            assert port_mutants.MUTANT_IDS[c["mutant"]] == c["mutant_name"], c["id"]


def test_controls_outcome(th, ctl):
    checks = {"g4_f32_K8": R.g4_f32(_f32_series(), _f32_mutants(), F_TINY, th, ctl),
              "g4_f16_K8": R.g4_f16(_f16(), {6: _f16(port=1.0)}, th, ctl),
              "g5_d32_K8": R.g5_d32(_d32(), _d32_mut(mean_kl=1e-6), th, ctl),
              **R.g5_r1(_r1_arms(), FLOOR, M26, th, ctl),
              "g2_real_weight_mutants": R.check(R.FAIL, {"undetected": ["route_scale_2826"]})}
    bp = R.g5_bp(_bp_results(), M27, th, ctl)
    out = R.controls_outcome(checks, bp, ctl)
    assert [k for k in out if k != "G2/registered"] == [c["id"] for c in ctl["required"]]
    assert all(out[c["id"]]["caught"] is True for c in ctl["required"])
    assert out["G5BP/27"]["if_not_caught"].startswith("allowed_B = {1}") and "G5BP/22" not in out
    assert out["G2/registered"]["caught"] is False and out["G2/registered"]["undetected"] == ["route_scale_2826"]
    out = R.controls_outcome({}, None, ctl)
    assert all(v["caught"] is None for v in out.values())
    cal = R.calibrators(checks)
    assert cal["f_tiny"] == F_TINY and set(cal["tau"]) == {"T1-8", "T9"} and set(cal["csort_ranges"]) == {"T1", "T3", "T9"}
    assert cal["r3"] == {"T1-8": 0.001, "T9": 0.001}


def test_checks_json_round_trip(th, ctl):
    """Every check entry is strict JSON (the gate record), and the rules accept
    their inputs back from JSON (str keys for B and mutant ids)."""
    muts = json.loads(json.dumps({str(k): v for k, v in _f32_mutants().items()}))
    c = R.g4_f32(json.loads(json.dumps(_f32_series())), muts, F_TINY, th, ctl)
    assert c["state"] == R.PASS
    json.dumps(c, allow_nan=False)
    bp = R.g5_bp(json.loads(json.dumps(_bp_results())), json.loads(json.dumps(M27)), th, ctl)
    json.dumps(bp, allow_nan=False)
    assert bp["allowed_B"]["K8"] == [1, 2, 4, 8, 16]
    for out in (R.g5_r1(_r1_arms(), FLOOR, json.loads(json.dumps(M26)), th, ctl),
                {"x": R.g5_d32(_d32(), json.loads(json.dumps(_d32_mut(mean_kl=1e-6))), th, ctl)},
                {"y": R.g4_f16(_f16(), json.loads(json.dumps({6: _f16(port=1.0)})), th, ctl)}):
        json.dumps(out, allow_nan=False)
        assert all(v["state"] == R.PASS for v in out.values())


# ------------------------------------------------------------------------------- tripwire (§7.2)


def test_tripwire_constants_and_rule():
    c = R.tripwire_constants()
    assert c == {"H2_tripwire_ub": -0.10, "H2_tripwire_did_ub": -0.10, "H2_tripwire_q": 0.95}
    c["H2_tripwire_ub"] = 0.0
    assert R.TRIPWIRE_CONSTANTS["H2_tripwire_ub"] == -0.10  # a copy
    ub = -0.10
    assert R.tripwire_rule(ub, ub) == R.NOT_TRIPPED  # equality does not trip
    assert R.tripwire_rule(down(ub), down(ub)) == R.TRIPPED
    assert R.tripwire_rule(down(ub), ub) == R.NOT_TRIPPED  # the DiD leg vetoes
    assert R.tripwire_rule(ub, down(ub)) == R.NOT_TRIPPED
    assert R.tripwire_rule(down(ub), None, protocol_run=False) == R.TRIPPED
    assert R.tripwire_rule(ub, None, protocol_run=False) == R.NOT_TRIPPED
    assert R.tripwire_rule(None, None, h2_run=False) == R.NOT_RUN
    # W9's synthetic replicates: UB = -0.0999 and -0.1001
    assert R.tripwire_rule(-0.1001, -0.1001) == R.TRIPPED and R.tripwire_rule(-0.0999, -0.1001) == R.NOT_TRIPPED
    with pytest.raises(R.RuleInputError):
        R.tripwire_rule(None, -0.2)
    with pytest.raises(R.RuleInputError):
        R.tripwire_rule(-0.2, math.nan)
    x = np.linspace(-0.3, 0.1, 10001)
    assert R.tripwire_bound(x) == pytest.approx(float(np.quantile(x, 0.95)))
