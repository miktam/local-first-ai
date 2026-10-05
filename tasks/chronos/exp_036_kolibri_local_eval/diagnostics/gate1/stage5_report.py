# SPDX-License-Identifier: MIT
"""Stage 5 (FROZEN_RULES.md §7): the frozen mapping, applied mechanically. Runs on
the mini over the committed restricted copies; minutes; no model, no GPU.

It reads only the top-level `rule_inputs` of the newest diagnostics/gate1/out/
stage{0..4}_*.json (or the files given) and the committed gate record
results/gate/gate_20261005T050112Z.json (+ its phase files and _layers.csv),
and writes diagnostics/gate1/out/stage5_report_<UTC>.json (JSON only):

  * per check (G5, G2, G4, G3): every sign with its inputs, its value and
    whether it fired; the reproduction spread; the outcome (D / N / 1 / 2 / 3 /
    invalid / failure stands); the frozen consequence, its type and who acts
  * the gate's own evaluate code (g2_layers.evaluate, g3_oracle.evaluate,
    g4_e2e.evaluate, run_gate.arm_verdict) re-applied to the recorded measured
    values with only the permitted new thresholds; a FAIL stays a FAIL, and
    G5-B1 is reported as "non-blocking under a verdict-rule change", never a pass
  * the package and its cycle cost under D1 (ii) (§8)

It searches for no passing rule and chooses nothing: Andrei types any threshold
change; the main session pushes any gate fix or verdict-rule change. A missing,
non-finite or ambiguous input gives "failure stands" for its check (§1.5).

usage (kit root, on the mini, after pulling the committed stage outputs):
  ~/models/exp036-mini/venv312/bin/python diagnostics/gate1/stage5_report.py

It refuses to run (FROZEN_RULES.md 7, Integrity) if this script, any other script
here or FROZEN_RULES.md differs from what the stage headers recorded, if the tree
is dirty, or if a stage has more than one output without --<stage> FILE and
--rerun-reason <stage> TEXT (1.6).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import itertools
import json
import math
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]

import diaglib as D  # noqa: E402
from gate import common  # noqa: E402

STAGES = ("stage0_kernels", "stage1_g5", "stage2_g2", "stage3_g4", "stage4_g3")
REGISTERED_MUTANTS = ("one_plus_w_norm", "qknorm_after_rope", "renorm_topk", "rope_on_full", "rope_traditional",
                      "sigmoid_bias_select", "swap_sandwich_norms")
G2_TAILS = ("att8", "att29", "moe13", "moe30")
G2_POSITIONS = {0, 1, 511, 512, 513, 514, 515, 2047, 2048, 2049}
RECORDED = {"g2_max": 6.077697277069092, "g4_T9": 0.98889449772842, "g4_T9_n_decisive": 5943,
            "g3_T1": 1.3105577068977687, "g3_ratio": 0.9142729969109077 / 0.7053979070863484}
STANDS = "failure stands"


# ---------------------------------------------------------------------------
# Input access
# ---------------------------------------------------------------------------


class Missing(Exception):
    pass


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def need(ri: dict, *path, kind="num"):
    """A rule input; Missing if absent, non-finite or of the wrong kind."""
    cur = ri
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            raise Missing("/".join(map(str, path)))
        cur = cur[p]
    if kind == "num" and not finite(cur):
        raise Missing("/".join(map(str, path)) + " (not a finite number)")
    if kind == "bool" and not isinstance(cur, bool):
        raise Missing("/".join(map(str, path)) + " (not a boolean)")
    if kind == "str" and not isinstance(cur, str):
        raise Missing("/".join(map(str, path)) + " (not a string)")
    if kind == "list" and not isinstance(cur, list):
        raise Missing("/".join(map(str, path)) + " (not a list)")
    if kind == "dict" and not isinstance(cur, dict):
        raise Missing("/".join(map(str, path)) + " (not an object)")
    return cur


def sign(name: str, rule: str, fn) -> dict:
    """Evaluate one frozen sign: fn() -> (fired: bool, inputs: dict)."""
    try:
        fired, inputs = fn()
        return {"sign": name, "rule": rule, "inputs": inputs, "fired": bool(fired), "missing": None}
    except Missing as e:
        return {"sign": name, "rule": rule, "inputs": None, "fired": None, "missing": str(e)}


def any_fired(signs: list) -> bool:
    return any(s["fired"] is True for s in signs)


def any_missing(signs: list) -> list:
    return [s["missing"] for s in signs if s["missing"]]


def ALT(type_: str, id_: str, who: str) -> dict:
    """One permitted consequence (FROZEN_RULES.md 1.10): its type, a label, who acts.
    A check whose `alternatives` is empty stands."""
    assert type_ in ("threshold change", "gate fix", "verdict-rule change")
    return {"type": type_, "id": id_, "who": who}


# ---------------------------------------------------------------------------
# G5 (FROZEN_RULES.md §3)
# ---------------------------------------------------------------------------


def g5(s0: dict, s1: dict) -> dict:
    signs = [
        sign("G5-D1", "C_mean_kl > 1e-4, or C_top1_dis x C_n_positions > 2",
             lambda: (need(s1, "C_mean_kl") > 1e-4 or round(need(s1, "C_top1_dis") * need(s1, "C_n_positions")) > 2,
                      {k: s1.get(k) for k in ("C_mean_kl", "C_top1_dis", "C_n_positions")})),
        sign("G5-D2", "R_mean_kl_ref_batched > 2 x R_mean_kl_ref_single",
             lambda: (need(s1, "R_mean_kl_ref_batched") > 2.0 * need(s1, "R_mean_kl_ref_single"),
                      {k: s1.get(k) for k in ("R_mean_kl_ref_batched", "R_mean_kl_ref_single")})),
        sign("G5-D3", "R_top1_decisive_batched < 0.995 while R_top1_decisive_single >= 0.995",
             lambda: (need(s1, "R_top1_decisive_batched") < 0.995 and need(s1, "R_top1_decisive_single") >= 0.995,
                      {k: s1.get(k) for k in ("R_top1_decisive_batched", "R_top1_decisive_single", "R_n_decisive")})),
        sign("G5-D4", "R_n_batched_wrong_single_right - R_n_single_wrong_batched_right >= 3",
             lambda: (need(s1, "R_n_batched_wrong_single_right") - need(s1, "R_n_single_wrong_batched_right") >= 3,
                      {k: s1.get(k) for k in ("R_n_batched_wrong_single_right", "R_n_single_wrong_batched_right")})),
        sign("G5-D5", "kprobe_verdict == 'DEFECT'",
             lambda: (need(s0, "kprobe_verdict", kind="str") == "DEFECT", {"kprobe_verdict": s0.get("kprobe_verdict")})),
    ]
    loc = {k: s1.get(k) for k in ("L1_rows_bitwise_equal", "L1_max_pre_divergence_kl", "L1_first_divergence_step",
                                  "L2_max_kl", "L3_max_abs_dlogprob_seq0_seq8", "L3_max_abs_dlogprob_seq3_seq11",
                                  "B_share_kl_top5", "B_mean_kl_first_wave", "B_mean_kl_mid_run", "B_mean_kl_t0",
                                  "B_mean_kl_t1_2", "B_mean_kl_t3plus", "B_mean_kl_prompt_ge_513")}
    loc["kprobe_defects"] = s0.get("kprobe_defects")
    loc["readings"] = {
        "L2_cache_classes_differ_alone": (s1.get("L2_max_kl") > 1e-6) if finite(s1.get("L2_max_kl")) else None,
        "concentration_top5_ge_50pct": (s1.get("B_share_kl_top5") >= 0.5) if finite(s1.get("B_share_kl_top5")) else None,
    }
    rec_kl = D.RECORD["g5_batch_parity_K8_mean_kl"]
    b_kl, inst = s1.get("B_mean_kl"), s1.get("B_instrumented_run_equals_gate_run")
    rep = {"B_mean_kl": b_kl, "record": rec_kl, "spread": abs(b_kl - rec_kl) if finite(b_kl) else None,
           "B_mean_kl_equals_record": (b_kl == rec_kl) if finite(b_kl) else None,
           "B_instrumented_run_equals_gate_run": inst if isinstance(inst, bool) else None}
    rep["reproduces_gate_draw"] = rep["B_mean_kl_equals_record"] is True and rep["B_instrumented_run_equals_gate_run"] is True
    out = {"check": "g5_batch_parity_K8", "signs": signs, "localisers": loc, "reproduction": rep}
    if any_fired(signs):
        defects = s0.get("kprobe_defects") or []
        if isinstance(defects, list) and defects:
            out.update(outcome="G5-D", consequence="localised to an M5 kernel (stage 0 kprobe_defects); the stage-0 JSON "
                       "goes into an upstream MLX report. Two workarounds, each citing the kernel, of different types; "
                       "stage 5 chooses neither: a path that avoids the kernel with G5 re-measured at B = 8 (gate fix), "
                       "or K8 and K4 at B = 1 (verdict-rule change: run_gate.BLOCKING, plan_fix and the guard; its own "
                       "amendment on Andrei's go) (FROZEN_RULES 3)", type="gate fix or verdict-rule change",
                       who="main session; the B = 1 variant needs Andrei's go", localised=True,
                       alternatives=[ALT("gate fix", "G5 path avoiding the M5 kernel (re-measured at B = 8)", "main session"),
                                     ALT("verdict-rule change", "G5 K8/K4 at B = 1 (M5 kernel)",
                                         "main session drafts, Andrei's go at the time")])
        else:
            out.update(outcome="G5-D", consequence=f"{STANDS}: not localised by this suite; a gate fix only if a "
                       "head_dim-128 tiny parity test reproduces the class in bf16 and fp32 (port make_cache / masks "
                       "under BatchKVCache / BatchRotatingKVCache, runner batch construction, or mlx_lm BatchGenerator)",
                       type=None, who="main session (localisation test)", localised=False, alternatives=[])
    elif any_missing(signs):
        out.update(outcome=STANDS, consequence=f"{STANDS}: missing rule input(s) {any_missing(signs)} (FROZEN_RULES 1.5)",
                   type=None, who=None, localised=None, alternatives=[])
    elif not rep["reproduces_gate_draw"]:
        out.update(outcome=STANDS, consequence=f"{STANDS}: no defect sign, but stage 1 did not reproduce the gate's draw "
                   f"(B_mean_kl equals the record: {rep['B_mean_kl_equals_record']}; instrumented run equals the gate run: "
                   f"{rep['B_instrumented_run_equals_gate_run']}); G5-N needs both (FROZEN_RULES 1.4, 3)",
                   type=None, who=None, localised=None, alternatives=[])
    else:
        out.update(outcome="G5-N", consequence="G5-B1 (D3, 'Run Kolibri at B = 1'): K8 and K4 run at B = 1; "
                   "run_gate.BLOCKING, plan_fix and the guard; its own amendment on Andrei's go, disclosed as a rule "
                   "change made after the result; the gate's 0.307 > 0.151 stands", type="verdict-rule change",
                   who="main session drafts, Andrei's go at the time", localised=None,
                   alternatives=[ALT("verdict-rule change", "G5-B1", "main session drafts, Andrei's go at the time")])
    return out


# ---------------------------------------------------------------------------
# G2 (FROZEN_RULES.md §4)
# ---------------------------------------------------------------------------


def g2(s2: dict, s3: dict, tiny: bool) -> dict:
    def tail_sign():
        tails = need(s2, "tails", kind="dict")
        top = need(s3, "top10_kl_positions", kind="list")
        keys = list(tails) if tiny else list(G2_TAILS)
        hits, inputs = [], {}
        for k in keys:
            ratio = need(s2, "tails", k, "port_max_over_emu_max")
            tx = need(s2, "tails", k, "port_max_text", kind="str")
            ps = need(s2, "tails", k, "port_max_position")
            near = [t for t in top if t.get("text") == tx and abs(int(t.get("position")) - int(ps)) <= 2]
            inputs[k] = {"port_max_over_emu_max": ratio, "port_max_text": tx, "port_max_position": ps,
                         "near_top10": near}
            if ratio > 3.0 and near:
                hits.append(k)
        return bool(hits), {"tails": inputs, "firing": hits}

    def t9_sign():
        p = need(s2, "t9_disagree_frac_port", kind="dict")
        e = need(s2, "t9_disagree_frac_emu", kind="dict")
        if set(p) != set(e) or not p:
            raise Missing("t9 bucket keys")
        hits = [b for b in p if need(p, b) > 2.0 * need(e, b)]
        return bool(hits), {"port": p, "emu": e, "firing": hits}

    signs = [
        sign("G2-D1", "flip_r_t > r_p99_layer20",
             lambda: (need(s2, "flip_r_t") > need(s2, "r_p99_layer20"),
                      {k: s2.get(k) for k in ("flip_r_t", "r_p50_layer20", "r_p99_layer20", "flip_sigma_t_port",
                                              "flip_sigma_t_emu")})),
        sign("G2-D2", "flip_position in 0-1, 511-515, 2047-2049, or flip_e_attn_max_pct_16_20 >= 99",
             lambda: (int(need(s2, "flip_position")) in G2_POSITIONS or need(s2, "flip_e_attn_max_pct_16_20") >= 99.0,
                      {k: s2.get(k) for k in ("flip_text", "flip_position", "flip_position_class",
                                              "flip_e_attn_max_pct_16_20")})),
        sign("G2-D3", "flip_emu_flips_same is false and flip_gap_over_sigma_t_emu > 6",
             lambda: ((not need(s2, "flip_emu_flips_same", kind="bool")) and need(s2, "flip_gap_over_sigma_t_emu") > 6.0,
                      {k: s2.get(k) for k in ("flip_emu_flips_same", "flip_gap", "flip_gap_over_sigma_t_emu")})),
        sign("G2-D4", "attention 8/29 or MoE 13/30: port_max_over_emu_max > 3 and the port's max within +-2 tokens "
                      "of a stage-3 top10_kl_positions entry in the same text", tail_sign),
        sign("G2-D5", "any T9 bucket: t9_disagree_frac_port > 2 x t9_disagree_frac_emu", t9_sign),
    ]
    rep = {}
    try:
        m = need(s2, "port_max_gap_over_sigma")
        rep = {"reproduces_gate": s2.get("reproduces_gate"), "port_max_gap_over_sigma": m, "record": RECORDED["g2_max"],
               "spread": 0.0 if s2.get("reproduces_gate") is True else abs(m - RECORDED["g2_max"])}
    except Missing as e:
        rep = {"missing": str(e)}
    out = {"check": "g2_bf16_natural_selection", "signs": signs, "reproduction": rep,
           "M_emu": s2.get("M_emu"), "emu_n_beyond_6sigma": s2.get("emu_n_beyond_6sigma")}
    if any_fired(signs):
        out.update(outcome="G2-D", consequence=f"{STANDS}; inspect. A gate fix in the port's mask or cache only if a "
                   "tiny G1 boundary test at that position class fails before the fix", type=None,
                   who="main session (inspection)", alternatives=[])
        return out
    miss = any_missing(signs)
    try:
        m_emu = need(s2, "M_emu")
        flips = need(s2, "flip_emu_flips_same", kind="bool")
    except Missing as e:
        miss = miss + [str(e)]
    if miss or "spread" not in rep:
        out.update(outcome=STANDS, consequence=f"{STANDS}: missing rule input(s) {miss or rep} (FROZEN_RULES 1.5)", type=None,
                   who=None, alternatives=[])
        return out
    if m_emu >= 6.0:
        new = min(8.0, 1.2 * m_emu)
        clears = new - RECORDED["g2_max"]
        if clears > rep["spread"]:
            out.update(outcome="G2-1", consequence=f"A1: bf16_selection_gap_sigma 6.0 -> {new!r} (min(8.0, 1.2 x M_emu)); "
                       "original 6.0, observed 6.0777", type="threshold change", who="Andrei types it",
                       new_thresholds={"G2.bf16_selection_gap_sigma": new}, clears_by=clears,
                       alternatives=[ALT("threshold change", "A1", "Andrei types it")])
        else:
            out.update(outcome="G2-3", consequence=f"{STANDS}: A1 abandoned, the recorded 6.0777 clears {new!r} by "
                       f"{clears!r}, not more than the reproduction spread {rep['spread']!r} (FROZEN_RULES 1.4)", type=None,
                       who=None, alternatives=[])
    elif flips:
        out.update(outcome="G2-2", consequence="A3, a verdict-rule change (FROZEN_RULES 1.10, 4): a port disagreement that "
                   "emu shares at the same (token, layer) is exempt from the sigma rule; every other disagreement keeps "
                   f"< 6 sigma_l; its own amendment on Andrei's go; if he declines, {STANDS}", type="verdict-rule change",
                   who="main session drafts, Andrei's go at the time",
                   alternatives=[ALT("verdict-rule change", "A3", "main session drafts, Andrei's go at the time")])
    else:
        out.update(outcome="G2-3", consequence=f"{STANDS} (M_emu < 6.0 and emu does not flip the token)", type=None,
                   who=None, alternatives=[])
    return out


# ---------------------------------------------------------------------------
# G4 (FROZEN_RULES.md §5)
# ---------------------------------------------------------------------------


def g4(s3: dict) -> dict:
    def bug():
        mk = {k: need(s3, "bug_mean_kl", k) for k in ("T1-8", "T9")}
        td = need(s3, "bug_top1_decisive", kind="dict")
        if "T1-8" not in td or len(td) != 4:
            raise Missing("bug_top1_decisive must hold T1-8 and the three buckets")
        tdv = {k: need(td, k) for k in td}
        bw = need(s3, "bug_boundary_window_max")
        return (any(v > 1e-3 for v in mk.values()) or any(v < 0.999 for v in tdv.values()) or bw > 1e-3,
                {"bug_mean_kl": mk, "bug_top1_decisive": tdv, "bug_boundary_window_max": bw})

    def chunk():
        mx_, nch = need(s3, "chunk64_max_kl"), need(s3, "chunk64_n_decisive_top1_changes")
        rk, rt = need(s3, "chunk64_vs_ref_mean_kl"), need(s3, "chunk64_vs_ref_top1_decisive")
        return (mx_ > 1e-6 or nch >= 1 or rk > 1e-3 or rt < 0.999,
                {"chunk64_max_kl": mx_, "chunk64_n_decisive_top1_changes": nch, "chunk64_vs_ref_mean_kl": rk,
                 "chunk64_vs_ref_top1_decisive": rt})

    def mcn():
        m = need(s3, "mcnemar", kind="dict")
        if "T9" not in m or len(m) != 4:
            raise Missing("mcnemar must hold T9 and the three buckets")
        hits, inputs = [], {}
        for k in m:
            p, pm, em = need(m, k, "p_one_sided"), need(m, k, "port_miss"), need(m, k, "emu_miss")
            inputs[k] = {"p_one_sided": p, "port_miss": pm, "emu_miss": em, "n10": m[k].get("n10"), "n01": m[k].get("n01")}
            if p < 0.05 and pm > 1.25 * em:
                hits.append(k)
        return bool(hits), {"sets": inputs, "firing": hits}

    def pdn():
        mean, lo, hi = (need(s3, "paired_dnll_T9", k) for k in ("mean", "ci99_lo", "ci99_hi"))
        return ((lo > 0 or hi < 0) and abs(mean) > 0.02), {"mean": mean, "ci99_lo": lo, "ci99_hi": hi}

    def exc():
        hits, inputs = [], {}
        for t in ("T5", "T6"):
            row = need(s3, "excess", t, kind="dict")
            for a, b in (("kl_port", "kl_emu"), ("miss_port", "miss_emu")):
                p, e = need(row, a), need(row, b)
                inputs[f"{t}.{a}"], inputs[f"{t}.{b}"] = p, e
                if p > 0 and p > 2.0 * max(e, 0.0):
                    hits.append(f"{t}.{a}")
        return bool(hits), {"values": inputs, "firing": hits}

    signs = [
        sign("G4-D1", "bug test: bug_mean_kl > 1e-3 (T1-8 or T9), bug_top1_decisive < 0.999 in any of the four sets, "
                      "or bug_boundary_window_max > 1e-3", bug),
        sign("G4-D2", "chunk64_max_kl > 1e-6 or chunk64_n_decisive_top1_changes >= 1; chunk64_vs_ref_mean_kl > 1e-3 "
                      "or chunk64_vs_ref_top1_decisive < 0.999", chunk),
        sign("G4-D3", "T9 or any bucket: p_one_sided < 0.05 and port_miss > 1.25 x emu_miss", mcn),
        sign("G4-D4", "paired_dnll_T9 [ci99_lo, ci99_hi] excludes 0 and |mean| > 0.02", pdn),
        sign("G4-D5", "n_kl_port_gt5_emu_lt1 >= 3",
             lambda: (need(s3, "n_kl_port_gt5_emu_lt1") >= 3, {"n_kl_port_gt5_emu_lt1": s3.get("n_kl_port_gt5_emu_lt1")})),
        sign("G4-D6", "T5 or T6, KL or miss rate: port excess > 0 and > 2 x max(emu excess, 0)", exc),
    ]
    rep = {}
    try:
        v = need(s3, "bf16_T9_top1_decisive")
        rep = {"bf16_T9_top1_decisive": v, "bf16_T9_decisive_miss": s3.get("bf16_T9_decisive_miss"),
               "record": RECORDED["g4_T9"], "spread": abs(v - RECORDED["g4_T9"])}
    except Missing as e:
        rep = {"missing": str(e)}
    out = {"check": "g4_K8_backstop", "signs": signs, "reproduction": rep,
           "E_T9": s3.get("E_T9"), "emu_T9_decisive_miss": s3.get("emu_T9_decisive_miss"),
           "emu_T9_n_decisive": s3.get("emu_T9_n_decisive"), "emu_decisive_set_equals_gate": s3.get("emu_decisive_set_equals_gate")}
    if any_fired(signs):
        out.update(outcome="G4-D", consequence=f"{STANDS}; a code question. A gate fix in port/kolibri1.py's long-context "
                   "path (NoPE layers, cache growth, chunk edges) only if a tiny G1 test at the boundary reproduces it",
                   type=None, who="main session (inspection)", alternatives=[])
        return out
    miss = any_missing(signs)
    try:
        e_t9 = need(s3, "E_T9")
        same = need(s3, "emu_decisive_set_equals_gate", kind="bool")
        n_dec = need(s3, "emu_T9_n_decisive")
    except Missing as e:
        miss = miss + [str(e)]
    if miss or "spread" not in rep:
        out.update(outcome=STANDS, consequence=f"{STANDS}: missing rule input(s) {miss or rep} (FROZEN_RULES 1.5)", type=None,
                   who=None, alternatives=[])
        return out
    new = max(0.985, min(0.99, 1.0 - 1.25 * (1.0 - e_t9)))
    out["C1_prime_value"] = new
    clears = RECORDED["g4_T9"] - new
    if not same or n_dec != RECORDED["g4_T9_n_decisive"]:
        out.update(outcome="G4-2", consequence=f"{STANDS}: E_T9 not measured on the gate's 5,943 decisive positions",
                   type=None, who=None, alternatives=[])
    elif not clears > rep["spread"]:
        out.update(outcome="G4-2", consequence=f"{STANDS}: the computed value {new!r} would not change the recorded "
                   f"verdict (0.98889 clears it by {clears!r}; reproduction spread {rep['spread']!r})", type=None,
                   who=None, alternatives=[])
    else:
        out.update(outcome="G4-1", consequence=f"C1': K8_backstop_decisive_top1_min 0.99 -> {new!r}; original 0.99, "
                   "observed 0.98889 (66 / 5,943); the key also applies to T1-8 (0.99652), disclosed",
                   type="threshold change", who="Andrei types it",
                   new_thresholds={"G4.K8_backstop_decisive_top1_min": new}, clears_by=clears,
                   alternatives=[ALT("threshold change", "C1'", "Andrei types it")])
    return out


# ---------------------------------------------------------------------------
# G3 (FROZEN_RULES.md §6, no anchor)
# ---------------------------------------------------------------------------


def g3(s4: dict) -> dict:
    validity = {}
    invalid = []
    try:
        d1 = {t: need(s4, "d1_k8_minus_ref_bpb", t) for t in ("T1", "T2", "T3", "T4", "T5", "T6")}
        validity["d1_k8_minus_ref_bpb"] = d1
        if any(abs(v) > 0.005 for v in d1.values()):
            invalid.append("|d1_k8_minus_ref_bpb| > 0.005")
    except Missing as e:
        invalid.append(f"missing {e}")
    for key in ("ref_bpb_reproduces_gate", "peer_bpb_reproduces_record"):
        try:
            v = need(s4, key, kind="bool")
            validity[key] = v
            if not v:
                invalid.append(f"{key} is false")
        except Missing as e:
            invalid.append(f"missing {e}")
    validity["peer_bpb_max_abs_diff"] = s4.get("peer_bpb_max_abs_diff")

    def windows():
        per = {}
        for t in ("T1", "T2", "T5", "T6"):
            p10, fr = need(s4, "window_ratio", t, "p10"), need(s4, "window_ratio", t, "frac_kolibri_better")
            per[t] = {"p10": p10, "frac_kolibri_better": fr, "holds": p10 >= 1.1 and fr < 0.10}
        return all(v["holds"] for v in per.values()), per

    def mutants():
        p01 = need(s4, "mutant_dnll_p01", kind="dict")
        if set(p01) != set(REGISTERED_MUTANTS):
            raise Missing(f"mutant_dnll_p01 must hold the 7 registered mutants, has {sorted(p01)}")
        vals = {m: {t: need(p01, m, t) for t in ("T1", "T5")} for m in REGISTERED_MUTANTS}
        hits = [f"{m}.{t}" for m, v in vals.items() for t, x in v.items() if x < 0.24]
        return bool(hits), {"mutant_dnll_p01": vals, "firing": hits}

    def from64():
        v = {t: need(s4, "ref_bpb_from64_rel_diff", t) for t in ("T1", "T2", "T3", "T4", "T5", "T6")}
        return any(abs(x) > 0.05 for x in v.values()), {"ref_bpb_from64_rel_diff": v}

    signs = [
        sign("G3-D1", "for each of T1, T2, T5 and T6: window_ratio p10 >= 1.1 and frac_kolibri_better < 0.10", windows),
        sign("G3-D2", "any registered mutant with mutant_dnll_p01 < 0.24 on T1 or T5", mutants),
        sign("G3-D3", "|ref_bpb_from64_rel_diff| > 0.05 on any of T1-T6", from64),
    ]
    out = {"check": "g3_bpb_per_text + g3_bpb_vs_peers", "validity": validity, "invalid": invalid, "signs": signs,
           "reproduction": {"ref_bpb_reproduces_gate": validity.get("ref_bpb_reproduces_gate"), "spread": 0.0
                            if validity.get("ref_bpb_reproduces_gate") is True else None}}
    if invalid:
        out.update(outcome="invalid", consequence=f"{STANDS}: stage 4 invalid ({'; '.join(invalid)})", type=None,
                   who=None, alternatives=[])
    elif any_fired(signs):
        out.update(outcome="G3-D", consequence=f"{STANDS}: misreading signs; no G3 amendment and no anchor route (D4 "
                   "withdrawn)", type=None, who=None, alternatives=[])
    elif any_missing(signs):
        out.update(outcome=STANDS, consequence=f"{STANDS}: missing rule input(s) {any_missing(signs)} (FROZEN_RULES 1.5)",
                   type=None, who=None, alternatives=[])
    else:
        out.update(outcome="G3-1", consequence="B1 permitted, labelled a post-hoc gross-sanity bound: "
                   "ref_bpb_per_text_max 1.2 -> 1.5 (observed T1 1.3106), ref_bpb_mean_vs_best_peer_max 1.25 -> 1.40 "
                   "(observed 1.2961); the amendment states the blind spot (FROZEN_RULES 9 item 7). If Andrei declines: G3-2, "
                   f"{STANDS}", type="threshold change", who="Andrei types it (or declines: G3-2)",
                   new_thresholds={"G3.ref_bpb_per_text_max": 1.5, "G3.ref_bpb_mean_vs_best_peer_max": 1.40},
                   clears_by={"per_text": 1.5 - RECORDED["g3_T1"], "ratio": 1.40 - RECORDED["g3_ratio"]},
                   alternatives=[ALT("threshold change", "B1", "Andrei types it (or declines: G3-2)")])
    return out


# ---------------------------------------------------------------------------
# The gate's own evaluate code on the recorded values
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def blocking_without(run_gate, removed: set):
    saved = copy.deepcopy(run_gate.BLOCKING)
    try:
        for arm in run_gate.BLOCKING:
            run_gate.BLOCKING[arm] = [c for c in run_gate.BLOCKING[arm] if c not in removed]
        yield
    finally:
        run_gate.BLOCKING.clear()
        run_gate.BLOCKING.update(saved)


def g2_res_from_record(phase2: dict, phase3: dict, csv_path: Path) -> tuple[dict, dict]:
    """The g2_layers.run() result rebuilt from the committed record (phase3 data +
    _layers.csv), and the emu statistics of phase 2."""
    g = phase3["data"]["g2"]
    rows = {"fp32_forced": [], "bf16_forced": [], "t9_bf16_forced": [], "g2q_K8": [], "g2q_K4": []}
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {"layer": int(r["layer"]), "branch": r["branch"], "median": float(r["median"]),
                   "p99": float(r["p99"]), "max": float(r["max"])}
            if r["bucket"]:
                row["bucket"] = [int(x) for x in r["bucket"].split("-")]
            rows[r["mode"]].append(row)
    res = {"layers_fp32": rows["fp32_forced"], "layers_bf16": rows["bf16_forced"], "layers_t9": rows["t9_bf16_forced"],
           "natural_fp32": copy.deepcopy(g["natural_fp32"]), "natural_bf16": copy.deepcopy(g["natural_bf16"]),
           "embedding": g["embedding"], "head": g["head"], "router_margin": g["router_margin"],
           "real_weight_mutants": g["real_weight_mutants"],
           "layers_g2q": {arm: {"layers": rows[f"g2q_{arm}"], **g["layers_g2q"][arm]} for arm in ("K8", "K4")}}
    return res, phase2["data"]["emu"]


def reevaluate(results: dict) -> dict:
    from gate import run_gate
    from gate.checks import g2_layers, g3_oracle, g4_e2e

    rec = common.read_json(D.GATE_RECORD)
    th = common.load_thresholds()
    checks = copy.deepcopy(rec["checks"])
    phase2 = common.read_json(D.GATE_PHASES / "phase2.json")
    phase3 = common.read_json(D.GATE_PHASES / "phase3.json")
    out = {"record": D.GATE_RECORD.name, "record_sha256": D.sha256_file(D.GATE_RECORD), "applied": {}}

    # G2: rebuild, check that the original thresholds reproduce every recorded G2 verdict, then apply.
    res, emu = g2_res_from_record(phase2, phase3, D.GATE_RECORD.with_name(D.GATE_RECORD.stem + "_layers.csv"))
    base = g2_layers.evaluate(copy.deepcopy(res), emu, th["G2"])
    same = {cid: base[cid]["pass"] == rec["checks"][cid]["pass"] for cid in base}
    out["g2_rebuild_reproduces_record"] = same
    g2r = results["G2"]
    nb = [r["n_beyond_6sigma"] for r in res["natural_bf16"]]
    if g2r["outcome"] == "G2-1" and all(same.values()):
        new = g2r["new_thresholds"]["G2.bf16_selection_gap_sigma"]
        if max(nb) > 1:
            out["applied"]["G2"] = {"note": "a layer holds > 1 disagreement beyond 6 sigma: the record cannot be "
                                            "re-evaluated at a new sigma exactly; not applied"}
        else:
            r2 = copy.deepcopy(res)
            for r in r2["natural_bf16"]:
                r["n_beyond_6sigma"] = int(r["n_beyond_6sigma"] > 0 and r["max_gap_over_sigma"] >= new)
            th2 = dict(th["G2"], bf16_selection_gap_sigma=new)
            ev = g2_layers.evaluate(r2, emu, th2)
            checks["g2_bf16_natural_selection"] = ev["g2_bf16_natural_selection"]
            out["applied"]["G2"] = {"threshold": new, "g2_bf16_natural_selection_pass": ev["g2_bf16_natural_selection"]["pass"]}
    elif g2r["outcome"] == "G2-2" and all(same.values()):
        r2 = copy.deepcopy(res)
        layer = next((r for r in r2["natural_bf16"] if r["n_beyond_6sigma"] > 0), None)
        if layer is not None and sum(nb) == 1:
            layer["n_beyond_6sigma"] = 0          # the shared flip is exempt under A3 (simulated verdict-rule change)
        ev = g2_layers.evaluate(r2, emu, th["G2"])
        checks["g2_bf16_natural_selection"] = ev["g2_bf16_natural_selection"]
        out["applied"]["G2"] = {"verdict_rule_change": "A3 (simulated: the single beyond-6-sigma disagreement is the "
                                                       "shared flip); only on Andrei's go",
                                "g2_bf16_natural_selection_pass": ev["g2_bf16_natural_selection"]["pass"]}

    # G4
    if results["G4"]["outcome"] == "G4-1":
        new = results["G4"]["new_thresholds"]["G4.K8_backstop_decisive_top1_min"]
        th4 = dict(th["G4"], K8_backstop_decisive_top1_min=new)
        ev = g4_e2e.evaluate(rec["checks"]["g4_K8_backstop"]["measured"], "K8", th4)
        checks["g4_K8_backstop"] = ev["g4_K8_backstop"]
        out["applied"]["G4"] = {"threshold": new, "g4_K8_backstop_pass": ev["g4_K8_backstop"]["pass"]}

    # G3
    if results["G3"]["outcome"] == "G3-1":
        th3 = dict(th["G3"], ref_bpb_per_text_max=1.5, ref_bpb_mean_vs_best_peer_max=1.40)
        m = rec["checks"]["g3_bpb_vs_peers"]["measured"]
        ev = g3_oracle.evaluate(rec["checks"]["g3_bpb_per_text"]["measured"], {"peers": m["peers"], "source": m["source"]},
                                rec["checks"]["g3_ref_mutants"]["measured"]["mutants"], th3, False)
        for cid in ("g3_bpb_per_text", "g3_bpb_vs_peers", "g3_ref_mutants"):
            checks[cid] = ev[cid]
        out["applied"]["G3"] = {"thresholds": {"ref_bpb_per_text_max": 1.5, "ref_bpb_mean_vs_best_peer_max": 1.40},
                                **{f"{cid}_pass": ev[cid]["pass"] for cid in ("g3_bpb_per_text", "g3_bpb_vs_peers")},
                                "condition": "only if Andrei types B1"}

    # G5: never a pass; non-blocking only under a verdict-rule change (G5-B1, or the M5-kernel B = 1 workaround).
    g5_types = {a["type"] for a in results["G5"].get("alternatives") or []}
    removed = set()
    if results["G5"]["outcome"] == "G5-N":
        removed.add("g5_batch_parity_K8")
        out["applied"]["G5"] = {"g5_batch_parity_K8": "non-blocking under a verdict-rule change (G5-B1); its recorded "
                                                      "pass stays False"}
    elif "verdict-rule change" in g5_types:
        out["applied"]["G5"] = {"g5_batch_parity_K8": "M5 kernel localised: blocking and failing as recorded here. Under "
                                                      "the B = 1 verdict-rule change it would be non-blocking "
                                                      "(verdict_if_kernel_B1 below); under the gate fix it must be "
                                                      "re-measured at B = 8 on the next gate run"}
    with blocking_without(run_gate, removed):
        v8 = run_gate.arm_verdict("K8", checks, False)
        v4 = run_gate.arm_verdict("K4", checks, False)
    out["verdict_with_permitted_changes"] = {"K8": {"verdict": v8[0], "failing": v8[1], "missing": v8[2]},
                                             "K4": {"verdict": v4[0], "failing": v4[1], "missing": v4[2]},
                                             "note": "offline re-evaluation of recorded values; not a gate record. "
                                                     "A FAIL stays a FAIL; G5-B1 is never a pass"}
    if results["G5"]["outcome"] != "G5-N" and "verdict-rule change" in g5_types:
        with blocking_without(run_gate, {"g5_batch_parity_K8"}):
            k8 = run_gate.arm_verdict("K8", checks, False)
        out["verdict_if_kernel_B1"] = {"K8": {"verdict": k8[0], "failing": k8[1], "missing": k8[2]},
                                       "note": "only under the M5-kernel B = 1 verdict-rule change, on Andrei's go"}
    return out


# ---------------------------------------------------------------------------
# Package and cycle cost (FROZEN_RULES.md §8)
# ---------------------------------------------------------------------------


def amendments_of(choice: tuple) -> list[str]:
    """FROZEN_RULES.md 1.10 and 8: every threshold change in one amendment (typed by Andrei), every gate
    fix in one amendment (main session), each verdict-rule change its own amendment."""
    thr = [a["id"] for a in choice if a["type"] == "threshold change"]
    gf = [a["id"] for a in choice if a["type"] == "gate fix"]
    vr = [a["id"] for a in choice if a["type"] == "verdict-rule change"]
    return (([f"threshold change ({', '.join(thr)})"] if thr else []) + ([f"gate fix ({'; '.join(gf)})"] if gf else [])
            + [f"verdict-rule change ({v})" for v in vr])


def package(results: dict, fix_cycles_max: int) -> dict:
    """Every combination of the permitted consequences and its cycle cost; stage 5 chooses none of them."""
    per_check = {k: {"outcome": r["outcome"], "type": r.get("type"), "who": r.get("who"),
                     "alternatives": r.get("alternatives") or []} for k, r in results.items()}
    stands = [k for k, r in results.items() if not r.get("alternatives")]
    if stands:
        return {"row": "any check at 'failure stands' with no localised, test-backed fix",
                "checks_standing": stands, "packages": [], "any_fits": False,
                "next_gate_run": "not useful; the route is stop and publish (D8), on Andrei's go", "per_check": per_check}
    keys = list(results)
    packages = []
    for choice in itertools.product(*(results[k]["alternatives"] for k in keys)):
        am = amendments_of(choice)
        fits = len(am) <= fix_cycles_max
        packages.append({"consequences": {k: a["id"] for k, a in zip(keys, choice)}, "amendments": am, "cycles": len(am),
                         "fits": fits, "next_gate_run": "the last" if fits else "the gate cannot pass: stop and publish (D8)"})
    return {"packages": packages, "any_fits": any(p["fits"] for p in packages), "cycles_used_so_far": 0,
            "fix_cycles_max": fix_cycles_max,
            "condition": "every threshold change is typed by Andrei and every verdict-rule change needs his go at the "
                         "time; if he declines one, that check's failure stands and the route is D8",
            "per_check": per_check}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def newest(d: Path, stage: str) -> list[Path]:
    return sorted(d.glob(f"{stage}_2*.json"))


def current_scripts_sha256() -> dict:
    """What diaglib.Run.header records as scripts_sha256, computed from the files here now."""
    return ({q.name: D.sha256_file(q) for q in sorted(D.HERE.glob("*.py"))}
            | {"run_stages.sh": D.sha256_file(D.HERE / "run_stages.sh")})


def integrity(objs: dict, rules: Path | None, smoke: bool) -> dict:
    """FROZEN_RULES.md 7 Integrity: the mapping that runs is the frozen one, over outputs of the frozen
    scripts, on a clean tree. Returns what was checked; raises SystemExit on any mismatch."""
    here = current_scripts_sha256()
    rules_sha = D.sha256_file(rules) if rules else None
    gate_sha = D.sha256_file(D.GATE_RECORD)
    problems = []
    if rules is None:
        problems.append("FROZEN_RULES.md not found")
    for s, obj in objs.items():
        h = obj.get("header") or {}
        rec = h.get("scripts_sha256")
        if not isinstance(rec, dict):
            problems.append(f"{s}: no scripts_sha256 in its header")
        elif rec != here:
            diff = sorted(k for k in set(rec) | set(here) if rec.get(k) != here.get(k))
            problems.append(f"{s}: scripts_sha256 differs from the files here for {diff}")
        if (h.get("frozen_rules") or {}).get("sha256") != rules_sha:
            problems.append(f"{s}: its FROZEN_RULES.md sha256 differs from this file's")
        if (h.get("gate_record") or {}).get("sha256") != gate_sha:
            problems.append(f"{s}: its gate record sha256 differs from the committed record's")
    dirty = common.git_dirty()
    if dirty is not False and not smoke:
        problems.append(f"the kit's working tree is dirty (git_dirty = {dirty})")
    if problems:
        raise SystemExit("stage5: refused (FROZEN_RULES.md 7, Integrity): " + "; ".join(problems))
    return {"scripts_sha256": here, "frozen_rules_sha256": rules_sha, "gate_record_sha256": gate_sha,
            "all_stage_headers_match": True, "git_dirty": dirty, "dirty_tree_allowed_smoke": bool(smoke and dirty)}


def main() -> int:
    ap = argparse.ArgumentParser(prog="stage5_report.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=str(D.REPO_OUT), help="where the restricted stage copies are")
    ap.add_argument("--out", help="output directory (default --dir)")
    for s in STAGES:
        ap.add_argument(f"--{s}", help=f"explicit {s} file (needed, with --rerun-reason, when --dir holds more than one)")
    ap.add_argument("--rerun-reason", nargs=2, action="append", default=[], metavar=("STAGE", "TEXT"),
                    help="FROZEN_RULES.md 1.6: the named script defect and the reason a stage was re-run")
    ap.add_argument("--tiny", action="store_true", help="smoke test on tiny outputs (tail keys as found)")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke tests only: allows a dirty tree; refuses to write into out/; not for the record")
    args = ap.parse_args()
    d = Path(args.dir)
    outdir = Path(args.out) if args.out else d
    if args.smoke and outdir.resolve() == D.REPO_OUT.resolve():
        raise SystemExit("stage5: --smoke never writes into diagnostics/gate1/out")
    reasons = {}
    for st, text in args.rerun_reason:
        if st not in STAGES:
            raise SystemExit(f"stage5: --rerun-reason names an unknown stage {st!r}")
        reasons[st] = text
    rules = D.frozen_rules_path()
    files, ri, objs = {}, {}, {}
    for s in STAGES:
        given = getattr(args, s)
        found = newest(d, s)
        if given:
            f = Path(given)
            others = [c for c in found if c.resolve() != f.resolve()]
        elif found:
            f, others = found[-1], found[:-1]
        else:
            raise SystemExit(f"stage5: no {s} output in {common.redact_path(d)}")
        if others and not (given and s in reasons):
            raise SystemExit(f"stage5: refused (FROZEN_RULES.md 1.6): {s} has {len(others) + 1} outputs "
                             f"({[c.name for c in [f] + others]}); name the one to use with --{s} and give the script "
                             f"defect and the reason for the re-run with --rerun-reason {s} TEXT")
        obj = common.read_json(f)
        if "rule_inputs" not in obj:
            raise SystemExit(f"stage5: {f.name} has no rule_inputs")
        files[s] = {"file": f.name, "sha256": D.sha256_file(f), "mode": obj.get("header", {}).get("mode"),
                    "utc": obj.get("header", {}).get("utc")}
        if others:
            files[s]["other_outputs"] = [{"file": c.name, "sha256": D.sha256_file(c)} for c in others]
            files[s]["rerun_reason"] = reasons[s]
        ri[s] = obj["rule_inputs"]
        objs[s] = obj
    modes = {v["mode"] for v in files.values()}
    tiny = args.tiny or modes == {"tiny"}
    if not tiny and modes != {"real"}:
        raise SystemExit(f"stage5: stage outputs mix modes {modes}")
    checked = integrity(objs, rules, args.smoke)
    results = {
        "G5": g5(ri["stage0_kernels"], ri["stage1_g5"]),
        "G2": g2(ri["stage2_g2"], ri["stage3_g4"], tiny),
        "G4": g4(ri["stage3_g4"]),
        "G3": g3(ri["stage4_g3"]),
    }
    th = common.load_thresholds()
    report = {
        "header": {"experiment": "exp_036", "stage": "stage5_report", "utc": common.utc_stamp(),
                   "mode": "tiny" if tiny else "real", "smoke": bool(args.smoke),
                   "for_the_record": not (args.smoke or tiny),
                   "what": "the frozen mapping applied to the stage outputs; not a gate record, not a verdict",
                   "frozen_rules": {"file": rules.relative_to(D.KIT).as_posix() if rules else None,
                                    "sha256": D.sha256_file(rules) if rules else None},
                   "script_sha256": D.sha256_file(Path(__file__)), "inputs": files, "integrity": checked,
                   "gate_record": {"file": D.GATE_RECORD.name, "sha256": D.sha256_file(D.GATE_RECORD)},
                   "git": {"head": common.git_head(), "dirty": common.git_dirty()}},
        "checks": results,
        "table": [{"check": r["check"], "signs_fired": [s["sign"] for s in r["signs"] if s["fired"]],
                   "signs_missing": [s["sign"] for s in r["signs"] if s["missing"]], "outcome": r["outcome"],
                   "consequence": r["consequence"], "type": r.get("type"), "who": r.get("who")}
                  for r in results.values()],
        "package": package(results, int(th["fix_cycles_max"])),
    }
    try:
        report["gate_reevaluation"] = reevaluate(results)
        report["gate_reevaluation"]["some_package_fits_the_cycle_budget"] = report["package"].get("any_fits")
    except Exception as e:   # reported, never a pass
        report["gate_reevaluation"] = {"error": f"{type(e).__name__}: {e}"}
    if tiny:
        report["gate_reevaluation_note"] = "tiny smoke outputs against the real gate record: mechanics only"
    report = common.jsonable(report)
    D.leak_scan(report, max_len=1500)
    path = D.write_new(outdir / f"stage5_report_{report['header']['utc']}.json", report)
    print(json.dumps({"stage": "stage5_report", "file": common.redact_path(path),
                      "outcomes": {k: r["outcome"] for k, r in results.items()},
                      "packages": [[p["amendments"], p["cycles"], p["fits"]] for p in report["package"].get("packages", [])],
                      "K8_reevaluated": (report["gate_reevaluation"].get("verdict_with_permitted_changes") or {})
                      .get("K8", {}).get("verdict")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
