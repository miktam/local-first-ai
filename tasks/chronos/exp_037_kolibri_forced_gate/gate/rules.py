# SPDX-License-Identifier: MIT
"""exp_037 gate rules: every pass/fail decision of the gate (DESIGN §3, §4.2, §5, §9.2; W6).

Frozen with GATE_RULES_SHA256: tools/hash_tree.py's scope "gate_rules" hashes this
file with thresholds.json, controls.json, calibration.json and port_mutants.py.
The modules under gate/checks/ only measure; every decision is a pure function
here of their measured dicts and gate/thresholds.json. Nothing here imports mlx,
the port or gate.common, draws a random number, or writes a file.

Moved from exp_036 (registered 2026-10-03, logic unchanged):
  G0 and G1          run_gate.py l.290-336, l.457-462
  G2 rows            g2_layers.py l.246-322 (plus the per-pair leg (p), §3.5)
  G3b                g3_oracle.py l.95-100 (the mutant classification), l.106-131
  G4-N(i)            g4_e2e.py l.122-132 (the KL leg; decisive top-1 now descriptive)
  G5                 g5_generation.py l.158-160 (parity_bound), l.312-320
                     (behaviour_verdict); run_gate.py l.478-496, l.552-553
New in exp_037: G4-F32, G4-F16, G5-D32, G5-R1, G5-BP-lean and allowed_B, the
controls map (controls.json), BLOCKING, arm_verdict, exit_code_for (the §3.16
table), cycle counting (§9.4), the P3 (G0k) and P5 rules, and the tripwire
constants that analysis/margins.json must equal (§7.2; W9's tests).

Check states
  PASS, FAIL, INCOMPLETE ("power not shown: <mutant>" or "VOID: <calibrator>
  above ceiling"), MISSING (no value: a crash, exit 3) and DESCRIPTIVE (reported,
  never blocking). Every check dict carries "state" and, for continuity with
  exp_036's record, "pass" (True for PASS, False for FAIL, else None).
  Precedence inside a check and for an arm: FAIL > MISSING > INCOMPLETE > PASS.

Comparisons are written so that a NaN never passes: "x within bound" is
`x <= bound`, and a leg that fires above a bound fires on `not (x <= bound)`.
Qualifying definitions stated with ">=" (an R2F-decisive position, lead >= 2
nats; F6's gap >= 1e-2) qualify at equality, as written in DESIGN §3.

Signatures. Rules moved from exp_036 keep their block-level threshold argument
(th0 = th["G0"], th2 = th["G2"], ...). Every new rule takes the whole
thresholds dict `th`, because it reads several blocks.

Measured-dict contracts (what each rule reads; W5a/W5b/W5c measure, W7 calls)
------------------------------------------------------------------------------
A "tail" is {"n": number of candidates, "top": the largest values, descending,
at most TAIL_K of them} (see `tail()`): a threshold-free summary from which a
rule reads the maximum and counts at least up to TAIL_K.

G2, g2_checks(res, emu, th2) and g2_row_verdicts(res, emu, th2):
  res = g2_layers.run(...): "layers_fp32", "layers_bf16" and "layers_t9" (rows
  {"layer", "branch", "median", "p99", "max", ["bucket"]}), "natural_fp32" (rows
  {"n", "n_violations", "n_exempt", "exempt_gaps"}), "natural_bf16" (rows {"n",
  "n_disagree", "sigma", "max_gap_over_sigma" (= max z_port of the layer, 0.0
  without a disagreement), "n_beyond_6sigma"}), "embedding" {"exact", "max_abs"},
  "head" {"max_abs_dlogit"}, optional "router_margin" {"diff_p99", ...},
  "layers_g2q" {arm: {"layers": rows, "embedding_max_rel", "head_rel_l2_max"} or
  {"capability_missing"}}, "real_weight_mutants" {"layers": [...], "mutants":
  {name: {"embedding_exact": bool, "layers": {layer: {"attn": {"median", "max"},
  "moe": {"median", "max"}, "selection_violations": int}}}}}, optional
  "flagged_pair".
  emu = emu_stats_exp037.json: {"layers": [{"attn": {"median", "p99"}, "moe":
  {...}, "natural_disagree_frac"}], "M_emu": max z_emu over layers and pairs,
  optional "n_beyond_6sigma", "flagged_pair"}.
G3, g3_checks(bpb, peers, mutants, th3): bpb = g3_oracle.ref_bpb(...); peers =
  g3_oracle.peer_bpb(...); mutants = {name: {"dnll_p01", "dnll_p99", ...}}.
G4-N(i), g4_n(res, th): res = g4_e2e.e2e(...) {"T1-8": {"mean_kl",
  "top1_decisive", "n_decisive"}, "T9": {...}}.
G4-F32, g4_f32(base, mutants, f_tiny, th): base and each mutants[m] =
  {"sets": {"T1-8": S, "T9": S}, "t9_buckets": {label: B}} with
  S = {"n", "mean_kl", "csort_mean", "csort_max", "top1_change_leads": tail of
  R2F's lead at positions where the top-1 differs, "kl": tail of per-position
  KL(R2F||port), "shadow_gaps": tail of R2F's 6th-7th biased gap at (token,
  layer) pairs whose shadow sets differ, "shadow_pairs": pairs compared} and
  B = {"lo", "hi", "n", "mean_kl", "mean_S", "csort_mean"}; csort_* are read
  from base only (mutants are judged with the unmutated run's tau and Csort).
  f_tiny from gate/calibration.json.
G4-F16, g4_f16(base, mutants, th): {"T1-8": {"mean_kl_port", "mean_kl_r3"},
  "T9": {...}} (both KL(R2F||.)); mutants[6] as base (mean_kl_r3 is base's).
G5 greedy, g5_greedy(res, th5): g5_generation.greedy_vs_ref(...).
Decode vs prefill, g5_decode_vs_prefill(fd, th5): floor_and_decode(...).
G5-D32, g5_d32(base, mutants, th): {range: {"csort": {"mean", "max"},
  "decode": R, "chunk64": R}} for the ranges th["G5D32"]["ranges"], with
  R = {"n", "mean_kl", "max_kl", "top1_change_leads": tail of the forced
  baseline's lead at positions where the top-1 differs}; mutants[20] has
  {range: {"decode": R}}.
G5-R1, g5_r1(arms, floor, mutants, th): arms = {"K8": {"anchor": A, "greedy":
  {"decisive_top1", "in_top5", "n_decisive", "n"}, "parity": {"mean_kl",
  "top1_dis", "n"}}, "K4": {"anchor": A}} with A = {"n", "mean_kl_r1_runner",
  "mean_kl_r1_single"}; floor = phase 4's noise floor {"floor_kl", "floor_dis"};
  mutants[26] = {"parity": {"mean_kl", "top1_dis"}} (K8).
G5-BP-lean, g5_bp(results, mutants, th): results = {arm: {B: P}} with
  P = {"B", "first_wave": D, "mid_run": D, "admitted_mid_run", "max_live"},
  D = {"n", "mean_kl_r1_batched", "mean_kl_r1_single"}; mutants[m] = P at K8,
  B = 8, for the mutant m of each required control of check g5_bp_K8_B8 in
  controls.json (controls.json's "probes" are never read here: run_gate.py
  judges them through control_result). B keys may be int or str (JSON).
Behaviour, g5_behaviour(cells, th5): {effort: g5_generation.behaviour(...)}.
P3 (G0k), p3_g0k(entries, th): [{"projection", "bits", "rows",
  "rel_f64_sorted", "rel_f64_unsorted", "bad_rows", "repeat_bitwise"}].
P5, p5(m, th): {"p5a": {"bitwise": {layer: bool}}, "p5b": {"bitwise": bool},
  "p5c": {"mean_kl": float}}.
"""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path

RULES_VERSION = "exp037-rules-1"
GATE_DIR = Path(__file__).resolve().parent
THRESHOLDS_PATH = GATE_DIR / "thresholds.json"
CONTROLS_PATH = GATE_DIR / "controls.json"

PASS, FAIL, INCOMPLETE, MISSING, DESCRIPTIVE = "PASS", "FAIL", "INCOMPLETE", "MISSING", "DESCRIPTIVE"
STATES = (PASS, FAIL, INCOMPLETE, MISSING, DESCRIPTIVE)
ARMS = ("K8", "K4")
TAIL_K = 32

CSORT_CEILING_REASON = "re-association control above its ceiling (cause not attributed)"


class RuleInputError(ValueError):
    """A measured dict does not carry what a rule reads (a gate bug: exit 3)."""


# ---------------------------------------------------------------------------
# Check ids and BLOCKING (§3.16; replaces exp_036 run_gate.py l.69-85)
# ---------------------------------------------------------------------------

SHARED_G0 = ["g0_census", "g0_strict_load_source", "g0_template_parity", "g0_tokenizer_parity"]


def _arm_g0(arm: str) -> list:
    return [f"g0_strict_load_{arm}", f"g0_converted_config_{arm}", f"g0_router_bf16_exact_{arm}",
            f"g0_head_bf16_exact_{arm}"]


BLOCKING = {
    "K8": SHARED_G0 + _arm_g0("K8") + [
        "g1",
        "g2_fp32_forced_branch", "g2_fp32_natural_selection", "g2_embedding", "g2_head",
        "g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_router_margin", "g2_t9_buckets", "g2q_K8",
        "g2_real_weight_mutants",
        "g3_ref_mutants",
        "g4_f32_K8", "g4_f16_K8", "g4_n_K8",
        "g5_greedy", "g5_decode_vs_prefill", "g5_d32_K8", "g5_r1_K8", "g5_behaviour_K8",
    ],
    "K4": SHARED_G0 + _arm_g0("K4") + ["same_port_sha", "g2q_K4", "g5_behaviour_K4", "g5_r1_K4"],
}

# thresholds.json "K4_blocking" names groups; each group is these check ids.
K4_BLOCKING_GROUPS = {
    "G0": SHARED_G0 + _arm_g0("K4"),
    "same_port_sha": ["same_port_sha"],
    "G2q": ["g2q_K4"],
    "G5_behaviour": ["g5_behaviour_K4"],
    "G5_R1_anchor": ["g5_r1_K4"],
}

# Reported, never blocking (§3.16 "Not blocking"; G5-BP-lean sets allowed_B).
NOT_BLOCKING = ["g3_bpb_per_text", "g3_bpb_vs_peers", "g4_K4"] + [
    f"g5_bp_{arm}_B{B}" for arm in ARMS for B in (8, 16)]


def group_of(cid: str) -> str:
    """g0 ... g5 for a check id (same_port_sha is G0, G2q is G2)."""
    return "g0" if cid == "same_port_sha" else "g2" if cid.startswith("g2q_") else cid.split("_", 1)[0]


def arms_of(cid: str) -> list:
    return [a for a in ARMS if cid in BLOCKING[a]]


# ---------------------------------------------------------------------------
# Files, check dicts, small helpers
# ---------------------------------------------------------------------------


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_controls(path: Path = CONTROLS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(state: str, measured=None, threshold=None, *, reason=None, legs=None, **extra) -> dict:
    """A check entry of the gate record."""
    if state not in STATES:
        raise ValueError(f"unknown check state {state!r}")
    out = {"state": state, "pass": True if state == PASS else False if state == FAIL else None,
           "reason": reason, "measured": measured, "threshold": threshold}
    if legs is not None:
        out["legs"] = legs
    if state == DESCRIPTIVE:
        out["descriptive"] = True
    out.update(extra)
    return out


def missing(reason: str, **extra) -> dict:
    return check(MISSING, None, None, reason=reason, **extra)


def state_of(c: dict | None) -> str:
    """The state of a check entry; an exp_036-style entry ({"pass": ...}) is read
    as PASS / FAIL, or MISSING when its pass is None."""
    if c is None:
        return MISSING
    if "state" in c:
        return c["state"]
    if c.get("descriptive"):
        return DESCRIPTIVE
    return PASS if c.get("pass") is True else FAIL if c.get("pass") is False else MISSING


def _num(d: dict, key: str, where: str) -> float:
    """d[key] as a number (int or float, never bool or None); NaN is allowed and
    fails every rule that reads it."""
    if not isinstance(d, dict) or key not in d:
        raise RuleInputError(f"{where}: no {key!r}")
    v = d[key]
    if not _is_number(v):
        raise RuleInputError(f"{where}: {key!r} is {v!r}, not a number")
    return float(v)


def _is_number(v) -> bool:
    """A real number (numpy scalars included), never a bool."""
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and type(v).__name__ not in ("bool_", "bool")


def _key(d: dict, k):
    """d[k] for an int key that a JSON round trip may have turned into a str."""
    if d is None:
        return None
    if k in d:
        return d[k]
    return d.get(str(k))


def within(x: float, bound: float) -> bool:
    """x <= bound; False for NaN."""
    return bool(x <= bound)


def at_least(x: float, thr: float) -> bool:
    """x >= thr, and True for NaN (a qualifying definition never lets a NaN through)."""
    return not (x < thr)


def at_least_strict(x: float, thr: float) -> bool:
    """x >= thr for a rule that passes at or above thr; False for NaN."""
    return bool(x >= thr)


def _ratio(num: float, den: float):
    """num / den, or None when den <= 0 (then the ratio is unbounded iff num > 0)."""
    return num / den if den > 0 else None


def tail(values, k: int = TAIL_K) -> dict:
    """{"n", "top"}: the count of candidates and the k largest values, descending.
    A NaN is placed first (treated as the largest), so it reaches every rule."""
    import numpy as np

    a = np.asarray(values, dtype=np.float64).reshape(-1)
    nan = np.isnan(a)
    rest = np.sort(a[~nan])[::-1][:k]
    top = [math.nan] * min(int(nan.sum()), k) + [float(x) for x in rest]
    return {"n": int(a.size), "top": top[:k]}


def _max(values, default: float = 0.0) -> float:
    """max() that returns NaN when any value is NaN (Python's max does not)."""
    vals = [float(v) for v in values]
    if any(math.isnan(v) for v in vals):
        return math.nan
    return max(vals, default=default)


def _tail(t: dict, where: str) -> dict:
    if not isinstance(t, dict) or "n" not in t or "top" not in t:
        raise RuleInputError(f"{where}: not a tail {{'n', 'top'}}")
    top = list(t["top"])
    if any(not at_least(a, b) for a, b in zip(top, top[1:])):
        raise RuleInputError(f"{where}: tail not in descending order")
    if len(top) > t["n"]:
        raise RuleInputError(f"{where}: tail longer than its count")
    return t


def tail_max(t: dict) -> float:
    return float(t["top"][0]) if t["top"] else -math.inf


def tail_count_at_least(t: dict, thr: float) -> int:
    """Entries >= thr (NaN counts). A lower bound once every listed entry qualifies
    and the tail was truncated."""
    return sum(1 for v in t["top"] if at_least(v, thr))


def tail_count_above(t: dict, thr: float) -> int:
    """Entries > thr (NaN counts)."""
    return sum(1 for v in t["top"] if not within(v, thr))


# ---------------------------------------------------------------------------
# G0 and G1 (moved from exp_036 run_gate.py l.290-336, l.457-462)
# ---------------------------------------------------------------------------


def g0_census(cen: dict, expected: dict, tiny: bool = False) -> dict:
    sha = cen.get("shard_sha") or {}
    sha_ok = sha.get("status") == "checked" and not sha.get("mismatches")
    ok = (cen["tensors"] == expected["tensor_count"] and cen["params"] == expected["param_count"]
          and cen["full_attention_layers"] == expected["full_attention_layers"] and (sha_ok or tiny))
    return check(PASS if ok else FAIL, cen, expected)


def g0_strict_load(sl: dict) -> dict:
    ok = (sl["n_unused_ckpt_tensors"] == 0 and sl["n_missing_params"] == 0 and not sl["shape_mismatch"]
          and sl["load_error"] is None and (sl["numel_equal"] or sl["quantized"]))
    return check(PASS if ok else FAIL, sl, "every tensor consumed once, no parameter absent")


def g0_template_parity(tp: dict, th0: dict) -> dict:
    n_want = th0["template_conversations"] * th0["template_settings"]
    ok = tp["n"] == n_want and not tp["mismatches"] and not tp["runner_chat"]["mismatches"]
    return check(PASS if ok else FAIL, tp, {"n": n_want, "mismatches": 0})


def g0_tokenizer_parity(tk: dict, th0: dict) -> dict:
    ok = (tk["n_lines"] >= th0["tokenizer_lines_min"] and tk["mismatches"] <= th0["tokenizer_mismatches_max"]
          and tk["roundtrip_fail"] == 0 and tk["fix_mistral_regex"] is not True)
    return check(PASS if ok else FAIL, tk, {"lines_min": th0["tokenizer_lines_min"],
                                            "mismatches_max": th0["tokenizer_mismatches_max"]})


def g0_converted_config(cc: dict, policy: str) -> dict:
    return check(PASS if cc["n_problems"] == 0 else FAIL, cc, {"policy": policy})


def same_port_sha(port: str, shas: dict) -> dict:
    ok = all(v == port for v in shas.values())
    return check(PASS if ok else FAIL, {"port": port, **shas}, "K8 and K4 converted with the current port/kolibri1.py")


def g0_dtypes(dt: dict, th0: dict) -> tuple[dict, dict]:
    """(router check, head check) from g0_static.dtype_asserts(...)."""
    router = (dt["router_logits_dtype"] == th0["router_logits_dtype"]
              and within(dt["router_bf16_exact_frac"], th0["router_bf16_exact_frac_max"]))
    head = (dt["head_dtype"] == th0["head_output_dtype"]
            and within(dt["head_bf16_exact_frac"], th0["head_bf16_exact_frac_max"]))
    return (check(PASS if router else FAIL, dt, {k: th0[k] for k in ("router_logits_dtype", "router_bf16_exact_frac_max")}),
            check(PASS if head else FAIL, dt, {k: th0[k] for k in ("head_output_dtype", "head_bf16_exact_frac_max")}))


def g1(res: dict, th1: dict, tiny: bool = False) -> dict:
    """0 failures, 0 errors, and no skip other than "build-host only:" (tiny runs
    may skip)."""
    skipped = res["skipped"] - res.get("skipped_build_host_only", 0)
    ok = res["tests"] > 0 and res["failures"] == 0 and res["errors"] == 0 and (tiny or skipped == 0)
    return check(PASS if ok else FAIL, res, {"failures": 0, "errors": 0, "skipped": 0,
                                             "skipped_except_build_host_only": 0, "source": th1["source"]})


# ---------------------------------------------------------------------------
# G2 (moved from exp_036 g2_layers.py l.246-322; leg (p) new, §3.5)
# ---------------------------------------------------------------------------


def _fp32_row(r: dict, th2: dict) -> dict:
    ok = within(r["median"], th2["fp32_forced_rel_err_median_max"]) and within(r["max"], th2["fp32_forced_rel_err_max_max"])
    return {"pass": ok, "limit_median": th2["fp32_forced_rel_err_median_max"], "limit_p99": None,
            "limit_max": th2["fp32_forced_rel_err_max_max"]}


def _bf16_row(r: dict, emu_layers: list, th2: dict) -> dict:
    e = emu_layers[r["layer"]][r["branch"]]
    lim_med = th2["bf16_forced_median_factor_vs_emu"] * e["median"]
    lim_p99 = max(th2["bf16_forced_p99_floor"], th2["bf16_forced_p99_factor_vs_emu"] * e["p99"])
    ok = within(r["median"], lim_med) and within(r["p99"], lim_p99)
    return {"pass": ok, "limit_median": lim_med, "limit_p99": lim_p99, "limit_max": None}


def bf16_rule(rows: list, emu_layers: list, th2: dict) -> tuple[bool, list, list]:
    """(ok, failing rows, per-row verdicts): per layer and branch, median e <= 3 x
    emu's median and p99 e <= max(5e-2, 3 x emu's p99)."""
    verdicts, bad = [], []
    for r in rows:
        v = _bf16_row(r, emu_layers, th2)
        verdicts.append(v)
        if not v["pass"]:
            bad.append({k: r[k] for k in ("layer", "branch", "median", "p99") if k in r}
                       | ({"bucket": r["bucket"]} if "bucket" in r else {}))
    return not bad, bad, verdicts


def g2_row_verdicts(res: dict, emu: dict | None, th2: dict) -> dict:
    """Per-row verdicts for gate_<UTC>_layers.csv, keyed by the CSV's mode and
    aligned with res's row lists: {"pass", "limit_median", "limit_p99",
    "limit_max"}; "pass" is None for a bf16 row without emulation statistics."""
    out = {"fp32_forced": [_fp32_row(r, th2) for r in res.get("layers_fp32", [])]}
    none = {"pass": None, "limit_median": None, "limit_p99": None, "limit_max": None}
    el = emu["layers"] if emu is not None else None
    for mode, key in (("bf16_forced", "layers_bf16"), ("t9_bf16_forced", "layers_t9")):
        out[mode] = [_bf16_row(r, el, th2) if el is not None else dict(none) for r in res.get(key, [])]
    for arm, g in (res.get("layers_g2q") or {}).items():
        out[f"g2q_{arm}"] = [_bf16_row(r, el, th2) if el is not None else dict(none) for r in g.get("layers", [])]
    return out


def g2_mutant_detected(m: dict, th2: dict) -> bool:
    """A real-weight mutant is detected iff, on some layer, a branch's forced
    error breaks the fp32 row (median > 1e-4 or max > 1e-3) or its natural
    selection has a violation, or its embedding is not exact (H36 l.237;
    exp_036 g2_layers.py l.208-214)."""
    for i, lay in m["layers"].items():
        if "attn" not in lay or "moe" not in lay:
            raise RuleInputError(f"real-weight mutant layer {i}: needs the branch stats 'attn' and 'moe' "
                                 "(the decision is rules.py's, not a measured 'forced_fail')")
        forced_fail = any(not _fp32_row(lay[br], th2)["pass"] for br in ("attn", "moe"))
        if forced_fail or lay["selection_violations"] > 0:
            return True
    return not m["embedding_exact"]


def g2_checks(res: dict, emu: dict | None, th2: dict) -> dict:
    """{check id: check} for the G2 checks that res and emu allow; without
    emulation statistics the bf16 rows and G2q are absent (the caller records
    them as MISSING with the reason)."""
    checks = {}
    fp = res["layers_fp32"]
    bad = [{k: r[k] for k in ("layer", "branch", "median", "max")} for r in fp if not _fp32_row(r, th2)["pass"]]
    checks["g2_fp32_forced_branch"] = check(
        FAIL if bad else PASS,
        {"worst_median": max(r["median"] for r in fp), "worst_max": max(r["max"] for r in fp), "failing": bad[:20]},
        {"median_max": th2["fp32_forced_rel_err_median_max"], "max_max": th2["fp32_forced_rel_err_max_max"]})
    nat = res["natural_fp32"]
    viol = sum(r["n_violations"] for r in nat)
    checks["g2_fp32_natural_selection"] = check(
        PASS if viol == 0 else FAIL,
        {"n_pairs": sum(r["n"] for r in nat), "violations": viol, "exempt": sum(r["n_exempt"] for r in nat),
         "exempt_gaps": [g for r in nat for g in r["exempt_gaps"]][:20]},
        {"near_tie_gap": th2["fp32_selection_near_tie_gap"]})
    checks["g2_embedding"] = check(PASS if res["embedding"]["exact"] is True else FAIL, res["embedding"], "exact")
    checks["g2_head"] = check(PASS if within(res["head"]["max_abs_dlogit"], th2["head_fp32_logit_maxabs"]) else FAIL,
                              res["head"], {"max_abs_dlogit": th2["head_fp32_logit_maxabs"]})
    if emu is not None and res["layers_bf16"]:
        el = emu["layers"]
        ok, bad, _ = bf16_rule(res["layers_bf16"], el, th2)
        checks["g2_bf16_forced_branch"] = check(
            PASS if ok else FAIL, {"failing": bad[:20]},
            {k: th2[k] for k in ("bf16_forced_median_factor_vs_emu", "bf16_forced_p99_floor", "bf16_forced_p99_factor_vs_emu")})
        checks["g2_bf16_natural_selection"] = g2_bf16_natural_selection(res, emu, th2)
        ok9, bad9, _ = bf16_rule(res["layers_t9"], el, th2)
        checks["g2_t9_buckets"] = check(PASS if ok9 else FAIL, {"failing": bad9[:20], "buckets": th2["t9_buckets"]},
                                        "the bf16-mode forced row, against emu's T1-T8 statistics")
        if "router_margin" in res:
            rm = res["router_margin"]
            # blocks only if the 99th percentile < 0: passes at or above it (never on NaN)
            ok = at_least_strict(rm["diff_p99"], th2["router_mutant_block_if_diff_p99_below"])
            checks["g2_router_margin"] = check(PASS if ok else FAIL, rm,
                                               {"block_if_diff_p99_below": th2["router_mutant_block_if_diff_p99_below"]})
        for arm, g in res["layers_g2q"].items():
            if "capability_missing" in g:
                continue
            okq, badq, _ = bf16_rule(g["layers"], el, th2)
            okq = (okq and within(g["embedding_max_rel"], th2["g2q_embedding_rel_err_max"])
                   and within(g["head_rel_l2_max"], th2["g2q_head_rel_l2_max"]))
            checks[f"g2q_{arm}"] = check(
                PASS if okq else FAIL,
                {"failing": badq[:20], "embedding_max_rel": g["embedding_max_rel"], "head_rel_l2_max": g["head_rel_l2_max"]},
                {"layers": "as g2_bf16_forced_branch", "embedding_rel_err_max": th2["g2q_embedding_rel_err_max"],
                 "head_rel_l2_max": th2["g2q_head_rel_l2_max"]})
    rwm = res["real_weight_mutants"]
    if rwm["mutants"]:
        detected = {n: g2_mutant_detected(m, th2) for n, m in rwm["mutants"].items()}
        undetected = sorted(n for n, d in detected.items() if not d)
        checks["g2_real_weight_mutants"] = check(
            PASS if not undetected else FAIL,
            {"layers": rwm["layers"], "undetected": undetected, "detected": sorted(n for n, d in detected.items() if d)},
            "each of mutants 1-8, 12-15 fails some fp32 check on >= 1 layer")
    return checks


def g2_bf16_natural_selection(res: dict, emu: dict, th2: dict) -> dict:
    """(a) the disagreement fraction <= max(0.2 %, 2 x emu's fraction), as
    registered; (p) M_port <= min(8, 1.2 x M_emu), A1's per-pair form (fix 7).
    The 6-sigma counts are descriptive (the registered 6-sigma rule is retired)."""
    rows = res["natural_bf16"]
    el = emu["layers"]
    n = sum(r["n"] for r in rows)
    frac = sum(r["n_disagree"] for r in rows) / max(n, 1)
    emu_frac = sum(l["natural_disagree_frac"] for l in el) / len(el)
    lim = max(th2["bf16_selection_disagree_floor"], th2["bf16_selection_disagree_factor_vs_emu"] * emu_frac)
    m_port = _max((r["max_gap_over_sigma"] for r in rows), default=0.0)
    m_emu = _num(emu, "M_emu", "emulation statistics")
    z_bound = min(th2["pairwise_z_cap"], th2["pairwise_z_factor_vs_emu"] * m_emu)
    leg_a = within(frac, lim)
    leg_p = within(m_port, z_bound)
    legs = {"a": {"ok": leg_a, "disagree_frac": frac, "emu_disagree_frac": emu_frac, "limit": lim},
            "p": {"ok": leg_p, "M_port": m_port, "M_emu": m_emu, "limit": z_bound}}
    failing = [k for k, v in legs.items() if not v["ok"]]
    measured = {"disagree_frac": frac, "emu_disagree_frac": emu_frac, "limit": lim, "M_port": m_port, "M_emu": m_emu,
                "z_limit": z_bound,
                "descriptive": {"n_beyond_6sigma_port": sum(r.get("n_beyond_6sigma", 0) for r in rows),
                                "n_beyond_6sigma_emu": emu.get("n_beyond_6sigma"),
                                "sigma_multiple": th2["bf16_selection_gap_sigma"],
                                "flagged_pair": res.get("flagged_pair", emu.get("flagged_pair"))}}
    return check(FAIL if failing else PASS, measured,
                 {k: th2[k] for k in ("bf16_selection_disagree_floor", "bf16_selection_disagree_factor_vs_emu",
                                      "pairwise_z_cap", "pairwise_z_factor_vs_emu")},
                 reason=("leg " + ", ".join(failing)) if failing else None, legs=legs)


# ---------------------------------------------------------------------------
# G3 (moved from exp_036 g3_oracle.py l.95-100, l.106-131; G3a descriptive, §3.6)
# ---------------------------------------------------------------------------

G3_REFERENCE_BETTER = "reference_better"
G3_MUTANT_BETTER = "mutant_better"
G3_NOT_RESOLVABLE = "not_resolvable (rests on vendor source)"


def g3_mutant_class(p01: float, p99: float) -> str:
    """The bootstrap of NLL_mutant - NLL_ref: reference better iff p01 > 0,
    mutant better iff p99 < 0, else not resolvable (it does not fail)."""
    if not (math.isfinite(p01) and math.isfinite(p99)):
        raise RuleInputError(f"G3b bootstrap quantiles not finite: p01 {p01!r}, p99 {p99!r}")
    if p01 > 0:
        return G3_REFERENCE_BETTER
    if p99 < 0:
        return G3_MUTANT_BETTER
    return G3_NOT_RESOLVABLE


def g3_checks(bpb: dict | None, peers: dict | None, mutants: dict | None, th3: dict) -> dict:
    """G3a (bits per byte) descriptive, with exp_036's limits printed beside it as
    "not an exp_037 criterion"; G3b blocking as registered: it fails only if some
    mutant is resolvably better than the reference or the mutant count is not 7."""
    checks = {}
    note = "exp_036's limit, printed beside the value; not an exp_037 criterion"
    if bpb is not None:
        worst = max(bpb["per_text_bpb"].values())
        checks["g3_bpb_per_text"] = check(
            DESCRIPTIVE, bpb, {"exp036_per_text_max": th3["ref_bpb_per_text_max"], "note": note},
            within_exp036_limit=within(worst, th3["ref_bpb_per_text_max"]))
        pv = (peers or {}).get("peers", {})
        best = min(pv.values()) if pv else None
        checks["g3_bpb_vs_peers"] = check(
            DESCRIPTIVE, {"mean_bpb": bpb["mean_bpb"], "peers": pv, "best_peer": best,
                          "ratio_to_best_peer": (bpb["mean_bpb"] / best) if best else None,
                          "source": (peers or {}).get("source")},
            {"exp036_mean_vs_best_peer_max": th3["ref_bpb_mean_vs_best_peer_max"], "note": note},
            within_exp036_limit=(within(bpb["mean_bpb"], th3["ref_bpb_mean_vs_best_peer_max"] * best) if best else None))
    if mutants is not None:
        classified = {}
        for name, m in mutants.items():
            cls = g3_mutant_class(_num(m, "dnll_p01", f"G3b {name}"), _num(m, "dnll_p99", f"G3b {name}"))
            classified[name] = {**m, "verdict": cls, "resolvable": cls != G3_NOT_RESOLVABLE}
        better = sorted(n for n, m in classified.items() if m["verdict"] == G3_MUTANT_BETTER)
        count_ok = len(classified) == th3["ref_mutants"]
        reason = None
        if better:
            reason = "mutant better than the reference: " + ", ".join(better)
        elif not count_ok:
            reason = f"{len(classified)} reference mutants, not {th3['ref_mutants']}"
        checks["g3_ref_mutants"] = check(
            FAIL if (better or not count_ok) else PASS,
            {"mutants": classified, "mutant_better": better,
             "rests_on_vendor_source": sorted(n for n, m in classified.items() if not m["resolvable"])},
            {"text": th3["ref_mutant_text"], "blocks": th3["ref_mutant_blocks"],
             "quantile": th3["ref_mutant_dnll_quantile"], "n_mutants": th3["ref_mutants"]},
            reason=reason)
    return checks


# ---------------------------------------------------------------------------
# G4-N(i) (moved from exp_036 g4_e2e.py l.122-132: the KL leg) and G4 K4
# ---------------------------------------------------------------------------


def g4_n(res: dict, th: dict) -> dict:
    """Free routing, production draw: mean KL(R1||K8) <= 0.10 on T1-8 and on T9,
    each pooled. Decisive top-1 is descriptive, printed beside the registered
    99 % and exp_036's emulation value."""
    lim = th["G4"]["K8_backstop_mean_kl_max"]
    legs = {}
    for s in th["G4N"]["sets"]:
        kl = _num(res.get(s), "mean_kl", f"G4-N(i) {s}")
        legs[s] = {"ok": within(kl, lim), "mean_kl": kl}
    emu = th["G4N"]["descriptive_emu_decisive_exp036"]
    desc = {"decisive_top1": {s: (res[s] or {}).get("top1_decisive") for s in th["G4N"]["sets"]},
            "registered_decisive_top1_min": th["G4"]["K8_backstop_decisive_top1_min"],
            "exp036_emulation_decisive_top1": {"set": emu["set"], "value": 1 - emu["misses"] / emu["n"]}}
    failing = [s for s, v in legs.items() if not v["ok"]]
    return check(FAIL if failing else PASS, res, {"mean_kl_max": lim, "sets": th["G4N"]["sets"]},
                 reason=("mean KL above the bound in " + ", ".join(failing)) if failing else None,
                 legs=legs, descriptive_rows=desc)


def g4_k4(res: dict, th: dict) -> dict:
    """K4 end to end against R1: descriptive, as registered (H36 l.241)."""
    return check(DESCRIPTIVE, res, th["G4"]["K4"])


# ---------------------------------------------------------------------------
# G4-F32 (§3.7), with F_tiny (§5.1) and the Csort ceiling (§5.2)
# ---------------------------------------------------------------------------


def _set(m: dict, s: str, where: str) -> dict:
    """m["sets"][s], or RuleInputError."""
    v = (m.get("sets") or {}).get(s) if isinstance(m, dict) else None
    if not isinstance(v, dict):
        raise RuleInputError(f"{where}: no series for set {s}")
    return v


def _buckets(m: dict, n_want: int, where: str) -> list:
    """The T9 buckets of a series, ordered by position: [(label, bucket)]."""
    b = m.get("t9_buckets")
    if not isinstance(b, dict) or len(b) != n_want:
        raise RuleInputError(f"{where}: needs {n_want} T9 buckets, got {None if b is None else len(b)}")
    return sorted(b.items(), key=lambda kv: _num(kv[1], "lo", f"{where} bucket {kv[0]}"))


def g4_f32_calibration(base: dict, f_tiny: float, th: dict) -> dict:
    """tau per set = max(F_tiny, 100 x mean Csort(set)); the T9 buckets use T9's
    tau. Refuses an F_tiny above its cap (stop for Andrei before the freeze)."""
    t = th["G4F32"]
    if not _is_number(f_tiny) or not (f_tiny > 0) or not math.isfinite(f_tiny):
        raise RuleInputError(f"F_tiny {f_tiny!r} is not a positive finite number")
    if not within(f_tiny, t["f_tiny_cap"]):
        raise RuleInputError(f"F_tiny {f_tiny!r} above its cap {t['f_tiny_cap']}: stop for Andrei (DESIGN §5.1)")
    tau, csort = {}, {}
    for s in t["sets"]:
        cm = _num(_set(base, s, "G4-F32 base"), "csort_mean", f"G4-F32 {s}")
        tau[s] = max(f_tiny, t["tau_csort_factor"] * cm)
        csort[s] = {"mean": cm, "max": _set(base, s, "G4-F32 base").get("csort_max")}
    n_b = len(th["G2"]["t9_buckets"])
    bucket_csort = {lab: _num(b, "csort_mean", f"G4-F32 bucket {lab}") for lab, b in _buckets(base, n_b, "G4-F32 base")}
    return {"f_tiny": float(f_tiny), "tau": tau, "csort": csort, "bucket_csort": bucket_csort}


def g4_f32_legs(m: dict, cal: dict, th: dict) -> dict:
    """F1, F2, F3, F5 and F6 on one series (the unmutated run or a mutant),
    judged with the unmutated run's calibration `cal`."""
    t = th["G4F32"]
    sets = t["sets"]
    n_b = len(th["G2"]["t9_buckets"])
    legs = {}
    # F1: a top-1 change at an R2F-decisive position (lead >= 2 nats).
    per = {}
    for s in sets:
        tl = _tail(_set(m, s, "G4-F32").get("top1_change_leads"), f"G4-F32 {s} top1_change_leads")
        per[s] = {"count": tail_count_at_least(tl, t["f1_decisive_lead_nats"]), "max_lead": tail_max(tl) if tl["top"] else None,
                  "n_top1_changes": tl["n"]}
    legs["F1"] = {"fired": any(v["count"] > 0 for v in per.values()), "per": per, "count": sum(v["count"] for v in per.values())}
    # F2: the mean KL above tau in T1-8, T9 or any T9 bucket (buckets use T9's tau).
    per = {}
    for s in sets:
        kl = _num(_set(m, s, "G4-F32"), "mean_kl", f"G4-F32 {s}")
        per[s] = {"mean_kl": kl, "tau": cal["tau"][s], "fired": not within(kl, cal["tau"][s]),
                  "ratio": _ratio(kl, cal["tau"][s])}
    buckets = _buckets(m, n_b, "G4-F32")
    for lab, b in buckets:
        kl = _num(b, "mean_kl", f"G4-F32 bucket {lab}")
        per[f"T9:{lab}"] = {"mean_kl": kl, "tau": cal["tau"]["T9"], "fired": not within(kl, cal["tau"]["T9"]),
                            "ratio": _ratio(kl, cal["tau"]["T9"])}
    legs["F2"] = {"fired": any(v["fired"] for v in per.values()), "per": per}
    # F3: position growth on T9: rho(B2) > 10 x rho(B0) and mean KL(B2) > 10 x Csort(B2).
    (lab0, b0), (lab2, b2) = buckets[0], buckets[-1]
    rho0 = _ratio(_num(b0, "mean_kl", "F3 B0"), _num(b0, "mean_S", "F3 B0"))
    rho2 = _ratio(_num(b2, "mean_kl", "F3 B2"), _num(b2, "mean_S", "F3 B2"))
    if rho0 is None or rho2 is None:
        raise RuleInputError("G4-F32 F3: S has no positive mean in a bucket")
    cs2 = cal["bucket_csort"][lab2]
    growth = not within(rho2, t["f3_rho_factor"] * rho0)
    above = not within(b2["mean_kl"], t["f3_csort_factor"] * cs2)
    legs["F3"] = {"fired": growth and above, "rho_B0": rho0, "rho_B2": rho2, "mean_kl_B2": b2["mean_kl"],
                  "csort_mean_B2": cs2, "buckets": [lab0, lab2]}
    # F5: any position with KL > 1e-2.
    per = {}
    for s in sets:
        tl = _tail(_set(m, s, "G4-F32").get("kl"), f"G4-F32 {s} kl")
        per[s] = {"max_kl": tail_max(tl) if tl["top"] else None, "count": tail_count_above(tl, t["f5_position_kl_max"])}
    legs["F5"] = {"fired": any(v["count"] > 0 for v in per.values()), "per": per,
                  "max_kl": max((v["max_kl"] for v in per.values() if v["max_kl"] is not None), default=None)}
    # F6: a (token, layer) whose shadow sets differ where R2F's 6th-7th gap >= 1e-2.
    per = {}
    for s in sets:
        tl = _tail(_set(m, s, "G4-F32").get("shadow_gaps"), f"G4-F32 {s} shadow_gaps")
        pairs = _set(m, s, "G4-F32").get("shadow_pairs")
        per[s] = {"count": tail_count_at_least(tl, t["f6_shadow_gap_min"]), "n_disagree": tl["n"],
                  "max_gap": tail_max(tl) if tl["top"] else None,
                  "disagree_rate": (tl["n"] / pairs) if pairs else None}
    legs["F6"] = {"fired": any(v["count"] > 0 for v in per.values()), "per": per, "count": sum(v["count"] for v in per.values())}
    return legs


def g4_f32(base: dict | None, mutants: dict | None, f_tiny: float, th: dict, controls: dict | None = None) -> dict:
    """G4-F32: the K8 fp32 port forced onto I against R2F. FAIL if F1, F2, F3, F5
    or F6 fires, or mean Csort is above its ceiling in either set; INCOMPLETE if
    a required control (mutants 1, 6, 19) misses its bound leg; else PASS."""
    if base is None:
        return missing("G4-F32: no unmutated series")
    t = th["G4F32"]
    cal = g4_f32_calibration(base, f_tiny, th)
    legs = g4_f32_legs(base, cal, th)
    ceiling = {s: {"csort_mean": cal["csort"][s]["mean"], "ceiling": t["csort_mean_ceiling"],
                   "breach": not within(cal["csort"][s]["mean"], t["csort_mean_ceiling"])} for s in t["sets"]}
    ctrl = _controls_for("g4_f32_K8", mutants, {"cal": cal}, th, controls)
    fired = [k for k, v in legs.items() if v["fired"]]
    breach = [s for s, v in ceiling.items() if v["breach"]]
    measured = {"tau": cal["tau"], "f_tiny": cal["f_tiny"], "csort": cal["csort"], "csort_ceiling": ceiling,
                "descriptive": {s: {"shadow_disagree_rate": legs["F6"]["per"][s]["disagree_rate"]} for s in t["sets"]}}
    threshold = {k: t[k] for k in ("tau_csort_factor", "f_tiny_cap", "csort_mean_ceiling", "f1_decisive_lead_nats",
                                   "f3_rho_factor", "f3_csort_factor", "f5_position_kl_max", "f6_shadow_gap_min")}
    return _with_controls(fired, breach, legs, ctrl, measured, threshold, "G4-F32")


def _with_controls(fired: list, breach: list, legs: dict, ctrl: dict, measured: dict, threshold, name: str,
                   void: str | None = None) -> dict:
    """The state of a check with required controls, first match wins: FAIL (a leg
    fired or a ceiling breached); MISSING (a control's series is absent: a crash,
    exit 3); INCOMPLETE "VOID: ..." (a calibrator above its ceiling); INCOMPLETE
    "power not shown: ..." (a control not caught); PASS."""
    reasons = []
    if breach:
        reasons.append(f"{CSORT_CEILING_REASON} ({', '.join(breach)})")
    if fired:
        reasons.append("legs fired: " + ", ".join(fired))
    if reasons:
        return check(FAIL, measured, threshold, reason="; ".join(reasons), legs=legs, controls=ctrl)
    absent = [cid for cid, r in ctrl.items() if r["caught"] is None]
    if absent:
        return check(MISSING, measured, threshold, reason=f"{name}: no series for control(s) " + ", ".join(absent),
                     legs=legs, controls=ctrl)
    if void:
        return check(INCOMPLETE, measured, threshold, reason=void, legs=legs, controls=ctrl)
    uncaught = [r for r in ctrl.values() if r["caught"] is False]
    if uncaught:
        return check(INCOMPLETE, measured, threshold,
                     reason="; ".join(f"power not shown: mutant {r['mutant']} ({r['mutant_name']})" for r in uncaught),
                     legs=legs, controls=ctrl)
    return check(PASS, measured, threshold, legs=legs, controls=ctrl)


# ---------------------------------------------------------------------------
# G4-F16 (§3.8)
# ---------------------------------------------------------------------------


def g4_f16_legs(m: dict, r3: dict, th: dict) -> dict:
    """The kappa rule per set: mean KL(R2F||port) > kappa x mean KL(R2F||R3)."""
    t = th["G4F16"]
    per = {}
    for s in t["sets"]:
        port = _num(m.get(s), "mean_kl_port", f"G4-F16 {s}")
        bound = t["kappa"] * r3[s]
        per[s] = {"mean_kl_port": port, "mean_kl_r3": r3[s], "bound": bound, "fired": not within(port, bound),
                  "ratio": _ratio(port, r3[s])}
    return {"kappa": {"fired": any(v["fired"] for v in per.values()), "per": per}}


def g4_f16(base: dict | None, mutants: dict | None, th: dict, controls: dict | None = None) -> dict:
    """G4-F16: K8 bf16 forced onto I against R2F, calibrated by R3. VOID (so
    INCOMPLETE) if mean KL(R2F||R3) > 1e-2 in either set: the calibrator is above
    its ceiling and the kappa rule is not judged; otherwise FAIL if the kappa rule
    fires in a set; INCOMPLETE if mutant 6 misses it; else PASS."""
    if base is None:
        return missing("G4-F16: no series")
    t = th["G4F16"]
    r3 = {s: _num(base.get(s), "mean_kl_r3", f"G4-F16 {s}") for s in t["sets"]}
    void_sets = [s for s in t["sets"] if not within(r3[s], t["r3_mean_kl_ceiling"])]
    legs = g4_f16_legs(base, r3, th)
    ctrl = _controls_for("g4_f16_K8", mutants, {"r3": r3}, th, controls)
    measured = {"r3": r3, "r3_ceiling": t["r3_mean_kl_ceiling"], "void_sets": void_sets,
                "descriptive": {s: {k: v for k, v in (base[s] or {}).items() if k not in ("mean_kl_port", "mean_kl_r3")}
                                for s in t["sets"]}}
    threshold = {k: t[k] for k in ("kappa", "r3_mean_kl_ceiling")}
    if void_sets:
        return _with_controls([], [], legs, ctrl, measured, threshold, "G4-F16",
                              void="VOID: R3 above ceiling (" + ", ".join(void_sets) + ")")
    fired = ["kappa"] if legs["kappa"]["fired"] else []
    return _with_controls(fired, [], legs, ctrl, measured, threshold, "G4-F16")


# ---------------------------------------------------------------------------
# G5 greedy, decode vs prefill, behaviour (moved; exp_036 run_gate.py
# l.478-483, l.552-553; g5_generation.py l.158-160, l.312-320)
# ---------------------------------------------------------------------------


def greedy_ok(res: dict, th5: dict) -> bool:
    """Decisive top-1 >= 99.5 % (vacuous without a decisive position) and the
    reference top-5 at >= 99 % of all positions."""
    dec = res["decisive_top1"]
    return ((dec is None or at_least_strict(dec, th5["greedy_decisive_top1_min"]))
            and res["in_top5"] is not None and at_least_strict(res["in_top5"], th5["greedy_ref_top5_min"]))


def g5_greedy(res: dict, th5: dict) -> dict:
    return check(PASS if greedy_ok(res, th5) else FAIL, res,
                 {k: th5[k] for k in ("greedy_decisive_lead_nats", "greedy_decisive_top1_min", "greedy_ref_top5_min",
                                      "greedy_tokens")})


def parity_bound(floor: dict, th5: dict) -> dict:
    """The decode-vs-prefill bound (registered): KL <= max(1e-4, 3 x floor_KL),
    top-1 disagreement <= 3 x floor_dis + 0.2 pp."""
    return {"kl_max": max(th5["parity_kl_floor"], th5["parity_kl_factor"] * floor["floor_kl"]),
            "dis_max": th5["parity_dis_factor"] * floor["floor_dis"] + th5["parity_dis_add"]}


def parity_ok(m: dict, bound: dict) -> bool:
    return within(m["mean_kl"], bound["kl_max"]) and within(m["top1_dis"], bound["dis_max"])


def g5_decode_vs_prefill(fd: dict, th5: dict) -> dict:
    bound = parity_bound(fd, th5)
    ok = within(fd["decode_kl"], bound["kl_max"]) and within(fd["decode_dis"], bound["dis_max"])
    return check(PASS if ok else FAIL, fd, bound)


def behaviour_verdict(cells: dict, block_count: int) -> tuple[list, dict]:
    """(blocked efforts, summary). An (arm, effort) cell of 20 blocks when
    >= block_count completions loop, or end without EOS, or get lang_tag
    "unknown" (HYPOTHESIS G5 behaviour row)."""
    blocked = [e for e, c in cells.items()
               if c["n_loop"] >= block_count or c["n_no_eos"] >= block_count or c["n_unknown_lang"] >= block_count]
    summary = {e: {k: c[k] for k in ("n", "n_loop", "n_no_eos", "n_unknown_lang", "think_closed_rate",
                                     "lang_match_rate", "cap", "seed")} for e, c in cells.items()}
    return blocked, summary


def g5_behaviour(cells: dict, th5: dict, rows: dict | None = None) -> dict:
    k = th5["behaviour_block_count"]
    blocked, summary = behaviour_verdict(cells, k)
    measured = {"cells": summary, "blocked_efforts": blocked}
    if rows is not None:
        measured["rows"] = rows
    return check(FAIL if blocked else PASS, measured,
                 {"block_count": k, "loop": f"{th5['loop_span_tokens']} x {th5['loop_repeats']}", "eos_ids": th5["eos_ids"]},
                 reason=("blocked at effort " + ", ".join(blocked)) if blocked else None)


# ---------------------------------------------------------------------------
# G5-D32 (§3.12)
# ---------------------------------------------------------------------------


def g5_d32_bounds(base: dict, th: dict) -> dict:
    """Per range: mean bound max(1e-8, 100 x Csort mean), max bound max(1e-6,
    100 x Csort max), and the Csort ceilings (mean <= 1e-6, max <= 1e-4). The
    factors are thresholds.json's; 100 is G4-F32's tau factor (final review
    W15-04: decode and chunk 64 differ from chunk 2,048 by kernel rounding in
    every layer, not by the single re-association Csort measures)."""
    t = th["G5D32"]
    out = {}
    for r in t["ranges"]:
        cs = (base.get(r) or {}).get("csort")
        cm, cx = _num(cs, "mean", f"G5-D32 {r} csort"), _num(cs, "max", f"G5-D32 {r} csort")
        out[r] = {"csort_mean": cm, "csort_max": cx,
                  "mean_bound": max(t["mean_floor"], t["mean_factor_vs_csort"] * cm),
                  "max_bound": max(t["max_floor"], t["max_factor_vs_csort"] * cx),
                  "ceiling_breach": [k for k, v, c in (("mean", cm, t["csort_mean_ceiling"]), ("max", cx, t["csort_max_ceiling"]))
                                     if not within(v, c)]}
    return out


def g5_d32_comparison(comp: dict, bounds: dict, th: dict, where: str) -> dict:
    """The three rules of one comparison ((iii) or (iv) against (ii)) in a range."""
    t = th["G5D32"]
    tl = _tail(comp["top1_change_leads"], f"{where} top1_change_leads")
    mean_kl, max_kl = _num(comp, "mean_kl", where), _num(comp, "max_kl", where)
    n_top1 = tail_count_at_least(tl, t["decisive_lead_nats"])
    rules = {"top1": {"fired": n_top1 > 0, "count": n_top1},
             "mean": {"fired": not within(mean_kl, bounds["mean_bound"]), "mean_kl": mean_kl, "bound": bounds["mean_bound"],
                      "ratio": _ratio(mean_kl, bounds["mean_bound"])},
             "max": {"fired": not within(max_kl, bounds["max_bound"]), "max_kl": max_kl, "bound": bounds["max_bound"],
                     "ratio": _ratio(max_kl, bounds["max_bound"])}}
    return {"fired": any(v["fired"] for v in rules.values()), "rules": rules}


def g5_d32(base: dict | None, mutants: dict | None, th: dict, controls: dict | None = None) -> dict:
    """G5-D32: forced decode and chunk 64 against the forced chunk-2,048 baseline,
    per range. FAIL if a rule fires or Csort is above a ceiling; INCOMPLETE if
    mutant 20 fires no decode rule in any range; else PASS."""
    if base is None:
        return missing("G5-D32: no series")
    t = th["G5D32"]
    bounds = g5_d32_bounds(base, th)
    legs = {}
    for r in t["ranges"]:
        for comp in ("decode", "chunk64"):
            legs[f"{r}:{comp}"] = g5_d32_comparison(base[r][comp], bounds[r], th, f"G5-D32 {r} {comp}")
    ctrl = _controls_for("g5_d32_K8", mutants, {"bounds": bounds}, th, controls)
    fired = [k for k, v in legs.items() if v["fired"]]
    breach = [f"{r} {k}" for r, b in bounds.items() for k in b["ceiling_breach"]]
    threshold = {k: t[k] for k in ("decisive_lead_nats", "mean_factor_vs_csort", "mean_floor", "max_factor_vs_csort",
                                   "max_floor", "csort_mean_ceiling", "csort_max_ceiling")}
    return _with_controls(fired, breach, legs, ctrl, {"bounds": bounds}, threshold, "G5-D32")


# ---------------------------------------------------------------------------
# G5-R1 (§3.13)
# ---------------------------------------------------------------------------


def r_anchor(a: dict, th: dict, where: str) -> dict:
    """mean KL(R1||runner) <= 3 x mean KL(R1||single), pooled."""
    runner, single = _num(a, "mean_kl_r1_runner", where), _num(a, "mean_kl_r1_single", where)
    bound = th["G5R1"]["anchor_factor"] * single
    return {"ok": within(runner, bound), "mean_kl_r1_runner": runner, "mean_kl_r1_single": single, "bound": bound,
            "ratio": _ratio(runner, single)}


def g5_r1(arms: dict | None, floor: dict | None, mutants: dict | None, th: dict, controls: dict | None = None) -> dict:
    """{"g5_r1_K8": check, "g5_r1_K4": check}. K8: R-anchor, R-greedy (the
    registered greedy rule) and R-parity (the registered decode-vs-prefill bound
    with phase 4's floor). K4: R-anchor. Mutant 26 at K8 must fire R-parity, or
    both arms are INCOMPLETE."""
    th5 = th["G5"]
    arms = arms or {}
    bound = parity_bound(floor, th5) if floor is not None else None
    if bound is not None:
        ctrl = _controls_for("g5_r1_K8", mutants, {"bound": bound}, th, controls)
    else:  # no floor: mutant 26 cannot be judged; its entry stays absent (caught None)
        ctrl = {c["id"]: _absent_control(c) for c in required_controls("g5_r1_K8", controls)}
    out = {}
    k8 = arms.get("K8")
    if k8 is None or bound is None:
        out["g5_r1_K8"] = missing("G5-R1 K8: no runner series" if k8 is None else "G5-R1 K8: no noise floor")
    else:
        legs = {"R-anchor": r_anchor(k8.get("anchor"), th, "G5-R1 K8 anchor"),
                "R-greedy": {"ok": greedy_ok(k8["greedy"], th5), **k8["greedy"]},
                "R-parity": {"ok": parity_ok(k8["parity"], bound), **k8["parity"], **bound}}
        fired = [k for k, v in legs.items() if not v["ok"]]
        out["g5_r1_K8"] = _with_controls(fired, [], legs, ctrl, {"floor": floor}, _g5_r1_threshold(th, "K8"), "G5-R1")
    k4 = arms.get("K4")
    if k4 is None:
        out["g5_r1_K4"] = missing("G5-R1 K4: no runner series")
    else:
        legs = {"R-anchor": r_anchor(k4.get("anchor"), th, "G5-R1 K4 anchor")}
        fired = [k for k, v in legs.items() if not v["ok"]]
        # Mutant 26 runs at K8 only; its outcome governs both arms (DESIGN §4.2).
        out["g5_r1_K4"] = _with_controls(fired, [], legs, ctrl, {}, _g5_r1_threshold(th, "K4"), "G5-R1")
    return out


def _g5_r1_threshold(th: dict, arm: str) -> dict:
    t = {"anchor_factor": th["G5R1"]["anchor_factor"], "legs": th["G5R1"]["legs"][arm]}
    if arm == "K8":
        t.update({k: th["G5"][k] for k in ("greedy_decisive_lead_nats", "greedy_decisive_top1_min", "greedy_ref_top5_min",
                                           "parity_kl_floor", "parity_kl_factor", "parity_dis_factor", "parity_dis_add")})
    return t


# ---------------------------------------------------------------------------
# G5-BP-lean and allowed_B (§3.14)
# ---------------------------------------------------------------------------


def bp_d(p: dict, th: dict, where: str) -> dict:
    """(d) per subset: mean KL(R1||batched) <= 3 x mean KL(R1||single)."""
    t = th["G5BP"]
    legs = {}
    for sub in t["subsets"]:
        d = p.get(sub)
        n = _num(d, "n", f"{where} {sub}")
        batched = _num(d, "mean_kl_r1_batched", f"{where} {sub}")
        single = _num(d, "mean_kl_r1_single", f"{where} {sub}")
        bound = t["anchor_factor"] * single
        legs[sub] = {"ok": n > 0 and within(batched, bound), "n": int(n), "mean_kl_r1_batched": batched,
                     "mean_kl_r1_single": single, "bound": bound, "ratio": _ratio(batched, single)}
    return legs


def bp_pass(p: dict | None, B: int, th: dict, where: str) -> dict:
    """pass(arm, B) iff (d) holds for the first wave and for mid-run, and (e):
    admitted_mid_run >= 1 and max_live <= B."""
    if p is None:
        return {"pass": False, "reason": "not measured", "legs": {}}
    d = bp_d(p, th, where)
    adm, live = _num(p, "admitted_mid_run", where), _num(p, "max_live", where)
    e = {"ok": at_least_strict(adm, th["G5BP"]["admitted_mid_run_min"]) and within(live, B),
         "admitted_mid_run": int(adm), "max_live": int(live), "B": B}
    legs = {**{f"d:{k}": v for k, v in d.items()}, "e": e}
    failing = [k for k, v in legs.items() if not v["ok"]]
    return {"pass": not failing, "legs": legs, "failing": failing}


def allowed_B(pass_8: bool, pass_16: bool, control_caught: bool, th: dict) -> list:
    """allowed_B(arm) = {1} U ({2, 4, 8} if pass(arm, 8)) U ({16} if pass(arm, 8)
    and pass(arm, 16)); {1} for every arm when a required control of check
    g5_bp_K8_B8 in controls.json is not caught ("power not shown"; no
    INCOMPLETE, no cycle)."""
    t = th["G5BP"]
    out = set(t["allowed_B_always"])
    if control_caught:
        if pass_8:
            out |= set(t["allowed_B_if_pass_8"])
            if pass_16:
                out |= set(t["allowed_B_if_pass_8_and_16"])
    return sorted(out)


def g5_bp(results: dict | None, mutants: dict | None, th: dict, controls: dict | None = None) -> dict:
    """{"checks": {"g5_bp_<arm>_B<B>": check}, "allowed_B": {arm: [B]},
    "controls": {...}}. Never fails the gate: a failure at (arm, B) restricts
    production to allowed_B and is published as a batched-path discrepancy."""
    t = th["G5BP"]
    results = results or {}
    ctrl = _controls_for("g5_bp_K8_B8", mutants, {}, th, controls)
    caught = all(r["caught"] is True for r in ctrl.values())
    checks, passed = {}, {}
    for arm in ARMS:
        for B in t["B"]:
            p = _key(results.get(arm) or {}, B)
            res = bp_pass(p, B, th, f"G5-BP-lean {arm} B={B}")
            passed[(arm, B)] = res["pass"]
            reason = None
            if not res["pass"]:
                reason = ("not measured" if p is None else
                          f"batched-path discrepancy detected at ({arm}, B={B}), leg {', '.join(res['failing'])}; "
                          "production restricted to allowed_B")
            checks[f"g5_bp_{arm}_B{B}"] = check(PASS if res["pass"] else FAIL, p,
                                                 {"anchor_factor": t["anchor_factor"],
                                                  "admitted_mid_run_min": t["admitted_mid_run_min"], "B": B},
                                                 reason=reason, legs=res["legs"], blocking=False, role="allowed_B")
    allowed = {arm: allowed_B(passed[(arm, 8)], passed[(arm, 16)], caught, th) for arm in ARMS}
    uncaught = [r for r in ctrl.values() if r["caught"] is not True]
    note = None if caught else "; ".join(
        [f"power not shown: mutant {r['mutant']} ({r['mutant_name']})" for r in uncaught] + ["allowed_B = {1}"])
    return {"checks": checks, "allowed_B": allowed, "controls": ctrl, "power_note": note}


# ---------------------------------------------------------------------------
# Controls (controls.json; §4.2): caught, and the tiny margin
# ---------------------------------------------------------------------------


def _margin(unit: str, value, required: float, unbounded: bool = False) -> dict:
    met = unbounded or (value is not None and at_least_strict(value, required))
    return {"unit": unit, "value": value, "required": required, "met": bool(met), "unbounded": bool(unbounded)}


def _best_ratio(items: list) -> tuple:
    """(largest finite ratio or None, any unbounded) over [(num, den)]."""
    best, unbounded = None, False
    for num, den in items:
        r = _ratio(num, den)
        if r is None:
            unbounded |= num > 0
        elif best is None or r > best:
            best = r
    return best, unbounded


def control_result(control: dict, m: dict | None, calib: dict, th: dict, factor: float = 10.0) -> dict:
    """Whether a required control's bound leg fires on the mutant's series `m`
    (judged with the unmutated run's calibration), and its tiny margin in the
    units of §4.2. caught is None when the series is absent."""
    out = _absent_control(control)
    if m is None:
        return out
    cid = control["id"]
    if cid.startswith("G4F32/"):
        legs = g4_f32_legs(m, calib["cal"], th)
        if control["leg"] == "F6":
            per = {s: v for s, v in legs["F6"]["per"].items() if s in control["where"]}
            caught = any(v["count"] > 0 for v in per.values())
            out["margin"] = _margin("count", sum(v["count"] for v in per.values()), factor)
        else:  # F2
            keys = [k for k in legs["F2"]["per"] if (k in control["where"]) or
                    ("T9_buckets" in control["where"] and k.startswith("T9:"))]
            per = {k: legs["F2"]["per"][k] for k in keys}
            caught = any(v["fired"] for v in per.values())
            best, unb = _best_ratio([(v["mean_kl"], v["tau"]) for v in per.values()])
            out["margin"] = _margin("mean", best, factor, unb)
        out["values"] = per
    elif cid.startswith("G4F16/"):
        legs = g4_f16_legs(m, calib["r3"], th)
        per = legs["kappa"]["per"]
        caught = legs["kappa"]["fired"]
        best, unb = _best_ratio([(v["mean_kl_port"], v["mean_kl_r3"]) for v in per.values()])
        out["margin"] = _margin("ratio", best, factor * th["G4F16"]["kappa"], unb)
        out["values"] = per
    elif cid.startswith("G5D32/"):
        # Caught iff a decode rule (top-1, mean or max) fires in >= 1 range; the
        # margin is the best of the three rules' units over the ranges.
        per, caught, cands = {}, False, []
        for r, b in calib["bounds"].items():
            comp = (m.get(r) or {}).get("decode")
            if comp is None:
                raise RuleInputError(f"control {cid}: no decode series for range {r}")
            res = g5_d32_comparison(comp, b, th, f"control {cid} {r}")
            per[r] = res
            caught |= res["fired"]
            cands.append(dict(_margin("count", res["rules"]["top1"]["count"], factor), range=r, rule="top1"))
            for k in ("mean", "max"):  # the bounds have positive floors, so the ratios exist
                cands.append(dict(_margin(k, res["rules"][k]["ratio"], factor), range=r, rule=k))
        out["margin"] = max(cands, key=lambda c: (c["met"], (c["value"] or 0.0) / c["required"]))
        out["values"] = per
    elif cid.startswith("G5R1/"):
        bound = calib["bound"]
        par = m.get("parity")
        if par is None:
            raise RuleInputError(f"control {cid}: no parity values")
        caught = not parity_ok(par, bound)
        r = _ratio(_num(par, "mean_kl", f"control {cid}"), bound["kl_max"])
        out["margin"] = _margin("mean", r, factor, r is None and par["mean_kl"] > 0)
        out["values"] = {**par, **bound}
    elif cid.startswith("G5BP/"):
        d = bp_d(m, th, f"control {cid}")
        caught = any(not v["ok"] for v in d.values())
        best, unb = _best_ratio([(v["mean_kl_r1_batched"], v["mean_kl_r1_single"]) for v in d.values()])
        out["margin"] = _margin("ratio", best, factor * th["G5BP"]["anchor_factor"], unb)
        out["values"] = d
    else:
        raise RuleInputError(f"unknown control {cid}")
    out["caught"] = bool(caught)
    return out


def _absent_control(control: dict) -> dict:
    """A control's entry before (or without) its series: caught None."""
    return {"id": control["id"], "check": control["check"], "mutant": control["mutant"],
            "mutant_name": control.get("mutant_name", ""), "leg": control["leg"], "caught": None, "margin": None,
            "values": None}


def required_controls(check_id: str, controls: dict | None = None) -> list:
    controls = controls if controls is not None else load_controls()
    return [c for c in controls["required"] if c["check"] == check_id]


def _controls_for(check_id: str, mutants: dict | None, calib: dict, th: dict, controls: dict | None) -> dict:
    controls = controls if controls is not None else load_controls()
    factor = float(controls["margin_factor"])
    return {c["id"]: control_result(c, _key(mutants or {}, c["mutant"]), calib, th, factor)
            for c in required_controls(check_id, controls)}


def controls_outcome(checks: dict, g5bp: dict | None = None, controls: dict | None = None) -> dict:
    """The record's "controls" field: every required control of controls.json
    with its result (caught True / False, or None when it never ran), gathered
    from the check entries (and from g5_bp's result)."""
    controls = controls if controls is not None else load_controls()
    found = {}
    for c in checks.values():
        found.update((c or {}).get("controls") or {})
    if g5bp is not None:
        found.update(g5bp.get("controls") or {})
    out = {}
    for c in controls["required"]:
        out[c["id"]] = {**(found.get(c["id"]) or _absent_control(c)), "if_not_caught": c["if_not_caught"]}
    reg = checks.get("g2_real_weight_mutants")
    if reg is not None and reg.get("measured"):
        out["G2/registered"] = {"check": "g2_real_weight_mutants", "undetected": reg["measured"].get("undetected"),
                                "caught": not reg["measured"].get("undetected"),
                                "if_not_caught": controls["registered"]["if_not_caught"]}
    return out


# ---------------------------------------------------------------------------
# Preconditions decided here: P3 (G0k, §3.1) and P5 (§3.1)
# ---------------------------------------------------------------------------


def p3_judged_shapes(th: dict) -> list:
    """Every (projection, bits, rows) G0k judges (one-expert rows included)."""
    t = th["P3_G0k"]
    rows = sorted({n for v in t["judged_rows"].values() for n in v} | set(t["judged_rows_one_expert"]))
    return [(p, b, n) for p in t["projections"] for b in t["bits"] for n in rows]


def p3_g0k(entries: list, th: dict) -> dict:
    """The fix-5 rule at every judged shape and bit width: rel_f64(sorted) <= 3 x
    rel_f64(unsorted), zero bad rows, and a bitwise repeat. A judged shape
    without an entry fails (exit 3; Andrei decides)."""
    t = th["P3_G0k"]
    seen, failing = set(), []
    for e in entries:
        if not isinstance(e, dict) or any(k not in e for k in ("projection", "bits", "rows", "repeat_bitwise")):
            raise RuleInputError(f"G0k entry {e!r}: needs projection, bits, rows and repeat_bitwise")
        key = (e["projection"], int(e["bits"]), int(e["rows"]))
        seen.add(key)
        rs, ru = _num(e, "rel_f64_sorted", f"G0k {key}"), _num(e, "rel_f64_unsorted", f"G0k {key}")
        legs = {"rel": within(rs, t["rel_sorted_max_factor_vs_unsorted"] * ru),
                "bad_rows": within(_num(e, "bad_rows", f"G0k {key}"), t["bad_rows_max"]),
                "repeat": e["repeat_bitwise"] is True}
        if not all(legs.values()):
            failing.append({"shape": list(key), "failed": [k for k, v in legs.items() if not v],
                            "rel_f64_sorted": rs, "rel_f64_unsorted": ru, "bad_rows": e["bad_rows"],
                            "repeat_bitwise": e["repeat_bitwise"]})
    absent = [list(k) for k in p3_judged_shapes(th) if k not in seen]
    return {"ok": not failing and not absent, "failing": failing, "absent": absent, "n_judged": len(seen)}


def p5(m: dict, th: dict) -> dict:
    """P5a bitwise on layers {0, 4}; P5b bitwise; P5c mean KL(free||forced onto
    t-1's ids) >= 0.10 over T1 positions 1-1,535."""
    t = th["P5"]
    bw = (m.get("p5a") or {}).get("bitwise") or {}
    p5a = {str(l): _key(bw, l) is True for l in t["p5a_layers"]}
    p5b = (m.get("p5b") or {}).get("bitwise") is True
    kl = _num(m.get("p5c"), "mean_kl", "P5c")
    p5c = at_least_strict(kl, t["p5c_mean_kl_min"])
    parts = {"P5a": {"ok": all(p5a.values()), "layers": p5a}, "P5b": {"ok": p5b},
             "P5c": {"ok": p5c, "mean_kl": kl, "min": t["p5c_mean_kl_min"]}}
    return {"ok": all(v["ok"] for v in parts.values()), "parts": parts}


# ---------------------------------------------------------------------------
# Arm verdict, exit codes, gate runs and cycles (§3.16, §9.4)
# ---------------------------------------------------------------------------


def arm_verdict(arm: str, checks: dict, tiny: bool = False) -> dict:
    """{"verdict", "failing", "incomplete", "missing", "reasons"}. FAIL if a
    blocking check FAILs; otherwise MISSING if a blocking value is missing (a
    crash: exit 3); otherwise INCOMPLETE if one is INCOMPLETE; otherwise PASS.
    A tiny run skips checks marked applicable: False."""
    failing, incomplete, absent = [], [], []
    for cid in BLOCKING[arm]:
        c = checks.get(cid)
        if c is not None and tiny and c.get("applicable") is False:
            continue
        s = state_of(c)
        if s == FAIL:
            failing.append(cid)
        elif s == INCOMPLETE:
            incomplete.append(cid)
        elif s != PASS:
            absent.append(cid)
    verdict = FAIL if failing else MISSING if absent else INCOMPLETE if incomplete else PASS
    reasons = {cid: (checks.get(cid) or {}).get("reason") for cid in failing + incomplete + absent}
    return {"verdict": verdict, "failing": failing, "incomplete": incomplete, "missing": absent, "reasons": reasons}


EXIT_TABLE = (
    ("a precondition (P1-P5) failed (the run stops there)", 3, False),
    ("K8 FAIL", 1, True),
    ("an exception, or any blocking value missing (and K8 not FAIL)", 3, False),
    ("K8 INCOMPLETE, or K4 INCOMPLETE", 5, True),
    ("K8 PASS, K4 FAIL", 4, True),
    ("K8 PASS, K4 PASS", 0, True),
)


def exit_code_for(verdicts: dict, precondition_failed: bool = False, error: str | None = None) -> int:
    """The §3.16 table, first match wins. verdicts: {"K8": state, "K4": state};
    an arm that did not run counts as MISSING."""
    k8, k4 = verdicts.get("K8", MISSING), verdicts.get("K4", MISSING)
    if precondition_failed:
        return 3
    if k8 == FAIL:
        return 1
    if error is not None or any(v not in (PASS, FAIL, INCOMPLETE) for v in (k8, k4)):
        return 3
    if INCOMPLETE in (k8, k4):
        return 5
    if k8 == PASS and k4 == FAIL:
        return 4
    if k8 == PASS and k4 == PASS:
        return 0
    return 3


def is_gate_run(exit_code: int) -> bool:
    """A real-mode --all run that wrote a record with exit 0, 1, 4 or 5 (§9.4)."""
    return exit_code in (0, 1, 4, 5)


def run_counts(exit_codes: list, th: dict) -> dict:
    """Gate runs and fix cycles from the exit codes of the real runs so far, in
    order. A cycle is a gate run after the first that follows a gate run that
    exited 1, 4 or 5; exit 3 is never a gate run nor a cycle. At most
    fix_cycles_max cycles, so at most 1 + fix_cycles_max gate runs."""
    runs = [c for c in exit_codes if is_gate_run(c)]
    cycles = sum(1 for prev in runs[:-1] if prev in (1, 4, 5)) if runs else 0
    cmax = th["fix_cycles_max"]
    next_is_cycle = bool(runs) and runs[-1] in (1, 4, 5)
    may_run = len(runs) < 1 + cmax and (not next_is_cycle or cycles < cmax)
    return {"gate_runs": len(runs), "cycles_used": cycles, "cycles_max": cmax, "gate_runs_max": 1 + cmax,
            "next_run_is_cycle": next_is_cycle, "may_run_again": may_run}


def calibrators(checks: dict) -> dict:
    """The record's "calibrators" field: Csort per set and per range, R3, F_tiny,
    tau per set, from the G4-F32, G4-F16 and G5-D32 entries."""
    f32 = ((checks.get("g4_f32_K8") or {}).get("measured")) or {}
    f16 = ((checks.get("g4_f16_K8") or {}).get("measured")) or {}
    d32 = ((checks.get("g5_d32_K8") or {}).get("measured")) or {}
    return {"csort_sets": f32.get("csort"), "tau": f32.get("tau"), "f_tiny": f32.get("f_tiny"),
            "r3": f16.get("r3"),
            "csort_ranges": {r: {k: b[k] for k in ("csort_mean", "csort_max")} for r, b in (d32.get("bounds") or {}).items()}}


# ---------------------------------------------------------------------------
# H2 tripwire constants (§7.2; analysis/margins.json must equal them, W9)
# ---------------------------------------------------------------------------

TRIPWIRE_CONSTANTS = {"H2_tripwire_ub": -0.10, "H2_tripwire_did_ub": -0.10, "H2_tripwire_q": 0.95}
TRIPPED, NOT_TRIPPED, NOT_RUN = "TRIPPED", "NOT_TRIPPED", "NOT_RUN"


def tripwire_constants() -> dict:
    """The §7.2 constants (fractions: -0.10 = -10 pp); exported for W9's tests,
    which check analysis/margins.json against them."""
    return dict(TRIPWIRE_CONSTANTS)


def tripwire_bound(samples, q: float = TRIPWIRE_CONSTANTS["H2_tripwire_q"]) -> float:
    """The one-sided 95 % upper bound of bootstrap replicates: np.quantile(x, 0.95)."""
    import numpy as np

    return float(np.quantile(np.asarray(samples, dtype=np.float64), q))


def tripwire_rule(ub_k: float | None, ub_did: float | None, *, h2_run: bool = True, protocol_run: bool = True) -> str:
    """The specification predicate of T-G3: NOT_RUN if H2 is NOT RUN; TRIPPED iff
    UB_K < -0.10 and (the protocol control is NOT_RUN or UB_DiD < -0.10).
    analysis/verdicts.py's tripwire (Tier 1) must agree with it."""
    c = TRIPWIRE_CONSTANTS
    if not h2_run:
        return NOT_RUN
    if ub_k is None or math.isnan(ub_k):
        raise RuleInputError("tripwire: no UB_K although H2 ran")
    k_leg = ub_k < c["H2_tripwire_ub"]
    if not protocol_run:
        return TRIPPED if k_leg else NOT_TRIPPED
    if ub_did is None or math.isnan(ub_did):
        raise RuleInputError("tripwire: no UB_DiD although the protocol control ran")
    return TRIPPED if (k_leg and ub_did < c["H2_tripwire_did_ub"]) else NOT_TRIPPED
