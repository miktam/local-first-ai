"""exp_036 exploratory and descriptive analyses E1-E12 and Control C1 (Tier 2).

HYPOTHESIS "Exploratory and descriptive (predictions recorded now; no
verdicts)". Frozen by its own numbered amendment ("Tier-2 analysis") before
scorers/score_all.py first runs (BUILD_SPEC Build tiers). Every output is
labelled exploratory; a prediction check is reported as holds / does not hold /
not computable, never as a verdict.

compute_all(ctx, hyp) -> {"E1": {...}, ..., "E12": {...}, "C1": {...}}, where ctx
is a verdicts.Context and hyp the Tier-1 per-hypothesis entries.

Extra inputs read here (besides scores and the bench files):
- results/raw/<session>/<arm>/<task>_<effort>.jsonl for completion_tokens and
  wall_s (E5, E7); header and resume lines are skipped, a torn line is counted
  and skipped, the first record per key counts.
- score records may carry "reasoning_lang" (de | en | mixed | unknown, from
  scorers/lang_tag.py on the reasoning segment) for E4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis import stats
from analysis import verdicts as V

LABEL = "exploratory (HYPOTHESIS E1-E12; no verdict)"

PREDICTIONS = {
    "E1": "Vendor table: 10 public rows K 75.6 / G 80.8 / Q36 80.4 / Q38 81.8 — Kolibri last. Selection 7 rows: K 83.6 / G 83.2 / Q36 81.7. RGB 3 rows: K 56.9 / G 75.3 / Q36 77.5. Tier-A 6-row composite (MMLU EN/DE, IFBench, RGB × 3): K 67.4 / G 78.6 / Q36 77.5. We predict K8 is last on every composite that contains RGB CB and FC, and within ±1.5 pp of G8 on the selection rows.",
    "E2": "Δwrong ≤ +3 pp; A ≥ 0.40 (vendor AA-Omniscience non-hallucination 44.0 vs Gemma 14.3 / Qwen3.6 56.7, p. 101/189). Under forced answering the K8 deficit shrinks by ≥ 5 pp relative to H6's.",
    "E3": "Negative: K8 within ±6 pp of G8 (vendor 85.6 vs 86.0) and above Q36-8 (79.6). Fact-Check: K8 lowest of the three (vendor 34 vs 61 / 74), with \"deferred\" its most common failure.",
    "E4": "≥ 95 % (vendor 99 % on AIME-DE after RL, Fig. 34, p. 79/189).",
    "E5": "Tokens none < low < medium < high; acc(high) − acc(medium) ≤ 3 pp; acc(low) − acc(none) ≥ 10 pp. The SFT checkpoint showed +0.7 and a 26.5-pp none gap, p. 63–64/189; \"none\" is outside the RL effort mix, p. 183/189.",
    "E6": "No cliff: prefill ms/token at 32k and 64k ≤ 2 × the 15k value (FLOP arithmetic: 1.27 × and 1.65 ×; 120k ≈ 2.4 ×). KV slope 20,480 B/token ± 15 %.",
    "E7": "Kolibri's German chars/s at batch 1 ≥ 1.0 × G4's (the tokenizer offsets the decode deficit).",
    "E8": "|D_peer| ≤ 5 pp on MMLU and IFBench.",
    "E9": "Kolibri's standing against the peers is better on the selection rows than on RGB CB/FC.",
    "E10": "Q38 > K8 (vendor +6.8).",
    "E11": "Head + embedding KL < 0.005 nats/token at 8-bit; 8-bit router agreement ≥ 97 %; layer 0–1 intervention ≥ 95 % (vendor 98.5 % at pre-training, p. 137–140/189); bound < 448 (p. 12/189; the 448 figure is stated on p. 180/189).",
    "E12": "≥ 10 pp below effort high on GPQA-D and MMLU.",
    "C1": "≤ 2 % of extracted answers flip.",
}

SELECTION_ROWS = ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "aime_en", "aime_de", "ifbench")
RGB_CB_FC = ("rgb_cb", "rgb_fact")
RGB_NEG = ("rgb_neg",)
PREDICTIONS_LABEL_E1 = V.PLAIN_ANSWERS["Q3_e1_label"]
PUBLIC_ROWS = SELECTION_ROWS + ("rgb_cb", "rgb_neg", "rgb_fact")


def _wrap(eid: str, fn, *args) -> dict:
    try:
        body = fn(*args)
    except (V.NotRun, V.ConfigError, KeyError, ValueError, ZeroDivisionError) as e:
        body = {"status": "not computed", "reason": f"{type(e).__name__}: {e}"}
    out = {"label": LABEL, "prediction": PREDICTIONS[eid]}
    out.update(body)
    return out


def compute_all(ctx: V.Context, hyp: dict) -> dict:
    raw = load_raw(ctx.results_dir) if ctx.results_dir is not None else {}
    return {
        "E1": _wrap("E1", e1, ctx),
        "E2": _wrap("E2", e2, ctx, hyp),
        "E3": _wrap("E3", e3, ctx),
        "E4": _wrap("E4", e4, ctx),
        "E5": _wrap("E5", e5, ctx, raw),
        "E6": _wrap("E6", e6, ctx),
        "E7": _wrap("E7", e7, ctx, raw),
        "E8": _wrap("E8", e8, ctx),
        "E9": _wrap("E9", e9, ctx),
        "E10": _wrap("E10", e10, ctx),
        "E11": _wrap("E11", e11, ctx),
        "E12": _wrap("E12", e12, ctx),
        "C1": _wrap("C1", c1, ctx),
    }


# --------------------------------------------------------------------------
# Raw records (token counts, wall time)
# --------------------------------------------------------------------------

def load_raw(results_dir: Path) -> dict:
    """(arm, task, effort, pass, item) -> {"completion_tokens", "wall_s", "truncated"}."""
    out: dict = {"_torn": 0}
    base = Path(results_dir) / "raw"
    if not base.is_dir():
        return out
    for p in sorted(base.glob("*/*/*.jsonl")):
        if p.name.endswith(".steps.jsonl"):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out["_torn"] += 1
                    continue
                if rec.get("type") in ("header", "resume") or "key" not in rec:
                    continue
                k = rec["key"]
                key = (k["arm"], k["task"], k.get("effort", "high"), int(k.get("pass", 0)), str(k["item"]))
                if key not in out:
                    out[key] = {"completion_tokens": rec.get("completion_tokens"),
                                "wall_s": rec.get("wall_s"), "truncated": rec.get("truncated")}
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _row_value(ctx, arm, row, items):
    """Our score on a row, on the vendor's metric. RGB Fact-Check: the vendor reports Error Correction
    (p. 101/189), so only "corrected" counts here (scorers/score_all.py's `correct` also counts
    "detected"). Other rows: verdicts.row_point (MMLU post-stratified)."""
    if row == "rgb_fact":
        c = ctx.cell(arm, row)
        return float(np.mean([c[i].category == "corrected" and not c[i].truncated for i in items]))
    return V.row_point(ctx, arm, row, items)


def _row_D(ctx, arm, row):
    c = ctx.cell(arm, row)
    items = ctx.primary_items(row, c)
    return _row_value(ctx, arm, row, items) - V.vendor_value(ctx.vendor, row, arm)


def _row_scores(ctx, arms, rows):
    """{row: {arm: our score}} for rows every arm ran (MMLU post-stratified; GPQA EN primary 197)."""
    out = {}
    for row in rows:
        try:
            items = ctx.common_items(arms, row)
        except V.NotRun:
            continue
        if not items:
            continue
        out[row] = {a: _row_value(ctx, a, row, items) for a in arms}
    return out


def _vendor_rows(ctx, arms, rows):
    return {r: {a: V.vendor_value(ctx.vendor, r, a) for a in arms} for r in rows}


def _mean_over(rows_dict, arm, rows):
    vals = [rows_dict[r][arm] for r in rows if r in rows_dict]
    return float(np.mean(vals)) if vals else None


# --------------------------------------------------------------------------
# E1 — public-only composite
# --------------------------------------------------------------------------

def e1(ctx):
    arms = [ctx.K] + ctx.peers
    ours = _row_scores(ctx, arms, PUBLIC_ROWS)
    rows = [r for r in PUBLIC_ROWS if r in ours]
    if not rows:
        raise V.NotRun("no public row that every compared model ran")
    vend = _vendor_rows(ctx, arms, rows)
    comp = {a: _mean_over(ours, a, rows) for a in arms}
    vcomp = {a: _mean_over(vend, a, rows) for a in arms}
    groups = {"selection": SELECTION_ROWS, "rgb_cb_fc": RGB_CB_FC, "rgb_negative": RGB_NEG}
    by_group = {g: {"rows": [r for r in rs if r in ours],
                    "ours": {a: _mean_over(ours, a, rs) for a in arms},
                    "vendor": {a: _mean_over(vend, a, rs) for a in arms}} for g, rs in groups.items()}
    loo = {r: {a: _mean_over(ours, a, [x for x in rows if x != r]) for a in arms} for r in rows}
    en_rows = [r for r in rows if not r.endswith("_de")]
    de_rows = [r for r in rows if r.endswith("_de")]
    peers_have_aime_gpqa = any(r.startswith("aime") for r in rows) and any(r.startswith("gpqa") for r in rows)
    out = {
        "rows": rows, "ours": comp, "vendor_same_rows": vcomp, "groups": by_group, "leave_one_row_out": loo,
        "en": {a: _mean_over(ours, a, en_rows) for a in arms}, "de": {a: _mean_over(ours, a, de_rows) for a in arms},
        "composite_label": None if peers_have_aime_gpqa else PREDICTIONS_LABEL_E1,
        "vendor_table": ctx.vendor["other"]["e1_vendor_composites"],
    }
    K = ctx.K
    checks = {}
    if "rgb_cb" in rows and "rgb_fact" in rows:
        checks["kolibri_last_on_composite_with_rgb_cb_fc"] = all(comp[K] < comp[a] for a in ctx.peers)
    sel = by_group["selection"]["ours"]
    if "G8" in arms and sel.get(K) is not None and sel.get("G8") is not None:
        checks["kolibri_within_1_5pp_of_g8_on_selection"] = abs(sel[K] - sel["G8"]) <= 0.015
    out["prediction_checks"] = checks
    return out



# --------------------------------------------------------------------------
# E2 — says so? knows when forced?
# --------------------------------------------------------------------------

def e2(ctx, hyp):
    m = ctx.margins
    K, peers = ctx.K, ctx.peers
    cells = {a: ctx.cell(a, "rgb_cb") for a in [K] + peers}
    items = ctx.common_items([K] + peers, "rgb_cb")

    def wrong(a, i):
        it = cells[a][i]
        return float((not it.truncated) and it.score == 0.0 and it.category != "abstain")

    dw = np.array([wrong(K, i) - np.mean([wrong(a, i) for a in peers]) for i in items])
    s_dw = stats.boot_paired({"all": dw}, ctx.B(), stats.rng("E2|dwrong"))
    nc = [i for i in items if cells[K][i].score == 0.0 and not cells[K][i].truncated]
    level = float(m["ci_level"])
    out = {"n_items": len(items), "delta_wrong": float(np.mean(dw)),
           "delta_wrong_ci95": list(stats.percentile_ci(s_dw, level))}
    if nc:
        ab = np.array([float(cells[K][i].category == "abstain") for i in nc])
        s_a = stats.boot_paired({"all": ab}, ctx.B(), stats.rng("E2|abstain"))
        out["A"] = float(np.mean(ab))
        out["A_ci95"] = list(stats.percentile_ci(s_a, level))
    else:
        out["A"] = None
        out["A_ci95"] = None
    lo_dw, hi_dw = out["delta_wrong_ci95"]
    if out["A_ci95"] is not None and hi_dw <= float(m["E2_dwrong_max"]) and out["A_ci95"][0] >= float(m["E2_abstain_min"]):
        reading = "says so"
    elif lo_dw > float(m["E2_dwrong_max"]):
        reading = "guesses"
    else:
        reading = "unclear"
    out["reading"] = reading
    h6d = (hyp.get("H6") or {}).get("detail") or {}
    forced = h6d.get("forced", {})
    out["forced"] = forced
    out["knows_less"] = bool(forced.get("upper_below_margin"))
    checks = {"delta_wrong_le_3pp": out["delta_wrong"] <= float(m["E2_dwrong_max"])}
    if out["A"] is not None:
        checks["A_ge_0_40"] = out["A"] >= float(m["E2_abstain_min"])
    if forced.get("ran") and hyp.get("H6", {}).get("estimate") is not None:
        checks["forced_deficit_shrinks_by_5pp"] = forced["deficit"] - hyp["H6"]["estimate"] >= 0.05
    out["prediction_checks"] = checks
    out["vendor_non_hallucination"] = ctx.vendor["other"]["aa_omniscience_non_hallucination"]
    return out


# --------------------------------------------------------------------------
# E3 — RGB Negative and Fact-Check
# --------------------------------------------------------------------------

def e3(ctx):
    arms = [ctx.K] + ctx.peers
    out: dict = {}
    neg = {}
    for a in arms:
        if ctx.has_cell(a, "rgb_neg"):
            c = ctx.cell(a, "rgb_neg")
            neg[a] = float(np.mean([it.score for it in c.values()]))
    out["negative_rejection_rate"] = neg
    out["negative_vendor"] = {a: V.vendor_value(ctx.vendor, "rgb_neg", a) for a in arms}
    fact = {}
    for a in arms:
        if not ctx.has_cell(a, "rgb_fact"):
            continue
        c = ctx.cell(a, "rgb_fact")
        split = {"corrected": 0, "deferred": 0, "detected": 0, "other": 0}
        for it in c.values():
            cat = it.category if it.category in split else "other"
            split[cat] += 1
        gated = None
        if ctx.has_cell(a, "rgb_cb"):
            cb = ctx.cell(a, "rgb_cb")
            known = [i for i in c if i in cb and cb[i].score == 1.0]
            g = {"corrected": 0, "deferred": 0, "detected": 0, "other": 0}
            for i in known:
                cat = c[i].category if c[i].category in g else "other"
                g[cat] += 1
            gated = {"n_known_closed_book": len(known), "split": g}
        fact[a] = {"n": len(c), "split": split, "corrected_rate": split["corrected"] / len(c) if c else None,
                   "knowledge_gated": gated}
    out["fact_check"] = fact
    out["fact_vendor"] = {a: V.vendor_value(ctx.vendor, "rgb_fact", a) for a in arms}
    if not neg and not fact:
        raise V.NotRun("no RGB Negative or Fact-Check cells (P0 only)")
    K = ctx.K
    checks = {}
    if K in neg and "G8" in neg:
        checks["neg_within_6pp_of_g8"] = abs(neg[K] - neg["G8"]) <= 0.06
    if K in neg and "Q36-8" in neg:
        checks["neg_above_q36"] = neg[K] > neg["Q36-8"]
    if K in fact and all(a in fact for a in ctx.peers):
        checks["fc_lowest"] = all(fact[K]["corrected_rate"] < fact[a]["corrected_rate"] for a in ctx.peers)
        fails = {k: v for k, v in fact[K]["split"].items() if k != "corrected"}
        checks["deferred_most_common_failure"] = max(fails, key=lambda k: (fails[k], k == "deferred")) == "deferred"
    out["prediction_checks"] = checks
    return out


# --------------------------------------------------------------------------
# E4 — German reasoning consistency
# --------------------------------------------------------------------------

def e4(ctx):
    K = ctx.K
    per_task, tags = {}, []
    for task in ("gpqa_de", "mmlu_de", "aime_de"):
        c = ctx.scores.cell(K, task, ctx.effort(K))
        if not c:
            continue
        t = [it.reasoning_lang for it in c.values()]
        if any(x is None for x in t):
            raise V.ConfigError(f"score records of {K} {task} carry no reasoning_lang")
        per_task[task] = {"n": len(t), "share_de": sum(x == "de" for x in t) / len(t)}
        tags += t
    if not tags:
        raise V.NotRun("no DE cells for the primary Kolibri arm")
    share = sum(x == "de" for x in tags) / len(tags)
    return {"share_de": share, "per_task": per_task, "n": len(tags),
            "prediction_checks": {"share_de_ge_95pct": share >= 0.95},
            "vendor": ctx.vendor["other"]["aime_de_reasoning_language_after_rl"]}


# --------------------------------------------------------------------------
# E5 — effort curve (K8, GPQA-D EN)
# --------------------------------------------------------------------------

def e5(ctx, raw):
    m = ctx.margins
    arm = "K8"
    out = {}
    for eff in ("none", "low", "medium", "high"):
        c = ctx.scores.cell(arm, "gpqa_en", eff)
        if not c:
            continue
        items = ctx.primary_items("gpqa_en", c)
        toks = [raw.get((arm, "gpqa_en", eff, 0, i), {}).get("completion_tokens") for i in items]
        toks = [t for t in toks if t is not None]
        out[eff] = {"n": len(items), "accuracy": float(np.mean([c[i].score for i in items])),
                    "mean_completion_tokens": float(np.mean(toks)) if toks else None}
    if len(out) < 2:
        raise V.NotRun("E5 needs K8 GPQA-D EN at two or more efforts (B7, B5)")
    checks = {}
    order = [e for e in ("none", "low", "medium", "high") if e in out and out[e]["mean_completion_tokens"] is not None]
    if len(order) >= 2:
        checks["tokens_increasing"] = all(out[a]["mean_completion_tokens"] < out[b]["mean_completion_tokens"]
                                          for a, b in zip(order, order[1:]))
    if "high" in out and "medium" in out:
        checks["high_minus_medium_le_3pp"] = out["high"]["accuracy"] - out["medium"]["accuracy"] <= float(m["E5_high_minus_medium_max"])
    if "low" in out and "none" in out:
        checks["low_minus_none_ge_10pp"] = out["low"]["accuracy"] - out["none"]["accuracy"] >= float(m["E5_low_minus_none_min"])
    return {"efforts": out, "prediction_checks": checks,
            "vendor_sft": ctx.vendor["other"]["sft_effort_14_benchmarks"]}


# --------------------------------------------------------------------------
# E6 — context ladder and prefill cliff
# --------------------------------------------------------------------------

def _nearest(sizes, target, tol):
    c = [s for s in sizes if abs(s - target) <= tol]
    return min(c, key=lambda s: abs(s - target)) if c else None


def e6(ctx):
    m = ctx.margins
    if "ladder" in ctx.bench.get("_incomplete", []):
        raise V.NotRun("the ladder bench file is incomplete")
    recs = ctx.bench.get("ladder") or []
    if not recs:
        raise V.NotRun("no bench/ladder_<UTC>.jsonl (B4)")
    by: dict = {}
    for r in recs:
        if not V._is_run(r) or r.get("arm") is None:
            continue
        n = r.get("n_context", r.get("n_tokens"))
        peak = r.get("peak_bytes", r.get("peak_memory_bytes"))
        d = by.setdefault(r["arm"], {}).setdefault(int(n), {"prompt_tps": [], "peak": []})
        if r.get("prompt_tps"):
            d["prompt_tps"].append(float(r["prompt_tps"]))
        if peak is not None:
            d["peak"].append(float(peak))
    out = {"hardware": "M5 Max, MLX 0.31.2", "arms": {}}
    checks = {}
    for arm, sizes in sorted(by.items()):
        ms = {n: 1000.0 / float(np.median(v["prompt_tps"])) for n, v in sizes.items() if v["prompt_tps"]}
        a = {"prefill_ms_per_token": {str(n): ms[n] for n in sorted(ms)}}
        base = _nearest(ms, 15000, 2000)
        if base is not None:
            for tgt in (32768, 65536):
                if tgt in ms:
                    a[f"ratio_{tgt}_vs_15k"] = ms[tgt] / ms[base]
                    checks[f"{arm}_no_cliff_{tgt}"] = ms[tgt] <= float(m["E6_cliff_factor"]) * ms[base]
        pts = [(n, float(np.median(v["peak"]))) for n, v in sizes.items() if v["peak"]]
        if len(pts) >= 2:
            x = np.array([p[0] for p in pts], dtype=np.float64)
            y = np.array([p[1] for p in pts], dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0])
            a["kv_slope_bytes_per_token"] = slope
            exp_ = float(m["E6_kv_bytes_per_token"])
            checks[f"{arm}_kv_slope_within_15pct"] = abs(slope - exp_) <= float(m["E6_kv_tolerance"]) * exp_
        out["arms"][arm] = a
    out["prediction_checks"] = checks
    return out


# --------------------------------------------------------------------------
# E7 — operator metrics
# --------------------------------------------------------------------------

def e7(ctx, raw):
    out: dict = {"cells": {}}
    for (arm, task, effort, pass_) in ctx.scores.keys():
        c = ctx.scores.cell(arm, task, effort, pass_)
        toks = [raw.get((arm, task, effort, pass_, i), {}).get("completion_tokens") for i in c]
        walls = [raw.get((arm, task, effort, pass_, i), {}).get("wall_s") for i in c]
        n_correct = sum(it.score for it in c.values())
        toks_ok = [t for t in toks if t is not None]
        walls_ok = [w for w in walls if w is not None]
        out["cells"][f"{arm}/{task}/{effort}/pass{pass_}"] = {
            "n": len(c),
            "truncation_rate": sum(it.truncated for it in c.values()) / len(c),
            "parse_failure_rate": sum((not it.parse_ok) and not it.truncated for it in c.values()) / len(c),
            "mean_completion_tokens": float(np.mean(toks_ok)) if toks_ok else None,
            "tokens_per_correct": (float(np.sum(toks_ok)) / n_correct) if toks_ok and n_correct else None,
            "sequence_seconds_per_correct": (float(np.sum(walls_ok)) / n_correct) if walls_ok and n_correct else None,
        }
    out["note"] = "sequence seconds are summed per-sequence wall times under batching, not cell wall time"
    # German chars/s at batch 1 = decode tok/s x chars/token (H5 corpus).
    tok = ctx.bench.get("tokenizer")
    runs = V.speed_runs(ctx.bench.get("speed") or [])
    if tok and runs:
        cols = V.tokenizer_docs(tok)
        chars = float(cols["chars"].sum())
        cpt = {fam: chars / float(cols["tokens"][fam].sum()) for fam in ("kolibri", "gemma4")}
        tps = {}
        for arm in ("K4", "G4"):
            v = [b[arm] for b in runs.values() if arm in b]
            if v:
                tps[arm] = float(np.median(v))
        if "K4" in tps and "G4" in tps:
            k = tps["K4"] * cpt["kolibri"]
            g = tps["G4"] * cpt["gemma4"]
            out["german_chars_per_s_batch1"] = {"K4": k, "G4": g, "ratio": k / g, "chars_per_token": cpt,
                                                "label": "M5 Max, MLX 0.31.2"}
            out["prediction_checks"] = {"k4_german_chars_per_s_ge_g4": k / g >= 1.0}
    out.setdefault("prediction_checks", {})
    out["raw_torn_lines_skipped"] = raw.get("_torn", 0) if raw else 0
    return out


# --------------------------------------------------------------------------
# E8 — relative replication and peer positive control
# --------------------------------------------------------------------------

def e8(ctx):
    m = ctx.margins
    level = float(m["ci_level"])
    arms = [ctx.K] + ctx.peers
    rows = [r for r in ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb", "gpqa_en", "gpqa_de", "aime_en", "aime_de")]
    table: dict = {}
    for row in rows:
        for a in arms:
            if not ctx.has_cell(a, row):
                continue
            s = V.h2_like(ctx, a, [row], ctx.B(), f"E8|{a}|{row}")
            table.setdefault(row, {})[a] = {"D": s["point"], "ci95": list(stats.percentile_ci(s["samples"], level))}
    if not table:
        raise V.NotRun("no rows")
    did = {}
    for row, d in table.items():
        if ctx.K in d and any(p in d for p in ctx.peers):
            did[row] = d[ctx.K]["D"] - float(np.mean([d[p]["D"] for p in ctx.peers if p in d]))
    flag_thr = float(m["E8_flag"])
    flags = [{"arm": a, "row": r, "D_peer": v["D"]} for r, d in table.items() for a, v in d.items()
             if a in ctx.peers and abs(v["D"]) > flag_thr]
    pred = float(m["E8_prediction_abs"])
    checks = {f"{a}_{r}_abs_le_5pp": abs(table[r][a]["D"]) <= pred
              for r in ("mmlu_en", "mmlu_de", "ifbench") if r in table for a in ctx.peers if a in table[r]}
    return {"D": table, "DiD": did, "flags_gt_8pp": flags, "prediction_checks": checks}


# --------------------------------------------------------------------------
# E9 — three groups
# --------------------------------------------------------------------------

def e9(ctx):
    arms = [ctx.K] + ctx.peers
    groups = {"selection": SELECTION_ROWS, "rgb_cb_fc": RGB_CB_FC, "rgb_negative": RGB_NEG}
    out = {}
    for g, rows in groups.items():
        ds = [_row_D(ctx, ctx.K, r) for r in rows if ctx.has_cell(ctx.K, r)]
        ours = _row_scores(ctx, arms, rows)
        gaps = [ours[r][ctx.K] - float(np.mean([ours[r][p] for p in ctx.peers])) for r in ours]
        out[g] = {"mean_D_kolibri": float(np.mean(ds)) if ds else None,
                  "kolibri_minus_peers": float(np.mean(gaps)) if gaps else None,
                  "rows": [r for r in rows if r in ours]}
    checks = {}
    if out["selection"]["kolibri_minus_peers"] is not None and out["rgb_cb_fc"]["kolibri_minus_peers"] is not None:
        checks["standing_better_on_selection"] = out["selection"]["kolibri_minus_peers"] > out["rgb_cb_fc"]["kolibri_minus_peers"]
    return {"groups": out, "prediction_checks": checks}


# --------------------------------------------------------------------------
# E10, E11, E12, C1
# --------------------------------------------------------------------------

def e10(ctx):
    items = ctx.common_items(["Q38-8", "K8"], "gpqa_de")
    cq, ck = ctx.cell("Q38-8", "gpqa_de"), ctx.cell("K8", "gpqa_de")
    d = np.array([cq[i].score - ck[i].score for i in items])
    s = stats.boot_paired({"all": d}, ctx.B(), stats.rng("E10"))
    return {"n": len(items), "q38_minus_k8": float(np.mean(d)), "ci95": list(stats.percentile_ci(s)),
            "vendor": ctx.vendor["vendor_gaps"]["E10_gpqa_de_qwen3_8_minus_kolibri"],
            "prediction_checks": {"q38_above_k8": float(np.mean(d)) > 0}}


def e11(ctx):
    gate = ctx.bench.get("gate")
    if not gate:
        raise V.NotRun("no results/gate/gate_<UTC>.json")
    diag = gate.get("E11") or gate.get("e11") or gate.get("diagnostics")
    if diag is None:
        raise V.NotRun("the gate record carries no E11 / diagnostics section")
    return {"gate_diagnostics": diag, "vendor": {
        "layer01_routed_intervention_rate": ctx.vendor["other"]["layer01_routed_intervention_rate"],
        "qk_norm_fp8_bound": ctx.vendor["other"]["qk_norm_fp8_bound"]}}


def e12(ctx):
    m = ctx.margins
    out, checks = {}, {}
    for task in ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench", "rgb_cb"):
        cn = ctx.scores.cell("K8", task, "none")
        ch = ctx.scores.cell("K8", task, "high")
        if not cn or not ch:
            continue
        items = ctx.primary_items(task, set(cn) & set(ch))
        if V._is_mmlu(task):
            an = V.row_point(ctx, "K8", task, items, cn)
            ah = V.row_point(ctx, "K8", task, items, ch)
        else:
            an = float(np.mean([cn[i].score for i in items]))
            ah = float(np.mean([ch[i].score for i in items]))
        out[task] = {"n": len(items), "none": an, "high": ah, "high_minus_none": ah - an}
        if task.startswith(("gpqa", "mmlu")):
            checks[f"{task}_none_ge_10pp_below_high"] = ah - an >= float(m["E12_gap_min"])
    if not out:
        raise V.NotRun("no K8 effort-none cells (B5)")
    return {"label_effort": "effort none — outside Kolibri's RL effort mix (p. 183/189)",
            "tasks": out, "prediction_checks": checks}


def c1(ctx):
    recs = ctx.bench.get("c1") or []
    if not recs:
        raise V.NotRun("no bench/c1_<UTC>.jsonl")
    summary = ctx.bench.get("c1_summary") or {}
    items = [r for r in recs if "flipped" in r]
    if items:
        rate = sum(bool(r["flipped"]) for r in items) / len(items)
        n = len(items)
    elif summary.get("flip_rate") is not None:
        rate, n = float(summary["flip_rate"]), summary.get("n_items")
    else:
        raise V.ConfigError("c1 records carry neither 'flipped' nor an end summary with 'flip_rate'")
    return {"flip_rate": rate, "n": n, "B": summary.get("B"),
            "n_diverged": summary.get("n_diverged"),
            "prediction_checks": {"flip_rate_le_2pct": rate <= float(ctx.margins["C1_flip_max"])}}
