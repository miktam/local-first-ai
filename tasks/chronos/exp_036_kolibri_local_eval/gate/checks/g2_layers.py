# SPDX-License-Identifier: MIT
"""G2: per layer, real weights (HYPOTHESIS Phase 0 G2; BUILD_SPEC §5.3 g2_layers.py).

Port layer l is fed the reference input h_l (all layers). Errors are measured
on the branch outputs before the residual add, each relative to its own
reference norm (item 29): e = ||r_port - r_ref|| / ||r_ref|| per token.

  fp32, forced     median e <= 1e-4, max e <= 1e-3 per layer and branch;
                   embedding exact; final norm + head max |dlogit| <= 1e-3
  fp32, natural    top-6 sets identical except where the reference's 6th-7th
                   biased gap < 1e-4 (exempt pairs reported with their gaps)
  bf16, forced     per layer and branch: median e <= 3 x emu's median and
                   p99 e <= max(5e-2, 3 x emu's p99)
  bf16, natural    disagreement fraction <= max(0.2 %, 2 x emu's), and every
                   disagreement has a reference gap < 6 sigma_l (sigma_l = RMS
                   of port - reference router logits in layer l)
  router margin    bf16-router mutant 10 vs the port, paired token bootstrap of
                   agree_port - agree_mutant; blocks only if its p99 < 0
  G2q              the quantised layer of each converted directory vs the
                   reference on that layer's dequantised weights, forced;
                   embedding max relative error <= 4e-3; head relative L2 <= 1e-3
  T9               bf16 forced, bucketed by position, against emu's T1-T8 stats
  mutants          port mutants 1-8 and 12-15 through the fp32 harness on
                   layers {0, 3, 4, 49}: each must fail some fp32 check

"Forced" puts the port on the reference's expert ids with the weights from its
own fp32 logits (item 12; gate/harness.py RouteTap). One layer is resident at
a time.
"""

from __future__ import annotations

import numpy as np

from gate import common
from gate.checks.ref_pass import CapabilityMissing, Dump, RefRunner, segments_of, set_disagree


def _rows(n8: int, L: int, t9: int):
    return slice(0, n8), slice(n8, n8 + t9)


def branch_stats(port: dict, r_attn_ref, r_moe_ref) -> dict:
    H = r_attn_ref.shape[-1]
    ea = common.rel_err_rows(port["r_attn"].reshape(-1, H), r_attn_ref)
    em = common.rel_err_rows(port["r_moe"].reshape(-1, H), r_moe_ref)
    return {"attn": common.stats(ea), "moe": common.stats(em), "_e": {"attn": ea, "moe": em}}


def selection(port_ids, ref_top6, ref_gap, near_tie: float) -> dict:
    k = ref_top6.shape[-1]
    dis = set_disagree(np.asarray(port_ids).reshape(-1, k), np.asarray(ref_top6))
    gaps = np.asarray(ref_gap)[dis]
    exempt = gaps < near_tie
    return {"n": int(dis.size), "n_disagree": int(dis.sum()), "n_exempt": int(exempt.sum()),
            "n_violations": int((~exempt).sum()), "exempt_gaps": sorted(map(float, gaps[exempt]))[:20],
            "violation_gaps": sorted(map(float, gaps[~exempt]))[:20], "_dis": dis}


def bf16_selection(port: dict, ref_top6, ref_gap, ref_logits) -> dict:
    k = ref_top6.shape[-1]
    dis = set_disagree(port["ids"].reshape(-1, k), np.asarray(ref_top6))
    diff = port["router_logits"].reshape(ref_logits.shape).astype(np.float64) - np.asarray(ref_logits, dtype=np.float64)
    sigma = float(np.sqrt(np.mean(diff * diff)))
    gaps = np.asarray(ref_gap)[dis]
    return {"n": int(dis.size), "n_disagree": int(dis.sum()), "sigma": sigma,
            "max_gap_over_sigma": float(gaps.max() / sigma) if gaps.size and sigma > 0 else 0.0,
            "_gaps": gaps, "_dis": dis}


def run(ctx, dump: Dump, emu: dict | None, textset, log=print) -> dict:
    """Phase 3. ctx: bf16_dir, arm_dir(arm), arms, mutant (tiny tests only),
    thresholds, want(check) -> bool."""
    from gate import harness, port_mutants

    th = ctx.thresholds["G2"]
    prof = textset.profile
    L, T9 = prof.text_len, prof.t9_len
    n8 = 8 * L
    s8, s9 = _rows(n8, L, T9)
    n_layers = int(dump.record()["num_layers"])
    k = int(np.asarray(dump.layer(0, "top6")).shape[-1])
    lengths8 = [L] * 8
    out = {"n_layers": n_layers, "layers_fp32": [], "layers_bf16": [], "layers_t9": [], "layers_g2q": {},
           "natural_fp32": [], "natural_bf16": [], "notes": []}
    want_bf16 = emu is not None
    if not want_bf16:
        out["notes"].append("bf16 rows not run: no bf16-emulation calibration (reference capability missing)")

    p32 = harness.PortLayers(ctx.bf16_dir, "fp32", mutant=ctx.mutant)
    p16 = harness.PortLayers(ctx.bf16_dir, "bf16", mutant=ctx.mutant) if want_bf16 else None
    p_r10 = (harness.PortLayers(ctx.bf16_dir, "bf16", mutant="bf16_router_logits")
             if want_bf16 and ctx.mutant is None else None)
    agree_port = np.zeros(n8)
    agree_mut = np.zeros(n8)

    # G2q: the quantised layer of each arm vs the dequantised reference.
    g2q = {}
    for arm in ctx.arms:
        try:
            g2q[arm] = {"port": harness.PortLayers(ctx.arm_dir(arm), "bf16", mutant=ctx.mutant),
                        "ref": RefRunner(ctx.bf16_dir, "dequant", dequant_dir=ctx.arm_dir(arm)), "rows": []}
        except CapabilityMissing as e:
            out["layers_g2q"][arm] = {"capability_missing": str(e)}
    mutant_layers = [i for i in prof.real_weight_mutant_layers if i < n_layers]
    mutants = {}
    if ctx.mutant is None:
        mutants = {name: {"layers": {}, "harness": None} for name in port_mutants.REAL_WEIGHT_MUTANTS}
    crosscheck = None

    for i in range(n_layers):
        h_all = dump.layer(i, "h_in")
        H = h_all.shape[-1]
        h8 = np.array(h_all[s8]).reshape(8, L, H)
        top_all = np.array(dump.layer(i, "top6"))
        top8 = top_all[s8].reshape(8, L, k)
        gap8 = np.array(dump.layer(i, "top6_gap")[s8])
        ra8, rm8 = np.array(dump.layer(i, "r_attn")[s8]), np.array(dump.layer(i, "r_moe")[s8])
        lg8 = np.array(dump.layer(i, "router_logits")[s8])

        # fp32
        layer = p32.build_layer(i)
        f = p32.run(layer, i, h8, force_ids=top8)
        st = branch_stats(f, ra8, rm8)
        for br in ("attn", "moe"):
            out["layers_fp32"].append({"layer": i, "branch": br, **st[br]})
        nat = p32.run(layer, i, h8)
        sel = selection(nat["ids"], top_all[s8], gap8, th["fp32_selection_near_tie_gap"])
        sel.pop("_dis")
        out["natural_fp32"].append({"layer": i, **sel})
        if i == 0 and ctx.mutant is None:
            crosscheck = p32.forcing_crosscheck(layer, i, h8, top8)
        del layer

        if want_bf16:
            layer = p16.build_layer(i)
            fb = p16.run(layer, i, h8, force_ids=top8)
            stb = branch_stats(fb, ra8, rm8)
            for br in ("attn", "moe"):
                out["layers_bf16"].append({"layer": i, "branch": br, **{k2: v for k2, v in stb[br].items()}})
            nb = p16.run(layer, i, h8)
            sb = bf16_selection(nb, top_all[s8], gap8, lg8)
            dis = sb.pop("_dis")
            gaps = sb.pop("_gaps")
            sb["n_beyond_6sigma"] = int(np.sum(gaps >= th["bf16_selection_gap_sigma"] * sb["sigma"]))
            out["natural_bf16"].append({"layer": i, **sb})
            agree_port += ~dis
            # T9, bucketed, forced, through the port's own cache in chunks.
            h9 = np.array(h_all[s9]).reshape(1, T9, H)
            top9 = top_all[s9].reshape(1, T9, k)
            f9 = p16.run(layer, i, h9, force_ids=top9, chunk=prof.prefill_chunk)
            ra9, rm9 = np.array(dump.layer(i, "r_attn")[s9]), np.array(dump.layer(i, "r_moe")[s9])
            st9 = branch_stats(f9, ra9, rm9)
            for a, b in prof.t9_buckets:
                for br in ("attn", "moe"):
                    out["layers_t9"].append({"layer": i, "branch": br, "bucket": [a, b], **common.stats(st9["_e"][br][a:b])})
            del layer
            if p_r10 is not None:
                lm = p_r10.build_layer(i)
                nm = p_r10.run(lm, i, h8)
                agree_mut += ~set_disagree(nm["ids"].reshape(-1, k), top_all[s8])
                del lm

        for arm in list(g2q):
            g = g2q[arm]
            try:
                rq = g["ref"].branches(i, np.array(h_all[s8]), segments_of(lengths8), force_ids=top_all[s8])
            except CapabilityMissing as e:
                out["layers_g2q"][arm] = {"capability_missing": str(e)}
                del g2q[arm]
                continue
            lq = g["port"].build_layer(i)
            fq = g["port"].run(lq, i, h8, force_ids=top8)
            stq = branch_stats(fq, rq["r_attn"], rq["r_moe"])
            for br in ("attn", "moe"):
                g["rows"].append({"layer": i, "branch": br, **stq[br]})
            del lq

        if i in mutant_layers:
            for name, m in mutants.items():
                if m["harness"] is None:
                    m["harness"] = harness.PortLayers(ctx.bf16_dir, "fp32", mutant=name)
                pm = m["harness"]
                lmu = pm.build_layer(i)
                fm = branch_stats(pm.run(lmu, i, h8, force_ids=top8, check_call=False), ra8, rm8)
                sm = selection(pm.run(lmu, i, h8, check_call=False)["ids"], top_all[s8], gap8,
                               th["fp32_selection_near_tie_gap"])
                forced_fail = any(fm[br]["median"] > th["fp32_forced_rel_err_median_max"]
                                  or fm[br]["max"] > th["fp32_forced_rel_err_max_max"] for br in ("attn", "moe"))
                m["layers"][i] = {"forced_fail": forced_fail, "selection_violations": sm["n_violations"],
                                  "attn_max": fm["attn"]["max"], "moe_max": fm["moe"]["max"]}
                del lmu
        log(f"[gate] G2 layer {i + 1}/{n_layers}")

    # Embedding and head, fp32 (from the BF16 shards).
    ids8 = np.array([textset.ids[t] for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")])
    emb_ref = np.array(dump.layer(0, "h_in")[s8])
    emb = p32.embedding_rows(ids8).reshape(n8, -1)
    out["embedding"] = {"max_abs": float(np.abs(emb.astype(np.float64) - emb_ref).max()),
                        "exact": bool(np.array_equal(emb.astype(np.float32), emb_ref))}
    h_last = np.array(dump.load("h_last.npy")[s8])
    ref_logits = dump.load("logits.npy")
    worst = 0.0
    for a, block in p32.head_chunks(h_last):
        ref = np.asarray(ref_logits[a:a + block.shape[0]], dtype=np.float32)
        worst = max(worst, float(np.abs(block.astype(np.float64) - ref.astype(np.float64)).max()))
    out["head"] = {"max_abs_dlogit": worst}
    for name, m in mutants.items():
        e = m["harness"].embedding_rows(ids8).reshape(n8, -1) if m["harness"] is not None else None
        m["embedding_exact"] = bool(e is not None and np.array_equal(e.astype(np.float32), emb_ref))
        m["detected"] = (any(v["forced_fail"] or v["selection_violations"] > 0 for v in m["layers"].values())
                         or not m["embedding_exact"])
        m.pop("harness")
    out["real_weight_mutants"] = {"layers": mutant_layers, "mutants": mutants}
    out["forcing_crosscheck"] = crosscheck

    if want_bf16 and p_r10 is not None:
        rng = np.random.Generator(np.random.PCG64(common.seed_from("G2", "router_margin")))
        d = (agree_port - agree_mut) / n_layers
        B = 10_000
        boot = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(B)])
        out["router_margin"] = {"agree_port": float(agree_port.mean() / n_layers),
                                "agree_mutant": float(agree_mut.mean() / n_layers),
                                "diff_mean": float(d.mean()), "diff_p01": float(np.quantile(boot, 0.01)),
                                "diff_p99": float(np.quantile(boot, 0.99)), "B": B}

    # G2q embedding and head, against the dequantised reference.
    final8 = np.array(dump.load("final_norm.npy")[s8])
    for arm, g in g2q.items():
        res = {"layers": g["rows"]}
        e_port = g["port"].embedding_rows(ids8).reshape(n8, -1)
        e_ref = np.asarray(g["ref"].ref.embed(ids8.reshape(-1)), dtype=np.float64)
        res["embedding_max_rel"] = float(common.rel_err_rows(e_port, e_ref).max())
        hr = g["ref"].ref.lm_head(final8)  # [N, V] fp32, the dequantised reference head, once
        res["head_rel_l2_max"] = max(float(common.rel_err_rows(block, hr[a:a + block.shape[0]]).max())
                                     for a, block in g["port"].head_chunks(final8, normed=True))
        out["layers_g2q"][arm] = res
    return out


# ---------------------------------------------------------------------------
# Verdicts (pure; thresholds from gate/thresholds.json)
# ---------------------------------------------------------------------------


def _bf16_rule(rows: list, emu_layers: list, th: dict) -> tuple[bool, list]:
    bad = []
    for r in rows:
        e = emu_layers[r["layer"]][r["branch"]]
        lim_med = th["bf16_forced_median_factor_vs_emu"] * e["median"]
        lim_p99 = max(th["bf16_forced_p99_floor"], th["bf16_forced_p99_factor_vs_emu"] * e["p99"])
        r["limit_median"], r["limit_p99"] = lim_med, lim_p99
        r["pass"] = r["median"] <= lim_med and r["p99"] <= lim_p99
        if not r["pass"]:
            bad.append({k: r[k] for k in ("layer", "branch", "median", "p99") if k in r} | ({"bucket": r["bucket"]} if "bucket" in r else {}))
    return not bad, bad


def evaluate(res: dict, emu: dict | None, th: dict) -> dict:
    """{check id: {"pass", "measured", "threshold"}} for the G2 checks."""
    checks = {}
    bad = []
    for r in res["layers_fp32"]:
        r["pass"] = r["median"] <= th["fp32_forced_rel_err_median_max"] and r["max"] <= th["fp32_forced_rel_err_max_max"]
        if not r["pass"]:
            bad.append({k: r[k] for k in ("layer", "branch", "median", "max")})
    checks["g2_fp32_forced_branch"] = {
        "pass": not bad, "measured": {"worst_median": max(r["median"] for r in res["layers_fp32"]),
                                      "worst_max": max(r["max"] for r in res["layers_fp32"]), "failing": bad[:20]},
        "threshold": {"median_max": th["fp32_forced_rel_err_median_max"], "max_max": th["fp32_forced_rel_err_max_max"]}}
    viol = sum(r["n_violations"] for r in res["natural_fp32"])
    checks["g2_fp32_natural_selection"] = {
        "pass": viol == 0,
        "measured": {"n_pairs": sum(r["n"] for r in res["natural_fp32"]), "violations": viol,
                     "exempt": sum(r["n_exempt"] for r in res["natural_fp32"]),
                     "exempt_gaps": [g for r in res["natural_fp32"] for g in r["exempt_gaps"]][:20]},
        "threshold": {"near_tie_gap": th["fp32_selection_near_tie_gap"]}}
    checks["g2_embedding"] = {"pass": res["embedding"]["exact"], "measured": res["embedding"], "threshold": "exact"}
    checks["g2_head"] = {"pass": res["head"]["max_abs_dlogit"] <= th["head_fp32_logit_maxabs"],
                         "measured": res["head"], "threshold": {"max_abs_dlogit": th["head_fp32_logit_maxabs"]}}
    if emu is not None and res["layers_bf16"]:
        el = emu["layers"]
        ok, bad = _bf16_rule(res["layers_bf16"], el, th)
        checks["g2_bf16_forced_branch"] = {"pass": ok, "measured": {"failing": bad[:20]},
                                           "threshold": {k: th[k] for k in ("bf16_forced_median_factor_vs_emu",
                                                                            "bf16_forced_p99_floor", "bf16_forced_p99_factor_vs_emu")}}
        n = sum(r["n"] for r in res["natural_bf16"])
        frac = sum(r["n_disagree"] for r in res["natural_bf16"]) / max(n, 1)
        emu_frac = float(np.mean([l["natural_disagree_frac"] for l in el]))
        lim = max(th["bf16_selection_disagree_floor"], th["bf16_selection_disagree_factor_vs_emu"] * emu_frac)
        beyond = sum(r["n_beyond_6sigma"] for r in res["natural_bf16"])
        checks["g2_bf16_natural_selection"] = {
            "pass": frac <= lim and beyond == 0,
            "measured": {"disagree_frac": frac, "emu_disagree_frac": emu_frac, "limit": lim, "n_beyond_6sigma": beyond,
                         "max_gap_over_sigma": max((r["max_gap_over_sigma"] for r in res["natural_bf16"]), default=0.0)},
            "threshold": {k: th[k] for k in ("bf16_selection_disagree_floor", "bf16_selection_disagree_factor_vs_emu",
                                             "bf16_selection_gap_sigma")}}
        ok9, bad9 = _bf16_rule(res["layers_t9"], el, th)
        checks["g2_t9_buckets"] = {"pass": ok9, "measured": {"failing": bad9[:20], "buckets": th["t9_buckets"]},
                                   "threshold": "the bf16-mode forced row, against emu's T1-T8 statistics"}
        if "router_margin" in res:
            rm = res["router_margin"]
            checks["g2_router_margin"] = {"pass": not (rm["diff_p99"] < th["router_mutant_block_if_diff_p99_below"]),
                                          "measured": rm, "threshold": {"block_if_diff_p99_below": th["router_mutant_block_if_diff_p99_below"]},
                                          "descriptive_unless_blocking": True}
        for arm, g in res["layers_g2q"].items():
            if "capability_missing" in g:
                continue
            okq, badq = _bf16_rule(g["layers"], el, th)
            okq = okq and g["embedding_max_rel"] <= th["g2q_embedding_rel_err_max"] and g["head_rel_l2_max"] <= th["g2q_head_rel_l2_max"]
            checks[f"g2q_{arm}"] = {"pass": okq, "measured": {"failing": badq[:20], "embedding_max_rel": g["embedding_max_rel"],
                                                              "head_rel_l2_max": g["head_rel_l2_max"]},
                                    "threshold": {"layers": "as g2_bf16_forced_branch", "embedding_rel_err_max": th["g2q_embedding_rel_err_max"],
                                                  "head_rel_l2_max": th["g2q_head_rel_l2_max"]}}
    rwm = res["real_weight_mutants"]
    if rwm["mutants"]:
        undetected = [n for n, m in rwm["mutants"].items() if not m["detected"]]
        checks["g2_real_weight_mutants"] = {"pass": not undetected,
                                            "measured": {"layers": rwm["layers"], "undetected": undetected,
                                                         "detected": sorted(n for n, m in rwm["mutants"].items() if m["detected"])},
                                            "threshold": "each of mutants 1-8, 12-15 fails some fp32 check on >= 1 layer"}
    return checks


def layer_csv_rows(res: dict, emu: dict | None) -> list[dict]:
    """Rows of gate_<UTC>_layers.csv."""
    rows = []
    for mode, key in (("fp32_forced", "layers_fp32"), ("bf16_forced", "layers_bf16"), ("t9_bf16_forced", "layers_t9")):
        for r in res.get(key, []):
            e = emu["layers"][r["layer"]][r["branch"]] if emu else {}
            rows.append({"mode": mode, "layer": r["layer"], "branch": r["branch"], "bucket": "-".join(map(str, r.get("bucket", []))),
                         "median": r["median"], "p99": r["p99"], "max": r["max"],
                         "emu_median": e.get("median"), "emu_p99": e.get("p99"), "pass": r.get("pass")})
    for arm, g in res.get("layers_g2q", {}).items():
        for r in g.get("layers", []):
            rows.append({"mode": f"g2q_{arm}", "layer": r["layer"], "branch": r["branch"], "bucket": "",
                         "median": r["median"], "p99": r["p99"], "max": r["max"], "emu_median": None, "emu_p99": None,
                         "pass": r.get("pass")})
    return rows
