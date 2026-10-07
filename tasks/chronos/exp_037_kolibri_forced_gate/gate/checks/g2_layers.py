# SPDX-License-Identifier: MIT
"""G2: per layer, real weights (HYPOTHESIS Phase 0 G2; BUILD_SPEC §5.3 g2_layers.py).

Port layer l is fed the reference input h_l (all layers). Errors are measured
on the branch outputs before the residual add, each relative to its own
reference norm (item 29): e = ||r_port - r_ref|| / ||r_ref|| per token.

  fp32, forced     per layer and branch: median, p99 and max of e;
                   embedding equality; final norm + head max |dlogit|
  fp32, natural    top-6 set disagreements with R1, each exempt where R1's
                   6th-7th biased gap < the registered near-tie gap (the
                   exempt and violating gaps reported)
  bf16, forced     per layer and branch: median, p99 and max of e
  bf16, natural    per layer: the disagreement count and sigma_port,l = RMS of
                   port - R1 router logits over T1-T8, and every disagreeing
                   pair's z_port = R1's gap / sigma_port,l (DESIGN §3.5); the
                   gap >= 6 sigma count (descriptive)
  router margin    bf16-router mutant 10 vs the port, a paired token bootstrap
                   of agree_port - agree_mutant (p01, p99); mutant 10's own
                   disagreement fraction and maximum z (descriptive, §3.5)
  G2q              the quantised layer of each converted directory vs the
                   reference on that layer's dequantised weights, forced;
                   embedding max relative error; head relative L2
  T9               bf16 forced, bucketed by position
  mutants          port mutants 1-8 and 12-15 through the fp32 harness on
                   layers {0, 3, 4, 49}: their fp32 branch statistics,
                   selection violations and embedding equality
  flagged pair     layer 20, T4 position 19 (exp_036's 6.078 sigma pair):
                   whether the port and the emulation disagree there, z_port,
                   z_emu, r_t = sigma_t(port) / sigma_t(emu) (DESIGN §3.5)

"Forced" puts the port on the reference's expert ids with the weights from its
own fp32 logits (item 12; gate/harness.py RouteTap). One layer is resident at
a time. PortLayers.run keeps exp_036's single call over all 8 sequences
(T1-T8, 8 x 1,536 x 6 = 73,728 rows in one expert call, a judged G0k shape;
DESIGN §3.4, decision F2); T9 runs through the port's cache in chunks.

exp_037 (DESIGN §3.4-§3.5; build task W5a): this module only measures. The
registered rules (exp_036's evaluate(), l.246-322) and the per-pair leg (p)
are gate/rules.py's g2_checks and g2_row_verdicts; layer_csv_rows takes
rules.py's per-row verdicts. Nothing here emits "pass" or any verdict.
"""

from __future__ import annotations

import numpy as np

from gate import common
from gate.checks.ref_pass import (CapabilityMissing, Dump, RefRunner, flagged_site, natural_vs_r1, segments_of,
                                  set_disagree, z_record)

BRANCHES = ("attn", "moe")


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


def bf16_selection(port: dict, ref_top6, ref_gap, ref_logits, at_row: int | None = None) -> dict:
    """The port's natural selection against R1's in one layer, keeping the
    per-pair z (ref_pass.natural_vs_r1): {"n", "n_disagree", "sigma"
    (sigma_port,l), "max_gap_over_sigma" (= max z_port of the layer)} and
    the arrays "_dis", "_rows", "_z", "_gaps" (and "_at" with at_row)."""
    return natural_vs_r1(port["ids"], ref_top6, ref_gap, port["router_logits"], ref_logits, at_row=at_row)


def flagged_pair_report(site: dict, port_at: dict | None, emu: dict | None) -> dict:
    """The flagged pair (DESIGN §3.5 "Descriptive"): the port's side from its
    bf16 natural run at the site's layer, the emulation's from emu_stats'
    "flagged_pair", and r_t = sigma_t(port) / sigma_t(emu) at that token."""
    out = {k: site[k] for k in ("layer", "text", "pos", "row", "stand_in")}
    if port_at is not None:
        out.update({"port_disagrees": port_at["disagrees"], "gap": port_at["gap"], "z_port": port_at["z"],
                    "sigma_l_port": port_at["sigma_l"], "sigma_t_port": port_at["sigma_t"],
                    "ref_top6": port_at["ref_top6"], "port_top6": port_at["top6"],
                    "port_dropped": port_at["dropped"], "port_added": port_at["added"]})
    ef = (emu or {}).get("flagged_pair")
    if ef is None:
        out["emu"] = None
        return out
    if any(ef.get(k) != site[k] for k in ("layer", "text", "pos", "row")):
        raise ValueError(f"flagged pair: the emulation statistics are for {[ef.get(k) for k in ('layer', 'text', 'pos')]}, "
                         f"the port run for {[site[k] for k in ('layer', 'text', 'pos')]}")
    out.update({"emu_disagrees": ef["emu_disagrees"], "z_emu": ef["z_emu"], "sigma_l_emu": ef["sigma_l_emu"],
                "sigma_t_emu": ef["sigma_t_emu"], "emu_top6": ef["emu_top6"],
                "emu_dropped": ef["emu_dropped"], "emu_added": ef["emu_added"]})
    if port_at is not None:
        out["r_t"] = port_at["sigma_t"] / ef["sigma_t_emu"] if ef["sigma_t_emu"] > 0 else None
        out["same_swap"] = (port_at["dropped"] == ef["emu_dropped"] and port_at["added"] == ef["emu_added"])
    return out


def run(ctx, dump: Dump, emu: dict | None, textset, log=print) -> dict:
    """Phase 3. ctx: bf16_dir, arm_dir(arm), arms, mutant (tiny tests only),
    thresholds, want(check) -> bool. emu: ref_pass.emu_stats' result (or
    None: the bf16 rows are not run)."""
    from gate import harness, port_mutants

    th = ctx.thresholds["G2"]
    prof = textset.profile
    L, T9 = prof.text_len, prof.t9_len
    n8 = 8 * L
    s8, s9 = _rows(n8, L, T9)
    n_layers = int(dump.record()["num_layers"])
    k = int(np.asarray(dump.layer(0, "top6")).shape[-1])
    lengths8 = [L] * 8
    site = flagged_site(n_layers, L)
    out = {"n_layers": n_layers, "layers_fp32": [], "layers_bf16": [], "layers_t9": [], "layers_g2q": {},
           "natural_fp32": [], "natural_bf16": [], "notes": [], "flagged_pair": None}
    want_bf16 = emu is not None
    if not want_bf16:
        out["notes"].append("bf16 rows not run: no bf16-emulation calibration (reference capability missing)")

    p32 = harness.PortLayers(ctx.bf16_dir, "fp32", mutant=ctx.mutant)
    p16 = harness.PortLayers(ctx.bf16_dir, "bf16", mutant=ctx.mutant) if want_bf16 else None
    p_r10 = (harness.PortLayers(ctx.bf16_dir, "bf16", mutant="bf16_router_logits")
             if want_bf16 and ctx.mutant is None else None)
    agree_port = np.zeros(n8)
    agree_mut = np.zeros(n8)
    r10_layers = []

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

        # fp32 (one call over T1-T8, as registered)
        layer = p32.build_layer(i)
        f = p32.run(layer, i, h8, force_ids=top8)
        st = branch_stats(f, ra8, rm8)
        for br in BRANCHES:
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
            for br in BRANCHES:
                out["layers_bf16"].append({"layer": i, "branch": br, **stb[br]})
            nb = p16.run(layer, i, h8)
            sb = bf16_selection(nb, top_all[s8], gap8, lg8, at_row=site["row"] if i == site["layer"] else None)
            out["natural_bf16"].append({"layer": i, **z_record(sb, L, th["bf16_selection_gap_sigma"])})
            if "_at" in sb:
                out["flagged_pair"] = flagged_pair_report(site, sb["_at"], emu)
            agree_port += ~sb["_dis"]
            # T9, bucketed, forced, through the port's own cache in chunks.
            h9 = np.array(h_all[s9]).reshape(1, T9, H)
            top9 = top_all[s9].reshape(1, T9, k)
            f9 = p16.run(layer, i, h9, force_ids=top9, chunk=prof.prefill_chunk)
            ra9, rm9 = np.array(dump.layer(i, "r_attn")[s9]), np.array(dump.layer(i, "r_moe")[s9])
            st9 = branch_stats(f9, ra9, rm9)
            for a, b in prof.t9_buckets:
                for br in BRANCHES:
                    out["layers_t9"].append({"layer": i, "branch": br, "bucket": [a, b], **common.stats(st9["_e"][br][a:b])})
            del layer
            if p_r10 is not None:
                lm = p_r10.build_layer(i)
                nm = p_r10.run(lm, i, h8)
                s10 = bf16_selection(nm, top_all[s8], gap8, lg8)
                agree_mut += ~s10["_dis"]
                r10_layers.append({"layer": i, **{key: v for key, v in z_record(s10, L, th["bf16_selection_gap_sigma"]).items()
                                                  if key != "z_pairs"}})
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
            for br in BRANCHES:
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
                m["layers"][i] = {"attn": fm["attn"], "moe": fm["moe"], "selection_violations": sm["n_violations"]}
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
        out["mutant10_natural_bf16"] = _mutant10_summary(r10_layers)

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


def _mutant10_summary(rows: list) -> dict:
    """Mutant 10 (bf16 router logits) in the router-margin run: its natural
    selection against R1, as the port's is measured (DESIGN §3.5: M for mutant
    10 and its disagreement fraction, descriptive)."""
    n = sum(r["n"] for r in rows)
    n_dis = sum(r["n_disagree"] for r in rows)
    top = max(rows, key=lambda r: r["max_gap_over_sigma"]) if rows else None
    return {"n": n, "n_disagree": n_dis, "disagree_frac": n_dis / n if n else None,
            "M": top["max_gap_over_sigma"] if top else 0.0,
            "M_at": ({"layer": top["layer"], **top["max_at"]} if top and top["max_at"] else None),
            "layers": rows}


# ---------------------------------------------------------------------------
# The layers CSV (verdict columns from gate/rules.py g2_row_verdicts)
# ---------------------------------------------------------------------------

CSV_MODES = (("fp32_forced", "layers_fp32"), ("bf16_forced", "layers_bf16"), ("t9_bf16_forced", "layers_t9"))
CSV_COLUMNS = ("mode", "layer", "branch", "bucket", "median", "p99", "max", "emu_median", "emu_p99",
               "limit_median", "limit_p99", "limit_max", "pass")


def layer_csv_rows(res: dict, emu: dict | None, row_verdicts: dict) -> list[dict]:
    """Rows of gate_<UTC>_layers.csv: each measured row with its emulation
    statistics and its verdict from rules.g2_row_verdicts(res, emu, th2)
    (keyed by mode, aligned with res's row lists). A mode whose verdict list
    is missing or misaligned is a gate bug (ValueError)."""
    rows = []

    def emit(mode: str, recs: list) -> None:
        if not recs:
            return
        verdicts = row_verdicts.get(mode)
        if verdicts is None or len(verdicts) != len(recs):
            raise ValueError(f"layers CSV: {len(recs)} {mode} rows but "
                             f"{'no' if verdicts is None else len(verdicts)} verdicts from rules.g2_row_verdicts")
        for r, v in zip(recs, verdicts):
            e = emu["layers"][r["layer"]][r["branch"]] if emu else {}
            rows.append({"mode": mode, "layer": r["layer"], "branch": r["branch"],
                         "bucket": "-".join(map(str, r.get("bucket", []))),
                         "median": r["median"], "p99": r["p99"], "max": r["max"],
                         "emu_median": e.get("median"), "emu_p99": e.get("p99"),
                         "limit_median": v.get("limit_median"), "limit_p99": v.get("limit_p99"),
                         "limit_max": v.get("limit_max"), "pass": v.get("pass")})

    for mode, key in CSV_MODES:
        emit(mode, res.get(key, []))
    for arm, g in (res.get("layers_g2q") or {}).items():
        emit(f"g2q_{arm}", g.get("layers", []))
    return rows
