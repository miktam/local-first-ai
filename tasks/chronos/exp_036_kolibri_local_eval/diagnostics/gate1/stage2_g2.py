# SPDX-License-Identifier: MIT
"""Stage 2 (DIAGNOSIS §4): G2 bf16 natural selection. About 30 minutes on the mbp:
the gate's harness one layer at a time (port, bf16 as stored) and the numpy
reference's bf16 emulation (emu), both fed the fp32 reference's h_l from the
gate's dump. No model is loaded whole; peak memory is about one layer plus the
dump rows of that layer.

Measured (reproducing layer 20's gate statistic first: 225 disagreements,
sigma 1.161e-3, max 6.078):

  flip     the layer-20 disagreement with the largest gap/sigma_l: text,
           position and class; e_attn rank in layers 16-20; sigma_t(port) and
           sigma_t(emu) (RMS over the 384 experts of the router-logit error at
           that token); r_t = sigma_t(port)/sigma_t(emu) with layer 20's p50/p99
           of r over all T1-T8 tokens; gap, gap/sigma_t(emu); whether emu also
           disagrees with the reference's selection at that (token, layer)
  layers   all 50 layers, emu natural on T1-T8: sigma_emu,l, emu's max
           gap/sigma_emu,l (M_emu is the max over layers), emu's count >= 6 sigma
  tails    attention 8 and 29, MoE 13 (T1-T8 rows, forced): (text, position) of
           the port's max and of emu's max, emu's error at the port's max token;
           MoE 30 on T9 (the 11.99 tail) by bucket
  T9       natural bf16 selection disagreement by bucket, port (through its own
           cache in prefill chunks) and emu (fed the fp32 h_l), all 50 layers

The output's rule_inputs are FROZEN_RULES.md §2's stage-2 keys (MoE 30's tail
on all T9 rows); stage 5 alone evaluates the frozen G2 rule (§4) on them, with
stage 3's top10_kl_positions for sign 4. Everything else is descriptive.

usage (kit root, env sourced): caffeinate -i "$PY" diagnostics/gate1/stage2_g2.py
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import time  # noqa: E402

import numpy as np  # noqa: E402

import diaglib as D  # noqa: E402
from gate import common  # noqa: E402
from gate.checks.ref_pass import RefRunner, segments_of, set_disagree  # noqa: E402


def ints(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def rms_rows(d: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(d * d, axis=-1))


def pct_rank(v: np.ndarray, x: float) -> float:
    return float((v < x).mean() * 100)


def where_max(e: np.ndarray, L: int, offset_text: str | None = None) -> dict:
    r = int(np.argmax(e))
    tid, pos = (offset_text, r) if offset_text else D.text_pos(r, L)
    return {"text": tid, "pos": pos, "value": float(e[r]), "row": r}


def main() -> int:
    ap = D.parser("stage2_g2", __doc__)
    ap.add_argument("--flip-layer", type=int, default=20)
    ap.add_argument("--rank-layers", default="16-20")
    ap.add_argument("--attn-tails", default="8,29")
    ap.add_argument("--moe-tails", default="13")
    ap.add_argument("--t9-moe-tail", type=int, default=30)
    ap.add_argument("--sigma-rule", type=float, default=None, help="default thresholds G2 bf16_selection_gap_sigma")
    args = ap.parse_args()
    run = D.Run("stage2_g2", args, PREC)
    from gate import harness

    ctx, ts, dump = run.ctx, run.textset(), run.dump()
    prof = ts.profile
    L, T9 = prof.text_len, prof.t9_len
    n8 = 8 * L
    n_layers = int(dump.record()["num_layers"])
    th2 = ctx.thresholds["G2"]
    sig_rule = args.sigma_rule or th2["bf16_selection_gap_sigma"]
    cfg = common.read_json(ctx.bf16_dir / "config.json")
    W, CH = int(cfg.get("sliding_window") or 0), prof.prefill_chunk
    if run.tiny and args.flip_layer >= n_layers:     # smoke tests: the same roles on a 10-layer checkpoint
        args.flip_layer, args.rank_layers, args.attn_tails, args.moe_tails, args.t9_moe_tail = \
            3, "0-3", "1,4", "2", n_layers - 1
    flip_layer = args.flip_layer
    rank_layers = [i for i in ints(args.rank_layers) if i < n_layers]
    attn_tails = [i for i in ints(args.attn_tails) if i < n_layers]
    moe_tails = [i for i in ints(args.moe_tails) if i < n_layers]
    t9_tail = args.t9_moe_tail if args.t9_moe_tail < n_layers else None
    port_t18_layers = sorted(set(rank_layers) | {flip_layer} | set(attn_tails) | set(moe_tails))
    buckets = [tuple(b) for b in prof.t9_buckets]
    seg8 = segments_of([L] * 8)
    seg_all = segments_of([L] * 8 + [T9])

    p16 = harness.PortLayers(ctx.bf16_dir, "bf16")
    emu = RefRunner(ctx.bf16_dir, "emu")
    rec = {"rule_sigma": sig_rule, "layers": {"flip": flip_layer, "rank": rank_layers, "attn_tails": attn_tails,
                                               "moe_tails": moe_tails, "t9_moe_tail": t9_tail},
           "window": W, "chunk": CH, "text_len": L, "t9_len": T9, "t9_buckets": [list(b) for b in buckets]}
    work = {}
    emu_layers, t9_rows = [], []
    e_attn_port = {}      # layer -> per-token e_attn (T1-T8), port bf16
    tails = {}
    flip = None
    t_all = time.time()
    for i in range(n_layers):
        t0 = time.time()
        h_all = np.array(dump.layer(i, "h_in"))                     # [n8 + T9, H] fp32
        H = h_all.shape[-1]
        top_all = np.array(dump.layer(i, "top6"))
        k = top_all.shape[-1]
        gap_all = np.array(dump.layer(i, "top6_gap"))
        lg_all = np.array(dump.layer(i, "router_logits")).astype(np.float64)

        # emu natural on T1-T8 and T9 (one call, independent sequences from position 0)
        en = emu.branches(i, h_all, seg_all)
        el = np.asarray(en["logits"], dtype=np.float64)
        etop = np.asarray(en["top6"])
        edis = set_disagree(etop, top_all)
        ed8 = el[:n8] - lg_all[:n8]
        esig = float(np.sqrt(np.mean(ed8 * ed8)))
        esig_t = rms_rows(ed8)
        eg = gap_all[:n8][edis[:n8]] / esig
        lrow = {"layer": i, "type": cfg["layer_types"][i], "emu_sigma": esig,
                "emu_n_disagree": int(edis[:n8].sum()), "emu_disagree_frac": float(edis[:n8].mean()),
                "emu_max_gap_over_sigma": float(eg.max()) if eg.size else 0.0,
                "emu_n_beyond_rule": int((eg >= sig_rule).sum())}
        lrow["emu_max_at"] = (dict(zip(("text", "pos"), D.text_pos(int(np.flatnonzero(edis[:n8])[int(np.argmax(eg))]), L)))
                              if eg.size else None)
        t9e = {}
        for a, b in buckets:
            t9e[f"{a}-{b}"] = int(edis[n8 + a:n8 + b].sum())
        lrow["emu_t9_disagree_by_bucket"] = t9e

        # emu forced at the tail layers (T1-T8) and at the T9 MoE tail
        if i in attn_tails or i in moe_tails:
            ef = emu.branches(i, h_all[:n8], seg8, force_ids=top_all[:n8])
            br = "attn" if i in attn_tails else "moe"
            ref_r = np.array(dump.layer(i, f"r_{br}")[:n8])
            tails.setdefault(i, {})["emu_e"] = common.rel_err_rows(ef[f"r_{br}"], ref_r)
            tails[i]["branch"] = br
        if t9_tail is not None and i == t9_tail:
            ef9 = emu.branches(i, h_all[n8:], [(0, T9)], force_ids=top_all[n8:])
            ref_r = np.array(dump.layer(i, "r_moe")[n8:])
            tails.setdefault(("T9", i), {})["emu_e"] = common.rel_err_rows(ef9["r_moe"], ref_r)
            tails[("T9", i)]["branch"] = "moe"
        del en

        # port, bf16 as stored, one layer
        layer = p16.build_layer(i)
        h8 = h_all[:n8].reshape(8, L, H)
        if i in port_t18_layers:
            nb = p16.run(layer, i, h8)                                  # natural, T1-T8
            ra8 = np.array(dump.layer(i, "r_attn")[:n8])
            e_attn_port[i] = common.rel_err_rows(nb["r_attn"].reshape(n8, H), ra8)
            if i == flip_layer:
                pl = nb["router_logits"].reshape(n8, -1).astype(np.float64)
                pid = nb["ids"].reshape(n8, k)
                dis = set_disagree(pid, top_all[:n8])
                diff = pl - lg_all[:n8]
                sigma = float(np.sqrt(np.mean(diff * diff)))
                sig_t = rms_rows(diff)
                rows = np.flatnonzero(dis)
                ratios = gap_all[:n8][rows] / sigma
                r_dist = sig_t / np.maximum(esig_t, 1e-300)
                flip = {"layer": i, "n_disagree": int(dis.sum()), "sigma_l": sigma,
                        "max_gap_over_sigma": float(ratios.max()) if ratios.size else 0.0,
                        "n_beyond_rule": int((ratios >= sig_rule).sum()),
                        "r_layer_p50": float(np.quantile(r_dist, 0.5)), "r_layer_p99": float(np.quantile(r_dist, 0.99)),
                        "emu_sigma_l": esig, "emu_n_disagree": int(edis[:n8].sum()),
                        "both_disagree": int((dis & edis[:n8]).sum())}
                if rows.size:
                    r = int(rows[int(np.argmax(ratios))])
                    tid, pos = D.text_pos(r, L)
                    pref = set(top_all[r].tolist())
                    eset = set(etop[r].tolist())
                    flip_tok = {"row": r, "text": tid, "pos": pos, "pos_class": D.pos_class(pos, W, CH),
                                "gap": float(gap_all[r]), "gap_over_sigma_l": float(gap_all[r] / sigma),
                                "sigma_t_port": float(sig_t[r]), "sigma_t_emu": float(esig_t[r]),
                                "r_t": float(sig_t[r] / max(esig_t[r], 1e-300)),
                                "gap_over_sigma_t_emu": float(gap_all[r] / max(esig_t[r], 1e-300)),
                                "gap_over_sigma_t_port": float(gap_all[r] / max(sig_t[r], 1e-300)),
                                "emu_disagrees_same_token_layer": bool(edis[r]),
                                "port_dropped": sorted(pref - set(pid[r].tolist())),
                                "port_added": sorted(set(pid[r].tolist()) - pref),
                                "emu_dropped": sorted(pref - eset), "emu_added": sorted(eset - pref)}
                    flip_tok["emu_same_swap"] = (flip_tok["emu_dropped"] == flip_tok["port_dropped"]
                                                 and flip_tok["emu_added"] == flip_tok["port_added"])
                    flip["token"] = flip_tok
                    work["flip_token_id"] = int(ts.ids[tid][pos])
                work["layer_flip_r_dist_quantiles"] = {str(q): float(np.quantile(r_dist, q))
                                                       for q in (0.5, 0.9, 0.99, 0.999, 1.0)}
        for li in (attn_tails + moe_tails):
            if li == i:
                br = tails[i]["branch"]
                fb = p16.run(layer, i, h8, force_ids=top_all[:n8].reshape(8, L, k))
                ref_r = np.array(dump.layer(i, f"r_{br}")[:n8])
                tails[i]["port_e"] = common.rel_err_rows(fb[f"r_{br}"].reshape(n8, H), ref_r)
        # port natural on T9 through its own cache; forced at the T9 MoE tail
        h9 = h_all[n8:].reshape(1, T9, H)
        nb9 = p16.run(layer, i, h9, chunk=CH)
        pdis9 = set_disagree(nb9["ids"].reshape(T9, k), top_all[n8:])
        t9p = {f"{a}-{b}": int(pdis9[a:b].sum()) for a, b in buckets}
        lrow["port_t9_disagree_by_bucket"] = t9p
        if t9_tail is not None and i == t9_tail:
            f9 = p16.run(layer, i, h9, force_ids=top_all[n8:].reshape(1, T9, k), chunk=CH)
            ref_r = np.array(dump.layer(i, "r_moe")[n8:])
            tails[("T9", i)]["port_e"] = common.rel_err_rows(f9["r_moe"].reshape(T9, H), ref_r)
        del layer
        emu_layers.append(lrow)
        D.log(f"[stage2] layer {i + 1}/{n_layers}: emu sigma {esig:.3e} max {lrow['emu_max_gap_over_sigma']:.2f} "
              f"beyond {lrow['emu_n_beyond_rule']} | T9 dis port {sum(t9p.values())} emu {sum(t9e.values())} "
              f"({time.time() - t0:.0f}s)")
        run.release()

    # ---- assemble -----------------------------------------------------------------
    M_emu = max(r["emu_max_gap_over_sigma"] for r in emu_layers)
    rec["emu_layers"] = emu_layers
    rec["M_emu"] = M_emu
    rec["M_emu_layer"] = int(max(emu_layers, key=lambda r: r["emu_max_gap_over_sigma"])["layer"])
    rec["emu_n_beyond_rule_total"] = int(sum(r["emu_n_beyond_rule"] for r in emu_layers))
    if flip is not None:
        rec["flip"] = flip
        if "token" in flip:
            r = flip["token"]["row"]
            ranks = {str(li): pct_rank(e_attn_port[li], e_attn_port[li][r]) for li in rank_layers if li in e_attn_port}
            flip["token"]["e_attn_rank_pct"] = ranks
            flip["token"]["e_attn_top1pct_any"] = any(v >= 99.0 for v in ranks.values())
            flip["token"].pop("row")
        if not run.tiny:
            g = D.RECORD["g2_layer20"]
            rec["reproduction"] = {"n_disagree": [g["n_disagree"], flip["n_disagree"]], "sigma": [g["sigma"], flip["sigma_l"]],
                                   "max_gap_over_sigma": [g["max_gap_over_sigma"], flip["max_gap_over_sigma"]],
                                   "exact": (g["n_disagree"] == flip["n_disagree"] and g["sigma"] == flip["sigma_l"]
                                             and g["max_gap_over_sigma"] == flip["max_gap_over_sigma"]),
                                   "max_gap_over_sigma_spread": abs(g["max_gap_over_sigma"] - flip["max_gap_over_sigma"])}
    tail_out = []
    for key, t in tails.items():
        if "port_e" not in t or "emu_e" not in t:
            continue
        if isinstance(key, tuple):
            segs = {"T9": (0, T9)} | {f"T9:{a}-{b}": (a, b) for a, b in buckets}
            for name, (a, b) in segs.items():
                pe, ee = t["port_e"][a:b], t["emu_e"][a:b]
                pm, em = int(np.argmax(pe)), int(np.argmax(ee))
                tail_out.append({"layer": key[1], "branch": "moe", "rows": name,
                                 "port_max": float(pe[pm]), "port_max_at": {"text": "T9", "pos": a + pm},
                                 "emu_max": float(ee[em]), "emu_max_at": {"text": "T9", "pos": a + em},
                                 "emu_e_at_port_max": float(ee[pm]),
                                 "port_max_over_emu_max": float(pe[pm] / max(ee[em], 1e-300)),
                                 "port_max_over_p99": float(pe[pm] / np.quantile(pe, 0.99)),
                                 "emu_max_over_p99": float(ee[em] / np.quantile(ee, 0.99))})
        else:
            pe, ee = t["port_e"], t["emu_e"]
            pm, em = int(np.argmax(pe)), int(np.argmax(ee))
            ptx, ppos = D.text_pos(pm, L)
            etx, epos = D.text_pos(em, L)
            tail_out.append({"layer": key, "branch": t["branch"], "rows": "T1-8",
                             "port_max": float(pe[pm]), "port_max_at": {"text": ptx, "pos": ppos},
                             "emu_max": float(ee[em]), "emu_max_at": {"text": etx, "pos": epos},
                             "emu_e_at_port_max": float(ee[pm]),
                             "port_max_over_emu_max": float(pe[pm] / max(ee[em], 1e-300)),
                             "port_max_over_p99": float(pe[pm] / np.quantile(pe, 0.99)),
                             "emu_max_over_p99": float(ee[em] / np.quantile(ee, 0.99))})
    rec["tails"] = tail_out
    t9 = {}
    for a, b in buckets:
        key = f"{a}-{b}"
        n = (b - a) * n_layers
        pf = sum(r["port_t9_disagree_by_bucket"][key] for r in emu_layers) / n
        ef = sum(r["emu_t9_disagree_by_bucket"][key] for r in emu_layers) / n
        t9[key] = {"port_disagree_frac": pf, "emu_disagree_frac": ef, "ratio": pf / ef if ef > 0 else None,
                   "pooled_over_layers": n_layers}
    rec["t9_natural_by_bucket"] = t9

    # ---- rule inputs (FROZEN_RULES.md §2, stage 2); stage 5 alone evaluates the rule --------
    flip_ = rec.get("flip") or {}
    tok = flip_.get("token") or {}
    ri = {"reproduces_gate": (rec.get("reproduction") or {}).get("exact"),
          "port_max_gap_over_sigma": flip_.get("max_gap_over_sigma"),
          "port_n_disagree_layer": flip_.get("n_disagree"),
          "flip_text": tok.get("text"), "flip_position": tok.get("pos"), "flip_position_class": tok.get("pos_class"),
          "flip_e_attn_max_pct_16_20": max(tok["e_attn_rank_pct"].values()) if tok.get("e_attn_rank_pct") else None,
          "flip_sigma_t_port": tok.get("sigma_t_port"), "flip_sigma_t_emu": tok.get("sigma_t_emu"),
          "flip_r_t": tok.get("r_t"), "r_p50_layer20": flip_.get("r_layer_p50"), "r_p99_layer20": flip_.get("r_layer_p99"),
          "flip_gap": tok.get("gap"), "flip_gap_over_sigma_t_emu": tok.get("gap_over_sigma_t_emu"),
          "flip_emu_flips_same": tok.get("emu_disagrees_same_token_layer"),
          "M_emu": M_emu, "emu_n_beyond_6sigma": rec["emu_n_beyond_rule_total"],
          "tails": {f"{'att' if t['branch'] == 'attn' else 'moe'}{t['layer']}":
                    {"port_max_over_emu_max": t["port_max_over_emu_max"], "port_max_text": t["port_max_at"]["text"],
                     "port_max_position": t["port_max_at"]["pos"], "rows": t["rows"]}
                    for t in tail_out if t["rows"] in ("T1-8", "T9")},
          "t9_disagree_frac_port": {b: v["port_disagree_frac"] for b, v in t9.items()},
          "t9_disagree_frac_emu": {b: v["emu_disagree_frac"] for b, v in t9.items()}}
    rec = {"rule_inputs": ri, **rec}
    rec["wall_s_layers"] = time.time() - t_all
    run.finish(rec, work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
