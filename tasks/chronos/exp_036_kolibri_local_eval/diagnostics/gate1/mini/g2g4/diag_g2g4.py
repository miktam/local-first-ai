# SPDX-License-Identifier: MIT
"""exp_036 gate diagnostic for the 20261005T050112Z K8 failure: G2 bf16
natural-selection outlier and G4 long-context backstop. Read-only on the kit:
imports gate/, writes only under --out. Not a gate check; nothing here is a
verdict.

  g2       per-disagreement detail of the bf16 natural-selection check for
           --layers (default 20), reproducing the gate's statistic first;
           with --emu, the same statistic for the bf16-emulated reference
           (what a correct bf16 implementation does on the same input).
  g4       per-position KL(ref || port), reference lead, top-1 agreement,
           dNLL over T1-T8 and T9, teacher-forced exactly as G4 does
           (runner.generate.iter_teacher_force_logprobs, chunk 2048), for
           --arm K8|K4 and --act bf16|fp32 (fp32: model.set_dtype(float32)
           on the converted model, the gate's own tiny-run path). With
           --ref-npz, the reference is that npz's logits (e.g. the numpy
           reference on the dequantised K8 weights) instead of the fp32 dump.
  analyse  summaries of one or more g4 npz files: G4's own numbers (as a
           reproduction check), T9 by 512-token bin, by document, around
           every 2048-chunk boundary, and the five content-matched controls
           (T1, T4, T3, T5, T6 standalone vs the same 1,536 tokens inside T9).
  t9ids    write T9's ids (and T1-T8) as a reference/kolibri_ref.py --ids-file.
  refcmp   a reference-mode npz (kolibri_ref --out, e.g. --emulate-bf16
           --dequant-dir K8) against the fp32 dump, in the g4 npz format.

Usage (mbp, kit root, "$PY" as in the RUNBOOK):
  "$PY" diag_g2g4.py g2 --layers 20 --emu --out $EXP036_WORK/diag
  "$PY" diag_g2g4.py g4 --arm K8 --act bf16 --out $EXP036_WORK/diag
  "$PY" diag_g2g4.py g4 --arm K8 --act fp32 --out $EXP036_WORK/diag
  "$PY" diag_g2g4.py analyse $EXP036_WORK/diag/g4_K8_bf16.npz $EXP036_WORK/diag/g4_K8_fp32.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path.cwd()
sys.path.insert(0, str(KIT))

from gate import common  # noqa: E402

ORDER8 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


def _ctx(tiny: str | None):
    from gate import run_gate

    th = common.load_thresholds()
    if tiny:
        root = Path(tiny).resolve()
        return run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=th, tiny=True,
                                    results_dir=root / "results", head_policy="quantised_head", subprocess_phases=False)
    return run_gate.GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(), thresholds=th,
                                subprocess_phases=False)


def _setup(tiny):
    from gate import run_gate
    from gate.checks import ref_pass

    ctx = _ctx(tiny)
    ts = run_gate.textset_for(ctx)
    dump = ref_pass.Dump(run_gate.dump_dir(ctx, ts))
    if not dump.complete:
        raise SystemExit(f"no complete fp32 dump at {common.redact_path(dump.dir)}")
    return ctx, ts, dump


def _t9_segments(ts) -> dict:
    """Document spans inside T9 and the offsets of the standalone texts in it,
    found by matching ids (no assumption about the separator)."""
    t9 = np.asarray(ts.t9)
    out = {}
    for tid in ("T1", "T4", "T3", "T5", "T6"):
        a = np.asarray(ts.ids[tid])
        L = a.size
        hits = [s for s in range(0, t9.size - L + 1) if t9[s] == a[0] and np.array_equal(t9[s:s + L], a)]
        if hits:
            out[tid] = int(hits[0])
    return out


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


def cmd_g2(args) -> int:
    import mlx.core as mx

    from gate import harness
    from gate.checks.ref_pass import RefRunner, segments_of, set_disagree

    ctx, ts, dump = _setup(args.tiny)
    prof = ts.profile
    L = prof.text_len
    n8 = 8 * L
    n_layers = int(dump.record()["num_layers"])
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    gate_rec = None
    if args.record:
        p3 = common.read_json(Path(args.record))
        gate_rec = {r["layer"]: r for r in p3["data"]["g2"]["natural_bf16"]}
    p16 = harness.PortLayers(ctx.bf16_dir, "bf16")
    emu = RefRunner(ctx.bf16_dir, "emu") if args.emu else None
    ids8 = np.concatenate([np.asarray(ts.ids[t]) for t in ORDER8])
    sig = ctx.thresholds["G2"]["bf16_selection_gap_sigma"]
    out = {"layers": [], "utc": common.utc_iso(), "sigma_rule": sig}
    for i in layers:
        t0 = time.time()
        h_all = dump.layer(i, "h_in")
        H = h_all.shape[-1]
        h8 = np.array(h_all[:n8]).reshape(8, L, H)
        top8 = np.array(dump.layer(i, "top6")[:n8])
        k = top8.shape[-1]
        gap8 = np.array(dump.layer(i, "top6_gap")[:n8])
        lg8 = np.array(dump.layer(i, "router_logits")[:n8]).astype(np.float64)
        ra8 = np.array(dump.layer(i, "r_attn")[:n8])
        layer = p16.build_layer(i)
        bias = np.array(layer.mlp.gate.expert_bias.astype(mx.float32)).astype(np.float64)
        nb = p16.run(layer, i, h8)
        del layer
        pl = nb["router_logits"].reshape(n8, -1).astype(np.float64)
        pid = nb["ids"].reshape(n8, k)
        dis = set_disagree(pid, top8)
        diff = pl - lg8
        sigma = float(np.sqrt(np.mean(diff * diff)))
        sig_t = np.sqrt(np.mean(diff * diff, axis=-1))
        gaps = gap8[dis]
        ratio = gaps / sigma
        rows = np.nonzero(dis)[0]
        e_attn = common.rel_err_rows(nb["r_attn"].reshape(n8, H), ra8)
        hn = np.linalg.norm(h8.reshape(n8, H), axis=-1)
        ref_b, port_b = lg8 + bias, pl + bias
        detail = []
        for r, g in sorted(zip(rows, gaps), key=lambda x: -x[1])[: args.top]:
            out_e = sorted(set(top8[r].tolist()) - set(pid[r].tolist()))  # ref's, dropped by the port
            in_e = sorted(set(pid[r].tolist()) - set(top8[r].tolist()))   # port's, not in ref
            pairs = []
            for eo in out_e:
                for ei in in_e:
                    ref_gap = ref_b[r, eo] - ref_b[r, ei]                  # > 0: ref prefers eo
                    d_pair = (port_b[r, ei] - port_b[r, eo]) + ref_gap    # port error that flipped it
                    pairs.append({"dropped": int(eo), "added": int(ei), "ref_pair_gap": float(ref_gap),
                                  "pair_err": float(d_pair), "pair_err_over_sigma_l": float(d_pair / sigma),
                                  "err_dropped": float(diff[r, eo]), "err_added": float(diff[r, ei]),
                                  "w_dropped": float(1 / (1 + np.exp(-lg8[r, eo]))),
                                  "w_added": float(1 / (1 + np.exp(-lg8[r, ei]))),
                                  "bias_dropped": float(bias[eo]), "bias_added": float(bias[ei])})
            rank = np.argsort(-ref_b[r])
            detail.append({"row": int(r), "text": ORDER8[r // L], "pos": int(r % L), "token_id": int(ids8[r]),
                           "prev_token_id": int(ids8[r - 1]) if r % L else None,
                           "gap": float(g), "gap_over_sigma_l": float(g / sigma), "sigma_t": float(sig_t[r]),
                           "sigma_t_over_sigma_l": float(sig_t[r] / sigma), "gap_over_sigma_t": float(g / sig_t[r]),
                           "sigma_t_rank_pct": float((sig_t < sig_t[r]).mean() * 100),
                           "h_norm_over_median": float(hn[r] / np.median(hn)), "e_attn": float(e_attn[r]),
                           "e_attn_rank_pct": float((e_attn < e_attn[r]).mean() * 100),
                           "ref_scores_5_to_8": [float(ref_b[r, e]) for e in rank[4:8]],
                           "pairs": pairs})
        rec = {"layer": i, "n": int(dis.size), "n_disagree": int(dis.sum()), "sigma": sigma,
               "max_gap_over_sigma": float(ratio.max()) if ratio.size else 0.0,
               "n_beyond": int((ratio >= sig).sum()),
               "gap_over_sigma_quantiles": {q: float(np.quantile(ratio, q)) for q in (0.5, 0.9, 0.99, 1.0)} if ratio.size else {},
               "gap_over_sigma_sorted_top50": sorted(map(float, ratio))[-50:],
               "sigma_t_quantiles_over_sigma_l": {q: float(np.quantile(sig_t / sigma, q)) for q in (0.01, 0.5, 0.99, 0.999, 1.0)},
               "top_disagreements": detail}
        if gate_rec and i in gate_rec:
            g = gate_rec[i]
            rec["reproduces_gate"] = {"n_disagree": [g["n_disagree"], rec["n_disagree"]], "sigma": [g["sigma"], sigma],
                                      "max_gap_over_sigma": [g["max_gap_over_sigma"], rec["max_gap_over_sigma"]]}
        if emu is not None:
            en = emu.branches(i, h8.reshape(n8, H), segments_of([L] * 8))
            el = np.asarray(en["logits"], dtype=np.float64)
            edis = set_disagree(np.asarray(en["top6"]), top8)
            ed = el - lg8
            esig = float(np.sqrt(np.mean(ed * ed)))
            er = gap8[edis] / esig
            esig_t = np.sqrt(np.mean(ed * ed, axis=-1))
            rec["emu"] = {"n_disagree": int(edis.sum()), "disagree_frac": float(edis.mean()), "sigma": esig,
                          "max_gap_over_sigma": float(er.max()) if er.size else 0.0, "n_beyond": int((er >= sig).sum()),
                          "gap_over_sigma_quantiles": {q: float(np.quantile(er, q)) for q in (0.5, 0.9, 0.99, 1.0)} if er.size else {},
                          "gap_over_sigma_sorted_top50": sorted(map(float, er))[-50:],
                          "sigma_t_quantiles_over_sigma_l": {q: float(np.quantile(esig_t / esig, q)) for q in (0.01, 0.5, 0.99, 0.999, 1.0)},
                          "port_vs_emu_router_rms": float(np.sqrt(np.mean((pl - el) ** 2))),
                          "both_disagree": int((dis & edis).sum())}
            # where the port's top disagreements sit in emu's own error scale
            for d in rec["top_disagreements"]:
                d["emu_disagrees_too"] = bool(edis[d["row"]])
                d["emu_sigma_t_over_emu_sigma_l"] = float(esig_t[d["row"]] / esig)
        rec["seconds"] = time.time() - t0
        out["layers"].append(rec)
        e = rec.get("emu", {})
        print(f"[diag g2] layer {i}: port dis {rec['n_disagree']} sigma {sigma:.3e} max {rec['max_gap_over_sigma']:.2f} "
              f"beyond {rec['n_beyond']} | emu dis {e.get('n_disagree')} sigma {e.get('sigma', float('nan')):.3e} "
              f"max {e.get('max_gap_over_sigma', float('nan')):.2f} beyond {e.get('n_beyond')}  ({rec['seconds']:.0f}s)",
              flush=True)
        if detail:
            d = detail[0]
            print(f"           top: {d['text']} pos {d['pos']} token {d['token_id']} gap/sigma_l {d['gap_over_sigma_l']:.2f} "
                  f"sigma_t/sigma_l {d['sigma_t_over_sigma_l']:.2f} (pct {d['sigma_t_rank_pct']:.1f}) "
                  f"gap/sigma_t {d['gap_over_sigma_t']:.2f} |h|/med {d['h_norm_over_median']:.2f}", flush=True)
    tot = lambda key, sub=None: sum((r[sub] if sub else r)[key] for r in out["layers"] if (sub is None or sub in r))
    out["summary"] = {"port_n_disagree": tot("n_disagree"), "port_n_beyond": tot("n_beyond"),
                      "port_max": max(r["max_gap_over_sigma"] for r in out["layers"])}
    if args.emu:
        out["summary"].update({"emu_n_disagree": tot("n_disagree", "emu"), "emu_n_beyond": tot("n_beyond", "emu"),
                               "emu_max": max(r["emu"]["max_gap_over_sigma"] for r in out["layers"])})
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = "all" if args.layers == "all" else args.layers.replace(",", "-")
    path = Path(args.out) / f"g2_layers_{tag}.json"
    common.write_json(path, out)
    print(json.dumps(out["summary"]), "->", common.redact_path(path))
    return 0


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------


def _per_position(model, ids, ref_rows, chunk, keep_lp=False):
    from gate.checks.g4_e2e import _compare
    from runner.generate import iter_teacher_force_logprobs

    ids = np.asarray(ids)
    cols = {k: np.empty(ids.size) for k in ("kl", "lead", "dnll", "nll_ref")}
    agree = np.zeros(ids.size, bool)
    top_ref = np.zeros(ids.size, np.int64)
    top_got = np.zeros(ids.size, np.int64)
    lps = []
    for a, block in iter_teacher_force_logprobs(model, list(map(int, ids)), chunk):
        b = a + block.shape[0]
        ref = np.asarray(ref_rows(a, b), dtype=np.float32)
        c = _compare(ref, block, ids[a + 1:b + 1])
        nxt = ids[a + 1:b + 1]
        lpr = common.log_softmax64(ref[:nxt.size])
        c["nll_ref"] = np.full(b - a, np.nan)
        c["nll_ref"][:nxt.size] = -lpr[np.arange(nxt.size), nxt]
        for k in cols:
            cols[k][a:b] = c[k]
        agree[a:b] = c["agree"]
        top_ref[a:b] = ref.argmax(-1)
        top_got[a:b] = block.argmax(-1)
        if keep_lp:
            lps.append(block)
    out = {**cols, "agree": agree, "top_ref": top_ref, "top_got": top_got}
    return out, (np.concatenate(lps) if keep_lp else None)


def cmd_g4(args) -> int:
    import mlx.core as mx

    from tools.precision import ensure_exact_fp32

    ensure_exact_fp32()
    ctx, ts, dump = _setup(args.tiny)
    prof = ts.profile
    L, n8 = prof.text_len, 8 * prof.text_len
    model, _, _ = common.load_port(ctx.arm_dir(args.arm))
    if args.act == "fp32":
        model.set_dtype(mx.float32)
    if args.ref_npz:
        z = np.load(args.ref_npz)
        ref_logits, lens = z["logits"], z["seq_lengths"]
        ref_name = Path(args.ref_npz).name
        # the npz holds the sequences of --ids-file; t9ids writes [T1..T8, T9] or [T9]
        offs = np.concatenate([[0], np.cumsum(lens)])
        seq_off = {("T9" if len(lens) == 1 else (ORDER8 + ("T9",))[j]): int(offs[j]) for j in range(len(lens))}
    else:
        ref_logits = dump.load("logits.npy")
        ref_name = "fp32 dump"
        seq_off = {t: j * L for j, t in enumerate(ORDER8)} | {"T9": n8}
    res = {}
    texts = [t for t in ORDER8 if t in seq_off] + ["T9"]
    for tid in texts:
        ids = ts.t9 if tid == "T9" else ts.ids[tid]
        off = seq_off[tid]
        t0 = time.time()
        res[tid], _ = _per_position(model, ids, lambda a, b, off=off: ref_logits[off + a:off + b], prof.prefill_chunk)
        print(f"[diag g4] {args.arm}/{args.act} vs {ref_name}: {tid} mean KL {res[tid]['kl'].mean():.4f} ({time.time() - t0:.0f}s)",
              flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tag = f"g4_{args.arm}_{args.act}" + (f"_vs_{Path(args.ref_npz).stem}" if args.ref_npz else "")
    path = Path(args.out) / f"{tag}.npz"
    flat = {f"{t}__{k}": v for t, d in res.items() for k, v in d.items()}
    np.savez(path, **flat, segments=json.dumps(_t9_segments(ts)), meta=json.dumps(
        {"arm": args.arm, "act": args.act, "ref": ref_name, "chunk": prof.prefill_chunk, "utc": common.utc_iso(),
         "decisive_lead": ctx.thresholds["G4"]["K8_backstop_decisive_lead_nats"], "text_len": L,
         "t9_buckets": [list(b) for b in prof.t9_buckets], "window": prof.window_position}))
    print("->", common.redact_path(path))
    return 0


def cmd_refcmp(args) -> int:
    """A reference-mode npz (e.g. --emulate-bf16 --dequant-dir K8) against the
    fp32 dump, per position, in the g4 npz format: what a correct bf16 (and
    8-bit) implementation achieves end to end, i.e. the calibration G4 lacks."""
    from gate.checks.g4_e2e import _compare

    ctx, ts, dump = _setup(args.tiny)
    prof = ts.profile
    L, n8 = prof.text_len, 8 * prof.text_len
    z = np.load(args.npz)
    cand, lens = z["logits"], z["seq_lengths"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    names = ["T9"] if len(lens) == 1 else list(ORDER8) + ["T9"]
    ref_logits = dump.load("logits.npy")
    ref_off = {t: j * L for j, t in enumerate(ORDER8)} | {"T9": n8}
    flat = {}
    for j, tid in enumerate(names):
        ids = np.asarray(ts.t9 if tid == "T9" else ts.ids[tid])
        n = ids.size
        cols = {k: np.empty(n) for k in ("kl", "lead", "dnll", "nll_ref")}
        agree, tr, tg = np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.int64)
        for a in range(0, n, 1024):
            b = min(n, a + 1024)
            ref = np.asarray(ref_logits[ref_off[tid] + a:ref_off[tid] + b], dtype=np.float32)
            got = np.asarray(cand[offs[j] + a:offs[j] + b], dtype=np.float32)
            c = _compare(ref, got, ids[a + 1:b + 1])
            nxt = ids[a + 1:b + 1]
            lpr = common.log_softmax64(ref[:nxt.size])
            c["nll_ref"] = np.full(b - a, np.nan)
            c["nll_ref"][:nxt.size] = -lpr[np.arange(nxt.size), nxt]
            for k in cols:
                cols[k][a:b] = c[k]
            agree[a:b], tr[a:b], tg[a:b] = c["agree"], ref.argmax(-1), got.argmax(-1)
        for k, v in {**cols, "agree": agree, "top_ref": tr, "top_got": tg}.items():
            flat[f"{tid}__{k}"] = v
        print(f"[diag refcmp] {tid}: mean KL(fp32 dump || {Path(args.npz).stem}) {cols['kl'].mean():.4f}", flush=True)
    path = Path(args.out) / f"refcmp_{Path(args.npz).stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **flat, segments=json.dumps(_t9_segments(ts)), meta=json.dumps(
        {"arm": Path(args.npz).stem, "act": "reference", "ref": "fp32 dump", "chunk": prof.prefill_chunk,
         "utc": common.utc_iso(), "decisive_lead": ctx.thresholds["G4"]["K8_backstop_decisive_lead_nats"],
         "text_len": L, "t9_buckets": [list(b) for b in prof.t9_buckets], "window": prof.window_position}))
    print("->", common.redact_path(path))
    return 0


def cmd_t9ids(args) -> int:
    _, ts, _ = _setup(args.tiny)
    seqs = ([list(map(int, ts.ids[t])) for t in ORDER8] if args.with8 else []) + [list(map(int, ts.t9))]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"ids": seqs}))
    print(f"{len(seqs)} sequence(s) -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# analyse
# ---------------------------------------------------------------------------


def _summ(d: dict, m=None, lead=2.0) -> dict:
    m = np.ones(d["kl"].size, bool) if m is None else m
    kl, ag, ld = d["kl"][m], d["agree"][m], d["lead"][m]
    dec = ld >= lead
    dn, nr = d["dnll"][m], d["nll_ref"][m]
    return {"n": int(m.sum()), "mean_kl": float(kl.mean()) if kl.size else None,
            "dnll": float(np.nanmean(dn)) if np.isfinite(dn).any() else None,
            "nll_ref": float(np.nanmean(nr)) if np.isfinite(nr).any() else None,
            "n_decisive": int(dec.sum()), "decisive_miss": int((~ag[dec]).sum()),
            "top1_decisive": float(ag[dec].mean()) if dec.any() else None, "top1": float(ag.mean()) if ag.size else None}


def _block_boot(x: np.ndarray, block=64, B=2000, seed=36) -> tuple:
    rng = np.random.default_rng(seed)
    nb = x.size // block
    xb = x[: nb * block].reshape(nb, block).mean(axis=1)
    bs = np.array([xb[rng.integers(0, nb, nb)].mean() for _ in range(B)])
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def cmd_analyse(args) -> int:
    for f in args.files:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        segs = json.loads(str(z["segments"]))
        lead = meta["decisive_lead"]
        d = {t: {k: z[f"{t}__{k}"] for k in ("kl", "lead", "dnll", "nll_ref", "agree", "top_ref", "top_got")}
             for t in ORDER8 + ("T9",) if f"{t}__kl" in z}
        print(f"\n=== {Path(f).name}: {meta['arm']} {meta['act']} vs {meta['ref']}")
        cat8 = {k: np.concatenate([d[t][k] for t in ORDER8 if t in d]) for k in d["T9"]}
        print("T1-8 ", json.dumps(_summ(cat8, lead=lead)))
        t9 = d["T9"]
        print("T9   ", json.dumps(_summ(t9, lead=lead)))
        pos = np.arange(t9["kl"].size)
        for lo, hi in meta["t9_buckets"]:
            print(f"  bucket {lo}-{hi}", json.dumps(_summ(t9, (pos >= lo) & (pos < hi), lead)))
        print("  T9 by 512-token bin: mean KL | decisive misses/decisive")
        line = []
        for a in range(0, pos.size, 512):
            s = _summ(t9, (pos >= a) & (pos < a + 512), lead)
            line.append(f"{a:5d}:{s['mean_kl']:.3f}|{s['decisive_miss']}/{s['n_decisive']}")
        for j in range(0, len(line), 4):
            print("   ", "  ".join(line[j:j + 4]))
        ch = meta["chunk"]
        near = np.zeros(pos.size, bool)
        for b in range(ch, pos.size, ch):
            near |= (pos >= b - 32) & (pos < b + 32)
        print(f"  within +-32 of a {ch}-chunk boundary:", json.dumps(_summ(t9, near, lead)),
              "| elsewhere:", json.dumps(_summ(t9, ~near & (pos >= ch), lead)))
        print("  content-matched controls (same 1,536 tokens: standalone at 0 vs inside T9):")
        for tid, off in segs.items():
            if tid not in d:
                continue
            Lt = d[tid]["kl"].size
            a, b = d[tid], {k: v[off:off + Lt] for k, v in t9.items()}
            sa, sb = _summ(a, lead=lead), _summ(b, lead=lead)
            delta = b["kl"] - a["kl"]
            lo, hi = _block_boot(delta)
            dn = b["nll_ref"][:Lt - 1] - a["nll_ref"][:Lt - 1]
            nlo, nhi = _block_boot(dn)
            print(f"    {tid} @T9[{off}]: standalone KL {sa['mean_kl']:.4f} miss {sa['decisive_miss']}/{sa['n_decisive']} "
                  f"dNLL {sa['dnll']:+.4f} refNLL {sa['nll_ref']:.3f} | in T9 KL {sb['mean_kl']:.4f} "
                  f"miss {sb['decisive_miss']}/{sb['n_decisive']} dNLL {sb['dnll']:+.4f} refNLL {sb['nll_ref']:.3f} | "
                  f"paired dKL {delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}], paired d(refNLL) {dn.mean():+.4f} "
                  f"[{nlo:+.4f}, {nhi:+.4f}] (64-token block bootstrap)")
        man = KIT / "gate" / "texts" / "MANIFEST.json"
        if man.is_file() and "T4" in segs:
            comps = common.read_json(man)["web"]["T9"]["components"]
            sep = segs["T4"] - comps[0]["n_tokens"]
            print(f"  T9 by document (separator {sep} token(s)):")
            p0 = 0
            for j, c in enumerate(comps):
                a0, b0 = p0, min(p0 + c["n_tokens"], pos.size)
                if a0 >= pos.size:
                    break
                s_ = _summ(t9, (pos >= a0) & (pos < b0), lead)
                print(f"    [{a0:5d},{b0:5d}) {c['what'][:24]:24s} {str(c.get('row', '')):5s} KL {s_['mean_kl']:.4f} "
                      f"miss {s_['decisive_miss']}/{s_['n_decisive']} dNLL {s_['dnll']:+.4f} refNLL {s_['nll_ref']:.3f}")
                p0 = b0 + sep
        order = np.argsort(-t9["kl"])[: args.top]
        print("  T9 top KL positions (pos, KL, lead, ref top1 -> port top1):",
              ", ".join(f"{int(p)}:{t9['kl'][p]:.1f}/{t9['lead'][p]:.1f}/{int(t9['top_ref'][p])}->{int(t9['top_got'][p])}" for p in order))
    if len(args.files) == 2:
        za, zb = (np.load(f) for f in args.files)
        a, b = za["T9__kl"], zb["T9__kl"]
        print(f"\nT9 per-position KL correlation between the two files: {np.corrcoef(a, b)[0, 1]:.3f}; "
              f"ratio of means {b.mean() / a.mean():.3f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tiny", help="tiny checkpoint root (dry run of this script)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g2 = sub.add_parser("g2")
    g2.add_argument("--layers", default="20")
    g2.add_argument("--emu", action="store_true")
    g2.add_argument("--top", type=int, default=10)
    g2.add_argument("--record", help="phase3.json of the gate run, to check the reproduction")
    g2.add_argument("--out", required=True)
    g4 = sub.add_parser("g4")
    g4.add_argument("--arm", default="K8", choices=("K8", "K4"))
    g4.add_argument("--act", default="bf16", choices=("bf16", "fp32"))
    g4.add_argument("--ref-npz")
    g4.add_argument("--out", required=True)
    an = sub.add_parser("analyse")
    an.add_argument("files", nargs="+")
    an.add_argument("--top", type=int, default=15)
    rc = sub.add_parser("refcmp")
    rc.add_argument("npz")
    rc.add_argument("--out", required=True)
    t9 = sub.add_parser("t9ids")
    t9.add_argument("--with8", action="store_true")
    t9.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    common.set_offline_env()
    return {"g2": cmd_g2, "g4": cmd_g4, "analyse": cmd_analyse, "t9ids": cmd_t9ids, "refcmp": cmd_refcmp}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
