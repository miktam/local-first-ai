# SPDX-License-Identifier: MIT
"""exp_036 diagnostic V: what does a vendor-faithful bf16 implementation score
on the gate's own statistics? (read-only on the kit; no thresholds, no gate
record touched)

Run on the mbp from the kit root, after gate 20261005T050112Z, with the same
environment as RUNBOOK step 10 ($EXP036_MODELS, $EXP036_WORK, MLX_ENABLE_TF32=0):

    caffeinate -i "$PY" <this file> --parts A,B,D

Parts (each independent; the fp32 dump of the failed gate is reused, never
rewritten):

  A  G2 6-sigma rule applied to the bf16 EMULATION (the gate's "emu", the
     reference rounded wherever vLLM stores bf16) on T1-T8, every layer:
     sigma_l, max gap/sigma, n_beyond_6sigma, plus a per-token-sigma version.
     Then the PORT (bf16) on layer 20 only: which token made the 6.08 sigma
     flip, its per-token sigma, and whether the emu flips the same token.
     ~6 min (emu natural only) + ~1 min.
  B  emu END TO END (KolibriReference(emulate_bf16=True).forward_packed) on
     T1-T8 and T9, compared with the fp32 dump's logits with G4's own
     statistics (g4_e2e._compare/_summary): mean KL, top1_decisive, buckets.
     ~12-15 min, ~30 GB RSS (logits 28,672 x 128,000 fp32).
  C  (optional) as B on T9 only, with q, k, v rounded to FP8 E4M3 at unit
     scale before attention (the vendor's evaluated KV/query precision,
     tech report p.180; vLLM attention.py query_quant + fp8 KV cache).
     ~8 min.
  D  G5 batch parity on K8 with PER-POSITION output, plus a B=1 control
     through the same BatchGenerator path (each sequence alone). ~3 min.

Writes $EXP036_WORK/diag_vendor/diag_vendor_<UTC>.json (and prints a summary).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path.cwd()
if not (EXP / "gate" / "run_gate.py").is_file():
    sys.exit("run from the exp_036 kit root")
sys.path.insert(0, str(EXP))

from tools.precision import ensure_exact_fp32  # noqa: E402

ensure_exact_fp32()

from gate import common  # noqa: E402
from gate import run_gate  # noqa: E402
from gate.checks import g4_e2e, g5_generation, ref_pass  # noqa: E402


def utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def make_ctx(tiny_root: str | None):
    if tiny_root:
        root = Path(tiny_root)
        return run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=common.load_thresholds(),
                                    tiny=True, results_dir=root / "results", head_policy="quantised_head",
                                    subprocess_phases=False)
    return run_gate.GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(),
                                thresholds=common.load_thresholds())


# ---------------------------------------------------------------- part A ---

def part_a(ctx, ts, dump, sigma_k: float, port_layer: int, log) -> dict:
    from gate import harness

    L = ts.profile.text_len
    n8 = 8 * L
    segs = ref_pass.segments_of([L] * 8)
    emu = ref_pass.RefRunner(ctx.bf16_dir, "emu")
    n_layers = int(dump.record()["num_layers"])
    rows, total_beyond, total_beyond_tok, worst = [], 0, 0, (0.0, None)
    keep = {}
    for i in range(n_layers):
        h = np.array(dump.layer(i, "h_in")[:n8])
        top6 = np.array(dump.layer(i, "top6")[:n8])
        gap = np.array(dump.layer(i, "top6_gap")[:n8], dtype=np.float64)
        lg = np.array(dump.layer(i, "router_logits")[:n8], dtype=np.float64)
        nat = emu.branches(i, h, segs)
        dis = ref_pass.set_disagree(np.asarray(nat["top6"]), top6)
        diff = np.asarray(nat["logits"], dtype=np.float64) - lg
        sigma = float(np.sqrt(np.mean(diff * diff)))
        sig_t = np.sqrt(np.mean(diff * diff, axis=-1))
        g = gap[dis]
        r = g / sigma if g.size else np.zeros(0)
        rt = g / np.maximum(sig_t[dis], 1e-30) if g.size else np.zeros(0)
        nb, nbt = int((r >= sigma_k).sum()), int((rt >= sigma_k).sum())
        total_beyond += nb
        total_beyond_tok += nbt
        mx_ = float(r.max()) if r.size else 0.0
        if mx_ > worst[0]:
            j = int(np.flatnonzero(dis)[int(np.argmax(r))])
            worst = (mx_, {"layer": i, "row": j, "text": f"T{j // L + 1}", "pos": j % L})
        rows.append({"layer": i, "n_disagree": int(dis.sum()), "disagree_frac": float(dis.mean()), "sigma": sigma,
                     "sigma_t_p50": float(np.median(sig_t)), "sigma_t_p99": float(np.quantile(sig_t, 0.99)),
                     "sigma_t_max": float(sig_t.max()), "max_gap_over_sigma": mx_, "n_beyond": nb,
                     "max_gap_over_sigma_t": float(rt.max()) if rt.size else 0.0, "n_beyond_per_token": nbt})
        if i == port_layer:
            keep = {"h": h, "top6": top6, "gap": gap, "lg": lg, "emu_dis": dis, "emu_sig_t": sig_t,
                    "emu_logits": np.asarray(nat["logits"], dtype=np.float64)}
        log(f"[A] emu layer {i + 1}/{n_layers}: disagree {dis.mean():.4f} max gap/sigma {mx_:.2f}")
    out = {"layers": rows, "n_beyond_total": total_beyond, "n_beyond_per_token_total": total_beyond_tok,
           "max_gap_over_sigma": worst[0], "worst": worst[1],
           "disagree_frac": float(np.mean([r["disagree_frac"] for r in rows]))}
    # The port on one layer: locate its worst flip and compare with the emu there.
    if keep:
        H = keep["h"].shape[-1]
        p16 = harness.PortLayers(ctx.bf16_dir, "bf16")
        layer = p16.build_layer(port_layer)
        nb_ = p16.run(layer, port_layer, keep["h"].reshape(8, L, H))
        pid = np.asarray(nb_["ids"]).reshape(n8, -1)
        plg = np.asarray(nb_["router_logits"], dtype=np.float64).reshape(n8, -1)
        pdis = ref_pass.set_disagree(pid, keep["top6"])
        pdiff = plg - keep["lg"]
        psig = float(np.sqrt(np.mean(pdiff * pdiff)))
        psig_t = np.sqrt(np.mean(pdiff * pdiff, axis=-1))
        g = keep["gap"][pdis]
        idx = np.flatnonzero(pdis)
        order = np.argsort(-g)[:5]
        top = []
        for o in order:
            j = int(idx[o])
            top.append({"row": j, "text": f"T{j // L + 1}", "pos": j % L, "gap": float(keep["gap"][j]),
                        "gap_over_sigma": float(keep["gap"][j] / psig),
                        "port_sigma_t": float(psig_t[j]), "gap_over_port_sigma_t": float(keep["gap"][j] / psig_t[j]),
                        "emu_flips_here": bool(keep["emu_dis"][j]), "emu_sigma_t": float(keep["emu_sig_t"][j]),
                        "port_minus_emu_logit_rms": float(np.sqrt(np.mean((plg[j] - keep["emu_logits"][j]) ** 2)))})
        out["port_layer"] = {"layer": port_layer, "sigma": psig, "n_disagree": int(pdis.sum()),
                             "both_flip": int((pdis & keep["emu_dis"]).sum()),
                             "port_only": int((pdis & ~keep["emu_dis"]).sum()),
                             "emu_only": int((~pdis & keep["emu_dis"]).sum()), "top_flips": top}
    return out


# ---------------------------------------------------------- parts B / C ---

def _e4m3(x: np.ndarray) -> np.ndarray:
    """Round to FP8 E4M3 (fn) at unit scale, nearest-even, saturating at 448
    (vLLM's fp8 quant clamps to the format's finite range)."""
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), np.float32(448.0))
    e = np.floor(np.log2(np.maximum(a, np.float32(2.0 ** -6))))
    step = np.exp2(e - 3).astype(np.float32)  # 3 mantissa bits; subnormal step 2^-9 below 2^-6
    y = np.round(a / step) * step
    return (np.sign(x) * np.minimum(y, np.float32(448.0))).astype(np.float32)


def g4_stats(ts, ref_logits, logits, lead: float) -> dict:
    prof = ts.profile
    L = prof.text_len
    parts8, per_text = [], {}
    for j, tid in enumerate(g4_e2e.ORDER8):
        ids = np.asarray(ts.ids[tid])
        c = g4_e2e._compare(np.asarray(ref_logits[j * L:(j + 1) * L], np.float32),
                            np.asarray(logits[j * L:(j + 1) * L], np.float32), ids[1:])
        parts8.append(c)
        per_text[tid] = g4_e2e._summary([c], lead)
    return {"T1-8": g4_e2e._summary(parts8, lead), "per_text": per_text}


def t9_stats(ts, ref_logits, logits, lead: float, chunk: int = 1024) -> dict:
    prof = ts.profile
    n8, t9 = 8 * prof.text_len, np.asarray(ts.t9)
    parts, buckets = [], {tuple(b): [] for b in prof.t9_buckets}
    for a in range(0, t9.size, chunk):
        b = min(t9.size, a + chunk)
        c = g4_e2e._compare(np.asarray(ref_logits[n8 + a:n8 + b], np.float32), np.asarray(logits[a:b], np.float32),
                            t9[a + 1:b + 1])
        parts.append(c)
        pos = np.arange(a, b)
        for lo, hi in buckets:
            m = (pos >= lo) & (pos < hi)
            if m.any():
                buckets[(lo, hi)].append(g4_e2e._masked(c, m))
    return {"T9": g4_e2e._summary(parts, lead),
            "kl_by_bucket": {f"{lo}-{hi}": g4_e2e._summary(v, lead) for (lo, hi), v in buckets.items() if v}}


def part_b(ctx, ts, dump, lead: float, log) -> dict:
    from reference import kolibri_ref

    ref = kolibri_ref.KolibriReference(str(ctx.bf16_dir), emulate_bf16=True, verbose=False)
    seqs = ts.packed8() + [ts.t9]
    t0 = time.time()
    logits, _, _, _ = ref.forward_packed(seqs)
    log(f"[B] emu forward {time.time() - t0:.0f} s")
    ref_logits = dump.load("logits.npy")
    n8 = 8 * ts.profile.text_len
    out = g4_stats(ts, ref_logits, logits[:n8], lead)
    out.update(t9_stats(ts, ref_logits, logits[n8:], lead))
    out["wall_s"] = time.time() - t0
    return out


def part_c(ctx, ts, dump, lead: float, log) -> dict:
    from reference import kolibri_ref

    orig = kolibri_ref.sdpa_ref

    def sdpa_fp8(q, k, v, scale, window, q_chunk=None):
        return orig(_e4m3(q), _e4m3(k), _e4m3(v), scale, window, q_chunk)

    kolibri_ref.sdpa_ref = sdpa_fp8
    try:
        ref = kolibri_ref.KolibriReference(str(ctx.bf16_dir), emulate_bf16=True, verbose=False)
        t0 = time.time()
        logits, _, _, _ = ref.forward_packed([ts.t9])
        log(f"[C] emu+fp8 qkv forward on T9 {time.time() - t0:.0f} s")
    finally:
        kolibri_ref.sdpa_ref = orig
    out = t9_stats(ts, dump.load("logits.npy"), logits, lead)
    out["note"] = "bf16 emulation + q/k/v rounded to E4M3 at unit scale before attention; P not rounded; weights bf16"
    return out


# ---------------------------------------------------------------- part D ---

def _parity_positions(model, prompts, max_tokens, B, eos) -> list:
    """g5_generation.batch_parity's loop, keeping every position."""
    import mlx.core as mx

    from gate.harness import logits_at

    gen, _ = g5_generation._batch_generator(model, B, eos, max(max_tokens))
    got, queue, uid_of, in_flight, step, any_finished, admitted_mid = {}, list(range(len(prompts))), {}, 0, 0, False, set()
    try:
        while queue or in_flight:
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                for u, j in zip(uids, group):
                    uid_of[u] = j
                    got[u] = {"tokens": [], "lp": [], "steps": [], "live": []}
                    if any_finished:
                        admitted_mid.add(j)
                in_flight += len(group)
            _, resps = gen.next()
            live = len(gen._generation_batch) + len(gen._prompt_batch)
            for r in resps:
                g = got[r.uid]
                g["tokens"].append(int(r.token))
                lp = r.logprobs.astype(mx.float32)
                mx.eval(lp)
                g["lp"].append(np.array(lp))
                g["steps"].append(step)
                g["live"].append(live)
                if r.finish_reason is not None:
                    in_flight -= 1
                    any_finished = True
            step += 1
    finally:
        gen.close()
    rows = []
    for u, j in sorted(uid_of.items(), key=lambda t: t[1]):
        g, p = got[u], prompts[j]
        P = len(p)
        single = g5_generation._lp(logits_at(model, list(p) + g["tokens"][:-1], np.arange(P - 1, P - 1 + len(g["tokens"]))))
        batched = g5_generation._lp(np.stack(g["lp"]))
        kl = g5_generation.kl_lp(single, batched)
        dis = single.argmax(-1) != batched.argmax(-1)
        lead = np.sort(single, axis=-1)[:, -1] - np.sort(single, axis=-1)[:, -2]
        for t in range(len(g["tokens"])):
            rows.append({"seq": j, "prompt_len": P, "t": t, "step": g["steps"][t], "live": g["live"][t],
                         "admitted_mid_run": j in admitted_mid, "kl": float(kl[t]), "dis": bool(dis[t]),
                         "single_lead": float(lead[t]), "token": g["tokens"][t]})
    return rows


def _summ(rows) -> dict:
    if not rows:
        return {"n": 0}
    kl = np.array([r["kl"] for r in rows])
    return {"n": len(rows), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
            "p90_kl": float(np.quantile(kl, 0.9)), "max_kl": float(kl.max()),
            "share_of_kl_top10": float(np.sort(kl)[-10:].sum() / max(kl.sum(), 1e-30)),
            "top1_dis": float(np.mean([r["dis"] for r in rows]))}


def part_d(ctx, ts, log) -> dict:
    model, _, _ = common.load_port(ctx.arm_dir("K8"))
    prof = ts.profile
    order = g4_e2e.ORDER8
    prompts = [ts.ids[order[j % 8]][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
    mt = list(prof.batch_max_tokens)
    B = ctx.thresholds["G5"]["batch_B"]
    window = prof.window_position
    rows = _parity_positions(model, prompts, mt, B, common.EOS_IDS)
    log("[D] B=8 done")
    solo = []
    for j in range(len(prompts)):
        for r in _parity_positions(model, [prompts[j]], [mt[j]], 1, common.EOS_IDS):
            r["seq"] = j
            solo.append(r)
    log("[D] B=1 control done")
    by = lambda rs, f: _summ([r for r in rs if f(r)])  # noqa: E731
    return {
        "B8": {"all": _summ(rows), "t0_first_token": by(rows, lambda r: r["t"] == 0),
               "t_ge_1": by(rows, lambda r: r["t"] >= 1),
               "prompt_le_window": by(rows, lambda r: r["prompt_len"] + r["t"] < window),
               "prompt_gt_window": by(rows, lambda r: r["prompt_len"] + r["t"] >= window),
               "admitted_mid_run": by(rows, lambda r: r["admitted_mid_run"]),
               "first_wave": by(rows, lambda r: not r["admitted_mid_run"]),
               "per_seq": {j: by(rows, lambda r, j=j: r["seq"] == j) for j in range(len(prompts))}},
        "B1_control": {"all": _summ(solo), "per_seq": {j: by(solo, lambda r, j=j: r["seq"] == j) for j in range(len(prompts))}},
        "rows_B8": rows, "rows_B1": solo, "window": window,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,B,D")
    ap.add_argument("--tiny-root", default=None, help="tests only: a tiny gate root (models + work)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    parts = {p.strip().upper() for p in args.parts.split(",") if p.strip()}
    ctx = make_ctx(args.tiny_root)
    ts = run_gate.textset_for(ctx)
    dump = ref_pass.Dump(run_gate.dump_dir(ctx, ts))
    if not ref_pass.reusable(dump, ctx.bf16_dir, ts.key):
        sys.exit("the fp32 dump of the gate is missing or stale: nothing to compare against")
    th = ctx.thresholds
    lead = th["G4"]["K8_backstop_decisive_lead_nats"]
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    rec = {"utc": utc(), "dump_record_sha256": common.sha256_file(Path(run_gate.dump_dir(ctx, ts)) / "record.json"),
           "parts": sorted(parts), "tiny": bool(args.tiny_root)}
    if "A" in parts:
        port_layer = 20 if not args.tiny_root else 1
        rec["A"] = part_a(ctx, ts, dump, th["G2"]["bf16_selection_gap_sigma"], port_layer, log)
    if "B" in parts:
        rec["B"] = part_b(ctx, ts, dump, lead, log)
    if "C" in parts:
        rec["C"] = part_c(ctx, ts, dump, lead, log)
    if "D" in parts:
        rec["D"] = part_d(ctx, ts, log)
    out = Path(args.out) if args.out else Path(ctx.work_dir) / "diag_vendor" / f"diag_vendor_{rec['utc']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=float))
    summ = {}
    if "A" in rec:
        a = rec["A"]
        summ["A_emu"] = {k: a[k] for k in ("disagree_frac", "max_gap_over_sigma", "n_beyond_total", "n_beyond_per_token_total", "worst")}
        if "port_layer" in a:
            summ["A_port_layer"] = {k: a["port_layer"][k] for k in ("layer", "n_disagree", "both_flip", "port_only", "emu_only")}
            summ["A_port_layer"]["top_flip"] = a["port_layer"]["top_flips"][:1]
    for p in ("B", "C"):
        if p in rec:
            r = rec[p]
            summ[p] = {k: {kk: r[k][kk] for kk in ("mean_kl", "top1_decisive", "n_decisive")} for k in ("T1-8", "T9") if k in r}
            summ[p]["kl_by_bucket"] = {b: {kk: v[kk] for kk in ("mean_kl", "top1_decisive")} for b, v in r["kl_by_bucket"].items()}
    if "D" in rec:
        summ["D"] = {"B8": rec["D"]["B8"]["all"], "B1": rec["D"]["B1_control"]["all"],
                     "B8_first_token": rec["D"]["B8"]["t0_first_token"], "B8_t_ge_1": rec["D"]["B8"]["t_ge_1"],
                     "B8_le_window": rec["D"]["B8"]["prompt_le_window"], "B8_gt_window": rec["D"]["B8"]["prompt_gt_window"],
                     "B8_mid_run": rec["D"]["B8"]["admitted_mid_run"]}
    print(json.dumps({"out": common.redact_path(out), "summary": summ}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
