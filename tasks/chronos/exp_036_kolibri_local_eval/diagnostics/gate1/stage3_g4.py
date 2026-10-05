# SPDX-License-Identifier: MIT
"""Stage 3 (DIAGNOSIS §4): G4 long-context backstop of K8. About 60-80 minutes on
the mbp, as separate steps, each its own process (one model-loading process at
a time) and each resumable (a finished step leaves <file>.done.json in
$EXP036_WORK/diag_gate1/stage3_g4/ and is skipped when run again):

  g4           K8 bf16 as loaded, then the same model with fp32 activations
               (model.set_dtype(float32), the quantised embedding kept at its
               stored bf16 dequantisation: diaglib.fp32_on_dequantised_weights),
               teacher-forced exactly as G4 does
               (runner.generate.iter_teacher_force_logprobs, chunk 2048) over
               T1-T8 and T9, against the gate's fp32 reference dump. Per
               position: KL(ref||port), reference lead, top-1 agreement, dNLL,
               reference NLL, entropy of both. The bf16 pass must reproduce
               G4's record (T9 top1_decisive 0.98889, 66 / 5,943).
  t9ids        T1-T8 and T9 token ids as a reference --ids-file (work dir only)
  ref_fp32     reference/kolibri_ref.py on the dequantised K8 weights, fp32
  bugtest      K8 fp32 activations on exactly the dequantised K8 weights
               (diaglib.fp32_on_dequantised_weights) against ref_fp32 (the bug
               test), then the
               chunk-64 exactness test in fp32 on T1 and T3 at 520-1,100 and
               T9 at 15,000-15,300 (prefill 2048 vs prefill 64 vs decode; and
               chunk 64 against ref_fp32 on the T9 range)
  ref_emu      the reference's bf16 emulation on the dequantised K8 weights
  refcmp       ref_emu against the fp32 dump, per position (E_T9, the calibrator)
  report       the analysis and the stage JSON: McNemar (port bf16 vs emu, on
               decisive positions), paired dNLL port - emu with 99 % 64-block
               bootstrap CI, dNLL/KL with 95 % CIs, entropy shift, max_kl
               positions, content-matched controls (T1, T4, T3, T5, T6
               standalone vs inside T9), the web tail as its own segment, chunk
               boundaries and document boundaries, and G4's top-10 KL positions
               (for stage 2's tail sign)
  all          every step in order, each in a fresh subprocess

The output's rule_inputs are FROZEN_RULES.md §2's stage-3 keys; stage 5 alone
evaluates the frozen G4 rule (§5) on them. Absolute port entropy stays in the
work file (k8abs_*); the restricted copy holds port-vs-reference and
port-vs-emu comparisons only.

usage (kit root, env sourced): caffeinate -i "$PY" diagnostics/gate1/stage3_g4.py all
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import diaglib as D  # noqa: E402
from gate import common  # noqa: E402

STEPS = ("g4", "t9ids", "ref_fp32", "bugtest", "ref_emu", "refcmp", "report")
COLS = ("kl", "lead", "dnll", "nll_ref", "agree", "ent_got", "ent_ref", "top_ref", "top_got")
WEB_TAIL_REAL = (13399, 16384)


# ---------------------------------------------------------------------------
# Per-position comparison (the gate's g4_e2e._compare plus entropies)
# ---------------------------------------------------------------------------


def compare_rows(ref: np.ndarray, got: np.ndarray, ids_next: np.ndarray, rows: int = 256) -> dict:
    """gate.checks.g4_e2e._compare (row-wise: KL(ref || got), reference lead,
    top-1 agreement, dNLL) plus reference NLL and both entropies, computed in
    blocks of `rows` so no [2048, V] float64 temporary is held at once."""
    from gate.checks.g4_e2e import _compare

    n_all = ref.shape[0]
    out = empty_cols(n_all)
    for a in range(0, n_all, rows):
        b = min(n_all, a + rows)
        nxt = ids_next[a:b] if a < ids_next.size else ids_next[:0]
        c = _compare(ref[a:b], got[a:b], nxt)
        lr = common.log_softmax64(ref[a:b])
        lg = common.log_softmax64(got[a:b])
        nll = np.full(b - a, np.nan)
        if nxt.size:
            nll[:nxt.size] = -lr[np.arange(nxt.size), nxt]
        for k in ("kl", "lead", "dnll", "agree"):
            out[k][a:b] = c[k]
        out["nll_ref"][a:b] = nll
        out["ent_got"][a:b] = D.entropy_lp(lg)
        out["ent_ref"][a:b] = D.entropy_lp(lr)
        out["top_ref"][a:b] = ref[a:b].argmax(-1)
        out["top_got"][a:b] = got[a:b].argmax(-1)
    return out


def empty_cols(n: int) -> dict:
    out = {k: np.empty(n) for k in COLS}
    out["agree"] = np.zeros(n, bool)
    out["top_ref"] = np.zeros(n, np.int64)
    out["top_got"] = np.zeros(n, np.int64)
    return out


def per_position(model, ids, ref_rows, chunk) -> dict:
    from runner.generate import iter_teacher_force_logprobs

    ids = np.asarray(ids)
    cols = empty_cols(ids.size)
    for a, block in iter_teacher_force_logprobs(model, list(map(int, ids)), chunk):
        b = a + block.shape[0]
        ref = np.asarray(ref_rows(a, b), dtype=np.float32)
        c = compare_rows(ref, block, ids[a + 1:b + 1])
        for k in COLS:
            cols[k][a:b] = c[k]
    return cols


def texts_and_offsets(ts, prof):
    """[(tid, ids, offset in the reference rows)] for T1-T8 and T9."""
    L = prof.text_len
    out = [(t, ts.ids[t], j * L) for j, t in enumerate(D.ORDER8)]
    out.append(("T9", ts.t9, 8 * L))
    return out


def save_cols(path: Path, res: dict, meta: dict) -> None:
    flat = {f"{t}__{k}": v for t, d in res.items() for k, v in d.items()}
    D.write_npz(path, **flat, meta=json.dumps(meta))


def load_cols(path: Path) -> tuple[dict, dict]:
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    res = {}
    for key in z.files:
        if "__" in key:
            t, k = key.split("__", 1)
            res.setdefault(t, {})[k] = z[key]
    return res, meta


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def step_g4(run) -> None:
    import mlx.core as mx

    if all(D.is_done(run.scratch / f"g4_K8_{act}.npz") for act in ("bf16", "fp32")):
        D.log("[stage3] g4: done earlier")      # run_stages.sh re-invokes every step on a resume
        return
    ts, dump = run.textset(), run.dump()
    prof = ts.profile
    ref_logits = dump.load("logits.npy")
    model, _ = run.load_arm("K8")
    for act in ("bf16", "fp32"):
        path = run.scratch / f"g4_K8_{act}.npz"
        if D.is_done(path):
            D.log(f"[stage3] g4 {act}: done earlier")
            continue
        if act == "fp32":
            model.set_dtype(mx.float32)
            D.fp32_on_dequantised_weights(model)
            mx.eval(model.parameters())
        res = {}
        for tid, ids, off in texts_and_offsets(ts, prof):
            t0 = time.time()
            res[tid] = per_position(model, ids, lambda a, b, off=off: ref_logits[off + a:off + b], prof.prefill_chunk)
            D.log(f"[stage3] g4 K8/{act} {tid}: mean KL {res[tid]['kl'].mean():.4f} ({time.time() - t0:.0f}s)")
        save_cols(path, res, {"arm": "K8", "act": act, "ref": "fp32 dump", "chunk": prof.prefill_chunk})
        D.done_marker(path)
    del model
    run.release()


def step_t9ids(run) -> None:
    path = run.scratch / "ids.json"
    if D.is_done(path):
        return
    ts = run.textset()
    seqs = [list(map(int, ts.ids[t])) for t in D.ORDER8] + [list(map(int, ts.t9))]
    path.write_text(json.dumps({"ids": seqs}))
    D.done_marker(path)


def step_ref(run, mode: str) -> None:
    """reference/kolibri_ref.py on the dequantised K8 weights (what the CLI's
    --dequant-dir [--emulate-bf16] --ids-file runs: KolibriReference.forward_packed),
    in this process with no MLX model loaded. The logits [N, V] fp32 go to a
    .npy so later steps can memory-map them (an .npz member cannot be mapped)."""
    from reference import kolibri_ref

    path = run.scratch / f"ref_deqK8_{mode}.logits.npy"
    if D.is_done(path):
        D.log(f"[stage3] ref {mode}: done earlier")
        return
    ids = run.scratch / "ids.json"
    if not D.is_done(ids):
        raise SystemExit("stage3: run the t9ids step first")
    seqs = kolibri_ref._read_ids(str(ids))
    t0 = time.time()
    ref = kolibri_ref.KolibriReference(str(run.ctx.bf16_dir), emulate_bf16=(mode == "emu"),
                                       dequant_dir=str(run.ctx.arm_dir("K8")))
    logits, _, _, lengths = ref.forward_packed(seqs)
    tmp = run.scratch / f"ref_deqK8_{mode}.logits.partial.npy"
    np.save(tmp, logits)
    tmp.replace(path)
    D.done_marker(path, {"wall_s": time.time() - t0, "mode": ref.mode, "dequantised_from": "K8",
                         "seq_lengths": [int(x) for x in lengths], "ref_version": kolibri_ref.REF_VERSION,
                         "peak_rss_bytes": common.peak_rss_bytes()})
    D.log(f"[stage3] ref {mode}: {time.time() - t0:.0f}s")


def _ref_npz_rows(path: Path, ts, prof):
    logits = np.load(path, mmap_mode="r")
    lens = common.read_json(Path(str(path) + ".done.json"))["seq_lengths"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    names = list(D.ORDER8) + ["T9"]
    if len(lens) != len(names) or logits.shape[0] != offs[-1]:
        raise SystemExit(f"{path.name}: {len(lens)} sequences / {logits.shape[0]} rows, expected 9 / {offs[-1]}")
    return logits, {t: int(offs[j]) for j, t in enumerate(names)}


def step_bugtest(run) -> None:
    import mlx.core as mx

    from gate.checks import g5_generation as g5

    ts = run.textset()
    prof = ts.profile
    refp = run.scratch / "ref_deqK8_fp32.logits.npy"
    if not D.is_done(refp):
        raise SystemExit("stage3: run the ref_fp32 step first")
    logits, off = _ref_npz_rows(refp, ts, prof)
    bug = run.scratch / "g4_K8_fp32_vs_ref_deqK8_fp32.npz"
    c64 = run.scratch / "chunk64.json"
    if D.is_done(bug) and D.is_done(c64):
        return
    model, _ = run.load_arm("K8", fp32=True, dequantised_embedding=True)
    if not D.is_done(bug):
        res = {}
        for tid, ids, _o in texts_and_offsets(ts, prof):
            o = off[tid]
            res[tid] = per_position(model, ids, lambda a, b, o=o: logits[o + a:o + b], prof.prefill_chunk)
            D.log(f"[stage3] bug test {tid}: mean KL {res[tid]['kl'].mean():.3e}")
        save_cols(bug, res, {"arm": "K8", "act": "fp32", "ref": "ref_deqK8_fp32", "chunk": prof.prefill_chunk})
        D.done_marker(bug)
    if not D.is_done(c64):
        big, small = run.ctx.thresholds["G5"]["noise_floor_prefill_steps"]
        texts = {"T1": ts.ids["T1"], "T3": ts.ids["T3"], "T9": ts.t9}
        out = {}
        for name, (lo, hi) in prof.decode_ranges.items():
            ids = texts[name]
            a = D.lsm(g5.prefill_rows(model, ids, lo, hi, big))
            b = D.lsm(g5.prefill_rows(model, ids, lo, hi, small))
            d = D.lsm(g5.decode_rows(model, ids, lo, hi, big))
            lead_a = D.lead_lp(a)
            pairs = {}
            for pname, (p, q) in {"prefill2048_vs_prefill64": (a, b), "prefill2048_vs_decode": (a, d),
                                  "prefill64_vs_decode": (b, d)}.items():
                k = D.kl_lp(p, q)
                ch = p.argmax(-1) != q.argmax(-1)
                pairs[pname] = {"max_kl": float(k.max()), "mean_kl": float(k.mean()),
                                "argmax_kl_pos": int(lo + int(k.argmax())), "n_gt_1e-6": int((k > 1e-6).sum()),
                                "top1_changes": int(ch.sum()),
                                "top1_changes_lead_ge_2": int((ch & (lead_a >= 2.0)).sum())}
            rec = {"positions": [lo, hi], "pairs": pairs}
            if name == "T9":
                o = off["T9"]
                r = D.lsm(np.asarray(logits[o + lo:o + hi + 1], dtype=np.float32))
                lr = D.lead_lp(r)
                for qn, q in (("prefill64", b), ("prefill2048", a), ("decode", d)):
                    k = D.kl_lp(r, q)
                    ch = r.argmax(-1) != q.argmax(-1)
                    dm = lr >= 2.0
                    rec[f"ref_deqK8_fp32_vs_{qn}"] = {"max_kl": float(k.max()), "mean_kl": float(k.mean()),
                                                      "n_gt_1e-6": int((k > 1e-6).sum()),
                                                      "top1_changes": int(ch.sum()),
                                                      "top1_changes_ref_lead_ge_2": int((ch & dm).sum()),
                                                      "n_decisive": int(dm.sum()),
                                                      "top1_decisive": float((~ch[dm]).mean()) if dm.any() else None}
            out[name] = rec
            D.log(f"[stage3] chunk64 {name}: " + ", ".join(f"{k} max {v['max_kl']:.2e}" for k, v in pairs.items()))
        common.write_json(c64, {"steps": [big, small], "ranges": out})
        D.done_marker(c64)
    del model
    run.release()


def step_refcmp(run) -> None:
    ts, dump = run.textset(), run.dump()
    prof = ts.profile
    path = run.scratch / "refcmp_ref_deqK8_emu.npz"
    if D.is_done(path):
        return
    emup = run.scratch / "ref_deqK8_emu.logits.npy"
    if not D.is_done(emup):
        raise SystemExit("stage3: run the ref_emu step first")
    cand, off = _ref_npz_rows(emup, ts, prof)
    ref_logits = dump.load("logits.npy")
    res = {}
    for tid, ids, roff in texts_and_offsets(ts, prof):
        ids = np.asarray(ids)
        cols = empty_cols(ids.size)
        for a in range(0, ids.size, 1024):
            b = min(ids.size, a + 1024)
            ref = np.asarray(ref_logits[roff + a:roff + b], dtype=np.float32)
            got = np.asarray(cand[off[tid] + a:off[tid] + b], dtype=np.float32)
            c = compare_rows(ref, got, ids[a + 1:b + 1])
            for k in COLS:
                cols[k][a:b] = c[k]
        res[tid] = cols
        D.log(f"[stage3] refcmp {tid}: mean KL(fp32 dump || emu) {cols['kl'].mean():.4f}")
    save_cols(path, res, {"arm": "ref_deqK8_emu", "act": "reference bf16 emulation", "ref": "fp32 dump"})
    D.done_marker(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summ(d: dict, m=None, lead=2.0) -> dict:
    m = np.ones(d["kl"].size, bool) if m is None else m
    kl, ag, ld = d["kl"][m], d["agree"][m], d["lead"][m]
    dec = ld >= lead
    dn = d["dnll"][m]
    fin = np.isfinite(dn)
    return {"n": int(m.sum()), "mean_kl": float(kl.mean()) if kl.size else None,
            "max_kl": float(kl.max()) if kl.size else None,
            "dnll_mean": float(dn[fin].mean()) if fin.any() else None,
            "n_decisive": int(dec.sum()), "decisive_miss": int((~ag[dec]).sum()),
            "top1_decisive": float(ag[dec].mean()) if dec.any() else None,
            "miss_rate": float((~ag[dec]).mean()) if dec.any() else None,
            "top1": float(ag.mean()) if ag.size else None,
            "entropy_shift": float((d["ent_got"][m] - d["ent_ref"][m]).mean()) if kl.size else None}


def t9_segments(ts) -> dict:
    t9 = np.asarray(ts.t9)
    out = {}
    for tid in ("T1", "T4", "T3", "T5", "T6"):
        a = np.asarray(ts.ids[tid])
        hits = [s for s in np.flatnonzero(t9[: t9.size - a.size + 1] == a[0]) if np.array_equal(t9[s:s + a.size], a)]
        if hits:
            out[tid] = int(hits[0])
    return out


def doc_boundaries(run, ts) -> list[int]:
    """T9 document starts (gate/texts/MANIFEST.json components, joined by the
    separator's token ids); [] for tiny runs."""
    if run.tiny:
        return []
    man = common.read_json(common.TEXTS_DIR / "MANIFEST.json")["web"]["T9"]
    tok = common.load_raw_tokenizer(common.tokenizer_dir())
    sep = len(tok.encode(man["separator"], add_special_tokens=False).ids)
    starts, p = [], 0
    for c in man["components"]:
        if p >= len(ts.t9):
            break
        starts.append(p)
        p += c["n_tokens"] + sep
    return starts[1:]


def ratio_block(d: dict, m: np.ndarray, seed: int) -> dict:
    dn, kl = d["dnll"][m], d["kl"][m]
    fin = np.isfinite(dn)
    r = D.moving_block_boot(dn[fin], kl[fin], block=64, seed=seed)
    return {"dnll_mean": r["mean"], "dnll_ci95": r["ci"], "kl_mean": float(kl[fin].mean()),
            "dnll_over_kl": r["ratio"], "dnll_over_kl_ci95": r["ratio_ci"]}


def step_report(run) -> None:
    from gate.checks import g3_oracle

    ts = run.textset()
    prof = ts.profile
    th4 = run.ctx.thresholds["G4"]
    lead = th4["K8_backstop_decisive_lead_nats"]
    files = {"bf16": "g4_K8_bf16.npz", "fp32": "g4_K8_fp32.npz", "bug": "g4_K8_fp32_vs_ref_deqK8_fp32.npz",
             "emu": "refcmp_ref_deqK8_emu.npz"}
    for f in list(files.values()) + ["chunk64.json"]:
        if not D.is_done(run.scratch / f):
            raise SystemExit(f"stage3: {f} missing; run the earlier steps")
    R = {k: load_cols(run.scratch / f)[0] for k, f in files.items()}
    cat8 = {k: {c: np.concatenate([R[k][t][c] for t in D.ORDER8]) for c in COLS} for k in R}
    T9 = len(ts.t9)
    pos = np.arange(T9)
    buckets = [tuple(b) for b in prof.t9_buckets]
    bmask = {f"{a}-{b}": (pos >= a) & (pos < b) for a, b in buckets}
    sets9 = {"T9": np.ones(T9, bool), **bmask}
    web_tail = WEB_TAIL_REAL if not run.tiny else (int(0.82 * T9), T9)
    wt_name = f"web_tail_{web_tail[0]}-{web_tail[1]}"
    desc_masks = {**sets9, wt_name: (pos >= web_tail[0]) & (pos < web_tail[1])}
    rec = {"decisive_lead": lead, "t9_buckets": [list(b) for b in buckets], "web_tail": list(web_tail),
           "runs": {"bf16": "K8 bf16 as loaded vs the fp32 dump (the gate's G4)",
                    "fp32": "K8 fp32 activations vs the fp32 dump (quantisation only, descriptive)",
                    "bug": "K8 fp32 activations vs the reference on the dequantised K8 weights (bug test)",
                    "emu": "reference bf16 emulation on the dequantised K8 weights vs the fp32 dump (E_T9)"}}

    # -- summaries per run (descriptive) ------------------------------------------------
    rec["summaries"] = {k: {"T1-8": summ(cat8[k], lead=lead), **{n: summ(R[k]["T9"], m, lead) for n, m in desc_masks.items()},
                            "per_text": {t: summ(R[k][t], lead=lead) for t in D.ORDER8}} for k in R}
    s9 = rec["summaries"]["bf16"]["T9"]
    if not run.tiny:
        rec["reproduction"] = {"T9_top1_decisive": [D.RECORD["g4_T9_top1_decisive"], s9["top1_decisive"]],
                               "T9_n_decisive": [D.RECORD["g4_T9_n_decisive"], s9["n_decisive"]],
                               "exact": s9["top1_decisive"] == D.RECORD["g4_T9_top1_decisive"]
                               and s9["n_decisive"] == D.RECORD["g4_T9_n_decisive"]}

    # -- bug test and its boundary windows ------------------------------------------------
    bug = R["bug"]
    docs = doc_boundaries(run, ts)
    seg = t9_segments(ts)
    bounds = sorted({b for b in range(prof.prefill_chunk, T9, prof.prefill_chunk)} | {b for b in docs if 0 < b < T9})
    near = []
    for b in bounds:
        w = (pos >= b - 32) & (pos < b + 32)
        near.append({"boundary": int(b), "kinds": (["chunk"] if b % prof.prefill_chunk == 0 else []) + (["document"] if b in docs else []),
                     "window_mean_kl": float(bug["T9"]["kl"][w].mean()), "window_max_kl": float(bug["T9"]["kl"][w].max()),
                     "window_decisive_miss": int((~bug["T9"]["agree"][w] & (bug["T9"]["lead"][w] >= lead)).sum())})
    rec["bug_test_boundaries"] = {"boundaries": near, "document_starts": docs,
                                  "T4_offset_check": {"segments": seg, "document_start_1": docs[0] if docs else None}}

    # -- chunk 64 ------------------------------------------------------------------------
    c64 = common.read_json(run.scratch / "chunk64.json")
    rec["chunk64"] = c64
    pairs = [p for r in c64["ranges"].values() for p in r["pairs"].values()]
    vs_ref = (c64["ranges"].get("T9") or {}).get("ref_deqK8_fp32_vs_prefill64") or {}

    # -- McNemar, port bf16 vs emu, on ref-decisive positions ------------------------------------
    P, E = R["bf16"], R["emu"]
    mc = {}
    for name, m in desc_masks.items():
        dec = (P["T9"]["lead"] >= lead) & m
        pm, em = ~P["T9"]["agree"] & dec, ~E["T9"]["agree"] & dec
        n10, n01 = int((pm & ~em).sum()), int((~pm & em).sum())
        mc[name] = {"n_decisive": int(dec.sum()), "port_miss": int(pm.sum()), "emu_miss": int(em.sum()),
                    "n10": n10, "n01": n01, "p_one_sided": D.mcnemar_one_sided(n10, n01)}
    rec["mcnemar_port_bf16_vs_emu"] = mc

    # -- paired dNLL port - emu (g3_oracle.block_bootstrap, 64 blocks, B = 10,000, q = 0.005) ----------
    seed = common.seed_from("diag_gate1", "paired_dnll_T9")
    pd = {}
    for name, m in desc_masks.items():
        dd = (P["T9"]["dnll"] - E["T9"]["dnll"])[m]
        dd = dd[np.isfinite(dd)]
        b = g3_oracle.block_bootstrap(dd, 64, 10_000, seed, 0.005)
        pd[name] = {"mean": b["mean"], "ci99_lo": b["p01"], "ci99_hi": b["p99"], "n": int(dd.size)}
    rec["paired_dnll_port_minus_emu"] = {"seed": seed, "blocks": 64, "B": 10_000, "q": 0.005, "sets": pd}

    # -- dNLL / KL with 95 % CIs (descriptive) ----------------------------------------------------
    rec["dnll_over_kl"] = {k: {"T1-8": ratio_block(cat8[k], np.ones(cat8[k]["kl"].size, bool), 36),
                               **{n: ratio_block(R[k]["T9"], m, 36) for n, m in desc_masks.items()},
                               **{t: ratio_block(R[k][t], np.ones(R[k][t]["kl"].size, bool), 36) for t in D.ORDER8}}
                           for k in ("bf16", "bug", "emu", "fp32")}

    # -- large-KL positions --------------------------------------------------------------------
    big = []
    for t in list(D.ORDER8) + ["T9"]:
        for p_ in np.flatnonzero(P[t]["kl"] > 5.0):
            big.append({"text": t, "pos": int(p_), "kl_port": float(P[t]["kl"][p_]), "kl_emu": float(E[t]["kl"][p_]),
                        "lead_ref": float(P[t]["lead"][p_])})
    rec["max_kl_positions"] = big

    # -- content-matched controls: standalone vs inside T9, same tokens -------------------------
    cm = {}
    for tid, o in seg.items():
        Lt = len(ts.ids[tid])
        row = {"offset_in_T9": o}
        for k in ("bf16", "emu", "fp32"):
            sa = summ(R[k][tid], lead=lead)
            sb = summ(R[k]["T9"], (pos >= o) & (pos < o + Lt), lead)
            row[k] = {"standalone": {x: sa[x] for x in ("mean_kl", "miss_rate", "decisive_miss", "n_decisive")},
                      "in_T9": {x: sb[x] for x in ("mean_kl", "miss_rate", "decisive_miss", "n_decisive")},
                      "excess_kl": sb["mean_kl"] - sa["mean_kl"],
                      "excess_miss_rate": (sb["miss_rate"] - sa["miss_rate"])
                      if sa["miss_rate"] is not None and sb["miss_rate"] is not None else None}
        cm[tid] = row
    rec["content_matched"] = cm

    # -- G4's top-10 KL positions over T1-T8 and T9 (stage 2's tail sign) -----------------------
    allk = [(float(P[t]["kl"][p_]), t, int(p_)) for t in list(D.ORDER8) + ["T9"] for p_ in np.argsort(-P[t]["kl"])[:10]]
    allk.sort(reverse=True)
    top10 = [{"text": t, "position": p_, "kl": k} for k, t, p_ in allk[:10]]

    # -- rule inputs (FROZEN_RULES.md §2, stage 3); stage 5 alone evaluates the rule ------------------
    e9 = rec["summaries"]["emu"]["T9"]
    same = bool(np.array_equal(R["emu"]["T9"]["lead"] >= lead, R["bf16"]["T9"]["lead"] >= lead))
    sb = {"T1-8": summ(cat8["bug"], lead=lead), "T9": summ(bug["T9"], lead=lead),
          **{n: summ(bug["T9"], m, lead) for n, m in bmask.items()}}
    ri = {
        "bf16_T9_top1_decisive": s9["top1_decisive"], "bf16_T9_decisive_miss": s9["decisive_miss"],
        "bf16_T9_n_decisive": s9["n_decisive"],
        "top10_kl_positions": [{"text": x["text"], "position": x["position"]} for x in top10],
        "bug_mean_kl": {"T1-8": sb["T1-8"]["mean_kl"], "T9": sb["T9"]["mean_kl"]},
        "bug_top1_decisive": {"T1-8": sb["T1-8"]["top1_decisive"], **{n: sb[n]["top1_decisive"] for n in bmask}},
        "bug_boundary_window_max": max((n["window_mean_kl"] for n in near), default=None),
        "chunk64_max_kl": max(p["max_kl"] for p in pairs),
        "chunk64_n_decisive_top1_changes": int(sum(p["top1_changes_lead_ge_2"] for p in pairs)),
        "chunk64_vs_ref_mean_kl": vs_ref.get("mean_kl"), "chunk64_vs_ref_top1_decisive": vs_ref.get("top1_decisive"),
        "E_T9": e9["top1_decisive"], "emu_T9_decisive_miss": e9["decisive_miss"], "emu_T9_n_decisive": e9["n_decisive"],
        "emu_decisive_set_equals_gate": same and (run.tiny or e9["n_decisive"] == D.RECORD["g4_T9_n_decisive"]),
        "mcnemar": {n: {k: mc[n][k] for k in ("n10", "n01", "p_one_sided", "port_miss", "emu_miss")} for n in sets9},
        "paired_dnll_T9": {k: pd["T9"][k] for k in ("mean", "ci99_lo", "ci99_hi")},
        "n_kl_port_gt5_emu_lt1": int(sum(1 for b_ in big if b_["kl_emu"] < 1.0)),
        "excess": {t: {"kl_port": cm[t]["bf16"]["excess_kl"], "kl_emu": cm[t]["emu"]["excess_kl"],
                       "miss_port": cm[t]["bf16"]["excess_miss_rate"], "miss_emu": cm[t]["emu"]["excess_miss_rate"]}
                   for t in ("T5", "T6") if t in cm},
    }
    rec = {"rule_inputs": ri, "top10_kl_positions_with_kl": top10, **rec}
    work = {"k8abs_entropy_mean": {k: {"T1-8": float(cat8[k]["ent_got"].mean()), "T9": float(R[k]["T9"]["ent_got"].mean())}
                                   for k in R},
            "npz_files": {k: common.redact_path(run.scratch / f) for k, f in files.items()}}
    run.finish(rec, work)


def main() -> int:
    ap = D.parser("stage3_g4", __doc__)
    ap.add_argument("step", choices=STEPS + ("all",))
    args = ap.parse_args()
    if args.step == "all":
        base = [sys.executable, str(Path(__file__).resolve())]
        extra = sum(([f"--{k.replace('_', '-')}", v] for k, v in (("tiny", args.tiny), ("work_out", args.work_out),
                                                                     ("repo_out", args.repo_out)) if v), [])
        for s in STEPS:
            D.log(f"[stage3] step {s}")
            p = subprocess.run(base + [s] + extra)
            if p.returncode != 0:
                return p.returncode
        return 0
    run = D.Run("stage3_g4", args, PREC)
    run.scratch = run.work_root / "stage3_g4"
    {"g4": step_g4, "t9ids": step_t9ids, "ref_fp32": lambda r: step_ref(r, "fp32"), "bugtest": step_bugtest,
     "ref_emu": lambda r: step_ref(r, "emu"), "refcmp": step_refcmp, "report": step_report}[args.step](run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
