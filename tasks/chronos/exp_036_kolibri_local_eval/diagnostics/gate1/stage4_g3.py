# SPDX-License-Identifier: MIT
"""Stage 4 (DIAGNOSIS §4, no-anchor branch; Andrei's D4 of 2026-10-05): G3, the
reference as an oracle. About 30 minutes on the mbp: the numpy reference on CPU
(the seven mutants on T1 and T5), then K8 (about 80 GB), then Q36-8 (about
37 GB), loaded one after the other in this process and released in between.

  mutants   gate.checks.ref_pass.mutant_nll on T1 and on T5 (the seven
            registered reference mutants and the base stream, one
            layer-streamed pass per text), scored with g3_oracle.ref_mutant_nll
            as the gate scores T3: each mutant's dNLL mean and p01. The base
            stream is compared with the dump's NLL (descriptive).
  D3''      on T1, T2, T5 and T6 (all six reported): per 128-byte window of
            the text's UTF-8 bytes, reference bits / Q36-8 bits, each token's
            NLL attributed to the window holding its last byte, windows where
            either model scores no token dropped; p10 / p50 / p90 and the
            fraction of windows where the reference is better. Kolibri side:
            the dump's NLL under the gate's rule (no prefix, tokens 1..T-1);
            Q36-8: the peer check's raw-text rule (context-start token, every
            token, log-softmax of bf16-rounded logits), which must reproduce
            the peers record the gate cites (g3_bpb_vs_peers.measured.source,
            results/peers_20261005T045649Z.json) within 1e-4 bpb per text
            (rule input peer_bpb_reproduces_record; FROZEN_RULES.md 1.4, 6).
  reference the reference's bpb from token 16 and from token 64 against the
            full text; its NLL on T4 tokens inside Q36-8's recall windows
            (128-byte windows where Q36-8 is under 0.1 bits per byte).
  K8        only D1 validity (K8 - reference, bpb per text: a difference) and
            the <|endoftext|> effect as a difference (K8 with the token before
            the text minus without, same tokens 1..T-1). No chat-wrapped K8 is
            computed, no recall probe is run, and K8's absolute bpb stays in the
            work file (k8abs_*, with the pooled relative <|endoftext|> effect,
            which would give it back). Disclosed limit (FROZEN_RULES.md 1.8):
            per text, reference bpb + d1 gives K8's bpb, as the gate record's
            per-text dnll_mean already does.
  tau*      descriptive: the NLL-optimal temperature of the reference (from the
            dump) and of Q36-8, per text.

The output's rule_inputs are FROZEN_RULES.md §2's stage-4 keys; stage 5 alone
evaluates the frozen G3 rule (§6, no anchor) on them. Everything else is
descriptive.

usage (kit root, env sourced): caffeinate -i "$PY" diagnostics/gate1/stage4_g3.py
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import math  # noqa: E402
import time  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import diaglib as D  # noqa: E402
from gate import common  # noqa: E402

LN2 = math.log(2)
TEXTS = ("T1", "T2", "T3", "T4", "T5", "T6")
D3_TEXTS = ("T1", "T2", "T5", "T6")
MUTANT_TEXTS = ("T1", "T5")
WINDOW = 128
RECALL_BPB = 0.1
PEER_BPB_TOL = 1e-4      # FROZEN_RULES.md 1.4: Q36-8 per-text bpb against the peers record the gate cites
TAUS = np.round(np.arange(0.90, 2.0001, 0.02), 3)


# ---------------------------------------------------------------------------
# Texts and byte alignment
# ---------------------------------------------------------------------------


def load_texts(run, ts) -> dict:
    """{tid: str}. Real: gate/texts T1-T4, $EXP036_WORK/gate_texts T5-T6 (the
    files the peer check read). Tiny: the stub tokenizer's words for the ids."""
    if run.tiny:
        return {t: " ".join(f"t{i}" for i in ts.ids[t]) for t in TEXTS}
    from bench import kl_8v4 as K

    recs = K.load_texts(K.gate_text_paths())
    return {t: recs[t]["text"] for t in TEXTS}


def tokenize(tok_json: Path, text: str):
    from bench import kl_8v4 as K

    return K.tokenize(tok_json, text)


def window_bits(nll, fb, nbytes: int, first_token: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """FROZEN_RULES.md §2 stage 4: a token's NLL (bits) is attributed to the
    128-byte window holding its last byte. nll[i] belongs to token
    first_token + i; a token's last byte is the byte before the next token's
    first byte (the text's last byte for the last token), or its own first
    byte when the next token starts on the same byte (a byte-level token
    splitting a character). Returns (bits per window, tokens per window)."""
    fb = np.asarray(fb, dtype=np.int64)
    nxt = np.append(fb[1:], nbytes)
    last = np.maximum(nxt - 1, fb)
    tok = np.arange(first_token, first_token + len(nll))
    w = last[tok] // WINDOW
    nw = (nbytes + WINDOW - 1) // WINDOW
    bits = np.bincount(w, weights=np.asarray(nll, dtype=np.float64) / LN2, minlength=nw)
    cnt = np.bincount(w, minlength=nw)
    return bits, cnt


# ---------------------------------------------------------------------------
# Model passes
# ---------------------------------------------------------------------------


def tf_nll(model, prefix: list, ids: list, chunk: int = 2048, round_bf16: bool = False, taus=None):
    """Per-token NLL (nats, float64) of ids[1:] (no prefix: the gate's rule) or
    of every token after `prefix`; optionally the summed NLL at each tau."""
    from bench import kl_8v4 as K

    if prefix:
        inputs, score_from = K.scored_input(list(prefix), list(ids))
        targets = list(ids)
    else:
        inputs, score_from, targets = list(ids), 0, list(ids[1:])
    rows, tau_sum = [], np.zeros(len(taus)) if taus is not None else None
    for t0, logits, _ in K.scored_chunks(model, inputs, score_from):
        n = min(logits.shape[0], len(targets) - t0)
        if n <= 0:
            continue
        lg = logits[:n]
        lp = K.logprobs(lg, round_bf16)
        tg = mx.array(np.asarray(targets[t0:t0 + n], dtype=np.int32))[:, None]
        v = -mx.take_along_axis(lp, tg, axis=-1)[:, 0]
        mx.eval(v)
        rows.append(np.array(v, dtype=np.float64))
        if taus is not None:
            for i, tau in enumerate(taus):
                x = lg / float(tau)
                s = mx.sum(mx.logsumexp(x, axis=-1) - mx.take_along_axis(x, tg, axis=-1)[:, 0])
                mx.eval(s)
                tau_sum[i] += float(s)
    out = np.concatenate(rows)
    if out.size != len(targets):
        raise RuntimeError(f"scored {out.size} of {len(targets)} targets")
    return out, tau_sum


def tau_star(tau_sum: np.ndarray, n: int) -> dict:
    i1 = int(np.argmin(np.abs(TAUS - 1.0)))
    io = int(np.argmin(tau_sum))
    return {"tau_star": float(TAUS[io]), "gain_nats_per_token": float((tau_sum[i1] - tau_sum[io]) / n),
            "grid": [float(TAUS[0]), float(TAUS[-1]), 0.02]}


def ref_tau(logits: np.ndarray, ids) -> dict:
    """tau* of the reference from the dump's fp32 logits (rows 0..T-2 predict ids[1:])."""
    ids = np.asarray(ids)
    tot = np.zeros(len(TAUS))
    for a in range(0, ids.size - 1, 512):
        b = min(ids.size - 1, a + 512)
        lg = mx.array(np.asarray(logits[a:b], dtype=np.float32))
        tg = mx.array(ids[a + 1:b + 1].astype(np.int32))[:, None]
        for i, tau in enumerate(TAUS):
            x = lg / float(tau)
            s = mx.sum(mx.logsumexp(x, axis=-1) - mx.take_along_axis(x, tg, axis=-1)[:, 0])
            mx.eval(s)
            tot[i] += float(s)
    return tau_star(tot, ids.size - 1)


def context_id(d: Path) -> int:
    from bench import kl_8v4 as K

    try:
        return int(K.context_prefix(d)["id"])
    except Exception:
        cfg = common.read_json(d / "config.json")
        return int(cfg["pad_token_id"])   # tiny stub tokenizer: Kolibri's <|endoftext|> is its pad id


def mutants_part(run, ts, ref_nll: dict) -> dict:
    """gate.checks.ref_pass.mutant_nll (one layer-streamed pass per text: the
    seven registered mutants plus the base stream), scored with
    g3_oracle.ref_mutant_nll exactly as the gate scores T3 (48 blocks,
    B = 10,000, q = 0.01, the gate's per-mutant seeds)."""
    from gate.checks import g3_oracle, ref_pass

    th3 = run.ctx.thresholds["G3"]
    res = {"texts": list(MUTANT_TEXTS), "path": "gate.checks.ref_pass.mutant_nll + g3_oracle.ref_mutant_nll, per text"}
    for t in MUTANT_TEXTS:
        t0 = time.time()
        nll, path = ref_pass.mutant_nll(run.ctx.bf16_dir, ts.ids[t])
        m = g3_oracle.ref_mutant_nll(nll, th3["ref_mutant_blocks"], th3["ref_mutant_dnll_quantile"])
        repro = float(np.max(np.abs(np.asarray(nll["ref"], dtype=np.float64) - ref_nll[t])))
        res[t] = {"mutants": m, "base_stream_vs_dump_max_abs_dnll": repro, "wall_s": time.time() - t0,
                  "min_p01": min(v["dnll_p01"] for v in m.values())}
        D.log(f"[stage4] mutants {t}: min p01 {res[t]['min_p01']:.3f} (base vs dump {repro:.2e}, {time.time() - t0:.0f}s)")
        run.release()
    return res


def gate_peers_record(run) -> Path:
    """The peers record the gate's g3_bpb_vs_peers cites (not the newest one)."""
    src = run.gate_record()["checks"]["g3_bpb_vs_peers"]["measured"]["source"]
    p = D.KIT / "results" / Path(src).name
    if not p.is_file():
        raise SystemExit(f"stage4: the peers record the gate cites ({Path(src).name}) is missing")
    return p


def main() -> int:
    ap = D.parser("stage4_g3", __doc__)
    ap.add_argument("--peer-dir", help="default the Q36-8 arm directory (tiny: the tiny K4 build)")
    ap.add_argument("--peers-record", help="default the record the gate's g3_bpb_vs_peers cites (peers_20261005T045649Z.json)")
    ap.add_argument("--skip", default="", help="comma list of parts to skip (smoke tests): mutants,tau")
    args = ap.parse_args()
    skip = {x for x in args.skip.split(",") if x}
    run = D.Run("stage4_g3", args, PREC)
    ts, dump = run.textset(), run.dump()
    prof = ts.profile
    L = prof.text_len
    th3 = run.ctx.thresholds["G3"]
    logits = dump.load("logits.npy")
    rec, work = {"texts": list(TEXTS), "window_bytes": WINDOW}, {}

    # ---- the reference, from the dump (deterministic: the gate's own numbers) -------------
    from gate.checks import ref_pass

    ref_nll, ref = {}, {}
    for j, t in enumerate(D.ORDER8):
        if t not in TEXTS:
            continue
        nll = ref_pass.next_token_nll(logits[j * L:(j + 1) * L], ts.ids[t])
        tb = ts.token_bytes[t]
        ref_nll[t] = nll
        full = float(nll.sum() / LN2 / sum(tb[1:]))
        row = {"bpb": full, "nll_per_token": float(nll.mean())}
        for k in (16, 64):
            v = float(nll[k - 1:].sum() / LN2 / sum(tb[k:]))
            row[f"bpb_from_token{k}"] = v
            row[f"rel_diff_from_token{k}"] = (v - full) / full
        ref[t] = row
    rec["reference"] = ref
    if not run.tiny:
        rec_bpb = run.phase(2)["checks"]["g3_bpb_per_text"]["measured"]["per_text_bpb"]
        rec["reference_reproduction"] = {t: {"record": rec_bpb[t], "diag": ref[t]["bpb"],
                                             "equal": rec_bpb[t] == ref[t]["bpb"]} for t in TEXTS}
        rec["reference_reproduction_spread_max"] = max(abs(rec_bpb[t] - ref[t]["bpb"]) for t in TEXTS)
    if "tau" not in skip:
        rec["reference_tau_star"] = {t: ref_tau(logits[j * L:(j + 1) * L], ts.ids[t])
                                     for j, t in enumerate(D.ORDER8) if t in TEXTS}

    # ---- the seven mutants on T1 and T5 -------------------------------------------------
    if "mutants" not in skip:
        rec["mutants"] = mutants_part(run, ts, ref_nll)
        run.release()

    # ---- K8: D1 validity and the <|endoftext|> effect (differences only) ----------------------
    texts = load_texts(run, ts)
    kdir = run.ctx.arm_dir("K8")
    k_fb = {}
    for t in TEXTS:
        ids, fb = tokenize(kdir / "tokenizer.json", texts[t])
        if list(ids) != list(ts.ids[t]):
            raise SystemExit(f"stage4: tokenising {t} with the K8 tokenizer does not give the gate ids")
        k_fb[t] = fb
    model, _ = run.load_arm("K8")
    eot = context_id(kdir)
    d1, eot_eff, k8abs = {}, {}, {}
    for t in TEXTS:
        ids = list(ts.ids[t])
        n0, _ = tf_nll(model, [], ids)
        ne, _ = tf_nll(model, [eot], ids)
        nb1 = sum(ts.token_bytes[t][1:])
        b0 = float(n0.sum() / LN2 / nb1)
        d1[t] = {"k8_minus_ref_bpb": b0 - ref[t]["bpb"],
                 "k8_minus_ref_nats_per_token": float((n0 - ref_nll[t]).mean())}
        eot_eff[t] = {"delta_bpb_tokens1plus": float((ne[1:].sum() - n0.sum()) / LN2 / nb1)}
        k8abs[t] = {"bpb_noprefix": b0, "bpb_eot_tokens1plus": float(ne[1:].sum() / LN2 / nb1),
                    "bpb_eot_all_tokens": float(ne.sum() / LN2 / sum(ts.token_bytes[t]))}
    nb = sum(sum(ts.token_bytes[t][1:]) for t in TEXTS)
    pooled0 = sum(k8abs[t]["bpb_noprefix"] * sum(ts.token_bytes[t][1:]) for t in TEXTS) / nb
    pooledE = sum(k8abs[t]["bpb_eot_tokens1plus"] * sum(ts.token_bytes[t][1:]) for t in TEXTS) / nb
    rec["D1_validity"] = {"per_text": d1, "max_abs_k8_minus_ref_bpb": max(abs(v["k8_minus_ref_bpb"]) for v in d1.values()),
                          "limit": 0.005}
    rec["D1_validity"]["valid"] = rec["D1_validity"]["max_abs_k8_minus_ref_bpb"] <= 0.005
    rec["eot_effect"] = {"per_text": eot_eff, "pooled_delta_bpb_tokens1plus": pooledE - pooled0,
                         "context_token_id_rule": "bench.kl_8v4.context_prefix"}
    # pooled_rel_delta = delta / pooled0 would give K8's pooled absolute bpb: work file only (FROZEN_RULES.md 1.8)
    work["k8abs_bpb"] = {"per_text": k8abs, "pooled_noprefix": pooled0, "pooled_eot_tokens1plus": pooledE,
                         "pooled_rel_delta": (pooledE - pooled0) / pooled0}
    del model
    run.release()

    # ---- Q36-8 (the peer check's raw-text rule) ---------------------------------------------
    if run.tiny:
        pdir = Path(args.peer_dir) if args.peer_dir else run.ctx.arm_dir("K4")
        peer = common.load_port(pdir)[0]
    else:
        from bench import common as bc
        from tools import peer_check

        pdir = Path(args.peer_dir) if args.peer_dir else bc.arm_dir("Q36-8")
        peer = peer_check._load(pdir)
    peot = context_id(pdir)
    q, qtau = {}, {}
    for t in TEXTS:
        qids, qfb = tokenize(pdir / "tokenizer.json", texts[t])
        qn, ts_ = tf_nll(peer, [peot], qids, round_bf16=True, taus=None if "tau" in skip else TAUS)
        nbytes = len(texts[t].encode("utf-8"))
        q[t] = {"ids_n": len(qids), "fb": qfb, "nll": qn, "bpb": float(qn.sum() / LN2 / nbytes), "bytes": nbytes}
        if ts_ is not None:
            qtau[t] = tau_star(ts_, len(qids))          # fp32 logits (the tau sums never round)
    del peer
    run.release()
    peers_rec = None
    if not run.tiny:
        prp = Path(args.peers_record) if args.peers_record else gate_peers_record(run)
        peers_rec = common.read_json(prp)["arms"]["Q36-8"]["nll"]["per_text_bpb"]
        rec["peer_reproduction"] = {"record": prp.name, "record_sha256": D.sha256_file(prp), "tolerance_bpb": PEER_BPB_TOL,
                                    **{t: {"record": peers_rec[t], "diag": q[t]["bpb"], "diff": q[t]["bpb"] - peers_rec[t]}
                                       for t in TEXTS}}
        rec["peer_reproduction"]["max_abs_diff"] = max(abs(q[t]["bpb"] - peers_rec[t]) for t in TEXTS)
    rec["peer"] = {"arm": "Q36-8" if not run.tiny else "tiny K4 stand-in", "per_text_bpb": {t: q[t]["bpb"] for t in TEXTS},
                   "rule": "tools/peer_check.text_stats: context-start token, every token, log-softmax of bf16-rounded logits"}
    if qtau:
        rec["peer_tau_star"] = qtau

    # ---- D3'': per 128-byte window, reference bits / Q36-8 bits (last-byte attribution) --------
    d3 = {}
    for t in TEXTS:
        nbytes = len(texts[t].encode("utf-8"))
        wk, ck = window_bits(ref_nll[t], k_fb[t], nbytes, first_token=1)
        wq, cq = window_bits(q[t]["nll"], q[t]["fb"], nbytes)
        keep = (ck > 0) & (cq > 0)
        ratio = wk[keep] / wq[keep]
        gap = wk[keep] - wq[keep]
        p10, p50, p90 = (float(x) for x in np.percentile(ratio, [10, 50, 90]))
        row = {"windows": int(keep.sum()), "windows_dropped": int((~keep).sum()), "p10": p10, "p50": p50, "p90": p90,
               "frac_kolibri_better": float((wk[keep] < wq[keep]).mean()), "gap_bits": float(gap.sum()),
               "top10pct_windows_share_of_gap": float(np.sort(gap)[::-1][:max(1, gap.size // 10)].sum() / gap.sum())
               if gap.sum() else None}
        if t == "T4":
            nb_w = np.minimum(WINDOW, nbytes - np.arange(len(wq)) * WINDOW)
            rec_w = keep & (wq / nb_w < RECALL_BPB)
            fbk = np.asarray(k_fb[t], dtype=np.int64)
            lastk = np.maximum(np.append(fbk[1:], nbytes) - 1, fbk)[1:]       # tokens 1..T-1
            inside = rec_w[lastk // WINDOW]
            row["recall_windows"] = {
                "rule": f"128-byte windows where Q36-8 is under {RECALL_BPB} bits per byte (descriptive)",
                "n_windows": int(rec_w.sum()), "n_tokens_inside": int(inside.sum()), "n_tokens_outside": int((~inside).sum()),
                "reference_nll_per_token_inside": float(ref_nll[t][inside].mean()) if inside.any() else None,
                "reference_nll_per_token_outside": float(ref_nll[t][~inside].mean()) if (~inside).any() else None,
                "reference_bits_inside": float(wk[rec_w].sum()), "q36_8_bits_inside": float(wq[rec_w].sum()),
                "gap_share_inside": float((wk[rec_w].sum() - wq[rec_w].sum()) / gap.sum()) if gap.sum() else None}
        d3[t] = row
    rec["D3pp_windows"] = d3
    rec["rule_inputs"] = {
        "ref_bpb_reproduces_gate": (all(v["equal"] for k, v in rec["reference_reproduction"].items())
                                    if "reference_reproduction" in rec else None),
        "d1_k8_minus_ref_bpb": {t: d1[t]["k8_minus_ref_bpb"] for t in TEXTS},
        "mutant_dnll_p01": ({m: {t: rec["mutants"][t]["mutants"][m]["dnll_p01"] for t in MUTANT_TEXTS}
                             for m in rec["mutants"][MUTANT_TEXTS[0]]["mutants"]} if "mutants" in rec else None),
        "mutant_dnll_mean": ({m: {t: rec["mutants"][t]["mutants"][m]["dnll_mean"] for t in MUTANT_TEXTS}
                              for m in rec["mutants"][MUTANT_TEXTS[0]]["mutants"]} if "mutants" in rec else None),
        "window_ratio": {t: {k: d3[t][k] for k in ("p10", "p50", "p90", "frac_kolibri_better")} for t in D3_TEXTS},
        "ref_bpb_from64_rel_diff": {t: ref[t]["rel_diff_from_token64"] for t in TEXTS},
        "peer_bpb_max_abs_diff": (rec["peer_reproduction"]["max_abs_diff"] if "peer_reproduction" in rec else None),
        "peer_bpb_reproduces_record": (rec["peer_reproduction"]["max_abs_diff"] <= PEER_BPB_TOL
                                       if "peer_reproduction" in rec else None),
    }
    rec = {"rule_inputs": rec.pop("rule_inputs"), **rec}
    run.finish(rec, work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
