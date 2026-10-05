# SPDX-License-Identifier: MIT
"""Stage 1 (DIAGNOSIS §4): G5 batch parity of K8, defect tests and localisers.
About 35-45 minutes on the mbp; K8 is loaded twice (bf16, then fp32
activations), then the numpy reference runs on CPU; one model at a time.

Parts, in order (the gate's own code: gate/checks/g5_generation.py and the
runner's BatchGenerator construction; nothing in the kit is edited):

  B   bf16 as loaded (production settings). The gate's batch_parity, which
      must reproduce 0.3073773544 (record), then an instrumented copy of the
      same loop keeping per-sequence tokens and log-probs. Per position:
      KL(single||batched) (the gate statistic; single = teacher-forced prefill,
      chunk 2048), KL(single||decode_tf), KL(decode_tf||batched), the matched
      floor KL(single2048||single64), the lead of single, flips where single
      leads by >= 2 nats. Classes: first wave / mid-run, t = 0 / 1-2 / 3+,
      prompt >= window; the top-5 positions' share of the KL sum.
  L1  identical-prompt localiser: B = 8 copies of prompt 2 (T3[:700]) against
      B = 1 through the same BatchGenerator, 24 tokens; KL per step up to the
      first greedy divergence; are the 8 rows bitwise equal.
  L2  B = 1 BatchGenerator against decode_tf on the same tokens (prompts 0, 2
      and 3: below, around and above the window); max KL.
  L3  between waves: batched log-probs of sequences 0 and 8 (both T1[:37]) and
      of 3 and 11 (both T4[:1100]) over their common token prefix.
  C   fp32 activations on the real K8 weights (model.set_dtype(float32),
      MLX_ENABLE_TF32=0): the same batch_parity and decomposition (mechanics).
  R   the fp32 reference (reference/kolibri_ref.py on the BF16 weights, the
      phase-5 machinery) teacher-forced on the 12 batched sequences (prompt +
      batched tokens). Per position KL(ref||single), KL(ref||batched),
      KL(ref||decode_tf); decisive (ref lead >= 2) top-1 of each path; on
      ref-decisive positions n10 = (batched wrong, single right) and n01.
  Q   optional (--with-q36, descriptive): the peer check's batched-path rule
      on Q36-8 with fp32 activations.

The output's rule_inputs are FROZEN_RULES.md §2's stage-1 keys; stage 5 alone
evaluates the frozen G5 rule (§3) on them. Everything else is descriptive.

usage (kit root, env sourced): caffeinate -i "$PY" diagnostics/gate1/stage1_g5.py [--with-q36]
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import time  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import diaglib as D  # noqa: E402
from gate import common  # noqa: E402
from gate.checks import g5_generation as g5  # noqa: E402
from gate.harness import logits_at  # noqa: E402

L1_STEPS = 24
L2_PROMPTS = (0, 2, 3)
L3_PAIRS = ((0, 8), (3, 11))


def instrumented(model, prompts, max_tokens, B, eos):
    """Verbatim admission logic of g5_generation.batch_parity, keeping per-sequence data."""
    gen, _ = g5._batch_generator(model, B, eos, max(max_tokens))
    got, queue = {}, list(range(len(prompts)))
    uid_of, in_flight, step, any_finished, admit = {}, 0, 0, False, {}
    try:
        while queue or in_flight:
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                for u, j in zip(uids, group):
                    uid_of[u] = j
                    got[u] = {"tokens": [], "lp": []}
                    admit[j] = {"step": step, "mid_run": any_finished}
                in_flight += len(group)
            _, resps = gen.next()
            for r in resps:
                g = got[r.uid]
                g["tokens"].append(int(r.token))
                x = r.logprobs.astype(mx.float32)
                mx.eval(x)
                g["lp"].append(np.array(x))
                if r.finish_reason is not None:
                    in_flight -= 1
                    any_finished = True
            step += 1
            if step > 100_000:
                raise RuntimeError("instrumented batch run did not finish")
    finally:
        gen.close()
    by_j = {j: got[u] for u, j in uid_of.items()}
    return [by_j[j] for j in range(len(prompts))], admit


def batched(model, prompts, max_tokens, B, eos):
    """Plain batched run (no admission bookkeeping): [{"tokens", "lp"}] by prompt index."""
    res, _ = instrumented(model, prompts, max_tokens, B, eos)
    return res


def decode_tf(model, prompt, toks, step=2048):
    """One sequence, the port's plain caches: prefill prompt[:-1], then feed
    prompt[-1] and the given tokens one at a time; the row of each step."""
    cache = model.make_cache()
    ids = np.asarray(prompt, dtype=np.int32)
    for a in range(0, ids.size - 1, step):
        mx.eval(model(mx.array(ids[a:min(ids.size - 1, a + step)])[None], cache=cache))
    rows, cur = [], int(ids[-1])
    for t in toks:
        o = model(mx.array([[cur]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(o)
        rows.append(np.array(o))
        cur = int(t)
    return np.stack(rows)


def summarise_positions(pos: list, W: int) -> tuple[dict, dict]:
    """(restricted summary, work-only extras). K8's own top-1 lead is a K8
    confidence number, not a comparison (FROZEN_RULES.md 1.8): the count of
    positions where it leads by >= 2 nats and the per-position lead stay in
    the work file."""
    A = {k: np.array([r[k] for r in pos]) for k in pos[0]}
    K = A["kl_sb"]
    srt = np.sort(K)[::-1]

    def m(mask, key="kl_sb"):
        return float(A[key][mask].mean()) if mask.any() else None

    classes = {
        "first_wave": ~A["mid_run"], "mid_run": A["mid_run"], "t0": A["t"] == 0,
        "t1_2": (A["t"] >= 1) & (A["t"] <= 2), "t3plus": A["t"] >= 3,
        f"plen_ge_{W}": A["plen"] >= W, f"plen_lt_{W}": A["plen"] < W,
        f"abs_pos_ge_{W}": A["abs_pos"] >= W, f"abs_pos_lt_{W}": A["abs_pos"] < W,
    }
    out = {
        "n_positions": int(K.size), "mean_kl_single_batched": float(K.mean()),
        "top1_dis_single_batched": float(A["flip_sb"].mean()),
        "mean_kl_single_decode": float(A["kl_sd"].mean()), "mean_kl_decode_batched": float(A["kl_db"].mean()),
        "share_of_kl_sum_top5_positions": float(srt[:5].sum() / max(K.sum(), 1e-300)),
        "share_of_kl_sum_top10_positions": float(srt[:10].sum() / max(K.sum(), 1e-300)),
        "flips_where_single_leads_2": int((A["flip_sb"] & (A["lead_single"] >= 2.0)).sum()),
        "by_class": {name: {"n": int(msk.sum()), "kl_sb": m(msk), "kl_sd": m(msk, "kl_sd"), "kl_db": m(msk, "kl_db"),
                            "top1_dis_sb": float(A["flip_sb"][msk].mean()) if msk.any() else None}
                     for name, msk in classes.items()},
    }
    if "kl_floor" in A and np.isfinite(A["kl_floor"]).all():
        fl = float(A["kl_floor"].mean())
        out["mean_kl_floor_matched"] = fl
        out["ratio_to_matched_floor"] = out["mean_kl_single_batched"] / fl if fl > 0 else None
        for name, msk in classes.items():
            out["by_class"][name]["kl_floor"] = m(msk, "kl_floor")
    out["ratio_to_decode_vs_prefill"] = (out["mean_kl_single_batched"] / out["mean_kl_single_decode"]
                                         if out["mean_kl_single_decode"] > 0 else None)
    order = np.argsort(-K)[:10]
    out["top10_positions"] = [{k: (bool(A[k][i]) if A[k].dtype == bool else float(A[k][i]) if A[k].dtype.kind == "f"
                                   else int(A[k][i]))
                               for k in ("j", "t", "abs_pos", "plen", "mid_run", "kl_sb", "kl_sd", "kl_db",
                                         "kl_floor", "flip_sb") if k in A}
                              for i in order]
    extra = {"n_single_leads_2": int((A["lead_single"] >= 2.0).sum()),
             "top10_positions_lead_single": [float(A["lead_single"][i]) for i in order]}
    return out, extra


def decompose(model, prompts, res, admit, W, with_floor=True, keep=None):
    """Per-position decomposition of one batched run `res` (from instrumented)."""
    pos, seqs = [], []
    for j, (p, g) in enumerate(zip(prompts, res)):
        P, n = len(p), len(g["tokens"])
        seq = list(p) + g["tokens"][:-1]
        at = np.arange(P - 1, P - 1 + n)
        single = D.lsm(logits_at(model, seq, at))
        bat = D.lsm(np.stack(g["lp"]))
        dec = D.lsm(decode_tf(model, p, g["tokens"]))
        s64 = D.lsm(logits_at(model, seq, at, chunk=64)) if with_floor else None
        k_sb, k_sd, k_db = D.kl_lp(single, bat), D.kl_lp(single, dec), D.kl_lp(dec, bat)
        k_fl = D.kl_lp(single, s64) if with_floor else np.full(n, np.nan)
        lead = D.lead_lp(single)
        flip = single.argmax(-1) != bat.argmax(-1)
        ent = D.entropy_lp(single)
        if keep is not None:
            keep["single"].append(single.astype(np.float32))
            keep["batched"].append(bat.astype(np.float32))
            keep["decode"].append(dec.astype(np.float32))
        seqs.append({"j": j, "P": P, "n": n, "mid_run": admit[j]["mid_run"], "admit_step": admit[j]["step"],
                     "kl_single_batched": float(k_sb.mean()), "kl_single_decode": float(k_sd.mean()),
                     "kl_decode_batched": float(k_db.mean()),
                     "kl_floor_matched": float(np.mean(k_fl)) if with_floor else None,
                     "dis_single_batched": float(flip.mean()),
                     "dis_decode_batched": float((dec.argmax(-1) != bat.argmax(-1)).mean()),
                     "max_abs_dlogprob_single_batched": float(np.abs(single - bat).max()),
                     "max_abs_dlogprob_decode_batched": float(np.abs(dec - bat).max()),
                     "first_step_dlogprob_gt_1e-3_decode_batched": next(
                         (int(t) for t in range(n) if np.abs(dec[t] - bat[t]).max() > 1e-3), None),
                     "kl_sb_max": float(k_sb.max()), "kl_sb_argmax_t": int(k_sb.argmax())})
        for t in range(n):
            pos.append({"j": j, "t": t, "abs_pos": int(P - 1 + t), "plen": P, "mid_run": bool(admit[j]["mid_run"]),
                        "kl_sb": float(k_sb[t]), "kl_sd": float(k_sd[t]), "kl_db": float(k_db[t]),
                        "kl_floor": float(k_fl[t]), "lead_single": float(lead[t]), "flip_sb": bool(flip[t]),
                        "entropy_single": float(ent[t])})
    summ, extra = summarise_positions([{k: v for k, v in r.items() if k != "entropy_single"} for r in pos], W)
    return summ, seqs, pos, extra


def localisers(model, prompts, res, admit, eos):
    out = {}
    # L1: B = 8 copies of prompt 2 against B = 1, through the same BatchGenerator construction.
    p = prompts[2]
    r8 = batched(model, [p] * 8, [L1_STEPS] * 8, 8, eos)
    r1 = batched(model, [p], [L1_STEPS], 1, eos)
    t1, t8 = r1[0]["tokens"], r8[0]["tokens"]
    m = min(len(t1), len(t8))
    div = next((i for i in range(m) if t1[i] != t8[i]), m)
    n = div + 1 if div < m else div
    kl = D.kl_lp(D.lsm(np.stack(r1[0]["lp"][:n])), D.lsm(np.stack(r8[0]["lp"][:n])))
    rows_eq = all(r["tokens"] == r8[0]["tokens"] and all(np.array_equal(a, b) for a, b in zip(r["lp"], r8[0]["lp"]))
                  for r in r8)
    out["L1_identical_prompt"] = {
        "prompt_index": 2, "prompt_len": len(p), "steps": L1_STEPS, "first_divergence_step": int(div),
        "kl_per_step_until_divergence": [float(x) for x in kl],
        "mean_kl_before_divergence": float(kl[:div].mean()) if div > 0 else None,
        "max_kl_before_divergence": float(kl[:div].max()) if div > 0 else None,
        "B8_rows_bitwise_equal": bool(rows_eq),
        "B8_row0_bitwise_equal_B1_before_divergence": bool(all(np.array_equal(r1[0]["lp"][i], r8[0]["lp"][i])
                                                              for i in range(div)))}
    # L2: B = 1 BatchGenerator against decode_tf on the same tokens.
    l2 = {}
    for pi in L2_PROMPTS:
        q = prompts[pi]
        r = r1 if pi == 2 else batched(model, [q], [L1_STEPS], 1, eos)
        b1 = D.lsm(np.stack(r[0]["lp"]))
        d1 = D.lsm(decode_tf(model, q, r[0]["tokens"]))
        k = D.kl_lp(d1, b1)
        l2[f"prompt{pi}_len{len(q)}"] = {"n": int(k.size), "max_kl": float(k.max()), "mean_kl": float(k.mean()),
                                         "max_abs_dlogprob": float(np.abs(d1 - b1).max()),
                                         "argmax_equal_all_steps": bool((d1.argmax(-1) == b1.argmax(-1)).all())}
    out["L2_B1_batchgen_vs_decode"] = l2
    out["L2_max_kl"] = max(v["max_kl"] for v in l2.values())
    # L3: first-wave and mid-run copies of the same prompt, from the instrumented run.
    l3 = {}
    for a, b in L3_PAIRS:
        if b >= len(prompts) or list(prompts[a]) != list(prompts[b]):
            l3[f"{a}_{b}"] = {"skipped": "prompts differ"}
            continue
        ta, tb = res[a]["tokens"], res[b]["tokens"]
        m = min(len(ta), len(tb))
        div = next((i for i in range(m) if ta[i] != tb[i]), m)
        n = div + 1 if div < m else div
        k = D.kl_lp(D.lsm(np.stack(res[a]["lp"][:n])), D.lsm(np.stack(res[b]["lp"][:n])))
        l3[f"{a}_{b}"] = {"prompt_len": len(prompts[a]), "first_wave": [not admit[a]["mid_run"], not admit[b]["mid_run"]],
                          "admit_step": [admit[a]["step"], admit[b]["step"]], "n_compared": int(n),
                          "first_token_divergence": int(div), "mean_kl": float(k.mean()) if n else None,
                          "max_kl": float(k.max()) if n else None,
                          "max_abs_dlogprob": float(max(np.abs(res[a]["lp"][i] - res[b]["lp"][i]).max() for i in range(n)))
                          if n else None,
                          "bitwise_equal": bool(all(np.array_equal(res[a]["lp"][i], res[b]["lp"][i]) for i in range(n)))}
    out["L3_between_waves"] = l3
    return out


def reference_part(run, prompts, res, keep, lead_min) -> tuple[dict, dict]:
    """R: the fp32 reference on prompt + batched tokens, against the three K8 paths."""
    from gate.checks import ref_pass

    seqs = [list(p) + g["tokens"][:-1] for p, g in zip(prompts, res)]
    t0 = time.time()
    rr = ref_pass.RefRunner(run.ctx.bf16_dir, "fp32")
    logits, _, _, lengths = rr.ref.forward_packed(seqs)
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    rows = np.concatenate([bounds[j] + len(p) - 1 + np.arange(len(g["tokens"]))
                           for j, (p, g) in enumerate(zip(prompts, res))])
    ref = D.lsm(logits[rows])
    del logits
    single, bat, dec = (np.concatenate(keep[k]).astype(np.float64) for k in ("single", "batched", "decode"))
    k_rs, k_rb, k_rd = D.kl_lp(ref, single), D.kl_lp(ref, bat), D.kl_lp(ref, dec)
    lead = D.lead_lp(ref)
    top = ref.argmax(-1)
    ok_s, ok_b, ok_d = single.argmax(-1) == top, bat.argmax(-1) == top, dec.argmax(-1) == top
    dec_m = lead >= lead_min
    n10 = int((dec_m & ~ok_b & ok_s).sum())
    n01 = int((dec_m & ~ok_s & ok_b).sum())
    summ = {
        "n_positions": int(ref.shape[0]), "n_ref_decisive": int(dec_m.sum()), "decisive_lead_nats": lead_min,
        "mean_kl_ref_single": float(k_rs.mean()), "mean_kl_ref_batched": float(k_rb.mean()),
        "mean_kl_ref_decode": float(k_rd.mean()),
        "ratio_batched_over_single": float(k_rb.mean() / k_rs.mean()) if k_rs.mean() > 0 else None,
        "decisive_top1_single": float(ok_s[dec_m].mean()) if dec_m.any() else None,
        "decisive_top1_batched": float(ok_b[dec_m].mean()) if dec_m.any() else None,
        "decisive_top1_decode": float(ok_d[dec_m].mean()) if dec_m.any() else None,
        "n10_batched_wrong_single_right": n10, "n01_single_wrong_batched_right": n01,
        "top1_single": float(ok_s.mean()), "top1_batched": float(ok_b.mean()),
        "reference": "reference.kolibri_ref fp32 on the BF16 weights (forward_packed, the phase-5 machinery)",
        "wall_s": time.time() - t0,
    }
    per = {"kl_ref_single": k_rs.tolist(), "kl_ref_batched": k_rb.tolist(), "kl_ref_decode": k_rd.tolist(),
           "ref_lead": lead.tolist()}
    return summ, per


def q36_part(run) -> tuple[dict, dict]:
    """Optional, descriptive: the peer check's batched path on Q36-8 with fp32 activations."""
    from bench import common as bc
    from tools import peer_check

    d = bc.arm_dir("Q36-8")
    model = peer_check._load(d)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    r = peer_check.batched_path(model, d, "Q36-8")
    del model
    run.release()
    wrapper = r.pop("wrapper", None)
    r["ratio_to_floor"] = r["parity"]["mean_kl"] / max(r["noise_floor"]["floor_kl"], 1e-300)
    return {"Q36_8_fp32_activations": r}, {"Q36_8_wrapper": wrapper}


def rule_inputs(rec: dict, W: int) -> dict:
    """FROZEN_RULES.md §2, stage 1: the only values stage 5 reads (None where a part did not run)."""
    bp = rec["B_gate_batch_parity_bf16"]
    dec = rec["B_decomposition_bf16"]
    cls = dec["by_class"]
    r = rec.get("R_reference_teacher_forcing") or {}
    c = rec.get("C_gate_batch_parity_fp32act") or {}
    l1 = rec["L1_identical_prompt"]
    l3 = rec["L3_between_waves"]
    return {
        "B_mean_kl": bp["mean_kl"], "B_top1_dis": bp["top1_dis"],
        "B_instrumented_run_equals_gate_run": rec["B_instrumented_run_equals_gate_run"],
        "R_mean_kl_ref_single": r.get("mean_kl_ref_single"), "R_mean_kl_ref_batched": r.get("mean_kl_ref_batched"),
        "R_mean_kl_ref_decode_tf": r.get("mean_kl_ref_decode"), "R_n_decisive": r.get("n_ref_decisive"),
        "R_top1_decisive_single": r.get("decisive_top1_single"), "R_top1_decisive_batched": r.get("decisive_top1_batched"),
        "R_n_batched_wrong_single_right": r.get("n10_batched_wrong_single_right"),
        "R_n_single_wrong_batched_right": r.get("n01_single_wrong_batched_right"),
        "C_mean_kl": c.get("mean_kl"), "C_top1_dis": c.get("top1_dis"), "C_n_positions": c.get("n_positions"),
        "L1_rows_bitwise_equal": l1["B8_rows_bitwise_equal"], "L1_max_pre_divergence_kl": l1["max_kl_before_divergence"],
        "L1_first_divergence_step": l1["first_divergence_step"],
        "L2_max_kl": rec["L2_max_kl"],
        "L3_max_abs_dlogprob_seq0_seq8": (l3.get("0_8") or {}).get("max_abs_dlogprob"),
        "L3_max_abs_dlogprob_seq3_seq11": (l3.get("3_11") or {}).get("max_abs_dlogprob"),
        "B_share_kl_top5": dec["share_of_kl_sum_top5_positions"],
        "B_mean_kl_first_wave": cls["first_wave"]["kl_sb"], "B_mean_kl_mid_run": cls["mid_run"]["kl_sb"],
        "B_mean_kl_t0": cls["t0"]["kl_sb"], "B_mean_kl_t1_2": cls["t1_2"]["kl_sb"], "B_mean_kl_t3plus": cls["t3plus"]["kl_sb"],
        f"B_mean_kl_prompt_ge_{W}": cls[f"plen_ge_{W}"]["kl_sb"],
    }


def main() -> int:
    ap = D.parser("stage1_g5", __doc__)
    ap.add_argument("--with-q36", action="store_true", help="also the optional descriptive Q36-8 fp32 check")
    ap.add_argument("--skip", default="", help="comma list of parts to skip (smoke tests): C,R")
    args = ap.parse_args()
    skip = {x for x in args.skip.split(",") if x}
    run = D.Run("stage1_g5", args, PREC)
    ts = run.textset()
    prof = ts.profile
    th5 = run.ctx.thresholds["G5"]
    B = th5["batch_B"]
    eos = common.EOS_IDS
    prompts = [ts.ids[D.ORDER8[j % 8]][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
    MT = list(prof.batch_max_tokens)
    rec, work = {"profile": {"batch_lengths": list(prof.batch_lengths), "batch_max_tokens": MT, "B": B}}, {}

    # ---- B, L1-L3: bf16 as loaded --------------------------------------------
    model, _ = run.load_arm("K8")
    W = model.args.sliding_window
    t0 = time.time()
    bp = g5.batch_parity(model, prompts, MT, B)
    rec["B_gate_batch_parity_bf16"] = bp
    if not run.tiny:
        rec["B_reproduction"] = {"record_mean_kl": D.RECORD["g5_batch_parity_K8_mean_kl"],
                                 "mean_kl_equal": bp["mean_kl"] == D.RECORD["g5_batch_parity_K8_mean_kl"],
                                 "mean_kl_diff": bp["mean_kl"] - D.RECORD["g5_batch_parity_K8_mean_kl"],
                                 "top1_dis_diff": bp["top1_dis"] - D.RECORD["g5_batch_parity_K8_top1_dis"],
                                 "registered_kl_max": D.RECORD["g5_kl_max"],
                                 "above_registered_bound": bp["mean_kl"] > D.RECORD["g5_kl_max"]}
    D.log(f"[stage1] B gate batch_parity {bp['mean_kl']:.10f} top1_dis {bp['top1_dis']:.4f} ({time.time() - t0:.0f}s)")
    res, admit = instrumented(model, prompts, MT, B, eos)
    keep = {"single": [], "batched": [], "decode": []}
    summ, seqs, pos, extra = decompose(model, prompts, res, admit, W, with_floor=True, keep=keep)
    rec["B_instrumented_run_equals_gate_run"] = summ["mean_kl_single_batched"] == bp["mean_kl"]
    rec["B_decomposition_bf16"] = summ
    rec["B_per_sequence"] = seqs
    work["B_positions"] = pos
    work["B_k8_own_lead"] = extra
    D.log(f"[stage1] B decomposition: KL {summ['mean_kl_single_batched']:.4f} floor {summ.get('mean_kl_floor_matched')} "
          f"decode {summ['mean_kl_single_decode']:.4f} top5 share {summ['share_of_kl_sum_top5_positions']:.2f}")
    rec.update(localisers(model, prompts, res, admit, eos))
    D.log(f"[stage1] L1 div {rec['L1_identical_prompt']['first_divergence_step']} rows equal "
          f"{rec['L1_identical_prompt']['B8_rows_bitwise_equal']}; L2 max {rec['L2_max_kl']:.3e}")
    D.write_npz(run.scratch / f"batched_tokens_{run.utc}.npz",
                tokens=np.concatenate([np.asarray(g["tokens"], dtype=np.int64) for g in res]),
                n=np.array([len(g["tokens"]) for g in res], dtype=np.int64))
    del model
    run.release()

    # ---- C: fp32 activations on the real weights --------------------------------
    if "C" not in skip:
        model, _ = run.load_arm("K8", fp32=True)
        bpc = g5.batch_parity(model, prompts, MT, B)
        resc, admitc = instrumented(model, prompts, MT, B, eos)
        summc, seqsc, posc, extrac = decompose(model, prompts, resc, admitc, W, with_floor=False)
        rec["C_gate_batch_parity_fp32act"] = bpc
        rec["C_decomposition_fp32act"] = summc
        rec["C_per_sequence"] = seqsc
        work["C_positions"] = posc
        work["C_k8_own_lead"] = extrac
        D.log(f"[stage1] C fp32-activation parity {bpc['mean_kl']:.3e} top1_dis {bpc['top1_dis']:.4f}")
        del model
        run.release()

    # ---- R: the fp32 reference on the batched sequences ------------------------
    if "R" not in skip:
        r, per = reference_part(run, prompts, res, keep, th5["greedy_decisive_lead_nats"])
        rec["R_reference_teacher_forcing"] = r
        work["R_positions"] = per
        D.log(f"[stage1] R KL(ref||single) {r['mean_kl_ref_single']:.4f} KL(ref||batched) {r['mean_kl_ref_batched']:.4f} "
              f"ratio {r['ratio_batched_over_single']}; decisive top-1 single {r['decisive_top1_single']} "
              f"batched {r['decisive_top1_batched']}; n10 {r['n10_batched_wrong_single_right']} n01 "
              f"{r['n01_single_wrong_batched_right']}")
    if args.with_q36 and not run.tiny:
        q, qw = q36_part(run)
        rec["Q_optional_descriptive"] = q
        work.update(qw)
    work["entropy_note"] = ("B_positions.entropy_single is K8's own entropy, and *_k8_own_lead K8's own top-1 lead: "
                            "work file only (FROZEN_RULES.md 1.8)")
    rec = {"rule_inputs": rule_inputs(rec, W), **rec}
    run.finish(rec, work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
