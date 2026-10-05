# SPDX-License-Identifier: MIT
"""exp_036 G5 batch-parity diagnosis for the run host (mbp). Read-only on the kit:
it imports gate/, runner/, tools/, bench/ and writes only under --out.

What it measures (K8 by default, K4 with --arms K8,K4):
  A  tiny fp32 mechanics on THIS GPU: w513 tiny checkpoints with head_dim 128
     (the real head_dim, so MLX's fused SDPA kernels run; the kit's tiny presets
     use head_dim 32, which always falls back to unfused attention), the gate's
     batch_parity at the registered lengths 37..1,100 and max_tokens 8..56.
     Expect mean KL ~1e-12 and max |dlogprob| <= 1e-5.
  B  the gate's exact batch_parity on the real arm, bf16 as loaded
     (reproducibility check against the gate record), plus an instrumented copy
     of the same loop that keeps per-sequence / per-position data, and for each
     sequence: single (prefill chunk 2048, the gate's comparator), single64
     (prefill chunk 64: the chunking floor ON THE SAME POSITIONS), decode_tf
     (one sequence, plain port caches, teacher-forced decode of the batched
     tokens). KL(single||batched) = KL(single||decode_tf) + batching part.
  C  the same with fp32 activations on the real quantised weights
     (model.set_dtype(float32), MLX_ENABLE_TF32=0): the mechanics check on the
     real weights. A padding / RoPE-offset / window-mask / admission bug shows
     here as KL >> 1e-4 for a class of sequences; bf16 routing noise does not.
  D  (descriptive) the peer check's batched-path rule on Kolibri, chat-wrapped
     (Amendment 5 regime: tools/peer_check.batched_path), for a like-for-like
     comparison with G8 (0.0083 vs floor 0.0137) and Q36-8 (0.059 vs 0.0096).

No token ids or text go into the output; only KL numbers, positions and indices.

usage (from the kit root, env sourced):
  caffeinate -i "$PY" <this file> --out "$EXP036_WORK/g5diag_$(date -u +%Y%m%dT%H%M%SZ)" [--arms K8] [--skip C,D]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

KIT = Path.cwd()
if not (KIT / "gate" / "checks" / "g5_generation.py").is_file():
    raise SystemExit("run from the exp_036 kit root")
sys.path[:0] = [str(KIT), str(KIT / "tests")]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from gate import common, textset  # noqa: E402
from gate.checks import g5_generation as g5  # noqa: E402
from gate.harness import logits_at  # noqa: E402

GATE_RECORD = {"K8": {"mean_kl": 0.3073773544462904, "top1_dis": 0.14285714285714285},
               "K4": {"mean_kl": 0.22264807020527483, "top1_dis": 0.15384615384615385}}
GATE_FLOOR = {"floor_kl": 0.05043081948260652, "floor_dis": 0.06561859193438141, "kl_max": 0.15129245844781955}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="K8")
ap.add_argument("--skip", default="")
a = ap.parse_args()
OUT = Path(a.out).expanduser()
OUT.mkdir(parents=True, exist_ok=True)
SKIP = set(x for x in a.skip.split(",") if x)
lp, kl = g5._lp, g5.kl_lp


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


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
    finally:
        gen.close()
    return got, uid_of, admit


def decode_tf(model, prompt, toks):
    cache = model.make_cache()
    ids = np.asarray(prompt, dtype=np.int32)
    for s in range(0, ids.size - 1, 2048):
        mx.eval(model(mx.array(ids[s:min(ids.size - 1, s + 2048)])[None], cache=cache))
    rows = []
    for t in [int(ids[-1])] + list(toks[:-1]):
        o = model(mx.array([[t]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(o)
        rows.append(np.array(o))
    return np.stack(rows)


def decompose(model, prompts, max_tokens, B, eos, W, with_floor=True):
    got, uid_of, admit = instrumented(model, prompts, max_tokens, B, eos)
    seqs, pos = [], []
    for u, j in sorted(uid_of.items(), key=lambda t: t[1]):
        g, p = got[u], prompts[j]
        P, n = len(p), len(g["tokens"])
        seq = list(p) + g["tokens"][:-1]
        at = np.arange(P - 1, P - 1 + n)
        single = lp(logits_at(model, seq, at))
        batched = lp(np.stack(g["lp"]))
        dec = lp(decode_tf(model, p, g["tokens"]))
        s64 = lp(logits_at(model, seq, at, chunk=64)) if with_floor else None
        k_sb, k_sd, k_db = kl(single, batched), kl(single, dec), kl(dec, batched)
        k_fl = kl(single, s64) if with_floor else np.full(n, np.nan)
        top2 = np.sort(single, axis=-1)[:, -2:]
        gap = top2[:, 1] - top2[:, 0]
        ent = -(np.exp(single) * single).sum(-1)
        seqs.append({
            "j": j, "P": P, "n": n, "mid_run": admit[j]["mid_run"], "admit_step": admit[j]["step"],
            "kl_single_batched": float(k_sb.mean()), "kl_single_decode": float(k_sd.mean()),
            "kl_decode_batched": float(k_db.mean()), "kl_floor_matched": float(np.mean(k_fl)),
            "dis_single_batched": float((single.argmax(-1) != batched.argmax(-1)).mean()),
            "dis_single_decode": float((single.argmax(-1) != dec.argmax(-1)).mean()),
            "dis_decode_batched": float((dec.argmax(-1) != batched.argmax(-1)).mean()),
            "dis_floor_matched": float((single.argmax(-1) != s64.argmax(-1)).mean()) if with_floor else None,
            "max_abs_dlogprob_single_batched": float(np.abs(single - batched).max()),
            "max_abs_dlogprob_decode_batched": float(np.abs(dec - batched).max()),
            "first_dlogprob_gt_1e-3_decode_batched": next((int(t) for t in range(n)
                                                          if np.abs(dec[t] - batched[t]).max() > 1e-3), None),
            "kl_sb_max": float(k_sb.max()), "kl_sb_argmax_t": int(k_sb.argmax()),
            "mean_entropy_single": float(ent.mean()),
        })
        for t in range(n):
            pos.append({"j": j, "t": t, "abs_pos": int(P - 1 + t), "mid_run": admit[j]["mid_run"],
                        "dis_sb": bool(single[t].argmax() != batched[t].argmax()), "kl_sb": float(k_sb[t]), "kl_sd": float(k_sd[t]), "kl_db": float(k_db[t]),
                        "kl_floor": float(k_fl[t]), "gap_single": float(gap[t]), "ent_single": float(ent[t])})
    K = np.array([r["kl_sb"] for r in pos])
    srt = np.sort(K)[::-1]

    def pooled(f, key="kl_sb"):
        v = [r[key] for r in pos if f(r)]
        return float(np.mean(v)) if v else None

    summ = {
        "n_positions": int(K.size), "mean_kl_single_batched": float(K.mean()),
        "top1_dis_single_batched": float(np.mean([r["dis_sb"] for r in pos])),
        "mean_kl_single_decode": pooled(lambda r: True, "kl_sd"),
        "mean_kl_decode_batched": pooled(lambda r: True, "kl_db"),
        "mean_kl_floor_matched": pooled(lambda r: True, "kl_floor"),
        "share_of_kl_sum_top5_positions": float(srt[:5].sum() / max(K.sum(), 1e-300)),
        "share_of_kl_sum_top10_positions": float(srt[:10].sum() / max(K.sum(), 1e-300)),
        "top10_position_kls": [float(x) for x in srt[:10]],
        "mean_kl_mid_run": pooled(lambda r: r["mid_run"]), "mean_kl_first_wave": pooled(lambda r: not r["mid_run"]),
        "mean_kl_first3_steps": pooled(lambda r: r["t"] < 3), "mean_kl_later_steps": pooled(lambda r: r["t"] >= 3),
        "mean_kl_abs_pos_lt_W": pooled(lambda r: r["abs_pos"] < W), "mean_kl_abs_pos_ge_W": pooled(lambda r: r["abs_pos"] >= W),
        "mean_kl_gap_lt_0.5": pooled(lambda r: r["gap_single"] < 0.5), "frac_gap_lt_0.5": float(np.mean([r["gap_single"] < 0.5 for r in pos])),
    }
    fl = summ["mean_kl_floor_matched"]
    summ["ratio_to_matched_floor"] = (summ["mean_kl_single_batched"] / fl) if fl else None
    summ["ratio_to_gate_floor"] = summ["mean_kl_single_batched"] / GATE_FLOOR["floor_kl"]
    return {"summary": summ, "per_sequence": seqs, "positions": pos}


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=1))
    log("wrote", name)


report = {"precision": PREC, "mlx": mx.__version__}
import mlx_lm  # noqa: E402

report["mlx_lm"] = mlx_lm.__version__
th = common.load_thresholds()
prof = textset.real_profile(th)
B = th["G5"]["batch_B"]

# A0: kernel probe on this GPU. Is an fp32 quantised matmul exact at M = 8
# (batched decode) and M = 1 (single decode)? If MLX_ENABLE_TF32=0 does not
# reach the quantised kernels, part C's fp32 run carries ~1e-3 relative
# errors and must be read per sequence (a bug is a class of sequences off
# from the first step; TF32-level noise is diffuse).
def kernel_probe():
    rng = np.random.default_rng(36)
    out = {}
    w = rng.standard_normal((1024, 2560)).astype(np.float32) * 0.02
    wq, s, b = mx.quantize(mx.array(w), group_size=64, bits=8)
    deq = np.array(mx.dequantize(wq, s, b, group_size=64, bits=8)).astype(np.float64)
    x = rng.standard_normal((8, 2560)).astype(np.float32)
    ref = x.astype(np.float64) @ deq.T
    for dt in (mx.float32, mx.bfloat16):
        xx = mx.array(x).astype(dt)
        y8 = mx.quantized_matmul(xx, wq, s.astype(dt), b.astype(dt), transpose=True, group_size=64, bits=8)
        y1 = mx.concatenate([mx.quantized_matmul(xx[i:i + 1], wq, s.astype(dt), b.astype(dt), transpose=True,
                                                 group_size=64, bits=8) for i in range(8)])
        y8n, y1n = np.array(y8.astype(mx.float32)), np.array(y1.astype(mx.float32))
        out[str(dt)] = {"rel_l2_M8_vs_f64": float(np.linalg.norm(y8n - ref) / np.linalg.norm(ref)),
                        "rel_l2_M1_vs_f64": float(np.linalg.norm(y1n - ref) / np.linalg.norm(ref)),
                        "M8_equals_M1_bitwise": bool((y8n == y1n).all()),
                        "max_abs_M8_minus_M1": float(np.abs(y8n - y1n).max())}
    return out


report["A0_kernel_probe"] = kernel_probe()
log("A0", json.dumps(report["A0_kernel_probe"]))

# A: tiny fp32 mechanics with the real head_dim on this GPU
if "A" not in SKIP:
    import tiny_checkpoint as tc

    tiny = {}
    with tempfile.TemporaryDirectory() as td:
        for name, ov in {"w513_h128_q12kv1": {"num_attention_heads": 12, "num_key_value_heads": 1, "head_dim": 128},
                         "w513_h128_q48kv4": {"num_attention_heads": 48, "num_key_value_heads": 4, "head_dim": 128}}.items():
            d = tc.write_tiny_checkpoint(Path(td) / name, seed=0, preset="w513", overrides=ov)
            m, _, _ = common.load_port(d)
            m.set_dtype(mx.float32)
            rng = np.random.default_rng(1000)
            texts = [rng.integers(0, m.args.vocab_size - 16, size=prof.text_len).tolist() for _ in range(8)]
            prompts = [texts[j % 8][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
            bp = g5.batch_parity(m, prompts, list(prof.batch_max_tokens), B)
            dec = decompose(m, prompts, list(prof.batch_max_tokens), B, common.EOS_IDS, m.args.sliding_window, with_floor=False)
            tiny[name] = {"gate_batch_parity": bp,
                          "max_abs_dlogprob_single_batched": max(s["max_abs_dlogprob_single_batched"] for s in dec["per_sequence"]),
                          "ok": bp["mean_kl"] <= 1e-8 and bp["top1_dis"] == 0.0}
            log("A", name, bp["mean_kl"], tiny[name]["max_abs_dlogprob_single_batched"])
            del m
    report["A_tiny_fp32_mechanics"] = tiny

ts = textset.load_real(common.tokenizer_dir(), common.work_dir() / "gate_texts", prof)
order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
prompts = [ts.ids[order[j % 8]][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
MT = list(prof.batch_max_tokens)

for arm in a.arms.split(","):
    adir = common.models_dir() / common.ARM_DIRS[arm]
    res = {}
    # B: bf16 as loaded (the gate's configuration)
    if "B" not in SKIP:
        model, _, _ = common.load_port(adir)
        W = model.args.sliding_window
        t0 = time.time()
        bp = g5.batch_parity(model, prompts, MT, B)
        res["B_gate_batch_parity_bf16"] = bp
        res["B_reproduces_gate_record"] = {"mean_kl_equal": bp["mean_kl"] == GATE_RECORD[arm]["mean_kl"],
                                           "mean_kl_diff": bp["mean_kl"] - GATE_RECORD[arm]["mean_kl"]}
        log(arm, "B gate batch_parity", bp["mean_kl"], bp["top1_dis"], f"{time.time() - t0:.0f}s")
        d = decompose(model, prompts, MT, B, common.EOS_IDS, W)
        res["B_decomposition_bf16"] = d["summary"]
        write(f"{arm}_B_bf16_per_sequence.json", d["per_sequence"])
        write(f"{arm}_B_bf16_positions.json", d["positions"])
        log(arm, "B", json.dumps(d["summary"])[:600])
        # D: the peer check's regime, chat-wrapped (descriptive). Kolibri's
        # Amendment 5 wrapper is 41 tokens, longer than the shortest registered
        # prompt (37), so peer_check.batched_prompts refuses it; here prompt i is
        # the wrapper + n_i stream tokens (stream offset 50 i, as the peer check),
        # and the floor runs on the wrapper + stream at (len(w)+520, len(w)+1100).
        if "D" not in SKIP:
            from bench import kl_8v4 as KL

            w = [int(i) for i in KL.chat_wrapper("kolibri", adir)["prompt_ids"]]
            texts = KL.load_texts(KL.gate_text_paths())
            base = []
            for t in ("T1", "T2", "T3", "T4"):
                base.extend(KL.tokenize(adir / "tokenizer.json", texts[t]["text"])[0])
            dprompts = [w + base[50 * i:50 * i + n] for i, n in enumerate(prof.batch_lengths)]
            fl = g5.floor_and_decode(model, {"base": w + base}, {"base": (len(w) + 520, len(w) + 1100)}, (2048, 64),
                                     log=lambda *_: None)
            bpd = g5.batch_parity(model, dprompts, MT, B)
            dd = decompose(model, dprompts, MT, B, common.EOS_IDS, W)
            res["D_chat_wrapped"] = {"wrapper_tokens": len(w), "noise_floor": fl, "parity": bpd,
                                     "ratio_to_floor": bpd["mean_kl"] / max(fl["floor_kl"], 1e-300),
                                     "decomposition": dd["summary"]}
            write(f"{arm}_D_chat_per_sequence.json", dd["per_sequence"])
            log(arm, "D chat-wrapped", bpd["mean_kl"], fl["floor_kl"], dd["summary"]["mean_kl_floor_matched"])
        del model
        mx.clear_cache()
    # C: fp32 activations on the real quantised weights
    if "C" not in SKIP:
        model, _, _ = common.load_port(adir)
        model.set_dtype(mx.float32)
        W = model.args.sliding_window
        bp = g5.batch_parity(model, prompts, MT, B)
        res["C_gate_batch_parity_fp32act"] = bp
        d = decompose(model, prompts, MT, B, common.EOS_IDS, W, with_floor=False)
        res["C_decomposition_fp32act"] = d["summary"]
        write(f"{arm}_C_fp32act_per_sequence.json", d["per_sequence"])
        log(arm, "C fp32-activation parity", bp["mean_kl"], bp["top1_dis"])
        del model
        mx.clear_cache()
    report[arm] = res
    write("g5diag_report.json", report)

# Verdict lines (the decision rule)
for arm in a.arms.split(","):
    r = report.get(arm, {})
    c = r.get("C_gate_batch_parity_fp32act")
    if c is not None:
        mech = "OK" if (c["mean_kl"] <= 1e-4 and c["top1_dis"] <= 2 / c["n_positions"]) else "BUG-SUSPECT"
        print(f"{arm} mechanics (fp32 activations, real weights): {mech}  mean_kl={c['mean_kl']:.3e} top1_dis={c['top1_dis']:.4f}")
    s = r.get("B_decomposition_bf16")
    if s is not None:
        print(f"{arm} bf16: gate KL {s['mean_kl_single_batched']:.4f}; matched floor {s['mean_kl_floor_matched']:.4f} "
              f"(ratio {s['ratio_to_matched_floor']:.2f}); single-vs-decode {s['mean_kl_single_decode']:.4f}; "
              f"decode-vs-batched {s['mean_kl_decode_batched']:.4f}; top-5 positions carry "
              f"{100 * s['share_of_kl_sum_top5_positions']:.0f}% of the KL sum; first-wave {s['mean_kl_first_wave']:.4f} "
              f"vs mid-run {s['mean_kl_mid_run']:.4f}")
write("g5diag_report.json", report)
