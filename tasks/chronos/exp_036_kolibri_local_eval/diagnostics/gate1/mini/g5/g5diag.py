"""G5 batch-parity diagnosis (read-only on the kit). Runs the gate's exact
batch_parity plus an instrumented copy of its loop, and decomposes the
batched-vs-single KL per sequence and per position.

usage: python g5diag.py <ckpt_dir> <precision: fp32|bf16|q8|q4> <out.json> [--profile real|tiny] [--seed N]
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32  # noqa: E402

ensure_exact_fp32()

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

from gate import common  # noqa: E402
from gate.checks import g5_generation as g5  # noqa: E402
from gate.harness import logits_at  # noqa: E402
from gate.textset import Profile, tiny_profile  # noqa: E402

ckpt, prec, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
args = sys.argv[4:]
profile_name = args[args.index("--profile") + 1] if "--profile" in args else "real"
seed = int(args[args.index("--seed") + 1]) if "--seed" in args else 0
skip_t9 = "--no-t9" in args

model, cfg, module = common.load_port(Path(ckpt))
W = model.args.sliding_window
if prec == "fp32":
    model.set_dtype(mx.float32)
elif prec in ("q8", "q4"):
    bits = 8 if prec == "q8" else 4
    nn.quantize(model, group_size=64, bits=bits, class_predicate=model.quant_predicate)
elif prec != "bf16":
    raise SystemExit(prec)
mx.eval(model.parameters())

V = model.args.vocab_size
prof = Profile() if profile_name == "real" else tiny_profile(W, model.args.num_hidden_layers)
rng = np.random.default_rng(1000 + seed)
L = prof.text_len
texts = {f"T{i}": rng.integers(0, V - 16, size=L).tolist() for i in range(1, 9)}
order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
prompts = [texts[order[j % 8]][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
max_tokens = list(prof.batch_max_tokens)
B = prof.batch_B
lp = g5._lp
kl = g5.kl_lp

res = {"ckpt": ckpt, "precision": prec, "profile": profile_name, "seed": seed, "W": W,
       "head_dim": model.args.head_dim, "n_layers": model.args.num_hidden_layers}

# 1) the gate's exact function
t0 = time.time()
gate_bp = g5.batch_parity(model, prompts, max_tokens, B)
res["gate_batch_parity"] = gate_bp
res["t_gate_bp"] = time.time() - t0


# 2) instrumented copy of the same loop (verbatim admission logic) keeping per-sequence data
def instrumented(model, prompts, max_tokens, B, eos=common.EOS_IDS):
    gen, path = g5._batch_generator(model, B, eos, max(max_tokens))
    got, queue = {}, list(range(len(prompts)))
    uid_of, in_flight, step, any_finished = {}, 0, 0, False
    admit = {}
    live_at = []
    try:
        while queue or in_flight:
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                for u, j in zip(uids, group):
                    uid_of[u] = j
                    got[u] = {"tokens": [], "lp": [], "steps": []}
                    admit[j] = {"step": step, "mid_run": any_finished}
                in_flight += len(group)
            _, resps = gen.next()
            gb = gen._generation_batch
            if len(gb):
                c0 = next((c for c in gb.prompt_cache if type(c).__name__ == "BatchRotatingKVCache"), None)
                c1 = next((c for c in gb.prompt_cache if type(c).__name__ == "BatchKVCache"), None)
                live_at.append({"step": step, "uids": list(gb.uids),
                                "rot_left_padding": None if c0 is None else np.array(c0.left_padding).tolist(),
                                "rot_offset": None if c0 is None else np.array(c0.offset).tolist(),
                                "full_left_padding": None if c1 is None else np.array(c1.left_padding).tolist(),
                                "full_offset": None if c1 is None else np.array(c1.offset).tolist()})
            for r in resps:
                g = got[r.uid]
                g["tokens"].append(int(r.token))
                x = r.logprobs.astype(mx.float32)
                mx.eval(x)
                g["lp"].append(np.array(x))
                g["steps"].append(step)
                if r.finish_reason is not None:
                    in_flight -= 1
                    any_finished = True
            step += 1
    finally:
        gen.close()
    return got, uid_of, admit, live_at


got, uid_of, admit, live_at = instrumented(model, prompts, max_tokens, B)


def decode_tf(model, prompt, toks):
    """Single sequence, plain port caches: prefill prompt[:-1] (chunks of 2048),
    then decode prompt[-1], toks[:-1] one token at a time (teacher-forced)."""
    cache = model.make_cache()
    ids = np.asarray(prompt, dtype=np.int32)
    for a in range(0, ids.size - 1, 2048):
        mx.eval(model(mx.array(ids[a:min(ids.size - 1, a + 2048)])[None], cache=cache))
    feed = [int(ids[-1])] + list(toks[:-1])
    rows = []
    for t in feed:
        o = model(mx.array([[t]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(o)
        rows.append(np.array(o))
    return np.stack(rows)


per_seq = []
all_kl, all_dis = [], []
pos_records = []
for u, j in sorted(uid_of.items(), key=lambda t: t[1]):
    g, p = got[u], prompts[j]
    P, n = len(p), len(g["tokens"])
    seq = list(p) + g["tokens"][:-1]
    positions = np.arange(P - 1, P - 1 + n)
    single = lp(logits_at(model, seq, positions))                    # gate's single: prefill chunk 2048
    single64 = lp(logits_at(model, seq, positions, chunk=64))         # position-matched chunking floor
    dec = lp(decode_tf(model, p, g["tokens"]))                       # single decode, teacher-forced
    batched = lp(np.stack(g["lp"]))
    k_sb = kl(single, batched)
    k_sd = kl(single, dec)
    k_db = kl(dec, batched)
    k_floor = kl(single, single64)
    dis_sb = single.argmax(-1) != batched.argmax(-1)
    dis_db = dec.argmax(-1) != batched.argmax(-1)
    dis_floor = single.argmax(-1) != single64.argmax(-1)
    maxabs_db = float(np.abs(dec - batched).max())
    maxabs_sb = float(np.abs(single - batched).max())
    all_kl.append(k_sb), all_dis.append(dis_sb)
    # top-2 gap of single at each position (near-tie indicator)
    s2 = np.sort(single, axis=-1)[:, -2:]
    gap = s2[:, 1] - s2[:, 0]
    ent = -(np.exp(single) * single).sum(-1)
    per_seq.append({
        "j": j, "P": P, "n": n, "mid_run": admit[j]["mid_run"], "admit_step": admit[j]["step"],
        "first_step": g["steps"][0], "beyond_window_from": max(0, W - (P - 1)),
        "kl_single_batched": float(k_sb.mean()), "kl_single_decode": float(k_sd.mean()),
        "kl_decode_batched": float(k_db.mean()), "kl_floor_matched": float(k_floor.mean()),
        "dis_single_batched": float(dis_sb.mean()), "dis_decode_batched": float(dis_db.mean()),
        "dis_floor_matched": float(dis_floor.mean()),
        "max_abs_dlogprob_single_batched": maxabs_sb, "max_abs_dlogprob_decode_batched": maxabs_db,
        "kl_sb_first3": float(k_sb[:3].mean()), "kl_sb_rest": float(k_sb[3:].mean()) if n > 3 else None,
        "kl_sb_max": float(k_sb.max()), "argmax_kl_pos": int(k_sb.argmax()),
        "mean_entropy": float(ent.mean()), "frac_gap_lt_0.05": float((gap < 0.05).mean()),
    })
    for t in range(n):
        pos_records.append({"j": j, "t": t, "abs_pos": int(P - 1 + t), "kl_sb": float(k_sb[t]), "kl_sd": float(k_sd[t]),
                            "kl_db": float(k_db[t]), "kl_floor": float(k_floor[t]), "gap": float(gap[t]),
                            "ent": float(ent[t])})

K_ = np.concatenate(all_kl)
D_ = np.concatenate(all_dis)
pr = pos_records
res["instrumented"] = {
    "mean_kl": float(K_.mean()), "top1_dis": float(D_.mean()), "n_positions": int(K_.size),
    "matches_gate": abs(float(K_.mean()) - gate_bp["mean_kl"]) <= 1e-12 * max(1.0, gate_bp["mean_kl"]) or float(K_.mean()) == gate_bp["mean_kl"],
    "pooled_kl_single_decode": float(np.mean([r["kl_sd"] for r in pr])),
    "pooled_kl_decode_batched": float(np.mean([r["kl_db"] for r in pr])),
    "pooled_kl_floor_matched": float(np.mean([r["kl_floor"] for r in pr])),
    "pooled_kl_mid_run": float(np.mean([r["kl_sb"] for r in pr if admit[r["j"]]["mid_run"]])),
    "pooled_kl_first_wave": float(np.mean([r["kl_sb"] for r in pr if not admit[r["j"]]["mid_run"]])),
    "pooled_kl_first3_steps": float(np.mean([r["kl_sb"] for r in pr if r["t"] < 3])),
    "pooled_kl_later_steps": float(np.mean([r["kl_sb"] for r in pr if r["t"] >= 3])),
    "pooled_kl_abs_pos_ge_W": float(np.mean([r["kl_sb"] for r in pr if r["abs_pos"] >= W] or [np.nan])),
    "pooled_kl_abs_pos_lt_W": float(np.mean([r["kl_sb"] for r in pr if r["abs_pos"] < W] or [np.nan])),
}
res["per_sequence"] = per_seq
res["positions"] = pr
res["live_at"] = live_at[:3] + live_at[-3:]

# 3) the gate's floor_and_decode at the floor positions, random-id texts
ranges = dict(prof.decode_ranges)
ftexts = {"T1": texts["T1"], "T3": texts["T3"]}
if not skip_t9:
    ftexts["T9"] = rng.integers(0, V - 16, size=prof.t9_len).tolist()
else:
    ranges.pop("T9")
t0 = time.time()
fd = g5.floor_and_decode(model, ftexts, ranges, (2048, 64), log=lambda *a: None)
res["floor_and_decode"] = fd
res["t_floor"] = time.time() - t0
res["ratio_gate_kl_over_floor"] = gate_bp["mean_kl"] / fd["floor_kl"] if fd["floor_kl"] > 0 else None
res["ratio_gate_kl_over_matched_floor"] = (gate_bp["mean_kl"] / res["instrumented"]["pooled_kl_floor_matched"]
                                           if res["instrumented"]["pooled_kl_floor_matched"] > 0 else None)
Path(out_path).write_text(json.dumps(res, indent=1))
s = res["instrumented"]
print(f"{ckpt} {prec} W={W} D={model.args.head_dim}: gate mean_kl={gate_bp['mean_kl']:.3e} dis={gate_bp['top1_dis']:.3f} "
      f"| instr={s['mean_kl']:.3e} same={s['matches_gate']} | single-vs-decode={s['pooled_kl_single_decode']:.3e} "
      f"decode-vs-batched={s['pooled_kl_decode_batched']:.3e} matched_floor={s['pooled_kl_floor_matched']:.3e} "
      f"| floor(T1/T3/T9)={fd['floor_kl']:.3e} decode={fd['decode_kl']:.3e} "
      f"| ratio_vs_floor={res['ratio_gate_kl_over_floor']} ratio_vs_matched={res['ratio_gate_kl_over_matched_floor']}")
