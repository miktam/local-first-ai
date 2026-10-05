"""G5 first-wave bf16 reproduction harness (investigation only).

Builds an in-memory Kolibri port model with configurable dims and weight scales,
runs an instrumented copy of gate/checks/g5_generation.batch_parity's loop
(verbatim admission logic, the gate's _batch_generator), and compares, per
position, batched vs single (the gate's comparator, logits_at chunk 2048) and
both against an fp32-activation truth on the same weights.

usage: rep.py --H 2560 --heads 48 --kv 4 --E 64 --k 6 --I 512 --layers 5 --prec q8 --out x.json [...]
"""
import argparse, json, os, sys, time
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, mlx.nn as nn, numpy as np
from gate import common
from gate.checks import g5_generation as g5
from gate.harness import logits_at

ap = argparse.ArgumentParser()
ap.add_argument("--H", type=int, default=2560); ap.add_argument("--heads", type=int, default=48)
ap.add_argument("--kv", type=int, default=4); ap.add_argument("--D", type=int, default=128)
ap.add_argument("--E", type=int, default=64); ap.add_argument("--k", type=int, default=6)
ap.add_argument("--I", type=int, default=512); ap.add_argument("--layers", type=int, default=5)
ap.add_argument("--V", type=int, default=4096); ap.add_argument("--W", type=int, default=513)
ap.add_argument("--prec", default="bf16"); ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--emb_std", type=float, default=1.0); ap.add_argument("--proj_std", type=float, default=0.02)
ap.add_argument("--head_std", type=float, default=0.08); ap.add_argument("--router_std", type=float, default=0.02)
ap.add_argument("--bias_std", type=float, default=2.0); ap.add_argument("--postnorm_scale", type=float, default=1.0)
ap.add_argument("--outlier", type=float, default=0.0, help="add a massive-activation dim to the embedding (value)")
ap.add_argument("--lengths", default="37,300,700,1100,64,520,900,150,37,700,300,1100")
ap.add_argument("--max_tokens", default="48,8,32,16,40,24,56,12,20,36,28,44")
ap.add_argument("--B", type=int, default=8)
ap.add_argument("--pad_id", type=int, default=0)
ap.add_argument("--no_truth", action="store_true")
ap.add_argument("--prefill_bs", type=int, default=0, help="0: the gate construction (min(B,8)); else override prefill_batch_size")
a = ap.parse_args()

module = common.port_module()
lt = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(a.layers)]
args = module.ModelArgs(model_type="kolibri1", hidden_size=a.H, num_hidden_layers=a.layers,
                        num_attention_heads=a.heads, num_key_value_heads=a.kv, head_dim=a.D, vocab_size=a.V,
                        num_experts=a.E, num_experts_per_tok=a.k, moe_intermediate_size=a.I,
                        shared_expert_intermediate_size=a.I, sliding_window=a.W, layer_types=lt,
                        rope_theta=10000.0, use_sliding_window=True)
model = module.Model(args)
mx.random.seed(a.seed)
def nrm(shape, std, dt=mx.bfloat16): return (mx.random.normal(shape) * std).astype(dt)
W = {}
W["model.embed_tokens.weight"] = nrm((a.V, a.H), a.emb_std)
if a.outlier:
    e = W["model.embed_tokens.weight"].astype(mx.float32)
    e[:, 7] = a.outlier
    W["model.embed_tokens.weight"] = e.astype(mx.bfloat16)
W["model.norm.weight"] = (1 + nrm((a.H,), 0.1, mx.float32)).astype(mx.bfloat16)
W["lm_head.weight"] = nrm((a.V, a.H), a.head_std)
for i in range(a.layers):
    p = f"model.layers.{i}"
    W[f"{p}.self_attn.q_proj.weight"] = nrm((a.heads * a.D, a.H), a.proj_std)
    W[f"{p}.self_attn.k_proj.weight"] = nrm((a.kv * a.D, a.H), a.proj_std)
    W[f"{p}.self_attn.v_proj.weight"] = nrm((a.kv * a.D, a.H), a.proj_std)
    W[f"{p}.self_attn.o_proj.weight"] = nrm((a.H, a.heads * a.D), a.proj_std)
    W[f"{p}.self_attn.q_norm.weight"] = (1 + nrm((a.D,), 0.1, mx.float32)).astype(mx.bfloat16)
    W[f"{p}.self_attn.k_norm.weight"] = (1 + nrm((a.D,), 0.1, mx.float32)).astype(mx.bfloat16)
    for n in ("input_layernorm", "post_attention_layernorm"):
        W[f"{p}.{n}.weight"] = (1 + nrm((a.H,), 0.1, mx.float32)).astype(mx.bfloat16)
    for n in ("post_attn_norm", "post_ffn_norm"):
        W[f"{p}.{n}.weight"] = ((1 + nrm((a.H,), 0.1, mx.float32)) * a.postnorm_scale).astype(mx.bfloat16)
    W[f"{p}.mlp.gate.weight"] = nrm((a.E, a.H), a.router_std).astype(mx.float32)
    W[f"{p}.mlp.gate.expert_bias"] = nrm((a.E,), a.bias_std).astype(mx.float32)
    W[f"{p}.mlp.switch_mlp.gate_proj.weight"] = nrm((a.E, a.I, a.H), a.proj_std)
    W[f"{p}.mlp.switch_mlp.up_proj.weight"] = nrm((a.E, a.I, a.H), a.proj_std)
    W[f"{p}.mlp.switch_mlp.down_proj.weight"] = nrm((a.E, a.H, a.I), a.proj_std)
    W[f"{p}.mlp.shared_experts.gate_proj.weight"] = nrm((a.I, a.H), a.proj_std)
    W[f"{p}.mlp.shared_experts.up_proj.weight"] = nrm((a.I, a.H), a.proj_std)
    W[f"{p}.mlp.shared_experts.down_proj.weight"] = nrm((a.H, a.I), a.proj_std)
    mx.eval([v for k_, v in W.items() if k_.startswith(p + ".")])
model.load_weights(list(W.items()))
del W
if a.prec in ("q8", "q4"):
    pred = module.make_quant_predicate(quantize_embeddings=False, quantize_lm_head=True)
    nn.quantize(model, group_size=64, bits=8 if a.prec == "q8" else 4, class_predicate=pred)
elif a.prec == "fp32":
    model.set_dtype(mx.float32)
mx.eval(model.parameters())

lengths = [int(x) for x in a.lengths.split(",")]
max_tokens = [int(x) for x in a.max_tokens.split(",")]
rng = np.random.default_rng(1000 + a.seed)
texts = {f"T{i}": rng.integers(1, a.V - 16, size=max(lengths)).tolist() for i in range(1, 9)}
order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
prompts = [texts[order[j % 8]][:lengths[j]] for j in range(len(lengths))]
B = a.B
lp, kl = g5._lp, g5.kl_lp

t0 = time.time()
gen, path = g5._batch_generator(model, B, common.EOS_IDS, max(max_tokens))
if a.prefill_bs:
    gen.prefill_batch_size = a.prefill_bs
    gen.completion_batch_size = max(B, a.prefill_bs)
got, queue, uid_of, in_flight, step, any_finished, admit = {}, list(range(len(prompts))), {}, 0, 0, False, {}
try:
    while queue or in_flight:
        free = B - in_flight
        if free > 0 and queue:
            group = queue[:free]; del queue[:free]
            uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
            for u, j in zip(uids, group):
                uid_of[u] = j; got[u] = {"tokens": [], "lp": []}; admit[j] = {"step": step, "mid_run": any_finished, "group": list(group)}
            in_flight += len(group)
        _, resps = gen.next()
        for r in resps:
            g = got[r.uid]; g["tokens"].append(int(r.token))
            x = r.logprobs.astype(mx.float32); mx.eval(x); g["lp"].append(np.array(x))
            if r.finish_reason is not None:
                in_flight -= 1; any_finished = True
        step += 1
finally:
    gen.close()
t_loop = time.time() - t0
seqs = {}
for u, j in uid_of.items():
    g = got[u]; p = prompts[j]; P = len(p); n = len(g["tokens"])
    seqs[j] = (list(p) + g["tokens"][:-1], np.arange(P - 1, P - 1 + n), np.stack(g["lp"]))
single = {j: lp(logits_at(model, s, pos)) for j, (s, pos, _) in seqs.items()}
truth = None
if not a.no_truth:
    model.set_dtype(mx.float32); mx.eval(model.parameters())
    truth = {j: lp(logits_at(model, s, pos)) for j, (s, pos, _) in seqs.items()}
rows = []
for j in sorted(seqs):
    s, pos, blp = seqs[j]
    b = lp(blp); si = single[j]
    for t in range(len(pos)):
        r = {"j": j, "t": t, "P": lengths[j], "mid": admit[j]["mid_run"],
             "kl_sb": float(kl(si[t:t+1], b[t:t+1])[0])}
        if truth is not None:
            tr = truth[j]
            r["kl_tb"] = float(kl(tr[t:t+1], b[t:t+1])[0]); r["kl_ts"] = float(kl(tr[t:t+1], si[t:t+1])[0])
        rows.append(r)
def agg(sel):
    rs = [r for r in rows if sel(r)]
    out = {"n": len(rs)}
    for k_ in ("kl_sb", "kl_tb", "kl_ts"):
        if rs and k_ in rs[0]: out[k_] = float(np.mean([r[k_] for r in rs]))
    return out
classes = {"all": lambda r: True, "first_wave": lambda r: not r["mid"], "mid_run": lambda r: r["mid"],
           "t0": lambda r: r["t"] == 0, "t1_2": lambda r: 1 <= r["t"] <= 2, "t3plus": lambda r: r["t"] >= 3,
           "fw_t0": lambda r: not r["mid"] and r["t"] == 0, "fw_t1_2": lambda r: not r["mid"] and 1 <= r["t"] <= 2,
           "fw_t3plus": lambda r: not r["mid"] and r["t"] >= 3, "mid_t0": lambda r: r["mid"] and r["t"] == 0,
           "mid_t1plus": lambda r: r["mid"] and r["t"] >= 1}
res = {"args": vars(a), "t_loop": t_loop, "classes": {k_: agg(f) for k_, f in classes.items()},
       "per_seq": {j: agg(lambda r, j=j: r["j"] == j) for j in sorted(seqs)},
       "admit": admit, "rows": rows}
open(a.out, "w").write(json.dumps(res, indent=1))
c = res["classes"]
fmt = lambda d: " ".join(f"{k_}={v:.3e}" if isinstance(v, float) else f"{k_}={v}" for k_, v in d.items())
for k_ in ("all", "first_wave", "mid_run", "fw_t0", "fw_t1_2", "fw_t3plus", "mid_t0", "mid_t1plus"):
    print(f"{k_:11s} {fmt(c[k_])}")
print("per_seq kl_sb:", {j: round(v["kl_sb"], 5) for j, v in res["per_seq"].items()})
if truth is not None:
    print("per_seq kl_tb:", {j: round(v["kl_tb"], 5) for j, v in res["per_seq"].items()})
    print("per_seq kl_ts:", {j: round(v["kl_ts"], 5) for j, v in res["per_seq"].items()})
