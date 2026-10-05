"""Where does the batched path's first-step (t=0) excess come from? Tiny w513 /
head_dim 128 checkpoint (the prior sweep's ckpt_w513h128), bf16 as stored.

For the first wave (8 prompts, registered lengths), compare the batched t=0
logprobs (GenerationBatch.__init__ step after the right-padded batched prefill)
with the single-sequence decode of the same last prompt token, and localise:
  (1) K/V caches after the batched prefill vs after a single prefill (per layer, bitwise)
  (2) t=0 with a uniform-length batch (no padding)
  (3) t=0 with B=1 through the same BatchGenerator
"""
import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx
import numpy as np
from pathlib import Path
from gate import common
from gate.checks import g5_generation as g5
from mlx_lm.generate import BatchGenerator
from runner.sampler import fp32_logits

CK = sys.argv[1]
prec = sys.argv[2] if len(sys.argv) > 2 else "bf16"
model, cfg, module = common.load_port(Path(CK))
if prec == "fp32":
    model.set_dtype(mx.float32)
mx.eval(model.parameters())
V = model.args.vocab_size
rng = np.random.default_rng(1000)
L = 1536
texts = [rng.integers(0, V - 16, size=L).tolist() for _ in range(8)]
lengths = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [texts[j][:n] for j, n in enumerate(lengths)]
lp = g5._lp
kl = g5.kl_lp


def single_t0(p):
    cache = model.make_cache()
    ids = np.asarray(p, dtype=np.int32)
    mx.eval(model(mx.array(ids[:-1])[None], cache=cache))
    o = model(mx.array([[int(ids[-1])]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
    mx.eval(o)
    return np.array(o), cache


def batched_t0(prompts, B):
    gen = BatchGenerator(fp32_logits(model), max_tokens=4, stop_tokens=[[e] for e in common.EOS_IDS],
                         sampler=lambda x: mx.argmax(x, axis=-1), completion_batch_size=B,
                         prefill_batch_size=min(B, 8), prefill_step_size=2048, max_kv_size=None)
    uids = gen.insert([list(p) for p in prompts], max_tokens=[4] * len(prompts))
    first = {}
    caches_after_prefill = None
    for _ in range(20):
        _, resps = gen.next()
        if caches_after_prefill is None and len(gen._generation_batch):
            caches_after_prefill = True
        for r in resps:
            if r.uid not in first:
                x = r.logprobs.astype(mx.float32); mx.eval(x)
                first[r.uid] = np.array(x)
        if len(first) == len(prompts):
            break
    gen.close()
    return [first[u] for u in uids]


out = {"ckpt": CK, "prec": prec}
sing = [single_t0(p) for p in prompts]
s_lp = [lp(s[0][None])[0] for s in sing]

# (A) the registered first wave, B=8 mixed lengths
b = batched_t0(prompts, 8)
out["A_mixed_B8_t0_kl_single_decode_vs_batched"] = [float(kl(s_lp[i][None], lp(b[i][None]))[0]) for i in range(8)]
# (B) B=1 through the same BatchGenerator, each prompt alone
b1 = [batched_t0([p], 1)[0] for p in prompts]
out["B_B1_t0_kl"] = [float(kl(s_lp[i][None], lp(b1[i][None]))[0]) for i in range(8)]
out["B_B1_t0_bitwise"] = [bool((b1[i] == (sing[i][0] - np.log(np.exp(sing[i][0] - sing[i][0].max()).sum()) - sing[i][0].max())).all()) for i in range(8)]
# (C) uniform length 700, B=8, different contents: no padding at all
up = [texts[j][:700] for j in range(8)]
su = [lp(single_t0(p)[0][None])[0] for p in up]
bu = batched_t0(up, 8)
out["C_uniform700_B8_t0_kl"] = [float(kl(su[i][None], lp(bu[i][None]))[0]) for i in range(8)]
# (D) same set as A but every prompt shorter than the window (lengths <= 513) -> mixed padding, no rotation
short = [texts[j][:n] for j, n in enumerate([37, 300, 500, 400, 64, 200, 450, 150])]
ss = [lp(single_t0(p)[0][None])[0] for p in short]
bs = batched_t0(short, 8)
out["D_mixed_short_B8_t0_kl"] = [float(kl(ss[i][None], lp(bs[i][None]))[0]) for i in range(8)]
# (E) mixed lengths all >= 1024 (no 1-pass/2-pass split between alone and batch in NoPE layers)
longp = [texts[j][:n] for j, n in enumerate([1030, 1100, 1060, 1090, 1040, 1100, 1070, 1050])]
sl = [lp(single_t0(p)[0][None])[0] for p in longp]
bl = batched_t0(longp, 8)
out["E_mixed_long_B8_t0_kl"] = [float(kl(sl[i][None], lp(bl[i][None]))[0]) for i in range(8)]
for k, v in out.items():
    if isinstance(v, list) and v and isinstance(v[0], float):
        out[k] = {"per_seq": [float(f"{x:.3e}") for x in v], "mean": float(np.mean(v))}
print(json.dumps(out, indent=1))
