"""E1: after the first-wave batched prefill (B=8, right-padded, finalize), is each
row's KV cache bitwise equal to the same prompt prefilled alone? And the t0 logits?"""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, mlx.nn as nn, numpy as np
from pathlib import Path
from gate import common
from mlx_lm.generate import BatchGenerator

ckpt = sys.argv[1]; prec = sys.argv[2]
model, cfg, module = common.load_port(Path(ckpt))
if prec == "fp32": model.set_dtype(mx.float32)
elif prec in ("q8", "q4"):
    nn.quantize(model, group_size=64, bits=8 if prec == "q8" else 4, class_predicate=model.quant_predicate)
mx.eval(model.parameters())
V = model.args.vocab_size
rng = np.random.default_rng(1)
lens = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [rng.integers(0, V - 16, size=n).tolist() for n in lens]

def f(a): return np.array(a.astype(mx.float32))

def batched(group):
    gen = BatchGenerator(model, max_tokens=4, stop_tokens=[[V - 2]], sampler=lambda x: mx.argmax(x, -1),
                         completion_batch_size=8, prefill_batch_size=8, prefill_step_size=2048, max_kv_size=None)
    gen.insert([prompts[j] for j in group], max_tokens=[4] * len(group))
    gen.next()  # prefill + finalize
    pc = gen._prompt_batch.prompt_cache
    caches = [[c.extract(i) for c in pc] for i in range(len(group))]
    for cs in caches: mx.eval([c.state for c in cs])
    gen.next()  # move to generation; t0 computed in GenerationBatch.__init__
    gb = gen._generation_batch
    lp0 = [f(x) for x in gb._next_logprobs]
    gen.close()
    return caches, lp0

def single(j):
    cache = model.make_cache()
    p = prompts[j]
    mx.eval(model(mx.array(p[:-1])[None], cache=cache))
    st = [(mx.array(c.state[0]), mx.array(c.state[1])) for c in cache]; mx.eval(st)
    out = model(mx.array(p[-1:])[None], cache=cache)[0, -1].astype(mx.float32)
    lp = out - mx.logsumexp(out)
    return st, f(lp)

cB, lpB = batched(list(range(8)))
for j in range(8):
    cS, lpS = single(j)
    c1, lp1 = batched([j])
    worst = []
    for li, (a, b, c) in enumerate(zip(cB[j], cS, c1[0])):
        ka, kb = f(a.keys), f(b[0])
        # temporal order of the rotating single cache: it holds the last max_size (or all) keys
        n = min(ka.shape[2], kb.shape[2])
        worst.append((li, ka.shape[2], kb.shape[2], float(np.abs(ka[..., -n:, :] - kb[..., -n:, :]).max()),
                      float(np.abs(f(a.values)[..., -n:, :] - f(b[1])[..., -n:, :]).max())))
    kl0 = float((np.exp(lpS) * (lpS - lpB[j])).sum())
    kl1 = float((np.exp(lpS) * (lpS - lp1[0])).sum())
    print(f"row {j} len {lens[j]}: KL(single||B8 t0)={kl0:.3e} KL(single||B1batch t0)={kl1:.3e}  max|dK|,|dV| per layer:",
          [(w[0], w[3], w[4]) for w in worst if w[3] or w[4]][:6] or "all bitwise")
