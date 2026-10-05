"""Which op in layer 0 makes a row's batched (B=8, right-padded) prefill differ from its B=1 prefill?"""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, mlx.nn as nn, numpy as np
from pathlib import Path
from gate import common
from mlx_lm.generate import _right_pad_prompts, _merge_caches
from mlx_lm.models.cache import make_prompt_cache
ckpt, prec = sys.argv[1], sys.argv[2]
model, cfg, module = common.load_port(Path(ckpt))
if prec == "fp32": model.set_dtype(mx.float32)
elif prec in ("q8", "q4"):
    nn.quantize(model, group_size=64, bits=8 if prec == "q8" else 4, class_predicate=model.quant_predicate)
mx.eval(model.parameters())
V = model.args.vocab_size
rng = np.random.default_rng(1)
lens = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [rng.integers(0, V - 16, size=n - 1).tolist() for n in lens]
def f(a): return np.array(a.astype(mx.float32))
def run(group):
    caches = _merge_caches([make_prompt_cache(model) for _ in group])
    toks = [prompts[j] for j in group]
    L = [len(t) for t in toks]; mlen = max(L); pad = [mlen - l for l in L]
    x = _right_pad_prompts(toks, mlen)
    if max(pad) > 0:
        for c in caches: c.prepare(lengths=L, right_padding=pad)
    inner = model.model
    h = inner.embed_tokens(x)
    fa, swa = inner.make_masks(h, caches)
    out = {}
    lay = inner.layers[0]
    hn = lay.input_layernorm(h)
    att = lay.self_attn
    B_, L_, _ = hn.shape
    q = att.q_proj(hn); k = att.k_proj(hn); v = att.v_proj(hn)
    out["q_proj"] = q; out["k_proj"] = k
    a = att(hn, swa if lay.use_sliding else fa, caches[0])
    out["attn_o"] = a
    r = lay.post_attn_norm(a); hm = h + r
    xm = lay.post_attention_layernorm(hm)
    out["router"] = lay.mlp.gate(xm)
    y, logits, ids = lay.mlp.forward_routed(xm)
    out["ids"] = ids
    out["shared"] = lay.mlp.shared_experts(xm)
    out["moe"] = y
    mx.eval(out)
    return {k_: (f(v_) if v_.dtype != mx.uint32 else np.array(v_)) for k_, v_ in out.items()}, L
oB, LB = run(list(range(8)))
for j in range(8):
    o1, _ = run([j])
    n = lens[j] - 1
    diffs = {}
    for k_ in oB:
        a, b = oB[k_][j, :n], o1[k_][0, :n]
        if k_ == "ids":
            diffs[k_] = int((np.sort(a, -1) != np.sort(b, -1)).any(-1).sum())
        else:
            diffs[k_] = float(np.abs(a - b).max())
    print(j, n, diffs)
