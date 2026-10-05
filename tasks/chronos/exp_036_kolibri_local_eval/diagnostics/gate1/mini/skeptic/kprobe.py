"""Kernel probe at Kolibri's REAL decode shapes: is the B=8 decode path the same
kernel family (bitwise) as the B=1 decode path, and how far is each from fp64?
Run on the mini (M4 Pro) now; the same file runs unchanged on the mbp (M5 Max).
No model weights needed (random weights at real shapes)."""
import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx
import numpy as np

rng = np.random.default_rng(36)
H, E, K, I, V = 2560, 384, 6, 512, 128000
out = {"device": mx.device_info()["device_name"], "arch": mx.device_info()["architecture"],
       "mlx": mx.__version__, "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}


def rel(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def rows(f, x, M):
    return mx.concatenate([f(x[i:i + 1]) for i in range(M)])


# 1. Router: fp32 activations @ fp32 weight [384, 2560] (port Router.__call__)
W = (rng.standard_normal((E, H)) * 0.02).astype(np.float32)
W = np.array(mx.array(W).astype(mx.bfloat16).astype(mx.float32))  # bf16 values held in fp32
Wm = mx.array(W)
res = {}
for M in (1, 2, 4, 8, 16, 64, 2048):
    x = rng.standard_normal((M, H)).astype(np.float32)
    x = np.array(mx.array(x).astype(mx.bfloat16).astype(mx.float32))
    ref = x.astype(np.float64) @ W.astype(np.float64).T
    f = lambda z: z @ Wm.T
    yb = f(mx.array(x)); y1 = rows(f, mx.array(x), M)
    mx.eval(yb, y1)
    yb, y1 = np.array(yb), np.array(y1)
    res[M] = {"rel_batched_vs_f64": rel(yb, ref), "rel_rowwise_vs_f64": rel(y1, ref),
              "bitwise_batched_eq_rowwise": bool((yb == y1).all()),
              "max_abs_batched_minus_rowwise": float(np.abs(yb - y1).max())}
out["router_fp32_gemm"] = res

# 2. q8 g64 quantized matmuls at real shapes, bf16 and fp32 activations, M=8 vs rowwise
shapes = {"q_proj": (6144, H), "k_proj": (512, H), "o_proj": (H, 6144), "shared_down": (H, I),
          "lm_head_fp32in": (V, H)}
qres = {}
for name, (N, Kd) in shapes.items():
    w = (rng.standard_normal((N, Kd)) * 0.02).astype(np.float32)
    wq, s, b = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=64, bits=8)
    deq = np.array(mx.dequantize(wq, s, b, group_size=64, bits=8).astype(mx.float32)).astype(np.float64)
    dts = (mx.float32,) if name == "lm_head_fp32in" else (mx.bfloat16, mx.float32)
    for dt in dts:
        x = rng.standard_normal((8, Kd)).astype(np.float32)
        xm = mx.array(x).astype(dt)
        ref = np.array(xm.astype(mx.float32)).astype(np.float64) @ deq.T
        f = lambda z: mx.quantized_matmul(z, wq, s.astype(dt) if dt == mx.float32 and name != "lm_head_fp32in" else s, b.astype(dt) if dt == mx.float32 and name != "lm_head_fp32in" else b, transpose=True, group_size=64, bits=8)
        y8 = f(xm); y1 = rows(f, xm, 8)
        mx.eval(y8, y1)
        y8n, y1n = np.array(y8.astype(mx.float32)), np.array(y1.astype(mx.float32))
        qres[f"{name}_{dt}"] = {"rel_M8_vs_f64": rel(y8n, ref), "rel_M1_vs_f64": rel(y1n, ref),
                                 "bitwise_M8_eq_M1": bool((y8n == y1n).all())}
out["qmm"] = qres

# 3. SwitchGLU-style gather_qmm decode: B=8 tokens x top-6 vs B=1 (q8, bf16)
from mlx_lm.models.switch_layers import SwitchGLU
import mlx.nn as nn
sg = SwitchGLU(H, I, E)
sg.set_dtype(mx.bfloat16)
nn.quantize(sg, group_size=64, bits=8)
mx.eval(sg.parameters())
x = mx.array(rng.standard_normal((8, 1, H)).astype(np.float32)).astype(mx.bfloat16)
idx = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(8)])[:, None, :].astype(np.uint32))
y8 = sg(x, idx); y1 = mx.concatenate([sg(x[i:i + 1], idx[i:i + 1]) for i in range(8)])
mx.eval(y8, y1)
out["switchglu_q8_bf16_B8_vs_B1_bitwise"] = bool((np.array(y8.astype(mx.float32)) == np.array(y1.astype(mx.float32))).all())
out["switchglu_q8_bf16_B8_vs_B1_maxabs"] = float(np.abs(np.array(y8.astype(mx.float32)) - np.array(y1.astype(mx.float32))).max())

# 4. Decode SDPA, 48 q / 4 kv heads, head_dim 128, bf16: one short sequence (valid
#    keys 40) alone vs inside a batch whose buffer is 1,164 keys (heavy left padding,
#    the 37-token prompt next to the 1,100-token prompt in the NoPE BatchKVCache)
Dh, Hq, Hk = 128, 48, 4
def sdpa_case(valid, total, dt):
    q = rng.standard_normal((1, Hq, 1, Dh)).astype(np.float32)
    k = rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)
    v = rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)
    qm, km, vm = (mx.array(a).astype(dt) for a in (q, k, v))
    alone = mx.fast.scaled_dot_product_attention(qm, km, vm, scale=Dh ** -0.5, mask=None)
    pad = total - valid
    kp = mx.concatenate([mx.array(rng.standard_normal((1, Hk, pad, Dh)).astype(np.float32)).astype(dt), km], axis=2)
    vp = mx.concatenate([mx.array(rng.standard_normal((1, Hk, pad, Dh)).astype(np.float32)).astype(dt), vm], axis=2)
    mask = mx.array(np.arange(total) >= pad)[None, None, None, :]
    padded = mx.fast.scaled_dot_product_attention(qm, kp, vp, scale=Dh ** -0.5, mask=mask)
    # fp64 truth
    qq, kk, vv = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (qm, km, vm))
    kk = np.repeat(kk, Hq // Hk, axis=1); vv = np.repeat(vv, Hq // Hk, axis=1)
    s = (qq @ kk.transpose(0, 1, 3, 2)) * Dh ** -0.5
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    ref = p @ vv
    mx.eval(alone, padded)
    a, b = np.array(alone.astype(mx.float32)), np.array(padded.astype(mx.float32))
    return {"bitwise_alone_eq_padded": bool((a == b).all()), "rel_alone_vs_f64": rel(a, ref), "rel_padded_vs_f64": rel(b, ref)}
out["sdpa_decode"] = {f"{dt}_valid{v}_total{t}": sdpa_case(v, t, dt)
                      for dt in (mx.bfloat16, mx.float32) for v, t in ((40, 1164), (40, 513), (600, 1164), (513, 513))}

# 5. RoPE: array offsets (batched) vs scalar offset (single), bf16
rope = nn.RoPE(Dh, traditional=False, base=10000.0)
offs = [36, 299, 699, 1099, 63, 519, 899, 149]
q = mx.array(rng.standard_normal((8, Hq, 1, Dh)).astype(np.float32)).astype(mx.bfloat16)
yb = rope(q, offset=mx.array(offs))
y1 = mx.concatenate([rope(q[i:i + 1], offset=o) for i, o in enumerate(offs)])
mx.eval(yb, y1)
out["rope_bf16_array_vs_scalar_bitwise"] = bool((np.array(yb.astype(mx.float32)) == np.array(y1.astype(mx.float32))).all())

print(json.dumps(out, indent=1))
