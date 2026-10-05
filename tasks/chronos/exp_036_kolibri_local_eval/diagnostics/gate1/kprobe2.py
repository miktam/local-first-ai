"""Kernel probe at Kolibri's real DECODE shapes, B=8 rows vs B=1 rows, against an EXACT
float64 reference. Runs unchanged on the mini (M4 Pro) and the mbp (M5 Max); random
weights at real shapes, no model weights. Fixes kprobe.py's reference, which dequantised
with bf16 scales (a bf16 rounding of every weight) and so put a ~1.6e-3 floor under every
qmm comparison, hiding any precision difference smaller than that.

Decision rule (frozen in DIAGNOSIS §4 stage 1): for any shape, if B=8 is not bitwise equal
to B=1 AND rel_B8 > 1.2 x rel_B1 (both vs float64 of the same rounded inputs), the B=8
kernel loses precision: a kernel defect on that chip, to be reported upstream, not noise.
"""
import os, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import numpy as np
import mlx.core as mx
import mlx.nn as nn

rng = np.random.default_rng(36)
H, E, K, I, V, Dh, Hq, Hk = 2560, 384, 6, 512, 128000, 128, 48, 4
info = mx.device_info()
out = {"device": info.get("device_name"), "arch": info.get("architecture"), "mlx": mx.__version__,
       "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}


def f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


def rel(a, ref):
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


def cmp(y8, y1, ref):
    a, b = f64(y8), f64(y1)
    if y1.dtype == mx.bfloat16:  # differences in bf16 ulps of the B=1 value (bf16 outputs only)
        d = np.abs(a - b) / np.exp2(np.floor(np.log2(np.maximum(np.abs(b), 1e-30))) - 7)
    else:
        d = np.zeros(1)
    return {"bitwise": bool(np.array_equal(a, b)), "rel_B8": rel(a, ref), "rel_B1": rel(b, ref),
            "ratio": rel(a, ref) / max(rel(b, ref), 1e-30), "max_abs_B8_minus_B1": float(np.abs(a - b).max()),
            "frac_gt_1ulp": float((d > 1.0).mean()), "max_ulp": float(d.max())}


def qweights(N, Kd, bits=8):
    w = mx.array((rng.standard_normal((N, Kd)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    # exact dequantisation: fp32 scales/biases, float64 product (no bf16 rounding of w)
    deq = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)).astype(np.float64)
    return wq, s, b, deq


res = {}
# 1. router: fp32 activations (cast from bf16) @ fp32 weight, [B,1,H] x [E,H]^T
W = mx.array((rng.standard_normal((E, H)) * 0.02).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
for M in (8, 16, 48):
    x = mx.array(rng.standard_normal((M, 1, H)).astype(np.float32)).astype(mx.bfloat16)
    f = lambda z: z.astype(mx.float32) @ W.T
    y8 = f(x); y1 = mx.concatenate([f(x[i:i + 1]) for i in range(M)]); mx.eval(y8, y1)
    res[f"router_fp32_M{M}"] = cmp(y8, y1, (f64(x)[:, 0] @ f64(W).T)[:, None, :])
# 2. q8 projections, bf16 activations [8,1,K] (decode), and the fp32-input head
for name, (N, Kd, dt) in {"q_proj": (6144, H, mx.bfloat16), "k_proj": (512, H, mx.bfloat16),
                          "o_proj": (H, 6144, mx.bfloat16), "shared_gate": (I, H, mx.bfloat16),
                          "shared_down": (H, I, mx.bfloat16), "lm_head_fp32in": (V, H, mx.float32)}.items():
    wq, s, b, deq = qweights(N, Kd)
    x = mx.array(rng.standard_normal((8, 1, Kd)).astype(np.float32)).astype(mx.bfloat16).astype(dt)
    f = lambda z: mx.quantized_matmul(z, wq, s, b, transpose=True, group_size=64, bits=8)
    y8 = f(x); y1 = mx.concatenate([f(x[i:i + 1]) for i in range(8)]); mx.eval(y8, y1)
    res[name] = cmp(y8, y1, f64(x) @ deq.T)
# 3. SwitchGLU q8 bf16, B=8 x top-6 (48 rows: the unsorted gather path, < 64) vs B=1 (6 rows),
#    and B=11 x 6 = 66 rows (the sorted path) vs B=1; reference = the same module in fp32
from mlx_lm.models.switch_layers import SwitchGLU
sg = SwitchGLU(H, I, E)
sg.set_dtype(mx.bfloat16)
nn.quantize(sg, group_size=64, bits=8)
mx.eval(sg.parameters())
sg32 = SwitchGLU(H, I, E)
sg32.set_dtype(mx.bfloat16)
nn.quantize(sg32, group_size=64, bits=8)
sg32.update(sg.parameters())
sg32.set_dtype(mx.float32)
mx.eval(sg32.parameters())
for Bt in (8, 11):
    x = mx.array(rng.standard_normal((Bt, 1, H)).astype(np.float32)).astype(mx.bfloat16)
    idx = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(Bt)])[:, None, :].astype(np.uint32))
    y8 = sg(x, idx); y1 = mx.concatenate([sg(x[i:i + 1], idx[i:i + 1]) for i in range(Bt)])
    ref = sg32(x.astype(mx.float32), idx); mx.eval(y8, y1, ref)
    res[f"switchglu_q8_B{Bt}x6"] = cmp(y8, y1, f64(ref))
# 4. decode SDPA bf16, 48q/4kv, head_dim 128: a short sequence (valid keys v) alone vs at the
#    same row inside a B=8 batch whose buffer is `total` keys (left padding, boolean mask)
def sdpa(valid, total, n=10):
    e1, e8, bw = [], [], True
    for _ in range(n):
        q = mx.array(rng.standard_normal((1, Hq, 1, Dh)).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array(rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array(rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)).astype(mx.bfloat16)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh ** -0.5, mask=None)
        pad = total - valid
        kp = mx.concatenate([mx.zeros((1, Hk, pad, Dh), dtype=mx.bfloat16), k], axis=2)
        vp = mx.concatenate([mx.zeros((1, Hk, pad, Dh), dtype=mx.bfloat16), v], axis=2)
        ko = mx.array(rng.standard_normal((7, Hk, total, Dh)).astype(np.float32)).astype(mx.bfloat16)
        qo = mx.array(rng.standard_normal((7, Hq, 1, Dh)).astype(np.float32)).astype(mx.bfloat16)
        mask = np.ones((8, 1, 1, total), bool); mask[0, :, :, :pad] = False
        b = mx.fast.scaled_dot_product_attention(mx.concatenate([q, qo]), mx.concatenate([kp, ko]),
                                                 mx.concatenate([vp, ko]), scale=Dh ** -0.5, mask=mx.array(mask))[:1]
        mx.eval(a, b)
        qq, kk, vv = f64(q), np.repeat(f64(k), Hq // Hk, 1), np.repeat(f64(v), Hq // Hk, 1)
        s_ = (qq @ kk.transpose(0, 1, 3, 2)) * Dh ** -0.5
        p = np.exp(s_ - s_.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        ref = p @ vv
        e1.append(rel(f64(a), ref)); e8.append(rel(f64(b), ref)); bw &= bool(np.array_equal(f64(a), f64(b)))
    return {"bitwise": bw, "rel_B1": float(np.mean(e1)), "rel_B8": float(np.mean(e8)), "ratio": float(np.mean(e8) / np.mean(e1))}
for v_, t_ in ((37, 1100), (300, 1100), (700, 1100), (300, 513), (513, 513), (1100, 1164)):
    res[f"sdpa_bf16_valid{v_}_buffer{t_}"] = sdpa(v_, t_)
out["probes"] = res
out["flagged"] = [k for k, r in res.items() if not r["bitwise"] and (r["ratio"] > 1.2 or r.get("frac_gt_1ulp", 0) > 0.01)]
# Verdict against a baseline from another chip (argv[1], e.g. kprobe2_m4pro.json): a probe is a
# chip-specific precision defect iff it is not bitwise here, its ratio exceeds 1.2 x the baseline's
# ratio (1.0 where the baseline is bitwise), and, for fp32 outputs, rel_B8 > 1e-5 (else it is
# accumulation order at fp32 eps, immaterial next to bf16 noise ~1e-3).
import sys
if len(sys.argv) > 1:
    base = json.load(open(sys.argv[1]))["probes"]
    defects = []
    for k, r in res.items():
        b = base.get(k)
        bratio = 1.0 if (b is None or b["bitwise"]) else b["ratio"]
        fp32_out = k.startswith("router") or k.startswith("lm_head")
        if (not r["bitwise"]) and r["ratio"] > 1.2 * bratio and (not fp32_out or r["rel_B8"] > 1e-5):
            defects.append(k)
    out["baseline"] = {"file": sys.argv[1], "chip_specific_defects": defects,
                       "verdict": "DEFECT" if defects else "no chip-specific B=8 precision loss"}
print(json.dumps(out, indent=1))
