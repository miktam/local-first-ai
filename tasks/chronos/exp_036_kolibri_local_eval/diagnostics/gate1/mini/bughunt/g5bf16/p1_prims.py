"""Primitive probe: does any op give a row a different (or less precise) bf16
result at B=8 (right-padded batch) than at B=1?"""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K]
from tools.precision import ensure_exact_fp32
print(ensure_exact_fp32())
import mlx.core as mx, numpy as np
rng = np.random.default_rng(0)
H, Hk, D = 12, 1, 128
L = 1099
lens = [36, 299, 699, 1099, 63, 519, 899, 149]

def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))

# ---------- P1 prefill SDPA
q = rng.standard_normal((8, H, L, D)).astype(np.float32) * 3
k = rng.standard_normal((8, Hk, L, D)).astype(np.float32) * 3
v = rng.standard_normal((8, Hk, L, D)).astype(np.float32)
scale = D ** -0.5
from mlx_lm.models.base import create_causal_mask
def sdpa(qq, kk, vv, mask, dt):
    o = mx.fast.scaled_dot_product_attention(mx.array(qq).astype(dt), mx.array(kk).astype(dt), mx.array(vv).astype(dt), scale=scale, mask=mask)
    return np.array(o.astype(mx.float32))
for win in (None, 513):
    mB = create_causal_mask(L, 0, window_size=win, left_padding=mx.zeros((8,), mx.int32))  # [8,1,L,L]
    oB = sdpa(q, k, v, mB, mx.bfloat16)
    worst_bit = 0; errs = []
    for r in range(8):
        m1 = create_causal_mask(L, 0, window_size=win, left_padding=mx.zeros((1,), mx.int32))
        o1 = sdpa(q[r:r+1], k[r:r+1], v[r:r+1], m1, mx.bfloat16)
        oc = sdpa(q[r:r+1], k[r:r+1], v[r:r+1], "causal" if win is None else create_causal_mask(L, 0, window_size=win), mx.bfloat16)
        o32 = sdpa(q[r:r+1], k[r:r+1], v[r:r+1], m1, mx.float32)
        n = lens[r]
        errs.append((r, float(np.abs(oB[r, :, :n] - o1[0, :, :n]).max()), float(np.abs(o1[0,:,:n]-oc[0,:,:n]).max()), rel(oB[r,:,:n], o32[0,:,:n]), rel(o1[0,:,:n], o32[0,:,:n])))
    print("P1 prefill sdpa win", win, "(row, max|B8-B1|, max|B1arr-B1causal|, relB8, relB1)")
    for e in errs: print("   ", e)

# ---------- P2 decode SDPA
for Kl in (513, 1100, 2100):
    qd = rng.standard_normal((8, H, 1, D)).astype(np.float32) * 3
    kd = rng.standard_normal((8, Hk, Kl, D)).astype(np.float32) * 3
    vd = rng.standard_normal((8, Hk, Kl, D)).astype(np.float32)
    lp = np.array([0, 100, 300, 0, 500, 50, 0, 200][:8])
    mask = mx.array(np.arange(Kl)[None, None, None, :] >= lp[:, None, None, None])
    oB = sdpa(qd, kd, vd, mask, mx.bfloat16)
    out = []
    for r in range(8):
        o1 = sdpa(qd[r:r+1], kd[r:r+1, :, lp[r]:], vd[r:r+1, :, lp[r]:], None, mx.bfloat16)
        o1m = sdpa(qd[r:r+1], kd[r:r+1], vd[r:r+1], mask[r:r+1], mx.bfloat16)
        o32 = sdpa(qd[r:r+1], kd[r:r+1, :, lp[r]:], vd[r:r+1, :, lp[r]:], None, mx.float32)
        out.append((r, int(lp[r]), float(np.abs(oB[r] - o1m[0]).max()), float(np.abs(oB[r]-o1[0]).max()), rel(oB[r], o32[0]), rel(o1[0], o32[0])))
    print("P2 decode sdpa K", Kl, "(row, lp, max|B8-B1mask|, max|B8-B1nopad|, relB8, relB1)")
    for e in out: print("   ", e)

# ---------- P3 rope
import mlx.nn as nn
rope = nn.RoPE(D, traditional=False, base=10000.0)
def rope_np(x, pos):  # x [B,H,L,D] float64 ; pos [B,L]
    half = D // 2
    inv = 10000.0 ** (-np.arange(half, dtype=np.float64) * 2 / D)
    th = pos[:, None, :, None] * inv[None, None, None, :]
    c, s = np.cos(th), np.sin(th)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], -1)
for Lq in (1099, 1):
    x = rng.standard_normal((8, H, Lq, D)).astype(np.float32)
    offs = np.array([1099, 299, 699, 0, 63, 519, 899, 149]) if Lq == 1 else np.zeros(8, int)
    xb = mx.array(x).astype(mx.bfloat16)
    xb64 = np.array(xb.astype(mx.float32)).astype(np.float64)
    oB = np.array(rope(xb, offset=mx.array(offs, dtype=mx.int32)).astype(mx.float32))
    res = []
    for r in range(8):
        o1a = np.array(rope(xb[r:r+1], offset=mx.array(offs[r:r+1], dtype=mx.int32)).astype(mx.float32))
        o1i = np.array(rope(xb[r:r+1], offset=int(offs[r])).astype(mx.float32))
        o1s = np.array(rope(xb[r:r+1], offset=mx.array(int(offs[r]), dtype=mx.int32)).astype(mx.float32))
        pos = (np.arange(Lq) + offs[r])[None]
        tr = rope_np(xb64[r:r+1], pos)
        res.append((r, int(offs[r]), float(np.abs(oB[r]-o1i[0]).max()), float(np.abs(o1a[0]-o1i[0]).max()), float(np.abs(o1s[0]-o1i[0]).max()), rel(oB[r], tr[0]), rel(o1i[0], tr[0])))
    print("P3 rope L", Lq, "(row, off, max|B8arr-B1int|, max|B1arr-B1int|, max|B1scalarr-B1int|, relB8, relB1int)")
    for e in res: print("   ", e)
