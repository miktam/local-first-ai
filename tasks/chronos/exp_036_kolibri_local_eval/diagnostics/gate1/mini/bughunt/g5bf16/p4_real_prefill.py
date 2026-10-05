import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, numpy as np
from mlx_lm.models.base import create_causal_mask
rng = np.random.default_rng(7)
def f64(a): return np.array(a.astype(mx.float32)).astype(np.float64)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
L, H, E = 1099, 2560, 384
lens = [36, 299, 699, 1099, 63, 519, 899, 149]
# router fp32
W = mx.array((rng.standard_normal((E, H)) * 0.02).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
x = mx.array(rng.standard_normal((8, L, H)).astype(np.float32)).astype(mx.bfloat16)
y8 = x.astype(mx.float32) @ W.T; mx.eval(y8)
res = []
for r, n in enumerate(lens):
    y1 = x[r:r+1, :n].astype(mx.float32) @ W.T; mx.eval(y1)
    res.append((r, n, bool(np.array_equal(np.array(y8[r:r+1, :n]), np.array(y1))), float(np.abs(np.array(y8[r:r+1, :n]) - np.array(y1)).max())))
print("router fp32", res)
# SDPA prefill 48/4 bf16, mask [8,1,L,L] causal(+window) vs [1,1,n,n]
for win in (None, 513):
    q = mx.array(rng.standard_normal((8, 48, L, 128)).astype(np.float32) * 2).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((8, 4, L, 128)).astype(np.float32) * 2).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((8, 4, L, 128)).astype(np.float32)).astype(mx.bfloat16)
    m8 = create_causal_mask(L, 0, window_size=win, left_padding=mx.zeros((8,), mx.int32))
    o8 = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5, mask=m8); mx.eval(o8)
    res = []
    for r, n in enumerate(lens):
        m1 = create_causal_mask(n, 0, window_size=win, left_padding=mx.zeros((1,), mx.int32))
        o1 = mx.fast.scaled_dot_product_attention(q[r:r+1, :, :n], k[r:r+1, :, :n], v[r:r+1, :, :n], scale=128 ** -0.5, mask=m1); mx.eval(o1)
        res.append((r, n, bool(np.array_equal(f64(o8[r:r+1, :, :n]), f64(o1))), float(np.abs(f64(o8[r:r+1, :, :n]) - f64(o1)).max())))
    print("sdpa prefill 48/4 win", win, res)
