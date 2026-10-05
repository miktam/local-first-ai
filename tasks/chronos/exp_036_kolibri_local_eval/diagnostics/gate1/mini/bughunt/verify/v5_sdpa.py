"""fp32 SDPA at Kolibri's attention shapes (48 q / 4 kv heads, D 128) and the T9 offsets, against float64:
full layers ("causal" string, kL = offset + qL) and sliding layers (boolean window mask from the port's own
cache), for qL = 1 (decode), 64 and 965/2048 (prefill chunks)."""
from common_v import *
from mlx_lm.models.base import create_causal_mask
D, Hq, Hk = 128, 48, 4
rng = np.random.default_rng(2)
def truth(q, k, v, q0, window=None):
    Lq, Lk = q.shape[1], k.shape[1]; rep = Hq // Hk
    qi = q0 + np.arange(Lq)[:, None]; kj = np.arange(Lk)[None]
    m = kj <= qi
    if window: m &= kj > qi - window
    out = np.empty((Hq, Lq, D))
    for h in range(Hq):
        s = (q[h].astype(np.float64) @ k[h // rep].astype(np.float64).T) * D ** -0.5
        s = np.where(m, s, -np.inf); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        out[h] = p @ v[h // rep].astype(np.float64)
    return out
def rel(a, b): return float(np.abs(a - b).max() / np.abs(b).max())
for sharp in (1.0, 6.0):
    Lk = 15301
    k = rng.standard_normal((Hk, Lk, D)).astype(np.float32); v = rng.standard_normal((Hk, Lk, D)).astype(np.float32)
    for qL in (1, 64, 965, 2048):
        q = (sharp * rng.standard_normal((Hq, qL, D))).astype(np.float32)
        t = truth(q, k, v, Lk - qL)
        y = np.array(mx.fast.scaled_dot_product_attention(mx.array(q)[None], mx.array(k)[None], mx.array(v)[None],
                     scale=D ** -0.5, mask="causal" if qL > 1 else None))[0]
        print(f"full  sharp {sharp} qL {qL:5d} kL {Lk}: max rel err vs f64 {rel(y, t):.1e}", flush=True)
    # sliding: cache holds offset-window keys (512) + qL new; mask from create_causal_mask(qL, 512, window 513)
    for qL in (1, 64, 965):
        Lks = 512 + qL
        ks, vs = k[:, :Lks], v[:, :Lks]
        q = (sharp * rng.standard_normal((Hq, qL, D))).astype(np.float32)
        t = truth(q, ks, vs, 512, window=513)
        mask = None if qL == 1 else create_causal_mask(qL, 512, window_size=513)
        y = np.array(mx.fast.scaled_dot_product_attention(mx.array(q)[None], mx.array(ks)[None], mx.array(vs)[None],
                     scale=D ** -0.5, mask=mask))[0]
        print(f"slide sharp {sharp} qL {qL:5d} kL {Lks}: max rel err vs f64 {rel(y, t):.1e}", flush=True)
# fp32 GEMM at the real hidden size, M = 64 vs 2048 rows, rows bitwise equal?
W = mx.array(rng.standard_normal((6144, 2560)).astype(np.float32) * 0.02)
X = mx.array(rng.standard_normal((2048, 2560)).astype(np.float32))
y_big = np.array(X @ W.T); y64 = np.concatenate([np.array(X[a:a+64] @ W.T) for a in range(0, 2048, 64)])
y1 = np.concatenate([np.array(X[a:a+1] @ W.T) for a in range(0, 64)])
t = np.array(X).astype(np.float64) @ np.array(W).astype(np.float64).T
print(f"GEMM K=2560 fp32: M=2048 rel {rel(y_big, t):.1e}, M=64 rel {rel(y64, t):.1e}, M=1 rel {rel(y1, t[:64]):.1e}; "
      f"M=64 rows bitwise == M=2048 rows: {np.array_equal(y64, y_big)}; M=1 == M=2048: {np.array_equal(y1, y_big[:64])}")
