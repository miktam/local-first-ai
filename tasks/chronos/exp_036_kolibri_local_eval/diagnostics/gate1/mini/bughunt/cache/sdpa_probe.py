from common_bh import *
from mlx_lm.models.base import create_causal_mask
D = 128
rng = np.random.default_rng(2)
def truth(q, k, v, scale, q0, window=None):
    # q [Hq, Lq, D], k/v [Hk, Lk, D]; query i at absolute key index q0+i
    Hq, Lq, _ = q.shape; Hk, Lk, _ = k.shape; rep = Hq // Hk
    out = np.empty((Hq, Lq, D))
    kk = k.astype(np.float64); vv = v.astype(np.float64)
    qi = q0 + np.arange(Lq)[:, None]; kj = np.arange(Lk)[None]
    m = kj <= qi
    if window: m &= kj > qi - window
    for h in range(Hq):
        s = (q[h].astype(np.float64) @ kk[h // rep].T) * scale
        s = np.where(m, s, -np.inf); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        out[h] = p @ vv[h // rep]
    return out
for Hq, Hk in ((48, 4), (12, 1)):
  for sharp in (1.0, 8.0):
    Lk = 15301
    k = rng.standard_normal((Hk, Lk, D)).astype(np.float32)
    v = rng.standard_normal((Hk, Lk, D)).astype(np.float32)
    # a "sink" key at position 0 and some others that every query loves
    for qL in (1, 8, 9, 64, 965, 2048):
        q = (sharp * rng.standard_normal((Hq, qL, D))).astype(np.float32)
        t = truth(q, k, v, D ** -0.5, Lk - qL)
        y = mx.fast.scaled_dot_product_attention(mx.array(q)[None], mx.array(k)[None], mx.array(v)[None], scale=D ** -0.5,
                                                 mask="causal" if qL > 1 else None)
        y = np.array(y)[0]
        e = np.abs(y - t).max()
        print(f"Hq {Hq} Hk {Hk} sharp {sharp} qL {qL:5d} kL {Lk}: causal-string max abs err vs f64 {e:.2e}", flush=True)
