"""Reference sdpa_ref: query chunking (attn_chunk) at large offsets, sliding and full, vs dense float64."""
import sys; sys.path.insert(0, "$KIT")
import numpy as np
from reference.kolibri_ref import sdpa_ref
rng = np.random.default_rng(4)
T, H, Hk, D = 9000, 8, 2, 128
q = (3 * rng.standard_normal((T, H, D))).astype(np.float32); k = rng.standard_normal((T, Hk, D)).astype(np.float32); v = rng.standard_normal((T, Hk, D)).astype(np.float32)
rows = np.r_[0:40, 500:530, 2040:2060, 8100:8192]
def dense(window):
    rep = H // Hk; out = np.empty((rows.size, H, D))
    kk = np.arange(T)[None]; qi = rows[:, None]; m = kk <= qi
    if window: m &= kk > qi - window
    for h in range(H):
        s = (q[rows, h].astype(np.float64) @ k[:, h // rep].astype(np.float64).T) * D ** -0.5
        s = np.where(m, s, -np.inf); s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        out[:, h] = p @ v[:, h // rep].astype(np.float64)
    return out
for window in (513, None):
    t = dense(window)
    for qc in (None, 1, 64, 513, 1000, 4096):
        o = sdpa_ref(q, k, v, D ** -0.5, window, qc)[rows]
        print(f"window {window} attn_chunk {qc}: max rel err vs dense f64 {np.abs(o - t).max() / np.abs(t).max():.1e}", flush=True)
