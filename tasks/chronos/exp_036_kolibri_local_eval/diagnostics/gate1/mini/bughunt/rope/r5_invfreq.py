"""Recover MLX fast.rope's per-dimension inv_freq exactly from p = 2^k probes
(fp32(p*inv) is exact for p a power of two), then compare in fp32 ulps with
vLLM/reference 1/theta^(2j/D) and with exp2(-(j/(D/2))*log2(theta))."""
import sys, json
sys.path.insert(0, "$KIT")
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, numpy as np
from reference.kolibri_ref import rope_inv_freq
D, TH = 128, 10000.0; H = D // 2
inv_ref = rope_inv_freq(D, TH)
def mlx_angle(p):
    x = np.zeros((1, 1, 1, D), np.float32); x[..., :H] = 1
    y = np.array(mx.fast.rope(mx.array(x), D, traditional=False, base=TH, scale=1.0, offset=int(p)))[0, 0, 0].astype(np.float64)
    return np.arctan2(y[H:], y[:H])
est = []
for k in (8, 10, 12):
    p = 2 ** k
    a = mlx_angle(p)
    n = np.round((p * inv_ref.astype(np.float64) - a) / (2 * np.pi))
    est.append((a + 2 * np.pi * n) / p)
inv_m = est[-1]
inv_m32 = inv_m.astype(np.float32)
ul = lambda a, b: (a.astype(np.float32).view(np.int32).astype(np.int64) - b.astype(np.float32).view(np.int32).astype(np.int64))
d = (np.arange(H, dtype=np.float32) / np.float32(H))
cands = {
 "exp2(-d*log2(theta)) fp32": np.exp2((-d * np.float32(np.log2(TH))).astype(np.float32)).astype(np.float32),
 "exp2 in float64 of fp32 product": np.exp2((-d * np.float32(np.log2(TH))).astype(np.float64)).astype(np.float32),
 "correctly rounded theta^-2j/D": (TH ** (-np.arange(0, D, 2) / D)).astype(np.float32),
}
out = {"rel_spread_between_k_estimates": float(np.abs(est[0] / est[-1] - 1).max()),
       "mlx_vs_ref_ulps": np.unique(ul(inv_m32, inv_ref), return_counts=True),
       "mlx_vs_ref_rel_max": float(np.abs(inv_m / inv_ref - 1).max())}
for k, v in cands.items():
    out[f"mlx_vs[{k}]_ulps"] = np.unique(ul(inv_m32, v), return_counts=True)
for k, v in out.items():
    print(k, (list(zip(v[0].tolist(), v[1].tolist())) if isinstance(v, tuple) else v))
