"""MLX fast.rope vs reference cos/sin per position band; candidate port-side
fixes: (a) fast.rope with freqs=1/inv_ref, (b) an fp32 cos/sin cache built in
MLX as vLLM builds it (angle = fp32(t * inv_freq), mx.cos/mx.sin), applied
NeoX-style. Bitwise/ulp comparison with the reference's numpy rope_neox."""
import json
import sys

sys.path.insert(0, "$KIT")
from tools.precision import ensure_exact_fp32

ensure_exact_fp32()
import mlx.core as mx
import numpy as np

from reference.kolibri_ref import rope_inv_freq, rope_neox

D, THETA, N = 128, 10000.0, 16385
H = D // 2
pos = np.arange(N)
inv_ref = rope_inv_freq(D, THETA)
inv64 = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
ang64 = pos[:, None] * inv64[None, :]

probe = np.zeros((N, 1, D), np.float32)
probe[..., :H] = 1.0
r = rope_neox(probe, pos, THETA)[:, 0]
c_ref, s_ref = r[:, :H], r[:, H:]


def mlx_probe(**kw):
    x = mx.array(probe.transpose(1, 0, 2)[None])  # [1, 1, N, D]
    y = np.array(mx.fast.rope(x, D, traditional=False, scale=1.0, offset=0, **kw))[0, 0]
    return y[:, :H], y[:, H:]


out = {}
bands = [(0, 512), (512, 1100), (1100, 4096), (4096, 8192), (8192, 12288), (12288, 15000), (15000, 15301), (15301, 16385)]


def band_table(c, s):
    e = np.maximum(np.abs(c.astype(np.float64) - c_ref), np.abs(s.astype(np.float64) - s_ref))
    t = np.maximum(np.abs(c.astype(np.float64) - np.cos(ang64)), np.abs(s.astype(np.float64) - np.sin(ang64)))
    return {f"{a}-{b}": {"vs_ref_max": float(e[a:b].max()), "vs_ref_mean": float(e[a:b].mean()),
                         "vs_ref_frac_bitwise_equal": float((e[a:b] == 0).mean()),
                         "vs_float64_max": float(t[a:b].max())} for a, b in bands}


c_m, s_m = mlx_probe(base=THETA)
out["mlx_fast_rope_base"] = band_table(c_m, s_m)
tr = np.maximum(np.abs(c_ref - np.cos(ang64)), np.abs(s_ref - np.sin(ang64)))
out["ref_vs_float64"] = {f"{a}-{b}": float(tr[a:b].max()) for a, b in bands}

# effective MLX inv_freq at p = 1 (angle = inv_freq; fast trig is accurate near 0 for small angles)
eff = np.arctan2(s_m[1].astype(np.float64), c_m[1].astype(np.float64))
out["mlx_inv_freq_vs_ref_rel_at_p1_max"] = float(np.abs(eff / inv_ref - 1).max())

# (a) fast.rope with freqs: kernel uses inv_freq = 1/freqs
c_a, s_a = mlx_probe(base=None, freqs=mx.array((1.0 / inv_ref.astype(np.float64)).astype(np.float32)))
out["fix_a_fast_rope_freqs"] = band_table(c_a, s_a)

# (b) fp32 cos/sin cache in MLX (vLLM rope_base.py:94-103), NeoX apply
t = mx.arange(N, dtype=mx.float32)
ang = t[:, None] * mx.array(inv_ref)[None, :]
cb, sb = mx.cos(ang), mx.sin(ang)
mx.eval(cb, sb)
out["fix_b_mlx_fp32_cache_precise_trig"] = band_table(np.array(cb), np.array(sb))
out["fix_b_angle_bitwise_equal_ref"] = bool(np.array_equal(np.array(ang), (pos[:, None].astype(np.float32) * inv_ref[None, :]).astype(np.float32)))

json.dump(out, open(sys.argv[1], "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict) and "0-512" in v and isinstance(v["0-512"], dict):
        print(k)
        for b, d in v.items():
            print(f"   {b:12s} vs_ref max {d['vs_ref_max']:.2e} mean {d['vs_ref_mean']:.2e} bitwise_eq {d['vs_ref_frac_bitwise_equal']:.3f} | vs_f64 max {d['vs_float64_max']:.2e}")
    else:
        print(k, v)
