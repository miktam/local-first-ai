"""Does giving mx.fast.rope vLLM's fp32 inv_freq (freqs = 1/inv_freq) align the port's angles with the reference's?"""
import sys
K = "$KIT"
sys.path.insert(0, K)
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import numpy as np, mlx.core as mx
from reference.kolibri_ref import rope_neox, rope_inv_freq
D, TH = 128, 10000.0
rng = np.random.default_rng(1)
inv32 = rope_inv_freq(D, TH)
for start in [0, 3911, 15000, 16000]:
    x = rng.standard_normal((300, 4, D)).astype(np.float32)
    pos = np.arange(start, start + 300)
    r = rope_neox(x, pos, TH)
    xm = mx.array(x.transpose(1, 0, 2))[None]
    a = np.array(mx.fast.rope(xm, D, traditional=False, base=TH, scale=1.0, offset=start)[0]).transpose(1, 0, 2)
    b = np.array(mx.fast.rope(xm, D, traditional=False, base=None, scale=1.0, offset=start,
                              freqs=mx.array((1.0 / inv32).astype(np.float32)))[0]).transpose(1, 0, 2)
    # what an exact cos/sin of the fp32 angle gives (numpy float64 cos on the fp32 angle)
    ang = (pos.astype(np.float32)[:, None] * inv32[None]).astype(np.float64)
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x64 = x.astype(np.float64); h = D // 2
    e = np.concatenate([x64[..., :h] * c - x64[..., h:] * s, x64[..., h:] * c + x64[..., :h] * s], -1)
    print(f"pos {start:>6}: |port(base)-ref| {np.abs(a-r).max():.2e}  |port(freqs=vLLM)-ref| {np.abs(b-r).max():.2e}  "
          f"|ref - exactcos(fp32 angle)| {np.abs(r-e).max():.2e}  |port(base)-exactcos(fp32 angle)| {np.abs(a-e).max():.2e}  |port(freqs)-exactcos| {np.abs(b-e).max():.2e}")
