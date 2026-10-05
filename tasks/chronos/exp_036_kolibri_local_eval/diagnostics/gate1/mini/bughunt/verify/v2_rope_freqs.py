"""Can the port reproduce vLLM's fp32 angle exactly via mx.fast.rope(freqs=...)?  The kernel's freqs are
'inverse frequencies' in MLX docs? check both freqs=inv and freqs=1/inv conventions."""
from common_v import *
from reference.kolibri_ref import rope_neox, rope_inv_freq
D = 128
inv = rope_inv_freq(D, 10000.0)
e = np.zeros((1, 1, 1, D), np.float32); e[..., :D // 2] = 1
def angles(fn, p):
    o = np.array(fn(mx.array(e), p))[0, 0, 0]
    return np.arctan2(o[D // 2:].astype(np.float64), o[:D // 2].astype(np.float64)), o
for nm, fr in (("freqs=1/inv (periods)", (np.float32(1) / inv).astype(np.float32)), ("freqs=inv", inv)):
    fn = lambda x, p, fr=fr: mx.fast.rope(x, D, traditional=False, base=None, scale=1.0, offset=p, freqs=mx.array(fr))
    for p in (1, 1000, 15000, 15300):
        a, o = angles(fn, p)
        a32 = (np.float32(p) * inv).astype(np.float64)
        d = np.abs(np.angle(np.exp(1j * (a - a32)))).max()
        # bitwise cos/sin vs fp32 cos/sin of the vLLM angle (numpy fp64 cos of the fp32 angle, rounded)
        c_ref = np.cos(a32).astype(np.float32); s_ref = np.sin(a32).astype(np.float32)
        print(f"{nm:24s} pos {p:6d}: max|angle - vLLM fp32 angle| {d:.2e} rad; max|cos-cos_vLLM| {np.abs(o[:D//2]-c_ref).max():.1e} max|sin-sin_vLLM| {np.abs(o[D//2:]-s_ref).max():.1e}")
rng = np.random.default_rng(1)
x = rng.standard_normal((64, 4, D)).astype(np.float32)
for off in (0, 1000, 15000):
    pos = np.arange(off, off + 64)
    r = rope_neox(x, pos, 10000.0)
    fr = (np.float32(1) / inv).astype(np.float32)
    y = np.array(mx.fast.rope(mx.array(x.transpose(1, 0, 2))[None], D, traditional=False, base=None, scale=1.0, offset=off, freqs=mx.array(fr)))[0].transpose(1, 0, 2)
    y0 = np.array(nn.RoPE(D, traditional=False, base=10000.0)(mx.array(x.transpose(1, 0, 2))[None], offset=off))[0].transpose(1, 0, 2)
    print(f"pos {off}+64: |freqs-aligned port - ref| max {np.abs(y-r).max():.2e}   |default port - ref| max {np.abs(y0-r).max():.2e}")
