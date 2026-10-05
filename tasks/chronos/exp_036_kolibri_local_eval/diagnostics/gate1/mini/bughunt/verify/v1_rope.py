"""RoPE: (a) chunked mx.fast.rope calls vs one call (bitwise); (b) MLX vs reference (vLLM fp32 angle) vs float64 truth;
(c) which inv_freq formula MLX uses (bitwise angle extraction)."""
from common_v import *
from reference.kolibri_ref import rope_neox, rope_inv_freq
D, H = 128, 4
rng = np.random.default_rng(0)
rope = nn.RoPE(D, traditional=False, base=10000.0)
# (a) chunk consistency at large offsets
P0, N = 14336, 965
x = rng.standard_normal((1, H, N, D)).astype(np.float32)
big = np.array(rope(mx.array(x), offset=P0))
for step in (1, 8, 64, 513, 2048):
    parts = [np.array(rope(mx.array(x[:, :, a:a + step]), offset=P0 + a)) for a in range(0, N, step)]
    y = np.concatenate(parts, 2)
    print(f"(a) chunk {step:5d} at offset {P0}: bitwise equal to one call: {np.array_equal(y, big)}  max|d| {np.abs(y-big).max():.1e}")
# (b) accuracy vs float64 truth
def truth(x, pos):
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float64) / D))
    ang = pos[:, None].astype(np.float64) * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x = x.astype(np.float64); x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
print("(b) max abs error per element (|x|~1), unit-norm-ish inputs")
for off in (0, 500, 1000, 2048, 4096, 8192, 12000, 15000, 16000):
    L = 64
    xx = rng.standard_normal((L, H, D)).astype(np.float32)
    pos = np.arange(off, off + L)
    t = truth(xx, pos); r = rope_neox(xx, pos, 10000.0)
    y = np.array(rope(mx.array(xx.transpose(1, 0, 2))[None], offset=off))[0].transpose(1, 0, 2)
    print(f"  pos {off:6d}+64: |mlx-truth| {np.abs(y-t).max():.2e}  |ref-truth| {np.abs(r-t).max():.2e}  |mlx-ref| {np.abs(y-r).max():.2e}")
# (c) extract MLX angles from unit vectors: x1=1, x2=0 -> out1=cos, out2=sin
pos = np.array([1, 100, 1000, 15000, 15300], np.int64)
e = np.zeros((1, 1, 1, D), np.float32); e[..., :D // 2] = 1
cs = []
for p in pos:
    o = np.array(rope(mx.array(e), offset=int(p)))[0, 0, 0]
    cs.append((o[:D // 2], o[D // 2:]))
inv_ref = rope_inv_freq(D, 10000.0)  # vLLM fp32
d = np.arange(0, D, 2, dtype=np.float32) / np.float32(D)
inv_exp2 = np.exp2(-d * np.log2(np.float32(10000.0))).astype(np.float32)
print("(c) inv_freq: vLLM-fp32 vs exp2-form max rel diff", float(np.max(np.abs(inv_ref.astype(np.float64) - inv_exp2) / inv_ref)))
for (c, s), p in zip(cs, pos):
    ang_mlx = np.arctan2(s.astype(np.float64), c.astype(np.float64))
    for nm, inv in (("vLLM fp32", inv_ref), ("exp2 fp32", inv_exp2)):
        a32 = (np.float32(p) * inv).astype(np.float32).astype(np.float64)
        dd = np.angle(np.exp(1j * (ang_mlx - a32)))
        print(f"  pos {p:6d} {nm}: max |angle_mlx - angle_formula| {np.abs(dd).max():.2e} rad")
    tr = p * (1.0 / (10000.0 ** (np.arange(0, D, 2) / D)))
    print(f"  pos {p:6d} mlx vs f64 truth max {np.abs(np.angle(np.exp(1j*(ang_mlx-tr)))).max():.2e} rad; vLLM-fp32 vs truth {np.abs(np.angle(np.exp(1j*((np.float32(p)*inv_ref).astype(np.float64)-tr)))).max():.2e} rad")
