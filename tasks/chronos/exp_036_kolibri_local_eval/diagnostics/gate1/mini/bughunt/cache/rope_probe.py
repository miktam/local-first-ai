from common_bh import *
from reference.kolibri_ref import rope_neox
D, H = 128, 4
rng = np.random.default_rng(0)
def truth(x, pos):  # float64, exact angle
    half = D // 2
    inv = 1.0 / (10000.0 ** (np.arange(0, D, 2, dtype=np.float64) / D))
    ang = pos[:, None].astype(np.float64) * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x = x.astype(np.float64); x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)
rope = nn.RoPE(D, traditional=False, base=10000.0)
for off in (0, 1000, 4000, 8000, 15000, 16000):
    for L in (1, 8, 64, 513, 2048):
        x = rng.standard_normal((L, H, D)).astype(np.float32)
        pos = np.arange(off, off + L)
        t = truth(x, pos)
        r = rope_neox(x, pos, 10000.0)
        xm = mx.array(x.transpose(1, 0, 2))[None]  # [1,H,L,D]
        y = np.array(rope(xm, offset=off))[0].transpose(1, 0, 2)
        print(f"off {off:6d} L {L:5d}  mlx-truth max {np.abs(y-t).max():.2e}  ref-truth max {np.abs(r-t).max():.2e}  mlx-ref max {np.abs(y-r).max():.2e}")
