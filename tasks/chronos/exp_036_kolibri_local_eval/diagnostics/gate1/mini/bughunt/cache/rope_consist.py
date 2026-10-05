from common_bh import *
D, H = 128, 4
rng = np.random.default_rng(1)
rope = nn.RoPE(D, traditional=False, base=10000.0)
P0, N = 14336, 965
x = rng.standard_normal((1, H, N, D)).astype(np.float32)
xm = mx.array(x)
big = np.array(rope(xm, offset=P0))
for step in (1, 8, 9, 64, 256, 512, 513, 514):
    parts = []
    for a in range(0, N, step):
        parts.append(np.array(rope(xm[:, :, a:a + step], offset=P0 + a)))
    y = np.concatenate(parts, axis=2)
    d = np.abs(y - big)
    print(f"step {step:4d} vs one call of {N}: max {d.max():.3e}  n_pos_diff {(d.max(axis=(0,1,3))>0).sum()}")
