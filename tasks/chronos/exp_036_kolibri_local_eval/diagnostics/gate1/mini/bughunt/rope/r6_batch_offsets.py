"""mx.fast.rope with per-example array offsets (BatchKVCache style, incl.
negative offsets from left padding) vs per-example scalar offsets, B=8,
[B, heads, L, D] transposed view as in the port; fp32 and bf16; L=1, 64, 1099."""
import sys
sys.path.insert(0, "$KIT")
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, numpy as np
D = 128
rng = np.random.default_rng(1)
for dt in (mx.float32, mx.bfloat16):
    for L in (1, 64, 1099):
        offs = np.array([-37, 0, 5, 512, 1099, 8191, 15000, 16000], np.int32)
        x = rng.standard_normal((8, L, 48, D)).astype(np.float32)
        xm = mx.array(x).astype(dt).transpose(0, 2, 1, 3)
        ya = np.array(mx.fast.rope(xm, D, traditional=False, base=1e4, scale=1.0, offset=mx.array(offs)).astype(mx.float32))
        eq = []
        for b in range(8):
            o = int(offs[b])
            if o < 0:   # scalar path cannot take a negative int offset; compare only the rows at >= 0 positions
                yb = np.array(mx.fast.rope(xm[b:b+1, :, -o:], D, traditional=False, base=1e4, scale=1.0, offset=0).astype(mx.float32))
                eq.append(bool(np.array_equal(ya[b:b+1, :, -o:], yb)))
            else:
                yb = np.array(mx.fast.rope(xm[b:b+1], D, traditional=False, base=1e4, scale=1.0, offset=o).astype(mx.float32))
                eq.append(bool(np.array_equal(ya[b:b+1], yb)))
        print(dt, "L", L, "array-offset == scalar-offset per example:", eq)
