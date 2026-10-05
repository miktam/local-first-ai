import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, numpy as np
from mlx_lm.models.switch_layers import SwitchGLU
mx.random.seed(1)
H, I, E, k = 2560, 512, 32, 6
glu = SwitchGLU(H, I, E)
glu.set_dtype(mx.bfloat16)
import mlx.nn as nn
nn.quantize(glu, group_size=64, bits=8)
X = mx.random.normal((64, H)).astype(mx.bfloat16)
idx = mx.array(np.stack([np.random.default_rng(i).choice(E, k, replace=False) for i in range(64)]).astype(np.uint32))
def run(M):
    return glu(X[:M], idx[:M])  # SwitchGLU sorts when indices.size >= 64
ref = run(1)
for M in (2, 8, 10, 11, 16, 64):
    out = run(M)[:1]
    a = np.array(ref.astype(mx.float32)); b = np.array(out.astype(mx.float32))
    print(f"SwitchGLU q8 row0 M=1 vs M={M:2d} (indices.size={M*k:3d}, sort={M*k>=64}): maxabs {np.abs(a-b).max():.3g} frac_diff {(a!=b).mean():.4f}")
