import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, numpy as np
mx.random.seed(0)
H, I = 2560, 512
X = mx.random.normal((2048, H)).astype(mx.bfloat16)
Wq = (mx.random.normal((I, H)) * 0.02).astype(mx.bfloat16)
wq, s, b = mx.quantize(Wq, group_size=64, bits=8)
f = lambda M: np.array(mx.quantized_matmul(X[:M], wq, s, b, transpose=True, group_size=64, bits=8)[:1].astype(mx.float32))
base = f(16)
for M in (32, 64, 512, 2048):
    print(f"q8 row0 M=16 vs M={M}: frac_diff {(base != f(M)).mean():.4f}")
W = (mx.random.normal((384, H)) * 0.02).astype(mx.float32)
g = lambda M: np.array((X[:M].astype(mx.float32) @ W.T)[:1])
base = g(8)
for M in (2, 4, 16, 64, 2048):
    print(f"fp32 router row0 M=8 vs M={M}: maxabs {np.abs(base-g(M)).max():.3g} frac_diff {(base != g(M)).mean():.4f}")
print(mx.metal.device_info() if hasattr(mx,'metal') else mx.device_info())
