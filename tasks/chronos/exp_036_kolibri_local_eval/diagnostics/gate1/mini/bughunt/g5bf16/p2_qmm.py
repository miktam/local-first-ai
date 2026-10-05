"""Prefill-shape probe: qmm / bf16 matmul / gather (SwitchGLU) at B=8 x L rows vs B=1 x L rows."""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, mlx.nn as nn, numpy as np
rng = np.random.default_rng(3)
def f64(a): return np.array(a.astype(mx.float32)).astype(np.float64)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
L = 1099
for (N, Kd) in ((128, 256), (512, 2560), (6144, 2560), (2560, 6144)):
    w = mx.array((rng.standard_normal((N, Kd)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    deq = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=8)).astype(np.float64)
    x = mx.array(rng.standard_normal((8, L, Kd)).astype(np.float32)).astype(mx.bfloat16)
    for name, fn, W64 in (("qmm8", lambda z: mx.quantized_matmul(z, wq, s, b, transpose=True, group_size=64, bits=8), deq),
                          ("bf16mm", lambda z: z @ w.T, f64(w))):
        y8 = fn(x); mx.eval(y8)
        y1 = mx.concatenate([fn(x[i:i+1]) for i in range(8)]); mx.eval(y1)
        yf = mx.concatenate([fn(x[i:i+1].reshape(1, -1, Kd)[:, :37]) for i in range(1)])
        ref = f64(x) @ W64.T
        a, c = f64(y8), f64(y1)
        print(f"{name} N={N} K={Kd}: bitwise={np.array_equal(a, c)} max|B8-B1|={np.abs(a-c).max():.3e} rel_B8={rel(a, ref):.4e} rel_B1={rel(c, ref):.4e}")
