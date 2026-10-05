"""SwitchGLU (sorted gather path) at prefill shapes: row r inside B=8 x 1099 tokens vs alone."""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, mlx.nn as nn, numpy as np
from mlx_lm.models.switch_layers import SwitchGLU
rng = np.random.default_rng(5)
def f64(a): return np.array(a.astype(mx.float32)).astype(np.float64)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
H, I, E, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
L = 1099
lens = [36, 299, 699, 1099, 63, 519, 899, 149]
for quant in (False, True):
    sg = SwitchGLU(H, I, E); sg.set_dtype(mx.bfloat16)
    if quant: nn.quantize(sg, group_size=64, bits=8)
    mx.eval(sg.parameters())
    sg32 = SwitchGLU(H, I, E); sg32.set_dtype(mx.bfloat16)
    if quant: nn.quantize(sg32, group_size=64, bits=8)
    sg32.update(sg.parameters()); sg32.set_dtype(mx.float32); mx.eval(sg32.parameters())
    x = mx.array(rng.standard_normal((8, L, H)).astype(np.float32)).astype(mx.bfloat16)
    idx = np.stack([[rng.choice(E, k, replace=False) for _ in range(L)] for _ in range(8)]).astype(np.uint32)
    # pad tokens route to the same experts (as identical pad ids would)
    for r, n in enumerate(lens):
        idx[r, n:] = idx[r, n - 1] if False else np.arange(k)
    idx = mx.array(idx)
    y8 = sg(x, idx); mx.eval(y8)
    out = []
    for r, n in enumerate(lens):
        y1 = sg(x[r:r+1, :n], idx[r:r+1, :n]); ref = sg32(x[r:r+1, :n].astype(mx.float32), idx[r:r+1, :n]); mx.eval(y1, ref)
        a, b, c = f64(y8[r:r+1, :n]), f64(y1), f64(ref)
        out.append((r, n, bool(np.array_equal(a, b)), round(rel(a, c), 6), round(rel(b, c), 6)))
    print("quant" if quant else "bf16", f"H={H} I={I} E={E} k={k}", out)
