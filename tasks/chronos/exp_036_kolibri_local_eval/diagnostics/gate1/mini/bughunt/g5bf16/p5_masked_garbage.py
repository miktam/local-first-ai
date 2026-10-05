"""Does a masked-out key/value's CONTENT change the SDPA output (bf16, decode and prefill)?
First-wave rows keep the pad tokens' real K/V in their masked left region; mid-run rows
get zeros (mx.pad in BatchKVCache.extend)."""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import mlx.core as mx, numpy as np
rng = np.random.default_rng(11)
Hq, Hk, D = 48, 4, 128
def f(a): return np.array(a.astype(mx.float32)).astype(np.float64)
for total, valid in ((1100, 37), (1100, 300), (1100, 700), (513, 37), (513, 300), (2100, 300)):
    for gscale in (1.0, 30.0):
        q = mx.array(rng.standard_normal((8, Hq, 1, D)).astype(np.float32) * 2).astype(mx.bfloat16)
        k = rng.standard_normal((8, Hk, total, D)).astype(np.float32) * 2
        v = rng.standard_normal((8, Hk, total, D)).astype(np.float32)
        pad = total - valid
        mask = np.ones((8, 1, 1, total), bool); mask[:, :, :, :pad] = False
        kz, vz = k.copy(), v.copy(); kz[:, :, :pad] = 0; vz[:, :, :pad] = 0
        kg, vg = k.copy(), v.copy(); kg[:, :, :pad] *= gscale; vg[:, :, :pad] *= gscale
        o_z = mx.fast.scaled_dot_product_attention(q, mx.array(kz).astype(mx.bfloat16), mx.array(vz).astype(mx.bfloat16), scale=D ** -0.5, mask=mx.array(mask))
        o_g = mx.fast.scaled_dot_product_attention(q, mx.array(kg).astype(mx.bfloat16), mx.array(vg).astype(mx.bfloat16), scale=D ** -0.5, mask=mx.array(mask))
        mx.eval(o_z, o_g)
        print(f"decode total={total} valid={valid} garbage x{gscale}: max|o_garbage - o_zeros| = {np.abs(f(o_g) - f(o_z)).max():.3e}")
# prefill: right-padded row, queries are valid tokens only, pad keys are in the causal future
from mlx_lm.models.base import create_causal_mask
L = 1099
for valid in (37, 300, 700):
    q = mx.array(rng.standard_normal((8, Hq, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
    k = rng.standard_normal((8, Hk, L, D)).astype(np.float32) * 2
    v = rng.standard_normal((8, Hk, L, D)).astype(np.float32)
    kg, vg = k.copy(), v.copy(); kg[:, :, valid:] *= 30; vg[:, :, valid:] *= 30
    m = create_causal_mask(L, 0, window_size=513, left_padding=mx.zeros((8,), mx.int32))
    o1 = mx.fast.scaled_dot_product_attention(q, mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16), scale=D ** -0.5, mask=m)
    o2 = mx.fast.scaled_dot_product_attention(q, mx.array(kg).astype(mx.bfloat16), mx.array(vg).astype(mx.bfloat16), scale=D ** -0.5, mask=m)
    mx.eval(o1, o2)
    print(f"prefill valid={valid}: max|Δ| on valid queries = {np.abs(f(o1)[:, :, :valid] - f(o2)[:, :, :valid]).max():.3e}")
