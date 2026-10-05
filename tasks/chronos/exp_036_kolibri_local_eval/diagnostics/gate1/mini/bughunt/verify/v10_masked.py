"""bf16 SDPA: masked-out keys' CONTENT must not change the output (first-wave rows keep pad-token K/V in their
masked region; mid-run rows get zeros). Decode (qL=1, B=8 boolean mask) and prefill (qL=1099) at Kolibri shapes."""
from common_v import *
from mlx_lm.models.base import create_causal_mask
rng = np.random.default_rng(11); Hq, Hk, D = 48, 4, 128
def f(a): return np.array(a.astype(mx.float32))
for total, valid in ((1100, 37), (1100, 700), (513, 37), (513, 300)):
    for g in (0.0, 30.0, 1e4):
        q = mx.array(rng.standard_normal((8, Hq, 1, D)).astype(np.float32) * 2).astype(mx.bfloat16)
        k = rng.standard_normal((8, Hk, total, D)).astype(np.float32) * 2; v = rng.standard_normal((8, Hk, total, D)).astype(np.float32)
        pad = total - valid; m = np.ones((8, 1, 1, total), bool); m[..., :pad] = False
        kz, vz = k.copy(), v.copy(); kz[:, :, :pad] = 0; vz[:, :, :pad] = 0
        kg, vg = k.copy(), v.copy(); kg[:, :, :pad] *= g if g else 1; vg[:, :, :pad] *= g if g else 1
        o1 = mx.fast.scaled_dot_product_attention(q, mx.array(kz).astype(mx.bfloat16), mx.array(vz).astype(mx.bfloat16), scale=D ** -0.5, mask=mx.array(m))
        o2 = mx.fast.scaled_dot_product_attention(q, mx.array(kg).astype(mx.bfloat16), mx.array(vg).astype(mx.bfloat16), scale=D ** -0.5, mask=mx.array(m))
        print(f"decode kL {total} valid {valid} pad content x{g:g} vs zeros: bitwise {np.array_equal(f(o1), f(o2))} max|d| {np.abs(f(o1)-f(o2)).max():.1e}")
L = 1099
for valid in (36, 699):
    q = mx.array(rng.standard_normal((8, Hq, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
    k = rng.standard_normal((8, Hk, L, D)).astype(np.float32) * 2; v = rng.standard_normal((8, Hk, L, D)).astype(np.float32)
    kg, vg = k.copy(), v.copy(); kg[:, :, valid:] *= 1e4; vg[:, :, valid:] *= 1e4
    for win in (513, None):
        msk = create_causal_mask(L, 0, window_size=win, left_padding=mx.zeros((8,), mx.int32))
        o1 = mx.fast.scaled_dot_product_attention(q, mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16), scale=D ** -0.5, mask=msk)
        o2 = mx.fast.scaled_dot_product_attention(q, mx.array(kg).astype(mx.bfloat16), mx.array(vg).astype(mx.bfloat16), scale=D ** -0.5, mask=msk)
        a, b = f(o1)[:, :, :valid], f(o2)[:, :, :valid]
        print(f"prefill right-pad valid {valid} window {win}: valid queries bitwise {np.array_equal(a, b)} max|d| {np.abs(a-b).max():.1e}")
