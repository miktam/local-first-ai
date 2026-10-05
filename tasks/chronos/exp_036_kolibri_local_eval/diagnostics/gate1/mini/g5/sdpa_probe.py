import os, time
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, numpy as np
def manual(q,k,v,scale,mask):
    rep = q.shape[1]//k.shape[1]
    k = mx.repeat(k, rep, axis=1); v = mx.repeat(v, rep, axis=1)
    s = (q*scale) @ k.swapaxes(-1,-2)
    if mask is not None:
        s = mx.where(mask, s, -mx.inf)
    return mx.softmax(s.astype(mx.float32), axis=-1).astype(q.dtype) @ v
for D in (32, 128):
    for L,K in ((1,513),(1,1156),(8,513),(300,812)):
        rng = np.random.default_rng(0)
        q = mx.array(rng.standard_normal((8,12,L,D)).astype(np.float32))
        k = mx.array(rng.standard_normal((8,1,K,D)).astype(np.float32))
        v = mx.array(rng.standard_normal((8,1,K,D)).astype(np.float32))
        mask = mx.array(rng.random((8,1,L,K))>0.1)
        for msk in (None, mask):
            a = mx.fast.scaled_dot_product_attention(q,k,v,scale=D**-0.5,mask=msk)
            b = manual(q,k,v,D**-0.5,msk)
            d = float(mx.abs(a-b).max())
            # timing hint: fused kernels are much faster at large sizes
            print(f"D={D} L={L} K={K} mask={'y' if msk is not None else 'n'} max|fused-manual|={d:.3e} {'BITWISE' if d==0 else ''}")
