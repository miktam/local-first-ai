"""Is MLX itself batch-invariant? Same row through the same op, alone vs in a
batch. Kolibri shapes: hidden 2560, expert inter 512, 8-bit affine gs64."""
import os
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx, numpy as np
mx.random.seed(0)
H, I, E = 2560, 512, 384
def rel(a, b):
    a = np.array(a.astype(mx.float32)); b = np.array(b.astype(mx.float32))
    return float(np.abs(a - b).max()), float((a != b).mean())
# 1. dense bf16 matmul (router-like, attention proj): M=1 vs M=8 vs M=64
W = (mx.random.normal((E, H)) * 0.02).astype(mx.bfloat16)
X = mx.random.normal((64, H)).astype(mx.bfloat16)
y1 = X[:1] @ W.T
for M in (2, 8, 16, 64):
    yM = (X[:M] @ W.T)[:1]
    print(f"bf16 matmul row0  M=1 vs M={M}: maxabs {rel(y1, yM)[0]:.3g}  frac_diff {rel(y1, yM)[1]:.4f}")
# fp32 output (router logits path): x.astype(f32) @ W.astype(f32)
y1 = X[:1].astype(mx.float32) @ W.astype(mx.float32).T
for M in (8, 64):
    yM = (X[:M].astype(mx.float32) @ W.astype(mx.float32).T)[:1]
    print(f"fp32 matmul row0  M=1 vs M={M}: maxabs {rel(y1, yM)[0]:.3g}  frac_diff {rel(y1, yM)[1]:.4f}")
# 2. 8-bit quantized matmul (qmv for M=1, qmm for larger M)
Wq = (mx.random.normal((I, H)) * 0.02).astype(mx.bfloat16)
wq, s, b = mx.quantize(Wq, group_size=64, bits=8)
y1 = mx.quantized_matmul(X[:1], wq, s, b, transpose=True, group_size=64, bits=8)
for M in (2, 8, 16, 64):
    yM = mx.quantized_matmul(X[:M], wq, s, b, transpose=True, group_size=64, bits=8)[:1]
    print(f"q8 matmul row0    M=1 vs M={M}: maxabs {rel(y1, yM)[0]:.3g}  frac_diff {rel(y1, yM)[1]:.4f}")
# 3. gather_qmm (SwitchGLU path), 6 experts per token, sorted vs unsorted
We = (mx.random.normal((16, I, H)) * 0.02).astype(mx.bfloat16)
weq, se, be = mx.quantize(We, group_size=64, bits=8)
idx = mx.array(np.random.default_rng(0).integers(0, 16, size=(64, 6)).astype(np.uint32))
x = X[:, None, None, :]
one = mx.gather_qmm(x[:1], weq, se, be, rhs_indices=idx[:1], transpose=True, group_size=64, bits=8)
for M in (8, 64):
    many = mx.gather_qmm(x[:M], weq, se, be, rhs_indices=idx[:M], transpose=True, group_size=64, bits=8)[:1]
    print(f"gather_qmm row0   M=1 vs M={M}: maxabs {rel(one, many)[0]:.3g}  frac_diff {rel(one, many)[1]:.4f}")
# 4. SDPA: decode query against 600 keys, alone vs batch of 8 with left padding + mask
nq, nkv, D, L = 48, 4, 128, 600
q = mx.random.normal((1, nq, 1, D)).astype(mx.bfloat16)
k = mx.random.normal((1, nkv, L, D)).astype(mx.bfloat16)
v = mx.random.normal((1, nkv, L, D)).astype(mx.bfloat16)
o1 = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
pad = 400
kb = mx.concatenate([mx.zeros((1, nkv, pad, D), dtype=mx.bfloat16), k], axis=2)
vb = mx.concatenate([mx.zeros((1, nkv, pad, D), dtype=mx.bfloat16), v], axis=2)
mask = mx.array(np.arange(pad + L) >= pad)[None, None, None, :]
ob = mx.fast.scaled_dot_product_attention(mx.repeat(q, 8, 0), mx.repeat(kb, 8, 0), mx.repeat(vb, 8, 0), scale=D ** -0.5, mask=mask)[:1]
print(f"sdpa decode       alone vs B=8 left-pad {pad} + bool mask: maxabs {rel(o1, ob)[0]:.3g}  frac_diff {rel(o1, ob)[1]:.4f}")
o2 = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=mx.ones((1,1,1,L), dtype=mx.bool_))
print(f"sdpa decode       no mask vs all-true mask: maxabs {rel(o1, o2)[0]:.3g}  frac_diff {rel(o1, o2)[1]:.4f}")
