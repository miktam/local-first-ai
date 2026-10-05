"""bf16 decode SDPA (48q/4kv, head_dim 128): error vs fp64 as a function of the
key-buffer length (alone, no mask) and of left padding in a longer buffer (mask).
Averaged over 20 draws. Shows where MLX switches to the 2-pass vector kernel."""
import os, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import mlx.core as mx
import numpy as np

Dh, Hq, Hk = 128, 48, 4
rng = np.random.default_rng(7)


def truth(q, k, v):
    qq, kk, vv = (np.array(a.astype(mx.float32)).astype(np.float64) for a in (q, k, v))
    kk = np.repeat(kk, Hq // Hk, axis=1); vv = np.repeat(vv, Hq // Hk, axis=1)
    s = (qq @ kk.transpose(0, 1, 3, 2)) * Dh ** -0.5
    p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return p @ vv


def err(valid, total, n=20, B=1, qscale=1.0):
    e_alone, e_pad = [], []
    for _ in range(n):
        q = mx.array((rng.standard_normal((1, Hq, 1, Dh)) * qscale).astype(np.float32)).astype(mx.bfloat16)
        k = mx.array(rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)).astype(mx.bfloat16)
        v = mx.array(rng.standard_normal((1, Hk, valid, Dh)).astype(np.float32)).astype(mx.bfloat16)
        ref = truth(q, k, v)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=Dh ** -0.5, mask=None)
        pad = total - valid
        kp = mx.concatenate([mx.zeros((1, Hk, pad, Dh), dtype=mx.bfloat16), k], axis=2)
        vp = mx.concatenate([mx.zeros((1, Hk, pad, Dh), dtype=mx.bfloat16), v], axis=2)
        if B > 1:  # same row inside a batch of B (other rows random, full length)
            ko = mx.array(rng.standard_normal((B - 1, Hk, total, Dh)).astype(np.float32)).astype(mx.bfloat16)
            qo = mx.array(rng.standard_normal((B - 1, Hq, 1, Dh)).astype(np.float32)).astype(mx.bfloat16)
            qp = mx.concatenate([q, qo]); kp = mx.concatenate([kp, ko]); vp = mx.concatenate([vp, ko])
            mask = np.ones((B, 1, 1, total), bool); mask[0, :, :, :pad] = False
        else:
            qp = q
            mask = np.ones((1, 1, 1, total), bool); mask[0, :, :, :pad] = False
        b = mx.fast.scaled_dot_product_attention(qp, kp, vp, scale=Dh ** -0.5, mask=mx.array(mask))[:1]
        mx.eval(a, b)
        a, b = np.array(a.astype(mx.float32)), np.array(b.astype(mx.float32))
        e_alone.append(np.linalg.norm(a - ref) / np.linalg.norm(ref))
        e_pad.append(np.linalg.norm(b - ref) / np.linalg.norm(ref))
    return float(np.mean(e_alone)), float(np.mean(e_pad))


out = {"device": mx.device_info()["device_name"]}
alone = {}
for L in (256, 512, 513, 800, 1000, 1023, 1024, 1025, 1100, 1164, 2048, 4096, 16384):
    alone[L] = err(L, L, n=10)[0]
out["alone_rel_err_by_K"] = alone
padded = {}
for valid, total in ((40, 1164), (300, 1164), (600, 1164), (900, 1164), (1100, 1164), (600, 1000), (600, 1023), (600, 1024), (800, 1100)):
    a, b = err(valid, total, n=20, B=8)
    padded[f"{valid}/{total}"] = {"alone": a, "in_batch_padded": b, "ratio": b / a}
out["padded_B8"] = padded
print(json.dumps(out, indent=1))
