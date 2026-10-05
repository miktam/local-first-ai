"""RoPE at large positions: MLX fast.rope vs reference rope_neox vs vLLM-style
fp32 cache vs float64 truth. head_dim 128, theta 1e4, positions 0..16384.

Probe: x = [1]*64 + [0]*64 per head -> NeoX output = [cos(a), sin(a)] exactly
(x2 = 0), so the kernel's cos/sin values are read back directly.
"""
import json
import sys

sys.path.insert(0, "$KIT")
from tools.precision import ensure_exact_fp32

print(ensure_exact_fp32())
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from reference.kolibri_ref import rope_inv_freq, rope_neox

D, THETA, N = 128, 10000.0, 16385
H = D // 2
pos = np.arange(N)

# ---------------- float64 truth --------------------------------------------
inv64 = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
ang64 = pos[:, None].astype(np.float64) * inv64[None, :]
cos64, sin64 = np.cos(ang64), np.sin(ang64)

# ---------------- reference (numpy fp32) -----------------------------------
inv_ref = rope_inv_freq(D, THETA)
probe = np.zeros((N, 1, D), np.float32)
probe[..., :H] = 1.0
r = rope_neox(probe, pos, THETA)[:, 0]
cos_ref, sin_ref = r[:, :H].astype(np.float64), r[:, H:].astype(np.float64)

# ---------------- vLLM 0.29 style (rope_base.py:86-103), numpy fp32 ----------
inv_v = (np.float32(1.0) / (np.float32(THETA) ** (np.arange(0, D, 2, dtype=np.float32) / np.float32(D)))).astype(np.float32)
t = np.arange(N, dtype=np.float32)
fr = (t[:, None] * inv_v[None, :]).astype(np.float32)
cos_v32, sin_v32 = np.cos(fr), np.sin(fr)
# bf16 cast (rope_base.py:61 cache.to(dtype) with dtype bf16)
def to_bf16(a):
    return np.array(mx.array(a.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
cos_vb, sin_vb = to_bf16(cos_v32), to_bf16(sin_v32)

# ---------------- MLX ------------------------------------------------------
rope = nn.RoPE(D, traditional=False, base=THETA)


def mlx_cs(chunk, layout="BHLD", dtype=mx.float32, array_offset=False):
    """cos/sin read from fast.rope applied to the probe in chunks of `chunk`
    with scalar offset = chunk start (as the port does through cache.offset)."""
    outs = []
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        L = b - a
        x = np.zeros((1, 4, L, D), np.float32)  # [B, heads, L, D]
        x[..., :H] = 1.0
        xm = mx.array(x).astype(dtype)
        if layout == "transposed":
            # as in the port: [B, L, heads, D] -> transpose(0, 2, 1, 3) (non-contiguous view)
            xm = mx.array(np.ascontiguousarray(x.transpose(0, 2, 1, 3))).astype(dtype).transpose(0, 2, 1, 3)
        off = mx.array([a], dtype=mx.int32) if array_offset else a
        y = mx.fast.rope(xm, D, traditional=False, base=THETA, scale=1.0, offset=off)
        y = np.array(y.astype(mx.float32))[0, 0]
        outs.append(y)
    y = np.concatenate(outs, 0).astype(np.float64)
    return y[:, :H], y[:, H:]


def stats(c, s, cref, sref):
    e = np.maximum(np.abs(c - cref), np.abs(s - sref))
    out = {}
    for lo, hi in ((0, 1024), (1024, 4096), (4096, 8192), (8192, 12288), (12288, 16385), (15000, 15301)):
        out[f"{lo}-{hi}"] = {"max_abs": float(e[lo:hi].max()), "mean_abs": float(e[lo:hi].mean())}
    return out


def eff_angle_err(c, s):
    """effective angle error vs float64 truth (wrapped), rad"""
    a = np.arctan2(s, c)
    d = np.angle(np.exp(1j * (a - ang64)))
    return d


res = {"inv_freq": {}}
# inv_freq comparisons (ulps of fp32)
inv_true32 = inv64.astype(np.float32)
def ulps(a, b):
    return (a.astype(np.float32).view(np.int32).astype(np.int64) - b.astype(np.float32).view(np.int32).astype(np.int64))
res["inv_freq"]["ref_vs_correctly_rounded_ulps"] = [int(x) for x in np.unique(ulps(inv_ref, inv_true32))]
res["inv_freq"]["vllm_numpy_vs_ref_ulps"] = [int(x) for x in np.unique(ulps(inv_v, inv_ref))]

variants = {}
variants["ref_numpy_fp32"] = (cos_ref, sin_ref)
variants["vllm_fp32_cache"] = (cos_v32.astype(np.float64), sin_v32.astype(np.float64))
variants["vllm_bf16_cache"] = (cos_vb.astype(np.float64), sin_vb.astype(np.float64))
for name, kw in {
    "mlx_scalar_chunk_whole": dict(chunk=N),
    "mlx_scalar_chunk2048": dict(chunk=2048),
    "mlx_scalar_chunk64": dict(chunk=64),
    "mlx_scalar_chunk1_decode": dict(chunk=1),
    "mlx_transposed_chunk2048": dict(chunk=2048, layout="transposed"),
    "mlx_transposed_chunk64": dict(chunk=64, layout="transposed"),
    "mlx_transposed_chunk1": dict(chunk=1, layout="transposed"),
    "mlx_arrayoffset_chunk64": dict(chunk=64, array_offset=True),
    "mlx_arrayoffset_chunk2048": dict(chunk=2048, array_offset=True),
    "mlx_arrayoffset_chunk1": dict(chunk=1, array_offset=True),
}.items():
    variants[name] = mlx_cs(**kw)

base = variants["mlx_scalar_chunk2048"]
for name, (c, s) in variants.items():
    ea = eff_angle_err(c, s)
    res[name] = {
        "vs_float64": stats(c, s, cos64, sin64),
        "eff_angle_err_rad_max_12288plus": float(np.abs(ea[12288:]).max()),
        "eff_angle_err_rad_mean_12288plus": float(np.abs(ea[12288:]).mean()),
        "vs_ref_max_abs_15000_15301": float(np.maximum(np.abs(c - cos_ref), np.abs(s - sin_ref))[15000:15301].max()),
        "bitwise_equal_to_mlx_scalar_chunk2048": bool(np.array_equal(c, base[0]) and np.array_equal(s, base[1])),
        "max_abs_vs_mlx_scalar_chunk2048": float(np.maximum(np.abs(c - base[0]), np.abs(s - base[1])).max()),
    }

# Reverse-engineer MLX's angle: compare with cos of candidate fp32 angles evaluated in float64
c_m, s_m = variants["mlx_scalar_chunk2048"]
cands = {
    "fp32(p*inv_ref)": (pos[:, None].astype(np.float32) * inv_ref[None, :]).astype(np.float32),
    "fp32(p*exp2(-d*log2(theta)))": None,
}
d = (np.arange(H, dtype=np.float32) / np.float32(H)).astype(np.float32)
inv_exp2 = np.exp2((-d * np.float32(np.log2(THETA))).astype(np.float32)).astype(np.float32)
cands["fp32(p*exp2(-d*log2(theta)))"] = (pos[:, None].astype(np.float32) * inv_exp2[None, :]).astype(np.float32)
res["inv_freq"]["exp2_form_vs_ref_ulps"] = [int(x) for x in np.unique(ulps(inv_exp2, inv_ref))]
for k, a32 in cands.items():
    a = a32.astype(np.float64)
    e = np.maximum(np.abs(c_m - np.cos(a)), np.abs(s_m - np.sin(a)))
    res[f"mlx_vs_exact_cos_of_{k}"] = {"max_abs_0_1024": float(e[:1024].max()), "max_abs_12288plus": float(e[12288:].max()),
                                       "mean_abs_12288plus": float(e[12288:].mean())}

# fp32 angle quantisation alone (reference's single fp32 rounding)
a_ref32 = (pos[:, None].astype(np.float32) * inv_ref[None, :]).astype(np.float64)
res["angle_fp32_rounding_vs_float64_max_rad_12288plus"] = float(np.abs(a_ref32 - ang64)[12288:].max())

# bf16 input path: rope on bf16 x, compare rotated output of random bf16 vector vs float64 truth
rng = np.random.default_rng(0)
xr = rng.standard_normal((1, 4, N, D)).astype(np.float32)
xb = np.array(mx.array(xr).astype(mx.bfloat16).astype(mx.float32))
x1, x2 = xb[..., :H].astype(np.float64), xb[..., H:].astype(np.float64)
truth = np.concatenate([x1 * cos64 - x2 * sin64, x2 * cos64 + x1 * sin64], -1)
for chunk in (2048, 64, 1):
    outs = []
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        y = mx.fast.rope(mx.array(xb[:, :, a:b]).astype(mx.bfloat16), D, traditional=False, base=THETA, scale=1.0, offset=a)
        outs.append(np.array(y.astype(mx.float32)))
    yb = np.concatenate(outs, 2).astype(np.float64)
    err = np.abs(yb - truth)
    res[f"bf16_input_chunk{chunk}_vs_float64"] = {"max_abs_0_1024": float(err[:, :, :1024].max()),
                                                   "max_abs_15000_15301": float(err[:, :, 15000:15301].max()),
                                                   "rel_l2_15000_15301": float(np.linalg.norm(err[:, :, 15000:15301]) / np.linalg.norm(truth[:, :, 15000:15301]))}
    if chunk == 2048:
        yb2048 = yb
    else:
        res[f"bf16_input_chunk{chunk}_bitwise_eq_chunk2048"] = bool(np.array_equal(yb, yb2048))
# bf16 rounding of exact result (the floor any bf16 output has)
tb = np.array(mx.array(truth.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)).astype(np.float64)
res["bf16_rounding_floor_max_abs_15000_15301"] = float(np.abs(tb - truth)[:, :, 15000:15301].max())

json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=1)
