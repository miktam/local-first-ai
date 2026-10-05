"""RoPE precision at large positions: reference rope_neox (numpy fp32 angles),
port nn.RoPE / mx.fast.rope (fp32, prefill chunks and single-token decode),
against float64 truth. Kolibri head_dim 128, theta 1e4."""
import os, sys
K = "$KIT"
sys.path.insert(0, K)
from tools.precision import ensure_exact_fp32
print(ensure_exact_fp32())
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from reference.kolibri_ref import rope_neox

D, THETA = 128, 10000.0
rng = np.random.default_rng(0)


def rope_f64(x, pos):
    x = np.asarray(x, np.float64)
    half = D // 2
    inv = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


rope = nn.RoPE(D, traditional=False, base=THETA)


def port_rope(x, offset):
    # x [T, H, D] -> port layout [B, H, T, D]
    xm = mx.array(x.transpose(1, 0, 2))[None]
    y = rope(xm, offset=offset)
    return np.array(y[0]).transpose(1, 0, 2)


def err(a, b):
    return float(np.abs(a - b).max())


print(f"{'start':>8} {'T':>5} | {'ref_np32-f64':>12} {'port_chunk-f64':>14} {'port_decode-f64':>15} {'port-ref':>10}")
for start in [0, 512, 2048, 3911, 8192, 15000, 16000, 65536, 131072, 262000]:
    T = 64
    x = rng.standard_normal((T, 4, D)).astype(np.float32)
    pos = np.arange(start, start + T)
    t = rope_f64(x, pos)
    r = rope_neox(x, pos, THETA)
    pc = port_rope(x, start)
    pd = np.concatenate([port_rope(x[i:i + 1], start + i) for i in range(T)], 0)
    print(f"{start:>8} {T:>5} | {err(r, t):12.3e} {err(pc, t):14.3e} {err(pd, t):15.3e} {err(pc, r):10.3e}  chunk==decode bitwise: {np.array_equal(pc, pd)}")

# Per frequency index: where does the error live at 15000-15300?
x = rng.standard_normal((300, 1, D)).astype(np.float32)
pos = np.arange(15000, 15300)
t = rope_f64(x, pos)
r = rope_neox(x, pos, THETA)
pc = port_rope(x, 15000)
e_r = np.abs(r - t).max(axis=(0, 1))
e_p = np.abs(pc - t).max(axis=(0, 1))
print("per-dim max err (first 8 dims, i.e. highest frequencies): ref", np.round(e_r[:8], 6), "port", np.round(e_p[:8], 6))
print("per-dim max err (dims 56-64, lowest freq): ref", np.round(e_r[56:64], 7), "port", np.round(e_p[56:64], 7))
# Equivalent position shift: the angle error divided by the frequency, for the highest frequency (1 rad/token)
print("worst equivalent position error (tokens) at 15000-15300, freq 0: ref",
      float(e_r[0]), "port", float(e_p[0]))
