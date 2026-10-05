"""Float64 Kolibri-1 forward written from the spec (BUILD_SPEC items 4-14), for
truth on tiny checkpoints at long context. Independent of reference/kolibri_ref.py:
weights read through mlx (tests/tiny_checkpoint.load_checkpoint_f32), RoPE angles
in float64 from exact integer positions, attention over keys [0, b) with an
explicit per-(query, key) mask, all experts evaluated densely then gathered.

Returns logits float64 [T, V] and the per-layer routing selections (sets)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

K = "$KIT"
sys.path.insert(0, K + "/tests")


def load_weights(ckpt):
    from tiny_checkpoint import load_checkpoint_f32
    W = load_checkpoint_f32(ckpt)
    return {k: v.astype(np.float64) for k, v in W.items()}


def rms(x, w, eps):
    return w * x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def rope64(x, pos, theta):
    D = x.shape[-1]
    half = D // 2
    import os
    if os.environ.get("F64_FP32_ANGLES") == "1":
        # vLLM's angle: fp32 inv_freq = 1 / base^(arange(0, D, 2) / D), fp32 t * inv_freq (one fp32 rounding);
        # cos/sin then evaluated in float64 on that fp32 angle.
        inv32 = np.float32(1.0) / (np.float32(theta) ** (np.arange(0, D, 2, dtype=np.float32) / np.float32(D)))
        ang = (pos.astype(np.float32)[:, None] * inv32[None]).astype(np.float64)
    else:
        inv = theta ** (-np.arange(0, D, 2, dtype=np.float64) / D)
        ang = pos.astype(np.float64)[:, None] * inv[None]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def attend(q, k, v, window, block=64):
    """q [T, nh, D], k/v [T, nkv, D] float64; query i sees key j iff j <= i and
    (window is None or i - j < window)."""
    T, nh, D = q.shape
    nkv = k.shape[1]
    rep = nh // nkv
    qg = q.reshape(T, nkv, rep, D).transpose(1, 2, 0, 3)  # [nkv, rep, T, D]
    kg = k.transpose(1, 0, 2)  # [nkv, T, D]
    vg = v.transpose(1, 0, 2)
    out = np.empty((nkv, rep, T, D))
    scale = D ** -0.5
    for a in range(0, T, block):
        b = min(T, a + block)
        k0 = 0 if window is None else max(0, a - window - 64)  # generous; the mask decides
        qi = np.arange(a, b)[:, None]
        kj = np.arange(k0, b)[None, :]
        m = kj <= qi
        if window is not None:
            m &= (qi - kj) < window
        Tb, Tk = b - a, b - k0
        s = np.matmul(qg[:, :, a:b].reshape(nkv, rep * Tb, D),
                      kg[:, k0:b].transpose(0, 2, 1)).reshape(nkv, rep, Tb, Tk) * scale
        s = np.where(m, s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        out[:, :, a:b] = np.matmul(p.reshape(nkv, rep * Tb, Tk), vg[:, k0:b]).reshape(nkv, rep, Tb, D)
    return out.transpose(2, 0, 1, 3).reshape(T, nh, D)


def silu(x):
    return x / (1 + np.exp(-x))


def forward(ckpt, ids, W=None, log=print):
    cfg = json.loads((Path(ckpt) / "config.json").read_text())
    W = W or load_weights(ckpt)
    eps = cfg["rms_norm_eps"]
    nh, nkv, D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    E, k = cfg["num_experts"], cfg["num_experts_per_tok"]
    ids = np.asarray(ids)
    T = ids.size
    pos = np.arange(T)
    h = W["model.embed_tokens.weight"][ids]
    sels = []
    for i, lt in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}"
        x = rms(h, W[f"{p}.input_layernorm.weight"], eps)
        q = (x @ W[f"{p}.self_attn.q_proj.weight"].T).reshape(T, nh, D)
        kk = (x @ W[f"{p}.self_attn.k_proj.weight"].T).reshape(T, nkv, D)
        v = (x @ W[f"{p}.self_attn.v_proj.weight"].T).reshape(T, nkv, D)
        q = rms(q, W[f"{p}.self_attn.q_norm.weight"], eps)
        kk = rms(kk, W[f"{p}.self_attn.k_norm.weight"], eps)
        sliding = lt == "sliding_attention"
        if sliding:
            q = rope64(q, pos, cfg["rope_theta"])
            kk = rope64(kk, pos, cfg["rope_theta"])
        o = attend(q, kk, v, cfg["sliding_window"] if sliding else None)
        a = o.reshape(T, nh * D) @ W[f"{p}.self_attn.o_proj.weight"].T
        h = h + rms(a, W[f"{p}.post_attn_norm.weight"], eps)
        y = rms(h, W[f"{p}.post_attention_layernorm.weight"], eps)
        logits = y @ W[f"{p}.mlp.gate.weight"].T
        score = logits + W[f"{p}.moe.router.expert_bias"][None]
        sel = np.argsort(-score, axis=-1, kind="stable")[:, :k]
        wts = 1 / (1 + np.exp(-np.take_along_axis(logits, sel, -1)))
        routed = np.zeros_like(y)
        for e in range(E):
            rows, slot = np.nonzero(sel == e)
            if rows.size == 0:
                continue
            ep = f"{p}.mlp.experts.{e}"
            ye = (silu(y[rows] @ W[f"{ep}.gate_proj.weight"].T) * (y[rows] @ W[f"{ep}.up_proj.weight"].T)) @ W[f"{ep}.down_proj.weight"].T
            routed[rows] += wts[rows, slot][:, None] * ye
        sp = f"{p}.mlp.shared_experts"
        shared = (silu(y @ W[f"{sp}.gate_proj.weight"].T) * (y @ W[f"{sp}.up_proj.weight"].T)) @ W[f"{sp}.down_proj.weight"].T
        h = h + rms(routed + shared, W[f"{p}.post_ffn_norm.weight"], eps)
        srt = np.sort(score, axis=-1)[:, ::-1]
        sels.append({"sel": np.sort(sel, -1), "margin": srt[:, k - 1] - srt[:, k]})
        log(f"[f64] layer {i} {lt} done")
    hn = rms(h, W["model.norm.weight"], eps)
    head = W["model.embed_tokens.weight"] if cfg["tie_word_embeddings"] else W["lm_head.weight"]
    return hn @ head.T, sels
