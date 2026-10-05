"""Tiny head_dim-128, window-513 Kolibri, fp32, 16k sequence:
port prefill 2048 / prefill 64 / decode at 15000..15300 vs the numpy reference,
with (a) MLX nn.RoPE, (b) exact cos/sin (float64 -> fp32) in the port,
(c) exact cos/sin in the reference too.

usage: r2_tiny16k.py <out.json> [random|realtok] [n_layers]
"""
import json
import sys
import time
from pathlib import Path

KIT = "$KIT"
sys.path.insert(0, KIT)
sys.path.insert(0, KIT + "/tests")
from tools.precision import ensure_exact_fp32

print(ensure_exact_fp32(), flush=True)
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from gate.checks import g5_generation as g5
from reference import kolibri_ref as KR
from tiny_checkpoint import write_tiny_checkpoint, random_ids, _layer_types
from port_harness import load_port

OUT = Path(sys.argv[1])
SEQ = sys.argv[2] if len(sys.argv) > 2 else "random"
NL = int(sys.argv[3]) if len(sys.argv) > 3 else 10
HERE = Path(__file__).resolve().parent
LO, HI = 15000, 15300
THETA = 10000.0

ck = HERE / f"ckpt_h128_w513_L{NL}"
if not (ck / "config.json").exists():
    write_tiny_checkpoint(ck, seed=7, preset="w513", overrides={
        "head_dim": 128, "num_attention_heads": 12, "num_key_value_heads": 4,
        "num_hidden_layers": NL, "layer_types": _layer_types(NL, 5),
        "max_position_embeddings": 32768})

if SEQ == "random":
    ids = np.array(random_ids(HI + 1, seed=3), dtype=np.int32)
else:
    ids = np.load(HERE / "realtok_ids.npy")[: HI + 1].astype(np.int32)
assert ids.size == HI + 1


def lsm(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):  # KL(p||q) rows, p,q log-probs
    return (np.exp(p) * (p - q)).sum(-1)


def lead(lp):
    s = np.sort(lp, -1)
    return s[:, -1] - s[:, -2]


def pair(p, q):
    k = kl(p, q)
    ch = p.argmax(-1) != q.argmax(-1)
    return {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "argmax_pos": int(LO + k.argmax()),
            "top1_changes": int(ch.sum()), "top1_changes_lead_ge_2": int((ch & (lead(p) >= 2.0)).sum())}


# ------------------------------- exact RoPE -----------------------------------
D = 128
inv64 = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)


def exact_cs(positions):
    a = np.asarray(positions, np.float64)[:, None] * inv64[None, :]
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


class ExactRoPE(nn.Module):
    """NeoX RoPE with cos/sin from float64 angles, rounded once to fp32."""

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x, offset=0):
        L = x.shape[-2]
        c, s = exact_cs(np.arange(int(offset), int(offset) + L))
        c, s = mx.array(c), mx.array(s)
        h = self.dims // 2
        x1, x2 = x[..., :h], x[..., h:]
        return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)


def port_runs(model, tag):
    t = time.time()
    a = lsm(g5.prefill_rows(model, ids, LO, HI, 2048))
    b = lsm(g5.prefill_rows(model, ids, LO, HI, 64))
    d = lsm(g5.decode_rows(model, ids, LO, HI, 2048))
    print(f"[{tag}] port runs {time.time() - t:.0f}s", flush=True)
    return {"prefill2048": a, "prefill64": b, "decode": d}


res = {"seq": SEQ, "n_layers": NL, "ckpt": str(ck), "range": [LO, HI]}

model = load_port(ck, float32=True)
P = {"mlx_rope": port_runs(model, "mlx_rope")}
for layer in model.layers:
    if layer.self_attn.rope is not None:
        layer.self_attn.rope = ExactRoPE(D)
P["exact_rope"] = port_runs(model, "exact_rope")
del model

t = time.time()
ref = KR.KolibriReference(str(ck))
R = {"ref_fp32": lsm(ref.forward(ids)[LO:HI + 1])}
print(f"ref {time.time() - t:.0f}s", flush=True)


class RefExact(KR.KolibriReference):
    def rope(self, x, positions):
        c, s = exact_cs(positions)
        c, s = c[:, None, :], s[:, None, :]
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return self._store(np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(np.float32))


t = time.time()
R["ref_exact_rope"] = lsm(RefExact(str(ck)).forward(ids)[LO:HI + 1])
print(f"ref exact {time.time() - t:.0f}s", flush=True)

out = {}
for rk, rv in P.items():
    out[f"port[{rk}] prefill2048_vs_prefill64"] = pair(rv["prefill2048"], rv["prefill64"])
    out[f"port[{rk}] prefill2048_vs_decode"] = pair(rv["prefill2048"], rv["decode"])
    out[f"port[{rk}] prefill64_vs_decode"] = pair(rv["prefill64"], rv["decode"])
    for refk, refv in R.items():
        for q in ("prefill2048", "prefill64", "decode"):
            out[f"{refk}_vs_port[{rk}].{q}"] = pair(refv, rv[q])
out["ref_fp32_vs_ref_exact_rope"] = pair(R["ref_fp32"], R["ref_exact_rope"])
out["port[mlx_rope].decode_vs_port[exact_rope].decode"] = pair(P["mlx_rope"]["decode"], P["exact_rope"]["decode"])
res["pairs"] = out
OUT.write_text(json.dumps(res, indent=1))
for k, v in out.items():
    print(f"{k:60s} mean {v['mean_kl']:.3e} max {v['max_kl']:.3e} @{v['argmax_pos']} top1ch {v['top1_changes']} (lead>=2: {v['top1_changes_lead_ge_2']})")
