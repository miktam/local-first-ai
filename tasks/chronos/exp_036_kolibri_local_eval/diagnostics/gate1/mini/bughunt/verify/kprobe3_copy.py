"""kprobe3: Kolibri's PREFILL shapes in the G5 first wave (B=8 right-padded to 1099) against
the same rows prefilled alone (B=1, M = len), both against float64 of the same rounded
inputs. kprobe2 covered decode shapes only. Random weights, no model weights. Runs unchanged
on the mini (M4 Pro) and the mbp (M5 Max); pass a baseline JSON from the other chip as argv[1].

A probe is flagged when a valid row of the B=8 call is not bitwise equal to the B=1 call AND
rel_B8 > 1.2 x rel_B1 (or, for 8 identical rows, the rows are not bitwise equal to each other)."""
import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.switch_layers import SwitchGLU

rng = np.random.default_rng(36)
H, E, K, I, Hq, Hk, D = 2560, 384, 6, 512, 48, 4, 128
L = 1099
LENS = [36, 299, 699, 1099, 63, 519, 899, 149]
info = mx.device_info()
out = {"device": info.get("device_name"), "arch": info.get("architecture"), "mlx": mx.__version__,
       "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"), "lens": LENS}

def f64(a): return np.array(a.astype(mx.float32)).astype(np.float64)
def rel(a, ref): return float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-300))

def rows_cmp(y8, ys1, refs):
    """y8 [8, L, ...]; ys1[i] [1, n_i, ...]; refs[i] float64 [n_i, ...] (valid rows only)."""
    per = []
    for i, n in enumerate(LENS):
        a = f64(y8[i, :n]); b = f64(ys1[i][0]); r = refs[i]
        per.append({"row": i, "n": n, "bitwise": bool(np.array_equal(a, b)), "rel_B8": rel(a, r), "rel_B1": rel(b, r),
                    "ratio": rel(a, r) / max(rel(b, r), 1e-300), "max_abs_B8_B1": float(np.abs(a - b).max())})
    return {"rows": per, "any_not_bitwise": any(not p["bitwise"] for p in per),
            "max_ratio": max(p["ratio"] for p in per)}

def identical_rows(fn, x1):
    """Same input in all 8 rows: are the 8 outputs bitwise equal to each other and to B=1?"""
    x8 = mx.concatenate([x1] * 8)
    y8, y1 = fn(x8), fn(x1); mx.eval(y8, y1)
    a = f64(y8); b = f64(y1)
    return {"rows_bitwise_equal": bool(all(np.array_equal(a[0], a[i]) for i in range(8))),
            "row0_equals_B1": bool(np.array_equal(a[0], b[0])), "max_abs_row_spread": float(np.abs(a - a[:1]).max())}

def qweights(N, Kd, bits=8):
    w = mx.array((rng.standard_normal((N, Kd)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    deq = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=bits)).astype(np.float64)
    return wq, s, b, deq

res = {}
# 1. q8 projections (prefill), bf16 activations [8, 1099, K]
for name, (N, Kd) in {"q_proj": (Hq * D, H), "k_proj": (Hk * D, H), "o_proj": (H, Hq * D),
                      "shared_gate": (I, H), "shared_down": (H, I)}.items():
    wq, s, b, deq = qweights(N, Kd)
    fn = lambda z: mx.quantized_matmul(z, wq, s, b, transpose=True, group_size=64, bits=8)
    x = mx.array(rng.standard_normal((8, L, Kd)).astype(np.float32)).astype(mx.bfloat16)
    y8 = fn(x); ys1 = [fn(x[i:i + 1, :n]) for i, n in enumerate(LENS)]; mx.eval(y8, ys1)
    xf = f64(x)
    res[f"qmm8_{name}"] = rows_cmp(y8, ys1, [xf[i, :n] @ deq.T for i, n in enumerate(LENS)])
    res[f"qmm8_{name}"]["identical"] = identical_rows(fn, x[2:3, :699])
# 2. router fp32 (prefill)
W = mx.array((rng.standard_normal((E, H)) * 0.02).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
fn = lambda z: z.astype(mx.float32) @ W.T
x = mx.array(rng.standard_normal((8, L, H)).astype(np.float32)).astype(mx.bfloat16)
y8 = fn(x); ys1 = [fn(x[i:i + 1, :n]) for i, n in enumerate(LENS)]; mx.eval(y8, ys1)
xf = f64(x); Wf = f64(W)
res["router_fp32"] = rows_cmp(y8, ys1, [xf[i, :n] @ Wf.T for i, n in enumerate(LENS)])
res["router_fp32"]["identical"] = identical_rows(fn, x[2:3, :699])
# 3. SwitchGLU q8, sorted gather (prefill); pad positions all routed to the same 6 experts
sg = SwitchGLU(H, I, E); sg.set_dtype(mx.bfloat16); nn.quantize(sg, group_size=64, bits=8); mx.eval(sg.parameters())
sg32 = SwitchGLU(H, I, E); sg32.set_dtype(mx.bfloat16); nn.quantize(sg32, group_size=64, bits=8)
sg32.update(sg.parameters()); sg32.set_dtype(mx.float32); mx.eval(sg32.parameters())
x = mx.array(rng.standard_normal((8, L, H)).astype(np.float32)).astype(mx.bfloat16)
idx = np.stack([[rng.choice(E, K, replace=False) for _ in range(L)] for _ in range(8)]).astype(np.uint32)
for i, n in enumerate(LENS):
    idx[i, n:] = np.array([3, 17, 42, 101, 200, 333], np.uint32)
idx = mx.array(idx)
y8 = sg(x, idx); mx.eval(y8)
ys1, refs = [], []
for i, n in enumerate(LENS):
    y1 = sg(x[i:i + 1, :n], idx[i:i + 1, :n]); r = sg32(x[i:i + 1, :n].astype(mx.float32), idx[i:i + 1, :n]); mx.eval(y1, r)
    ys1.append(y1); refs.append(f64(r)[0])
res["switchglu_q8_sorted"] = rows_cmp(y8, ys1, refs)
res["switchglu_q8_sorted"]["identical"] = identical_rows(lambda z: sg(z, mx.concatenate([idx[2:3, :699]] * z.shape[0])), x[2:3, :699])
# 4. SDPA prefill bf16 48q/4kv, boolean mask [8,1,L,L] (causal, and causal+window 513) vs alone
for win in (None, 513):
    q = mx.array(rng.standard_normal((8, Hq, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((8, Hk, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((8, Hk, L, D)).astype(np.float32)).astype(mx.bfloat16)
    m8 = create_causal_mask(L, 0, window_size=win, left_padding=mx.zeros((8,), mx.int32))
    y8 = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5, mask=m8); mx.eval(y8)
    ys1, refs = [], []
    for i, n in enumerate(LENS):
        m1 = create_causal_mask(n, 0, window_size=win, left_padding=mx.zeros((1,), mx.int32))
        y1 = mx.fast.scaled_dot_product_attention(q[i:i + 1, :, :n], k[i:i + 1, :, :n], v[i:i + 1, :, :n], scale=D ** -0.5, mask=m1)
        mx.eval(y1)
        qq, kk, vv = f64(q[i, :, :n]), np.repeat(f64(k[i, :, :n]), Hq // Hk, 0), np.repeat(f64(v[i, :, :n]), Hq // Hk, 0)
        sc = (qq @ kk.transpose(0, 2, 1)) * D ** -0.5
        mm = np.array(m1[0, 0]); sc = np.where(mm[None], sc, -np.inf)
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        ys1.append(y1.transpose(0, 2, 1, 3)); refs.append((p @ vv).transpose(1, 0, 2))
    res[f"sdpa_prefill_bf16_win{win}"] = rows_cmp(y8.transpose(0, 2, 1, 3), ys1, refs)
# 5. RoPE, array offset [8] (BatchRotatingKVCache.offset) vs int offset, bf16, prefill and decode
rope = nn.RoPE(D, traditional=False, base=10000.0)
for Lq, offs in ((L, [0] * 8), (1, [n for n in LENS])):
    x = mx.array(rng.standard_normal((8, Hk, Lq, D)).astype(np.float32)).astype(mx.bfloat16)
    y8 = rope(x, offset=mx.array(offs, dtype=mx.int32)); mx.eval(y8)
    eq = []
    for i in range(8):
        y1 = rope(x[i:i + 1], offset=int(offs[i])); mx.eval(y1)
        eq.append(bool(np.array_equal(f64(y8[i]), f64(y1[0]))))
    res[f"rope_bf16_array_offset_L{Lq}"] = {"rows_bitwise_vs_int_offset": eq, "any_not_bitwise": not all(eq), "max_ratio": 1.0}

# 6. Neighbour invariance: the valid rows of a B=8 right-padded call must not depend on the
#    CONTENT of the pad rows (in exact arithmetic they cannot). Pad rows: N(0,1) vs x1000
#    (massive activations of a run of ~1000 identical pad tokens). Valid rows compared bitwise,
#    and against float64.
def neighbour(fn, x, ref_fn, name, pad_axis=1):
    xs = np.array(x.astype(mx.float32))
    big = xs.copy()
    for i, n in enumerate(LENS):
        if pad_axis == 1:
            big[i, n:] *= 1000.0
        else:
            big[i, :, n:] *= 1000.0
    xb = mx.array(big).astype(x.dtype)
    y_a, y_b = fn(x), fn(xb); mx.eval(y_a, y_b)
    per = []
    for i, n in enumerate(LENS):
        a = f64(y_a[i, :n]) if pad_axis == 1 else f64(y_a[i, :, :n])
        b = f64(y_b[i, :n]) if pad_axis == 1 else f64(y_b[i, :, :n])
        r = ref_fn(i, n)
        per.append({"row": i, "n": n, "bitwise": bool(np.array_equal(a, b)), "rel_small_pads": rel(a, r), "rel_big_pads": rel(b, r),
                    "ratio": rel(b, r) / max(rel(a, r), 1e-300)})
    res[f"neighbour_{name}"] = {"rows": per, "any_not_bitwise": any(not p["bitwise"] for p in per),
                                "max_ratio": max(p["ratio"] for p in per)}
wq, s, b, deq = qweights(Hq * D, H)
x = mx.array(rng.standard_normal((8, L, H)).astype(np.float32)).astype(mx.bfloat16)
xf = f64(x)
neighbour(lambda z: mx.quantized_matmul(z, wq, s, b, transpose=True, group_size=64, bits=8), x,
          lambda i, n: xf[i, :n] @ deq.T, "qmm8_q_proj")
neighbour(lambda z: z @ mx.array(deq.astype(np.float32)).astype(mx.bfloat16).T, x,
          lambda i, n: xf[i, :n] @ f64(mx.array(deq.astype(np.float32)).astype(mx.bfloat16)).T, "bf16mm_q_proj")
refs_sg = {}
def ref_sg(i, n):
    r = sg32(x[i:i + 1, :n].astype(mx.float32), idx[i:i + 1, :n]); mx.eval(r); return f64(r)[0]
neighbour(lambda z: sg(z, idx), x, ref_sg, "switchglu_q8_sorted")
# attention: pad KEYS/VALUES/QUERIES x1000 (they sit in the causal future of every valid query)
q = mx.array(rng.standard_normal((8, Hq, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
k = mx.array(rng.standard_normal((8, Hk, L, D)).astype(np.float32) * 2).astype(mx.bfloat16)
v = mx.array(rng.standard_normal((8, Hk, L, D)).astype(np.float32)).astype(mx.bfloat16)
m8 = create_causal_mask(L, 0, window_size=513, left_padding=mx.zeros((8,), mx.int32))
def attn_ref(i, n):
    m1 = create_causal_mask(n, 0, window_size=513, left_padding=mx.zeros((1,), mx.int32))
    qq, kk, vv = f64(q[i, :, :n]), np.repeat(f64(k[i, :, :n]), Hq // Hk, 0), np.repeat(f64(v[i, :, :n]), Hq // Hk, 0)
    sc = (qq @ kk.transpose(0, 2, 1)) * D ** -0.5
    sc = np.where(np.array(m1[0, 0])[None], sc, -np.inf)
    p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    return p @ vv
def kv_big(arr):
    a_ = np.array(arr.astype(mx.float32)).copy()
    for i, n in enumerate(LENS):
        a_[i, :, n:] *= 1000.0
    return mx.array(a_).astype(mx.bfloat16)
neighbour(lambda z: mx.fast.scaled_dot_product_attention(z, k if z is q else kv_big(k), v if z is q else kv_big(v),
                                                         scale=D ** -0.5, mask=m8), q, attn_ref, "sdpa_prefill_win513", pad_axis=2)

out["probes"] = res
out["flagged"] = sorted(k for k, r in res.items()
                        if (r.get("any_not_bitwise") and r.get("max_ratio", 1.0) > 1.2)
                        or (k.startswith("neighbour_") and r.get("any_not_bitwise"))
                        or (r.get("identical") and not r["identical"]["rows_bitwise_equal"]))
if len(sys.argv) > 1:
    base = json.load(open(sys.argv[1]))["probes"]
    chip = []
    for k_, r in res.items():
        b = base.get(k_, {})
        if r.get("max_ratio", 1.0) > 1.2 * max(1.0, b.get("max_ratio", 1.0)) and r.get("any_not_bitwise"):
            chip.append(k_)
        if r.get("identical") and not r["identical"]["rows_bitwise_equal"] and b.get("identical", {}).get("rows_bitwise_equal", True):
            chip.append(k_ + ".identical_rows")
    out["baseline"] = {"file": sys.argv[1], "chip_specific": chip}
print(json.dumps(out, indent=1))
