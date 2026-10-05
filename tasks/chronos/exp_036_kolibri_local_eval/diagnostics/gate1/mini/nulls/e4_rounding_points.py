"""E4: size of the two rounding points the skeptic says make 'vs emu' differ from 'vs vendor':
(a) fused_add_rms_norm: variance of the bf16-rounded sum (port; vLLM CUDA kernel if it rounds
first) vs of the unrounded fp32 sum (vLLM 0.29 IR op, ir_layernorm.py l.53-58; the reference's
emulation); (b) FlashAttention rounds P to bf16 before P.V, the emulation does not.
Each is compared with one bf16 rounding of the output, which every path already pays."""
import numpy as np, json
rng = np.random.default_rng(5)
def bf(x):
    x = np.asarray(x, np.float32).copy(); u = x.view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> 16) & 1); u &= np.uint32(0xFFFF0000); return x
def rel(a, t): return float(np.linalg.norm(a - t) / np.linalg.norm(t))
H, N = 2560, 2000
h = rng.standard_normal((N, H)).astype(np.float32) * 2.0
h[:, :4] *= 200.0                                   # massive-activation channels
h = bf(h); r = bf(rng.standard_normal((N, H)).astype(np.float32) * 0.5)
w = bf(1.0 + 0.1 * rng.standard_normal(H).astype(np.float32)); eps = 1e-6
s64 = h.astype(np.float64) + r
truth = s64 / np.sqrt((s64 ** 2).mean(-1, keepdims=True) + eps) * w
s_r = bf(h + r)                                      # port: bf16 sum, then norm in fp32, one rounding
port = bf(s_r / np.sqrt((s_r.astype(np.float64) ** 2).mean(-1, keepdims=True) + eps) * w)
s32 = (h + r).astype(np.float32)                     # IR: unrounded sum, round normed, *w, round
ir = bf(bf(s32 / np.sqrt((s32 ** 2).mean(-1, keepdims=True) + eps)) * w)
one_round = bf(truth.astype(np.float32))
res = {"norm": {"rel_one_rounding": rel(one_round, truth), "rel_port_order": rel(port, truth), "rel_ir_order": rel(ir, truth),
                "rel_port_vs_ir": rel(port, ir)}}
# attention: 48q/4kv, D=128, T keys, with a sink at key 0
D, T, Hq = 128, 1100, 48
q = bf(rng.standard_normal((Hq, D)).astype(np.float32)); k = bf(rng.standard_normal((T, D)).astype(np.float32))
k[0] *= 4.0; v = bf(rng.standard_normal((T, D)).astype(np.float32))
s = (q.astype(np.float64) @ k.T.astype(np.float64)) / np.sqrt(D)
p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
truth = p @ v
emu = bf((p.astype(np.float32) @ v).astype(np.float32))                       # emulation: P fp32, out rounded
fa = bf((bf(p.astype(np.float32)) @ v).astype(np.float32))                     # FA: P rounded to bf16, out rounded
res["attention"] = {"rel_emu_P_fp32": rel(emu, truth), "rel_fa_P_bf16": rel(fa, truth), "rel_one_rounding": rel(bf(truth.astype(np.float32)), truth),
                    "rel_fa_vs_emu": rel(fa, emu)}
print(json.dumps(res, indent=1))
