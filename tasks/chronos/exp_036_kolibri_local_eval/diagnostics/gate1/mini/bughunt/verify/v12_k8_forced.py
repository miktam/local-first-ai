"""As v3 (port-internal part) on a K8-quantised tiny model with fp32 activations on the dequantised weights
(the bug test's arm: quantize 8-bit incl. embedding and head, set_dtype(fp32), diaglib.fp32_on_dequantised_weights)."""
from common_v import *
import time, importlib.util
from pathlib import Path
from gate import common
V = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("diaglib", "$KIT/diagnostics/gate1/diaglib.py")
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
ck = V / sys.argv[1]; T = 16384; LO, HI = 15000, 15300
ids = np.random.default_rng(5).integers(0, 1008, size=T).astype(np.int32)
pf = V / "kit_base" / "port" / "kolibri1.py"; mod = common.port_module(None, pf)
m, _, _ = common.load_port(ck, port_file=pf, module=mod)
nn.quantize(m, group_size=64, bits=8, class_predicate=mod.make_quant_predicate(quantize_embeddings=True, quantize_lm_head=True))
m.set_dtype(mx.float32); D.fp32_on_dequantised_weights(m); mx.eval(m.parameters())
print("embedding quantised:", hasattr(m.model.embed_tokens, "inner"), "q_proj quantised:", hasattr(m.layers[0].self_attn.q_proj, "scales"))
for i, l in enumerate(m.layers): l._v = i
ST = {"force": None, "p": 0, "rec": []}
def call(self, x, mask=None, cache=None):
    L = x.shape[1]; f = None
    if ST["force"] is not None: f = ST["force"][self._v][ST["p"]:ST["p"] + L][None]
    r = self.branches(x, mask, cache, force_ids=f); ST["rec"].append(r[5]); return r[3]
mod.DecoderLayer.__call__ = call
def run(sched, force=None):
    ST["force"] = force; cache = m.make_cache(); lps, sel = [], []; p = 0
    for L in sched:
        ST["rec"] = []; ST["p"] = p
        out = m(mx.array(ids[p:p + L])[None], cache=cache)[0].astype(mx.float32)
        lp = out - mx.logsumexp(out, axis=-1, keepdims=True); s = mx.stack([mx.sort(r[0], axis=-1) for r in ST["rec"]]); mx.eval(lp, s)
        lps.append(np.array(lp)); sel.append(np.array(s)); p += L
    return np.concatenate(lps), np.concatenate(sel, axis=1)
fixed = lambda s, n=T: [min(s, n - a) for a in range(0, n, s)]
a, sel = run(fixed(2048)); force = mx.array(sel)
b_free, _ = run(fixed(64)); b, _ = run(fixed(64), force); d, _ = run(fixed(2048, LO) + [1] * (HI + 1 - LO), force)
def kl(x, y): return (np.exp(x.astype(np.float64)) * (x.astype(np.float64) - y)).sum(-1)
for nm, y in (("p64 forced", b), ("p64 free", b_free)):
    k = kl(a, y)
    print(f"K8 fp32-deq: KL(p2048||{nm}) per-2048 mean " + " ".join(f"{k[x:x+2048].mean():.1e}" for x in range(0, T, 2048)) + f" | max {k.max():.1e} | [15000-15300] mean {k[LO:HI+1].mean():.1e}")
k = kl(a[LO:HI + 1], d[LO:HI + 1]); print(f"K8 fp32-deq: KL(p2048||decode forced) [15000-15300] mean {k.mean():.1e} max {k.max():.1e}")
