"""Forced-routing comparison at 16k: every path uses the expert ids the base port's prefill-2048 run selected,
so the discrete router cannot amplify rounding. Any remaining position-growing gap is a cache / mask / RoPE /
attention defect or a systematic numeric difference. Paths: port (kit) and port (vLLM-angle RoPE, scratch),
prefill 2048 / prefill 64 / prefill-2048-then-decode from 15000; reference (vLLM fp32 angle) and reference with
float64 RoPE angles."""
from common_v import *
import time, json
from pathlib import Path
from gate import common
from reference import kolibri_ref as KR
V = Path(__file__).resolve().parent
ck = V / sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
LO, HI = 15000, min(T - 1, 15300)
rng = np.random.default_rng(5)
ids = rng.integers(0, 1008, size=T).astype(np.int32)

def load(variant):
    pf = V / ("kit_base" if variant == "base" else "kit_rope") / "port" / "kolibri1.py"
    mod = common.port_module(None, pf)
    m, _, _ = common.load_port(ck, port_file=pf, module=mod)
    m.set_dtype(mx.float32); mx.eval(m.parameters())
    for i, l in enumerate(m.layers): l._v = i
    ST = {"force": None, "p": 0, "rec": []}
    def call(self, x, mask=None, cache=None):
        L = x.shape[1]; f = None
        if ST["force"] is not None:
            f = ST["force"][self._v][ST["p"]:ST["p"] + L][None]
        r = self.branches(x, mask, cache, force_ids=f)
        ST["rec"].append(r[5]); return r[3]
    mod.DecoderLayer.__call__ = call
    return m, ST

def run(m, ST, sched, force=None):
    ST["force"] = force; cache = m.make_cache(); lps, sel = [], []; p = 0
    for L in sched:
        ST["rec"] = []; ST["p"] = p
        out = m(mx.array(ids[p:p + L])[None], cache=cache)[0].astype(mx.float32)
        lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
        s = mx.stack([mx.sort(r[0], axis=-1) for r in ST["rec"]]); mx.eval(lp, s)
        lps.append(np.array(lp)); sel.append(np.array(s)); p += L
    return np.concatenate(lps), np.concatenate(sel, axis=1)

def sched_fixed(step, n=T): return [min(step, n - a) for a in range(0, n, step)]
def sched_dec(lo=LO, hi=HI):  # the gate's decode_rows: prefill ids[:lo] in 2048 chunks, then one token at a time
    return sched_fixed(2048, lo) + [1] * (hi + 1 - lo)

res = {}
m, ST = load("base")
t = time.time(); res["base_p2048_free"], sel = run(m, ST, sched_fixed(2048)); print(f"base p2048 free {time.time()-t:.0f}s", flush=True)
force = mx.array(sel)
for name, sc in (("p2048", sched_fixed(2048)), ("p64", sched_fixed(64)), ("dec", sched_dec())):
    t = time.time(); res[f"base_{name}"], s2 = run(m, ST, sc, force); print(f"base {name} forced {time.time()-t:.0f}s", flush=True)
m, ST = load("rope")
for name, sc in (("p2048", sched_fixed(2048)), ("p64", sched_fixed(64)), ("dec", sched_dec())):
    t = time.time(); res[f"rope_{name}"], _ = run(m, ST, sc, force); print(f"rope {name} forced {time.time()-t:.0f}s", flush=True)
del m

class RopeF64(KR.KolibriReference):
    def rope(self, x, positions):
        D = x.shape[-1]; half = D // 2
        inv = 1.0 / (self.cfg.rope_theta ** (np.arange(0, D, 2, dtype=np.float64) / D))
        ang = np.asarray(positions, np.float64)[:, None] * inv[None]
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        x = x.astype(np.float64); x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(np.float32)
def ref_forced(cls):
    r = cls(str(ck))
    h = r.embed(ids.astype(np.int64))
    for i in range(r.cfg.num_hidden_layers):
        h = r.layer_forward(i, h, None, force_ids=sel[i].astype(np.int64))
    lg = r.lm_head(r.final_norm(h)).astype(np.float64)
    lg -= lg.max(-1, keepdims=True)
    return (lg - np.log(np.exp(lg).sum(-1, keepdims=True))).astype(np.float32)
t = time.time(); res["ref"] = ref_forced(KR.KolibriReference); print(f"ref {time.time()-t:.0f}s", flush=True)
t = time.time(); res["ref_f64rope"] = ref_forced(RopeF64); print(f"ref f64rope {time.time()-t:.0f}s", flush=True)
np.savez_compressed(V / f"v3_{ck.name}_{T}.npz", ids=ids, **res)

def kl(a, b): return (np.exp(a.astype(np.float64)) * (a.astype(np.float64) - b)).sum(-1)
def seg(x, name):
    if x.shape[0] == T: return x
    return None
out = {}
blocks = [(a, min(T, a + 2048)) for a in range(0, T, 2048)]
pairs = [("base_p2048", "base_p64"), ("base_p2048", "base_dec"), ("base_p64", "base_dec"),
         ("rope_p2048", "rope_p64"), ("rope_p2048", "rope_dec"),
         ("ref", "base_p2048"), ("ref", "base_p64"), ("ref", "base_dec"),
         ("ref", "rope_p2048"), ("ref", "rope_p64"), ("ref", "rope_dec"),
         ("ref_f64rope", "ref"), ("ref_f64rope", "base_p2048"), ("ref_f64rope", "rope_p2048")]
for a, b in pairs:
    A, B = res[a], res[b]
    # decode rows cover LO..HI only
    if A.shape[0] != T or B.shape[0] != T:
        def rows(X):
            return X[LO:HI + 1] if X.shape[0] == T else X[-(HI + 1 - LO):]
        k = kl(rows(A), rows(B))
        out[f"{a}|{b}"] = {"range": [LO, HI], "mean": float(k.mean()), "max": float(k.max())}
        print(f"KL({a}||{b}) [{LO}-{HI}] mean {k.mean():.2e} max {k.max():.2e}")
        continue
    k = kl(A, B)
    out[f"{a}|{b}"] = {"block_mean": [float(k[x:y].mean()) for x, y in blocks], "block_max": [float(k[x:y].max()) for x, y in blocks],
                       "range_mean": float(k[LO:HI + 1].mean()), "range_max": float(k[LO:HI + 1].max())}
    print(f"KL({a}||{b}) per-2048 mean " + " ".join(f"{v:.1e}" for v in out[f'{a}|{b}']['block_mean'])
          + " | max " + " ".join(f"{v:.1e}" for v in out[f'{a}|{b}']['block_max']) + f" | [{LO}-{HI}] mean {k[LO:HI+1].mean():.1e}")
json.dump(out, open(V / f"v3_{ck.name}_{T}.json", "w"), indent=1)
