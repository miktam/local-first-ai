"""Port (prefill 2048 and 64, forced to the same expert ids) vs the numpy
reference (same ids) vs a float64-angle RoPE reference, per 2048 block, 16k."""
from common_bh import *
import time
from reference import kolibri_ref as KR
ck = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
m, _, mod = common.load_port(Path(ck)); m.set_dtype(mx.float32); mx.eval(m.parameters())
for i, l in enumerate(m.layers): l._bh = i
ST = {"force": None, "p": 0, "rec": []}
def call(self, x, mask=None, cache=None):
    L = x.shape[1]; f = None
    if ST["force"] is not None:
        f = ST["force"][self._bh][ST["p"]:ST["p"] + L][None]
    r = self.branches(x, mask, cache, force_ids=f)
    ST["rec"].append(r[5]); return r[3]
mod.DecoderLayer.__call__ = call
rng = np.random.default_rng(5)
ids = rng.integers(0, 1008, size=T).astype(np.int32)
def run(sched, force=None):
    ST["force"] = force; cache = m.make_cache(); lps, sel = [], []; p = 0
    for L in sched:
        ST["rec"] = []; ST["p"] = p
        out = m(mx.array(ids[p:p + L])[None], cache=cache)[0].astype(mx.float32)
        lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
        s = mx.stack([r[0] for r in ST["rec"]]); mx.eval(lp, s)
        lps.append(np.array(lp)); sel.append(np.array(s)); p += L
    return np.concatenate(lps), np.concatenate(sel, axis=1)
p2048, sel = run([2048] * (T // 2048))
p64, _ = run([64] * (T // 64), mx.array(sel))
class Rope64(KR.KolibriReference):
    def rope(self, x, positions):
        D = x.shape[-1]; half = D // 2
        inv = 1.0 / (self.cfg.rope_theta ** (np.arange(0, D, 2, dtype=np.float64) / D))
        ang = np.asarray(positions, np.float64)[:, None] * inv[None]
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        x = x.astype(np.float64); x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(np.float32)
def ref_forced(cls):
    r = cls(str(Path(ck)))
    h = r.embed(ids.astype(np.int64))
    for i in range(r.cfg.num_hidden_layers):
        h = r.layer_forward(i, h, None, force_ids=sel[i].astype(np.int64))
    lg = r.lm_head(r.final_norm(h))
    lg = lg - lg.max(-1, keepdims=True)
    return (lg - np.log(np.exp(lg).sum(-1, keepdims=True))).astype(np.float32)
t = time.time(); ref = ref_forced(KR.KolibriReference); print(f"ref {time.time()-t:.0f}s", flush=True)
t = time.time(); r64 = ref_forced(Rope64); print(f"ref f64-rope {time.time()-t:.0f}s", flush=True)
np.savez_compressed(f"exp4_{ck}.npz", p2048=p2048, p64=p64, ref=ref, r64=r64)
def kl(a, b): return (np.exp(a) * (a - b)).sum(-1)
for an, a, bn, b in (("ref", ref, "p2048", p2048), ("ref", ref, "p64", p64), ("p2048", p2048, "p64", p64),
                     ("f64rope", r64, "ref", ref), ("f64rope", r64, "p2048", p2048)):
    k = kl(a, b)
    print(f"KL({an}||{bn}) per-2048-block mean " + " ".join(f"{k[x:x+2048].mean():.1e}" for x in range(0, T, 2048))
          + "  | max " + " ".join(f"{k[x:x+2048].max():.1e}" for x in range(0, T, 2048)))
