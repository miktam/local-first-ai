"""Full-length (16k) per-position comparison of port prefill step sizes and
token-by-token decode, with router-selection flips per layer."""
from common_bh import *
import time, json
ck = sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
m, _, mod = common.load_port(Path(ck)); m.set_dtype(mx.float32); mx.eval(m.parameters())
REC = []
orig = mod.DecoderLayer.__call__
def rec_call(self, x, mask=None, cache=None):
    r = self.branches(x, mask, cache)
    REC.append(r[5])
    return r[3]
mod.DecoderLayer.__call__ = rec_call
nL = len(m.layers)
rng = np.random.default_rng(5)
ids = rng.integers(0, 1008, size=T).astype(np.int32)
def run(sched):
    cache = m.make_cache(); lps, sel = [], []
    p = 0
    for L in sched:
        REC.clear()
        out = m(mx.array(ids[p:p + L])[None], cache=cache)[0].astype(mx.float32)
        lp = out - mx.logsumexp(out, axis=-1, keepdims=True)
        s = mx.stack([mx.sort(r[0], axis=-1) for r in REC])  # [layers, L, k]
        mx.eval(lp, s); lps.append(np.array(lp)); sel.append(np.array(s)); p += L
    return np.concatenate(lps), np.concatenate(sel, axis=1)
res = {}
for name, sched in [("p2048", [2048] * (T // 2048)), ("p64", [64] * (T // 64)), ("p256", [256] * (T // 256)),
                    ("p512", [512] * (T // 512)), ("p513", [513] * (T // 513) + ([T % 513] if T % 513 else [])),
                    ("p514", [514] * (T // 514) + ([T % 514] if T % 514 else [])), ("dec", [1] * T)]:
    t = time.time(); res[name] = run(sched); print(name, f"{time.time()-t:.1f}s", flush=True)
np.savez_compressed(f"exp2_{ck}.npz", ids=ids, **{f"{k}_lp": v[0] for k, v in res.items()}, **{f"{k}_sel": v[1] for k, v in res.items()})
a_lp, a_sel = res["p2048"]
for name, (lp, sel) in res.items():
    if name == "p2048": continue
    kl = (np.exp(a_lp) * (a_lp - lp)).sum(-1)
    flips = (sel != a_sel).any(-1)  # [layers, T]
    fpos = np.where(flips.any(0))[0]
    first = int(np.argmax(kl > 1e-6)) if (kl > 1e-6).any() else None
    print(f"{name} vs p2048: mean KL {kl.mean():.3e} max {kl.max():.3e} at {int(kl.argmax())}; "
          f"KL>1e-6 first at {first}, n {int((kl>1e-6).sum())}; positions with any router flip {fpos.size}, first {fpos[:5].tolist()}; "
          f"mean KL 15000-15300 {kl[15000:15301].mean():.3e}  0-1100 {kl[:1101].mean():.3e}")
