"""Forced routing: every schedule uses the expert ids p2048 selected, so the
discrete router cannot amplify rounding. Any remaining growth of the gap with
position would be a cache / mask / RoPE defect."""
from common_bh import *
import time
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
ref_lp, ref_sel = run([2048] * (T // 2048))
force = mx.array(ref_sel)  # [layers, T, k]
def kl(a, b): return (np.exp(a) * (a - b)).sum(-1)
lo = 15000
for name, sched in [("p2048 forced (self)", [2048] * (T // 2048)), ("p64", [64] * (T // 64)), ("p256", [256] * (T // 256)),
                    ("p512", [512] * (T // 512)), ("p513", [513] * (T // 513) + [T % 513]), ("p514", [514] * (T // 514) + [T % 514]),
                    ("2048 to 15000 then decode", [2048] * 7 + [lo - 7 * 2048] + [1] * (T - lo)),
                    ("64 to 15000 then decode", [64] * (lo // 64) + [lo % 64] + [1] * (T - lo))]:
    sched = [s for s in sched if s]
    t = time.time(); lp, sel = run(sched, force)
    k = kl(ref_lp, lp); d = np.abs(ref_lp - lp).max(-1)
    blocks = [f"{k[a:a+2048].max():.1e}" for a in range(0, T, 2048)]
    print(f"{name:28s} {time.time()-t:5.1f}s  KL mean {k.mean():.2e} max {k.max():.2e} at {int(k.argmax())} | per-2048-block max KL {blocks} | 15000-15300 mean {k[15000:15301].mean():.2e} | max|dlp| {d.max():.2e}", flush=True)
