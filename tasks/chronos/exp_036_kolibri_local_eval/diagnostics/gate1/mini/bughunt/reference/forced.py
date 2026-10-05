"""Same as longctx_truth.py but with the routing of every implementation forced
to the float64 truth's selection: removes the near-tie routing channel, so what
remains is attention / RoPE / mask / cache / chunking arithmetic at large
positions."""
import json, sys, time
from pathlib import Path
K = "$KIT"
sys.path.insert(0, K)
sys.path.insert(0, K + "/tests")
from tools.precision import ensure_exact_fp32
print(ensure_exact_fp32())
import numpy as np
import mlx.core as mx

HERE = Path(__file__).resolve().parent
CK = HERE / (sys.argv[1] if len(sys.argv) > 1 else "ckpt_k16_L10")
T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
OUT = HERE / f"out_{CK.name}_{T}"
T9 = np.array(json.load(open(HERE.parent / "work/gate_texts/T9.ids.json"))["ids"])[:T]
ids = np.where(T9 == 263, 263, T9 % 1008).astype(np.int64)
docs = [int(p) + 1 for p in np.flatnonzero(T9 == 263) if p + 1 < T]
f64 = np.load(OUT / "f64.npy")
fsel = np.load(OUT / "f64_sel.npy")  # [L, T, k] sorted


def lsm(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    lp, lq = lsm(p), lsm(q)
    return (np.exp(lp) * (lp - lq)).sum(-1)


def cached(name, fn):
    p = OUT / f"{name}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    x = fn()
    np.save(p, x)
    print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    return x


# ---- reference, forced --------------------------------------------------------------
from reference import kolibri_ref


class ForcedRef(kolibri_ref.KolibriReference):
    def route(self, logits, bias, k):
        i = len(self.last_routing)  # layers done so far (appended after each layer)
        return kolibri_ref.forced_route_ref(logits, fsel[i], self.cfg.norm_topk_prob)


def ref_forced(chunk):
    r = ForcedRef(str(CK), attn_chunk=chunk)
    r._log = lambda m: None
    return r.forward(ids)


rf_def = cached("forced_ref_default", lambda: ref_forced(None))
rf_64 = cached("forced_ref_chunk64", lambda: ref_forced(64))

# ---- port, forced -------------------------------------------------------------------
from port_harness import load_port, port_namespace, ForcedRouting
model = load_port(CK, float32=True)
ns = port_namespace(model)
forced = ForcedRouting([fsel[i] for i in range(fsel.shape[0])])
ns["route"] = forced.route


def port_prefill(step, lo=0, hi=T - 1):
    cache = model.make_cache()
    rows = []
    x = ids[:hi + 1].astype(np.int32)
    for a in range(0, x.size, step):
        b = min(x.size, a + step)
        forced.start, forced.layer = a, 0
        out = model(mx.array(x[a:b])[None], cache=cache)[0].astype(mx.float32)
        mx.eval(out)
        if b > lo:
            rows.append(np.array(out)[max(lo, a) - a:])
    return np.concatenate(rows, 0)


def port_decode(lo, hi, step=2048):
    cache = model.make_cache()
    x = ids.astype(np.int32)
    for a in range(0, lo, step):
        forced.start, forced.layer = a, 0
        mx.eval(model(mx.array(x[a:min(lo, a + step)])[None], cache=cache))
    rows = []
    for p in range(lo, hi + 1):
        forced.start, forced.layer = p, 0
        out = model(mx.array(x[p:p + 1])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out)
        rows.append(np.array(out))
    return np.stack(rows)


pf2048 = cached("forced_port_prefill2048", lambda: port_prefill(2048))
pf64 = cached("forced_port_prefill64", lambda: port_prefill(64))
pf7 = cached("forced_port_prefill7_tail", lambda: port_prefill(7, 15000, 15300))  # odd step, tail rows only
LO, HI = 15000, min(15300, T - 1)
pdec = cached("forced_port_decode", lambda: port_decode(LO, HI))

buckets = [(0, 513), (513, 2048), (2048, 8192), (8192, 15000), (15000, T)]
res = {}
for name, x in [("ref_default", rf_def), ("ref_chunk64", rf_64), ("port_prefill2048", pf2048), ("port_prefill64", pf64)]:
    k = kl(f64, x)
    d = np.abs(lsm(f64) - lsm(x)).max(-1)
    res[name] = {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "argmax_pos": int(k.argmax()),
                 "top1_changes": int((f64.argmax(-1) != x.argmax(-1)).sum()),
                 "buckets": {f"{a}-{b}": [float(k[a:b].mean()), float(k[a:b].max()), float(d[a:b].max())] for a, b in buckets},
                 "doc_windows_max_kl": float(np.max([k[max(0, s - 32):s + 32].max() for s in docs])),
                 "doc_windows_max_abs_dlogprob": float(np.max([d[max(0, s - 32):s + 32].max() for s in docs]))}
for name, x in [("port_decode", pdec), ("port_prefill7", pf7)]:
    k = kl(f64[LO:HI + 1], x)
    res[name + f"_{LO}_{HI}"] = {"mean_kl": float(k.mean()), "max_kl": float(k.max()),
                                 "max_abs_dlogprob": float(np.abs(lsm(f64[LO:HI + 1]) - lsm(x)).max())}
for an, a, bn, b in [("port_prefill2048", pf2048, "port_prefill64", pf64), ("ref_default", rf_def, "port_prefill2048", pf2048),
                     ("ref_default", rf_def, "ref_chunk64", rf_64)]:
    k = kl(a, b)
    res[f"{an}_vs_{bn}"] = {"mean_kl": float(k.mean()), "max_kl": float(k.max()),
                            "max_abs_dlogprob": float(np.abs(lsm(a) - lsm(b)).max())}
print(json.dumps(res, indent=1))
(OUT / "report_forced.json").write_text(json.dumps(res, indent=1))
