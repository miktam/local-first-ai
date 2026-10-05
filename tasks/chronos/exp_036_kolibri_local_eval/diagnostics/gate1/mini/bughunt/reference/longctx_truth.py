"""Reference and port (fp32) against float64 truth on a tiny checkpoint with the
real attention geometry at 16,384 tokens built from the real T9 ids (mapped into
the tiny vocab; '\\n\\n' = id 263 kept). Per-position KL(f64 || X), buckets,
document-start windows and chunk boundaries."""
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
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
OUT = HERE / f"out_{CK.name}_{T}{TAG}"
OUT.mkdir(exist_ok=True)
T9 = np.array(json.load(open(HERE.parent / "work/gate_texts/T9.ids.json"))["ids"])[:T]
ids = np.where(T9 == 263, 263, T9 % 1008).astype(np.int64)
docs = [int(p) + 1 for p in np.flatnonzero(T9 == 263) if p + 1 < T]


def cached(name, fn):
    p = OUT / f"{name}.npy"
    if p.exists():
        return np.load(p)
    t0 = time.time()
    x = fn()
    np.save(p, x)
    print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
    return x


def lsm(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    lp, lq = lsm(p), lsm(q)
    return (np.exp(lp) * (lp - lq)).sum(-1)


# ---- truth --------------------------------------------------------------------------
import f64_dense
sel_path = OUT / "f64_margins.npy"
if not (OUT / "f64.npy").exists():
    lg, sels = f64_dense.forward(CK, ids, log=lambda m: None)
    np.save(OUT / "f64.npy", lg)
    np.save(sel_path, np.stack([s["margin"] for s in sels]))
    np.save(OUT / "f64_sel.npy", np.stack([s["sel"] for s in sels]))
f64 = np.load(OUT / "f64.npy")
margins = np.load(sel_path)
f64sel = np.load(OUT / "f64_sel.npy")

# ---- reference -------------------------------------------------------------------------
from reference import kolibri_ref


def ref_run(chunk):
    ref = kolibri_ref.KolibriReference(str(CK), attn_chunk=chunk)
    lg = ref.forward(ids)
    sel = np.stack([np.sort(r[1], -1) for r in ref.last_routing])
    np.save(OUT / f"ref_sel_{chunk}.npy", sel)
    return lg


ref_def = cached("ref_default", lambda: ref_run(None))
ref_c64 = cached("ref_chunk64", lambda: ref_run(64))
ref_c1000 = cached("ref_chunk1000", lambda: ref_run(1000))

# ---- port ------------------------------------------------------------------------------
from port_harness import load_port
from runner.generate import iter_teacher_force_logprobs
model = None


def port():
    global model
    if model is None:
        model = load_port(CK, float32=True)
    return model


def port_prefill(step):
    rows = [lp for _, lp in iter_teacher_force_logprobs(port(), list(map(int, ids)), step)]
    return np.concatenate(rows, 0)


p2048 = cached("port_prefill2048", lambda: port_prefill(2048))
p64 = cached("port_prefill64", lambda: port_prefill(64))
LO, HI = (15000, 15300) if T > 15300 else (T - 301, T - 1)


def port_decode():
    from gate.checks import g5_generation as g5
    return g5.decode_rows(port(), ids, LO, HI, 2048)


pdec = cached("port_decode", port_decode)

# ---- report ----------------------------------------------------------------------------
pos = np.arange(T)
buckets = [(0, 513), (513, 2048), (2048, 8192), (8192, T)]
res = {}
for name, x in [("ref_default", ref_def), ("ref_chunk64", ref_c64), ("ref_chunk1000", ref_c1000),
                ("port_prefill2048", p2048), ("port_prefill64", p64)]:
    k = kl(f64, x)
    d = np.abs(lsm(f64) - lsm(x)).max(-1)
    top = (f64.argmax(-1) != x.argmax(-1))
    r = {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "argmax_pos": int(k.argmax()),
         "top1_changes": int(top.sum()),
         "buckets": {f"{a}-{b}": {"mean_kl": float(k[a:b].mean()), "max_kl": float(k[a:b].max()),
                                   "max_abs_dlogprob": float(d[a:b].max())} for a, b in buckets if a < T},
         "doc_windows_mean_kl": float(np.mean([k[max(0, s - 32):s + 32].mean() for s in docs])) if docs else None,
         "doc_windows_max_kl": float(np.max([k[max(0, s - 32):s + 32].max() for s in docs])) if docs else None}
    res[name] = r
for name, x in [("port_decode", pdec)]:
    k = kl(f64[LO:HI + 1], x)
    res[name] = {"range": [LO, HI], "mean_kl": float(k.mean()), "max_kl": float(k.max())}
for a_name, a, b_name, b in [("port_prefill2048", p2048, "port_prefill64", p64),
                             ("ref_default", ref_def, "port_prefill2048", p2048),
                             ("ref_default", ref_def, "ref_chunk64", ref_c64)]:
    k = kl(a, b)
    res[f"{a_name}_vs_{b_name}"] = {"mean_kl": float(k.mean()), "max_kl": float(k.max()),
                                    "range_15000_15300_mean": float(k[LO:HI + 1].mean())}
k = kl(p2048[LO:HI + 1], pdec)
res["port_prefill2048_vs_decode"] = {"mean_kl": float(k.mean()), "max_kl": float(k.max())}
# routing agreement with f64
for c in ("None", "64", "1000"):
    p = OUT / f"ref_sel_{c}.npy"
    if p.exists():
        s = np.load(p)
        dis = (s != f64sel).any(-1)
        res[f"ref_sel_{c}_routing_disagreements_vs_f64"] = int(dis.sum())
res["f64_min_selection_margin"] = float(margins.min())
res["f64_n_margin_lt_1e-5"] = int((margins < 1e-5).sum())
res["docs"] = docs
print(json.dumps(res, indent=1))
(OUT / "report.json").write_text(json.dumps(res, indent=1))
