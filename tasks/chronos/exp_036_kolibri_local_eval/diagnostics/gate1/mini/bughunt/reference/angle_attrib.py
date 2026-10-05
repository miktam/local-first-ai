"""Attribution of the residual position-dependent error (forced routing): f64
with vLLM-style fp32 RoPE angles, routing forced to the f64 truth's selection,
against the forced reference and the forced port. Also: shape dependence of
ref and port for identical tokens (ids[:1536] standalone vs the 16k run's rows)."""
import json, os, sys
from pathlib import Path
os.environ["F64_FP32_ANGLES"] = "1"
K = "$KIT"
sys.path.insert(0, K); sys.path.insert(0, K + "/tests")
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import numpy as np
import mlx.core as mx
HERE = Path(__file__).resolve().parent
CK = HERE / "ckpt_k16_L10"
T = 16384
OUT = HERE / f"out_{CK.name}_{T}"
T9 = np.array(json.load(open(HERE.parent / "work/gate_texts/T9.ids.json"))["ids"])[:T]
ids = np.where(T9 == 263, 263, T9 % 1008).astype(np.int64)
fsel = np.load(OUT / "f64_sel.npy")


def lsm(x):
    x = np.asarray(x, np.float64); m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    lp, lq = lsm(p), lsm(q)
    return (np.exp(lp) * (lp - lq)).sum(-1)


import f64_dense
p = OUT / "f64_fp32angles_forced.npy"
if not p.exists():
    # force the f64 run's routing to the f64 (exact-angle) selection so all runs share one routing
    W = f64_dense.load_weights(CK)
    orig_argsort = np.argsort
    layer = {"i": 0}

    def forward_forced():
        import types
        src = Path(f64_dense.__file__).read_text().replace(
            'sel = np.argsort(-score, axis=-1, kind="stable")[:, :k]', 'sel = FSEL[i]')
        ns = {}
        exec(compile(src, "f64_forced", "exec"), ns)
        ns["FSEL"] = fsel
        return ns["forward"](CK, ids, W=W, log=lambda m: None)[0]
    np.save(p, forward_forced())
f32a = np.load(p)
f64 = np.load(OUT / "f64.npy")
rf = np.load(OUT / "forced_ref_default.npy")
pf = np.load(OUT / "forced_port_prefill2048.npy")
buckets = [(0, 513), (513, 2048), (2048, 8192), (8192, 15000), (15000, T)]
res = {}
for name, x in [("ref_forced", rf), ("port_forced", pf), ("f64_exact_angles", f64)]:
    k = kl(f32a, x)
    d = np.abs(lsm(f32a) - lsm(x)).max(-1)
    res[f"KL(f64_fp32angles || {name})"] = {f"{a}-{b}": [float(k[a:b].mean()), float(k[a:b].max()), float(d[a:b].max())] for a, b in buckets}

# shape dependence for identical tokens (unforced, the default runs)
from reference import kolibri_ref
from port_harness import load_port
from runner.generate import iter_teacher_force_logprobs
r = kolibri_ref.KolibriReference(str(CK)); r._log = lambda m: None
ref1536 = r.forward(ids[:1536])
ref16k = np.load(OUT / "ref_default.npy")[:1536]
model = load_port(CK, float32=True)
port1536 = np.concatenate([lp for _, lp in iter_teacher_force_logprobs(model, list(map(int, ids[:1536])), 2048)], 0)
port16k = np.load(OUT / "port_prefill2048.npy")[:1536]
res["ref_T1like_standalone_vs_ref_T9_prefix"] = {"bit_identical": bool(np.array_equal(ref1536, ref16k)),
                                                 "max_abs_dlogprob": float(np.abs(lsm(ref1536) - lsm(ref16k)).max()),
                                                 "max_kl": float(kl(ref1536, ref16k).max())}
res["port_T1like_standalone_vs_port_T9_prefix"] = {"bit_identical": bool(np.array_equal(port1536, port16k)),
                                                   "max_abs_dlogprob": float(np.abs(lsm(port1536) - lsm(port16k)).max()),
                                                   "max_kl": float(kl(port1536, port16k).max())}
print(json.dumps(res, indent=1))
(OUT / "report_angles_shapes.json").write_text(json.dumps(res, indent=1))
