"""E1: precision null on a CORRECT MLX implementation (mlx_lm Qwen3), mini only.

Question (skeptic objection 4 / integrity G3 test 2): when a correct implementation
runs in bf16 (and with q8 / q4 g64 weights, bf16 activations) and is compared with
the SAME model in fp32 on the gate texts, is dNLL ~= +KL (the zero-mean-noise
prediction), or is dNLL << KL (a systematic, temperature-like deviation)?
Also: the fitted temperature tau* (KL-optimal and NLL-optimal) per text, decisive
top-1 by T9 bucket, entropy shift. Writes e1_<model>.json (+ per-position npz).
"""
import os, sys, json, time
os.environ["MLX_ENABLE_TF32"] = "0"
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MP = sys.argv[1]
TAG = sys.argv[2]
TEXTS = json.load(open("texts.json"))
CHUNK = 2048
TAUS = np.round(np.arange(0.85, 1.1501, 0.01), 4)
LEAD = 2.0


def load_variant(kind):
    m, tok = load(MP)
    if kind == "fp32":
        m.set_dtype(mx.float32)
    elif kind in ("q8", "q4"):
        nn.quantize(m, group_size=64, bits=8 if kind == "q8" else 4)
    mx.eval(m.parameters())
    return m, tok


ref, tok = load_variant("fp32")
variants = {k: load_variant(k)[0] for k in ("bf16", "q8", "q4")}


def lsm(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def run_text(name, ids):
    ids = list(ids)
    T = len(ids)
    caches = {"ref": make_prompt_cache(ref), **{k: make_prompt_cache(m) for k, m in variants.items()}}
    x = mx.array([ids])
    per = {k: {f: [] for f in ("kl", "nll", "agree", "H")} for k in variants}
    ref_per = {f: [] for f in ("nll", "lead", "H")}
    taus = mx.array(TAUS.astype(np.float32))
    klt = {k: np.zeros(len(TAUS)) for k in variants}
    nllt = {k: np.zeros(len(TAUS)) for k in variants}
    nllt_ref = np.zeros(len(TAUS))
    for s in range(0, T - 1, CHUNK):
        e = min(T - 1, s + CHUNK)  # rows s..e-1 predict ids[s+1..e]
        xin = x[:, s:min(T, s + CHUNK)]
        lr = ref(xin, cache=caches["ref"]).astype(mx.float32)[0][: e - s]
        y = mx.array(ids[s + 1:e + 1])
        lpr = lsm(lr)
        pr = mx.exp(lpr)
        nll_r = -mx.take_along_axis(lpr, y[:, None], axis=-1)[:, 0]
        top2 = mx.topk(lpr, 2, axis=-1)
        lead = mx.max(top2, axis=-1) - mx.min(top2, axis=-1)
        Hr = -mx.sum(pr * lpr, axis=-1)
        am_r = mx.argmax(lpr, axis=-1)
        mx.eval(nll_r, lead, Hr, am_r)
        ref_per["nll"].append(np.array(nll_r)); ref_per["lead"].append(np.array(lead)); ref_per["H"].append(np.array(Hr))
        for i, t in enumerate(TAUS):
            lt = lsm(lr / float(t))
            v = -mx.sum(mx.take_along_axis(lt, y[:, None], axis=-1))
            mx.eval(v)
            nllt_ref[i] += float(v)
        for k, m in variants.items():
            lv = m(xin, cache=caches[k]).astype(mx.float32)[0][: e - s]
            lpv = lsm(lv)
            kl = mx.sum(pr * (lpr - lpv), axis=-1)
            nll_v = -mx.take_along_axis(lpv, y[:, None], axis=-1)[:, 0]
            agree = mx.argmax(lpv, axis=-1) == am_r
            Hv = -mx.sum(mx.exp(lpv) * lpv, axis=-1)
            mx.eval(kl, nll_v, agree, Hv)
            per[k]["kl"].append(np.array(kl)); per[k]["nll"].append(np.array(nll_v))
            per[k]["agree"].append(np.array(agree)); per[k]["H"].append(np.array(Hv))
            for i, t in enumerate(TAUS):
                lt = lsm(lv / float(t))
                a = mx.sum(pr * (lpr - lt))
                b = -mx.sum(mx.take_along_axis(lt, y[:, None], axis=-1))
                mx.eval(a, b)
                klt[k][i] += float(a); nllt[k][i] += float(b)
            del lv, lpv
        del lr, lpr, pr
        mx.clear_cache()
    R = {f: np.concatenate(v) for f, v in ref_per.items()}
    out = {"n": int(R["nll"].size), "ref_nll": float(R["nll"].mean()),
           "ref_tau_nll_opt": float(TAUS[int(np.argmin(nllt_ref))]),
           "ref_nll_at_opt_minus_at1": float((nllt_ref.min() - nllt_ref[list(TAUS).index(1.0)]) / R["nll"].size)}
    arrays = {"ref_nll": R["nll"], "ref_lead": R["lead"], "ref_H": R["H"]}
    for k in variants:
        V = {f: np.concatenate(v) for f, v in per[k].items()}
        arrays.update({f"{k}_{f}": v for f, v in V.items()})
        out[k] = {"kl": float(V["kl"].mean()), "dnll": float((V["nll"] - R["nll"]).mean()),
                  "tau_kl_opt": float(TAUS[int(np.argmin(klt[k]))]),
                  "kl_at_opt_over_kl_at1": float(klt[k].min() / klt[k][list(TAUS).index(1.0)]),
                  "tau_nll_opt": float(TAUS[int(np.argmin(nllt[k]))]),
                  "dH": float((V["H"] - R["H"]).mean())}
    np.savez_compressed(f"e1_{TAG}_{name}.npz", **arrays)
    return out


res = {"model": MP, "tag": TAG, "device": mx.device_info()["device_name"], "mlx": mx.__version__, "texts": {}}
for name in ("T1", "T2", "T3", "T4", "T5", "T6", "T9"):
    ids = tok.encode(TEXTS[name])
    if name == "T9":
        ids = ids[:16384]
    t0 = time.time()
    res["texts"][name] = run_text(name, ids)
    print(name, len(ids), f"{time.time() - t0:.0f}s", json.dumps(res["texts"][name]), flush=True)
json.dump(res, open(f"e1_{TAG}.json", "w"), indent=1)
