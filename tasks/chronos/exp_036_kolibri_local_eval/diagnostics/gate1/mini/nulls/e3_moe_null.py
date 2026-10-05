"""E3: activation-precision null on a CORRECT MLX MoE (mlx_lm gemma4 26B-A4B, OptiQ 4-bit
weights). Same weights; activations bf16 (as loaded) vs fp32 (model.set_dtype(float32),
MLX_ENABLE_TF32=0). Texts T1-T6 and T9 as the assistant turn after the Amendment 5
wrapper ("Write a text.", effort none), only text tokens scored. Reports KL(fp32||bf16),
dNLL, dNLL/KL, decisive top-1 (fp32 lead >= 2 nats), by T9 bucket.
Question: is Kolibri K8's activation-driven KL (~0.03-0.07) MoE-normal in MLX?"""
import os, sys, json, time
os.environ["MLX_ENABLE_TF32"] = "0"
K = "$KIT"
sys.path.insert(0, K)
import numpy as np
import mlx.core as mx
from pathlib import Path
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MP = sys.argv[1]
TEXTS = json.load(open("texts.json"))
CHUNK = 1024
ref, tok = load(MP)
ref.set_dtype(mx.float32)
mx.eval(ref.parameters())
bf, _ = load(MP)
mx.eval(bf.parameters())

from runner import chat
from bench import kl_8v4
_, wrap_ids, _ = chat.render("gemma4", tok, [{"role": "user", "content": kl_8v4.WRAP_USER_MESSAGE}], kl_8v4.WRAP_EFFORT)


def lsm(x):
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


rng = np.random.default_rng(36)


def boot(d, k, block=64, B=2000):
    n = d.size; nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    r = d[idx].mean(1) / k[idx].mean(1)
    return [float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])], [float(x) for x in np.percentile(r, [2.5, 97.5])]


def run(ids, n_wrap):
    T = len(ids)
    cr, cb = make_prompt_cache(ref), make_prompt_cache(bf)
    x = mx.array([ids])
    KL, DN, LEAD, AG = [], [], [], []
    for s in range(0, T - 1, CHUNK):
        e = min(T - 1, s + CHUNK)
        xin = x[:, s:min(T, s + CHUNK)]
        lr = lsm(ref(xin, cache=cr).astype(mx.float32)[0][: e - s])
        lb = lsm(bf(xin, cache=cb).astype(mx.float32)[0][: e - s])
        y = mx.array(ids[s + 1:e + 1])
        pr = mx.exp(lr)
        kl = mx.sum(pr * (lr - lb), axis=-1)
        dn = mx.take_along_axis(lr - lb, y[:, None], axis=-1)[:, 0]
        t2 = mx.topk(lr, 2, axis=-1)
        lead = mx.max(t2, axis=-1) - mx.min(t2, axis=-1)
        ag = mx.argmax(lr, axis=-1) == mx.argmax(lb, axis=-1)
        mx.eval(kl, dn, lead, ag)
        KL.append(np.array(kl)); DN.append(np.array(dn)); LEAD.append(np.array(lead)); AG.append(np.array(ag))
        mx.clear_cache()
    sl = slice(n_wrap - 1, None)  # rows predicting the text's tokens
    return {k: np.concatenate(v)[sl] for k, v in (("kl", KL), ("dnll", DN), ("lead", LEAD), ("agree", AG))}


def summ(z):
    dec = z["lead"] >= 2.0
    ci, rci = boot(z["dnll"], z["kl"])
    return {"n": int(z["kl"].size), "kl": float(z["kl"].mean()), "dnll": float(z["dnll"].mean()), "dnll_ci": ci,
            "ratio": float(z["dnll"].mean() / z["kl"].mean()), "ratio_ci": rci, "max_kl": float(z["kl"].max()),
            "n_dec": int(dec.sum()), "dec_miss": int((dec & ~z["agree"].astype(bool)).sum()),
            "top1_dec": float(z["agree"][dec].mean()) if dec.any() else None, "top1": float(z["agree"].mean())}


res = {"model": MP, "device": mx.device_info()["device_name"], "wrapper_tokens": len(wrap_ids), "texts": {}}
for name in ("T1", "T2", "T3", "T4", "T5", "T6", "T9"):
    body = tok.encode(TEXTS[name], add_special_tokens=False)
    if name == "T9":
        body = body[:16384 - len(wrap_ids)]
    t0 = time.time()
    z = run(list(wrap_ids) + body, len(wrap_ids))
    res["texts"][name] = summ(z)
    if name == "T9":
        off = len(wrap_ids)
        for b, (a, c) in {"0-2k": (0, 2048), "2-8k": (2048, 8192), "8-16k": (8192, 16384)}.items():
            sl = slice(max(0, a - off), c - off)
            res["texts"][f"T9_{b}"] = summ({k: v[sl] for k, v in z.items()})
    print(name, len(body), f"{time.time() - t0:.0f}s", json.dumps(res["texts"][name]), flush=True)
json.dump(res, open("e3_gemma_moe.json", "w"), indent=1)
for k in ("T9_0-2k", "T9_2-8k", "T9_8-16k"):
    print(k, json.dumps(res["texts"][k]))
