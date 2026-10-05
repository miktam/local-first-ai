"""NLL-optimal temperature tau* of an fp32 model on raw T1-T6 (and T9), finer grid."""
import os, sys, json
os.environ["MLX_ENABLE_TF32"] = "0"
import numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
MP, TAG = sys.argv[1], sys.argv[2]
TEXTS = json.load(open("texts.json"))
m, tok = load(MP); m.set_dtype(mx.float32); mx.eval(m.parameters())
TAUS = np.round(np.arange(0.9, 2.001, 0.02), 3)
res = {}
for name in ("T1", "T2", "T3", "T4", "T5", "T6", "T9"):
    ids = tok.encode(TEXTS[name])[:16384]
    c = make_prompt_cache(m); x = mx.array([ids]); tot = np.zeros(len(TAUS)); n = 0
    for s in range(0, len(ids) - 1, 2048):
        e = min(len(ids) - 1, s + 2048)
        l = m(x[:, s:min(len(ids), s + 2048)], cache=c).astype(mx.float32)[0][: e - s]
        y = mx.array(ids[s + 1:e + 1])
        for i, t in enumerate(TAUS):
            lt = l / float(t); v = mx.sum(mx.logsumexp(lt, axis=-1) - mx.take_along_axis(lt, y[:, None], axis=-1)[:, 0])
            mx.eval(v); tot[i] += float(v)
        n += e - s
    i1 = list(TAUS).index(1.0); io = int(np.argmin(tot))
    res[name] = {"tau_star": float(TAUS[io]), "nll_at_1": tot[i1] / n, "gain_nats_per_tok": (tot[i1] - tot[io]) / n}
    print(name, res[name], flush=True)
json.dump(res, open(f"e1b_tau_{TAG}.json", "w"), indent=1)
