"""Model-level KV precision of the G5 first-wave prefill (B=8, right-padded to 1099, finalize)
against the same rows prefilled alone (B=1, the mid-run and single path), both against an
fp32-activation truth on the same weights. Realistic dims, random weights (mk.py); runs on
the mini and the mbp. Output: per row, per layer, rel L2 error of K and V (valid entries)."""
import sys, json; sys.path.insert(0, ".")
import mk
ap = mk.parser(); ap.add_argument("--out", default=None)
a = ap.parse_args()
mx, np = mk.mx, mk.np
from mlx_lm.generate import _right_pad_prompts, _merge_caches
from mlx_lm.models.cache import make_prompt_cache
model, module = mk.build(a)
rng = np.random.default_rng(1)
lens = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [rng.integers(1, a.V - 16, size=n - 1).tolist() for n in lens]
def f64(x): return np.array(x.astype(mx.float32)).astype(np.float64)
def prefill(toks_list):
    caches = _merge_caches([make_prompt_cache(model) for _ in toks_list])
    L = [len(t) for t in toks_list]; m = max(L); pad = [m - l for l in L]
    x = _right_pad_prompts(toks_list, m)
    if max(pad) > 0:
        for c in caches: c.prepare(lengths=L, right_padding=pad)
    model(x, cache=caches)
    if max(pad) > 0:
        for c in caches: c.finalize()
    mx.eval([c.state for c in caches])
    return [[(f64(c.extract(i).keys), f64(c.extract(i).values)) for c in caches] for i in range(len(toks_list))]
cB = prefill(prompts)
c1 = [prefill([p])[0] for p in prompts]
model.set_dtype(mx.float32); mx.eval(model.parameters())
cT = [prefill([p])[0] for p in prompts]
def rel(x, r): return float(np.linalg.norm(x - r) / np.linalg.norm(r))
out = []
for j in range(8):
    row = []
    for li in range(a.layers):
        (kb, vb), (k1, v1), (kt, vt) = cB[j][li], c1[j][li], cT[j][li]
        row.append({"layer": li, "relK_B8": rel(kb, kt), "relK_B1": rel(k1, kt), "relV_B8": rel(vb, vt), "relV_B1": rel(v1, vt),
                    "bitwise": bool(np.array_equal(kb, k1) and np.array_equal(vb, v1))})
    out.append({"row": j, "len": lens[j], "layers": row})
    print(j, lens[j], " ".join(f"L{r['layer']}:K {r['relK_B8']:.2e}/{r['relK_B1']:.2e} V {r['relV_B8']:.2e}/{r['relV_B1']:.2e}{'' if not r['bitwise'] else ' =='}" for r in row))
agg = {k: float(np.mean([r[k] for o in out for r in o["layers"]])) for k in ("relK_B8", "relK_B1", "relV_B8", "relV_B1")}
print("mean over rows and layers (B8 / B1 vs fp32 truth):", agg)
if a.out:
    json.dump({"device": mx.device_info().get("device_name"), "args": vars(a), "rows": out, "mean": agg}, open(a.out, "w"), indent=1)
