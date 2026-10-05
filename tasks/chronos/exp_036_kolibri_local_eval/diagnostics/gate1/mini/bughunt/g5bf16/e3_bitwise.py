"""Realistic dims: is each first-wave row's prefill KV (B=8, right-padded) bitwise equal to its
B=1 prefill (the mid-run / single path)? Which layer first differs? Also 8 identical prompts."""
import sys; sys.path.insert(0, ".")
import mk
ap = mk.parser(); a = ap.parse_args()
mx, np = mk.mx, mk.np
from mlx_lm.generate import _right_pad_prompts, _merge_caches
from mlx_lm.models.cache import make_prompt_cache
model, module = mk.build(a)
V = a.V
rng = np.random.default_rng(1)
def f(x): return np.array(x.astype(mx.float32))
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
    return [[c.extract(i) for c in caches] for i in range(len(toks_list))]
def cmp(ca, cb):
    first = None; mx_d = 0.0
    for li, (x, y) in enumerate(zip(ca, cb)):
        kx, ky = f(x.keys), f(y.keys); vx, vy = f(x.values), f(y.values)
        d = max(float(np.abs(kx - ky).max()), float(np.abs(vx - vy).max()))
        if d > 0 and first is None: first = li
        mx_d = max(mx_d, d)
    return first, mx_d
lens = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [rng.integers(1, V - 16, size=n - 1).tolist() for n in lens]
cB = prefill(prompts)
print("mixed first wave: row, len, first differing layer, max|d| (vs B=1 prefill)")
for j in range(8):
    c1 = prefill([prompts[j]])
    print("  ", j, lens[j], *cmp(cB[j], c1[0]))
same = [prompts[2]] * 8
cS = prefill(same); c1 = prefill([prompts[2]])
print("8 identical (699 tokens): rows vs row0:", [cmp(cS[i], cS[0]) for i in range(8)], " row0 vs B=1:", cmp(cS[0], c1[0]))
