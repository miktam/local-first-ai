import sys; sys.path.insert(0, ".")
import mk
a = mk.parser().parse_args()
mx, np = mk.mx, mk.np
from mlx_lm.generate import _merge_caches
from mlx_lm.models.cache import make_prompt_cache
model, module = mk.build(a)
rng = np.random.default_rng(1)
toks = rng.integers(1, a.V - 16, size=699).tolist()
def f(x): return np.array(x.astype(mx.float32))
lay = model.model.layers[0]; att = lay.self_attn
for B in (8, 1):
    x = mx.array([toks] * B)
    h = model.model.embed_tokens(x)
    hn = lay.input_layernorm(h)
    k = att.k_proj(hn).reshape(B, 699, a.kv, a.D)
    kn = att.k_norm(k).transpose(0, 2, 1, 3)
    caches = _merge_caches([make_prompt_cache(model) for _ in range(B)])
    off = caches[0].offset
    kr = att.rope(kn, offset=off)
    kr_int = att.rope(kn, offset=0)
    mx.eval(hn, k, kn, kr, kr_int)
    globals()[f"r{B}"] = dict(h=f(h[0]), hn=f(hn[0]), k=f(k[0]), kn=f(kn[0]), kr=f(kr[0]), kr_int=f(kr_int[0]), off=np.array(off))
for name in ("h", "hn", "k", "kn", "kr", "kr_int"):
    d = np.abs(r8[name] - r1[name])
    print(name, "max|B8-B1|", float(d.max()), "argmax", np.unravel_index(d.argmax(), d.shape))
print("off8", r8["off"], "off1", r1["off"])
d = np.abs(r8["kr"] - r8["kr_int"]); print("B8 array-offset vs int-offset rope max", float(d.max()), np.unravel_index(d.argmax(), d.shape))
d = np.abs(r1["kr"] - r1["kr_int"]); print("B1 array-offset vs int-offset rope max", float(d.max()))
