"""First p64-vs-p2048 divergence on the sharpened tiny model: router margin,
size of the pre-router difference, cache state and mask at that chunk."""
from common_bh import *
ck = sys.argv[1]; T = 2048
m, _, mod = common.load_port(Path(ck)); m.set_dtype(mx.float32); mx.eval(m.parameters())
for i, l in enumerate(m.layers): l._bh = i
ST = {"rec": [], "cache_log": []}
def call(self, x, mask=None, cache=None):
    if cache is not None and self._bh in (0, 4):
        ST["cache_log"].append((self._bh, type(cache).__name__, cache.offset, None if cache.keys is None else cache.keys.shape[2],
                                getattr(cache, "_idx", None), mask if isinstance(mask, (str, type(None))) else tuple(mask.shape)))
    r = self.branches(x, mask, cache)
    ST["rec"].append((r[4], r[5], self.post_attention_layernorm(r[1]))); return r[3]
mod.DecoderLayer.__call__ = call
rng = np.random.default_rng(5); ids = rng.integers(0, 1008, size=16384).astype(np.int32)
def run(step):
    cache = m.make_cache(); lg, sel, xin, logs = [], [], [], []
    for p in range(0, T, step):
        ST["rec"] = []; ST["cache_log"] = []
        mx.eval(m(mx.array(ids[p:p + step])[None], cache=cache))
        lg.append(np.stack([np.array(r[0][0]) for r in ST["rec"]])); sel.append(np.stack([np.sort(np.array(r[1][0]), -1) for r in ST["rec"]]))
        xin.append(np.stack([np.array(r[2][0]) for r in ST["rec"]])); logs.append((p, list(ST["cache_log"])))
    return np.concatenate(lg, 1), np.concatenate(sel, 1), np.concatenate(xin, 1), logs
A = run(2048); B = run(64)
fl = (A[1] != B[1]).any(-1)  # [layers, T]
pos = int(np.where(fl.any(0))[0][0]); lay = int(np.where(fl[:, pos])[0][0])
bias = np.array(m.layers[lay].mlp.gate.expert_bias)
k = m.args.num_experts_per_tok
for nm, R in (("p2048", A), ("p64", B)):
    s = np.sort(R[0][lay, pos] + bias)[::-1]
    print(f"{nm}: layer {lay} pos {pos} selected {R[1][lay, pos].tolist()} margin(k-th minus k+1-th biased score) {s[k-1]-s[k]:.3e}")
x0, x1 = A[2][lay, pos], B[2][lay, pos]
print(f"router input rel diff at that point {np.linalg.norm(x0-x1)/np.linalg.norm(x0):.2e}; router logit max abs diff {np.abs(A[0][lay,pos]-B[0][lay,pos]).max():.2e}")
# median margin over all positions/layers, p2048
sc = np.sort(A[0] + np.stack([np.array(l.mlp.gate.expert_bias) for l in m.layers])[:, None, :], -1)[..., ::-1]
mg = sc[..., k-1] - sc[..., k]
print(f"p2048 margins: median {np.median(mg):.3e}, frac < 1e-4 {np.mean(mg<1e-4):.2e}, frac < 1e-5 {np.mean(mg<1e-5):.2e}")
pre = [l for p, l in B[3] if p <= pos < p + 64][0]
print("p64 chunk containing it: cache log (layer, class, offset before update, keys len before, _idx before, mask):", pre)
