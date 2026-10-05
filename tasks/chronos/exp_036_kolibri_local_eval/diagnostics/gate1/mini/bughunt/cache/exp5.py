"""The gate's own chunk-64 procedure (gate.checks.g5_generation.prefill_rows /
decode_rows, T9-like range 15000-15300) on the tiny real-layout model, free
routing; then the first divergence: router margin and cache/mask state."""
from common_bh import *
from gate.checks import g5_generation as g5
ck = sys.argv[1]
m, _, mod = common.load_port(Path(ck)); m.set_dtype(mx.float32); mx.eval(m.parameters())
T = 16384; rng = np.random.default_rng(5); ids = rng.integers(0, 1008, size=T).astype(np.int32)
def lsm(x): x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
def kl(a, b): return (np.exp(a) * (a - b)).sum(-1)
for lo, hi in ((520, 1100), (15000, 15300)):
    a = lsm(g5.prefill_rows(m, ids, lo, hi, 2048)); b = lsm(g5.prefill_rows(m, ids, lo, hi, 64)); d = lsm(g5.decode_rows(m, ids, lo, hi, 2048))
    for n, (p, q) in {"prefill2048_vs_prefill64": (a, b), "prefill2048_vs_decode": (a, d), "prefill64_vs_decode": (b, d)}.items():
        k = kl(p, q); ch = p.argmax(-1) != q.argmax(-1)
        print(f"[{lo}-{hi}] {n:26s} mean KL {k.mean():.3e} max {k.max():.3e} n>1e-6 {int((k>1e-6).sum())}/{k.size} top1 changes {int(ch.sum())}")
