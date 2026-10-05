from common_bh import *
import time
m, mod = load(sys.argv[1] if len(sys.argv) > 1 else "ckpt_w513h128")
T = 16384
rng = np.random.default_rng(7)
ids = rng.integers(0, m.args.vocab_size - 16, size=T)
lo, hi = 15000, 15300
res = {}
for step in (2048, 64, 256, 512, 513, 514):
    t = time.time(); res[step] = prefill_hidden(m, ids, step, lo, hi); print("step", step, f"{time.time()-t:.1f}s", flush=True)
t = time.time(); res["dec"] = decode_hidden(m, ids, lo, hi); print("decode", f"{time.time()-t:.1f}s")
base = res["dec"]
for k, v in res.items():
    r = rel(v, base)
    print(k, "rel vs decode: max", f"{r.max():.3e}", "mean", f"{r.mean():.3e}", "first>1e-4", (lo + int(np.argmax(r > 1e-4))) if (r > 1e-4).any() else None)
