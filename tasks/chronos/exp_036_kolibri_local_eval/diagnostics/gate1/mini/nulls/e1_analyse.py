import numpy as np, json, sys
rng = np.random.default_rng(36)
def boot(d, k, block=64, B=2000):
    n = d.size; nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(1, n - block + 1), size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    dm, km = d[idx].mean(1), k[idx].mean(1)
    r = dm / km
    return np.percentile(dm, [2.5, 97.5]), np.percentile(r, [2.5, 97.5])
for tag in sys.argv[1:]:
    print("=====", tag)
    for t in ("T1", "T2", "T3", "T4", "T5", "T6", "T9"):
        z = np.load(f"e1_{tag}_{t}.npz")
        segs = {t: slice(None)} if t != "T9" else {"T9": slice(None), "0-2k": slice(0, 2047), "2-8k": slice(2047, 8191), "8-16k": slice(8191, None)}
        for name, sl in segs.items():
            row = [f"{name:6s}"]
            for v in ("bf16", "q8", "q4"):
                kl = z[f"{v}_kl"][sl]; dn = (z[f"{v}_nll"] - z["ref_nll"])[sl]
                dec = z["ref_lead"][sl] >= 2.0
                ci, rci = boot(dn, kl)
                miss = int((~z[f"{v}_agree"][sl].astype(bool) & dec).sum())
                row.append(f"{v}: KL {kl.mean():.4f} dNLL {dn.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] r {dn.mean()/kl.mean():+.2f} [{rci[0]:+.2f},{rci[1]:+.2f}] dec {1-miss/dec.sum():.4f} ({miss}/{dec.sum()})")
            print(" | ".join(row))
