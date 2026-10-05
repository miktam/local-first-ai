"""Free-routing reference vs free-routing port (exp2 rows) on the sharpened tiny model."""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"; sys.path[:0] = [K]
import numpy as np
from reference import kolibri_ref as KR
z = np.load("exp2_ck_real_s3.npz"); ids = z["ids"]
r = KR.KolibriReference("ck_real_s3")
lg = r.forward(ids.astype(np.int64)); lg = lg - lg.max(-1, keepdims=True); ref = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
def kl(a, b): return (np.exp(a) * (a - b)).sum(-1)
for n in ("p2048", "p64", "dec"):
    k = kl(ref, z[f"{n}_lp"])
    print(f"KL(ref||{n}) free routing, per-2048-block mean " + " ".join(f"{k[x:x+2048].mean():.1e}" for x in range(0, 16384, 2048)) + f" | 15000-15300 mean {k[15000:15301].mean():.3e} max {k[15000:15301].max():.3e}")
