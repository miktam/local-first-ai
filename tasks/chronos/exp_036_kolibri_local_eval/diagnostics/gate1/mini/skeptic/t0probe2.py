import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
exec(open("t0probe.py").read().split("out = {")[0])  # reuse setup + helpers
def t0kl(ps, idx=0):
    s = lp(single_t0(ps[idx])[0][None])[0]
    b = batched_t0(ps, len(ps))[idx]
    return float(kl(s[None], lp(b[None]))[0])
res = {}
T = texts
res["B2 [T0:37, T1:300] idx0"] = t0kl([T[0][:37], T[1][:300]], 0)
res["B2 [T1:300, T0:37] idx1"] = t0kl([T[1][:300], T[0][:37]], 1)
res["B2 [T3:37, T1:300] idx0"] = t0kl([T[3][:37], T[1][:300]], 0)
res["B2 [T5:37, T1:300] idx0"] = t0kl([T[5][:37], T[1][:300]], 0)
for n in (20, 30, 36, 37, 38, 45, 50, 64, 100):
    res[f"B2 [T0:{n}, T1:300] idx0"] = t0kl([T[0][:n], T[1][:300]], 0)
    res[f"B2 [T2:{n}, T1:300] idx0"] = t0kl([T[2][:n], T[1][:300]], 0)
res["B2 [T0:37, T1:38] idx0"] = t0kl([T[0][:37], T[1][:38]], 0)
res["B2 [T0:37, T1:40] idx0"] = t0kl([T[0][:37], T[1][:40]], 0)
res["B2 [T0:37, T1:37] idx0 (no pad)"] = t0kl([T[0][:37], T[1][:37]], 0)
for k, v in res.items(): print(f"{k:32s} {v:.3e}")
