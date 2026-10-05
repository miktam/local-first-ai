import numpy as np, sys, itertools
z = np.load(sys.argv[1])
names = ["p2048", "p64", "p256", "p512", "p513", "p514", "dec"]
lp = {n: z[f"{n}_lp"] for n in names}; sel = {n: z[f"{n}_sel"] for n in names}
for a, b in itertools.combinations(names, 2):
    d = np.abs(lp[a] - lp[b]).max(-1)
    fl = (sel[a] != sel[b]).any(-1)  # [layers, T]
    anyf = np.where(fl.any(0))[0]
    f1 = int(np.argmax(d > 1e-3)) if (d > 1e-3).any() else None
    print(f"{a:6s} {b:6s} maxabs-lp<=1e-5 frac {np.mean(d<=1e-5):.3f}  first >1e-3 at {f1}  first flip {anyf[:3].tolist()} (layer {np.where(fl[:, anyf[0]])[0].tolist() if anyf.size else None})  bitwise-equal positions {np.mean(d==0):.3f}")
