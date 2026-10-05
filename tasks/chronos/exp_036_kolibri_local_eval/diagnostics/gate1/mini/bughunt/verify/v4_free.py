"""Free routing (as the gate runs): the same paths as v3 without forcing. Shows how rounding seeds of different size
(GEMM/SDPA order ~1e-7; RoPE angle up to ~1e-3 rad deep in the sequence) are amplified by router flips."""
from common_v import *
import time, json
from pathlib import Path
from gate import common
from reference import kolibri_ref as KR
V = Path(__file__).resolve().parent
ck = V / sys.argv[1]; T = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
LO, HI = 15000, min(T - 1, 15300)
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 5)
ids = rng.integers(0, 1008, size=T).astype(np.int32)
exec(open(V / "v3_forced.py").read().split("res = {}")[0].split("ids = rng.integers")[1].split("\n", 1)[1])
res, sels = {}, {}
for variant in ("base", "rope"):
    m, ST = load(variant)
    for name, sc in (("p2048", sched_fixed(2048)), ("p64", sched_fixed(64)), ("dec", sched_dec())):
        t = time.time(); res[f"{variant}_{name}"], sels[f"{variant}_{name}"] = run(m, ST, sc); print(f"{variant} {name} free {time.time()-t:.0f}s", flush=True)
    del m
r = KR.KolibriReference(str(ck)); r.last_routing = []
lg = r.forward(ids.astype(np.int64)).astype(np.float64); lg -= lg.max(-1, keepdims=True)
res["ref"] = (lg - np.log(np.exp(lg).sum(-1, keepdims=True))).astype(np.float32)
sels["ref"] = np.stack([np.sort(i_, -1) for (_, i_) in r.last_routing[-r.cfg.num_hidden_layers:]])
np.savez_compressed(V / f"v4_{ck.name}_{T}.npz", ids=ids, **res)
def kl(a, b): return (np.exp(a.astype(np.float64)) * (a.astype(np.float64) - b)).sum(-1)
blocks = [(a, min(T, a + 2048)) for a in range(0, T, 2048)]
out = {}
def rows(X): return X[LO:HI + 1] if X.shape[0] == T else X[-(HI + 1 - LO):]
for a, b in [("base_p2048", "base_p64"), ("base_p2048", "base_dec"), ("base_p64", "base_dec"),
             ("ref", "base_p2048"), ("ref", "base_p64"), ("ref", "rope_p2048"), ("ref", "rope_p64"), ("rope_p2048", "rope_p64")]:
    A, B = res[a], res[b]
    if A.shape[0] != T or B.shape[0] != T:
        k = kl(rows(A), rows(B)); ch = rows(A).argmax(-1) != rows(B).argmax(-1)
        out[f"{a}|{b}"] = {"range_mean": float(k.mean()), "range_max": float(k.max()), "range_top1_changes": int(ch.sum())}
        print(f"KL({a}||{b}) [{LO}-{HI}] mean {k.mean():.2e} max {k.max():.2e} top1 changes {int(ch.sum())}"); continue
    k = kl(A, B)
    fl = (sels[a] != sels[b]).any(-1) if sels[a].shape == sels[b].shape else None
    out[f"{a}|{b}"] = {"block_mean": [float(k[x:y].mean()) for x, y in blocks], "block_max": [float(k[x:y].max()) for x, y in blocks],
                       "range_mean": float(k[LO:HI + 1].mean()), "range_max": float(k[LO:HI + 1].max()),
                       "first_flip_pos": (int(np.where(fl.any(0))[0][0]) if fl is not None and fl.any() else None),
                       "n_flips_layer_pos": (int(fl.sum()) if fl is not None else None)}
    print(f"KL({a}||{b}) per-2048 mean " + " ".join(f"{v:.1e}" for v in out[f'{a}|{b}']['block_mean'])
          + f" | [{LO}-{HI}] mean {k[LO:HI+1].mean():.1e} max {k[LO:HI+1].max():.1e} | router flips (layer,pos) {out[f'{a}|{b}']['n_flips_layer_pos']} first at {out[f'{a}|{b}']['first_flip_pos']}")
json.dump(out, open(V / f"v4_{ck.name}_{T}.json", "w"), indent=1)
