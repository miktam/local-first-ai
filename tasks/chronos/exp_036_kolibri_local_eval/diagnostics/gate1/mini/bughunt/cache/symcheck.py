"""Symbolic check of the port's cache + mask logic: keys carry their absolute
position; for every query the set of visible absolute positions must equal
[i-512, i] (sliding) / [0, i] (full)."""
from common_bh import *
m, mod = load("ckpt_w513h128")
W = m.args.sliding_window
def visible(mask, kpos, qL):
    kL = kpos.shape[0]
    if mask is None:
        assert qL == 1; allowed = np.ones((1, kL), bool)
    elif isinstance(mask, str):
        assert mask == "causal"
        allowed = np.arange(kL)[None] <= (kL - qL) + np.arange(qL)[:, None]
    else:
        a = np.array(mask)
        allowed = np.broadcast_to(a, (qL, kL)) if a.ndim <= 2 else None
        assert a.shape[-1] == kL and a.shape[-2] in (1, qL), (a.shape, qL, kL)
        allowed = np.broadcast_to(a.reshape(a.shape[-2:]), (qL, kL))
    return allowed
def run(schedule, T, tag):
    cache = m.make_cache()
    si, fi = m.model.swa_idx, m.model.fa_idx
    pos = 0; bad = []
    for L in schedule(T):
        h = mx.zeros((1, L, m.args.hidden_size))
        fa, swa = m.model.make_masks(h, cache)
        kp = mx.array(np.arange(pos, pos + L, dtype=np.float32))[None, None, :, None] * mx.ones((1, 1, 1, 128))
        res = {}
        for name, idx, mask, win in (("swa", si, swa, W), ("fa", fi, fa, None)):
            k, v = cache[idx].update_and_fetch(kp, kp)
            kpos = np.array(k[0, 0, :, 0]).astype(np.int64)
            allowed = visible(mask, kpos, L)
            for r in range(L):
                i = pos + r
                got = np.sort(kpos[allowed[r]])
                lo = 0 if win is None else max(0, i - win + 1)
                exp = np.arange(lo, i + 1)
                if got.shape != exp.shape or (got != exp).any():
                    bad.append((name, i, L, len(got), len(exp)))
        # other layers of the same kind share the logic; keep caches in sync
        for j, c in enumerate(cache):
            if j not in (si, fi):
                c.update_and_fetch(kp, kp)
        pos += L
    print(f"{tag}: T={T} chunks={sum(1 for _ in schedule(T))} bad={len(bad)}", bad[:5], flush=True)
def fixed(step):
    return lambda T: [min(step, T - a) for a in range(0, T, step)]
def pre_then_decode(step, lo):
    return lambda T: [min(step, lo - a) for a in range(0, lo, step)] + [1] * (T - lo)
T = 16384 + 37
for step in (64, 256, 512, 513, 514, 2048):
    run(fixed(step), T, f"prefill {step}")
run(pre_then_decode(2048, 15000), 15400, "2048 then decode")
run(pre_then_decode(64, 15000), 15400, "64 then decode")
run(lambda T: [1] * T, 3000, "decode from 0")
rng = np.random.default_rng(3)
sched = list(rng.integers(1, 700, size=60)); run(lambda T: sched, sum(sched), "random chunk sizes incl 1")
