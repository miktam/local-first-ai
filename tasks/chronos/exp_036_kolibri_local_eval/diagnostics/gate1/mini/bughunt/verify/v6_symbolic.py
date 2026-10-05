"""Exact visibility check of the port's caches + masks: keys carry their absolute position; for every query
the visible set must be exactly [i-512, i] (sliding) / [0, i] (full), for chunk schedules up to 16.4k."""
from common_v import *
from pathlib import Path
from gate import common
V = Path(__file__).resolve().parent
m, _, mod = common.load_port(V / "ckv_s1")
W = m.args.sliding_window
def allowed_of(mask, qL, kL):
    if mask is None: return np.ones((qL, kL), bool)
    if isinstance(mask, str):
        assert mask == "causal"; return np.arange(kL)[None] <= (kL - qL) + np.arange(qL)[:, None]
    a = np.array(mask); return np.broadcast_to(a.reshape(a.shape[-2:]), (qL, kL))
def run(sched, tag):
    cache = m.make_cache(); si, fi = m.model.swa_idx, m.model.fa_idx; pos = 0; bad = 0; nq = 0
    for L in sched:
        h = mx.zeros((1, L, m.args.hidden_size)); fa, swa = m.model.make_masks(h, cache)
        kp = mx.array(np.arange(pos, pos + L, dtype=np.float32))[None, None, :, None] * mx.ones((1, 1, 1, 8))
        for idx, mask, win in ((si, swa, W), (fi, fa, None)):
            k, _ = cache[idx].update_and_fetch(kp, kp); kpos = np.array(k[0, 0, :, 0]).astype(np.int64)
            al = allowed_of(mask, L, kpos.size)
            for r in range(L):
                i = pos + r; got = np.sort(kpos[al[r]]); lo = 0 if win is None else max(0, i - win + 1)
                if got.size != i + 1 - lo or (got != np.arange(lo, i + 1)).any(): bad += 1
                nq += 1
        for j, c in enumerate(cache):
            if j not in (si, fi): c.update_and_fetch(kp, kp)
        pos += L
    print(f"{tag:40s} tokens {pos:6d} queries checked {nq:6d} wrong visible sets {bad}", flush=True)
fixed = lambda s, T: [min(s, T - a) for a in range(0, T, s)]
T = 16421
for s in (64, 2048, 513, 514):
    run(fixed(s, T), f"prefill chunk {s}")
run(fixed(2048, 15000) + [1] * 301, "gate decode_rows: 2048 to 15000, then 1x301")
run(fixed(64, 15000) + [1] * 301, "64 to 15000, then 1x301")
rng = np.random.default_rng(9); sch = []
while sum(sch) < 17000: sch += [int(rng.choice([1, 2, 63, 64, 65, 511, 512, 513, 514, 1025, 2048]))] * int(rng.integers(1, 4))
run(sch, f"mixed transitions ({len(sch)} chunks)")
