"""Same short prompts across partner lengths; KL(single||batched) at t=0 (median and mean),
and accuracy of each path against the fp32 truth. Tiny w513 head_dim 128, bf16."""
import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
exec(open("t0probe.py").read().split("out = {")[0])
mf, _, _ = common.load_port(Path(CK)); mf.set_dtype(mx.float32); mx.eval(mf.parameters())
def single_f32(p):
    cache = mf.make_cache(); ids = np.asarray(p, dtype=np.int32)
    mx.eval(mf(mx.array(ids[:-1])[None], cache=cache))
    o = mf(mx.array([[int(ids[-1])]], dtype=mx.int32), cache=cache)[0, -1]; mx.eval(o); return np.array(o)
N = int(sys.argv[3])
PLS = [int(x) for x in sys.argv[4].split(",")]
rp = np.random.default_rng(11); rq = np.random.default_rng(12)
prompts_ = [rp.integers(0, V - 16, size=int(rp.integers(20, 121))).tolist() for _ in range(N)]
partner = rq.integers(0, V - 16, size=max(PLS)).tolist()
tru = [lp(single_f32(p)[None])[0] for p in prompts_]
sng = [lp(single_t0(p)[0][None])[0] for p in prompts_]
ts = np.array([float(kl(t[None], s[None])[0]) for t, s in zip(tru, sng)])
print(f"truth-single: mean {ts.mean():.2e} median {np.median(ts):.2e}")
for PL in PLS:
    sb, tb = [], []
    for p, t, s in zip(prompts_, tru, sng):
        b = lp(batched_t0([p, partner[:PL]], 2)[0][None])[0]
        sb.append(float(kl(s[None], b[None])[0])); tb.append(float(kl(t[None], b[None])[0]))
    sb, tb = np.array(sb), np.array(tb)
    print(f"partner {PL:5d}: KL(single||batched) median {np.median(sb):.2e} mean {sb.mean():.2e} | truth-batched mean {tb.mean():.2e} median {np.median(tb):.2e} | batched further: {np.mean(tb > ts):.2f}", flush=True)
