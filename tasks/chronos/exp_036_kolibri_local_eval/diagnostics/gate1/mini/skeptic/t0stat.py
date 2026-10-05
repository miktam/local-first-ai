"""Is the right-padded batched t=0 step LESS ACCURATE than the single path, or just
different? Truth = the same tiny checkpoint in fp32 (single prefill+decode).
N random short prompts (len 20..120), each batched with a 300-token partner."""
import os, sys, json
os.environ.setdefault("MLX_ENABLE_TF32", "0")
exec(open("t0probe.py").read().split("out = {")[0])
mf, _, _ = common.load_port(Path(CK)); mf.set_dtype(mx.float32); mx.eval(mf.parameters())
def single_f32(p):
    cache = mf.make_cache(); ids = np.asarray(p, dtype=np.int32)
    mx.eval(mf(mx.array(ids[:-1])[None], cache=cache))
    o = mf(mx.array([[int(ids[-1])]], dtype=mx.int32), cache=cache)[0, -1]; mx.eval(o); return np.array(o)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 120
partner_len = int(sys.argv[4]) if len(sys.argv) > 4 else 300
r2 = np.random.default_rng(5)
rows = []
for i in range(N):
    n = int(r2.integers(20, 121))
    p = r2.integers(0, V - 16, size=n).tolist()
    q = r2.integers(0, V - 16, size=partner_len).tolist()
    tru = lp(single_f32(p)[None])[0]
    s = lp(single_t0(p)[0][None])[0]
    b = lp(batched_t0([p, q], 2)[0][None])[0]
    rows.append((float(kl(tru[None], s[None])[0]), float(kl(tru[None], b[None])[0]), float(kl(s[None], b[None])[0])))
a = np.array(rows)
print(json.dumps({"N": N, "partner_len": partner_len,
  "mean_KL_truth_single": a[:,0].mean(), "mean_KL_truth_batched": a[:,1].mean(), "mean_KL_single_batched": a[:,2].mean(),
  "median_truth_single": float(np.median(a[:,0])), "median_truth_batched": float(np.median(a[:,1])),
  "frac_batched_further_from_truth": float((a[:,1] > a[:,0]).mean()),
  "n_single_gt_1e-3": int((a[:,0]>1e-3).sum()), "n_batched_gt_1e-3": int((a[:,1]>1e-3).sum())}, indent=1))
