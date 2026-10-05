"""Identical-prompt localiser done right: B=8 copies of one prompt vs B=1 through BatchGenerator,
compared only up to the first generated-token divergence (free-running greedy)."""
import sys, json
sys.argv = [sys.argv[0], sys.argv[1], "loc", sys.argv[2], sys.argv[3]]
src = open("e2_batch_null.py").read().split("t0 = time.time()")[0]
src = src.replace("truth, _ = get(True)", "truth = None")
exec(src)
out = {}
for pi in (2, 3, 0):
    p = prompts[pi]
    r8 = batched(model, [p] * 8, [24] * 8, 8)
    r1 = batched(model, [p], [24], 1)
    t1 = r1[0]["tokens"]; t8 = r8[0]["tokens"]
    div = next((i for i in range(min(len(t1), len(t8))) if t1[i] != t8[i]), min(len(t1), len(t8)))
    n = div + 1 if div < min(len(t1), len(t8)) else div  # include the divergence step's distribution
    k = kl(lp(np.stack(r1[0]["lp"][:n])), lp(np.stack(r8[0]["lp"][:n])))
    out[f"prompt{pi}_len{len(p)}"] = {"first_divergence_step": div, "kl_until_div": [float(f"{x:.3e}") for x in k],
                                      "rows_equal": bool(all(r["tokens"] == t8 for r in r8))}
print(json.dumps(out, indent=1))
