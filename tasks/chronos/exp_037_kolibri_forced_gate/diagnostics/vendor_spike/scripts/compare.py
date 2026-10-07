"""Compare vendor-code outputs against the fp32 reference (W13 spike).

For each (checkpoint, sequence), per decoder layer l (residual stream after
layer l), the final-norm output and the logits:
  rel_l2       ||vendor - ref||_2 / ||ref||_2 over the whole [T, *] tensor
               (the §10.5 statistic; mismatch if > 1e-4)
  rel_l2_row   max over positions of the per-position relative L2 error
  rel_max      max|d| / max|ref| (exp_036 G1's fp32 logits estimator)
plus argmax agreement of the logits.
Usage: compare.py <V> <prefix> <ck,ck,...>  -> out/compare_<prefix>.json
"""
import json
import sys

import numpy as np

V, prefix, cks = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
BOUND = 1e-4


def stats(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = a - b
    rows = np.linalg.norm(d.reshape(d.shape[0], -1), axis=1) / np.maximum(
        np.linalg.norm(b.reshape(b.shape[0], -1), axis=1), 1e-300)
    return {
        "rel_l2": float(np.linalg.norm(d) / np.linalg.norm(b)),
        "rel_l2_row": float(rows.max()),
        "rel_l2_row_argmax_pos": int(rows.argmax()),
        "rel_max": float(np.abs(d).max() / np.abs(b).max()),
    }


res = {"bound": BOUND, "cases": []}
worst = {"rel_l2": 0.0, "rel_l2_row": 0.0, "rel_max": 0.0}
for ck in cks:
    for name in ("ids64", "ids600"):
        r = np.load(f"{V}/out/ref_{ck}_{name}.npz")
        g = np.load(f"{V}/out/{prefix}_{ck}_{name}.npz")
        assert (r["ids"] == g["ids"]).all()
        case = {"ck": ck, "seq": name, "T": int(r["ids"].size), "layers": []}
        for l in range(r["hidden"].shape[0]):
            s = stats(g["hidden"][l], r["hidden"][l])
            case["layers"].append(s)
        case["final_norm"] = stats(g["final"], r["final"])
        case["logits"] = stats(g["logits"], r["logits"])
        case["argmax_agree"] = float((g["logits"].argmax(-1) == r["logits"].argmax(-1)).mean())
        all_s = case["layers"] + [case["final_norm"], case["logits"]]
        case["max"] = {k: max(s[k] for s in all_s) for k in worst}
        case["mismatch"] = any(s["rel_l2"] > BOUND for s in case["layers"] + [case["logits"]])
        for k in worst:
            worst[k] = max(worst[k], case["max"][k])
        res["cases"].append(case)
        print(f"{ck:10s} {name:6s} max rel_l2 {case['max']['rel_l2']:.2e}  max row rel_l2 {case['max']['rel_l2_row']:.2e}  "
              f"max rel_max {case['max']['rel_max']:.2e}  logits rel_l2 {case['logits']['rel_l2']:.2e}  "
              f"argmax {case['argmax_agree']:.4f}  {'MISMATCH' if case['mismatch'] else 'ok'}")
        print("   per-layer rel_l2:", " ".join(f"{s['rel_l2']:.1e}" for s in case["layers"]))
res["worst"] = worst
res["any_mismatch"] = any(c["mismatch"] for c in res["cases"])
json.dump(res, open(f"{V}/out/compare_{prefix}.json", "w"), indent=1)
print("WORST", json.dumps(worst), "ANY_MISMATCH", res["any_mismatch"])
