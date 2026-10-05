"""Minimal standalone reproduction for an MLX issue report (exp_036 follow-up; decides nothing).

mx.gather_qmm(..., sorted_indices=True) against the same call with sorted_indices=False, which
mlx_repro.py showed matches a float64 reference (rel ~0.002, bf16 output rounding). On the M5 Max
(applegpu_g17s, macOS 27.0, MLX 0.31.2) the sorted call was wrong above roughly 2^15 rows and
correct below; this script sweeps the row count tightly around 32,768, checks determinism, and
repeats at a second shape, at 4 bits and in float16.

Seeded; no model, no download. Usage: python mlx_repro_min.py [--out result.json]
"""
import os, sys, json, platform, argparse
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import numpy as np
import mlx.core as mx


def weights(E, n_out, d_in, bits, dtype, seed=0):
    w = mx.array(np.random.default_rng(seed).standard_normal((E, n_out, d_in), dtype=np.float32) * 0.02).astype(dtype)
    q, s, b = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(q, s, b)
    return q, s, b


def call(W, x, idx, sorted_flag, bits):
    q, s, b = W
    y = mx.gather_qmm(x, q, s, b, rhs_indices=idx, transpose=True, group_size=64, bits=bits,
                      mode="affine", sorted_indices=sorted_flag)
    mx.eval(y)
    return np.array(y.astype(mx.float32))


def case(W, E, d_in, n, bits, dtype, seed=1):
    r = np.random.default_rng([seed, n])
    x = mx.array(r.standard_normal((n, 1, d_in), dtype=np.float32)).astype(dtype)
    idx = mx.array(np.sort(r.integers(0, E, n)).astype(np.uint32))
    a = call(W, x, idx, True, bits)
    a2 = call(W, x, idx, True, bits)
    c = call(W, x, idx, False, bits)
    rel = float(np.linalg.norm(a - c) / max(np.linalg.norm(c), 1e-30))
    bad_rows = int(np.sum(np.linalg.norm((a - c)[:, 0, :], axis=1) > 0.05 * np.linalg.norm(c[:, 0, :], axis=1)))
    first_bad = int(np.argmax(np.linalg.norm((a - c)[:, 0, :], axis=1) > 0.05 * np.linalg.norm(c[:, 0, :], axis=1))) if bad_rows else None
    return {"rows": n, "rel_sorted_vs_unsorted": rel, "rows_off_by_more_than_5pct": bad_rows,
            "first_bad_row": first_bad, "sorted_deterministic": bool(np.array_equal(a, a2))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); a = ap.parse_args()
    info = mx.device_info()
    res = {"env": {"mlx": mx.__version__, "device": info.get("device_name"), "arch": info.get("architecture"),
                   "macos": platform.mac_ver()[0], "python": platform.python_version()}, "runs": []}
    sweep = [16384, 24576, 30000, 32000, 32704, 32760, 32767, 32768, 32769, 32770, 32776, 32832, 33000, 34000, 40000, 49152, 65536]
    configs = [("kolibri_gate_proj", 384, 512, 2560, 8, mx.bfloat16, sweep),
               ("kolibri_down_proj", 384, 2560, 512, 8, mx.bfloat16, [16384, 32768, 32769, 49152]),
               ("kolibri_gate_4bit", 384, 512, 2560, 4, mx.bfloat16, [16384, 32768, 32769, 49152]),
               ("kolibri_gate_fp16", 384, 512, 2560, 8, mx.float16, [16384, 32768, 32769, 49152]),
               ("other_shape_E128", 128, 768, 2048, 8, mx.bfloat16, [16384, 32768, 32769, 49152])]
    for name, E, n_out, d_in, bits, dtype, ns in configs:
        W = weights(E, n_out, d_in, bits, dtype)
        for n in ns:
            r = case(W, E, d_in, n, bits, dtype)
            r.update({"config": name, "E": E, "n_out": n_out, "d_in": d_in, "bits": bits, "dtype": str(dtype)})
            res["runs"].append(r)
            print(f"{name:18s} rows {n:6d}  rel {r['rel_sorted_vs_unsorted']:.4f}  bad rows {r['rows_off_by_more_than_5pct']:6d}"
                  f"  first bad {r['first_bad_row']}  deterministic {r['sorted_deterministic']}", flush=True)
        del W
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print(json.dumps(res["env"]))


if __name__ == "__main__":
    main()
