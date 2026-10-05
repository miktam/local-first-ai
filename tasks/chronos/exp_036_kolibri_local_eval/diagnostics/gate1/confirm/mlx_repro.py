"""mlx_repro: a seeded, model-free reproduction of the gate-1 kernel finding (BUGHUNT §6.4).

kprobe3 found that mlx-lm's sorted 8-bit SwitchGLU, at the G5 first wave's prefill shape
(8 sequences right-padded to 1,099 tokens, top-6 of 384 experts), returned outputs with a
relative error of 0.46-0.49 on the M5 Max (macOS 27) but ~0.005 on the M4 Pro (macOS 26).
kprobe3 could not say which op, which trigger, or whether MLX, the OS or the chip is at fault.

This script calls the single op that SwitchGLU's projections call, mx.gather_qmm, on inputs
built exactly as mlx_lm.models.switch_layers.SwitchGLU builds them (rows gathered per
(token, expert), sorted by expert, sorted_indices=True), with:
  - weights and inputs from a seeded numpy generator (bitwise identical on every machine),
    bf16-rounded, then mx.quantize'd (affine, group 64, 8 bits; 4 bits with --bits4);
  - a float64 reference computed per expert from the dequantised weights;
  - the OS, MLX / mlx-lm versions and the GPU architecture recorded in the output.

Tests (each reports the relative L2 error of the rows that matter, against float64):
  A  first wave: 8 rows right-padded to 1,099, pad positions routed to 6 fixed experts
     (as identical pad hidden states are), sorted_indices=True       -> the kprobe3 shape
  A2 the same sorted data with sorted_indices=False                    -> sorted path at fault?
  A3 repeat of A                                                       -> deterministic?
  B  each sequence alone (B = 1), as SwitchGLU would call it          -> the baseline
  C  A with the pad rows' input scaled x1000                           -> do valid rows change?
  D  8 x 1,099, no padding, uniform routing (52,752 rows)             -> size without a hot expert
  E  hot-expert sweep: ~8k uniform rows plus S rows on one expert      -> segment-size trigger?
  F  uniform size sweep up to 105,504 rows                             -> total-size trigger?
  G  A with float32 activations and scales                            -> bf16-only?
  H  mx.argsort of the flattened indices against numpy                  -> is the sort itself right?
  (A4 with --bits4: A at 4 bits.)

Runs in about a minute or two; nothing is downloaded and no model is loaded.
Usage: python mlx_repro.py --out repro_<UTC>.json [--quick] [--bits4]
"""
import os, sys, json, time, platform, argparse
os.environ.setdefault("MLX_ENABLE_TF32", "0")
import numpy as np
import mlx.core as mx

SEED = 37
H, I, E, K, GS = 2560, 512, 384, 6, 64
L = 1099
LENS = [36, 299, 699, 1099, 63, 519, 899, 149]
PAD_EXPERTS = [3, 17, 42, 101, 200, 333]


def env_record():
    info = mx.device_info()
    try:
        import mlx_lm
        mlx_lm_v = mlx_lm.__version__
    except Exception:
        mlx_lm_v = None
    return {"mlx": mx.__version__, "mlx_lm": mlx_lm_v, "device": info.get("device_name"),
            "arch": info.get("architecture"), "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
            "seed": SEED}


def bf16_round(a32):
    return np.array(mx.array(a32).astype(mx.bfloat16).astype(mx.float32))


def make_weights(bits):
    """Seeded per-expert weights [E, I, H], bf16-rounded, quantised; returns MLX parts and float32 dequant."""
    wq, sc, bi, deq = [], [], [], np.empty((E, I, H), dtype=np.float32)
    for e in range(E):
        r = np.random.default_rng([SEED, e])
        w = mx.array(r.standard_normal((I, H), dtype=np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=GS, bits=bits)
        d = mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=GS, bits=bits)  # exact: fp32 scales
        mx.eval(q, s, b, d)
        wq.append(q); sc.append(s); bi.append(b); deq[e] = np.array(d)
    return mx.stack(wq), mx.stack(sc), mx.stack(bi), deq


def reference(xs64, idx, deq):
    out = np.zeros((xs64.shape[0], I), dtype=np.float64)
    for e in np.unique(idx):
        r = np.nonzero(idx == e)[0]
        out[r] = xs64[r] @ deq[e].astype(np.float64).T
    return out


def call(xs, idx, W, sorted_flag, dtype=mx.bfloat16, bits=8):
    q, s, b = W
    if dtype == mx.float32:
        s, b = s.astype(mx.float32), b.astype(mx.float32)
    x = mx.array(xs).astype(dtype)[:, None, :]
    y = mx.gather_qmm(x, q, s, b, rhs_indices=mx.array(idx.astype(np.uint32)), transpose=True,
                      group_size=GS, bits=bits, mode="affine", sorted_indices=sorted_flag)
    mx.eval(y)
    return np.array(y.astype(mx.float32))[:, 0, :].astype(np.float64)


def rel(a, ref):
    return float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-300))


def flatten_sorted(tok_x, tok_idx):
    """As SwitchGLU: rows per (token, expert), stably sorted by expert. Returns xs, idx, token id, slot."""
    T = tok_idx.shape[0]
    flat = tok_idx.reshape(-1)
    order = np.argsort(flat, kind="stable")
    return tok_x[order // K], flat[order], order // K, order


def route_uniform(rng, T):
    return np.argsort(rng.random((T, E)), axis=1)[:, :K].astype(np.int64)


def first_wave(rng, lens, pad_scale=1.0):
    """8 sequences right-padded to L; returns token-level x [8*L, H], idx [8*L, K], valid mask, seq id."""
    B = len(lens)
    x = rng.standard_normal((B * L, H), dtype=np.float32)
    idx = route_uniform(rng, B * L)
    valid = np.zeros(B * L, dtype=bool); seq = np.repeat(np.arange(B), L)
    for i, n in enumerate(lens):
        valid[i * L:i * L + n] = True
    pad = ~valid
    idx[pad] = np.array(PAD_EXPERTS)
    x[pad] *= pad_scale
    return bf16_round(x), idx, valid, seq


def summarise(y, ref, rows_tok, valid_tok, seq_tok, idx):
    """Per-sequence and per-expert-class errors over valid rows."""
    v = valid_tok[rows_tok]
    out = {"rel_valid": rel(y[v], ref[v]), "max_abs_valid": float(np.abs(y[v] - ref[v]).max())}
    if seq_tok is not None:
        out["rel_by_seq"] = [rel(y[v & (seq_tok[rows_tok] == s)], ref[v & (seq_tok[rows_tok] == s)])
                             for s in range(int(seq_tok.max()) + 1)]
    pad_e = np.isin(idx, PAD_EXPERTS)
    if (v & pad_e).any():
        out["rel_valid_rows_on_pad_experts"] = rel(y[v & pad_e], ref[v & pad_e])
    if (v & ~pad_e).any():
        out["rel_valid_rows_on_other_experts"] = rel(y[v & ~pad_e], ref[v & ~pad_e])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="fewer sweep points")
    ap.add_argument("--bits4", action="store_true", help="also run test A at 4 bits")
    a = ap.parse_args()
    t0 = time.time()
    res = {"what": "mlx_repro: seeded single-op gather_qmm reproduction (exp_036 gate-1 follow-up; decides nothing)",
           "env": env_record(), "shapes": {"H": H, "I": I, "E": E, "K": K, "group_size": GS, "L": L, "lens": LENS,
                                           "pad_experts": PAD_EXPERTS}, "tests": {}}
    W8 = make_weights(8)
    deq = W8[3]; Wq = W8[:3]
    T = res["tests"]

    # A, A2, A3: first wave
    rng = np.random.default_rng([SEED, 1000])
    tx, tidx, valid, seq = first_wave(rng, LENS)
    xs, idx, tok, _ = flatten_sorted(tx, tidx)
    ref = reference(xs.astype(np.float64), idx, deq)
    yA = call(xs, idx, Wq, True)
    T["A_first_wave_sorted"] = {"rows": int(xs.shape[0]), **summarise(yA, ref, tok, valid, seq, idx)}
    yA2 = call(xs, idx, Wq, False)
    T["A2_first_wave_unsorted_flag"] = {"rows": int(xs.shape[0]), **summarise(yA2, ref, tok, valid, seq, idx)}
    yA3 = call(xs, idx, Wq, True)
    T["A3_repeat_bitwise_equal_to_A"] = bool(np.array_equal(yA, yA3))

    # B: each sequence alone, as SwitchGLU calls it (sorted iff >= 64 (token, expert) rows)
    perseq = []
    for s, n in enumerate(LENS):
        m = (seq == s) & valid
        bx, bidx, btok, _ = flatten_sorted(tx[m], tidx[m])
        bref = reference(bx.astype(np.float64), bidx, deq)
        yb = call(bx, bidx, Wq, bidx.size >= 64)
        perseq.append({"n_tokens": int(n), "rows": int(bx.shape[0]), "rel": rel(yb, bref)})
    T["B_each_sequence_alone"] = perseq

    # C: pad rows' input x1000; same routing and valid inputs
    xs_c = xs.copy(); padrow = ~valid[tok]
    xs_c[padrow] = bf16_round(xs_c[padrow] * 1000.0)
    yC = call(xs_c, idx, Wq, True)
    v = valid[tok]
    T["C_pad_scaled_x1000"] = {"valid_rows_bitwise_unchanged": bool(np.array_equal(yC[v], yA[v])),
                               "rel_valid": rel(yC[v], ref[v])}

    # D: 8 x 1,099 with no padding
    rng = np.random.default_rng([SEED, 2000])
    dx = bf16_round(rng.standard_normal((8 * L, H), dtype=np.float32)); didx = route_uniform(rng, 8 * L)
    xs_d, idx_d, tok_d, _ = flatten_sorted(dx, didx)
    ref_d = reference(xs_d.astype(np.float64), idx_d, deq)
    T["D_no_padding_52752_rows"] = {"rows": int(xs_d.shape[0]), "rel": rel(call(xs_d, idx_d, Wq, True), ref_d),
                                    "max_rows_per_expert": int(np.bincount(idx_d, minlength=E).max())}

    # E: hot-expert sweep (uniform base plus S rows on expert 3)
    rng = np.random.default_rng([SEED, 3000])
    base_T = 1400
    bx = bf16_round(rng.standard_normal((base_T, H), dtype=np.float32)); bidx = route_uniform(rng, base_T)
    xs_b, idx_b, _, _ = flatten_sorted(bx, bidx)
    sweep = []
    S_list = [0, 1024, 4096, 8192, 16384, 30174] if a.quick else [0, 256, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 30174]
    for S in S_list:
        hx = bf16_round(rng.standard_normal((S, H), dtype=np.float32))
        X = np.concatenate([xs_b, hx]); IDX = np.concatenate([idx_b, np.full(S, 3)])
        o = np.argsort(IDX, kind="stable"); X, IDX = X[o], IDX[o]
        r_ = reference(X.astype(np.float64), IDX, deq); y = call(X, IDX, Wq, True)
        hot = IDX == 3
        sweep.append({"hot_rows": int(hot.sum()), "total_rows": int(X.shape[0]), "rel_all": rel(y, r_),
                      "rel_hot_expert": rel(y[hot], r_[hot]), "rel_other_experts": rel(y[~hot], r_[~hot])})
    T["E_hot_expert_sweep"] = sweep

    # F: uniform total-size sweep
    rng = np.random.default_rng([SEED, 4000])
    fs = []
    for Tn in ([64, 1100, 8792, 17584] if a.quick else [64, 256, 1100, 2200, 4400, 8792, 13188, 17584]):
        fx = bf16_round(rng.standard_normal((Tn, H), dtype=np.float32)); fidx = route_uniform(rng, Tn)
        X, IDX, _, _ = flatten_sorted(fx, fidx)
        r_ = reference(X.astype(np.float64), IDX, deq)
        fs.append({"tokens": Tn, "rows": int(X.shape[0]), "rel": rel(call(X, IDX, Wq, True), r_),
                   "max_rows_per_expert": int(np.bincount(IDX, minlength=E).max())})
    T["F_uniform_size_sweep"] = fs

    # G: float32 activations and scales, first-wave data
    yG = call(xs, idx, Wq, True, dtype=mx.float32)
    T["G_first_wave_fp32"] = summarise(yG, ref, tok, valid, seq, idx)

    # H: the sort itself
    flat = tidx.reshape(-1)
    order_mx = np.array(mx.argsort(mx.array(flat.astype(np.uint32))))
    T["H_mx_argsort"] = {"keys_sorted": bool(np.all(np.diff(flat[order_mx]) >= 0)),
                         "is_permutation": bool(np.array_equal(np.sort(order_mx), np.arange(flat.size))),
                         "keys_equal_numpy_sorted": bool(np.array_equal(flat[order_mx], np.sort(flat)))}

    if a.bits4:
        W4 = make_weights(4)
        ref4 = reference(xs.astype(np.float64), idx, W4[3])
        T["A4_first_wave_sorted_4bit"] = summarise(call(xs, idx, W4[:3], True, bits=4), ref4, tok, valid, seq, idx)
        perseq4 = []
        for s, n in enumerate(LENS):
            m = (seq == s) & valid
            bx4, bidx4, _, _ = flatten_sorted(tx[m], tidx[m])
            perseq4.append(rel(call(bx4, bidx4, W4[:3], bidx4.size >= 64, bits=4), reference(bx4.astype(np.float64), bidx4, W4[3])))
        T["B4_each_sequence_alone_4bit"] = perseq4

    res["seconds"] = round(time.time() - t0, 1)
    flags = []
    if T["A_first_wave_sorted"]["rel_valid"] > 10 * max(p["rel"] for p in perseq):
        flags.append("A: batched first wave is >10x worse than the same rows alone")
    if not T["C_pad_scaled_x1000"]["valid_rows_bitwise_unchanged"]:
        flags.append("C: valid rows depend on pad rows' content")
    if not T["A3_repeat_bitwise_equal_to_A"]:
        flags.append("A3: not deterministic")
    res["flags"] = flags
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"env": res["env"], "flags": flags,
                      "A_rel_valid": T["A_first_wave_sorted"]["rel_valid"],
                      "A2_rel_valid": T["A2_first_wave_unsorted_flag"]["rel_valid"],
                      "B_rel_max": max(p["rel"] for p in perseq),
                      "D_rel": T["D_no_padding_52752_rows"]["rel"],
                      "E": [(e["hot_rows"], round(e["rel_hot_expert"], 4), round(e["rel_other_experts"], 4)) for e in sweep],
                      "F": [(e["rows"], round(e["rel"], 4)) for e in fs],
                      "G_rel_valid": T["G_first_wave_fp32"]["rel_valid"], "seconds": res["seconds"]}, indent=1))


if __name__ == "__main__":
    main()
