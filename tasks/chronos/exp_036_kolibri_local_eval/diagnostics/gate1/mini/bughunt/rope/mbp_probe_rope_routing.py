"""PROPOSED real-weights probe (not part of the frozen suite): are gate-1
findings (1) and (2) MoE routing-flip amplification, and how much of (2) is the
port's RoPE angle (mx.fast.rope inv_freq = exp2(-(j/64)*log2 1e4), up to 8 fp32
ulps off vLLM's 1/1e4^(2j/128))?

On K8, fp32 activations on the dequantised weights (as stage 3's bug test),
T9 rows 15000..15300:
  A   port prefill 2048, natural routing (routing recorded for all positions)
  B   port prefill 64, routing FORCED to A's                -> KL(A||B)  [(1) if ~1e-8]
  Bn  port prefill 64, natural                              -> KL(A||Bn) [reproduces 0.0523]
  C   port prefill 2048, RoPE = fast.rope(freqs=1/inv_freq_vLLM), natural
                                                            -> KL(ref||C) vs KL(ref||A) [(2)]
  Cn  port prefill 64, same RoPE, natural                   -> KL(C||Cn) [(1) is RoPE-independent?]
and the bug test over T1-T8 (chunk = profile prefill chunk) with the RoPE of C.

Writes only port-vs-port and port-vs-reference comparisons, to
$EXP036_WORK/diag_gate1/bughunt_rope/ (never the kit).

usage (mbp):   python mbp_probe_rope_routing.py
       (mini): python mbp_probe_rope_routing.py --selftest-tiny CKPT_DIR
"""
import argparse
import json
import sys
import time
from pathlib import Path

KIT = Path("$KIT")
sys.path.insert(0, str(KIT))
sys.path.insert(0, str(KIT / "diagnostics" / "gate1"))
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

LO, HI = 15000, 15300
THETA, D = 10000.0, 128


def inv_freq_vllm(dims=D, theta=THETA):
    # vLLM 0.29 rope_base.py:86-91 (fp32): 1.0 / base ** (arange(0, dim, 2, float) / dim)
    e = np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims)
    return (np.float32(1.0) / (np.float32(theta) ** e)).astype(np.float32)


class VllmFp32RoPE(nn.Module):
    """NeoX RoPE whose angle is fp32(pos * inv_freq_vLLM): mx.fast.rope with
    explicit freqs (kernel inv_freq = 1/freqs). cos/sin within 1.2e-7 of the
    reference rope_neox at positions 0..16384 (r4_rope_fixpaths.py)."""

    def __init__(self, dims=D, theta=THETA):
        super().__init__()
        self.dims = dims
        self._freqs = mx.array((1.0 / inv_freq_vllm(dims, theta).astype(np.float64)).astype(np.float32))

    def __call__(self, x, offset=0):
        return mx.fast.rope(x, self.dims, traditional=False, base=None, scale=1.0, offset=offset, freqs=self._freqs)


def set_rope(model, kind):
    for layer in model.layers:
        a = layer.self_attn
        if getattr(a, "rope", None) is None:
            continue
        if not hasattr(a, "_orig_rope"):
            a._orig_rope = a.rope
        a.rope = a._orig_rope if kind == "mlx" else VllmFp32RoPE(a.head_dim, a._orig_rope.base)


def prefill(model, ids, step, lo, hi, force=None, record=True):
    """Teacher-forced prefill of ids[:hi+1] in chunks of `step` through the
    model's cache. Returns (fp32 logits rows lo..hi, routing ids [layers, T, k]
    or None). force: [layers, T, k] expert ids replacing every selection."""
    ids = np.asarray(ids[:hi + 1], dtype=np.int32)
    T = ids.size
    recs = [[] for _ in model.layers]
    pos = [0] * len(model.layers)
    for i, layer in enumerate(model.layers):
        orig = type(layer.mlp).forward_routed.__get__(layer.mlp)

        def fr(x, force_ids=None, i=i, orig=orig):
            L = x.shape[1]
            if force is not None:
                force_ids = mx.array(force[i][pos[i]:pos[i] + L][None].astype(np.uint32))
            y, lg, sel = orig(x, force_ids)
            if record:
                recs[i].append(sel)
            pos[i] += L
            return y, lg, sel

        layer.mlp.forward_routed = fr
    try:
        cache = model.make_cache()
        rows = []
        for a in range(0, T, step):
            b = min(T, a + step)
            out = model(mx.array(ids[a:b])[None], cache=cache)
            sel = out[0, max(lo, a) - a:b - a].astype(mx.float32) if b > lo else out[0, :0]
            mx.eval(sel, *[r[-1] for r in recs if r])
            if b > lo:
                rows.append(np.array(sel))
            if record:
                for r in recs:  # move to host to keep GPU memory flat
                    r[-1] = np.array(r[-1][0])
    finally:
        for layer in model.layers:
            del layer.mlp.forward_routed
    routing = np.stack([np.concatenate(r, 0) for r in recs]) if record else None
    return np.concatenate(rows, 0), routing


def lsm(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def cmp(p_logits, q_logits, lo):
    p, q = lsm(p_logits), lsm(q_logits)
    k = (np.exp(p) * (p - q)).sum(-1)
    part = np.partition(p, -2, -1)
    lead = part[:, -1] - part[:, -2]
    ch = p.argmax(-1) != q.argmax(-1)
    return {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "argmax_kl_pos": int(lo + k.argmax()),
            "n_gt_1e-6": int((k > 1e-6).sum()), "top1_changes": int(ch.sum()),
            "top1_changes_lead_ge_2": int((ch & (lead >= 2.0)).sum())}


def routing_diff(A, B):
    d = (np.sort(A, -1) != np.sort(B, -1)).any(-1)
    anyl = d.any(0)
    return {"n_layer_positions_flipped": int(d.sum()), "n_positions_any_layer": int(anyl.sum()),
            "first_position": int(np.argmax(anyl)) if anyl.any() else None,
            "per_layer": [int(x) for x in d.sum(1)]}


def probe(model, ids, ref_rows, lo, hi, log=print):
    """ref_rows: reference fp32 logits rows lo..hi (or None)."""
    out = {"range": [lo, hi]}
    t = time.time()
    set_rope(model, "mlx")
    A, RA = prefill(model, ids, 2048, lo, hi)
    log(f"A {time.time() - t:.0f}s")
    B, _ = prefill(model, ids, 64, lo, hi, force=RA, record=False)
    log(f"B {time.time() - t:.0f}s")
    Bn, RBn = prefill(model, ids, 64, lo, hi)
    log(f"Bn {time.time() - t:.0f}s")
    set_rope(model, "vllm_fp32")
    C, RC = prefill(model, ids, 2048, lo, hi)
    Cn, _ = prefill(model, ids, 64, lo, hi, record=False)
    set_rope(model, "mlx")
    log(f"C, Cn {time.time() - t:.0f}s")
    out["A_vs_B_p64_forced_to_A_routing"] = cmp(A, B, lo)
    out["A_vs_Bn_p64_natural"] = cmp(A, Bn, lo)
    out["routing_A_vs_Bn"] = routing_diff(RA, RBn)
    out["C_vs_Cn_vllmrope_p2048_vs_p64"] = cmp(C, Cn, lo)
    out["A_vs_C_mlxrope_vs_vllmrope"] = cmp(A, C, lo)
    out["routing_A_vs_C"] = routing_diff(RA, RC)
    if ref_rows is not None:
        out["ref_vs_A_mlxrope"] = cmp(ref_rows, A, lo)
        out["ref_vs_C_vllmrope"] = cmp(ref_rows, C, lo)
        out["ref_vs_Bn"] = cmp(ref_rows, Bn, lo)
    return out


def main_mbp():
    import diaglib as D
    import stage3_g4 as S3

    ap = D.parser("bughunt_rope", __doc__)
    args = ap.parse_args()
    run = D.Run("bughunt_rope", args, PREC)
    ts = run.textset()
    prof = ts.profile
    refp = run.work_root / "stage3_g4" / "ref_deqK8_fp32.logits.npy"
    logits, off = S3._ref_npz_rows(refp, ts, prof)
    lo, hi = prof.decode_ranges["T9"]
    model, _ = run.load_arm("K8", fp32=True, dequantised_embedding=True)
    o9 = off["T9"]
    res = {"t9": probe(model, ts.t9, np.asarray(logits[o9 + lo:o9 + hi + 1], np.float32), lo, hi, D.log)}
    # bug test T1-T8 with the vLLM-fp32 RoPE (stage 3 per_position, chunk = profile prefill chunk)
    set_rope(model, "vllm_fp32")
    bt = {}
    for tid, ids, _o in S3.texts_and_offsets(ts, prof):
        if tid == "T9":
            continue
        o = off[tid]
        r = S3.per_position(model, ids, lambda a, b, o=o: logits[o + a:o + b], prof.prefill_chunk)
        bt[tid] = float(np.asarray(r["kl"]).mean())
        D.log(f"bug test (vLLM-fp32 RoPE) {tid}: mean KL {bt[tid]:.3e}")
    res["bugtest_T1_8_vllm_fp32_rope_mean_kl_per_text"] = bt
    out = run.work_root / "bughunt_rope" / f"probe_{run.utc}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    D.log(f"wrote {out}")


def main_tiny(ck):
    sys.path.insert(0, str(KIT / "tests"))
    from port_harness import load_port
    from reference import kolibri_ref as KR
    from tiny_checkpoint import random_ids

    ids = np.array(random_ids(HI + 1, seed=3), dtype=np.int32)
    model = load_port(ck, float32=True)
    ref = KR.KolibriReference(str(ck)).forward(ids)[LO:HI + 1]
    print(json.dumps(probe(model, ids, ref, LO, HI), indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--selftest-tiny":
        main_tiny(sys.argv[2])
    else:
        main_mbp()
