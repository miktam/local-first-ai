"""exp_036 gate-1 bug-hunt verification on the real K8 (mbp). Investigation only: no gate record, nothing
under results/ or diagnostics/gate1/out, no kit file edited, aggregates only (no token ids, no text).

Reads the stage-3 work files that already exist in $EXP036_WORK/diag_gate1/stage3_g4/ (ids.json,
ref_deqK8_fp32.logits.npy, g4_K8_fp32_vs_ref_deqK8_fp32.npz). K8 is loaded as stage 3 loads it
(fp32 activations on exactly the dequantised K8 weights, diaglib.fp32_on_dequantised_weights).

Part 1 (claim 1, chunk 64 at large offsets): T9 rows of prof.decode_ranges["T9"] (15,000-15,300).
  p2048 free (the gate's prefill_rows) records every layer's expert ids for positions 0..15,300;
  then p64 free (must reproduce stage 3's 0.0523), p64 FORCED to the p2048 ids, decode FORCED.
  Prediction if there is no cache/mask/RoPE/attention defect: forced KL at fp32 noise (<= ~1e-6 per
  position, 0 top-1 changes) while free stays ~0.05 (router-flip amplification).
Part 2 (claim 2, port vs reference): the stage-3 bug test re-run with ONE in-memory change, the
  port's RoPE angle computed as vLLM computes it (inv_freq = 1/theta**(arange(0,D,2)/D) in fp32,
  angle = pos * inv_freq in fp32; vllm rotary_embedding base.py _compute_inv_freq /
  _compute_cos_sin_cache), side by side with the recorded bug test (kit port).

usage (kit root, env sourced):
  caffeinate -i "$PY" /path/to/mbp_verify_rope_chaos.py --out "$EXP036_WORK/bughunt_verify/rope_chaos_$(date -u +%Y%m%dT%H%M%SZ).json" [--parts 12] [--tiny ROOT]
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")
KIT = Path(os.environ.get("EXP036_KIT", Path(__file__).resolve().parents[3])).resolve()  # the kit root
sys.path[:0] = [str(KIT), str(KIT / "diagnostics" / "gate1")]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

import diaglib as D  # noqa: E402
import stage3_g4 as S3  # noqa: E402
from gate import common, run_gate  # noqa: E402
from gate.checks import g5_generation as g5  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--parts", default="12")
ap.add_argument("--tiny", default=None)
args = ap.parse_args()
out_path = Path(args.out).expanduser().resolve()
if KIT in out_path.parents:
    raise SystemExit("refusing to write inside the kit")
if not (KIT / "port" / "kolibri1.py").is_file():
    raise SystemExit(f"run from the kit root (or set EXP036_KIT); {KIT} has no port/kolibri1.py")
ctx = D.gate_context(args.tiny)
ts = run_gate.textset_for(ctx)
prof = ts.profile
scratch = Path(ctx.work_dir) / "diag_gate1" / "stage3_g4"
refp = scratch / "ref_deqK8_fp32.logits.npy"
for f in (refp, scratch / "g4_K8_fp32_vs_ref_deqK8_fp32.npz"):
    if not D.is_done(f):
        raise SystemExit(f"missing stage-3 work file {common.redact_path(f)}")
LEAD = ctx.thresholds["G4"]["K8_backstop_decisive_lead_nats"]
rec = {"what": "bughunt verification: chunk-64 forced routing (claim 1) and vLLM-angle RoPE bug test (claim 2)",
       "investigation_only": True, "precision": PREC, "mlx": mx.__version__,
       "device": mx.device_info().get("device_name"), "tiny": bool(args.tiny)}


class VllmRoPE(nn.Module):
    def __init__(self, dims, base):
        super().__init__()
        exps = np.arange(0, dims, 2, dtype=np.float32) / np.float32(dims)
        inv = np.float32(1.0) / (np.float32(base) ** exps)
        self.dims = dims
        self._freqs = mx.array((np.float32(1.0) / inv).astype(np.float32))

    def __call__(self, x, offset=0):
        return mx.fast.rope(x, self.dims, traditional=False, base=None, scale=1.0, offset=offset, freqs=self._freqs)


def load(variant):
    mod = common.port_module()
    if variant == "rope":
        Base = mod.Attention

        class Attention(Base):
            def __init__(self, a, use_rope):
                super().__init__(a, use_rope)
                if self.rope is not None:
                    self.rope = VllmRoPE(self.head_dim, a.rope_theta)
        mod.Attention = Attention
    d = ctx.arm_dir("K8")
    if not args.tiny:
        from port.convert import check_port_file
        check_port_file(d)
    model, _, _ = common.load_port(d, module=mod)
    model.set_dtype(mx.float32)
    D.fp32_on_dequantised_weights(model)
    mx.eval(model.parameters())
    return model, mod


def kl(p, q):
    return (np.exp(p) * (p - q)).sum(-1)


# ---------------------------------------------------------------- part 1
if "1" in args.parts:
    t0 = time.time()
    model, mod = load("base")
    for i, l in enumerate(model.layers):
        l._bh = i
    ST = {"force": None, "p": 0, "rec": []}

    def call(self, x, mask=None, cache=None):
        L = x.shape[1]
        f = None if ST["force"] is None else ST["force"][self._bh][ST["p"]:ST["p"] + L][None]
        r = self.branches(x, mask, cache, force_ids=f)
        if ST["record"]:
            ST["rec"].append(mx.sort(r[5][0], axis=-1).astype(mx.uint16))
        return r[3]
    mod.DecoderLayer.__call__ = call
    lo, hi = prof.decode_ranges["T9"]
    ids = np.asarray(ts.t9[:hi + 1], dtype=np.int32)

    def run(sched, force=None, record=False):
        ST["force"], ST["record"] = force, record
        cache = model.make_cache()
        rows, sel, p = [], [], 0
        for L in sched:
            ST["rec"], ST["p"] = [], p
            o = model(mx.array(ids[p:p + L])[None], cache=cache)
            if record:
                s = mx.stack(ST["rec"]); mx.eval(s); sel.append(np.array(s))
            if p + L > lo:
                r = o[0, max(lo, p) - p:].astype(mx.float32); mx.eval(r); rows.append(np.array(r))
            else:
                mx.eval(o)
            p += L
        lp = D.lsm(np.concatenate(rows))
        return lp, (np.concatenate(sel, axis=1) if record else None)
    fixed = lambda s, n: [min(s, n - a) for a in range(0, n, s)]
    a, sel = run(fixed(2048, hi + 1), record=True)
    force = mx.array(sel.astype(np.uint32))
    res = {"positions": [int(lo), int(hi)]}
    for name, sched, f in (("p64_free", fixed(64, hi + 1), None), ("p64_forced", fixed(64, hi + 1), force),
                           ("decode_forced", fixed(2048, lo) + [1] * (hi + 1 - lo), force),
                           ("p2048_forced_self", fixed(2048, hi + 1), force)):
        b, _ = run(sched, f)
        k = kl(a, b)
        ch = a.argmax(-1) != b.argmax(-1)
        res[f"p2048_vs_{name}"] = {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "n_gt_1e-6": int((k > 1e-6).sum()),
                                   "top1_changes": int(ch.sum())}
        D.log(f"[verify] part1 p2048 vs {name}: mean KL {k.mean():.3e} max {k.max():.3e} top1 changes {int(ch.sum())}")
    if not args.tiny:
        c64 = common.read_json(scratch / "chunk64.json")["ranges"]["T9"]["pairs"]["prefill2048_vs_prefill64"]
        res["stage3_record_p2048_vs_p64_mean_kl"] = c64["mean_kl"]
        res["p64_free_reproduces_stage3"] = abs(res["p2048_vs_p64_free"]["mean_kl"] - c64["mean_kl"]) <= 1e-9
    res["seconds"] = time.time() - t0
    rec["part1_chunk64_forced"] = res
    del model
    D.Run.release()

# ---------------------------------------------------------------- part 2
if "2" in args.parts:
    t0 = time.time()
    logits, off = S3._ref_npz_rows(refp, ts, prof)
    before, _ = S3.load_cols(scratch / "g4_K8_fp32_vs_ref_deqK8_fp32.npz")
    model, _ = load("rope")
    after = {}
    for tid, ids, _o in S3.texts_and_offsets(ts, prof):
        o = off[tid]
        after[tid] = S3.per_position(model, ids, lambda x, y, o=o: logits[o + x:o + y], prof.prefill_chunk)
        D.log(f"[verify] part2 rope bug test {tid}: mean KL {after[tid]['kl'].mean():.3e} (kit port {before[tid]['kl'].mean():.3e})")
    del model
    D.Run.release()

    class _R:
        tiny = bool(args.tiny)
    docs = S3.doc_boundaries(_R, ts)
    T9 = len(ts.t9)
    pos = np.arange(T9)
    sets = {"T9": np.ones(T9, bool), **{f"{x}-{y}": (pos >= x) & (pos < y) for x, y in prof.t9_buckets}}
    bounds = sorted({b for b in range(prof.prefill_chunk, T9, prof.prefill_chunk)} | {b for b in docs if 0 < b < T9})

    def summary(R):
        cat8 = {c: np.concatenate([R[t][c] for t in D.ORDER8]) for c in S3.COLS}
        s = {"T1-8": S3.summ(cat8, lead=LEAD), **{n: S3.summ(R["T9"], m, LEAD) for n, m in sets.items()}}
        keep = ("mean_kl", "max_kl", "n_decisive", "decisive_miss", "top1_decisive")
        s = {k: {x: v[x] for x in keep} for k, v in s.items()}
        win = []
        for b in bounds:
            w = (pos >= b - 32) & (pos < b + 32)
            win.append({"boundary": int(b), "kind": "document" if b in docs else "chunk",
                        "window_mean_kl": float(R["T9"]["kl"][w].mean()), "window_max_kl": float(R["T9"]["kl"][w].max())})
        s["boundary_window_max"] = max(x["window_mean_kl"] for x in win) if win else None
        s["windows"] = win
        return s
    rec["part2_bug_test"] = {"kit_port_recorded": summary(before), "vllm_angle_rope": summary(after),
                             "frozen_bounds": {"bug_mean_kl_max": 1e-3, "bug_top1_decisive_min": 0.999,
                                               "boundary_window_max": 1e-3},
                             "seconds": time.time() - t0}

D.leak_scan(rec)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
D.log(f"[verify] wrote {common.redact_path(out_path)}")
