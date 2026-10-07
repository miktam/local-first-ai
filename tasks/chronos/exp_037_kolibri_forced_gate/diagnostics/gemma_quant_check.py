"""Is the G8 (gemma-4-26b-a4b-it-8bit) build or MLX's 8-bit path faulty? (exp_036, 2026-10-04)

Diagnostic only: not part of the pre-registered kit, in no hash scope, scores nothing.
Asked by Andrei after the step-9 re-run (aborted/20261004T175538Z-peercheck/NOTE.md):
chat-wrapped NLL(8) 2.938 > NLL(4) 2.910 and KL(8||4) 0.323 for Gemma 4, while Qwen's
8-bit beats its 4-bit and KL(8||4) is 0.07-0.10. The two mlx-community configs and
safetensors headers are structurally identical (same 1,697 tensors, BF16 + U32, group 64,
router 8-bit in both), so this checks values and kernels:

  A. kernels: quantized_matmul and sorted gather_qmm at Gemma 4's shapes (K and N in
     2816, 704, 2112, 4096, 8192; bits 4 and 8; M 1..256) against dequantize + fp32
     matmul, on this GPU. A healthy kernel's error is the bf16 activation rounding
     (~1e-3 relative) at both bit widths.
  B. weights: for every quantised tensor, d84 = |deq(G8) - deq(G4)| / |deq(G8)| against
     e4 = |deq(Q4(deq(G8))) - deq(G8)| / |deq(G8)|, the 4-bit error of G8's own values.
     If both builds come from the same bf16 source, d84 / e4 is about 1 everywhere; a
     tensor far from 1 is a corrupted or differently sourced tensor in one build.
  C. unquantised tensors (norms etc., BF16 in both): must be bit-identical.

Usage (mbp, venv, nothing else running on the GPU):
  "$PY" diagnostics/gemma_quant_check.py --g8 "$EXP036_MODELS/gemma-4-26b-a4b-it-8bit" \
      --g4 "$EXP036_MODELS/gemma-4-26b-a4b-it-4bit" --out "$EXP036_WORK/gemma_quant_check.json"
Prints a summary; writes the full table to --out (no model text, only tensor names and numbers).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def kernels(bits_list=(4, 8), group=64) -> list[dict]:
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(704)
    rows = []
    shapes = [(2816, 704), (704, 2816), (2816, 2112), (2112, 2816), (2816, 4096), (4096, 2816),
              (2816, 8192), (8192, 2816), (2560, 512), (512, 2560)]   # last two: Kolibri-like control
    for K, N in shapes:
        w = mx.array(rng.standard_normal((N, K)).astype(np.float32) * 0.02)
        for bits in bits_list:
            wq, s, b = mx.quantize(w, group_size=group, bits=bits)
            wd = mx.dequantize(wq, s, b, group_size=group, bits=bits).astype(mx.float32)
            for M in (1, 7, 64, 256):
                x = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
                ref = x.astype(mx.float32) @ wd.T
                out = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=group, bits=bits).astype(mx.float32)
                rel = float(mx.linalg.norm(out - ref) / mx.linalg.norm(ref))
                rows.append({"op": "quantized_matmul", "K": K, "N": N, "M": M, "bits": bits, "rel_err": rel})
            # sorted gather_qmm over 8 "experts" (the SwitchGLU path when T*k >= 64)
            E = 8
            we = mx.stack([w] * E)
            weq, se, be = mx.quantize(we, group_size=group, bits=bits)
            wed = mx.dequantize(weq, se, be, group_size=group, bits=bits).astype(mx.float32)
            M = 128
            # As mlx_lm switch_layers: x [M, 1, 1, K] (expand_dims(x, (-2, -3))), indices [M, 1] sorted.
            xs = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
            x = mx.expand_dims(xs, (-2, -3))
            idx_np = np.sort(rng.integers(0, E, size=M)).astype(np.uint32).reshape(M, 1)
            idx = mx.array(idx_np, dtype=mx.uint32)
            out = mx.gather_qmm(x, weq, se, be, rhs_indices=idx, transpose=True, group_size=group, bits=bits,
                                sorted_indices=True).astype(mx.float32).reshape(M, N)
            ref = mx.stack([xs[i].astype(mx.float32) @ wed[int(idx_np[i, 0])].T for i in range(M)])
            rel = float(mx.linalg.norm(out - ref) / mx.linalg.norm(ref))
            rows.append({"op": "gather_qmm_sorted", "K": K, "N": N, "M": M, "bits": bits, "rel_err": rel})
    return rows


def load_all(d: Path) -> dict:
    import mlx.core as mx

    out = {}
    for f in sorted(d.glob("*.safetensors")):
        out.update(mx.load(str(f)))
    return out


def bits_for(cfg: dict, base: str) -> int:
    q = cfg.get("quantization") or {}
    ov = q.get(base)
    return int(ov["bits"]) if isinstance(ov, dict) else int(q["bits"])


def weights(g8: Path, g4: Path, group=64) -> tuple[list[dict], list[dict]]:
    import mlx.core as mx

    c8 = json.loads((g8 / "config.json").read_text())
    c4 = json.loads((g4 / "config.json").read_text())
    a, b = load_all(g8), load_all(g4)
    qrows, frows = [], []
    for k in sorted(a):
        if k.endswith((".scales", ".biases")):
            continue
        if k.endswith(".weight") and (k[:-7] + ".scales") in a:
            base = k[:-7]
            b8, b4 = bits_for(c8, base), bits_for(c4, base)
            w8 = mx.dequantize(a[k], a[base + ".scales"], a[base + ".biases"], group_size=group, bits=b8).astype(mx.float32)
            w4 = mx.dequantize(b[k], b[base + ".scales"], b[base + ".biases"], group_size=group, bits=b4).astype(mx.float32)
            n8 = mx.linalg.norm(w8)
            d84 = float(mx.linalg.norm(w8 - w4) / n8)
            if b4 == 4:
                q = mx.quantize(w8.astype(mx.bfloat16), group_size=group, bits=4)
                e4 = float(mx.linalg.norm(mx.dequantize(*q, group_size=group, bits=4).astype(mx.float32) - w8) / n8)
            else:
                e4 = None
            qrows.append({"tensor": base, "bits8": b8, "bits4": b4, "d84": d84, "e4": e4,
                          "ratio": (d84 / e4) if e4 else None})
            mx.clear_cache()
        else:
            same = bool(mx.array_equal(a[k], b[k])) if k in b else False
            frows.append({"tensor": k, "dtype": str(a[k].dtype), "identical": same})
    return qrows, frows


def reference_kl(bf16: Path, g8: Path, g4: Path, texts: list[Path]) -> list[dict]:
    """D. Chat-wrapped (Amendment 5 wrapper) NLL of bf16, G8 and G4 on the gate texts, and
    KL(bf16 || G8), KL(bf16 || G4) per token. A healthy 8-bit build sits far closer to bf16
    than the 4-bit one. bf16 stays loaded; G8 and G4 are loaded one at a time."""
    import gc

    import mlx.core as mx
    from mlx_lm import load
    from bench import kl_8v4 as K

    def lp_rows(model, seq, start):
        rows = [K.logprobs(lg, round_bf16=False) for _, lg, _ in K.scored_chunks(model, seq, score_from=start)]
        return mx.concatenate(rows, axis=0)

    ref, _ = load(str(bf16))
    out = []
    cache = {}
    for t in texts:
        body = t.read_text(encoding="utf-8")
        w = K.chat_wrapper("gemma4", bf16)
        ids, _ = K.tokenize(bf16 / "tokenizer.json", body)
        seq = w["prompt_ids"] + ids
        start = len(w["prompt_ids"]) - 1
        lp = lp_rows(ref, seq[:-1], start)
        mx.eval(lp)
        cache[t.name] = (seq, start, lp)
    for name, d in (("G8", g8), ("G4", g4)):
        m, _ = load(str(d))
        for t in texts:
            seq, start, lpr = cache[t.name]
            lpm = lp_rows(m, seq[:-1], start)
            tgt = mx.array(seq[start + 1:])
            nll_ref = float(-mx.mean(mx.take_along_axis(lpr, tgt[:, None], axis=-1)))
            nll_m = float(-mx.mean(mx.take_along_axis(lpm, tgt[:, None], axis=-1)))
            kl = float(mx.mean(K.kl_rows(lpr, lpm)))
            out.append({"text": t.name, "model": name, "nll_bf16": nll_ref, "nll_model": nll_m, "kl_bf16_model": kl,
                        "n_tokens": int(tgt.shape[0])})
        del m
        gc.collect()
        mx.clear_cache()
    return out


def main(argv=None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--g8", required=True, type=Path)
    ap.add_argument("--g4", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--skip-weights", action="store_true", help="kernel check only (no model files needed)")
    ap.add_argument("--bf16", type=Path, help="D. optional: an upstream bf16 MLX build (mlx-community/gemma-4-26b-a4b-it-bf16)")
    args = ap.parse_args(argv)

    import mlx.core as mx
    rec = {"device": mx.device_info().get("device_name"), "mlx": mx.__version__}
    rec["kernels"] = kernels()
    worst = {}
    for r in rec["kernels"]:
        key = (r["op"], r["bits"])
        worst[key] = max(worst.get(key, 0.0), r["rel_err"])
    print("A. kernels, worst relative error per op and bit width:")
    for (op, bits), v in sorted(worst.items()):
        print(f"   {op:18s} {bits}-bit  {v:.2e}")
    gem = [r for r in rec["kernels"] if r["K"] in (2816, 704, 2112) or r["N"] in (2816, 704, 2112)]
    ctl = [r for r in rec["kernels"] if r["K"] in (2560, 512)]
    print(f"   Gemma shapes worst {max(r['rel_err'] for r in gem):.2e}; Kolibri-like control worst {max(r['rel_err'] for r in ctl):.2e}")

    if not args.skip_weights:
        qrows, frows = weights(args.g8, args.g4)
        rec["quantised"], rec["unquantised"] = qrows, frows
        rat = [r for r in qrows if r["ratio"] is not None]
        rat.sort(key=lambda r: r["ratio"])
        print(f"B. quantised tensors: {len(qrows)}; d84/e4 median {rat[len(rat)//2]['ratio']:.3f}, "
              f"min {rat[0]['ratio']:.3f} ({rat[0]['tensor']}), max {rat[-1]['ratio']:.3f} ({rat[-1]['tensor']})")
        out = [r for r in rat if not 0.7 <= r["ratio"] <= 1.4]
        print(f"   outside [0.7, 1.4]: {len(out)}")
        for r in out[:15]:
            print(f"     {r['tensor']}: d84 {r['d84']:.3e} e4 {r['e4']:.3e} ratio {r['ratio']:.2f}")
        r8 = [r for r in qrows if r["ratio"] is None]
        if r8:
            print(f"   8-bit in both (router): max d84 {max(r['d84'] for r in r8):.2e} over {len(r8)}")
        diff = [r for r in frows if not r["identical"]]
        print(f"C. unquantised tensors: {len(frows)}, not bit-identical: {len(diff)}")
        for r in diff[:15]:
            print(f"     {r['tensor']} ({r['dtype']})")
    if args.bf16:
        exp = Path(__file__).resolve().parents[1]
        texts = [p for p in sorted((exp / "gate" / "texts").glob("T[1-4]_*.txt"))]
        rec["reference_kl"] = rows = reference_kl(args.bf16, args.g8, args.g4, texts)
        print("D. chat-wrapped against bf16 (per text: NLL bf16 / NLL model, KL(bf16||model)):")
        for r in rows:
            print(f"   {r['model']} {r['text']:32s} {r['nll_bf16']:.3f} / {r['nll_model']:.3f}   KL {r['kl_bf16_model']:.4f}")
        for name in ("G8", "G4"):
            rs = [r for r in rows if r["model"] == name]
            n = sum(r["n_tokens"] for r in rs)
            print(f"   {name}: token-weighted KL(bf16||{name}) {sum(r['kl_bf16_model'] * r['n_tokens'] for r in rs) / n:.4f}, "
                  f"NLL {sum(r['nll_model'] * r['n_tokens'] for r in rs) / n:.3f} vs bf16 "
                  f"{sum(r['nll_bf16'] * r['n_tokens'] for r in rs) / n:.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=1))
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
