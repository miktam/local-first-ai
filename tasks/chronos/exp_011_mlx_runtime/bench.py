#!/usr/bin/env python3
"""
Exp 011 — MLX runtime vs Ollama: prefill latency and generation throughput
Mac Mini M4 Pro · gemma-4-26B-A4B-it-OptiQ-4bit

Usage:
  python3 bench.py --phase A --machine mini
  python3 bench.py --phase B --machine mini --baseline 1.774
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_ID = "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"
NUM_PREDICT = 128
IDLE_SECONDS = 60

SIZES_A = [4_000, 8_000, 15_000, 25_000, 35_000]
SIZES_B = [20_000, 22_500, 25_000, 27_500, 30_000, 32_500, 35_000, 37_500, 40_000]
REPS_A, REPS_B = 3, 2

BASELINES = {
    "exp008_FA0_fp16": {"4k": 1.557, "8k": 1.642, "15k": 1.774, "25k": 1.983, "35k": 2.241,
                        "gen_35k": 26.69},
    "exp010_FA0_q8":   {"4k": 1.497, "8k": 1.570, "15k": 1.694, "25k": 1.859, "35k": 2.087,
                        "gen_35k": 27.90},
}

FIXTURE_DIR = Path(__file__).parent.parent / "exp_007_hardware_comparison" / "fixtures" / "padding"


def load_fixture(target_tokens: int) -> str:
    key = {4000: "4k", 8000: "8k", 15000: "15k", 20000: "20k", 22500: "22500",
           25000: "25k", 27500: "27500", 30000: "30k", 32500: "32500",
           35000: "35k", 37500: "37500", 40000: "40k"}.get(target_tokens)
    if key:
        p = FIXTURE_DIR / f"{key}.txt"
        if p.exists():
            return p.read_text()
    # fallback: repeat a token-dense phrase
    phrase = "The quick brown fox jumps over the lazy dog. " * 20
    return (phrase * (target_tokens // 10))[:target_tokens * 4]


def measure_one(model, tokenizer, prompt_text: str, max_tokens: int = NUM_PREDICT):
    """Return (prompt_tokens, prefill_ms_per_tok, gen_tps)."""
    import mlx.core as mx
    from mlx_lm.utils import generate_step

    tokens = tokenizer.encode(prompt_text)
    n_prompt = len(tokens)
    prompt_mx = mx.array(tokens)

    first_token_t = None
    out_tokens = []
    t_start = time.perf_counter()

    for token, _ in zip(generate_step(prompt_mx, model, temp=0.0), range(max_tokens)):
        if first_token_t is None:
            mx.eval(token)          # force sync for accurate prefill timing
            first_token_t = time.perf_counter()
        out_tokens.append(token)

    t_end = time.perf_counter()

    prefill_s = first_token_t - t_start
    gen_s = t_end - first_token_t
    prefill_ms_per_tok = (prefill_s * 1000) / n_prompt
    gen_tps = len(out_tokens) / gen_s if gen_s > 0 else 0.0

    return n_prompt, prefill_ms_per_tok, gen_tps


def run(phase: str, machine: str, baseline_ms: float | None):
    print(f"\nExp 011 — Phase {phase} [MLX: {MODEL_ID}]")
    print(f"Machine: {machine}  |  Model: gemma-4-26B-A4B")
    if baseline_ms:
        threshold = baseline_ms * 2
        print(f"Cliff threshold: {threshold:.3f} ms/tok  (2 × {baseline_ms:.3f})")

    print("Loading model (first run downloads ~14–16 GB if not cached)...")
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    print("Model loaded.\n")

    sizes = SIZES_A if phase == "A" else SIZES_B
    reps  = REPS_A  if phase == "A" else REPS_B

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evdir = Path(__file__).parent / "evidence" / f"{ts}-phase_{phase.lower()}-{machine}"
    evdir.mkdir(parents=True, exist_ok=True)

    summary_sizes = []

    for target in sizes:
        key = {4000:"4k", 8000:"8k", 15000:"15k", 20000:"20k", 22500:"22500",
               25000:"25k", 27500:"27500", 30000:"30k", 32500:"32500",
               35000:"35k", 37500:"37500", 40000:"40k"}.get(target, str(target))
        text = load_fixture(target)

        print(f"  [{key}] idle {IDLE_SECONDS}s... ", end="", flush=True)
        time.sleep(IDLE_SECONDS)
        print("start")

        rep_data = []
        for r in range(1, reps + 1):
            n_tok, pre_ms, gen_tps = measure_one(model, tokenizer, text)
            cliff = baseline_ms is not None and pre_ms > (baseline_ms * 2)
            print(f"           rep {r}/{reps}  prefill={round(pre_ms,3)} ms/tok"
                  f"  gen={round(gen_tps,2)} t/s")
            rep_data.append({"rep": r, "prompt_tokens": n_tok,
                             "prefill_ms_per_tok": round(pre_ms, 3),
                             "gen_tps": round(gen_tps, 2)})

        rep1 = rep_data[0]
        cliff = baseline_ms is not None and rep1["prefill_ms_per_tok"] > (baseline_ms * 2)
        mean_gen = round(sum(r["gen_tps"] for r in rep_data) / len(rep_data), 2)
        print(f"           → rep1={rep1['prefill_ms_per_tok']} ms/tok"
              f"  mean_gen={mean_gen} t/s"
              f"  cliff={'YES ⚠' if cliff else 'no'}\n")

        record = {
            "experiment": "011", "phase": phase, "machine": machine,
            "runtime": "mlx", "model_id": MODEL_ID,
            "size_key": key, "target_tokens": target,
            "rep1_prefill_ms_per_tok": rep1["prefill_ms_per_tok"],
            "rep1_prompt_tokens": rep1["prompt_tokens"],
            "mean_gen_tps": mean_gen,
            "cliff_triggered": cliff,
            "reps": rep_data,
            "baselines": BASELINES,
        }
        (evdir / f"size_{key}.json").write_text(json.dumps(record, indent=2))

        summary_sizes.append({
            "size_key": key,
            "prompt_tokens": rep1["prompt_tokens"],
            "rep1_prefill_ms_per_tok": rep1["prefill_ms_per_tok"],
            "mean_gen_tps": mean_gen,
            "cliff_triggered": cliff,
        })

    # summary
    summary = {
        "experiment": "011", "phase": phase, "machine": machine,
        "runtime": "mlx", "model_id": MODEL_ID,
        "baseline_ms_per_tok": baseline_ms,
        "cliff_threshold_ms_per_tok": round(baseline_ms * 2, 3) if baseline_ms else None,
        "started": ts, "sizes": summary_sizes, "baselines": BASELINES,
    }
    (evdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== Phase {phase} Summary (Exp 011 MLX) ===")
    print(f"{'Size':<10} {'Tokens':<10} {'Prefill ms/tok':<18} {'Gen t/s':<12} {'Cliff'}")
    for s in summary_sizes:
        print(f"{s['size_key']:<10} {s['prompt_tokens']:<10} "
              f"{s['rep1_prefill_ms_per_tok']:<18} {s['mean_gen_tps']:<12} "
              f"{'YES' if s['cliff_triggered'] else 'no'}")

    if phase == "A":
        p15 = next((s["rep1_prefill_ms_per_tok"] for s in summary_sizes
                    if s["size_key"] == "15k"), None)
        if p15:
            print(f"\nNext: python3 bench.py --phase B --machine {machine} --baseline {p15}")

    print(f"Evidence: {evdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase",    required=True, choices=["A", "B"])
    ap.add_argument("--machine",  default="mini")
    ap.add_argument("--baseline", type=float, default=None,
                    help="15K prefill ms/tok from Phase A (required for Phase B cliff detection)")
    args = ap.parse_args()

    if args.phase == "B" and args.baseline is None:
        ap.error("--baseline required for Phase B")

    run(args.phase, args.machine, args.baseline)
