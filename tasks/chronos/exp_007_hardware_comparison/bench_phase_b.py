#!/usr/bin/env python3
"""
Exp 007 — Phase B: Prefill cliff localisation.

Fine-grained sweep from 20K to 40K tokens (nine points, 2.5K spacing).
Cliff defined as the smallest N where prefill_ms_per_tok > 2× the 15K baseline.

Usage:
    python3 bench_phase_b.py --machine mini --baseline 8.36
    python3 bench_phase_b.py --machine mbp  --baseline 8.36
    python3 bench_phase_b.py --machine mini --baseline 8.36 --sizes 20k 22500 25k

    The --baseline value comes from Phase A's mean_prefill_ms_per_tok at 15K.
    Check evidence/<ts>-phase_a-<machine>/summary.json.

Pre-conditions:
    - Phase A complete on this machine (to get the 15K baseline)
    - Padding fixtures generated (run generate_padding.py first)
    - Machine on AC power, Wi-Fi off

Evidence written to: evidence/<timestamp>-phase_b-<machine>/
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
IDLE_SECONDS = 60
NUM_REPEATS = 2        # 2 repeats per cell — cliff localisation, not statistics
NUM_PREDICT = 64
TIMEOUT_S = 3600       # 60 min — worst-case 40K on Mini could be very slow

BASE_DIR = Path(__file__).resolve().parent
FIXTURES = BASE_DIR / "fixtures" / "padding"
EVIDENCE = BASE_DIR / "evidence"

SIZES = {
    "20k":   "pad_20k.txt",
    "22500": "pad_22500.txt",
    "25k":   "pad_25k.txt",
    "27500": "pad_27500.txt",
    "30k":   "pad_30k.txt",
    "32500": "pad_32500.txt",
    "35k":   "pad_35k.txt",
    "37500": "pad_37500.txt",
    "40k":   "pad_40k.txt",
}

INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."


def unload_model() -> None:
    try:
        payload = json.dumps({
            "model": MODEL, "prompt": "x", "stream": False,
            "keep_alive": "0", "options": {"num_predict": 1},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_GENERATE, data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60)
    except Exception:
        pass
    try:
        subprocess.run(["ollama", "stop", MODEL], capture_output=True, timeout=15)
    except Exception:
        pass


def call_generate(prompt: str) -> dict:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": NUM_PREDICT,
            "num_ctx": 131072,
        },
    }).encode()
    req = urllib.request.Request(
        OLLAMA_GENERATE, data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        raw = json.loads(resp.read())
    raw["_wall_seconds"] = round(time.time() - t0, 3)
    return raw


def derive_metrics(r: dict) -> dict:
    pc = r.get("prompt_eval_count", 0) or 0
    pd_ = r.get("prompt_eval_duration", 1) or 1
    ec = r.get("eval_count", 0) or 0
    ed = r.get("eval_duration", 1) or 1
    return {
        "prefill_ms_per_tok": round((pd_ / 1e6) / pc, 3) if pc else None,
        "gen_tps":            round(ec / (ed / 1e9), 2) if ec and ed else None,
        "prompt_tokens":      pc,
        "gen_tokens":         ec,
        "wall_seconds":       r["_wall_seconds"],
    }


def run(machine: str, baseline_ms: float, sizes: list[str]) -> None:
    cliff_threshold = baseline_ms * 2.0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_b-{machine}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 007 — Phase B: Prefill cliff localisation")
    print(f"Machine: {machine}  |  Model: {MODEL}  |  Sizes: {sizes}")
    print(f"Baseline (15K): {baseline_ms} ms/tok  |  Cliff threshold: {cliff_threshold:.3f} ms/tok")
    print(f"Evidence: {ev_dir}\n")

    summary_rows = []
    cliff_confirmed_at = None

    for size_key in sizes:
        fixture_path = FIXTURES / SIZES[size_key]
        if not fixture_path.exists():
            print(f"  [{size_key}] SKIP — fixture missing: {fixture_path}")
            continue

        prompt = fixture_path.read_text(encoding="utf-8") + INSTRUCTION

        print(f"  [{size_key}] unloading...", end="", flush=True)
        unload_model()
        print(f" idle {IDLE_SECONDS}s...", end="", flush=True)
        time.sleep(IDLE_SECONDS)
        print(" start")

        cell_records = []
        for rep in range(1, NUM_REPEATS + 1):
            print(f"           rep {rep}/{NUM_REPEATS}  ", end="", flush=True)
            try:
                raw = call_generate(prompt)
                m = derive_metrics(raw)
                record = {
                    "experiment": "007", "phase": "B",
                    "machine": machine, "size_key": size_key, "repeat": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "baseline_ms_per_tok": baseline_ms,
                    "cliff_threshold_ms_per_tok": cliff_threshold,
                    "metrics": m,
                    "ollama_raw": {k: raw.get(k) for k in (
                        "prompt_eval_count", "prompt_eval_duration",
                        "eval_count", "eval_duration", "done_reason",
                    )},
                }
                cell_records.append(record)
                cliff_hit = (m.get("prefill_ms_per_tok") or 0) > cliff_threshold
                flag = " *** CLIFF ***" if cliff_hit else ""
                print(f"prefill={m['prefill_ms_per_tok']} ms/tok  "
                      f"gen={m['gen_tps']} t/s  "
                      f"({m['wall_seconds']:.0f}s){flag}", flush=True)
            except Exception as e:
                print(f"FAILED — {e}", flush=True)
                cell_records.append({
                    "experiment": "007", "phase": "B",
                    "machine": machine, "size_key": size_key,
                    "repeat": rep, "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        (ev_dir / f"size_{size_key}.json").write_text(
            json.dumps(cell_records, indent=2, ensure_ascii=False))

        valid = [r["metrics"] for r in cell_records if "metrics" in r]
        if valid:
            pre_vals = [m["prefill_ms_per_tok"] for m in valid if m.get("prefill_ms_per_tok") is not None]
            mean_pre = round(sum(pre_vals) / len(pre_vals), 3) if pre_vals else None
            cliff_hit = mean_pre is not None and mean_pre > cliff_threshold
            row = {
                "size_key": size_key,
                "prompt_tokens": valid[0].get("prompt_tokens"),
                "mean_prefill_ms_per_tok": mean_pre,
                "cliff_triggered": cliff_hit,
                "n": len(valid),
            }
            summary_rows.append(row)
            if cliff_hit and cliff_confirmed_at is None:
                cliff_confirmed_at = size_key
                print(f"           → CLIFF ONSET CONFIRMED at {size_key} "
                      f"({mean_pre} ms/tok > threshold {cliff_threshold:.3f})\n")
            else:
                print(f"           → mean prefill={mean_pre} ms/tok  cliff={cliff_hit}\n")

    summary = {
        "machine": machine, "model": MODEL,
        "baseline_ms_per_tok": baseline_ms,
        "cliff_threshold_ms_per_tok": cliff_threshold,
        "cliff_confirmed_at": cliff_confirmed_at,
        "started": timestamp,
        "sizes": summary_rows,
    }
    (ev_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Phase B Summary ===")
    print(f"Baseline: {baseline_ms} ms/tok at 15K  |  Cliff threshold: {cliff_threshold:.3f} ms/tok")
    print(f"{'Size':8}  {'Tokens':8}  {'Prefill ms/tok':16}  {'Cliff':6}")
    for row in summary_rows:
        flag = "YES ***" if row.get("cliff_triggered") else "no"
        print(f"{row['size_key']:8}  {str(row.get('prompt_tokens','?')):8}  "
              f"{str(row.get('mean_prefill_ms_per_tok','?')):16}  {flag}")
    if cliff_confirmed_at:
        print(f"\nCliff onset: {cliff_confirmed_at}")
    else:
        print(f"\nCliff not reached in tested range.")
    print(f"\nResults in: {ev_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, choices=["mini", "mbp"],
                   help="Machine identifier")
    p.add_argument("--baseline", required=True, type=float, metavar="MS_PER_TOK",
                   help="Phase A mean_prefill_ms_per_tok at 15K (from summary.json)")
    p.add_argument("--sizes", nargs="+", choices=list(SIZES), default=list(SIZES),
                   help="Which sizes to run (default: all 9)")
    args = p.parse_args()
    run(machine=args.machine, baseline_ms=args.baseline, sizes=args.sizes)
