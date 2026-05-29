#!/usr/bin/env python3
"""
Exp 007 — Phase A: Generation throughput sweep.

For each context size: unload model → 60s idle → 3 repeats.
Captures gen t/s and prefill ms/tok from Ollama's timing fields.

Usage:
    python3 bench_phase_a.py --machine mini
    python3 bench_phase_a.py --machine mbp
    python3 bench_phase_a.py --machine mini --sizes 4k 8k   # partial run

Pre-conditions:
    - Ollama running with gemma4:26b available
    - Padding fixtures generated (run generate_padding.py first)
    - Machine on AC power, Wi-Fi off, no other foreground apps

Evidence written to: evidence/<timestamp>-phase_a-<machine>/
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
NUM_REPEATS = 3
NUM_PREDICT = 128      # enough to measure gen t/s; short to bound wall time
TIMEOUT_S = 2400       # 40 min — covers worst-case 35K prefill on Mini

BASE_DIR = Path(__file__).resolve().parent
FIXTURES = BASE_DIR / "fixtures" / "padding"
EVIDENCE = BASE_DIR / "evidence"

SIZES = {
    "4k":  "pad_4k.txt",
    "8k":  "pad_8k.txt",
    "15k": "pad_15k.txt",
    "25k": "pad_25k.txt",
    "35k": "pad_35k.txt",
}

INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."


def unload_model() -> None:
    """Unload model from memory via keep_alive=0 ping, then ollama stop fallback."""
    try:
        payload = json.dumps({
            "model": MODEL,
            "prompt": "x",
            "stream": False,
            "keep_alive": "0",
            "options": {"num_predict": 1},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_GENERATE, data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60)
    except Exception:
        pass
    try:
        subprocess.run(["ollama", "stop", MODEL],
                       capture_output=True, timeout=15)
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
    ec = r.get("eval_count", 0) or 0
    ed = r.get("eval_duration", 1) or 1
    pc = r.get("prompt_eval_count", 0) or 0
    pd_ = r.get("prompt_eval_duration", 1) or 1
    return {
        "gen_tps":             round(ec / (ed / 1e9), 2) if ec and ed else None,
        "prefill_ms_per_tok":  round((pd_ / 1e6) / pc, 3) if pc and pd_ else None,
        "prompt_tokens":       pc,
        "gen_tokens":          ec,
        "wall_seconds":        r["_wall_seconds"],
        "load_duration_ms":    round(r.get("load_duration", 0) / 1e6, 1),
    }


def run(machine: str, sizes: list[str]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_a-{machine}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 007 — Phase A: Generation throughput sweep")
    print(f"Machine: {machine}  |  Model: {MODEL}  |  Sizes: {sizes}  |  Repeats: {NUM_REPEATS}")
    print(f"Evidence: {ev_dir}\n")

    summary_rows = []

    for size_key in sizes:
        fixture_path = FIXTURES / SIZES[size_key]
        if not fixture_path.exists():
            print(f"  [{size_key}] SKIP — fixture missing: {fixture_path}")
            print(f"           Run: python3 generate_padding.py")
            continue

        prompt = fixture_path.read_text(encoding="utf-8") + INSTRUCTION

        print(f"  [{size_key}] unloading model...", end="", flush=True)
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
                    "experiment": "007", "phase": "A",
                    "machine": machine, "size_key": size_key, "repeat": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "metrics": m,
                    "ollama_raw": {k: raw.get(k) for k in (
                        "total_duration", "load_duration",
                        "prompt_eval_count", "prompt_eval_duration",
                        "eval_count", "eval_duration", "done_reason",
                    )},
                }
                cell_records.append(record)
                print(f"gen={m['gen_tps']} t/s  "
                      f"prefill={m['prefill_ms_per_tok']} ms/tok  "
                      f"({m['wall_seconds']:.0f}s)", flush=True)
            except Exception as e:
                print(f"FAILED — {e}", flush=True)
                cell_records.append({
                    "experiment": "007", "phase": "A",
                    "machine": machine, "size_key": size_key,
                    "repeat": rep, "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        (ev_dir / f"size_{size_key}.json").write_text(
            json.dumps(cell_records, indent=2, ensure_ascii=False))

        valid = [r["metrics"] for r in cell_records if "metrics" in r]
        if valid:
            tps_vals = [m["gen_tps"] for m in valid if m.get("gen_tps") is not None]
            pre_vals = [m["prefill_ms_per_tok"] for m in valid if m.get("prefill_ms_per_tok") is not None]
            row = {
                "size_key": size_key,
                "prompt_tokens": valid[0].get("prompt_tokens"),
                "mean_gen_tps": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None,
                "mean_prefill_ms_per_tok": round(sum(pre_vals) / len(pre_vals), 3) if pre_vals else None,
                "n": len(valid),
            }
            summary_rows.append(row)
            print(f"           → mean gen={row['mean_gen_tps']} t/s  "
                  f"prefill={row['mean_prefill_ms_per_tok']} ms/tok\n")

    summary = {
        "machine": machine, "model": MODEL,
        "num_predict": NUM_PREDICT, "idle_seconds": IDLE_SECONDS,
        "started": timestamp,
        "sizes": summary_rows,
    }
    (ev_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Phase A Summary ===")
    print(f"{'Size':8}  {'Tokens':8}  {'Gen t/s':10}  {'Prefill ms/tok':16}")
    for row in summary_rows:
        print(f"{row['size_key']:8}  {str(row.get('prompt_tokens','?')):8}  "
              f"{str(row.get('mean_gen_tps','?')):10}  "
              f"{str(row.get('mean_prefill_ms_per_tok','?')):16}")
    print(f"\nResults in: {ev_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, choices=["mini", "mbp"],
                   help="Machine identifier for evidence tagging")
    p.add_argument("--sizes", nargs="+", choices=list(SIZES), default=list(SIZES),
                   help="Which sizes to run (default: all 5)")
    args = p.parse_args()
    run(machine=args.machine, sizes=args.sizes)
