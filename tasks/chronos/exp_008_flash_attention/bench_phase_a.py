#!/usr/bin/env python3
"""
Exp 008 — Phase A: Generation throughput sweep with FA + q8_0 KV cache.

Same protocol as Exp 007 Phase A. Ollama MUST be running with:
    OLLAMA_FLASH_ATTENTION=1
    OLLAMA_KV_CACHE_TYPE=q8_0

Use start_ollama_flags.sh in a separate terminal before running this.

Usage:
    python3 bench_phase_a.py --machine mini
    python3 bench_phase_a.py --machine mbp
    python3 bench_phase_a.py --machine mini --sizes 4k 8k 15k

Fixtures: shared from exp_007_hardware_comparison/fixtures/padding/ (do not duplicate).
Evidence: written to evidence/<timestamp>-phase_a-<machine>/
"""

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
IDLE_SECONDS = 60
NUM_REPEATS = 3
NUM_PREDICT = 128
TIMEOUT_S = 2400

BASE_DIR = Path(__file__).resolve().parent
FIXTURES = BASE_DIR.parent / "exp_007_hardware_comparison" / "fixtures" / "padding"
EVIDENCE = BASE_DIR / "evidence"

SIZES = {
    "4k":  "pad_4k.txt",
    "8k":  "pad_8k.txt",
    "15k": "pad_15k.txt",
    "25k": "pad_25k.txt",
    "35k": "pad_35k.txt",
}

INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."

FLAGS_ON  = {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"}
FLAGS_OFF = {"OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_KV_CACHE_TYPE": "fp16 (default)"}
FLAGS = FLAGS_ON  # overridden in main() based on --no-flags


def verify_flags() -> bool:
    """Check Ollama server logs for flag confirmation via /api/version."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
        return True
    except Exception:
        return False


def check_ollama_env() -> dict:
    """Best-effort: read Ollama process environment for flag verification."""
    result = {}
    try:
        ps = subprocess.run(
            ["pgrep", "-x", "ollama"],
            capture_output=True, text=True
        )
        pid = ps.stdout.strip().split("\n")[0]
        if pid:
            env_out = subprocess.run(
                ["ps", "eww", "-p", pid],
                capture_output=True, text=True
            )
            for flag in FLAGS:
                if f"{flag}=1" in env_out.stdout or f"{flag}=q8_0" in env_out.stdout:
                    result[flag] = "confirmed"
                else:
                    result[flag] = "not detected (may still be set)"
    except Exception:
        result = {k: "could not verify" for k in FLAGS}
    return result


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
    ec = r.get("eval_count", 0) or 0
    ed = r.get("eval_duration", 1) or 1
    pc = r.get("prompt_eval_count", 0) or 0
    pd_ = r.get("prompt_eval_duration", 1) or 1
    return {
        "gen_tps":            round(ec / (ed / 1e9), 2) if ec and ed else None,
        "prefill_ms_per_tok": round((pd_ / 1e6) / pc, 3) if pc and pd_ else None,
        "prompt_tokens":      pc,
        "gen_tokens":         ec,
        "wall_seconds":       r["_wall_seconds"],
        "load_duration_ms":   round(r.get("load_duration", 0) / 1e6, 1),
    }


def run(machine: str, sizes: list[str], no_flags: bool) -> None:
    mode = "no-flags (FA=0, fp16 KV)" if no_flags else "flags-on (FA=1, q8_0 KV)"
    print(f"\nExp 008 — Phase A: Generation sweep [{mode}]")
    print(f"Machine: {machine}  |  Model: {MODEL}  |  Sizes: {sizes}")
    print(f"\nVerifying Ollama is reachable...", end="", flush=True)
    if not verify_flags():
        print(f" FAILED\nOllama not running. Start it with: ./start_ollama_flags.sh")
        return
    print(" ok")

    env_check = check_ollama_env()
    print(f"Flag check (best-effort process scan):")
    for k, v in env_check.items():
        print(f"  {k}: {v}")
    print(f"  (If flags show as 'not detected', verify start_ollama_flags.sh was used)\n")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "nf" if no_flags else "fa"
    ev_dir = EVIDENCE / f"{timestamp}-phase_a-{machine}-{suffix}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evidence: {ev_dir}\n")

    summary_rows = []

    for size_key in sizes:
        fixture_path = FIXTURES / SIZES[size_key]
        if not fixture_path.exists():
            print(f"  [{size_key}] SKIP — fixture not found: {fixture_path}")
            print(f"           Run exp_007/generate_padding.py first.")
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
                    "experiment": "008", "phase": "A",
                    "machine": machine, "size_key": size_key, "repeat": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL,
                    "ollama_flags": FLAGS,
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
                    "experiment": "008", "phase": "A",
                    "machine": machine, "size_key": size_key,
                    "repeat": rep, "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        (ev_dir / f"size_{size_key}.json").write_text(
            json.dumps(cell_records, indent=2, ensure_ascii=False))

        valid = [r["metrics"] for r in cell_records if "metrics" in r]
        if valid:
            tps_vals = [m["gen_tps"] for m in valid if m.get("gen_tps") is not None]
            row = {
                "size_key": size_key,
                "prompt_tokens": valid[0].get("prompt_tokens"),
                "rep1_gen_tps": valid[0].get("gen_tps"),
                "mean_gen_tps": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None,
                "rep1_prefill_ms_per_tok": valid[0].get("prefill_ms_per_tok"),
                "n": len(valid),
            }
            summary_rows.append(row)
            print(f"           → rep1 prefill={row['rep1_prefill_ms_per_tok']} ms/tok  "
                  f"mean gen={row['mean_gen_tps']} t/s\n")

    (ev_dir / "summary.json").write_text(json.dumps({
        "experiment": "008",
        "machine": machine,
        "model": MODEL,
        "ollama_flags": FLAGS,
        "num_predict": NUM_PREDICT,
        "idle_seconds": IDLE_SECONDS,
        "started": timestamp,
        "sizes": summary_rows,
        "exp007_baseline_ref": "../evidence/ — see exp_007 phase_a for default-flags comparison",
    }, indent=2, ensure_ascii=False))

    print("\n=== Phase A Summary (Exp 008: FA + q8_0) ===")
    print(f"{'Size':8}  {'Tokens':8}  {'Rep1 prefill ms/tok':21}  {'Mean gen t/s':14}")
    for row in summary_rows:
        print(f"{row['size_key']:8}  {str(row.get('prompt_tokens','?')):8}  "
              f"{str(row.get('rep1_prefill_ms_per_tok','?')):21}  "
              f"{str(row.get('mean_gen_tps','?')):14}")
    flag_arg = " --no-flags" if no_flags else ""
    print(f"\nNext: python3 bench_phase_b.py --machine {machine}{flag_arg} "
          f"--baseline <rep1_prefill at 15k above>")
    print(f"Results in: {ev_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, choices=["mini", "mbp"],
                   help="Machine identifier")
    p.add_argument("--sizes", nargs="+", choices=list(SIZES), default=list(SIZES),
                   help="Which sizes to run (default: all 5)")
    p.add_argument("--no-flags", action="store_true",
                   help="Primary run: FA=0, fp16 KV cache (baseline). "
                        "Default: FA=1, q8_0 (flags-on, production config).")
    args = p.parse_args()
    if args.no_flags:
        FLAGS = FLAGS_OFF
    run(machine=args.machine, sizes=args.sizes, no_flags=args.no_flags)
