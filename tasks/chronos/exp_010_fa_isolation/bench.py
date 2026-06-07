#!/usr/bin/env python3
"""
Exp 010 — FA vs q8_0 KV Cache: Factorial Isolation.

Tests two remaining conditions of the 2x2 factorial to isolate
whether OLLAMA_FLASH_ATTENTION=1 or OLLAMA_KV_CACHE_TYPE=q8_0
(or their combination) caused the 20K prefill cliff in Exp 007.

Known baselines (from prior experiments):
  Condition A: FA=0, fp16  → Exp 008  (no cliff through 40K)
  Condition D: FA=1, q8_0  → Exp 007  (cliff at 20K)

This script runs:
  Condition B: FA=1, fp16  → flash attention isolated
  Condition C: FA=0, q8_0  → KV quantization isolated

Usage:
  # Start Ollama for the condition first, then:
  python3 bench.py --condition B --phase A --machine mini
  python3 bench.py --condition B --phase B --machine mini --baseline <ms_tok>
  python3 bench.py --condition C --phase A --machine mini
  python3 bench.py --condition C --phase B --machine mini --baseline <ms_tok>

Fixtures: shared from exp_007_hardware_comparison/fixtures/padding/
Evidence: written to evidence/<timestamp>-phase_<P>-<machine>-cond<X>/
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
TIMEOUT_S = 3600
INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."

BASE_DIR = Path(__file__).resolve().parent
FIXTURES = BASE_DIR.parent / "exp_007_hardware_comparison" / "fixtures" / "padding"
EVIDENCE = BASE_DIR / "evidence"

SIZES_A = {
    "4k":  ("pad_4k.txt",  3),
    "8k":  ("pad_8k.txt",  3),
    "15k": ("pad_15k.txt", 3),
    "25k": ("pad_25k.txt", 3),
    "35k": ("pad_35k.txt", 3),
}
SIZES_B = {
    "20k":   ("pad_20k.txt",   2),
    "22500": ("pad_22500.txt", 2),
    "25k":   ("pad_25k.txt",   2),
    "27500": ("pad_27500.txt", 2),
    "30k":   ("pad_30k.txt",   2),
    "32500": ("pad_32500.txt", 2),
    "35k":   ("pad_35k.txt",   2),
    "37500": ("pad_37500.txt", 2),
    "40k":   ("pad_40k.txt",   2),
}

CONDITIONS = {
    "B": {"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "fp16 (default)"},
    "C": {"OLLAMA_FLASH_ATTENTION": "0", "OLLAMA_KV_CACHE_TYPE": "q8_0"},
}

# Known baselines for comparison in output
EXP007_PHASE_A = {"4k": 34.76, "8k": 31.38, "15k": 25.08, "25k": 14.40, "35k": 10.75}
EXP008_PHASE_A = {"4k": 40.42, "8k": 42.60, "15k": 36.97, "25k": 31.08, "35k": 26.69}


def verify_ollama() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5):
            return True
    except Exception:
        return False


def check_ollama_env(condition: str) -> dict:
    flags = CONDITIONS[condition]
    result = {}
    try:
        ps = subprocess.run(["pgrep", "-x", "ollama"], capture_output=True, text=True)
        pid = ps.stdout.strip().split("\n")[0]
        if pid:
            env_out = subprocess.run(["ps", "eww", "-p", pid],
                                     capture_output=True, text=True)
            for flag, expected in [("OLLAMA_FLASH_ATTENTION", flags["OLLAMA_FLASH_ATTENTION"]),
                                    ("OLLAMA_KV_CACHE_TYPE", flags.get("OLLAMA_KV_CACHE_TYPE", ""))]:
                result[flag] = "confirmed" if f"{flag}={expected}" in env_out.stdout \
                    else "not detected"
    except Exception:
        result = {k: "could not verify" for k in flags}
    return result


def unload_model() -> None:
    try:
        payload = json.dumps({
            "model": MODEL, "prompt": "x", "stream": False,
            "keep_alive": "0", "options": {"num_predict": 1},
        }).encode()
        req = urllib.request.Request(OLLAMA_GENERATE, data=payload,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60)
    except Exception:
        pass
    try:
        subprocess.run(["ollama", "stop", MODEL], capture_output=True, timeout=15)
    except Exception:
        pass


def call_generate(prompt: str, num_predict: int) -> dict:
    payload = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": 131072},
    }).encode()
    req = urllib.request.Request(OLLAMA_GENERATE, data=payload,
                                 headers={"Content-Type": "application/json"})
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
        "gen_tps":            round(ec / (ed / 1e9), 2) if ec and ed else None,
        "prefill_ms_per_tok": round((pd_ / 1e6) / pc, 3) if pc else None,
        "prompt_tokens":      pc,
        "gen_tokens":         ec,
        "wall_seconds":       r["_wall_seconds"],
        "load_duration_ms":   round(r.get("load_duration", 0) / 1e6, 1),
    }


def run_phase_a(machine: str, condition: str) -> None:
    flags = CONDITIONS[condition]
    label = f"Condition {condition}: FA={flags['OLLAMA_FLASH_ATTENTION']}, KV={flags['OLLAMA_KV_CACHE_TYPE']}"
    print(f"\nExp 010 — Phase A [{label}]")
    print(f"Machine: {machine}  |  Model: {MODEL}")

    if not verify_ollama():
        print("Ollama not running. Start it with the appropriate start_condition_X.sh")
        return

    env = check_ollama_env(condition)
    print("Flag check (best-effort):")
    for k, v in env.items():
        print(f"  {k}: {v}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_a-{machine}-cond{condition}"
    ev_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evidence: {ev_dir}\n")

    summary_rows = []
    for size_key, (fname, reps) in SIZES_A.items():
        fixture_path = FIXTURES / fname
        if not fixture_path.exists():
            print(f"  [{size_key}] SKIP — fixture not found")
            continue
        prompt = fixture_path.read_text(encoding="utf-8") + INSTRUCTION

        print(f"  [{size_key}] unloading...", end="", flush=True)
        unload_model()
        print(f" idle {IDLE_SECONDS}s...", end="", flush=True)
        time.sleep(IDLE_SECONDS)
        print(" start")

        cell_records = []
        for rep in range(1, reps + 1):
            print(f"           rep {rep}/{reps}  ", end="", flush=True)
            try:
                raw = call_generate(prompt, 128)
                m = derive_metrics(raw)
                cell_records.append({
                    "experiment": "010", "phase": "A", "condition": condition,
                    "machine": machine, "size_key": size_key, "repeat": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL, "ollama_flags": flags, "metrics": m,
                    "ollama_raw": {k: raw.get(k) for k in (
                        "total_duration", "load_duration", "prompt_eval_count",
                        "prompt_eval_duration", "eval_count", "eval_duration", "done_reason")},
                })
                print(f"gen={m['gen_tps']} t/s  prefill={m['prefill_ms_per_tok']} ms/tok  "
                      f"({m['wall_seconds']:.0f}s)", flush=True)
            except Exception as e:
                print(f"FAILED — {e}", flush=True)
                cell_records.append({"experiment": "010", "phase": "A", "condition": condition,
                                     "machine": machine, "size_key": size_key,
                                     "repeat": rep, "error": str(e),
                                     "timestamp": datetime.now(timezone.utc).isoformat()})

        (ev_dir / f"size_{size_key}.json").write_text(
            json.dumps(cell_records, indent=2, ensure_ascii=False))

        valid = [r["metrics"] for r in cell_records if "metrics" in r]
        if valid:
            tps_vals = [m["gen_tps"] for m in valid if m.get("gen_tps")]
            row = {"size_key": size_key,
                   "prompt_tokens": valid[0].get("prompt_tokens"),
                   "rep1_prefill_ms_per_tok": valid[0].get("prefill_ms_per_tok"),
                   "mean_gen_tps": round(sum(tps_vals) / len(tps_vals), 2) if tps_vals else None,
                   "exp007_gen_tps": EXP007_PHASE_A.get(size_key),
                   "exp008_gen_tps": EXP008_PHASE_A.get(size_key),
                   "n": len(valid)}
            summary_rows.append(row)
            print(f"           → rep1 prefill={row['rep1_prefill_ms_per_tok']} ms/tok  "
                  f"mean gen={row['mean_gen_tps']} t/s\n")

    (ev_dir / "summary.json").write_text(json.dumps({
        "experiment": "010", "phase": "A", "condition": condition,
        "machine": machine, "model": MODEL, "ollama_flags": flags,
        "idle_seconds": IDLE_SECONDS, "started": timestamp, "sizes": summary_rows,
        "baselines": {"exp007_FA1_q8": EXP007_PHASE_A, "exp008_FA0_fp16": EXP008_PHASE_A},
    }, indent=2, ensure_ascii=False))

    print(f"\n=== Phase A Summary (Exp 010 Condition {condition}) ===")
    print(f"{'Size':8}  {'Tokens':8}  {'Rep1 prefill':14}  {'Mean gen t/s':14}  "
          f"{'Exp007 (FA+q8)':16}  {'Exp008 (FA0)':12}")
    for row in summary_rows:
        print(f"{row['size_key']:8}  {str(row.get('prompt_tokens','?')):8}  "
              f"{str(row.get('rep1_prefill_ms_per_tok','?')):14}  "
              f"{str(row.get('mean_gen_tps','?')):14}  "
              f"{str(row.get('exp007_gen_tps','?')):16}  "
              f"{str(row.get('exp008_gen_tps','?')):12}")
    rep1_15k = next((r["rep1_prefill_ms_per_tok"] for r in summary_rows if r["size_key"] == "15k"), None)
    if rep1_15k:
        print(f"\nNext: python3 bench.py --condition {condition} --phase B "
              f"--machine {machine} --baseline {rep1_15k}")
    print(f"Results in: {ev_dir}/\n")


def run_phase_b(machine: str, condition: str, baseline_ms: float) -> None:
    flags = CONDITIONS[condition]
    cliff_threshold = baseline_ms * 2.0
    label = f"Condition {condition}: FA={flags['OLLAMA_FLASH_ATTENTION']}, KV={flags['OLLAMA_KV_CACHE_TYPE']}"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_b-{machine}-cond{condition}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 010 — Phase B [{label}]")
    print(f"Machine: {machine}  |  Baseline: {baseline_ms} ms/tok  |  "
          f"Cliff threshold: {cliff_threshold:.3f} ms/tok")
    print(f"Exp 007 cliff was at 20K (FA+q8). Exp 008 had no cliff to 40K (FA0).")
    print(f"Evidence: {ev_dir}\n")

    summary_rows = []
    cliff_confirmed_at = None

    for size_key, (fname, reps) in SIZES_B.items():
        fixture_path = FIXTURES / fname
        if not fixture_path.exists():
            print(f"  [{size_key}] SKIP — fixture not found")
            continue
        prompt = fixture_path.read_text(encoding="utf-8") + INSTRUCTION

        print(f"  [{size_key}] unloading...", end="", flush=True)
        unload_model()
        print(f" idle {IDLE_SECONDS}s...", end="", flush=True)
        time.sleep(IDLE_SECONDS)
        print(" start")

        cell_records = []
        for rep in range(1, reps + 1):
            print(f"           rep {rep}/{reps}  ", end="", flush=True)
            try:
                raw = call_generate(prompt, 64)
                m = derive_metrics(raw)
                cliff_hit = (m.get("prefill_ms_per_tok") or 0) > cliff_threshold
                cell_records.append({
                    "experiment": "010", "phase": "B", "condition": condition,
                    "machine": machine, "size_key": size_key, "repeat": rep,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL, "ollama_flags": flags,
                    "baseline_ms_per_tok": baseline_ms,
                    "cliff_threshold_ms_per_tok": cliff_threshold,
                    "metrics": m,
                    "ollama_raw": {k: raw.get(k) for k in (
                        "prompt_eval_count", "prompt_eval_duration",
                        "eval_count", "eval_duration", "done_reason")},
                })
                flag_str = " *** CLIFF ***" if cliff_hit else ""
                print(f"prefill={m['prefill_ms_per_tok']} ms/tok  "
                      f"gen={m['gen_tps']} t/s  ({m['wall_seconds']:.0f}s){flag_str}", flush=True)
            except Exception as e:
                print(f"FAILED — {e}", flush=True)
                cell_records.append({"experiment": "010", "phase": "B", "condition": condition,
                                     "machine": machine, "size_key": size_key,
                                     "repeat": rep, "error": str(e),
                                     "timestamp": datetime.now(timezone.utc).isoformat()})

        (ev_dir / f"size_{size_key}.json").write_text(
            json.dumps(cell_records, indent=2, ensure_ascii=False))

        valid = [r["metrics"] for r in cell_records if "metrics" in r]
        if valid:
            cold_pre = valid[0].get("prefill_ms_per_tok")
            cliff_hit = cold_pre is not None and cold_pre > cliff_threshold
            row = {"size_key": size_key,
                   "prompt_tokens": valid[0].get("prompt_tokens"),
                   "cold_prefill_ms_per_tok": cold_pre,
                   "cliff_triggered": cliff_hit, "n": len(valid)}
            summary_rows.append(row)
            if cliff_hit and cliff_confirmed_at is None:
                cliff_confirmed_at = size_key
                print(f"           → CLIFF at {size_key} "
                      f"(rep1={cold_pre} ms/tok > threshold {cliff_threshold:.3f})\n")
            else:
                print(f"           → rep1={cold_pre} ms/tok  cliff={cliff_hit}\n")

    (ev_dir / "summary.json").write_text(json.dumps({
        "experiment": "010", "phase": "B", "condition": condition,
        "machine": machine, "model": MODEL, "ollama_flags": flags,
        "baseline_ms_per_tok": baseline_ms, "cliff_threshold_ms_per_tok": cliff_threshold,
        "cliff_confirmed_at": cliff_confirmed_at, "started": timestamp, "sizes": summary_rows,
        "reference_cliffs": {"exp007_FA1_q8": "20k", "exp008_FA0_fp16": "not reached at 40k"},
    }, indent=2, ensure_ascii=False))

    print(f"\n=== Phase B Summary (Exp 010 Condition {condition}) ===")
    print(f"Baseline: {baseline_ms} ms/tok  |  Threshold: {cliff_threshold:.3f} ms/tok")
    print(f"{'Size':8}  {'Tokens':8}  {'Rep1 prefill ms/tok':21}  {'Cliff':6}")
    for row in summary_rows:
        print(f"{row['size_key']:8}  {str(row.get('prompt_tokens','?')):8}  "
              f"{str(row.get('cold_prefill_ms_per_tok','?')):21}  "
              f"{'YES ***' if row.get('cliff_triggered') else 'no'}")
    if cliff_confirmed_at:
        print(f"\nCliff onset: {cliff_confirmed_at}")
    else:
        print(f"\nCliff not reached in tested range.")
    print(f"Results in: {ev_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=["B", "C"],
                   help="B = FA=1/fp16 (FA isolated), C = FA=0/q8_0 (KV isolated)")
    p.add_argument("--phase", required=True, choices=["A", "B"],
                   help="A = generation sweep, B = cliff localisation")
    p.add_argument("--machine", required=True, choices=["mini", "mbp"])
    p.add_argument("--baseline", type=float, metavar="MS_PER_TOK",
                   help="Rep1 prefill ms/tok at 15K from Phase A (required for Phase B)")
    args = p.parse_args()

    if args.phase == "B" and args.baseline is None:
        p.error("--baseline is required for Phase B")

    if args.phase == "A":
        run_phase_a(machine=args.machine, condition=args.condition)
    else:
        run_phase_b(machine=args.machine, condition=args.condition, baseline_ms=args.baseline)
