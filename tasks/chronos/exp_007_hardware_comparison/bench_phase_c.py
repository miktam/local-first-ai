#!/usr/bin/env python3
"""
Exp 007 — Phase C: Thermal endurance (MacBook Pro Max 5 only).

Runs continuous generation at 8K context for 90 minutes.
Samples gen t/s every 5 minutes. Captures powermetrics if available (requires sudo).

Usage:
    python3 bench_phase_c.py                    # full 90-minute run
    python3 bench_phase_c.py --duration 1800    # custom duration in seconds
    python3 bench_phase_c.py --no-powermetrics  # skip thermal capture (no sudo)

Powermetrics: the script tries `sudo -n powermetrics` (passwordless sudo).
If it fails, thermal data is absent but gen t/s data is still valid.
To enable: add `<user> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics` to sudoers.

Evidence written to: evidence/<timestamp>-phase_c-mbp/
"""

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_GENERATE = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
CONTEXT_SIZE = "8k"
NUM_PREDICT = 256      # long enough to measure steady-state gen t/s per call
TIMEOUT_S = 600        # 10 min per call — 8K should complete in <60s
SAMPLE_INTERVAL_S = 300  # 5 minutes

TOTAL_DURATION_DEFAULT = 5400  # 90 minutes

BASE_DIR = Path(__file__).resolve().parent
FIXTURES = BASE_DIR / "fixtures" / "padding"
EVIDENCE = BASE_DIR / "evidence"

INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."

thermal_log: list[dict] = []
thermal_lock = threading.Lock()
stop_thermal = threading.Event()


def sample_powermetrics() -> dict | None:
    """Single powermetrics snapshot. Returns None if unavailable."""
    try:
        result = subprocess.run(
            ["sudo", "-n", "powermetrics",
             "-n", "1", "-i", "500",
             "--samplers", "gpu_power,cpu_power,thermal",
             "--format", "json"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
    except Exception:
        return None


def parse_thermal(pm: dict | None) -> dict:
    if pm is None:
        return {"available": False}
    try:
        gpu = pm.get("gpu", {})
        cpu = pm.get("processor", {})
        temp = pm.get("thermal_pressure", None)
        fans = pm.get("fans", [])
        fan_rpm = [f.get("rpm") for f in fans if f.get("rpm") is not None] if fans else []
        return {
            "available": True,
            "gpu_power_w": gpu.get("gpu_power") or gpu.get("power"),
            "cpu_power_w": cpu.get("cpu_power") or cpu.get("power"),
            "die_temp_c": cpu.get("die_temp") or cpu.get("temperature"),
            "thermal_pressure": temp,
            "fan_rpm": fan_rpm,
        }
    except Exception as e:
        return {"available": True, "parse_error": str(e)}


def thermal_sampler_thread(interval: int) -> None:
    """Background thread: sample powermetrics every `interval` seconds."""
    while not stop_thermal.wait(timeout=interval):
        pm = sample_powermetrics()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": None,  # filled by main thread on flush
            "thermal": parse_thermal(pm),
        }
        with thermal_lock:
            thermal_log.append(entry)


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


def run(duration_s: int, use_powermetrics: bool) -> None:
    fixture_path = FIXTURES / "pad_8k.txt"
    if not fixture_path.exists():
        print(f"ERROR: fixture not found: {fixture_path}")
        print("Run: python3 generate_padding.py")
        sys.exit(1)

    prompt = fixture_path.read_text(encoding="utf-8") + INSTRUCTION

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_c-mbp"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 007 — Phase C: Thermal endurance")
    print(f"Machine: mbp (MacBook Pro Max 5 only)")
    print(f"Model: {MODEL}  |  Context: {CONTEXT_SIZE}  |  Duration: {duration_s}s  "
          f"|  Powermetrics: {'yes' if use_powermetrics else 'no'}")
    print(f"Evidence: {ev_dir}\n")

    if use_powermetrics:
        pm_check = sample_powermetrics()
        if pm_check is None:
            print("  [warn] powermetrics unavailable — thermal data will be absent.")
            print("         To enable: sudo visudo — add NOPASSWD rule for powermetrics")
            use_powermetrics = False
        else:
            print("  [ok] powermetrics available — thermal sampling active")
            t = threading.Thread(target=thermal_sampler_thread,
                                 args=(SAMPLE_INTERVAL_S,), daemon=True)
            t.start()

    print(f"\n  Unloading model before first call...")
    unload_model()
    time.sleep(10)

    generation_log: list[dict] = []
    start_time = time.time()
    call_count = 0

    print(f"  {'Elapsed':8}  {'Gen t/s':10}  {'Wall s':8}  {'Note'}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*20}")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration_s:
            break

        call_count += 1
        try:
            raw = call_generate(prompt)
            ec = raw.get("eval_count", 0) or 0
            ed = raw.get("eval_duration", 1) or 1
            pc = raw.get("prompt_eval_count", 0) or 0
            gen_tps = round(ec / (ed / 1e9), 2) if ec and ed else None
            wall = raw["_wall_seconds"]
            elapsed_now = round(time.time() - start_time, 1)

            record = {
                "call": call_count,
                "elapsed_s": elapsed_now,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "gen_tps": gen_tps,
                "prompt_tokens": pc,
                "gen_tokens": ec,
                "wall_seconds": wall,
                "ollama_raw": {k: raw.get(k) for k in (
                    "eval_count", "eval_duration",
                    "prompt_eval_count", "prompt_eval_duration",
                )},
            }
            generation_log.append(record)
            elapsed_min = elapsed_now / 60
            note = ""
            if call_count == 1:
                note = "(cold start)"
            print(f"  {elapsed_min:6.1f}m    {str(gen_tps):10}  {wall:8.0f}  {note}",
                  flush=True)

        except Exception as e:
            elapsed_now = round(time.time() - start_time, 1)
            print(f"  {elapsed_now/60:6.1f}m    FAILED — {e}", flush=True)
            generation_log.append({
                "call": call_count,
                "elapsed_s": elapsed_now,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    stop_thermal.set()
    total_elapsed = round(time.time() - start_time, 1)

    with thermal_lock:
        snapshots = list(thermal_log)

    (ev_dir / "generation_log.json").write_text(
        json.dumps(generation_log, indent=2, ensure_ascii=False))
    (ev_dir / "thermal_log.json").write_text(
        json.dumps(snapshots, indent=2, ensure_ascii=False))

    valid_tps = [r["gen_tps"] for r in generation_log if r.get("gen_tps") is not None]
    summary = {
        "machine": "mbp",
        "model": MODEL,
        "context_size": CONTEXT_SIZE,
        "duration_s": total_elapsed,
        "total_calls": call_count,
        "powermetrics_active": use_powermetrics,
        "started": timestamp,
        "gen_tps_min": min(valid_tps) if valid_tps else None,
        "gen_tps_max": max(valid_tps) if valid_tps else None,
        "gen_tps_mean": round(sum(valid_tps) / len(valid_tps), 2) if valid_tps else None,
        "gen_tps_peak_to_trough": round(max(valid_tps) - min(valid_tps), 2) if valid_tps else None,
        "h3_confirmed": (max(valid_tps) - min(valid_tps)) >= 5.0 if valid_tps else None,
    }
    (ev_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n=== Phase C Summary ===")
    print(f"Duration: {total_elapsed:.0f}s  |  Calls: {call_count}")
    if valid_tps:
        print(f"Gen t/s — min: {summary['gen_tps_min']}  "
              f"max: {summary['gen_tps_max']}  "
              f"mean: {summary['gen_tps_mean']}  "
              f"peak-to-trough: {summary['gen_tps_peak_to_trough']}")
        h3 = summary["h3_confirmed"]
        print(f"H3 (thermal decay ≥5 t/s): {'CONFIRMED' if h3 else 'rejected'}")
    print(f"\nResults in: {ev_dir}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=TOTAL_DURATION_DEFAULT,
                   metavar="SECONDS", help="Total run duration (default: 5400 = 90 min)")
    p.add_argument("--no-powermetrics", action="store_true",
                   help="Skip thermal capture even if powermetrics is available")
    args = p.parse_args()
    run(duration_s=args.duration, use_powermetrics=not args.no_powermetrics)
