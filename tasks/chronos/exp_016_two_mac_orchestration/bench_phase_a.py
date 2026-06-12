#!/usr/bin/env python3
"""
Exp 016 Phase A — Model selection benchmark.

Runs on: MBP (service-user). Results are written locally, then copied to mini.

Usage:
    python3 bench_phase_a.py                   # all models in order
    python3 bench_phase_a.py --model-index 0   # control only (0=control, 1=primary, 2=coding, 3=optional)
    python3 bench_phase_a.py --output-dir ~/somewhere

Copy results to mini when done (printed again at end):
    scp ~/exp_016_measurements/*.json \\
        miktam02@mini.local:/Users/miktam02/REPOS/local-first-ai/tasks/chronos/exp_016_two_mac_orchestration/measurements/
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK_PROMPT = (
    "Add a price_per_sqm computed field to the get_property response. "
    "The field is price_EUR divided by living_area_sqm, rounded to the nearest integer. "
    "Return null if either field is absent or zero. "
    "Write the implementation in scripts/mcp_server.py and add a regression test."
)

MAX_TOKENS = 2000
MACHINE = "mbp"
REPS = 3

MODELS = [
    {
        "id": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "slug": "gemma4-26b",
        "priority": "control",
        "note": "Baseline — same model as mini",
    },
    {
        "id": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "slug": "qwen35-122b",
        "priority": "primary",
        "note": "Main smart-tier candidate",
    },
    {
        "id": "mlx-community/Qwen3-Coder-Next-4bit",
        "slug": "qwen3-coder",
        "priority": "coding",
        "note": "Coding specialist",
    },
    {
        "id": "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "slug": "llama4-scout",
        "priority": "optional",
        "note": "Run if time allows",
    },
]

QUALITY_RUBRIC = """
Quality rubric (0–5):
  0 — no useful output / completely wrong
  1 — understands task, implementation incorrect
  2 — mostly correct reasoning, code incomplete
  3 — correct implementation + regression test present (conceptually)
  4 — correct + clean
  5 — correct + clean + idiomatic, all edge cases handled
"""

MINI_MEASUREMENTS_PATH = (
    "/Users/miktam02/REPOS/local-first-ai/tasks/chronos"
    "/exp_016_two_mac_orchestration/measurements/"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_metrics(output: str) -> dict:
    metrics = {}
    m = re.search(r"Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec", output)
    if m:
        metrics["prompt_tokens"] = int(m.group(1))
        metrics["prompt_tps"] = float(m.group(2))
    m = re.search(r"Generation: (\d+) tokens, ([\d.]+) tokens-per-sec", output)
    if m:
        metrics["gen_tokens"] = int(m.group(1))
        metrics["gen_tps"] = float(m.group(2))
    m = re.search(r"Peak memory: ([\d.]+) GB", output)
    if m:
        metrics["peak_memory_gb"] = float(m.group(1))
    return metrics


def extract_model_output(stdout: str) -> str:
    """Extract text between the ========== delimiters."""
    parts = stdout.split("==========")
    if len(parts) >= 3:
        return parts[1].strip()
    return stdout.strip()


def ask_quality_score(rep: int) -> tuple[int, str]:
    print(QUALITY_RUBRIC)
    while True:
        try:
            raw = input(f"Quality score for rep {rep} (0–5, or 's' to skip): ").strip()
            if raw.lower() == "s":
                return -1, "skipped"
            score = int(raw)
            if 0 <= score <= 5:
                break
            print("  Enter 0–5.")
        except (ValueError, EOFError):
            return -1, "skipped"
    notes = input("  Notes (Enter to skip): ").strip()
    return score, notes


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------

def run_rep(model_id: str, model_slug: str, rep: int, output_dir: Path) -> dict:
    stamp = ts_now()
    print(f"\n{'─'*60}")
    print(f"  {model_id}  |  rep {rep}/{REPS}  |  {stamp}")
    print(f"{'─'*60}")

    cmd = [
        "mlx_lm.generate",
        "--model", model_id,
        "--prompt", BENCHMARK_PROMPT,
        "--max-tokens", str(MAX_TOKENS),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_clock_s = round(time.time() - t0, 2)

    combined = proc.stdout + proc.stderr
    metrics = parse_metrics(combined)
    model_output = extract_model_output(proc.stdout)

    print("\n── model output ──────────────────────────────────────")
    print(model_output[:3000])  # truncate display, full text saved to JSON
    if len(model_output) > 3000:
        print(f"  [truncated — full output saved to JSON]")
    print("── metrics ────────────────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  wall_clock_s: {wall_clock_s}")
    print("───────────────────────────────────────────────────────\n")

    quality_score, quality_notes = ask_quality_score(rep)

    record = {
        "experiment": "016",
        "phase": "A",
        "machine": MACHINE,
        "model_id": model_id,
        "rep": rep,
        "timestamp": stamp,
        "benchmark_prompt": BENCHMARK_PROMPT,
        "max_tokens": MAX_TOKENS,
        "wall_clock_s": wall_clock_s,
        **metrics,
        "quality_score": quality_score,
        "quality_notes": quality_notes,
        "model_output": model_output,
        "note": "wall_clock_s includes model load from disk; each rep is a fresh subprocess",
    }

    slug = model_slug[:20]
    filename = f"{stamp}-phase_a-{MACHINE}-{slug}-rep{rep}.json"
    dest = output_dir / filename
    dest.write_text(json.dumps(record, indent=2))
    print(f"  Saved: {dest.name}")
    return record


def run_model(model: dict, output_dir: Path) -> list[dict]:
    print(f"\n{'═'*60}")
    print(f"  MODEL {model['priority'].upper()}: {model['id']}")
    print(f"  {model['note']}")
    print(f"{'═'*60}")
    try:
        input("\n  Press Enter to start (Ctrl-C to skip this model)...")
    except KeyboardInterrupt:
        print(f"\n  Skipping {model['slug']}.")
        return []

    results = []
    for rep in range(1, REPS + 1):
        try:
            r = run_rep(model["id"], model["slug"], rep, output_dir)
            results.append(r)
        except KeyboardInterrupt:
            print(f"\n  Stopped at rep {rep}.")
            break
    return results


def write_summary(all_results: list[dict], output_dir: Path):
    by_model: dict = {}
    for r in all_results:
        mid = r["model_id"]
        if mid not in by_model:
            by_model[mid] = {"model_id": mid, "reps": []}
        by_model[mid]["reps"].append({
            k: r.get(k) for k in [
                "rep", "gen_tps", "prompt_tps", "gen_tokens",
                "prompt_tokens", "peak_memory_gb", "wall_clock_s",
                "quality_score", "quality_notes", "timestamp",
            ]
        })

    dest = output_dir / "phase_a_summary.json"
    dest.write_text(json.dumps({"experiment": "016", "phase": "A",
                                "machine": MACHINE, "models": list(by_model.values())},
                               indent=2))
    print(f"\n  Summary: {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp 016 Phase A benchmark (runs on MBP)")
    parser.add_argument("--output-dir", default=str(Path.home() / "exp_016_measurements"),
                        help="Local directory for JSON output (default: ~/exp_016_measurements)")
    parser.add_argument("--model-index", type=int, default=None,
                        help="Run only one model: 0=control 1=primary 2=coding 3=optional")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExp 016 Phase A — output: {output_dir}\n")

    models = MODELS if args.model_index is None else [MODELS[args.model_index]]

    all_results = []
    for model in models:
        try:
            reps = run_model(model, output_dir)
            all_results.extend(reps)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

    if all_results:
        write_summary(all_results, output_dir)

    scp_cmd = (
        f"scp {output_dir}/*.json "
        f"miktam02@mini.local:{MINI_MEASUREMENTS_PATH}"
    )
    print(f"\n{'═'*60}")
    print("  Phase A done. Copy results to mini:")
    print(f"\n  {scp_cmd}\n")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
