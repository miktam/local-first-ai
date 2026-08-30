#!/usr/bin/env python3
"""
Exp 025 — Context-allocation isolation.

Part 1: num_ctx sweep for gemma4:31b vs gemma4:26b (control) — isolates
whether gemma4:31b's exp_023 slowdown is a context-size cliff specific to
that model, not shared by a same-default-context sibling.

Part 2: re-run exp_023's exact 5-prompt comparison across all 5 models with
num_ctx explicitly fixed at 4096 — produces a non-confounded efficiency
ranking, directly diffable against exp_023's original results.
"""
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_BASE_URL = "http://localhost:11434"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

SWEEP_MODELS = ["gemma4:31b", "gemma4:26b"]
CTX_VALUES = [2048, 4096, 8192, 16384, 262144]
SWEEP_REPS = 2

# Exact same 5 prompts as exp_023, for Part 2 comparability
FULL_MODELS = ["gemma4:e4b", "gemma4:26b", "gemma4:31b", "qwen3.5:35b", "qwen3.6:35b"]
PROMPTS = [
    "Explain how a hash table resolves collisions, comparing chaining and open addressing.",
    "Write a function that finds the longest palindromic substring in a string, and explain its time complexity.",
    "Explain the difference between TCP and UDP, and give one real-world scenario where each is the better choice.",
    "A user reports that their API returns 200 OK but the response body is empty about 1% of the time under load. Walk through how you'd debug this.",
    "Explain what a Bloom filter is, how it works, and one real-world use case.",
]


def call(model: str, prompt: str, num_ctx: int, num_predict: int = None) -> dict:
    options = {"temperature": 0.2, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": options,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def derive(resp: dict) -> dict:
    eval_count = resp.get("eval_count", 0)
    eval_duration = resp.get("eval_duration", 1)
    prompt_eval_count = resp.get("prompt_eval_count", 0)
    prompt_eval_duration = resp.get("prompt_eval_duration", 1)
    return {
        "gen_tok_s": round(eval_count / (eval_duration / 1e9), 3) if eval_duration else None,
        "prompt_tok_s": round(prompt_eval_count / (prompt_eval_duration / 1e9), 3) if prompt_eval_duration else None,
        "wall_clock_s": round(resp.get("total_duration", 0) / 1e9, 2),
        "load_duration_s": round(resp.get("load_duration", 0) / 1e9, 2),
        "tokens_per_answer": eval_count,
        "prompt_tokens": prompt_eval_count,
    }


def part1_sweep():
    print("\n" + "=" * 60)
    print("PART 1 — context sweep: gemma4:31b vs gemma4:26b (control)")
    print("=" * 60)
    results = []
    for model in SWEEP_MODELS:
        for ctx in CTX_VALUES:
            reps = []
            for rep in range(SWEEP_REPS):
                try:
                    resp = call(model, "say hi", num_ctx=ctx, num_predict=10)
                    m = derive(resp)
                    reps.append(m)
                    print(f"  {model:12s} ctx={ctx:7d} rep{rep}: gen={m['gen_tok_s']} tok/s "
                          f"prompt={m['prompt_tok_s']} tok/s load={m['load_duration_s']}s")
                except Exception as e:
                    reps.append({"error": str(e)})
                    print(f"  {model:12s} ctx={ctx:7d} rep{rep}: ERROR {e}")
            results.append({"model": model, "num_ctx": ctx, "reps": reps})
    out = RESULTS_DIR / f"sweep_{TS}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return results


def part2_corrected():
    print("\n" + "=" * 60)
    print("PART 2 — corrected 5-model re-run, num_ctx=4096 fixed")
    print("=" * 60)
    all_runs = []
    summary = {}
    for model in FULL_MODELS:
        print(f"\n=== {model} ===")
        try:
            call(model, "Say ready.", num_ctx=4096)  # untimed warm-up
        except Exception as e:
            print(f"  [ERROR] warm-up failed: {e}")
            continue
        model_runs = []
        for idx, prompt in enumerate(PROMPTS):
            try:
                resp = call(model, prompt, num_ctx=4096)
                m = derive(resp)
                run = {"model": model, "prompt_idx": idx, **m}
                print(f"  [{idx}] gen={m['gen_tok_s']} tok/s tokens={m['tokens_per_answer']} "
                      f"wall_clock={m['wall_clock_s']}s")
            except Exception as e:
                run = {"model": model, "prompt_idx": idx, "error": str(e)}
                print(f"  [{idx}] ERROR: {e}")
            model_runs.append(run)
            all_runs.append(run)
        valid = [r for r in model_runs if "error" not in r]
        if valid:
            summary[model] = {
                "n_runs": len(valid),
                "mean_gen_tok_s": round(statistics.mean(r["gen_tok_s"] for r in valid), 2),
                "mean_wall_clock_s": round(statistics.mean(r["wall_clock_s"] for r in valid), 2),
                "mean_tokens_per_answer": round(statistics.mean(r["tokens_per_answer"] for r in valid), 1),
            }
    result = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "num_ctx_fixed": 4096, "runs": all_runs, "summary": summary}
    out = RESULTS_DIR / f"corrected_{TS}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print("\n=== SUMMARY (num_ctx=4096 fixed) ===")
    for model, s in summary.items():
        print(f"  {model:15s} gen={s['mean_gen_tok_s']:7.2f} tok/s  wall={s['mean_wall_clock_s']:6.1f}s  "
              f"tokens/answer={s['mean_tokens_per_answer']:6.1f}")
    return result


if __name__ == "__main__":
    part1_sweep()
    part2_corrected()
