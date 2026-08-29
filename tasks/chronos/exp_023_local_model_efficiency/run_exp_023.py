#!/usr/bin/env python3
"""
Exp 023 — Generation efficiency across the local model family.

Modeled on terminalbytes.com's Qwen3.8-vs-Qwen3.6 methodology (5 timed
generations per model, varied technical prompts, tok/s + wall-clock + token
economy), applied to the five models already pulled and used somewhere in
this project rather than any newly-downloaded model.

Uses the Ollama HTTP API directly (not `ollama run --verbose` CLI text) —
the API's JSON response carries the identical timing fields `--verbose`
prints, without needing to parse human-readable output.
"""
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

OLLAMA_BASE_URL = "http://100.100.251.84:11434"

MODELS = ["gemma4:e4b", "gemma4:26b", "gemma4:31b", "qwen3.5:35b", "qwen3.6:35b"]

PROMPTS = [
    "Explain how a hash table resolves collisions, comparing chaining and open addressing.",
    "Write a function that finds the longest palindromic substring in a string, and explain its time complexity.",
    "Explain the difference between TCP and UDP, and give one real-world scenario where each is the better choice.",
    "A user reports that their API returns 200 OK but the response body is empty about 1% of the time under load. Walk through how you'd debug this.",
    "Explain what a Bloom filter is, how it works, and one real-world use case.",
]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ollama_chat_raw(model: str, prompt: str) -> dict:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def derive_metrics(resp: dict) -> dict:
    eval_count = resp.get("eval_count", 0)
    eval_duration = resp.get("eval_duration", 1)  # ns
    prompt_eval_count = resp.get("prompt_eval_count", 0)
    prompt_eval_duration = resp.get("prompt_eval_duration", 1)  # ns
    total_duration = resp.get("total_duration", 0)  # ns
    load_duration = resp.get("load_duration", 0)  # ns

    return {
        "gen_tok_s": round(eval_count / (eval_duration / 1e9), 2) if eval_duration else None,
        "prompt_tok_s": round(prompt_eval_count / (prompt_eval_duration / 1e9), 2) if prompt_eval_duration else None,
        "wall_clock_s": round(total_duration / 1e9, 2),
        "load_duration_s": round(load_duration / 1e9, 2),
        "tokens_per_answer": eval_count,
    }


def main():
    all_runs = []
    summary = {}

    for model in MODELS:
        print(f"\n=== {model} ===")

        print("  warm-up (untimed, forces model into memory)...")
        t0 = time.time()
        try:
            ollama_chat_raw(model, "Say ready.")
        except Exception as e:
            print(f"  [ERROR] warm-up failed for {model}: {e}")
            continue
        print(f"  warm-up took {time.time() - t0:.1f}s wall-clock (includes model load)")

        model_runs = []
        for idx, prompt in enumerate(PROMPTS):
            try:
                resp = ollama_chat_raw(model, prompt)
                metrics = derive_metrics(resp)
                reply = resp.get("message", {}).get("content", "")
                run = {
                    "model": model,
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "reply_excerpt": reply[:200],
                    **metrics,
                }
                print(f"  [{idx}] gen={metrics['gen_tok_s']} tok/s  "
                      f"tokens={metrics['tokens_per_answer']}  "
                      f"wall_clock={metrics['wall_clock_s']}s")
            except Exception as e:
                run = {"model": model, "prompt_idx": idx, "prompt": prompt, "error": str(e)}
                print(f"  [{idx}] ERROR: {e}")
            model_runs.append(run)
            all_runs.append(run)

        valid = [r for r in model_runs if "error" not in r]
        if valid:
            summary[model] = {
                "n_runs": len(valid),
                "mean_gen_tok_s": round(statistics.mean(r["gen_tok_s"] for r in valid), 2),
                "mean_prompt_tok_s": round(statistics.mean(r["prompt_tok_s"] for r in valid), 2),
                "mean_wall_clock_s": round(statistics.mean(r["wall_clock_s"] for r in valid), 2),
                "mean_tokens_per_answer": round(statistics.mean(r["tokens_per_answer"] for r in valid), 1),
                "range_gen_tok_s": [round(min(r["gen_tok_s"] for r in valid), 2),
                                    round(max(r["gen_tok_s"] for r in valid), 2)],
                "range_wall_clock_s": [round(min(r["wall_clock_s"] for r in valid), 2),
                                       round(max(r["wall_clock_s"] for r in valid), 2)],
            }

    # ── Verdicts ──────────────────────────────────────────────────────────
    verdicts = {}

    if "gemma4:e4b" in summary:
        e4b_speed = summary["gemma4:e4b"]["mean_gen_tok_s"]
        others_faster = [m for m in summary if m != "gemma4:e4b" and summary[m]["mean_gen_tok_s"] >= e4b_speed]
        verdicts["H1"] = "REFUTED" if others_faster else "CONFIRMED"
        verdicts["H1_detail"] = f"gemma4:e4b={e4b_speed} tok/s; models matching/beating it: {others_faster or 'none'}"

    if "qwen3.5:35b" in summary and "qwen3.6:35b" in summary:
        s5, s6 = summary["qwen3.5:35b"], summary["qwen3.6:35b"]
        speed_order_matches_wallclock_order = (
            (s5["mean_gen_tok_s"] < s6["mean_gen_tok_s"]) == (s5["mean_wall_clock_s"] < s6["mean_wall_clock_s"])
        )
        verdicts["H2"] = "REFUTED" if speed_order_matches_wallclock_order else "CONFIRMED"
        verdicts["H2_detail"] = (
            f"qwen3.5:35b gen={s5['mean_gen_tok_s']} tok/s wall={s5['mean_wall_clock_s']}s tokens={s5['mean_tokens_per_answer']}; "
            f"qwen3.6:35b gen={s6['mean_gen_tok_s']} tok/s wall={s6['mean_wall_clock_s']}s tokens={s6['mean_tokens_per_answer']}"
        )

    sizes_gb = {"gemma4:e4b": 9.6, "gemma4:26b": 17, "gemma4:31b": 19, "qwen3.5:35b": 23, "qwen3.6:35b": 23}
    present = [m for m in MODELS if m in summary]
    by_size = sorted(present, key=lambda m: sizes_gb[m])
    by_wallclock = sorted(present, key=lambda m: summary[m]["mean_wall_clock_s"])
    verdicts["H3"] = "REFUTED" if by_size == by_wallclock else "CONFIRMED"
    verdicts["H3_detail"] = f"by_size_order={by_size}; by_wallclock_order={by_wallclock}"

    result = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hardware": "miktam-mini M4 Pro 64GB",
        "prompts": PROMPTS,
        "runs": all_runs,
        "summary": summary,
        "verdicts": verdicts,
    }

    out_path = RESULTS_DIR / f"run_{TS}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

    print("\n=== SUMMARY ===")
    for model, s in summary.items():
        print(f"  {model:15s} gen={s['mean_gen_tok_s']:6.1f} tok/s  "
              f"prompt={s['mean_prompt_tok_s']:7.1f} tok/s  "
              f"wall_clock={s['mean_wall_clock_s']:6.1f}s  "
              f"tokens/answer={s['mean_tokens_per_answer']:6.1f}")

    print("\n=== VERDICTS ===")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
