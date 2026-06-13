#!/usr/bin/env python3
"""
Exp 016 Phase B — LAN routing overhead benchmark.

Sends 10 identical short prompts to both:
  - local:  http://localhost:11434/v1/chat/completions  (gemma4:26b via Ollama)
  - remote: http://miktam-mbp.local:8080/v1/chat/completions  (Qwen3-Coder-Next via MLX)

Measures TTFT (time to first token) using streaming.
Records median and p95 overhead. Writes phase_b_lan_latency.json.

Gate: median overhead < 500 ms  →  H4 confirmed, Phase C unblocked.
"""

import json
import statistics
import time
import urllib.request
from datetime import datetime, timezone

LOCAL_URL = "http://localhost:11434/v1/chat/completions"
REMOTE_URL = "http://miktam-mbp.local:8080/v1/chat/completions"

LOCAL_MODEL = "gemma4:26b"
REMOTE_MODEL = "mlx-community/Qwen3-Coder-Next-4bit"

PROMPT = "Return exactly one line: The quick brown fox jumps over the lazy dog."
N_REPS = 10
MAX_TOKENS = 32
TIMEOUT_S = 60


def ttft_streaming(url: str, model: str) -> dict:
    """Send a streaming chat completion request; return TTFT and total wall time."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    first_token_t = None
    chunks = 0

    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or delta.get("reasoning", "")
            if content and first_token_t is None:
                first_token_t = time.perf_counter()
            chunks += 1

    t_end = time.perf_counter()

    if first_token_t is None:
        raise RuntimeError(f"No content tokens received from {url}")

    return {
        "ttft_ms": round((first_token_t - t0) * 1000, 1),
        "wall_ms": round((t_end - t0) * 1000, 1),
        "chunks": chunks,
    }


def run_endpoint(label: str, url: str, model: str, n: int) -> list[dict]:
    results = []
    print(f"\n{'='*60}")
    print(f"  {label}  ({url})")
    print(f"  model: {model}  reps: {n}")
    print(f"{'='*60}")
    for i in range(1, n + 1):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        try:
            r = ttft_streaming(url, model)
            r.update({"rep": i, "ts": ts, "endpoint": label, "model": model, "ok": True})
            print(f"  rep {i:2d}  ttft={r['ttft_ms']:7.1f} ms  wall={r['wall_ms']:7.1f} ms")
        except Exception as exc:
            r = {"rep": i, "ts": ts, "endpoint": label, "model": model, "ok": False, "error": str(exc)}
            print(f"  rep {i:2d}  ERROR: {exc}")
        results.append(r)
    return results


def summarise(label: str, reps: list[dict]) -> dict:
    good = [r["ttft_ms"] for r in reps if r.get("ok")]
    if not good:
        return {"endpoint": label, "n_ok": 0, "n_fail": len(reps)}
    good_sorted = sorted(good)
    p95_idx = max(0, int(len(good_sorted) * 0.95) - 1)
    return {
        "endpoint": label,
        "n_ok": len(good),
        "n_fail": len(reps) - len(good),
        "ttft_ms_median": round(statistics.median(good), 1),
        "ttft_ms_p95": round(good_sorted[p95_idx], 1),
        "ttft_ms_min": round(min(good), 1),
        "ttft_ms_max": round(max(good), 1),
    }


def main():
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\nExp 016 Phase B — LAN routing overhead benchmark")
    print(f"Started: {run_ts}  reps: {N_REPS}  prompt: {PROMPT!r}")

    local_reps = run_endpoint("local", LOCAL_URL, LOCAL_MODEL, N_REPS)
    remote_reps = run_endpoint("remote", REMOTE_URL, REMOTE_MODEL, N_REPS)

    local_summary = summarise("local", local_reps)
    remote_summary = summarise("remote", remote_reps)

    overhead_ms = None
    verdict = "UNKNOWN"
    if local_summary.get("ttft_ms_median") and remote_summary.get("ttft_ms_median"):
        overhead_ms = round(remote_summary["ttft_ms_median"] - local_summary["ttft_ms_median"], 1)
        if overhead_ms < 500:
            verdict = "H4_CONFIRMED"
        elif overhead_ms < 1500:
            verdict = "H4_INCONCLUSIVE"
        else:
            verdict = "H4_FALSIFIED"

    output = {
        "experiment": "exp_016_phase_b",
        "run_ts": run_ts,
        "n_reps": N_REPS,
        "prompt": PROMPT,
        "local": {
            "url": LOCAL_URL,
            "model": LOCAL_MODEL,
            "summary": local_summary,
            "reps": local_reps,
        },
        "remote": {
            "url": REMOTE_URL,
            "model": REMOTE_MODEL,
            "summary": remote_summary,
            "reps": remote_reps,
        },
        "overhead_ms_median": overhead_ms,
        "verdict": verdict,
    }

    out_path = "measurements/phase_b_lan_latency.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Local   TTFT median: {local_summary.get('ttft_ms_median')} ms  (p95: {local_summary.get('ttft_ms_p95')} ms)")
    print(f"  Remote  TTFT median: {remote_summary.get('ttft_ms_median')} ms  (p95: {remote_summary.get('ttft_ms_p95')} ms)")
    print(f"  Overhead (median):   {overhead_ms} ms")
    print(f"  Verdict: {verdict}")
    print(f"\n  Written: {out_path}")


if __name__ == "__main__":
    main()
