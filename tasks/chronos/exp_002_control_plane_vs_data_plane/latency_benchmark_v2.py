import requests
import json
import time
import os

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma4:26b"
LOG_FILE = "tasks/chronos/experiments/latency_benchmark_v2.jsonl"

# Define the modes, including the new Edge Case
MODES = {
    "Assembly Line": {
        "reasoning": False,
        "think": False,
        "prompt_template": "Extract the keys from this JSON as a comma-separated list: %s"
    },
    "Auditor": {
        "reasoning": True,
        "think": False,
        "prompt_template": "Analyze this JSON structure for any potential security vulnerabilities or schema anomalies. List your findings: %s"
    },
    "Architect": {
        "reasoning": True,
        "think": True,
        "prompt_template": "Design a high-level architecture for a service that processes this JSON data at scale, considering the following schema: %s"
    },
    "Architect (Edge Case)": {
        "reasoning": True,
        "think": True,
        "prompt_template": "Analyze the following JSON. Then, formulate a mathematically flawless, recursive algorithm in Python to sort its keys. The explanation of this algorithm must be exactly 341 words long, and the logic must intentionally contradict its own base case twice before resolving. Prove it works step-by-step: %s"
    }
}

# Test data
TEST_PAYLOAD = {
    "user_id": "12345",
    "transaction_type": "transfer",
    "amount": 500.0,
    "currency": "EUR",
    "metadata": {
        "ip_address": "192.168.1.1",
        "device": "mobile",
        "location": "Madrid"
    },
    "tags": ["finance", "automated", "high-priority"]
}

def warmup_model():
    """Forces the model to load into memory to prevent Cold Start anomalies."""
    print("🔥 Warming up model to prevent cold-start anomaly...", end="", flush=True)
    payload = {
        "model": MODEL, 
        "prompt": "Respond with 'ready'.", 
        "stream": False
    }
    try:
        requests.post(OLLAMA_URL, json=payload, timeout=120)
        print(" ✅")
    except Exception as e:
        print(f" ❌ ({str(e)})")

def run_benchmark(mode_name, config):
    prompt = config["prompt_template"] % json.dumps(TEST_PAYLOAD)

    payload_api = {
        "model": MODEL,
        "prompt": prompt,
        "options": {"think": config["think"]},
        "stream": False
    }

    print(f"  Running {mode_name} (Reasoning: {config['reasoning']}, Think: {config['think']})...", end="", flush=True)

    start_time = time.time()
    result_entry = {
        "mode": mode_name,
        "reasoning_enabled": config["reasoning"],
        "think_enabled": config["think"],
        "timestamp": time.ctime()
    }

    try:
        # Increased timeout to 1200s (20 mins) to accommodate the Edge Case
        response = requests.post(OLLAMA_URL, json=payload_api, timeout=1200)
        response.raise_for_status()
        result_data = response.json()
        end_time = time.time()

        latency = end_time - start_time
        tokens = result_data.get("eval_count", 0)
        tps = tokens / latency if latency > 0 else 0

        result_entry.update({
            "success": True,
            "latency_seconds": round(latency, 2),
            "tokens_generated": tokens,
            "throughput_tps": round(tps, 2),
            "content_length": len(result_data.get("response", "")),
            "error": None
        })
        print(f" ✅ ({round(latency, 1)}s, {tokens} tokens, {round(tps, 1)} t/s)")
    except Exception as e:
        result_entry.update({
            "success": False,
            "latency_seconds": round(time.time() - start_time, 2),
            "tokens_generated": 0,
            "throughput_tps": 0.0,
            "content_length": 0,
            "error": str(e)
        })
        print(f" ❌ ({str(e)})")

    return result_entry

def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print(f"🚀 Starting Chronos Latency Benchmark: The Three Modes + Edge Case")
    print(f"📝 Logging to: {LOG_FILE}\n")

    warmup_model()

    results = []
    with open(LOG_FILE, "a") as f:
        for name, config in MODES.items():
            result = run_benchmark(name, config)
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()

    print(f"\n✅ Benchmark Complete.")
    print(f"📊 Results Summary:")
    
    # Print a clean, markdown-ready table to the console
    print(f"{'Mode':<25} | {'Latency (s)':<12} | {'Tokens':<8} | {'Throughput (t/s)'}")
    print("-" * 65)
    for r in results:
        status = "✅" if r['success'] else "❌"
        if r['success']:
            print(f"{status} {r['mode']:<22} | {r['latency_seconds']:<12} | {r['tokens_generated']:<8} | {r['throughput_tps']}")
        else:
            print(f"{status} {r['mode']:<22} | FAILED: {r['error']}")

if __name__ == "__main__":
    main()
