#!/usr/bin/env python3
"""exp_051: prefill and generation speed of gemma 4 26B-A4B in Ollama and in MLX on the Mac mini, on three texts.

One arm per invocation:
  O   Ollama 0.40.2, gemma4:26b (GGUF Q4_K_M, llama.cpp runner), production settings, on exp_007's fixtures
  X   mlx-lm, mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit (exp_011's build), on exp_007's fixtures
  F   as X, on exp_011's fallback text (one sentence repeated; what exp_011 actually measured)
  ON  as O, on natural text (published synthetic enquiries, kit/build_natural.py)
  XN  as X, on natural text

Every call: user message = "Request <nonce>.\n\n" + text + instruction, through the runtime's chat template with
thinking off, greedy, at most 128 new tokens. The nonce is new for every call, so no prompt cache can be reused.
After every call the Ollama log is read: a POST that is not this call's own marks the call contaminated, and the
call is repeated (at most twice). In the MLX arms no Ollama model may be loaded; one found loaded is stopped.

    HF_HUB_OFFLINE=1 python3.14 -I kit/bench51.py --arm X --out runs/<ts>/X
"""
import argparse
import json
import statistics
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIX = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_007_hardware_comparison/fixtures/padding"
NAT = HERE / "natural"
TARGET = {"4k": 4000, "8k": 8000, "15k": 15000, "20k": 20000, "25k": 25000, "30k": 30000, "35k": 35000, "40k": 40000}
SIZES = {"O": list(TARGET), "X": list(TARGET), "F": list(TARGET), "ON": ["4k", "15k", "25k", "35k"],
         "XN": ["4k", "15k", "25k", "35k"]}
REPS, IDLE_S, MAX_NEW, MAX_REDO = 3, 20, 128, 2
INSTRUCTION = "\n\nWrite a one-sentence summary of the text above."
OLLAMA, OMODEL, NUM_CTX = "http://localhost:11434", "gemma4:26b", 65536
MLX_ID = "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"
LOG = Path("/opt/homebrew/var/log/ollama.log")


def fallback(target: int) -> str:
    """exp_011's load_fixture fallback, verbatim."""
    phrase = "The quick brown fox jumps over the lazy dog. " * 20
    return (phrase * (target // 10))[:target * 4]


def text_for(arm: str, key: str) -> str:
    if arm == "F":
        return fallback(TARGET[key])
    if arm in ("ON", "XN"):
        return (NAT / f"nat_{key}.txt").read_text()
    return (FIX / f"pad_{key}.txt").read_text()


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def log_offset() -> int:
    return LOG.stat().st_size


def posts_since(offset: int) -> list:
    with open(LOG, "rb") as f:
        f.seek(offset)
        data = f.read().decode("utf-8", "replace")
    return [ln for ln in data.splitlines() if ln.startswith("[GIN]") and "POST" in ln]


def http(path: str, payload: dict | None = None, timeout: int = 3600) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def loaded_models() -> list:
    return [m["name"] for m in http("/api/ps").get("models", [])]


def stop_all_ollama() -> list:
    stopped = []
    for name in loaded_models():
        subprocess.run(["ollama", "stop", name], capture_output=True, timeout=60)
        stopped.append(name)
    return stopped


class OllamaArm:
    runtime = "ollama"

    def __init__(self):
        v = http("/api/version")["version"]
        self.meta = {"runtime": "ollama", "ollama_version": v, "model": OMODEL, "num_ctx": NUM_CTX,
                     "settings_documented": "OLLAMA_FLASH_ATTENTION=0, default fp16 KV cache (production since 2026-08-30)"}
        for name in loaded_models():                  # only gemma4:26b may be resident
            if name != OMODEL:
                subprocess.run(["ollama", "stop", name], capture_output=True, timeout=60)
        self.call("Warm-up. Reply with one word.")    # loads at NUM_CTX; not measured

    def call(self, msg: str) -> dict:
        r = http("/api/chat", {"model": OMODEL, "messages": [{"role": "user", "content": msg}], "stream": False,
                               "think": False, "keep_alive": -1,
                               "options": {"temperature": 0, "num_predict": MAX_NEW, "num_ctx": NUM_CTX, "seed": 51}})
        pc, pd = r.get("prompt_eval_count") or 0, r.get("prompt_eval_duration") or 0
        ec, ed = r.get("eval_count") or 0, r.get("eval_duration") or 0
        return {"prompt_tokens": pc, "prefill_s": pd / 1e9, "prefill_ms_per_tok": (pd / 1e6) / pc if pc else None,
                "gen_tokens": ec, "gen_tps": ec / (ed / 1e9) if ec and ed else None,
                "load_s": (r.get("load_duration") or 0) / 1e9, "finish": r.get("done_reason"),
                "output": (r.get("message") or {}).get("content", "")[:400]}

    def own_posts(self) -> int:
        return 1


class MlxArm:
    runtime = "mlx"

    def __init__(self):
        import mlx.core as mx
        import mlx_lm
        from mlx_lm import load
        self.mx = mx
        self.stopped_before = stop_all_ollama()
        self.model, self.tok = load(MLX_ID)
        self.meta = {"runtime": "mlx", "mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
                     "model": MLX_ID, "kv_cache": "mlx-lm default (unquantised); the build's kv_config.json is not read",
                     "ollama_models_stopped_at_start": self.stopped_before}
        self.call("Warm-up. Reply with one word.")

    def call(self, msg: str) -> dict:
        from mlx_lm import stream_generate
        prompt = self.tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        self.mx.reset_peak_memory()
        last, out = None, []
        for r in stream_generate(self.model, self.tok, prompt=prompt, max_tokens=MAX_NEW):
            last = r
            out.append(r.text)
        return {"prompt_tokens": last.prompt_tokens, "prefill_s": last.prompt_tokens / last.prompt_tps,
                "prefill_ms_per_tok": 1000.0 / last.prompt_tps, "gen_tokens": last.generation_tokens,
                "gen_tps": last.generation_tps, "load_s": 0.0, "finish": last.finish_reason,
                "peak_memory_gb": round(self.mx.get_peak_memory() / 1e9, 2), "output": "".join(out)[:400]}

    def own_posts(self) -> int:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(SIZES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="*")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = a.sizes or SIZES[a.arm]
    arm = OllamaArm() if a.arm in ("O", "ON") else MlxArm()
    meta = {"experiment": "051", "arm": a.arm, "tag": a.tag, "started": now(), "sizes": sizes, "reps": REPS,
            "idle_s": IDLE_S, "max_new_tokens": MAX_NEW, "instruction": INSTRUCTION, **arm.meta}
    print(json.dumps(meta), flush=True)
    calls = out / "calls.jsonl"
    summary = []
    for key in sizes:
        text = text_for(a.arm, key)
        time.sleep(IDLE_S)
        reps = []
        for rep in range(1, REPS + 1):
            for attempt in range(1, MAX_REDO + 2):
                stopped = stop_all_ollama() if arm.runtime == "mlx" else []
                nonce = uuid.uuid4().hex[:8]
                off, t0 = log_offset(), now()
                m = arm.call(f"Request {nonce}.\n\n{text}{INSTRUCTION}")
                t1 = now()
                time.sleep(0.5)                               # let the log line land
                posts = posts_since(off)
                contaminated = len(posts) > arm.own_posts()
                rec = {"arm": a.arm, "size": key, "rep": rep, "attempt": attempt, "nonce": nonce, "start": t0,
                       "end": t1, "text_chars": len(text), "contaminated": contaminated,
                       "ollama_posts": posts, "ollama_stopped_before": stopped, **m}
                with open(calls, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{a.arm} {key} rep{rep} try{attempt} tok={m['prompt_tokens']} "
                      f"prefill={m['prefill_ms_per_tok']:.3f} ms/tok gen={m['gen_tps'] or 0:.2f} t/s "
                      f"n={m['gen_tokens']}{' CONTAMINATED' if contaminated else ''}", flush=True)
                if not contaminated:
                    break
            reps.append(rec)
        clean = [r for r in reps if not r["contaminated"]]
        summary.append({"size": key, "prompt_tokens": statistics.median(r["prompt_tokens"] for r in reps),
                        "prefill_ms_per_tok_median": statistics.median(r["prefill_ms_per_tok"] for r in clean) if clean else None,
                        "gen_tps_median": statistics.median(r["gen_tps"] for r in clean if r["gen_tps"]) if clean else None,
                        "clean_reps": len(clean)})
    meta["ended"] = now()
    (out / "summary.json").write_text(json.dumps({**meta, "sizes_summary": summary}, indent=2) + "\n")
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
