#!/usr/bin/env python3
"""exp_053: send every frozen question to one Ollama build of gemma 4 26B-A4B, exactly as the live bot calls its
answering model, and time it.

  G4  gemma4:26b                (GGUF Q4_K_M, the live build)
  GQ  gemma4:26b-a4b-it-qat     (Google's quantisation-aware-trained 4-bit)
  G8  gemma4:26b-a4b-it-q8_0    (8-bit)

The call is the bot's: /api/chat, {"temperature": 0.3} plus a fixed seed, no think flag (Ollama's default for this
model: thinking on), the daemon's context (8192). The bot's failure behaviour is kept: a 120 s timeout, one retry
after 3 s, then the bot's error message. Only the build under test (and the router, gemma4:e4b) may be loaded:
other gemma4 26B builds are stopped first. After every question the Ollama log is read; a POST that is not this
question's own marks it contaminated, and it is asked again once. Fixed shuffled order, seed 53, same for all arms.

    python3.14 -I kit/arms53.py --arm G4 --frozen data/frozen.jsonl --out runs/<ts>/G4 [--first N]
"""
import argparse
import json
import random
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA = "http://localhost:11434"
BUILDS = {"G4": "gemma4:26b", "GQ": "gemma4:26b-a4b-it-qat", "G8": "gemma4:26b-a4b-it-q8_0"}
LOG = Path("/opt/homebrew/var/log/ollama.log")
TIMEOUT_S, RETRY_SLEEP_S, TEMP, SEED = 120, 3, 0.3, 53
LLM_ERROR = "__LLM_ERROR__"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def post(path: str, payload: dict, timeout: int = TIMEOUT_S) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def only_this_build_loaded(model: str) -> list:
    with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=10) as r:
        loaded = [m["name"] for m in json.loads(r.read()).get("models", [])]
    stopped = [n for n in loaded if n != model and n not in ("gemma4:e4b",)]
    for n in stopped:
        subprocess.run(["ollama", "stop", n], capture_output=True, timeout=60)
    return stopped


def call(model: str, messages: list) -> dict:
    r = post("/api/chat", {"model": model, "messages": messages, "stream": False,
                           "options": {"temperature": TEMP, "seed": SEED}})
    m = r["message"]
    return {"answer": m["content"].strip(), "thinking_chars": len(m.get("thinking") or ""),
            "prompt_tokens": r.get("prompt_eval_count"), "gen_tokens": r.get("eval_count"),
            "prefill_s": (r.get("prompt_eval_duration") or 0) / 1e9, "gen_s": (r.get("eval_duration") or 0) / 1e9,
            "finish": r.get("done_reason")}


def bot_call(model: str, messages: list) -> dict:
    """The bot's generate_response loop: two attempts, 3 s apart, then the error message."""
    t0, errors = time.time(), []
    for attempt in (1, 2):
        try:
            return {**call(model, messages), "attempts": attempt, "errors": errors, "user_wait_s": time.time() - t0}
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}"[:300])
            if attempt == 1:
                time.sleep(RETRY_SLEEP_S)
    return {"answer": LLM_ERROR, "attempts": 2, "errors": errors, "user_wait_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(BUILDS))
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    model = BUILDS[a.arm]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(a.frozen)) if r["static"] is None]
    random.Random(SEED).shuffle(rows)
    if a.first:
        rows = rows[:a.first]
    stopped = only_this_build_loaded(model)
    post("/api/generate", {"model": model, "keep_alive": -1}, timeout=600)       # load, not measured
    print(json.dumps({"experiment": "053", "arm": a.arm, "model": model, "n": len(rows), "stopped": stopped,
                      "started": now()}), flush=True)
    with open(out / "answers.jsonl", "a") as f:
        for i, r in enumerate(rows, 1):
            for attempt in (1, 2):
                off, t0 = LOG.stat().st_size, now()
                res = bot_call(model, r["messages"])
                t1 = now()
                time.sleep(0.5)
                with open(LOG, "rb") as lf:
                    lf.seek(off)
                    posts = [ln for ln in lf.read().decode("utf-8", "replace").splitlines()
                             if ln.startswith("[GIN]") and "POST" in ln]
                contaminated = len(posts) > res["attempts"]
                rec = {"id": r["id"], "arm": a.arm, "model": model, "order": i, "try": attempt, "start": t0,
                       "end": t1, "contaminated": contaminated, "ollama_posts": posts, "prefix": r["prefix"], **res}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{a.arm} {i}/{len(rows)} {r['id']} {res['user_wait_s']:.1f}s tok={res.get('gen_tokens')} "
                      f"think={res.get('thinking_chars')} fin={res.get('finish')}"
                      f"{' ERROR' if res['answer'] == LLM_ERROR else ''}{' CONTAMINATED' if contaminated else ''}",
                      flush=True)
                if not contaminated:
                    break
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
