#!/usr/bin/env python3
"""exp_052: send every frozen question to one answering backend, as the live bot would, and time it.

  G  Ollama /api/chat, gemma4:26b, exactly the bot's call: {"temperature": 0.3}, no think flag (Ollama's default:
     thinking on), the daemon's context (8192), plus a fixed seed
  M  mlx-lm server /v1/chat/completions, mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit, temperature 0.3,
     enable_thinking true, max_tokens 6144

Both arms keep the bot's failure behaviour: a 120 s timeout, one retry after 3 s, then the bot's error message.
The time recorded is what the user waits: from the first request to the final answer, retries included.
After every question the Ollama log is read: a POST that is not this question's own marks it contaminated, and
it is asked again once. Questions run in one fixed shuffled order (seed 52), the same in both arms.

    python3.14 -I kit/arms52.py --arm G --frozen data/frozen.jsonl --out runs/<ts>/G [--first N]
"""
import argparse
import json
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA, OMODEL = "http://localhost:11434", "gemma4:26b"
MLX, MMODEL = "http://127.0.0.1:8090", "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"
LOG = Path("/opt/homebrew/var/log/ollama.log")
TIMEOUT_S, RETRY_SLEEP_S, TEMP, SEED, MAX_TOKENS = 120, 3, 0.3, 52, 6144
LLM_ERROR = "__LLM_ERROR__"          # the bot shows LLM_ERROR_MSG[lang] here


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def call_g(messages: list) -> dict:
    r = post(f"{OLLAMA}/api/chat", {"model": OMODEL, "messages": messages, "stream": False,
                                    "options": {"temperature": TEMP, "seed": SEED}})
    m = r["message"]
    return {"answer": m["content"].strip(), "thinking_chars": len(m.get("thinking") or ""),
            "prompt_tokens": r.get("prompt_eval_count"), "gen_tokens": r.get("eval_count"),
            "prefill_s": (r.get("prompt_eval_duration") or 0) / 1e9, "gen_s": (r.get("eval_duration") or 0) / 1e9,
            "finish": r.get("done_reason")}


def call_m(messages: list) -> dict:
    r = post(f"{MLX}/v1/chat/completions", {"model": MMODEL, "messages": messages, "temperature": TEMP,
                                            "max_tokens": MAX_TOKENS,
                                            "chat_template_kwargs": {"enable_thinking": True}})
    c = r["choices"][0]
    u = r.get("usage") or {}
    return {"answer": (c["message"].get("content") or "").strip(),
            "thinking_chars": len(c["message"].get("reasoning") or ""),
            "prompt_tokens": u.get("prompt_tokens"), "gen_tokens": u.get("completion_tokens"),
            "prefill_s": None, "gen_s": None, "finish": c.get("finish_reason")}


def bot_call(fn, messages: list) -> dict:
    """The bot's generate_response loop: two attempts, 3 s apart, then the error message."""
    t0, errors = time.time(), []
    for attempt in (1, 2):
        try:
            res = fn(messages)
            return {**res, "attempts": attempt, "errors": errors, "user_wait_s": time.time() - t0}
        except Exception as e:                       # timeout, HTTP error, bad JSON
            errors.append(f"{type(e).__name__}: {e}"[:300])
            if attempt == 1:
                time.sleep(RETRY_SLEEP_S)
    return {"answer": LLM_ERROR, "attempts": 2, "errors": errors, "user_wait_s": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["G", "M"])
    ap.add_argument("--frozen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first", type=int, default=0, help="only the first N in the fixed order (drift check)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in map(json.loads, open(a.frozen)) if r["static"] is None]
    random.Random(SEED).shuffle(rows)
    if a.first:
        rows = rows[:a.first]
    fn = call_g if a.arm == "G" else call_m
    own = 1 if a.arm == "G" else 0
    print(json.dumps({"experiment": "052", "arm": a.arm, "n": len(rows), "started": now()}), flush=True)
    with open(out / "answers.jsonl", "a") as f:
        for i, r in enumerate(rows, 1):
            for attempt in (1, 2):
                off, t0 = LOG.stat().st_size, now()
                res = bot_call(fn, r["messages"])
                t1 = now()
                time.sleep(0.5)
                with open(LOG, "rb") as lf:
                    lf.seek(off)
                    posts = [ln for ln in lf.read().decode("utf-8", "replace").splitlines()
                             if ln.startswith("[GIN]") and "POST" in ln]
                contaminated = len(posts) > own * res["attempts"]
                rec = {"id": r["id"], "arm": a.arm, "order": i, "try": attempt, "start": t0, "end": t1,
                       "contaminated": contaminated, "ollama_posts": posts, "prefix": r["prefix"], **res}
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
