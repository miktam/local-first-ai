#!/usr/bin/env python3
"""exp_060: three fixes for the local reader's empty answers, on exp_055's corpus, read in place.

  T  think: false; the context stays the daemon's 8,192
  F  exp_055's G call unchanged (no think key = thinking on, 8,192); when no parsable band comes back, one retry of
     the same prompt with think: false
  C  options.num_ctx 32768, thinking on; nothing else changed

Every call records the parsed band and items, the full content, the full thinking text (thinking.jsonl), done_reason,
prompt_eval_count, eval_count, the durations and wall-clock seconds. Arm F keeps pass 1 and the retry apart
(pass1.jsonl, retry.jsonl); its results.jsonl holds the final answer, with the message's total wall-clock.
The daemon's state (ollama --version, ollama ps, /api/ps) is written before, during (after the first call) and after
each arm; an arm that raised num_ctx ends with one dummy call at 8,192 and records the state again (daemon_restored).

    python3.14 kit/arms60.py --all --run runs/<ts>             T, F, C in order, each under exp_054's audit hook
    python3.14 kit/arms60.py --arm T --out runs/<ts>/T         one arm
    python3.14 kit/arms60.py --arm F --out dev/<ts>/F --dummy  smoke test on kit/dummy60.eml, not a corpus message

Nothing leaves the machine: the only network act is the call to 127.0.0.1:11434.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXP55 = HERE.parent / "exp_055_exposure_review_lures"
HOOK = HERE.parent / "exp_054_exposure_review_egress/kit/hook"
sys.path.insert(0, str(EXP55 / "kit"))
import arms55  # noqa: E402  (exp_055's prompt builder and parser, unchanged)

MODEL = "gemma4:26b"
HOST = "http://127.0.0.1:11434"
ARMS = {  # think None = no think key in the request (Ollama's default for this model: thinking on)
    "T": {"think": False, "num_ctx": None, "timeout": 600, "fallback": False},
    "F": {"think": None, "num_ctx": None, "timeout": 600, "fallback": True},
    "C": {"think": None, "num_ctx": 32768, "timeout": 900, "fallback": False},
}
RETRY = {"think": False, "num_ctx": None, "timeout": 600}
RESTORE_CTX = 8192
DUMMY = {"D001": {"file": "kit/dummy60.eml", "kind": "dummy", "channel": "email", "to_item": "C-526e13",
                  "domains": {"harborlight-speakers.example": "2026-10-20"}, "used_items": [], "ask": "click",
                  "tactic": "dummy: speaker-portal confirmation on a 5-day-old domain"}}

COUNTERS = ("done_reason", "prompt_eval_count", "eval_count", "total_duration", "load_duration",
            "prompt_eval_duration", "eval_duration")


def build_prompt(m: dict, text: str) -> str:
    """exp_055's prompt, byte for byte (asserted against arms55.prompt_for on corpus messages)."""
    footprint = "\n".join((EXP55 / "data/FOOTPRINT.txt").read_text().splitlines()[1:])
    return (EXP55 / "kit/prompt55.md").read_text().format(
        footprint=footprint, registry=arms55.registry_facts(m), channel=m["channel"], to=arms55.to_label(m),
        message=text)


def prompt_for(m: dict, dummy: bool) -> str:
    if dummy:
        return build_prompt(m, (HERE / m["file"]).read_text(encoding="utf-8"))
    p = build_prompt(m, arms55.message_text(m))
    assert p == arms55.prompt_for(m), "prompt differs from exp_055's"
    return p


def call(prompt: str, think, num_ctx, timeout: int):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0, "seed": 55}}
    if think is not None:
        body["think"] = think
    if num_ctx:
        body["options"]["num_ctx"] = num_ctx
    req = urllib.request.Request(HOST + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    rec = {"band": None, "used_items": [], "why": None, "parsed": False, "content": None, "error": None}
    thinking = ""
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        msg = r.get("message") or {}
        content = msg.get("content") or ""
        thinking = msg.get("thinking") or ""
        d = arms55.parse_json(content)
        rec.update({"band": d["band"] if d else None, "used_items": list((d or {}).get("used_items") or []),
                    "why": (d or {}).get("why"), "parsed": bool(d), "content": content})
        rec.update({k: r.get(k) for k in COUNTERS})
    except urllib.error.HTTPError as e:
        rec["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # timeouts included
        rec["error"] = f"{type(e).__name__}: {e}"
    rec.update({"seconds": round(time.time() - t0, 2), "think": think, "num_ctx": num_ctx,
                "thinking_chars": len(thinking)})
    return rec, thinking


def _ollama(*args) -> str:
    try:
        return subprocess.run(["ollama", *args], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def daemon_state(path: Path):
    try:
        ps = json.loads(urllib.request.urlopen(HOST + "/api/ps", timeout=30).read())
    except Exception as e:
        ps = {"error": f"{type(e).__name__}: {e}"}
    state = {"t": time.time(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "ollama_version": _ollama("--version").strip(), "ollama_ps": _ollama("ps"), "api_ps": ps}
    path.write_text(json.dumps(state, indent=1) + "\n")
    ctx = {m.get("name"): m.get("context_length") for m in (ps.get("models") or []) if isinstance(ps, dict)}
    print(f"daemon {path.name}: {ctx}", flush=True)
    return state


def append(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def restore(out: Path):
    """Leave the daemon as found: one dummy call at the production context, then record the state."""
    rec, _ = call("Reply with the single word: ok", False, RESTORE_CTX, 300)
    append(out / "restore_call.jsonl", rec)
    daemon_state(out / "daemon_restored.json")


def run_arm(arm: str, out: Path, dummy: bool):
    cfg = ARMS[arm]
    out.mkdir(parents=True, exist_ok=True)
    manifest = DUMMY if dummy else json.loads((EXP55 / "data/MANIFEST.json").read_text())
    res_path = out / "results.jsonl"
    done = {json.loads(l)["id"] for l in open(res_path)} if res_path.exists() else set()
    daemon_state(out / "daemon_before.json")
    first = True
    for mid in sorted(manifest):
        if mid in done:
            continue
        m = manifest[mid]
        prompt = prompt_for(m, dummy)
        rec, thinking = call(prompt, cfg["think"], cfg["num_ctx"], cfg["timeout"])
        if cfg["fallback"]:
            append(out / "pass1.jsonl", {"id": mid, "pass": "pass1", **rec})
            append(out / "thinking.jsonl", {"id": mid, "pass": "pass1", "thinking": thinking})
            if rec["band"] is None:
                rec2, thinking2 = call(prompt, RETRY["think"], RETRY["num_ctx"], RETRY["timeout"])
                append(out / "retry.jsonl", {"id": mid, "pass": "retry", **rec2})
                append(out / "thinking.jsonl", {"id": mid, "pass": "retry", "thinking": thinking2})
                final = {**rec2, "fallback": True, "pass1_done_reason": rec.get("done_reason"),
                         "pass1_eval_count": rec.get("eval_count"), "pass1_seconds": rec["seconds"],
                         "seconds": round(rec["seconds"] + rec2["seconds"], 2)}
            else:
                final = {**rec, "fallback": False}
        else:
            append(out / "thinking.jsonl", {"id": mid, "pass": arm, "thinking": thinking})
            final = rec
        append(res_path, {"id": mid, **final})
        print(arm, mid, final["band"], final["seconds"], final.get("done_reason"), final.get("eval_count"),
              "fallback" if final.get("fallback") else "", final["error"] or "", flush=True)
        if first:
            daemon_state(out / "daemon_during.json")
            first = False
    daemon_state(out / "daemon_after.json")
    if cfg["num_ctx"]:
        restore(out)
    print("done", arm, out, flush=True)


def run_all(run: Path, dummy: bool):
    run.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
            "python": sys.version, "ollama_version": _ollama("--version").strip(), "ollama_list": _ollama("list"),
            "macos": _sw_vers(), "arms": {}}
    for arm in ("T", "F", "C"):
        out = run / arm
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(HOOK), "AUDIT54_LOG": str(out / "audit.jsonl")}
        argv = [sys.executable, __file__, "--arm", arm, "--out", str(out)] + (["--dummy"] if dummy else [])
        t0 = time.time()
        rc = subprocess.run(argv, env=env).returncode
        meta["arms"][arm] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("all done", run, meta["arms"], flush=True)


def _sw_vers() -> str:
    try:
        return subprocess.run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS))
    ap.add_argument("--out")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--dummy", action="store_true", help="kit/dummy60.eml only; never a corpus message")
    a = ap.parse_args()
    if a.all:
        if not a.run:
            ap.error("--all needs --run <dir>")
        run_all(Path(a.run), a.dummy)
    elif a.arm and a.out:
        run_arm(a.arm, Path(a.out), a.dummy)
    else:
        ap.error("either --all --run <dir>, or --arm X --out <dir>")


if __name__ == "__main__":
    main()
