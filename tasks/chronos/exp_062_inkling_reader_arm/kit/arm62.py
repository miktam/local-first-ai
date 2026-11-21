#!/usr/bin/env python3
"""exp_062: Inkling 2 Mini in the local reader's seat, reasoning on. One arm, on exp_055's corpus, read in place.

  I  exp_055's prompt as the single user message to a llama.cpp side server on 127.0.0.1:18450:
     temperature 0, seed 55, no reasoning parameter (the model's default), no output cap, context 8,192, one slot

The driver (--all) checks the frozen inputs and the model file, starts the server, runs the arm in a child process
under exp_054's audit hook, and stops the server. The child talks to the side server only. The reasoning text goes
to a private folder outside this repository; its sha256 is written to the run folder.

    python3.14 kit/arm62.py --all --run runs/<ts> --private <dir>            the measured run (Andrei starts it)
    python3.14 kit/arm62.py --all --run dev/<ts> --private <dir> --dummy     smoke test on exp_060's dummy message

Nothing here calls the production Ollama except `ollama ps`, which is read before and after.
"""
import argparse
import hashlib
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
EXP60 = HERE.parent / "exp_060_reader_empty_answers"
HOOK = HERE.parent / "exp_054_exposure_review_egress/kit/hook"
sys.path.insert(0, str(EXP55 / "kit"))
sys.path.insert(0, str(EXP60 / "kit"))
import arms55  # noqa: E402  (exp_055's prompt builder and parser, unchanged)
import arms60  # noqa: E402  (exp_060's byte-for-byte prompt check and its dummy message)

PORT = 18450
HOST = f"http://127.0.0.1:{PORT}"
SERVER = Path.home() / "models/llama.cpp-pr32704/build/bin/llama-server"
SERVER_REPO = Path.home() / "models/llama.cpp-pr32704"
SERVER_COMMIT = "42fd370cf9bc758f02f39d5cdfa15822f86f23a3"
MODEL = Path.home() / "models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf"
MODEL_SHA256 = "78fcedcf93c5d1bfe83e225513b71392bb52259be0eaa5f8899c856a2e5834ae"
MODEL_BYTES = 16610544608
SERVER_ARGS = ["-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1", "--port", str(PORT)]
TIMEOUT = 900
FROZEN = {  # exp_060's Frozen inputs table
    "data/MANIFEST.json": "20ec58300860097529e2330201c25ead470c233068d8a7e26d9bd4259223cfc5",
    "data/cache/lookups.json": "1208be020334af7e9faa649c9e21ae177a28d738f006fa18152b45fbc4dca113",
    "data/FOOTPRINT.txt": "2217e48c7971ad72bccd4a8210369db8c67cc666d4fe12553c4b10b43fe6ed02",
    "kit/prompt55.md": "512959511bca927bb0cab5f932f0ca2d8b95ed8ffffa0a752b2b895617af95c9",
}
FROZEN_MESSAGES = "d7d605dfbbfeed24a934fd0b06584712ba19e0f83863a4c142e7e9906fbbda28"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def frozen_check() -> dict:
    """C1, first half: the five sha256 values of exp_060's Frozen inputs table."""
    got = {p: sha256_file(EXP55 / p) for p in FROZEN}
    h = hashlib.sha256()
    for ext in ("eml", "txt"):
        for f in sorted((EXP55 / "data/messages").glob(f"*.{ext}")):
            h.update(f.read_bytes())
    got["data/messages"] = h.hexdigest()
    want = {**FROZEN, "data/messages": FROZEN_MESSAGES}
    return {"pass": got == want, "mismatch": sorted(p for p in want if got.get(p) != want[p])}


def thermal() -> str:
    try:
        out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True,
                             text=True, timeout=10).stdout
        return out.split()[-1] if out.split() else ""
    except Exception as e:
        return f"{type(e).__name__}"


def _run(*argv, timeout=30) -> str:
    try:
        return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def append(path: Path, row: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call(prompt: str):
    body = {"messages": [{"role": "user", "content": prompt}], "temperature": 0, "seed": 55}
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec = {"band": None, "used_items": [], "why": None, "parsed": False, "content": None, "error": None,
           "thermal_before": thermal()}
    reasoning = ""
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=TIMEOUT).read())
        choice = (r.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        d = arms55.parse_json(content)
        usage, timings = r.get("usage") or {}, r.get("timings") or {}
        rec.update({"band": d["band"] if d else None, "used_items": list((d or {}).get("used_items") or []),
                    "why": (d or {}).get("why"), "parsed": bool(d), "content": content,
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "prompt_ms": timings.get("prompt_ms"), "predicted_ms": timings.get("predicted_ms"),
                    "predicted_per_second": timings.get("predicted_per_second")})
    except urllib.error.HTTPError as e:
        rec["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
    except Exception as e:  # timeouts included
        rec["error"] = f"{type(e).__name__}: {e}"
    rec.update({"seconds": round(time.time() - t0, 2), "reasoning_chars": len(reasoning)})
    return rec, reasoning


def run_arm(out: Path, private: Path, dummy: bool):
    """The child: runs under the audit hook and connects to the side server only."""
    out.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    manifest = arms60.DUMMY if dummy else json.loads((EXP55 / "data/MANIFEST.json").read_text())
    res_path = out / "results.jsonl"
    done = {json.loads(l)["id"] for l in open(res_path)} if res_path.exists() else set()
    for mid in sorted(manifest):
        if mid in done:
            continue
        prompt = arms60.prompt_for(manifest[mid], dummy)  # asserts equality with exp_055's builder on corpus messages
        rec, reasoning = call(prompt)
        if rec["error"]:
            # Amendment 1: an error is not an answer. It is kept apart, the arm stops, and a resume asks again.
            append(out / "errors.jsonl", {"id": mid, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec})
            print("STOP: call failed on", mid, rec["error"], "- nothing was scored for it; resume with the same --run",
                  flush=True)
            sys.exit(4)
        append(private / "reasoning.jsonl", {"id": mid, "reasoning": reasoning})
        append(res_path, {"id": mid, **rec})
        print("I", mid, rec["band"], rec["seconds"], rec.get("finish_reason"), rec.get("completion_tokens"),
              "thermal", rec["thermal_before"], rec["error"] or "", flush=True)
    if dummy is False and len({json.loads(l)["id"] for l in open(res_path)}) != len(manifest):
        print("STOP: not every message has an answer row", flush=True)
        sys.exit(5)
    (out / "reasoning.sha256").write_text(sha256_file(private / "reasoning.jsonl") + "  reasoning.jsonl (private)\n")
    print("done I", out, flush=True)


def wait_ready(proc, seconds=300) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            return False
        try:
            if b'"ok"' in urllib.request.urlopen(HOST + "/health", timeout=3).read():
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def run_all(run: Path, private: Path, dummy: bool):
    """The driver: checks, server, child under the audit hook, server stopped, state recorded."""
    out = run / "I"
    out.mkdir(parents=True, exist_ok=True)
    meta = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
            "python": sys.version, "macos": _run("sw_vers", "-productVersion"), "macos_build": _run("sw_vers", "-buildVersion"),
            "dummy": dummy, "server_args": SERVER_ARGS, "ollama_ps_before": _run("ollama", "ps"),
            "thermal_start": thermal()}

    def save():
        (run / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
        (run / f"meta-{meta['started_utc'].replace(':', '')}.json").write_text(json.dumps(meta, indent=1) + "\n")

    meta["frozen_inputs"] = frozen_check()
    meta["server_commit"] = _run("git", "-C", str(SERVER_REPO), "rev-parse", "HEAD")
    meta["model_bytes"] = MODEL.stat().st_size if MODEL.exists() else None
    print("hashing the model file (about half a minute) ...", flush=True)
    meta["model_sha256"] = sha256_file(MODEL) if MODEL.exists() else None
    meta["pinned"] = {"frozen_inputs": meta["frozen_inputs"]["pass"],
                      "server_commit": meta["server_commit"] == SERVER_COMMIT,
                      "model_file": meta["model_sha256"] == MODEL_SHA256 and meta["model_bytes"] == MODEL_BYTES}
    save()
    if not all(meta["pinned"].values()):
        print("STOP: a pinned input does not match:", meta["pinned"], meta["frozen_inputs"], flush=True)
        sys.exit(2)

    try:
        urllib.request.urlopen(HOST + "/health", timeout=3).read()
        busy = True
    except urllib.error.HTTPError:
        busy = True          # something answered, even if not with a health page
    except (urllib.error.URLError, OSError):
        busy = False
    meta["port_was_free"] = not busy
    if busy:
        print(f"STOP: something already answers on 127.0.0.1:{PORT}; this run starts its own server", flush=True)
        save()
        sys.exit(6)
    log = open(out / "server.log", "ab")
    server = subprocess.Popen([str(SERVER), "-m", str(MODEL), *SERVER_ARGS], stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(server):
            print("STOP: the server did not become ready; see", out / "server.log", flush=True)
            meta["server_ready"] = False
            save()
            sys.exit(3)
        meta["server_ready"] = True
        save()
        env = {**os.environ, "PYTHONPATH": str(HOOK), "AUDIT54_LOG": str(out / "audit.jsonl")}
        argv = [sys.executable, __file__, "--arm", "--out", str(out), "--private", str(private)]
        argv += ["--dummy"] if dummy else []
        t0 = time.time()
        meta["arm_rc"] = subprocess.run(argv, env=env).returncode
        meta["arm_seconds"] = round(time.time() - t0, 1)
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
    meta.update({"ollama_ps_after": _run("ollama", "ps"), "thermal_end": thermal(),
                 "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    save()
    print("all done" if meta.get("arm_rc") == 0 else "STOPPED EARLY, see above; resume with the same --run", run,
          {k: meta.get(k) for k in ("arm_rc", "arm_seconds", "thermal_start", "thermal_end")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="driver: checks, server, arm, server stopped")
    ap.add_argument("--run", help="run folder for --all")
    ap.add_argument("--arm", action="store_true", help="child: the arm only; the server must be up")
    ap.add_argument("--out")
    ap.add_argument("--private", required=True, help="folder outside the repository for the reasoning text")
    ap.add_argument("--dummy", action="store_true", help="exp_060's kit/dummy60.eml only; never a corpus message")
    a = ap.parse_args()
    if a.all and a.run:
        run_all(Path(a.run), Path(a.private), a.dummy)
    elif a.arm and a.out:
        run_arm(Path(a.out), Path(a.private), a.dummy)
    else:
        ap.error("either --all --run <dir> --private <dir>, or --arm --out <dir> --private <dir>")


if __name__ == "__main__":
    main()
