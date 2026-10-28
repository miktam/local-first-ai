#!/usr/bin/env python3
"""exp_057: run the session controls and, per model, a load window, an idle window and a driven window, with the
runtime's process tree and the whole machine watched throughout.

Layout written under <run>/:
  meta.json                    ollama version and `ollama list` at the start, macOS version, times, arguments
  steps.json                   one record per step (kind, model, rc, seconds; root_pid for the tree control)
  logs/<step>.egress.jsonl     lsof samples of the watched tree during the step (egress_watch.Watcher, 0.2 s)
  logs/<step>.machine.jsonl    nettop samples of every process on the machine (egress_watch.MachineWatcher, 1 s)
  logs/<step>.audit.jsonl      the driver's Python audit log (loopback ground truth), driven steps only
  logs/<step>.drive.jsonl      per-prompt windows and answers, driven steps only

Steps:
  C_pos           control: one process connects to example.com:80; the tree watcher, pointed at that PID at the
                  measurement interval, must see the non-loopback endpoint
  C_tree          control: the same connection from a grandchild of the watched root (sh -> python), so the
                  descendant expansion that must catch a runner spawned mid-window is what is tested
  load_<model>    the one-token warm-up that loads the model (its runner spawns here), watched
  idle_<model>    P0: 30 s watching the loaded tree with no generation
  drive_<model>   P1: the frozen prompts, tree watched throughout (C_loop lives here)

    python3.14 run57.py <run_dir> [--idle SECONDS] [--prompts N] [model ...]

--idle and --prompts exist for smoke tests only. The registered run uses the defaults: 30 s and every prompt.
"""
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import egress_watch as ew
from drive import is_embed

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PROMPTS = EXP / "data/prompts.json"
HOOK = HERE / "hook"
OLLAMA = "http://127.0.0.1:11434"
PY = sys.executable

DEFAULT_MODELS = [
    "qwen3-4b-b11577:q4km", "qwen3-4b-b11577:q8", "qwen3-4b-b11577:q4km-imat",
    "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", "hf.co/unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    "qwen3.8:27b", "qwen3.6:35b", "qwen3.5:35b",
    "gemma4:26b", "nomic-embed-text",
]
IDLE_SECONDS = 30


def _ollama(args):
    try:
        r = subprocess.run(["ollama", *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e!r}"


def warmup(model):
    """Load the model with a one-token generate (or one embedding). Returns None on success, else the error."""
    if is_embed(model):
        path, body = "/api/embed", {"model": model, "input": "hi"}
    else:
        path, body = "/api/generate", {"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1}}
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        return None
    except Exception as e:                              # noqa: BLE001
        return repr(e)


def run_driver(model, audit_log, drive_log, max_prompts):
    env = {**os.environ, "PYTHONPATH": str(HOOK), "AUDIT57_LOG": str(audit_log)}
    argv = [PY, str(HERE / "drive.py"), model, str(PROMPTS), str(drive_log)]
    if max_prompts:
        argv.append(str(max_prompts))
    return subprocess.run(argv, env=env).returncode


class Window:
    """Both samplers around one step: tree (lsof over root_pids_fn) and machine (nettop)."""

    def __init__(self, logs, step, root_pids_fn, interval=ew.LSOF_INTERVAL):
        self.tree = ew.Watcher(root_pids_fn, logs / f"{step}.egress.jsonl", interval)
        self.machine = ew.MachineWatcher(logs / f"{step}.machine.jsonl")

    def __enter__(self):
        self.tree.start()
        self.machine.start()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.tree.stop()
        self.machine.stop()
        self.seconds = round(time.time() - self.t0, 1)


def control(logs, step, argv):
    """Spawn argv, watch it (and its descendants) at the measurement interval until it exits."""
    p = subprocess.Popen(argv)
    with Window(logs, step, lambda pid=p.pid: {pid}) as w:
        p.wait()
        time.sleep(0.5)                                  # one more sample after the socket closed
    return {"kind": "control", "step": step, "rc": p.returncode, "root_pid": p.pid, "seconds": w.seconds}


def main():
    args = sys.argv[1:]
    run = Path(args.pop(0))
    idle, max_prompts = IDLE_SECONDS, None
    while args and args[0].startswith("--"):
        flag = args.pop(0)
        if flag == "--idle":
            idle = float(args.pop(0))
        elif flag == "--prompts":
            max_prompts = int(args.pop(0))
        else:
            sys.exit(f"unknown flag {flag}")
    models = args or DEFAULT_MODELS
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    meta = {"started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "ollama_version": _ollama(["--version"]), "ollama_list": _ollama(["list"]),
            "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "idle_seconds": idle, "max_prompts": max_prompts, "models": models,
            "registered_defaults": idle == IDLE_SECONDS and max_prompts is None and models == DEFAULT_MODELS}
    steps = []

    # --- controls: the watcher must see an external connection, and one made by a descendant spawned later ---
    ctl = str(HERE / "control_pos.py")
    steps.append(control(logs, "C_pos", [PY, ctl]))
    steps.append(control(logs, "C_tree", ["/bin/sh", "-c", f"sleep 0.5; {PY} {ctl}"]))
    print(f"controls done: {[s['rc'] for s in steps]}", flush=True)

    # --- per model: load (warm-up, watched), idle (P0), drive (P1) ---
    for model in models:
        safe = model.replace("/", "_")
        with Window(logs, f"load_{safe}", ew.resolve_ollama_pids) as w:
            err = warmup(model)
        steps.append({"kind": "load", "step": f"load_{model}", "model": model, "error": err, "seconds": w.seconds})
        if err:
            print(f"[skip] {model}: warm-up failed ({err})", flush=True)
            continue

        with Window(logs, f"idle_{safe}", ew.resolve_ollama_pids) as w:
            time.sleep(idle)
        steps.append({"kind": "idle", "step": f"idle_{model}", "model": model, "seconds": w.seconds})

        with Window(logs, f"drive_{safe}", ew.resolve_ollama_pids) as w:
            rc = run_driver(model, logs / f"drive_{safe}.audit.jsonl", logs / f"drive_{safe}.drive.jsonl", max_prompts)
        steps.append({"kind": "drive", "step": f"drive_{model}", "model": model, "rc": rc, "seconds": w.seconds})
        print(f"{model}: load {steps[-3]['seconds']}s, idle {steps[-2]['seconds']}s, drive {w.seconds}s", flush=True)

    meta["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (run / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
    print(f"\nwrote {run / 'steps.json'} ({len(steps)} steps)")


if __name__ == "__main__":
    main()
