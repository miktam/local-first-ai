#!/usr/bin/env python3
"""exp_054: run every registered step with the audit hook on, and sample open connections with lsof.

Steps:
  C2         control: kit/control_c2.py (must be seen: a resolution, an HTTP request, a spawned dig)
  cmd_<x>    exposure.py <x> on a fresh copy of the fictional demo subject: build, snapshot, delta, week, report
  reg_<Mnn>  exposure.py triage on each message at the default tier (registry), model leaf on
  off_<Mnn>  the same with --offline, model leaf on
  C1         control: Parry's check.py --net full on M01 (must be seen: dig for the sender's mail records)

Each step gets its own audit log (logs/<step>.jsonl) and lsof samples (logs/<step>.lsof). Parry's lookup cache is
a fresh folder per tier (TRIAGE_CACHE), so every lookup the code would make is made.

    python3.14 kit/run54.py runs/<ts>
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
WS = Path.home() / ".openclaw/workspace"
EXPOSURE = WS / "tools/exposure-report/exposure.py"
CHECK = WS / "tools/triage/check.py"
DEMO = WS / "tasks/exposure-report/demo"
MSGS = HERE / "data/messages"
COMMANDS = ["build", "snapshot", "delta", "week", "report"]


def descendants(pid):
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        r = subprocess.run(["pgrep", "-P", str(p)], capture_output=True, text=True)
        for c in r.stdout.split():
            if int(c) not in out:
                out.add(int(c))
                todo.append(int(c))
    return out


def run_step(name, argv, logs, env_extra, stdin=None):
    log = logs / f"{name}.jsonl"
    env = {**os.environ, "PYTHONPATH": str(HERE / "kit/hook"), "AUDIT54_LOG": str(log), **env_extra}
    t0 = time.time()
    p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE if stdin else None, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    seen, stop = set(), threading.Event()

    def sample():
        while not stop.is_set():
            pids = {p.pid} | descendants(p.pid)
            r = subprocess.run(["lsof", "-a", "-i", "-n", "-P", "-p", ",".join(map(str, pids))],
                               capture_output=True, text=True)
            for line in r.stdout.splitlines()[1:]:
                seen.add(" ".join(line.split()[7:]))     # NODE NAME columns: protocol and addresses
            time.sleep(0.2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    out, err = p.communicate(stdin)
    stop.set()
    th.join(timeout=5)
    (logs / f"{name}.lsof").write_text("\n".join(sorted(seen)) + ("\n" if seen else ""))
    (logs / f"{name}.out").write_bytes(out[-20000:] + b"\n--- stderr ---\n" + err[-5000:])
    return {"step": name, "rc": p.returncode, "seconds": round(time.time() - t0, 1)}


def main():
    run = Path(sys.argv[1])
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    subject = run / "subject"
    shutil.copytree(DEMO, subject, ignore=shutil.ignore_patterns("out", ".lock"))
    manifest = json.loads((MSGS / "MANIFEST.json").read_text())
    steps = []
    py = "python3.14"
    steps.append(run_step("C2", [py, str(HERE / "kit/control_c2.py")], logs, {}))
    for c in COMMANDS:
        steps.append(run_step(f"cmd_{c}", [py, str(EXPOSURE), c, str(subject)], logs,
                              {"TRIAGE_CACHE": str(run / "cache_cmd")}))
    for tier, flag, cache in (("reg", [], "cache_registry"), ("off", ["--offline"], "cache_offline")):
        for mid, m in manifest.items():
            argv = [py, str(EXPOSURE), "triage", str(subject), str(MSGS / m["file"]), "--json"] + flag
            if m["channel"]:
                argv += ["--channel", m["channel"]]
            steps.append(run_step(f"{tier}_{mid}", argv, logs, {"TRIAGE_CACHE": str(run / cache)}))
            print(steps[-1], flush=True)
    steps.append(run_step("C1", [py, str(CHECK), "--net", "full", "--json", str(MSGS / "M01.eml")], logs,
                          {"TRIAGE_CACHE": str(run / "cache_c1")}))
    (run / "steps.json").write_text(json.dumps(steps, indent=2) + "\n")
    print("done", run, flush=True)


if __name__ == "__main__":
    main()
