#!/usr/bin/env python3
"""exp_057 Amendment 2: with Ollama's cloud features disabled, does the daemon still refresh model recommendations?

Run after the flag is set and the daemon restarted. Reads the daemon's log for its last start and the cloud line
beside it, runs C_pos (the watcher must still see an external connection), then watches the Ollama tree (lsof,
0.2 s) and the whole machine (nettop, 1 s) for N seconds with no generation, then reads the log again for any
refresh line since the start. Nothing here generates, pulls or resolves a name.

    python3.14 idle57.py <run_dir> <seconds> [--log PATH]

Writes steps.json, logs/C_pos.*.jsonl, logs/idle.*.jsonl and verdict.json. The daemon log defaults to the
Homebrew service's path on this machine; pass --log elsewhere. Reading it is the only thing this script does
outside the run folder.
"""
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import egress_watch as ew          # noqa: E402
import score57 as sc               # noqa: E402

DEFAULT_LOG = "/opt/homebrew/var/log/ollama.log"
PY = sys.executable


def log_state(log):
    """(start_line, cloud_disabled, refresh_lines) from the daemon log: the last 'Listening on' line, the cloud
    line just before it, and every model_recommendations line after it."""
    r = subprocess.run(["grep", "-n", "-E", "Listening on|Ollama cloud disabled|model_recommendations", log],
                       capture_output=True, text=True)
    lines = r.stdout.splitlines()
    starts = [i for i, l in enumerate(lines) if "Listening on" in l]
    if not starts:
        return None, None, []
    i = starts[-1]
    cloud = next((l for l in reversed(lines[:i]) if "Ollama cloud disabled" in l), "")
    disabled = None if not cloud else ("disabled: true" in cloud)
    refresh = [l.split(":", 1)[1][:200] for l in lines[i + 1:] if "model_recommendations" in l]
    return lines[i].split(":", 1)[1][:120], disabled, refresh


def main():
    args = sys.argv[1:]
    run, seconds = Path(args.pop(0)), float(args.pop(0))
    log = DEFAULT_LOG
    if args[:1] == ["--log"]:
        log = args[1]
    logs = run / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    start_line, disabled, refresh_before = log_state(log)
    print(f"daemon start: {start_line}\ncloud disabled: {disabled}\nrefresh lines since start: {len(refresh_before)}", flush=True)

    p = subprocess.Popen([PY, str(HERE / "control_pos.py")])
    w = ew.Watcher(lambda pid=p.pid: {pid}, logs / "C_pos.egress.jsonl")
    w.start(); p.wait(); time.sleep(0.5); w.stop()
    c_pos = sc.externals(sc.load_jsonl(logs / "C_pos.egress.jsonl"))

    t0 = time.time()
    print(f"watching the Ollama tree and the machine for {seconds:.0f} s with no generation…", flush=True)
    ew.watch_for(ew.resolve_ollama_pids, seconds, logs / "idle.egress.jsonl", machine_path=logs / "idle.machine.jsonl")
    watched = round(time.time() - t0, 1)

    _, disabled_after, refresh_after = log_state(log)
    tree_samples = sc.load_jsonl(logs / "idle.egress.jsonl")
    tree_ext = sc.externals(tree_samples)
    machine = sc.machine_externals(sc.load_jsonl(logs / "idle.machine.jsonl"), sc.tree_pids(tree_samples))
    verdict = {
        "registered": "exp_057 Amendment 2",
        "started": dt.datetime.fromtimestamp(t0, dt.timezone.utc).isoformat(timespec="seconds"),
        "watched_seconds": watched,
        "daemon_start_line": start_line,
        "C_flag_cloud_disabled": disabled,
        "C_pos": {"pass": bool(c_pos), "externals_seen": sorted(c_pos)},
        "H5_no_refresh_line_since_start": {"pass": not refresh_after, "lines": refresh_after[:20]},
        "H6_tree_opened_nothing": {"pass": not tree_ext, "externals": sorted(tree_ext)},
        "tree_pids": sorted(sc.tree_pids(tree_samples)),
        "machine_tree_flows": [i["flow"] for i in machine["tree"]],
        "machine_other_processes": {p: len(v) for p, v in machine["other"].items()},
    }
    (run / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    (run / "steps.json").write_text(json.dumps([{"kind": "control", "step": "C_pos", "rc": p.returncode},
                                                {"kind": "idle", "step": "idle", "seconds": watched}], indent=2) + "\n")
    m = lambda ok: "PASS" if ok else "FAIL"   # noqa: E731
    print(f"\nexp_057 Amendment 2  ({run})")
    print(f"  C_flag (log says cloud disabled)     : {m(disabled is True)}  {disabled}")
    print(f"  C_pos  (watcher sees an external)    : {m(bool(c_pos))}  {sorted(c_pos)}")
    print(f"  H5 no refresh line since the start   : {m(not refresh_after)}  {len(refresh_after)} line(s)")
    print(f"  H6 tree opened nothing in {watched:.0f} s : {m(not tree_ext)}  {sorted(tree_ext)}")
    print(f"  other processes with external flows  : {verdict['machine_other_processes'] or 'none'}")
    print(f"\n  wrote {run / 'verdict.json'}")


if __name__ == "__main__":
    main()
