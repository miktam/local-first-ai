#!/usr/bin/env python3
"""exp_057: sample the open internet sockets of a process tree, and the whole machine beside it.

Two instruments, both without root:

1. Watcher: `lsof -nP -i -a -p <pids>` every `interval` seconds (0.2 s registered) over a set of root PIDs
   re-expanded to their descendants each sample, so a runner spawned mid-window is caught. The model runtime
   (Ollama's daemon and the runner it spawns per model: on 0.40.2 `libexec/lib/ollama/llama-server`, on older
   builds `ollama runner`) is Go/C++, not Python, so the exp_054 audit hook cannot see it; this samples it from
   outside. Each distinct (command, pid, endpoint) is recorded once, with the time of first sight. Endpoint is
   lsof's NODE + NAME columns, e.g. "TCP 127.0.0.1:11434->127.0.0.1:52345 (ESTABLISHED)".

2. MachineWatcher: `nettop -n -x -L 1` every second, every process of every user, with the interface, state
   and byte counts nettop keeps per flow. This checks the first instrument's blind spot: a connection made by a
   process outside the tree (anything the runtime reaches through launchd or XPC, or a runner that is not
   parent-linked) still appears here, named by process. It has a 1 s floor (nettop refuses sub-second
   intervals) and no PID filter, so a flow it shows is attributed by process name and PID, not proven to be
   the runtime's. Each distinct (process, pid, flow) is written at first sight; the byte counts last seen
   are written when the watch stops.

Neither instrument resolves a name (`-n` everywhere): watching makes no DNS query of its own.
Classification into loopback vs external is left to score57.py.

CLI (standalone watch of the Ollama tree for N seconds, both instruments):
    python3.14 egress_watch.py <seconds> <tree.jsonl> [machine.jsonl]
"""
import json
import subprocess
import sys
import threading
import time

LSOF_INTERVAL = 0.2
NETTOP_INTERVAL = 1.0
NETTOP_ARGV = ["nettop", "-n", "-x", "-L", "1", "-J", "interface,state,bytes_in,bytes_out"]


def _pgrep(args):
    r = subprocess.run(["pgrep", *args], capture_output=True, text=True)
    return {int(x) for x in r.stdout.split()} if r.returncode == 0 else set()


def resolve_ollama_pids():
    """Daemon and runner PIDs, by name and by command line, so both are caught even if not parent-linked.

    0.40.2 spawns the runner as `.../libexec/lib/ollama/llama-server --model ...` (a child of the daemon, so
    descendants() finds it too); older builds named it `ollama runner` or `ollama_llama_server`.
    """
    return (_pgrep(["-x", "ollama"]) | _pgrep(["-f", "ollama runner"]) | _pgrep(["-f", "ollama_llama_server"])
            | _pgrep(["-f", "libexec/lib/ollama/"]))


def descendants(pid):
    out, todo = set(), [pid]
    while todo:
        for c in _pgrep(["-P", str(todo.pop())]):
            if c not in out:
                out.add(c)
                todo.append(c)
    return out


def expand(root_pids):
    all_pids = set(root_pids)
    for p in list(root_pids):
        all_pids |= descendants(p)
    return all_pids


def sample(pids):
    """One lsof pass over pids. Yields (command, pid, endpoint) for each internet socket."""
    if not pids:
        return
    r = subprocess.run(["lsof", "-nP", "-i", "-a", "-p", ",".join(map(str, sorted(pids)))],
                       capture_output=True, text=True)
    for line in r.stdout.splitlines()[1:]:
        f = line.split()
        if len(f) >= 9:
            yield f[0], f[1], " ".join(f[7:])      # COMMAND, PID, NODE NAME...


class Watcher(threading.Thread):
    """Background lsof sampler over a live PID set. root_pids_fn() is re-called each sample."""

    def __init__(self, root_pids_fn, out_path, interval=LSOF_INTERVAL):
        super().__init__(daemon=True)
        self._root = root_pids_fn
        self._fh = open(out_path, "a", buffering=1, encoding="utf-8")
        self._interval = interval
        self._stop = threading.Event()
        self._seen = set()

    def run(self):
        while not self._stop.is_set():
            pids = expand(self._root())
            now = time.time()
            for cmd, pid, endpoint in sample(pids):
                key = (cmd, pid, endpoint)
                if key not in self._seen:
                    self._seen.add(key)
                    self._fh.write(json.dumps({"t": now, "command": cmd, "pid": pid, "endpoint": endpoint}) + "\n")
            self._stop.wait(self._interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        self._fh.close()


def _int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def parse_nettop(text):
    """Rows of one `nettop -L 1 -J interface,state,bytes_in,bytes_out` sample.

    nettop prints a process row (`name.pid,,,bytes_in,bytes_out,`) followed by that process's flow rows
    (`tcp4 local<->remote,iface,state,bytes_in,bytes_out,`). The header starts with a comma.
    """
    rows, proc, pid = [], None, None
    for line in text.splitlines():
        if not line or line.startswith(","):
            continue
        f = line.split(",")
        name = f[0]
        if "<->" not in name:
            proc, _, pid = name.rpartition(".")
            continue
        rows.append({"process": proc, "pid": pid, "flow": name,
                     "iface": f[1] if len(f) > 1 else "", "state": f[2] if len(f) > 2 else "",
                     "bytes_in": _int(f[3]) if len(f) > 3 else None,
                     "bytes_out": _int(f[4]) if len(f) > 4 else None})
    return rows


def machine_sample():
    r = subprocess.run(NETTOP_ARGV, capture_output=True, text=True)
    return parse_nettop(r.stdout) if r.returncode == 0 else []


class MachineWatcher(threading.Thread):
    """Background nettop sampler over every process on the machine."""

    def __init__(self, out_path, interval=NETTOP_INTERVAL):
        super().__init__(daemon=True)
        self._fh = open(out_path, "a", buffering=1, encoding="utf-8")
        self._interval = interval
        self._stop = threading.Event()
        self._last = {}

    def run(self):
        while not self._stop.is_set():
            now = time.time()
            for row in machine_sample():
                key = (row["process"], row["pid"], row["flow"])
                if key not in self._last:
                    self._fh.write(json.dumps({"kind": "flow", "t": now, **row}) + "\n")
                self._last[key] = row
            self._stop.wait(self._interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        for row in self._last.values():
            self._fh.write(json.dumps({"kind": "final", **row}) + "\n")
        self._fh.close()


def watch_for(root_pids_fn, seconds, out_path, interval=LSOF_INTERVAL, machine_path=None):
    """Watch for `seconds` with the tree sampler, and with the machine sampler too when machine_path is given."""
    w = Watcher(root_pids_fn, out_path, interval)
    m = MachineWatcher(machine_path) if machine_path else None
    w.start()
    if m:
        m.start()
    time.sleep(seconds)
    w.stop()
    if m:
        m.stop()


if __name__ == "__main__":
    secs, out = float(sys.argv[1]), sys.argv[2]
    machine = sys.argv[3] if len(sys.argv) > 3 else None
    watch_for(resolve_ollama_pids, secs, out, machine_path=machine)
    print(f"watched the Ollama tree for {secs}s -> {out}" + (f", the machine -> {machine}" if machine else ""))
