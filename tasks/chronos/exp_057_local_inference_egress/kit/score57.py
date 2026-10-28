#!/usr/bin/env python3
"""exp_057: classify every endpoint seen, and evaluate the controls and H1-H4 from a run's logs.

Reads <run>/logs and <run>/steps.json. Writes <run>/score.json and prints a summary. Descriptive where the
hypothesis is descriptive; it never decides adoption.

    python3.14 score57.py <run_dir> [--pcap-text FILE]

--pcap-text is the optional full-capture arm as text: `tcpdump -n -k NP -r capture.pcapng > capture.txt`. The
parser takes the standard `IP src > dst` line shape plus the `proc name:pid` metadata Apple's tcpdump prints from
a pktap capture. It is marked experimental: it has not yet met a real capture. Endpoints the capture shows that
neither sampler saw are listed, never merged into the sampler's result.
"""
import ast
import json
import re
import sys
from pathlib import Path

TOOLCALL_MARKERS = ["<tool_call", "tool_call", "web_search", "function_call", "\"name\":", "search(",
                    "i'll search", "i will search", "let me search", "googling", "browsing the web"]
LOOPBACK_PREFIXES = ("127.", "::1", "localhost", "[::1]")
PCAP_LINE = re.compile(r"IP6?\s+(\S+)\s+>\s+(\S+):\s")      # dst ends at ": "; IPv6 holds colons itself
PCAP_PROC = re.compile(r"proc\s+([^:,()\s]+):(\d+)")


def load_jsonl(p):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def lit(s):
    try:
        return ast.literal_eval(s)
    except Exception:                                   # noqa: BLE001
        return s


def split_hostport(addr):
    """'192.0.2.4:443' -> ('192.0.2.4', '443'); '[::1]:443' -> ('::1', '443'); nettop's v6 'fe80::1.5353' -> split on '.'."""
    if addr.startswith("["):
        return addr[1:addr.index("]")], addr[addr.index("]") + 2:]
    if addr.count(":") > 1:                              # bare IPv6: nettop separates the port with a dot
        host, _, port = addr.rpartition(".")
        return host, port
    host, _, port = addr.rpartition(":")
    return host, port


def is_loopback(host):
    return host is not None and host.startswith(LOOPBACK_PREFIXES)


def remote_endpoint(endpoint):
    """(kind, proto, host, port) from an lsof NODE+NAME string. kind is 'listen', 'egress' or 'other'."""
    if "(LISTEN)" in endpoint:
        return "listen", None, None, None
    m = re.match(r"\s*(TCP|UDP)\s+(\S+)", endpoint)
    if not m:
        return "other", None, None, None
    proto, addr = m.group(1), m.group(2)
    remote = addr.split("->")[1] if "->" in addr else addr
    host, port = split_hostport(remote)
    return "egress", proto, host, port


def externals(samples):
    """{'TCP host:port': earliest_t} for non-loopback remote endpoints in a list of tree samples."""
    out = {}
    for s in samples:
        kind, proto, host, port = remote_endpoint(s["endpoint"])
        if kind == "egress" and host and host != "*" and not is_loopback(host):
            key = f"{proto} {host}:{port}"
            out[key] = min(out.get(key, s["t"]), s["t"])
    return out


def tree_pids(samples):
    return {str(s["pid"]) for s in samples}


def loopback_11434(samples):
    for s in samples:
        kind, proto, host, port = remote_endpoint(s["endpoint"])
        if kind == "egress" and is_loopback(host) and "11434" in s["endpoint"] and "ESTABLISHED" in s["endpoint"]:
            return True
    return False


def audit(path):
    """The driver's own network acts from the exp_054 hook: loopback model calls vs anything else."""
    out = {"loopback_11434": False, "externals": [], "resolved": [], "spawned": []}
    for e in load_jsonl(path):
        if e["kind"] == "getaddrinfo":
            host = str(e["host"])
            if not is_loopback(host):
                out["resolved"].append(host)
            continue
        name, args = e["event"], e.get("args", [])
        if name == "socket.connect" and len(args) > 1:
            addr = lit(args[1])
            if isinstance(addr, tuple) and len(addr) >= 2:
                host, port = str(addr[0]), int(addr[1])
                if is_loopback(host) and port == 11434:
                    out["loopback_11434"] = True
                elif not is_loopback(host):
                    out["externals"].append(f"{host}:{port}")
        elif name in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.system"):
            out["spawned"].append(str(args[0])[:120] if args else "")
    return out


def machine_externals(rows, tree):
    """Machine-wide non-loopback flows from one window, split into the watched tree's and other processes'."""
    first, final = {}, {}
    for r in rows:
        key = (r["process"], r["pid"], r["flow"])
        if r.get("kind") == "final":
            final[key] = r
        else:
            first.setdefault(key, r)
    mine, other = [], {}
    for key, r in first.items():
        local, _, remote = r["flow"].partition("<->")
        if "*" in remote:
            continue
        host, port = split_hostport(remote)
        if is_loopback(host):
            continue
        fin = final.get(key, r)
        item = {"flow": r["flow"], "iface": r["iface"], "state": fin.get("state"), "t": r.get("t"),
                "bytes_in": fin.get("bytes_in"), "bytes_out": fin.get("bytes_out"), "pid": r["pid"]}
        if r["pid"] in tree:
            mine.append(item)
        else:
            other.setdefault(r["process"], []).append(item)
    return {"tree": mine, "other": other}


def pcap_text(path):
    """Experimental: non-loopback destinations from `tcpdump -n -k NP` text, with the process when present."""
    seen = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = PCAP_LINE.search(line)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        dhost, _, dport = dst.rpartition(".")                 # tcpdump -n: 1.2.3.4.443 and 2001:db8::1.443
        if is_loopback(dhost) or is_loopback(src):
            continue
        p = PCAP_PROC.search(line)
        key = f"{dhost}:{dport}"
        seen.setdefault(key, {"process": p.group(1) if p else None, "pid": p.group(2) if p else None, "packets": 0})
        seen[key]["packets"] += 1
    return seen


def main():
    args = sys.argv[1:]
    run = Path(args.pop(0))
    pcap = None
    if args[:1] == ["--pcap-text"]:
        pcap = args[1]
    logs = run / "logs"
    steps = json.loads((run / "steps.json").read_text())
    models = [s["model"] for s in steps if s["kind"] == "drive"]
    by_step = {s["step"]: s for s in steps}

    # --- controls ---
    c_pos_ext = externals(load_jsonl(logs / "C_pos.egress.jsonl"))
    c_pos = {"pass": bool(c_pos_ext), "externals_seen": sorted(c_pos_ext)}
    tree_samples = load_jsonl(logs / "C_tree.egress.jsonl")
    root = str(by_step.get("C_tree", {}).get("root_pid"))
    c_tree_ext = externals([s for s in tree_samples if str(s["pid"]) != root])
    c_tree = {"pass": bool(c_tree_ext), "externals_seen_from_descendant": sorted(c_tree_ext),
              "pids_seen": sorted(tree_pids(tree_samples)), "root_pid": root}

    # --- per model ---
    per_model, h1_new, c_loop_lsof, c_loop_hook, driver_leaks = {}, {}, False, False, {}
    for model in models:
        safe = model.replace("/", "_")
        load_s = load_jsonl(logs / f"load_{safe}.egress.jsonl")
        idle_s = load_jsonl(logs / f"idle_{safe}.egress.jsonl")
        drive_s = load_jsonl(logs / f"drive_{safe}.egress.jsonl")
        load, idle, drive = externals(load_s), externals(idle_s), externals(drive_s)
        new = {h: t for h, t in drive.items() if h not in idle}        # H1: external during P1, absent at P0
        c_loop_lsof |= loopback_11434(drive_s)
        a = audit(logs / f"drive_{safe}.audit.jsonl")
        c_loop_hook |= a["loopback_11434"]
        if a["externals"] or a["resolved"]:
            driver_leaks[model] = {"connects": a["externals"], "resolved": a["resolved"]}

        prompts = load_jsonl(logs / f"drive_{safe}.drive.jsonl")
        windows = [(p["t0"], p["t1"], p["i"]) for p in prompts]
        tracked = {}                                                    # H2: new external first seen inside a prompt
        for host, t in new.items():
            for t0, t1, i in windows:
                if t0 <= t <= t1:
                    tracked[host] = i
        toolish = [p["i"] for p in prompts
                   if any(m in (p.get("answer") or "").lower() for m in TOOLCALL_MARKERS)]   # H4, coarse markers

        tree = tree_pids(load_s) | tree_pids(idle_s) | tree_pids(drive_s)
        machine = {w: machine_externals(load_jsonl(logs / f"{w}_{safe}.machine.jsonl"), tree)
                   for w in ("load", "idle", "drive")}
        per_model[model] = {
            "load_externals": sorted(load),                             # descriptive: egress while loading
            "idle_externals": sorted(idle),                             # H3: enumerate background egress
            "drive_externals": sorted(drive),
            "new_during_drive": new,                                    # H1 evidence
            "tracked_to_prompt": tracked,                               # H2 evidence
            "toolish_answer_indices": toolish,                          # H4 evidence
            "prompt_errors": [p["i"] for p in prompts if p.get("error")],
            "tree_pids": sorted(tree),
            "machine_wide": machine,                                    # descriptive: the rest of the machine
            "driver_spawned": a["spawned"],
        }
        h1_new[model] = sorted(new)

    h1_pass = bool(models) and all(not v for v in h1_new.values())
    h2_pass = bool(models) and all(not per_model[m]["tracked_to_prompt"] for m in models)
    c_loop = {"pass": c_loop_lsof and c_loop_hook and not driver_leaks,
              "lsof_saw_loopback_11434": c_loop_lsof, "hook_saw_loopback_11434": c_loop_hook,
              "driver_non_loopback_acts": driver_leaks}

    # machine-wide, across all windows: every other process that talked to a non-loopback endpoint
    others = {}
    for m in models:
        for w in per_model[m]["machine_wide"].values():
            for proc, items in w["other"].items():
                others.setdefault(proc, set()).update(i["flow"].partition("<->")[2] for i in items)
    tree_machine = sorted({i["flow"] for m in models for w in per_model[m]["machine_wide"].values() for i in w["tree"]})

    score = {
        "controls": {"C_pos": c_pos, "C_tree": c_tree, "C_loop": c_loop},
        "H1_inference_stays_local": {"pass": h1_pass, "new_externals_by_model": h1_new},
        "H2_no_endpoint_tracks_generation": {"pass": h2_pass},
        "H3_idle_egress_enumerated": {m: per_model[m]["idle_externals"] for m in models},
        "H3_load_egress_enumerated": {m: per_model[m]["load_externals"] for m in models},
        "H4_toolcall_output": {m: per_model[m]["toolish_answer_indices"] for m in models},
        "machine_wide_tree_flows": tree_machine,
        "machine_wide_other_processes": {p: sorted(v) for p, v in sorted(others.items())},
        "per_model": per_model,
        "models_run": models,
        "models_skipped": [s["model"] for s in steps if s["kind"] == "load" and s.get("error")],
    }
    if pcap:
        cap = pcap_text(pcap)
        sampled = {k.split(" ", 1)[1] for m in models for k in
                   per_model[m]["load_externals"] + per_model[m]["idle_externals"] + per_model[m]["drive_externals"]}
        sampled |= {split_hostport(f.partition("<->")[2])[0] + ":" + split_hostport(f.partition("<->")[2])[1]
                    for f in tree_machine} | {r for v in others.values() for r in v}
        score["pcap_arm"] = {"file": pcap, "endpoints": cap,
                             "unseen_by_samplers": sorted(k for k in cap if k not in sampled), "experimental": True}
    (run / "score.json").write_text(json.dumps(score, indent=2) + "\n")

    def mark(ok):
        return "PASS" if ok else "FAIL"
    print(f"\nexp_057 score  ({run})")
    print(f"  C_pos  (watcher sees an external connection) : {mark(c_pos['pass'])}  {c_pos['externals_seen']}")
    print(f"  C_tree (…from a descendant spawned later)    : {mark(c_tree['pass'])}  "
          f"{c_tree['externals_seen_from_descendant']}")
    print(f"  C_loop (driver's only act is the local call) : {mark(c_loop['pass'])}  lsof={c_loop_lsof} "
          f"hook={c_loop_hook} leaks={driver_leaks or 'none'}")
    print(f"  H1 inference stays local                     : {mark(h1_pass)}")
    for m in models:
        pm = per_model[m]
        flag = "  <-- NEW EXTERNAL" if pm["new_during_drive"] else ""
        print(f"       {m}: load={pm['load_externals']} idle={pm['idle_externals']} "
              f"new_in_drive={sorted(pm['new_during_drive'])}{flag}")
    print(f"  H2 no endpoint tracks a prompt               : {mark(h2_pass)}")
    print(f"  machine-wide, other processes with external flows: "
          f"{ {p: len(v) for p, v in score['machine_wide_other_processes'].items()} or 'none'}")
    if pcap:
        print(f"  pcap arm (experimental parse): {len(cap)} endpoints, unseen by samplers: "
              f"{score['pcap_arm']['unseen_by_samplers']}")
    print(f"  skipped: {score['models_skipped']}")
    print(f"\n  wrote {run / 'score.json'}")


if __name__ == "__main__":
    main()
