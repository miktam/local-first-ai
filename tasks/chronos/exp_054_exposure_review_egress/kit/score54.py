#!/usr/bin/env python3
"""exp_054: score the registered claims from one run's logs.

    python3.14 kit/score54.py runs/<ts>
"""
import ast
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
IANA_DATA, IANA_WHOIS = "data.iana.org", "whois.iana.org"
NETWORK_PROGRAMS = {"whois", "dig", "curl", "wget", "nslookup", "host", "nc", "ssh", "scp", "rsync", "ping"}


def lit(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse(path: Path) -> dict:
    ev = {"resolved": [], "connects": [], "urls": [], "spawns": [], "http": []}
    if not path.exists():
        return ev
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        if e["kind"] == "getaddrinfo":
            ev["resolved"].append({"host": str(e["host"]), "port": e.get("port"), "addrs": e.get("addrs") or [],
                                   "error": e.get("error")})
            continue
        name, args = e["event"], e["args"]
        if name == "socket.connect":
            addr = lit(args[1]) if len(args) > 1 else None
            ev["connects"].append(addr)
        elif name == "urllib.Request":
            ev["urls"].append(lit(args[0]))
        elif name == "subprocess.Popen":
            argv = lit(args[1]) if len(args) > 1 else []
            argv = [argv] if isinstance(argv, (str, bytes)) else list(argv or [])
            ev["spawns"].append([str(x) for x in argv])
        elif name == "http.client.connect":
            ev["http"].append((lit(args[1]) if len(args) > 1 else None, lit(args[2]) if len(args) > 2 else None))
    return ev


def prog(argv):
    return Path(argv[0]).name if argv else ""


def is_local_addr(a):
    return isinstance(a, str) or (isinstance(a, tuple) and str(a[0]) in LOCAL_HOSTS)   # str: a unix socket path


def rdap_hosts(cache: Path) -> set:
    p = cache / "rdap-bootstrap.json"
    if not p.exists():
        return set()
    d = json.loads(p.read_text())
    d = d.get("data", d)
    return {urlparse(u).hostname for s in d.get("services", []) for u in s[1] if urlparse(u).hostname}


def whois_servers(cache: Path) -> set:
    p = cache / "lookups.json"
    if not p.exists():
        return set()
    return {v["server"] for k, v in json.loads(p.read_text()).items()
            if k.startswith("whois-tld:") and isinstance(v, dict) and v.get("server")}


def network_findings(ev, allowed_hosts):
    """Everything in a step's log that is not a loopback model call or an allowed host."""
    bad = []
    ok_ips = {a for r in ev["resolved"] if r["host"] in allowed_hosts for a in r["addrs"]}
    for r in ev["resolved"]:
        if r["host"] not in allowed_hosts and r["host"] not in LOCAL_HOSTS:
            bad.append(f"resolved {r['host']}")
    for c in ev["connects"]:
        if not is_local_addr(c) and not (isinstance(c, tuple) and c[0] in ok_ips):
            bad.append(f"connect {c}")
    for u in ev["urls"]:
        h = urlparse(u).hostname if isinstance(u, str) else None
        if h not in allowed_hosts and h not in LOCAL_HOSTS:
            bad.append(f"url {u}")
    return bad


def main():
    run = Path(sys.argv[1])
    logs = run / "logs"
    manifest = json.loads((Path(__file__).resolve().parent.parent / "data/messages/MANIFEST.json").read_text())
    res = {}

    # Controls: the recorder must see these, or the audit is void.
    c2, c1 = parse(logs / "C2.jsonl"), parse(logs / "C1.jsonl")
    c2_ok = (any(r["host"] == "example.com" for r in c2["resolved"]) and
             any(isinstance(u, str) and "example.com" in u for u in c2["urls"]) and
             any(prog(s) == "dig" for s in c2["spawns"]))
    c1_ok = any(prog(s) == "dig" for s in c1["spawns"])
    res["controls"] = {"C2_recorder_sees_resolution_http_and_spawn": c2_ok, "C1_recorder_sees_dns_at_full_tier": c1_ok,
                       "C1_dig_queries": [s for s in c1["spawns"] if prog(s) == "dig"]}
    void = not (c2_ok and c1_ok)

    # H1: the review's other commands make no network call at all.
    h1 = {}
    for p in sorted(logs.glob("cmd_*.jsonl")) or []:
        ev = parse(p)
        bad = network_findings(ev, set()) + [f"spawned {s}" for s in ev["spawns"] if prog(s) in NETWORK_PROGRAMS]
        bad += [f"model connect {c}" for c in ev["connects"] if isinstance(c, tuple) and c[1] == 11434]
        h1[p.stem] = bad
    missing_cmds = [p.stem for p in [logs / f"cmd_{c}.jsonl" for c in ("build", "snapshot", "delta", "week", "report")]
                    if not (logs / f"{p.stem}.out").exists()]
    res["H1_review_commands_make_no_network_call"] = {"findings": h1, "steps_missing": missing_cmds,
                                                      "verdict": "pass" if not any(h1.values()) and not missing_cmds
                                                      else "fail"}

    # H2: every model call is on this machine.
    model_bad = []
    for p in sorted(logs.glob("*.jsonl")):
        ev = parse(p)
        for c in ev["connects"]:
            if isinstance(c, tuple) and len(c) > 1 and c[1] == 11434 and str(c[0]) not in LOCAL_HOSTS:
                model_bad.append(f"{p.stem}: {c}")
        for r in ev["resolved"]:
            if r["port"] == 11434 and r["host"] not in LOCAL_HOSTS:
                model_bad.append(f"{p.stem}: resolved {r['host']}:11434")
    model_calls = sum(1 for p in logs.glob("*_M*.jsonl") for c in parse(p)["connects"]
                      if isinstance(c, tuple) and len(c) > 1 and c[1] == 11434)
    res["H2_model_calls_stay_on_this_machine"] = {"model_connects_seen": model_calls, "findings": model_bad,
                                                  "verdict": "pass" if not model_bad and model_calls > 0 else
                                                  ("fail" if model_bad else "not decided (no model call seen)")}

    # H3 and H4: the registry tier.
    reg_cache = run / "cache_registry"
    rdap, wservers = rdap_hosts(reg_cache), whois_servers(reg_cache)
    allowed = {IANA_DATA} | rdap
    h3, h4, per_msg = {}, {}, {}
    for mid, m in manifest.items():
        ev = parse(logs / f"reg_{mid}.jsonl")
        msg_domains = set(m["domains"])
        bad = network_findings(ev, allowed)
        disclosed, iana_queries = [], []
        for s in ev["spawns"]:
            if prog(s) == "dig":
                bad.append(f"dns record lookup {s}")
            elif prog(s) == "whois":
                server = s[s.index("-h") + 1] if "-h" in s else None
                query = s[-1]
                if server == IANA_WHOIS:
                    iana_queries.append(query)
                    if "." in query:
                        bad.append(f"IANA sent more than a TLD: {query}")
                elif server in wservers:
                    disclosed.append(query.lower())
                else:
                    bad.append(f"whois to an unlisted server {server}")
            elif prog(s) in NETWORK_PROGRAMS:
                bad.append(f"spawned {s}")
        for u in ev["urls"]:
            if isinstance(u, str) and "/domain/" in u:
                disclosed.append(u.rsplit("/domain/", 1)[1].split("?")[0].lower())
        for r in ev["resolved"]:
            if any(r["host"] == d or r["host"].endswith("." + d) for d in msg_domains):
                bad.append(f"resolver asked about a message domain: {r['host']}")
        h3[mid] = bad
        dset = sorted(set(disclosed))
        h4[mid] = ([f"{len(dset)} domains disclosed (cap 12)"] if len(dset) > 12 else []) + \
                  [f"excluded domain disclosed: {d}" for d in dset if d in set(m["excluded"])] + \
                  [f"disclosed a domain not in the message: {d}" for d in dset if d not in msg_domains]
        per_msg[mid] = {"purpose": m["purpose"], "domains_in_message": len(msg_domains), "disclosed": dset,
                        "iana_whois_queries": iana_queries,
                        "rdap_requests": [u for u in ev["urls"] if isinstance(u, str) and "/domain/" in u],
                        "bootstrap_fetches": sum(1 for u in ev["urls"] if isinstance(u, str) and IANA_DATA in u)}
    res["H3_registry_tier_contacts_only_registries"] = {
        "allowed": {"rdap_hosts_in_bootstrap": len(rdap), "whois_servers_named_by_iana": sorted(wservers)},
        "findings": {k: v for k, v in h3.items() if v}, "verdict": "pass" if not any(h3.values()) else "fail"}
    res["H4_disclosure_capped_and_exclusions_hold"] = {
        "findings": {k: v for k, v in h4.items() if v}, "verdict": "pass" if not any(h4.values()) else "fail"}
    res["per_message"] = per_msg

    # H5: offline means nothing leaves but the loopback model call.
    h5 = {}
    for mid in manifest:
        ev = parse(logs / f"off_{mid}.jsonl")
        h5[mid] = network_findings(ev, set()) + [f"spawned {s}" for s in ev["spawns"] if prog(s) in NETWORK_PROGRAMS]
    res["H5_offline_sends_nothing"] = {"findings": {k: v for k, v in h5.items() if v},
                                       "verdict": "pass" if not any(h5.values()) else "fail"}

    # Cross-check: lsof's view of open internet sockets, any remote endpoint the logs do not explain.
    lsof = {}
    for p in sorted(logs.glob("*.lsof")):
        lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
        odd = [ln for ln in lines if "->" in ln and not any(x in ln.split("->")[1] for x in ("127.0.0.1", "[::1]"))
               and ":443 " not in ln + " " and ":43 " not in ln + " "]
        if odd:
            lsof[p.stem] = odd
    res["lsof_unexplained_endpoints"] = lsof
    res["steps"] = json.loads((run / "steps.json").read_text()) if (run / "steps.json").exists() else []
    res["void"] = void
    (run / "score.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    for k, v in res.items():
        if k.startswith("H"):
            print(k, "VOID" if void else v["verdict"])
    print("controls", res["controls"]["C2_recorder_sees_resolution_http_and_spawn"],
          res["controls"]["C1_recorder_sees_dns_at_full_tier"])


if __name__ == "__main__":
    main()
