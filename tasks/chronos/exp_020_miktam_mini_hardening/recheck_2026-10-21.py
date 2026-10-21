#!/usr/bin/env python3
"""Re-check of exp_020's sudo findings, 2026-10-21 (see AUDIT_FINDINGS.md, "Correction (2026-10-21)").

Nothing here runs a granted command. `sudo -n -l <command> <args>` asks the sudo policy whether it
would allow that exact command for this account, and runs nothing. The script also records, for every
command named in the grants, who owns it and whether this account can write it or its directory
(a root grant on a file the account can change is a root grant on anything), sudo's environment
defaults for this account, and FileVault status.

Run as the automation account (miktam02). Writes results/recheck_<UTC timestamp>.json.
Home paths of other local accounts are written as /Users/<other-account>.
"""
import datetime
import getpass
import json
import os
import pwd
import re
import subprocess
from pathlib import Path

QUERIES = [  # (name, argv to query, expected answer after the 2026-10-21 fixes)
    ("cp, any file", ["/bin/cp", "/dev/null", "/tmp/recheck-x"], False),
    ("tail, any file", ["/usr/bin/tail", "/private/var/root/recheck-x"], False),
    ("tail -f, the one log it is scoped to", None, True),        # argv read from the grant itself
    ("deploy script, root-owned copy", ["/usr/local/sbin/pharos-deploy-bot"], True),
    ("deploy script, old path in the account's home", ["/Users/miktam02/REPOS/pharos/tools/deploy-bot.sh"], False),
    ("shutdown", ["/sbin/shutdown", "-h", "+999"], True),
]


def run(argv):
    return subprocess.run(argv, capture_output=True, text=True)


def allowed(argv):
    return run(["sudo", "-n", "-l", *argv]).returncode == 0


def redact(text):
    return re.sub(r"/Users/(?!" + re.escape(getpass.getuser()) + r"\b)[^/\s\"]+", "/Users/<other-account>", text)


def scoped_tail(listing):
    """The narrow tail grant, e.g. '/usr/bin/tail -f <one file>', as an argv list."""
    for part in re.split(r"[,\n]", listing):
        part = part.strip().split("NOPASSWD:")[-1].strip()
        if part.startswith("/usr/bin/tail -f "):
            return part.split()
    return None


def owner(path):
    return pwd.getpwuid(os.stat(path).st_uid).pw_name


def path_record(p):
    rec = {"path": p, "exists": os.path.exists(p)}
    if rec["exists"]:
        d = os.path.dirname(p)
        rec.update(owner=owner(p), mode=oct(os.stat(p).st_mode & 0o7777),
                   writable_by_account=os.access(p, os.W_OK),
                   dir_owner=owner(d), dir_writable_by_account=os.access(d, os.W_OK))
    return rec


def main():
    listing = run(["sudo", "-n", "-l"]).stdout
    granted = set()
    for line in listing.splitlines():
        m = re.match(r"\s*\([^)]*\)\s*NOPASSWD:\s*(.*)", line)
        if m:
            for part in m.group(1).split(","):
                if part.strip():
                    granted.add(part.strip().split()[0])
    queries = []
    for name, argv, expected in QUERIES:
        argv = argv or scoped_tail(listing)
        if argv is None:
            queries.append({"query": name, "argv": None, "allowed": False, "expected": expected, "ok": False})
            continue
        got = allowed(argv)
        queries.append({"query": name, "argv": argv, "allowed": got, "expected": expected, "ok": got == expected})
    paths = [path_record(p) for p in sorted(granted)]
    out = {
        "experiment": "exp_020 re-check",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "account": getpass.getuser(),
        "method": "sudo -n -l <argv> (policy query, nothing runs); stat of every granted path; sudo defaults; fdesetup status",
        "sudo_defaults": {"secure_path": "secure_path" in listing, "always_set_home": "always_set_home" in listing},
        "queries": queries,
        "granted_paths": paths,
        "granted_path_problems": [r["path"] for r in paths if r["exists"] and
                                  (r["owner"] != "root" or r["writable_by_account"] or r["dir_writable_by_account"])],
        "filevault": run(["fdesetup", "status"]).stdout.strip(),
    }
    out["all_ok"] = (all(q["ok"] for q in queries) and not out["granted_path_problems"]
                     and all(out["sudo_defaults"].values()) and out["filevault"] == "FileVault is On.")
    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    f = results / ("recheck_" + out["timestamp"].replace(":", "-").replace("+00-00", "Z") + ".json")
    f.write_text(redact(json.dumps(out, indent=1)) + "\n")
    print(json.dumps({k: out[k] for k in ("timestamp", "sudo_defaults", "granted_path_problems", "filevault", "all_ok")}, indent=1))
    for q in queries:
        print(("ok  " if q["ok"] else "BAD ") + q["query"] + ": allowed=" + str(q["allowed"]))
    print("written:", f)


if __name__ == "__main__":
    main()
