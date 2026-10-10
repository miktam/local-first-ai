"""exp_038 blindness audit (DESIGN §4.3 step 5, X3): which files each subagent opened, from its own transcript.

Reads a Claude Code subagent transcript (JSONL) and extracts every tool call's path or command: Read, Write, Edit,
Glob, Grep, NotebookEdit file paths, and every Bash command in full. Read paths, and paths named in Bash commands,
are compared with the agent's allowed list. Forbidden markers in any path or command are flagged. Nothing from
the transcript's message text or tool results is copied; only tool-call inputs.

    python3 tools/audit_agents.py <label> <transcript> <allowed-glob> [<allowed-glob> ...]
writes blindness/<label>.json and prints a one-line verdict.
"""
from __future__ import annotations

import fnmatch
import json
import re
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"conversations\.jsonl", r"tasks/coapi/materials", r"beta-lessons", r"knowledge/areas\.md", r"/internal/",
             r"eval_v2", r"coapi_eval_v1", r"/reference/", r"/runs/", r"judge", r"doc2query", r"/train/", r"WebFetch", r"WebSearch"]
HOME = str(Path.home()) + "/"
PATH_RE = re.compile(r"(?:" + re.escape(HOME) + r"|~/|(?<![\w/])(?:private|vendor|runs|reference|rights|harness|kit|tools)/)[^\s'\"\)\]\},;|&<>]+")


def calls(transcript: Path):
    for line in transcript.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = (o.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                yield c.get("name"), c.get("input") or {}


def norm(p: str) -> str:
    p = p.replace("~/", HOME)
    if not p.startswith("/"):
        p = str(P38 / p)
    return p


def main(argv) -> int:
    label, transcript, allowed = argv[0], Path(argv[1]), [norm(a) for a in argv[2:]]
    rec = {"label": label, "transcript": transcript.name, "allowed": allowed, "reads": [], "writes": [], "searches": [],
           "bash": [], "other_tools": [], "flags": []}
    for name, inp in calls(transcript):
        if name in ("Read", "NotebookRead"):
            rec["reads"].append(norm(inp.get("file_path", "")))
        elif name in ("Write", "Edit", "NotebookEdit"):
            rec["writes"].append(norm(inp.get("file_path", "")))
        elif name in ("Glob", "Grep"):
            rec["searches"].append({"tool": name, "pattern": inp.get("pattern"), "path": inp.get("path")})
        elif name == "Bash":
            rec["bash"].append(inp.get("command", ""))
        elif name not in ("SubagentHandback", "TodoWrite", "StructuredOutput"):
            rec["other_tools"].append(name)
    parents = {str(Path(a).parent) for a in allowed}

    def ok(p):
        return any(fnmatch.fnmatch(p, a) for a in allowed) or p.rstrip("/") in parents or p.rstrip("/") == str(P38)
    for p in rec["reads"]:
        if not ok(p):
            rec["flags"].append(f"read outside allowed list: {p}")
    for cmd in rec["bash"]:
        for m in PATH_RE.findall(cmd):
            p = norm(m.rstrip(".:"))
            if not ok(p) and not p.startswith(str(P38 / "tools/")) and "/tmp/" not in p:
                rec["flags"].append(f"bash names a path outside the allowed list: {p}")
        stripped = cmd
        for a in allowed:  # an explicitly allowed path never trips a marker
            stripped = stripped.replace(a, "").replace(a.replace(str(P38) + "/", ""), "")
        for pat in FORBIDDEN:
            if re.search(pat, stripped):
                rec["flags"].append(f"bash mentions forbidden marker {pat!r}")
    for p in rec["reads"] + rec["writes"]:
        if ok(p):
            continue
        for pat in FORBIDDEN:
            if re.search(pat, p):
                rec["flags"].append(f"path matches forbidden marker {pat!r}: {p}")
    if rec["searches"]:
        rec["flags"].append(f"{len(rec['searches'])} filesystem search(es)")
    if rec["other_tools"]:
        rec["flags"].append(f"other tools used: {sorted(set(rec['other_tools']))}")
    rec["flags"] = sorted(set(rec["flags"]))
    rec["verdict"] = "CLEAN" if not rec["flags"] else "REVIEW"
    out = P38 / "blindness" / f"{label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(f"{label}: {rec['verdict']} (reads {len(rec['reads'])}, writes {len(rec['writes'])}, bash {len(rec['bash'])}, "
          f"flags {len(rec['flags'])})" + ("".join(f"\n  - {f}" for f in rec["flags"]) if rec["flags"] else ""))
    return 0 if rec["verdict"] == "CLEAN" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
