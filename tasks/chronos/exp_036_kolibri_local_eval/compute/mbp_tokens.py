#!/usr/bin/env python3
"""Sum Claude Code token usage on this machine between two UTC times.

Run on the mbp to close the gap in the exp_036 / exp_037 compute records: the
mbp's own Claude session transcripts never reached the mini.

    python3 mbp_tokens.py 2026-10-03T15:00:00Z 2026-10-05T15:00:00Z
    python3 mbp_tokens.py 2026-10-06T00:00:00Z 2026-10-08T00:00:00Z --match local-first-ai

What it reads: ~/.claude/projects/*/*.jsonl (sessions) and
~/.claude/projects/*/<session>/subagents/**/*.jsonl (agents and workflows).
A session is counted when any record in it, or in one of its agent transcripts,
has a `cwd` containing --match (default 'local-first-ai'). Note that the default
also matches a cwd such as '.../local-first-ai-blog'; use --exclude to drop one.

How it counts (same method as the mini's ledger):
  * one usage record per API message id, de-duplicated across all files
    (resumed sessions copy history into new files); the final usage line is
    kept (largest output count);
  * a message is in the window when its first timestamp t satisfies
    START <= t < END;
  * '<synthetic>' placeholders (API errors, 0 tokens) are excluded.

What it prints: only aggregates (counts and token totals by type, model and
source). No message text, tool input, path, cwd or session id is printed.
--show-scope adds the matched project-folder names and the first/last counted
message times, to check that the window and --match caught the right sessions.
Standard library only; read-only; no network.
"""
import argparse
import collections
import glob
import json
import os
import sys
from datetime import datetime, timezone

TYPES = ("input", "cache_write_5m", "cache_write_1h", "cache_write_unsplit", "cache_read", "output")


def utc(s):
    s = s.strip().replace("Z", "+00:00")
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc)


def split_usage(u):
    """Map one API usage block to the six token types; never double-count cache writes."""
    cw = int(u.get("cache_creation_input_tokens") or 0)
    cc = u.get("cache_creation") if isinstance(u.get("cache_creation"), dict) else {}
    w5 = int(cc.get("ephemeral_5m_input_tokens") or 0)
    w1 = int(cc.get("ephemeral_1h_input_tokens") or 0)
    if w5 + w1 > cw:  # inconsistent record: trust the total, keep it unsplit
        w5 = w1 = 0
    stu = u.get("server_tool_use") if isinstance(u.get("server_tool_use"), dict) else {}
    return {
        "input": int(u.get("input_tokens") or 0),
        "cache_write_5m": w5,
        "cache_write_1h": w1,
        "cache_write_unsplit": cw - w5 - w1,
        "cache_read": int(u.get("cache_read_input_tokens") or 0),
        "output": int(u.get("output_tokens") or 0),
    }, int(stu.get("web_search_requests") or 0), int(stu.get("web_fetch_requests") or 0)


def scan(path):
    """Return (cwd_matched, cwd_excluded, messages, compactions) for one transcript file."""
    msgs = {}
    comps = []
    cwds = set()
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if not isinstance(d, dict):
                continue
            cwd = d.get("cwd")
            if isinstance(cwd, str):
                cwds.add(cwd)
            ts = d.get("timestamp")
            if d.get("type") == "system" and d.get("subtype") == "compact_boundary" and ts:
                meta = d.get("compactMetadata") if isinstance(d.get("compactMetadata"), dict) else {}
                comps.append((ts, int(meta.get("preTokens") or 0), int(meta.get("postTokens") or 0)))
                continue
            if d.get("type") != "assistant" or not ts:
                continue
            m = d.get("message") if isinstance(d.get("message"), dict) else {}
            u = m.get("usage")
            if not isinstance(u, dict):
                continue
            mid = m.get("id") or d.get("uuid")
            if not mid:
                continue
            model = m.get("model") or "unknown"
            toks, ws, wf = split_usage(u)
            prev = msgs.get(mid)
            if prev is None:
                msgs[mid] = [ts, model, toks, ws, wf]
            else:
                prev[0] = min(prev[0], ts)
                if toks["output"] >= prev[2]["output"]:
                    prev[1:] = [model, toks, ws, wf]
    return cwds, msgs, comps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("start", help="window start, UTC, e.g. 2026-10-03T15:00:00Z (inclusive)")
    ap.add_argument("end", help="window end, UTC (exclusive)")
    ap.add_argument("--match", default="local-first-ai", help="substring a session's cwd must contain")
    ap.add_argument("--exclude", action="append", default=[], help="drop sessions whose every matching cwd contains this (repeatable)")
    ap.add_argument("--root", default=os.path.expanduser("~/.claude/projects"))
    ap.add_argument("--show-scope", action="store_true", help="also print matched project folders and first/last counted message times")
    a = ap.parse_args()
    t0, t1 = utc(a.start), utc(a.end)
    if t1 <= t0:
        sys.exit("end must be after start")

    # session id -> list of (kind, path); kind is 'main' or 'agent'
    files = collections.defaultdict(list)
    proj_of = {}
    for p in glob.glob(os.path.join(a.root, "*", "*.jsonl")):
        sid = os.path.basename(p)[:-6]
        files[sid].append(("main", p))
        proj_of[sid] = os.path.basename(os.path.dirname(p))
    for p in glob.glob(os.path.join(a.root, "*", "*", "subagents", "**", "*.jsonl"), recursive=True):
        parts = p.split(os.sep)
        sid = parts[len(parts) - 1 - parts[::-1].index("subagents") - 1]
        files[sid].append(("agent", p))
        proj_of.setdefault(sid, parts[len(parts) - 1 - parts[::-1].index("subagents") - 2])

    seen = {}  # message id -> [first_ts, model, toks, web_search, web_fetch, source]
    comps = []
    n_files = collections.Counter()
    unreadable = 0
    sessions = collections.Counter()  # project folder -> matched sessions
    for sid, fl in files.items():
        # a file last written before the window cannot hold a message inside it
        fl = [(k, p) for k, p in fl if os.path.getmtime(p) >= t0.timestamp()]
        if not fl:
            continue
        scanned = []
        cwd_hit = False
        hit_cwds = set()
        for kind, p in fl:
            r = scan(p)
            if r is None:
                unreadable += 1
                continue
            cwds, msgs, cps = r
            hits = {c for c in cwds if a.match in c}
            if hits:
                cwd_hit = True
                hit_cwds |= hits
            scanned.append((kind, msgs, cps))
        if not cwd_hit:
            continue
        if a.exclude and all(any(x in c for x in a.exclude) for c in hit_cwds):
            continue
        in_window = False
        for kind, msgs, cps in scanned:
            n_files[kind] += 1
            comps.extend(cps)
            for mid, (ts, model, toks, ws, wf) in msgs.items():
                prev = seen.get(mid)
                if prev is None:
                    seen[mid] = [ts, model, toks, ws, wf, kind]
                else:
                    prev[0] = min(prev[0], ts)
                    if toks["output"] >= prev[2]["output"]:
                        prev[1:5] = [model, toks, ws, wf]
                    if kind == "main":
                        prev[5] = "main"
        sessions[proj_of.get(sid, "?")] += 1

    by_model = collections.defaultdict(collections.Counter)
    by_source = collections.defaultdict(collections.Counter)
    n_model = collections.Counter()
    n_source = collections.Counter()
    synthetic = 0
    web_s = web_f = 0
    first = last = None
    for mid, (ts, model, toks, ws, wf, src) in seen.items():
        t = utc(ts)
        if not (t0 <= t < t1):
            continue
        if model == "<synthetic>":
            synthetic += 1
            continue
        by_model[model].update(toks)
        by_source[src].update(toks)
        n_model[model] += 1
        n_source[src] += 1
        web_s += ws
        web_f += wf
        first = t if first is None or t < first else first
        last = t if last is None or t > last else last
    win_comps = [c for c in comps if t0 <= utc(c[0]) < t1]

    total = collections.Counter()
    for c in by_model.values():
        total.update(c)
    fmt = "{:<24s} {:>8s} " + " ".join("{:>15s}" for _ in TYPES)
    num = lambda v: f"{v:,}"
    print(f"window (UTC): {t0:%Y-%m-%dT%H:%M:%SZ} -> {t1:%Y-%m-%dT%H:%M:%SZ}   cwd contains: {a.match!r}"
          + (f"   excluded: {a.exclude}" if a.exclude else ""))
    print(f"sessions matched: {sum(sessions.values())}"
          + (f"   by project folder: {dict(sessions)}" if a.show_scope else ""))
    print(f"transcript files read: {n_files['main']} session, {n_files['agent']} agent;   unreadable: {unreadable}")
    print(f"messages counted (unique ids, first timestamp in window): {sum(n_model.values()):,}"
          f"   '<synthetic>' placeholders excluded: {synthetic}")
    if first and a.show_scope:
        print(f"first / last counted message: {first:%Y-%m-%dT%H:%M:%SZ} / {last:%Y-%m-%dT%H:%M:%SZ}")
    print()
    print(fmt.format("by model", "messages", *TYPES))
    for model in sorted(by_model):
        print(fmt.format(model[:24], num(n_model[model]), *(num(by_model[model][k]) for k in TYPES)))
    print(fmt.format("TOTAL", num(sum(n_model.values())), *(num(total[k]) for k in TYPES)))
    print()
    print(fmt.format("by source", "messages", *TYPES))
    for src in ("main", "agent"):
        if n_source[src]:
            print(fmt.format("main session" if src == "main" else "agents and workflows", num(n_source[src]),
                             *(num(by_source[src][k]) for k in TYPES)))
    print()
    print(f"cache_write total: {total['cache_write_5m'] + total['cache_write_1h'] + total['cache_write_unsplit']:,}"
          f"   all four types (a size, not a cost): {sum(total.values()):,}")
    print(f"server-side requests: web_search {web_s}, web_fetch {web_f}")
    print(f"compactions in window: {len(win_comps)}"
          + (f"   (the summary call leaves no usage record; context before: {sum(c[1] for c in win_comps):,},"
             f" after: {sum(c[2] for c in win_comps):,} tokens)" if win_comps else ""))


if __name__ == "__main__":
    main()
