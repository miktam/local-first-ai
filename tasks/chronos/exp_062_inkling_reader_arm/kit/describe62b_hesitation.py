#!/usr/bin/env python3
"""exp_062 Addendum B: hesitation words in the reasoning, per decision and per arm. Declared in ADDENDUM_B.md on
2026-11-22 while the run was in progress and before any research-arm result had been read. Descriptive: it decides
nothing. Counts only; no reasoning text is printed.

    python3.14 -I kit/describe62b_hesitation.py runs_b/<ts> <the private folder of that run>     writes hesitation.json

The words are the three the maker names as its examples of "uncertainty indicators" in the announcement of Inkling 2
Mini ("wait", "actually", "hmm"; 9.7, 3.9 and 3.6 an answer at 1, 3 and 5 passes on GPQA Diamond). Whole words, any
case. The maker's own list may be longer; this one is fixed here.
"""
import json
import re
import statistics
import sys
from pathlib import Path

WORDS = re.compile(r"\b(?:wait|actually|hmm+)\b", re.I)


def main():
    run, private = Path(sys.argv[1]), Path(sys.argv[2]).expanduser()
    out = {"words": ["wait", "actually", "hmm"]}
    for arm in ("G", "I", "V1", "V2", "V3", "V4", "V5"):
        p = private / f"reasoning-{arm}.jsonl"
        if not p.exists():
            continue
        per, chars = {}, {}
        res = run / arm / "results.jsonl"       # decisions that ended at the context wall are left out, and counted
        out_ids = {r["id"] for r in map(json.loads, open(res, encoding="utf-8")) if r.get("end") == "context"} if res.exists() else set()
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["id"] in out_ids:
                    continue
                per[r["id"]] = per.get(r["id"], 0) + len(WORDS.findall(r["reasoning"]))
                chars[r["id"]] = chars.get(r["id"], 0) + len(r["reasoning"])
        total_chars = sum(chars.values())
        if not per:
            continue
        out[arm] = {"decisions": len(per), "left_out_at_the_context_wall": len(out_ids), "mean_per_decision": round(statistics.mean(per.values()), 2),
                    "median_per_decision": statistics.median(per.values()), "decisions_with_none": sum(1 for v in per.values() if v == 0),
                    "per_1000_reasoning_chars": round(1000 * sum(per.values()) / total_chars, 3) if total_chars else None}
    (run / "hesitation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
