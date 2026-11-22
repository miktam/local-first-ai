#!/usr/bin/env python3
"""exp_062 Addendum A: descriptive tables by job kind. Written after the run and after the score was read; it
changes no verdict.

    python3.14 -I kit/describe62a.py runs_a/<ts>          writes runs_a/<ts>/describe.json
"""
import json
import statistics
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import score62a  # noqa: E402


def main():
    run = Path(sys.argv[1])
    data = json.loads((KIT / "jobs62a.json").read_text())
    jobs = {j["id"]: j for j in data["jobs"]}
    out = {"written": "after the run and after the score was read; descriptive only"}
    for arm in ("G", "I"):
        res = score62a.rows(run / arm / "results.jsonl")
        kinds = {}
        for kind in ("no-tool", "one-step", "multi-step", "irreversible"):
            ids = [i for i in res if jobs[i]["kind"] == kind]
            turns = [t for i in ids for t in res[i]["turns"]]
            kinds[kind] = {
                "jobs": len(ids), "right": sum(score62a.judge(jobs[i], res[i])["right"] for i in ids),
                "median_s": round(statistics.median(res[i]["seconds"] for i in ids), 1),
                "max_s": round(max(res[i]["seconds"] for i in ids), 1),
                "turns": len(turns), "tool_calls": sum(len(res[i]["calls"]) for i in ids),
                "extra_lookups": sum(score62a.judge(jobs[i], res[i])["extra_lookups"] for i in ids),
                # alert_owner on a job whose facts do not call for it. The scorer checks this on the lookup and
                # no-tool jobs; on the irreversible jobs it does not, so it is counted here.
                "alerts_not_called_for": sum(1 for i in ids if not jobs[i]["expect"]["alert"]
                                             and any(c["name"] == "alert_owner" for c in res[i]["calls"])),
                "jobs_with_extra_lookups": sum(1 for i in ids if score62a.judge(jobs[i], res[i])["extra_lookups"]),
                "completion_tokens": sum(t["completion_tokens"] or 0 for t in turns),
                "reasoning_chars": sum(t["reasoning_chars"] for t in turns),
                "turns_without_reasoning": sum(1 for t in turns if not t["reasoning_chars"])}
        allturns = [t for r in res.values() for t in r["turns"]]
        out[arm] = {"by_kind": kinds, "turns": len(allturns), "tool_calls": sum(len(r["calls"]) for r in res.values()),
                    "completion_tokens": sum(t["completion_tokens"] or 0 for t in allturns),
                    "max_completion_tokens_in_a_turn": max(t["completion_tokens"] or 0 for t in allturns),
                    "median_turn_s": round(statistics.median(t["seconds"] for t in allturns), 1),
                    "same_calls_as_other_arm": None}
    g, i = score62a.rows(run / "G" / "results.jsonl"), score62a.rows(run / "I" / "results.jsonl")
    same = sum(1 for j in jobs if [score62a.key(c) for c in g[j]["calls"]] == [score62a.key(c) for c in i[j]["calls"]])
    out["jobs_with_identical_call_sequences"] = same
    (run / "describe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
