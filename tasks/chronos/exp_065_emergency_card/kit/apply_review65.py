"""exp_065: make the frozen task file from the writers' output and the reviewer's patches. No hand edits.
    python3.14 -I kit/apply_review65.py
Reads writer/tasks-types-1-3.json, writer/tasks-types-4-6.json and writer/review-patches.json; writes kit/tasks65.json
and prints its sha256. The reviewer's convention: indexes are 0-based and refer to the original numbering; in a task,
replacements are applied before deletions. Every patch must apply exactly, or nothing is written."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
tasks = {}
for name in ("tasks-types-1-3.json", "tasks-types-4-6.json"):
    for t in json.loads((EXP / "writer" / name).read_text()):
        assert t["id"] not in tasks
        t.setdefault("must_not", [])
        tasks[t["id"]] = t
assert len(tasks) == 48, len(tasks)

by_task = defaultdict(list)
for p in json.loads((EXP / "writer" / "review-patches.json").read_text()):
    by_task[p["id"]].append(p)

applied = 0
for tid, patches in by_task.items():
    t = tasks[tid]
    deletes = defaultdict(list)
    for p in patches:
        f = p["field"]
        if f == "prompt":
            assert t["prompt"].count(p["find"]) == 1, (tid, "prompt text not found exactly once", p["find"][:50])
            t["prompt"] = t["prompt"].replace(p["find"], p["replace"])
        elif f == "judge_note":
            t["judge_note"] = p["new"]
        elif f in ("must", "must_not"):
            if "append" in p:
                t[f].append(p["append"])
            elif p.get("delete"):
                assert 0 <= p["index"] < len(t[f]), (tid, f, "index")
                deletes[f].append(p["index"])
            else:
                assert 0 <= p["index"] < len(t[f]), (tid, f, "index")
                t[f][p["index"]] = p["new"]
        else:
            raise AssertionError((tid, "unknown field", f))
        applied += 1
    for f, idxs in deletes.items():
        assert len(set(idxs)) == len(idxs)
        for i in sorted(idxs, reverse=True):
            del t[f][i]

out = []
for tid in sorted(tasks):
    t = tasks[tid]
    assert 3 <= len(t["must"]) <= 5, (tid, "must-points", len(t["must"]))
    assert len(t["must_not"]) <= 2, (tid, "must_not points", len(t["must_not"]))
    row = {"id": tid, "type": t["type"]}
    if "class" in t:
        row["class"] = t["class"]
    row.update({"pattern": t.get("pattern", ""), "prompt": t["prompt"], "must": t["must"], "must_not": t["must_not"],
                "judge_note": t.get("judge_note", "")})
    out.append(row)

text = json.dumps(out, ensure_ascii=False, indent=1) + "\n"
(EXP / "kit" / "tasks65.json").write_text(text)
print(f"{applied} patches applied to {len(by_task)} tasks; {len(out)} tasks written")
print("must-points per task:", sorted({len(t['must']) for t in out}), "| words in the longest prompt:", max(len(t["prompt"].split()) for t in out))
print("sha256", hashlib.sha256(text.encode()).hexdigest())
