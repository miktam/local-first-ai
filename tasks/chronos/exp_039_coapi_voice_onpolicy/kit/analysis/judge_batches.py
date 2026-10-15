#!/usr/bin/env python3
"""exp_039 — blind judging batches for the scoring cells, and collection of the judge output.

  python3.14 judge_batches.py build   OUT_DIR   # reads OUT_DIR/cells.json, writes batches/ + key.jsonl
  python3.14 judge_batches.py collect OUT_DIR   # reads replies/*.jsonl, writes judge.jsonl + judge_rows.jsonl

Per the pre-registration: the arms' answers to one row are presented together, shuffled (seed 39 per row),
unlabelled; the judge never sees arm names; the key lives in key.jsonl, which judges are told not to open.
Five rows per batch file; one fresh Fable instance per batch.
"""
import json
import random
import sys
from pathlib import Path

ARMS = ["B0r", "R1", "A12", "R1G"]
SETS = ["v1", "v21"]
SLOTS = "abcd"
PER_BATCH = 5


def load(run):
    return {json.loads(l)["id"]: json.loads(l) for l in (Path(run) / "outputs.jsonl").read_text().splitlines() if l.strip()}


def build(out):
    cells = json.loads((out / "cells.json").read_text())
    (out / "batches").mkdir(exist_ok=True)
    rows, key = [], []
    for s in SETS:
        data = {a: load(cells[f"{a}:{s}"]) for a in ARMS}
        for rid in sorted(data["B0r"]):
            base = data["B0r"][rid]
            arms = ARMS[:]
            random.Random(f"39:{s}:{rid}").shuffle(arms)
            answers = []
            for slot, arm in zip(SLOTS, arms):
                r = data[arm][rid]
                ans = {"slot": slot, "answer": r["answer"], "citations_dropped": r.get("citations_dropped", []),
                       "scope": r.get("scope", "in")}
                if r["context"] != base["context"]:
                    ans["retrieved_material_for_this_answer"] = r["context"] or "(none: this answer was produced without retrieval)"
                answers.append(ans)
                key.append({"set": s, "id": rid, "slot": slot, "arm": arm})
            rows.append({"row": f"{s}:{rid}", "language": base["lang"], "type": base["type"], "question": base["q"],
                         "gold": base["gold"], "retrieved_material": base["context"], "answers": answers})
    for i in range(0, len(rows), PER_BATCH):
        (out / "batches" / f"batch_{i // PER_BATCH + 1:02d}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + PER_BATCH]), encoding="utf-8")
    (out / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    print(f"{len(rows)} rows, {len(key)} answers, {(len(rows) + PER_BATCH - 1) // PER_BATCH} batches")


def collect(out):
    key = {(k["set"], k["id"], k["slot"]): k["arm"] for k in map(json.loads, (out / "key.jsonl").read_text().splitlines())}
    J, R = [], {}
    for p in sorted((out / "replies").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            s, rid = r["row"].split(":", 1)
            R[(s, rid)] = {"set": s, "id": rid, "substance_in_context": r["substance_in_context"]}
            for a in r["answers"]:
                J.append({"set": s, "id": rid, "arm": key[(s, rid, a["slot"])], **{k: v for k, v in a.items() if k != "slot"}})
    want = len(key)
    assert len(J) == want, f"{len(J)} judged answers, {want} expected"
    (out / "judge.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in J))
    (out / "judge_rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in R.values()))
    print(f"collected {len(J)} answers over {len(R)} rows")


if __name__ == "__main__":
    {"build": build, "collect": collect}[sys.argv[1]](Path(sys.argv[2]))
