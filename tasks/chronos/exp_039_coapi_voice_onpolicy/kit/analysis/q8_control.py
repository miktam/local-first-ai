#!/usr/bin/env python3
"""exp_039 amendment 5 — the Q8_0 quantisation control: build blind batches, collect, analyse.

  python3.14 q8_control.py build   OUT_DIR   # OUT_DIR/cells.json: {"B0rq8:v1": run dir, "R1q8:v1": run dir}
  python3.14 q8_control.py collect OUT_DIR
  python3.14 q8_control.py analyze OUT_DIR

Rows: the 40 v1 rows flagged substance_in_context = true in the main scoring (exp039_results/judge_rows.jsonl).
Two answers per row (B0-r-Q8, R2-Q8), shuffled per row (seed "39:q8:<id>"), unlabelled; 5 rows per batch.
Primary rule (amendment 5, fixed before any Q8 answer existed):
  "Training gain shown at Q8_0" iff stratified paired bootstrap (lang x type, 10,000, seed 39) 95% lower > 0
  AND exact one-sided sign-flip permutation p < 0.05; "Not worse at Q8_0" if lower >= -0.10; else "not shown".
"""
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MAIN = Path.home() / "REPOS/casasol/coapi_voice/runs/exp039_results"
ARMS = ["B0rq8", "R1q8"]


def load(run):
    return {json.loads(l)["id"]: json.loads(l) for l in (Path(run) / "outputs.jsonl").read_text().splitlines() if l.strip()}


def rows40():
    return sorted(r["id"] for r in map(json.loads, (MAIN / "judge_rows.jsonl").read_text().splitlines())
                  if r["set"] == "v1" and r["substance_in_context"])


def build(out):
    cells = json.loads((out / "cells.json").read_text())
    data = {a: load(cells[f"{a}:v1"]) for a in ARMS}
    ids = rows40()
    assert len(ids) == 40, len(ids)
    (out / "batches").mkdir(exist_ok=True)
    rows, key = [], []
    for rid in ids:
        base = data["B0rq8"][rid]
        arms = ARMS[:]
        random.Random(f"39:q8:{rid}").shuffle(arms)
        answers = []
        for slot, arm in zip("ab", arms):
            r = data[arm][rid]
            a = {"slot": slot, "answer": r["answer"], "citations_dropped": r.get("citations_dropped", []), "scope": r.get("scope", "in")}
            if r["context"] != base["context"]:
                a["retrieved_material_for_this_answer"] = r["context"]
            answers.append(a)
            key.append({"set": "v1", "id": rid, "slot": slot, "arm": arm})
        rows.append({"row": f"v1:{rid}", "language": base["lang"], "type": base["type"], "question": base["q"],
                     "gold": base["gold"], "retrieved_material": base["context"], "answers": answers})
    for i in range(0, len(rows), 5):
        (out / "batches" / f"batch_{i // 5 + 1:02d}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i:i + 5]))
    (out / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    print(f"{len(rows)} rows, {len(key)} answers, {len(rows) // 5} batches")


def collect(out):
    key = {(k["id"], k["slot"]): k["arm"] for k in map(json.loads, (out / "key.jsonl").read_text().splitlines())}
    J = []
    for p in sorted((out / "replies").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rid = r["row"].split(":", 1)[1]
                for a in r["answers"]:
                    J.append({"id": rid, "arm": key[(rid, a["slot"])], **{k: v for k, v in a.items() if k != "slot"}})
    assert len(J) == len(key), (len(J), len(key))
    (out / "judge.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in J))
    print(f"collected {len(J)}")


def analyze(out):
    J = {(j["id"], j["arm"]): j for j in map(json.loads, (out / "judge.jsonl").read_text().splitlines())}
    meta = load(json.loads((out / "cells.json").read_text())["B0rq8:v1"])
    strata = defaultdict(list)
    diffs = []
    for rid in rows40():
        d = J[(rid, "R1q8")]["correctness"] - J[(rid, "B0rq8")]["correctness"]
        strata[(meta[rid]["lang"], meta[rid]["type"])].append(d)
        diffs.append(d)
    rng = random.Random(39)
    groups = [v for v in strata.values() if v]
    means = sorted(sum(x for v in groups for x in (rng.choice(v) for _ in v)) / len(diffs) for _ in range(10000))
    mean, lo, hi = sum(diffs) / len(diffs), means[250], means[9749]
    nz = [d for d in diffs if d != 0]
    obs = sum(nz)
    p = sum(1 for signs in itertools.product((1, -1), repeat=len(nz)) if sum(s * abs(d) for s, d in zip(signs, nz)) >= obs) / 2 ** len(nz)
    verdict = ("Training gain shown at Q8_0" if lo > 0 and p < 0.05 else
               "Not worse at Q8_0" if lo >= -0.10 else "Not shown")
    harm = {a: sorted(rid for (rid, arm), j in J.items() if arm == a and j.get("harmful")) for a in ARMS}
    corr = {a: sum(J[(rid, a)]["correctness"] for rid in rows40()) / 40 for a in ARMS}
    res = {"verdict": verdict, "mean_diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "sign_flip_p_one_sided": round(p, 4),
           "up": sum(d > 0 for d in diffs), "down": sum(d < 0 for d in diffs), "same": sum(d == 0 for d in diffs),
           "mean_correctness": corr, "harmful": harm,
           "descriptive_vs_q4_main_run": {"B0r_q4": 1.20, "R1_q4": 1.40, "note": "different judging session; drift applies"}}
    (out / "verdict.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"build": build, "collect": collect, "analyze": analyze}[sys.argv[1]](Path(sys.argv[2]))
