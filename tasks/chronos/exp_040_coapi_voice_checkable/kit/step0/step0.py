#!/usr/bin/env python3
"""exp_040 step 0 — the quantiser (STEP0.md): build blind batches, collect, analyse.

  python3.14 step0.py build   OUT_DIR   # OUT_DIR/cells.json: {"B0i:v1": run dir, "B0u:v1": run dir, "B0rq8:v1": run dir}
  python3.14 step0.py collect OUT_DIR
  python3.14 step0.py analyze OUT_DIR

Rows: the 40 v1 rows flagged substance_in_context = true in exp_039's main scoring.
Three answers per row (B0i, B0u, B0rq8), shuffled per row (seed "40:s0:<id>"), unlabelled; 5 rows per batch.
Rule (DESIGN.md §8, STEP0.md, fixed before any answer existed):
  d = mean correctness(B0i) - mean correctness(B0rq8), same batches.
  d >= -0.10 -> adopt the imatrix Q4_K_M recipe; else Q8_0 is exp_040's primary quantisation.
CI (stratified paired bootstrap, lang x type, 10,000, seed 40) and exact sign-flip p are descriptive.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

MAIN = Path.home() / "REPOS/casasol/coapi_voice/runs/exp039_results"
ARMS = ["B0i", "B0u", "B0rq8"]
REF = {"B0rq8_exp039_q8_control": 1.425, "B0r_q4_exp039": 1.20, "A12_exp039": 1.425}


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
        random.Random(f"40:s0:{rid}").shuffle(arms)
        answers = []
        for slot, arm in zip("abc", arms):
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
                    J.append({"id": rid, "arm": key[(rid, a["slot"])], "row_substance_in_context": r.get("substance_in_context"),
                              **{k: v for k, v in a.items() if k != "slot"}})
    assert len(J) == len(key), (len(J), len(key))
    (out / "judge.jsonl").write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in J))
    print(f"collected {len(J)}")


def sign_flip_p(diffs):
    """Exact two-sided sign-flip p for integer paired differences (distribution of the signed sum by DP)."""
    nz = [abs(int(d)) for d in diffs if d != 0]
    dist = {0: 1}
    for v in nz:
        nxt = defaultdict(int)
        for s, c in dist.items():
            nxt[s + v] += c
            nxt[s - v] += c
        dist = nxt
    obs = abs(sum(diffs))
    return sum(c for s, c in dist.items() if abs(s) >= obs) / 2 ** len(nz)


def paired(J, meta, a, b, seed=40):
    strata, diffs = defaultdict(list), []
    for rid in rows40():
        d = J[(rid, a)]["correctness"] - J[(rid, b)]["correctness"]
        strata[(meta[rid]["lang"], meta[rid]["type"])].append(d)
        diffs.append(d)
    rng = random.Random(seed)
    groups = [v for v in strata.values() if v]
    means = sorted(sum(x for v in groups for x in (rng.choice(v) for _ in v)) / len(diffs) for _ in range(10000))
    return {"diff": round(sum(diffs) / len(diffs), 4), "ci95": [round(means[250], 4), round(means[9749], 4)],
            "sign_flip_p_two_sided": round(sign_flip_p(diffs), 4),
            "up": sum(d > 0 for d in diffs), "down": sum(d < 0 for d in diffs), "same": sum(d == 0 for d in diffs)}


def analyze(out):
    J = {(j["id"], j["arm"]): j for j in map(json.loads, (out / "judge.jsonl").read_text().splitlines())}
    meta = load(json.loads((out / "cells.json").read_text())["B0rq8:v1"])
    ids = rows40()
    corr = {a: round(sum(J[(rid, a)]["correctness"] for rid in ids) / 40, 4) for a in ARMS}
    useful = {a: round(sum(J[(rid, a)]["usefulness"] for rid in ids) / 40, 3) for a in ARMS}
    harm = {a: sorted(rid for rid in ids if J[(rid, a)].get("harmful")) for a in ARMS}
    primary = paired(J, meta, "B0i", "B0rq8")
    d = primary["diff"]
    decision = ("ADOPT imatrix Q4_K_M for exp_040 (d >= -0.10)" if d >= -0.10 - 1e-9 else
                "Q8_0 is exp_040's primary quantisation (d < -0.10)")
    a12 = {json.loads(l)["id"]: json.loads(l) for l in (MAIN / "judge.jsonl").read_text().splitlines()
           if l.strip() and json.loads(l).get("arm") == "A12" and json.loads(l).get("set", "v1") == "v1"}
    res = {"decision": decision, "primary_B0i_minus_B0rq8": primary, "mean_correctness": corr, "mean_usefulness": useful,
           "harmful": harm,
           "descriptive": {"B0u_minus_B0rq8": paired(J, meta, "B0u", "B0rq8"), "B0i_minus_B0u": paired(J, meta, "B0i", "B0u"),
                           "judge_drift_B0rq8": {"now": corr["B0rq8"], "exp039_q8_control": REF["B0rq8_exp039_q8_control"]},
                           "A12_minus_B0u_cross_session": round(sum(a12[r]["correctness"] for r in ids if r in a12) / max(1, sum(r in a12 for r in ids)) - corr["B0u"], 4) if a12 else None,
                           "B0r_q4_exp039_cross_session": REF["B0r_q4_exp039"]}}
    (out / "verdict.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"build": build, "collect": collect, "analyze": analyze}[sys.argv[1]](Path(sys.argv[2]))
