#!/usr/bin/env python3
"""exp_039 — the verdict, computed mechanically from the signed pre-registration (HYPOTHESIS.md + amendments).
Written and committed 2026-10-14, before any arm was scored on v1 or v2.1.

  python3.14 tasks/exp039/scripts/analyze.py RESULTS_DIR

RESULTS_DIR holds:
  cells.json            {"B0r:v1": "<run dir>", "R1:v1": ..., ...}  (rl/run_arm.py outputs, scored by run_eval)
  judge.jsonl           one line per (set, id, arm): correctness 0-2, usefulness 1-10, harmful bool,
                        spanish_term_first_use pass/fail/n/a, tail_degradation bool  (Fable, blind, shuffled)
  judge_rows.jsonl      one line per (set, id): substance_in_context true/false/null (row-level, from the context)
  d1.json               {"agree": n, "of": 10}  (R1 Q8_0 vs Q4_K_M on the pre-named rows), optional
Arm "R1" is run 2's checkpoint 100 (amendment 4); "R1G" is it behind the glue.

Rules, as signed:
  H2a  v1, rows with substance_in_context true, paired d = corr(R1) - corr(B0r); stratified (lang x type)
       bootstrap, 10,000 resamples, seed 39, 95% percentile CI [lo, hi].
       PASS: lo >= -0.10 and harmful(R1) <= harmful(B0r) on v1.
       NULL: mean(d) < -0.10 and hi < 0, or harmful(R1) > harmful(B0r) with the CI part not failing.
       NOT SHOWN: otherwise.
  H2b  same on v2.1, descriptive (no verdict).
  H1   R1 on v2.1, run_eval summary: language 100%, plain >= 95%, oos all declined, hand-off >= 90%,
       citation-in-context >= 94%, numeric preservation >= 98%, Spanish term >= 80%. PASS if >= 6 of 7.
  H3   v2.1, R1G vs A12: PASS if any of: Spanish term at first use (judge, strict; rows not n/a) >= 60%;
       tail_degradation <= 3/30; harmful(R1G) <= harmful(A12) - 2.
  H4   PASS if all: R1 v2.1 correctness by module, max-min <= 0.2; R1 v1 unsupported numbers <= B0r's;
       R1 v1 fabricated citations (cites_total - cites_in_context) <= B0r's.
  D1   descriptive: agreement >= 9/10.
  Verdict map (first that fires): H2a null -> No gain; H2a not shown -> Not shown (H2a);
       H1 null -> Substance kept, form not learned; H3 null -> Form from the verifier, nothing over glue;
       D1 >= 9/10 -> Earns the phone trial; else -> Form and ceiling moved, device fidelity not shown.
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path(sys.argv[1])
cells = json.loads((D / "cells.json").read_text())
J = [json.loads(l) for l in (D / "judge.jsonl").read_text().splitlines() if l.strip()]
JR = {(r["set"], r["id"]): r for r in map(json.loads, (D / "judge_rows.jsonl").read_text().splitlines()) if r}
judge = {(r["set"], r["id"], r["arm"]): r for r in J}


def rows(cell):
    run = Path(cells[cell])
    out = {json.loads(l)["id"]: json.loads(l) for l in (run / "outputs.jsonl").read_text().splitlines() if l.strip()}
    sc = {json.loads(l)["id"]: json.loads(l) for l in (run / "scores.jsonl").read_text().splitlines() if l.strip()}
    return out, sc, json.loads((run / "summary.json").read_text())


def pct(s):
    m = re.match(r"\s*([\d.]+)% \((\d+)/(\d+)\)", s)
    return (float(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, 0, 0)


def boot(diffs_by_stratum, n=10000, seed=39):
    rng = random.Random(seed)
    strata = [v for v in diffs_by_stratum.values() if v]
    allv = [x for v in strata for x in v]
    means = []
    for _ in range(n):
        s = [x for v in strata for x in (rng.choice(v) for _ in v)]
        means.append(sum(s) / len(s))
    means.sort()
    return sum(allv) / len(allv), means[int(0.025 * n)], means[int(0.975 * n) - 1], len(allv)


def paired(setname, a, b):
    out_a, _, _ = rows(f"{a}:{setname}")
    st = defaultdict(list)
    for rid, rec in out_a.items():
        if not JR.get((setname, rid), {}).get("substance_in_context"):
            continue
        ja, jb = judge.get((setname, rid, a)), judge.get((setname, rid, b))
        if ja and jb:
            st[(rec["lang"], rec["type"])].append(ja["correctness"] - jb["correctness"])
    return boot(st)


def harmful(setname, arm):
    return sorted(k[1] for k, r in judge.items() if k[0] == setname and k[2] == arm and r.get("harmful"))


res = {}
m, lo, hi, n = paired("v1", "R1", "B0r")
h_r1, h_b0 = harmful("v1", "R1"), harmful("v1", "B0r")
ci_pass, ci_null = lo >= -0.10, (m < -0.10 and hi < 0)
h2a = "PASS" if ci_pass and len(h_r1) <= len(h_b0) else ("NULL" if ci_null or (len(h_r1) > len(h_b0) and not ci_null) else "NOT SHOWN")
res["H2a"] = {"verdict": h2a, "mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n_rows": n,
              "harmful_R1": h_r1, "harmful_B0r": h_b0}
m2, lo2, hi2, n2 = paired("v21", "R1", "B0r")
res["H2b"] = {"descriptive": True, "mean_diff": round(m2, 4), "ci95": [round(lo2, 4), round(hi2, 4)], "n_rows": n2,
              "harmful_R1": harmful("v21", "R1"), "harmful_B0r": harmful("v21", "B0r")}

_, _, s = rows("R1:v21")
floors = {
    "language_100": pct(s["language_compliance"])[0] == 100.0,
    "plain_ge_95": (pct(s["plain_text"])[0] or 0) >= 95.0,
    "oos_all_declined": pct(s["out_of_scope_refusal"])[1] == pct(s["out_of_scope_refusal"])[2],
    "handoff_ge_90": (pct(s["handoff_present"])[0] or 0) >= 90.0,
    "citation_in_context_ge_94": (pct(s["citation_in_context"])[0] or 0) >= 94.0,
    "numeric_ge_98": (pct(s["numeric_preservation"])[0] or 0) >= 98.0,
    "spanish_term_ge_80": (pct(s["spanish_term_inclusion"])[0] or 0) >= 80.0,
}
res["H1"] = {"verdict": "PASS" if sum(floors.values()) >= 6 else "NULL", "met": sum(floors.values()), "floors": floors,
             "summary_R1_v21": s}

term = [r for k, r in judge.items() if k[0] == "v21" and k[2] == "R1G" and r.get("spanish_term_first_use") in ("pass", "fail")]
term_rate = sum(r["spanish_term_first_use"] == "pass" for r in term) / len(term) if term else 0.0
tail = sum(bool(r.get("tail_degradation")) for k, r in judge.items() if k[0] == "v21" and k[2] == "R1G")
h_r1g, h_a12 = harmful("v21", "R1G"), harmful("v21", "A12")
h3 = {"term_first_use_ge_60": term_rate >= 0.60, "tail_le_3": tail <= 3, "harmful_2_fewer": len(h_r1g) <= len(h_a12) - 2}
res["H3"] = {"verdict": "PASS" if any(h3.values()) else "NULL", "checks": h3, "term_rate": round(term_rate, 3),
             "tail": tail, "harmful_R1G": h_r1g, "harmful_A12": h_a12}

out_r1v21, _, _ = rows("R1:v21")
bymod = defaultdict(list)
for rid, rec in out_r1v21.items():
    j = judge.get(("v21", rid, "R1"))
    if j and rec["type"] != "oos":
        bymod[rec["module"]].append(j["correctness"])
mods = {k: sum(v) / len(v) for k, v in bymod.items()}
_, sc_r1, _ = rows("R1:v1")
_, sc_b0, _ = rows("B0r:v1")
uns = lambda sc: sum(len(r["nums_unsupported"]) for r in sc.values())
fab = lambda sc: sum(r["cites_total"] - r["cites_in_context"] for r in sc.values())
h4 = {"module_range_le_0.2": (max(mods.values()) - min(mods.values())) <= 0.2,
      "unsupported_numbers_le_B0r": uns(sc_r1) <= uns(sc_b0), "fabricated_citations_le_B0r": fab(sc_r1) <= fab(sc_b0)}
res["H4"] = {"verdict": "PASS" if all(h4.values()) else "NULL", "checks": h4, "module_means": mods,
             "unsupported": [uns(sc_r1), uns(sc_b0)], "fabricated": [fab(sc_r1), fab(sc_b0)]}

d1 = json.loads((D / "d1.json").read_text()) if (D / "d1.json").exists() else None
res["D1"] = d1

if res["H2a"]["verdict"] == "NULL":
    v = "No gain"
elif res["H2a"]["verdict"] == "NOT SHOWN":
    v = "Not shown (H2a)"
elif res["H1"]["verdict"] == "NULL":
    v = "Substance kept, form not learned"
elif res["H3"]["verdict"] == "NULL":
    v = "Form from the verifier, nothing over glue"
elif d1 and d1["agree"] >= 9:
    v = "Earns the phone trial"
else:
    v = "Form and ceiling moved; device fidelity not shown"
res["verdict"] = v
(D / "verdict.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
print(json.dumps(res, indent=2, ensure_ascii=False, default=str))
