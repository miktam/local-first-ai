#!/usr/bin/env python3
"""exp_056: what the model step asked, per message, in the re-run (Amendment 4), from the original run folders.

Written for the re-run's result on 2026-11-10, after the run; it is not part of the kit pinned in Amendment 3. With
the fix, the check's verdict records, for every message, the views of it that the model step was asked about. This
script reads, from each run's `subject/runs/<run_id>/verdict.json`, only the model step's block (`persona`) and in
it only:
  - `views`: for each view its paragraph count, paragraphs dropped, length in characters, whether it answered,
    its points and its three closed labels (`approach`, `asks`, `confidence`), and whether an error was recorded
    (never the error's text);
  - `chosen`, `views_failed`, `views_capped`, and the step's `latency_s`.
No other field is read. Those folders also hold the messages; they are not published, so this script runs only on
the machine that holds the run folders. Its output is published as `runs/views-output.txt`. It prints counts, ids,
lengths and label names, never text.

  python3.14 kit/views56.py --run <six-r2 folder> --score runs/six-r2/score.json \
                            --run <n1-r2 folder> --score runs/n1-r2/score.json \
                            --p55 <exp055-views folder>/P --manifest55 ../exp_055_exposure_review_lures/data/MANIFEST.json

Amendment 4's added description: for every H1 or H2 violation of the re-run, whether the base's whole-message
question was among the variant's questions. A view of the variant is that question when it has the paragraph count
and the length of the base's whole-message view (the variant is the base, a blank line, then the text).
"""
import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

LABELS = ("approach", "asks", "confidence")


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def model_step(run_dir):
    """The selected fields of the model step's block, nothing else."""
    per = json.loads((run_dir / "verdict.json").read_text()).get("persona") or {}
    views = []
    for v in per.get("views") or []:
        out = v.get("output") or {}
        views.append({"paragraphs": v.get("paragraphs"), "dropped": v.get("dropped"), "chars": v.get("chars"),
                      "ok": v.get("ok"), "points": v.get("points"), "error": v.get("error") is not None,
                      "labels": {k: out.get(k) for k in LABELS}})
    return {"views": views, "chosen": per.get("chosen"), "views_failed": per.get("views_failed"),
            "views_capped": per.get("views_capped"), "latency_s": per.get("latency_s")}


def same_question(a, b):
    return a["paragraphs"] == b["paragraphs"] and a["chars"] == b["chars"]


def step_summary(name, recs, steps):
    print(f"  {name}")
    for step, items in steps.items():
        if not items:
            continue
        calls = [len(m["views"]) for _, m in items]
        lat = [m["latency_s"] for _, m in items if m["latency_s"] is not None]
        print(f"    {step:8} records {len(items)}: model calls {sum(calls)} ({sum(calls) / len(items):.2f} per message, "
              f"min {min(calls)}, max {max(calls)}); views_failed > 0: "
              f"{sum(1 for _, m in items if (m['views_failed'] or 0) > 0)}; views_capped: "
              f"{sum(1 for _, m in items if m['views_capped'])}; views not answered: "
              f"{sum(1 for _, m in items for v in m['views'] if not v['ok'])}; views with an error: "
              f"{sum(1 for _, m in items for v in m['views'] if v['error'])}; model-step seconds max "
              f"{max(lat) if lat else '-'}")


def exp056(run, score_path):
    recs = {r["key"]: r for r in jsonl(run / "results.jsonl")}
    score = json.loads(Path(score_path).read_text())
    ms = {k: model_step(run / "subject" / "runs" / r["run_id"]) for k, r in recs.items() if r.get("run_id")}
    print(f"== {run.name}")
    steps = {s: [(k, ms[k]) for k in sorted(ms) if recs[k]["step"] == s]
             for s in ("bases1", "variants", "bases2", "nollm", "dead")}
    step_summary("per step", recs, steps)

    print("  the view kept (index 0 is the whole message):")
    for s in ("bases1", "variants", "bases2"):
        c = Counter("whole" if m["chosen"] == 0 else "shorter" for _, m in steps[s])
        worth = [recs[k]["id"] for k, m in steps[s]
                 if m["chosen"] not in (None, 0) and m["views"][m["chosen"]]["points"] > m["views"][0]["points"]]
        drops = Counter(m["views"][m["chosen"]]["dropped"] for _, m in steps[s] if m["chosen"] not in (None, 0))
        print(f"    {s:8} {dict(c)}; a shorter view worth more points than the whole message: {len(worth)}; "
              f"paragraphs the kept shorter view drops: {dict(drops)}")
    print("  records whose points differ from the kept view's:",
          sum(1 for k, m in ms.items() if m["views"] and m["chosen"] is not None
              and recs[k]["persona_points"] != m["views"][m["chosen"]]["points"]))

    def base_of(k):
        return "bases1:" + recs[k]["base"]

    viol = [("H1", v) for v in score["H1"]["violations"]] + [("H2", v) for v in score["H2"]["violations"]]
    print(f"  Amendment 4's description: H1 violations {len(score['H1']['violations'])}, "
          f"H2 violations {len(score['H2']['violations'])}")
    for h, v in viol:
        k = "variants:" + v["variant"]
        whole = ms[base_of(k)]["views"][0]
        among = any(same_question(x, whole) for x in ms[k]["views"])
        print(f"    {h} {v['variant']}: the base's whole-message question among the variant's: {among}")
    if not viol:
        print("    no violation to describe")

    var = [(k, m) for k, m in steps["variants"]]
    among = sum(1 for k, m in var if any(same_question(x, ms[base_of(k)]["views"][0]) for x in m["views"]))
    print(f"  every variant: the base's whole-message question among the variant's questions: {among} of {len(var)}")
    extra = Counter(len(m["views"]) - len(ms[base_of(k)]["views"]) for k, m in var)
    print(f"  questions per variant minus questions per base: {dict(extra)}")

    asked = pts = labs = 0
    where = Counter()
    for k, m in var:
        for bv in ms[base_of(k)]["views"]:
            hit = [x for x in m["views"] if same_question(x, bv)]
            if not hit:
                continue
            asked += 1
            pts += hit[0]["points"] != bv["points"]
            diff = tuple(l for l in LABELS if hit[0]["labels"][l] != bv["labels"][l])
            if diff:
                labs += 1
                where[(recs[k]["base"], f"{bv['paragraphs']} paragraphs", "+".join(diff))] += 1
    print(f"  a base's questions asked again inside its variants: {asked}; different points {pts}; "
          f"different labels {labs} {dict(where)}")
    same12 = sum(1 for k, m in steps["bases1"]
                 if [(x["chars"], x["points"], x["labels"]) for x in m["views"]]
                 == [(x["chars"], x["points"], x["labels"]) for x in ms["bases2:" + recs[k]["id"]]["views"]])
    print(f"  bases1 against bases2, every view (length, points, labels) identical: {same12} of {len(steps['bases1'])}")
    paras = [m["views"][0]["paragraphs"] for _, m in steps["bases1"]]
    print(f"  paragraphs of the bases' whole-message views: min {min(paras)}, max {max(paras)}")


def exp055_p(p, manifest):
    """exp_055's runner records no run id: its run folders are matched to its records in order (it runs the ids
    sorted, one at a time), and the match is checked by the channel in the folder's name and by the model step's
    seconds not exceeding the record's."""
    man = json.loads(Path(manifest).read_text())
    recs = {r["id"]: r for r in jsonl(p / "results.jsonl")}
    ids = sorted(man)
    folders = sorted(d for d in (p / "subject" / "runs").iterdir() if d.is_dir())
    print(f"== exp_055 P re-run ({p.parent.name}/{p.name})")
    assert len(folders) == len(ids), (len(folders), len(ids))
    ms = {i: model_step(d) for i, d in zip(ids, folders)}
    chan = sum(1 for i, d in zip(ids, folders) if d.name.split("-", 2)[2] == man[i]["channel"])
    secs = sum(1 for i in ids if (ms[i]["latency_s"] or 0) <= recs[i]["seconds"])
    print(f"  folders matched to records in order: channel agrees {chan} of {len(ids)}; model-step seconds within "
          f"the record's {secs} of {len(ids)}")
    calls = [len(ms[i]["views"]) for i in ids]
    lat = [ms[i]["latency_s"] or 0 for i in ids]
    print(f"  model calls {sum(calls)} ({sum(calls) / len(ids):.2f} per message, min {min(calls)}, max {max(calls)}); "
          f"distribution {dict(sorted(Counter(calls).items()))}")
    print(f"  views_failed > 0: {sum(1 for i in ids if (ms[i]['views_failed'] or 0) > 0)}; views_capped: "
          f"{sum(1 for i in ids if ms[i]['views_capped'])}; views not answered: "
          f"{sum(1 for i in ids for v in ms[i]['views'] if not v['ok'])}; model-step seconds mean "
          f"{st.mean(lat):.2f}, max {max(lat)}")
    short = [i for i in ids if ms[i]["chosen"] not in (None, 0)]
    print(f"  a shorter view kept: {len(short)} of {len(ids)} "
          f"({dict(Counter(man[i]['kind'] for i in short))})")
    for i in short:
        m = ms[i]
        print(f"    {i} {man[i]['kind']:6} views {len(m['views'])}, kept view drops {m['views'][m['chosen']]['dropped']} "
              f"paragraph(s); points whole {m['views'][0]['points']} -> kept {m['views'][m['chosen']]['points']}; "
              f"the record counts {recs[i]['persona_points']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--score", action="append", default=[])
    ap.add_argument("--p55")
    ap.add_argument("--manifest55", default="../exp_055_exposure_review_lures/data/MANIFEST.json")
    a = ap.parse_args()
    assert len(a.run) == len(a.score), "one --score per --run"
    for run, score in zip(a.run, a.score):
        exp056(Path(run), score)
    if a.p55:
        exp055_p(Path(a.p55), a.manifest55)


if __name__ == "__main__":
    main()
