"""R2: the vendored scorer re-scores R1's A12 outputs to exp_035's published A12 summary exactly.

It copies an R1 run's outputs.jsonl into a fresh runs/r2_<UTC>/ directory and calls the vendored
`run_eval.cmd_score` on it, unchanged. Then it compares the result field by field:
- summary.json against exp_035's results/designA_A12_summary_20260922.json and against the
  reference run's summary.json, all keys except "run" (the directory name);
- scores.jsonl against the reference run's scores.jsonl, every row and field.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
REF = P38 / "reference" / "runs" / "glue_Qwen3-4B-GGUF-Q4-K-M_A12_20260922-2118"
E35_SUMMARY = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_035_coapi_voice/results/designA_A12_summary_20260922.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("r1_run", type=Path)
    a = ap.parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = P38 / "runs" / f"r2_{stamp}"
    out.mkdir(parents=True)
    shutil.copyfile(a.r1_run / "outputs.jsonl", out / "outputs.jsonl")

    sys.path.insert(0, str(VENDOR))
    import run_eval as ev  # noqa: E402
    ev.cmd_score(argparse.Namespace(run_dir=str(out)))

    got = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    res = {"check": "R2", "r1_run": a.r1_run.name, "out": out.name, "summary_diffs": {}, "scores_diffs": []}
    for name, path in (("e35_designA_A12_summary", E35_SUMMARY), ("reference_run_summary", REF / "summary.json")):
        want = json.loads(path.read_text(encoding="utf-8"))
        keys = sorted((set(want) | set(got)) - {"run"})
        res["summary_diffs"][name] = [k for k in keys if want.get(k) != got.get(k)]
    want_rows = [json.loads(l) for l in (REF / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    got_rows = [json.loads(l) for l in (out / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(want_rows) != len(got_rows):
        res["scores_diffs"].append({"row_count": [len(want_rows), len(got_rows)]})
    for w, g in zip(want_rows, got_rows):
        bad = sorted(k for k in set(w) | set(g) if w.get(k) != g.get(k))
        if bad:
            res["scores_diffs"].append({"id": w.get("id"), "fields": bad})
    res["verdict"] = "PASS" if not any(res["summary_diffs"].values()) and not res["scores_diffs"] else "FAIL"
    (out / "r2_result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
