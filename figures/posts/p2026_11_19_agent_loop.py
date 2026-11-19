"""Figures for 'Tests One AI Agent Wrote for Another Were Wrong in 7 of 13 Runs' (field note 2026-11-19, the agent loop trial)."""
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-11-19-tests-one-agent-wrote-for-another.md"
N = "field-notes/2026-11-19-agent-loop-trial"


def rows(repo: Path, name: str):
    with open(repo / N / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tiles(runs, misses) -> str:
    """One tile per run, in order; the tile says whether that run's frozen test was right."""
    kind = {m["run"]: m["type"] for m in misses}
    var = {"passed while false": "--s2", "failed on a correct result": "--s1"}
    W, side, gap, x0, y0 = 600, 36, 8, 14, 8
    svg = ""
    for i, r in enumerate(runs):
        x = x0 + i * (side + gap)
        k = kind.get(r["run"])
        fill = f"var({var[k]})" if k else "var(--same)"
        tip = f"run {r['run']} ({r['kind']}): " + (f"the test {k}" if k else "the test's verdict was not found wrong")
        svg += (f'<rect x="{x}" y="{y0}" width="{side}" height="{side}" rx="4" fill="{fill}"><title>{V.esc(tip)}</title></rect>'
                f'<text x="{x + side / 2}" y="{y0 + side + 16}" text-anchor="middle" font-size="12" fill="var(--ink2)">{r["run"]}</text>')
    H = y0 + side + 24
    n_false = sum(1 for k in kind.values() if k == "passed while false")
    n_fail = sum(1 for k in kind.values() if k == "failed on a correct result")
    leg = V.legend([(f"passed while the thing it guarded was broken ({n_false})", "--s2"),
                    (f"failed on a correct result ({n_fail})", "--s1"),
                    (f"verdict not found wrong ({len(runs) - len(kind)})", "--same")])
    label = (f"The test written before the work gave a wrong verdict in {len(kind)} of {len(runs)} runs: "
             + "; ".join(f"run {m['run']} {m['type']}" for m in misses))
    cap = (f"One square per run, in order. In {len(kind)} of {len(runs)} runs the test that was written and frozen before the work "
           f"turned out to be wrong: {n_false} times it passed while the thing it guarded was broken, {n_fail} times it failed on a correct result. "
           "A grey square means the test's pass or fail was not found wrong, not that the test was proved right: in runs 01 and 11 the reviewer still found cases that could not fail.")
    return V.figure(leg + f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{V.esc(label)}">{svg}</svg>', cap, label)


def figures(repo: Path) -> dict:
    runs, defects, misses = rows(repo, "runs.tsv"), rows(repo, "defects.tsv"), rows(repo, "check_misses.tsv")
    c = Counter(d["caught_by"] for d in defects)
    who = [("the reviewer alone", c["reviewer"], ""), ("worker and reviewer", c["both"], ""), ("the worker alone", c["worker"], ""),
           ("the task-writer's extra check", c["dispatcher"], ""), ("the test written first", c["frozen check"], "run by the worker"),
           ("the human owner", 0, "")]
    caught = V.hbars(who, f"Who caught each of the {len(defects)} recorded defects in {len(runs)} runs. A row can group several findings of one kind "
                     "from one run, so these are rows of a table, not a count of every flaw. The reviewer was not given the worker's report.",
                     label=f"Who caught the {len(defects)} recorded defects")
    night = rows(repo, "night_run.tsv")
    lost = [r for r in night if r["measure"].startswith("everyday searches")]
    before, after = int(lost[0]["value"]), int(lost[1]["value"])
    of = int(lost[0]["of"])
    fix = V.hbars([("after the fix", before, f"of {of}"), ("after one more round", after, f"of {of}")],
                  f"Everyday searches with one extra detail, out of {of}, that no longer got the list of homes the old code gave. The frozen tests for this part read "
                  "253 of 254 both times. The second count is the task-writer's re-run of the reviewer's comparison.", top=of, label="Searches that lost their answer behind passing tests", var="--s2")
    return {"loop-tiles": tiles(runs, misses), "loop-caught": caught, "loop-fix": fix}
