"""Figure for 'The Cost-Capability Curve Has One Step' (exp_012, exp_014), parsed from the published scientific log."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-06-09-cost-capability-curve.md"
MODELS = [("gemma4:26b", "gemma4:26b", "$0"), ("Claude Haiku 4.5", "Haiku 4.5", "$0.095"), ("Claude Sonnet 4.6", "Sonnet 4.6", "$0.291"),
          ("Claude Opus 4.8", "Opus 4.8", "$0.611")]


def _table_after(text, heading):
    i = text.index(heading)
    rows = []
    for line in text[i:].splitlines()[1:]:
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Model") and not line.startswith("| Rep"):
            rows.append([c.strip() for c in line.strip("|").split("|")])
        elif rows and not line.startswith("|"):
            break
    return rows


def _net(cell):                      # "3−1†" → 2, "2" → 2
    cell = cell.replace("†", "").replace("−", "-").replace("*", "")
    return sum(int(x) for x in re.findall(r"-?\d+", cell)) if "-" in cell[1:] else int(re.findall(r"-?\d+", cell)[0])


def figures(repo: Path) -> dict:
    log = (repo / "tasks/chronos/scientific_log.md").read_text()
    ta = {r[0]: [_net(c) for c in r[1:4]] for r in _table_after(log, "**Task A scores (compliance extraction")}
    tb = {r[0]: [_net(c) for c in r[1:4]] for r in _table_after(log, "**Task B scores (impl gap detection")}
    comp = {r[0]: int(re.search(r"\*\*(-?\d+)/8\*\*", "|".join(r)).group(1)) for r in _table_after(log, "**Composite scores (Task A + Task B")}
    net = lambda r: int(re.search(r"\*\*(-?\d+)/8\*\*", r[-1].replace("−", "-")).group(1))
    e14g = [net(r) for r in _table_after(log, "#### Gemma4:26b — all 5 reps")]
    e14h = [net(r) for r in _table_after(log, "#### Claude Haiku 4.5 — all 5 reps")]
    cats = []
    for full, short, cost in MODELS:
        groups = [([a + b for a, b in zip(ta[full], tb[full])], "--s1", True)]
        if short == "gemma4:26b":
            groups.append((e14g, "--s2", False))
        if short == "Haiku 4.5":
            groups.append((e14h, "--s2", False))
        cats.append({"label": short, "sub": f"exp_012 cost {cost}", "groups": groups, "tick": comp[full]})
    fig = V.cat_dots(cats,
        "Net rubric score per repetition (Task A + Task B, max 8; −1 per confirmed false positive), from the published "
        "scientific log. Filled dots: exp_012's three repetitions per model; hollow dots: exp_014's five repetitions (strict "
        "scoring), run the same day for gemma4:26b and Haiku only. The tick is the score the post reported, which counts an item "
        "as found when it appears in at least 2 of 3 repetitions. gemma4:26b ran with thinking off. Costs are exp_012's API "
        "totals for all six calls per model, not per audit. The log's per-run Task B scores for Haiku and Sonnet do not add up "
        "to its item tables (log note, 24 October 2026); the dots follow the per-run tables.",
        ylab="net score of 8", ymin=-2, ymax=8,
        marks_legend=[("filled", "--s1", "exp_012 repetition"), ("hollow", "--s2", "exp_014 repetition"), ("tick", "--ink", "score in the post")],
        label="Scores per repetition, local against cloud")
    return {"cost-per-rep": fig}
