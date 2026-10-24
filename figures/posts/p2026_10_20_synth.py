"""Figure for 'Two AIs Disagreed on 1 in 5 Test Enquiries. We Deleted Those.' (exp_044 hard groups)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-20-synthetic-data-lessons.md"
E = "tasks/chronos/exp_044_kev_lead_triage_rev4"


def budget_agree(a, b):          # as leads44.budget_agree (exp_044 kit): within 1 %
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 0.01 * max(a, b)


def figures(repo: Path) -> dict:
    s = json.load(open(repo / E / "results/test/score_rev31.json"))["by_stratum"]
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(repo / E / "data/test.jsonl")}
    pred = {json.loads(l)["id"]: json.loads(l) for l in open(repo / E / "results/test/pred_rev31.jsonl")}

    def row(lbl, k, n, var):
        lo, hi = V.wilson(k, n)
        return {"label": f"{lbl} ({k}/{n})", "est": k / n, "lo": lo, "hi": hi, "var": var}
    req = [row("ordinary", round(s["standard"]["intent"]["accuracy"] * s["standard"]["intent"]["n"]), s["standard"]["intent"]["n"], "--s1"),
           row("holiday wording", round(s["hard_holiday"]["intent"]["accuracy"] * s["hard_holiday"]["intent"]["n"]), s["hard_holiday"]["intent"]["n"], "--s2"),
           row("business look-alikes", round(s["hard_b2b"]["intent"]["accuracy"] * s["hard_b2b"]["intent"]["n"]), s["hard_b2b"]["intent"]["n"], "--s2")]
    std = [i for i, r in rows.items() if r.get("stratum") == "standard" and r["labels"].get("budget_eur") not in (None, "disagree")]
    k_std = sum(budget_agree(pred[i]["facts"]["budget_eur"], rows[i]["labels"]["budget_eur"]) for i in std if i in pred)
    hb = s["hard_budget"]["budget_eur"]
    bud = [row("ordinary, budget stated", k_std, len(std), "--s1"), row("budgets in words", round(hb["accuracy"] * hb["n"]), hb["n"], "--s2")]
    body = ('<div class="ttl">Request read right</div>' + V.dot_ci(req, None, dec=2, xmin=0, xmax=1, signed=False)
            + '<div class="ttl" style="margin-top:.6em">Budget read within 1 %, where a budget is stated</div>' + V.dot_ci(bud, None, dec=2, xmin=0, xmax=1, signed=False))
    fig = V.figure(V.legend([("ordinary messages", "--s1"), ("groups written to be hard", "--s2")]) + body,
        "exp_044's test set (331 synthetic enquiries), the old version (rev 3.1), one pass; 95 % Wilson intervals, computed here "
        "from the published predictions. With 29–31 messages per hard group, a gap of up to about 0.2 cannot be ruled out: the "
        "groups were not dramatically harder, not shown to be equally easy. Only budgets in words collapsed, a gap already known "
        "in the code.", label="The groups written to be hard, against ordinary messages")
    return {"synth-hard-groups": fig}
