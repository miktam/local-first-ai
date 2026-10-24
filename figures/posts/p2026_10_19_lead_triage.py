"""Figure for 'Sorting Estate-Agency Enquiries on a Mac Mini' (exp_043): the business-pitch margin."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-19-lead-triage.md"
E = "tasks/chronos/exp_043_kev_lead_triage"


def figures(repo: Path) -> dict:
    lab = {json.loads(l)["id"]: json.loads(l)["labels"]["intent"] for l in open(repo / E / "data/test.jsonl")}
    pred = [json.loads(l) for l in open(repo / E / "results/test/pred.jsonl")]
    gen = [(p["b2b"], p["b2b"] >= 0.6, f"{p['id']}: pitch score {p['b2b']:.2f}") for p in pred if lab[p["id"]] != "spam"]
    spam = [(p["b2b"], False, f"{p['id']}: pitch score {p['b2b']:.2f}") for p in pred if lab[p["id"]] == "spam"]
    fig = V.strip([{"label": f"genuine ({len(gen)})", "var": "--s1", "points": gen}, {"label": f"spam ({len(spam)})", "var": "--s2", "points": spam}],
        f"Kev-4B's business-pitch score for every test enquiry (exp_043, 235 synthetic messages, one pass); raw probabilities, not "
        f"calibrated. At or above the cut (0.6) a message is treated as a pitch. In the genuine row, the {sum(1 for g in gen if g[1])} "
        "highlighted dots on and past the line are genuine enquiries flagged as spam (0.60–0.63), with more just below it. That is why spam goes to a "
        "review folder, not the bin. The cut was set on 120 tuning messages.",
        cut=0.6, cut_label="pitch cut 0.6", label="How thin the business-pitch margin is")
    return {"triage-pitch-cut": fig}
