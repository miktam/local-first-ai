"""Figure for 'A Local Decision Model Made Our 4B Refuse 92 % of Unanswerable Questions' (exp_041, exp_042)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-18-answerability-gate.md"
NAMES = {"G1": "Qwen3-4B yes/no", "G2": "Kev-4B", "G3": "Clef-Flash 9B", "G4": "Clef 27B"}


def figures(repo: Path) -> dict:
    a = json.load(open(repo / "tasks/chronos/exp_041_answerability_gate/results/v3/score.json"))
    b = json.load(open(repo / "tasks/chronos/exp_042_answerability_gate_retrieval/results/v3/score.json"))
    g41 = a["gates"]
    base = g41["G2"]["over_B0i"]
    pts41 = [{"label": "no gate", "x": 100 * base["false_abstention"]["y"], "y": 100 * base["abstention"]["y"], "var": "--base", "dx": 9, "dy": 4}]
    nudge = {"G1": (9, 4), "G2": (-9, 15), "G3": (9, 4), "G4": (-9, 4)}
    for g in ("G1", "G2", "G3", "G4"):
        o = g41[g]["over_B0i"]
        dx, dy = nudge[g]
        pts41.append({"label": NAMES[g], "x": 100 * o["false_abstention"]["x"], "y": 100 * o["abstention"]["x"],
                      "var": "--s1", "dx": dx, "dy": dy, "anchor": "end" if dx < 0 else "start"})
    s0 = b["S0"]
    g4, g2 = b["gates"]["G4"], b["gates"]["G2"]
    pts42 = [{"label": "no gate", "x": 100 * s0["false_abstention"], "y": 100 * s0["abstain"], "var": "--base", "dx": 9, "dy": 4},
             {"label": "Clef 27B", "x": 100 * g4["false_abstention"]["count"] / g4["false_abstention"]["of"], "y": 100 * g4["abstain"]["gated"],
              "var": "--s1", "dx": 9, "dy": -9},
             {"label": "Kev-4B", "x": 100 * g2["false_abstention"]["count"] / g2["false_abstention"]["of"], "y": 100 * g2["abstain"]["gated"],
              "var": "--s1", "dx": 9, "dy": 15}]
    fig = V.scatter_panels(
        [{"title": "exp_041: first test", "note": "answerable = source in the guide (112)", "points": pts41},
         {"title": "exp_042: retrieval fixed", "note": "answerable = source in the retrieved text (101)", "points": pts42}],
        "Each dot is one gate in front of the untrained Qwen3-4B voice (imatrix Q4), on the same 150 sealed questions. Up is "
        "better (correct refusals of the 38 unanswerable questions); left is better (answerable questions wrongly refused). Each "
        "gate's threshold was set on its own dev set to block at most 3 % of answerable rows; the 5 % limit was pre-registered. "
        "The two panels use different denominators: in exp_042, 'answerable' means the source is in the retrieved text, and the 11 "
        "rows whose source retrieval lost are reported apart (Clef refused 8 of them; under exp_041's label that alone would be "
        "at least 7 %). The pass rests on the registered relabel as well as on the retrieval fix.",
        "wrong refusals of answerable questions", "correct refusals of unanswerable questions (of 38)",
        xmax=14, ymin=40, ymax=100, limit=(5, "over the 5 % limit"), xdec=0, ydec=0,
        label="Answerability gates: correct against wrong refusals")
    return {"gate-operating-points": fig}
