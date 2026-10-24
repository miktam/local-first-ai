"""Figures for 'Our 4B Property Model Gained 6.7 Points at Q4 and None at Q8' (exp_040)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-17-checkable-reward-gain-vanished-at-q8.md"
SRC = "tasks/chronos/exp_040_coapi_voice_checkable/results/v3"


def figures(repo: Path) -> dict:
    d = repo / SRC
    q4, q8 = json.load(open(d / "H1q4.json")), json.load(open(d / "H1q8.json"))
    b0, r3 = json.load(open(d / "desc_B0_q4_vs_q8.json")), json.load(open(d / "desc_R3_q4_vs_q8.json"))

    def row(lbl, h, var):
        p = h.get("mcnemar_p_one_sided")
        return {"label": lbl, "est": h["diff"], "lo": h["ci95"][0], "hi": h["ci95"][1], "var": var,
                "note": f"p {p:.3f}" if p is not None and h["diff"] != 0 else ""}
    gain = V.dot_ci(
        [row("Exact match · Q4", q4["H1_exact"], "--s1"), row("Exact match · Q8", q8["H1_exact"], "--s2"),
         row("Both twins right · Q4", q4["H3_pairs_both_right"], "--s1"), row("Both twins right · Q8", q8["H3_pairs_both_right"], "--s2")],
        "Trained (R3, checkpoint 50) minus untrained Qwen3-4B, with 95 % intervals from 10,000 stratified paired resamples; "
        "p is exact one-sided McNemar. Exact match by code on the 150 sealed questions; pairs: 30 contrast pairs with both twins "
        "right. Q4 is our importance-matrix Q4_K_M, Q8 is Q8_0. One greedy pass per arm. The light line marks the registered "
        "minimum gain for the pairs. Q8 was the pre-registered robustness arm: verdict not robust.",
        xlab="change, trained minus untrained", dec=3, series=[("Q4_K_M", "--s1"), ("Q8_0", "--s2")],
        refs=[(0.10, "registered minimum, pairs")], label="Training gain at Q4 and Q8 with intervals")

    def drow(lbl, h, lt, rt):
        return (lbl, h["discordant"]["y_only"], lt, h["discordant"]["x_only"], rt)
    churn = V.diverging(
        [drow("Training, at Q4", q4["H1_exact"], "untrained", "trained"),
         drow("Training, at Q8", q8["H1_exact"], "untrained", "trained"),
         drow("Untrained: Q4 against Q8", b0["H1_exact"], "Q8", "Q4"),
         drow("Trained: Q4 against Q8", r3["H1_exact"], "Q8", "Q4")],
        "Answers right in only one of two arms, of the same 150 questions (exact match by code, greedy decoding). Changing "
        "only the quantiser flips about as many answers (17–19) as training did at Q4 (18). The two Q4-against-Q8 rows are "
        "descriptive, not registered.", label="Answers that changed: training against quantisation",
        header="answers right only in this arm")
    return {"e40-gain-ci": gain, "e40-churn": churn}
