"""Figure for 'We Trained Our 4B Property Model With a Code-Checked Reward. It Did Not Get More Correct.' (exp_039)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-15-code-checked-reward-did-not-make-it-more-correct.md"
R = "tasks/chronos/exp_039_coapi_voice_onpolicy/results"


def figures(repo: Path) -> dict:
    q8 = json.load(open(repo / R / "q8_control/verdict.json"))
    m = json.load(open(repo / R / "main/verdict.json"))["H2a"]
    q4 = q8["descriptive_vs_q4_main_run"]
    mc = q8["mean_correctness"]
    fig = V.slope(
        [("untrained", q4["B0r_q4"], mc["B0rq8"], "--s1"), ("trained", q4["R1_q4"], mc["R1q8"], "--s2")],
        f"Mean judged correctness (0–2) on the {m['n_rows']} questions whose rule was in the retrieved text; blind Fable judge, no "
        f"glue. Trained minus untrained: at Q4 {m['mean_diff']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]; at Q8 "
        f"{q8['mean_diff']:+.3f} [{q8['ci95'][0]:+.3f}, {q8['ci95'][1]:+.3f}] (within-session paired bootstrap). The untrained model "
        "rose at Q8 while the trained one stayed flat, so the Q4 gain was damage to the base, not a gain from training. 'Q4' is our "
        "plain llama.cpp Q4_K_M with no importance matrix. The Q4 and Q8 cells were judged in different sessions (measured judge "
        "drift about 0.025).", "Q4_K_M (ours, no imatrix)", "Q8_0", dec=3, top=1.6, bottom=1.0,
        series=[("untrained Qwen3-4B", "--s1"), ("trained (run 2, checkpoint 100)", "--s2")], label="Which arm moved between Q4 and Q8")
    return {"e39-q4q8": fig}
