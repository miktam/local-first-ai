"""Figure for 'We Trained It Three Times. Then We Stopped.' (exp_035)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-09-22-we-trained-it-three-times-then-stopped.md"
R = "tasks/chronos/exp_035_coapi_voice/results"
BUILDS = [("4B untrained", "b2m_Qwen3-4B_judge_20260922.json", "--s1"), ("v2, all projections", "b3_v2_judge_20260922.json", "--s2"),
          ("v3, q/v only", "b3_v3_judge_20260922.json", "--ink2"), ("glue, untrained 4B", "designA_A12_judge_20260922.json", "--s3")]


def figures(repo: Path) -> dict:
    s = [json.load(open(repo / R / f))["summary"] for _, f, _ in BUILDS]
    lang = lambda x, l: x["language_native_counts_by_question_language"][l]
    panels = [
        {"title": "Correctness with the rule in context", "note": "0–2 scale, 38 rows, higher is better",
         "values": [x["correctness_mean_when_substance_in_context"] for x in s], "top": 2, "dec": 2},
        {"title": "Native Spanish", "note": f"of {lang(s[0], 'es')['n']} Spanish questions, higher is better",
         "values": [lang(x, "es")["pass"] for x in s], "top": 12, "dec": 0},
        {"title": "Native Polish", "note": f"of {lang(s[0], 'pl')['n']} Polish questions, higher is better",
         "values": [lang(x, "pl")["pass"] for x in s], "top": 17, "dec": 0},
        {"title": "Hand-off to an agent present", "note": "share of 43 rows scored, higher is better",
         "values": [100 * x["handoff_pass_rate"] for x in s], "top": 100, "dec": 0, "unit": "%"},
        {"title": "Answers that collapse after a correct opener", "note": "of 60, fewer is better",
         "values": [x["rows_with_tail_degradation"] for x in s], "top": 30, "dec": 0},
    ]
    fig = V.bar_pairs(panels, [(n, v) for n, _, v in BUILDS],
        "Sealed v1 set, 60 questions (31 English, 17 Polish, 12 Spanish); one blind Fable judge with a frozen rubric; no intervals "
        "are published. All builds are Qwen3-4B at Q4_K_M: v2 is a LoRA on all seven projections, v3 on q/v only. The glue is the "
        "untrained 4B behind code: its 100 % hand-off is a template appended by code, and its native-language gain comes from "
        "rendering the prompt in the question's language. The glue was tuned using the judge's notes on these 60 rows, so for the "
        "glue this set became a development set.", label="Both fine-tunes lost substance; the glue did not")
    return {"e35-builds": fig}
