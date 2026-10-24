"""Figure for 'Kolibri Scored Below Gemma 4 in Our Property Assistant' (exp_038)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-10-kolibri-scored-below-gemma-4.md"


def figures(repo: Path) -> dict:
    a = json.load(open(repo / "tasks/chronos/exp_038_kolibri_casasol_dropin/results/analysis.json"))
    d = a["descriptive"]
    L = d["D1_by_language"]

    def r(lbl, x, var="--s1", note=""):
        return {"label": lbl, "est": x["point"], "lo": x["ci95"][0], "hi": x["ci95"][1], "var": var, "note": note}
    rows = [r("K4 − Gemma, all 60 (H1)", a["H1"]), r("K4 − Gemma, English (31)", L["en"]), r("K4 − Gemma, Polish (17)", L["pl"]),
            r("K4 − Gemma, Spanish (12)", L["es"]), r("K4 − Gemma, German slice (33)", a["H3"]),
            r("K8 − Gemma, all 60", d["D8_K8"]["K8_minus_G26"], "--s2"), r("K4-MED − Gemma, all 60", d["D9_K4MED"]["MED_minus_G26"], "--s3")]
    h = a["headroom"]
    fig = V.dot_ci(rows,
        "Paired differences in judged correctness (0–2 scale) against gemma4:26b, with 95 % intervals (paired bootstrap stratified "
        "by language, 10,000 replicates for H1). The registered margin was +0.20; no interval reaches it. The set itself tops out: "
        f"the best score it allows is {h['C']:.3f} and Gemma already sits at {h['G26_mean']:.3f}, {h['C'] - h['G26_mean']:.3f} below it. "
        "The German slice is a separate LLM-written set fed English context: not evidence against a German edge below about +0.4. "
        "K4 = MLX affine 4-bit (M5 Max); Gemma = Q4_K_M on Ollama 0.40.2 (Mac mini). One blind Fable judge family.",
        xlab="change in correctness, Kolibri minus Gemma", dec=3, refs=[(0.20, "registered margin +0.20")],
        label="Kolibri against Gemma, by slice")
    return {"e38-diff": fig}
