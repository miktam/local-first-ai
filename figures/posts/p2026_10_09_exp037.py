"""Figures for 'We Stopped the Kolibri Benchmark' (exp_037)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lfa_viz as V  # noqa: E402

POST = "2026-10-09-we-stopped-the-kolibri-benchmark.md"
E = "tasks/chronos/exp_037_kolibri_forced_gate"


def figures(repo: Path) -> dict:
    plan = json.load(open(repo / E / "results/plan_fixed_20261009T081207Z.json"))
    line = next(l for l in (repo / E / "HYPOTHESIS.md").read_text().splitlines() if l.startswith("**Ladder totals**"))
    reg = {m.group(1): tuple(float(x) for x in m.group(2, 3, 4)) for m in re.finditer(r"(P\d+) (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) / (\d+(?:\.\d+)?)", line)}
    items = [(f"P{k}", *reg[f"P{k}"], rung["total_h"]) for k, rung in enumerate(plan["ladder"])]
    rungs = V.range_vs_point(items,
        "For each of the 11 registered plans (P0 the fullest, P10 the smallest): the light bar runs from the registered nominal to "
        "the adverse estimate, with a tick at the pessimistic one; the dot is the projection from the pilot. These are "
        "projections, not measurements, and not a speed result. Both are Tier A hours × 1.15; the projection uses the pilot's "
        "answer lengths (8 questions per cell, plus one standard error, × 1.25 for GPQA Diamond) with truncations counted at the "
        "raised cap. No plan fits the budget, and none splits into two sessions of 16 h or less.",
        budget=plan["B_main_h"], budget_label=f"budget {plan['B_main_h']:.0f} h", ylab="hours", label="No plan fits the budget")
    ps = json.load(open(repo / E / "results/pilot_summary_20261009T045611Z.json"))
    cells = {(c["arm"], c["task"]): c for c in ps["cells"]}
    arms = [("K8", "Kolibri K8", "--s1"), ("K4", "Kolibri K4", "--s1"), ("G8", "Gemma 4 8-bit", "--s2"), ("Q36-8", "Qwen3.6 8-bit", "--s3")]
    parts = []
    for task, title, trig in (("gpqa_main_en", "GPQA main, English", 0.4), ("gpqa_main_de", "GPQA main, German", 0.4),
                              ("mmlu_full_en", "MMLU-ProX, English", 0.5), ("mmlu_full_de", "MMLU-ProX, German", 0.5)):
        groups = []
        for a, name, var in arms:
            c = cells.get((a, task))
            if not c:
                continue
            groups.append({"label": name, "var": var,
                           "points": [(n, tr, f"{name}: {n:,} tokens{' (cut off)' if tr else ''}") for n, tr in zip(c["lengths"], c["truncated"])]})
        cap = next(c["cap"] for c in ps["cells"] if c["task"] == task)
        parts.append(f'<div class="ttl" style="margin-top:.6em">{title}</div>' + V.strip(
            groups, None, lo=0, hi=34000, cuts=[(cap, f"cap {cap:,}"), (trig * cap, f"cap-raise trigger {trig * cap:,.0f}")],
            tick_fmt=lambda t: f"{t / 1000:.0f}k"))
    tails = V.figure("".join(parts),
        "Answer length in tokens, one dot per pilot question (8 per cell, GPQA main, high effort; outputs not scored). The "
        "highlighted dots on the cap line were cut off mid-reasoning. Each model counts with its own tokenizer; Kolibri packs about 1.17× more bytes "
        "per German token. All four models triggered the GPQA cap raise, so it is not Kolibri's alone; what Kolibri added was "
        "the truncations at the raised cap and a halved batch size for K8.", label="Answer lengths in the pilot")
    return {"e37-rungs": rungs, "e37-tails": tails}
