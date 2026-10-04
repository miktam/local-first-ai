"""runner/plan_fix.py and runner/plan_rules.json (HYPOTHESIS "Pilot and the
rule that fixes n and max_tokens"; BUILD_SPEC §5.4, §7.2).

Synthetic pilots are built from the HYPOTHESIS "Planning assumptions" table
(mean completion tokens, aggregate decode tok/s at the chosen B, prefill
tok/s, and the prompt tokens of the budget arithmetic): every pilot item has
the planning mean length (so the (mean + 1 SE)/mean scale is 1), GPQA main
lengths are the Diamond planning lengths / 1.25 (the rule multiplies them
back), and the decode steps follow step_seconds = n_live / rate (a = c = 0),
i.e. a constant aggregate rate as in the planning arithmetic.

What the frozen rule gives with those pilots:
- nominal: P0 + Tier B {B1, B2, B5}, as in the HYPOTHESIS scenario table;
- pessimistic: P8 (as in the table) + Tier B {B5}. The table says "none":
  it priced B5 with fixed effort-none lengths and 600 MMLU items (2.4 h),
  while rule 4 derives K8 effort-none lengths as 0.08 x the effort-high
  lengths and B5's MMLU cells run at the plan's n_M (196), which gives 1.3 h
  and fits B_main = 30.9 h. The rule is what is pre-registered; the table is
  its planning illustration (deviation reported to the integrator);
- adverse: P10, no Tier B; slightly slower than adverse: STOP.
"""

from __future__ import annotations

import json
import re

import pytest

from runner import plan_fix
from runner.common import EXP_DIR, load_rules

GiB = 1 << 30
RULES = load_rules()

TOK = {
    "K": {"gpqa": (5000, 8000, 10000), "mmlu": (2000, 3000, 3500), "ifbench": (1500, 2500, 3000),
          "rgb_cb": (600, 1000, 1200), "rgb_forced": (400, 800, 1000), "aime": (14000, 20000, 24000)},
    "G": {"gpqa": (4000, 6000, 7000), "mmlu": (1500, 2500, 2800), "ifbench": (1200, 2000, 2200),
          "rgb_cb": (600, 1000, 1000), "rgb_forced": (400, 800, 900)},
    "Q36": {"gpqa": (7000, 9000, 10000), "mmlu": (3000, 3500, 4000), "ifbench": (2000, 3000, 3200),
            "rgb_cb": (900, 1200, 1300), "rgb_forced": (600, 1000, 1100)},
}
PROMPT = {"gpqa": 300, "mmlu": 400, "ifbench": 150, "rgb_cb": 50, "rgb_forced": 60, "aime": 200}
DECODE = {"K8": (180, 120, 100), "K4": (240, 160, 140), "G8": (220, 150, 140), "Q36-8": (250, 160, 150)}
PREFILL = {"K8": (1500, 800, 700), "K4": (1700, 900, 800), "G8": (1500, 800, 700), "Q36-8": (1800, 900, 800)}
FAM = {"K8": "K", "K4": "K", "G8": "G", "Q36-8": "Q36"}
KIND = {"gpqa_main_en": "gpqa", "gpqa_main_de": "gpqa", "mmlu_full_en": "mmlu", "mmlu_full_de": "mmlu",
        "ifbench_pilot": "ifbench", "rgb_int_cb": "rgb_cb", "rgb_int_forced": "rgb_forced", "aime_en_pilot": "aime"}
S1_HOURS = (5.4, 9.1, 9.6)
WEIGHTS = {"K8": 83_100_000_000, "K4": 44_000_000_000, "G8": 28_000_000_000, "Q36-8": 37_700_000_000,
           "Q38-8": 29_500_000_000}
NOW = "2026-10-04T18:00:00.000Z"


def synthetic(s: int, rate_scale: float = 1.0, edit=None):
    """(pilot, steps, s1, ctx) for scenario s (0 nominal, 1 pessimistic, 2 adverse)."""
    cells, steps, arms = [], {}, {}
    t0 = 1_800_000_000.0
    for arm in RULES["pilot"]["arms"]:
        rate = DECODE[arm][s] * rate_scale
        steps[arm] = [{"type": "step", "step": i, "t": t0 + 10 * i, "n_live": 1 + i % 8, "padded_len": 200 + 37 * i,
                       "step_seconds": (1 + i % 8) / rate, "prompt_tokens": 0} for i in range(90)]
        arms[arm] = {"t_start": t0, "t_end": t0 + 890, "prefill_tokens": 30000.0,
                     "prefill_seconds": 30000.0 / (PREFILL[arm][s] * rate_scale)}
        for c in RULES["pilot"]["cells"][arm]:
            kind = KIND[c["task"]]
            L = TOK[FAM[arm]][kind][s] / (1.25 if kind == "gpqa" else 1.0)
            n = c["n"]
            cells.append({"arm": arm, "task": c["task"], "effort": "high", "n": n, "lengths": [L] * n,
                          "prompt_tokens": [PROMPT[kind]] * n, "truncated": [False] * n, "n_truncated": 0,
                          "n_parse_fail": 0, "diagnoses": {}, "status_counts": {"closed": n}, "n_defect": 0})
    pilot = {"cells": cells, "arms": arms, "caps": RULES["caps"]}
    if edit:
        edit(pilot)
    h = S1_HOURS[s]
    s1 = [{"phase": "preflight", "t_start": "2026-10-04T08:00:00Z", "t_end": "2026-10-04T08:30:00Z"},
          {"phase": "rest", "t_start": "2026-10-04T09:00:00Z",
           "t_end": f"2026-10-04T{9 + int(h - 0.5):02d}:{int(round(((h - 0.5) % 1) * 60)):02d}:00Z"}]
    ctx = {"now_utc": NOW, "L_bytes": 115_448_725_504, "weight_bytes": dict(WEIGHTS),
           "scorers_tree_sha256": "a" * 64, "manifests": {"mmlu_prox_lite_en": "b" * 64}}
    return pilot, steps, s1, ctx


def run_fix(s, rate_scale=1.0, edit=None, existing=()):
    pilot, steps, s1, ctx = synthetic(s, rate_scale, edit)
    return plan_fix.fix(pilot, steps, s1, RULES, list(existing), context=ctx)


# ------------------------------------------------------------ scenarios


def test_nominal_gives_P0_with_B1_B2_B5():
    plan, md, k = run_fix(0)
    assert plan["status"] == "FIXED" and plan["plan"] == "P0" and plan["n_M"] == 588
    assert plan["tier_b"] == ["B1", "B2", "B5"]
    assert plan["S1_hours"] == pytest.approx(5.4)
    assert plan["B_main_h"] == 31.0
    assert plan["S2_h"] <= 16 and plan["S3_h"] <= 16 and plan["main_h"] <= 31.0
    assert plan["S2_h"] == pytest.approx(14.0, abs=0.05)  # the HYPOTHESIS split: K8 Tier A + G8/Q36-8 MMLU EN
    why = {t["item"]: t["reason"] for t in plan["tier_b_considered"]}
    assert why["B3"].startswith("total") and why["B4"].startswith("split fails")
    assert plan["rows"]["H2"] == ["gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench"]
    assert plan["rows"]["H2_secondary_aime"] == ["aime_en", "aime_de"]
    assert plan["rows"]["H7"] == ["mmlu_en", "mmlu_de", "ifbench", "rgb_cb"]
    assert plan["forced_cells"] and plan["neg_fc_cells"] and plan["k8_runs"]
    first = plan["queue"][0]
    assert first["cell"] == "K8:gpqa_en:high" and first["B"] == 8 and first["cap"] == 32768
    assert {q["cell"]: q["B"] for q in plan["queue"]}["K8:ifbench:high"] == 16
    assert all(q["session"] == "S2" for q in plan["queue"][: plan["split_index"]])
    assert "## Amendment 1 — plan (2026-10-04T18:00:00.000Z)" in md
    assert "Plan: **P0**" in md and "Tier B: B1, B2, B5" in md


def test_pessimistic_gives_P8_and_only_the_rule_derived_B5():
    plan, _, _ = run_fix(1)
    assert plan["plan"] == "P8" and plan["n_M"] == 196 and not plan["forced_cells"]
    assert plan["tier_b"] == ["B5"]  # see the module docstring: the table's "none" priced B5 differently
    why = {t["item"]: t for t in plan["tier_b_considered"]}
    assert why["B4"]["reason"].startswith("total")  # 1.4 h x 1.15 does not fit 30.9 h
    assert why["B5"]["projected_h"] < 1.5


def test_adverse_gives_P10_and_no_tier_b():
    plan, md, _ = run_fix(2)
    assert plan["plan"] == "P10" and plan["n_M"] == 154 and plan["tier_b"] == []
    assert plan["rows"]["H2"] == ["gpqa_en", "mmlu_en", "mmlu_de", "ifbench"]  # 4 rows under P10
    assert not plan["k8_gpqa_de"]
    assert all(q["cell"] != "K8:gpqa_de:high" for q in plan["queue"])
    assert plan["main_h"] == pytest.approx(30.3, abs=0.05)


def test_slower_than_adverse_gives_STOP():
    plan, md, _ = run_fix(2, rate_scale=0.9)
    assert plan["status"] == "STOP" and plan["plan"] is None and plan["queue"] == []
    assert "Status: STOP" in md and "4th session" in md


def test_a_session_over_16h_steps_down_the_ladder():
    # P0 fits the 31 h budget but its queue cannot be split into two <= 16 h
    # sessions (10 + 7 > 16 leaves 20 h for S3); P1 can.
    hours = {"P0": [10.0, 7.0, 10.0, 3.0], "P1": [10.0, 6.0, 10.0, 3.0], "P2": [1.0]}
    ch = plan_fix.choose_plan(hours, ["P0", "P1", "P2"], 31.0, 16.0)
    assert ch["plan"] == "P1"
    assert ch["tried"][0]["fits_budget"] and not ch["tried"][0]["split_ok"]
    assert plan_fix.split_two([10.0, 7.0, 10.0, 3.0], 16.0) == (False, 1, 10.0, 20.0)
    assert plan_fix.split_two([10.0, 6.0, 10.0, 3.0], 16.0) == (True, 2, 16.0, 13.0)
    assert plan_fix.choose_plan({"P0": [17.0]}, ["P0"], 31.0, 16.0)["plan"] is None  # one cell > 16 h


def test_one_cell_over_16h_in_a_real_projection_steps_down():
    # K8 GPQA-D DE alone > 16 h: every plan with it fails the split; P10 drops it.
    def slow_gpqa_de(p):
        for c in p["cells"]:
            if c["arm"] == "K8" and c["task"] == "gpqa_main_de":
                c["lengths"] = [52_000 / 1.25] * c["n"]
                c["truncated"] = [False] * c["n"]

    plan, _, _ = run_fix(0, edit=slow_gpqa_de)
    tried = {t["plan"]: t for t in plan["ladder"]}
    assert not tried["P0"]["split_ok"]
    assert plan["plan"] in ("P10", None)


# ---------------------------------------------------------- cap raise


def _edit_cell(arm, task, **kw):
    def f(p):
        for c in p["cells"]:
            if c["arm"] == arm and c["task"] == task:
                c.update(kw)
    return f


def test_one_truncation_raises_the_task_cap_to_the_ceiling_for_all_models():
    tr = [True] + [False] * 7
    plan, _, _ = run_fix(0, edit=_edit_cell("G8", "mmlu_full_de", truncated=tr, n_truncated=1))
    assert plan["caps"]["mmlu"] == 65536 and plan["cap_raised"]["mmlu"]
    assert plan["caps"]["gpqa"] == 32768 and not plan["cap_raised"]["gpqa"]
    caps = {q["cell"]: q["cap"] for q in plan["queue"]}
    assert caps["K8:mmlu_en:high"] == caps["Q36-8:mmlu_de:high"] == 65536
    assert {q["cell"]: q["B"] for q in plan["queue"]}["K8:mmlu_en:high"] == 4  # memory rule re-evaluated


def test_one_long_completion_raises_the_cap():
    long = [2000.0] * 7 + [0.51 * 16384]
    plan, _, _ = run_fix(0, edit=_edit_cell("K4", "ifbench_pilot", lengths=long))
    assert plan["caps"]["ifbench"] == 32768
    gpqa = [4000.0] * 7 + [0.41 * 32768]  # GPQA triggers at 0.4 x the cap
    plan, _, _ = run_fix(0, edit=_edit_cell("Q36-8", "gpqa_main_en", lengths=gpqa))
    assert plan["caps"]["gpqa"] == 65536
    ok = [4000.0] * 7 + [0.39 * 32768]
    plan, _, _ = run_fix(0, edit=_edit_cell("Q36-8", "gpqa_main_en", lengths=ok))
    assert plan["caps"]["gpqa"] == 32768


def test_truncated_pilot_items_count_at_the_post_raise_cap():
    tr = [True] + [False] * 7
    base, _, _ = run_fix(0)
    plan, _, _ = run_fix(0, edit=_edit_cell("K8", "mmlu_full_en", truncated=tr, n_truncated=1))
    h = {q["cell"]: q["projected_h"] for q in plan["queue"]}
    h0 = {q["cell"]: q["projected_h"] for q in base["queue"]}
    assert h["K8:mmlu_en:high"] > 3 * h0["K8:mmlu_en:high"]


# ------------------------------------------------- parse / reasoning gate


def test_one_parse_failure_does_not_block():
    plan, _, _ = run_fix(0, edit=_edit_cell("K8", "mmlu_full_de", n_parse_fail=1,
                                            diagnoses={"extractor_mismatch": 1}))
    assert plan["status"] == "FIXED" and not plan["pilot_gate"]["blocked"]


def test_two_parse_failures_with_an_extractor_mismatch_block():
    plan, md, _ = run_fix(0, edit=_edit_cell("K8", "mmlu_full_de", n_parse_fail=2,
                                             diagnoses={"extractor_mismatch": 1, "model": 1}))
    assert plan["status"] == "BLOCKED" and plan["pilot_gate"]["blocked"] == ["K8:mmlu_full_de:high"]
    assert plan["queue"] == [] and "BLOCKED cells: K8:mmlu_full_de:high" in md


def test_two_parse_failures_without_a_diagnosed_mismatch_do_not_block():
    plan, _, _ = run_fix(0, edit=_edit_cell("G8", "gpqa_main_en", n_parse_fail=2, diagnoses={"model": 2}))
    assert plan["status"] == "FIXED"
    assert plan["pilot_gate"]["cells"]["G8:gpqa_main_en:high"]["n_parse_fail"] == 2


def test_ifbench_and_aime_pilots_never_block():
    for arm, task in (("K8", "ifbench_pilot"), ("K8", "aime_en_pilot")):
        plan, _, _ = run_fix(0, edit=_edit_cell(arm, task, n_parse_fail=4, diagnoses={"extractor_mismatch": 4}))
        assert plan["status"] == "FIXED", task


def test_summary_counts_unclosed_at_cap_as_truncation_and_gemma_none_as_fine():
    from runner.run import summarise_pilot_cell

    def rec(arm, L, finish, status, text, answer, max_tokens=16384):
        return {"key": {"arm": arm, "task": "rgb_int_cb", "effort": "high", "item": "i", "pass": 0},
                "completion_tokens": L, "finish_reason": finish, "truncated": finish == "length",
                "reasoning_status": status, "prompt_prefill": "none", "rendered_prompt_tokens": 50,
                "answer_text": answer, "text": text, "max_tokens": max_tokens}

    recs = [
        rec("K8", 16384, "length", "unclosed", "<think>long", ""),        # at the cap, unclosed: truncation
        rec("K8", 16384, "stop", "unclosed", "<think>long<|im_end|>", ""),  # EOS exactly at the cap: truncation too
        rec("K8", 900, "stop", "unclosed", "<think>gave up<|im_end|>", ""),  # unclosed, ended by EOS: parse failure
        rec("K8", 300, "stop", "closed", "<think>r</think>Paris.<|im_end|>", "Paris."),
    ]
    s = summarise_pilot_cell(recs, "K8", "rgb_int_cb", "high", 16384, 16)
    assert s["n_truncated"] == 2 and s["truncated"] == [True, True, False, False]
    assert s["n_parse_fail"] == 1 and s["diagnoses"] == {"model": 1}
    assert s["status_counts"] == {"unclosed": 3, "closed": 1}
    gem = rec("G8", 200, "stop", "none", "Paris.<turn|>", "Paris.")
    g = summarise_pilot_cell([gem] * 3, "G8", "rgb_int_cb", "high", 16384, 16)
    assert g["n_parse_fail"] == 0 and g["n_defect"] == 0  # Gemma 4 not opening its thought channel is legitimate
    k = summarise_pilot_cell([rec("K8", 200, "stop", "none", "Paris.<|im_end|>", "Paris.")], "K8", "rgb_int_cb",
                             "high", 16384, 16)
    assert k["n_defect"] == 1 and k["n_parse_fail"] == 0  # recorded as a defect, not a reasoning failure


def test_diagnose():
    d = plan_fix.diagnose
    assert d("gpqa_main_en", "So the answer is (C).", "", "closed", "stop") == "extractor_mismatch"
    assert d("mmlu_full_de", "Die richtige Antwort ist J.", "", "closed", "stop") == "extractor_mismatch"
    assert d("gpqa_main_en", "Therefore \\boxed{B}", "", "closed", "stop") == "extractor_mismatch"
    assert d("gpqa_main_en", "I am not sure about any of these.", "", "closed", "stop") is None
    assert d("gpqa_main_en", "The answer is (H).", "", "closed", "stop") is None  # out of A-D
    assert d("rgb_int_cb", "", "thinking ... </thinking> Paris", "unclosed", "stop") == "delimiter_mismatch"
    assert d("rgb_int_cb", "", "thinking without end", "unclosed", "stop") is None
    assert d("rgb_int_cb", "", "<think>opened, never closed", "unclosed", "stop") is None  # own opener is normal
    assert d("aime_en_pilot", "so the result is \\boxed{ 70 }", "", "closed", "stop") == "extractor_mismatch"


# ---------------------------------------------- amendments, determinism


def test_amendment_numbering_after_a_gate_fix_amendment():
    existing = [{"k": 0, "type": "sign-off choices"}, {"k": 1, "type": "gate fix"}]
    plan, md, k = run_fix(0, existing=existing)
    assert k == 2 and plan["amendment_k"] == 2 and plan["supersedes"] is None
    assert md.startswith("## Amendment 2 — plan (")
    assert "Supersedes: none" in md


def test_a_re_pilot_writes_a_new_amendment_that_supersedes():
    existing = [{"k": 1, "type": "gate fix"}, {"k": 2, "type": "plan"}, {"k": 3, "type": "extractor fix"}]
    plan, md, k = run_fix(0, existing=existing)
    assert k == 4 and plan["supersedes"] == 2
    assert "Supersedes: Amendment 2" in md


def test_same_input_byte_identical_output():
    a_plan, a_md, _ = run_fix(0)
    b_plan, b_md, _ = run_fix(0)
    assert plan_fix.plan_bytes(a_plan) == plan_fix.plan_bytes(b_plan)
    assert a_md == b_md


def test_amendment_names_the_plan_file_and_its_sha256_for_the_guard():
    from runner.guard import PLAN_FILE_RE, SCORERS_RE
    from runner.common import sha256_bytes

    plan, md, _ = run_fix(0)
    m = PLAN_FILE_RE.search(md)
    assert m and m.group(1) == "results/plan_fixed_20261004T180000Z.json"
    assert m.group(2) == sha256_bytes(plan_fix.plan_bytes(plan))
    assert SCORERS_RE.search(md).group(1) == "a" * 64


def test_write_outputs_never_overwrites(tmp_path):
    plan, md, k = run_fix(0)
    pf, am = plan_fix.write_outputs(tmp_path, plan, md, k, NOW)
    assert pf.read_bytes() == plan_fix.plan_bytes(plan) and am.read_text() == md
    with pytest.raises(FileExistsError):
        plan_fix.write_outputs(tmp_path, plan, md, k, NOW)


def test_k8_that_does_not_fit_is_replaced_by_k4():
    pilot, steps, s1, ctx = synthetic(0)
    ctx["L_bytes"] = 90 * GiB
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["k8_runs"] is False
    assert all(q["arm"] != "K8" for q in plan["queue"])
    assert plan["rows"]["H2_arm"] == "K4" and plan["rows"]["H7"] == []


def test_s1_hours_sum_the_records():
    recs = [{"t_start": "2026-10-04T08:00:00Z", "t_end": "2026-10-04T09:30:00Z"},
            {"t_start": "2026-10-04T10:00:00.000Z", "t_end": "2026-10-04T10:15:00.000Z"},
            {"t_start": None, "t_end": None}]
    assert plan_fix.s1_hours(recs) == pytest.approx(1.75)


def test_merge_summaries_replaces_re_piloted_cells():
    a = {"cells": [{"arm": "K8", "task": "rgb_int_cb", "effort": "high", "lengths": [1]},
                   {"arm": "K8", "task": "ifbench_pilot", "effort": "high", "lengths": [2]}],
         "arms": {"K8": {"t_start": 1, "t_end": 2}}}
    b = {"cells": [{"arm": "K8", "task": "rgb_int_cb", "effort": "high", "lengths": [9]}],
         "arms": {"K8": {"t_start": 5, "t_end": 6}}}
    m = plan_fix.merge_summaries([a, b])
    by = {c["task"]: c["lengths"] for c in m["cells"]}
    assert by == {"rgb_int_cb": [9], "ifbench_pilot": [2]}
    assert m["arms"]["K8"] == {"t_start": 1, "t_end": 2}  # one cell re-piloted: the arm's timing stays
    c = {"cells": a["cells"], "arms": {"K8": {"t_start": 7, "t_end": 8}}}
    assert plan_fix.merge_summaries([a, c])["arms"]["K8"] == {"t_start": 7, "t_end": 8}  # whole arm re-run


# ------------------------------------------- plan_rules vs HYPOTHESIS


def test_plan_rules_frozen_values():
    r = RULES
    assert r["version"] == "exp036-plan-2"
    assert r["budget"] == {"total_h": 40.0, "main_cap_h": 31.0, "margin": 1.15, "session_cap_h": 16.0,
                           "overrun_ceiling_h": 44.0}
    assert r["B_choices"] == [16, 8, 4, 2, 1] and r["B_max_aime"] == 8 and r["limit_fraction"] == 0.9
    assert r["caps"] == {"gpqa": 32768, "mmlu": 32768, "aime": 65536, "ifbench": 16384, "rgb": 16384}
    assert r["cap_raise"]["ceilings"] == {"gpqa": 65536, "mmlu": 65536, "aime": 98304, "ifbench": 32768, "rgb": 32768}
    assert r["nM_ladder"] == [588, 504, 406, 350, 294, 252, 196, 154]
    assert r["crash_fallback"] == {"heartbeat_stale_min": 30, "max_fallbacks": 2}
    assert r["ladder_estimate_h"] == {"B4": 1.4}
    assert r["sysctl_advice_mb"] == 114688


def _hyp() -> str:
    return (EXP_DIR / "HYPOTHESIS.md").read_text(encoding="utf-8")


def test_plan_defs_match_the_hypothesis_ladder_table():
    text = _hyp()
    rows = dict(re.findall(r"^\| (P\d+) \| (.+?) \|$", text, re.M))
    assert sorted(rows, key=lambda p: int(p[1:])) == RULES["plans"]
    nM = None
    for p in RULES["plans"]:
        d = RULES["plan_defs"][p]
        m = re.search(r"n_M = (\d+)", rows[p])
        if m:
            nM = int(m.group(1))
        assert d["nM"] == nM, p
        assert d["neg_fc"] == (p == "P0"), p
        assert d["forced"] == (int(p[1:]) <= 7), p
        assert d["k8_gpqa_de"] == (p != "P10"), p
    assert [RULES["plan_defs"][p]["nM"] for p in ("P0", "P2", "P3", "P4", "P5", "P6", "P7", "P9")] == [
        588] + RULES["nM_ladder"][1:]
    # Amendment 4 (2026-10-04): n_M need not be a multiple of 14 any more (the Lite allocation is proportional, not
    # n_M/14 per category); the registered ladder values are unchanged and stay within Lite's 588.
    assert RULES["nM_ladder"] == sorted(RULES["nM_ladder"], reverse=True) and RULES["nM_ladder"][0] == 588


def test_tier_a_queue_matches_the_hypothesis_list():
    q = [(c["arm"], c["task"]) for c in RULES["tier_a_queue"]]
    assert q[:6] == [("K8", t) for t in ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench", "rgb_cb")]
    assert set(q[6:14]) == {(a, t) for a in ("G8", "Q36-8") for t in ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb")}
    assert q.index(("Q36-8", "ifbench")) < q.index(("G8", "ifbench"))  # H4 complete first
    assert q[14:18] == [("K4", t) for t in ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb")]
    assert q[18:21] == [(a, "rgb_forced") for a in ("K8", "G8", "Q36-8")]
    assert q[21:] == [(a, t) for a in ("K8", "G8", "Q36-8") for t in ("rgb_neg", "rgb_fact")]
    n = {(c["arm"], c["task"]): c["n"] for c in RULES["tier_a_queue"]}
    assert n[("K8", "gpqa_en")] == 198 and n[("G8", "ifbench")] == 300 and n[("K4", "rgb_cb")] == 400
    assert n[("K8", "rgb_neg")] == 300 and n[("Q36-8", "rgb_fact")] == 100 and n[("G8", "mmlu_de")] == "nM"


def test_tier_b_cells_match_the_hypothesis_table():
    text = _hyp()
    rows = dict(re.findall(r"^\| (B\d+) \| (.+?) \| .+? \|$", text, re.M))
    assert sorted(rows, key=lambda b: int(b[1:])) == RULES["tier_b_order"]
    for item, cells in RULES["tier_b_cells"].items():
        arms = {c["arm"] for c in cells}
        for arm in arms - {"bench"}:
            assert arm in rows[item], (item, arm)
    efforts = {c["effort"] for c in RULES["tier_b_cells"]["B7"]}
    assert efforts == {"low", "medium"}
    assert {c["effort"] for c in RULES["tier_b_cells"]["B5"]} == {"none"}
    assert {c.get("pass") for c in RULES["tier_b_cells"]["B10"]} == {1}


def test_ladder_definitions_agree_with_analysis_power_if_present():
    power = pytest.importorskip("analysis.power")
    for name, nM, negfc, forced, de in power.LADDER:
        d = RULES["plan_defs"][name]
        assert (d["nM"], d["neg_fc"], d["forced"], d["k8_gpqa_de"]) == (nM, negfc, forced, de), name


def test_pilot_cells_match_the_hypothesis_pilot_table():
    c = {a: [x["task"] for x in v] for a, v in RULES["pilot"]["cells"].items()}
    assert len(c["K8"]) == 8 and "aime_en_pilot" in c["K8"]
    assert c["G8"] == c["Q36-8"] == [t for t in c["K8"] if t != "aime_en_pilot"]
    assert c["K4"] == [t for t in c["G8"] if t != "rgb_int_forced"]
    n = {x["task"]: x["n"] for x in RULES["pilot"]["cells"]["K8"]}
    assert n == {"gpqa_main_en": 8, "gpqa_main_de": 8, "mmlu_full_en": 8, "mmlu_full_de": 8, "ifbench_pilot": 8,
                 "rgb_int_cb": 16, "rgb_int_forced": 8, "aime_en_pilot": 4}


def test_q38_step_model_from_descriptive_speed_records():
    recs = [{"kind": "batch", "arm": "Q38-8", "B": B, "prompt_tokens": 1024 * B, "generation_tokens": 256 * B,
             "generation_tps": B / (0.03 + 0.004 * B)} for B in (1, 2, 4)]
    recs += [{"kind": "prefill", "arm": "Q38-8", "prompt_tps": v} for v in (480.0, 500.0, 520.0)]
    recs += [{"kind": "batch", "arm": "G8", "B": 1, "generation_tps": 99.0}]
    sm, pf = plan_fix.q38_from_speed_desc(recs)
    a, b, c = sm["Q38-8"]
    assert a == pytest.approx(0.03) and b == pytest.approx(0.004) and c == 0.0
    assert pf["Q38-8"] == 500.0
    # B9 becomes projectable once the context carries them.
    pilot, steps, s1, ctx = synthetic(0)
    ctx["step_models_extra"], ctx["prefill_tps_extra"] = {"Q38-8": [a, b, c]}, {"Q38-8": 500.0}
    plan, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    b9 = {t["item"]: t for t in plan["tier_b_considered"]}["B9"]
    assert b9["projected_h"] is not None and b9["reason"] != "unprojectable"


def test_cli_end_to_end_on_a_temporary_experiment_dir(tmp_path, monkeypatch, capsys):
    """plan_fix.py --pilot … reads the summary, the step logs, the S1 records,
    the manifests record and the arm weight sizes, and writes the two files."""
    import shutil
    import subprocess

    exp = tmp_path / "exp"
    (exp / "runner").mkdir(parents=True)
    shutil.copy(EXP_DIR / "runner" / "plan_rules.json", exp / "runner" / "plan_rules.json")
    (exp / "scorers").mkdir()
    (exp / "scorers" / "x.py").write_text("X = 1\n")
    subprocess.run(["git", "init", "-q", str(exp)], check=True)
    subprocess.run(["git", "-C", str(exp), "config", "user.name", "Miktam"], check=True)
    subprocess.run(["git", "-C", str(exp), "config", "user.email", "hello@localfirstai.eu"], check=True)
    (exp / "HYPOTHESIS.md").write_text("# H\n\n## Amendment 1 — gate fix (2026-10-04T12:00:00Z)\n\nx\n")
    models = tmp_path / "models"
    from runner.common import ARMS

    for arm, size in WEIGHTS.items():
        d = models / ARMS[arm]["dir"]
        d.mkdir(parents=True)
        with open(d / "model.safetensors", "wb") as f:
            f.truncate(size)  # sparse file of the arm's weight size
    monkeypatch.setenv("EXP036_MODELS", str(models))
    from runner import memory

    monkeypatch.setattr(memory, "effective_limit", lambda *a, **k: (115_448_725_504, {"chosen": "test"}))

    pilot, steps, s1, _ = synthetic(0)
    res = exp / "results"
    pdir = res / "pilot" / "20261004T150000Z"
    for c in pilot["cells"]:
        sf = pdir / c["arm"] / f"{c['task']}_high.steps.jsonl"
        sf.parent.mkdir(parents=True, exist_ok=True)
        if not sf.exists():
            sf.write_text("".join(json.dumps(st) + "\n" for st in steps[c["arm"]]))
        c["steps_file"] = sf.relative_to(exp).as_posix()
    # The arms' step logs are read from the cells' files: keep one copy per arm.
    seen = set()
    for c in pilot["cells"]:
        if c["arm"] in seen:
            c["steps_file"] = None
        seen.add(c["arm"])
    pilot.update({"type": "pilot_summary", "t_start": "2026-10-04T13:00:00Z", "t_end": "2026-10-04T13:00:00Z",
                  "complete": True})
    (res / "pilot_summary_20261004T150000Z.json").write_text(json.dumps(pilot))
    (res / "preflight_20261004T080000Z.json").write_text(json.dumps(
        {"mode": "deep", "t_start": "2026-10-04T08:00:00Z", "t_end": "2026-10-04T08:24:00Z"}))
    (res / "preflight_20261004T075000Z.json").write_text(json.dumps(
        {"mode": "quick", "t_start": "2026-10-04T07:50:00Z", "t_end": "2026-10-04T07:51:00Z"}))
    (res / "gate").mkdir()
    (res / "gate" / "gate_20261004T120000Z.json").write_text(json.dumps(
        {"t_start": "2026-10-04T09:00:00Z", "t_end": "2026-10-04T14:00:00Z", "verdict": {"K8": "PASS"}}))
    (res / "bench").mkdir()
    (res / "bench" / "speed_20261004T140000Z.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        {"type": "header", "t_start": "2026-10-04T14:00:00Z"},
        {"kind": "run", "t_start": "2026-10-04T14:01:00Z", "t_end": "2026-10-04T14:02:00Z"},
        {"type": "end", "complete": True, "t_start": "2026-10-04T14:00:00Z", "t_end": "2026-10-04T14:30:00Z"}]))
    (res / "manifests_20261004T085000Z.json").write_text(json.dumps(
        {"manifests": {"tasks/manifests/mmlu_prox_lite_en.json": "c" * 64},
         "private": {"manifests/gpqa_diamond_en.jsonl": "d" * 64}}))
    # Amendment 4: the plan reads the Lite listing (categories in the Webster seat order) for the per-category n
    from tasks import mmlu_prox

    lite_counts = json.loads((EXP_DIR / "tasks" / "selection_rules.json").read_text())["mmlu_prox"]["lite_category_counts"]
    (exp / "tasks" / "manifests").mkdir(parents=True)
    (exp / "tasks" / "manifests" / "mmlu_prox_lite_en.json").write_text(json.dumps(
        {"items": [{"id": str(i), "category": c} for i, c in enumerate(mmlu_prox.webster_order(lite_counts))]}))

    rc = plan_fix.main(["--pilot", "results/pilot_summary_20261004T150000Z.json", "--exp-dir", str(exp)])
    out = capsys.readouterr().out
    assert rc == 0, out
    pf = sorted(res.glob("plan_fixed_*.json"))
    am = sorted(res.glob("AMENDMENT_*_*.md"))
    assert len(pf) == 1 and len(am) == 1 and am[0].name.startswith("AMENDMENT_2_")
    plan = json.loads(pf[0].read_text())
    # deep preflight 0.4 h + gate 5 h + bench cell 0.5 h; the quick preflight and the
    # bench file's inner run lines do not count
    assert plan["S1_hours"] == pytest.approx(0.4 + 5.0 + 0.5)
    assert plan["manifests"] == {"tasks/manifests/mmlu_prox_lite_en.json": "c" * 64,
                                 "$EXP036_PRIVATE/manifests/gpqa_diamond_en.jsonl": "d" * 64}
    assert plan["gate_record"]["path"] == "results/gate/gate_20261004T120000Z.json"
    assert plan["status"] == "FIXED" and plan["plan"] == "P0"
    assert plan["mmlu_allocation"] == dict(sorted(lite_counts.items()))  # P0: n_M = 588, all of Lite
    from runner.guard import PLAN_FILE_RE

    m = PLAN_FILE_RE.search(am[0].read_text())
    assert m.group(1) == pf[0].relative_to(exp).as_posix()
    from runner.common import sha256_file

    assert m.group(2) == sha256_file(pf[0])


# ------------------------------------------------- review fixes 2026-10-03


def _without(pilot, arms):
    pilot["cells"] = [c for c in pilot["cells"] if c["arm"] not in arms]


def test_k4_excluded_by_the_gate_has_no_cell_and_four_hypotheses_not_run():
    pilot, steps, s1, ctx = synthetic(0, edit=lambda p: _without(p, {"K4"}))
    steps.pop("K4")
    ctx["excluded_arms"] = {"K4": "gate K4 FAIL (gate_20261004T120000Z.json)"}
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["status"] == "FIXED"
    assert all(q["arm"] != "K4" and q["arm"] != "bench" for q in plan["queue"])   # B4 ladder needs K4
    assert "B8" not in plan["tier_b"] and "B4" not in plan["tier_b"]
    assert set(plan["not_run"]) == {"H1", "H7", "H8", "D1"}
    assert plan["rows"]["H7"] == [] and plan["rows"]["H7_status"].startswith("NOT RUN")
    assert "Excluded arms (no cell queued): K4" in md and "NOT RUN by this amendment" in md


def test_a_dropped_peer_has_no_cell_and_h4_is_not_run_without_qwen():
    pilot, steps, s1, ctx = synthetic(0, edit=lambda p: _without(p, {"Q36-8"}))
    steps.pop("Q36-8")
    ctx["excluded_arms"] = {"Q36-8": "peer check fail (peers_20261004T110000Z.json)"}
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["status"] == "FIXED"
    assert all(q["arm"] != "Q36-8" for q in plan["queue"])
    assert plan["peers"] == ["G8"] and plan["not_run"] == {"H4": plan["not_run"]["H4"]}
    assert "MoE peers for H3, H6 and the H2 protocol control: G8" in md


def test_h7_rows_of_b8_under_p10_leave_out_gpqa_de():
    rules = json.loads(json.dumps(RULES))
    rules["plans"] = ["P10"]                       # force P10; nominal lengths leave room for Tier B
    pilot, steps, s1, ctx = synthetic(0)
    plan, _, _ = plan_fix.fix(pilot, steps, s1, rules, [], context=ctx)
    assert plan["plan"] == "P10" and "B8" in plan["tier_b"]
    assert plan["rows"]["H7"] == ["mmlu_en", "mmlu_de", "ifbench", "rgb_cb", "gpqa_en"]


def test_pilot_inputs_must_be_complete_and_cover_every_cell():
    pilot, _, _, _ = synthetic(0)
    full = dict(pilot, complete=True)
    assert plan_fix.check_pilot_inputs([full], RULES, {}) == []
    assert "not a complete pilot summary" in plan_fix.check_pilot_inputs([dict(pilot)], RULES, {})[0]
    partial = dict(full, cells=full["cells"][:3])
    assert "missing" in plan_fix.check_pilot_inputs([partial], RULES, {})[0]
    repilot = dict(full, cells=full["cells"][:1], repilot=True)
    assert plan_fix.check_pilot_inputs([repilot], RULES, {})              # alone: refused
    assert plan_fix.check_pilot_inputs([full, repilot], RULES, {}) == []   # full first, then the re-pilot
    no_k4 = dict(full, cells=[c for c in full["cells"] if c["arm"] != "K4"])
    assert plan_fix.check_pilot_inputs([no_k4], RULES, {"K4": "gate K4 FAIL"}) == []
    assert plan_fix.check_pilot_inputs([full], RULES, {"K4": "gate K4 FAIL"})  # K4 cells of an excluded arm


def test_plan_fix_output_with_a_dropped_peer_feeds_the_verdicts():
    """Integration (review fix 2026-10-03): a real plan_fix plan with Qwen3.6 dropped gives H3 and H6 on Gemma 4
    alone and H4 NOT RUN in analysis/verdicts.py."""
    from datetime import datetime, timezone

    from analysis import verdicts as V
    from test_analysis_world import GPQA_EN_EXCLUDED, POST_STRAT, all_confirmed_world

    pilot, steps, s1, ctx = synthetic(0, edit=lambda p: _without(p, {"Q36-8"}))
    steps.pop("Q36-8")
    ctx["excluded_arms"] = {"Q36-8": "peer check fail"}
    plan, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    w = all_confirmed_world()
    for k in [k for k in w.rates if k[0] == "Q36-8"]:
        del w.rates[k]
    plan.update({"gpqa_en_excluded": list(GPQA_EN_EXCLUDED), "post_strat_counts": POST_STRAT, "cells": []})
    w.plan = plan
    v = w.compute(now=datetime(2026, 10, 10, tzinfo=timezone.utc), exploratory=False)
    assert v["config"]["peers"] == ["G8"]
    assert v["verdicts"]["H3_detail"]["verdict"] != V.NOT_RUN and v["verdicts"]["H3_detail"]["detail"]["peers"] == ["G8"]
    assert v["verdicts"]["H6_detail"]["verdict"] != V.NOT_RUN
    assert v["verdicts"]["H4_detail"]["verdict"] == V.NOT_RUN


# ------------------------------------------------- Amendment 4 (2026-10-04)


def _lite_listing():
    from tasks import mmlu_prox

    counts = json.loads((EXP_DIR / "tasks" / "selection_rules.json").read_text())["mmlu_prox"]["lite_category_counts"]
    return counts, mmlu_prox.webster_order(counts)


@pytest.mark.parametrize("s,want_plan,n_M", [(0, "P0", 588), (1, "P8", 196), (2, "P10", 154)])
def test_plan_records_the_mmlu_allocation_at_n_M(s, want_plan, n_M):
    """The plan states the per-category n of the MMLU-ProX-Lite rows: the first n_M entries of the Lite listing
    (the Webster seat order), not n_M/14 each; the amendment prints it and says the power figures are conservative."""
    from tasks import mmlu_prox

    counts, listing = _lite_listing()
    pilot, steps, s1, ctx = synthetic(s)
    ctx["mmlu_lite_categories"] = listing
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["plan"] == want_plan and plan["n_M"] == n_M
    assert plan["mmlu_allocation"] == dict(sorted(mmlu_prox.allocation(counts, n_M).items()))
    assert sum(plan["mmlu_allocation"].values()) == n_M and len(set(plan["mmlu_allocation"].values())) > 1
    line = next(l for l in md.splitlines() if l.startswith("- MMLU-ProX-Lite items per category at n_M"))
    assert "history " + str(plan["mmlu_allocation"]["history"]) in line
    assert "conservative" in md and "design effect 1.11" in md
    assert "1.114" in plan["power"]["assumptions"] and plan["power"]["mmlu_deff"]["amendment4_proportional"] < 1.01


def test_plan_without_the_lite_manifest_records_no_allocation():
    plan, md, _ = run_fix(0)
    assert plan["mmlu_allocation"] is None and "items per category" not in md
    pilot, steps, s1, ctx = synthetic(0)
    ctx["mmlu_lite_categories"] = _lite_listing()[1][:100]  # a listing shorter than n_M is named, not truncated
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert "fewer than n_M = 588" in plan["mmlu_allocation"]["error"] and "fewer than n_M = 588" in md
