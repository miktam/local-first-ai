"""analysis/verdicts.py (BUILD_SPEC §5.8; HYPOTHESIS H1-H8, D1, Common rules, Plain answers).

Synthetic score sets whose true verdict is known by construction hit CONFIRMED,
REFUTED and INCONCLUSIVE for every hypothesis (each alone, the others NOT RUN,
so it sits at Holm step 1 with threshold alpha/8), NOT RUN for every hypothesis
with m frozen, the Holm interplay of both families, "leans refuted", the
E8 flag, the truncation-sensitivity headline rule, H8's 5-of-6 condition, the
H6 wording state, D1, the plain-answer states, escalation to B = 100,000, the
rescore refusal and byte-identical re-runs.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from datetime import datetime, timezone

import numpy as np
import pytest

from analysis import stats
from analysis import verdicts as V
from test_analysis_world import (GPQA_EXCLUDED, World, all_confirmed_world, fit_records, kl_blob,
                                 speed_blocks, tokenizer_docs, vend)

NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
FAMILY = list(V.FAMILY)


def words(v):
    return {h: v["verdicts"][f"{h}_detail"]["verdict"] for h in FAMILY}


# ======================================================================
# The all-CONFIRMED world
# ======================================================================

@pytest.fixture(scope="module")
def confirmed():
    return all_confirmed_world().compute(now=NOW)


def test_all_confirmed(confirmed):
    v = confirmed
    assert words(v) == {h: V.CONFIRMED for h in FAMILY}
    assert v["verdicts"]["D1"] == V.CONFIRMED
    assert v["summary"]["holm"]["m"] == 8
    for h in FAMILY:
        d = v["verdicts"][f"{h}_detail"]
        assert d["p_adjusted"] <= 0.05
        assert d["headline_eligible"] and d["headline_verdict"] == V.CONFIRMED
        assert d["seed"] == stats.seed_of(h)
        assert "bound_confirm" in d and "bound_refute" in d
    # the shape of exp_023: ts, hardware, config, manifests, summary, verdicts
    assert set(v) == {"ts", "hardware", "config", "manifests", "summary", "verdicts"}
    assert v["ts"] == "2026-10-10T12:00:00Z" and v["hardware"] == V.HOST_LABEL


def test_all_confirmed_details(confirmed):
    vd = confirmed["verdicts"]
    h2 = vd["H2_detail"]["detail"]
    assert set(h2["rows"]) == {"gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench"}
    assert h2["rows"]["gpqa_en"]["n"] == 197                      # eval-framework primary set
    assert h2["gpqa198_sensitivity"]["gpqa_en_n"] == 198
    assert h2["reconstructed_rows"] == ["mmlu_en", "mmlu_de", "ifbench"]
    assert h2["protocol_control"]["peers_within_band"]
    assert vd["H1_detail"]["estimate"] == pytest.approx(0.90, rel=1e-9)
    assert vd["H1_detail"]["detail"]["n_blocks"] == 10
    assert vd["H5_detail"]["detail"]["sub_verdict"] == "report-consistent"
    assert vd["H8_detail"]["detail"]["per_text_condition"]
    ts = vd["H3_detail"]["truncation_sensitivity"]
    assert ts["n_excluded"] == 0 and ts["agrees"]


# ======================================================================
# Every hypothesis alone: CONFIRMED / REFUTED / INCONCLUSIVE by construction
# ======================================================================

def world_only(h, mode):
    w = World()
    w.plan["peers"] = ["G8", "Q36-8"]
    if h == "H1":
        w.speed = {"C": speed_blocks(0.90), "R": speed_blocks(0.60),
                   "I": speed_blocks(0.75, noise_sd=0.05)}[mode]
    elif h == "H2":
        shift = {"C": 0.0, "R": -0.12, "I": -0.04}[mode]
        for row in ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench"):
            w.set("K8", row, vend(row, "K8") + shift)
    elif h == "H3":
        k = {"C": 0.70, "R": 0.88, "I": 0.79}[mode]
        for row in ("mmlu_en", "mmlu_de"):
            w.set("K8", row, k)
            w.set("G8", row, 0.80)
            w.set("Q36-8", row, 0.80)
    elif h == "H4":
        k, q = {"C": (0.85, 0.60), "R": (0.50, 0.70), "I": (0.68, 0.66)}[mode]
        w.set("K8", "ifbench", k)
        w.set("Q36-8", "ifbench", q)
    elif h == "H5":
        w.tokenizer = {"C": tokenizer_docs(), "R": tokenizer_docs(bpt_g=4.90 / 1.05, bpt_q=4.90 / 1.05),
                       "I": tokenizer_docs(bpt_q=4.90 / 1.15)}[mode]
    elif h == "H6":
        k = {"C": 0.51, "R": 0.79, "I": 0.69}[mode]
        w.set("K8", "rgb_cb", k)
        w.set("G8", "rgb_cb", 0.79)
        w.set("Q36-8", "rgb_cb", 0.79)
    elif h == "H7":
        for row, r in (("mmlu_en", 0.80), ("mmlu_de", 0.75), ("ifbench", 0.78), ("rgb_cb", 0.51)):
            w.set("K8", row, r)
            if mode == "C":
                w.rates[("K4", row, "high", 0)] = None          # identical to K8
            else:
                w.set("K4", row, r + {"R": -0.10, "I": -0.03}[mode])
    elif h == "H8":
        w.kl = {"C": kl_blob((1.0,) * 6), "R": kl_blob((3.0,) * 6), "I": kl_blob((1.5,) * 6, noise=0.2)}[mode]
    return w


EXPECTED = {"C": V.CONFIRMED, "R": V.REFUTED, "I": V.INCONCLUSIVE}


@pytest.mark.parametrize("h", FAMILY)
@pytest.mark.parametrize("mode", ["C", "R", "I"])
def test_each_hypothesis_alone(h, mode):
    v = world_only(h, mode).compute(now=NOW, exploratory=False)
    w = words(v)
    assert w[h] == EXPECTED[mode], (h, mode, v["verdicts"][f"{h}_detail"])
    for other in FAMILY:
        if other != h:
            assert w[other] == V.NOT_RUN
            d = v["verdicts"][f"{other}_detail"]
            assert d["p"] == 1.0 and d["p_rev"] == 1.0
    d = v["verdicts"][f"{h}_detail"]
    # alone in the family it is at step 1 (unless its p ties the NOT RUN p = 1):
    # adjusted = 8 x raw, because m = 8 stays frozen
    assert d["p_adjusted"] == pytest.approx(min(1.0, 8 * d["p"]))
    assert d["p_rev_adjusted"] == pytest.approx(min(1.0, 8 * d["p_rev"]))
    if d["p"] < 1.0:
        assert d["holm_step"] == 1
    if d["p_rev"] < 1.0:
        assert d["holm_step_rev"] == 1


# ======================================================================
# Holm interplay, both families, leans refuted (decision layer, exact p)
# ======================================================================

def _res(p, p_rev, **kw):
    return {"status": "RUN", "p": p, "p_rev": p_rev, **kw}


def _decide(rs):
    full = {h: rs.get(h, V.not_run("absent")) for h in FAMILY}
    return {h: d["verdict"] for h, d in V.decide(full, FAMILY, V.load_margins()).items()}, \
        V.decide(full, FAMILY, V.load_margins())


def test_holm_same_raw_p_different_verdicts():
    # H1 at p = 0.02: CONFIRMED when the seven others are tiny (step 8, threshold 0.05) ...
    w, _ = _decide({**{h: _res(1e-5, 1.0) for h in FAMILY}, "H1": _res(0.02, 0.9)})
    assert w["H1"] == V.CONFIRMED
    # ... INCONCLUSIVE when it is the only small one among eight (adjusted 0.16).
    w, _ = _decide({**{h: _res(0.9, 0.9) for h in FAMILY}, "H1": _res(0.02, 0.9)})
    assert w["H1"] == V.INCONCLUSIVE


def test_holm_step_down_stops_at_first_failure():
    p = {"H1": 0.001, "H2": 0.0072, "H3": 0.0075}
    w, d = _decide({h: _res(p.get(h, 0.5), 0.9) for h in FAMILY})
    assert w["H1"] == V.CONFIRMED
    assert w["H2"] == V.INCONCLUSIVE and w["H3"] == V.INCONCLUSIVE    # H3 raw clears alpha/6, adjusted does not
    assert d["H3"]["confirm"]["adjusted"] == pytest.approx(0.0504)


def test_not_run_keeps_m_at_8():
    w, d = _decide({"H1": _res(0.006, 0.9)})
    assert d["H1"]["confirm"]["adjusted"] == pytest.approx(0.048)     # 8 x 0.006, not 7 x
    assert w["H1"] == V.CONFIRMED
    w, d = _decide({"H1": _res(0.007, 0.9)})
    assert d["H1"]["confirm"]["adjusted"] == pytest.approx(0.056) and w["H1"] == V.INCONCLUSIVE


@pytest.mark.parametrize("h", ["H3", "H4"])
def test_confirmed_and_refuted_both_possible_resolves_to_confirmed(h):
    w, d = _decide({h: _res(1e-4, 1e-4)})
    assert d[h]["confirm"]["adjusted"] <= 0.05 and d[h]["refute"]["adjusted"] <= 0.05
    assert w[h] == V.CONFIRMED


def test_refuted_family_is_separate():
    rs = {h: _res(0.9, 0.9) for h in FAMILY}
    rs["H2"] = _res(0.9, 0.001)
    rs["H5"] = _res(0.9, 0.012)          # refute family: step 2 threshold 0.05/7; adjusted 0.084
    w, d = _decide(rs)
    assert w["H2"] == V.REFUTED and w["H5"] == V.INCONCLUSIVE
    assert d["H5"]["refute"]["adjusted"] == pytest.approx(7 * 0.012)


def test_leans_refuted_label():
    full = {h: V.not_run("absent") for h in FAMILY}
    full["H6"] = _res(0.5, 0.02)
    d = V.decide(full, FAMILY, V.load_margins())
    assert d["H6"]["verdict"] == V.INCONCLUSIVE and d["H6"]["leans_refuted"]
    assert V._label(d["H6"]["verdict"], d["H6"]["leans_refuted"], []) == V.LEANS_REFUTED
    full["H6"] = _res(0.5, 0.03)
    d = V.decide(full, FAMILY, V.load_margins())
    assert not d["H6"]["leans_refuted"]


def test_h8_confirm_condition_blocks_confirmed():
    w, _ = _decide({"H8": _res(1e-4, 0.99, confirm_condition=False)})
    assert w["H8"] == V.INCONCLUSIVE


# ======================================================================
# NOT RUN for every hypothesis, m frozen
# ======================================================================

def _drop(w, h):
    if h == "H1":
        w.speed = None
    elif h == "H2":
        del w.rates[("K8", "gpqa_de", "high", 0)]
    elif h == "H3":
        del w.rates[("G8", "mmlu_de", "high", 0)]
    elif h == "H4":
        w.plan["peers"] = ["G8"]
    elif h == "H5":
        w.tokenizer = None
    elif h == "H6":
        del w.rates[("Q36-8", "rgb_cb", "high", 0)]
    elif h == "H7":
        w.plan["cells"] = [{"arm": "K4", "task": "ifbench", "effort": "high", "n": 301}]
    elif h == "H8":
        w.kl = None
    return w


@pytest.mark.parametrize("h", FAMILY)
def test_not_run_each(h):
    v = _drop(all_confirmed_world(), h).compute(now=NOW, exploratory=False)
    d = v["verdicts"][f"{h}_detail"]
    assert d["verdict"] == V.NOT_RUN and v["verdicts"][h] == V.NOT_RUN
    assert d["p"] == 1.0 and d["p_rev"] == 1.0 and d["reason"]
    assert v["summary"]["holm"]["m"] == 8
    assert v["summary"]["holm"]["confirm"][h]["raw"] == 1.0
    assert not d["headline_eligible"]
    others = {k: x for k, x in words(v).items() if k != h}
    if h == "H4":       # the peer drop also leaves H3 and H6 on the one remaining peer
        assert v["verdicts"]["H3_detail"]["detail"]["peers"] == ["G8"]
    assert all(x == V.CONFIRMED for x in others.values()), others


def test_not_run_from_plan_reason_and_incomplete_cell():
    w = all_confirmed_world()
    w.plan["not_run"] = {"H6": "S3b overrun ceiling reached"}
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H6_detail"]["reason"] == "S3b overrun ceiling reached"
    w = _drop(all_confirmed_world(), "H7")
    v = w.compute(now=NOW, exploratory=False)
    assert "300 of 301 items scored" in v["verdicts"]["H7_detail"]["reason"]


def test_k8_cannot_run_k4_primary():
    w = all_confirmed_world()
    for (arm, task, eff, ps), r in list(w.rates.items()):
        if arm == "K8":
            w.rates[("K4", task, eff, ps)] = r
    w.plan["kolibri_primary"] = "K4"
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H7_detail"]["verdict"] == V.NOT_RUN
    assert "K4 is primary" in v["verdicts"]["H7_detail"]["reason"]
    assert v["verdicts"]["H2_detail"]["detail"]["title"] == "at MLX 4-bit"
    assert v["verdicts"]["H2_detail"]["detail"]["arm"] == "K4"
    assert v["verdicts"]["H3_detail"]["detail"]["arm"] == "K4"


# ======================================================================
# D1
# ======================================================================

@pytest.mark.parametrize("peaks,want", [
    ([45.0, 46.0, 47.0], V.CONFIRMED),            # median 46.0 <= 46.66
    ([46.66, 46.66, 60.0], V.CONFIRMED),          # equality confirms
    ([50.0, 50.5, 51.0], V.INCONCLUSIVE),          # fits only without headroom
    ([51.84, 51.84, 51.84], V.INCONCLUSIVE),       # REFUTED needs > 51.84
    ([52.0, 53.0, 52.5], V.REFUTED),               # 64k does not fit
])
def test_d1(peaks, want):
    w = World()
    w.fit = fit_records(peaks)
    ctx = w.ctx()
    r = V.d1(ctx)
    assert r["verdict"] == want
    assert r["median_peak_gib_64k"] == pytest.approx(float(np.median(peaks)), abs=1e-6)
    if want == V.INCONCLUSIVE:
        assert "without the 10 % headroom" in r["label"]


def test_d1_not_run():
    assert V.d1(World().ctx())["verdict"] == V.NOT_RUN


# ======================================================================
# E8 flag, truncation sensitivity, headline eligibility
# ======================================================================

def test_e8_flag_marks_h3_and_blocks_headline():
    w = all_confirmed_world()
    w.set("G8", "mmlu_en", vend("mmlu_en", "G8") - 0.10)     # |D_peer| > 8 pp on a row H3 uses
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H3_detail"]
    assert d["e8_flags"] and d["e8_flags"][0]["arm"] == "G8" and d["e8_flags"][0]["row"] == "mmlu_en"
    assert v["verdicts"]["H3"].startswith(d["verdict"] + " — peer control failed (G8, mmlu_en")
    assert not d["headline_eligible"]
    assert not v["verdicts"]["H4_detail"]["e8_flags"]          # H4 uses IFBench only


def test_e8_flag_h4_uses_qwen_only():
    w = all_confirmed_world()
    w.set("G8", "ifbench", vend("ifbench", "G8") - 0.15)
    v = w.compute(now=NOW, exploratory=False)
    assert not v["verdicts"]["H4_detail"]["e8_flags"]
    w.set("Q36-8", "ifbench", vend("ifbench", "Q36-8") - 0.10)
    w.set("K8", "ifbench", 0.70)
    v = w.compute(now=NOW, exploratory=False)
    assert [f["arm"] for f in v["verdicts"]["H4_detail"]["e8_flags"]] == ["Q36-8"]


def test_truncation_sensitivity_disagreement_is_not_headline_eligible():
    w = World()
    # K8 at 0.80; Q36-8 truncates 120 of 300 items (scored 0), and on the other 180
    # it matches K8, so H4 is CONFIRMED overall but not once the truncated items go.
    w.set("K8", "ifbench", 0.80)
    w.set("Q36-8", "ifbench", 0.80)
    w.trunc[("Q36-8", "ifbench")] = set(range(0, 300, 5)) | set(range(1, 300, 5))
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H4_detail"]
    assert d["verdict"] == V.CONFIRMED
    ts = d["truncation_sensitivity"]
    assert ts["n_excluded"] == 120 and ts["verdict"] != V.CONFIRMED and not ts["agrees"]
    assert d["headline_verdict"] == V.TRUNCATION_SENSITIVE
    assert not d["headline_eligible"]
    assert d["detail"]["truncation_rate"]["Q36-8"] == pytest.approx(0.4)


def test_truncated_item_scores_zero_even_if_marked_correct():
    s = V.Scores([{"key": {"arm": "K8", "task": "gpqa_de", "effort": "high", "item": "a", "pass": 0},
                   "correct": True, "truncated": True}])
    assert s.cell("K8", "gpqa_de")["a"].score == 0.0


# ======================================================================
# H8 5-of-6, per-token note; H6 wording; plain answers
# ======================================================================

def test_h8_fails_five_of_six():
    w = World()
    w.kl = kl_blob((0.5, 0.5, 0.5, 0.5, 2.0, 2.0), noise=0.01)
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H8_detail"]
    assert d["p_adjusted"] <= 0.05                               # the ratio test alone would confirm
    assert d["detail"]["texts_at_or_below_threshold"] == 4
    assert not d["detail"]["per_text_condition"]
    assert d["verdict"] == V.INCONCLUSIVE


def test_h8_drops_a_peer_family_excluded_by_the_peer_check():
    w = all_confirmed_world()
    w.plan["excluded_arms"] = {"Q38-4": "peer check fail (peers_x.json)"}
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H8_detail"]
    assert d["detail"]["models"] == ["kolibri", "gemma4", "qwen3_6"]
    assert d["verdict"] == V.CONFIRMED


def test_h8_not_run_when_every_peer_family_is_excluded():
    w = all_confirmed_world()
    w.plan["excluded_arms"] = {"G4": "x", "Q36-8": "x", "Q38-4": "x"}
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H8"] == V.NOT_RUN


def test_h8_per_token_note():
    w = all_confirmed_world()
    blob = kl_blob((1.2,) * 6, noise=0.01)
    for t in blob["families"]["kolibri"]["texts"].values():   # few Kolibri tokens: per-token KL is high
        for b in t["blocks"]:
            b["tokens"] = max(1, b["bytes"] // 12)
    w.kl = blob
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H8_detail"]
    assert d["verdict"] == V.CONFIRMED and d["detail"]["per_token"]["exceeds_threshold"]
    assert "per_token_note" in v["summary"]["plain_answers"]["Q5"]


def test_h6_wording_forced_knows_less():
    w = all_confirmed_world()
    w.plan["forced_cells"] = True
    w.set("K8", "rgb_forced", 0.40)
    w.set("G8", "rgb_forced", 0.80)
    w.set("Q36-8", "rgb_forced", 0.80)
    v = w.compute(now=NOW, exploratory=False)
    f = v["verdicts"]["H6_detail"]["detail"]["forced"]
    assert f["ran"] and f["upper_below_margin"]
    assert v["summary"]["plain_answers"]["Q4"]["id"] == "Q4_knows_less"


def test_h6_wording_without_forced_cells():
    w = all_confirmed_world()
    w.plan["plan"] = "P8"            # P8-P10: no forced-answer cells
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H6_detail"]["detail"]["forced"] == {"ran": False}
    assert v["summary"]["plain_answers"]["Q4"]["id"] == "Q4_confirmed_rate"
    assert "no 'recall'" in v["verdicts"]["H6_detail"]["detail"]["wording"]


def test_plain_answers_q2_peers_outside_band_takes_precedence():
    w = all_confirmed_world()
    for arm in ("G8", "Q36-8"):
        for row in ("mmlu_en", "mmlu_de", "ifbench"):
            w.set(arm, row, vend(row, arm) + 0.05)
    v = w.compute(now=NOW, exploratory=False)
    pc = v["verdicts"]["H2_detail"]["detail"]["protocol_control"]
    assert not pc["peers_within_band"] and pc["peers_mean_D_shared"] > 0.02
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "Q2_peers_outside" and q2["direction"] == "higher"


@pytest.mark.parametrize("mode,state", [("R", "Q2_refuted"), ("I", "Q2_inconclusive"), ("C", "Q2_confirmed")])
def test_plain_answers_q2_by_verdict(mode, state):
    w = world_only("H2", mode)
    for arm in ("G8", "Q36-8"):
        for row in ("mmlu_en", "mmlu_de", "ifbench"):
            w.set(arm, row, vend(row, arm))
    v = w.compute(now=NOW, exploratory=False)
    assert v["summary"]["plain_answers"]["Q2"]["id"] == state


def test_plain_answers_not_run_and_q5():
    w = all_confirmed_world()
    w.kl = None
    w.speed = None
    w.fit = None
    v = w.compute(now=NOW, exploratory=False)
    pa = v["summary"]["plain_answers"]
    assert pa["Q5"]["id"] == "Q5_not_run" and pa["Q5"]["missing"] == ["H8"]
    assert pa["Q1"][1]["id"] == "not_run" and pa["Q1"][1]["H"] == "H1"
    assert pa["Q1"][2]["id"] == "not_run" and pa["Q1"][2]["H"] == "D1"
    assert pa["Q4"]["id"] == "Q4_confirmed_rate"


def test_plain_answers_q3_flags():
    """German basis and the E1 label from the cells that ran (review fix 2026-10-03)."""
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B1", "B2", "B3", "B6"]
    for arm in ("K8", "G8", "Q36-8"):
        for row in ("gpqa_en", "gpqa_de", "aime_en", "aime_de"):
            w.set(arm, row, 0.5)
    v = w.compute(now=NOW, exploratory=False)
    q3 = v["summary"]["plain_answers"]["Q3"]
    assert not q3["e1_label_applies"]
    assert q3["german_basis"] == ["MMLU-ProX DE", "GPQA-D DE", "AIME DE"]


def test_plain_answers_q3_planned_tier_b_that_never_ran_does_not_count():
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B1", "B2", "B3", "B6"]          # planned, none of their cells ran
    v = w.compute(now=NOW, exploratory=False)
    q3 = v["summary"]["plain_answers"]["Q3"]
    assert q3["e1_label_applies"]
    assert q3["german_basis"] == ["MMLU-ProX DE"]


# ======================================================================
# H2 details
# ======================================================================

def test_h2_gpqa_primary_excludes_the_overlong_item(tmp_path):
    w = World()
    w.set("K8", "gpqa_en", 0.5)
    w.plan["h2_rows"] = ["gpqa_en"]
    ctx = w.ctx()
    cell = ctx.cell("K8", "gpqa_en")
    items = ctx.primary_items("gpqa_en", cell)
    assert len(items) == 197 and GPQA_EXCLUDED not in items
    r = V.h2(ctx, 2000)
    want = np.mean([cell[i].score for i in items]) - vend("gpqa_en", "K8")
    assert r["estimate"] == pytest.approx(want)
    w.plan.pop("gpqa_en_excluded")
    # No exclusion in the plan and no manifest with in_primary flags: an empty manifests dir, so the test does not
    # depend on whether the real tasks/manifests/gpqa_diamond_en.json exists yet (RUNBOOK step 7; review fix
    # 2026-10-03: found by running the suite on a late-run copy in run-host mode).
    w.plan["manifests_dir"] = str(tmp_path)
    with pytest.raises(V.ConfigError):
        V.h2(w.ctx(), 2000)


def test_h2_mmlu_rows_are_post_stratified():
    w = World()
    rates = [0.2 + 0.05 * c for c in range(14)]
    w.set("K8", "mmlu_en", rates)
    ctx = w.ctx()
    pt = V.row_point(ctx, "K8", "mmlu_en")
    counts = np.array(list(w.plan["post_strat_counts"]["en"].values()), dtype=float)
    per = [round(r * 42) / 42 for r in rates]
    assert pt == pytest.approx(float(np.sum(counts / counts.sum() * np.array(per))))
    assert pt != pytest.approx(float(np.mean(per)))              # differs from the unweighted Lite mean


def test_h2_two_pass_b10():
    w = World()
    for row in ("gpqa_en", "gpqa_de"):
        w.set("K8", row, 0.80)
        w.set("K8", row, 0.86, pass_=1)
    w.plan["tier_b"] = ["B10"]
    w.plan["h2_rows"] = ["gpqa_en", "gpqa_de"]
    ctx = w.ctx()
    r = V.h2(ctx, 4000)
    rows = r["detail"]["rows"]
    assert rows["gpqa_en"]["two_pass"] and rows["gpqa_de"]["two_pass"]
    c0, c1 = ctx.cell("K8", "gpqa_de"), ctx.cell("K8", "gpqa_de", pass_=1)
    est = np.mean([(c0[i].score + c1[i].score) / 2 for i in c0])
    assert rows["gpqa_de"]["ours"] == pytest.approx(est)


def test_h2_rows_named():
    w = world_only("H2", "C")
    w.set("K8", "ifbench", vend("ifbench", "K8") - 0.15)
    r = V.h2(w.ctx(), 4000)
    assert "ifbench" in r["detail"]["rows_abs_D_gt_5pp"]
    assert "ifbench" in r["detail"]["rows_upper_bound_below_minus_8pp"]
    assert "gpqa_de" not in r["detail"]["rows_abs_D_gt_5pp"]


def test_h2_secondary_aime_estimate_only():
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B1"]
    w.set("K8", "aime_en", 0.9)
    w.set("K8", "aime_de", 0.8)
    v = w.compute(now=NOW, exploratory=False)
    sec = v["verdicts"]["H2_detail"]["detail"]["secondary_with_aime"]
    assert set(sec["rows"]) == {"aime_en", "aime_de"} and "estimate" in sec and "p" not in sec


def test_normalise_plan_defaults():
    p = V.normalise_plan({"plan": "P10"})
    assert p["h2_rows"] == ["gpqa_en", "mmlu_en", "mmlu_de", "ifbench"]
    assert not p["forced_cells"]
    p = V.normalise_plan({"plan": "P3", "tier_b": ["B1", "B8", "B10"]})
    assert p["h2_secondary_rows"] == ["aime_en", "aime_de"]
    assert p["h7_rows"][-2:] == ["gpqa_en", "gpqa_de"] and p["b10"] and p["forced_cells"]
    assert V.normalise_plan({"plan": "P0"})["negfc_cells"]
    with pytest.raises(V.ConfigError):
        V.normalise_plan({"m": 7})


# ======================================================================
# Seeds and escalation
# ======================================================================

def test_h_results_do_not_depend_on_order(confirmed):
    w = all_confirmed_world()
    ctx = w.ctx()
    alone = V.h3(ctx, 10_000)
    assert alone["p"] == confirmed["verdicts"]["H3_detail"]["p"]
    _ = V.h6(ctx, 10_000)
    again = V.h3(ctx, 10_000)
    assert np.array_equal(alone["_samples"], again["_samples"])


def test_escalation_to_100k_near_the_threshold():
    # Find an H4 effect whose B = 10,000 p lands within a factor 2 of alpha/8 (step 1, others NOT RUN).
    thr = 0.05 / 8
    chosen = None
    for k in (0.74, 0.75, 0.76, 0.73, 0.77, 0.72, 0.78):
        w = World()
        w.set("K8", "ifbench", k)
        w.set("Q36-8", "ifbench", 0.66)
        p = V.h4(w.ctx(), 10_000)["p"]
        if thr / 2 <= p <= thr * 2:
            chosen = w
            break
    assert chosen is not None, "no candidate effect near the threshold"
    v = chosen.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H4_detail"]
    assert d["B"] == 100_000 and d["escalated_from_B"] == 10_000
    assert v["summary"]["holm"]["escalated"] == ["H4"]
    assert d["mc_se_p"] == pytest.approx(stats.mc_se(d["p"], 100_000))


def test_no_escalation_far_from_threshold(confirmed):
    assert confirmed["summary"]["holm"]["escalated"] == []
    assert all(confirmed["verdicts"][f"{h}_detail"]["B"] in (0, 10_000) for h in FAMILY)


# ======================================================================
# Files: rescore refusal, byte-identical re-runs, the CLI
# ======================================================================

def test_refuses_without_matching_rescore(tmp_path):
    w = all_confirmed_world()
    res = w.write_results(tmp_path / "a" / "results", rescore=False)
    with pytest.raises(V.RefuseError, match="rescore_mini"):
        V.compute(res, w.plan, now=NOW, exploratory=False)

    def tamper(entries):
        entries[0]["sha256_mini"] = "0" * 64

    res = w.write_results(tmp_path / "b" / "results", tamper=tamper)
    with pytest.raises(V.RefuseError, match="mini sha256 differs"):
        V.compute(res, w.plan, now=NOW, exploratory=False)

    def drop(entries):
        entries.pop()

    res = w.write_results(tmp_path / "c" / "results", tamper=drop)
    with pytest.raises(V.RefuseError, match="not in rescore_mini"):
        V.compute(res, w.plan, now=NOW, exploratory=False)

    def mbp_mismatch(entries):
        e = next(e for e in entries if not e["mini_only"])
        e["sha256_mbp"] = "1" * 64

    res = w.write_results(tmp_path / "d" / "results", tamper=mbp_mismatch)
    with pytest.raises(V.RefuseError, match="mbp sha256 differs"):
        V.compute(res, w.plan, now=NOW, exploratory=False)


def test_file_round_trip_matches_in_memory(tmp_path, confirmed):
    w = all_confirmed_world()
    res = w.write_results(tmp_path / "results")
    v = V.compute(res, w.plan, now=NOW)
    assert words(v) == words(confirmed)
    assert v["verdicts"]["H3_detail"]["p"] == confirmed["verdicts"]["H3_detail"]["p"]
    ins = v["manifests"]["inputs"]
    assert any(k.startswith("scores/K8/") for k in ins) and "bench/speed_20261005T000000Z.jsonl" in ins
    assert "bench/speed_desc_20261005T000000Z.jsonl" in ins
    assert v["manifests"]["rescore"]["n_files"] == len([k for k in ins if k.startswith("scores/")])


def test_rerun_is_byte_identical(tmp_path):
    w = all_confirmed_world()
    a = V.dumps(w.compute(now=NOW))
    b = V.dumps(all_confirmed_world().compute(now=NOW))
    assert a == b
    paths = []
    for sub in ("x", "y"):
        res = w.write_results(tmp_path / sub / "results")
        v = V.compute(res, w.plan, now=NOW)
        jp, mp = V.write(res, v)
        paths.append((jp, mp))
    assert paths[0][0].name == "verdicts_20261010T120000Z.json"
    assert paths[0][0].read_bytes() == paths[1][0].read_bytes()
    assert paths[0][1].read_bytes() == paths[1][1].read_bytes()
    with pytest.raises(FileExistsError):
        V.write(paths[0][0].parent, V.compute(paths[0][0].parent, w.plan, now=NOW))
    json.loads(paths[0][0].read_text(encoding="utf-8"))          # valid JSON, no NaN


def test_markdown_table_lists_every_hypothesis(confirmed):
    md = V.verdict_table_md(confirmed)
    for h in FAMILY:
        assert f"| {h} | " in md
    assert "D1 (outside Holm)" in md


def test_cli_writes_once(tmp_path, monkeypatch):
    w = all_confirmed_world()
    res = w.write_results(tmp_path / "results")
    (res / "plan_fixed_20261006T000000Z.json").write_text(json.dumps(w.plan), encoding="utf-8")
    calls = []
    guard = types.ModuleType("runner.guard")
    guard.require_identity = lambda: calls.append("identity")
    pkg = types.ModuleType("runner")
    pkg.guard = guard
    monkeypatch.setitem(sys.modules, "runner", pkg)
    monkeypatch.setitem(sys.modules, "runner.guard", guard)
    assert V.main(["--results", str(res)]) == 0
    assert calls == ["identity"]
    out = sorted(res.glob("verdicts_*.json"))
    assert len(out) == 1
    v = json.loads(out[0].read_text(encoding="utf-8"))
    assert v["manifests"]["plan_file"]["path"] == "plan_fixed_20261006T000000Z.json"


def test_scores_duplicates_and_headers():
    r = {"key": {"arm": "K8", "task": "ifbench", "effort": "high", "item": "x", "pass": 0},
         "correct": True, "correct_loose": False, "truncated": False}
    r2 = dict(r, correct_loose=True)
    s = V.Scores([{"type": "header"}, r, r2])
    assert s.duplicates == 1
    assert s.cell("K8", "ifbench")["x"].score == 0.0              # first record counts; loose is used
    with pytest.raises(V.ConfigError):
        V.Scores([{"correct": True}])


# ======================================================================
# Integration contracts with the sibling producers
# ======================================================================

def test_incomplete_bench_file_is_not_run(tmp_path):
    w = all_confirmed_world()
    res = w.write_results(tmp_path / "results")
    p = res / "bench" / "speed_20261005T000000Z.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()[:-1]          # drop the end record
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    v = V.compute(res, w.plan, now=NOW, exploratory=False)
    d = v["verdicts"]["H1_detail"]
    assert d["verdict"] == V.NOT_RUN and "incomplete" in d["reason"]
    assert "bench/speed_20261005T000000Z.jsonl" not in v["manifests"]["inputs"]
    # an older complete file is used instead of a newer incomplete one
    (res / "bench" / "speed_20261004T000000Z.jsonl").write_text(
        "\n".join([json.dumps({"type": "header"})] + [json.dumps(r) for r in w.speed]
                  + [json.dumps({"type": "end", "complete": True, "summary": {}})]) + "\n", encoding="utf-8")
    v = V.compute(res, w.plan, now=NOW, exploratory=False)
    assert v["verdicts"]["H1_detail"]["verdict"] == V.CONFIRMED
    assert "bench/speed_20261004T000000Z.jsonl" in v["manifests"]["inputs"]


def _plan_fixed(plan="P0", h2_arm="K8", peers=("G8", "Q36-8"), tier_b=()):
    """A dict shaped like runner/plan_fix.py's plan_fixed output (the keys verdicts.py reads)."""
    queue = [{"arm": "K8", "task": t, "effort": "high", "pass": 0, "n": n, "tier": "A"}
             for t, n in (("gpqa_en", 198), ("gpqa_de", 198), ("mmlu_en", 588), ("mmlu_de", 588),
                          ("ifbench", 300), ("rgb_cb", 400))]
    for a in peers:
        queue += [{"arm": a, "task": t, "effort": "high", "pass": 0, "n": 588, "tier": "A"} for t in ("mmlu_en", "mmlu_de")]
    queue.append({"arm": "bench", "task": "ladder", "effort": None, "pass": 0, "n": 0, "tier": "B"})
    return {"status": "FIXED", "plan": plan, "n_M": 588, "forced_cells": plan in V.FORCED_PLANS,
            "neg_fc_cells": plan == "P0", "tier_b": list(tier_b), "queue": queue,
            "rows": {"H2": ["gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench"], "H2_arm": h2_arm,
                     "H2_secondary_aime": ["aime_en", "aime_de"] if "B1" in tier_b else [],
                     "H7": ["mmlu_en", "mmlu_de", "ifbench", "rgb_cb"] if h2_arm == "K8" else [],
                     "H7_status": "runs" if h2_arm == "K8" else "NOT RUN (K8 cannot run)"}}


def test_normalise_plan_reads_plan_fix_output():
    p = V.normalise_plan(_plan_fixed(tier_b=("B1",)))
    assert p["h2_rows"] == ["gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench"]
    assert p["h2_secondary_rows"] == ["aime_en", "aime_de"] and p["kolibri_primary"] == "K8"
    assert p["h7_rows"] == ["mmlu_en", "mmlu_de", "ifbench", "rgb_cb"]
    assert p["peers"] == ["G8", "Q36-8"] and p["negfc_cells"] and p["forced_cells"]
    assert {"arm": "K8", "task": "ifbench", "effort": "high", "pass": 0, "n": 300} in p["cells"]
    assert all(c["arm"] != "bench" for c in p["cells"])
    p = V.normalise_plan(_plan_fixed(peers=("G8",)))
    assert p["peers"] == ["G8"]                                     # Qwen3.6 dropped: no Tier-A cells
    p = V.normalise_plan(_plan_fixed(h2_arm="K4"))
    assert p["kolibri_primary"] == "K4"


def test_plan_fix_shaped_plan_end_to_end():
    w = all_confirmed_world()
    plan = _plan_fixed(peers=("G8",))
    plan.update({"gpqa_en_excluded": [GPQA_EXCLUDED], "post_strat_counts": w.plan["post_strat_counts"]})
    w.plan = plan
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H4_detail"]["verdict"] == V.NOT_RUN       # Qwen3.6 dropped
    assert v["verdicts"]["H3_detail"]["detail"]["peers"] == ["G8"]
    assert v["config"]["peers"] == ["G8"]


def test_gpqa_primary_from_the_manifest_in_primary_flags(tmp_path):
    w = World()
    w.set("K8", "gpqa_en", 0.5)
    w.plan.pop("gpqa_en_excluded")
    mdir = tmp_path / "manifests"
    mdir.mkdir()
    items = [{"id": f"gq_en_{j:03d}", "item_sha256": "0" * 64, "prompt_sha256": "0" * 64, "in_primary": j != 42}
             for j in range(198)]
    (mdir / "gpqa_diamond_en.json").write_text(json.dumps({"items": items}), encoding="utf-8")
    w.plan["manifests_dir"] = str(mdir)
    ctx = w.ctx()
    prim = ctx.primary_items("gpqa_en", ctx.cell("K8", "gpqa_en"))
    assert len(prim) == 197 and "gq_en_042" not in prim


# ======================================================================
# Review fixes 2026-10-03 (verdict-correctness lens)
# ======================================================================

def test_h2_b10_planned_but_pass1_absent_is_single_pass_not_not_run():
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B10"]
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H2_detail"]
    assert d["verdict"] != V.NOT_RUN
    row = d["detail"]["rows"]["gpqa_en"]
    assert row["two_pass"] is False and "single pass" in row["two_pass_note"]


def test_h2_b10_two_pass_per_row_and_truncation_over_both_passes():
    w = World()
    for row in ("gpqa_en", "gpqa_de"):
        w.set("K8", row, 0.80)
    w.set("K8", "gpqa_en", 0.86, pass_=1)              # pass 1 only for EN
    w.trunc[("K8", "gpqa_en")] = set(range(10))      # 10 items truncated in pass 0 (pattern applies to both passes)
    w.plan.update({"tier_b": ["B10"], "h2_rows": ["gpqa_en", "gpqa_de"]})
    r = V.h2(w.ctx(), 2000)
    rows = r["detail"]["rows"]
    assert rows["gpqa_en"]["two_pass"] and not rows["gpqa_de"]["two_pass"]
    assert rows["gpqa_en"]["D_excluding_truncated"] is not None
    assert rows["gpqa_en"]["truncation_rate"] == pytest.approx(10 / 197, abs=0.01)


def test_h7_b8_rows_that_did_not_run_are_dropped_and_recorded():
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B8"]                          # planned, K4 GPQA never ran
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H7_detail"]
    assert d["verdict"] != V.NOT_RUN
    assert set(d["detail"]["rows_dropped"]) == {"gpqa_en", "gpqa_de"}
    assert set(d["detail"]["rows"]) == {"mmlu_en", "mmlu_de", "ifbench", "rgb_cb"}


def test_h7_p10_with_b8_has_no_gpqa_de_row():
    assert V.normalise_plan({"plan": "P10", "tier_b": ["B8"]})["h7_rows"][-1] == "gpqa_en"
    w = all_confirmed_world(n_M=154)
    w.plan.update({"plan": "P10", "n_M": 154, "tier_b": ["B8"]})
    del w.rates[("K8", "gpqa_de", "high", 0)]
    w.set("K4", "gpqa_en", vend("gpqa_en", "K4"))
    w.set("K4", "gpqa_de", vend("gpqa_de", "K4"))
    v = w.compute(now=NOW, exploratory=False)
    d = v["verdicts"]["H7_detail"]
    assert d["verdict"] != V.NOT_RUN and "gpqa_en" in d["detail"]["rows"] and "gpqa_de" not in d["detail"]["rows"]


def test_h7_missing_tier_a_row_is_still_not_run():
    w = all_confirmed_world()
    del w.rates[("K4", "rgb_cb", "high", 0)]
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H7_detail"]["verdict"] == V.NOT_RUN


@pytest.mark.parametrize("arm,task,n", [("K8", "rgb_cb", 400), ("K4", "ifbench", 300)])
def test_truncation_sensitivity_not_computable_is_not_headline_eligible(arm, task, n):
    w = all_confirmed_world()
    if arm == "K4":
        w.rates[("K4", "ifbench", "high", 0)] = 0.78
    w.trunc[(arm, task)] = set(range(n))
    v = w.compute(now=NOW, exploratory=False)
    h = "H6" if task == "rgb_cb" else "H7"
    d = v["verdicts"][f"{h}_detail"]
    assert d["truncation_sensitivity"]["status"] == V.NOT_RUN
    assert d["headline_eligible"] is False and d["headline_verdict"] == V.TRUNCATION_SENSITIVE


def test_h2_excluding_truncated_with_an_emptied_category_does_not_crash():
    w = all_confirmed_world(n_M=154)
    w.plan["n_M"] = 154
    w.trunc[("K8", "mmlu_en")] = set(range(11))       # every item of category 0
    v = w.compute(now=NOW, exploratory=False)
    row = v["verdicts"]["H2_detail"]["detail"]["rows"]["mmlu_en"]
    assert row["D_excluding_truncated"] is None and "cat00" in row["D_excluding_truncated_reason"]
    assert v["verdicts"]["H2_detail"]["verdict"] != V.NOT_RUN


@pytest.mark.parametrize("k", [284, 348])             # exactly -8.0 pp and +8.0 pp on RGB CB (vendor 79.0)
def test_e8_exact_tie_at_the_threshold_is_not_flagged(k):
    w = all_confirmed_world()
    w.set("G8", "rgb_cb", k / 400)
    flags, _ = V.e8_flags(w.ctx(), ["G8"], ("rgb_cb",))
    assert flags == []
    w.set("G8", "rgb_cb", (k - 1) / 400 if k < 300 else (k + 1) / 400)   # one item beyond: flagged
    flags, _ = V.e8_flags(w.ctx(), ["G8"], ("rgb_cb",))
    assert [f["arm"] for f in flags] == ["G8"]


def test_protocol_control_and_q2_without_any_peer():
    w = all_confirmed_world()
    w.plan["peers"] = []
    v = w.compute(now=NOW, exploratory=False)
    pc = v["verdicts"]["H2_detail"]["detail"]["protocol_control"]
    assert pc["status"] == V.NOT_RUN
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "Q2_peer_control_not_run" and q2["headline_eligible"] is False


def test_q2_with_the_protocol_control_not_run_uses_no_preregistered_sentence():
    from analysis import tables

    w = all_confirmed_world()
    del w.rates[("G8", "ifbench", "high", 0)]
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H2_detail"]["verdict"] == V.CONFIRMED
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "Q2_peer_control_not_run" and not q2["headline_eligible"]
    assert tables.plain_answer_sentences(v)["Q2"][0].startswith("[No pre-registered Q2 sentence applies")


def test_plain_sentences_follow_p10_k4_primary_and_the_margin():
    from analysis import tables

    w = all_confirmed_world()
    v = w.compute(now=NOW, exploratory=False)
    s = tables.plain_answer_sentences(v)["Q2"][0]
    assert s.startswith(V.PLAIN_ANSWERS["Q2_confirmed"].split("(D̄")[0])     # defaults: verbatim
    w = all_confirmed_world(n_M=154)
    w.plan.update({"plan": "P10", "n_M": 154})
    del w.rates[("K8", "gpqa_de", "high", 0)]
    s = tables.plain_answer_sentences(w.compute(now=NOW, exploratory=False))["Q2"][0]
    assert "the four public rows" in s and "three of the four rows" in s
    m = V.load_margins()
    m["H2"] = -0.03
    s = tables.plain_answer_sentences(all_confirmed_world().compute(now=NOW, exploratory=False, margins=m))["Q2"][0]
    assert "no more than 3 pp below" in s
    w = all_confirmed_world()
    for k in list(w.rates):
        if k[0] in ("K8", "K4"):
            del w.rates[k]
    for row in ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench", "rgb_cb"):
        w.set("K4", row, vend(row, "K4"))
    w.plan["kolibri_primary"] = "K4"
    out = tables.plain_answer_sentences(w.compute(now=NOW, exploratory=False))
    assert "at MLX 4-bit" in out["Q2"][0] and "8-bit" not in out["Q2"][0]
    assert out["Q3"][0].startswith("In English and German on MMLU-ProX-Lite, Kolibri 4-bit scores")


def test_h1_needs_ten_blocks_and_d1_three_reps():
    w = World()
    w.speed = [r for r in speed_blocks(0.95) if r.get("block") is None or r["block"] < 3]
    assert V.h1.__name__ and w.compute(now=NOW, exploratory=False)["verdicts"]["H1_detail"]["verdict"] == V.NOT_RUN
    w = World()
    w.fit = fit_records([45.0, 45.0])
    r = V.d1(w.ctx())
    assert r["verdict"] == V.NOT_RUN and "needs 3" in r["reason"]
