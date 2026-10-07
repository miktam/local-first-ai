"""The H2 tripwire T-G3 (exp_037; analysis/verdicts.py, Tier 1) and its power tables (analysis/power.py).

The rule: UB_K and UB_DiD are the H2_tripwire_q (0.95) quantiles of H2's D-bar replicates and of the protocol
control's DiD replicates; TRIPPED iff UB_K < -10 pp and (the protocol control is NOT RUN or UB_DiD < -10 pp); NOT RUN
iff H2 is NOT RUN. Tested here: synthetic replicates on both sides of the bound (UB = -0.0999 and -0.1001), each with
the DiD leg tripped, not tripped and NOT RUN; the inputs are H2's own replicates and no random number is drawn; the
labels on H2-H8 and E1-E5, E9, E10, E12 (H1, H5, D1 untouched); the caveat when not tripped; the precedence of the
tripped Q2 rows; the stripping of "_did_samples"; the seeds; the power tables to 3 decimals.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from analysis import power as P
from analysis import stats
from analysis import tables as T
from analysis import verdicts as V
from test_analysis_world import all_confirmed_world, vend

NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
EXP_DIR = Path(__file__).resolve().parents[1]
MARGINS = V.load_margins()
H2_ROWS = ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench")
SHARED = ("mmlu_en", "mmlu_de", "ifbench")
LABELLED_H = ("H2", "H3", "H4", "H6", "H7", "H8")
LABELLED_E = ("E1", "E2", "E3", "E4", "E5", "E9", "E10", "E12")


def reps(ub: float, n: int = 10_001, q: float = 0.95) -> np.ndarray:
    """Replicates whose q quantile (numpy's linear method) is ub: an even grid with ub at index q (n - 1)."""
    step = 1e-5
    k = round(q * (n - 1))
    out = ub + step * (np.arange(n) - k)
    assert np.quantile(out, q) == pytest.approx(ub, abs=1e-15)
    return out


def h2_result(ub_k: float, did: str | float = "nr", status: str = "RUN") -> dict:
    """A synthetic H2 result: D-bar replicates with UB ub_k; the protocol control NOT RUN ("nr") or with DiD UB did."""
    if status == V.NOT_RUN:
        return V.not_run("synthetic")
    pc = ({"status": V.NOT_RUN, "reason": "no MoE peer left (peer check)"} if did == "nr"
          else {"DiD": float(did), "_did_samples": reps(float(did))})
    return {"status": "RUN", "B": 10_001, "_samples": reps(ub_k), "detail": {"arm": "K8", "protocol_control": pc}}


# ======================================================================
# The rule on synthetic replicates
# ======================================================================

@pytest.mark.parametrize("ub_k", [-0.0999, -0.1001])
@pytest.mark.parametrize("did", ["tripped", "not tripped", "NOT_RUN"])
def test_rule_on_both_sides_of_the_bound(ub_k, did):
    d = {"tripped": -0.1001, "not tripped": -0.0999, "NOT_RUN": "nr"}[did]
    tw = V.tripwire(h2_result(ub_k, d), MARGINS)
    want = V.TRIPPED if (ub_k < -0.10 and did in ("tripped", "NOT_RUN")) else V.NOT_TRIPPED
    assert tw["status"] == want, tw
    assert tw["UB_K"] == pytest.approx(ub_k, abs=1e-12) and tw["K_beyond"] == (ub_k < -0.10)
    assert tw["arm"] == "K8" and tw["q"] == 0.95 and tw["id"] == "T-G3"
    assert tw["UB_K_threshold"] == -0.10 and tw["UB_DiD_threshold"] == -0.10
    if did == "NOT_RUN":
        assert tw["protocol_control"] == V.NOT_RUN and tw["UB_DiD"] is None and tw["DiD_beyond"] is None
        assert "no MoE peer" in tw["protocol_control_reason"]
    else:
        assert tw["protocol_control"] == "RUN"
        assert tw["UB_DiD"] == pytest.approx(d, abs=1e-12) and tw["DiD_beyond"] == (did == "tripped")


@pytest.mark.parametrize("ub", [-0.10, -0.10 - 1e-12, -0.10 + 1e-12])
def test_a_bound_at_the_threshold_does_not_trip(ub):
    """'<' is strict; a bound equal to -10 pp to floating-point error is not beyond it (the module's EPS)."""
    assert V.tripwire(h2_result(ub, ub), MARGINS)["status"] == V.NOT_TRIPPED
    assert V.tripwire(h2_result(ub), MARGINS)["status"] == V.NOT_TRIPPED
    assert V.tripwire(h2_result(ub - 1e-6), MARGINS)["status"] == V.TRIPPED


def test_the_bound_is_the_95_percent_quantile_of_the_replicates():
    gen = np.random.default_rng(7)
    s = gen.normal(-0.115, 0.01, size=10_000)
    did = gen.normal(-0.12, 0.02, size=10_000)
    tw = V.tripwire({"status": "RUN", "B": 10_000, "_samples": s,
                     "detail": {"arm": "K4", "protocol_control": {"_did_samples": did}}}, MARGINS)
    assert tw["UB_K"] == float(np.quantile(s, 0.95)) and tw["UB_DiD"] == float(np.quantile(did, 0.95))
    assert tw["arm"] == "K4" and tw["B"] == 10_000 and tw["n_replicates_K"] == tw["n_replicates_DiD"] == 10_000
    assert tw["status"] == (V.TRIPPED if tw["UB_K"] < -0.10 and tw["UB_DiD"] < -0.10 else V.NOT_TRIPPED)
    m = dict(MARGINS, H2_tripwire_q=0.5)                   # the quantile comes from margins.json
    assert V.tripwire({"status": "RUN", "_samples": s, "detail": {"protocol_control": {"_did_samples": did}}},
                      m)["UB_K"] == pytest.approx(float(np.median(s)))


def test_not_run_when_h2_is_not_run():
    for r in (V.not_run("plan STOP: budget"), None):
        tw = V.tripwire(r, MARGINS)
        assert tw["status"] == V.NOT_RUN and "H2 was not run" in tw["reason"]
        assert "UB_K" not in tw
    assert "plan STOP: budget" in V.tripwire(V.not_run("plan STOP: budget"), MARGINS)["reason"]


def test_missing_replicates_are_a_config_error():
    with pytest.raises(V.ConfigError, match="_samples"):
        V.tripwire({"status": "RUN", "detail": {"protocol_control": {}}}, MARGINS)
    with pytest.raises(V.ConfigError, match="_did_samples"):
        V.tripwire({"status": "RUN", "_samples": reps(-0.2), "detail": {"protocol_control": {"DiD": -0.2}}}, MARGINS)
    with pytest.raises(V.ConfigError, match="protocol_control"):
        V.tripwire({"status": "RUN", "_samples": reps(-0.2), "detail": {}}, MARGINS)


def test_margins_keys_and_version():
    assert MARGINS["version"] == "exp037-margins-1"
    assert (MARGINS["H2_tripwire_ub"], MARGINS["H2_tripwire_did_ub"], MARGINS["H2_tripwire_q"]) == (-0.10, -0.10, 0.95)
    assert MARGINS["H2_tripwire_ub"] == pytest.approx(2.5 * MARGINS["H2"])     # 2.5 x H2's -4 pp margin
    assert -MARGINS["H2_tripwire_ub"] > MARGINS["E8_flag"]                     # beyond E8's 8-pp protocol band
    assert "H2_tripwire" in MARGINS["notes"]


def test_margins_match_gate_rules_tripwire_constants():
    """gate/rules.py exports tripwire_constants (W6); the analysis margins must carry the same three values."""
    if not (EXP_DIR / "gate" / "rules.py").is_file():
        pytest.skip("gate/rules.py is not built yet")
    rules = importlib.import_module("gate.rules")
    tc = getattr(rules, "tripwire_constants")
    tc = tc() if callable(tc) else tc
    alias = {"H2_tripwire_ub": "ub", "H2_tripwire_did_ub": "did_ub", "H2_tripwire_q": "q"}
    for k, short in alias.items():
        got = tc[k] if k in tc else tc[short]
        assert float(got) == MARGINS[k], k


def test_agrees_with_the_gate_rules_predicate():
    """gate/rules.py's tripwire_rule (W6) states the same predicate; both agree on a grid around the bounds (status
    words mapped by name: verdicts.py writes "NOT TRIPPED" / "NOT RUN", in its own vocabulary). They differ only for a
    bound within EPS = 1e-9 below -10 pp, which verdicts.py, by its convention, does not count as beyond."""
    if not (EXP_DIR / "gate" / "rules.py").is_file():
        pytest.skip("gate/rules.py is not built yet")
    rules = importlib.import_module("gate.rules")
    if not hasattr(rules, "tripwire_rule"):
        pytest.skip("gate/rules.py has no tripwire_rule")
    word = {rules.TRIPPED: V.TRIPPED, rules.NOT_TRIPPED: V.NOT_TRIPPED, rules.NOT_RUN: V.NOT_RUN}
    grid = (-0.2, -0.1001, -0.10, -0.0999, -0.05, 0.03)
    for ub_k in grid:
        for did in ("nr",) + grid:
            tw = V.tripwire(h2_result(ub_k, did), MARGINS)
            want = rules.tripwire_rule(tw["UB_K"], tw["UB_DiD"], protocol_run=did != "nr")
            assert tw["status"] == word[want], (ub_k, did)
    assert word[rules.tripwire_rule(None, None, h2_run=False)] == V.tripwire(V.not_run("x"), MARGINS)["status"]


# ======================================================================
# Inputs: H2's own replicates; no random numbers drawn
# ======================================================================

def tripped_world(peers=True):
    """all_confirmed_world with K8 15 pp below its table on every H2 row (peers at theirs): UB_K, UB_DiD about -13."""
    w = all_confirmed_world()
    for row in H2_ROWS:
        w.set("K8", row, vend(row, "K8") - 0.15)
    if not peers:
        w.plan["peers"] = []
    return w


def test_inputs_are_h2s_replicates_and_no_number_is_drawn(monkeypatch):
    w = tripped_world()
    ctx = w.ctx()
    r = V.h2(ctx, 10_000)
    direct = V.tripwire(r, ctx.margins)
    assert direct["UB_K"] == float(np.quantile(r["_samples"], 0.95))
    assert direct["UB_DiD"] == float(np.quantile(r["detail"]["protocol_control"]["_did_samples"], 0.95))
    assert r["detail"]["protocol_control"]["DiD_ub95"] == direct["UB_DiD"]

    def no_draws(*a, **k):
        raise AssertionError("the tripwire drew a random number")

    monkeypatch.setattr(stats, "rng", no_draws)
    monkeypatch.setattr(np.random, "default_rng", no_draws)
    assert V.tripwire(r, ctx.margins) == direct
    monkeypatch.undo()
    v = w.compute(now=NOW, exploratory=False)
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["UB_K"] == direct["UB_K"] and tw["UB_DiD"] == direct["UB_DiD"] and tw["B"] == 10_000


def test_seeds_are_exp037():
    tw = V.tripwire(h2_result(-0.2, -0.2), MARGINS)
    for h in ("H2", "H2|protocol"):
        assert tw["seeds"][h] == int.from_bytes(hashlib.sha256(b"exp037|" + h.encode()).digest()[:8], "big")
    # the protocol control's DiD replicates come from rng("H2|protocol") and reproduce
    ctx = tripped_world().ctx()
    a = V.protocol_control(ctx, 2000)["_did_samples"]
    b = V.protocol_control(ctx, 2000)["_did_samples"]
    assert np.array_equal(a, b) and a.shape == (2000,)


# ======================================================================
# End to end: TRIPPED
# ======================================================================

@pytest.fixture(scope="module")
def tripped():
    return tripped_world().compute(now=NOW)


def test_tripped_labels_h2_to_h8_and_leaves_h1_h5_d1(tripped):
    v = tripped
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["status"] == V.TRIPPED and tw["protocol_control"] == "RUN"
    assert tw["UB_K"] < -0.10 and tw["UB_DiD"] < -0.10 and tw["arm"] == "K8"
    assert v["summary"]["tripwire"] == V.TRIPPED
    for h in LABELLED_H:
        d = v["verdicts"][f"{h}_detail"]
        assert d["verdict"] != V.NOT_RUN
        assert v["verdicts"][h] == d["verdict"] + ", implementation-uncertain (G3 blind spot)"
        assert d["label"] == v["verdicts"][h] and d["implementation_uncertain"] is True
        assert d["headline_eligible"] is False and "caveat" not in d
        assert v["summary"]["verdict_words"][h] == d["verdict"]          # the verdict word itself is unchanged
    for h in ("H1", "H5"):
        d = v["verdicts"][f"{h}_detail"]
        assert v["verdicts"][h] == V.CONFIRMED and d["headline_eligible"] and "implementation_uncertain" not in d
    assert v["verdicts"]["D1"] == V.CONFIRMED


def test_tripped_labels_the_computed_e_rows_only(tripped):
    from analysis import exploratory as X

    v = tripped
    suffix = X.LABEL + V.TRIPWIRE_LABEL_SUFFIX
    computed = [k for k in LABELLED_E if v["verdicts"][k].get("status") != "not computed"]
    assert set(computed) >= {"E1", "E2", "E4", "E9"}
    for k in LABELLED_E:
        e = v["verdicts"][k]
        if k in computed:
            assert e["label"] == suffix and e["implementation_uncertain"] and e["headline_eligible"] is False
        else:
            assert e["label"] == X.LABEL
    for k in ("E6", "E7", "E8", "E11", "C1"):
        assert v["verdicts"][k]["label"] == X.LABEL and "implementation_uncertain" not in v["verdicts"][k]


def test_tripping_changes_no_verdict_word_p_value_or_holm(monkeypatch, tripped):
    """H2 is computed as registered: with the tripwire forced NOT TRIPPED, every number and word is the same."""
    real = V.tripwire
    monkeypatch.setattr(V, "tripwire", lambda r, m: {**real(r, m), "status": V.NOT_TRIPPED})
    other = tripped_world().compute(now=NOW)
    assert other["summary"]["verdict_words"] == tripped["summary"]["verdict_words"]
    assert other["summary"]["holm"] == tripped["summary"]["holm"]
    for h in V.FAMILY:
        a, b = other["verdicts"][f"{h}_detail"], tripped["verdicts"][f"{h}_detail"]
        assert (a["p"], a["p_rev"], a.get("estimate"), a.get("ci95")) == (b["p"], b["p_rev"], b.get("estimate"),
                                                                          b.get("ci95"))
    assert other["verdicts"]["H2"] == tripped["verdicts"]["H2_detail"]["verdict"]


def test_tripped_q2_row_and_sentences(tripped):
    v = tripped
    pa = v["summary"]["plain_answers"]
    q2 = pa["Q2"]
    assert q2["id"] == "Q2_tripped" and q2["headline_eligible"] is False and q2["tripwire"] == V.TRIPPED
    assert "caveat" not in q2
    assert pa["Q3"]["implementation_uncertain"] == ["H3", "H4"] and pa["Q4"]["implementation_uncertain"] == ["H6"]
    assert pa["Q5"]["implementation_uncertain"] == ["H7", "H8"]
    s = T.plain_answer_sentences(v)
    h2 = v["verdicts"]["H2_detail"]
    x = f"{-100.0 * h2['estimate']:.1f}"
    lo, hi = h2["ci95"]
    assert s["Q2"] == [V.PLAIN_ANSWERS["Q2_tripped"].replace("<x>", x).replace(
        "[CI]", f"[{-100.0 * hi:.1f}, {-100.0 * lo:.1f}]")]
    assert "more than 10 pp beyond what the peers show" in s["Q2"][0]
    assert s["Q4"][-1] == "Implementation-uncertain (G3 blind spot): H6."
    assert "CONFIRMED, implementation-uncertain (G3 blind spot)" in s["Q3"][0]
    text = " ".join(" ".join(x) for x in s.values())
    assert not re.search(r"<[A-Za-z][^>]*>", text) and "[CI]" not in text
    assert V.TRIPWIRE_CAVEAT not in text


def test_did_samples_are_stripped_and_only_the_bound_is_written(tripped, tmp_path):
    v = tripped
    pc = v["verdicts"]["H2_detail"]["detail"]["protocol_control"]
    assert "_did_samples" not in pc and pc["DiD_ub95"] == v["verdicts"]["H2_tripwire"]["UB_DiD"]
    text = V.dumps(v)
    assert "_did_samples" not in text and "_samples" not in text
    jp, mp = V.write(tmp_path, v)
    assert "_did_samples" not in jp.read_text(encoding="utf-8")
    md = mp.read_text(encoding="utf-8")
    assert "H2 tripwire T-G3: TRIPPED (arm K8" in md and "implementation-uncertain (G3 blind spot)" in md
    assert "Implementation-uncertain (G3 blind spot): H6." in md


def test_tripped_with_the_protocol_control_not_run():
    v = tripped_world(peers=False).compute(now=NOW, exploratory=False)
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["status"] == V.TRIPPED and tw["protocol_control"] == V.NOT_RUN and tw["UB_DiD"] is None
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "Q2_tripped_peer_control_not_run" and not q2["headline_eligible"]
    s = T.plain_answer_sentences(v)["Q2"][0]
    assert "without the peer control we cannot separate it from our protocol either" in s
    assert "more than 10 pp beyond" not in s
    for h in ("H3", "H4", "H6"):                               # no peer: NOT RUN, so no label
        assert v["verdicts"][h] == V.NOT_RUN
    for h in ("H2", "H7", "H8"):
        assert v["verdicts"][h].endswith(V.TRIPWIRE_LABEL_SUFFIX)


# ======================================================================
# End to end: NOT TRIPPED, and NOT RUN
# ======================================================================

def test_not_tripped_every_quality_verdict_carries_the_caveat():
    v = all_confirmed_world().compute(now=NOW)
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["status"] == V.NOT_TRIPPED and tw["UB_K"] > -0.10 and tw["UB_DiD"] > -0.10
    for h in LABELLED_H:
        d = v["verdicts"][f"{h}_detail"]
        assert d["caveat"] == V.TRIPWIRE_CAVEAT and v["verdicts"][h] == V.CONFIRMED and d["headline_eligible"]
    for h in ("H1", "H5"):
        assert "caveat" not in v["verdicts"][f"{h}_detail"]
    assert "caveat" not in v["verdicts"]["D1_detail"]
    pa = v["summary"]["plain_answers"]
    for q in ("Q2", "Q3", "Q4", "Q5"):
        assert pa[q]["caveat"] == V.TRIPWIRE_CAVEAT
    assert pa["Q2"]["id"] == "Q2_confirmed" and pa["Q2"]["tripwire"] == V.NOT_TRIPPED
    s = T.plain_answer_sentences(v)
    for q in ("Q2", "Q3", "Q4", "Q5"):
        assert s[q][-1] == V.TRIPWIRE_CAVEAT and s[q][0] != V.TRIPWIRE_CAVEAT
    assert all(V.TRIPWIRE_CAVEAT not in x for x in s["Q1"])
    assert "Every Kolibri quality verdict (H2, H3, H4, H6, H7, H8) carries the caveat" in V.verdict_table_md(v)


def test_not_tripped_when_only_the_kolibri_leg_is_beyond():
    """Kolibri and the peers 15 pp below their tables: UB_K < -10 pp but DiD about 0, so no trip (protocol drift that
    hits every model cannot fire it), and Q2 is the peers-outside row with the caveat."""
    w = tripped_world()
    for arm in ("G8", "Q36-8"):
        for row in SHARED:
            w.set(arm, row, vend(row, arm) - 0.15)
    v = w.compute(now=NOW, exploratory=False)
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["UB_K"] < -0.10 and tw["UB_DiD"] > -0.10 and tw["status"] == V.NOT_TRIPPED
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "Q2_peers_outside" and q2["caveat"] == V.TRIPWIRE_CAVEAT
    assert not v["verdicts"]["H2"].endswith(V.TRIPWIRE_LABEL_SUFFIX)


def test_not_run_with_h2_not_run_publishes_no_sentence_for_it():
    w = all_confirmed_world()
    del w.rates[("K8", "gpqa_de", "high", 0)]
    w.plan["h2_rows"] = list(H2_ROWS)
    v = w.compute(now=NOW, exploratory=False)
    assert v["verdicts"]["H2"] == V.NOT_RUN
    tw = v["verdicts"]["H2_tripwire"]
    assert tw["status"] == V.NOT_RUN and "H2 was not run" in tw["reason"]
    q2 = v["summary"]["plain_answers"]["Q2"]
    assert q2["id"] == "not_run" and "caveat" not in q2
    assert "no sentence is published for it" in V.verdict_table_md(v)
    for h in ("H3", "H4", "H6", "H7", "H8"):                   # not tripped: the caveat stays on what ran
        assert v["verdicts"][f"{h}_detail"]["caveat"] == V.TRIPWIRE_CAVEAT


# ======================================================================
# Plain-answer precedence (state level)
# ======================================================================

def _hyp(word="REFUTED", estimate=-0.15, pc=None):
    pc = {"peers_within_band": True, "peers_mean_D_shared": 0.0} if pc is None else pc
    hyp = {h: {"verdict": V.NOT_RUN, "label": V.NOT_RUN, "reason": "synthetic"} for h in V.FAMILY}
    hyp["H2"] = {"verdict": word, "label": word, "estimate": estimate, "headline_eligible": True,
                 "detail": {"protocol_control": pc}}
    return hyp


def _tw(status, pc_run=True):
    ub = -0.2 if status == V.TRIPPED else -0.05
    tw = V.tripwire(h2_result(ub, -0.2 if pc_run else "nr"), MARGINS)
    assert tw["status"] == status
    return tw


@pytest.mark.parametrize("pc", [
    {"peers_within_band": True, "peers_mean_D_shared": 0.0},         # would be Q2_refuted / confirmed / inconclusive
    {"peers_within_band": False, "peers_mean_D_shared": -0.05},      # would be Q2_peers_outside
    {"status": V.NOT_RUN, "reason": "no MoE peer left"},            # would be Q2_peer_control_not_run
])
@pytest.mark.parametrize("word", [V.CONFIRMED, V.REFUTED, V.INCONCLUSIVE])
def test_the_tripped_row_takes_precedence_over_every_other_q2_row(pc, word):
    pc_run = pc.get("status") != V.NOT_RUN
    q2 = V.plain_answer_states(_hyp(word, pc=pc), {"verdict": V.NOT_RUN}, {}, tripwire=_tw(V.TRIPPED, pc_run))["Q2"]
    assert q2["id"] == ("Q2_tripped" if pc_run else "Q2_tripped_peer_control_not_run")
    assert q2["state"] == word and q2["headline_eligible"] is False and "caveat" not in q2
    untripped = V.plain_answer_states(_hyp(word, pc=pc), {"verdict": V.NOT_RUN}, {}, tripwire=_tw(V.NOT_TRIPPED))
    assert untripped["Q2"]["id"] not in V.Q2_TRIPPED_IDS and untripped["Q2"]["caveat"] == V.TRIPWIRE_CAVEAT


def test_without_a_tripwire_record_the_states_are_exp036s():
    q2 = V.plain_answer_states(_hyp(), {"verdict": V.NOT_RUN}, {})["Q2"]
    assert q2["id"] == "Q2_refuted" and "tripwire" not in q2 and "caveat" not in q2


@pytest.mark.parametrize("est,want", [(-0.07, True), (-0.10, True), (-0.12, False)])
def test_refuted_within_the_tripwire_band_is_flagged(est, want):
    """An H2 REFUTED between the H2 margin and -10 pp cannot be told apart from a shared misreading: flagged, with
    the caveat after its sentence."""
    q2 = V.plain_answer_states(_hyp(V.REFUTED, est), {"verdict": V.NOT_RUN}, {}, tripwire=_tw(V.NOT_TRIPPED))["Q2"]
    assert q2["id"] == "Q2_refuted" and q2["refuted_within_tripwire_band"] is want
    assert q2["caveat"] == V.TRIPWIRE_CAVEAT
    q2c = V.plain_answer_states(_hyp(V.CONFIRMED, -0.01), {"verdict": V.NOT_RUN}, {}, tripwire=_tw(V.NOT_TRIPPED))["Q2"]
    assert q2c["refuted_within_tripwire_band"] is False


def test_wording_is_verbatim():
    assert V.PLAIN_ANSWERS["Q2_tripped"] == (
        "Through our port, Kolibri scores <x> pp [CI] below its own table on the public rows, more than 10 pp beyond "
        "what the peers show against theirs. A misreading of the architecture shared by our port and our reference "
        "would look exactly like this. No local test can rule it out, so we make no claim that Kolibri itself falls "
        "short.")
    assert V.PLAIN_ANSWERS["Q2_tripped_peer_control_not_run"] == (
        "Through our port, Kolibri scores <x> pp [CI] below its own table on the public rows. A misreading of the "
        "architecture shared by our port and our reference would look exactly like this, and without the peer control "
        "we cannot separate it from our protocol either. No local test can rule it out, so we make no claim that "
        "Kolibri itself falls short.")
    assert V.TRIPWIRE_CAVEAT == (
        "A misreading shared by port and reference that moves Kolibri's measured shortfall on these rows by up to "
        "about 12–15 pp is more likely missed than caught locally; only one beyond about 15 pp is reliably caught.")
    assert V.TRIPWIRE_LABEL_SUFFIX == ", implementation-uncertain (G3 blind spot)"
    assert V.TRIPWIRE_QUALITY_H == LABELLED_H and V.TRIPWIRE_LABELLED_E == LABELLED_E


# ======================================================================
# Power tables (analysis/power.py) to 3 decimals
# ======================================================================

# The tripwire table (D-bar leg; normal approximation): column -> (SE pp, {true D pp: power}, 50 % point pp).
TRIPWIRE_TABLE = {
    "n_M 588": (1.09, {-4: 0.000, -8: 0.000, -10: 0.050, -12.5: 0.742, -15: 0.998}, -11.793),
    "n_M 196 (P7-P8)": (1.31, {-4: 0.000, -8: 0.001, -10: 0.050, -12.5: 0.604, -15: 0.985}, -12.155),
    "n_M 154 (P9)": (1.39, {-4: 0.000, -8: 0.001, -10: 0.050, -12.5: 0.561, -15: 0.975}, -12.286),
    "P10": (1.57, {-4: 0.000, -8: 0.002, -10: 0.050, -12.5: 0.479, -15: 0.938}, -12.582),
}
# As the pre-registration prints them (its own rounding): -8 pp to 4 decimals, the rest to 2 or 3.
TRIPWIRE_PRINTED = {
    "n_M 588": {-8: (0.0003, 4), -12.5: (0.74, 2), -15: (0.998, 3)},
    "n_M 196 (P7-P8)": {-8: (0.0008, 4), -12.5: (0.60, 2), -15: (0.985, 3)},
    "n_M 154 (P9)": {-8: (0.0010, 4), -12.5: (0.56, 2), -15: (0.97, 2)},
    "P10": {-8: (0.0018, 4), -12.5: (0.48, 2), -15: (0.94, 2)},
}


HALF_PRINTED = {"n_M 588": -11.8, "n_M 196 (P7-P8)": -12.2, "n_M 154 (P9)": -12.3, "P10": -12.6}


def test_tripwire_power_table_to_3_decimals():
    t = P.tripwire_power_table()
    assert list(t) == list(TRIPWIRE_TABLE)
    for col, (se, powers, half) in TRIPWIRE_TABLE.items():
        assert t[col]["se_pp"] == se
        for d, want in powers.items():
            assert round(t[col]["power"][f"{d:g}"], 3) == want, (col, d)
        for d, (want, nd) in TRIPWIRE_PRINTED[col].items():
            assert round(t[col]["power"][f"{d:g}"], nd) == want, (col, d)
        assert round(t[col]["half_point_pp"], 3) == half
        assert round(t[col]["half_point_pp"], 1) == HALF_PRINTED[col]
        assert t[col]["power"]["-10"] == pytest.approx(0.05, abs=1e-12)      # at the bound: 1 - q


def test_tripwire_columns_are_the_registered_h2_ses():
    assert [P.tripwire_se(n, four) for _, n, four in P.TRIPWIRE_COLUMNS] == pytest.approx([0.0109, 0.0131, 0.0139,
                                                                                           0.0157])
    assert P.tripwire_power(-0.10, 0.0109) == pytest.approx(0.05)
    assert P.tripwire_power(-0.20, 0.0109) > 0.999 and P.tripwire_power(-0.04, 0.0157) < 1e-6
    assert P.tripwire_margins() == (-0.10, 0.95)


GREEDY = {"0.05 %": (0.000, "0.0001", 4), "0.1 %": (0.003, "0.003", 3), "0.32 %": (0.186, "0.19", 2)}


def test_greedy_false_fail_table_to_3_decimals():
    g = P.greedy_table()
    assert g["n_decisive"] == 941 and g["fail_at_misses"] == 5 and g["top1_min"] == 0.995
    for k, (want3, printed, nd) in GREEDY.items():
        p = g["rows"][k]["p_false_fail"]
        assert round(p, 3) == want3 and f"{round(p, nd):.{nd}f}" == printed, (k, p)
    assert round(g["zero_miss_upper_bound_95"], 5) == 0.00318                  # 95 % bound from 0 / 941
    assert round(g["rows"]["0.3179 %"]["p_false_fail"], 3) == 0.183           # at the exact bound
    assert round(g["rows"]["0.32 %"]["joint_with_R_greedy_max"], 3) == 0.338    # "<= 0.34"
    # exact: 4 misses in 941 pass (99.575 %), 5 fail (99.469 %)
    assert P.greedy_fail_count(941, 0.995) == 5 and P.greedy_fail_count(1000, 0.995) == 6
    assert P.binom_sf(5, 941, 0.001) == pytest.approx(1 - sum(
        math.comb(941, j) * 0.001 ** j * 0.999 ** (941 - j) for j in range(5)))


def test_greedy_rule_constant_matches_the_gate_thresholds():
    """The table's 99.5 % is the registered G5 greedy rule (gate/thresholds.json, kept value for value)."""
    def find(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from ([v] if k == "greedy_decisive_top1_min" else find(v))
        elif isinstance(x, list):
            for v in x:
                yield from find(v)

    found = list(find(json.loads((EXP_DIR / "gate" / "thresholds.json").read_text(encoding="utf-8"))))
    assert found and all(v == P.GREEDY_TOP1_MIN for v in found)


def test_power_tables_carry_the_exp037_tables(monkeypatch):
    monkeypatch.setattr(P, "h8_power", lambda a, alpha: 0.0)                  # skip the slow H8 simulation
    t = P.power_tables()
    assert t["H2_tripwire"] == P.tripwire_power_table() and t["G5_greedy_false_fail"] == P.greedy_table()
