"""analysis/exploratory.py (E1-E12, C1) and analysis/tables.py (Tier 2).

A synthetic run with every Tier-A and Tier-B cell present; each exploratory
analysis is computed, labelled exploratory, and its prediction check comes out
as constructed. tables.py renders the key-numbers table and fills every
plain-answer placeholder.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from analysis import exploratory as X
from analysis import tables as T
from analysis import verdicts as V
from test_analysis_world import World, all_confirmed_world, vend

NOW = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
GIB = 2 ** 30


def rich_world(k8_abstains=True):
    w = all_confirmed_world()
    w.plan["tier_b"] = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9"]
    w.plan["forced_cells"] = True
    w.plan["negfc_cells"] = True
    if k8_abstains:
        w.categories_rgb[("K8", "rgb_cb")] = 1          # every non-correct K8 item is an abstention
    else:
        w.categories_rgb.pop(("K8", "rgb_cb"), None)
    for arm in ("K8", "G8", "Q36-8"):
        w.set(arm, "rgb_forced", 0.45 if arm == "K8" else 0.80)
        w.set(arm, "rgb_neg", vend("rgb_neg", arm))
        w.set(arm, "rgb_fact", vend("rgb_fact", arm))
        w.set(arm, "aime_en", vend("aime_en", arm))
        w.set(arm, "aime_de", vend("aime_de", arm))
    for arm in ("G8", "Q36-8"):
        w.set(arm, "gpqa_en", vend("gpqa_en", arm))
        w.set(arm, "gpqa_de", vend("gpqa_de", arm))
    w.set("Q38-8", "gpqa_de", 0.88)
    for eff, r in (("none", 0.30), ("low", 0.70), ("medium", 0.80)):
        w.set("K8", "gpqa_en", r, effort=eff)
    for task, r in (("gpqa_de", 0.30), ("mmlu_en", 0.55), ("mmlu_de", 0.50), ("ifbench", 0.60), ("rgb_cb", 0.45)):
        w.set("K8", task, r, effort="none")
    w.ladder = []                    # bench/ladder.py layout
    for arm in ("K4", "K8", "G4"):
        for rung, n, tps in (("4k", 4096, 1500.0), ("15k", 15000, 1400.0), ("32k", 32768, 1200.0),
                             ("64k", 65536, 900.0)):
            for rep in range(3):
                w.ladder.append({"kind": "run", "arm": arm, "rung": rung, "rep": rep, "n_context": n,
                                 "prompt_tps": tps, "prefill_ms_per_token": 1000.0 / tps,
                                 "peak_bytes": int(40 * GIB + 20480 * n)})
        w.ladder.append({"kind": "idle", "arm": arm})
    w.c1 = [{"key": {"arm": "K8", "task": "c1", "effort": "none", "item": f"c1_{j:03d}"}, "flipped": j == 7}
            for j in range(100)]
    w.c1_summary = {"n_items": 100, "B": 8, "n_flips": 1, "flip_rate": 0.01, "n_diverged": 3}
    w.gate = {"E11": {"head_embed_kl_nats_per_token_k8": 0.003, "router8_agreement": 0.985}}
    return w


def write_raw(res, w):
    """results/raw/S2/<arm>/<task>_<effort>.jsonl with completion_tokens (no text: synthetic)."""
    toks = {"none": 300, "low": 2000, "medium": 3500, "high": 5000}
    for (arm, task, eff, ps) in w.rates:
        p = res / "raw" / "S2" / arm / f"{task}_{eff}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "header"}) + "\n")
            for r in w.records():
                k = r["key"]
                if (k["arm"], k["task"], k["effort"], k["pass"]) == (arm, task, eff, ps):
                    f.write(json.dumps({"type": "record", "key": k, "completion_tokens": toks.get(eff, 1000),
                                        "wall_s": 10.0, "truncated": False}) + "\n")
            f.write('{"torn": ')                         # a torn last line is skipped and counted


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    w = rich_world()
    res = w.write_results(tmp_path_factory.mktemp("rich") / "results")
    write_raw(res, w)
    ctx = V.Context(res, w.plan, w.scores(), w.bench(), V.load_margins(), V.load_vendor())
    fam = V.evaluate_family(ctx)
    return ctx, fam["hypotheses"], X.compute_all(ctx, fam["hypotheses"])


def test_every_exploratory_analysis_is_labelled_and_computed(rich):
    _, _, e = rich
    assert set(e) == {f"E{i}" for i in range(1, 13)} | {"C1"}
    for k, v in e.items():
        assert v["label"] == X.LABEL and v["prediction"] == X.PREDICTIONS[k]
        assert v.get("status") != "not computed", (k, v)
        assert "verdict" not in v


def test_e1_composite(rich):
    _, _, e = rich
    e1 = e["E1"]
    assert len(e1["rows"]) == 10 and e1["composite_label"] is None
    assert set(e1["groups"]) == {"selection", "rgb_cb_fc", "rgb_negative"}
    assert "kolibri_last_on_composite_with_rgb_cb_fc" in e1["prediction_checks"]
    assert e1["vendor_same_rows"]["K8"] == pytest.approx(0.7558, abs=1e-3)   # 75.6 on the 10 public rows


def test_e2_says_so_and_guesses(rich, tmp_path):
    _, hyp, e = rich
    e2 = e["E2"]
    assert e2["reading"] == "says so" and e2["A"] == 1.0
    assert e2["forced"]["ran"] and e2["knows_less"]
    w = rich_world(k8_abstains=False)
    ctx = V.Context(None, w.plan, w.scores(), w.bench(), V.load_margins(), V.load_vendor())
    assert X.e2(ctx, hyp)["reading"] == "guesses"


def test_e3_negative_and_fact(rich):
    _, _, e = rich
    e3 = e["E3"]
    assert e3["negative_rejection_rate"]["K8"] == pytest.approx(vend("rgb_neg", "K8"), abs=0.005)
    assert e3["prediction_checks"]["neg_within_6pp_of_g8"] and e3["prediction_checks"]["neg_above_q36"]
    assert e3["prediction_checks"]["fc_lowest"]
    assert e3["fact_check"]["K8"]["knowledge_gated"]["n_known_closed_book"] > 0


def test_e4_e5_e6_e7(rich):
    _, _, e = rich
    assert e["E4"]["share_de"] == pytest.approx(0.95, abs=0.002)
    e5 = e["E5"]
    assert set(e5["efforts"]) == {"none", "low", "medium", "high"}
    assert e5["prediction_checks"]["tokens_increasing"] and e5["prediction_checks"]["low_minus_none_ge_10pp"]
    assert e5["efforts"]["high"]["n"] == 198                       # every Diamond item (Amendment 3)
    e6 = e["E6"]
    assert e6["prediction_checks"]["K4_no_cliff_65536"]
    assert e6["arms"]["K4"]["kv_slope_bytes_per_token"] == pytest.approx(20480, rel=1e-6)
    e7 = e["E7"]
    assert e7["german_chars_per_s_batch1"]["ratio"] == pytest.approx(0.9 * 4.90 / 4.13, rel=0.02)
    assert e7["prediction_checks"]["k4_german_chars_per_s_ge_g4"]
    assert e7["raw_torn_lines_skipped"] > 0
    assert e7["cells"]["K8/gpqa_en/none/pass0"]["mean_completion_tokens"] == 300


def test_e8_e9_e10_e11_e12_c1(rich):
    _, _, e = rich
    e8 = e["E8"]
    assert not e8["flags_gt_8pp"] and all(e8["prediction_checks"].values())
    assert set(e8["D"]) >= {"mmlu_en", "ifbench", "gpqa_de", "aime_en"}
    assert "standing_better_on_selection" in e["E9"]["prediction_checks"]
    assert e["E10"]["prediction_checks"]["q38_above_k8"]
    assert e["E11"]["gate_diagnostics"]["router8_agreement"] == 0.985
    assert e["E12"]["prediction_checks"]["gpqa_en_none_ge_10pp_below_high"]
    assert e["C1"]["flip_rate"] == pytest.approx(0.01) and e["C1"]["prediction_checks"]["flip_rate_le_2pct"]


def test_missing_inputs_are_not_computed_not_errors():
    w = all_confirmed_world()
    v = w.compute(now=NOW)
    for k in ("E5", "E10", "E11", "E12", "C1"):
        assert v["verdicts"][k]["status"] == "not computed"
        assert v["verdicts"][k]["label"] == X.LABEL


# ---------------------------------------------------------------- tables

def test_render_markdown_fills_every_placeholder(rich):
    w = rich_world()
    v = w.compute(now=NOW)
    md = T.render_markdown(v)
    assert "## Key numbers" in md and "## Plain answers" in md
    sents = T.plain_answer_sentences(v)
    assert set(sents) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    text = " ".join(" ".join(s) for s in sents.values())
    assert not re.search(r"<[A-Za-z][^>]*>", text), text
    assert "[CI]" not in text
    assert "CONFIRMED" in sents["Q3"][0]
    assert sents["Q4"][0].startswith("Kolibri knows less without documents")
    assert sents["Q3"][0].endswith("The German comparison rests on MMLU-ProX DE plus GPQA-D DE and AIME DE.")
    assert "{" not in T.render_key_numbers(v)                      # no raw dicts in the table
    assert "says so" in sents["Q4"][0]


def test_plain_answer_not_run_sentence():
    w = all_confirmed_world()
    w.kl = None
    w.speed = None
    v = w.compute(now=NOW)
    s = T.plain_answer_sentences(v)
    assert any(x.startswith("H1 was not run (") and x.endswith("we make no claim about it.") for x in s["Q1"])
    assert s["Q5"][0].startswith("We could not measure the task cost / fidelity loss of 4-bit (H8:")


def test_key_numbers_rows_order(rich):
    w = rich_world()
    rows = T.key_numbers_rows(w.compute(now=NOW))
    qs = [r["q"] for r in rows]
    assert qs == sorted(qs)                          # ordered Q1 -> Q5
    h2 = [r for r in rows if "(H2 row)" in r["what"]]
    assert len(h2) == 5 and all(r["label"] in ("V", "R") for r in h2)


def test_fact_check_rows_use_the_error_correction_rate(rich):
    ctx, _, e = rich
    c = ctx.cell("K8", "rgb_fact")
    corrected = sum(it.category == "corrected" for it in c.values()) / len(c)
    correct = sum(it.score for it in c.values()) / len(c)
    assert correct >= corrected                     # the scorer's correct counts "detected" too
    assert e["E1"]["groups"]["rgb_cb_fc"]["rows"] == ["rgb_cb", "rgb_fact"]
    assert X._row_value(ctx, "K8", "rgb_fact", sorted(c)) == pytest.approx(corrected)
