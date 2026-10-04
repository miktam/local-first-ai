"""Amendment 6: a failed family fidelity rule attributed to one build
by a pinned bf16 reference record (tools/peer_check.py, tools/fidelity_reference.json).

Synthetic: the pin loader's checks (sha256, rows, texts, schema), the pure attribution (pass / speed-only, and the
agreement of the record with the run), run() with and without a reference, a pin that does not match the run or
cannot be read (the registered failure stands), the verdict and exit code; then the record downstream: the guard,
the pilot's --without, plan_fix.build_context and fix(), and analysis/verdicts.py (H8 without Gemma 4, H1 with G4).
Real: the committed pin against the committed D record (G8 passes at 0.0167, G4 is speed-only at 0.378), its hash
coverage, and the committed step-9 record re-attributed (the table the mbp's re-run should print).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from test_tools_peer_check import _stats, fake_run  # noqa: F401  (fake_run is a fixture)
from tools import peer_check as pc

# Gemma-like chat-wrapped stats of the step-9 re-run after Amendment 5: NLL(8) 2.94 > NLL(4) 2.91 + 0.02 and
# KL(8||4) 0.32: the family rule fails.
FAILING = _stats(2.94, 2.91, 0.32)
TEXTS = ("T1", "T2")   # fake_run's texts


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_reference(root: Path, rows: dict, family: str = "gemma4", texts=TEXTS, sha: str | None = None,
                    schema: str | None = None, extra_rows=()) -> Path:
    """A D-like record under root/diagnostics/ and its pin under root/tools/. rows: {arm: (nll, kl, tokens)}."""
    (root / "diagnostics").mkdir(parents=True, exist_ok=True)
    (root / "tools").mkdir(parents=True, exist_ok=True)
    recs = [{"text": f"{t}_some_text.txt", "model": arm, "nll_bf16": 2.876, "nll_model": nll, "kl_bf16_model": kl,
             "n_tokens": n} for arm, (nll, kl, n) in rows.items() for t in texts] + list(extra_rows)
    rp = root / "diagnostics" / "ref_D.json"
    rp.write_text(json.dumps({"device": "test", "reference_kl": recs}))
    pin = {"schema": schema or pc.FIDELITY_REFERENCE_SCHEMA,
           "families": {family: {"record": "diagnostics/ref_D.json", "record_sha256": sha or _sha(rp),
                                 "texts": list(texts), "reference_model": {"repo": "x/y-bf16", "revision": "0" * 40}}}}
    pp = root / "tools" / "fidelity_reference.json"
    pp.write_text(json.dumps(pin))
    return pp


GEMMA_ROWS = {"G8": (2.94, 0.0167, 100), "G4": (2.91, 0.3778, 100)}


def load(tmp_path, rows=GEMMA_ROWS, **kw) -> dict:
    return pc.load_fidelity_reference(write_reference(tmp_path, rows, **kw), exp_dir=tmp_path)


# --------------------------------------------------------------------------- the pin


def test_the_loader_checks_the_pinned_record(tmp_path):
    ref = load(tmp_path)
    a = ref["families"]["gemma4"]["arms"]
    assert set(a) == {"G8", "G4"} and a["G8"]["tokens"] == 200
    assert abs(a["G8"]["kl_ref_per_token"] - 0.0167) < 1e-12 and abs(a["G4"]["kl_ref_per_token"] - 0.3778) < 1e-12
    assert set(a["G4"]["per_text"]) == set(TEXTS) and a["G4"]["per_text"]["T1"]["nll_arm"] == 2.91
    assert ref["pin"]["path"] == "tools/fidelity_reference.json" and len(ref["pin"]["sha256"]) == 64
    # Token-weighted over the texts, not a plain mean of the per-text values.
    rows = [{"text": "T1_a.txt", "model": "G8", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 0.1, "n_tokens": 300},
            {"text": "T2_b.txt", "model": "G8", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 0.5, "n_tokens": 100}]
    two = tmp_path / "w"
    write_reference(two, {"G4": (1.0, 0.2, 10)}, extra_rows=rows)
    assert abs(pc.load_fidelity_reference(two / "tools" / "fidelity_reference.json", two)["families"]["gemma4"]
               ["arms"]["G8"]["kl_ref_per_token"] - 0.2) < 1e-12        # (0.1 x 300 + 0.5 x 100) / 400
    # No pin file: {} (the registered behaviour).
    assert pc.load_fidelity_reference(tmp_path / "nope.json", tmp_path) == {}


@pytest.mark.parametrize("kw, msg", [
    ({"sha": "f" * 64}, "has sha256"),
    ({"rows": {"G8": (2.94, 0.0167, 100)}}, "G4 has rows for"),          # the 4-bit build is missing
    ({"family": "llama"}, "unknown family"),
    ({"schema": "something else"}, "schema"),
    ({"extra_rows": [{"text": "T5_x.txt", "model": "G4", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 1,
                      "n_tokens": 1}]}, "not one of the pinned texts"),
    ({"extra_rows": [{"text": "T1_again.txt", "model": "G4", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 1,
                      "n_tokens": 1}]}, "repeats"),
])
def test_the_loader_refuses_a_pin_it_cannot_trust(tmp_path, kw, msg):
    with pytest.raises(pc.ReferenceError, match=msg):
        load(tmp_path, **kw)


def test_the_loader_refuses_a_missing_record(tmp_path):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    (tmp_path / "diagnostics" / "ref_D.json").unlink()
    with pytest.raises(pc.ReferenceError, match="missing"):
        pc.load_fidelity_reference(pp, tmp_path)


@pytest.mark.parametrize("body, msg", [("[1, 2]", "has no list"), ("{not json", "is not JSON"),
                                       ('{"reference_kl": {"a": 1}}', "has no list")])
def test_the_loader_refuses_a_malformed_record_and_run_survives_it(tmp_path, body, msg, monkeypatch):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    rp = tmp_path / "diagnostics" / "ref_D.json"
    rp.write_text(body)
    pin = json.loads(pp.read_text())
    pin["families"]["gemma4"]["record_sha256"] = _sha(rp)
    pp.write_text(json.dumps(pin))
    with pytest.raises(pc.ReferenceError, match=msg):
        pc.load_fidelity_reference(pp, tmp_path)
    monkeypatch.setattr(pc, "FIDELITY_REFERENCE_FILE", pp)
    monkeypatch.setattr(pc.common, "EXP_DIR", tmp_path)
    ref, summary = pc._reference_for_run(None)
    assert summary["status"] == "unusable" and msg in summary["error"] and ref.get("families") is None


# --------------------------------------------------------------------------- the attribution (pure)


def _nll_chat(nll, tokens=100, texts=TEXTS):
    return {"per_text": {t: {"nll_per_token": nll, "tokens": tokens} for t in texts}}


def test_attribution_judges_each_arm_by_its_own_kl_to_the_reference(tmp_path):
    fam = load(tmp_path)["families"]["gemma4"]
    att = pc.attribute_fidelity(fam, {"G8": _nll_chat(2.94), "G4": _nll_chat(2.91)})
    assert att["usable"] and att["problems"] == [] and att["threshold"] == pc.KL_MAX == 0.2
    assert att["arms"]["G8"]["fidelity"] == "pass" and att["arms"]["G4"]["fidelity"] == pc.SPEED_ONLY
    assert att["arms"]["G4"]["agreement"]["T1"]["nll_diff"] == 0.0
    # The threshold is the registered KL bound: just below passes, exactly 0.2 is speed-only.
    edge = load(tmp_path / "e", rows={"G8": (2.94, 0.1999, 100), "G4": (2.91, 0.2, 100)})["families"]["gemma4"]
    att = pc.attribute_fidelity(edge, {"G8": _nll_chat(2.94), "G4": _nll_chat(2.91)})
    assert att["arms"]["G8"]["fidelity"] == "pass" and att["arms"]["G4"]["fidelity"] == pc.SPEED_ONLY


@pytest.mark.parametrize("g4_run, msg", [
    (_nll_chat(2.91, tokens=99), "chat-wrapped tokens in this run"),     # another tokenisation / text
    (_nll_chat(2.91 + 0.021), "in the reference record"),                # other weights or measurement
    (_nll_chat(2.91, texts=("T1",)), "no chat-wrapped NLL"),             # a text the run did not score
    (None, "no chat-wrapped NLL"),
])
def test_a_record_that_does_not_describe_this_run_is_not_used(tmp_path, g4_run, msg):
    fam = load(tmp_path)["families"]["gemma4"]
    att = pc.attribute_fidelity(fam, {"G8": _nll_chat(2.94), "G4": g4_run})
    assert not att["usable"] and any(msg in p for p in att["problems"])
    ok = pc.attribute_fidelity(fam, {"G8": _nll_chat(2.94), "G4": _nll_chat(2.91 + 0.019)})
    assert ok["usable"]                                                  # within REFERENCE_NLL_TOL (0.02)


def test_verdict_order_and_exit_code_with_speed_only():
    so = ["NLL(8) / KL(8‖4) rule failed; KL(bf16‖G4) 0.3778 ≥ 0.2 on the Amendment 6 reference"]
    assert pc._verdict({"problems": [], "speed_only_problems": so}) == pc.SPEED_ONLY == "speed-only"
    assert pc._verdict({"problems": ["x"], "speed_only_problems": so}) == "fail"           # fail wins
    assert pc._verdict({"problems": [], "speed_only_problems": so,
                        "batched_path_problems": ["greedy flip rate 0.033 > 0.02"]}) == pc.SPEED_ONLY
    assert pc._verdict({"problems": [], "speed_only_problems": []}) == "ok"
    assert pc.exit_code({"arms": {"G8": {"verdict": "B=1"}, "G4": {"verdict": pc.SPEED_ONLY}}}) == 2
    assert pc.exit_code({"arms": {"G4": {"verdict": pc.SPEED_ONLY}}}) == 2
    assert pc.exit_code({"arms": {"G4": {"verdict": pc.SPEED_ONLY}, "Q36-4": {"verdict": "fail"}}}) == 1


# --------------------------------------------------------------------------- run()


def test_run_attributes_a_failed_family_rule_with_a_matching_reference(fake_run, tmp_path):  # noqa: F811
    fake_run["chat"], fake_run["flip_ok"] = FAILING, False
    ref = load(tmp_path)
    rec = pc.run(["G8", "G4", "Q36-8", "Q36-4"], root=fake_run["root"], reference=ref)
    f = rec["families"]["gemma4"]
    assert not f["fidelity"]["nll_ok"] and not f["fidelity"]["kl_ok"]                   # the rule still fails
    att = f["fidelity_attribution"]
    assert att["used"] and att["amendment"] == "Amendment 6" and att["problems"] == []
    assert att["reference"]["record"] == "diagnostics/ref_D.json" and att["reference"]["pin_sha256"] == ref["pin"]["sha256"]
    g8, g4 = rec["arms"]["G8"], rec["arms"]["G4"]
    # G8 passes fidelity, so its own greedy flip decides: B = 1, as registered for the batched-path check.
    assert g8["problems"] == [] and g8["fidelity_reference"]["fidelity"] == "pass" and g8["verdict"] == "B=1"
    assert "fidelity passes" in pc.arm_line("G8", g8) and "greedy flip" in pc.arm_line("G8", g8)
    # G4 carries the loss: speed-only.
    assert g4["problems"] == [] and g4["verdict"] == pc.SPEED_ONLY
    assert g4["speed_only_problems"][0].startswith(pc.FAMILY_RULE_PROBLEM) and "0.3778 ≥ 0.2" in g4["speed_only_problems"][0]
    assert pc.arm_line("G4", g4).split()[1] == "speed-only"
    # The reference pins only Gemma: Qwen3.6's failure is the registered one.
    for a in ("Q36-8", "Q36-4"):
        assert rec["arms"][a]["verdict"] == "fail" and rec["arms"][a]["problems"] == [pc.FAMILY_RULE_PROBLEM]
    q_att = rec["families"]["qwen3_6"]["fidelity_attribution"]                        # and the record says why
    assert q_att["used"] is False and "no Amendment 6 fidelity reference is pinned" in q_att["reason"]
    assert rec["fidelity_reference"]["status"] == "pinned"
    assert pc.exit_code(rec) == 1
    only_gemma = pc.run(["G8", "G4"], root=fake_run["root"], reference=ref)
    assert [only_gemma["arms"][a]["verdict"] for a in ("G8", "G4")] == ["B=1", pc.SPEED_ONLY]
    assert pc.exit_code(only_gemma) == 2


def test_run_without_a_reference_keeps_the_registered_failure(fake_run):  # noqa: F811
    fake_run["chat"] = FAILING
    rec = pc.run(["G8", "G4"], root=fake_run["root"], reference={})
    for a in ("G8", "G4"):
        assert rec["arms"][a]["verdict"] == "fail" and rec["arms"][a]["problems"] == [pc.FAMILY_RULE_PROBLEM]
        assert "speed_only_problems" not in rec["arms"][a] and "fidelity_reference" not in rec["arms"][a]
    att = rec["families"]["gemma4"]["fidelity_attribution"]
    assert att == {"used": False, "amendment": "Amendment 6",
                   "reason": "no Amendment 6 fidelity reference is pinned for this family: the registered rule applies"}
    assert rec["fidelity_reference"]["status"] == "absent" and pc.exit_code(rec) == 1


def test_a_passing_family_rule_never_reads_the_reference(fake_run, tmp_path):  # noqa: F811
    ref = load(tmp_path, rows={"G8": (2.0, 0.5, 100), "G4": (2.01, 0.5, 100)})   # would make both speed-only
    rec = pc.run(["G8", "G4"], root=fake_run["root"], reference=ref)              # CHAT: the rule passes
    assert [rec["arms"][a]["verdict"] for a in ("G8", "G4")] == ["ok", "ok"]
    assert "fidelity_attribution" not in rec["families"]["gemma4"]


def test_a_reference_that_does_not_match_the_run_keeps_the_registered_failure(fake_run, tmp_path):  # noqa: F811
    fake_run["chat"] = FAILING
    ref = load(tmp_path, rows={"G8": (2.94, 0.0167, 100), "G4": (3.30, 0.3778, 100)})   # G4 NLL 3.30 vs 2.91
    rec = pc.run(["G8", "G4"], root=fake_run["root"], reference=ref)
    assert not rec["families"]["gemma4"]["fidelity_attribution"]["used"]
    for a in ("G8", "G4"):
        assert rec["arms"][a]["verdict"] == "fail" and rec["arms"][a]["problems"][0] == pc.FAMILY_RULE_PROBLEM
        assert "does not match this run" in rec["arms"][a]["problems"][1]
    # The committed pin (T1-T4 of the real Gemma builds) does not describe the fake run either: the default load.
    real = pc.run(["G8", "G4"], root=fake_run["root"])
    assert real["fidelity_reference"]["status"] == "pinned"
    assert not real["families"]["gemma4"]["fidelity_attribution"]["used"]
    assert [real["arms"][a]["verdict"] for a in ("G8", "G4")] == ["fail", "fail"]


def test_an_unreadable_pin_keeps_the_registered_failure(fake_run, tmp_path, monkeypatch):  # noqa: F811
    fake_run["chat"] = FAILING
    pp = write_reference(tmp_path, GEMMA_ROWS, sha="0" * 64)
    monkeypatch.setattr(pc, "FIDELITY_REFERENCE_FILE", pp)
    monkeypatch.setattr(pc.common, "EXP_DIR", tmp_path)
    rec = pc.run(["G8", "G4"], root=fake_run["root"])
    assert rec["fidelity_reference"]["status"] == "unusable" and "sha256" in rec["fidelity_reference"]["error"]
    att = rec["families"]["gemma4"]["fidelity_attribution"]
    assert att["used"] is False and "unusable" in att["reason"]
    assert [rec["arms"][a]["verdict"] for a in ("G8", "G4")] == ["fail", "fail"]


# --------------------------------------------------------------------------- downstream


def test_a_speed_only_record_downstream(fake_run, tmp_path, monkeypatch):  # noqa: F811
    """The record run() writes with the reference: the guard keeps G4 out of every quality cell and does not
    exclude it, the pilot's --without accepts it, plan_fix.build_context (the production reader of the record)
    carries it into speed_only_arms, fix() queues exactly the all-ok plan's cells (G4 has no task cell; the B4
    ladder keeps it) with Gemma out of H8's peer families, and analysis/verdicts.py runs H1 on K4 / G4 and H8
    without Gemma 4."""
    import importlib
    import shutil
    from datetime import datetime, timezone

    from analysis import verdicts as V
    from runner import memory
    from test_analysis_world import GPQA_EN_EXCLUDED, POST_STRAT, all_confirmed_world
    from test_runner_plan_fix import RULES, synthetic

    guard = importlib.import_module("runner.guard")
    plan_fix = importlib.import_module("runner.plan_fix")
    run_mod = importlib.import_module("runner.run")

    fake_run["chat"], fake_run["flip_ok"] = FAILING, False
    rec = pc.run(["G8", "G4"], root=fake_run["root"], reference=load(tmp_path / "ref"))
    rec["arms"].update({a: {"verdict": "ok", "problems": []} for a in ("Q36-8", "Q36-4", "Q38-8", "Q38-4")})
    exp = tmp_path / "exp"
    (exp / "runner").mkdir(parents=True)
    shutil.copy(pc.common.EXP_DIR / "runner" / "plan_rules.json", exp / "runner" / "plan_rules.json")
    (exp / "scorers").mkdir()
    (exp / "scorers" / "x.py").write_text("X = 1\n")
    res = exp / "results"
    res.mkdir()
    (res / "peers_20261004T210000Z.json").write_text(json.dumps(rec))
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "no_models"))
    monkeypatch.setattr(memory, "effective_limit", lambda *a, **k: (115_448_725_504, {"chosen": "test"}))

    # Guard: no quality cell for G4, but G4 is not excluded; G8 runs (at B = 1).
    with pytest.raises(guard.GuardError, match="speed-only"):
        guard.require_peers("G4", res)
    assert guard.require_peers("G8", res)["verdict"] == "B=1"
    assert guard.speed_only_arms(res) == {"G4": "peer check speed-only (peers_20261004T210000Z.json)"}
    assert guard.excluded_arms(res) == {}

    class _Ctx:
        results = res

    assert run_mod.pilot_without("G4", _Ctx()) == {"G4": "peer check speed-only (peers_20261004T210000Z.json)"}
    with pytest.raises(guard.GuardError, match="do not exclude G8"):
        run_mod.pilot_without("G8", _Ctx())

    built = plan_fix.build_context(exp, [])
    assert built["speed_only_arms"] == {"G4": "peer check speed-only (peers_20261004T210000Z.json)"}
    assert built["peer_b1"] == ["G8"] and built["excluded_arms"] == {}
    pilot, steps, s1, ctx = synthetic(0)
    base, base_md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=dict(ctx, peer_b1=["G8"]))
    ctx.update({k: built[k] for k in ("peer_b1", "excluded_arms", "speed_only_arms", "peers_record")})
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["status"] == "FIXED" and plan["speed_only_arms"] == built["speed_only_arms"]
    assert plan["queue"] == base["queue"] and plan["tier_b"] == base["tier_b"]   # G4 had no task cell to lose
    assert not any(q["arm"] == "G4" for q in plan["queue"])
    assert plan["peers"] == ["G8", "Q36-8"] and plan["not_run"] == {}            # H1 and H8 are not NOT RUN
    assert plan["h8_peers"] == ["qwen3_6", "qwen3_8"] and list(plan["h8_families_left"]) == ["gemma4"]
    assert base["h8_peers"] == ["gemma4", "qwen3_6", "qwen3_8"] and base["h8_families_left"] == {}
    assert "Speed-only arms" in md and "G4 — peer check speed-only" in md and "Speed-only arms" not in base_md
    assert "- H8 peer families: qwen3_6, qwen3_8; left H8's peer median" in md and "gemma4 (G4 speed-only" in md

    # Analysis on this plan: H8 runs on Qwen3.6 and Qwen3.8 only; H1 still runs on K4 vs G4.
    w = all_confirmed_world()
    plan.update({"gpqa_en_excluded": list(GPQA_EN_EXCLUDED), "post_strat_counts": POST_STRAT, "cells": []})
    w.plan = plan
    v = w.compute(now=datetime(2026, 10, 10, tzinfo=timezone.utc), exploratory=False)
    h8 = v["verdicts"]["H8_detail"]
    assert h8["detail"]["models"] == ["kolibri", "qwen3_6", "qwen3_8"]
    assert h8["detail"]["families_left"] == {"gemma4": "G4"}
    assert v["verdicts"]["H1_detail"]["verdict"] != V.NOT_RUN
    assert v["verdicts"]["H1_detail"]["detail"]["n_blocks"] == 10


# --------------------------------------------------------------------------- the committed pin


def test_the_committed_pin_puts_gemmas_loss_on_g4():
    """tools/fidelity_reference.json against the committed D record (diagnostics/, commit 70fedbf): token-weighted
    over T1-T4 (7,005 tokens), KL(bf16||G8) 0.0167 passes, KL(bf16||G4) 0.378 is speed-only."""
    ref = pc.load_fidelity_reference()
    fam = ref["families"]["gemma4"]
    assert fam["record"] == "diagnostics/gemma_quant_check_20261004T192910Z_D.json"
    assert fam["record_sha256"] == "9334e973e1db8ddf5cceaf76524c2dc3a865257657e3e325560662c1691143ab"
    assert fam["reference_model"]["revision"] == "e13fae2a81ec07e3092a3ebb70c80970b640dc3c"
    assert fam["texts"] == ["T1", "T2", "T3", "T4"] and set(ref["families"]) == {"gemma4"}
    g8, g4 = fam["arms"]["G8"], fam["arms"]["G4"]
    assert g8["tokens"] == g4["tokens"] == 7005
    assert round(g8["kl_ref_per_token"], 4) == 0.0167 and round(g4["kl_ref_per_token"], 4) == 0.3778
    assert round(g8["nll_arm_per_token"], 3) == 2.863 and round(g4["nll_arm_per_token"], 3) == 2.821
    assert round(g8["nll_ref_per_token"], 3) == 2.876
    assert pc.reference_classes(ref) == {"G8": "pass", "G4": pc.SPEED_ONLY}


def test_the_pin_is_hash_covered_and_the_diagnostics_are_not_a_scope():
    """The pin is in the TOOLS tree (its sha256 binds the record's), and diagnostics/ stays outside every scope."""
    from tools import hash_tree

    tools = hash_tree.BY_NAME["tools"]
    rels = [ln.split("\t", 1)[0] for ln in hash_tree.tree_lines(pc.common.EXP_DIR / tools.path, tools.include,
                                                                tools.exclude)]
    assert "fidelity_reference.json" in rels
    assert not any(s.path == "diagnostics" or s.path.startswith("diagnostics/") for s in hash_tree.SCOPES)


def test_the_committed_step9_record_re_attributed():
    """What the mbp's step-9 re-run should print, from the committed record of the run that failed (3461c8c): the
    same chat-wrapped numbers agree with the D record to ~1e-7, so G8 passes fidelity and reads B=1 on its own
    greedy flip, and G4 is speed-only; Qwen unchanged; exit code 2."""
    p = pc.common.EXP_DIR / "results" / "peers_20261004T175538Z.json"
    if not p.is_file():
        pytest.skip("the committed step-9 record is not in this checkout")
    rec = json.loads(p.read_text(encoding="utf-8"))
    for a in rec["arms"].values():
        a["problems"] = [x for x in a["problems"] if x != pc.FAMILY_RULE_PROBLEM]
    pc.apply_family_rule_failure(rec, "gemma4", ("G8", "G4"), pc.load_fidelity_reference())
    att = rec["families"]["gemma4"]["fidelity_attribution"]
    assert att["used"] and max(abs(x["nll_diff"]) for a in att["arms"].values()
                               for x in a["agreement"].values()) < 1e-5
    got = {a: pc._verdict(e) for a, e in rec["arms"].items()}
    assert got == {"G8": "B=1", "G4": pc.SPEED_ONLY, "Q36-8": "B=1", "Q36-4": "ok", "Q38-8": "ok", "Q38-4": "ok"}
    for a, v in got.items():
        rec["arms"][a]["verdict"] = v
    assert pc.exit_code(rec) == 2
