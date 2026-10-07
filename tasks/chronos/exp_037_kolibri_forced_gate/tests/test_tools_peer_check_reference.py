"""Amendment 6: a failed family fidelity rule attributed to one build
by a pinned bf16 reference record (tools/peer_check.py, tools/fidelity_reference.json).

exp_037 (DESIGN §3.18, decision (c)): the pin names the record by rule, not by sha256. The record is produced in
S1 under MLX 0.32.3 (RUNBOOK step 9a) by a byte-identical copy of exp_036's producer,
diagnostics/gemma_quant_check.py; exp_036's record (MLX 0.31.2) is never read (§2.11).

Synthetic: the S1 rule (exactly one matching record, earlier than the peer check's t_start, the pinned producer
sha256, P2's MLX and the M5 Max in the record; refused with none, two, a wrong producer, another MLX or device, or
a later UTC), the pin loader's other checks (rows, texts, schema), the pure attribution (pass / speed-only, and the
agreement of the record with the run), run() with and without a reference, a pin or record that does not match the
run or cannot be used (the registered failure stands), the verdict and exit code; then the record downstream: the
guard, the pilot's --without, plan_fix.build_context and fix(), and analysis/verdicts.py (H8 without Gemma 4, H1
with G4). Real: the committed pin (its rule's constants against P2 and the copied producer), its hash coverage,
this checkout's record state (none before S1, exactly one after), and the copied producer itself: its part A runs
on this host and its bench.kl_8v4 calls resolve against this kit.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_tools_peer_check import _stats, fake_run  # noqa: F401  (fake_run is a fixture)
from tools import peer_check as pc

# Gemma-like chat-wrapped stats of the step-9 re-run after Amendment 5: NLL(8) 2.94 > NLL(4) 2.91 + 0.02 and
# KL(8||4) 0.32: the family rule fails.
FAILING = _stats(2.94, 2.91, 0.32)
TEXTS = ("T1", "T2")   # fake_run's texts

EXP = pc.common.EXP_DIR
E36 = EXP.parent / "exp_036_kolibri_local_eval"
PRODUCER = "diagnostics/gemma_quant_check.py"
PRODUCER_SHA256 = "9393765dbad457e288e2085fc5c439596088947e00b893da3c7748f714e548e9"   # DESIGN §2.2, §3.18
PATTERN = "diagnostics/gemma_quant_check_<YYYYMMDDTHHMMSSZ>.json"
RULE = json.loads(pc.FIDELITY_REFERENCE_FILE.read_text(encoding="utf-8"))["families"]["gemma4"]["record_rule"]
REC_UTC = "20261005T120000Z"          # a synthetic step-9a record, in the past of any real run
T_START = "2026-10-05T13:00:00Z"      # a synthetic peer check's t_start, one hour later


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def record_path(root: Path, utc: str = REC_UTC) -> Path:
    return root / "diagnostics" / f"gemma_quant_check_{utc}.json"


def write_record(root: Path, rows: dict, texts=TEXTS, utc: str = REC_UTC, extra_rows=(), **top) -> Path:
    """A step-9a-like record (the copied producer's shape: device, mlx, kernels, reference_kl).
    rows: {arm: (nll, kl, tokens)}; top overrides device / mlx (None drops the key)."""
    recs = [{"text": f"{t}_some_text.txt", "model": arm, "nll_bf16": 2.876, "nll_model": nll, "kl_bf16_model": kl,
             "n_tokens": n} for arm, (nll, kl, n) in rows.items() for t in texts] + list(extra_rows)
    body = {"device": RULE["device"], "mlx": RULE["mlx"], "kernels": [], "reference_kl": recs}
    for k, v in top.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    rp = record_path(root, utc)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(body))
    return rp


def write_reference(root: Path, rows: dict, family: str = "gemma4", texts=TEXTS, schema: str | None = None,
                    extra_rows=(), utc: str = REC_UTC, producer_sha: str | None = None, **top) -> Path:
    """A stand-in producer and one record under root/diagnostics/, and the pin under root/tools/ in the committed
    pin's format, with its record rule (P2's MLX, the M5 Max)."""
    (root / "tools").mkdir(parents=True, exist_ok=True)
    prod = root / PRODUCER
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text("# a stand-in producer\n")
    write_record(root, rows, texts=texts, utc=utc, extra_rows=extra_rows, **top)
    pin = {"schema": schema or pc.FIDELITY_REFERENCE_SCHEMA,
           "families": {family: {"record": PATTERN, "record_rule": dict(RULE),
                                 "producer": {"script": PRODUCER, "script_sha256": producer_sha or _sha(prod)},
                                 "texts": list(texts), "reference_model": {"repo": "x/y-bf16", "revision": "0" * 40}}}}
    pp = root / "tools" / "fidelity_reference.json"
    pp.write_text(json.dumps(pin))
    return pp


GEMMA_ROWS = {"G8": (2.94, 0.0167, 100), "G4": (2.91, 0.3778, 100)}


def load(tmp_path, rows=GEMMA_ROWS, before=T_START, **kw) -> dict:
    return pc.load_fidelity_reference(write_reference(tmp_path, rows, **kw), exp_dir=tmp_path, before=before)


# --------------------------------------------------------------------------- the pin


def test_the_loader_checks_the_pinned_record(tmp_path):
    ref = load(tmp_path)
    fam = ref["families"]["gemma4"]
    a = fam["arms"]
    assert set(a) == {"G8", "G4"} and a["G8"]["tokens"] == 200
    assert abs(a["G8"]["kl_ref_per_token"] - 0.0167) < 1e-12 and abs(a["G4"]["kl_ref_per_token"] - 0.3778) < 1e-12
    assert set(a["G4"]["per_text"]) == set(TEXTS) and a["G4"]["per_text"]["T1"]["nll_arm"] == 2.91
    assert ref["pin"]["path"] == "tools/fidelity_reference.json" and len(ref["pin"]["sha256"]) == 64
    # The record chosen by the rule: its path, the sha256 of its bytes, its UTC, MLX and device, and the producer.
    assert fam["record"] == f"diagnostics/gemma_quant_check_{REC_UTC}.json"
    assert fam["record_sha256"] == _sha(record_path(tmp_path))
    assert (fam["record_utc"], fam["mlx"], fam["device"]) == (REC_UTC, RULE["mlx"], RULE["device"])
    assert fam["producer"] == {"script": PRODUCER, "script_sha256": _sha(tmp_path / PRODUCER)}
    # Token-weighted over the texts, not a plain mean of the per-text values.
    rows = [{"text": "T1_a.txt", "model": "G8", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 0.1, "n_tokens": 300},
            {"text": "T2_b.txt", "model": "G8", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 0.5, "n_tokens": 100}]
    two = tmp_path / "w"
    write_reference(two, {"G4": (1.0, 0.2, 10)}, extra_rows=rows)
    assert abs(pc.load_fidelity_reference(two / "tools" / "fidelity_reference.json", two, before=T_START)
               ["families"]["gemma4"]["arms"]["G8"]["kl_ref_per_token"] - 0.2) < 1e-12   # (0.1 x 300 + 0.5 x 100) / 400
    # No pin file: {} (the registered behaviour).
    assert pc.load_fidelity_reference(tmp_path / "nope.json", tmp_path) == {}


@pytest.mark.parametrize("kw, msg", [
    ({"rows": {"G8": (2.94, 0.0167, 100)}}, "G4 has rows for"),          # the 4-bit build is missing
    ({"family": "llama"}, "unknown family"),
    ({"schema": "something else"}, "schema"),
    ({"schema": "exp036 fidelity reference v1"}, "schema"),              # exp_036's pin format (sha256-pinned)
    ({"extra_rows": [{"text": "T5_x.txt", "model": "G4", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 1,
                      "n_tokens": 1}]}, "not one of the pinned texts"),
    ({"extra_rows": [{"text": "T1_again.txt", "model": "G4", "nll_bf16": 1, "nll_model": 1, "kl_bf16_model": 1,
                      "n_tokens": 1}]}, "repeats"),
])
def test_the_loader_refuses_a_pin_it_cannot_trust(tmp_path, kw, msg):
    with pytest.raises(pc.ReferenceError, match=msg):
        load(tmp_path, **kw)


# --------------------------------------------------------------------------- the S1 rule (DESIGN §3.18)


def _stamp(iso: str) -> str:
    return iso.replace("-", "").replace(":", "")


def test_the_s1_rule_accepts_exactly_one_matching_record(tmp_path):
    """Exactly one file matches the pattern; names that only resemble it do not count (exp_036's `_D` record, the
    producer, a stamp-less or malformed name, a directory)."""
    pp = write_reference(tmp_path, GEMMA_ROWS)
    diag = tmp_path / "diagnostics"
    (diag / "gemma_quant_check_20261004T192910Z_D.json").write_text("{}")   # exp_036's name: not the pattern
    (diag / "gemma_quant_check.json").write_text("{}")
    (diag / "gemma_quant_check_20261005T1200Z.json").write_text("{}")
    (diag / "gemma_quant_check_20261005T120000Z.json.tmp").write_text("{}")
    (diag / "gemma_quant_check_20261005T110000Z.json").mkdir()               # a directory is not a record
    # A repeat after a crash moves the earlier output to aborted/<UTC>-gemmacheck/ (§3.18): it no longer counts.
    write_record(tmp_path / "aborted" / "20261005T113000Z-gemmacheck", GEMMA_ROWS, utc="20261005T110000Z")
    ref = pc.load_fidelity_reference(pp, tmp_path, before=T_START)
    assert ref["families"]["gemma4"]["record"] == f"diagnostics/gemma_quant_check_{REC_UTC}.json"


def test_the_s1_rule_refuses_none_and_two(tmp_path):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    record_path(tmp_path).unlink()
    with pytest.raises(pc.ReferenceError, match="no record matches"):
        pc.load_fidelity_reference(pp, tmp_path, before=T_START)
    write_record(tmp_path, GEMMA_ROWS)
    write_record(tmp_path, GEMMA_ROWS, utc="20261005T110000Z")              # a repeat left in place
    with pytest.raises(pc.ReferenceError, match="2 records match .*none is chosen"):
        pc.load_fidelity_reference(pp, tmp_path, before=T_START)


@pytest.mark.parametrize("kw, msg", [
    ({"producer_sha": "f" * 64}, "producer .* has sha256"),
    ({"mlx": "0.31.2"}, "has mlx '0.31.2'"),                             # exp_036's MLX
    ({"mlx": "0.32.2"}, "has mlx '0.32.2'"),
    ({"mlx": None}, "has mlx None"),
    ({"device": "Apple M4 Pro"}, "has device 'Apple M4 Pro'"),            # the mini
    ({"device": "Apple M5 Pro"}, "has device"),
    ({"device": None}, "has device None"),
])
def test_the_s1_rule_refuses_a_record_from_another_producer_runtime_or_device(tmp_path, kw, msg):
    with pytest.raises(pc.ReferenceError, match=msg):
        load(tmp_path, **kw)


def test_the_s1_rule_refuses_a_missing_producer(tmp_path):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    (tmp_path / PRODUCER).unlink()
    with pytest.raises(pc.ReferenceError, match="producer .* is missing"):
        pc.load_fidelity_reference(pp, tmp_path, before=T_START)


def test_the_s1_rule_needs_a_record_earlier_than_the_peer_checks_t_start(tmp_path):
    """Strictly earlier, at the second: one second after the record is fine; the same second or earlier is not."""
    pp = write_reference(tmp_path, GEMMA_ROWS)
    ok = pc.load_fidelity_reference(pp, tmp_path, before="2026-10-05T12:00:01Z")
    assert ok["families"]["gemma4"]["record_utc"] == REC_UTC
    for before in ("2026-10-05T12:00:00Z", "2026-10-05T11:59:59Z", "2026-10-04T13:00:00Z"):
        with pytest.raises(pc.ReferenceError, match="is not earlier than the peer check's t_start"):
            pc.load_fidelity_reference(pp, tmp_path, before=before)
    # The default `before` is now: a record stamped in the future is refused.
    future = tmp_path / "f"
    with pytest.raises(pc.ReferenceError, match="is not earlier"):
        pc.load_fidelity_reference(write_reference(future, GEMMA_ROWS, utc="20991231T235959Z"), future)
    # A stamp that is not a time.
    bad = tmp_path / "b"
    with pytest.raises(pc.ReferenceError, match="no valid UTC stamp"):
        pc.load_fidelity_reference(write_reference(bad, GEMMA_ROWS, utc="20261345T250000Z"), bad, before=T_START)
    assert _stamp(T_START) == "20261005T130000Z" and _stamp(T_START) > REC_UTC


def test_the_s1_rule_refuses_a_malformed_pin(tmp_path):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    pin = json.loads(pp.read_text())
    for mutate, msg in [
        (lambda f: f.pop("record_rule"), "needs record, record_rule"),
        (lambda f: f["record_rule"].pop("device"), "needs record, record_rule"),
        (lambda f: f.pop("producer"), "needs record, record_rule"),
        (lambda f: f.update(record="diagnostics/gemma_quant_check.json"), "must hold <YYYYMMDDTHHMMSSZ> once"),
        (lambda f: f.update(record="diagnostics/<YYYYMMDDTHHMMSSZ>/x_<YYYYMMDDTHHMMSSZ>.json"), "once, in the file"),
        (lambda f: f.update(record_sha256="0" * 64, record="diagnostics/x.json"), "must hold"),   # exp_036's shape
    ]:
        p = json.loads(json.dumps(pin))
        mutate(p["families"]["gemma4"])
        pp.write_text(json.dumps(p))
        with pytest.raises(pc.ReferenceError, match=msg):
            pc.load_fidelity_reference(pp, tmp_path, before=T_START)


@pytest.mark.parametrize("body, msg", [("[1, 2]", "is not a JSON object"), ("{not json", "is not JSON"),
                                       ('{"mlx": "0.32.3", "device": "Apple M5 Max", "reference_kl": {"a": 1}}',
                                        "has no list")])
def test_the_loader_refuses_a_malformed_record_and_run_survives_it(tmp_path, body, msg, monkeypatch):
    pp = write_reference(tmp_path, GEMMA_ROWS)
    record_path(tmp_path).write_text(body)
    with pytest.raises(pc.ReferenceError, match=msg):
        pc.load_fidelity_reference(pp, tmp_path, before=T_START)
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
    assert att["reference"]["record"] == f"diagnostics/gemma_quant_check_{REC_UTC}.json"
    assert att["reference"]["pin_sha256"] == ref["pin"]["sha256"]
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


def test_run_finds_the_s1_record_by_rule_and_records_its_path_and_sha256(fake_run, tmp_path, monkeypatch):  # noqa: F811
    """The production path: run() loads the pin itself, with this run's t_start, and the peer record carries the
    chosen record's path and sha256 (fidelity_reference.families.gemma4; DESIGN §3.18)."""
    fake_run["chat"], fake_run["flip_ok"] = FAILING, False
    pp = write_reference(tmp_path, GEMMA_ROWS)
    monkeypatch.setattr(pc, "FIDELITY_REFERENCE_FILE", pp)
    monkeypatch.setattr(pc.common, "EXP_DIR", tmp_path)
    rec = pc.run(["G8", "G4"], root=fake_run["root"])
    fr = rec["fidelity_reference"]
    assert fr["status"] == "pinned" and fr["pin"] == "tools/fidelity_reference.json"
    assert fr["families"]["gemma4"] == {"record": f"diagnostics/gemma_quant_check_{REC_UTC}.json",
                                        "record_sha256": _sha(record_path(tmp_path)), "record_utc": REC_UTC}
    assert REC_UTC < _stamp(rec["t_start"])
    assert [rec["arms"][a]["verdict"] for a in ("G8", "G4")] == ["B=1", pc.SPEED_ONLY]


@pytest.mark.parametrize("case", ["later", "none", "two", "producer", "mlx", "device"])
def test_an_unusable_s1_record_keeps_the_registered_failure(fake_run, tmp_path, monkeypatch, case):  # noqa: F811
    """None, two, a record made after the peer check started, another producer, MLX or device: the pin is
    unusable, run() records why and both Gemma builds fail as registered (none is speed-only, none passes)."""
    fake_run["chat"] = FAILING
    kw = {"later": {"utc": "20991231T235959Z"}, "producer": {"producer_sha": "e" * 64},
          "mlx": {"mlx": "0.31.2"}, "device": {"device": "Apple M4 Pro"}}.get(case, {})
    pp = write_reference(tmp_path, GEMMA_ROWS, **kw)
    if case == "none":
        record_path(tmp_path).unlink()
    if case == "two":
        write_record(tmp_path, GEMMA_ROWS, utc="20261005T110000Z")
    monkeypatch.setattr(pc, "FIDELITY_REFERENCE_FILE", pp)
    monkeypatch.setattr(pc.common, "EXP_DIR", tmp_path)
    rec = pc.run(["G8", "G4"], root=fake_run["root"])
    assert rec["fidelity_reference"]["status"] == "unusable"
    att = rec["families"]["gemma4"]["fidelity_attribution"]
    assert att["used"] is False and "unusable" in att["reason"]
    for a in ("G8", "G4"):
        assert rec["arms"][a]["verdict"] == "fail" and rec["arms"][a]["problems"] == [pc.FAMILY_RULE_PROBLEM]
        assert "speed_only_problems" not in rec["arms"][a] and "fidelity_reference" not in rec["arms"][a]
    assert pc.exit_code(rec) == 1


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
    # The committed pin, by default: before S1 no record exists (unusable); after S1 the real Gemma record does
    # not describe the fake run (T1-T4 of the real builds). Either way the registered failure stands.
    real = pc.run(["G8", "G4"], root=fake_run["root"])
    if _records_in_this_checkout():
        assert real["fidelity_reference"]["status"] in ("pinned", "unusable")
    else:
        assert real["fidelity_reference"]["status"] == "unusable"
        assert "no record matches" in real["fidelity_reference"]["error"]
    assert not real["families"]["gemma4"]["fidelity_attribution"]["used"]
    assert [real["arms"][a]["verdict"] for a in ("G8", "G4")] == ["fail", "fail"]


def test_an_unreadable_pin_keeps_the_registered_failure(fake_run, tmp_path, monkeypatch):  # noqa: F811
    fake_run["chat"] = FAILING
    pp = write_reference(tmp_path, GEMMA_ROWS, producer_sha="0" * 64)
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


# --------------------------------------------------------------------------- the committed pin and producer


def _records_in_this_checkout() -> list[str]:
    """The step-9a records in this checkout (none before S1; exactly one once S1 has run)."""
    rx = re.compile(r"gemma_quant_check_[0-9]{8}T[0-9]{6}Z\.json\Z")
    d = EXP / "diagnostics"
    return sorted(p.name for p in d.iterdir() if rx.match(p.name)) if d.is_dir() else []


def test_the_committed_pin_names_the_s1_record_by_rule():
    """tools/fidelity_reference.json (DESIGN §3.18): exp_036's family, texts, row fields, bf16 source and
    measurement; the record named by rule, with no sha256 and no exp_036 record; its MLX is P2's pin and its
    device the M5 Max; its producer the byte-identical copy of exp_036's."""
    pin = json.loads(pc.FIDELITY_REFERENCE_FILE.read_text(encoding="utf-8"))
    assert pin["schema"] == pc.FIDELITY_REFERENCE_SCHEMA == "exp037 fidelity reference v1"
    assert set(pin["families"]) == {"gemma4"}
    fam = pin["families"]["gemma4"]
    assert fam["record"] == PATTERN and "record_sha256" not in fam and "record_commit" not in fam
    assert fam["record_rule"] == {"mlx": "0.32.3", "device": "Apple M5 Max"}
    p2 = json.loads((EXP / "gate" / "thresholds.json").read_text(encoding="utf-8"))["P2"]["packages"]["mlx"]
    versions = json.loads((EXP / "env" / "versions.json").read_text(encoding="utf-8"))["core"]["mlx"]
    assert fam["record_rule"]["mlx"] == p2 == versions                      # P2's MLX pin, one value
    assert fam["producer"]["script"] == PRODUCER and fam["producer"]["script_sha256"] == PRODUCER_SHA256
    assert fam["texts"] == ["T1", "T2", "T3", "T4"] and fam["rows"] == "reference_kl"
    assert fam["fields"] == pc.REFERENCE_FIELDS
    assert fam["reference_model"]["repo"] == "mlx-community/gemma-4-26b-a4b-it-bf16"
    assert fam["reference_model"]["revision"] == "e13fae2a81ec07e3092a3ebb70c80970b640dc3c"
    # The rule's file name pattern does not match exp_036's records (never read, DESIGN §2.11).
    rx = re.compile(r"gemma_quant_check_([0-9]{8}T[0-9]{6}Z)\.json\Z")
    for name in ("gemma_quant_check_20261004T192910Z_D.json", "gemma_quant_check_20261004T181520Z_AC.json"):
        assert not rx.match(name)
    # This checkout's state: no record before S1 (the pin is then unusable and the registered rule applies);
    # exactly one after it, which the loader accepts as of now.
    found = _records_in_this_checkout()
    assert len(found) <= 1, f"two or more step-9a records: {found}"
    if not found:
        with pytest.raises(pc.ReferenceError, match="no record matches"):
            pc.load_fidelity_reference()
        assert pc._reference_for_run(None)[1]["status"] == "unusable"
    else:
        ref = pc.load_fidelity_reference()
        g = ref["families"]["gemma4"]
        assert g["record"] == f"diagnostics/{found[0]}" and g["mlx"] == p2 and g["device"] == "Apple M5 Max"
        assert set(g["arms"]) == {"G8", "G4"} and all(a["tokens"] > 0 for a in g["arms"].values())


def test_the_pin_is_hash_covered_and_the_diagnostics_are_not_a_scope():
    """The pin is in the TOOLS tree (its sha256 binds the rule and the producer's), and diagnostics/ stays outside
    every scope."""
    from tools import hash_tree

    tools = hash_tree.BY_NAME["tools"]
    rels = [ln.split("\t", 1)[0] for ln in hash_tree.tree_lines(pc.common.EXP_DIR / tools.path, tools.include,
                                                                tools.exclude)]
    assert "fidelity_reference.json" in rels
    assert not any(s.path == "diagnostics" or s.path.startswith("diagnostics/") for s in hash_tree.SCOPES)


def test_the_producer_is_exp036s_byte_for_byte():
    """diagnostics/gemma_quant_check.py is a byte-identical copy of exp_036's (DESIGN §2.2), the sha256 the pin
    names; exp_036's file is read here as a comparison only."""
    assert _sha(EXP / PRODUCER) == PRODUCER_SHA256
    assert _sha(E36 / PRODUCER) == PRODUCER_SHA256


def _producer_tree() -> ast.Module:
    return ast.parse((EXP / PRODUCER).read_text(encoding="utf-8"))


def test_the_producers_imports_and_calls_resolve_in_this_kit():
    """The copied script reaches this kit's bench/kl_8v4.py and tools/precision.py (it puts the experiment
    directory first on sys.path), and mlx_lm's load. Every K.<name>(...) call it makes names a function of
    bench.kl_8v4 whose signature accepts that call's positional count and keywords; part D's text glob finds
    exactly the pinned texts T1-T4 in this kit's gate/texts."""
    import mlx_lm

    from bench import kl_8v4 as K
    from tools import precision

    tree = _producer_tree()
    imports = {(n.module, a.name, a.asname) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("bench", "kl_8v4", "K") in imports and ("mlx_lm", "load", None) in imports
    assert ("tools.precision", "ensure_exact_fp32", None) in imports
    assert callable(precision.ensure_exact_fp32) and callable(mlx_lm.load)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "K"]
    names = sorted({c.func.attr for c in calls})
    assert names == ["chat_wrapper", "kl_rows", "logprobs", "scored_chunks", "tokenize"]
    for c in calls:
        fn = getattr(K, c.func.attr)
        assert callable(fn), c.func.attr
        inspect.signature(fn).bind(*([None] * len(c.args)), **{k.arg: None for k in c.keywords})
    texts = sorted(p.name.split("_", 1)[0] for p in (EXP / "gate" / "texts").glob("T[1-4]_*.txt"))
    assert texts == ["T1", "T2", "T3", "T4"]
    pin = json.loads(pc.FIDELITY_REFERENCE_FILE.read_text(encoding="utf-8"))
    assert texts == pin["families"]["gemma4"]["texts"]


def test_the_producers_part_a_runs_here(tmp_path):
    """--skip-weights (part A, the kernel check) runs under this interpreter's MLX: it needs no model files and
    writes device, mlx and 100 kernel rows (10 shapes x 2 bit widths x (4 quantized_matmul + 1 gather_qmm))."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    out = tmp_path / "partA.json"
    r = subprocess.run([sys.executable, str(EXP / PRODUCER), "--g8", str(tmp_path / "no_g8"), "--g4",
                        str(tmp_path / "no_g4"), "--out", str(out), "--skip-weights"],
                       capture_output=True, text=True, timeout=600, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads(out.read_text())
    assert set(rec) == {"device", "mlx", "kernels"}                      # no weights, no reference_kl
    assert rec["mlx"] == mx.__version__ and rec["device"] == mx.device_info().get("device_name")
    ks = rec["kernels"]
    assert len(ks) == 100 and {(k["op"], k["bits"]) for k in ks} == {
        (op, b) for op in ("quantized_matmul", "gather_qmm_sorted") for b in (4, 8)}
    assert all(math.isfinite(k["rel_err"]) for k in ks)
    assert "A. kernels" in r.stdout and "written" in r.stdout
