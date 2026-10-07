# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5d).
"""F_tiny: gate/checks/ftiny.py and its record gate/calibration.json
(DESIGN §5.1, §3.3 G1 item 7, §4.2). Heavy: the gate's G1 run leaves this file
out (gate/checks/g1_synthetic.py EXCLUDE); the mini's full suite runs it (about
10 minutes).

Build host only (DESIGN §5.1, decision (c), 2026-10-06): F_tiny is a constant
produced on the mini, and its validation margin is thin by construction, so
another host's numerics could miss it by a few percent. The whole file runs
only on the host that wrote gate/calibration.json, recognised from the
record's device (the host label from the chip and memory, never a host name;
the GPU architecture) and package versions. Anywhere else, the mbp's RUNBOOK
step 3 included, it skips with a reason starting "build-host only:", which
EXP036_REQUIRE_ALL=run-host accepts and EXP036_REQUIRE_ALL=1 (the mini) does not.

* the record is well formed: F_tiny = 10 x the largest of its eight
  2,048-position block means, at or below the cap (thresholds.json G4F32
  f_tiny_cap, 1e-5), which rules.g4_f32_calibration accepts; the checkpoint
  seeds (23, 29), the token draws (default_rng(5), default_rng(6)) and the
  validation layout are §5.1's, written out here independently;
* the tiny sets this suite builds (tests/tiny_real_layout.py) are the recorded
  checkpoints: BF16 config and shard sha256 everywhere; the K8 build's on the
  host that wrote the record;
* G1 item 7: the validation re-run from calibration.json's F_tiny, on the
  validation checkpoint, through the exact G4-F32 code path
  (gate/checks/g4_forced.py) and gate/rules.py: rules.g4_f32 PASS; F1, F3, F5
  and F6 silent; every F2 set and bucket mean <= tau / 10; mutants 1, 6 and 19
  caught by their bound legs with the §4.2 margin; F_tiny within the cap;
  under the recorded conditions (host, packages, port, reference tree and
  measurement code) the re-run's values are the recorded ones, bitwise;
* the calibration re-run stays within the cap, and reproduces the recorded
  F_tiny exactly under the recorded conditions;
* the criteria at their boundaries, on constructed rules.g4_f32 results.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

try:
    import mlx.core  # noqa: F401
except ImportError as e:  # no MLX: not the build host, and nothing below imports without it
    pytest.skip(f"build-host only: F_tiny's tests need MLX, which is not importable here ({e})",
                allow_module_level=True)

from gate import common, rules  # noqa: E402
from gate.checks import ftiny  # noqa: E402

CAL_RECORD = Path(__file__).resolve().parents[1] / "gate" / "calibration.json"
BUILD_HOST_KEYS = ("host", "architecture", "packages")   # the recorded device and package versions


def _build_host_skip_reason() -> str | None:
    """None on the host that wrote gate/calibration.json (DESIGN §5.1, decision (c)); else the skip reason.

    The host is the record's provenance.environment: its host label (chip and memory, from
    tools/redact.host_label, never a host name), GPU architecture and packages (mlx, mlx-metal, mlx-lm,
    numpy), compared with gate.checks.ftiny.environment() here. The Python version is not compared: the
    mini runs this file under both exp_037 venvs. A missing or unreadable record does not skip, so that the
    record's own test fails where it should."""
    try:
        want = json.loads(CAL_RECORD.read_text(encoding="utf-8"))["provenance"]["environment"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    got = ftiny.environment()
    differs = [k for k in BUILD_HOST_KEYS if got.get(k) != want.get(k)]
    if differs:
        return (f"build-host only: gate/calibration.json was written on {want.get('host')} "
                f"({want.get('architecture')}, {want.get('packages')}); this host differs in "
                f"{', '.join(differs)} (DESIGN §5.1)")
    return None


# A per-test mark, not a module-level skip: under EXP036_REQUIRE_ALL=1 a module-level skip is a collection error
# that interrupts the whole session, while skipped tests are reported one by one as failures (tests/conftest.py).
_SKIP_REASON = _build_host_skip_reason()
pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

CAL = ftiny.CALIBRATION_PATH
SECTION_5_1 = {  # DESIGN §5.1, written out independently of ftiny.py
    "cal_seed": 23, "val_seed": 29, "cal_tokens": (5, 16384), "val_tokens": (6, 28672), "draw_high": 1008,
    "val_lengths": [1536] * 8 + [16384], "val_labels": ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"],
    "buckets": [[0, 2048], [2048, 8192], [8192, 16384]], "cap": 1e-5, "factor": 10.0, "block": 2048,
    "n_blocks": 8, "controls": {"G4F32/1": ("F6", 1), "G4F32/6": ("F2", 6), "G4F32/19": ("F2", 19)},
}


@pytest.fixture(scope="module")
def record() -> dict:
    assert CAL.is_file(), "gate/calibration.json is missing: run python -m gate.checks.ftiny --root <dir> --write"
    return ftiny.load()


@pytest.fixture(scope="module")
def th() -> dict:
    return common.load_thresholds()


@pytest.fixture(scope="module")
def work37(tmp_path_factory):
    """EXP036_WORK -> a temporary directory for this module (every dump this file makes lies under it)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EXP036_WORK", str(tmp_path_factory.mktemp("ftiny_work")))
        yield common.work37_dir()


def _same_conditions(record: dict) -> bool:
    """This process runs under the conditions the record was written under:
    host label, packages, port, reference tree and the measurement code
    (values then compare bitwise). After any change there, only the criteria
    and the cap are asserted."""
    prov = record["provenance"]
    env = ftiny.environment()
    return (env["host"] == prov["environment"]["host"] and env["packages"] == prov["environment"]["packages"]
            and env["architecture"] == prov["environment"]["architecture"]
            and common.sha256_file(common.PORT_FILE) == prov["port_sha256"]
            and common.reference_tree_sha256() == prov["reference_tree_sha256"]
            and ftiny.code_sha256() == prov.get("code_sha256"))


# ----------------------------------------------------------------------------- the record


def test_record_is_well_formed(record, th):
    assert record["version"] == ftiny.VERSION == "exp037-calibration-1"
    means = record["block_means"]
    assert len(means) == SECTION_5_1["n_blocks"] and record["block"] == SECTION_5_1["block"]
    assert all(isinstance(m, float) and math.isfinite(m) and m > 0 for m in means)
    assert record["factor"] == SECTION_5_1["factor"]
    assert record["f_tiny"] == SECTION_5_1["factor"] * max(means)  # exactly: one multiplication
    assert record["calibration"]["measured"]["argmax_block"] == int(np.argmax(means))
    assert record["calibration"]["measured"]["n_positions"] == SECTION_5_1["cal_tokens"][1]
    # The cap: the record's, thresholds.json's and DESIGN §5.1's are one value, and F_tiny is within it.
    assert record["cap"] == th["G4F32"]["f_tiny_cap"] == SECTION_5_1["cap"]
    assert record["f_tiny"] <= record["cap"] and record["calibration"]["cap_met"] is True
    assert ftiny.load_f_tiny() == record["f_tiny"]
    rules.g4_f32_calibration({"sets": {s: {"csort_mean": 0.0} for s in th["G4F32"]["sets"]},
                              "t9_buckets": {f"{a}-{b}": {"lo": a, "csort_mean": 0.0} for a, b in SECTION_5_1["buckets"]}},
                             record["f_tiny"], th)  # rules.py accepts it (it refuses a value above the cap)
    assert record["validation"]["all_met"] is True
    assert all(c["met"] for c in record["validation"]["criteria"])
    assert record["validation"]["rules_g4_f32"]["state"] == rules.PASS
    assert "kl_rows" in record["estimator"] and "float64" in record["estimator"]
    assert "6.4e-7" in record["context"] and "not F_tiny" in record["context"]
    assert record["provenance"]["reference_tree_sha256"] == common.reference_tree_sha256()


def test_record_names_section_5_1_seeds_and_layout(record):
    cal, val = record["calibration"], record["validation"]
    assert cal["checkpoint"]["seed"] == SECTION_5_1["cal_seed"] and val["checkpoint"]["seed"] == SECTION_5_1["val_seed"]
    for which in (cal, val):
        ck = which["checkpoint"]
        assert ck["preset"] == "w513" and ck["sharpen_qk_norms"] == 3.0
        assert (ck["k8"]["bits"], ck["k8"]["group_size"]) == (8, 64)
        lay = ck["layout"]
        assert (lay["num_hidden_layers"], lay["num_attention_heads"], lay["num_key_value_heads"], lay["head_dim"],
                lay["sliding_window"], lay["num_experts"], lay["num_experts_per_tok"], lay["hidden_size"],
                lay["moe_intermediate_size"], lay["shared_expert_intermediate_size"]) == \
            (10, 48, 4, 128, 513, 64, 6, 256, 64, 64)
        assert [i for i, t in enumerate(lay["layer_types"]) if t == "full_attention"] == [4, 9]
        assert ck["bf16"]["shards_sha256"] and ck["k8"]["shards_sha256"]
    seed, n = SECTION_5_1["cal_tokens"]
    want = np.random.default_rng(seed).integers(0, SECTION_5_1["draw_high"], n)
    assert cal["tokens"]["token_seed"] == seed and cal["tokens"]["lengths"] == [n]
    assert cal["tokens"]["pack_ids_sha256"] == common.ids_sha256(want)
    seed, n = SECTION_5_1["val_tokens"]
    want = np.random.default_rng(seed).integers(0, SECTION_5_1["draw_high"], n)
    assert val["tokens"]["token_seed"] == seed and val["tokens"]["lengths"] == SECTION_5_1["val_lengths"]
    assert val["tokens"]["labels"] == SECTION_5_1["val_labels"]
    assert val["tokens"]["pack_ids_sha256"] == common.ids_sha256(want)
    assert val["layout"]["t9_buckets"] == SECTION_5_1["buckets"]


def test_record_carries_no_local_path(record):
    text = CAL.read_text(encoding="utf-8")
    assert str(Path.home()) not in text and "/private/" not in text and "/tmp/" not in text


def test_build_host_rule(record, monkeypatch):
    """DESIGN §5.1 (decision (c)): this file runs here because this host is the recorded one; a host that differs
    in its label, GPU architecture or any recorded package gets a skip reason starting "build-host only:"."""
    assert CAL_RECORD == CAL
    assert _build_host_skip_reason() is None
    here = ftiny.environment()
    assert {k: here[k] for k in BUILD_HOST_KEYS} == {k: record["provenance"]["environment"][k] for k in BUILD_HOST_KEYS}
    for k, other in [("host", "mbp (M5 Max, 128 GB)"), ("architecture", "applegpu_g17s"),
                     ("packages", {**here["packages"], "mlx": "0.32.4"})]:
        monkeypatch.setattr(ftiny, "environment", lambda k=k, other=other: {**here, k: other})
        reason = _build_host_skip_reason()
        assert reason.startswith("build-host only:") and reason.endswith(f"differs in {k} (DESIGN §5.1)"), reason
    monkeypatch.setattr(ftiny, "environment", lambda: {**here, "python": "3.99.0"})
    assert _build_host_skip_reason() is None   # the Python version is not part of the host


# ----------------------------------------------------------------------------- layouts and arithmetic


def test_layouts_are_section_5_1():
    seed, n = SECTION_5_1["cal_tokens"]
    assert np.array_equal(ftiny.CAL_LAYOUT.pack(), np.random.default_rng(seed).integers(0, 1008, n))
    assert [lab for lab, _ in ftiny.CAL_LAYOUT.texts()] == ["T9"]
    seed, n = SECTION_5_1["val_tokens"]
    pack = np.random.default_rng(seed).integers(0, 1008, n)
    texts = ftiny.VAL_LAYOUT.texts()
    assert [lab for lab, _ in texts] == SECTION_5_1["val_labels"]
    assert [ids.size for _, ids in texts] == SECTION_5_1["val_lengths"]
    assert np.array_equal(np.concatenate([ids for _, ids in texts]), pack)
    assert ftiny.VAL_LAYOUT.segments() == [(1536 * j, 1536 * (j + 1)) for j in range(8)] + [(12288, 28672)]
    assert ftiny.CAL_LAYOUT.text_key() != ftiny.VAL_LAYOUT.text_key()
    ts = ftiny.validation_textset()
    assert [len(ts.ids[t]) for t in SECTION_5_1["val_labels"][:8]] == [1536] * 8 and len(ts.t9) == 16384
    assert [list(b) for b in ts.profile.t9_buckets] == SECTION_5_1["buckets"]


def test_f_tiny_of():
    assert ftiny.f_tiny_of([1e-8, 3e-8, 2e-8]) == 10.0 * 3e-8
    for bad in ([], [1e-8, math.nan], [math.inf]):
        with pytest.raises(ValueError):
            ftiny.f_tiny_of(bad)


# ----------------------------------------------------------------------------- the criteria on constructed results


def _set(mean_kl=1e-8, csort=1e-10, leads=(), kls=(1e-7,), gaps=(), pairs=1000):
    return {"n": 1000, "mean_kl": mean_kl, "csort_mean": csort, "csort_max": csort * 10,
            "top1_change_leads": rules.tail(leads), "kl": rules.tail(kls), "shadow_gaps": rules.tail(gaps),
            "shadow_pairs": pairs}


def _buckets(means=(1e-8, 1e-8, 1e-8)):
    return {f"{lo}-{hi}": {"lo": lo, "hi": hi, "n": hi - lo, "mean_kl": m, "mean_S": 1e-3, "csort_mean": 1e-10}
            for (lo, hi), m in zip(SECTION_5_1["buckets"], means)}


def _series(t18=None, t9=None, buckets=None):
    return {"sets": {"T1-8": t18 or _set(), "T9": t9 or _set()}, "t9_buckets": buckets or _buckets()}


def _mutants():
    return {1: _series(t18=_set(gaps=[0.5] * 12)), 6: _series(t18=_set(mean_kl=1e-4)),
            19: _series(buckets=_buckets(means=(1e-8, 1e-4, 1e-4)))}


def _criteria(base, mutants, f_tiny, th):
    return ftiny.validation_criteria(rules.g4_f32(base, mutants, f_tiny, th, rules.load_controls()))


def _unmet(res):
    return [c["name"] for c in res["criteria"] if not c["met"]]


def test_criteria_at_their_boundaries(th):
    f_tiny = 1e-6  # tau = F_tiny (100 x Csort = 1e-8 is below it)
    res = _criteria(_series(), _mutants(), f_tiny, th)
    assert res["all_met"], _unmet(res)
    names = {c["name"] for c in res["criteria"]}
    assert {"F1 silent", "F3 silent", "F5 silent", "F6 silent"} <= names
    assert {f"control {cid} caught with the §4.2 margin" for cid in SECTION_5_1["controls"]} <= names
    # F2: a set mean exactly at tau / 10 meets the margin; one ulp above does not (the rule itself still passes).
    bound = f_tiny / 10.0
    at = _criteria(_series(t9=_set(mean_kl=bound)), _mutants(), f_tiny, th)
    assert at["all_met"], _unmet(at)
    above = _criteria(_series(t9=_set(mean_kl=float(np.nextafter(bound, 1.0)))), _mutants(), f_tiny, th)
    assert _unmet(above) == ["F2 T9 mean <= tau / 10"]
    # (B0 raised too, so that F3's growth leg, B2 / B0 > 10, stays silent.)
    bucket = _criteria(_series(buckets=_buckets(means=(5e-8, 1e-8, 2 * bound))), _mutants(), f_tiny, th)
    assert _unmet(bucket) == ["F2 T9:8192-16384 mean <= tau / 10"]
    # A fired leg is unmet, and the rules state with it.
    f6 = _criteria(_series(t9=_set(gaps=[0.5])), _mutants(), f_tiny, th)
    assert set(_unmet(f6)) == {"state", "F6 silent"}
    f1 = _criteria(_series(t18=_set(leads=[2.0])), _mutants(), f_tiny, th)
    assert set(_unmet(f1)) == {"state", "F1 silent"}
    f5 = _criteria(_series(t18=_set(kls=[0.02])), _mutants(), f_tiny, th)
    assert set(_unmet(f5)) == {"state", "F5 silent"}


def test_criteria_need_each_control_with_its_margin(th):
    f_tiny = 1e-6
    # Mutant 6 fires F2 but only 5x above tau: caught, margin not met (§4.2 wants 10x).
    weak = {**_mutants(), 6: _series(t18=_set(mean_kl=5 * f_tiny))}
    res = _criteria(_series(), weak, f_tiny, th)
    assert _unmet(res) == ["control G4F32/6 caught with the §4.2 margin"]
    # Mutant 1 caught by F6 on 9 pairs: below the count margin of 10.
    few = {**_mutants(), 1: _series(t18=_set(gaps=[0.5] * 9))}
    assert _unmet(_criteria(_series(), few, f_tiny, th)) == ["control G4F32/1 caught with the §4.2 margin"]
    # Mutant 19 not caught: INCOMPLETE, so the state is unmet too.
    blind = {**_mutants(), 19: _series()}
    assert set(_unmet(_criteria(_series(), blind, f_tiny, th))) == {"state", "control G4F32/19 caught with the §4.2 margin"}


def test_module_emits_no_verdict_flag(record):
    """ftiny measures: its record keeps rules.py's state, never the exp_036-style boolean."""
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)

    assert "pass" not in set(keys(record))


# ----------------------------------------------------------------------------- the checkpoints


@pytest.mark.parametrize("which", ["calibration", "validation"])
def test_bf16_checkpoints_are_the_recorded_ones(record, which, tiny_real_cal, tiny_real_val):
    tset = {"calibration": tiny_real_cal, "validation": tiny_real_val}[which]
    got = ftiny.checkpoint_record(tset.bf16, tset.k8, tset.seed, tset.sharpen)
    want = record[which]["checkpoint"]
    assert got["seed"] == want["seed"] and got["layout"] == want["layout"]
    assert got["bf16"] == want["bf16"]


@pytest.mark.parametrize("which", ["calibration", "validation"])
def test_k8_builds_are_the_recorded_ones(record, which, tiny_real_cal, tiny_real_val):
    """The K8 conversion is MLX's quantisation on this host's GPU: compared on
    the host that wrote the record (the mini)."""
    if ftiny.environment()["host"] != record["provenance"]["environment"]["host"]:
        pytest.skip("build-host only: the K8 shards are compared on the host that wrote gate/calibration.json")
    tset = {"calibration": tiny_real_cal, "validation": tiny_real_val}[which]
    got = ftiny.checkpoint_record(tset.bf16, tset.k8, tset.seed, tset.sharpen)
    assert got["k8"] == record[which]["checkpoint"]["k8"]


# ----------------------------------------------------------------------------- G1 item 7: the validation re-run


@pytest.fixture(scope="module")
def validation(record, work37, tiny_real_val):
    return ftiny.validate(tiny_real_val.bf16, tiny_real_val.k8, ftiny.load_f_tiny(), log=lambda m: None)


def test_validation_rerun_meets_every_criterion(validation, record, th):
    """DESIGN §5.1 / §3.3 G1 item 7: the correct port passes every rule with F2
    at >= 10x margin, and each required control is caught with its margin."""
    assert validation["f_tiny"] == record["f_tiny"] <= th["G4F32"]["f_tiny_cap"]
    check = validation["check"]
    assert check["state"] == rules.PASS, check.get("reason")
    for leg in ("F1", "F3", "F5", "F6"):
        assert check["legs"][leg]["fired"] is False, (leg, check["legs"][leg])
    per = check["legs"]["F2"]["per"]
    assert set(per) == {"T1-8", "T9", "T9:0-2048", "T9:2048-8192", "T9:8192-16384"}
    for where, v in per.items():
        assert v["tau"] == max(record["f_tiny"], th["G4F32"]["tau_csort_factor"] *
                               validation["csort"]["T9" if where.startswith("T9") else where]["mean"])
        assert v["mean_kl"] <= v["tau"] / 10.0, (where, v)
    for cid, (leg, mutant) in SECTION_5_1["controls"].items():
        c = check["controls"][cid]
        assert (c["leg"], c["mutant"], c["caught"]) == (leg, mutant, True), c
        assert c["margin"]["met"] and c["margin"]["required"] == 10.0, c
    assert validation["all_met"], [c["name"] for c in validation["criteria"] if not c["met"]]
    if _same_conditions(record):  # the recorded host, packages, port and reference tree: bitwise the record
        rec = record["validation"]
        assert json.loads(common.dumps(validation["summary"])) == rec["summary"]
        assert json.loads(common.dumps(validation["tau"])) == rec["tau"]
        assert json.loads(common.dumps(validation["csort"])) == rec["csort"]


# ----------------------------------------------------------------------------- the calibration re-run


def test_calibration_rerun(record, work37, tiny_real_cal, th):
    """The calibration on the seed-23 set stays within the cap; under the
    recorded conditions it reproduces the recorded F_tiny and block means bitwise."""
    cal = ftiny.calibrate(tiny_real_cal.bf16, tiny_real_cal.k8, log=lambda m: None)
    assert cal["cap"] == th["G4F32"]["f_tiny_cap"]
    assert cal["cap_met"] and cal["f_tiny"] <= cal["cap"]
    assert cal["f_tiny"] == ftiny.F_TINY_FACTOR * max(cal["block_means"])
    assert cal["n_positions"] == 16384 and len(cal["block_means"]) == 8
    if _same_conditions(record):
        assert cal["f_tiny"] == record["f_tiny"]
        assert cal["block_means"] == record["block_means"]
        assert cal["references"]["i_sha256"] == record["calibration"]["measured"]["references"]["i_sha256"]
