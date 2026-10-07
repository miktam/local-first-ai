# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5b).
"""G4-F32, G4-F16 and G4-N(i) measurement (gate/checks/g4_forced.py,
gate/checks/g4_e2e.py; DESIGN §3.7-§3.10, §4.2, §5.2; §3.3 G1 item 3, last
bullet). Light: a small layout on the tiny real-layout checkpoint.

* synthetic series: the reductions build rules.py's measured dicts exactly
  (means, tails, buckets, Csort fields, the G4-F16 descriptive rows, the F4
  windows), and rules.g4_f32 / rules.g4_f16 read them;
* the mutant loop: controls.json's lists, each mutant loaded once, bf16 runs
  before the in-place fp32 conversion, memory released around every mutant;
* on the tiny build: the K8 fp32 port forced onto I against R2F passes every
  G4-F32 leg with tau at F_tiny's cap, its three required controls (mutants 1,
  6, 19) are caught with the 10x margin, and the weight-path mutants 6, 7 and
  14 fail G4-F32 under ForcedTrunk; G4-F16's series reach rules.g4_f16 and
  mutant 6 fires the kappa rule; g4_e2e is measurement only and its output is
  rules.g4_n's input;
* fp32_on_dequantised_weights is exp_036's diaglib.py l.247-279 verbatim.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from gate import common, rules
from gate.checks import g4_e2e, ref_pass
from gate.checks import g4_forced as g4
from gate.checks import ref_drivers as rd
from gate.harness import HarnessError
from gate.textset import Profile, TextSet

TL, T9L, CH = 96, 768, 256  # T1-8 analogue (one forward each), T9 analogue (3 forwards, past the 513 window)
BUCKETS = ((0, 256), (256, 512), (512, 768))
L, K = 10, 6
F_TINY = 1e-5  # F_tiny's cap (DESIGN §5.1); calibration.json is W5d's
NO_CONTROLS = {"required": [], "margin_factor": 10}  # a rule judged without its controls (synthetic)


@pytest.fixture(scope="module")
def th():
    return common.load_thresholds()


def _no_pass_key(obj) -> bool:
    if isinstance(obj, dict):
        return "pass" not in obj and all(_no_pass_key(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_no_pass_key(v) for v in obj)
    return True


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------


def _series(n8=4, n9=12, layers=2, seed=0):
    """A forced series by hand: T1..T8 of n8 positions, T9 of n9."""
    rng = np.random.default_rng(seed)
    names = list(g4.ORDER8) + ["T9"]
    cols = {}
    for nm in names:
        n = n9 if nm == "T9" else n8
        cols[nm] = {"kl": rng.uniform(1e-9, 2e-9, n), "lead": rng.uniform(0.0, 1.0, n), "agree": np.ones(n, bool),
                    "S": np.full(n, 0.5), "r3_agree": np.ones(n, bool), "shadow_dis": np.zeros((layers, n), bool),
                    "gap": rng.uniform(0.0, 1e-3, (layers, n)).astype(np.float32),
                    "r3_shadow_dis": np.zeros((layers, n), bool)}
    return {"kind": "g4_forced_series", "names": names, "sets": dict(g4.STANDARD_SETS),
            "pack": g4.pack_spans([(nm, np.zeros(cols[nm]["kl"].size)) for nm in names]), "num_layers": layers,
            "top_k": 6, "chunk": 4, "seconds": 0.0, "cols": cols}


def _csort(series, value=1e-10):
    return {"kind": "csort_series", "names": series["names"], "pack": series["pack"], "chunk": series["chunk"],
            "seconds": 0.0, "cols": {nm: {"kl": np.full(c["kl"].size, value)} for nm, c in series["cols"].items()}}


SYN_BUCKETS = ((0, 4), (4, 8), (8, 12))


def test_f32_reduction_matches_the_rules_contract(th):
    s = _series()
    cs = _csort(s)
    m = g4.g4_f32_measured(s, cs, buckets=SYN_BUCKETS, th=th)
    assert _no_pass_key(m)
    json.loads(common.dumps(m))
    t18 = m["sets"]["T1-8"]
    kl8 = np.concatenate([s["cols"][t]["kl"] for t in g4.ORDER8])
    assert t18["n"] == 32 and t18["mean_kl"] == pytest.approx(kl8.mean()) and t18["kl"]["n"] == 32
    assert t18["kl"]["top"] == sorted(kl8.tolist(), reverse=True)[:rules.TAIL_K]
    assert t18["top1_change_leads"] == {"n": 0, "top": []} and t18["shadow_gaps"] == {"n": 0, "top": []}
    assert t18["shadow_pairs"] == 2 * 32 and t18["mean_S"] == 0.5
    assert t18["csort_mean"] == pytest.approx(1e-10) and t18["csort_max_share"] == pytest.approx(1 / 32)
    assert set(m["t9_buckets"]) == {"0-4", "4-8", "8-12"}
    b = m["t9_buckets"]["4-8"]
    assert (b["lo"], b["hi"], b["n"]) == (4, 8, 4) and b["mean_kl"] == pytest.approx(s["cols"]["T9"]["kl"][4:8].mean())
    assert b["mean_S"] == 0.5 and b["csort_mean"] == pytest.approx(1e-10)
    assert set(m["descriptive"]["per_text"]) == set(s["names"]) and "F4" in m["descriptive"]
    res = rules.g4_f32(m, None, F_TINY, th, controls=NO_CONTROLS)
    assert res["state"] == rules.PASS, res
    # A mutant series carries no Csort; rules read Csort from the unmutated run.
    mm = g4.g4_f32_measured(s, None, buckets=SYN_BUCKETS, th=th)
    assert "csort_mean" not in mm["sets"]["T9"] and "csort_mean" not in mm["t9_buckets"]["0-4"]
    assert "F4" not in mm["descriptive"]
    cal = rules.g4_f32_calibration(m, F_TINY, th)
    assert not any(v["fired"] for v in rules.g4_f32_legs(mm, cal, th).values())


def test_f32_reduction_feeds_each_leg(th):
    cal = rules.g4_f32_calibration(g4.g4_f32_measured(_series(), _csort(_series()), buckets=SYN_BUCKETS, th=th),
                                   F_TINY, th)

    def legs(edit):
        s = _series()
        edit(s["cols"])
        return {k: v["fired"] for k, v in rules.g4_f32_legs(g4.g4_f32_measured(s, None, buckets=SYN_BUCKETS, th=th),
                                                            cal, th).items()}

    def f1(c):  # a top-1 change where R2F leads by 2 nats; one at 1.9 does not count
        c["T3"]["agree"][1] = False
        c["T3"]["lead"][1] = 2.0
        c["T9"]["agree"][2] = False
        c["T9"]["lead"][2] = 1.9

    def f2(c):
        c["T9"]["kl"][9] = 12 * 1e-5 * 4  # the 8-12 bucket's mean above tau

    def f5(c):
        c["T1"]["kl"][0] = 0.011

    def f6(c):  # a shadow disagreement where R2F's gap is 0.02 (gaps are fp32: 1e-2 itself rounds below)
        c["T2"]["shadow_dis"][1, 3] = True
        c["T2"]["gap"][1, 3] = 0.02

    assert legs(f1) == {"F1": True, "F2": False, "F3": False, "F5": False, "F6": False}
    assert legs(f2)["F2"] and not legs(f2)["F1"]
    assert legs(f5)["F5"] and legs(f5)["F2"]
    assert legs(f6) == {"F1": False, "F2": False, "F3": False, "F5": False, "F6": True}
    s = _series()
    f6(s["cols"])
    t = g4.g4_f32_measured(s, None, buckets=SYN_BUCKETS, th=th)["sets"]["T1-8"]
    assert t["shadow_gaps"]["n"] == 1 and t["shadow_gaps"]["top"] == [pytest.approx(0.02)]
    assert t["n_shadow_disagree"] == 1 and t["shadow_disagree_rate"] == pytest.approx(1 / 64)
    # The Csort ceiling: a FAIL of G4-F32 (cause not attributed).
    s = _series()
    res = rules.g4_f32(g4.g4_f32_measured(s, _csort(s, 2e-6), buckets=SYN_BUCKETS, th=th), None, F_TINY, th,
                       controls=NO_CONTROLS)
    assert res["state"] == rules.FAIL and rules.CSORT_CEILING_REASON in res["reason"]


def test_f32_reduction_refuses_inconsistent_inputs(th):
    s = _series()
    cs = _csort(s)
    cs["chunk"] = 8
    with pytest.raises(HarnessError, match="Csort"):
        g4.g4_f32_measured(s, cs, buckets=SYN_BUCKETS, th=th)
    with pytest.raises(ValueError, match="no position"):
        g4.g4_f32_measured(s, None, buckets=((0, 4), (4, 12), (12, 20)), th=th)


def test_f16_reduction(th):
    s = _series()
    for c in s["cols"].values():
        c["S"][:] = 0.005
    c9 = s["cols"]["T9"]
    c9["kl"][:] = 0.02
    c9["kl"][0] = 1.5  # a sparse spike: port above 1 nat where R3 is below 0.1
    c9["S"][1] = 0.05
    c9["lead"][:3] = 2.5  # three decisive positions
    c9["agree"][0] = False
    c9["r3_agree"][1:3] = False
    c9["shadow_dis"][0, :6] = True
    c9["r3_shadow_dis"][1, :3] = True
    m = g4.g4_f16_measured(s, th=th)
    assert _no_pass_key(m)
    json.loads(common.dumps(m))
    r = m["T9"]
    assert r["n"] == 12 and r["mean_kl_port"] == pytest.approx(c9["kl"].mean()) and r["mean_kl_r3"] == pytest.approx(c9["S"].mean())
    assert r["port_over_r3"] == pytest.approx(c9["kl"].mean() / c9["S"].mean())
    assert r["p99_ratio"] == pytest.approx(np.quantile(c9["kl"], 0.99) / np.quantile(c9["S"], 0.99))
    assert (r["n_decisive"], r["decisive_miss_port"], r["decisive_miss_r3"], r["n_sparse_spikes"]) == (3, 1, 2, 1)
    assert (r["shadow_pairs"], r["shadow_disagree_port"], r["shadow_disagree_r3"]) == (24, 6, 3)
    assert r["shadow_rate_port"] == 0.25 and r["shadow_rate_r3"] == 0.125
    res = rules.g4_f16(m, None, th, controls=NO_CONTROLS)
    assert res["state"] == rules.FAIL and res["legs"]["kappa"]["per"]["T9"]["fired"]  # 0.143 > 9 x 0.00875
    assert not res["legs"]["kappa"]["per"]["T1-8"]["fired"]
    c9["S"][:] = 0.02
    res = rules.g4_f16(g4.g4_f16_measured(s, th=th), None, th, controls=NO_CONTROLS)
    assert res["state"] == rules.INCOMPLETE and res["reason"].startswith("VOID")  # R3 above its 1e-2 ceiling
    for nm in s["cols"]:
        s["cols"][nm]["S"] = None
    with pytest.raises(HarnessError, match="R3"):
        g4.g4_f16_measured(s, th=th)


def test_f4_windows():
    T = 256
    kl = np.full(T, 1e-9)
    kl[60:70] = 1e-6  # a burst at the chunk start 64
    S = np.ones(T)
    S[128:192] = 2.0
    cs = np.full(T, 1e-10)
    c = {"kl": kl, "S": S, "lead": np.zeros(T), "agree": np.ones(T, bool)}
    f4 = g4.f4_windows(c, cs, (), doc_starts=(100, 0, 300), chunk=64, half=8)
    assert f4["chunk_starts"] == [64, 128, 192] and f4["document_starts"] == [100]
    rows = {r["boundary"]: r for r in f4["windows"]}
    assert set(rows) == {64, 100, 128, 192}
    assert rows[64]["kinds"] == ["chunk"] and rows[100]["kinds"] == ["document"]
    w = rows[64]
    assert (w["lo"], w["hi"]) == (56, 72)
    assert w["mean_kl"] == pytest.approx(kl[56:72].mean()) and w["rho"] == pytest.approx(kl[56:72].mean())
    # the other 16-position tiles: [0, 16), ..., [240, 256) minus [48, 64) and [64, 80), which overlap [56, 72)
    others = [kl[a:a + 16].mean() / S[a:a + 16].mean() for a in range(0, 256, 16) if a + 16 <= 56 or a >= 72]
    assert w["rho_others_median"] == pytest.approx(np.median(others))
    assert w["rho_ratio"] == pytest.approx(w["rho"] / np.median(others)) and w["rho_ratio"] > 100
    assert w["kl_over_csort"] == pytest.approx(kl[56:72].mean() / 1e-10)
    assert rows[128]["rho_ratio"] < 1 and _no_pass_key(f4)


def test_layout_helpers():
    assert g4.block_means(np.arange(10.0), 4) == [1.5, 5.5, 8.5]
    assert g4.chunk_boundaries(10, 4) == [4, 8] and g4.chunk_boundaries(10, [(0, 3), (3, 10)]) == [3]
    assert g4.pack_spans([("a", [1, 2]), ("b", [3])]) == {"a": (0, 2), "b": (2, 3)}
    assert g4.default_sets(list(g4.ORDER8) + ["T9"]) == {"T1-8": g4.ORDER8, "T9": ("T9",)}
    assert g4.default_sets(["cal"]) == {"cal": ("cal",)}


def test_t9_document_starts():
    t9 = np.arange(100)

    class TS:
        pass

    ts = TS()
    ts.t9 = t9
    man = {"ids_sha256": common.ids_sha256(t9), "separator": "\n\n",
           "components": [{"n_tokens": 30}, {"n_tokens": 40}, {"n_tokens": 50}, {"n_tokens": 9}]}
    assert g4.t9_document_starts(ts, manifest=man, sep_tokens=2) == [32, 74]  # the 4th starts past T9's end
    assert g4.t9_document_starts(ts, manifest=dict(man, ids_sha256="0" * 64), sep_tokens=2) == []
    ts.t9 = np.arange(16)  # a tiny or synthetic T9 is not the manifest's: no documents, no tokenizer needed
    assert g4.t9_document_starts(ts) == []


# ---------------------------------------------------------------------------
# The mutant loop
# ---------------------------------------------------------------------------


class _Model:
    def __init__(self, name):
        self.name, self.fp32 = name, False


def test_control_plan_is_controls_json():
    plan = g4.control_plan(g4.G4_CONTROL_CHECKS)
    assert [(e["mutant"], e["name"]) for e in plan] == [(1, "sigmoid_bias_select"), (6, "renorm_topk"),
                                                        (19, "rope_restart_per_chunk")]
    runs = {e["mutant"]: [(r["check"], r["precision"], r["leg"]) for r in e["runs"]] for e in plan}
    assert runs == {1: [("g4_f32_K8", "fp32", "F6")],
                    6: [("g4_f16_K8", "bf16", "kappa"), ("g4_f32_K8", "fp32", "F2")],
                    19: [("g4_f32_K8", "fp32", "F2")]}
    assert [e["mutant"] for e in g4.control_plan(["g4_f16_K8"])] == [6]
    with pytest.raises(ValueError, match="no control"):
        g4.control_plan(["g4_n_K8"])
    bad = rules.load_controls()
    bad["required"] = [dict(c, mutant_name="renorm") if c["mutant"] == 6 else c for c in bad["required"]]
    with pytest.raises(ValueError, match="port_mutants"):
        g4.control_plan(["g4_f16_K8"], bad)


def test_mutant_loop_order_conversion_and_release():
    events = []

    def load(name):
        events.append(("load", name))
        return _Model(name), f"module:{name}"

    def to_fp32(model):
        events.append(("fp32", model.name))
        model.fp32 = True

    def measure(check):
        def run(model, module):
            assert module == f"module:{model.name}"
            events.append((check, model.name, "fp32" if model.fp32 else "bf16"))
            return {"value": len(events), "meta": {"kind": check}}
        return run

    out = g4.mutant_loop(g4.G4_CONTROL_CHECKS, load, {c: measure(c) for c in g4.G4_CONTROL_CHECKS},
                         to_fp32=to_fp32, release=lambda: events.append(("release",)), log=lambda m: None)
    assert events == [("release",), ("load", "sigmoid_bias_select"), ("fp32", "sigmoid_bias_select"),
                      ("g4_f32_K8", "sigmoid_bias_select", "fp32"), ("release",),
                      ("release",), ("load", "renorm_topk"), ("g4_f16_K8", "renorm_topk", "bf16"),
                      ("fp32", "renorm_topk"), ("g4_f32_K8", "renorm_topk", "fp32"), ("release",),
                      ("release",), ("load", "rope_restart_per_chunk"), ("fp32", "rope_restart_per_chunk"),
                      ("g4_f32_K8", "rope_restart_per_chunk", "fp32"), ("release",)]
    assert set(out["g4_f32_K8"]) == {1, 6, 19} and set(out["g4_f16_K8"]) == {6}
    m6 = out["g4_f32_K8"][6]
    assert m6["meta"] == {"kind": "g4_f32_K8", "mutant": "renorm_topk", "mutant_id": 6, "control": "G4F32/6",
                          "precision": "fp32"}
    # A failing measure propagates, and the mutant is still released.
    events.clear()

    def boom(model, module):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        g4.mutant_loop(["g4_f16_K8"], load, {"g4_f16_K8": boom}, to_fp32=to_fp32,
                       release=lambda: events.append(("release",)), log=lambda m: None)
    assert events == [("release",), ("load", "renorm_topk"), ("release",)]
    with pytest.raises(ValueError, match="no measure"):
        g4.mutant_loop(g4.G4_CONTROL_CHECKS, load, {"g4_f16_K8": boom})
    with pytest.raises(ValueError, match="not a G4 forced check"):
        g4.g4_mutants(load, None, [0], None, None, checks=("g5_d32_K8",), ref={})


# ---------------------------------------------------------------------------
# The tiny build
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def work37(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EXP036_WORK", str(tmp_path_factory.mktemp("work")))
        yield common.work37_dir()


@pytest.fixture(scope="module")
def ts():
    rng = np.random.default_rng(6)
    ids = {t: [int(x) for x in rng.integers(0, 1008, TL)] for t in g4.ORDER8}
    t9 = [int(x) for x in rng.integers(0, 1008, T9L)]
    prof = Profile(name="w5b-tiny", text_len=TL, t9_len=T9L, t9_buckets=BUCKETS, prefill_chunk=CH, window_position=48)
    return TextSet(prof, ids, t9, [], {}, "w5b-tiny", {"tiny": True})


@pytest.fixture(scope="module")
def refs(work37, tiny_real_cal, ts):
    """R1 on the BF16 checkpoint (natural) -> I; R2F and R3 on the K8 build's
    dequantised weights, forced onto I."""
    pytest.importorskip("mlx.core")
    texts = g4.text_layout(ts)
    seqs = [x for _, x in texts]
    pack = np.concatenate(seqs)
    segs = ref_pass.segments_of([x.size for x in seqs])
    r1 = work37 / "ref" / "w5b" / "fp32"
    ref_pass.fp32_dump(tiny_real_cal.bf16, seqs, r1, "w5b-tiny", log=lambda m: None)
    I, _ = rd.i_from_r1(r1)
    dumps = {}
    for mode in rd.MODES:
        key = rd.derived_key(tiny_real_cal.k8, pack, segs, I, mode)
        rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, pack, segs, I, mode, rd.forced_dump_dir(key), key=key,
                           log=lambda m: None)
        dumps[mode] = rd.ForcedDump(rd.forced_dump_dir(key))
    prof = g4.reference_profile(texts, dumps["fp32"], dumps["emu"], num_layers=L, log=lambda m: None)
    return {"I": I, "R2F": dumps["fp32"], "R3": dumps["emu"], "r1": r1, "ref": prof}


@pytest.fixture(scope="module")
def k8_bf16(tiny_real_cal):
    model, _, module = common.load_port(tiny_real_cal.k8)
    return model, module


@pytest.fixture(scope="module")
def f32_base(tiny_real_cal, ts, refs, th):
    model, _, module = common.load_port(tiny_real_cal.k8)
    g4.make_fp32_port(model)
    return g4.g4_f32_run(model, ts, refs["I"], refs["R2F"], refs["R3"], CH, module=module, ref=refs["ref"], th=th,
                         with_series=True, log=lambda m: None)


def test_fp32_on_dequantised_weights_is_exp036s_verbatim():
    src = (common.EXP_DIR.parent / "exp_036_kolibri_local_eval" / "diagnostics" / "gate1" / "diaglib.py")
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)[246:279]  # l.247-279
    assert lines[0].startswith("def fp32_on_dequantised_weights(model)")
    assert inspect.getsource(g4.fp32_on_dequantised_weights) == "".join(lines)


def test_make_fp32_port_is_applied_once(tiny_real_cal):
    import mlx.core as mx

    model, _, _ = common.load_port(tiny_real_cal.k8)
    assert not g4.is_fp32_port(model)
    g4.make_fp32_port(model)
    emb = model.model.embed_tokens.inner
    assert g4.is_fp32_port(model) and emb.scales.dtype == mx.bfloat16
    assert model.model.norm.weight.dtype == mx.float32
    g4.make_fp32_port(model)  # a second call leaves the restored bf16 scales alone
    assert model.model.embed_tokens.inner is emb and emb.scales.dtype == mx.bfloat16


def test_forced_series_on_tiny(tiny_real_cal, ts, refs, f32_base):
    _, series, cs = f32_base
    assert series["names"] == list(g4.ORDER8) + ["T9"] and series["num_layers"] == L and series["top_k"] == K
    assert series["pack"]["T9"] == (8 * TL, 8 * TL + T9L) and series["chunk"] == CH
    c9 = series["cols"]["T9"]
    assert c9["kl"].shape == (T9L,) and c9["shadow_dis"].shape == (L, T9L) and c9["gap"].shape == (L, T9L)
    assert np.all(c9["kl"] >= 0) and c9["kl"].mean() <= 1e-6  # the fp32 port on R2F's own weights and routing
    a, b = series["pack"]["T9"]
    assert np.allclose(c9["S"], g4.s_profile(refs["R2F"], refs["R3"], a, b - a))
    assert np.array_equal(c9["lead"], common.top_lead(np.array(refs["R2F"].logits()[a:b])))
    for nm in g4.ORDER8:
        assert cs["cols"][nm]["kl"].shape == (TL,)
    assert np.all(cs["cols"]["T9"]["kl"] >= 0) and cs["cols"]["T9"]["kl"].mean() <= 1e-6


def test_g4_f32_passes_on_tiny_with_its_controls_caught(tiny_real_cal, ts, refs, f32_base, th):
    base, _, _ = f32_base
    assert _no_pass_key(base)
    json.loads(common.dumps(base))
    mutants = g4.g4_mutants(g4.port_loader(tiny_real_cal.k8), ts, refs["I"], refs["R2F"], refs["R3"], CH,
                            ref=refs["ref"], checks=("g4_f32_K8",), th=th, log=lambda m: None)["g4_f32_K8"]
    assert set(mutants) == {1, 6, 19} and all(_no_pass_key(m) for m in mutants.values())
    res = rules.g4_f32(base, mutants, F_TINY, th)
    assert res["state"] == rules.PASS, res
    assert not any(v["fired"] for v in res["legs"].values())
    for cid, c in res["controls"].items():
        assert c["caught"] and c["margin"]["met"], (cid, c)
    for s in ("T1-8", "T9"):  # tau = F_tiny (Csort far below its ceiling); the port's mean >= 10x below tau
        assert res["measured"]["tau"][s] == F_TINY and base["sets"][s]["mean_kl"] <= F_TINY / 10
        assert base["sets"][s]["csort_mean"] <= th["G4F32"]["csort_mean_ceiling"] / 100
    m19 = rules.g4_f32_legs(mutants[19], rules.g4_f32_calibration(base, F_TINY, th), th)["F2"]["per"]
    assert not m19["T1-8"]["fired"] and m19["T9"]["fired"]  # one forward per T1-8 text: no later chunk to restart
    f4 = base["descriptive"]["F4"]
    assert f4["chunk_starts"] == [256, 512] and f4["document_starts"] == []


@pytest.mark.parametrize("mutant", ["renorm_topk", "route_scale_2826", "biased_weights"])
def test_weight_path_mutants_fail_g4_f32_under_forced_trunk(tiny_real_cal, ts, refs, f32_base, th, mutant):
    """G1 item 3, last bullet: mutants 6, 7 and 14 fail G4-F32 (their weights
    come from route(), so forcing the selection leaves them visible)."""
    base, _, _ = f32_base
    model, module = g4.port_loader(tiny_real_cal.k8)(mutant)
    g4.make_fp32_port(model)
    m = g4.g4_f32_run(model, ts, refs["I"], refs["R2F"], refs["R3"], CH, module=module, ref=refs["ref"], csort=False,
                      th=th, log=lambda m: None)
    legs = rules.g4_f32_legs(m, rules.g4_f32_calibration(base, F_TINY, th), th)
    assert legs["F2"]["fired"] and all(v["fired"] for v in legs["F2"]["per"].values())


def test_g4_f16_on_tiny(k8_bf16, tiny_real_cal, ts, refs, th):
    model, module = k8_bf16
    base = g4.g4_f16_run(model, ts, refs["I"], refs["R2F"], refs["R3"], CH, module=module, ref=refs["ref"], th=th,
                         log=lambda m: None)
    assert _no_pass_key(base)
    json.loads(common.dumps(base))
    for s in ("T1-8", "T9"):
        r = base[s]
        assert r["mean_kl_port"] > 0 and r["mean_kl_r3"] > 0
        assert r["port_over_r3"] < th["G4F16"]["kappa"]  # the K8 build in bf16 is near the forced emulation
        names = g4.STANDARD_SETS[s]
        r3d = sum(int(refs["ref"]["cols"][n]["r3_shadow_dis"].sum()) for n in names)
        assert r["shadow_disagree_r3"] == r3d and r["shadow_pairs"] == L * sum(len(ts.ids[n]) if n != "T9" else T9L
                                                                                for n in names)
    muts = g4.g4_mutants(g4.port_loader(tiny_real_cal.k8), ts, refs["I"], refs["R2F"], refs["R3"], CH, ref=refs["ref"],
                         checks=("g4_f16_K8",), th=th, log=lambda m: None)["g4_f16_K8"]
    res = rules.g4_f16(base, muts, th)
    assert res["state"] in rules.STATES and res["controls"]["G4F16/6"]["caught"]  # mutant 6 fires the kappa rule
    with pytest.raises(HarnessError, match="bf16"):
        m2, _, mod2 = common.load_port(tiny_real_cal.k8)
        g4.make_fp32_port(m2)
        g4.g4_f16_run(m2, ts, refs["I"], refs["R2F"], refs["R3"], CH, module=mod2, ref=refs["ref"], th=th)
    with pytest.raises(HarnessError, match="fp32 port"):
        g4.g4_f32_run(model, ts, refs["I"], refs["R2F"], refs["R3"], CH, module=module, ref=refs["ref"], th=th)


def test_csort_sequence(k8_bf16, ts):
    from gate import harness

    model, module = k8_bf16
    ids = np.asarray(ts.t9[:300])
    pos = np.array([299, 5, 140])
    cs = g4.csort_sequence(model, ids, 128, module=module, positions=pos)
    free, rec = harness.free_run(model, module, ids, 128)
    assert np.array_equal(cs["table"], rec["ids"]) and cs["spans"] == [[0, 128], [128, 256], [256, 300]]
    assert harness.bitwise_equal(cs["free_at"], free[pos])
    assert np.all(cs["kl"] >= 0) and np.allclose(cs["kl"][pos], common.kl_rows(free[pos], cs["sorted_at"]))
    with pytest.raises(ValueError, match="positions"):
        g4.csort_sequence(model, ids, 128, module=module, positions=[300])


def test_forced_series_refuses_a_wrong_layout(k8_bf16, ts, refs):
    model, module = k8_bf16
    with pytest.raises(HarnessError, match="I has"):
        g4.forced_series(model, ts, refs["I"][:-1], refs["R2F"], refs["R3"], CH, module=module, ref=refs["ref"])
    texts = g4.text_layout(ts)[:-1] + [("T9", np.asarray(ts.t9[:-1]))]
    with pytest.raises(HarnessError):
        g4.forced_texts(model, texts, refs["I"], refs["R2F"], refs["R3"], CH, module=module)
    with pytest.raises(HarnessError, match="another layout"):
        g4.forced_texts(model, g4.text_layout(ts), refs["I"], refs["R2F"], refs["R3"], CH, module=module,
                        ref=dict(refs["ref"], num_layers=L + 1))


# ---------------------------------------------------------------------------
# G4-N(i): g4_e2e measures, rules.g4_n decides
# ---------------------------------------------------------------------------


def test_g4_e2e_is_measurement_only():
    assert not hasattr(g4_e2e, "evaluate")
    for mod in (g4_e2e, g4):
        assert '"pass"' not in inspect.getsource(mod) and "'pass'" not in inspect.getsource(mod)


def test_g4_e2e_output_is_rules_g4_n_input(k8_bf16, ts, refs, th):
    model, _ = k8_bf16
    res = g4_e2e.e2e(model, ts, ref_pass.Dump(refs["r1"]), th["G4"]["K8_backstop_decisive_lead_nats"],
                     log=lambda m: None)
    assert _no_pass_key(res)
    n = rules.g4_n(res, th)
    assert n["state"] in (rules.PASS, rules.FAIL) and set(n["legs"]) == {"T1-8", "T9"}
    assert n["legs"]["T9"]["mean_kl"] == res["T9"]["mean_kl"]
    assert rules.g4_k4(res, th)["state"] == rules.DESCRIPTIVE
    assert set(res["kl_by_bucket"]) == {f"{lo}-{hi}" for lo, hi in BUCKETS}
