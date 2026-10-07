# SPDX-License-Identifier: MIT
"""Unit tests of the gate's check helpers on constructed cases (BUILD_SPEC §5.3).

exp_037 (DESIGN §9.2; W6): the modules under gate/checks/ only measure, and
every decision is gate/rules.py's. Each test here drives the measurement
helper and then applies the rule from rules.py.

G0: the bf16-exactness measure, converted-config policy checks, census.
G2: the self-calibrated bf16 rule. G3: the mutant bootstrap and its
classification, and the peer bpb reader (G3a is descriptive in exp_037).
G4-N(i): the registered backstop's KL leg (decisive top-1 now descriptive).
G5: the loop detector, the greedy rule, the behaviour rule and its run_cell
plumbing, and decode vs prefill catching a decode-only cache defect while
passing the port.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling


# --------------------------------------------------------------------- G0


def test_bf16_exact_fraction():
    from gate import common

    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000).astype(np.float32)
    assert common.bf16_exact_fraction(x) < 1e-3  # fp32 compute: about 2**-16
    xb = (x.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
    assert common.bf16_exact_fraction(xb) == 1.0
    assert np.isnan(common.bf16_exact_fraction(np.array([np.nan], np.float32)))


@pytest.fixture(scope="module")
def tiny_models(tmp_path_factory):
    import_sibling("port.convert")
    from test_gate_drivers_tiny import build_tiny_models

    return build_tiny_models(tmp_path_factory.mktemp("gate_checks"), preset="vendor")


def test_converted_config_policy(tiny_models):
    from gate import common
    from gate.checks import g0_static

    th0 = common.load_thresholds()["G0"]
    for arm, bits in (("K8", 8), ("K4", 4)):
        d = tiny_models / common.ARM_DIRS[arm]
        ok = g0_static.converted_config_check(d, bits, "quantised_head", th0)
        assert ok["n_problems"] == 0, ok["problems"]
        assert ok["observed_policy"] == {"embed_tokens": f"{bits}bit", "lm_head": f"{bits}bit"}
        bad = g0_static.converted_config_check(d, bits, "vendor_faithful", th0)
        assert bad["n_problems"] > 0 and any("signed-off" in p for p in bad["problems"])
        wrong_bits = g0_static.converted_config_check(d, 12 - bits, "quantised_head", th0)
        assert any("quantization block" in p for p in wrong_bits["problems"])


def test_census_counts_on_tiny(tiny_models):
    from reference.kolibri_ref import expected_shapes, load_config, param_count

    from gate.checks import g0_static

    cfg = load_config(str(tiny_models / "Kolibri-1-BF16"))
    exp = {"tensor_count": len(expected_shapes(cfg)), "param_count": param_count(cfg),
           "full_attention_layers": [4, 5]}
    c = g0_static.census(tiny_models / "Kolibri-1-BF16", exp, hash_shards=False)
    assert (c["tensors"], c["params"], c["full_attention_layers"]) == (exp["tensor_count"], exp["param_count"], [4, 5])
    assert c["shard_sha"]["status"] == "unavailable"


def test_census_checks_shards_against_etags(tiny_models, tmp_path):
    """Shard sha256 against the hf-download metadata etag (line 2)."""
    import shutil

    from gate import common
    from gate.checks import g0_static

    src = tmp_path / "src"
    shutil.copytree(tiny_models / "Kolibri-1-BF16", src)
    meta = src / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    shards = sorted(p.name for p in src.glob("model-*.safetensors"))
    for i, s in enumerate(shards):
        etag = common.sha256_file(src / s) if i else "0" * 64
        (meta / f"{s}.metadata").write_text(f"commit\n{etag}\n0\n")
    c = g0_static.census(src, {}, results_dir=tmp_path, hash_shards=True)
    assert c["shard_sha"]["status"] == "checked" and c["shard_sha"]["mismatches"] == [shards[0]]


# --------------------------------------------------------------------- G2


def test_bf16_rule_is_self_calibrated():
    from gate import common, rules

    th = common.load_thresholds()["G2"]
    emu = [{"attn": {"median": 0.01, "p99": 0.02}, "moe": {"median": 0.01, "p99": 0.03}}]
    rows = [{"layer": 0, "branch": "attn", "median": 0.029, "p99": 0.049},   # within 3x emu and the 5e-2 floor
            {"layer": 0, "branch": "moe", "median": 0.031, "p99": 0.05}]    # median above 3x emu
    ok, bad, verdicts = rules.bf16_rule(rows, emu, th)
    assert not ok and bad == [{"layer": 0, "branch": "moe", "median": 0.031, "p99": 0.05}]
    assert verdicts[0]["pass"] and verdicts[0]["limit_p99"] == 0.06 and verdicts[1]["limit_median"] == pytest.approx(0.03)
    assert "pass" not in rows[0]  # the rule never writes into the measured rows


# --------------------------------------------------------------------- G3


def test_mutant_bootstrap_verdicts():
    """g3_oracle measures the 48-block bootstrap; rules.py classifies (exp_036
    g3_oracle.py l.95-100) and judges G3b as registered."""
    from gate import rules
    from gate.checks import g3_oracle

    rng = np.random.default_rng(1)
    base = rng.normal(2.0, 0.5, 1535)
    nll = {"ref": base, "worse": base + 0.5, "better": base - 0.5,
           "same": base + rng.normal(0.0, 0.3, 1535)}  # zero-mean noise: not resolvable
    out = g3_oracle.ref_mutant_nll(nll, 48, 0.01, B=2000)
    cls = {k: rules.g3_mutant_class(v["dnll_p01"], v["dnll_p99"]) for k, v in out.items()}
    assert cls == {"worse": "reference_better", "better": "mutant_better",
                   "same": "not_resolvable (rests on vendor source)"}
    th = {"ref_bpb_per_text_max": 1.2, "ref_bpb_mean_vs_best_peer_max": 1.25, "ref_mutant_text": "T3",
          "ref_mutant_blocks": 48, "ref_mutant_dnll_quantile": 0.01, "ref_mutants": 3}
    c = rules.g3_checks(None, None, out, th)["g3_ref_mutants"]
    assert c["state"] == "FAIL" and c["pass"] is False and c["measured"]["mutant_better"] == ["better"]
    assert c["measured"]["mutants"]["worse"]["resolvable"] and not c["measured"]["mutants"]["same"]["resolvable"]
    c = rules.g3_checks(None, None, {k: v for k, v in out.items() if k != "better"}, dict(th, ref_mutants=2))
    assert c["g3_ref_mutants"]["state"] == "PASS"  # "not resolvable" does not fail the gate


def test_peer_bpb_reader_and_rule(tmp_path):
    """The peer bpb reader (measurement) and G3a, descriptive in exp_037 (§3.6):
    exp_036's limits are printed beside the values and never fail the gate."""
    from gate import rules
    from gate.checks import g3_oracle

    rec = {"arms": {"G8": {"nll": {"mean_bpb": 0.80, "per_text_bpb": {"T1": 0.7}}},
                    "Q36-8": {"nll": {"mean_bpb": 0.90}}, "G4": {"nll": {"mean_bpb": 0.5}}}}
    (tmp_path / "peers_20261004T090000Z.json").write_text(json.dumps(rec))
    peers = g3_oracle.peer_bpb(tmp_path)
    assert peers["peers"] == {"G8": 0.80, "Q36-8": 0.90}  # G4 is not a G3 peer
    th = {"ref_bpb_per_text_max": 1.2, "ref_bpb_mean_vs_best_peer_max": 1.25, "ref_mutant_text": "T3",
          "ref_mutant_blocks": 48, "ref_mutant_dnll_quantile": 0.01, "ref_mutants": 7}
    bpb = {"per_text_bpb": {f"T{i}": 0.9 for i in range(1, 7)}, "mean_bpb": 0.99}
    c = rules.g3_checks(bpb, peers, None, th)["g3_bpb_vs_peers"]
    assert c["state"] == "DESCRIPTIVE" and c["pass"] is None and c["within_exp036_limit"] is True  # 0.99 <= 1.25 x 0.80
    bpb["mean_bpb"] = 1.01
    c = rules.g3_checks(bpb, peers, None, th)["g3_bpb_vs_peers"]
    assert c["state"] == "DESCRIPTIVE" and c["within_exp036_limit"] is False
    bpb["per_text_bpb"]["T4"] = 1.21
    c = rules.g3_checks(bpb, peers, None, th)["g3_bpb_per_text"]
    assert c["state"] == "DESCRIPTIVE" and c["within_exp036_limit"] is False
    assert "g3_bpb_per_text" not in rules.BLOCKING["K8"] and "g3_bpb_vs_peers" not in rules.BLOCKING["K8"]
    assert g3_oracle.peer_bpb(tmp_path / "none")["peers"] == {}


# --------------------------------------------------------------------- G4


def test_g4_backstop_rule():
    """G4-N(i) (§3.9): the registered backstop's KL leg, mean KL <= 0.10 on T1-8
    and on T9; decisive top-1 is descriptive (the registered 99 % printed beside it)."""
    from gate import common, rules

    th = common.load_thresholds()
    good = {"T1-8": {"mean_kl": 0.02, "top1_decisive": 0.995}, "T9": {"mean_kl": 0.03, "top1_decisive": 0.999}}
    assert rules.g4_n(good, th)["state"] == "PASS"
    bad = {"T1-8": {"mean_kl": 0.02, "top1_decisive": 0.995}, "T9": {"mean_kl": 0.11, "top1_decisive": 0.999}}
    assert rules.g4_n(bad, th)["state"] == "FAIL"
    low_top1 = {"T1-8": {"mean_kl": 0.02, "top1_decisive": 0.98}, "T9": {"mean_kl": 0.01, "top1_decisive": 0.999}}
    c = rules.g4_n(low_top1, th)
    assert c["state"] == "PASS" and c["descriptive_rows"]["decisive_top1"]["T1-8"] == 0.98  # descriptive in exp_037
    assert c["descriptive_rows"]["registered_decisive_top1_min"] == th["G4"]["K8_backstop_decisive_top1_min"]
    assert rules.g4_k4(low_top1, th)["state"] == "DESCRIPTIVE" and rules.g4_k4(low_top1, th)["pass"] is None


# --------------------------------------------------------------------- G5


def test_loop_detector():
    from gate.checks.g5_generation import has_loop

    rng = np.random.default_rng(2)
    plain = rng.integers(0, 1000, 600).tolist()
    assert not has_loop(plain)
    span = rng.integers(0, 1000, 32).tolist()
    assert has_loop(plain[:100] + span * 4 + plain[:50])          # a 32-token span, 4 times
    assert not has_loop(plain[:100] + span * 3 + plain[:50])      # only 3 times
    assert has_loop(plain[:10] + [7, 8, 9] * 50)                  # a shorter cycle over >= 128 tokens
    assert not has_loop([5] * 127)                                # shorter than one window
    assert has_loop([5] * 128)


def test_greedy_rule():
    from gate.checks.g5_generation import greedy_vs_ref

    V = 50
    logits = np.tile(-0.01 * np.arange(V, dtype=np.float32), (10, 1))  # distinct: top-5 = {3, 0, 1, 2, 4}
    logits[:, 3] = 5.0   # token 3 leads by about 5 nats
    logits[5:, 4] = 4.9  # rows 5..9: a near-tie between 3 and 4
    # prompt of 4 tokens: generated token j is predicted by row 3 + j
    conts = [{"id": "p", "prompt_tokens": 4, "continuation": [3] * 6, "finished": "length"}]
    res = greedy_vs_ref(conts, {}, {"p": logits}, 2.0)
    assert res["n_positions"] == 6 and res["decisive_top1"] == 1.0 and res["in_top5"] == 1.0
    assert res["n_decisive"] == 2  # rows 3 and 4
    from gate import common, rules

    th5 = common.load_thresholds()["G5"]
    assert rules.g5_greedy(res, th5)["state"] == "PASS"
    conts[0]["continuation"] = [3, 9, 3, 4, 3, 3]  # 9 at decisive row 4 (outside the top-5); 4 at a near-tie row
    res = greedy_vs_ref(conts, {}, {"p": logits}, 2.0)
    assert res["decisive_top1"] == 0.5 and res["in_top5"] == pytest.approx(5 / 6)
    assert rules.g5_greedy(res, th5)["state"] == "FAIL"


def test_behaviour_through_run_cell(monkeypatch):
    """behaviour() sends the items through runner.generate.run_cell and
    summarises its records; rules.behaviour_verdict applies the >= 2 / 20 rule."""
    gen = import_sibling("runner.generate")
    from gate import common, rules
    from gate.checks import g5_generation

    seen = {}

    def fake_run_cell(model, tokenizer, items, arm, task, effort, cap, B, seed, writer, family, **kw):
        seen.update(arm=arm, task=task, effort=effort, cap=cap, B=B, family=family, eos=kw.get("eos"), n=len(items))
        for j, it in enumerate(items):
            loop = j == 0
            writer.append({"key": {"item": it["id"]}, "completion_ids": ([1, 2] * 80) if loop else list(range(40)),
                           "completion_tokens": 160 if loop else 40,
                           "finish_reason": "length" if j < 2 else "stop", "answer_text": "Die Antwort ist Wien." if j % 2 else "The answer is Lisbon.",
                           "reasoning_status": "closed"})
        return {}

    monkeypatch.setattr(gen, "run_cell", fake_run_cell)
    items = [{"id": f"B{j:02d}", "lang": "en" if j < 10 else "de", "messages": [{"role": "user", "content": "q"}]}
             for j in range(20)]
    lang = lambda text: "de" if "Antwort" in text else "en"  # noqa: E731
    cell = g5_generation.behaviour(object(), object(), items, "K8", "high", 8192, 8, (127906, 127901), lang)
    assert seen == {"arm": "K8", "task": "gate_behaviour", "effort": "high", "cap": 8192, "B": 8, "family": "kolibri",
                    "eos": [127906, 127901], "n": 20}
    assert (cell["n"], cell["n_loop"], cell["n_no_eos"], cell["n_unknown_lang"]) == (20, 1, 2, 0)
    assert cell["think_closed_rate"] == 1.0
    blocked, summary = rules.behaviour_verdict({"high": cell, "none": dict(cell, n_no_eos=1)}, 2)
    assert blocked == ["high"] and set(summary) == {"high", "none"}
    blocked, _ = rules.behaviour_verdict({"high": dict(cell, n_no_eos=1, n_loop=1)}, 2)
    assert blocked == []
    th5 = common.load_thresholds()["G5"]
    c = rules.g5_behaviour({"high": cell, "none": dict(cell, n_no_eos=1)}, th5)
    assert c["state"] == "FAIL" and c["measured"]["blocked_efforts"] == ["high"]


def test_decode_vs_prefill_catches_a_decode_cache_defect(tiny_models):
    import mlx.core as mx

    from gate import common, rules
    from gate.checks import g5_generation as g5

    th5 = common.load_thresholds()["G5"]
    model, _, module = common.load_port(tiny_models / "Kolibri-1-BF16")
    model.set_dtype(mx.float32)
    ids = tc.random_ids(200, seed=5)
    rng = {"T": (70, 190)}  # past the vendor preset's 65-token window
    ok = g5.floor_and_decode(model, {"T": ids}, rng, tuple(th5["noise_floor_prefill_steps"]), log=lambda m: None)
    bound = rules.parity_bound(ok, th5)
    assert ok["decode_kl"] <= bound["kl_max"] and ok["decode_dis"] <= bound["dis_max"]
    assert rules.g5_decode_vs_prefill(ok, th5)["state"] == "PASS"
    orig = model.make_cache
    model.make_cache = lambda: [module.SlidingKVCache(max_size=c.max_size - 1, keep=0)
                                if isinstance(c, module.SlidingKVCache) else c for c in orig()]
    bad = g5.floor_and_decode(model, {"T": ids}, rng, tuple(th5["noise_floor_prefill_steps"]), log=lambda m: None)
    bound = rules.parity_bound(bad, th5)
    assert bad["decode_kl"] > bound["kl_max"]
    assert rules.g5_decode_vs_prefill(bad, th5)["state"] == "FAIL"


def test_batch_parity_on_the_port(tiny_models):
    import mlx.core as mx

    from gate import common
    from gate.checks import g5_generation as g5

    model, _, _ = common.load_port(tiny_models / "Kolibri-1-BF16")
    model.set_dtype(mx.float32)
    prompts = [tc.random_ids(n, seed=n) for n in (7, 30, 70, 110, 12, 52, 90, 15, 9, 64)]
    res = g5.batch_parity(model, prompts, [12, 3, 9, 5, 11, 7, 14, 4, 6, 10], 4, eos=(tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID))
    assert res["max_live"] <= 4 and res["admitted_mid_run"] > 0
    assert res["mean_kl"] < 1e-8 and res["top1_dis"] == 0.0


def test_census_reuses_preflight_hashes_and_checks_committed_oids(tiny_models, tmp_path, monkeypatch):
    """G0 reuses preflight --deep shard hashes of the same HEAD and checks
    every shard against tools/kolibri_lfs_oids.json as well."""
    import shutil

    from gate import common
    from gate.checks import g0_static

    src = tmp_path / "src"
    shutil.copytree(tiny_models / "Kolibri-1-BF16", src)
    meta = src / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    shards = sorted(p.name for p in src.glob("model-*.safetensors"))
    true = {s: common.sha256_file(src / s) for s in shards}
    for s in shards:
        (meta / f"{s}.metadata").write_text(f"commit\n{true[s]}\n0\n")
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(common, "git_head", lambda: "abc")
    rec = {"git": {"head": "abc"}, "assets": {"models": [{"repo": "Aleph-Alpha/Kolibri-1-BF16",
                                                          "deep": {"sha256": dict(true)}}]}}
    (results / "preflight_20261004T080000Z.json").write_text(json.dumps(rec))
    c = g0_static.census(src, {}, results_dir=results, hash_shards=True, oids=dict(true))
    assert c["shard_sha"]["n_reused"] == len(shards) and c["shard_sha"]["mismatches"] == []
    bad_oids = dict(true, **{shards[-1]: "1" * 64})
    c = g0_static.census(src, {}, results_dir=results, hash_shards=True, oids=bad_oids)
    assert c["shard_sha"]["mismatches"] == [shards[-1]]
    rec["git"]["head"] = "other"  # a preflight of another HEAD is not reused
    (results / "preflight_20261004T090000Z.json").write_text(json.dumps(rec))
    (results / "preflight_20261004T080000Z.json").unlink()
    c = g0_static.census(src, {}, results_dir=results, hash_shards=True)
    assert c["shard_sha"]["n_reused"] == 0 and c["shard_sha"]["source"] == "computed"
    assert len(g0_static.lfs_oids()) in (0, 32)  # the committed list covers the 32 BF16 shards
