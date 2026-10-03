# SPDX-License-Identifier: MIT
"""bench/fit.py (D1) and bench/ladder.py (E6, Tier 2) on tiny checkpoints."""

from __future__ import annotations

import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    assert_sorted_keys_jsonl,
    bench_env,
    read_jsonl,
    tiny_loader,
    tiny_models,
)

from bench import common, fit, ladder


def test_d1_constants_match_the_preregistration():
    assert (fit.D1_ARM, fit.D1_SIZES, fit.D1_REPS, fit.D1_GEN_TOKENS) == ("K4", (32768, 65536), 3, 512)
    assert fit.D1_FIXTURE == "pad_120k.txt"


def test_d1_cell_at_preregistered_context_sizes(bench_env, tiny_models):
    calls = []
    path = fit.run_cell(bench_env.results, loader=tiny_loader({"K4": tiny_models["a"]}, calls), reps=1)
    assert common.is_complete(path) and path.name.startswith("fit_")
    assert_sorted_keys_jsonl(path)
    recs = read_jsonl(path)
    runs = [r for r in recs if r.get("kind") == "run"]
    assert [r["n_context"] for r in runs] == [32768, 65536]
    for r in runs:
        assert r["prompt_tokens"] == r["n_context"] and r["generation_tokens"] == 512
        assert r["peak_bytes"] >= r["memory_before"]["active_bytes"] > 0
        assert r["peak_gib"] == pytest.approx(r["peak_bytes"] / 2**30)
        assert r["prefill_step_size"] == 2048 and r["prompt_tps"] > 0
    # KV grows with context, so the 64k peak is the larger one
    assert runs[1]["peak_bytes"] > runs[0]["peak_bytes"]
    assert calls == ["K4"] and bench_env.calls[:2] == [("identity",), ("gate", "K4")]
    summary = recs[-1]["summary"]["median_peak_gib"]
    assert set(summary) == {"32768", "65536"}


def test_d1_three_reps_each_and_prefixes_are_exact(bench_env, tiny_models):
    path = fit.run_cell(bench_env.results, loader=tiny_loader({"K4": tiny_models["a"]}), sizes=(512, 1024), gen_tokens=16)
    recs = read_jsonl(path)
    runs = [r for r in recs if r.get("kind") == "run"]
    assert [(r["n_context"], r["rep"]) for r in runs] == [(512, 0), (512, 1), (512, 2), (1024, 0), (1024, 1), (1024, 2)]
    # exact-length prefixes of one tokenisation: the same ids per size
    by_n = {}
    for r in runs:
        by_n.setdefault(r["n_context"], set()).add(r["prompt_ids_sha256"])
    assert all(len(v) == 1 for v in by_n.values())
    assert recs[0]["prompt_source"]["fixture"] == "pad_120k.txt"


def test_ladder_requires_tier2_and_fixtures_and_restricts_120k(bench_env, tiny_models):
    rungs = (
        ("4k", "pad_4k.txt", 300, None),
        ("15k", "pad_15k.txt", 400, None),
        ("120k", "pad_120k.txt", 500, ("K4",)),
    )
    loader = tiny_loader({"K4": tiny_models["a"], "K8": tiny_models["b"], "G4": tiny_models["a"]})
    path = ladder.run_cell(bench_env.results, loader=loader, rungs=rungs, gen_tokens=8)
    assert common.is_complete(path)
    assert bench_env.calls[:2] == [("identity",), ("tier2",)]
    assert {c for c in bench_env.calls if c[0] == "gate"} == {("gate", "K4"), ("gate", "K8")}
    recs = read_jsonl(path)
    runs = [r for r in recs if r.get("kind") == "run"]
    got = sorted({(r["arm"], r["rung"]) for r in runs})
    assert ("K4", "120k") in got and ("K8", "120k") not in got and ("G4", "120k") not in got
    reps = {}
    for r in runs:
        reps[(r["arm"], r["rung"])] = reps.get((r["arm"], r["rung"]), 0) + 1
    assert reps[("K4", "120k")] == 2 and reps[("K8", "4k")] == 3
    assert all(r["generation_tokens"] == 8 and r["prefill_ms_per_token"] > 0 for r in runs)
    # 60 s idle after each rung of each arm: K4 3 rungs, K8 2, G4 2
    assert bench_env.sleeps == [60] * 7
    assert set(recs[0]["fixtures_sha256"]) == {"pad_4k.txt", "pad_15k.txt", "pad_120k.txt"}


def test_ladder_default_rungs_and_fixture_list():
    labels = [r[0] for r in ladder.RUNGS]
    assert labels == ["4k", "15k", "32k", "64k", "120k"]
    assert dict((r[0], r[2]) for r in ladder.RUNGS) == {"4k": None, "15k": None, "32k": 32768, "64k": 65536, "120k": None}
    assert [r for r in ladder.RUNGS if r[3]] == [("120k", "pad_120k.txt", None, ("K4",))]
    assert ladder.required_fixtures() == ["pad_120k.txt", "pad_15k.txt", "pad_4k.txt"]
    assert ladder.LADDER_ARMS == ("K4", "K8", "G4") and ladder.IDLE_BETWEEN_SIZES_S == 60


def test_ladder_refuses_a_missing_fixture(bench_env, tiny_models):
    with pytest.raises(common.BenchError, match="pad_7k.txt is missing"):
        ladder.run_cell(bench_env.results, loader=tiny_loader({}), rungs=(("7k", "pad_7k.txt", None, None),))
    assert common.cell_outputs(bench_env.results, "ladder") == []
