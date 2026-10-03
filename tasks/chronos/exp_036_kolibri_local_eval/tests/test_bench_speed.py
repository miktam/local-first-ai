# SPDX-License-Identifier: MIT
"""bench/speed.py: the H1 cell and the descriptive speed cell, on tiny
checkpoints with the H1 protocol at its pre-registered sizes; and the
fixture arithmetic with the real tokenizers (why the 4,096-token prompts come
from pad_120k.txt)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    assert_sorted_keys_jsonl,
    bench_env,
    kolibri_tok_dir,
    mini_dir,
    read_jsonl,
    tiny_loader,
    tiny_models,
)

from bench import common, speed

# HYPOTHESIS H1 / BUILD_SPEC §5.7: 10 blocks, order from seed 36. Pinned so a
# change of numpy's permutation stream cannot change the order silently.
SEED36_ORDERS = [
    ["G4", "K4"], ["K4", "G4"], ["G4", "K4"], ["K4", "G4"], ["G4", "K4"],
    ["K4", "G4"], ["K4", "G4"], ["K4", "G4"], ["K4", "G4"], ["G4", "K4"],
]


def test_protocol_constants_match_the_preregistration():
    assert speed.H1_ARMS == ("K4", "G4")
    assert (speed.H1_BLOCKS, speed.H1_ORDER_SEED, speed.IDLE_S) == (10, 36, 30)
    assert (speed.H1_PROMPT_FIXTURE, speed.H1_PROMPT_TOKENS, speed.H1_GEN_TOKENS) == ("pad_4k.txt", 1024, 256)
    assert speed.DESC_ARMS == ("K8", "G8", "Q36-8", "Q36-4", "Q38-8", "Q38-4")
    assert (speed.DESC_REPS, speed.DESC_BATCH, speed.PREFILL_TOKENS) == (5, (1, 2, 4), 4096)


def test_block_orders_are_seed_36_permutations():
    assert speed.block_orders() == SEED36_ORDERS
    assert speed.block_orders() == speed.block_orders()


def test_h1_cell_on_tiny_models_at_preregistered_sizes(bench_env, tiny_models):
    calls = []
    loader = tiny_loader({"K4": tiny_models["a"], "G4": tiny_models["b"]}, calls)
    path = speed.run_h1(bench_env.results, loader=loader)
    assert path.parent == bench_env.results / "bench" and path.name.startswith("speed_")
    assert common.is_complete(path)
    assert_sorted_keys_jsonl(path)
    recs = read_jsonl(path)
    head, body, end = recs[0], recs[1:-1], recs[-1]

    # guards: identity first, the gate for the Kolibri arm only
    assert bench_env.calls[:2] == [("identity",), ("gate", "K4")]
    assert ("gate", "G4") not in bench_env.calls
    assert calls == ["K4", "G4"]  # both resident

    assert head["block_orders"] == SEED36_ORDERS
    assert head["arms"]["K4"]["prompt"]["n_tokens"] == 1024
    assert head["arms"]["K4"]["prefill_prompt"]["fixture"] == "pad_120k.txt"
    assert head["environment"]["metal"]["max_recommended_working_set_size"] > 0

    kinds = [r["kind"] for r in body]
    assert kinds.count("warmup") == 2 and kinds.count("run") == 20 and kinds.count("block") == 10
    assert kinds.count("prefill") == 10
    runs = [r for r in body if r["kind"] == "run"]
    assert all(r["generation_tokens"] == 256 and r["finish_reason"] == "length" for r in runs)
    assert all(r["prompt_tokens"] == 1024 and r["eos_masked"] for r in runs)
    for b in [r for r in body if r["kind"] == "block"]:
        mine = {r["arm"]: r["generation_tps"] for r in runs if r["block"] == b["block"]}
        assert [r["arm"] for r in runs if r["block"] == b["block"]] == b["order"] == SEED36_ORDERS[b["block"]]
        assert b["log_ratio_k4_g4"] == pytest.approx(math.log(mine["K4"] / mine["G4"]))
        assert b["before"]["thermal_label"] == "nominal" and b["after"]["power_source"] == "AC Power"
    prefill = [r for r in body if r["kind"] == "prefill"]
    assert all(r["prompt_tokens"] == 4096 for r in prefill)
    # 30 s idle after the warm-ups and after every timed run and prefill
    assert bench_env.sleeps == [30] * (1 + 20 + 10)
    assert end["summary"]["n_blocks"] == 10
    assert end["summary"]["mean_log_ratio"] == pytest.approx(sum(end["summary"]["log_ratios_k4_g4"]) / 10)
    assert str(Path.home()) not in path.read_text()


def test_desc_cell_on_tiny_models(bench_env, tiny_models):
    calls = []
    loader = tiny_loader({"K8": tiny_models["a"], "G8": tiny_models["b"]}, calls)
    path = speed.run_desc(bench_env.results, loader=loader, arms=("K8", "G8"), reps=2)
    assert common.is_complete(path) and path.name.startswith("speed_desc_")
    recs = read_jsonl(path)
    body = recs[1:-1]
    assert calls == ["K8", "G8"]  # one at a time, in order
    assert ("gate", "K8") in bench_env.calls and ("gate", "G8") not in bench_env.calls
    for arm in ("K8", "G8"):
        mine = [r for r in body if r.get("arm") == arm]
        kinds = [r["kind"] for r in mine]
        assert kinds.count("arm") == 1 and kinds.count("warmup") == 1
        assert kinds.count("prefill") == 2 and kinds.count("decode_b1") == 2 and kinds.count("rep") == 2
        batch = [r for r in mine if r["kind"] == "batch"]
        assert [r["B"] for r in batch] == [1, 2, 4, 1, 2, 4]
        assert all(r["generation_tokens"] == 256 * r["B"] for r in batch)
        info = [r for r in mine if r["kind"] == "arm"][0]["batch_prompts"]
        assert info["n_tokens"] == 4 * 1024 and len(set(info["rows_ids_sha256"])) == 4
        assert all(r["prompt_tokens"] == 4096 for r in mine if r["kind"] == "prefill")
    summary = recs[-1]["summary"]["median_by_arm"]
    assert set(summary) == {"K8", "G8"} and summary["K8"]["aggregate_tps_B4"] > 0


class _RawTok:
    """encode(text, add_special_tokens=False) over a raw tokenizer.json."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.t = Tokenizer.from_file(str(path))

    def encode(self, text, add_special_tokens=False):
        return self.t.encode(text, add_special_tokens=add_special_tokens).ids


def _real_tokenizers() -> dict[str, Path]:
    return {
        "kolibri": kolibri_tok_dir() / "tokenizer.json",
        "gemma4": mini_dir("mlx-community", "gemma-4-26b-a4b-it-4bit") / "tokenizer.json",
        "qwen3_6": mini_dir("mlx-community", "Qwen3.6-35B-A3B-4bit") / "tokenizer.json",
        "qwen3_8": mini_dir("mlx-community", "Qwen3.8-27B-4bit") / "tokenizer.json",
    }


def test_fixture_arithmetic_with_the_real_tokenizers():
    """Every prompt the bench cells cut from the padding fixtures exists in
    every tokenizer, and the two documented substitutions are needed:
    pad_4k.txt has fewer than 4,096 tokens (so the 4,096-token prompts come
    from pad_120k.txt), and pad_120k.txt has fewer than 120k Kolibri tokens
    (so the E6 120k rung is the whole file)."""
    from bench import genutil

    for name, f in _real_tokenizers().items():
        tok = _RawTok(f)
        _, small = genutil.fixture_ids(tok, "pad_4k.txt", speed.H1_PROMPT_TOKENS)
        assert small["fixture_tokens"] < speed.PREFILL_TOKENS, name
        genutil.fixture_ids(tok, speed.PREFILL_FIXTURE, speed.PREFILL_TOKENS)
        genutil.fixture_ids(tok, speed.BATCH_FIXTURE, max(speed.DESC_BATCH) * speed.H1_PROMPT_TOKENS)
        _, big = genutil.fixture_ids(tok, "pad_120k.txt", 65536)  # D1 and E6 64k
        if name == "kolibri":
            assert big["fixture_tokens"] < 120_000
        # pad_4k.txt is a byte prefix of pad_120k.txt, so its 1,024-token
        # prompt equals the first 1,024 tokens of pad_120k.txt
        a, _ = genutil.fixture_ids(tok, "pad_4k.txt", 1024)
        b, _ = genutil.fixture_ids(tok, "pad_120k.txt", 1024)
        assert a == b, name


def test_no_text_of_the_fixture_in_the_record(bench_env, tiny_models):
    """Records carry token counts and hashes, not prompt text."""
    loader = tiny_loader({"K4": tiny_models["a"], "G4": tiny_models["b"]})
    path = speed.run_h1(bench_env.results, loader=loader, n_blocks=1, prefill_reps=1, gen_tokens=8)
    text = path.read_text()
    assert "The throughput of a distributed system" not in text
    assert json.loads(text.splitlines()[0])["protocol"]["blocks"] == 1
