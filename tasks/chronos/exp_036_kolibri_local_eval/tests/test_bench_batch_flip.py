# SPDX-License-Identifier: MIT
"""bench/batch_flip.py (Control C1): divergence arithmetic, the cell on a
tiny checkpoint with injected items and extractor, the memory-rule B, and the
default item preparation and extraction through the real sibling modules
(runner.chat, scorers) with the real Kolibri tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    assert_sorted_keys_jsonl,
    bench_env,
    kolibri_tok_dir,
    read_jsonl,
    tiny_loader,
    tiny_models,
)

from bench import batch_flip, common


@pytest.mark.parametrize(
    "a,b,expected",
    [([1, 2, 3], [1, 2, 3], None), ([1, 2, 3], [1, 9, 3], 1), ([1, 2], [1, 2, 3], 2), ([], [], None), ([5], [], 0)],
)
def test_first_divergence(a, b, expected):
    assert batch_flip.first_divergence(a, b) == expected


def _items(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "id": f"syn{i}",
            "item_sha256": f"{i:064x}",
            "prompt_sha256": f"{i + 100:064x}",
            "rendered_prompt_sha256": f"{i + 200:064x}",
            "prompt_ids": [int(x) for x in rng.integers(0, 1000, size=int(rng.integers(20, 80)))],
        }
        for i in range(n)
    ]


def _letter(prompt_ids, completion_ids):
    return "ABCD"[completion_ids[0] % 4] if completion_ids else None


def _run(bench_env, tiny_models, B, max_tokens=24, extract=_letter):
    return batch_flip.run_cell(
        bench_env.results,
        loader=tiny_loader({"K8": tiny_models["a"]}),
        prepare=lambda tok: _items(),
        extract=extract,
        B=B,
        max_tokens=max_tokens,
        full_dir=bench_env.root / "work" / "c1",
    )


def test_c1_cell_records_are_hashed_and_full_records_stay_private(bench_env, tiny_models):
    path = _run(bench_env, tiny_models, B=4)
    assert common.is_complete(path) and path.name.startswith("c1_")
    assert_sorted_keys_jsonl(path)
    recs = read_jsonl(path)
    head, body, end = recs[0], recs[1:-1], recs[-1]
    assert bench_env.calls[:2] == [("identity",), ("gate", "K8")]
    assert head["B_rule"] == {"B": 4, "source": "argument"} and head["effort"] == "none"
    assert head["runs"]["B"]["B"] == 4 and head["runs"]["B1"]["B"] == 1
    assert [r["key"]["item"] for r in body] == [f"syn{i}" for i in range(10)]
    text = path.read_text()
    assert "completion_ids\"" not in text and "prompt_ids" not in text
    full = sorted((bench_env.root / "work" / "c1").glob("c1_*.full.jsonl"))
    assert len(full) == 1 and full[0].stem.startswith(path.stem)
    frecs = read_jsonl(full[0])[1:-1]
    for r, f in zip(body, frecs):
        cb, c1 = f["completion_ids"]["B"], f["completion_ids"]["B1"]
        assert r["runs"]["B"]["completion_ids_sha256"] == common.sha256_ids(cb)
        assert r["first_divergence"] == batch_flip.first_divergence(cb, c1)
        truncated = r["runs"]["B"]["finish_reason"] == "length"
        assert r["runs"]["B"]["extracted"] == (None if truncated else "ABCD"[cb[0] % 4])
        assert r["flipped"] == (r["runs"]["B"]["extracted"] != r["runs"]["B1"]["extracted"])
        assert r["runs"]["B"]["finish_reason"] in ("stop", "length")
    s = end["summary"]
    assert s["n_items"] == 10 and s["flip_rate"] == s["n_flips"] / 10
    assert s["full_records"]["sha256"] == common.sha256_file(full[0])


def test_c1_identical_runs_have_no_flips(bench_env, tiny_models):
    path = _run(bench_env, tiny_models, B=1)
    s = read_jsonl(path)[-1]["summary"]
    assert s["n_flips"] == 0 and s["n_diverged"] == 0 and s["first_divergence_positions"] == []


def test_c1_truncated_completion_counts_as_no_answer(bench_env, tiny_models):
    seen = []

    def extract(p, c):
        seen.append(len(c))
        return "A"

    path = _run(bench_env, tiny_models, B=2, max_tokens=4, extract=extract)
    for r in read_jsonl(path)[1:-1]:
        for run in r["runs"].values():
            if run["finish_reason"] == "length":
                assert run["truncated"] and run["extracted"] is None and run["parse_status"] == "truncated"
            else:
                assert run["extracted"] == "A"


def test_memory_rule_B_prefers_the_preflight_value_and_records_both(monkeypatch, tmp_path):
    from runner import memory

    calls = []
    monkeypatch.setattr(memory, "effective_limit", lambda: (115_448_725_504, {"chosen": "max_recommended_working_set_size"}))
    monkeypatch.setattr(memory, "choose_B", lambda arm, task, cap, L, **kw: calls.append((arm, task, cap, L)) or 4)
    rec = batch_flip.memory_rule_B(tmp_path)  # no preflight record yet
    assert calls == [("K8", "mmlu", 32768, 115_448_725_504)]
    assert rec["B"] == 4 and rec["preflight"] is None and rec["agree"] is None
    (tmp_path / "preflight_20261003T170000Z.json").write_text(json.dumps({"memory_rule": {"c1_B": 8}}))
    rec = batch_flip.memory_rule_B(tmp_path)
    assert rec["B"] == 8 and rec["B_now"] == 4 and rec["agree"] is False
    assert rec["preflight"] == {"file": "preflight_20261003T170000Z.json", "B": 8} and rec["cap"] == 32768


def test_c1_refuses_when_k8_does_not_fit(bench_env, tiny_models):
    with pytest.raises(common.CouldNotRun, match="B = 0"):
        _run(bench_env, tiny_models, B=0)
    assert common.cell_outputs(bench_env.results, "c1") == []


def test_initial_cap_is_the_plan_rules_mmlu_cap():
    assert batch_flip.initial_cap() == 32768 == batch_flip.C1_INITIAL_CAP


def _kolibri_tokenizer():
    from mlx_lm.tokenizer_utils import load as load_tok

    return load_tok(kolibri_tok_dir())


def test_default_prepare_and_extract_through_the_sibling_modules(monkeypatch):
    """tasks.mmlu_prox items (faked: no MMLU-ProX data on the mini) rendered by
    runner.chat at effort none, and the answer extracted by scorers."""
    import tasks.assets
    import tasks.mmlu_prox
    from tasks.common import make_item

    tok = _kolibri_tokenizer()
    fake = [
        make_item("mmlu_en", f"q{i}", {"q": i}, [{"role": "user", "content": f"Synthetic question {i}? (A) x (B) y"}])
        for i in range(batch_flip.C1_N)
    ]
    monkeypatch.setattr(tasks.assets, "asset", lambda name: type("A", (), {"path": Path("/nonexistent") / name})())
    seen = {}
    monkeypatch.setattr(tasks.mmlu_prox, "c1_items", lambda full, lite, n=100: seen.update(full=full, lite=lite, n=n) or fake)
    items = batch_flip.default_prepare(tok)
    assert seen["n"] == 100 and seen["full"].name == "MMLU-ProX" and seen["lite"].name == "MMLU-ProX-Lite"
    assert len(items) == 100 and items[0]["id"] == "q0" and items[0]["prompt_sha256"] == fake[0].prompt_sha256
    tail = tok.decode(items[0]["prompt_ids"][-6:])
    assert tail.endswith("<think>\n\n</think>\n\n")
    extract = batch_flip.default_extract(tok)
    done = tok.encode("Short reasoning. The answer is (B).<|im_end|>", add_special_tokens=False)
    assert extract(items[0]["prompt_ids"], done) == "B"
    assert extract(items[0]["prompt_ids"], tok.encode("No letter here.<|im_end|>", add_special_tokens=False)) is None
