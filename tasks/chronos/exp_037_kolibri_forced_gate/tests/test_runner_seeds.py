"""runner/seeds.py: the per-cell seed rule H("exp037", arm, task, effort, pass)
(HYPOTHESIS "Fixed before any run", Sampling; BUILD_SPEC §5.4; exp_037 DESIGN
§2.10: exp_036's rule with its own prefix)."""

from __future__ import annotations

import hashlib

import pytest

from runner import seeds
from runner.seeds import cell_seed, seed_string

# Pinned for exp_037 on 2026-10-06 (DESIGN §2.10, prefix "exp037"); a change here
# changes every cell's seed and needs a numbered amendment after the freeze.
PINNED = {
    ("K8", "gpqa_en", "high", 0): 2653512673,
    ("K8", "gpqa_en", "high", 1): 1023157708,
    ("G8", "mmlu_de", "high", 0): 3708826703,
    ("K8", "rgb_cb", "none", 0): 2044799679,
    ("Q36-8", "ifbench", "high", 0): 2193709987,
}
# exp_036's values, pinned on 2026-10-03 under the prefix "exp036".
PINNED_EXP036 = {
    ("K8", "gpqa_en", "high", 0): 1404998068,
    ("K8", "gpqa_en", "high", 1): 2316956487,
    ("G8", "mmlu_de", "high", 0): 2685946012,
    ("K8", "rgb_cb", "none", 0): 1591033383,
    ("Q36-8", "ifbench", "high", 0): 2036686678,
}


@pytest.mark.parametrize("key,value", sorted(PINNED.items()))
def test_seed_values_are_stable(key, value):
    assert cell_seed(*key) == value


def test_prefix_is_exp037_and_only_the_prefix_changed(monkeypatch):
    """G1 item 13: the cell seeds are exp_037 draws; with exp_036's prefix the same code gives exp_036's pinned
    values, so the rule itself is unchanged."""
    assert seeds.SEED_PREFIX == "exp037"
    assert not set(PINNED.values()) & set(PINNED_EXP036.values())
    monkeypatch.setattr(seeds, "SEED_PREFIX", "exp036")
    assert {k: cell_seed(*k) for k in PINNED_EXP036} == PINNED_EXP036


def test_seed_is_first_four_bytes_big_endian_of_sha256():
    s = seed_string("K4", "mmlu_en", "high", 0)
    assert s == "exp037|K4|mmlu_en|high|0"
    assert cell_seed("K4", "mmlu_en", "high", 0) == int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big")


def test_seeds_differ_by_every_component_and_fit_uint32():
    base = ("K8", "gpqa_en", "high", 0)
    variants = [("K4",) + base[1:], base[:1] + ("gpqa_de",) + base[2:], base[:2] + ("low", 0), base[:3] + (1,)]
    seen = {cell_seed(*base)} | {cell_seed(*v) for v in variants}
    assert len(seen) == 5
    assert all(0 <= s < 2**32 for s in seen)


@pytest.mark.parametrize("bad", [("", "t", "e", 0), ("K8|x", "t", "e", 0), ("K8", "t", "e", -1)])
def test_bad_components_are_refused(bad):
    with pytest.raises(ValueError):
        cell_seed(*bad)
