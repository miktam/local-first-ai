"""runner/seeds.py: the per-cell seed rule H("exp036", arm, task, effort, pass)
(HYPOTHESIS "Fixed before any run", Sampling; BUILD_SPEC §5.4)."""

from __future__ import annotations

import hashlib

import pytest

from runner.seeds import cell_seed, seed_string

# Pinned on 2026-10-03 when the rule was frozen; a change here changes every
# cell's seed and needs a numbered amendment.
PINNED = {
    ("K8", "gpqa_en", "high", 0): 1404998068,
    ("K8", "gpqa_en", "high", 1): 2316956487,
    ("G8", "mmlu_de", "high", 0): 2685946012,
    ("K8", "rgb_cb", "none", 0): 1591033383,
    ("Q36-8", "ifbench", "high", 0): 2036686678,
}


@pytest.mark.parametrize("key,value", sorted(PINNED.items()))
def test_seed_values_are_stable(key, value):
    assert cell_seed(*key) == value


def test_seed_is_first_four_bytes_big_endian_of_sha256():
    s = seed_string("K4", "mmlu_en", "high", 0)
    assert s == "exp036|K4|mmlu_en|high|0"
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
