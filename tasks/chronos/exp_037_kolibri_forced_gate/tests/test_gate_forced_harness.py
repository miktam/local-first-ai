# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W3).
"""ForcedTrunk and the P5b / P5c helpers on the tiny real-layout checkpoint
(gate/harness.py; DESIGN §3.0, §3.1 P5b-P5c, §3.3 G1 item 3).

* forced onto its own free-run ids in their own order, the port reproduces the
  free run bitwise at chunk sizes 64, 513 and 2,048 (fp32), and on the K8 build
  as loaded (bf16);
* forced onto the same ids sorted ascending (Csort), within 1e-6 mean KL;
* misaligned rows raise HarnessError, misaligned ids make the set check fire,
  and a route() that ignores the boost makes the set check fire;
* a route-call count other than num_hidden_layers raises HarnessError;
* forced onto the ids of position t - 1, mean KL(free || forced) >= 0.10 (P5c);
* the shadow is the port's own selection on the unboosted scores, and the
  weight-path mutants 6, 7 and 14 stay visible under forcing (their weights
  come from route()). Mutants 6, 7 and 14 against G4-F32's rules are W5b's test.

The port is the BF16 tiny checkpoint loaded through gate.common.load_port and
cast to fp32 (the fp32 port), and the 8-bit build as loaded; token ids are the
first positions of F_tiny's calibration draw (DESIGN §5.1).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from gate import common, harness  # noqa: E402
from gate.harness import ForcedTrunk, HarnessError  # noqa: E402

T = 2100  # crosses one 2,048 chunk, four 513 chunks and 32 64-token chunks
L = 10  # the tiny real layout's num_hidden_layers (DESIGN §5.1)
K = 6


@pytest.fixture(scope="module")
def ids() -> np.ndarray:
    return np.random.default_rng(5).integers(0, 1008, 16384)[:T]


@pytest.fixture(scope="module")
def fp32_port(tiny_real_cal):
    import mlx.core as mx

    model, _, module = common.load_port(tiny_real_cal.bf16)
    model.set_dtype(mx.float32)
    return model, module


@pytest.fixture(scope="module")
def k8_port(tiny_real_cal):
    model, _, module = common.load_port(tiny_real_cal.k8)
    return model, module


@pytest.fixture(scope="module")
def free2048(fp32_port, ids):
    model, module = fp32_port
    return harness.free_run(model, module, ids, 2048)


def _forced(model, module, mode, table, ids, chunk=2048, rows=None):
    ft = ForcedTrunk(module, L, mode, table=table)
    rows = np.arange(len(ids)) if rows is None else rows
    return harness.forced_logits(model, ft, rows, ids, chunk), ft


# --- construction and installation --------------------------------------------


def test_modes_and_tables_are_checked(fp32_port, free2048):
    _, module = fp32_port
    table = free2048[1]["ids"]
    with pytest.raises(ValueError):
        ForcedTrunk(module, L, "forced")
    with pytest.raises(ValueError):
        ForcedTrunk(module, L, "force")
    with pytest.raises(ValueError):
        ForcedTrunk(module, L, "free", table=table)
    with pytest.raises(HarnessError):
        ForcedTrunk(module, L, "force", table=table[:-1])


def test_installation_is_scoped_and_never_nested(fp32_port, free2048, ids):
    model, module = fp32_port
    orig = module.route
    _forced(model, module, "force", free2048[1]["ids"], ids[:200])
    assert module.route is orig
    ft = ForcedTrunk(module, L)
    with pytest.raises(HarnessError):
        ft.begin(np.arange(4))  # not installed
    with ft:
        with pytest.raises(HarnessError):
            ft.__enter__()
        with pytest.raises(HarnessError):  # route() outside a forward
            harness.stream_logits(model, ids[:8]).__next__()
    assert module.route is orig


def test_a_globals_dict_takes_the_tap_like_the_module(fp32_port, free2048, ids):
    """A model loaded by mlx_lm's model_file path exposes its port module only
    as a globals dict (tests/port_harness.port_namespace)."""
    model, module = fp32_port
    free, rec = free2048
    ft = ForcedTrunk(vars(module), L, "force_own_order", table=rec["ids"])
    out = harness.forced_logits(model, ft, np.arange(T), ids, 2048)
    assert harness.bitwise_equal(out, free)
    assert module.route is not ft


# --- P5b: forced onto its own ids in their own order is the free run ----------


@pytest.mark.parametrize("chunk", [64, 513, 2048])
def test_forced_onto_own_ids_in_own_order_is_bitwise_free(fp32_port, ids, chunk):
    model, module = fp32_port
    res = harness.p5b_exactness(model, module, ids, chunk)
    assert res["ok"], res
    assert res["values"]["bitwise_equal"] and res["values"]["rows_differing"] == 0
    assert res["values"]["shadow_set_disagreements"] == 0  # bitwise states: the shadow is the free selection


def test_p5b_holds_on_the_k8_build_as_loaded(k8_port, ids):
    model, module = k8_port
    for chunk in (64, 2048):
        res = harness.p5b_exactness(model, module, ids, chunk)
        assert res["ok"], res


def test_p5b_default_is_one_chunk(fp32_port, ids):
    model, module = fp32_port
    res = harness.p5b_exactness(model, module, ids[:1536])
    assert res["ok"] and res["values"]["chunk"] == 1536 and res["values"]["n_positions"] == 1536


def test_free_records(free2048):
    _, rec = free2048
    assert rec["ids"].shape == (L, T, K) and rec["shadow"].shape == (L, T, K)
    assert np.array_equal(rec["rows"], np.arange(T))
    assert np.array_equal(rec["ids"], rec["shadow"])  # free mode: the selection is the shadow
    assert all(len(set(r)) == K for r in rec["ids"].reshape(-1, K)[:500])


# --- Csort: sorted order --------------------------------------------------------


def test_sorted_own_ids_within_1e6_mean_kl(fp32_port, free2048, ids):
    model, module = fp32_port
    free, rec = free2048
    unsorted = np.any(np.diff(rec["ids"], axis=-1) < 0, axis=-1)
    assert unsorted.mean() > 0.5  # argpartition's order: sorting is not a no-op
    forced, ft = _forced(model, module, "force", rec["ids"], ids)
    used = ft.records()["ids"]
    assert np.array_equal(used, np.sort(rec["ids"], axis=-1))  # handed to the port ascending
    kl = common.kl_rows(free, forced)
    assert kl.mean() <= 1e-6, kl.mean()


# --- row bookkeeping and misalignment -----------------------------------------


def test_pack_rows_index_the_forcing_table(fp32_port, free2048, ids):
    """The driver's rows select the table rows: the sequence at pack offset 700
    of a larger table reproduces the free run bitwise."""
    model, module = fp32_port
    free, rec = free2048
    table = np.random.default_rng(1).permuted(np.tile(np.arange(64), (L, 3000, 1)), axis=-1)[..., :K]
    table[:, 700:700 + T] = rec["ids"]
    out, ft = _forced(model, module, "force_own_order", table, ids, rows=700 + np.arange(T))
    assert harness.bitwise_equal(out, free)
    assert np.array_equal(ft.records()["rows"], 700 + np.arange(T))
    assert ft.forwards == 2


def test_a_free_pack_run_becomes_its_own_forcing_table(fp32_port, ids):
    """Two sequences at pack rows [0, 600) and [600, 1100) through one free-mode
    ForcedTrunk; records_table puts each row's ids at its pack row, and forcing
    the pack onto it in its own order reproduces both runs bitwise. A row the
    free run never recorded cannot be forced."""
    model, module = fp32_port
    seqs = [(ids[:600], np.arange(600)), (ids[1000:1500], 600 + np.arange(500))]
    ft = ForcedTrunk(module, L)
    free = [harness.forced_logits(model, ft, rows, s, 256) for s, rows in seqs]
    rec = ft.records()
    assert np.array_equal(rec["rows"], np.arange(1100)) and ft.forwards == 3 + 2
    table = harness.records_table(rec)
    assert table.shape == (L, 1100, K)
    ft2 = ForcedTrunk(module, L, "force_own_order", table=table)
    for (s, rows), want in zip(seqs, free):
        assert harness.bitwise_equal(harness.forced_logits(model, ft2, rows, s, 256), want)
    gap = harness.records_table(rec, n_rows=1200)
    with pytest.raises(HarnessError, match=r"outside \[0, 64\)"):
        harness.forced_logits(model, ForcedTrunk(module, L, "force", table=gap), 1100 + np.arange(64), ids[:64], 64)
    with pytest.raises(HarnessError, match="recorded twice"):
        harness.records_table({"ids": rec["ids"][:, :2], "rows": np.array([5, 5])})


def test_rows_misaligned_by_one_raise(fp32_port, free2048, ids):
    import mlx.core as mx

    model, module = fp32_port
    table = free2048[1]["ids"]
    ft = ForcedTrunk(module, L, "force", table=table)
    with pytest.raises(HarnessError, match="misaligned"):  # one row short for the sequence
        harness.forced_logits(model, ft, np.arange(T - 1), ids, 2048)
    with ft:  # the driver sets one row fewer than the forward feeds
        ft.begin(np.arange(63))
        with pytest.raises(HarnessError, match="misaligned"):
            model(mx.array(ids[:64].astype(np.int32))[None], cache=model.make_cache())
        ft.abort()
    with pytest.raises(HarnessError, match="outside the forcing table"):  # the second chunk's last row
        harness.forced_logits(model, ft, np.arange(T) + 1, ids, 2048)
    assert ft.forwards == 1  # only the first chunk passed every check and was recorded
    assert np.array_equal(ft.records()["rows"], np.arange(2048) + 1)
    assert module.route is not ft


def test_ids_misaligned_by_one_slot_fire_the_set_check(fp32_port, free2048, ids):
    """Each row's ids shifted by one slot across the flattened [T, k] table: a
    row that then holds an id twice cannot be forced (the set check fires)."""
    model, module = fp32_port
    rec = free2048[1]
    flat = rec["ids"].reshape(L, -1)
    shifted = np.concatenate([flat[:, -1:], flat[:, :-1]], axis=1).reshape(rec["ids"].shape)
    dup = np.array([len(set(r)) < K for r in shifted[0]])
    assert dup.any()
    with pytest.raises(HarnessError, match="forcing did not take"):
        _forced(model, module, "force_own_order", shifted, ids)


def test_a_route_that_ignores_the_boost_fires_the_set_check(fp32_port, free2048, ids):
    import mlx.core as mx

    model, module = fp32_port
    orig = module.route
    module.route = lambda logits, bias, k, renorm=False: orig(logits, mx.zeros_like(bias), k, renorm)
    try:
        shifted = harness.shift_ids(free2048[1]["ids"])
        with pytest.raises(HarnessError, match="forcing did not take"):
            _forced(model, module, "force", shifted, ids[:300])
    finally:
        module.route = orig


def test_route_call_count_mismatch_raises(fp32_port, ids):
    model, module = fp32_port
    with pytest.raises(HarnessError, match="expected num_hidden_layers = 11"):
        harness.forced_logits(model, ForcedTrunk(module, L + 1), np.arange(64), ids[:64], 64)
    with pytest.raises(HarnessError, match="more than num_hidden_layers = 9"):
        harness.forced_logits(model, ForcedTrunk(module, L - 1), np.arange(64), ids[:64], 64)


# --- P5c and the shadow --------------------------------------------------------


def test_ids_of_the_previous_position_move_the_logits(fp32_port, ids):
    model, module = fp32_port
    res = harness.p5c_forcing_active(model, module, ids[:1536], min_mean_kl=0.10)
    assert res["ok"], res
    assert res["values"]["mean_kl"] >= 0.10 and res["values"]["n_positions"] == 1535
    low = harness.p5c_forcing_active(model, module, ids[:300], min_mean_kl=1e3)
    assert not low["ok"] and low["reason"]


def test_shift_ids_keeps_position_zero():
    rec = np.arange(2 * 4 * 3).reshape(2, 4, 3)
    out = harness.shift_ids(rec)
    assert np.array_equal(out[:, 0], rec[:, 0]) and np.array_equal(out[:, 1:], rec[:, :-1])


def test_shadow_is_the_unboosted_selection(fp32_port, free2048, ids):
    """Forced onto the ids of t - 1, layer 0's router input is unchanged (it sees
    only the embeddings), so its shadow is the free selection; at later layers
    the shadow is the port's own choice, not the forced ids."""
    model, module = fp32_port
    rec = free2048[1]
    table = harness.shift_ids(rec["ids"])
    _, ft = _forced(model, module, "force_own_order", table, ids)
    got = ft.records()
    assert np.array_equal(np.sort(got["shadow"][0], -1), np.sort(rec["ids"][0], -1))
    assert np.array_equal(got["ids"], table)
    later = np.any(np.sort(got["shadow"][L - 1], -1) != np.sort(table[L - 1], -1), axis=-1)
    assert later.mean() > 0.5


@pytest.mark.parametrize("mutant", ["renorm_topk", "route_scale_2826", "biased_weights"])
def test_weight_path_mutants_stay_visible_when_forced(tiny_real_cal, fp32_port, free2048, ids, mutant):
    import mlx.core as mx

    model, module = fp32_port
    rec = free2048[1]
    clean, _ = _forced(model, module, "force", rec["ids"], ids[:1024])
    mm, _, mmod = common.load_port(tiny_real_cal.bf16, mutant=mutant)
    mm.set_dtype(mx.float32)
    bad, ft = _forced(mm, mmod, "force", rec["ids"], ids[:1024])
    assert np.array_equal(ft.records()["ids"], np.sort(rec["ids"][:, :1024], -1))  # forcing took
    assert common.kl_rows(clean, bad).mean() >= 0.10


# --- spans, decode, positions --------------------------------------------------


def test_spans():
    assert harness.chunk_spans(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert harness.chunk_spans(3, [(0, 1), (1, 3)]) == [(0, 1), (1, 3)]
    for bad in ([(0, 1), (2, 3)], [(0, 2)], [(0, 0), (0, 3)]):
        with pytest.raises(ValueError):
            harness.chunk_spans(3, bad)
    with pytest.raises(ValueError):
        harness.chunk_spans(3, 0)
    assert harness.decode_spans(10, 6, 4) == [(0, 4), (4, 6)] + [(t, t + 1) for t in range(6, 10)]
    assert harness.decode_spans(3, 0) == [(0, 1), (1, 2), (2, 3)]
    with pytest.raises(ValueError):
        harness.decode_spans(3, 4)


def test_forced_decode_and_positions(fp32_port, free2048, ids):
    """Forced decode (prefix in chunks, then one token per forward) against the
    forced chunked run on the same sorted ids: the routing is identical, so only
    attention rounding remains."""
    model, module = fp32_port
    table = free2048[1]["ids"][:, :600]
    n, start = 600, 520
    chunked, _ = _forced(model, module, "force", table, ids[:n], 256)
    pos = np.arange(start, n)
    ft = ForcedTrunk(module, L, "force", table=table)
    dec = harness.forced_logits(model, ft, np.arange(n), ids[:n], harness.decode_spans(n, start, 256), positions=pos)
    assert ft.forwards == 3 + (n - start)
    assert common.kl_rows(chunked[pos], dec).mean() <= 1e-6
    ft2 = ForcedTrunk(module, L, "force", table=table)
    sub = harness.forced_logits(model, ft2, np.arange(n), ids[:n], 256, positions=pos[::-7])
    assert harness.bitwise_equal(sub, chunked[pos[::-7]])
