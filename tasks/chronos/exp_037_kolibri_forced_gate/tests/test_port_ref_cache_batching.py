# SPDX-License-Identifier: MIT
"""The port under mlx_lm's continuous batching (spec items 19, 24, 27;
`test_cache_batching.py` in BUILD_SPEC 6, named test_port_ref_* here).

mlx_lm 0.32.0 (exp_037 decision F2; DESIGN §6.3, W0r item 2): BatchGenerator
no longer has _make_cache. Each inserted sequence gets its own cache from
models.cache.make_prompt_cache (the port's make_cache: KVCache on full layers,
SlidingKVCache, a RotatingKVCache with keep 0, on sliding layers), and the
batch caches are built by each cache class's merge (mlx_lm.generate
_merge_caches): KVCache -> BatchKVCache, SlidingKVCache ->
BatchRotatingKVCache(max_size = sliding_window). Prompts admitted together are
right-padded, then finalised to left padding; finished sequences are replaced
mid-run ("refill"). None of that may change a sequence's numbers:

* make_prompt_cache plus merge map the port's caches as above, and an empty
  merged batch takes left padding as the old _make_cache(model, [0, 3, 1])
  did; SlidingKVCache.to_quantized returns itself (item 19);
* the cache semantics mlx_lm 0.32.0 changed, re-validated on the classes the
  port maps to: the offset is rebound, never mutated in place (#1848); left
  padding is read per row from the host, without an MLX reduction (#1824);
  extend() keeps the cache dtype when one side is empty (#1491, no promotion
  to float32); state returns the full state and round-trips (#1778);
* a left-padded batch of 4 reproduces each sequence run alone (fp32, 1e-4);
* test_refill_rotated: the tiny vendor preset in fp32 at W = 65 and at the
  real W = 513, completion_batch_size 8, 4 and 3, twelve prompts of up to 3W
  tokens generating up to 3W tokens, with at least two admissions while
  another live sequence is past 2W: every sequence's greedy log-probs equal
  its B = 1 run's within 1e-4, and the B = 1 run equals a one-shot forward
  over prompt + generation within 1e-4.

Generation goes through runner.generate.make_batch_generator, the runner's one
BatchGenerator construction (prefill_batch_size = min(B, 8), prefill_step_size
2,048, max_kv_size None, the fp32-logit wrapper), with no stop ids.

Tolerance: fp32 log-probs of these logits (std ~0.8, |logprob| < 10) carry
a few 1e-6 of summation-order noise between batch shapes, so 1e-4 leaves
more than 20x headroom. Greedy decoding can still diverge at an exact
near-tie; the comparison then stops at the first differing token, which must
sit at a top-2 log-prob gap below 1e-4 in the B = 1 run, and at most one such
divergence is allowed per configuration.
Measured under mlx_lm 0.31.3 (W = 65 and 513): batched vs B = 1 <= 4.3e-6;
B = 1 vs one-shot forward <= 4.8e-6; no divergence; 4, 8 and 7 admissions
while another sequence is past 2W at completion_batch_size 8, 4 and 3.
"""

from __future__ import annotations

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc

TOL = 1e-4


def _prompts(W: int, vocab: int):
    rng = np.random.default_rng(7)
    lens = [10, 3 * W // 2, 30, 2 * W + 5, 40, W + 3, 5, 3 * W, 20, W // 2, 2 * W, 15]
    maxt = [25, 2 * W, 60, W, 3 * W, 30, 2 * W + 10, 50, 40, W, 20, 3 * W]
    prompts = [rng.integers(0, vocab - tc.RESERVED_TOP_IDS, size=n).tolist() for n in lens]
    return prompts, maxt


def _run(model, prompts, maxt, cbs):
    """Greedy generation through the runner's BatchGenerator construction; per
    sequence the tokens and the full fp32 log-prob vectors, plus the refill
    events."""
    import mlx.core as mx

    from runner.generate import make_batch_generator

    gen = make_batch_generator(model, cbs, [], max(maxt), lambda x: mx.argmax(x, axis=-1))
    try:
        uids = gen.insert(prompts, max_tokens=maxt)
        index = {u: i for i, u in enumerate(uids)}
        toks = {u: [] for u in uids}
        lps = {u: [] for u in uids}
        live_max, admissions_while_long = set(), 0
        while True:
            resp = gen.next_generated()
            if not resp:
                break
            step_uids = {r.uid for r in resp}
            new = step_uids - live_max
            if new and live_max:
                lengths = [len(prompts[index[u]]) + len(toks[u]) for u in live_max & step_uids]
                if any(n > 2 * model.args.sliding_window for n in lengths):
                    admissions_while_long += len(new)
            live_max |= new
            for r in resp:
                toks[r.uid].append(int(r.token))
                lps[r.uid].append(np.array(r.logprobs.astype(mx.float32)))
            for r in resp:
                if getattr(r, "finish_reason", None) is not None:
                    live_max.discard(r.uid)
    finally:
        gen.close()
    return [toks[u] for u in uids], [np.stack(lps[u]) for u in uids], admissions_while_long


def _compare(single_toks, single_lps, toks, lps):
    """Max |d logprob| up to the first token divergence, and the divergences."""
    worst, divergences = 0.0, []
    for i in range(len(single_toks)):
        a, b = single_toks[i], toks[i]
        n = min(len(a), len(b))
        first = next((k for k in range(n) if a[k] != b[k]), None)
        upto = n if first is None else first + 1  # the diverging step's distribution is still comparable
        worst = max(worst, float(np.abs(single_lps[i][:upto] - lps[i][:upto]).max()))
        if first is not None:
            s = np.sort(single_lps[i][first])
            divergences.append((i, first, float(s[-1] - s[-2])))
        else:
            assert len(a) == len(b), (i, len(a), len(b))
    return worst, divergences


@pytest.fixture(scope="module")
def models(tiny_checkpoint_factory):
    import mlx.core as mx

    from exp036_helpers import import_sibling

    import_sibling("port.kolibri1")

    out = {}
    for preset in ("vendor", "w513"):
        model = ph.load_port(tiny_checkpoint_factory(preset, 0))
        model.set_dtype(mx.float32)
        out[model.args.sliding_window] = model
    return out


# --------------------------------------------- the cache mapping (mlx_lm 0.32.0)


def _merged(model, n: int):
    """What BatchGenerator builds for n inserted sequences: make_prompt_cache per sequence, then the batch cache of
    each layer from its class's merge (mlx_lm.generate._merge_caches)."""
    from mlx_lm.models.cache import make_prompt_cache

    per_seq = [make_prompt_cache(model) for _ in range(n)]
    return [type(layer_caches[0]).merge(list(layer_caches)) for layer_caches in zip(*per_seq)]


def test_make_prompt_cache_and_merge_map_the_port_caches(models):
    import importlib

    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, KVCache, make_prompt_cache

    G = importlib.import_module("mlx_lm.generate")  # the module (mlx_lm.generate the attribute is a function)
    assert hasattr(G, "BatchGenerator") and hasattr(G, "_merge_caches")
    assert not hasattr(G, "_make_cache")  # removed in 0.32.0: this test replaces the 0.31.3 one
    for W, model in models.items():
        single = make_prompt_cache(model)
        assert [type(c).__name__ for c in single] == [type(c).__name__ for c in model.make_cache()]
        caches = _merged(model, 3)
        assert len(caches) == len(model.layers)
        for layer, s, c in zip(model.layers, single, caches):
            if layer.use_sliding:
                assert type(s).__name__ == "SlidingKVCache" and s.max_size == W and s.keep == 0
                assert type(c) is BatchRotatingKVCache and c.max_size == W
            else:
                assert type(s) is KVCache
                assert type(c) is BatchKVCache
            # The empty batch takes the admitted prompts' left padding, as _make_cache(model, [0, 3, 1]) did.
            assert c.left_padding.tolist() == [0, 0, 0] and c.offset.tolist() == [0, 0, 0]
            c.prepare(left_padding=[0, 3, 1])
            assert c.left_padding.tolist() == [0, 3, 1] and c.offset.tolist() == [0, -3, -1]
        # The generator's own helper builds the same classes.
        assert [type(c) for c in G._merge_caches([make_prompt_cache(model) for _ in range(2)])] == [
            type(c) for c in caches]
        sliding = [c for c in model.make_cache() if type(c).__name__ == "SlidingKVCache"]
        assert sliding and all(c.to_quantized(group_size=64, bits=8) is c for c in sliding)


def test_merged_caches_decode_each_sequence_as_alone(models):
    """Sequences prefilled alone (one past the window), merged into one batch cache per layer (left padding from
    the length differences) and decoded one step together give each sequence's own next-token logits."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    model = models[65]
    rng = np.random.default_rng(3)
    vocab = model.args.vocab_size - tc.RESERVED_TOP_IDS
    prompts = [rng.integers(0, vocab, size=n).tolist() for n in (40, 7, 23)]
    nxt = [int(t) for t in rng.integers(0, vocab, size=3)]
    singles = []
    for p in prompts:
        c = make_prompt_cache(model)
        model(mx.array([p]), cache=c)
        singles.append(c)
    batch = [type(cs[0]).merge(list(cs)) for cs in zip(*singles)]
    for c in batch:
        assert c.left_padding.tolist() == [0, 33, 17]
    got = np.array(model(mx.array([[t] for t in nxt]), cache=batch)[:, -1, :].astype(mx.float32))
    for i, (c, t) in enumerate(zip(singles, nxt)):
        want = np.array(model(mx.array([[t]]), cache=c)[0, -1, :].astype(mx.float32))
        assert np.abs(got[i] - want).max() <= TOL, i


# ------------------------------------------- cache semantics changed in 0.32.0


def _kv(B, S, dtype=None, seed=0):
    import mlx.core as mx

    rng = np.random.default_rng(seed)
    k = mx.array(rng.standard_normal((B, 2, S, 4)).astype(np.float32))
    v = mx.array(rng.standard_normal((B, 2, S, 4)).astype(np.float32))
    return (k, v) if dtype is None else (k.astype(dtype), v.astype(dtype))


def _batch_caches(W: int = 8):
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    return {"BatchKVCache": BatchKVCache([0, 2, 1]), "BatchRotatingKVCache": BatchRotatingKVCache(W, [0, 2, 1])}


@pytest.mark.parametrize("name", ["BatchKVCache", "BatchRotatingKVCache"])
def test_offset_is_rebound_not_mutated(name):
    """#1848: update_and_fetch and trim rebind self.offset to a new array, so a reference taken before (as mutant
    26 keeps one) still holds the old offsets."""
    c = _batch_caches()[name]
    before = c.offset
    want = before.tolist()
    for S in (3, 1, 1):  # a prefill chunk, then decode steps (BatchRotatingKVCache's in-place path)
        held = c.offset
        c.update_and_fetch(*_kv(3, S, seed=S))
        assert c.offset is not held and held.tolist() == want
        want = [o + S for o in want]
        assert c.offset.tolist() == want
    assert before.tolist() == [0, -2, -1]
    held = c.offset
    assert c.trim(1) == 1 and held.tolist() == want and c.offset.tolist() == [o - 1 for o in want]


def test_left_padding_is_read_per_row_without_a_reduction():
    """#1824: BatchKVCache reads left padding per row on the host (filter's shift, extract's slice), and the decode
    mask hides exactly each row's own padding."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache, KVCache

    rows = []
    singles = []
    for i, n in enumerate((5, 2, 3)):
        s = KVCache()
        k, v = _kv(1, n, seed=10 + i)
        s.update_and_fetch(k, v)
        singles.append(s)
        rows.append((np.array(k), np.array(v)))
    c = BatchKVCache.merge(singles)
    assert c.left_padding.tolist() == [0, 3, 2] and c._idx == 5 and c.offset.tolist() == [5, 2, 3]
    mask = np.array(c.make_mask(1))  # one decode position over the 5 cached + 1 new keys
    assert mask.shape[0] == 3
    for b, pad in enumerate((0, 3, 2)):
        assert mask[b].reshape(-1).tolist() == [False] * pad + [True] * (6 - pad)
    c.filter([1, 2])  # drop the longest row: the common padding (2) is shifted out
    assert c.left_padding.tolist() == [1, 0] and c._idx == 3
    for j, i in enumerate((1, 2)):
        e = c.extract(j)
        assert e.offset == rows[i][0].shape[2]
        np.testing.assert_array_equal(np.array(e.keys), rows[i][0])
        np.testing.assert_array_equal(np.array(e.values), rows[i][1])
    lazy = BatchKVCache.merge(singles)
    lazy.left_padding = lazy.left_padding + mx.array([1, 1, 1])  # unevaluated: read without .min().item()
    lazy.filter([0, 1, 2])
    assert lazy.left_padding.tolist() == [0, 3, 2] and lazy._idx == 4


@pytest.mark.parametrize("name", ["BatchKVCache", "BatchRotatingKVCache"])
def test_extend_keeps_the_cache_dtype_when_one_side_is_empty(name):
    """#1491: extending a bf16 batch cache with an empty one (a sequence admitted with no history) keeps bf16;
    under 0.31.3 the empty side was a float32 array and promoted the batch to float32."""
    import mlx.core as mx

    for dtype in (mx.bfloat16, mx.float32):
        a = _batch_caches()[name]
        a.update_and_fetch(*_kv(3, 3, dtype=dtype))
        b = type(a)(*([8, [0]] if name == "BatchRotatingKVCache" else [[0]]))
        a.extend(b)
        k, v = a.keys_and_values() if hasattr(a, "keys_and_values") else (a.keys, a.values)
        assert a.keys.dtype == dtype and a.values.dtype == dtype and k.dtype == dtype
        assert a.left_padding.shape[0] == 4 and a.offset.shape[0] == 4


def test_state_is_the_full_state_and_round_trips():
    """#1778: state returns the full state (unsliced buffers plus every index), so a cache rebuilt from it is the
    same cache, also after the rotating cache has wrapped."""
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    c = BatchKVCache([0, 2, 1])
    c.update_and_fetch(*_kv(3, 5, seed=1))
    st = c.state
    assert len(st) == 5 and st[0].shape[2] >= c._idx == 5 and st[4] == 5
    d = BatchKVCache([0, 0, 0])
    d.state = st
    for x, y in zip(c.keys_and_values(), d.keys_and_values()):
        np.testing.assert_array_equal(np.array(x), np.array(y))
    assert d.offset.tolist() == c.offset.tolist() and d.left_padding.tolist() == c.left_padding.tolist()

    r = BatchRotatingKVCache(8, [0, 2, 1])
    r.update_and_fetch(*_kv(3, 6, seed=2))
    for t in range(5):  # decode past the window: the buffer wraps
        r.update_and_fetch(*_kv(3, 1, seed=20 + t))
    assert r.rotated
    st = r.state
    assert len(st) == 8 and st[4] == 8 and st[7] is True
    q = BatchRotatingKVCache(8, [0, 0, 0])
    q.state = st
    assert (q.max_size, q._offset, q._idx, q.rotated) == (r.max_size, r._offset, r._idx, r.rotated)
    for x, y in zip(r.keys_and_values(), q.keys_and_values()):
        np.testing.assert_array_equal(np.array(x), np.array(y))
    np.testing.assert_array_equal(np.array(q.make_mask(1)), np.array(r.make_mask(1)))


# ------------------------------------------------ batched generation = alone


def test_left_padded_batch_of_four_equals_single(models):
    model = models[65]
    prompts, _ = _prompts(65, model.args.vocab_size)
    chosen = [prompts[i] for i in (0, 1, 5, 9)]  # 10, 97, 68, 32 tokens: left padding up to 87
    maxt = [20] * 4
    single = [_run(model, [p], [m], 1)[:2] for p, m in zip(chosen, maxt)]
    toks, lps, _ = _run(model, chosen, maxt, 4)
    worst, div = _compare([s[0][0] for s in single], [s[1][0] for s in single], toks, lps)
    assert worst <= TOL and not div, (worst, div)


@pytest.mark.parametrize("W", [65, 513])
def test_refill_rotated(models, W):
    import mlx.core as mx

    model = models[W]
    prompts, maxt = _prompts(W, model.args.vocab_size)
    single_t, single_l = [], []
    for p, m in zip(prompts, maxt):
        t, lp, _ = _run(model, [p], [m], 1)
        single_t.append(t[0])
        single_l.append(lp[0])

    # B = 1 against a one-shot forward over prompt + generation (no cache).
    for p, t, lp in zip(prompts, single_t, single_l):
        logits = model(mx.array(p + t)[None])[0]
        full = np.array(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        full = full[len(p) - 1 : len(p) - 1 + len(t)]
        assert np.abs(full - lp).max() <= TOL

    for cbs in (8, 4, 3):
        toks, lps, refills_while_long = _run(model, prompts, maxt, cbs)
        assert refills_while_long >= 2, (cbs, refills_while_long)
        worst, div = _compare(single_t, single_l, toks, lps)
        assert worst <= TOL, (cbs, worst)
        assert len(div) <= 1 and all(gap < TOL for _, _, gap in div), (cbs, div)
