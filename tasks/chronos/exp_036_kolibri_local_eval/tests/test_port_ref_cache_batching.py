# SPDX-License-Identifier: MIT
"""The port under mlx_lm's continuous batching (spec items 19, 24, 27;
`test_cache_batching.py` in BUILD_SPEC 6, named test_port_ref_* here).

BatchGenerator (mlx_lm 0.31.3) turns the port's caches into batch caches:
KVCache -> BatchKVCache, SlidingKVCache (a RotatingKVCache with keep 0) ->
BatchRotatingKVCache(max_size = sliding_window). Sequences are left-padded
to the longest live one, and finished sequences are replaced mid-run
("refill"). None of that may change a sequence's numbers:

* _make_cache maps the port's caches as above; SlidingKVCache.to_quantized
  returns itself (item 19);
* a left-padded batch of 4 reproduces each sequence run alone (fp32, 1e-4);
* test_refill_rotated: the tiny vendor preset in fp32 at W = 65 and at the
  real W = 513, completion_batch_size 8, 4 and 3, twelve prompts of up to 3W
  tokens generating up to 3W tokens, with at least two admissions while
  another live sequence is past 2W: every sequence's greedy log-probs equal
  its B = 1 run's within 1e-4, and the B = 1 run equals a one-shot forward
  over prompt + generation within 1e-4.

Tolerance: fp32 log-probs of these logits (std ~0.8, |logprob| < 10) carry
a few 1e-6 of summation-order noise between batch shapes, so 1e-4 leaves
more than 20x headroom. Greedy decoding can still diverge at an exact
near-tie; the comparison then stops at the first differing token, which must
sit at a top-2 log-prob gap below 1e-4 in the B = 1 run, and at most one such
divergence is allowed per configuration.
Measured (W = 65 and 513): batched vs B = 1 <= 4.3e-6; B = 1 vs one-shot
forward <= 4.8e-6; no divergence; 4, 8 and 7 admissions while another
sequence is past 2W at completion_batch_size 8, 4 and 3.
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
    """Greedy generation through BatchGenerator; per sequence the tokens and
    the full fp32 log-prob vectors, plus the refill events."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    gen = BatchGenerator(model, max_tokens=max(maxt), completion_batch_size=cbs,
                         prefill_batch_size=min(cbs, 8), sampler=lambda x: mx.argmax(x, axis=-1))
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


def test_make_cache_maps_the_port_caches(models):
    from mlx_lm.generate import _make_cache
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    for W, model in models.items():
        caches = _make_cache(model, [0, 3, 1], None)
        for layer, c in zip(model.layers, caches):
            if layer.use_sliding:
                assert type(c) is BatchRotatingKVCache and c.max_size == W
            else:
                assert type(c) is BatchKVCache
        sliding = [c for c in model.make_cache() if type(c).__name__ == "SlidingKVCache"]
        assert sliding and all(c.to_quantized(group_size=64, bits=8) is c for c in sliding)


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
