# SPDX-License-Identifier: MIT
"""KV-cache path of the port: prefill + token-by-token decode, and chunked
prefill, must reproduce the full-sequence forward (spec items 8-10, 19, 24).

Port only (no reference): the full-sequence forward without a cache is the
oracle, and test_port_vs_ref.py ties that forward to the reference.

What can go wrong here and is not visible without a cache:
  * RoPE offsets: sliding layers must rotate with absolute positions
    (cache.offset), also after the rotating buffer has wrapped;
  * the rotating buffer (SlidingKVCache, max_size = sliding_window, keep = 0)
    must hold exactly the current token plus the W-1 before it at decode,
    and W-1 old + the new chunk at prefill, with the right mask;
  * full layers (KVCache) must keep every key, with no window and no RoPE.

Sequence lengths make every buffer wrap several times: vendor W=65 with 340
tokens, pattern5 W=17 with 160 tokens. Plans:
  decode-from-short   prefill W//2 tokens (below the window), decode the rest
  decode-from-long    prefill 2W+3 tokens (the buffer starts over-full), decode the rest
  chunks-below-window prefill steps of W//3 tokens
  chunks-above-window prefill steps of W + W//2 tokens

fp32: every position within 1e-4 relative of the full forward, same argmax
everywhere. Measured: 1.2e-6 to 1.6e-6.
bf16: see the two bf16 tests; the thresholds follow the same error model as
test_port_vs_ref.py (b1, b2) and INTEGRATION_LOG.md entry 2.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling

PRESETS = {"vendor": 340, "pattern5": 160}  # preset -> sequence length
SEEDS = (0, 1)
PLANS = ("decode-from-short", "decode-from-long", "chunks-below-window", "chunks-above-window")


def make_plan(name: str, T: int, W: int) -> list[int]:
    """Token counts of the successive model calls; they sum to T."""
    def chunks(step):
        return [step] * (T // step) + ([T % step] if T % step else [])

    if name == "decode-from-short":
        return [W // 2] + [1] * (T - W // 2)
    if name == "decode-from-long":
        return [2 * W + 3] + [1] * (T - 2 * W - 3)
    if name == "chunks-below-window":
        return chunks(W // 3)
    if name == "chunks-above-window":
        return chunks(W + W // 2)
    raise ValueError(name)


@pytest.fixture(
    scope="module",
    params=[(p, s) for p in PRESETS for s in SEEDS],
    ids=lambda c: f"{c[0]}-seed{c[1]}",
)
def seq(request, tiny_checkpoint_factory):
    import_sibling("port.kolibri1")
    preset, seed = request.param
    cfg = tc.preset_config(preset)
    T, W = PRESETS[preset], cfg["sliding_window"]
    return SimpleNamespace(
        preset=preset,
        dir=tiny_checkpoint_factory(preset, seed),
        ids=tc.random_ids(T, seed=2000 + seed),
        T=T,
        W=W,
        cfg=cfg,
    )


def run_cached(model, ids, plan, forced=None):
    """Feed `ids` through make_cache() caches in calls of `plan` tokens.
    Returns (logits [T, V] float32, caches)."""
    cache = model.make_cache()
    out, pos = [], 0
    for n in plan:
        if forced is not None:
            forced.start = pos
        out.append(ph.port_logits(model, ids[pos : pos + n], cache=cache))
        pos += n
    assert pos == len(ids)
    return np.concatenate(out), cache


def check_caches(seq, cache, plan):
    """The caches are the kinds the spec asks for and ended where they should."""
    for layer_type, c in zip(seq.cfg["layer_types"], cache):
        assert c.offset == seq.T
        if layer_type == "sliding_attention":
            # Item 24: rotating buffer of exactly W, no sink tokens.
            assert type(c).__name__ == "SlidingKVCache"
            assert c.max_size == seq.W and c.keep == 0
            # Decode leaves exactly W keys; a prefill chunk of S leaves W-1+S.
            assert c.keys.shape[2] <= seq.W - 1 + max(plan)
        else:
            assert type(c).__name__ == "KVCache"
            assert c.keys.shape[2] >= seq.T


@pytest.mark.parametrize("plan_name", PLANS)
def test_fp32_cached_matches_full_forward(seq, plan_name):
    model = ph.load_port(seq.dir, float32=True)
    full = ph.port_logits(model, seq.ids)
    plan = make_plan(plan_name, seq.T, seq.W)
    if plan_name.startswith("decode"):
        # The decode phase alone wraps the W-slot buffer at least three times.
        assert plan.count(1) >= 3 * seq.W
    got, cache = run_cached(model, seq.ids, plan)
    check_caches(seq, cache, plan)
    rel = np.abs(got - full).max(axis=-1) / np.abs(full).max()
    bad = np.flatnonzero(rel > 1e-4)
    assert bad.size == 0, f"positions {bad[:10].tolist()} differ, worst rel {rel.max():.2e}"
    assert (got.argmax(-1) == full.argmax(-1)).all()


@pytest.mark.parametrize("plan_name", PLANS)
def test_bf16_cached_matches_full_forward_forced_routing(seq, plan_name, monkeypatch):
    """bf16 cached run vs bf16 full forward, with both runs on the same
    experts. The cached run is forced onto the ids that the full forward
    selected.

    The two runs differ only in GEMM shapes (one row per decode step versus T
    rows), which MLX rounds differently in bf16. The thresholds are those of
    test_port_vs_ref (b1): max|d| / max|full| <= 5e-2 (sqrt(n) * u with n ~ 160
    sequential roundings), and top-1 agreement >= 0.98 on positions where the
    full forward leads by DECISIVE_GAP = 0.1 or more. Measured: rel 1.1e-2 to
    1.8e-2, decisive top-1 1.000, overall top-1 0.97-1.00.
    """
    model = ph.load_port(seq.dir)
    routing = ph.record_routing(monkeypatch, model)
    full = ph.port_logits(model, seq.ids)
    assert len(routing) == seq.cfg["num_hidden_layers"]
    forced = ph.force_routing(monkeypatch, model, routing)
    got, _ = run_cached(model, seq.ids, make_plan(plan_name, seq.T, seq.W), forced)
    stats = ph.compare(full, got)
    assert stats["rel_max"] <= 5e-2, ph.fmt(stats)
    assert stats["top1_decisive"] >= 0.98, ph.fmt(stats)


@pytest.mark.parametrize("plan_name", PLANS)
def test_bf16_cached_vs_full_forward_natural_routing(seq, plan_name):
    """bf16 generation path as used (own routing) vs the bf16 full forward.

    The GEMM-shape rounding differences flip near-tied routes, just as bf16
    vs fp32 does. The bounds are the worst-case bounds of test_port_vs_ref
    (b2): top-1 >= 0.60 and mean KL <= 3e-2. Measured: top-1 0.92-0.98, mean
    KL 1.9e-4 to 2.3e-3.
    """
    model = ph.load_port(seq.dir)
    full = ph.port_logits(model, seq.ids)
    got, _ = run_cached(model, seq.ids, make_plan(plan_name, seq.T, seq.W))
    stats = ph.compare(full, got)
    assert np.isfinite(got).all()
    assert stats["top1"] >= 0.60, ph.fmt(stats)
    assert stats["kl_mean"] <= 3e-2, ph.fmt(stats)


def test_kv_bits_quantises_only_full_layers(tiny_pattern5_dir):
    """mlx_lm's --kv-bits path (generate.maybe_quantize_kv_cache) calls
    to_quantized on every cache. Sliding caches must stay SlidingKVCache and
    only the full-attention KVCaches become QuantizedKVCache (spec item 19)."""
    import_sibling("port.kolibri1")
    from mlx_lm.generate import maybe_quantize_kv_cache

    model = ph.load_port(tiny_pattern5_dir)
    cache = model.make_cache()
    ids = tc.random_ids(40, seed=7)
    ph.port_logits(model, ids, cache=cache)
    maybe_quantize_kv_cache(cache, quantized_kv_start=0, kv_group_size=32, kv_bits=8)
    kinds = [type(c).__name__ for c in cache]
    for layer_type, kind in zip(tc.preset_config("pattern5")["layer_types"], kinds):
        expected = "SlidingKVCache" if layer_type == "sliding_attention" else "QuantizedKVCache"
        assert kind == expected, kinds
    # Decoding continues on the mixed caches.
    out = ph.port_logits(model, [5], cache=cache)
    assert np.isfinite(out).all()
