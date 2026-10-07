# SPDX-License-Identifier: MIT
"""exp_037's hook mutants 19, 20, 22, 26 and 27 (gate/port_mutants.py; DESIGN
§4.1, §12 W2 and Stage 3b, G1 item 6). 19, 20, 26 and 27 are required
controls; 22 is a reported probe since decision (b), and 27 replaces it as
G5-BP-lean's control (gate/controls.json).

Each overrides only the port's Attention.rope_offset or Attention.attn_mask
hook (DESIGN §6.1), never __call__, and changes the tiny model's output only
in its stated regime, judged bitwise against the unmutated port on the same
weights (the tiny "w513" checkpoint: the real 513-token window):

  19 rope_restart_per_chunk     prefill chunks after the first
  20 decode_window_256          decode steps that see >= 257 keys
  22 batched_rope_positions_in_padded_frame
                                bitwise a no-op at B = 1 (prompts of 600 and
                                1,100 tokens, past the window); changes the
                                logits at B = 2 with unequal prompt lengths
  26 batch_decode_pos_frozen    batch-cache decode, B = 1 included
  27 batch_decode_pad_keys_visible
                                bitwise a no-op where no row is padded (B = 1,
                                rows of equal length) and on prefill and
                                single-sequence caches; at B = 2 with unequal
                                prompts it changes the padded row's decode
                                steps only, the first generated token included

The structure mutant 27 relies on is pinned through mlx-lm 0.32.0's own
BatchGenerator (chunked right-padded prefill, the split to generation, the
mid-run extend, filter, rotation and trims): at every decode step the mask
the port builds for a batch cache is False exactly at the slots that hold
padding (DESIGN §4.1), so its all-True mask is exactly "batched rows ignore
their own left padding".

Batch caches are built and prefilled as mlx-lm 0.32.0's BatchGenerator does
(decision F2): one sequence's caches merged into batch caches (_merge_caches,
generate.py l.815-828), a right-padded prefill finalised to left padding
(PromptProcessingBatch.prompt, l.1163-1209), the last prompt token as the
first decode input (l.1210-1222); and through BatchGenerator itself with the
registered arguments (prefill_batch_size = min(B, 8), prefill_step_size 2,048,
max_kv_size None).

Also: the unmutated port is untouched by a mutated module, and mutants 12 and
13 (registered, unchanged) still run with the port's VllmRoPE.
"""

from __future__ import annotations

import hashlib
import inspect
import types

import numpy as np
import pytest

import tiny_checkpoint as tc

HOOK = {
    "rope_restart_per_chunk": "rope_offset",
    "decode_window_256": "attn_mask",
    "batched_rope_positions_in_padded_frame": "rope_offset",
    "batch_decode_pos_frozen": "rope_offset",
    "batch_decode_pad_keys_visible": "attn_mask",
}
WINDOW = 513  # the w513 preset's sliding_window, as the real model's
MATERIAL = 0.1  # a "change" is a max |delta logit| above this; the observed changes are ~0.5-3


# --- fixtures and drivers -----------------------------------------------------


@pytest.fixture(scope="module")
def w513(tiny_checkpoint_factory):
    d = tiny_checkpoint_factory("w513", 0)
    from gate import common

    assert common.read_json(d / "config.json")["sliding_window"] == WINDOW
    return d


@pytest.fixture(scope="module")
def models(w513):
    """name -> model loaded through gate.common.load_port; None is the port."""
    from gate import common, port_mutants as pm

    return {name: common.load_port(w513, mutant=name)[0] for name in (None,) + pm.HOOK_MUTANTS}


@pytest.fixture(scope="module")
def ids():
    return tc.random_ids(1400, seed=37)


def _np(a):
    import mlx.core as mx

    return np.array(a.astype(mx.float32))


def _prefill(model, toks, chunk, cache):
    import mlx.core as mx

    return [_np(model(mx.array([toks[s:s + chunk]]), cache=cache)) for s in range(0, len(toks), chunk)]


def _decode(model, cache, toks):
    import mlx.core as mx

    return [_np(model(mx.array([[t]]), cache=cache)) for t in toks]


def _single(model, toks, chunk, decode_toks):
    """(prefill logits per chunk, decode logits per step) through
    model.make_cache(): single-sequence KVCache / SlidingKVCache."""
    cache = model.make_cache()
    return _prefill(model, toks, chunk, cache), _decode(model, cache, decode_toks)


def _batch_caches(model, n):
    """mlx-lm 0.32.0's _merge_caches over n fresh make_prompt_cache lists
    (generate.py l.815-828): BatchKVCache / BatchRotatingKVCache."""
    per_seq = [model.make_cache() for _ in range(n)]
    return [per_seq[0][i].merge([c[i] for c in per_seq]) for i in range(len(per_seq[0]))]


def _batched(model, prompts, decode_toks, step=2048, caches=None):
    """A prompt group as BatchGenerator handles it, by hand: prompt[:-1]
    through PromptProcessingBatch.prompt's loop (right-padded, `step`-token
    chunks, finalised to left padding); then the last prompt token, then
    decode_toks[s] (one per row), one step at a time. Returns (prefill logits
    per chunk [B, n, V], decode logits per step [B, 1, V], caches)."""
    import mlx.core as mx

    caches = caches if caches is not None else _batch_caches(model, len(prompts))
    body = [p[:-1] for p in prompts]
    lengths = [len(p) for p in body]
    pad = [max(lengths) - n for n in lengths]
    toks = mx.array([p + [0] * k for p, k in zip(body, pad)])
    if max(pad) > 0:
        for c in caches:
            c.prepare(lengths=lengths, right_padding=pad)
    pre = []
    while toks.shape[1] > 0:
        pre.append(_np(model(toks[:, :step], cache=caches)))
        toks = toks[:, step:]
    if max(pad) > 0:
        for c in caches:
            c.finalize()
    dec, nxt = [], [p[-1] for p in prompts]
    for row in [nxt] + list(decode_toks):
        dec.append(_np(model(mx.array(row)[:, None], cache=caches)))
    return pre, dec, caches


def _generate(model, prompts, max_tokens):
    """mlx-lm 0.32.0's BatchGenerator with the registered arguments
    (BUILD_SPEC item 27; DESIGN §6.3) and a greedy sampler; all prompts
    inserted at once, B = len(prompts). Per prompt: [(token, logprobs fp32)]."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    B = len(prompts)
    gen = BatchGenerator(model, max_tokens=max_tokens, stop_tokens=None,
                         sampler=lambda lp: mx.argmax(lp, axis=-1), completion_batch_size=B,
                         prefill_batch_size=min(B, 8), prefill_step_size=2048, max_kv_size=None)
    try:
        uids = gen.insert(prompts, max_tokens=[max_tokens] * B)
        out = {u: [] for u in uids}
        done: set = set()
        while len(done) < B:
            for r in gen.next_generated():
                out[r.uid].append((int(r.token), _np(r.logprobs)))
                if r.finish_reason is not None:
                    done.add(r.uid)
    finally:
        gen.close()
    return [out[u] for u in uids]


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def _maxdiff(a, b) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


# --- registry and structure ------------------------------------------------------


def test_registry_adds_the_hook_mutants():
    from gate import port_mutants as pm

    assert {i: pm.MUTANT_IDS[i] for i in (19, 20, 22, 26, 27)} == {
        19: "rope_restart_per_chunk", 20: "decode_window_256",
        22: "batched_rope_positions_in_padded_frame", 26: "batch_decode_pos_frozen",
        27: "batch_decode_pad_keys_visible"}
    assert pm.HOOK_MUTANTS == tuple(HOOK)
    assert len(pm.PORT_MUTANTS) == 20 and set(pm.PORT_MUTANTS) == set(pm.PATCHES) == set(pm.TARGETS)
    assert all(pm.NAME_TO_ID[pm.MUTANT_IDS[i]] == i for i in pm.MUTANT_IDS)
    # Each control is bound to one leg (DESIGN §4.2); 22, a probe, names the leg it is reported on.
    assert {n: pm.TARGETS[n] for n in pm.HOOK_MUTANTS} == {
        "rope_restart_per_chunk": "g4_f32_F2_T9", "decode_window_256": "g5_d32_decode",
        "batched_rope_positions_in_padded_frame": "g5_bp_lean_d_K8_B8",
        "batch_decode_pos_frozen": "g5_r1_R_parity_K8",
        "batch_decode_pad_keys_visible": "g5_bp_lean_d_K8_B8"}
    # Not real-weight G2 mutants: the registered tuple is unchanged.
    assert pm.REAL_WEIGHT_MUTANTS == tuple(pm.MUTANT_IDS[i] for i in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15))


def test_controls_json_binds_27_to_g5_bp_lean_and_lists_22_as_a_probe():
    """Decision (b) (DESIGN §4.2): G5BP/27 is G5-BP-lean's required control, with
    Decision 8's fallback unchanged; G5BP/22 is a probe, tiny mode only, and the
    control mutant loop never runs it."""
    from gate import port_mutants as pm, rules
    from gate.checks import g4_forced

    ctl = rules.load_controls()
    bp = [c for c in ctl["required"] if c["check"] == "g5_bp_K8_B8"]
    assert [(c["id"], c["mutant"], c["mutant_name"], c["arm"], c["B"], c["leg"], c["leg_type"], c["where"])
            for c in bp] == [("G5BP/27", 27, "batch_decode_pad_keys_visible", "K8", 8, "d", "ratio",
                              ["first_wave", "mid_run"])]
    assert bp[0]["incomplete_checks"] == [] and bp[0]["if_not_caught"].startswith("allowed_B = {1} for both arms")
    assert 22 not in {c["mutant"] for c in ctl["required"]}
    assert [(p["id"], p["mutant"], p["mutant_name"], p["check"], p["leg"], p["leg_type"], p["modes"])
            for p in ctl["probes"]] == [("G5BP/22", 22, "batched_rope_positions_in_padded_frame", "g5_bp_K8_B8",
                                         "d", "ratio", ["tiny"])]
    for c in ctl["required"] + ctl["probes"]:
        assert pm.MUTANT_IDS[c["mutant"]] == c["mutant_name"], c["id"]
    plan = g4_forced.control_plan(["g5_bp_K8_B8"], ctl)
    assert [(e["mutant"], e["name"]) for e in plan] == [(27, "batch_decode_pad_keys_visible")]


@pytest.mark.parametrize("name", list(HOOK))
def test_hook_mutants_override_only_their_hook(name):
    """Classes 19, 20, 22, 26 and 27 do not override __call__ (or __init__): a port
    fix to Attention.__call__ reaches them (DESIGN §4.1)."""
    from gate import port_mutants as pm

    module = pm.mutant_module(name)
    cls = module.Attention
    own = {k for k in vars(cls) if not (k.startswith("__") and k.endswith("__"))}
    assert own == {HOOK[name]}, own
    assert "__call__" not in vars(cls) and "__init__" not in vars(cls)
    base = cls.__mro__[1]
    assert base.__module__ == module.__name__ and base.__name__ == "Attention"  # the port's own class
    assert cls.__call__ is base.__call__
    assert "rope_offset" in vars(base) and "attn_mask" in vars(base)  # the port's hooks (W1)
    # The port's __call__ reaches both hooks.
    src = inspect.getsource(base.__call__)
    assert "self.rope_offset(" in src and "self.attn_mask(" in src
    assert module.EXP036_PORT_MUTANT == name


@pytest.mark.parametrize("name", list(HOOK))
def test_hook_mutants_refuse_a_port_without_the_hook(tmp_path, name):
    """On a port without the hooks a hook mutant would be a silent no-op (and
    its control would never fire): it is refused instead."""
    from gate import common, port_mutants as pm

    src = common.PORT_FILE.read_text()
    assert f"def {HOOK[name]}(" in src
    bare = tmp_path / "kolibri1.py"
    bare.write_text(src.replace(f"def {HOOK[name]}(", f"def _no_{HOOK[name]}("))
    with pytest.raises(AttributeError, match=HOOK[name]):
        pm.mutant_module(name, port_file=bare)


# --- the unmutated port ------------------------------------------------------------


def _port_snapshot(model, ids):
    pre, dec = _single(model, ids[:600], 256, ids[600:604])
    bpre, bdec, _ = _batched(model, [ids[:300], ids[300:1000]], [[ids[1000], ids[1001]]], step=256)
    return pre + dec + bpre + bdec


def test_unmutated_port_is_untouched_by_a_mutated_module(w513, models, ids):
    from gate import common, port_mutants as pm

    sha = hashlib.sha256(common.PORT_FILE.read_bytes()).hexdigest()
    before = _port_snapshot(common.load_port(w513)[0], ids)
    for name in pm.HOOK_MUTANTS:  # build and run every control mutant
        _port_snapshot(common.load_port(w513, mutant=name)[0], ids)
    assert hashlib.sha256(common.PORT_FILE.read_bytes()).hexdigest() == sha
    fresh = common.port_module(None)
    assert fresh.Attention.__bases__[0].__name__ == "Module"  # no mutant subclass leaked in
    for hook in ("rope_offset", "attn_mask"):
        assert vars(fresh.Attention)[hook].__qualname__ == f"Attention.{hook}"
    after = _port_snapshot(common.load_port(w513)[0], ids)
    assert _same(before, after)
    # The shared port model of the other tests is the unmutated one.
    assert type(models[None].model.layers[0].self_attn).__mro__[1].__name__ == "Module"


# --- 19: rope_restart_per_chunk ------------------------------------------------------


def test_m19_single_chunk_and_decode_are_the_port(models, ids):
    import mlx.core as mx

    port, mut = models[None], models["rope_restart_per_chunk"]
    for T in (300, 600, 1100):  # one prefill chunk, then decode (past the window for 600 and 1,100)
        a = _single(port, ids[:T], T, ids[T:T + 6])
        b = _single(mut, ids[:T], T, ids[T:T + 6])
        assert _same(a[0] + a[1], b[0] + b[1])
    x = mx.array([ids[:300]])
    assert np.array_equal(_np(port(x)), _np(mut(x)))  # no cache: one chunk from 0


@pytest.mark.parametrize("chunk", [100, 256])
def test_m19_changes_prefill_chunks_after_the_first(models, ids, chunk):
    port, mut = models[None], models["rope_restart_per_chunk"]
    a, _ = _single(port, ids[:600], chunk, [])
    b, _ = _single(mut, ids[:600], chunk, [])
    assert np.array_equal(a[0], b[0])  # the first chunk starts at 0 anyway
    assert all(_maxdiff(x, y) > MATERIAL for x, y in zip(a[1:], b[1:])), [_maxdiff(x, y) for x, y in zip(a, b)]


# --- 20: decode_window_256 -------------------------------------------------------------


def test_m20_prefill_and_short_decode_are_the_port(models, ids):
    port, mut = models[None], models["decode_window_256"]
    for T, chunk in ((600, 600), (600, 256), (1100, 2048)):
        assert _same(_single(port, ids[:T], chunk, [])[0], _single(mut, ids[:T], chunk, [])[0])
    # 250 cached keys, then 6 decode steps: 251 ... 256 keys, every key kept.
    cp, cm = port.make_cache(), mut.make_cache()
    assert _same(_prefill(port, ids[:250], 250, cp), _prefill(mut, ids[:250], 250, cm))
    assert _same(_decode(port, cp, ids[250:256]), _decode(mut, cm, ids[250:256]))
    # The 257th key: the oldest one (of 257) drops out of the mutant's view, a
    # smaller change than the 257 of 513 dropped past the window.
    assert _maxdiff(_decode(port, cp, ids[256:257])[0], _decode(mut, cm, ids[256:257])[0]) > 1e-3


@pytest.mark.parametrize("T,chunk", [(300, 300), (600, 600), (1100, 256)])
def test_m20_changes_decode_with_257_or_more_keys(models, ids, T, chunk):
    port, mut = models[None], models["decode_window_256"]
    _, a = _single(port, ids[:T], chunk, ids[T:T + 6])
    _, b = _single(mut, ids[:T], chunk, ids[T:T + 6])
    assert all(_maxdiff(x, y) > MATERIAL for x, y in zip(a, b)), [_maxdiff(x, y) for x, y in zip(a, b)]


def test_m20_leaves_batch_caches_and_full_layers_alone(models, ids):
    port, mut = models[None], models["decode_window_256"]
    a = _batched(port, [ids[:600]], [[ids[600]], [ids[601]]])
    b = _batched(mut, [ids[:600]], [[ids[600]], [ids[601]]])
    assert _same(a[0] + a[1], b[0] + b[1])
    layer_types = [layer.self_attn.rope is not None for layer in mut.model.layers]
    assert True in layer_types and False in layer_types  # w513 has sliding and full layers


@pytest.mark.parametrize("prefill,chunk,steps", [(100, 100, 600), (600, 600, 300), (750, 250, 120), (513, 513, 40)])
def test_m20_keeps_exactly_the_256_most_recent_keys(prefill, chunk, steps):
    """decode_window_keep on a RotatingKVCache(513, keep=0) whose keys carry
    their own position: through growth, the first rotation and the trim after
    an over-long prefill, the kept slots are positions offset-256 ... offset-1."""
    import mlx.core as mx
    from mlx_lm.models.cache import RotatingKVCache

    from gate import port_mutants as pm

    def kv(p0, n):
        k = mx.arange(p0, p0 + n, dtype=mx.float32).reshape(1, 1, n, 1)
        return k, k

    cache = RotatingKVCache(max_size=WINDOW, keep=0)
    for s in range(0, prefill, chunk):
        cache.update_and_fetch(*kv(s, min(chunk, prefill - s)))
    for t in range(prefill, prefill + steps):
        keys, _ = cache.update_and_fetch(*kv(t, 1))
        assert cache.offset == t + 1
        keep = pm.decode_window_keep(cache)
        n = keys.shape[2]
        if n <= pm.DECODE_WINDOW_KEPT:
            assert keep is None
            continue
        assert keep.shape == (n,) and keep.dtype == mx.bool_
        kept = sorted(np.array(keys[0, 0, :, 0])[np.array(keep)].astype(int).tolist())
        assert kept == list(range(t + 1 - pm.DECODE_WINDOW_KEPT, t + 1)), (t, n)
    with pytest.raises(ValueError, match="keep = 0"):
        sink = RotatingKVCache(max_size=WINDOW, keep=4)
        sink.update_and_fetch(*kv(0, 300))
        pm.decode_window_keep(sink)


# --- 22: batched_rope_positions_in_padded_frame ------------------------------------------


@pytest.mark.parametrize("T", [600, 1100])
@pytest.mark.parametrize("step", [2048, 256])
def test_m22_is_bitwise_a_no_op_at_b1(models, ids, T, step):
    port, mut = models[None], models["batched_rope_positions_in_padded_frame"]
    dec = [[t] for t in ids[T:T + 8]]
    a = _batched(port, [ids[:T]], dec, step=step)
    b = _batched(mut, [ids[:T]], dec, step=step)
    assert T > WINDOW and len(a[1]) == 9
    assert _same(a[0] + a[1], b[0] + b[1])


def test_m22_leaves_single_sequence_caches_alone(models, ids):
    port, mut = models[None], models["batched_rope_positions_in_padded_frame"]
    a = _single(port, ids[:1100], 256, ids[1100:1106])
    b = _single(mut, ids[:1100], 256, ids[1100:1106])
    assert _same(a[0] + a[1], b[0] + b[1])


@pytest.mark.parametrize("lens", [(300, 700), (1100, 600)])
def test_m22_changes_the_logits_at_b2_with_unequal_prompts(models, ids, lens):
    port, mut = models[None], models["batched_rope_positions_in_padded_frame"]
    prompts = [ids[:lens[0]], ids[100:100 + lens[1]]]
    dec = [[ids[1200 + s], ids[1300 + s]] for s in range(5)]
    a = _batched(port, prompts, dec)
    b = _batched(mut, prompts, dec)
    short, long_ = int(np.argmin(lens)), int(np.argmax(lens))
    assert _same(a[0], b[0])  # right-padded prefill: every row starts at offset 0
    for x, y in zip(a[1], b[1]):
        assert np.array_equal(x[long_], y[long_])  # the longest row is at its own position
        assert _maxdiff(x[short], y[short]) > MATERIAL  # the shorter row decodes at a shifted position


def test_m22_through_batch_generator(models, ids):
    port, mut = models[None], models["batched_rope_positions_in_padded_frame"]
    one = [ids[:600]]
    assert _same([lp for _, lp in _generate(port, one, 6)[0]], [lp for _, lp in _generate(mut, one, 6)[0]])
    two = [ids[:300], ids[200:900]]
    a, b = _generate(port, two, 6), _generate(mut, two, 6)
    assert _same([lp for _, lp in a[1]], [lp for _, lp in b[1]])
    assert _maxdiff(a[0][0][1], b[0][0][1]) > MATERIAL


# --- 26: batch_decode_pos_frozen ----------------------------------------------------------


def test_m26_leaves_prefill_and_single_sequence_caches_alone(models, ids):
    port, mut = models[None], models["batch_decode_pos_frozen"]
    a = _single(port, ids[:600], 256, ids[600:608])
    b = _single(mut, ids[:600], 256, ids[600:608])
    assert _same(a[0] + a[1], b[0] + b[1])


@pytest.mark.parametrize("T", [300, 600, 1100])
def test_m26_changes_batch_cache_decode_at_b1(models, ids, T):
    from gate import port_mutants as pm

    port, mut = models[None], models["batch_decode_pos_frozen"]
    dec = [[t] for t in ids[T:T + 6]]
    a = _batched(port, [ids[:T]], dec)
    b = _batched(mut, [ids[:T]], dec)
    assert _same(a[0], b[0])  # prefill
    assert np.array_equal(a[1][0], b[1][0])  # the first decode step is the port's
    assert all(_maxdiff(x, y) > MATERIAL for x, y in zip(a[1][1:], b[1][1:]))
    # The frozen value is a copy of the offset at the first decode step (prompt[:-1]
    # prefilled, so T - 1), while the cache's own offset moved on.
    rope_caches = [c for c, layer in zip(b[2], mut.model.layers) if layer.self_attn.rope is not None]
    assert rope_caches
    for c in rope_caches:
        assert np.array(getattr(c, pm.FROZEN_OFFSET_ATTR)).tolist() == [T - 1]
        assert np.array(c.offset).tolist() == [T - 1 + len(dec) + 1]
    full = [c for c, layer in zip(b[2], mut.model.layers) if layer.self_attn.rope is None]
    assert full and not any(hasattr(c, pm.FROZEN_OFFSET_ATTR) for c in full)  # NoPE layers: never asked


def test_m26_refreezes_when_the_batch_shape_changes(models, ids):
    from gate import port_mutants as pm

    port, mut = models[None], models["batch_decode_pos_frozen"]
    prompts = [ids[:300], ids[300:1000]]
    dec = [[ids[1100 + s], ids[1200 + s]] for s in range(3)]
    a = _batched(port, prompts, dec)
    b = _batched(mut, prompts, dec)
    assert _same(a[0], b[0]) and np.array_equal(a[1][0], b[1][0])
    assert all(_maxdiff(x, y) > MATERIAL for x, y in zip(a[1][1:], b[1][1:]))
    # Drop row 0, as BatchGenerator does when a sequence finishes: the next decode
    # step re-freezes on the remaining row's current offset.
    caches = b[2]
    rope = next(c for c, layer in zip(caches, mut.model.layers) if layer.self_attn.rope is not None)
    for c in caches:
        c.filter([1])
    now = np.array(rope.offset).tolist()
    assert np.array(getattr(rope, pm.FROZEN_OFFSET_ATTR)).shape == (2,)
    import mlx.core as mx

    mut(mx.array([[ids[1300]]]), cache=caches)
    assert np.array(getattr(rope, pm.FROZEN_OFFSET_ATTR)).tolist() == now
    mut(mx.array([[ids[1301]]]), cache=caches)
    assert np.array(getattr(rope, pm.FROZEN_OFFSET_ATTR)).tolist() == now
    assert np.array(rope.offset).tolist() == [now[0] + 2]


def test_m26_freezes_a_copy_not_a_reference():
    """mlx-lm 0.32.0 rebinds cache.offset (#1848); an in-place update (`+=` on an
    mx.array mutates every reference to it) must not move the frozen value
    either. A stub batch cache: only `left_padding`'s presence is used."""
    import mlx.core as mx

    from gate import port_mutants as pm

    m = pm.mutant_module("batch_decode_pos_frozen")
    attn = m.Attention(m.ModelArgs.from_dict(tc.preset_config("w513")), use_rope=True)

    class StubBatchCache:
        def __init__(self, offsets):
            self.left_padding = None  # present, never read
            self.offset = mx.array(offsets)

    c = StubBatchCache([10, 7])
    assert np.array(attn.rope_offset(c, 5)).tolist() == [10, 7]  # prefill: the port's offset
    assert not hasattr(c, pm.FROZEN_OFFSET_ATTR)
    first = attn.rope_offset(c, 1)
    c.offset += 1  # in place
    assert np.array(attn.rope_offset(c, 1)).tolist() == [10, 7] == np.array(first).tolist()
    c.offset = c.offset + 1  # rebound, as 0.32.0 does
    assert np.array(attn.rope_offset(c, 1)).tolist() == [10, 7]
    c.offset = mx.array([12])  # a row filtered out: re-frozen
    assert np.array(attn.rope_offset(c, 1)).tolist() == [12]
    assert attn.rope_offset(types.SimpleNamespace(offset=30), 1) == 30  # single-sequence cache: the port's


def test_m26_through_batch_generator_at_b1(models, ids):
    port, mut = models[None], models["batch_decode_pos_frozen"]
    a, b = _generate(port, [ids[:700]], 8)[0], _generate(mut, [ids[:700]], 8)[0]
    assert np.array_equal(a[0][1], b[0][1])  # the first generated token's log-probs
    later = [_maxdiff(x[1], y[1]) for x, y in zip(a[1:], b[1:])]
    assert max(later) > MATERIAL, later


# --- 27: batch_decode_pad_keys_visible (decision (b)) ---------------------------------------


@pytest.mark.parametrize("T", [600, 1100])
@pytest.mark.parametrize("step", [2048, 256])
def test_m27_is_bitwise_a_no_op_at_b1(models, ids, T, step):
    port, mut = models[None], models["batch_decode_pad_keys_visible"]
    dec = [[t] for t in ids[T:T + 8]]
    a = _batched(port, [ids[:T]], dec, step=step)
    b = _batched(mut, [ids[:T]], dec, step=step)
    assert T > WINDOW and len(a[1]) == 9
    assert _same(a[0] + a[1], b[0] + b[1])


def test_m27_is_bitwise_a_no_op_with_rows_of_equal_length(models, ids):
    port, mut = models[None], models["batch_decode_pad_keys_visible"]
    prompts = [ids[:700], ids[300:1000]]
    dec = [[ids[1100 + s], ids[1200 + s]] for s in range(6)]
    a, b = _batched(port, prompts, dec), _batched(mut, prompts, dec)
    assert _same(a[0] + a[1], b[0] + b[1])


def test_m27_leaves_single_sequence_caches_alone(models, ids):
    port, mut = models[None], models["batch_decode_pad_keys_visible"]
    for T, chunk in ((300, 300), (1100, 256)):
        a = _single(port, ids[:T], chunk, ids[T:T + 6])
        b = _single(mut, ids[:T], chunk, ids[T:T + 6])
        assert _same(a[0] + a[1], b[0] + b[1])


@pytest.mark.parametrize("lens", [(300, 700), (1100, 600)])
def test_m27_changes_only_the_padded_row_at_decode(models, ids, lens):
    """Prefill (L > 1) is the port's, the longest row (no padding) is the port's at
    every step, and the padded row changes at every decode step. That includes the
    first one: the last prompt token is an L == 1 step over the padded caches."""
    port, mut = models[None], models["batch_decode_pad_keys_visible"]
    prompts = [ids[:lens[0]], ids[100:100 + lens[1]]]
    dec = [[ids[1200 + s], ids[1300 + s]] for s in range(5)]
    a, b = _batched(port, prompts, dec), _batched(mut, prompts, dec)
    short, long_ = int(np.argmin(lens)), int(np.argmax(lens))
    assert _same(a[0], b[0])
    for x, y in zip(a[1], b[1]):
        assert np.array_equal(x[long_], y[long_])
        assert _maxdiff(x[short], y[short]) > MATERIAL, [_maxdiff(x[short], y[short]) for x, y in zip(a[1], b[1])]


def test_m27_through_batch_generator(models, ids):
    port, mut = models[None], models["batch_decode_pad_keys_visible"]
    one = [ids[:600]]
    assert _same([lp for _, lp in _generate(port, one, 6)[0]], [lp for _, lp in _generate(mut, one, 6)[0]])
    two = [ids[:300], ids[200:900]]
    a, b = _generate(port, two, 6), _generate(mut, two, 6)
    assert _same([lp for _, lp in a[1]], [lp for _, lp in b[1]])  # the longer prompt's row: no padding
    # The shorter prompt's row is padded; its first generated token comes from the last
    # prompt token's L == 1 step over the group's padded caches, so it already moves.
    assert _maxdiff(a[0][0][1], b[0][0][1]) > MATERIAL


class _MarkerModel:
    """A stand-in for the port inside mlx-lm 0.32.0's BatchGenerator: a full layer
    (KVCache) and a sliding layer (RotatingKVCache(window, keep=0), as the port's
    make_cache), both masks built before any update, as Kolibri1Model.make_masks
    builds them. Every key is its token's id: prompt and generated tokens are >= 1,
    _right_pad_prompts pads with 0 and extend pads with zeros, so a slot holds
    padding iff its key is < 1. Records every L == 1 step's masks and keys."""

    VOCAB = 4

    def __init__(self, window: int):
        self.window = window
        self.steps = []

    def make_cache(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        return [KVCache(), RotatingKVCache(max_size=self.window, keep=0)]

    def __call__(self, inputs, cache=None, **kwargs):
        import mlx.core as mx
        from mlx_lm.models.base import create_attention_mask

        B, T = inputs.shape
        h = mx.zeros((B, T, 1))
        masks = [create_attention_mask(h, cache[0]), create_attention_mask(h, cache[1], window_size=self.window)]
        k = inputs.astype(mx.float32)[:, None, :, None]
        for layer, (c, mask) in enumerate(zip(cache, masks)):
            keys, _ = c.update_and_fetch(k, k)
            if T == 1:
                self.steps.append({"layer": layer, "is_array": isinstance(mask, mx.array),
                                   "mask": np.array(mask) if isinstance(mask, mx.array) else None,
                                   "keys": np.array(keys[:, 0, :, 0]), "rotated": bool(getattr(c, "rotated", False))})
        return mx.broadcast_to(mx.array([0.0, 4.0, 0.0, 0.0]), (B, T, self.VOCAB))  # greedy: token 1


@pytest.mark.parametrize("window,step,lengths", [
    (WINDOW, 64, (7, 30, 70, 110, 12, 52, 90, 15, 600, 1100, 37, 700, 64, 520)),
    (33, 16, (5, 40, 9, 70, 17, 3, 33, 50, 26, 8, 61, 12)),
])
def test_m27_decode_masks_are_false_exactly_at_padding(window, step, lengths):
    """DESIGN §4.1's structure, pinned through mlx-lm 0.32.0's BatchGenerator with
    the registered construction at B = 4 (prefill_batch_size = min(B, 8), chunked
    right-padded prefill, the split to generation, mid-run extend, filter, and the
    rotating cache's trims and rotations): at every decode step, on both cache kinds,
    the mask is an array that is False exactly at the slots holding padding. Mutant
    27's all-True mask is therefore exactly "rows attend to their own padding"."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    model = _MarkerModel(window)
    B = 4
    gen = BatchGenerator(model, max_tokens=64, stop_tokens=None, sampler=lambda lp: mx.argmax(lp, axis=-1),
                         completion_batch_size=B, prefill_batch_size=min(B, 8), prefill_step_size=step,
                         max_kv_size=None)
    rng = np.random.default_rng(37)
    prompts = [[int(t) for t in rng.integers(2, 1000, size=n)] for n in lengths]
    max_tokens = [3 + (7 * j) % 29 for j in range(len(prompts))]
    try:
        uids = gen.insert(prompts, max_tokens=max_tokens)
        got = {u: 0 for u in uids}
        while sum(got.values()) < sum(max_tokens):
            for r in gen.next_generated():
                got[r.uid] += 1
                assert int(r.token) == 1
    finally:
        gen.close()
    assert [got[u] for u in uids] == max_tokens
    padded = {0: 0, 1: 0}
    rows = []
    for s in model.steps:
        assert s["is_array"], s["layer"]
        m, keys = s["mask"], s["keys"]
        assert m.dtype == np.bool_ and m.shape == (keys.shape[0], 1, 1, keys.shape[1])
        assert np.array_equal(m[:, 0, 0, :], keys >= 1), (s["layer"], keys.shape)
        padded[s["layer"]] += int((keys < 1).any())
        rows.append(keys.shape[0])
    # The run exercised what the claim is about: padded decode steps on both cache
    # kinds, a batch that grew mid-run (extend) and shrank (filter), and rotation.
    assert padded[0] > 0 and padded[1] > 0, padded
    assert any(b > a for a, b in zip(rows, rows[1:])) and any(b < a for a, b in zip(rows, rows[1:]))
    assert any(s["rotated"] for s in model.steps if s["layer"] == 1)


# --- registered mutants 12 and 13 under VllmRoPE ------------------------------------------


def test_mutants_12_and_13_still_run_with_vllm_rope(w513, models, ids):
    """12 replaces self.rope (traditional nn.RoPE); 13 overrides __call__ and
    keeps calling self.rope(q, offset=...), which is the port's VllmRoPE."""
    import mlx.core as mx

    from gate import common, port_mutants as pm

    port = models[None]
    sliding = [layer.self_attn for layer in port.model.layers if layer.self_attn.rope is not None]
    assert sliding and all(type(a.rope).__name__ == "VllmRoPE" for a in sliding)

    m13 = pm.mutant_module("qknorm_after_rope")
    assert "__call__" in vars(m13.Attention)
    calls = []
    rope_call = m13.VllmRoPE.__call__

    def spy(self, x, offset=0):
        calls.append(offset)
        return rope_call(self, x, offset=offset)

    m13.VllmRoPE.__call__ = spy  # this fresh module's class only
    model13 = common.load_port(w513, module=m13)[0]
    m12 = common.load_port(w513, mutant="rope_traditional")[0]
    assert all(type(layer.self_attn.rope).__name__ == "VllmRoPE"
               for layer in model13.model.layers if layer.self_attn.rope is not None)
    assert all(type(layer.self_attn.rope).__name__ == "RoPE" and layer.self_attn.rope.traditional
               for layer in m12.model.layers if layer.self_attn.rope is not None)

    ref = _single(port, ids[:600], 256, ids[600:603])
    for model in (model13, m12):
        got = _single(model, ids[:600], 256, ids[600:603])
        assert all(np.isfinite(x).all() for x in got[0] + got[1])
        assert _maxdiff(ref[0][0], got[0][0]) > MATERIAL
    n_rope_layers = len(sliding)
    assert len(calls) == 2 * n_rope_layers * (3 + 3)  # q and k, per rope layer, 3 chunks + 3 steps
    assert sorted({int(np.array(o)) if not isinstance(o, int) else o for o in calls}) == [0, 256, 512, 600, 601, 602]
    assert m13.VllmRoPE is not common.exec_port_module().VllmRoPE  # the spy stayed in m13
