# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06).
"""G1 item 1 (DESIGN §3.3, §6.1): the port's vLLM-angle RoPE and its attention hooks.

* Angle. VllmRoPE._freqs is float32(1 / rope_inv_freq(128, 1e4)) bitwise, and
  float32(1 / _freqs) is rope_inv_freq bitwise, so the kernel's angle
  pos * (1 / freqs) is vLLM's fp32 angle pos * inv_freq.
* Rotation. Rotated vectors match reference.kolibri_ref.rope_neox within 1e-6
  relative L2 per vector at every position 0-20,479 (ten prefill-sized calls of
  2,048 positions at integer offsets) and at every 97th position up to 262,143,
  plus 262,143 itself (one decode-sized call each). The per-octave maxima are
  the committed output tests/fixtures/rope_angle_262143.json: this test checks
  that the file belongs to its definition and is within the bound, and, on the
  runtime that wrote it (same mlx, numpy and GPU architecture), that it equals
  this run. EXP037_WRITE_ROPE_FIXTURE=1 rewrites it (build host, before the
  pre-registration). This run's maxima also go into the junit record as
  test-suite properties.
* Array offsets. A scalar array offset, and one offset per sequence (what
  mlx-lm 0.32.0's batch caches pass), give output bitwise equal to integer
  offsets.
* Hook refactor. The port with its hooks gives logits bitwise equal to the same
  port running exp_036's Attention.__call__ (no hooks; l.276-299, held below
  and checked against exp_036's file), on two tiny checkpoints, without a
  cache, through single-sequence caches (chunked prefill past the window, then
  decode) and through left-padded batch caches.
* Hook contract (what gate/port_mutants.py's mutants 19, 20, 22 and 26
  override): rope_offset(cache, L) runs on RoPE layers only, before the cache
  update, and returns cache.offset (0 without a cache); attn_mask(mask, cache,
  L) runs on every layer, after the update, and returns mask; an override of
  either reaches the forward.
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
import tiny_real_layout as trl
from exp036_helpers import import_sibling

EXP = Path(__file__).resolve().parent.parent
E36_PORT = EXP.parent / "exp_036_kolibri_local_eval" / "port" / "kolibri1.py"
FIXTURE = EXP / "tests" / "fixtures" / "rope_angle_262143.json"
FIXTURE_SCHEMA = "exp037 rope angle v1"
WRITE_ENV = "EXP037_WRITE_ROPE_FIXTURE"

HEAD_DIM = 128  # Kolibri-1
THETA = 10000.0
BOUND = 1e-6  # relative L2 per rotated vector (DESIGN §3.3 item 1)
DENSE_END = 20480  # every position 0..20,479
DENSE_CHUNK = 2048  # prefill_step_size
SPARSE_STEP = 97
LAST = 262143  # max_position_embeddings - 1
HEADS = 4
SEED = 37

# exp_036's Attention.__call__ (port/kolibri1.py l.276-299 at 222c845), verbatim:
# the pre-hook forward the refactor must reproduce bit for bit.
EXP036_ATTENTION_CALL = '''
def __call__(self, x: mx.array, mask: Any = None, cache: Optional[Any] = None) -> mx.array:
    B, L, _ = x.shape
    q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
    k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
    v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

    # Norm over head_dim on the [B, L, heads, D] view, then to [B, heads, L, D].
    q = self.q_norm(q).transpose(0, 2, 1, 3)
    k = self.k_norm(k).transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)

    if self.rope is not None:
        # Absolute positions: cache.offset counts every token seen, also
        # past the window (RotatingKVCache.offset is unbounded).
        offset = cache.offset if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

    if cache is not None:
        k, v = cache.update_and_fetch(k, v)

    out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(out)
'''


def _kolibri1():
    return import_sibling("port.kolibri1")


def _ref():
    return import_sibling("reference.kolibri_ref")


def _bits(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _np(a) -> np.ndarray:
    import mlx.core as mx

    return np.array(a.astype(mx.float32))


# ---------------------------------------------------------------- the angle


@pytest.mark.parametrize("dims", [HEAD_DIM, 32], ids=["d128", "tiny_d32"])
def test_freqs_are_the_exact_reciprocal_of_vllm_inv_freq(dims):
    import mlx.core as mx

    inv = _ref().rope_inv_freq(dims, THETA)
    assert inv.dtype == np.float32 and inv.shape == (dims // 2,)
    rope = _kolibri1().VllmRoPE(dims, THETA)
    freqs = np.array(rope._freqs)
    assert rope._freqs.dtype == mx.float32 and rope.dims == dims
    # _freqs is float32(1 / inv_freq) bitwise ...
    np.testing.assert_array_equal(_bits(freqs), _bits((np.float32(1.0) / inv).astype(np.float32)))
    # ... and its fp32 reciprocal is inv_freq bitwise, so pos * (1 / freqs) is vLLM's angle.
    np.testing.assert_array_equal(_bits((np.float32(1.0) / freqs).astype(np.float32)), _bits(inv))
    # Private: not a parameter, so strict loads of exp_036's shards are unaffected.
    assert dict(rope.parameters()) == {}


def test_sliding_layers_use_vllm_rope_and_full_layers_none(tiny_pattern5_dir):
    model = ph.load_port(tiny_pattern5_dir)
    for layer in model.layers:
        rope = layer.self_attn.rope
        if layer.use_sliding:
            assert type(rope).__name__ == "VllmRoPE" and rope.dims == model.args.head_dim
        else:
            assert rope is None


# ---------------------------------------------------------------- the rotation


def _octaves() -> list[tuple[int, int]]:
    """[0, 1), then [2^(k-1), 2^k) up to [131072, 262144)."""
    return [(0, 1)] + [(2 ** (k - 1), 2**k) for k in range(1, LAST.bit_length() + 1)]


def _sparse_positions() -> list[int]:
    pos = list(range(0, LAST + 1, SPARSE_STEP))
    if pos[-1] != LAST:
        pos.append(LAST)
    return pos


def _rel_l2(y: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per vector (last axis) ||y - r|| / ||r||, in float64."""
    y = y.astype(np.float64)
    r = r.astype(np.float64)
    return np.linalg.norm(y - r, axis=-1) / np.linalg.norm(r, axis=-1)


def _measure() -> dict:
    """The port's rotation against the reference: per position, the worst
    relative L2 over HEADS random vectors; then the maximum per octave."""
    import mlx.core as mx

    ref = _ref()
    rope = _kolibri1().VllmRoPE(HEAD_DIM, THETA)
    rng = np.random.default_rng(SEED)
    worst: dict[int, float] = {}

    def note(positions, errs):
        for p, e in zip(positions, errs):
            worst[int(p)] = max(worst.get(int(p), 0.0), float(e))

    # Dense: positions 0..20,479 as ten 2,048-position calls at integer offsets.
    x = rng.standard_normal((DENSE_END, HEADS, HEAD_DIM)).astype(np.float32)  # [T, heads, D]
    for start in range(0, DENSE_END, DENSE_CHUNK):
        xc = x[start : start + DENSE_CHUNK]
        pos = np.arange(start, start + len(xc))
        y = rope(mx.array(xc.transpose(1, 0, 2))[None], offset=int(start))  # [1, heads, T, D]
        y = np.array(y)[0].transpose(1, 0, 2)
        note(pos, _rel_l2(y, ref.rope_neox(xc, pos, THETA)).max(axis=1))

    # Sparse: every 97th position to 262,143, one position per call (decode-sized).
    sparse = _sparse_positions()
    xs = rng.standard_normal((len(sparse), HEADS, HEAD_DIM)).astype(np.float32)
    for i, p in enumerate(sparse):
        y = rope(mx.array(xs[i][:, None, :])[None], offset=int(p))  # [1, heads, 1, D]
        y = np.array(y)[0].transpose(1, 0, 2)
        note([p], _rel_l2(y, ref.rope_neox(xs[i][None], np.array([p]), THETA)).max(axis=1))

    octaves = []
    for lo, hi in _octaves():
        ps = sorted(p for p in worst if lo <= p < hi)
        errs = [worst[p] for p in ps]
        k = int(np.argmax(errs))
        octaves.append(
            {"lo": lo, "hi": hi, "n_positions": len(ps), "max_rel_l2": errs[k], "argmax_position": ps[k]}
        )
    try:
        arch = mx.device_info()["architecture"]
    except Exception:  # pragma: no cover - older mlx
        arch = mx.metal.device_info()["architecture"]
    return {
        "schema": FIXTURE_SCHEMA,
        "what": (
            "Per-octave maxima of the per-vector relative L2 between port/kolibri1.py VllmRoPE "
            "(mx.fast.rope with freqs = 1 / inv_freq) and reference/kolibri_ref.py rope_neox, "
            "written by tests/test_port_rope_vllm.py (DESIGN §3.3 G1 item 1)."
        ),
        "definition": {
            "head_dim": HEAD_DIM,
            "theta": THETA,
            "heads": HEADS,
            "seed": SEED,
            "dtype": "float32",
            "dense_positions": [0, DENSE_END - 1],
            "dense_chunk": DENSE_CHUNK,
            "sparse_step": SPARSE_STEP,
            "last_position": LAST,
            "bound_rel_l2": BOUND,
        },
        "runtime": {
            "mlx": mx.__version__,
            "numpy": np.__version__,
            "gpu_architecture": arch,
            "device": str(mx.default_device()),
        },
        "n_positions": len(worst),
        "max_rel_l2": max(worst.values()),
        "octaves": octaves,
    }


@pytest.fixture(scope="module")
def rope_measurement() -> dict:
    return _measure()


def test_rotation_matches_rope_neox_to_262143(rope_measurement, record_testsuite_property):
    m = rope_measurement
    for o in m["octaves"]:  # this host's maxima, into the junit record (G1's on the run host)
        record_testsuite_property(f"rope_max_rel_l2_{o['lo']}_{o['hi']}", repr(o["max_rel_l2"]))
    assert m["n_positions"] == len(set(range(DENSE_END)) | set(_sparse_positions()))
    assert all(o["n_positions"] > 0 for o in m["octaves"])
    over = [o for o in m["octaves"] if not o["max_rel_l2"] <= BOUND]
    assert not over, f"VllmRoPE vs rope_neox above {BOUND} relative L2: {over}"


def test_committed_rope_output(rope_measurement):
    m = rope_measurement
    if os.environ.get(WRITE_ENV) == "1":
        FIXTURE.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
    assert FIXTURE.is_file(), f"{FIXTURE.name} missing; write it with {WRITE_ENV}=1 on the build host"
    rec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # The committed output belongs to this test's definition ...
    assert rec["schema"] == FIXTURE_SCHEMA
    assert rec["definition"] == m["definition"]
    assert rec["n_positions"] == m["n_positions"]
    shape = lambda r: [(o["lo"], o["hi"], o["n_positions"]) for o in r["octaves"]]  # noqa: E731
    assert shape(rec) == shape(m)
    # ... is within the bound ...
    assert all(o["max_rel_l2"] <= BOUND for o in rec["octaves"]) and rec["max_rel_l2"] <= BOUND
    assert rec["max_rel_l2"] == max(o["max_rel_l2"] for o in rec["octaves"])
    # ... and, on the runtime that wrote it, is this run's output.
    if rec["runtime"] == m["runtime"]:
        assert rec["octaves"] == m["octaves"] and rec["max_rel_l2"] == m["max_rel_l2"]


# ---------------------------------------------------------------- array offsets


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_array_offsets_equal_integer_offsets(dtype):
    import mlx.core as mx

    rope = _kolibri1().VllmRoPE(HEAD_DIM, THETA)
    rng = np.random.default_rng(SEED + 1)
    offs = [0, 513, 20479, LAST + 1 - 5]
    x = mx.array(rng.standard_normal((len(offs), HEADS, 5, HEAD_DIM)).astype(np.float32)).astype(
        getattr(mx, dtype)
    )
    for L in (1, 5):
        xs = x[:, :, :L, :]
        singles = [rope(xs[b : b + 1], offset=o) for b, o in enumerate(offs)]
        # A scalar array offset.
        for b, o in enumerate(offs):
            np.testing.assert_array_equal(_np(rope(xs[b : b + 1], offset=mx.array(o))), _np(singles[b]))
        # One offset per sequence, as BatchKVCache / BatchRotatingKVCache carry it.
        batched = rope(xs, offset=mx.array(offs))
        for b in range(len(offs)):
            np.testing.assert_array_equal(_np(batched[b : b + 1]), _np(singles[b]))


# ---------------------------------------------------------------- the hook refactor


def _attention_call_source(path: Path) -> str:
    src = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == "Attention":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__call__":
                    lines = src.splitlines()[fn.lineno - 1 : fn.end_lineno]
                    return textwrap.dedent("\n".join(lines)).strip()
    raise AssertionError(f"no Attention.__call__ in {path}")


def test_pre_hook_call_is_exp036_verbatim():
    assert EXP036_ATTENTION_CALL.strip() == _attention_call_source(E36_PORT)
    # The port's own __call__ differs from it only by the two hook calls.
    own = _attention_call_source(EXP / "port" / "kolibri1.py")
    assert "self.rope_offset(cache, L)" in own and "self.attn_mask(mask, cache, L)" in own
    assert "cache.offset" not in own


def _load_pre_hook(model_dir):
    """The port from model_dir with exp_036's Attention.__call__ (no hooks).
    mlx_lm executes kolibri1.py afresh per load, so the patch reaches this
    model's classes only."""
    model = ph.load_port(model_dir)
    cls = type(model.layers[0].self_attn)
    ns = dict(ph.port_namespace(model))
    exec(compile(EXP036_ATTENTION_CALL, "<exp036 Attention.__call__>", "exec"), ns)
    cls.__call__ = ns["__call__"]
    return model


def _batch_caches(model, left_padding):
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache, RotatingKVCache

    caches = []
    for c in model.make_cache():
        if isinstance(c, RotatingKVCache):
            caches.append(BatchRotatingKVCache(c.max_size, list(left_padding)))
        else:
            caches.append(BatchKVCache(list(left_padding)))
    return caches


def _scenarios(model, window: int) -> list[np.ndarray]:
    """Logits of one fixed workload: no cache; single-sequence caches (prefill
    in chunks past the window, then decode); left-padded batch caches."""
    import mlx.core as mx

    out = []
    n = 2 * window + 41
    ids = tc.random_ids(n, seed=71)
    out.append(_np(model(mx.array([ids]))))  # no cache, T past the window

    cache = model.make_cache()
    for a, b in ((0, 37), (37, 37 + window), (37 + window, n)):
        out.append(_np(model(mx.array([ids[a:b]]), cache=cache)))
    for t in tc.random_ids(6, seed=72):
        out.append(_np(model(mx.array([[t]]), cache=cache)))

    short = tc.random_ids(window // 2 + 3, seed=73)
    lp = [0, n - len(short)]
    caches = _batch_caches(model, lp)
    batch = mx.array([ids, [0] * lp[1] + short])
    out.append(_np(model(batch, cache=caches)))
    for t in np.array(tc.random_ids(8, seed=74)).reshape(4, 2):
        out.append(_np(model(mx.array(t[:, None]), cache=caches)))
    return out


@pytest.fixture(scope="module")
def tiny_real_layout_dir(tmp_path_factory) -> Path:
    """The tiny real-layout checkpoint (head_dim 128, window 513, 10 layers) of
    the validation seed, in the loadable form (model_file and a port copy)."""
    return trl.write_real_layout(tmp_path_factory.mktemp("rope_real_layout") / "ck", trl.SEED_VAL)


@pytest.mark.parametrize("which", ["pattern5", "real_layout"])
def test_hook_refactor_changes_no_output(which, tiny_pattern5_dir, tiny_real_layout_dir):
    d = tiny_pattern5_dir if which == "pattern5" else tiny_real_layout_dir
    hooked = ph.load_port(d)
    pre = _load_pre_hook(d)
    assert type(hooked.layers[0].self_attn).__call__ is not type(pre.layers[0].self_attn).__call__
    window = hooked.args.sliding_window
    a = _scenarios(hooked, window)
    b = _scenarios(pre, window)
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and np.isfinite(x).all(), i
        np.testing.assert_array_equal(x, y, err_msg=f"workload step {i}")


# ---------------------------------------------------------------- the hook contract


def test_hook_contract(tiny_pattern5_dir):
    import mlx.core as mx

    model = ph.load_port(tiny_pattern5_dir)
    cls = type(model.layers[0].self_attn)
    orig_ro, orig_am = cls.rope_offset, cls.attn_mask
    calls = []
    order = {id(layer.self_attn): i for i, layer in enumerate(model.layers)}

    def ro(self, cache, L):
        off = orig_ro(self, cache, L)
        calls.append(("rope_offset", order[id(self)], L, None if cache is None else int(cache.offset), off))
        return off

    def am(self, mask, cache, L):
        got = orig_am(self, mask, cache, L)
        assert got is mask
        calls.append(("attn_mask", order[id(self)], L, None if cache is None else int(cache.offset), None))
        return got

    cls.rope_offset, cls.attn_mask = ro, am
    sliding = [i for i, layer in enumerate(model.layers) if layer.use_sliding]
    n_layers = len(model.layers)

    def expected(L, before):
        e = []
        for i in range(n_layers):
            if i in sliding:
                e.append(("rope_offset", i, L, before, 0 if before is None else before))
            e.append(("attn_mask", i, L, None if before is None else before + L, None))
        return e

    ids = tc.random_ids(30, seed=75)
    model(mx.array([ids]))  # no cache
    assert calls == expected(30, None)
    calls.clear()
    cache = model.make_cache()
    model(mx.array([ids[:20]]), cache=cache)
    assert calls == expected(20, 0)
    calls.clear()
    model(mx.array([ids[20:21]]), cache=cache)
    assert calls == expected(1, 20)
    cls.rope_offset, cls.attn_mask = orig_ro, orig_am


def test_hook_overrides_reach_the_forward(tiny_pattern5_dir):
    import mlx.core as mx

    ids = tc.random_ids(40, seed=76)

    def run(model):
        cache = model.make_cache()
        model(mx.array([ids[:39]]), cache=cache)
        return _np(model(mx.array([ids[39:]]), cache=cache))

    base = run(ph.load_port(tiny_pattern5_dir))

    # rope_offset: position 0 for every call (the decode step's RoPE position is wrong).
    m = ph.load_port(tiny_pattern5_dir)
    type(m.layers[0].self_attn).rope_offset = lambda self, cache, L: 0
    assert np.abs(run(m) - base).max() > 1e-3

    # attn_mask: the decode query sees only its own key.
    m = ph.load_port(tiny_pattern5_dir)

    def own_key_only(self, mask, cache, L):
        if L != 1:
            return mask
        keys = min(int(cache.offset), getattr(cache, "max_size", None) or int(cache.offset))
        return mx.array([[[[j == keys - 1 for j in range(keys)]]]])

    type(m.layers[0].self_attn).attn_mask = own_key_only
    assert np.abs(run(m) - base).max() > 1e-3
