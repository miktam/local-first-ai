"""Structural invariants of the numpy reference, on tiny random checkpoints.

These do not compare the reference against anything else. They check
properties that any correct Kolibri 1 forward pass must have, so that the
reference can be trusted as the oracle for the port:

  causality          logits at t never depend on tokens after t
  sliding window     a sliding layer sees exactly `sliding_window` keys,
                     itself included: W-1 positions back is inside, W is
                     outside (spec items 9-10)
  full layers        no window, and no positional encoding (NoPE): with a
                     causal mask and no position signal, the output at the
                     last position is invariant to permuting the earlier
                     tokens (spec item 9); a RoPE sliding layer is not
  RMSNorm            y = w * x / sqrt(mean(x^2) + eps), NOT (1 + w) * ... (item 6)
  embedding          plain row lookup, no scaling (item 4)

The reference is float32 throughout. "Unchanged" means a max abs difference
<= 1e-4 (observed summation-order noise <= 3e-6); "changed" means > 1e-3
(observed >= 0.02).
"""

from __future__ import annotations

import numpy as np
import pytest

import tiny_checkpoint as tc
from exp036_helpers import as_logits, import_sibling

# Differences below this count as "unchanged": fp32 summation-order noise on
# these checkpoints is <= 3e-6 (logits up to ~3.5); "changed" cases are >= 0.02.
SAME_ATOL = 1e-4
# Differences above this count as "changed".
CHANGED_MIN = 1e-3

PRESETS = ["vendor", "pattern5"]


def _reference(model_dir):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    return kolibri_ref.KolibriReference(str(model_dir))


def _logits(ref, ids) -> np.ndarray:
    return as_logits(ref.forward(np.asarray(ids, dtype=np.int64)))


def _layers(ref, ids) -> list[np.ndarray]:
    """Residual stream after each decoder layer, each [T, H]."""
    out = ref.forward(np.asarray(ids, dtype=np.int64), return_hidden=True)
    assert isinstance(out, tuple) and len(out) >= 2, "forward(return_hidden=True) must return (logits, layers, ...)"
    layers = [np.asarray(h, dtype=np.float32) for h in out[1]]
    return layers


def _maxdiff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def _swap_token(ids: list[int], pos: int) -> list[int]:
    """Same sequence with a different token at `pos`."""
    out = list(ids)
    out[pos] = (out[pos] + 517) % (tc.VOCAB_SIZE - tc.RESERVED_TOP_IDS)
    assert out[pos] != ids[pos]
    return out


@pytest.fixture(scope="module", params=PRESETS)
def preset_ref(request, tiny_vendor_dir, tiny_pattern5_dir):
    model_dir = tiny_vendor_dir if request.param == "vendor" else tiny_pattern5_dir
    cfg = tc.preset_config(request.param)
    return request.param, cfg, _reference(model_dir)


@pytest.fixture(scope="module")
def one_layer_dir(tmp_path_factory):
    """Factory: a 1-layer checkpoint of a preset with the given layer type."""
    made = {}

    def make(preset: str, layer_type: str, edit=None, tag: str = ""):
        key = (preset, layer_type, tag)
        if key not in made:
            out = tmp_path_factory.mktemp(f"one_{preset}_{layer_type}{tag}")
            made[key] = tc.write_tiny_checkpoint(
                out,
                preset=preset,
                copy_port=False,
                overrides={"num_hidden_layers": 1, "layer_types": [layer_type]},
                edit_tensors=edit,
            )
        return made[key]

    return make


# --- causality ---------------------------------------------------------------


def test_causality_perturbation(preset_ref):
    """Changing token p leaves logits[:p] alone and changes logits[p]."""
    name, cfg, ref = preset_ref
    W = cfg["sliding_window"]
    T = 2 * W + 10  # longer than the window, so full and sliding layers both matter
    ids = tc.random_ids(T, seed=11)
    base = _logits(ref, ids)
    assert base.shape == (T, cfg["vocab_size"])
    assert base.dtype == np.float32
    for p in (W // 2, W + 3, T - 1):
        other = _logits(ref, _swap_token(ids, p))
        assert _maxdiff(other[:p], base[:p]) <= SAME_ATOL, f"{name}: future token {p} leaked into earlier logits"
        assert _maxdiff(other[p], base[p]) > CHANGED_MIN, f"{name}: changing token {p} did not change logits[{p}]"


def test_causality_truncation(preset_ref):
    """forward(ids[:t]) equals the first t rows of forward(ids)."""
    name, cfg, ref = preset_ref
    W = cfg["sliding_window"]
    T = 2 * W + 10
    ids = tc.random_ids(T, seed=12)
    full = _logits(ref, ids)
    for t in (1, W - 1, W, W + 1, T - 1):
        part = _logits(ref, ids[:t])
        assert part.shape == (t, cfg["vocab_size"])
        assert _maxdiff(part, full[:t]) <= SAME_ATOL, f"{name}: prefix of length {t} disagrees with the full run"


# --- sliding window ----------------------------------------------------------


def test_sliding_window_boundary_layer0(preset_ref):
    """Layer 0 is a sliding layer in both presets. Its output at position i
    depends on the token W-1 positions back and not on the one W back."""
    name, cfg, ref = preset_ref
    assert cfg["layer_types"][0] == "sliding_attention"
    W = cfg["sliding_window"]
    T = W + 8
    i = T - 1
    ids = tc.random_ids(T, seed=13)
    base = _layers(ref, ids)[0]

    # W positions back: outside the window of i, inside the window of i - 1.
    out = _layers(ref, _swap_token(ids, i - W))[0]
    assert _maxdiff(out[i], base[i]) <= SAME_ATOL, f"{name}: query {i} sees the key {W} positions back"
    assert _maxdiff(out[i - 1], base[i - 1]) > CHANGED_MIN, f"{name}: query {i - 1} misses the key {W - 1} positions back"

    # W-1 positions back: the oldest key still inside the window of i.
    inside = _layers(ref, _swap_token(ids, i - (W - 1)))[0]
    assert _maxdiff(inside[i], base[i]) > CHANGED_MIN, f"{name}: query {i} misses the key {W - 1} positions back"


def test_sliding_window_stacks(preset_ref):
    """Two stacked sliding layers reach back 2(W-1) positions, not further."""
    name, cfg, ref = preset_ref
    assert cfg["layer_types"][:2] == ["sliding_attention", "sliding_attention"]
    W = cfg["sliding_window"]
    reach = 2 * (W - 1)
    T = reach + 6
    i = T - 1
    ids = tc.random_ids(T, seed=14)
    base = _layers(ref, ids)[1]
    out = _layers(ref, _swap_token(ids, i - reach - 1))[1]
    assert _maxdiff(out[i], base[i]) <= SAME_ATOL, f"{name}: two sliding layers reach past {reach}"
    inside = _layers(ref, _swap_token(ids, i - reach))[1]
    assert _maxdiff(inside[i], base[i]) > CHANGED_MIN, f"{name}: two sliding layers do not reach {reach}"


@pytest.mark.parametrize("preset", PRESETS)
def test_full_layer_has_no_window(one_layer_dir, preset):
    """A full-attention layer sees position 0 from far beyond the window."""
    W = tc.preset_config(preset)["sliding_window"]
    ref = _reference(one_layer_dir(preset, "full_attention"))
    T = W + 8
    ids = tc.random_ids(T, seed=15)
    base = _logits(ref, ids)
    out = _logits(ref, _swap_token(ids, 0))
    assert _maxdiff(out[T - 1], base[T - 1]) > CHANGED_MIN, "full layer does not see token 0"
    # Control: the same edit is invisible through a single sliding layer.
    sref = _reference(one_layer_dir(preset, "sliding_attention"))
    sbase = _logits(sref, ids)
    sout = _logits(sref, _swap_token(ids, 0))
    assert _maxdiff(sout[T - 1], sbase[T - 1]) <= SAME_ATOL


# --- positional encoding -----------------------------------------------------


@pytest.mark.parametrize("preset", PRESETS)
def test_full_layer_is_nope(one_layer_dir, preset):
    """Full layer, causal mask, no positional encoding: the last position's
    output depends on the earlier tokens only as a set, so permuting them
    leaves it unchanged. The same permutation through a RoPE sliding layer
    (sequence shorter than the window, so the mask plays no part) changes it.
    """
    W = tc.preset_config(preset)["sliding_window"]
    T = min(24, W - 2)
    ids = tc.random_ids(T, seed=16)
    perm = np.random.default_rng(16).permutation(T - 1)
    assert not np.array_equal(perm, np.arange(T - 1))
    shuffled = [ids[j] for j in perm] + [ids[-1]]

    full = _reference(one_layer_dir(preset, "full_attention"))
    d_full = _maxdiff(_logits(full, shuffled)[-1], _logits(full, ids)[-1])
    assert d_full <= SAME_ATOL, f"full layer is position-dependent (max diff {d_full:.3g}): RoPE applied on a NoPE layer?"

    sliding = _reference(one_layer_dir(preset, "sliding_attention"))
    d_slide = _maxdiff(_logits(sliding, shuffled)[-1], _logits(sliding, ids)[-1])
    assert d_slide > CHANGED_MIN, f"sliding layer is permutation-invariant (max diff {d_slide:.3g}): RoPE missing?"


# --- RMSNorm and embedding ---------------------------------------------------


def test_rms_norm_function_is_w_times_x():
    kolibri_ref = import_sibling("reference.kolibri_ref")
    if not hasattr(kolibri_ref, "rms_norm"):
        pytest.skip("reference exposes no module-level rms_norm(x, w, eps)")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 64)).astype(np.float32) * 3
    w = (1.0 + 0.1 * rng.standard_normal(64)).astype(np.float32)
    eps = 1e-6
    want = w * x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    got = np.asarray(kolibri_ref.rms_norm(x, w, eps))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    gemma = (1.0 + w) * x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    assert _maxdiff(got, gemma) > 0.5


@pytest.mark.parametrize("preset", PRESETS)
def test_final_norm_scales_logits_linearly(tmp_path, preset):
    """Doubling model.norm.weight doubles every logit exactly under w * x.

    Under (1 + w) * x the logits would change by (1 + 2w) / (1 + w) instead.
    Doubling is exact in bf16 and in fp32, so the check is tight.
    """
    ids = tc.random_ids(20, seed=17)
    a = tc.write_tiny_checkpoint(tmp_path / "a", preset=preset, copy_port=False)

    def double_final_norm(t):
        t["model.norm.weight"] = t["model.norm.weight"] * 2.0

    b = tc.write_tiny_checkpoint(tmp_path / "b", preset=preset, copy_port=False, edit_tensors=double_final_norm)
    la = _logits(_reference(a), ids)
    lb = _logits(_reference(b), ids)
    np.testing.assert_allclose(lb, 2.0 * la, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("preset", PRESETS)
def test_zero_sandwich_norms_remove_the_branches(tmp_path, preset):
    """With every post_attn_norm and post_ffn_norm weight at zero, a w * x norm
    zeroes both branch outputs, so the residual stream stays at the embedding
    and logits = model.norm(embed[ids]) @ lm_head^T, computed here from the
    checkpoint read back through mlx. With (1 + w) the branches would survive.
    """
    cfg = tc.preset_config(preset)

    def zero_sandwich(t):
        for i in range(cfg["num_hidden_layers"]):
            for n in ("post_attn_norm", "post_ffn_norm"):
                key = f"model.layers.{i}.{n}.weight"
                t[key] = np.zeros_like(t[key])

    d = tc.write_tiny_checkpoint(tmp_path / "z", preset=preset, copy_port=False, edit_tensors=zero_sandwich)
    w = tc.load_checkpoint_f32(d)
    ids = tc.random_ids(30, seed=18)
    e = w["model.embed_tokens.weight"][ids].astype(np.float64)
    hn = w["model.norm.weight"] * e / np.sqrt(np.mean(e * e, axis=-1, keepdims=True) + cfg["rms_norm_eps"])
    want = hn @ w["lm_head.weight"].astype(np.float64).T
    got = _logits(_reference(d), ids)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_embedding_is_plain_lookup(tiny_vendor_dir):
    """ref.embed(ids) is embed_tokens[ids], no sqrt(hidden) or other scaling."""
    ref = _reference(tiny_vendor_dir)
    if not hasattr(ref, "embed"):
        pytest.skip("reference exposes no embed(ids)")
    w = tc.load_checkpoint_f32(tiny_vendor_dir)
    ids = np.asarray(tc.random_ids(10, seed=19), dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(ref.embed(ids), dtype=np.float32), w["model.embed_tokens.weight"][ids])
