# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
# Modified by Miktam for Chronos exp_037, 2026-10-06 (mutants 19, 20, 22, 26 and 27 as attention-hook overrides; 27 from decision (b)).
"""The port mutants: exp_036's 15 of HYPOTHESIS Phase 0 ("Port mutants";
BUILD_SPEC §5.3) and exp_037's hook mutants 19, 20, 22, 26 and 27.

Each mutant is a patch applied to a *fresh* module executed from
port/kolibri1.py (gate.common.exec_port_module): subclasses of the port's
classes, or replacements of its module-level functions, installed in that
module's own globals. port/kolibri1.py itself is never edited, and a patched
module affects only the models built from it.

Patches act on what both DecoderLayer.__call__ and DecoderLayer.branches use
(module globals, child-module classes, attribute access), so the gate's
per-layer harness, end-to-end runs and generation all see the same mutant.

  1 sigmoid_bias_select   select on sigmoid(logits) + bias (the afmoe / llama.cpp trap)
  2 window_512            sliding window 512 instead of 513 (mask and cache)
  3 rope_on_full          RoPE also on the full-attention (NoPE) layers
  4 one_plus_w_norm       (1 + w) RMSNorm everywhere
  5 swap_sandwich_norms   post_attn_norm <-> post_attention_layernorm
  6 renorm_topk           routing weights renormalised over the top-k
  7 route_scale_2826      afmoe route_scale 2.826 on the routing weights
  8 mup_embed_scale       muP sqrt(hidden) embedding scaling
  9 attn_output_gate      a retained afmoe attention output gate (extra parameter)
 10 bf16_router_logits    router logits computed in bf16
 11 bf16_head_logits      lm_head logits computed in bf16
 12 rope_traditional      interleaved (GPT-J) RoPE instead of NeoX half-rotation
 13 qknorm_after_rope     q/k RMSNorm applied after RoPE
 14 biased_weights        routing weights sigmoid(logits + bias)
 15 swiglu_swapped        silu on `up` instead of `gate`

On real weights mutant 9 must fail G0 (strict load) and mutants 10 and 11 the
value-level bf16-exactness check; the others must fail a G2 fp32 check on at
least one of layers {0, 3, 4, 49} (TARGETS below).

exp_037's hook mutants (DESIGN §4.1). 19, 20, 26 and 27 are each the
required control of one leg (§4.2); 22 is a reported probe since decision
(b) (§4.1-§4.2: run in tiny mode only, its (d) ratio reported, no margin).
Each overrides only one of the port's two attention hooks,
Attention.rope_offset(cache, L) or Attention.attn_mask(mask, cache, L)
(DESIGN §6.1), never Attention.__call__, so a port fix to __call__ reaches
them too:

 19 rope_restart_per_chunk               rope_offset: 0 for every prefill chunk (L > 1)
                                         -> G4-F32 F2 on T9 or a T9 bucket
 20 decode_window_256                    attn_mask: a decode step (L == 1) of a sliding
                                         layer sees only its 256 most recent keys
                                         -> G5-D32 decode leg
 22 batched_rope_positions_in_padded_frame
                                         rope_offset on a batch cache: every row at the
                                         longest row's position (positions counted in
                                         the padded frame) -> reported on G5-BP-lean (d),
                                         K8 B = 8 (a probe, not required)
 26 batch_decode_pos_frozen              rope_offset on a batch cache in decode: the
                                         offset of the first decode step, frozen
                                         -> G5-R1 R-parity, K8
 27 batch_decode_pad_keys_visible        attn_mask on a batch cache in decode: all True,
                                         so a padded row attends to its own pad slots
                                         -> G5-BP-lean (d), K8 B = 8 (decision (b))

They are not real-weight G2 mutants: REAL_WEIGHT_MUTANTS is unchanged.

mlx-lm 0.32.0 batch caches (decision F2; cache.py l.880-1461). BatchKVCache
and BatchRotatingKVCache start `offset` at -left_padding (l.905, l.1109) and
rebind it, an mx.array [B], on every update (#1848), instead of mutating it.
mlx-lm reads `left_padding` without a reduction (#1824), and
BatchRotatingKVCache changes it on every trim, rotation and roll (l.1157,
l.1164, l.1208, l.1215, l.1261). Mutants 22, 26 and 27 therefore recognise a
batch cache by the presence of a `left_padding` attribute and never read its
value; mutant 26 stores a copy of the offset, so it holds under rebinding and
under in-place mutation alike. BatchGenerator prefills a newly admitted group
right-padded, every row starting at offset 0, and finalises to left padding
(generate.py l.1163-1209); it merges even one sequence's caches into batch
caches (_merge_caches, l.815-828), so the runner at B = 1 decodes through
batch caches, where mutant 26 acts.

At decode (L == 1) a batch cache's mask (BatchKVCache.make_mask, cache.py
l.977-980; BatchRotatingKVCache.make_mask, l.1304-1331) is False only at a
row's padded slots: one query is causal to every cached key, and a rotating
batch cache holds at most max_size keys, max_size being the window, so the
window term is all True. Mutant 27's all-True mask is therefore exactly
"batched rows ignore their own left padding at decode": the pad-token keys of
a right-padded prefill that finalize rolls to the left (l.951-958, l.1256-1263)
and the zero slots that extend adds when a group is merged mid-run (l.1001-1045,
l.1343-1390). Where no row is padded (B = 1, rows of equal length) the mask
is already all True and mutant 27 is bitwise a no-op.
"""

from __future__ import annotations

import dataclasses
import math
import types
from pathlib import Path
from typing import Callable

from gate.common import PORT_FILE, exec_port_module

MUTANT_IDS = {
    1: "sigmoid_bias_select",
    2: "window_512",
    3: "rope_on_full",
    4: "one_plus_w_norm",
    5: "swap_sandwich_norms",
    6: "renorm_topk",
    7: "route_scale_2826",
    8: "mup_embed_scale",
    9: "attn_output_gate",
    10: "bf16_router_logits",
    11: "bf16_head_logits",
    12: "rope_traditional",
    13: "qknorm_after_rope",
    14: "biased_weights",
    15: "swiglu_swapped",
    # exp_037 (DESIGN §4.1): the hook mutants, attention-hook overrides only
    # (22 a probe since decision (b); 27 its replacement as G5-BP-lean's control).
    19: "rope_restart_per_chunk",
    20: "decode_window_256",
    22: "batched_rope_positions_in_padded_frame",
    26: "batch_decode_pos_frozen",
    27: "batch_decode_pad_keys_visible",
}
NAME_TO_ID = {v: k for k, v in MUTANT_IDS.items()}
# exp_037's hook mutants (DESIGN §4.1); each overrides only the port's
# Attention.rope_offset or Attention.attn_mask hook (DESIGN §6.1).
HOOK_MUTANTS = tuple(MUTANT_IDS[i] for i in (19, 20, 22, 26, 27))

# The check that must catch each mutant (HYPOTHESIS Phase 0, "Port mutants").
TARGETS = {
    "sigmoid_bias_select": "g2_fp32_natural_selection",
    "window_512": "g2_fp32_forced_branch",
    "rope_on_full": "g2_fp32_forced_branch",
    "one_plus_w_norm": "g2_fp32_forced_branch",
    "swap_sandwich_norms": "g2_fp32_forced_branch",
    "renorm_topk": "g2_fp32_forced_branch",
    "route_scale_2826": "g2_fp32_forced_branch",
    "mup_embed_scale": "g2_embedding",
    "attn_output_gate": "g0_strict_load",
    "bf16_router_logits": "g0_router_bf16_exact",
    "bf16_head_logits": "g0_head_bf16_exact",
    "rope_traditional": "g2_fp32_forced_branch",
    "qknorm_after_rope": "g2_fp32_forced_branch",
    "biased_weights": "g2_fp32_forced_branch",
    "swiglu_swapped": "g2_fp32_forced_branch",
    # exp_037: the one leg each control is bound to (DESIGN §4.2; gate/controls.json).
    "rope_restart_per_chunk": "g4_f32_F2_T9",          # F2 fires on T9 or a T9 bucket
    "decode_window_256": "g5_d32_decode",              # a decode-leg rule fires in >= 1 range
    # A probe since decision (b): the leg it is reported on (controls.json "probes"), not a required catch.
    "batched_rope_positions_in_padded_frame": "g5_bp_lean_d_K8_B8",  # (d) ratio, tiny mode only
    "batch_decode_pos_frozen": "g5_r1_R_parity_K8",    # R-parity fires at K8
    "batch_decode_pad_keys_visible": "g5_bp_lean_d_K8_B8",  # (d), first wave or mid-run (decision (b))
}
# HYPOTHESIS G2 "Real-weight mutants 1-8 and 12-15".
REAL_WEIGHT_MUTANTS = tuple(MUTANT_IDS[i] for i in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15))

AFMOE_ROUTE_SCALE = 2.826


def _nn_proxy(module, **overrides):
    """A stand-in for the module's `nn` global: mlx.nn with some classes
    replaced. Only instances created after the patch use it."""
    import mlx.nn as nn

    proxy = types.SimpleNamespace(**{k: getattr(nn, k) for k in dir(nn) if not k.startswith("__")})
    for k, v in overrides.items():
        setattr(proxy, k, v)
    module.nn = proxy


# --- the patches (module -> None) -------------------------------------------


def _sigmoid_bias_select(m):
    import mlx.core as mx

    def route(logits_f32, expert_bias_f32, top_k, renormalize=False):
        logits = logits_f32.astype(mx.float32)
        choice = mx.sigmoid(logits) + expert_bias_f32.astype(mx.float32)
        ids = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.sigmoid(mx.take_along_axis(logits, ids, axis=-1))
        if renormalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return weights, ids

    m.route = route


def _window_512(m):
    Base = m.Model

    class Model(Base):
        def __init__(self, args):
            super().__init__(dataclasses.replace(args, sliding_window=args.sliding_window - 1))

    m.Model = Model


def _rope_on_full(m):
    Base = m.Attention

    class Attention(Base):
        def __init__(self, args, use_rope):
            super().__init__(args, use_rope=True)

    m.Attention = Attention


def _one_plus_w_norm(m):
    import mlx.core as mx
    import mlx.nn as nn

    class OnePlusRMSNorm(nn.RMSNorm):
        def __call__(self, x):
            return mx.fast.rms_norm(x, 1.0 + self["weight"], self.eps)

    _nn_proxy(m, RMSNorm=OnePlusRMSNorm)


def _swap_sandwich_norms(m):
    Base = m.DecoderLayer
    swap = {"post_attn_norm": "post_attention_layernorm", "post_attention_layernorm": "post_attn_norm"}

    class DecoderLayer(Base):
        # Child modules live in the Module dict and are reached through
        # __getattr__; the checkpoint names (and loading) are unchanged, only
        # the use is swapped, in __call__ and branches alike. The swap is
        # switched on after __init__, whose attribute assignments probe
        # hasattr() on these names.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.exp036_swap_norms = True  # a bool: kept in __dict__, not a parameter

        def __getattr__(self, key):
            if self.__dict__.get("exp036_swap_norms"):
                key = swap.get(key, key)
            return super().__getattr__(key)

    m.DecoderLayer = DecoderLayer


def _renorm_topk(m):
    route = m.route
    m.route = lambda logits_f32, expert_bias_f32, top_k, renormalize=False: route(
        logits_f32, expert_bias_f32, top_k, True)
    if hasattr(m, "forced_route"):
        forced = m.forced_route
        m.forced_route = lambda logits_f32, ids, renormalize=False: forced(logits_f32, ids, True)


def _route_scale_2826(m):
    route = m.route

    def scaled(*args, **kwargs):
        w, ids = route(*args, **kwargs)
        return w * AFMOE_ROUTE_SCALE, ids

    m.route = scaled
    if hasattr(m, "forced_route"):
        forced = m.forced_route

        def forced_scaled(*args, **kwargs):
            w, ids = forced(*args, **kwargs)
            return w * AFMOE_ROUTE_SCALE, ids

        m.forced_route = forced_scaled


def _mup_embed_scale(m):
    import mlx.nn as nn

    class MupEmbedding(nn.Embedding):
        def __call__(self, x):
            return super().__call__(x) * math.sqrt(self.weight.shape[-1])

    _nn_proxy(m, Embedding=MupEmbedding)


def _attn_output_gate(m):
    import mlx.core as mx
    import mlx.nn as nn

    Base = m.Attention

    class Attention(Base):
        def __init__(self, args, use_rope):
            super().__init__(args, use_rope)
            # afmoe's gate: an extra [n_heads * head_dim, hidden] projection,
            # absent from the Kolibri checkpoint (spec item 3).
            self.gate_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim, bias=False)

        def __call__(self, x, mask=None, cache=None):
            B, L, _ = x.shape
            q = self.q_norm(self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
            k = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            if self.rope is not None:
                offset = cache.offset if cache is not None else 0
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = m.scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(out * mx.sigmoid(self.gate_proj(x)))

    m.Attention = Attention


def _bf16_router_logits(m):
    import mlx.core as mx

    Base = m.Router

    class Router(Base):
        def __call__(self, x):
            return (x.astype(mx.bfloat16) @ self.weight.astype(mx.bfloat16).T).astype(mx.float32)

    m.Router = Router


def _bf16_head_logits(m):
    import mlx.core as mx

    Base = m.Model

    def head_bf16(self, h):
        x = h.astype(mx.bfloat16)
        if hasattr(self.lm_head, "scales"):
            return self.lm_head(x).astype(mx.float32)
        return (x @ self.lm_head.weight.astype(mx.bfloat16).T).astype(mx.float32)

    if hasattr(Base, "compute_logits"):
        class Model(Base):
            def compute_logits(self, h):
                return head_bf16(self, h)
    else:  # an older port without compute_logits
        class Model(Base):
            def __call__(self, inputs, cache=None, input_embeddings=None):
                return head_bf16(self, self.model(inputs, cache, input_embeddings))

    m.Model = Model


def _rope_traditional(m):
    import mlx.nn as nn

    Base = m.Attention

    class Attention(Base):
        def __init__(self, args, use_rope):
            super().__init__(args, use_rope)
            if self.rope is not None:
                self.rope = nn.RoPE(self.head_dim, traditional=True, base=args.rope_theta)

    m.Attention = Attention


def _qknorm_after_rope(m):
    Base = m.Attention

    class Attention(Base):
        def __call__(self, x, mask=None, cache=None):
            B, L, _ = x.shape
            q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
            if self.rope is not None:
                offset = cache.offset if cache is not None else 0
                q = self.rope(q, offset=offset)
                k = self.rope(k, offset=offset)
            q = self.q_norm(q)  # the norm is over the last axis (head_dim) in either layout
            k = self.k_norm(k)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = m.scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    m.Attention = Attention


def _biased_weights(m):
    import mlx.core as mx

    route = m.route

    def biased(logits_f32, expert_bias_f32, top_k, renormalize=False):
        _, ids = route(logits_f32, expert_bias_f32, top_k, renormalize)
        choice = logits_f32.astype(mx.float32) + expert_bias_f32.astype(mx.float32)
        weights = mx.sigmoid(mx.take_along_axis(choice, ids, axis=-1))
        if renormalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return weights, ids

    m.route = biased


def _swiglu_swapped(m):
    import mlx.nn as nn

    m.swiglu = lambda gate, x: nn.silu(x) * gate

    class SwappedSwiGLU(nn.Module):
        def __call__(self, x, gate):  # SwitchGLU calls activation(x_up, x_gate)
            return nn.silu(x) * gate

    Base = m.SwitchGLU

    class SwitchGLU(Base):
        def __init__(self, input_dims, hidden_dims, num_experts, activation=None, bias=False):
            super().__init__(input_dims, hidden_dims, num_experts, activation=SwappedSwiGLU(), bias=bias)

    m.SwitchGLU = SwitchGLU


# --- exp_037's control mutants: attention-hook overrides only (DESIGN §4.1) --


def _hooked_attention(m, hook: str):
    """The module's Attention class, which must define `hook` (DESIGN §6.1):
    on a port without the hooks a hook mutant would be a silent no-op."""
    Base = m.Attention
    if not callable(getattr(Base, hook, None)):
        raise AttributeError(f"port Attention has no {hook}() hook (DESIGN §6.1); "
                             "exp_037's control mutants override only the hooks")
    return Base


def _is_batch_cache(cache) -> bool:
    """mlx-lm 0.32.0's BatchKVCache / BatchRotatingKVCache (cache.py
    l.880-1461), recognised by the attribute alone: `left_padding`'s value is
    never read here (it changes on every trim, rotation and roll, l.1157,
    l.1164, l.1208, l.1215, l.1261)."""
    return cache is not None and hasattr(cache, "left_padding")


def _rope_restart_per_chunk(m):
    """19: RoPE positions restart at 0 in every prefill chunk (L > 1). The first
    chunk of a sequence starts at 0 anyway; chunks 2+ get positions that are
    wrong relative to the keys already cached. Decode (L == 1) keeps the
    port's offset."""
    Base = _hooked_attention(m, "rope_offset")

    class Attention(Base):
        def rope_offset(self, cache, L):
            if L > 1:
                return 0
            return super().rope_offset(cache, L)

    m.Attention = Attention


DECODE_WINDOW_KEPT = 256  # mutant 20: the keys a decode query still sees, its own included


def decode_window_keep(cache, n_keep: int = DECODE_WINDOW_KEPT):
    """Mutant 20's mask for a single-sequence RotatingKVCache after a decode
    step's update_and_fetch: a boolean [n] over the n key slots
    update_and_fetch returned, True for the n_keep most recent keys (age <
    n_keep; the query's own key has age 0). None when every slot is kept
    (n <= n_keep), so the step is then the port's, bitwise.

    Slot j's age (keep = 0): (cache._idx - 1 - j) mod max_size once the
    cache is full (offset > n, n = max_size: it rotates in place), else
    cache.offset - 1 - j (slots in temporal order). n mirrors
    RotatingKVCache.keys_and_values (mlx-lm 0.32.0 cache.py l.527; the
    decode update, _update_in_place, is l.482-520)."""
    import mlx.core as mx

    if cache.keep != 0:
        raise ValueError(f"decode_window_256: defined for keep = 0 only, got keep = {cache.keep}")
    buf = cache.keys.shape[2]
    n = cache.offset if cache.offset < buf else buf
    if n <= n_keep:
        return None
    j = mx.arange(n)
    if cache.offset > n:
        if n != cache.max_size:
            raise RuntimeError(f"decode_window_256: a full rotating cache returned {n} slots, max_size is "
                               f"{cache.max_size}; re-validate the mutant against this mlx-lm")
        # _idx >= 1 after the update and j <= max_size - 1, so the sum is >= 0.
        age = (cache._idx - 1 + cache.max_size - j) % cache.max_size
    else:
        age = cache.offset - 1 - j
    return age < n_keep


def _decode_window_256(m):
    """20: a decode step (L == 1) of a sliding (RoPE) layer attends to its 256
    most recent keys only, instead of the window's 513. Single-sequence
    RotatingKVCache only (the G5-D32 decode path); prefill, the full layers
    and batch caches keep the port's mask."""
    import mlx.core as mx

    Base = _hooked_attention(m, "attn_mask")

    class Attention(Base):
        def attn_mask(self, mask, cache, L):
            mask = super().attn_mask(mask, cache, L)
            if (L != 1 or self.rope is None or cache is None or _is_batch_cache(cache)
                    or not hasattr(cache, "max_size")):
                return mask
            keep = decode_window_keep(cache)
            if keep is None:
                return mask
            if mask is None or isinstance(mask, str):  # "causal" is no restriction for one query
                return keep
            if mask.dtype == mx.bool_:
                return mx.logical_and(mask, keep)
            return mx.where(keep, mask, mx.array(-mx.inf, dtype=mask.dtype))

    m.Attention = Attention


def _batched_rope_positions_in_padded_frame(m):
    """22: on a batch cache every row gets the longest row's position
    (positions counted in the padded frame, not per sequence):
    mx.broadcast_to(mx.max(cache.offset), cache.offset.shape). At B = 1 that
    equals cache.offset bitwise; single-sequence caches are untouched. Rows
    whose offsets differ (unequal prompts after BatchGenerator's
    finalisation to left padding, mid-run admissions) decode at shifted
    positions relative to their own cached keys."""
    import mlx.core as mx

    Base = _hooked_attention(m, "rope_offset")

    class Attention(Base):
        def rope_offset(self, cache, L):
            if _is_batch_cache(cache):
                return mx.broadcast_to(mx.max(cache.offset), cache.offset.shape)
            return super().rope_offset(cache, L)

    m.Attention = Attention


FROZEN_OFFSET_ATTR = "exp037_m26_frozen_offset"  # set on the batch cache object by mutant 26


def _batch_decode_pos_frozen(m):
    """26: on a batch cache, every decode step (L == 1) uses the offset of the
    sequence's first decode step. The value is stored on the cache object as
    a copy, mx.array(cache.offset), so it does not depend on how mlx-lm
    advances the offset (0.32.0 rebinds it, #1848), and it is re-frozen
    whenever cache.offset's shape changes (rows filtered out or admitted).
    The first decode step is the port's; later steps sit at that position.
    Prefill and single-sequence caches are untouched. The runner decodes
    through batch caches at every B, B = 1 included."""
    import mlx.core as mx

    Base = _hooked_attention(m, "rope_offset")

    class Attention(Base):
        def rope_offset(self, cache, L):
            offset = super().rope_offset(cache, L)
            if L != 1 or not _is_batch_cache(cache):
                return offset
            frozen = getattr(cache, FROZEN_OFFSET_ATTR, None)
            if frozen is None or frozen.shape != cache.offset.shape:
                frozen = mx.array(cache.offset)
                setattr(cache, FROZEN_OFFSET_ATTR, frozen)
            return frozen

    m.Attention = Attention


def _batch_decode_pad_keys_visible(m):
    """27 (decision (b)): at decode (L == 1) on a batch cache, when the mask is
    an array, return mx.ones(mask.shape, dtype=mx.bool_): every cached slot is
    visible to the decode query, on every layer (sliding and full). The only
    False entries of a batch cache's decode mask are a row's padded slots
    (module docstring), so a padded row attends to the pad-token keys and
    values of its right-padded prefill and to the zero slots of a mid-run
    merge. Reads only the mask's shape, never `left_padding`. Prefill (L > 1),
    single-sequence caches and a mask that is not an array keep the port's
    mask; where no row is padded the mask is already all True, so the step is
    the port's, bitwise. Only G5-BP-lean's batched path can see it."""
    import mlx.core as mx

    Base = _hooked_attention(m, "attn_mask")

    class Attention(Base):
        def attn_mask(self, mask, cache, L):
            mask = super().attn_mask(mask, cache, L)
            if L == 1 and _is_batch_cache(cache) and isinstance(mask, mx.array):
                return mx.ones(mask.shape, dtype=mx.bool_)
            return mask

    m.Attention = Attention


PATCHES: dict[str, Callable] = {
    "sigmoid_bias_select": _sigmoid_bias_select,
    "window_512": _window_512,
    "rope_on_full": _rope_on_full,
    "one_plus_w_norm": _one_plus_w_norm,
    "swap_sandwich_norms": _swap_sandwich_norms,
    "renorm_topk": _renorm_topk,
    "route_scale_2826": _route_scale_2826,
    "mup_embed_scale": _mup_embed_scale,
    "attn_output_gate": _attn_output_gate,
    "bf16_router_logits": _bf16_router_logits,
    "bf16_head_logits": _bf16_head_logits,
    "rope_traditional": _rope_traditional,
    "qknorm_after_rope": _qknorm_after_rope,
    "biased_weights": _biased_weights,
    "swiglu_swapped": _swiglu_swapped,
    "rope_restart_per_chunk": _rope_restart_per_chunk,
    "decode_window_256": _decode_window_256,
    "batched_rope_positions_in_padded_frame": _batched_rope_positions_in_padded_frame,
    "batch_decode_pos_frozen": _batch_decode_pos_frozen,
    "batch_decode_pad_keys_visible": _batch_decode_pad_keys_visible,
}
assert set(PATCHES) == set(MUTANT_IDS.values()) == set(TARGETS)


def mutant_module(name: str, port_file: Path = PORT_FILE):
    """A fresh port module with mutant `name` applied."""
    if name not in PATCHES:
        raise KeyError(f"unknown port mutant {name!r}; known: {sorted(PATCHES)}")
    module = exec_port_module(port_file)
    PATCHES[name](module)
    module.EXP036_PORT_MUTANT = name
    return module


def _builder(name: str):
    def build(args):
        return mutant_module(name).Model(args)

    build.__name__ = f"build_{name}"
    return build


# BUILD_SPEC §5.3: PORT_MUTANTS: dict[str, Callable[[ModelArgs], nn.Module]].
PORT_MUTANTS: dict[str, Callable] = {name: _builder(name) for name in MUTANT_IDS.values()}
