# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""The 15 port mutants of HYPOTHESIS Phase 0 ("Port mutants"; BUILD_SPEC §5.3).

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
}
NAME_TO_ID = {v: k for k, v in MUTANT_IDS.items()}

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
}
assert set(PATCHES) == set(MUTANT_IDS.values())


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
