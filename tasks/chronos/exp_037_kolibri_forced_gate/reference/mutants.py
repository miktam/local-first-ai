# SPDX-License-Identifier: Apache-2.0
# Derived from aleph-alpha-inference (Copyright 2026 Aleph Alpha GmbH, commit 049a6a7), modified for MLX/numpy by Miktam for Chronos exp_036, 2026-10-03.
"""The seven G3 reference mutants (HYPOTHESIS "Checks", G3; BUILD_SPEC 5.2).

Each mutant is a subclass of KolibriReference that overrides exactly one
method, so the frozen reference file is never edited and every other part
of the forward pass (weights, attention, mask, MoE loop, head) is shared.
Each encodes a plausible misreading of the architecture, the same traps the
port mutants of gate/port_mutants.py cover:

    sigmoid_bias_select   select top-k on sigmoid(logits) + bias (afmoe,
                          DeepSeek-V3, llama.cpp) instead of logits + bias
    rope_on_full          RoPE on the full-attention layers too (they are NoPE)
    one_plus_w_norm       (1 + w) RMSNorm (Gemma) instead of w * x
    renorm_topk           routing weights renormalised over the top-k
    swap_sandwich_norms   post_attn_norm <-> post_attention_layernorm
    rope_traditional      interleaved (GPT-J) RoPE instead of NeoX half rotation
    qknorm_after_rope     q/k RMSNorm applied after RoPE instead of before

G3 runs them as parallel streams of one layer-streamed pass
(KolibriReference.forward_streams, or the CLI's --mutants) and asks whether
each makes T3 measurably less likely than the reference does: if the
reference's semantics were the misreading, the mutant would score better.

Imports only numpy and the reference.
"""

import numpy as np

try:  # package import (python -m reference.kolibri_ref, tests)
    from .kolibri_ref import KolibriReference, route_ref, rope_inv_freq, sigmoid
except ImportError:  # direct script execution from inside reference/
    from kolibri_ref import KolibriReference, route_ref, rope_inv_freq, sigmoid


def rope_interleaved(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """GPT-J / "traditional" RoPE: rotates the pairs (x[2j], x[2j+1]) by
    pos * inv_freq[j] (mlx nn.RoPE(traditional=True)). x: [T, heads, D]."""
    x = np.asarray(x, dtype=np.float32)
    inv_freq = rope_inv_freq(x.shape[-1], theta)
    ang = np.asarray(positions, dtype=np.float32).reshape(-1)[:, None] * inv_freq[None, :]
    cos = np.cos(ang)[:, None, :]
    sin = np.sin(ang)[:, None, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = odd * cos + even * sin
    return out


class SigmoidBiasSelect(KolibriReference):
    """Selection on sigmoid(logits) + bias; weights still sigmoid(logits[ids])."""

    def route(self, logits, bias, k):
        logits = np.asarray(logits, dtype=np.float32)
        score = sigmoid(logits) + np.asarray(bias, dtype=np.float32)[None, :]
        ids = np.argsort(-score, axis=-1, kind="stable")[:, :k].astype(np.int64)
        weights = sigmoid(np.take_along_axis(logits, ids, axis=-1))
        if self.cfg.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + np.float32(1e-20))
        return weights.astype(np.float32), ids


class RopeOnFull(KolibriReference):
    """RoPE in every layer, the full-attention (NoPE) layers included."""

    def uses_rope(self, i):
        return True


class OnePlusWNorm(KolibriReference):
    """(1 + w) * x / rms(x) in every RMSNorm (Gemma's GemmaRMSNorm)."""

    def norm(self, x, w):
        return super().norm(x, np.asarray(w, dtype=np.float32) + np.float32(1.0))


class RenormTopk(KolibriReference):
    """Routing weights renormalised to sum to 1 over the selected experts."""

    def route(self, logits, bias, k):
        return route_ref(logits, bias, k, renormalize=True)


class SwapSandwichNorms(KolibriReference):
    """post_attn_norm and post_attention_layernorm swapped (the naming trap)."""

    def layer_norm_names(self):
        names = super().layer_norm_names()
        names["post_attn"], names["pre_moe"] = names["pre_moe"], names["post_attn"]
        return names


class RopeTraditional(KolibriReference):
    """Interleaved (GPT-J) RoPE instead of the NeoX half rotation."""

    def rope(self, x, positions):
        return self._store(rope_interleaved(x, positions, self.cfg.rope_theta))


class QKNormAfterRope(KolibriReference):
    """q/k RMSNorm applied after RoPE."""

    def qk_norm_rope(self, i, q, k, positions):
        p = f"model.layers.{i}.self_attn"
        if self.uses_rope(i):
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        q = self.norm(q, self._w(f"{p}.q_norm.weight"))
        k = self.norm(k, self._w(f"{p}.k_norm.weight"))
        return q, k


MUTANTS = {
    "sigmoid_bias_select": SigmoidBiasSelect,
    "rope_on_full": RopeOnFull,
    "one_plus_w_norm": OnePlusWNorm,
    "renorm_topk": RenormTopk,
    "swap_sandwich_norms": SwapSandwichNorms,
    "rope_traditional": RopeTraditional,
    "qknorm_after_rope": QKNormAfterRope,
}
