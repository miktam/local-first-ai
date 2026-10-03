# SPDX-License-Identifier: Apache-2.0
# filter_np is a numpy transcription of vLLM v0.29.0
# vllm/v1/sample/ops/topk_topp_sampler.py::apply_top_k_top_p and apply_top_k_only
# (Copyright contributors to the vLLM project, Apache-2.0).
# Modified by Miktam for Chronos exp_036, 2026-10-03: rewritten from PyTorch to numpy
# (the reference for the tests) and re-expressed for MLX as a value threshold.
"""The vLLM-order sampler (BUILD_SPEC §4 item 26, HYPOTHESIS C5).

Order, for every model, on fp32 logits:

  1. temperature (all models use T = 1.0; T = 0 means greedy argmax);
  2. top-k: threshold = the k-th largest value; values below it are masked,
     so ties at the threshold are kept;
  3. top-p over the renormalised top-k distribution: probabilities of the kept
     values in ascending order, cumulative sum, mask where cum <= 1 - p, the
     largest value always kept;
  4. categorical sampling from the masked fp32 log-probs.

No min-p, no penalties. mlx_lm's own make_sampler applies top-p on the full
distribution before top-k (C5) and is never used.

Tie order. vLLM sorts with torch.sort; on ties at the top-p boundary the
order of equal values decides which of them is masked. Both implementations
here use a stable ascending order (among equal values the lower vocabulary
index ranks lower and is masked first). For real fp32 logits ties at that
boundary have probability ~0; the tests construct them on purpose.

The MLX implementation sorts the row once (values only), finds the top-k
threshold and the top-p cut-off value v* in sorted space, and builds the
mask on the unsorted row: keep x > v*, and of the values equal to v* keep all
but the first n (by index) that the cumulative rule drops. That is the same
mask vLLM's sort / mask / scatter produces, without a scatter.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

NEG_INF = float("-inf")


# --------------------------------------------------------------------- numpy


def _softmax_np(s: np.ndarray) -> np.ndarray:
    m = np.max(s, axis=-1, keepdims=True)
    e = np.exp((s - m).astype(np.float32), dtype=np.float32)
    return (e / np.sum(e, axis=-1, keepdims=True, dtype=np.float32)).astype(np.float32)


def filter_np(logits: np.ndarray, top_k: int | None, top_p: float | None) -> np.ndarray:
    """vLLM v0.29 apply_top_k_top_p, transcribed to numpy (fp32).

    logits: [V] or [B, V]. top_k None/0/>=V disables top-k; top_p None or
    >= 1 disables top-p. Returns the masked logits (-inf where masked)."""
    x = np.array(logits, dtype=np.float32, copy=True)
    squeeze = x.ndim == 1
    x = np.atleast_2d(x)
    V = x.shape[-1]
    k = int(top_k) if top_k and 0 < int(top_k) < V else None
    p = np.float32(top_p) if top_p is not None and float(top_p) < 1.0 else None

    if p is None:
        if k is None:
            out = x
        else:
            # apply_top_k_only: the k-th largest value per row; mask values below it.
            thr = np.sort(x, axis=-1)[:, V - k][:, None]
            out = np.where(x < thr, np.float32(NEG_INF), x)
        return out[0] if squeeze else out

    # logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    idx = np.argsort(x, axis=-1, kind="stable")
    s = np.take_along_axis(x, idx, axis=-1)
    if k is not None:
        # top_k_mask = logits_sort.gather(1, (V - k)); logits_sort < top_k_mask -> -inf
        thr = s[:, V - k][:, None]
        s = np.where(s < thr, np.float32(NEG_INF), s)
    # probs_sort = logits_sort.softmax(-1); probs_sum = cumsum(probs_sort)
    probs = _softmax_np(s)
    cum = np.cumsum(probs, axis=-1, dtype=np.float32)
    # top_p_mask = probs_sum <= 1 - p; top_p_mask[:, -1] = False
    mask = cum <= (np.float32(1.0) - p)
    mask[:, -1] = False
    s = np.where(mask, np.float32(NEG_INF), s)
    # logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    out = np.empty_like(s)
    np.put_along_axis(out, idx, s, axis=-1)
    return out[0] if squeeze else out


def top_p_cum_np(logits: np.ndarray, top_k: int | None) -> np.ndarray:
    """The ascending cumulative probabilities filter_np compares with 1 - p,
    scattered back to vocabulary order (for diagnosing boundary cases)."""
    x = np.atleast_2d(np.asarray(logits, dtype=np.float32))
    V = x.shape[-1]
    idx = np.argsort(x, axis=-1, kind="stable")
    s = np.take_along_axis(x, idx, axis=-1)
    if top_k and 0 < top_k < V:
        thr = s[:, V - top_k][:, None]
        s = np.where(s < thr, np.float32(NEG_INF), s)
    cum = np.cumsum(_softmax_np(s), axis=-1, dtype=np.float32)
    out = np.empty_like(cum)
    np.put_along_axis(out, idx, cum, axis=-1)
    return out


# ----------------------------------------------------------------------- mlx


def filter_top_k_top_p(logprobs_f32, top_k: int | None, top_p: float | None):
    """MLX: mask [..., V] fp32 log-probs (or logits) with top-k then top-p,
    vLLM order. Returns an fp32 array with -inf where masked."""
    import mlx.core as mx

    x = logprobs_f32.astype(mx.float32)
    V = x.shape[-1]
    use_k = top_k is not None and 0 < int(top_k) < V
    use_p = top_p is not None and float(top_p) < 1.0
    if not use_k and not use_p:
        return x

    neg_inf = mx.array(NEG_INF, dtype=mx.float32)
    s = mx.sort(x, axis=-1)  # ascending, values only
    if use_k:
        k = int(top_k)
        thr = s[..., V - k : V - k + 1]  # the k-th largest value
        x = mx.where(x < thr, neg_inf, x)
        if not use_p:
            return x
        s = mx.where(s < thr, neg_inf, s)

    # Renormalised distribution over the kept values, ascending cumulative sum.
    probs = mx.softmax(s, axis=-1, precise=True)
    cum = mx.cumsum(probs, axis=-1)
    one_minus_p = mx.array(1.0, dtype=mx.float32) - mx.array(float(top_p), dtype=mx.float32)
    not_last = mx.arange(V) < (V - 1)  # the largest value is always kept
    drop = mx.logical_and(cum <= one_minus_p, not_last)
    # cum is non-decreasing, so the dropped positions are a prefix of length n_drop.
    n_drop = mx.sum(drop.astype(mx.int32), axis=-1, keepdims=True)
    v_star = mx.take_along_axis(s, n_drop, axis=-1)  # smallest kept value
    first_pos = mx.sum((s < v_star).astype(mx.int32), axis=-1, keepdims=True)
    n_tie_drop = n_drop - first_pos  # how many values equal to v* are dropped
    eq = x == v_star
    tie_rank = mx.cumsum(eq.astype(mx.int32), axis=-1) - 1  # rank by index among ties
    keep = mx.logical_or(x > v_star, mx.logical_and(eq, tie_rank >= n_tie_drop))
    return mx.where(keep, x, neg_inf)


def make_vllm_sampler(temperature: float, top_p: float, top_k: int) -> Callable:
    """Batched sampler for BatchGenerator (item 26): logprobs [B, V] -> ids [B].

    BatchGenerator hands the sampler log-softmax-normalised values; top-k and
    top-p are invariant to the per-row shift, and dividing by T scales the
    shift too, so this equals vLLM's processing of the raw logits."""
    import mlx.core as mx

    T = float(temperature)
    if T < 0:
        raise ValueError("temperature must be >= 0")

    def sampler(logprobs):
        x = logprobs.astype(mx.float32)
        if T == 0.0:
            return mx.argmax(x, axis=-1)
        if T != 1.0:
            x = x / T
        x = filter_top_k_top_p(x, top_k, top_p)
        return mx.random.categorical(x, axis=-1)

    sampler.config = {"order": "vllm", "temperature": T, "top_p": float(top_p), "top_k": int(top_k)}
    return sampler


def fp32_logits(model):
    """Thin wrapper so every model hands BatchGenerator fp32 logits (item 26,
    C6): BatchGenerator's log-softmax and the sampler then run in fp32. It
    exposes `layers` and, when the wrapped model has one, `make_cache`."""
    import mlx.core as mx
    import mlx.nn as nn

    class FP32Logits(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            if hasattr(inner, "make_cache"):
                self.make_cache = inner.make_cache

        @property
        def layers(self):
            return self.inner.layers

        def __call__(self, inputs, cache=None, **kwargs):
            return self.inner(inputs, cache=cache, **kwargs).astype(mx.float32)

    if isinstance(model, FP32Logits) or getattr(model, "_exp036_fp32", False):
        return model
    w = FP32Logits(model)
    w._exp036_fp32 = True
    return w
