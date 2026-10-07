# SPDX-License-Identifier: MIT
"""Helpers for the tests that run the MLX port (port/kolibri1.py) against the
numpy reference or against itself.

The port is always loaded the way the experiment loads it: mlx_lm.utils.load_model
on a checkpoint directory whose config.json names "model_file": "kolibri1.py".
mlx_lm executes that file afresh on every load and does not register it in
sys.modules, so each loaded model has its own module namespace; patches made
through port_namespace() or on type(model.layers[0]) affect only that model.

Precision vocabulary used by the tests (see INTEGRATION_LOG.md, entries 1-3):

* "natural routing": the port selects its own experts. In bf16 or after
  quantisation, near-tied router scores flip, and each flip changes that
  token's hidden state by tens of percent. On tiny random weights this
  cascade, not the arithmetic, dominates the logit error.
* "forced routing": the port's `route` is replaced by one that takes the
  expert ids from a recorded run and computes the weights exactly as spec
  item 12 says (sigmoid of the port's own fp32 router logits at those ids, no
  renormalisation). What remains is the reduced-precision arithmetic itself.
* "decisive positions": positions where the comparison run's top-1 logit
  leads the runner-up by at least DECISIVE_GAP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# bf16 has 8 significant bits: unit roundoff u = 2**-8.
BF16_U = 2.0**-8

# Top-1 margin (in logits) above which a reduced-precision run is expected to
# keep the reference's argmax. With forced routing the measured median
# |delta logit| is 0.005-0.03 and the largest 0.08 (logit std 0.8), so a 0.1
# lead is 3-20x the typical error. Positions with smaller leads are coin flips.
DECISIVE_GAP = 0.1


def load_port(model_dir, float32: bool = False):
    """Load the port from a checkpoint dir via mlx_lm.utils.load_model.

    float32=True casts every floating parameter to fp32 (expert_bias and
    lm_head are fp32 already after sanitize).

    mlx-lm 0.32.0 runs a model_file only with trust_remote_code=True (#1385,
    CVE-2026-5843; exp_037 decision F2). It is passed directly here: every
    directory these tests load was written by the kit's own writers
    (tiny_checkpoint.py, port/convert.py), with a copy of port/kolibri1.py."""
    import mlx.core as mx
    from mlx_lm.utils import load_model

    model, _ = load_model(Path(model_dir), trust_remote_code=True)
    if float32:
        model.set_dtype(mx.float32)
    return model


def port_namespace(model) -> dict:
    """Module globals of the kolibri1.py this model was built from."""
    return type(model.layers[0].mlp).__call__.__globals__


def port_logits(model, ids, cache=None) -> np.ndarray:
    """Logits [T, V] float32 for one sequence (batch of 1).

    With cache=None and T above the sliding window, the port builds a dense
    [T, T] window mask (memory ~T^2; see Kolibri1Model.__call__). Fine for the
    tiny test sequences; real-length texts go through a cache in chunks."""
    import mlx.core as mx

    out = model(mx.array(np.asarray(ids, dtype=np.int32))[None], cache=cache)
    return np.array(out[0].astype(mx.float32))


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def kl_per_position(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """KL(p || q) per position, in nats, computed in float64."""
    lp, lq = log_softmax(p_logits), log_softmax(q_logits)
    return (np.exp(lp) * (lp - lq)).sum(axis=-1)


def top1_gap(logits: np.ndarray) -> np.ndarray:
    """Lead of the top-1 logit over the runner-up, per position."""
    s = np.sort(np.asarray(logits, dtype=np.float64), axis=-1)
    return s[:, -1] - s[:, -2]


def compare(ref_logits: np.ndarray, got_logits: np.ndarray) -> dict:
    """The statistics every comparison reports."""
    agree = ref_logits.argmax(-1) == got_logits.argmax(-1)
    decisive = top1_gap(ref_logits) >= DECISIVE_GAP
    kl = kl_per_position(ref_logits, got_logits)
    return {
        "rel_max": float(np.abs(got_logits - ref_logits).max() / np.abs(ref_logits).max()),
        "top1": float(agree.mean()),
        "top1_decisive": float(agree[decisive].mean()),
        "n_decisive": int(decisive.sum()),
        "kl_mean": float(kl.mean()),
        "kl_max": float(kl.max()),
    }


def fmt(stats: dict) -> str:
    return (
        f"rel_max {stats['rel_max']:.2e}  top1 {stats['top1']:.3f}  "
        f"top1(decisive, n={stats['n_decisive']}) {stats['top1_decisive']:.3f}  "
        f"KL mean {stats['kl_mean']:.2e} max {stats['kl_max']:.2e}"
    )


class ForcedRouting:
    """Replacement for the port's `route` that uses recorded expert ids.

    ids_per_layer: one [T, k] integer array per decoder layer, for the whole
    sequence. Set `.start` to the position of the first token before each model
    call; every call consumes one layer, in layer order, and slices the ids of
    the tokens in that call. Weights follow spec item 12: sigmoid of the
    port's own (unbiased, fp32) router logits at the forced ids.
    """

    def __init__(self, ids_per_layer):
        import mlx.core as mx

        self.ids = [mx.array(np.asarray(i, dtype=np.uint32)) for i in ids_per_layer]
        self.layer = 0
        self.start = 0

    def route(self, logits_f32, expert_bias_f32, top_k, renormalize=False):
        import mlx.core as mx

        n = logits_f32.shape[-2]
        ids = self.ids[self.layer][self.start : self.start + n]
        assert ids.shape == (n, top_k), (ids.shape, n, top_k)
        self.layer = (self.layer + 1) % len(self.ids)
        ids = mx.broadcast_to(ids, logits_f32.shape[:-1] + (top_k,))
        weights = mx.sigmoid(mx.take_along_axis(logits_f32.astype(mx.float32), ids, axis=-1))
        if renormalize:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return weights, ids


def force_routing(monkeypatch, model, ids_per_layer) -> ForcedRouting:
    forced = ForcedRouting(ids_per_layer)
    monkeypatch.setitem(port_namespace(model), "route", forced.route)
    return forced


def record_routing(monkeypatch, model) -> list:
    """Wrap the port's `route`; returns a list that receives the selected ids
    of every call as a [T, k] int64 array sorted within each row."""
    ns = port_namespace(model)
    original = ns["route"]
    calls: list = []

    def recording(*args, **kwargs):
        weights, ids = original(*args, **kwargs)
        flat = np.array(ids).reshape(-1, ids.shape[-1]).astype(np.int64)
        calls.append(np.sort(flat, axis=-1))
        return weights, ids

    monkeypatch.setitem(ns, "route", recording)
    return calls


def record_layer_outputs(monkeypatch, model) -> list:
    """Wrap DecoderLayer.__call__; returns a list that receives each layer's
    output (the residual stream after the layer) as float32 [T, H], batch 0."""
    import mlx.core as mx

    cls = type(model.layers[0])
    original = cls.__call__
    outputs: list = []

    def recording(self, x, mask=None, cache=None):
        y = original(self, x, mask, cache)
        outputs.append(np.array(y[0].astype(mx.float32)))
        return y

    monkeypatch.setattr(cls, "__call__", recording)
    return outputs


def record_reference_scores(monkeypatch, kolibri_ref) -> list:
    """Wrap the reference's route_ref; returns a list that receives the biased
    selection scores (router logits + expert_bias) [T, E] of every layer."""
    original = kolibri_ref.route_ref
    scores: list = []

    def recording(logits_f32, bias_f32, k, renormalize=False):
        scores.append(np.asarray(logits_f32, np.float32) + np.asarray(bias_f32, np.float32)[None, :])
        return original(logits_f32, bias_f32, k, renormalize)

    monkeypatch.setattr(kolibri_ref, "route_ref", recording)
    return scores


def selection_margin(scores: np.ndarray, k: int) -> np.ndarray:
    """k-th minus (k+1)-th largest biased score per token: how far the
    selection is from a tie."""
    s = -np.sort(-np.asarray(scores, dtype=np.float64), axis=-1)
    return s[:, k - 1] - s[:, k]
