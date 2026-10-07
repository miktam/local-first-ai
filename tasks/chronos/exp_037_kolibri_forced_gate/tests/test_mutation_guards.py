# SPDX-License-Identifier: MIT
"""Guards for port deviations that the end-to-end comparisons cannot see.

Written after a mutation run against port/kolibri1.py (2026-10-03): 30 mutants
of the spec's semantic traps, each loaded through a tiny checkpoint with
$EXP036_PORT_FILE (see tiny_checkpoint.py). The existing suite killed 29.
The survivor is guarded here.

lm_head in bf16 (spec item 14). sanitize leaves lm_head.weight bf16 and the
head runs in the model dtype, so the logits are rounded to bf16. The fp32
tests cast the whole model to fp32, which hides it. In the bf16 tests the
rounding (about u/2 = 2e-3 of a logit) sits below the bf16 noise bounds,
so they cannot see it either. This is the same blind spot as the router
logits, guarded by test_router_logits_are_fp32_of_bf16_operands.
"""

from __future__ import annotations

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling


@pytest.mark.parametrize("preset_dir", ["tiny_vendor_dir", "tiny_pattern5_dir"])
def test_bf16_model_logits_are_fp32_of_bf16_operands(preset_dir, request):
    """Spec item 14 (head_dtype float32). The model is bf16 as loaded, so the
    final norm emits bf16 hidden states. The logits must equal
    h.f32 @ W.f32^T, where W is the bf16 lm_head weight in the checkpoint,
    read here with mlx's plain loader and not through the port.

    The oracle is a float64 product of the same operands. fp32 accumulation
    over hidden_size = 256 products is about 1e-7 relative, and the bound is
    1e-5. Measured: 3.0e-7 to 3.4e-7. Rounding the logits to bf16, or
    computing the head in bf16, is off by 2.2e-3 to 2.4e-3 relative
    (mutants M18a and M18b), so the bound separates the two by over 200x.

    Only behaviour is checked, not lm_head.weight.dtype. Keeping the weight
    bf16 and upcasting it at each step is an equivalent implementation."""
    import mlx.core as mx

    import_sibling("port.kolibri1")
    model_dir = request.getfixturevalue(preset_dir)
    model = ph.load_port(model_dir)  # bf16 as loaded
    ids = mx.array(np.asarray(tc.random_ids(64, seed=11), dtype=np.int32))[None]

    h = model.model(ids)  # final-norm output, the lm_head input
    assert h.dtype == mx.bfloat16, "the loaded model should run its trunk in bf16"
    logits = model(ids)
    assert logits.dtype == mx.float32

    w = tc.load_checkpoint_f32(model_dir)["lm_head.weight"].astype(np.float64)
    exact = np.array(h.astype(mx.float32))[0].astype(np.float64) @ w.T
    got = np.array(logits[0]).astype(np.float64)
    scale = np.abs(exact).max()
    err = np.abs(got - exact).max() / scale
    bf16_err = np.abs(
        np.array(mx.array(exact.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)) - exact
    ).max() / scale
    assert err <= 1e-5, f"logits off by {err:.2e} relative (bf16 rounding would be {bf16_err:.2e})"
    assert bf16_err > 1e-4  # the check above can see bf16 rounding
