# SPDX-License-Identifier: MIT
"""Config handling the port and the reference must share with the vendor.

use_sliding_window (review 2026-10-03). The vendor's Kolibri1Config subclasses
transformers' Qwen3MoeConfig, which defaults use_sliding_window to False and
then sets sliding_window to None; the vendor attention raises on sliding
layers without a window. A config that omits the key, or sets it false, must
therefore be refused by both of ours too, not run with a window. The
released config.json sets it to true, so the shipped model is unaffected.
"""

from __future__ import annotations

import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling


def _config(use_sliding_window, layer_types=None) -> dict:
    cfg = tc.preset_config("vendor")
    if use_sliding_window is None:
        cfg.pop("use_sliding_window")
    else:
        cfg["use_sliding_window"] = use_sliding_window
    if layer_types is not None:
        cfg["layer_types"] = layer_types
        cfg["num_hidden_layers"] = len(layer_types)
    return cfg


def _port_args(cfg):
    return import_sibling("port.kolibri1").ModelArgs.from_dict(cfg)


def _ref_config(cfg):
    return import_sibling("reference.kolibri_ref").KolibriConfig.from_dict(cfg)


@pytest.mark.parametrize("parse", [_port_args, _ref_config], ids=["port", "reference"])
@pytest.mark.parametrize("value", [None, False], ids=["missing", "false"])
def test_sliding_layers_need_use_sliding_window_true(parse, value):
    with pytest.raises(ValueError, match="use_sliding_window"):
        parse(_config(value))


@pytest.mark.parametrize("parse", [_port_args, _ref_config], ids=["port", "reference"])
def test_sliding_layers_with_use_sliding_window_true_parse(parse):
    out = parse(_config(True))
    assert out.sliding_window == 65


@pytest.mark.parametrize("parse", [_port_args, _ref_config], ids=["port", "reference"])
def test_all_full_layers_need_no_sliding_flag(parse):
    parse(_config(None, layer_types=["full_attention", "full_attention"]))


def test_vendor_base_config_defaults_to_no_window():
    """The fact the two refusals rest on, checked against the installed
    transformers (the vendor plugin's own config base class)."""
    qwen = pytest.importorskip("transformers.models.qwen3_moe.configuration_qwen3_moe")
    assert qwen.Qwen3MoeConfig(sliding_window=513).sliding_window is None
    assert qwen.Qwen3MoeConfig(sliding_window=513, use_sliding_window=True).sliding_window == 513
