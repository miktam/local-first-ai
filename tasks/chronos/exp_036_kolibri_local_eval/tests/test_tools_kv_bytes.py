"""tools/kv_bytes.py (BUILD_SPEC §5.9; HYPOTHESIS rule 1, spec item 20)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import tiny_checkpoint as tc
from tools import kv_bytes as kvb


def _kolibri_like(n_layers=50, kv=4, hd=128, window=513) -> dict:
    return {"model_type": "kolibri1", "num_hidden_layers": n_layers, "num_attention_heads": 48,
            "num_key_value_heads": kv, "head_dim": hd, "hidden_size": 2560, "sliding_window": window,
            "layer_types": ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(n_layers)]}


def test_kolibri_prediction():
    g, f = kvb.kv_bytes_per_token(_kolibri_like())
    assert (g, f) == (20480, 42_024_960)
    assert kvb.check(_kolibri_like())["mismatch"] == []


def test_mismatch_is_recorded_not_raised():
    r = kvb.check(_kolibri_like(kv=8))
    assert r["growing_bytes_per_token"] == 40960 and r["mismatch"]


def test_tiny_preset_by_hand():
    cfg = tc.preset_config("vendor")
    g, f = kvb.kv_bytes_per_token(cfg)
    n_full = sum(t == "full_attention" for t in cfg["layer_types"])
    n_slide = len(cfg["layer_types"]) - n_full
    per = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    assert (g, f) == (n_full * per, n_slide * cfg["sliding_window"] * per)


def test_agrees_with_runner_memory_kolibri_formula():
    try:
        from runner.memory import kolibri_kv_from_config
    except ModuleNotFoundError:
        pytest.skip("runner/memory.py is not present yet")
    for cfg in (_kolibri_like(), tc.preset_config("vendor"), tc.preset_config("pattern5")):
        assert kvb.kv_bytes_per_token(cfg) == kolibri_kv_from_config(cfg)


def _gemma_like() -> dict:
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 5
    return {"model_type": "gemma4", "text_config": {
        "model_type": "gemma4_text", "num_hidden_layers": 30, "layer_types": types, "num_key_value_heads": 8,
        "head_dim": 256, "num_global_key_value_heads": 2, "global_head_dim": 512, "attention_k_eq_v": True,
        "sliding_window": 1024, "num_kv_shared_layers": 0, "num_attention_heads": 16}}


def _qwen_like(n_layers=40, v_heads=32, kv=2) -> dict:
    return {"model_type": "qwen3_5_moe", "text_config": {
        "num_hidden_layers": n_layers, "full_attention_interval": 4, "num_key_value_heads": kv, "head_dim": 256,
        "num_attention_heads": 16, "hidden_size": 2048, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": v_heads, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}}


def test_peer_formulas():
    assert kvb.kv_bytes_per_token(_gemma_like()) == (20480, 209_715_200)
    g, f = kvb.kv_bytes_per_token(_qwen_like())
    assert g == 20480
    conv = 3 * (2 * 16 * 128 + 32 * 128) * 2
    state = 32 * 128 * 128 * 4
    assert f == 30 * (conv + state)
    q38 = _qwen_like(n_layers=64, v_heads=48, kv=4)
    q38["model_type"] = "qwen3_5"
    assert kvb.kv_bytes_per_token(q38)[0] == 65536


def test_real_peer_configs_if_present():
    from exp036_helpers import peer_root

    root = peer_root(["gemma-4-26b-a4b-it-8bit", "Qwen3.6-35B-A3B-8bit", "Qwen3.8-27B-8bit"])
    if root is None:
        pytest.skip("peer configs not found (set EXP036_MINI_ASSETS on the mini, EXP036_MODELS on the run host)")
    want = {"gemma-4-26b-a4b-it-8bit": 20480, "Qwen3.6-35B-A3B-8bit": 20480, "Qwen3.8-27B-8bit": 65536}
    for folder, g in want.items():
        cfg = json.loads((root / folder / "config.json").read_text())
        r = kvb.check(cfg)
        assert r["growing_bytes_per_token"] == g and r["mismatch"] == [], (folder, r)
