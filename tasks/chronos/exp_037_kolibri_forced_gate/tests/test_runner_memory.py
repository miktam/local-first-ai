"""runner/memory.py: the memory rule, B and the Metal limit (HYPOTHESIS rule 1,
"K8 working-set rule"; BUILD_SPEC §5.4).

Measured on the mbp (host/mbp_preflight_20261003T170152Z.json): L = 115,448,725,504 B
(107.52 GiB), iogpu.wired_limit_mb = 0. K8 weights ≈ 83.1 GB (spec item 20).
"""

from __future__ import annotations

import json
import os

import pytest

from exp036_helpers import kolibri_tok_env

from runner import memory
from runner.common import load_rules

GiB = 1 << 30
L_MBP = 115_448_725_504
K8_BYTES = 83_100_000_000
RULES = load_rules()


def B(task, cap, L, wb=K8_BYTES, arm="K8"):
    return memory.choose_B(arm, task, cap, L, weight_bytes=wb, rules=RULES)


def test_need_bytes_arithmetic():
    need = memory.need_bytes(10 * GiB, 4, 1000, 9000, 20480, 42_024_960)
    assert need == 10 * GiB + 2 * 4 * 10_000 * 20480 + 4 * 42_024_960 + 4 * GiB


def test_kolibri_kv_bytes_from_config():
    cfg = {"layer_types": ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(50)],
           "num_key_value_heads": 4, "head_dim": 128, "sliding_window": 513}
    assert memory.kolibri_kv_from_config(cfg) == (20_480, 42_024_960)
    tok = kolibri_tok_env()
    real = os.path.join(tok, "config.json") if tok else None
    if real and os.path.isfile(real):  # the released config, when present
        assert memory.kolibri_kv_from_config(json.load(open(real))) == (20_480, 42_024_960)
    assert RULES["kv_bytes_per_token"]["kolibri"] == 20_480
    assert RULES["fixed_window_bytes"]["kolibri"] == 42_024_960


def test_effective_limit_is_the_max_of_mlx_and_iogpu():
    di = lambda: {"max_recommended_working_set_size": L_MBP}  # noqa: E731
    L, src = memory.effective_limit(sysctl_reader=lambda name: "0", device_info=di)
    assert L == L_MBP and src["chosen"] == "max_recommended_working_set_size"
    L, src = memory.effective_limit(sysctl_reader=lambda name: "114688", device_info=di)
    assert L == 114688 * 2**20 and src["chosen"] == "iogpu.wired_limit_mb"
    L, _ = memory.effective_limit(sysctl_reader=lambda name: None, device_info=di)
    assert L == L_MBP


@pytest.mark.parametrize("L", [L_MBP, 112 * GiB])
def test_k8_B_at_the_measured_and_the_raised_limit(L):
    assert B("gpqa_en", 32_768, L) == 8
    assert B("gpqa_en", 65_536, L) == 4
    assert B("mmlu_en", 32_768, L) == 8
    assert B("ifbench", 16_384, L) == 16
    assert B("rgb_cb", 16_384, L) == 16


def test_aime_is_capped_at_B8():
    small = 10 * GiB
    assert B("aime_en", 65_536, L_MBP, wb=small) == 8
    assert B("ifbench", 16_384, L_MBP, wb=small) == 16


def test_no_sysctl_advice_at_the_measured_limit_and_initial_caps():
    cells = [("gpqa_en", 32_768), ("gpqa_de", 32_768), ("mmlu_en", 32_768), ("ifbench", 16_384), ("rgb_cb", 16_384),
             ("rgb_neg", 16_384)]
    assert memory.sysctl_advice(cells, L_MBP, weight_bytes=K8_BYTES, rules=RULES) is None
    assert memory.sysctl_advice([("gpqa_en", 65_536), ("aime_en", 65_536)], L_MBP, weight_bytes=K8_BYTES, rules=RULES) is None


def test_sysctl_advice_when_B1_does_not_fit():
    L90 = 90 * GiB
    assert B("gpqa_en", 32_768, L90) == 0
    adv = memory.sysctl_advice([("gpqa_en", 32_768)], L90, weight_bytes=K8_BYTES, rules=RULES)
    assert adv is not None and adv.startswith("sudo sysctl iogpu.wired_limit_mb=114688")
    assert "does not fit at B = 1" in adv


def test_sysctl_advice_names_the_B_gain():
    # A limit at which 112 GiB raises GPQA from B = 4 to B = 8.
    L = 100 * GiB
    assert B("gpqa_en", 32_768, L) == 4 and B("gpqa_en", 32_768, 112 * GiB) == 8
    adv = memory.sysctl_advice([("gpqa_en", 32_768)], L, weight_bytes=K8_BYTES, rules=RULES)
    assert "B 4 -> 8" in adv


def test_weight_bytes_sums_safetensors(tmp_path):
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"x" * 23)
    (tmp_path / "config.json").write_text("{}")
    assert memory.arm_weight_bytes(tmp_path) == 123
    with pytest.raises(FileNotFoundError):
        memory.arm_weight_bytes(tmp_path / "nothing")
