# SPDX-License-Identifier: MIT
"""gate/thresholds.json (BUILD_SPEC §7.1; HYPOTHESIS Phase 0 "Checks").

The file is frozen: it must equal the JSON block of BUILD_SPEC §7.1 value for
value, carry every key the gate code reads, and agree with the numbers of the
HYPOTHESIS Phase 0 table (HYPOTHESIS wins over BUILD_SPEC).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
THRESHOLDS = EXP / "gate" / "thresholds.json"


def _spec_block() -> dict:
    text = (EXP / "BUILD_SPEC.md").read_text(encoding="utf-8")
    sec = text[text.index("### 7.1 `gate/thresholds.json`"):]
    block = sec[sec.index("```json") + len("```json"):]
    return json.loads(block[: block.index("```")])


@pytest.fixture(scope="module")
def th() -> dict:
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


def test_equals_build_spec_block(th):
    assert th == _spec_block()


def test_schema(th):
    assert th["version"] == "exp036-gate-2"
    assert set(th) == {"version", "G0", "G1", "G2", "G3", "G4", "G5", "K4_blocking", "fix_cycles_max"}
    g0 = th["G0"]
    assert isinstance(g0["tensor_count"], int) and isinstance(g0["param_count"], int)
    assert g0["full_attention_layers"] == [i for i in range(50) if i % 5 == 4]
    assert set(g0["tensor_policy"]) == {"quantised_head", "vendor_faithful"}
    for pol in g0["tensor_policy"].values():
        assert set(pol) == {"embed_tokens", "lm_head"}
    g2 = th["G2"]
    assert all(b[0] < b[1] for b in g2["t9_buckets"]) and g2["t9_buckets"][-1][1] == 16384
    assert g2["real_weight_mutants"] == [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15]
    assert th["G5"]["eos_ids"] == [127906, 127901]
    assert th["K4_blocking"] == ["G0", "same_port_sha", "G2q", "G5_behaviour"]
    assert th["fix_cycles_max"] == 2


def test_values_match_hypothesis_phase0(th):
    """The numbers of the HYPOTHESIS Phase 0 table, cited row by row."""
    hyp = (EXP / "HYPOTHESIS.md").read_text(encoding="utf-8").replace("\\|", "|")  # markdown-escaped pipes
    assert "58,353 tensors; 78,103,074,560 parameters" in hyp
    assert (th["G0"]["tensor_count"], th["G0"]["param_count"]) == (58353, 78103074560)
    assert "each fraction ≤ 1 %" in hyp and th["G0"]["router_bf16_exact_frac_max"] == 0.01
    assert "median e ≤ 1e-4 and max e ≤ 1e-3" in hyp
    assert (th["G2"]["fp32_forced_rel_err_median_max"], th["G2"]["fp32_forced_rel_err_max_max"]) == (1e-4, 1e-3)
    assert "max |Δlogit| ≤ 1e-3" in hyp and th["G2"]["head_fp32_logit_maxabs"] == 1e-3
    assert "gap < 1e-4" in hyp and th["G2"]["fp32_selection_near_tie_gap"] == 1e-4
    assert "median e ≤ 3 × emu's median, and p99 e ≤ max(5e-2, 3 × emu's p99)" in hyp
    assert (th["G2"]["bf16_forced_median_factor_vs_emu"], th["G2"]["bf16_forced_p99_floor"],
            th["G2"]["bf16_forced_p99_factor_vs_emu"]) == (3.0, 5e-2, 3.0)
    assert "≤ max(0.2 %, 2 × emu's fraction)" in hyp and "< 6·σ_l" in hyp
    assert (th["G2"]["bf16_selection_disagree_floor"], th["G2"]["bf16_selection_disagree_factor_vs_emu"],
            th["G2"]["bf16_selection_gap_sigma"]) == (0.002, 2.0, 6.0)
    assert "embedding max relative error ≤ 4e-3" in hyp and "head relative L2 error ≤ 1e-3" in hyp
    assert (th["G2"]["g2q_embedding_rel_err_max"], th["G2"]["g2q_head_rel_l2_max"]) == (4e-3, 1e-3)
    assert "each text ≤ 1.2 bpb, and the mean ≤ 1.25 × the best peer" in hyp
    assert (th["G3"]["ref_bpb_per_text_max"], th["G3"]["ref_bpb_mean_vs_best_peer_max"]) == (1.2, 1.25)
    assert "48-block bootstrap" in hyp and th["G3"]["ref_mutant_blocks"] == 48
    assert "mean KL(ref‖K8) ≤ 0.10 nats/token and top-1 agreement ≥ 99 %" in hyp
    assert (th["G4"]["K8_backstop_mean_kl_max"], th["G4"]["K8_backstop_decisive_top1_min"]) == (0.10, 0.99)
    assert "≥ 99.5 %; within the reference top-5 at ≥ 99 %" in hyp
    assert (th["G5"]["greedy_decisive_top1_min"], th["G5"]["greedy_ref_top5_min"]) == (0.995, 0.99)
    assert "mean KL ≤ max(1e-4, 3 × floor_KL); top-1 disagreement ≤ 3 × floor_dis + 0.2 pp" in hyp
    assert (th["G5"]["parity_kl_floor"], th["G5"]["parity_kl_factor"], th["G5"]["parity_dis_factor"],
            th["G5"]["parity_dis_add"]) == (1e-4, 3.0, 3.0, 0.002)
    assert re.search(r"≥ 2/20 loop \(a 32-token span repeated ≥ 4 times", hyp)
    assert (th["G5"]["loop_span_tokens"], th["G5"]["loop_repeats"], th["G5"]["behaviour_block_count"]) == (32, 4, 2)


def test_every_key_the_gate_reads_exists(th):
    """Every th["..."] / thresholds["Gn"]["..."] key in gate/ is in the file."""
    keys = set()
    for p in (EXP / "gate").rglob("*.py"):
        keys |= set(re.findall(r'\bth[0-9]?\["([A-Za-z0-9_]+)"\]', p.read_text(encoding="utf-8")))
    flat = set(th)
    for v in th.values():
        if isinstance(v, dict):
            flat |= set(v)
    missing = sorted(keys - flat)
    assert not missing, missing


def test_sha_helpers(th):
    from gate import common

    import hashlib

    assert common.thresholds_sha256() == hashlib.sha256(THRESHOLDS.read_bytes()).hexdigest()
    assert common.load_thresholds() == th
