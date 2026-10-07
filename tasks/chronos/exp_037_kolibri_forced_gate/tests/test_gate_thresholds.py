# SPDX-License-Identifier: MIT
"""gate/thresholds.json, version exp037-gate-1 (DESIGN §3-§6, §12 W6; BUILD_SPEC §7.1).

The file is frozen with GATE_RULES_SHA256. It must:
* keep exp_036's registered blocks G0, G1, G2, G3, G4 and G5 value for value
  (G2 adds A1's per-pair constants; the "descriptive" lists name the keys that
  now feed only descriptive rows);
* carry the new blocks P2, P3_G0k, P4, P5, G4N, G4F32, G4F16, G5D32, G5R1 and
  G5BP with the constants of DESIGN §3-§6, each derived constant equal to its
  derivation and each adopted registered constant equal to the registered one;
* cite the sha256 of every source of P2's pins, P3's lifted scripts and P4's
  convert records, and agree with those sources;
* carry every key the gate code reads;
* agree with exp_037's HYPOTHESIS.md gate table and with BUILD_SPEC §7.1's
  JSON block (HYPOTHESIS wins over BUILD_SPEC).

The two document assertions run as soon as the documents carry the table and
the block. Until W12 writes them they are off: the tests pass with a warning,
because under EXP036_REQUIRE_ALL a skip is a failure and W14 (stage 4) runs
before W12 (stage 5). DOC_ASSERTIONS_ON is switched to True after W12's
hand-off note (DESIGN §12 W6, W15); from then on a missing table or block fails.
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
E36 = EXP.parent / "exp_036_kolibri_local_eval"
THRESHOLDS = EXP / "gate" / "thresholds.json"

# Switched to True after W12's hand-off note (DESIGN §12 W6, W15; final review W15-01).
DOC_ASSERTIONS_ON = True

REGISTERED_BLOCKS = ("G0", "G1", "G2", "G3", "G4", "G5")
NEW_BLOCKS = ("P2", "P3_G0k", "P4", "P5", "G4N", "G4F32", "G4F16", "G5D32", "G5R1", "G5BP")
# Keys exp_037 adds to registered blocks (no registered value changes).
ADDED_TO_REGISTERED = {"G2": {"pairwise_z_cap", "pairwise_z_factor_vs_emu", "descriptive"},
                       "G3": {"descriptive"}, "G4": {"descriptive"}}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def th() -> dict:
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- schema and inheritance


def test_schema(th):
    assert th["version"] == "exp037-gate-1"
    assert set(th) == {"version", *REGISTERED_BLOCKS, *NEW_BLOCKS, "K4_blocking", "fix_cycles_max"}
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
    assert th["K4_blocking"] == ["G0", "same_port_sha", "G2q", "G5_behaviour", "G5_R1_anchor"]
    assert th["fix_cycles_max"] == 2
    for block in ("G2", "G3", "G4"):
        assert set(th[block]["descriptive"]) <= set(th[block]) - {"descriptive"}, block


def test_registered_blocks_inherited_value_for_value(th):
    """G0, G1, G2, G3, G4 and G5 equal exp_036's registered file; exp_037 only adds keys."""
    e36 = json.loads((E36 / "gate" / "thresholds.json").read_text(encoding="utf-8"))
    assert e36["version"] == "exp036-gate-2"
    for block in REGISTERED_BLOCKS:
        added = ADDED_TO_REGISTERED.get(block, set())
        assert set(th[block]) - set(e36[block]) == added, block
        assert {k: v for k, v in th[block].items() if k not in added} == e36[block], block
    assert th["fix_cycles_max"] == e36["fix_cycles_max"]
    assert th["K4_blocking"][:4] == e36["K4_blocking"]  # exp_037 adds G5-R1's anchor leg for K4 (§3.16)


def test_descriptive_keys_are_not_rules(th):
    """The registered 6-sigma rule, G3a's bounds and G4's decisive top-1 are
    retired to descriptive rows: gate/rules.py never reads them for a decision."""
    assert th["G2"]["descriptive"] == ["bf16_selection_gap_sigma"] and th["G2"]["bf16_selection_gap_sigma"] == 6.0
    assert th["G3"]["descriptive"] == ["ref_bpb_per_text_max", "ref_bpb_mean_vs_best_peer_max"]
    assert th["G4"]["descriptive"] == ["K8_backstop_decisive_top1_min"]
    src = (EXP / "gate" / "rules.py").read_text(encoding="utf-8")
    assert '["bf16_selection_gap_sigma"]' in src  # printed beside the descriptive counts only
    for line in src.splitlines():
        if "bf16_selection_gap_sigma" in line or "K8_backstop_decisive_top1_min" in line:
            assert "within(" not in line and "at_least" not in line and "<" not in line, line


# ------------------------------------------------------------------------------- new blocks: values and derivations


def test_new_block_values(th):
    """The constants of DESIGN §3.1, §3.5, §3.7-§3.9, §3.12-§3.14, §5."""
    g2 = th["G2"]
    assert (g2["pairwise_z_cap"], g2["pairwise_z_factor_vs_emu"]) == (8.0, 1.2)
    f32 = th["G4F32"]
    assert f32["sets"] == ["T1-8", "T9"]
    assert (f32["tau_csort_factor"], f32["f_tiny_cap"], f32["tau_cap"], f32["csort_mean_ceiling"]) == (100.0, 1e-5, 1e-4, 1e-6)
    assert (f32["f1_decisive_lead_nats"], f32["f3_rho_factor"], f32["f3_csort_factor"]) == (2.0, 10.0, 10.0)
    assert (f32["f5_position_kl_max"], f32["f6_shadow_gap_min"], f32["descriptive_f4_window"]) == (1e-2, 1e-2, 32)
    f16 = th["G4F16"]
    assert (f16["kappa"], f16["r3_mean_kl_ceiling"], f16["sets"]) == (9.0, 1e-2, ["T1-8", "T9"])
    assert (f16["descriptive_spike_port_kl_min"], f16["descriptive_spike_r3_kl_max"]) == (1.0, 0.1)
    d32 = th["G5D32"]
    assert d32["ranges"] == ["T1", "T3", "T9"] and (d32["chunk"], d32["chunk_small"]) == (2048, 64)
    # Final review W15-04 (2026-10-06, before the freeze): 10 -> 100, G4-F32's tau factor (decode and chunk 64 differ
    # from chunk 2,048 by kernel rounding in every layer); the floors and ceilings are unchanged.
    assert (d32["mean_factor_vs_csort"], d32["mean_floor"], d32["max_factor_vs_csort"], d32["max_floor"]) == (100.0, 1e-8, 100.0, 1e-6)
    assert (d32["csort_mean_ceiling"], d32["csort_max_ceiling"], d32["decisive_lead_nats"]) == (1e-6, 1e-4, 2.0)
    r1 = th["G5R1"]
    assert (r1["B"], r1["max_tokens"], r1["prompts"], r1["prefill_step_size"], r1["anchor_factor"]) == (1, 256, 8, 2048, 3.0)
    assert r1["legs"] == {"K8": ["R-anchor", "R-greedy", "R-parity"], "K4": ["R-anchor"]}
    bp = th["G5BP"]
    assert (bp["B"], bp["prefill_batch_size_max"], bp["anchor_factor"], bp["admitted_mid_run_min"]) == ([8, 16], 8, 3.0, 1)
    assert bp["subsets"] == ["first_wave", "mid_run"]
    assert (bp["allowed_B_always"], bp["allowed_B_if_pass_8"], bp["allowed_B_if_pass_8_and_16"]) == ([1], [2, 4, 8], [16])
    p5 = th["P5"]
    assert (p5["p5a_layers"], p5["p5b_text"], p5["p5c_text"], p5["p5c_positions"]) == ([0, 4], "T1", "T1", [1, 1535])
    assert (p5["p5c_id_shift"], p5["p5c_mean_kl_min"], p5["p5a_bitwise"], p5["p5b_bitwise"]) == (1, 0.10, True, True)
    n = th["G4N"]
    assert (n["sets"], n["routing"], n["prefill_chunk"], n["comparator"]) == (["T1-8", "T9"], "free", 2048, "R1")
    emu = n["descriptive_emu_decisive_exp036"]
    assert (emu["set"], emu["misses"], emu["n"]) == ("T9", 68, 5943) and round(1 - 68 / 5943, 5) == 0.98856


def test_derived_constants(th):
    """Each derived constant equals its derivation (DESIGN §3.7, §3.8, §3.12, §5.1-§5.2, §11.5)."""
    f32, d32, f16, g2 = th["G4F32"], th["G5D32"], th["G4F16"], th["G2"]
    assert f32["f_tiny_cap"] == pytest.approx(10 * f32["csort_mean_ceiling"], rel=1e-12)       # cap = 10 x Csort ceiling
    assert f32["tau_cap"] == pytest.approx(max(f32["f_tiny_cap"], f32["tau_csort_factor"] * f32["csort_mean_ceiling"]), rel=1e-12)
    assert f32["f5_position_kl_max"] == pytest.approx(100 * f32["tau_cap"], rel=1e-12)          # 100 x tau's cap
    assert f32["f6_shadow_gap_min"] == pytest.approx(100 * g2["fp32_selection_near_tie_gap"], rel=1e-12)  # 100 x near-tie
    assert d32["csort_max_ceiling"] == pytest.approx(100 * d32["csort_mean_ceiling"], rel=1e-12)
    assert d32["mean_floor"] == pytest.approx(d32["csort_mean_ceiling"] / 100, rel=1e-12)
    assert d32["max_floor"] == d32["csort_mean_ceiling"] == f32["csort_mean_ceiling"]
    assert d32["mean_factor_vs_csort"] == d32["max_factor_vs_csort"] == f32["tau_csort_factor"]   # W15-04: one derivation
    assert f16["kappa"] == g2["bf16_forced_median_factor_vs_emu"] ** 2                           # 3 squared
    # Csort mean ceiling: 2 x the KL of a head-tolerance logit change, 1/2 (1e-3)^2 (§5.2)
    assert f32["csort_mean_ceiling"] == pytest.approx(2 * 0.5 * g2["head_fp32_logit_maxabs"] ** 2, rel=1e-12)


def test_adopted_constants_equal_the_registered_ones(th):
    """New rules that adopt a registered constant carry the registered value."""
    g2, g4, g5 = th["G2"], th["G4"], th["G5"]
    lead = g4["K8_backstop_decisive_lead_nats"]
    assert lead == g5["greedy_decisive_lead_nats"] == th["G4F32"]["f1_decisive_lead_nats"] == th["G5D32"]["decisive_lead_nats"]
    assert th["G4F16"]["descriptive_decisive_lead_nats"] == lead
    assert th["P5"]["p5c_mean_kl_min"] == g4["K8_backstop_mean_kl_max"]                    # the gross-bug bound
    factor = g2["bf16_forced_median_factor_vs_emu"]
    assert th["P3_G0k"]["rel_sorted_max_factor_vs_unsorted"] == factor                       # G2's bf16 factor
    assert th["G5R1"]["anchor_factor"] == th["G5BP"]["anchor_factor"] == factor == g5["parity_kl_factor"]
    assert th["G4N"]["kl_leg"] == "G4.K8_backstop_mean_kl_max" and g4["K8_backstop_mean_kl_max"] == 0.10
    assert th["G5R1"]["max_tokens"] == g5["greedy_tokens"] and th["G5R1"]["prompts"] == g5["greedy_prompts"]
    assert th["G5D32"]["chunk"] == th["G4N"]["prefill_chunk"] == th["G5R1"]["prefill_step_size"] == g5["noise_floor_prefill_steps"][0]
    assert th["G5D32"]["chunk_small"] == g5["noise_floor_prefill_steps"][1]
    assert th["G5BP"]["prefill_batch_size_max"] == g5["batch_B"] == 8


def test_g5bp_layout(th):
    """The 26 prompts (§3.14): 17,708 tokens and 792 generated positions per configuration."""
    lay = th["G5BP"]["layout"]
    assert lay["text_order"] == [f"T{i}" for i in range(1, 9)] and lay["b_text_offset"] == 4
    assert lay["lengths"] == [37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100]
    assert lay["max_tokens"] == [48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44]
    assert len(set(lay["lengths"])) == 8 and len(lay["lengths"]) == len(lay["max_tokens"]) == 12
    l1, l2 = lay["L1"], lay["L2"]
    assert (l1["t9_ids"], l1["max_tokens"], l2["t9_ids"], l2["max_tokens"]) == ([0, 2100], 24, [4096, 7096], 40)
    prompt = 2 * sum(lay["lengths"]) + (l1["t9_ids"][1] - l1["t9_ids"][0]) + (l2["t9_ids"][1] - l2["t9_ids"][0])
    generated = 2 * sum(lay["max_tokens"]) + l1["max_tokens"] + l2["max_tokens"]
    assert (prompt, generated, prompt + generated) == (lay["prompt_tokens"], lay["generated_positions"], 17708)
    assert lay["queue"] == ["A0-A11", "L1", "L2", "B0-B11"]
    assert 2 * len(lay["lengths"]) + 2 == 26
    for L in (l1, l2):  # both cross a 2,048-token prefill chunk
        assert L["t9_ids"][0] // 2048 != (L["t9_ids"][1] - 1) // 2048
    tiny = th["G5BP"]["tiny_layout"]
    assert (tiny["L1"], tiny["L2"]) == ({"t9_ids": [0, 100], "max_tokens": 6}, {"t9_ids": [128, 300], "max_tokens": 10})


def test_p3_judged_rows(th):
    """G0k's shapes (§3.1 P3, decision F2): every count the run issues, the B = 16
    bound and mlx_repro_min.py's boundary sweep; nothing recorded only."""
    p3 = th["P3_G0k"]
    assert (p3["seed"], p3["experts"], p3["group_size"], p3["mode"], p3["bits"]) == (37, 384, 64, "affine", [8, 4])
    assert p3["activation_dtype"] == "bfloat16"
    assert p3["projections"] == {"gate_up": {"n_out": 512, "d_in": 2560}, "down": {"n_out": 2560, "d_in": 512}}
    j = p3["judged_rows"]
    assert j["decode_kolibri"] == [6 * B for B in (1, 2, 4, 8, 16)]
    assert j["decode_peers"] == [8 * B for B in (1, 2, 4, 8, 16)]
    assert j["prefill_chunk_kolibri"] == [2048 * 6] and j["prefill_chunk_peers"] == [2048 * 8]
    assert p3["judged_rows_one_expert"] == [2048]
    assert j["batched_prefill_kolibri"] == [n * 2048 * 6 for n in (2, 4, 8)]
    assert j["batched_prefill_peers"] == [n * 2048 * 8 for n in (2, 4, 8)]
    assert j["first_wave_exp036"] == [8 * 1099 * 6] and 52752 % 64 != 0
    assert j["g2_single_call"] == [8 * 1536 * 6] and j["b16_bound"] == [16 * 2048 * 6]
    src = (E36 / "diagnostics" / "gate1" / "confirm" / "mlx_repro_min.py").read_text(encoding="utf-8")
    sweep = json.loads(re.search(r"^\s*sweep = (\[[0-9, ]+\])$", src, re.M).group(1))
    assert j["boundary_mlx_repro_min"] == sweep
    assert max(n for v in j.values() for n in v) == 196608
    assert p3["recorded_only"] == []
    assert (p3["rel_sorted_max_factor_vs_unsorted"], p3["bad_row_rel"], p3["bad_rows_max"], p3["repeat_bitwise"]) == (3.0, 0.05, 0, True)
    for name, prefix in p3["lifted_from"].items():
        assert _sha(E36 / "diagnostics" / "gate1" / "confirm" / name).startswith(prefix), name
    assert p3["lifted_from"] == {"mlx_repro.py": "9c98cbc4c6f971e0", "mlx_repro_min.py": "7004dcf44754c598"}


def test_p2_pins_and_their_sources(th):
    """P2 (§3.1, decision F2): the pins equal their cited sources, by sha256."""
    p2 = th["P2"]
    assert (p2["macos"], p2["macos_build"], p2["architecture_prefix"], p2["MLX_ENABLE_TF32"]) == ("27.0", "26A428", "applegpu_g17", "0")
    assert p2["packages"] == {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}
    assert (p2["version_record_schema"], p2["preflight_schema"]) == ("exp037 version record v1", "exp037 preflight v1")
    src = p2["sources"]
    for name, s in src.items():
        assert _sha((EXP / s["path"]).resolve()) == s["sha256"], name
    req = (EXP / src["packages"]["path"]).read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([^\s]+)$", req, re.M))
    for pkg, ver in p2["packages"].items():
        assert pins[pkg] == ver, pkg
    versions = json.loads((EXP / "env" / "versions.json").read_text(encoding="utf-8"))
    assert {k: versions["core"][k] for k in p2["packages"]} == p2["packages"]
    vr = json.loads((EXP / src["os"]["path"]).read_text(encoding="utf-8"))
    assert (vr["os"]["macos"], vr["os"]["build"]) == (p2["macos"], p2["macos_build"])
    pf = json.loads((EXP / src["architecture"]["path"]).read_text(encoding="utf-8"))
    assert pf["mx_device_info"]["architecture"].startswith(p2["architecture_prefix"])
    assert pf["mx_device_info"]["architecture"] == "applegpu_g17s"
    assert not Path(src["os"]["path"]).is_absolute() and not Path(src["architecture"]["path"]).is_absolute()


def test_p4_build_identity_sources(th):
    from gate import common

    p4 = th["P4"]
    assert set(p4["builds"]) == {"K8", "K4"}
    for arm, b in p4["builds"].items():
        assert b["dir"] == common.ARM_DIRS[arm]
        rec = EXP / b["convert_record"]
        assert _sha(rec) == b["convert_record_sha256"], arm
        r = json.loads(rec.read_text(encoding="utf-8"))
        assert r["bits"] == common.ARM_BITS[arm] and r["out_name"] == b["dir"]
        assert "config.json" in r["files"] and any(f.startswith("model") and f.endswith(".safetensors") for f in r["files"])
    assert (p4["shards"], p4["config"], p4["port_check"]) == ("model*.safetensors", "config.json", "port.convert.check_port_file")


def test_k4_blocking_names_the_rules_groups(th):
    from gate import rules

    assert list(rules.K4_BLOCKING_GROUPS) == th["K4_blocking"]
    flat = [cid for g in th["K4_blocking"] for cid in rules.K4_BLOCKING_GROUPS[g]]
    assert sorted(flat) == sorted(rules.BLOCKING["K4"]) and len(flat) == len(set(flat))


def test_no_absolute_or_home_paths(th):
    """Every path the frozen files cite is relative to the experiment directory."""
    text = THRESHOLDS.read_text(encoding="utf-8") + (EXP / "gate" / "controls.json").read_text(encoding="utf-8")
    for bad in ("/Users/", "/home/", "/private/", "~/", "$HOME"):
        assert bad not in text, bad


# ------------------------------------------------------------------------------- the code reads only existing keys


def test_every_key_the_gate_reads_exists(th):
    """Every th["..."] / thN["..."] key in gate/ is in the file."""
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

    assert common.thresholds_sha256() == hashlib.sha256(THRESHOLDS.read_bytes()).hexdigest()
    assert common.load_thresholds() == th


# ------------------------------------------------------------------------------- documents (on after W12)


def _doc_off(what: str) -> None:
    if DOC_ASSERTIONS_ON:
        pytest.fail(f"{what} is missing, and the document assertions are switched on (W12 has handed off)")
    warnings.warn(f"{what} is not written yet (W12); its assertions are off until W12's hand-off note")


def _spec_block() -> dict | None:
    p = EXP / "BUILD_SPEC.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    if "### 7.1 `gate/thresholds.json`" not in text:
        return None
    sec = text[text.index("### 7.1 `gate/thresholds.json`"):]
    if "```json" not in sec:
        return None
    block = sec[sec.index("```json") + len("```json"):]
    return json.loads(block[: block.index("```")])


def test_equals_build_spec_block(th):
    block = _spec_block()
    if block is None:
        _doc_off("BUILD_SPEC.md §7.1's JSON block")
        return
    assert th == block


def _gate_table() -> str | None:
    """The Phase 0 section of HYPOTHESIS.md, if it carries the gate table (a
    markdown table whose header starts with "| Check")."""
    p = EXP / "HYPOTHESIS.md"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8").replace("\\|", "|")  # markdown-escaped pipes
    m = re.search(r"^## Phase 0.*?(?=^## (?!#))", text + "\n## end\n", re.S | re.M)
    if m is None or not re.search(r"^\| Check\b", m.group(0), re.M):
        return None
    return m.group(0)


# (regex in the HYPOTHESIS gate table, the thresholds value it states)
HYPOTHESIS_ROWS = [
    (r"58,353 tensors", ("G0", "tensor_count", 58353)),
    (r"78,103,074,560 parameters", ("G0", "param_count", 78103074560)),
    (r"≥ 2,000", ("G0", "tokenizer_lines_min", 2000)),
    (r"each fraction ≤ 1 %", ("G0", "router_bf16_exact_frac_max", 0.01)),
    (r"median e ≤ 1e-4 and max e ≤ 1e-3", ("G2", "fp32_forced_rel_err_median_max", 1e-4)),
    (r"max \|Δlogit\| ≤ 1e-3", ("G2", "head_fp32_logit_maxabs", 1e-3)),
    (r"gap < 1e-4", ("G2", "fp32_selection_near_tie_gap", 1e-4)),
    (r"median e ≤ 3 × emu's median, and p99 e ≤ max\(5e-2, 3 × emu's p99\)", ("G2", "bf16_forced_p99_floor", 5e-2)),
    (r"max\(0\.2 %, 2 × emu's fraction\)", ("G2", "bf16_selection_disagree_floor", 0.002)),
    (r"min\(8(?:\.0)?, 1\.2 × M_emu\)", ("G2", "pairwise_z_cap", 8.0)),
    (r"embedding max relative error ≤ 4e-3", ("G2", "g2q_embedding_rel_err_max", 4e-3)),
    (r"head relative L2 error ≤ 1e-3", ("G2", "g2q_head_rel_l2_max", 1e-3)),
    (r"48-block bootstrap", ("G3", "ref_mutant_blocks", 48)),
    (r"mean KL ≤ 0\.10", ("G4", "K8_backstop_mean_kl_max", 0.10)),
    (r"100 × C̄sort", ("G4F32", "tau_csort_factor", 100.0)),
    (r"F_tiny[^\n]*≤ 1e-5", ("G4F32", "f_tiny_cap", 1e-5)),
    (r"KL > 1e-2", ("G4F32", "f5_position_kl_max", 1e-2)),
    (r"gap ≥ 1e-2", ("G4F32", "f6_shadow_gap_min", 1e-2)),
    (r"κ = 9", ("G4F16", "kappa", 9.0)),
    (r"max\(1e-8, 100 × Csort", ("G5D32", "mean_floor", 1e-8)),
    (r"max\(1e-6, 100 × Csort", ("G5D32", "max_floor", 1e-6)),
    (r"≥ 99\.5 %", ("G5", "greedy_decisive_top1_min", 0.995)),
    (r"max\(1e-4, 3 × floor_KL\)", ("G5", "parity_kl_floor", 1e-4)),
    (r"3 × floor_dis \+ 0\.2 pp", ("G5", "parity_dis_add", 0.002)),
    (r"≥ 2 ?/ ?20", ("G5", "behaviour_block_count", 2)),
    (r"196,608", ("P3_G0k", "b16_bound", None)),
    (r"52,752", ("P3_G0k", "first_wave_exp036", None)),
]


def test_values_match_hypothesis_gate_table(th):
    """The numbers of exp_037's HYPOTHESIS Phase 0 gate table, cited row by row."""
    table = _gate_table()
    if table is None:
        _doc_off("HYPOTHESIS.md's Phase 0 gate table")
        return
    missing = [pat for pat, _ in HYPOTHESIS_ROWS if not re.search(pat, table)]
    assert not missing, missing
    for _, (block, key, value) in HYPOTHESIS_ROWS:
        if value is None:
            assert key in th[block]["judged_rows"], key
        else:
            assert th[block][key] == value, (block, key)
