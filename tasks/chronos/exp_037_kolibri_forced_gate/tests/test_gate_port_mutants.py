# SPDX-License-Identifier: MIT
"""gate/port_mutants.py (HYPOTHESIS Phase 0 "Port mutants"; BUILD_SPEC §5.3,
the test BUILD_SPEC calls test_mutants.py).

On the tiny checkpoints each mutant fails its target check by more than 100 x
that check's tolerance:
  * semantic mutants (2-7, 12-15): the G2 fp32 forced branch error max e
    exceeds 100 x the G2 max threshold (1e-3) on some layer, through the
    gate's own harness (gate/harness.py) against the reference;
  * mutant 1 (selection): forced routing is blind to it by design; the
    natural-selection disagreement fraction exceeds 100 x the bf16 floor (0.2 %);
  * mutant 8 (embedding): the trunk's layer-0 input is off by more than
    100 x the G2q embedding tolerance (4e-3), relative;
  * mutant 9: the strict load fails (a parameter absent from the checkpoint);
  * mutants 10 and 11: on a near-tie fixture (router) and a large-logit
    fixture (head, |logit| >= 16) bf16 rounding changes the outcome: every
    value is bf16-exact (fraction 1 against the <= 1 % rule, 100 x) and the
    selection / logit moves by more than 100 x the tolerance.
port/kolibri1.py itself is never edited (checked by hash).

exp_037 adds the control mutants 19, 20, 22, 26 and 27 (attention-hook
overrides, DESIGN §4.1; 27 since decision (b), with 22 kept as a probe). They are
registered here and built below with the others; their regimes are tested in
test_gate_port_mutants_exp037.py.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint as tc

EXP = Path(__file__).resolve().parents[1]
SEMANTIC = ("window_512", "rope_on_full", "one_plus_w_norm", "swap_sandwich_norms", "renorm_topk",
            "route_scale_2826", "rope_traditional", "qknorm_after_rope", "biased_weights", "swiglu_swapped")
FACTOR = 100


def test_registry():
    from gate import port_mutants as pm

    # exp_036's 15 registered mutants plus exp_037's five hook mutants (DESIGN §4.1):
    # the controls 19, 20, 26 and 27, and 22 (a probe since decision (b)).
    assert len(pm.PORT_MUTANTS) == 20 and set(pm.PORT_MUTANTS) == set(pm.PATCHES) == set(pm.MUTANT_IDS.values())
    assert pm.MUTANT_IDS == {
        1: "sigmoid_bias_select", 2: "window_512", 3: "rope_on_full", 4: "one_plus_w_norm",
        5: "swap_sandwich_norms", 6: "renorm_topk", 7: "route_scale_2826", 8: "mup_embed_scale",
        9: "attn_output_gate", 10: "bf16_router_logits", 11: "bf16_head_logits", 12: "rope_traditional",
        13: "qknorm_after_rope", 14: "biased_weights", 15: "swiglu_swapped",
        19: "rope_restart_per_chunk", 20: "decode_window_256",
        22: "batched_rope_positions_in_padded_frame", 26: "batch_decode_pos_frozen",
        27: "batch_decode_pad_keys_visible"}
    assert set(pm.TARGETS) == set(pm.PORT_MUTANTS)
    # The registered 15 keep their exp_036 targets.
    assert {pm.MUTANT_IDS[i]: pm.TARGETS[pm.MUTANT_IDS[i]] for i in range(1, 16)} == {
        "sigmoid_bias_select": "g2_fp32_natural_selection", "window_512": "g2_fp32_forced_branch",
        "rope_on_full": "g2_fp32_forced_branch", "one_plus_w_norm": "g2_fp32_forced_branch",
        "swap_sandwich_norms": "g2_fp32_forced_branch", "renorm_topk": "g2_fp32_forced_branch",
        "route_scale_2826": "g2_fp32_forced_branch", "mup_embed_scale": "g2_embedding",
        "attn_output_gate": "g0_strict_load", "bf16_router_logits": "g0_router_bf16_exact",
        "bf16_head_logits": "g0_head_bf16_exact", "rope_traditional": "g2_fp32_forced_branch",
        "qknorm_after_rope": "g2_fp32_forced_branch", "biased_weights": "g2_fp32_forced_branch",
        "swiglu_swapped": "g2_fp32_forced_branch"}
    assert pm.REAL_WEIGHT_MUTANTS == tuple(pm.MUTANT_IDS[i] for i in (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15))


def test_mutants_build_from_args_and_leave_the_port_untouched(tiny_pattern5_dir):
    from gate import common, port_mutants as pm

    before = hashlib.sha256(common.PORT_FILE.read_bytes()).hexdigest()
    module = common.exec_port_module()
    args = module.ModelArgs.from_dict(common.read_json(tiny_pattern5_dir / "config.json"))
    for name, build in pm.PORT_MUTANTS.items():
        model = build(args)
        assert type(model).__name__ == "Model" and hasattr(model, "make_cache"), name
    assert hashlib.sha256(common.PORT_FILE.read_bytes()).hexdigest() == before
    # The unmutated module is unaffected by the patches of the others.
    assert common.exec_port_module().route.__module__ != "gate.port_mutants"


@pytest.fixture(scope="module", params=["vendor", "pattern5"])
def layer_errors(request, tiny_checkpoint_factory):
    """Per mutant: the worst fp32 forced branch error and the natural
    disagreement fraction over all layers, via the gate harness."""
    from gate import common, harness
    from gate.checks import ref_pass

    d = tiny_checkpoint_factory(request.param, 0)
    S, T = 3, 96
    seqs = [tc.random_ids(T, seed=100 + s) for s in range(S)]
    rr = ref_pass.RefRunner(d, "fp32")
    seg = ref_pass.segments_of([T] * S)
    h = rr.ref.embed(np.concatenate(seqs))
    ref_layers = []
    for i in range(rr.cfg.num_hidden_layers):
        b = rr.branches(i, h, seg)
        ref_layers.append((h, b))
        h = b["h_out"]
    out = {}
    for name in (None,) + SEMANTIC + ("sigmoid_bias_select",):
        pl = harness.PortLayers(d, "fp32", mutant=name)
        worst, dis, n = 0.0, 0, 0
        for i, (hin, b) in enumerate(ref_layers):
            layer = pl.build_layer(i)
            H = hin.reshape(S, T, -1)
            top = np.asarray(b["top6"]).reshape(S, T, -1)
            f = pl.run(layer, i, H, force_ids=top, check_call=False)
            for br in ("r_attn", "r_moe"):
                e = common.rel_err_rows(f[br].reshape(S * T, -1), b[br])
                worst = max(worst, float(e.max()))
            nat = pl.run(layer, i, H, check_call=False)
            dis += int(ref_pass.set_disagree(nat["ids"].reshape(S * T, -1), np.asarray(b["top6"])).sum())
            n += S * T
        out[name] = {"worst": worst, "dis_frac": dis / n}
    out["_dir"] = d
    out["_tokens"] = seqs
    return out


def test_port_passes_the_fp32_bounds(layer_errors):
    from gate import common

    th = common.load_thresholds()["G2"]
    assert layer_errors[None]["worst"] <= th["fp32_forced_rel_err_max_max"]
    assert layer_errors[None]["dis_frac"] == 0.0


@pytest.mark.parametrize("name", SEMANTIC)
def test_semantic_mutant_fails_the_forced_branch_check(layer_errors, name):
    from gate import common

    tol = common.load_thresholds()["G2"]["fp32_forced_rel_err_max_max"]
    assert layer_errors[name]["worst"] > FACTOR * tol, layer_errors[name]


def test_selection_mutant_fails_natural_selection(layer_errors):
    from gate import common

    floor = common.load_thresholds()["G2"]["bf16_selection_disagree_floor"]
    m = layer_errors["sigmoid_bias_select"]
    assert m["worst"] <= common.load_thresholds()["G2"]["fp32_forced_rel_err_max_max"]  # forced is blind by design
    assert m["dis_frac"] > FACTOR * floor, m


def test_embedding_mutant_fails_the_embedding_check(tiny_pattern5_dir):
    from gate import common, harness

    tol = common.load_thresholds()["G2"]["g2q_embedding_rel_err_max"]
    ids = np.array([tc.random_ids(40, seed=7)])
    ok = harness.PortLayers(tiny_pattern5_dir, "fp32").embedding_rows(ids)
    bad = harness.PortLayers(tiny_pattern5_dir, "fp32", mutant="mup_embed_scale").embedding_rows(ids)
    w = tc.load_checkpoint_f32(tiny_pattern5_dir)["model.embed_tokens.weight"][ids[0]]
    assert np.array_equal(ok[0], w)  # the port: exact
    assert common.rel_err_rows(bad[0], w).max() > FACTOR * tol


def test_attention_gate_mutant_fails_strict_load(tiny_pattern5_dir):
    from gate.checks import g0_static

    ok = g0_static.strict_load_check(tiny_pattern5_dir)
    assert ok["load_error"] is None and ok["n_missing_params"] == 0 and ok["numel_equal"]
    bad = g0_static.strict_load_check(tiny_pattern5_dir, mutant="attn_output_gate")
    assert bad["n_missing_params"] > 0 and "gate_proj" in bad["missing_params"][0] and bad["load_error"]


def test_bf16_router_mutant_on_a_near_tie():
    """Router-level fixture: biased scores tied to within 1e-6 at the top-k
    boundary, arranged so that bf16 rounding of the logits flips the choice."""
    import mlx.core as mx

    from gate import common, port_mutants as pm

    rng = np.random.default_rng(10)
    E, H, k = 8, 256, 2
    port = common.exec_port_module()
    mut = pm.mutant_module("bf16_router_logits")
    w = mx.array(rng.standard_normal((E, H)).astype(np.float32) * 0.1).astype(mx.bfloat16).astype(mx.float32)
    x = mx.array(rng.standard_normal((1, H)).astype(np.float32)).astype(mx.bfloat16)
    r_port, r_mut = port.Router(H, E), mut.Router(H, E)
    r_port.weight = w
    r_mut.weight = w
    lp = np.array(r_port(x))[0].astype(np.float64)
    lm = np.array(r_mut(x))[0].astype(np.float64)
    d = lm - lp  # the bf16 rounding of each logit
    a, b = int(np.argmin(d)), int(np.argmax(d))  # a loses most, b gains most under rounding
    eps = 1e-6
    assert d[b] - d[a] > FACTOR * eps
    base = np.linspace(-10.0, -3.0, E)
    base[a], base[b] = 0.0 + eps, 0.0  # a just ahead of b for the port
    others = [i for i in range(E) if i not in (a, b)]
    base[others[0]] = 5.0  # one clear winner; the second slot is the a/b near-tie
    bias = mx.array((base - lp).astype(np.float32))
    _, ids_p = port.route(mx.array(lp.astype(np.float32))[None], bias, k)
    _, ids_m = mut.route(r_mut(x), bias, k)
    sel_p, sel_m = set(np.array(ids_p)[0].tolist()), set(np.array(ids_m)[0].tolist())
    assert a in sel_p and b not in sel_p
    assert b in sel_m and a not in sel_m  # bf16 rounding flipped the selection
    assert common.bf16_exact_fraction(np.array(r_mut(x))) == 1.0
    assert common.bf16_exact_fraction(np.array(r_port(x))) <= 0.01  # the port: fp32 logits


def test_bf16_head_mutant_on_large_logits():
    """Head fixture: logits 40.00 ... 40.12 (|logit| >= 16; bf16 spacing 0.25
    there), so bf16 rounding moves them by up to 0.12 > 100 x the 1e-3 head
    tolerance, while the port's fp32 head reproduces them exactly."""
    import mlx.core as mx

    from gate import common, port_mutants as pm

    tol = common.load_thresholds()["G2"]["head_fp32_logit_maxabs"]
    H, V = 64, 16
    port, mut = common.exec_port_module(), pm.mutant_module("bf16_head_logits")
    cfg = tc.preset_config("vendor")
    cfg.update(hidden_size=H, vocab_size=V, num_hidden_layers=1, layer_types=["full_attention"], head_dim=8,
               num_attention_heads=8, num_key_value_heads=2, moe_intermediate_size=64,
               shared_expert_intermediate_size=64)
    args = port.ModelArgs.from_dict(cfg)
    m_port, m_mut = port.Model(args), mut.Model(args)
    w = np.zeros((V, H), dtype=np.float32)
    w[:, 0] = 40.0
    w[:, 1] = np.linspace(0.0, 0.12, V)
    w = np.array(mx.array(w).astype(mx.bfloat16).astype(mx.float32))  # the stored bf16 weight
    h = np.zeros((1, 1, H), dtype=np.float32)
    h[..., :2] = 1.0  # bf16-exact hidden state: logit_v = 40 + w[v, 1]
    exact = h[0, 0].astype(np.float64) @ w.T.astype(np.float64)
    for m in (m_port, m_mut):
        m.lm_head.weight = mx.array(w).astype(mx.bfloat16)
    lp = np.array(m_port.compute_logits(mx.array(h).astype(mx.bfloat16)))[0, 0]
    lm = np.array(m_mut.compute_logits(mx.array(h).astype(mx.bfloat16)))[0, 0]
    assert np.abs(exact).min() >= 16
    assert np.abs(lp - exact).max() <= tol
    assert np.abs(lm - exact).max() > FACTOR * tol
    assert common.bf16_exact_fraction(lm) == 1.0
    assert common.bf16_exact_fraction(lp) < 1.0
