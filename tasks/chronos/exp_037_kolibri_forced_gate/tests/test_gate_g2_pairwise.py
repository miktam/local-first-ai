# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5a).
"""G2 per-pair z, the emulation statistics and G3's measurement side (DESIGN §3.4-§3.6, §12 W5a).

* natural_vs_r1 / bf16_selection keep every disagreeing pair's z = R1's
  6th-7th biased gap / sigma_l (synthetic arrays, and on a tiny checkpoint
  against an independent recomputation from the port and the reference);
* the emulation statistics (emu_stats_exp037.json): sigma_emu,l, every z_emu,
  M_emu and where it is, the summed counts, the flagged pair's flags, kept
  only under work37_dir() and reused only for the same R1 dump and settings;
* g2_layers.run keeps the registered single call over T1-T8 and reports the
  flagged pair (layer 20, T4 position 19; a stand-in on tiny);
* the measurement modules emit no "pass" and no verdict: gate/rules.py judges
  their dicts (g2_checks, g2_row_verdicts, g3_checks), and layer_csv_rows
  takes rules.py's per-row verdicts;
* G3b's bootstrap seeds are drawn under the exp037 prefix (via common).

The tiny checkpoint is tiny_checkpoint's pattern5 preset (10 layers, 16
experts, top-6, window 17), converted to 8 bits for G2q. A run of G2 on it
takes a few seconds.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tiny_checkpoint as tc
import tiny_real_layout as trl
from gate import common, rules, textset
from gate.checks import g2_layers, g3_oracle, ref_pass

EXP = Path(__file__).resolve().parents[1]
CHECKS = EXP / "gate" / "checks"
OWNED = ("g2_layers.py", "ref_pass.py", "g3_oracle.py")
# Keys that carry a decision; the measurement modules never write them (gate/rules.py does).
DECISION_KEYS = {"pass", "state", "detected", "forced_fail", "verdict", "resolvable", "limit_median", "limit_p99",
                 "limit_max", "legs", "failing"}


def _quiet(_msg) -> None:
    return None


def _decision_keys(obj, path="") -> list[str]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in DECISION_KEYS:
                found.append(f"{path}/{k}")
            found += _decision_keys(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            found += _decision_keys(v, f"{path}[{j}]")
    return found


def _no_constant(c):
    raise ValueError(f"non-finite JSON constant {c}")


def _strict_json(obj):
    """common.dumps then json.loads, refusing NaN and Infinity (the gate record is strict JSON)."""
    return json.loads(common.dumps(obj), parse_constant=_no_constant)


# ---------------------------------------------------------------------------
# Synthetic: the per-pair z
# ---------------------------------------------------------------------------


def test_natural_vs_r1_keeps_every_pairs_z():
    rng = np.random.default_rng(37)
    N, E, k = 40, 8, 2
    ref_top = np.stack([rng.choice(E, k, replace=False) for _ in range(N)])
    sel = ref_top.copy()
    flip = np.array([3, 11, 12, 30])
    for r in flip:  # swap one expert for one outside the set
        sel[r, 0] = next(e for e in range(E) if e not in ref_top[r])
    sel[5] = sel[5][::-1]  # the same set in another order: not a disagreement
    ref_logits = rng.normal(size=(N, E)).astype(np.float32)
    logits = (ref_logits + rng.normal(scale=1e-2, size=(N, E))).astype(np.float32)
    gap = rng.uniform(0.0, 0.05, N).astype(np.float32)

    s = ref_pass.natural_vs_r1(sel, ref_top, gap, logits, ref_logits, at_row=11)
    diff = logits.astype(np.float64) - ref_logits.astype(np.float64)
    sigma = float(np.sqrt(np.mean(diff * diff)))
    assert s["sigma"] == sigma and s["n"] == N and s["n_disagree"] == flip.size
    assert np.array_equal(s["_rows"], flip)
    z = gap.astype(np.float64)[flip] / sigma
    assert np.array_equal(s["_z"], z) and s["max_gap_over_sigma"] == float(z.max())
    at = s["_at"]
    assert at["disagrees"] is True and at["z"] == float(gap[11]) / sigma and at["sigma_l"] == sigma
    assert at["sigma_t"] == float(np.sqrt(np.mean(diff[11] * diff[11])))
    assert at["dropped"] == sorted(set(ref_top[11].tolist()) - set(sel[11].tolist()))
    assert at["added"] == [int(sel[11, 0])]

    rec = ref_pass.z_record(s, text_len=10, gap_sigma=float(z.max()))  # 4 texts of 10 rows
    assert rec["z_pairs"] == {"rows": flip.tolist(), "z": z.tolist()}
    r_max = int(flip[int(np.argmax(z))])
    assert rec["max_at"] == {"row": r_max, "text": f"T{r_max // 10 + 1}", "pos": r_max % 10}
    # the descriptive count is exp_036's (gap >= multiple x sigma): the maximum counts at equality
    assert rec["n_beyond_6sigma"] == int(np.sum(gap.astype(np.float64)[flip] >= float(z.max()) * sigma))
    assert rec["n_beyond_6sigma"] >= 1
    assert _strict_json(rec)["z_pairs"]["z"] == z.tolist()


def test_natural_vs_r1_without_disagreement_and_at_zero_sigma():
    ref_top = np.array([[0, 1], [2, 3], [1, 2]])
    logits = np.zeros((3, 4), dtype=np.float32)
    s = ref_pass.natural_vs_r1(ref_top[:, ::-1], ref_top, np.ones(3), logits, logits)
    assert s["n_disagree"] == 0 and s["max_gap_over_sigma"] == 0.0 and s["_z"].size == 0
    assert ref_pass.z_record(s, 3, 6.0)["max_at"] is None
    # sigma_l = 0 with a disagreement: +inf for a positive gap, 0 for a tie (never NaN)
    sel = np.array([[0, 2], [2, 3], [1, 3]])
    s = ref_pass.natural_vs_r1(sel, ref_top, np.array([0.5, 0.5, 0.0]), logits, logits)
    assert s["sigma"] == 0.0 and s["_z"].tolist() == [math.inf, 0.0]
    assert s["max_gap_over_sigma"] == math.inf  # rules.py's leg (p) fails it


def test_flagged_site_registered_and_tiny_stand_in():
    assert ref_pass.FLAGGED_PAIR == {"layer": 20, "text": "T4", "pos": 19}
    assert ref_pass.flagged_site(50, 1536) == {"layer": 20, "text": "T4", "pos": 19, "row": 3 * 1536 + 19,
                                               "stand_in": False}
    assert ref_pass.flagged_site(10, 160) == {"layer": 9, "text": "T4", "pos": 19, "row": 3 * 160 + 19,
                                              "stand_in": True}
    assert ref_pass.text_pos(3 * 1536 + 19, 1536) == ("T4", 19)


def test_flagged_pair_report_combines_both_sides():
    site = ref_pass.flagged_site(50, 1536)
    port_at = {"disagrees": True, "gap": 0.006, "z": 6.0, "sigma_l": 1e-3, "sigma_t": 2e-3, "ref_top6": [1, 2],
               "top6": [1, 3], "dropped": [2], "added": [3]}
    emu = {"flagged_pair": {**site, "emu_disagrees": False, "z_emu": 5.0, "sigma_l_emu": 1.2e-3,
                            "sigma_t_emu": 1e-3, "emu_top6": [1, 2], "emu_dropped": [], "emu_added": []}}
    out = g2_layers.flagged_pair_report(site, port_at, emu)
    assert out["port_disagrees"] is True and out["emu_disagrees"] is False
    assert out["z_port"] == 6.0 and out["z_emu"] == 5.0 and out["r_t"] == 2.0 and out["same_swap"] is False
    assert g2_layers.flagged_pair_report(site, port_at, None)["emu"] is None
    other = {"flagged_pair": dict(emu["flagged_pair"], layer=21)}
    with pytest.raises(ValueError, match="flagged pair"):
        g2_layers.flagged_pair_report(site, port_at, other)


# ---------------------------------------------------------------------------
# Tiny: one R1 dump, the emulation statistics and one G2 run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def work37(tmp_path_factory):
    """EXP036_WORK -> a temporary directory for this module; yields work37_dir()."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EXP036_WORK", str(tmp_path_factory.mktemp("work")))
        yield common.work37_dir()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory, work37):
    from gate import harness
    from port import convert

    root = tmp_path_factory.mktemp("g2_pairwise")
    src = tc.write_tiny_checkpoint(root / "Kolibri-1-BF16", seed=0, preset="pattern5", copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    trl.write_stub_tokenizer(src)
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(src, root / common.ARM_DIRS["K8"], 8, 64, guard=False)

    ts = textset.make_tiny(cfg.get("sliding_window") or 0, cfg["num_hidden_layers"], cfg["vocab_size"])
    L = ts.profile.text_len
    d = ref_pass.cache_dir("fp32", ts.key, src, work=work37)
    ref_pass.fp32_dump(src, ts.packed8() + [ts.t9], d, ts.key, log=_quiet)
    dump = ref_pass.Dump(d)
    emu_path = ref_pass.emu_stats_path(src, ts.key)
    emu = ref_pass.emu_stats_exp037(src, dump, 8 * L, [L] * 8, emu_path, log=_quiet)

    th = common.load_thresholds()
    ctx = SimpleNamespace(bf16_dir=src, arms=("K8",), arm_dir=lambda arm: root / common.ARM_DIRS[arm], mutant=None,
                          thresholds=th, want=lambda check: True)
    calls = []
    original = harness.PortLayers.run

    def spy(self, layer, i, h, force_ids=None, chunk=None, check_call=True):
        calls.append({"mode": self.mode, "mutant": self.mutant, "layer": i, "shape": tuple(h.shape), "chunk": chunk})
        return original(self, layer, i, h, force_ids=force_ids, chunk=chunk, check_call=check_call)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(harness.PortLayers, "run", spy)
        res = g2_layers.run(ctx, dump, emu, ts, log=_quiet)
    return SimpleNamespace(root=root, bf16=src, ts=ts, L=L, dump=dump, emu=emu, emu_path=emu_path, res=res, th=th,
                           calls=calls, n_layers=cfg["num_hidden_layers"], work37=work37)


def test_registered_single_call_over_t1_8_and_chunked_t9(tiny):
    """§3.4 (decision F2): every uncached call is the registered one over all 8
    texts; T9 runs through the port's cache in prefill chunks."""
    H = tiny.dump.layer(0, "h_in").shape[-1]
    p = tiny.ts.profile
    assert tiny.calls
    for c in tiny.calls:
        if c["chunk"] is None:
            assert c["shape"] == (8, p.text_len, H), c
        else:
            assert c["shape"] == (1, p.t9_len, H) and c["chunk"] == p.prefill_chunk and c["mode"] == "bf16", c
    assert sum(c["chunk"] is not None for c in tiny.calls) == tiny.n_layers


def test_port_per_pair_z_on_tiny(tiny):
    """Each layer's z_pairs are R1's gap / sigma_port,l at exactly the rows where
    the port's bf16 natural selection differs from R1's (independent re-run)."""
    from gate import harness

    L, n8 = tiny.L, 8 * tiny.L
    rows_seen = 0
    for rec in tiny.res["natural_bf16"]:
        z = np.array(rec["z_pairs"]["z"])
        rows = np.array(rec["z_pairs"]["rows"], dtype=np.int64)
        assert rec["n"] == n8 and rec["n_disagree"] == rows.size == z.size
        assert np.all(np.diff(rows) > 0)
        assert rec["max_gap_over_sigma"] == (float(z.max()) if z.size else 0.0)
        if z.size:
            r = rec["max_at"]["row"]
            assert z[list(rows).index(r)] == z.max() and rec["max_at"]["text"] == f"T{r // L + 1}"
        rows_seen += rows.size
    assert rows_seen > 0, "the tiny checkpoint must have bf16 disagreements for this test to mean anything"

    p16 = harness.PortLayers(tiny.bf16, "bf16")
    site = ref_pass.flagged_site(tiny.n_layers, L)
    for i in sorted({0, site["layer"]}):
        layer = p16.build_layer(i)
        h8 = np.array(tiny.dump.layer(i, "h_in")[:n8]).reshape(8, L, -1)
        nb = p16.run(layer, i, h8)
        ids = nb["ids"].reshape(n8, -1)
        top = np.array(tiny.dump.layer(i, "top6")[:n8])
        dis = np.any(np.sort(ids, axis=-1) != np.sort(top, axis=-1), axis=-1)
        diff = (nb["router_logits"].reshape(n8, -1).astype(np.float64)
                - np.array(tiny.dump.layer(i, "router_logits")[:n8], dtype=np.float64))
        sigma = float(np.sqrt(np.mean(diff * diff)))
        rows = np.flatnonzero(dis)
        gap = np.array(tiny.dump.layer(i, "top6_gap")[:n8], dtype=np.float64)
        rec = tiny.res["natural_bf16"][i]
        assert rec["layer"] == i and rec["sigma"] == sigma
        assert rec["z_pairs"]["rows"] == rows.tolist()
        assert rec["z_pairs"]["z"] == (gap[rows] / sigma).tolist()
        assert rec["n_beyond_6sigma"] == int(np.sum(gap[rows] >= tiny.th["G2"]["bf16_selection_gap_sigma"] * sigma))
        if i == site["layer"]:
            fp = tiny.res["flagged_pair"]
            r = site["row"]
            assert fp["port_disagrees"] == bool(dis[r]) and fp["sigma_l_port"] == sigma
            assert fp["z_port"] == gap[r] / sigma
            assert fp["sigma_t_port"] == float(np.sqrt(np.mean(diff[r] * diff[r])))
        del layer

    # leg (p) reads M_port = the maximum over every layer and pair
    m_port = max(max(r["z_pairs"]["z"], default=0.0) for r in tiny.res["natural_bf16"])
    c = rules.g2_bf16_natural_selection(tiny.res, tiny.emu, tiny.th["G2"])
    assert c["measured"]["M_port"] == m_port and c["measured"]["M_emu"] == tiny.emu["M_emu"]
    assert c["measured"]["z_limit"] == min(tiny.th["G2"]["pairwise_z_cap"],
                                           tiny.th["G2"]["pairwise_z_factor_vs_emu"] * tiny.emu["M_emu"])


def test_m_emu_bookkeeping_on_tiny(tiny):
    emu = tiny.emu
    L, n8 = tiny.L, 8 * tiny.L
    assert emu["version"] == ref_pass.EMU_STATS_VERSION and emu["reused"] is False
    assert len(emu["layers"]) == tiny.n_layers and emu["n_rows"] == n8 and emu["lengths"] == [L] * 8
    all_z = [(lay["layer"], r, z) for lay in emu["layers"] for r, z in zip(lay["z_pairs"]["rows"], lay["z_pairs"]["z"])]
    assert all_z, "the tiny emulation must disagree somewhere for this test to mean anything"
    assert emu["M_emu"] == max(z for _, _, z in all_z) == max(lay["max_gap_over_sigma"] for lay in emu["layers"])
    at = emu["M_emu_at"]
    assert (at["layer"], at["row"], emu["M_emu"]) in all_z
    assert (at["text"], at["pos"]) == ref_pass.text_pos(at["row"], L)
    assert emu["n_disagree"] == sum(lay["n_disagree"] for lay in emu["layers"]) == len(all_z)
    assert emu["n_beyond_6sigma"] == sum(lay["n_beyond_6sigma"] for lay in emu["layers"])
    assert emu["gap_sigma"] == tiny.th["G2"]["bf16_selection_gap_sigma"]
    for lay in emu["layers"]:
        assert lay["natural_disagree_frac"] == lay["n_disagree"] / lay["n"]
        assert {"median", "p99"} <= set(lay["attn"]) and {"median", "p99"} <= set(lay["moe"])

    # one layer against an independent run of the emulated reference
    site = ref_pass.flagged_site(tiny.n_layers, L)
    i = site["layer"]
    rr = ref_pass.RefRunner(tiny.bf16, "emu")
    nat = rr.branches(i, np.array(tiny.dump.layer(i, "h_in")[:n8]), ref_pass.segments_of([L] * 8))
    top = np.array(tiny.dump.layer(i, "top6")[:n8])
    dis = ref_pass.set_disagree(np.asarray(nat["top6"]), top)
    diff = np.asarray(nat["logits"], dtype=np.float64) - np.array(tiny.dump.layer(i, "router_logits")[:n8], dtype=np.float64)
    sigma = float(np.sqrt(np.mean(diff * diff)))
    gap = np.array(tiny.dump.layer(i, "top6_gap")[:n8], dtype=np.float64)
    lay = emu["layers"][i]
    assert lay["sigma"] == sigma and lay["z_pairs"]["rows"] == np.flatnonzero(dis).tolist()
    assert lay["z_pairs"]["z"] == (gap[dis] / sigma).tolist()
    fp = emu["flagged_pair"]
    r = site["row"]
    assert {k: fp[k] for k in ("layer", "text", "pos", "row", "stand_in")} == site
    assert fp["emu_disagrees"] == bool(dis[r]) and fp["z_emu"] == gap[r] / sigma and fp["sigma_l_emu"] == sigma
    assert fp["sigma_t_emu"] == float(np.sqrt(np.mean(diff[r] * diff[r])))

    # the port's report takes the emulation's side and r_t from both
    rep = tiny.res["flagged_pair"]
    assert rep["z_emu"] == fp["z_emu"] and rep["emu_disagrees"] == fp["emu_disagrees"]
    assert rep["r_t"] == rep["sigma_t_port"] / fp["sigma_t_emu"] and rep["stand_in"] is True


def test_emu_stats_file_is_kept_under_work37_and_reused_only_when_it_matches(tiny, tmp_path):
    path = tiny.emu_path
    assert path.name == ref_pass.EMU_STATS_NAME == "emu_stats_exp037.json"
    assert tiny.work37.resolve() in path.resolve().parents
    on_disk = json.loads(path.read_text(), parse_constant=_no_constant)
    assert on_disk["dump_record_sha256"] == common.sha256_file(tiny.dump.dir / "record.json")
    assert on_disk["M_emu"] == tiny.emu["M_emu"] and on_disk["flagged_pair"] == _strict_json(tiny.emu["flagged_pair"])
    L = tiny.L
    args = (tiny.bf16, tiny.dump, 8 * L, [L] * 8)

    again = ref_pass.emu_stats_exp037(*args, path, log=_quiet)
    assert again["reused"] is True and again["M_emu"] == tiny.emu["M_emu"]

    other = tiny.work37 / "w5a-test" / ref_pass.EMU_STATS_NAME
    first = ref_pass.emu_stats_exp037(*args, other, log=_quiet, gap_sigma=5.0)
    assert first["reused"] is False and first["gap_sigma"] == 5.0
    assert ref_pass.emu_stats_exp037(*args, other, log=_quiet, gap_sigma=5.0)["reused"] is True
    assert ref_pass.emu_stats_exp037(*args, other, log=_quiet)["reused"] is False  # 6.0 again: recomputed
    moved = dict(ref_pass.flagged_site(tiny.n_layers, L), layer=0)
    st = ref_pass.emu_stats_exp037(*args, other, log=_quiet, flagged=moved)
    assert st["reused"] is False and st["flagged_pair"]["layer"] == 0
    assert st["layers"] == _strict_json(tiny.emu)["layers"]  # the site moves only the flags

    outside = tmp_path / "elsewhere" / ref_pass.EMU_STATS_NAME
    with pytest.raises(ValueError, match="exp_037 writers stay under"):
        ref_pass.emu_stats_exp037(*args, outside, log=_quiet)
    assert not outside.parent.exists()


def test_fp32_dump_writes_only_under_work37(tiny, tmp_path):
    """A matching dump is read wherever it is (R1 under exp_036's registered key);
    a (re)computation outside work37_dir() is refused before anything is
    created or removed (DESIGN §2.9)."""
    import shutil

    seqs = tiny.ts.packed8() + [tiny.ts.t9]
    elsewhere = tmp_path / "exp036_work" / "ref" / "fp32"
    shutil.copytree(tiny.dump.dir, elsewhere)
    assert ref_pass.fp32_dump(tiny.bf16, seqs, elsewhere, tiny.ts.key, log=_quiet)["reused"] is True
    with pytest.raises(ValueError, match="exp_037 writers stay under"):
        ref_pass.fp32_dump(tiny.bf16, seqs, elsewhere, "another text key", log=_quiet)
    assert (elsewhere / "record.json").is_file()  # not removed
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError):
        ref_pass.fp32_dump(tiny.bf16, seqs, fresh, tiny.ts.key, log=_quiet)
    assert not fresh.exists()


def test_rules_judge_the_measured_dicts(tiny):
    """g2_layers.run's dict and the emulation statistics are what rules.py reads,
    in memory and after a strict-JSON round trip (the phase files)."""
    th2 = tiny.th["G2"]
    for res, emu in ((tiny.res, tiny.emu), (_strict_json(tiny.res), _strict_json(tiny.emu))):
        checks = rules.g2_checks(res, emu, th2)
        assert set(checks) == {"g2_fp32_forced_branch", "g2_fp32_natural_selection", "g2_embedding", "g2_head",
                               "g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_t9_buckets",
                               "g2_router_margin", "g2q_K8", "g2_real_weight_mutants"}
        for cid in ("g2_fp32_forced_branch", "g2_fp32_natural_selection", "g2_embedding", "g2_head",
                    "g2_real_weight_mutants"):
            assert checks[cid]["state"] == rules.PASS, (cid, checks[cid])
        assert checks["g2_real_weight_mutants"]["measured"]["undetected"] == []
        assert all(c["state"] in (rules.PASS, rules.FAIL) for c in checks.values())
        sel = checks["g2_bf16_natural_selection"]
        assert set(sel["legs"]) == {"a", "p"} and sel["measured"]["descriptive"]["flagged_pair"]["layer"] == 9

    muts = tiny.res["real_weight_mutants"]
    assert len(muts["mutants"]) == 12 and muts["layers"] == list(tiny.ts.profile.real_weight_mutant_layers)
    for m in muts["mutants"].values():
        assert isinstance(m["embedding_exact"], bool)
        for lay in m["layers"].values():
            assert set(lay) == {"attn", "moe", "selection_violations"}
            assert {"median", "p99", "max"} <= set(lay["attn"]) and {"median", "p99", "max"} <= set(lay["moe"])
    m10 = tiny.res["mutant10_natural_bf16"]
    assert m10["n"] == 8 * tiny.L * tiny.n_layers and len(m10["layers"]) == tiny.n_layers
    assert m10["M"] == max(r["max_gap_over_sigma"] for r in m10["layers"])
    assert m10["disagree_frac"] == m10["n_disagree"] / m10["n"]


def test_layer_csv_rows_take_rules_verdicts(tiny):
    res, emu = tiny.res, tiny.emu
    verdicts = rules.g2_row_verdicts(res, emu, tiny.th["G2"])
    rows = g2_layers.layer_csv_rows(res, emu, verdicts)
    n = {"fp32_forced": len(res["layers_fp32"]), "bf16_forced": len(res["layers_bf16"]),
         "t9_bf16_forced": len(res["layers_t9"]), "g2q_K8": len(res["layers_g2q"]["K8"]["layers"])}
    assert all(v > 0 for v in n.values())
    assert [r["mode"] for r in rows] == [m for m, k in n.items() for _ in range(k)]
    assert all(tuple(r) == g2_layers.CSV_COLUMNS for r in rows)
    flat = [v for mode in n for v in verdicts[mode]]
    assert [r["pass"] for r in rows] == [v["pass"] for v in flat]
    assert [r["limit_p99"] for r in rows] == [v["limit_p99"] for v in flat]
    t9 = [r for r in rows if r["mode"] == "t9_bf16_forced"]
    assert {r["bucket"] for r in t9} == {f"{a}-{b}" for a, b in tiny.ts.profile.t9_buckets}
    first = rows[n["fp32_forced"]]  # the first bf16 row carries its layer's emulation statistics
    assert first["emu_median"] == emu["layers"][first["layer"]][first["branch"]]["median"]

    short = dict(verdicts, bf16_forced=verdicts["bf16_forced"][:-1])
    with pytest.raises(ValueError, match="bf16_forced"):
        g2_layers.layer_csv_rows(res, emu, short)
    with pytest.raises(ValueError, match="g2q_K8"):
        g2_layers.layer_csv_rows(res, emu, {k: v for k, v in verdicts.items() if k != "g2q_K8"})
    # without emulation statistics the bf16 verdicts are None, never invented
    none = rules.g2_row_verdicts(res, None, tiny.th["G2"])
    rows0 = g2_layers.layer_csv_rows(res, None, none)
    assert all(r["pass"] is None and r["emu_median"] is None for r in rows0 if r["mode"] != "fp32_forced")


# ---------------------------------------------------------------------------
# Measurement only: no "pass" emitted
# ---------------------------------------------------------------------------


def test_no_pass_emitted_by_the_measurements(tiny):
    assert _decision_keys(tiny.res) == []
    assert _decision_keys(tiny.emu) == []
    assert _decision_keys(g3_oracle.ref_bpb(tiny.dump, tiny.ts)) == []
    rng = np.random.default_rng(3)
    base = rng.normal(2.0, 0.5, 200)
    out = g3_oracle.ref_mutant_nll({"ref": base, "worse": base + 0.5}, 8, 0.01, B=200)
    assert _decision_keys(out) == []


def test_measurement_modules_have_no_evaluate_and_do_not_import_rules():
    for name in OWNED:
        tree = ast.parse((CHECKS / name).read_text(encoding="utf-8"))
        defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert "evaluate" not in defs and "_bf16_rule" not in defs, name
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "gate.rules" and not (node.module == "gate" and
                                                             any(a.name == "rules" for a in node.names)), name
            if isinstance(node, ast.Import):
                assert all(a.name != "gate.rules" for a in node.names), name
    for mod in (g2_layers, g3_oracle):
        assert not hasattr(mod, "evaluate")


# ---------------------------------------------------------------------------
# G3: the bootstrap seeds (exp037 prefix, via common)
# ---------------------------------------------------------------------------


def test_g3_bootstrap_seeds_use_the_exp037_prefix():
    def rule(prefix, *parts):
        return int.from_bytes(hashlib.sha256((prefix + "|" + "|".join(parts)).encode()).digest()[:8], "big")

    rng = np.random.default_rng(11)
    base = rng.normal(2.0, 0.5, 1535)
    nll = {"ref": base, "attn_no_scale": base + rng.normal(0.01, 0.3, 1535),
           "rope_off": base + rng.normal(0.02, 0.3, 1535)}
    out = g3_oracle.ref_mutant_nll(nll, 48, 0.01, B=2000)
    for name in ("attn_no_scale", "rope_off"):
        seed = rule("exp037", "G3", name)
        assert g3_oracle.bootstrap_seed(name) == common.seed_from("G3", name) == seed != rule("exp036", "G3", name)
        assert out[name]["seed"] == seed
        s = g3_oracle.block_bootstrap(nll[name] - base, 48, 2000, seed, 0.01)
        assert (out[name]["dnll_p01"], out[name]["dnll_p99"], out[name]["dnll_mean"]) == (s["p01"], s["p99"], s["mean"])
        s36 = g3_oracle.block_bootstrap(nll[name] - base, 48, 2000, rule("exp036", "G3", name), 0.01)
        assert (s36["p01"], s36["p99"]) != (s["p01"], s["p99"])
        assert out[name]["blocks"] == 48 and out[name]["B"] == 2000 and out[name]["quantile"] == 0.01
    # rules.py classifies the measured quantiles (exp_036 g3_oracle.py l.95-100, moved)
    th3 = dict(common.load_thresholds()["G3"], ref_mutants=2)
    c = rules.g3_checks(None, None, out, th3)["g3_ref_mutants"]
    assert c["state"] in (rules.PASS, rules.FAIL)
    assert set(c["measured"]["mutants"]) == {"attn_no_scale", "rope_off"}
