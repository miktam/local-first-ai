# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W3).
"""The reference drivers on the tiny real-layout checkpoint
(gate/checks/ref_drivers.py; DESIGN §3.0, §2.9, §3.1 P5a, §3.3 G1 item 4).

* R2F-tiny (the reference on the dequantised K8 weights) forced onto its own
  natural ids reproduces its natural run bitwise: logits, shadow and gap; the
  same for R3-tiny (bf16 emulation). This is P5a's logic;
* P5a on an R1 dump of the tiny BF16 checkpoint: layers 0 and 4 bitwise, and a
  one-ulp change in the dump is caught;
* forced dumps are keyed: the key changes with each of its five components, a
  complete dump under the same key is reused, anything else is rebuilt;
* every writer stays under $EXP036_WORK/exp037 (work37_dir());
* r1_rows_pass gives R1's logits at the wanted rows bitwise, without the
  [N, V] matrix of the whole pack.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from gate import common
from gate.checks import ref_drivers as rd
from gate.checks import ref_pass
from gate.harness import HarnessError, bitwise_equal
from reference import kolibri_ref as kr

LENGTHS = (600, 300)  # the first crosses the 513 window
K = 6


@pytest.fixture(scope="module")
def work37(tmp_path_factory):
    """EXP036_WORK -> a temporary directory for this module; yields work37_dir()."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("EXP036_WORK", str(tmp_path_factory.mktemp("work")))
        yield common.work37_dir()


@pytest.fixture(scope="module")
def pack():
    rng = np.random.default_rng(6)
    seqs = [rng.integers(0, 1008, n) for n in LENGTHS]
    return seqs, np.concatenate(seqs), ref_pass.segments_of(LENGTHS)


def _natural(bf16, k8, seqs, mode):
    ref = kr.KolibriReference(str(bf16), dequant_dir=str(k8), emulate_bf16=mode == "emu")
    top6, biased = [], []

    def on_layer(i, h_in, h_out, info):
        top6.append(np.array(info["top6"]))
        biased.append(np.array(info["biased"]))

    logits, _, _, _ = ref.forward_packed(seqs, on_layer=on_layer)
    return logits, np.stack(top6), biased


@pytest.fixture(scope="module")
def r1_dump(work37, tiny_real_cal, pack):
    seqs = pack[0]
    d = work37 / "ref" / "tiny" / "fp32"
    ref_pass.fp32_dump(tiny_real_cal.bf16, seqs, d, "tiny-text-key", log=lambda m: None)
    return d


# --- R2F / R3 forced onto their own natural ids --------------------------------


@pytest.mark.parametrize("mode", ["fp32", "emu"])
def test_forced_on_its_own_natural_ids_is_bitwise_natural(work37, tiny_real_cal, pack, mode):
    seqs, ids, segs = pack
    logits, top6, biased = _natural(tiny_real_cal.bf16, tiny_real_cal.k8, seqs, mode)
    I = np.sort(top6, axis=-1)
    key = rd.derived_key(tiny_real_cal.k8, ids, segs, I, mode)
    out = rd.forced_dump_dir(key)
    assert out.parent.parent == work37 / "ref_forced" and out.name == mode
    rec = rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, mode, out, log=lambda m: None)
    fd = rd.ForcedDump(out)
    assert fd.complete and rec["key"] == key and rec["mode"] == mode
    assert rec["check"] == ("R2F" if mode == "fp32" else "R3")
    assert rec["num_tokens"] == ids.size and rec["seq_lengths"] == list(LENGTHS)
    assert bitwise_equal(np.array(fd.logits()), logits)
    for i in range(len(top6)):
        assert np.array_equal(np.array(fd.shadow(i)), top6[i])  # same order as route_ref's
        assert bitwise_equal(np.array(fd.gap(i)), ref_pass.top_gap(biased[i], K))
    assert np.array_equal(np.load(out / "ids.npy"), ids)


def test_r2f_follows_the_forcing(work37, tiny_real_cal, pack):
    """Forced onto other ids (each position's natural ids of t - 1), R2F moves:
    the reference really takes I."""
    seqs, ids, segs = pack
    logits, top6, _ = _natural(tiny_real_cal.bf16, tiny_real_cal.k8, seqs, "fp32")
    I = np.sort(top6, axis=-1)
    I[:, 1:] = I[:, :-1].copy()
    out = rd.forced_dump_dir(rd.derived_key(tiny_real_cal.k8, ids, segs, I, "fp32"))
    rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", out, log=lambda m: None)
    kl = common.kl_rows(logits, np.array(rd.ForcedDump(out).logits()))
    assert kl.mean() >= 0.10
    with pytest.raises(HarnessError):
        rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I[:-1], "fp32",
                           work37 / "ref_forced" / "short" / "fp32", key=rd.derived_key(
                               tiny_real_cal.k8, ids, segs, I[:-1], "fp32"), log=lambda m: None)


# --- keys and reuse ------------------------------------------------------------


def _base_key(mode="fp32", **kw):
    parts = dict(reference_tree_sha256="a" * 64, k8_shards=[("model-00001.safetensors", "b" * 64),
                                                           ("model-00002.safetensors", "c" * 64)],
                 text_key="d" * 64, i_sha256="e" * 64, mode=mode)
    parts.update(kw)
    return rd.forced_dump_key(**parts)


def test_the_key_changes_with_each_component():
    base = _base_key()
    variants = [
        _base_key(reference_tree_sha256="f" * 64),
        _base_key(k8_shards=[("model-00001.safetensors", "b" * 64), ("model-00002.safetensors", "0" * 64)]),
        _base_key(text_key="0" * 64),
        _base_key(i_sha256="0" * 64),
        _base_key(mode="emu"),
    ]
    shas = {base["sha256"]} | {v["sha256"] for v in variants}
    assert len(shas) == 6
    assert len({rd.forced_dump_dir(k, work=common.work37_dir()) for k in [base] + variants}) == 6
    # The shard list is a set of (name, sha) pairs: order and container do not matter.
    again = _base_key(k8_shards={"model-00002.safetensors": "c" * 64, "model-00001.safetensors": "b" * 64})
    assert again == base
    with pytest.raises(ValueError):
        _base_key(mode="bf16")


def test_i_from_r1_and_its_sha(work37, r1_dump, tmp_path):
    I, sha = rd.i_from_r1(r1_dump)
    rec = common.read_json(r1_dump / "record.json")
    L = rec["num_layers"]
    assert I.shape == (L, sum(LENGTHS), K) and I.dtype == np.int64
    for i in range(L):
        assert np.array_equal(I[i], np.sort(np.load(r1_dump / f"layer{i:02d}.top6.npy"), axis=-1))
    lines = "".join(f"layer{i:02d}.top6.npy\t{common.sha256_file(r1_dump / f'layer{i:02d}.top6.npy')}\n"
                    for i in range(L))
    assert sha == common.sha256_bytes(lines.encode())
    # A changed top6 file: refused against the record, and a new sha without one.
    copy = tmp_path / "r1copy"
    shutil.copytree(r1_dump, copy)
    t = np.load(copy / "layer03.top6.npy")
    t[0] = t[0][::-1]
    np.save(copy / "layer03.top6.npy", t)
    with pytest.raises(HarnessError, match="layer03.top6.npy"):
        rd.i_from_r1(copy)
    os.remove(copy / "record.json")
    I2, sha2 = rd.i_from_r1(copy)
    assert sha2 != sha and np.array_equal(I2, I)  # same set per row, other file bytes


def test_a_forced_dump_is_reused_only_under_its_key(work37, tiny_real_cal, pack, monkeypatch):
    seqs, ids, segs = pack
    I = np.sort(_natural(tiny_real_cal.bf16, tiny_real_cal.k8, seqs, "fp32")[1], axis=-1)
    key = rd.forced_dump_key(reference_tree_sha256=common.reference_tree_sha256(),
                             k8_shards=rd.build_shards(tiny_real_cal.k8), text_key="reuse-test",
                             i_sha256="reuse-test", mode="fp32")
    out = rd.forced_dump_dir(key)
    first = rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", out, key=key,
                               log=lambda m: None)
    assert not first.get("reused")
    stamp = (out / "logits.npy").stat().st_mtime_ns

    class NoRef:
        def __init__(self, *a, **kw):
            raise AssertionError("the reference ran although the dump is reusable")

    with monkeypatch.context() as mp:
        mp.setattr(kr, "KolibriReference", NoRef)
        again = rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", out, key=key,
                                   log=lambda m: None)
        assert again["reused"] and again["key"] == key
        with pytest.raises(ValueError, match="mode"):  # a key for the other mode
            rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "emu", out, key=key)
    assert (out / "logits.npy").stat().st_mtime_ns == stamp
    # Another key in the same directory, or an incomplete directory: rebuilt.
    other = dict(key, sha256="0" * 64)
    rebuilt = rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", out, key=other,
                                 log=lambda m: None)
    assert not rebuilt.get("reused") and rebuilt["key"] == other
    os.remove(out / "record.json")
    again = rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", out, key=other,
                               log=lambda m: None)
    assert not again.get("reused") and (out / "record.json").is_file()


# --- writer paths --------------------------------------------------------------


def test_writers_stay_under_exp037(work37, tiny_real_cal, pack, tmp_path):
    seqs, ids, segs = pack
    I = np.zeros((10, ids.size, K), dtype=np.int64) + np.arange(K)
    exp036_area = common.work_dir() / "ref" / "x" / "fp32"
    for bad in (exp036_area, tmp_path / "elsewhere", work37, work37 / ".." / "ref_forced" / "x"):
        with pytest.raises(ValueError, match="exp_037 writers stay under"):
            rd.forced_ref_pass(tiny_real_cal.bf16, tiny_real_cal.k8, ids, segs, I, "fp32", bad,
                               key=_base_key(), log=lambda m: None)
        with pytest.raises(ValueError, match="exp_037 writers stay under"):
            rd.r1_rows_pass(tiny_real_cal.bf16, seqs, [np.array([0]), np.array([0])], out_dir=bad)
    assert not exp036_area.exists() and not (tmp_path / "elsewhere").exists()
    assert rd.require_under_work37(work37 / "ref_forced" / "k" / "fp32") == (work37 / "ref_forced" / "k" / "fp32").resolve()
    assert common.work37_dir().name == "exp037" and common.work37_dir().parent == common.work_dir()
    assert rd.forced_dump_dir(_base_key()).is_relative_to(work37)


# --- P5a -----------------------------------------------------------------------


def test_p5a_reproduces_the_dump_bitwise(tiny_real_cal, r1_dump):
    res = rd.p5a(tiny_real_cal.bf16, r1_dump, log=lambda m: None)
    assert res["ok"], res
    layers = res["values"]["layers"]
    assert set(layers) == {"layer00", "layer04"}
    assert layers["layer00"]["layer_type"] == "sliding_attention" and layers["layer04"]["layer_type"] == "full_attention"
    assert all(v["bitwise_equal"] and v["n_rows"] == sum(LENGTHS) for v in layers.values())
    last = rd.p5a(tiny_real_cal.bf16, r1_dump, layers=(9,), log=lambda m: None)  # against h_last.npy
    assert last["ok"], last


def test_p5a_catches_one_ulp(tiny_real_cal, r1_dump, tmp_path):
    copy = tmp_path / "r1ulp"
    shutil.copytree(r1_dump, copy)
    h = np.load(copy / "layer01.h_in.npy")
    h.view(np.uint32)[123, 7] += 1
    np.save(copy / "layer01.h_in.npy", h)
    res = rd.p5a(tiny_real_cal.bf16, copy, log=lambda m: None)
    assert not res["ok"] and "layer00" in res["reason"]
    assert res["values"]["layers"]["layer00"]["rows_differing"] == 1
    assert res["values"]["layers"]["layer04"]["bitwise_equal"]


# --- R1 at wanted rows ---------------------------------------------------------


def test_r1_rows_pass_is_r1_at_the_wanted_rows(work37, tiny_real_cal, pack, r1_dump):
    seqs, ids, _ = pack
    full = np.load(r1_dump / "logits.npy")
    want = [np.array([599, 0, 5]), np.array([7])]
    out_dir = work37 / "run" / "test" / "anchor"
    res = rd.r1_rows_pass(tiny_real_cal.bf16, seqs, want, out_dir=out_dir, log=lambda m: None)
    assert np.array_equal(res["rows"], [599, 0, 5, 607])
    assert np.array_equal(res["seq_index"], [0, 0, 0, 1]) and np.array_equal(res["positions"], [599, 0, 5, 7])
    assert bitwise_equal(res["logits"], full[res["rows"]])
    assert res["record"]["num_rows"] == 4 and res["record"]["pack_ids_sha256"] == common.ids_sha256(ids)
    assert bitwise_equal(np.load(out_dir / "logits.npy"), res["logits"])
    assert common.read_json(out_dir / "record.json")["files"]["logits.npy"]["shape"] == [4, full.shape[1]]
    one = rd.r1_rows_pass(tiny_real_cal.bf16, seqs, np.array([650]), log=lambda m: None)  # packed index, one row
    assert bitwise_equal(one["logits"], full[[650]]) and one["seq_index"].tolist() == [1]
    assert one["positions"].tolist() == [50]
    with pytest.raises(ValueError):
        rd.r1_rows_pass(tiny_real_cal.bf16, seqs, [np.array([600]), np.array([0])])
    with pytest.raises(ValueError):
        rd.r1_rows_pass(tiny_real_cal.bf16, seqs, [np.array([0])])
    with pytest.raises(ValueError):
        rd.r1_rows_pass(tiny_real_cal.bf16, seqs, np.array([900]))
