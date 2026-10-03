# SPDX-License-Identifier: MIT
"""The reference's gate-facing outputs (review 2026-10-03; BUILD_SPEC §5.2).

* moe_block's info carries logits, biased, top6 and weights;
* `--dump DIR` writes per-layer h_in, r_attn, h_mid, r_moe, h_out,
  router_logits, biased, top6 and the top-6 near-tie gap, the final logits
  and log-probs, and ref_record.json with the sha256 of every file, the
  checkpoint fingerprint, the reference tree sha, the gate-text sha, the mode
  and t_start / t_end (BUILD_SPEC 5.1b item 2); the files agree with the
  same run's .npz and with the checkpoint;
* `--emulate-bf16` dumps the bf16-emulation mode; `--mutants` carries the G3
  mutants through the same pass and writes their next-token NLL;
* `--out` without .npz gets the suffix np.savez would add, and the printed
  path is the file actually written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling

PRESETS = ["vendor", "pattern5"]


@pytest.fixture(params=PRESETS)
def preset_dir(request, tiny_vendor_dir, tiny_pattern5_dir):
    return tiny_vendor_dir if request.param == "vendor" else tiny_pattern5_dir


def _ids_file(tmp_path: Path, seqs) -> Path:
    path = tmp_path / "ids.json"
    path.write_text(json.dumps([list(map(int, s)) for s in seqs]))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_moe_block_info_has_spec_keys(tiny_pattern5_dir):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ref = kolibri_ref.KolibriReference(str(tiny_pattern5_dir))
    x = np.random.default_rng(0).standard_normal((12, ref.cfg.hidden_size)).astype(np.float32)
    _, info = ref.moe_block(3, x)
    assert {"logits", "biased", "top6", "weights"} <= set(info)

    bias = ref.ckpt.get("model.layers.3.moe.router.expert_bias")
    np.testing.assert_array_equal(info["biased"], info["logits"] + bias[None, :])
    w, ids = kolibri_ref.route_ref(info["logits"], bias, ref.cfg.num_experts_per_tok)
    np.testing.assert_array_equal(info["top6"], ids)
    np.testing.assert_array_equal(info["ids"], ids)
    np.testing.assert_array_equal(info["weights"], w)
    assert info["logits"].dtype == np.float32 and info["biased"].dtype == np.float32


def test_out_without_npz_suffix_names_the_written_file(tiny_vendor_dir, tmp_path, capsys):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(8, seed=1)])
    kolibri_ref.main(["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids),
                      "--out", str(tmp_path / "out_noext"), "--quiet"])
    printed = capsys.readouterr().out
    assert (tmp_path / "out_noext.npz").is_file()
    assert not (tmp_path / "out_noext").exists()
    assert str(tmp_path / "out_noext.npz") in printed


def test_cli_needs_out_or_dump(tiny_vendor_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(8, seed=1)])
    with pytest.raises(SystemExit):
        kolibri_ref.main(["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids), "--quiet"])


def test_dump_refuses_a_non_empty_dir(tiny_vendor_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(8, seed=1)])
    dump = tmp_path / "dump"
    dump.mkdir()
    (dump / "old.npy").write_bytes(b"")
    with pytest.raises(SystemExit, match="not an empty directory"):
        kolibri_ref.main(["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids),
                          "--dump", str(dump), "--quiet"])


def test_dump_matches_the_run_and_the_checkpoint(preset_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ref = kolibri_ref.KolibriReference(str(preset_dir))
    cfg = ref.cfg
    k, L = cfg.num_experts_per_tok, cfg.num_hidden_layers
    seqs = [tc.random_ids(90, seed=21), tc.random_ids(37, seed=22)]
    ids_path = _ids_file(tmp_path, seqs)
    dump = tmp_path / "ref" / "gate_text_sha"
    npz_path = tmp_path / "run.npz"
    assert kolibri_ref.main(["--model-dir", str(preset_dir), "--ids-file", str(ids_path),
                             "--out", str(npz_path), "--hidden", "--dump", str(dump), "--quiet"]) == 0
    run = np.load(npz_path)

    # The record: every file listed with its sha256 and size, sorted keys, run identity.
    text = (dump / "ref_record.json").read_text()
    record = json.loads(text)
    assert text == json.dumps(record, indent=2, sort_keys=True) + "\n"
    on_disk = {p.name for p in dump.iterdir() if p.name != "ref_record.json"}
    assert set(record["files"]) == on_disk
    for name, entry in record["files"].items():
        assert entry["sha256"] == _sha256(dump / name), name
        assert entry["bytes"] == (dump / name).stat().st_size, name
    assert record["ref_version"] == kolibri_ref.REF_VERSION
    assert record["checkpoint_fingerprint"] == str(run["checkpoint_fingerprint"])
    assert record["num_layers"] == L and record["num_tokens"] == 127
    assert record["seq_lengths"] == [90, 37]
    assert record["peak_rss_bytes"] > 0 and record["wall_seconds"] > 0
    assert record["mode"] == "fp32" and record["dequantised_from"] == "" and record["streams"] == []
    assert record["reference_tree_sha256"] == kolibri_ref.reference_tree_sha256()
    assert record["ids_file_sha256"] == _sha256(ids_path) == record["gate_text_sha256"]
    ids_all = np.concatenate([np.asarray(s, dtype="<i8") for s in seqs])
    assert record["ids_sha256"] == hashlib.sha256(ids_all.tobytes()).hexdigest()
    assert record["t_start"] <= record["t_end"] and record["t_end"].endswith("Z")

    def load(name):
        return np.load(dump / name)

    np.testing.assert_array_equal(load("logits.npy"), run["logits"])
    np.testing.assert_array_equal(load("final_norm.npy"), run["final_norm"])
    np.testing.assert_array_equal(load("ids.npy"), run["ids"])
    np.testing.assert_array_equal(load("seq_lengths.npy"), run["seq_lengths"])
    logprobs = load("logprobs.npy")
    assert logprobs.dtype == np.float32
    exact = ph.log_softmax(run["logits"])  # float64
    assert np.abs(logprobs - exact).max() <= 1e-6 * max(1.0, np.abs(exact).max())

    embed = tc.load_checkpoint_f32(preset_dir)["model.embed_tokens.weight"]
    np.testing.assert_array_equal(load("layer00.h_in.npy"), embed[run["ids"]])
    for i in range(L):
        p = f"layer{i:02d}"
        h_in, h_out = load(f"{p}.h_in.npy"), load(f"{p}.h_out.npy")
        np.testing.assert_array_equal(h_out, run["hidden"][i])
        if i > 0:
            np.testing.assert_array_equal(h_in, run["hidden"][i - 1])
        # The branches (spec item 29) and the stream between them, exactly.
        r_attn, h_mid, r_moe = load(f"{p}.r_attn.npy"), load(f"{p}.h_mid.npy"), load(f"{p}.r_moe.npy")
        np.testing.assert_array_equal(h_mid, h_in + r_attn)
        np.testing.assert_array_equal(h_out, h_mid + r_moe)
        logits = load(f"{p}.router_logits.npy")
        top6 = load(f"{p}.top6.npy")
        gap = load(f"{p}.top6_gap.npy")
        assert logits.dtype == np.float32 and logits.shape == (127, cfg.num_experts)
        assert top6.shape == (127, k) and gap.shape == (127,)
        np.testing.assert_array_equal(top6, run["router_ids"][i])
        # The dumped logits reproduce the run's routing and weights exactly.
        bias = ref.ckpt.get(f"model.layers.{i}.moe.router.expert_bias")
        np.testing.assert_array_equal(load(f"{p}.biased.npy"), logits + bias[None, :])
        w, ids = kolibri_ref.route_ref(logits, bias, k)
        np.testing.assert_array_equal(ids, top6)
        np.testing.assert_array_equal(w, run["router_weights"][i])
        # The near-tie margin G2 exempts on: k-th minus (k+1)-th biased score.
        margin = ph.selection_margin(logits + bias[None, :], k)
        np.testing.assert_allclose(gap, margin, rtol=1e-6, atol=0)
        assert (gap >= 0).all()


def test_dump_with_num_layers_writes_only_those_layers(tiny_vendor_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(20, seed=4)])
    dump = tmp_path / "dump"
    kolibri_ref.main(["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids),
                      "--dump", str(dump), "--num-layers", "2", "--quiet"])
    layers = sorted({p.name.split(".")[0] for p in dump.glob("layer*.npy")})
    assert layers == ["layer00", "layer01"]
    assert json.loads((dump / "ref_record.json").read_text())["num_layers"] == 2


def test_dump_with_the_gate_text_sha(tiny_vendor_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(20, seed=4)])
    dump = tmp_path / "dump"
    kolibri_ref.main(["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids), "--dump", str(dump),
                      "--gate-text-sha", "ab" * 32, "--quiet"])
    record = json.loads((dump / "ref_record.json").read_text())
    assert record["gate_text_sha256"] == "ab" * 32 and record["ids_file_sha256"] == _sha256(ids)


def test_dump_in_bf16_emulation(tiny_pattern5_dir, tmp_path):
    """--emulate-bf16: the branch outputs are bf16 values (vLLM stores them in
    bf16); the stream between layers is the unrounded fp32 sum that vLLM's
    fused add + norm normalises, built from the stored (rounded) residual;
    router logits and the head stay fp32 (BUILD_SPEC 5.1b item 3)."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    R = kolibri_ref.round_bf16
    ids = _ids_file(tmp_path, [tc.random_ids(60, seed=8)])
    dump = tmp_path / "emu"
    kolibri_ref.main(["--model-dir", str(tiny_pattern5_dir), "--ids-file", str(ids), "--dump", str(dump),
                      "--emulate-bf16", "--quiet"])
    record = json.loads((dump / "ref_record.json").read_text())
    assert record["mode"] == "bf16_emulation"

    def load(name):
        return np.load(dump / name)

    def bf16_frac(x):
        return float(np.mean(R(x) == x))

    for i in range(tc.preset_config("pattern5")["num_hidden_layers"]):
        p = f"layer{i:02d}"
        h_in, r_attn, h_mid = load(f"{p}.h_in.npy"), load(f"{p}.r_attn.npy"), load(f"{p}.h_mid.npy")
        r_moe, h_out = load(f"{p}.r_moe.npy"), load(f"{p}.h_out.npy")
        assert bf16_frac(r_attn) == 1.0 and bf16_frac(r_moe) == 1.0
        np.testing.assert_array_equal(h_mid, R(h_in) + r_attn)
        np.testing.assert_array_equal(h_out, R(h_mid) + r_moe)
        assert bf16_frac(load(f"{p}.router_logits.npy")) < 0.05  # fp32 router logits (item 11)
    assert bf16_frac(load("final_norm.npy")) == 1.0
    assert bf16_frac(load("logits.npy")) < 0.05  # fp32 head (item 14)


def test_cli_mutants_ride_along_in_one_pass(tiny_vendor_dir, tmp_path):
    """--mutants: the reference's own dump is unchanged, and each mutant's
    next-token NLL equals the one from its own separate forward."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    mutants = import_sibling("reference.mutants")
    seqs = [tc.random_ids(40, seed=31), tc.random_ids(25, seed=32)]
    ids_path = _ids_file(tmp_path, seqs)
    plain, streamed = tmp_path / "plain", tmp_path / "streamed"
    base = ["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids_path), "--quiet"]
    kolibri_ref.main(base + ["--dump", str(plain)])
    kolibri_ref.main(base + ["--dump", str(streamed), "--mutants", "all"])
    rec_plain = json.loads((plain / "ref_record.json").read_text())
    rec = json.loads((streamed / "ref_record.json").read_text())
    assert rec["streams"] == ["ref"] + sorted(mutants.MUTANTS)  # "all" runs them in name order
    for name, entry in rec_plain["files"].items():
        assert rec["files"][name]["sha256"] == entry["sha256"], name
    ids = np.concatenate([np.asarray(s) for s in seqs])
    lengths = [len(s) for s in seqs]
    ref = kolibri_ref.KolibriReference(str(tiny_vendor_dir))
    nll_ref = np.load(streamed / "stream_ref.nll.npy")
    np.testing.assert_array_equal(nll_ref, kolibri_ref.next_token_nll(np.load(plain / "logits.npy"), ids, lengths))
    assert np.isnan(nll_ref[[39, 64]]).all() and np.isfinite(np.delete(nll_ref, [39, 64])).all()
    for name, cls in mutants.MUTANTS.items():
        logits = ref._clone_as(cls).forward_packed(seqs)[0]
        np.testing.assert_array_equal(np.load(streamed / f"stream_{name}.nll.npy"),
                                      kolibri_ref.next_token_nll(logits, ids, lengths), err_msg=name)


def test_cli_mutants_needs_dump_and_known_names(tiny_vendor_dir, tmp_path):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = _ids_file(tmp_path, [tc.random_ids(8, seed=1)])
    base = ["--model-dir", str(tiny_vendor_dir), "--ids-file", str(ids), "--quiet"]
    import contextlib
    import io

    for extra in (["--out", str(tmp_path / "x"), "--mutants", "all"],
                  ["--dump", str(tmp_path / "d1"), "--mutants", "no_such_mutant"],
                  ["--dump", str(tmp_path / "d2"), "--mutants", "all", "--hidden"]):
        with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            kolibri_ref.main(base + extra)
