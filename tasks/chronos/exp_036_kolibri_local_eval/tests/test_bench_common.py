# SPDX-License-Identifier: MIT
"""bench/common.py and bench/genutil.py: paths, fixtures, result files,
thermals, EOS masking and the timed / batched generation helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    EOS_ID,
    PAD_ID,
    assert_sorted_keys_jsonl,
    bench_env,
    eos_first_model_dir,
    read_jsonl,
    tiny_models,
)

from bench import common


# --------------------------------------------------------------------------
# Paths and fixtures
# --------------------------------------------------------------------------


def test_env_dirs_follow_build_spec_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    monkeypatch.delenv("EXP036_DATA", raising=False)
    monkeypatch.delenv("EXP036_WORK", raising=False)
    assert common.data_dir() == tmp_path / "m" / "data"
    assert common.work_dir() == tmp_path / "m" / "work"
    assert common.fineweb_parquet() == tmp_path / "m" / "data/fineweb-2/data/deu_Latn/test/000_00000.parquet"
    monkeypatch.setenv("EXP036_DATA", str(tmp_path / "d"))
    assert common.data_dir() == tmp_path / "d"


def test_offline_flags_are_set_on_import():
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("name", ["pad_4k.txt", "pad_15k.txt", "pad_120k.txt"])
def test_padding_fixtures_exist_by_real_name(name):
    text, sha = common.read_fixture(name)
    assert len(text) > 20_000 and len(sha) == 64


def test_missing_fixture_is_a_hard_failure():
    with pytest.raises(common.BenchError, match="pad_5k.txt is missing"):
        common.fixture_path("pad_5k.txt")


def test_pad_4k_and_pad_15k_are_prefixes_of_pad_120k():
    """speed.py and ladder.py rely on this when they take 4,096-token and
    batch prompts from pad_120k.txt."""
    big = common.read_fixture("pad_120k.txt")[0]
    assert big.startswith(common.read_fixture("pad_4k.txt")[0])
    assert big.startswith(common.read_fixture("pad_15k.txt")[0])


def test_arm_table_matches_assets_json():
    assets = {m["path"] for m in common.assets_json()["models"]}
    for arm in common.ARMS.values():
        if arm.family != "kolibri":
            assert arm.folder in assets, arm
    assert {a for pair in common.FAMILY_PAIRS.values() for a in pair} == set(common.ARMS)


# --------------------------------------------------------------------------
# Result files
# --------------------------------------------------------------------------


def test_jsonl_cell_header_records_end_sorted_and_exclusive(tmp_path):
    p = tmp_path / "bench" / "fit_20261003T120000Z.jsonl"
    with common.JsonlCell(p, {"cell": "fit", "z": 1, "a": {"y": 2, "b": 3}}) as w:
        w.append({"kind": "run", "x": 1.5})
        w.finish({"n": 1})
    recs = read_jsonl(p)
    assert [r["type"] for r in recs] == ["header", "record", "end"]
    assert recs[-1]["complete"] is True and recs[-1]["n_records"] == 1
    assert recs[0]["t_start"] == recs[-1]["t_start"] and recs[-1]["t_end"] >= recs[-1]["t_start"]
    assert_sorted_keys_jsonl(p)
    assert common.is_complete(p)
    with pytest.raises(FileExistsError):
        common.JsonlCell(p, {"cell": "fit"})


def test_jsonl_cell_failure_writes_incomplete_end_without_message(tmp_path):
    from pathlib import Path

    p = tmp_path / "c.jsonl"
    secret = str(Path.home() / "secret" / "path")  # an exception message may hold a home path
    with pytest.raises(ValueError):
        with common.JsonlCell(p, {"cell": "x"}) as w:
            w.append({"a": 1})
            raise ValueError(secret)
    end = read_jsonl(p)[-1]
    assert end == {**end, "type": "end", "complete": False, "error_type": "ValueError"}
    assert secret not in p.read_text()
    assert not common.is_complete(p)


def test_nan_is_refused(tmp_path):
    with pytest.raises(ValueError):
        common.dumps({"x": float("nan")})


def test_write_json_new_refuses_overwrite(tmp_path):
    p = tmp_path / "kl_8v4_20261003T120000Z.json"
    common.write_json_new(p, {"complete": True})
    assert common.is_complete(p)
    with pytest.raises(common.BenchError):
        common.write_json_new(p, {"complete": True})


def test_cell_outputs_keep_speed_and_speed_desc_apart(tmp_path):
    b = tmp_path / "bench"
    b.mkdir()
    for name in ("speed_20261003T120000Z.jsonl", "speed_desc_20261003T120001Z.jsonl", "speed_notes.jsonl"):
        (b / name).write_text("")
    assert [p.name for p in common.cell_outputs(tmp_path, "speed")] == ["speed_20261003T120000Z.jsonl"]
    assert [p.name for p in common.cell_outputs(tmp_path, "speed_desc")] == ["speed_desc_20261003T120001Z.jsonl"]
    (tmp_path / "tokenizer_20261003T120002Z.json").write_text('{"complete": true}')
    assert common.complete_output(tmp_path, "tokenizer").name == "tokenizer_20261003T120002Z.json"
    assert common.complete_output(tmp_path, "speed") is None


def test_quarantine_moves_incomplete_files_with_a_note(tmp_path):
    res = tmp_path / "results"
    p = res / "bench" / "fit_20261003T120000Z.jsonl"
    with pytest.raises(RuntimeError):
        with common.JsonlCell(p, {"cell": "fit"}):
            raise RuntimeError
    done = res / "bench" / "fit_20261003T130000Z.jsonl"
    with common.JsonlCell(done, {"cell": "fit"}) as w:
        w.finish({})
    moved = common.quarantine_incomplete(res, "fit")
    assert [m.name for m in moved] == [p.name] and not p.exists() and done.exists()
    note = (moved[0].parent / "NOTE.md").read_text()
    assert p.name in note and common.sha256_file(moved[0]) in note
    assert moved[0].parent.parent == tmp_path / "aborted"
    assert common.quarantine_incomplete(res, "fit") == []


# --------------------------------------------------------------------------
# Thermals and power (C17)
# --------------------------------------------------------------------------


def test_snapshot_parses_thermal_and_power(bench_env):
    s = common.snapshot()
    assert s["thermal_state"] == 0 and s["thermal_label"] == "nominal"
    assert s["power_source"] == "AC Power" and s["powermode"] == 2 and s["lowpowermode"] == 0


def test_wait_cool_idles_ten_minutes_then_requires_nominal(bench_env):
    bench_env.thermal = ["0", "2", "1", "0"]  # before, after idle, polls
    rec = common.wait_cool()
    assert bench_env.sleeps == [600, 60, 60]
    assert rec["after"]["thermal_label"] == "nominal" and rec["idle_s"] == 720


def test_wait_cool_refuses_when_never_nominal(bench_env):
    bench_env.thermal = ["2"]
    with pytest.raises(common.CouldNotRun, match="serious"):
        common.wait_cool()
    assert sum(bench_env.sleeps) == 600 + 1200


def test_wait_cool_refuses_when_unreadable(bench_env):
    bench_env.thermal = [None]
    with pytest.raises(common.CouldNotRun, match="could not be read"):
        common.wait_cool()


def test_environment_without_mlx_does_not_import_it(bench_env):
    import subprocess
    import sys

    code = (
        "import sys; sys.path.insert(0, %r); from bench import common; "
        "common.host_label = lambda: 'h'; common.environment(include_mlx=False); "
        "print('mlx' in sys.modules)" % str(common.EXP_DIR)
    )
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


# --------------------------------------------------------------------------
# genutil: EOS masking and generation
# --------------------------------------------------------------------------


def test_eos_mask_and_masked_greedy():
    import mlx.core as mx

    from bench import genutil

    logits = mx.array(np.array([[0.0, 5.0, 1.0, 9.0]], dtype=np.float32))
    out = np.array(genutil.EosMask([3, 1])(None, logits))
    assert np.isneginf(out[0, 1]) and np.isneginf(out[0, 3]) and out[0, 2] == 1.0
    assert int(genutil.masked_greedy([3, 1])(logits)[0]) == 2
    assert int(genutil.greedy(logits)[0]) == 3


def test_eos_ids_union(tiny_models):
    from mlx_lm import load

    from bench import genutil

    _, tok = load(str(tiny_models["a"]))
    assert genutil.eos_ids(tiny_models["a"], tok) == [PAD_ID, EOS_ID]


def test_timed_generate_masks_eos_and_counts_exactly(tmp_path):
    from mlx_lm import load

    from bench import genutil

    d = eos_first_model_dir(tmp_path / "eosfirst")
    model, tok = load(str(d))
    eos = genutil.eos_ids(d, tok)
    assert 0 in eos
    prompt = list(range(40, 80))
    unmasked = genutil.timed_generate(model, tok, prompt, 16, eos, mask_eos=False)
    assert unmasked["finish_reason"] == "stop" and unmasked["generation_tokens"] == 1
    masked = genutil.timed_generate(model, tok, prompt, 16, eos)
    assert masked["finish_reason"] == "length" and masked["generation_tokens"] == 16
    assert masked["completion_ids_sha256"] == common.sha256_ids([1] * 16)
    for k in ("prompt_tps", "generation_tps"):
        assert masked[k] > 0


def test_timed_generate_refuses_a_short_run(tmp_path, monkeypatch):
    from mlx_lm import load

    from bench import genutil

    d = eos_first_model_dir(tmp_path / "eosfirst2")
    model, tok = load(str(d))
    # Mask a different id only: EOS id 0 is still greedy's pick, so the run
    # stops after one token and the helper must refuse the record.
    with pytest.raises(common.BenchError, match="expected exactly 8"):
        genutil.timed_generate(model, tok, [5, 6, 7], 8, [999])


def test_fixture_ids_hard_fails_when_too_short(tiny_models):
    from mlx_lm import load

    from bench import genutil

    _, tok = load(str(tiny_models["a"]))
    ids, info = genutil.fixture_ids(tok, "pad_4k.txt", 1024)
    assert len(ids) == 1024 and info["fixture_tokens"] == len(common.read_fixture("pad_4k.txt")[0].encode())
    assert info["ids_sha256"] == common.sha256_ids(ids)
    with pytest.raises(common.BenchError, match="tokens in this tokenizer"):
        genutil.fixture_ids(tok, "pad_4k.txt", 10**7)


def test_batch_aggregate_exact_counts_distinct_rows(tiny_models):
    from mlx_lm import load

    from bench import genutil

    model, tok = load(str(tiny_models["a"]))
    rows = [list(range(10 + i, 74 + i)) for i in range(4)]
    for B in (1, 2, 4):
        g = genutil.batch_aggregate(model, rows[:B], 24, [PAD_ID, EOS_ID], B)
        assert g["B"] == B and g["generation_tokens"] == 24 * B and g["generation_tps"] > 0
        assert len(set(g["prompt_ids_sha256"])) == B
    with pytest.raises(common.BenchError):
        genutil.batch_aggregate(model, rows[:2], 8, [EOS_ID], 4)


def test_batch_completions_in_order_and_stop_token_last(tmp_path):
    from mlx_lm import load

    from bench import genutil

    d = eos_first_model_dir(tmp_path / "eosfirst3")
    model, _ = load(str(d))
    outs = genutil.batch_completions(model, [[5, 6, 7], [8, 9], [10, 11, 12, 13]], 2, 8, [0])
    assert [o["completion_ids"] for o in outs] == [[0], [0], [0]]
    assert all(o["finish_reason"] == "stop" for o in outs)


def test_load_arm_checks_the_port_file_for_kolibri(monkeypatch, tmp_path, bench_env):
    import port.convert

    from bench import genutil

    checked = []
    monkeypatch.setattr(port.convert, "check_port_file", lambda d: checked.append(d))
    import mlx_lm

    monkeypatch.setattr(mlx_lm, "load", lambda p: ("model", "tok"))
    d = tmp_path / "k"
    d.mkdir()
    assert genutil.load_arm("K4", d) == ("model", "tok", d) and checked == [d]
    assert genutil.load_arm("G4", d) == ("model", "tok", d) and checked == [d]
    with pytest.raises(common.BenchError, match="missing"):
        genutil.load_arm("K8", tmp_path / "absent")


def test_model_fingerprint_reads_the_convert_record(tmp_path):
    d = tmp_path / "k8"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "exp036_convert_record.json").write_text(json.dumps({"manifest_sha256": "ab" * 32, "port_sha256": "cd" * 32}))
    (d / "model.safetensors").write_bytes(b"\0" * 10)
    fp = common.model_fingerprint("K8", d)
    assert fp["manifest_sha256"] == "ab" * 32 and fp["port_sha256"] == "cd" * 32 and fp["weights_bytes"] == 10
    g = common.model_fingerprint("G8", d)
    assert g["assets_revision"] == "33c6d23798a0af159529890f79329206dbfbd73c"
