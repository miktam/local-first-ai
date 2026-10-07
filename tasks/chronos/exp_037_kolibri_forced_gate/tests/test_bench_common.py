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

    _, tok = load(str(tiny_models["a"]), trust_remote_code=True)
    assert genutil.eos_ids(tiny_models["a"], tok) == [PAD_ID, EOS_ID]


def test_timed_generate_masks_eos_and_counts_exactly(tmp_path):
    from mlx_lm import load

    from bench import genutil

    d = eos_first_model_dir(tmp_path / "eosfirst")
    model, tok = load(str(d), trust_remote_code=True)
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
    model, tok = load(str(d), trust_remote_code=True)
    # Mask a different id only: EOS id 0 is still greedy's pick, so the run
    # stops after one token and the helper must refuse the record.
    with pytest.raises(common.BenchError, match="expected exactly 8"):
        genutil.timed_generate(model, tok, [5, 6, 7], 8, [999])


def test_fixture_ids_hard_fails_when_too_short(tiny_models):
    from mlx_lm import load

    from bench import genutil

    _, tok = load(str(tiny_models["a"]), trust_remote_code=True)
    ids, info = genutil.fixture_ids(tok, "pad_4k.txt", 1024)
    assert len(ids) == 1024 and info["fixture_tokens"] == len(common.read_fixture("pad_4k.txt")[0].encode())
    assert info["ids_sha256"] == common.sha256_ids(ids)
    with pytest.raises(common.BenchError, match="tokens in this tokenizer"):
        genutil.fixture_ids(tok, "pad_4k.txt", 10**7)


def test_batch_aggregate_exact_counts_distinct_rows(tiny_models):
    from mlx_lm import load

    from bench import genutil

    model, tok = load(str(tiny_models["a"]), trust_remote_code=True)
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
    model, _ = load(str(d), trust_remote_code=True)
    outs = genutil.batch_completions(model, [[5, 6, 7], [8, 9], [10, 11, 12, 13]], 2, 8, [0])
    assert [o["completion_ids"] for o in outs] == [[0], [0], [0]]
    assert all(o["finish_reason"] == "stop" for o in outs)


def test_load_arm_checks_the_port_file_for_kolibri(monkeypatch, tmp_path, bench_env):
    """Kolibri: check_port_file, then the load with **model_file_trust(d) (mlx-lm 0.32.0, #1385), whose own
    check_port_file precedes the flag. A peer (no model_file) loads without the flag; a foreign model_file is
    refused before mlx_lm is called."""
    import port.convert

    from bench import genutil

    checked = []
    monkeypatch.setattr(port.convert, "check_port_file", lambda d, **kw: checked.append(d))
    import mlx_lm

    loads = []
    monkeypatch.setattr(mlx_lm, "load", lambda p, **kw: loads.append(kw) or ("model", "tok"))
    k = tmp_path / "k"
    k.mkdir()
    (k / "config.json").write_text(json.dumps({"model_type": "kolibri1", "model_file": "kolibri1.py"}))
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    assert genutil.load_arm("K4", k) == ("model", "tok", k)
    assert checked == [k, k] and loads == [{"trust_remote_code": True}]
    assert genutil.load_arm("G4", g) == ("model", "tok", g)
    assert checked == [k, k] and loads[-1] == {}
    f = tmp_path / "f"
    f.mkdir()
    (f / "config.json").write_text(json.dumps({"model_type": "x", "model_file": "evil.py"}))
    with pytest.raises(port.convert.UntrustedModelFile):
        genutil.load_arm("G8", f)
    assert len(loads) == 2
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


# --------------------------------------------------------------------------
# exp_037 (decision F2: MLX 0.32.3 / mlx-lm 0.32.0; DESIGN §2.8, §3.14, §6.3, §6.5)
# --------------------------------------------------------------------------


def test_kolibri_arms_resolve_under_the_builds_dir(monkeypatch, tmp_path):
    """DESIGN §2.8: K8 and K4 under $EXP037_BUILDS (the refreshed clones) when set and non-empty, else under
    $EXP036_MODELS; the peers always under $EXP036_MODELS."""
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    assert common.builds_dir() == tmp_path / "m"
    assert common.arm_dir("K8") == tmp_path / "m" / "Kolibri-1-MLX-8bit-g64"
    monkeypatch.setenv("EXP037_BUILDS", "")
    assert common.builds_dir() == tmp_path / "m"
    monkeypatch.setenv("EXP037_BUILDS", str(tmp_path / "b"))
    assert common.builds_dir() == tmp_path / "b"
    assert common.arm_dir("K8") == tmp_path / "b" / "Kolibri-1-MLX-8bit-g64"
    assert common.arm_dir("K4") == tmp_path / "b" / "Kolibri-1-MLX-4bit-g64"
    for arm in ("G8", "G4", "Q36-8", "Q36-4", "Q38-8", "Q38-4"):
        assert common.arm_dir(arm) == tmp_path / "m" / common.ARMS[arm].folder


def test_allowed_B_is_read_by_the_runner_guard_in_every_form():
    """allowed_B per arm through runner.guard.allowed_B_of (the one implementation): the list require_gate
    returns, the gate record (or a dict) holding allowed_B per arm or as one list, or a GateRecord-like object.
    No allowed_B gives {1} (§3.14's base set); a set G5-BP-lean cannot produce is refused; a peer is not
    restricted."""

    class GateRecordLike:
        def __init__(self, record):
            self.record = record

    class WithAttr:
        allowed_B = [8, 1, 4, 2]

    want = (1, 2, 4, 8)
    rec = {"verdict": {"K8": "PASS", "K4": "PASS"}, "allowed_B": {"K8": [8, 4, 2, 1], "K4": [1]}}
    assert common.allowed_B("K8", [1, 2, 4, 8]) == want
    assert common.allowed_B("K8", rec) == want and common.allowed_B("K4", rec) == (1,)
    assert common.allowed_B("K8", {"allowed_B": [2, 1, 4, 8]}) == want
    assert common.allowed_B("K8", GateRecordLike(rec)) == want
    assert common.allowed_B("K8", WithAttr()) == want
    assert common.allowed_B("K8", (1, 2, 4, 8, 16)) == (1, 2, 4, 8, 16)
    for no_b in ({"verdict": {"K8": "PASS"}}, {"allowed_B": {"K4": [1]}}):
        assert common.allowed_B("K8", no_b) == (1,)
    for peer in ("G8", "Q36-4", "Q38-8"):
        assert common.allowed_B(peer, None) is None
    for bad in ([2, 4], [1, 2], [1, 2, 4], [1, 1, 2, 4, 8], [1, "2", 4, 8], [1, True, 4, 8], [0, 1]):
        with pytest.raises(SystemExit):
            common.allowed_B("K8", bad)


def test_clip_B_is_the_largest_allowed_B_not_above_the_rule():
    assert common.clip_B(6, (1, 2, 4, 8)) == 4
    assert common.clip_B(8, (1, 2, 4, 8, 16)) == 8
    assert common.clip_B(32, (1, 2, 4, 8, 16)) == 16
    assert common.clip_B(16, (1,)) == 1
    assert common.clip_B(1, (1, 2, 4, 8)) == 1
    assert common.clip_B(3, None) == 3
    with pytest.raises(common.CouldNotRun):
        common.clip_B(1, (2, 4))


def test_require_gates_returns_what_the_guard_returned_for_kolibri_arms(bench_env):
    bench_env.allowed_B["K4"] = [1]
    out = common.require_gates(["K8", "G8", "K4", "Q36-4"])
    assert set(out) == {"K8", "K4"}
    assert [c for c in bench_env.calls if c[0] == "gate"] == [("gate", "K8"), ("gate", "K4")]
    assert common.allowed_B("K8", out["K8"]) == (1, 2, 4, 8, 16) and common.allowed_B("K4", out["K4"]) == (1,)


def test_require_environment_hands_the_guard_the_newest_gate_record(monkeypatch, tmp_path):
    """DESIGN §6.3 version binding: run_bench calls runner.guard.require_environment(rec) with rec the newest
    results/gate/gate_<UTC>.json under its results directory (None when there is none), and returns what it
    returns."""
    from runner import guard

    seen = []
    monkeypatch.setattr(guard, "require_environment", lambda rec: seen.append(rec) or {"ok": True})
    assert common.require_environment(tmp_path) == {"ok": True} and seen == [None]
    g = tmp_path / "gate"
    g.mkdir()
    (g / "gate_20261006T010000Z.json").write_text(json.dumps({"mode": "tiny", "n": 1}))
    (g / "gate_20261006T020000Z.json").write_text(json.dumps({"mode": "real", "n": 2}))
    (g / "gate_20261006T030000Z_layers.csv").write_text("x")
    (g / "gate_20261006T030000Z_mutants.json").write_text("{}")
    assert common.require_environment(tmp_path) == {"ok": True}
    assert {k: v for k, v in seen[-1].items() if k != "_file"} == {"mode": "real", "n": 2}
    assert seen[-1].get("_file", "gate_20261006T020000Z.json") == "gate_20261006T020000Z.json"
    assert common.latest_gate_record(tmp_path)["file"] == "results/gate/gate_20261006T020000Z.json"


def test_the_binding_records_but_does_not_judge_a_tiny_record(tmp_path):
    """Through the real guard: a tiny gate record is recorded, not judged (a real one would be judged)."""
    g = tmp_path / "gate"
    g.mkdir()
    (g / "gate_20261006T010000Z.json").write_text(json.dumps({"mode": "tiny"}))
    out = common.require_environment(tmp_path)
    assert out["binding"] == "recorded" and out["gate_mode"] == "tiny"
    assert out["pins"]["mlx"] == "0.32.3" and out["pins"]["mlx-lm"] == "0.32.0"


def test_load_arm_loads_a_tiny_conversion_through_model_file_trust(tiny_real_cal, monkeypatch):
    """mlx-lm 0.32.0 (#1385): a converted Kolibri build (model_file kolibri1.py, written by port/convert.py)
    loads through load_arm, which grants trust_remote_code only through model_file_trust; mlx_lm alone refuses
    the same directory. A Kolibri-architecture stand-in loaded as a peer arm (the dry run) gets the flag too."""
    import mlx.core as mx
    import mlx_lm

    from bench import genutil
    from port import convert

    seen = []
    real = convert.model_file_trust
    monkeypatch.setattr(convert, "model_file_trust", lambda d, **kw: seen.append(d) or real(d, **kw))
    d = tiny_real_cal.k8
    assert json.loads((d / "config.json").read_text())["model_file"] == "kolibri1.py"
    model, tok, out_dir = genutil.load_arm("K8", d)
    assert out_dir == d and seen == [d] and real(d) == {"trust_remote_code": True}
    logits = model(mx.array([[1, 2, 3]], dtype=mx.int32))
    assert logits.shape[:2] == (1, 3)
    _, _, d4 = genutil.load_arm("G4", tiny_real_cal.k4)
    assert seen == [d, d4]
    with pytest.raises(ValueError, match="trust_remote_code"):
        mlx_lm.load(str(d))


def test_load_arm_refuses_a_stale_port_file_before_any_load(tiny_real_cal, tmp_path, monkeypatch):
    import shutil

    import mlx_lm

    from bench import genutil
    from port import convert

    d = tmp_path / "Kolibri-1-MLX-8bit-g64"
    shutil.copytree(tiny_real_cal.k8, d)
    p = d / "kolibri1.py"
    p.write_text(p.read_text() + "\n# edited after the conversion\n")
    loads = []
    monkeypatch.setattr(mlx_lm, "load", lambda *a, **kw: loads.append(kw))
    for arm in ("K8", "G8"):  # a Kolibri arm, and a Kolibri-architecture stand-in as a peer
        with pytest.raises(convert.StalePortFile):
            genutil.load_arm(arm, d)
    assert loads == []


def test_eos_mask_output_is_independent_of_tokens():
    """mlx-lm 0.32.0 (#1777) puts the prefilled prompt tokens into the logits processors' history. EosMask
    reads only the logits, so its output is bitwise the same whatever the history holds."""
    import mlx.core as mx

    from bench import genutil

    rng = np.random.default_rng(7)
    logits = mx.array(rng.standard_normal((1, 1024)).astype(np.float32))
    mask = genutil.EosMask([EOS_ID, PAD_ID, 0])
    ref = np.array(mask(None, logits))
    histories = (
        mx.array(np.zeros(0, dtype=np.int32)),
        mx.array(np.array([5], dtype=np.int32)),
        mx.array(np.array(list(range(40, 80)) + [1, 1], dtype=np.int32)),
        mx.array(rng.integers(0, 1024, 3000).astype(np.int32)),
    )
    for tokens in histories:
        assert np.array_equal(np.array(mask(tokens, logits)), ref)
    assert np.isneginf(ref[0, [0, PAD_ID, EOS_ID]]).all()
    keep = np.ones(1024, dtype=bool)
    keep[[0, PAD_ID, EOS_ID]] = False
    assert np.array_equal(ref[0, keep], np.array(logits)[0, keep])


def test_stream_generate_gives_processors_the_prompt_history(tmp_path):
    """Pins mlx-lm 0.32.0 #1777, the reason for the test above: the first processor call sees the whole
    prompt (prefilled in chunks) as history, the next one the prompt plus the first token; timed_generate's
    EosMask still yields exactly max_tokens tokens."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import stream_generate

    from bench import genutil

    d = eos_first_model_dir(tmp_path / "eosfirst4")
    model, tok = load(str(d), trust_remote_code=True)
    eos = genutil.eos_ids(d, tok)
    mask = genutil.EosMask(eos)
    seen = []

    def recording(tokens, logits):
        seen.append([int(t) for t in np.array(tokens)])
        return mask(tokens, logits)

    prompt = list(range(40, 80))
    out = [
        int(r.token)
        for r in stream_generate(model, tok, mx.array(prompt, dtype=mx.int32), max_tokens=4, sampler=genutil.greedy,
                                 logits_processors=[recording], prefill_step_size=16)
    ]
    assert out == [1, 1, 1, 1]
    assert seen[0] == prompt and seen[1] == prompt + out[:1]
    r = genutil.timed_generate(model, tok, prompt, 4, eos, prefill_step_size=16)
    assert r["generation_tokens"] == 4 and r["completion_ids_sha256"] == common.sha256_ids(out)


def test_batch_aggregate_rate_is_the_stats_window_definition(tiny_models):
    """mlx-lm 0.32.0 #1829: BatchGenerator.stats() is a window over monotonic counters, and its generation_tps
    is generation_tokens / (wall_time - prompt_time) (generate_utils.BatchStats). Pinned on BatchStats and on
    the batch_aggregate record, which carries wall_time_s and prompt_time_s."""
    from mlx_lm import load
    from mlx_lm.generate_utils import BatchStats

    from bench import genutil

    s = BatchStats(prompt_tokens=10, prompt_time=0.5, generation_tokens=300, generation_steps=100,
                   decode_time=1.0, wall_time=2.0, peak_memory=0.0)
    assert s.generation_tps == 300 / (2.0 - 0.5) and s.prompt_tps == 10 / 0.5
    model, _ = load(str(tiny_models["a"]), trust_remote_code=True)
    rows = [list(range(10 + i, 74 + i)) for i in range(2)]
    g = genutil.batch_aggregate(model, rows, 16, [PAD_ID, EOS_ID], 2)
    # mlx-lm 0.32.0 counts L - 1 prompt tokens per sequence: the last prompt token goes in with the first decode step
    assert g["generation_tokens"] == 32 and g["prompt_tokens"] == 2 * (64 - 1)
    assert 0 < g["prompt_time_s"] < g["wall_time_s"]
    assert g["generation_tps"] == pytest.approx(g["generation_tokens"] / (g["wall_time_s"] - g["prompt_time_s"]), rel=1e-9)
    assert g["prompt_tps"] == pytest.approx(g["prompt_tokens"] / g["prompt_time_s"], rel=1e-9)


def test_the_batch_generator_api_the_bench_reads_exists():
    """What bench/genutil calls on mlx-lm 0.32.0's BatchGenerator (a future mlx-lm change fails here loudly)."""
    import inspect

    from mlx_lm.generate import BatchGenerator

    for name in ("insert", "next_generated", "stats", "close"):
        assert callable(getattr(BatchGenerator, name, None)), name
    params = inspect.signature(BatchGenerator.__init__).parameters
    for p in ("max_tokens", "stop_tokens", "sampler", "completion_batch_size", "prefill_batch_size",
              "prefill_step_size", "max_kv_size"):
        assert p in params, p


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
def test_the_bench_batch_generator_is_the_runner_construction(B, monkeypatch, tiny_models):
    """DESIGN §6.3: one BatchGenerator construction for the kit. genutil goes through
    runner.generate.make_batch_generator, with the registered arguments (BUILD_SPEC item 27)."""
    import mlx.core as mx
    from mlx_lm import load

    from bench import genutil
    from runner import generate

    real = generate.make_batch_generator
    seen = []

    def spy(model, b, **kw):
        seen.append((b, kw))
        return real(model, b, **kw)

    monkeypatch.setattr(generate, "make_batch_generator", spy)
    model, _ = load(str(tiny_models["a"]), trust_remote_code=True)
    gen = genutil._batch_generator(model, B, 8, [EOS_ID, PAD_ID], genutil.greedy)
    try:
        assert seen == [(B, {"eos": [EOS_ID, PAD_ID], "max_tokens": 8, "sampler": genutil.greedy,
                             "prefill_step_size": 2048})]
        assert gen.prefill_batch_size == min(B, 8) and gen.completion_batch_size == B
        assert gen.prefill_step_size == 2048 and gen.max_kv_size is None and gen.max_tokens == 8
        assert gen.model(mx.array([[1, 2, 3]], dtype=mx.int32)).dtype == mx.float32
    finally:
        gen.close()


def test_no_bench_module_constructs_a_batch_generator_itself():
    from pathlib import Path

    for p in sorted((Path(common.EXP_DIR) / "bench").glob("*.py")):
        assert "BatchGenerator(" not in p.read_text(), p.name
