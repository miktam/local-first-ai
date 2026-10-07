"""runner/generate.py and the cell machinery of runner/run.py on a tiny
Kolibri checkpoint (BUILD_SPEC §5.4 test_runner_tiny; HYPOTHESIS rule 6).

- an end-to-end tiny cell and the record schema (public and withheld);
- truncation flagged (finish_reason "length"), EOS stops, per-sequence max_tokens;
- B = 4 vs B = 1 greedy equality (fp32);
- in-flight sequences (queued + prefilling + decoding) never exceed B, for B in {1, 2, 4, 8};
- the same seed gives the same sampled outputs;
- resume after a simulated crash (stop + torn last line) gives the same records;
- the crash fallback: a stale heartbeat resumes the cell at the next lower B with
  b_fallback_from set; after two fallbacks the cell moves to aborted/ and is NOT RUN;
- exp_037: private writes go under $EXP036_PRIVATE/exp037/ (DESIGN §2.9) and the
  Kolibri arms resolve under builds_dir() (§2.8).

Under mlx-lm 0.32.0 the step log reads BatchGenerator.stats() windows (exp_037
decision F2; tests/test_runner_mlxlm032.py pins them); every assertion here is
exp_036's, unchanged.

The tokenizer is a stub over the tiny vocabulary (ids 1000/1001 play <think>/</think>).
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from exp036_helpers import import_sibling  # noqa: E402

import tiny_checkpoint as tc  # noqa: E402
from runner import jsonl  # noqa: E402
from runner.generate import run_cell, split_ids  # noqa: E402

EOS = [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]
GREEDY = {"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0}
TIMING = ("t_submit", "t_first_token", "t_done", "wall_s", "batch_id")


class StubTok:
    unk_token_id = None
    eos_token_ids = set(EOS)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": 1000, "</think>": 1001, "<|channel>": 1002, "<channel|>": 1003}.get(t)


@pytest.fixture(scope="module")
def model(tiny_vendor_dir):
    harness = import_sibling("port_harness")
    return harness.load_port(tiny_vendor_dir, float32=True)


def items(n, base=12, step=9, max_tokens=None):
    out = []
    for k in range(n):
        it = {"id": f"q{k:03d}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k + 100:064x}",
              "prompt_ids": tc.random_ids(base + step * k, seed=k), "prompt_text": "user turn"}
        if max_tokens is not None:
            it["max_tokens"] = max_tokens(k)
        out.append(it)
    return out


def run(model, path, its, B, cap=24, seed=36, cfg=GREEDY, private=None, **kw):
    w = jsonl.open_run(path, {"cell": "t", "B": B}, aborted_root=path.parent / "aborted",
                       private_root=path.parent / "private")
    pw = None
    if private is not None:
        pw = jsonl.open_run(private, {"cell": "t"}, aborted_root=path.parent / "aborted",
                            private_root=path.parent / "private")
    try:
        summ = run_cell(model, StubTok(), its, "K8", "mmlu_en", "high", cap, B, seed, w, "kolibri",
                        sampler_cfg=cfg, eos=kw.pop("eos", EOS), private_writer=pw, **kw)
    finally:
        w.close()
        if pw:
            pw.close()
    return summ


def records(path):
    return {r["key"]["item"]: r for r in jsonl.first_records(jsonl.read_jsonl(path))}


def content(recs):
    return {k: {f: v for f, v in r.items() if f not in TIMING} for k, r in recs.items()}


# ---------------------------------------------------------------- schema

SCHEMA = {"type", "key", "item_sha256", "prompt_sha256", "rendered_sha256", "rendered_prompt_tokens", "prompt_prefill",
          "completion_ids", "completion_tokens", "reasoning_tokens", "answer_tokens", "text", "reasoning_text",
          "answer_text", "finish_reason", "truncated", "reasoning_status", "split_by", "stop_token", "t_submit",
          "t_first_token", "t_done", "wall_s", "batch_id", "batch_size", "cell_seed", "max_tokens", "sampler",
          "eos_ids"}


def test_end_to_end_tiny_cell_record_schema_and_steps(model, tmp_path):
    p = tmp_path / "raw/S2/K8/mmlu_en_high.jsonl"
    its = items(6, max_tokens=lambda k: 10 + 3 * k)
    summ = run(model, p, its, B=4, cap=20, steps_path=tmp_path / "steps.jsonl", heartbeat_path=tmp_path / ".hb")
    assert summ["n_done"] == 6 and not summ["stopped"]
    recs = records(p)
    assert sorted(recs) == [i["id"] for i in its]
    for k, r in enumerate(recs[i["id"]] for i in its):
        assert set(r) == SCHEMA, set(r) ^ SCHEMA
        assert r["key"] == {"arm": "K8", "task": "mmlu_en", "effort": "high", "item": its[k]["id"], "pass": 0}
        assert r["max_tokens"] == min(10 + 3 * k, 20)  # per-sequence max_tokens, clamped to the cap
        assert r["completion_tokens"] == len(r["completion_ids"]) <= r["max_tokens"]
        assert r["rendered_prompt_tokens"] == len(its[k]["prompt_ids"])
        assert r["batch_size"] == 4 and r["cell_seed"] == 36 and r["eos_ids"] == EOS
        assert r["sampler"] == GREEDY
        assert r["truncated"] == (r["finish_reason"] == "length")
        if r["truncated"]:
            assert r["completion_tokens"] == r["max_tokens"]
        assert r["reasoning_tokens"] + r["answer_tokens"] <= r["completion_tokens"]
        assert r["t_submit"] <= r["t_first_token"] <= r["t_done"]
    steps = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    dec = [s for s in steps if s["type"] == "step"]
    assert dec and all({"n_live", "padded_len", "step_seconds", "prompt_tokens", "t"} <= set(s) for s in dec)
    assert max(s["n_live"] for s in dec) == 4
    assert sum(s["prompt_tokens"] for s in dec) == sum(len(i["prompt_ids"]) - 1 for i in its)
    assert any(s["type"] == "env" for s in steps)
    hb = json.loads((tmp_path / ".hb").read_text())
    assert hb["state"] == "running" and hb["n_done"] == 6 and hb["B"] == 4 and hb["pid"] == os.getpid()


def test_withheld_records_hashed_in_repo_full_in_private(model, tmp_path):
    from runner.common import ids_sha256, sha256_text

    p = tmp_path / "raw/S2/K8/gpqa_en_high.jsonl"
    priv = tmp_path / "private/raw/S2/K8/gpqa_en_high.jsonl"
    run(model, p, items(3), B=2, private=priv)
    pub, full = records(p), records(priv)
    assert sorted(pub) == sorted(full)
    for k in pub:
        r, f = pub[k], full[k]
        assert not {"text", "completion_ids", "reasoning_text", "answer_text"} & set(r)
        assert r["completion_ids_sha256"] == ids_sha256(f["completion_ids"])
        assert r["text_sha256"] == sha256_text(f["text"])
        assert not any(k.startswith(("answer_text", "reasoning_text")) for k in r)  # short texts: not even hashed
        assert r["completion_tokens"] == f["completion_tokens"]


def test_eos_stops_and_truncation_is_flagged(model, tmp_path):
    its = items(4)
    run(model, tmp_path / "a.jsonl", its, B=1, cap=16)
    first = records(tmp_path / "a.jsonl")
    assert all(r["finish_reason"] == "length" and r["truncated"] for r in first.values())
    eos_tok = first["q000"]["completion_ids"][5]  # a token greedy decoding emits early
    run(model, tmp_path / "b.jsonl", its, B=1, cap=16, eos=[eos_tok])
    second = records(tmp_path / "b.jsonl")
    r = second["q000"]
    assert r["finish_reason"] == "stop" and not r["truncated"]
    assert r["completion_ids"][-1] == eos_tok and r["stop_token"] == eos_tok
    assert r["completion_tokens"] == first["q000"]["completion_ids"].index(eos_tok) + 1
    assert r["answer_tokens"] <= r["completion_tokens"] - 1  # the EOS is in neither segment


def test_B4_equals_B1_greedy_in_fp32(model, tmp_path):
    its = items(7, base=10, step=13, max_tokens=lambda k: 12 + 4 * k)
    run(model, tmp_path / "b1.jsonl", its, B=1, cap=40)
    run(model, tmp_path / "b4.jsonl", its, B=4, cap=40)
    one, four = records(tmp_path / "b1.jsonl"), records(tmp_path / "b4.jsonl")
    assert {k: r["completion_ids"] for k, r in one.items()} == {k: r["completion_ids"] for k, r in four.items()}


@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_in_flight_never_exceeds_B(model, tmp_path, B):
    seen = []

    def probe(gen):
        seen.append(len(gen._generation_batch) + len(gen._prompt_batch) + len(gen._unprocessed_sequences))
        assert len(gen._generation_batch) + len(gen._prompt_batch) <= B

    its = items(11, base=8, step=5, max_tokens=lambda k: 4 + (7 * k) % 13)
    summ = run(model, tmp_path / f"b{B}.jsonl", its, B=B, cap=20, on_step=probe)
    assert summ["n_done"] == 11
    assert max(seen) <= B and (B == 1 or max(seen) == B)


def test_same_seed_same_sampled_outputs(model, tmp_path):
    cfg = {"order": "vllm", "temperature": 1.0, "top_p": 0.97, "top_k": 128}
    its = items(5)
    run(model, tmp_path / "s1.jsonl", its, B=2, cfg=cfg, seed=1234)
    run(model, tmp_path / "s2.jsonl", its, B=2, cfg=cfg, seed=1234)
    run(model, tmp_path / "s3.jsonl", its, B=2, cfg=cfg, seed=4321)
    a, b, c = (records(tmp_path / f"s{i}.jsonl") for i in (1, 2, 3))
    ids = lambda r: {k: v["completion_ids"] for k, v in r.items()}  # noqa: E731
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_resume_after_simulated_crash_gives_identical_records(model, tmp_path):
    its = items(6, max_tokens=lambda k: 8 + 2 * k)
    ref = tmp_path / "ref.jsonl"
    run(model, ref, its, B=1, cap=20)

    p = tmp_path / "raw/S2/K8/mmlu_en_high.jsonl"

    def stop_after_three():
        recs = [r for r in jsonl.read_jsonl(p) if r.get("type") == "record"] if p.exists() else []
        return len(recs) >= 3

    summ = run(model, p, its, B=1, cap=20, stop=stop_after_three)
    assert summ["stopped"] and summ["n_done"] == 3
    with open(p, "ab") as f:  # killed while writing the 4th line
        f.write(b'{"completion_ids": [1, 2, 3], "key": {"it')
    done = jsonl.completed_keys("K8", "mmlu_en", "high", root=tmp_path / "raw")
    assert len(done) == 3
    summ = run(model, p, its, B=1, cap=20, done=done)
    assert summ["n_todo"] == 3 and summ["n_done"] == 3
    recs, bad = jsonl.scan(p)
    assert bad == 1 and [r["type"] for r in recs if r["type"] != "record"] == ["header", "resume"]
    assert content(records(p)) == content(records(ref))
    assert jsonl.completed_keys("K8", "mmlu_en", "high", root=tmp_path / "raw") == {(i["id"], 0) for i in its}


def test_split_ids_rules():
    o, c = 1000, 1001
    assert split_ids("kolibri", "empty_think", [5, 6, tc.EOS_TOKEN_ID], o, c, EOS) == ([], [5, 6], "none")
    assert split_ids("qwen3_6", "open_think", [5, c, 7, 8], o, c, EOS) == ([5], [7, 8], "closed")
    assert split_ids("qwen3_6", "open_think", [5, 6], o, c, EOS) == ([5, 6], [], "unclosed")
    assert split_ids("kolibri", "none", [o, 5, c, 9], o, c, EOS) == ([5], [9], "closed")
    assert split_ids("kolibri", "none", [9, o, 5], o, c, EOS) == ([], [9, o, 5], "none")
    assert split_ids("gemma4", "none", [7, 8], 1002, 1003, EOS) == ([], [7, 8], "none")


# --------------------------------------------------------- crash fallback


@pytest.fixture
def cellctx(tmp_path, monkeypatch, model):
    run_mod = import_sibling("runner.run")
    monkeypatch.setenv("EXP036_PRIVATE", str(tmp_path / "private"))
    ctx = run_mod.Ctx(tmp_path / "exp")
    ctx.raw.mkdir(parents=True)
    its = items(5, max_tokens=lambda k: 6 + k)
    monkeypatch.setattr(run_mod, "items_for_cell", lambda task, n: [dict(i) for i in its][:n])

    def fake_get(self, arm):
        return model, StubTok(), {"eos_ids": EOS, "model_dir": "$EXP036_MODELS/tiny"}

    monkeypatch.setattr(run_mod.Loaded, "get", fake_get)
    monkeypatch.setattr(run_mod.chat, "sampling_for", lambda fam: dict(GREEDY))
    return run_mod, ctx, its


def _hb(ctx, session, cell, B, pid=2**22 + 12345, age_min=45):
    from datetime import timedelta

    from runner.common import utc_iso, utc_now
    from runner.generate import write_heartbeat

    write_heartbeat(ctx.raw / session / ".heartbeat", {
        "state": "running", "pid": pid, "cell": cell, "arm": "K8", "task": "mmlu_en", "effort": "high",
        "pass": 0, "B": B, "utc": utc_iso(utc_now() - timedelta(minutes=age_min))})


def test_crash_fallback_then_abort_after_two_fallbacks(cellctx):
    run_mod, ctx, its = cellctx
    rules = ctx.rules
    cell = {"arm": "K8", "task": "mmlu_en", "effort": "high", "pass": 0, "n": 5, "cap": 20, "B": 8, "tier": "A"}
    # First run: two records, then a "crash" (stop + stale running heartbeat).

    class StopAfter:
        def __init__(self, k):
            self.k = k

        def __call__(self):
            repo = ctx.raw / "S2/K8/mmlu_en_high.jsonl"
            return repo.exists() and sum(1 for r in jsonl.read_jsonl(repo) if r.get("type") == "record") >= self.k

    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, StopAfter(2))
    _hb(ctx, "S2", "K8:mmlu_en:high", 8)
    hb = run_mod.detect_crash(ctx, rules)
    assert hb is not None and hb["cell"] == "K8:mmlu_en:high"
    fb = run_mod.plan_fallback(ctx, rules, hb)
    assert fb == {"cell": "K8:mmlu_en:high", "from": 8, "to": 4}

    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, StopAfter(3), fallback=fb)
    repo = ctx.raw / "S2/K8/mmlu_en_high.jsonl"
    hs = jsonl.headers(jsonl.read_jsonl(repo))
    assert [h["B"] for h in hs] == [8, 4] and hs[-1]["b_fallback_from"] == 8 and hs[-1]["type"] == "resume"
    third = [r for r in jsonl.read_jsonl(repo) if r.get("type") == "record"][2]
    assert third["b_fallback_from"] == 8 and third["batch_size"] == 4

    # A plain resume (Ctrl-C, no crash) stays at B = 4 and keeps b_fallback_from.
    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, StopAfter(4))
    fourth = [r for r in jsonl.read_jsonl(repo) if r.get("type") == "record"][3]
    assert fourth["b_fallback_from"] == 8 and fourth["batch_size"] == 4
    assert jsonl.fallback_count(repo) == 1

    # Second crash at B = 4: falls back to 2 (the second fallback).
    _hb(ctx, "S2", "K8:mmlu_en:high", 4)
    fb2 = run_mod.plan_fallback(ctx, rules, run_mod.detect_crash(ctx, rules))
    assert fb2 == {"cell": "K8:mmlu_en:high", "from": 4, "to": 2}
    run_mod.execute_cell(ctx, "S3", cell, run_mod.Loaded(), None, StopAfter(4), fallback=fb2)
    assert jsonl.fallback_count(repo) == 2
    assert not (ctx.raw / "S3/K8").exists()  # resumed in S3, still the S2 file

    # Third crash: after two fallbacks the cell is aborted and NOT RUN.
    _hb(ctx, "S3", "K8:mmlu_en:high", 2)
    fb3 = run_mod.plan_fallback(ctx, rules, run_mod.detect_crash(ctx, rules))
    assert "abort" in fb3
    dest = run_mod.abort_cell(ctx, "S3", cell, fb3["abort"])
    assert (dest / "NOTE.md").is_file() and (dest / "S2-mmlu_en_high.jsonl").is_file()
    assert not repo.exists()
    assert jsonl.completed_keys("K8", "mmlu_en", "high", root=ctx.raw) == set()
    nr = run_mod.not_run_cells(ctx)
    assert nr["K8:mmlu_en:high"]["reason"] == fb3["abort"]


def test_no_lower_B_aborts_and_a_live_run_is_refused(cellctx):
    run_mod, ctx, _ = cellctx
    _hb(ctx, "S2", "K8:mmlu_en:high", 1)
    fb = run_mod.plan_fallback(ctx, ctx.rules, run_mod.detect_crash(ctx, ctx.rules))
    assert "abort" in fb and "no lower B" in fb["abort"]
    _hb(ctx, "S2", "K8:mmlu_en:high", 8, pid=os.getppid(), age_min=1)  # alive and fresh
    with pytest.raises(SystemExit):
        run_mod.detect_crash(ctx, ctx.rules)
    _hb(ctx, "S2", "K8:mmlu_en:high", 8, pid=os.getppid(), age_min=45)  # alive but stale: a hung run
    assert run_mod.detect_crash(ctx, ctx.rules)["B"] == 8


# ---------------------------------------- exp_037 directories (DESIGN §2.8, §2.9)


def test_private_writes_go_under_exp037(cellctx, tmp_path, monkeypatch):
    """exp_037's private records (raw, aborted; the pilot's are under the same root) go to $EXP036_PRIVATE/exp037/,
    labelled so in the repo's manifest; nothing is written beside exp_036's private files."""
    run_mod, ctx, its = cellctx
    from runner.common import private_base_dir, private_dir, redact_path

    base = tmp_path / "private"
    assert private_base_dir() == base and private_dir() == base / "exp037" and ctx.private == base / "exp037"
    cell = {"arm": "K8", "task": "gpqa_en", "effort": "high", "pass": 0, "n": 3, "cap": 12, "B": 2, "tier": "A"}
    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, lambda: False)
    priv = base / "exp037" / "raw" / "S2" / "K8" / "gpqa_en_high.jsonl"
    assert priv.is_file() and sorted(p.name for p in base.iterdir()) == ["exp037"]
    assert redact_path(priv) == "$EXP036_PRIVATE/exp037/raw/S2/K8/gpqa_en_high.jsonl"
    assert run_mod.update_withheld_manifest(ctx, "S2") == 1
    man = jsonl.read_jsonl(ctx.evidence / "withheld_manifest.jsonl")
    assert [r["path"] for r in man] == ["$EXP036_PRIVATE/exp037/raw/S2/K8/gpqa_en_high.jsonl"]
    run_mod.abort_cell(ctx, "S2", cell, "test")
    moved = list((base / "exp037" / "aborted").glob("*/S2-gpqa_en_high.jsonl"))
    assert len(moved) == 1 and sorted(p.name for p in base.iterdir()) == ["exp037"]
    # runner/common.redact_path's own fallback (without tools/redact.py) keeps the exp037 label.
    from runner import common

    real = common.importlib.import_module

    def no_tools(name, *a, **k):
        if name == "tools.redact":
            raise ModuleNotFoundError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(common.importlib, "import_module", no_tools)
    assert common.redact_path(moved[0]).startswith("$EXP036_PRIVATE/exp037/aborted/")
    assert common.redact_path(base / "manifests" / "x.jsonl") == "$EXP036_PRIVATE/manifests/x.jsonl"


def test_kolibri_arm_dirs_resolve_under_builds_dir(tmp_path, monkeypatch):
    """DESIGN §2.8: K8 and K4 load from $EXP037_BUILDS (the refreshed clones) when set, else from $EXP036_MODELS;
    peers always from $EXP036_MODELS. The directory names are exp_036's."""
    from runner.common import arm_dir, builds_dir

    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    assert builds_dir() == tmp_path / "models"
    assert arm_dir("K8") == tmp_path / "models" / "Kolibri-1-MLX-8bit-g64"
    monkeypatch.setenv("EXP037_BUILDS", str(tmp_path / "builds"))
    assert builds_dir() == tmp_path / "builds"
    assert arm_dir("K8") == tmp_path / "builds" / "Kolibri-1-MLX-8bit-g64"
    assert arm_dir("K4") == tmp_path / "builds" / "Kolibri-1-MLX-4bit-g64"
    assert arm_dir("G8") == tmp_path / "models" / "gemma-4-26b-a4b-it-8bit"


def test_rotated_sliding_caches_with_mid_run_admission(tiny_pattern5_dir, tmp_path):
    """pattern5 (window 17): prompts and generations several windows long,
    sequences admitted while others decode past 2 windows; B = 3 and B = 1
    give the same greedy tokens through BatchRotatingKVCache."""
    harness = import_sibling("port_harness")
    m = harness.load_port(tiny_pattern5_dir, float32=True)
    its = items(7, base=20, step=7, max_tokens=lambda k: 25 + (11 * k) % 40)
    run(m, tmp_path / "r1.jsonl", its, B=1, cap=64)
    run(m, tmp_path / "r3.jsonl", its, B=3, cap=64)
    one, three = records(tmp_path / "r1.jsonl"), records(tmp_path / "r3.jsonl")
    assert max(len(i["prompt_ids"]) for i in its) > 3 * 17
    assert {k: r["completion_ids"] for k, r in one.items()} == {k: r["completion_ids"] for k, r in three.items()}


def test_teacher_force_logprobs_chunked_equals_one_forward(model):
    import mlx.core as mx
    import numpy as np

    from runner.generate import iter_teacher_force_logprobs, teacher_force_logprobs

    ids = tc.random_ids(150, seed=9)  # > 2 windows of the vendor preset (65)
    full = model(mx.array([ids])).astype(mx.float32)[0]
    want = np.array(full - mx.logsumexp(full, axis=-1, keepdims=True))
    got = teacher_force_logprobs(model, ids, chunk=32)
    assert got.shape == want.shape and got.dtype == np.float32
    np.testing.assert_allclose(got, want, atol=2e-5)
    starts = [s for s, _ in iter_teacher_force_logprobs(model, ids, chunk=64)]
    assert starts == [0, 64, 128]
