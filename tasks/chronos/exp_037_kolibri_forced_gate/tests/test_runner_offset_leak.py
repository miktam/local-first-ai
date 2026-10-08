"""exp_037 Amendment 2, gate fix: run_cell leaves no lazy graph behind BatchKVCache.offset.

Pilot 20261008T082355Z crashed in K8 aime_en_pilot_high (B = 4, cap 65,536) after about 49,664 decode steps with
"RuntimeError: [metal::malloc] Resource limit (499000) exceeded", raised in mlx_lm's GenerationBatch._step. mlx_lm
0.32.0 BatchKVCache.update_and_fetch rebinds `offset` lazily on every decode step (cache.py:929) and never forces it;
on Kolibri's NoPE layers (i % 5 == 4) nothing in decode reads it, so every step left one more live Metal buffer per
NoPE layer for the whole cell, and resource_limit counts buffers. run_cell now evaluates those offsets once per
gen.next() (runner.generate._settle_batch_offsets).

T1 (structural; oracle: MLX's own graph export) counts the primitive nodes behind every BatchKVCache.offset of the
decode batch through on_step: 0 at every step with the fix; without it the count grows with the decode steps.
T2 (behavioural; oracle: MLX's own limit, mx.device_info()["resource_limit"]) holds resource_limit - 15,000 buffers
and runs run_cell for 4,000 decode steps on a 50-layer / 10-NoPE build, in a subprocess, so that it neither depends
on nor disturbs the rest of the suite: it completes with the fix and raises the pilot's error without it.
T3 (output-neutral; oracle: the pre-fix code path, i.e. the settle step patched to a no-op) runs the same items with
the settle step a no-op and active, greedy and under Kolibri's vendor sampler: the records and the step logs are
identical, and the active arm called the step once per gen.next().

Every model here is a tiny real-layout build written by the tests' own writers (tests/tiny_real_layout.py,
conftest.py), never real weights. EOS is an id outside the tiny vocabulary, so every sequence runs to its max_tokens
and every step count is fixed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

import port_harness  # noqa: E402
import tiny_checkpoint as tc  # noqa: E402
import tiny_real_layout as trl  # noqa: E402
from runner import chat, generate, jsonl  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
EXP_DIR = TESTS_DIR.parent

NO_EOS = 5000  # outside the tiny vocabulary (tiny_checkpoint.VOCAB_SIZE 1024): never sampled
assert NO_EOS >= tc.VOCAB_SIZE
GREEDY = {"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0}
WALL_CLOCK = ("t_submit", "t_first_token", "t_done", "wall_s")
STEP_FIELDS = ("type", "step", "n_live", "n_prefill", "padded_len", "prompt_tokens", "admitted")

# T2's sizes (Amendment 2).
T2_MARGIN = 15_000  # buffers left free under resource_limit
T2_STEPS = 4_000  # decode steps per sequence (all B sequences admitted together)
T2_B = 4
T2_LAYERS = 50  # the released depth; NoPE at i % 5 == 4: 10 layers
T2_TIMEOUT_S = 900.0


class StubTok:
    """tests/test_runner_tiny.py's stub over the tiny vocabulary (ids 1000/1001 play <think>/</think>)."""

    unk_token_id = None
    eos_token_ids = {NO_EOS}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": 1000, "</think>": 1001, "<|channel>": 1002, "<channel|>": 1003}.get(t)


@pytest.fixture(scope="module")
def k8_model(tiny_real_val_unsharp):
    """The whole tiny gate's validation build (seed 29, unsharpened), 8 bits group 64, loaded as the kit loads K8."""
    return port_harness.load_port(tiny_real_val_unsharp.k8)


def _items(n: int, max_tokens, base: int = 40, seed: int = 700) -> list[dict]:
    return [{"id": f"q{k:03d}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k + 100:064x}",
             "prompt_ids": tc.random_ids(base + 7 * k, seed=seed + k), "prompt_text": "user turn",
             "max_tokens": int(max_tokens(k))} for k in range(n)]


def _run(model, path: Path, its, B: int, cap: int, cfg: dict, seed: int, **kw) -> dict:
    w = jsonl.open_run(path, {"cell": "offset_leak", "B": B}, aborted_root=path.parent / "aborted",
                       private_root=path.parent / "private")
    try:
        return generate.run_cell(model, StubTok(), its, "K8", "offset_leak", "high", cap, B, seed, w, "kolibri",
                                 sampler_cfg=cfg, eos=[NO_EOS], **kw)
    finally:
        w.close()


def _records(path: Path) -> dict:
    return {r["key"]["item"]: {f: v for f, v in r.items() if f not in WALL_CLOCK}
            for r in jsonl.first_records(jsonl.read_jsonl(path))}


def _steps(path: Path) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    return [{f: s.get(f) for f in STEP_FIELDS} for s in rows if s.get("type") == "step"]


_RECT = re.compile(r"shape=rectangle")


def _graph_nodes(arr) -> int:
    """Primitive (unevaluated) nodes in the graph behind `arr`, as mx.export_to_dot draws them (rectangles)."""
    buf = io.StringIO()
    mx.export_to_dot(buf, arr)
    return len(_RECT.findall(buf.getvalue()))


class OffsetProbe:
    """on_step hook: for every gen.next() whose decode batch is non-empty, the primitive-node count behind each
    BatchKVCache.offset of that batch, the live uids, and the number of such caches."""

    def __init__(self):
        self.calls = 0
        self.rows: list[dict] = []
        self.uids: set = set()

    def __call__(self, gen):
        self.calls += 1
        gb = gen._generation_batch
        if not len(gb):
            return
        self.uids.update(gb.uids)
        offs = [c.offset for c in gb.prompt_cache if isinstance(c, BatchKVCache)]
        self.rows.append({"decode_step": len(self.rows) + 1, "n_live": len(gb), "n_batch_kv": len(offs),
                          "nodes": [_graph_nodes(o) for o in offs]})


# ------------------------------------------------------------------------------------------------ T1 (structural)


def test_t1_no_graph_behind_batch_kv_offsets(k8_model, tmp_path):
    """B = 2, 3 items with staggered max_tokens (one refill), about 210 decode steps: after every gen.next() of
    run_cell, no BatchKVCache.offset of the decode batch has a primitive node behind it."""
    chain = mx.array([0, 0], dtype=mx.int32)
    for _ in range(5):
        chain = chain + 1  # the shape of update_and_fetch's rebinding
    assert _graph_nodes(chain) >= 5  # the oracle sees a lazy chain ...
    mx.eval(chain)
    assert _graph_nodes(chain) == 0  # ... and none once it is evaluated
    nope = [i for i, layer in enumerate(k8_model.layers) if not layer.use_sliding]
    assert nope == [4, 9], nope  # the real layout at 10 layers: full attention (NoPE) at i % 5 == 4
    probe = OffsetProbe()
    its = _items(3, max_tokens=lambda k: (120, 200, 90)[k])
    summ = _run(k8_model, tmp_path / "t1.jsonl", its, B=2, cap=256, cfg=GREEDY, seed=11, on_step=probe)
    assert summ["n_done"] == 3 and not summ["stopped"], summ
    recs = _records(tmp_path / "t1.jsonl")
    assert sorted(r["completion_tokens"] for r in recs.values()) == [90, 120, 200]
    assert probe.uids and len(probe.uids) == 3 and max(r["n_live"] for r in probe.rows) == 2
    assert {r["n_batch_kv"] for r in probe.rows} == {len(nope)}  # every NoPE layer's cache was looked at
    n = len(probe.rows)
    assert n >= 200, n
    last = probe.rows[-1]
    assert last["nodes"] == [0] * len(nope), (
        f"after {n} decode steps the BatchKVCache offsets still have {last['nodes']} primitive nodes behind them "
        f"(one lazy add per decode step: the chain that exhausted resource_limit in pilot 20261008T082355Z)")
    grown = [r for r in probe.rows if any(r["nodes"])]
    assert not grown, f"{len(grown)} of {n} steps left a graph behind an offset; first {grown[0]}"


# ------------------------------------------------------------------------------------------------ T2 (behavioural)


def _write_real50(root: Path) -> Path:
    """tiny_real_layout's gate-validation recipe (seed 29, q/k norms unsharpened, the real head and expert layout)
    at the released depth, 50 layers with full attention (NoPE) at i % 5 == 4, converted by port/convert.py to
    8 bits, group 64, as K8 is. Written with the tests' own writers; no real weights."""
    from port import convert  # the experiment directory is on sys.path (conftest.py)

    ov = trl.real_layout_overrides()
    ov["num_hidden_layers"] = T2_LAYERS
    ov["layer_types"] = tc._layer_types(T2_LAYERS, 5)
    bf16 = tc.write_tiny_checkpoint(root / trl.BF16_NAME, seed=trl.SEED_VAL, preset=trl.PRESET, copy_port=False,
                                    overrides=ov, edit_tensors=trl.sharpen_qk_norms(trl.SHARPEN_GATE))
    cfg = json.loads((bf16 / "config.json").read_text())
    cfg.pop("model_file", None)  # the released BF16 layout (write_real_layout, hf_layout=True)
    (bf16 / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    trl.write_stub_tokenizer(bf16)
    out = root / trl.build_dir_name(8)
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(bf16, out, 8, trl.GROUP_SIZE, guard=False)
    return out


# The subprocess. It loads the build, holds resource_limit - margin scalar buffers, then runs run_cell greedy with
# B sequences of `steps` tokens each, and writes one JSON object. Any exception from run_cell is reported, not raised.
_CHILD = textwrap.dedent("""
    import json, sys, time, traceback
    sys.dont_write_bytecode = True
    exp_dir, tests_dir, ckpt, work, steps, margin, B, no_eos = sys.argv[1:9]
    steps, margin, B, no_eos = int(steps), int(margin), int(B), int(no_eos)
    sys.path[:0] = [exp_dir, tests_dir]
    from pathlib import Path
    import mlx.core as mx
    import port_harness, tiny_checkpoint as tc
    from runner import jsonl
    from runner import generate

    class StubTok:
        unk_token_id = None
        eos_token_ids = {no_eos}
        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"<{int(i)}>" for i in ids)
        def convert_tokens_to_ids(self, t):
            return {"<think>": 1000, "</think>": 1001, "<|channel>": 1002, "<channel|>": 1003}.get(t)

    work = Path(work)
    res = {"generate_file": generate.__file__, "has_settle_step": hasattr(generate, "_settle_batch_offsets")}
    model = port_harness.load_port(Path(ckpt))
    mx.eval(model.parameters())
    res["n_layers"] = len(model.layers)
    res["nope_layers"] = [i for i, l in enumerate(model.layers) if not l.use_sliding]
    limit = int(mx.device_info()["resource_limit"])
    res["resource_limit"] = limit
    mx.clear_cache()
    hold = [mx.array(i) for i in range(limit - margin)]  # one live Metal buffer each
    mx.eval(hold)
    res["held"] = len(hold)
    its = [{"id": f"q{k:03d}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k + 100:064x}",
            "prompt_ids": tc.random_ids(16 + k, seed=900 + k), "prompt_text": "user turn", "max_tokens": steps}
           for k in range(B)]
    seen = {"decode_steps": 0}
    def on_step(gen):
        if len(gen._generation_batch):
            seen["decode_steps"] += 1
    raw = work / "t2.jsonl"
    w = jsonl.open_run(raw, {"cell": "offset_leak_t2", "B": B}, aborted_root=work / "aborted",
                       private_root=work / "private")
    t0 = time.time()
    res["error"] = None
    try:
        summ = generate.run_cell(model, StubTok(), its, "K8", "offset_leak", "high", steps, B, 13, w, "kolibri",
                                 sampler_cfg={"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0},
                                 eos=[no_eos], on_step=on_step)
        res["n_done"] = summ["n_done"]
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = ["/".join(f.filename.rsplit("/", 2)[-2:]) + f":{f.lineno} {f.name}"
                            for f in traceback.extract_tb(e.__traceback__)]
    finally:
        w.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["decode_steps"] = seen["decode_steps"]
    recs = list(jsonl.first_records(jsonl.read_jsonl(raw))) if raw.exists() else []
    res["records"] = [[r["completion_tokens"], r["finish_reason"]] for r in recs]
    (work / "t2_result.json").write_text(json.dumps(res))
""")


def test_t2_resource_limit_not_reached_with_the_buffers_nearly_exhausted(tmp_path):
    """resource_limit - 15,000 buffers held, then 4 x 4,000 tokens through run_cell on a 50-layer / 10-NoPE build
    (B = 4, greedy): with the fix it completes; without it, 10 buffers per decode step exhaust the remaining ~12,400
    after about 1,240 steps and run_cell raises the pilot's "[metal::malloc] Resource limit" error."""
    ckpt = _write_real50(tmp_path / "real50")
    work = tmp_path / "work"
    work.mkdir()
    cmd = [sys.executable, "-c", _CHILD, str(EXP_DIR), str(TESTS_DIR), str(ckpt), str(work), str(T2_STEPS),
           str(T2_MARGIN), str(T2_B), str(NO_EOS)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(EXP_DIR), env=env, capture_output=True, text=True, timeout=T2_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail(f"T2's subprocess did not finish within {T2_TIMEOUT_S:.0f} s "
                    f"({T2_STEPS} decode steps at B = {T2_B})")
    out = work / "t2_result.json"
    assert p.returncode == 0 and out.is_file(), (
        f"T2's subprocess exited {p.returncode} after {time.time() - t0:.0f} s without a result; stderr tail:\n"
        + "\n".join(p.stderr.splitlines()[-25:]))
    res = json.loads(out.read_text())
    assert res["generate_file"] == generate.__file__, res  # the child ran this process's runner.generate
    assert res["n_layers"] == T2_LAYERS and res["nope_layers"] == list(range(4, T2_LAYERS, 5)), res
    assert res["resource_limit"] > T2_MARGIN and res["held"] == res["resource_limit"] - T2_MARGIN, res
    assert res["error"] is None, (
        f"run_cell raised {res['error']!r} after {res['decode_steps']} decode steps with {res['held']} of "
        f"{res['resource_limit']} buffers held (the pilot's crash; traceback {res.get('traceback')})")
    assert res["decode_steps"] == T2_STEPS, res
    assert res["records"] == [[T2_STEPS, "length"]] * T2_B, res


# ---------------------------------------------------------------------------------------------- T3 (output-neutral)


@pytest.mark.parametrize("sampling", ["greedy", "vendor"])
def test_t3_settle_step_leaves_records_and_steps_unchanged(k8_model, tmp_path, monkeypatch, sampling):
    """The same 5 items (B = 2, staggered max_tokens, three refills, the decode batch never empty in between) through
    run_cell with the settle step patched to a no-op (the pre-fix code path) and active: identical records (every field
    but the wall-clock times) and identical step logs (every field but the times). The no-op arm grows the offset
    graph; the active arm leaves none and called the step once per gen.next()."""
    cfg = GREEDY if sampling == "greedy" else dict(chat.sampling_for("kolibri"))
    if sampling == "vendor":
        assert cfg["temperature"] > 0, cfg
    its = _items(5, max_tokens=lambda k: (260, 200, 170, 150, 90)[k], seed=500)
    real = getattr(generate, "_settle_batch_offsets", None)
    n_settle = {"n": 0}

    def counted(gen):
        n_settle["n"] += 1
        if real is not None:
            real(gen)

    arms = {}
    for arm, hook in (("noop", lambda gen: None), ("active", counted)):
        monkeypatch.setattr(generate, "_settle_batch_offsets", hook, raising=False)
        probe = OffsetProbe()
        d = tmp_path / arm
        summ = _run(k8_model, d / "t3.jsonl", its, B=2, cap=512, cfg=cfg, seed=1683829245,
                    steps_path=d / "t3.steps.jsonl", on_step=probe)
        arms[arm] = {"summary": summ, "records": _records(d / "t3.jsonl"), "steps": _steps(d / "t3.steps.jsonl"),
                     "probe": probe}
    noop, active = arms["noop"], arms["active"]
    assert noop["summary"]["n_done"] == 5 and active["summary"]["n_done"] == 5
    assert [noop["records"][k]["completion_tokens"] for k in sorted(noop["records"])] == [260, 200, 170, 150, 90]
    diff = sorted(k for k in noop["records"] if noop["records"][k] != active["records"].get(k))
    assert not diff and set(noop["records"]) == set(active["records"]), f"records differ for {diff}"
    assert noop["steps"] == active["steps"], "the step logs differ"
    if sampling == "vendor":
        assert len(set(noop["records"]["q000"]["completion_ids"])) > 50  # the sampler was in effect
    calls = active["probe"].calls
    assert n_settle["n"] == calls, f"run_cell called the settle step {n_settle['n']} times in {calls} gen.next() calls"
    n = len(noop["probe"].rows)
    assert min(noop["probe"].rows[-1]["nodes"]) >= n, "the no-op arm is not the pre-fix path (no chain grew)"
    assert all(not any(r["nodes"]) for r in active["probe"].rows), "the active arm left a graph behind an offset"
