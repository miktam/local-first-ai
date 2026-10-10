"""exp_038 MLXGatedBackend on tiny builds (DESIGN rev 2 §0b.2-3, §2.1).

B1 one fresh batch generator per call, and the Amendment 2 settle once per decode step, with no graph left
   behind any BatchKVCache offset (oracle: MLX's own graph export, as E37 T1).
B2 the fix matters: in a subprocess holding resource_limit - 15,000 Metal buffers, one 4,000-token call on a
   50-layer / 10-NoPE build completes with the settle and raises the pilot's error with it patched to a no-op
   (oracle: MLX's own limit, as E37 T2).
B3 no build-up across calls: under the same margin, 40 consecutive 500-token calls complete with the settle
   patched out, because each call's generator and caches are freed (the fresh-generator rule on its own).
B4 the penalty: the answer call's processor equals llama.cpp's penalties rule on random logits and a token
   window that includes prompt tokens; the gate call gets none; a 1.0 penalty installs none.
B5 the reasoning cap and answer cap logic at effort medium, on synthetic token streams.
B6 the reply has run_eval.ollama_chat's shape, so the vendored glue can call it unchanged.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm.models.cache import BatchKVCache  # noqa: E402

import mlx_backend as mb  # noqa: E402  (kit/)
import port_harness  # noqa: E402  (E37 tests/)
import tiny_checkpoint as tc  # noqa: E402
from runner import generate  # noqa: E402  (E37)

NO_EOS = 5000  # outside the tiny vocabulary: every call runs to its max_tokens, so step counts are fixed
THINK, UNTHINK = 1000, 1001
KIT = Path(mb.__file__).resolve().parent


class StubTok:
    """E37 tests' stub over the tiny vocabulary (ids 1000/1001 play <think>/</think>)."""

    unk_token_id = None
    eos_token_ids = {NO_EOS}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": THINK, "</think>": UNTHINK, "<|channel>": 1002, "<channel|>": 1003}.get(t)


def stub_render(n_prompt: int, seed: int = 700):
    def render(family, tok, messages, effort):
        ids = tc.random_ids(n_prompt, seed=seed)
        return f"user turn ({effort})", ids, f"{seed:064x}"
    return render


@pytest.fixture(scope="module")
def k8(tiny_set):
    return port_harness.load_port(tiny_set.k8)


def backend(model, n_prompt=40, **kw):
    b = mb.MLXGatedBackend.tiny(model, StubTok(), [NO_EOS], **kw)
    b._render = stub_render(n_prompt)
    return b


ANSWER = {"temperature": 0.0, "seed": 42, "num_predict": 300, "top_k": 20, "top_p": 0.8, "min_p": 0.05,
          "repeat_penalty": 1.05, "presence_penalty": 0.0, "repeat_last_n": 256, "num_ctx": 4096}
GATE = {"temperature": 0.0, "seed": 42, "num_predict": 4, "num_ctx": 1024}
_RECT = re.compile(r"shape=rectangle")


def _graph_nodes(arr) -> int:
    buf = io.StringIO()
    mx.export_to_dot(buf, arr)
    return len(_RECT.findall(buf.getvalue()))


# ------------------------------------------------------------------------------------------------------ B1
def test_b1_fresh_generator_per_call_and_settle_every_step(k8, monkeypatch):
    made, settles = [], []
    real_make, real_settle = generate.make_batch_generator, generate._settle_batch_offsets
    monkeypatch.setattr(generate, "make_batch_generator", lambda *a, **k: made.append(1) or real_make(*a, **k))
    monkeypatch.setattr(generate, "_settle_batch_offsets", lambda g: settles.append(1) or real_settle(g))
    b = backend(k8)
    nodes = []

    def probe(gen):
        gb = gen._generation_batch
        if len(gb):
            nodes.append([_graph_nodes(c.offset) for c in gb.prompt_cache if isinstance(c, BatchKVCache)])

    b._on_step = probe
    r1 = b.chat("K4", [{"role": "user", "content": "x"}], ANSWER)
    r2 = b.chat("K4", [{"role": "user", "content": "x"}], GATE)
    assert len(made) == 2, made  # one generator per call
    x1, x2 = r1["exp038"], r2["exp038"]
    assert x1["completion_tokens"] == 300 and x2["completion_tokens"] == 4
    assert len(settles) == x1["decode_calls"] + x2["decode_calls"]  # once per gen.next()
    assert nodes and all(n == [0] * len(n) for n in nodes), "a graph was left behind a BatchKVCache offset"
    assert {len(n) for n in nodes} == {2}  # both NoPE layers of the 10-layer build were looked at


# ------------------------------------------------------------------------------------------------- B2, B3
_CHILD = textwrap.dedent("""
    import json, sys, time, traceback
    sys.dont_write_bytecode = True
    kit, e37, ckpt, out, mode, n_calls, n_tok, margin = sys.argv[1:9]
    n_calls, n_tok, margin = int(n_calls), int(n_tok), int(margin)
    sys.path[:0] = [kit, e37, e37 + "/tests", e37 + "/tools"]
    from pathlib import Path
    import mlx.core as mx
    import port_harness, tiny_checkpoint as tc
    from runner import generate
    import mlx_backend as mb

    class StubTok:
        unk_token_id = None
        eos_token_ids = {5000}
        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"<{int(i)}>" for i in ids)
        def convert_tokens_to_ids(self, t):
            return {"<think>": 1000, "</think>": 1001}.get(t)

    if mode == "no_settle":
        generate._settle_batch_offsets = lambda gen: None
    res = {"mode": mode}
    model = port_harness.load_port(Path(ckpt))
    mx.eval(model.parameters())
    res["nope_layers"] = [i for i, l in enumerate(model.layers) if not l.use_sliding]
    limit = int(mx.device_info()["resource_limit"])
    mx.clear_cache()
    hold = [mx.array(i) for i in range(limit - margin)]
    mx.eval(hold)
    res.update(resource_limit=limit, held=len(hold))
    b = mb.MLXGatedBackend.tiny(model, StubTok(), [5000])
    b._render = lambda f, t, m, e: ("user turn", tc.random_ids(16, seed=900), "0" * 64)
    opts = {"temperature": 0.0, "num_predict": n_tok, "repeat_penalty": 1.0, "num_ctx": 4096}
    res["calls_done"], res["error"] = 0, None
    t0 = time.time()
    try:
        for _ in range(n_calls):
            r = b.chat("K4", [{"role": "user", "content": "x"}], opts)
            assert r["exp038"]["completion_tokens"] == n_tok, r["exp038"]
            res["calls_done"] += 1
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    res["seconds"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(res))
""")


@pytest.fixture(scope="module")
def real50(tmp_path_factory):
    """E37 T2's 50-layer / 10-NoPE build (8 bits, group 64), written with E37's own writers."""
    import test_runner_offset_leak as t2  # E37 tests/

    return t2._write_real50(tmp_path_factory.mktemp("real50"))


def _child(real50, tmp_path, mode, n_calls, n_tok, margin=15_000, timeout=1500):
    out = tmp_path / f"{mode}_{n_calls}x{n_tok}.json"
    cmd = [sys.executable, "-c", _CHILD, str(KIT), str(mb.E37), str(real50), str(out), mode, str(n_calls),
           str(n_tok), str(margin)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    assert p.returncode == 0 and out.is_file(), p.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res["nope_layers"] == list(range(4, 50, 5)) and res["held"] == res["resource_limit"] - margin, res
    return res


def test_b2_settle_prevents_the_pilot_crash(real50, tmp_path):
    ok = _child(real50, tmp_path, "settle", 1, 4000)
    assert ok["error"] is None and ok["calls_done"] == 1, ok
    bad = _child(real50, tmp_path, "no_settle", 1, 4000)
    assert bad["calls_done"] == 0 and bad["error"] and "Resource limit" in bad["error"], bad


def test_b3_no_buildup_across_calls(real50, tmp_path):
    """40 x 500 = 20,000 decode steps in one process, 10 NoPE layers: a single long-lived generator without the
    settle would leave ~200,000 buffers, far beyond the 15,000 margin. Fresh generators free each call's chain."""
    res = _child(real50, tmp_path, "no_settle", 40, 500)
    assert res["error"] is None and res["calls_done"] == 40, res


# ------------------------------------------------------------------------------------------------------ B4
def llama_cpp_penalty(logits: np.ndarray, history: list[int], penalty: float, last_n: int) -> np.ndarray:
    """llama.cpp llama_sampler_penalties at presence = frequency = 0: once per distinct token in the window."""
    out = logits.copy()
    for t in set(history[-last_n:]):
        out[t] = out[t] * penalty if out[t] <= 0 else out[t] / penalty
    return out


def test_b4_penalty_rule_matches_llama_cpp():
    from mlx_lm.sample_utils import make_repetition_penalty

    rng = np.random.default_rng(38)
    for trial in range(20):
        V = 1024
        logits = rng.normal(0, 3, V).astype(np.float32)
        hist = [int(x) for x in rng.integers(0, V, size=int(rng.integers(1, 600)))]  # duplicates included
        proc = make_repetition_penalty(1.05, 256)
        got = np.array(proc(mx.array(hist, dtype=mx.int32), mx.array(logits[None, :]))[0])
        want = llama_cpp_penalty(logits, hist, 1.05, 256)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6, err_msg=f"trial {trial}")


def test_b4_processor_installed_only_on_penalised_answer_calls(k8, monkeypatch):
    seen = []
    real = generate.make_batch_generator

    def spy(*a, **k):
        g = real(*a, **k)
        ins = g.insert

        def insert(prompts, **kw):
            seen.append(kw.get("logits_processors"))
            return ins(prompts, **kw)

        g.insert = insert
        return g

    monkeypatch.setattr(generate, "make_batch_generator", spy)
    b = backend(k8)
    b.chat("K4", [{"role": "user", "content": "x"}], GATE)
    b.chat("K4", [{"role": "user", "content": "x"}], dict(ANSWER, num_predict=8))
    b.chat("K4", [{"role": "user", "content": "x"}], dict(ANSWER, num_predict=8, repeat_penalty=1.0))
    assert seen[0] is None and seen[2] is None
    assert seen[1] is not None and len(seen[1]) == 1 and len(seen[1][0]) == 1


def _ids(content: str) -> list[int]:
    return [int(x) for x in re.findall(r"<(\d+)>", content)]


def test_b4_penalty_window_holds_the_prompt(k8):
    """With a huge penalty over a window longer than prompt + completion, no generated token may repeat a prompt
    token or an earlier generated one: mlx-lm 0.32.0 seeds the processor's TokenBuffer with the prompt, as G26's
    engine counts prompt tokens (W0r). Without the penalty, the greedy continuation does repeat itself."""
    n_prompt = 200
    prompt = set(tc.random_ids(n_prompt, seed=700))
    b = backend(k8, n_prompt=n_prompt)
    plain = _ids(b.chat("K4", [], dict(ANSWER, num_predict=40, repeat_penalty=1.0))["message"]["content"])
    strong = _ids(b.chat("K4", [], dict(ANSWER, num_predict=40, repeat_penalty=1e9, repeat_last_n=4096))["message"]["content"])
    assert len(plain) == len(strong) == 40
    assert plain != strong
    assert not (set(strong) & prompt), sorted(set(strong) & prompt)
    assert len(set(strong)) == len(strong)


# ------------------------------------------------------------------------------------------------------ B5
def test_b5_reasoning_state():
    b = mb.MLXGatedBackend.__new__(mb.MLXGatedBackend)
    b.open_id, b.close_id, b._blank = THINK, UNTHINK, {}
    b._gen = generate
    b.tokenizer = StubTok()
    assert b._reasoning_state([5, 6]) == (True, 0, 2)  # no segment opened
    assert b._reasoning_state([THINK, 1, 2, 3]) == (False, 3, 0)
    assert b._reasoning_state([THINK, 1, 2, UNTHINK, 7, 8]) == (True, 2, 2)


def test_b5_medium_needs_a_cap(k8):
    with pytest.raises(ValueError):
        mb.MLXGatedBackend.tiny(k8, StubTok(), [NO_EOS], effort_answer="medium")
    with pytest.raises(ValueError):
        mb.MLXGatedBackend.tiny(k8, StubTok(), [NO_EOS], effort_answer="high", reasoning_cap=1024)


def test_b5_gate_call_never_reasons(k8):
    b = backend(k8, effort_answer="medium", reasoning_cap=64)
    efforts = []
    real = b._render
    b._render = lambda f, t, m, e: efforts.append(e) or real(f, t, m, e)
    b.chat("K4", [], GATE)
    b.chat("K4", [], dict(ANSWER, num_predict=8))
    assert efforts == ["none", "medium"]


# ------------------------------------------------------------------------------------------------------ B6
def test_b6_reply_shape_for_the_glue(k8):
    b = backend(k8)
    r = b.chat("K4", [{"role": "user", "content": "x"}], dict(ANSWER, num_predict=12))
    assert isinstance(r["message"]["content"], str)
    assert r["prompt_eval_count"] == 40 and r["eval_count"] == 12
    assert r["exp038"]["finish_reason"] == "length" and r["exp038"]["binding"] == {"tiny": True}
    with pytest.raises(ValueError):
        b.chat("K4", [], dict(ANSWER, temperature=0.7))
    with pytest.raises(ValueError):
        b.chat("K4", [], dict(ANSWER, presence_penalty=0.5))
