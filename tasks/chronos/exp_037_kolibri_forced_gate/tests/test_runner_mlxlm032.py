"""The runner under mlx-lm 0.32.0 (exp_037 decision F2; DESIGN §6.3, §6.5;
G1 item 2; W0r items 1, 2, 4 and 6).

- the one BatchGenerator construction, runner.generate.make_batch_generator,
  passes BUILD_SPEC item 27's registered arguments (completion_batch_size B,
  prefill_batch_size min(B, 8), prefill_step_size 2,048, max_kv_size None)
  at B = 1, 2, 4, 8 and 16, and run_cell builds its generator only through it;
- the step log's prompt tokens and seconds come from BatchGenerator.stats()
  windows (#1829; 0.31.3's _prompt_tokens_counter / _prompt_time_counter are
  gone) and sum to the generator's totals over a tiny cell;
- every private BatchGenerator and cache attribute the kit reads exists, so a
  future mlx-lm change fails here, loudly;
- a Kolibri load goes through port/convert.py:model_file_trust (#1385,
  CVE-2026-5843: trust_remote_code=True only for the kit's own kolibri1.py);
  a stale port file is refused before any load;
- runner/sampler.py imports nothing from mlx_lm.sample_utils (#1372, #1825,
  #1912 do not reach the kit) and run_cell passes no logits processor (#1777).
"""

from __future__ import annotations

import ast
import importlib
import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from exp036_helpers import import_sibling  # noqa: E402

import tiny_checkpoint as tc  # noqa: E402
from runner import generate as rg  # noqa: E402
from runner import jsonl  # noqa: E402
from runner.common import EXP_DIR  # noqa: E402

EOS = [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]
GREEDY = {"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0}
G = importlib.import_module("mlx_lm.generate")  # the module (the package attribute mlx_lm.generate is a function)


class StubTok:
    unk_token_id = None
    eos_token_ids = set(EOS)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": 1000, "</think>": 1001}.get(t)


@pytest.fixture(scope="module")
def model(tiny_vendor_dir):
    harness = import_sibling("port_harness")
    return harness.load_port(tiny_vendor_dir, float32=True)


def _greedy():
    import mlx.core as mx

    return lambda x: mx.argmax(x, axis=-1)


def items(n, base=11, step=7, max_tokens=None):
    out = []
    for k in range(n):
        it = {"id": f"q{k:03d}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k + 100:064x}",
              "prompt_ids": tc.random_ids(base + step * k, seed=40 + k), "prompt_text": "user turn"}
        if max_tokens is not None:
            it["max_tokens"] = max_tokens(k)
        out.append(it)
    return out


class Spy:
    """Wraps the real generator make_batch_generator returns: records the call and every insert, and keeps one
    outer stats() window open from construction to close (the cell's totals; nested windows are independent)."""

    def __init__(self, gen, call):
        self.gen = gen
        self.call = call
        self.inserts = []
        self._outer = gen.stats()
        self.total = self._outer.__enter__()

    def __getattr__(self, name):
        return getattr(self.gen, name)

    def insert(self, *a, **k):
        self.inserts.append(dict(k))
        return self.gen.insert(*a, **k)

    def close(self):
        self._outer.__exit__(None, None, None)
        self.gen.close()


@pytest.fixture
def spy(monkeypatch):
    made = []
    real = rg.make_batch_generator

    def factory(*a, **k):
        s = Spy(real(*a, **k), (a, k))
        made.append(s)
        return s

    monkeypatch.setattr(rg, "make_batch_generator", factory)
    return made


def run(model, path, its, B, cap=16, **kw):
    w = jsonl.open_run(path, {"cell": "t", "B": B}, aborted_root=path.parent / "aborted",
                       private_root=path.parent / "private")
    try:
        return rg.run_cell(model, StubTok(), its, "K8", "mmlu_en", "high", cap, B, 37, w, "kolibri",
                           sampler_cfg=GREEDY, eos=EOS, **kw)
    finally:
        w.close()


# ------------------------------------------------------- the one construction


@pytest.mark.parametrize("B", [1, 2, 4, 8, 16])
def test_make_batch_generator_passes_the_registered_arguments(monkeypatch, model, B):
    """G1 item 2: BUILD_SPEC item 27's arguments, as passed and as the generator holds them."""
    captured = {}

    class Fake:
        def __init__(self, m, **k):
            captured.update(model=m, **k)

    monkeypatch.setattr(G, "BatchGenerator", Fake)
    sampler = _greedy()
    rg.make_batch_generator(model, B, [7, 9], 123, sampler)
    m = captured.pop("model")
    assert getattr(m, "_exp036_fp32", False) and m.inner is model  # the fp32-logit wrapper (item 26, C6)
    assert captured == {"max_tokens": 123, "stop_tokens": [[7], [9]], "sampler": sampler,
                        "completion_batch_size": B, "prefill_batch_size": min(B, 8), "prefill_step_size": 2048,
                        "max_kv_size": None}
    monkeypatch.undo()
    gen = rg.make_batch_generator(model, B, EOS, 32, sampler)
    try:
        assert (gen.completion_batch_size, gen.prefill_batch_size, gen.prefill_step_size, gen.max_kv_size) == (
            B, min(B, 8), 2048, None)
        assert gen.logits_processors == [] and gen.max_tokens == 32
    finally:
        gen.close()


def test_make_batch_generator_refuses_B_below_one(model):
    with pytest.raises(ValueError):
        rg.make_batch_generator(model, 0, EOS, 8, _greedy())


def _batchgenerator_calls(path: Path) -> list[str]:
    """Names of the functions in `path` that call BatchGenerator(...) (any spelling of the callee)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
                    if name == "BatchGenerator":
                        out.append(fn.name)
    return out


def test_the_runner_constructs_its_batch_generator_in_one_place():
    calls = {p.relative_to(EXP_DIR).as_posix(): _batchgenerator_calls(p)
             for p in sorted((EXP_DIR / "runner").glob("*.py"))}
    calls = {k: v for k, v in calls.items() if v}
    assert calls == {"runner/generate.py": ["make_batch_generator"]}


def test_run_cell_builds_its_generator_through_make_batch_generator(model, spy, tmp_path):
    summ = run(model, tmp_path / "c.jsonl", items(5), B=4, cap=12)
    assert summ["n_done"] == 5 and len(spy) == 1
    (m, B, eos, cap, sampler), kw = spy[0].call
    assert m is model and B == 4 and eos == EOS and cap == 12 and callable(sampler)
    assert kw == {"prefill_step_size": rg.PREFILL_STEP_SIZE} and rg.PREFILL_STEP_SIZE == 2048


# ------------------------------------------------------------- the step log


def test_step_log_prompt_tokens_and_seconds_are_the_stats_totals(model, spy, tmp_path):
    """G1 item 2: the per-step prompt tokens and seconds come from stats() windows around each next(); over the cell
    they sum to the generator's own totals (an outer window held from construction to close)."""
    its = items(9, base=9, step=13, max_tokens=lambda k: 3 + (5 * k) % 11)
    summ = run(model, tmp_path / "c.jsonl", its, B=3, cap=16, steps_path=tmp_path / "steps.jsonl")
    assert summ["n_done"] == 9
    total = spy[0].total
    steps = [json.loads(x) for x in (tmp_path / "steps.jsonl").read_text().splitlines()]
    dec = [s for s in steps if s["type"] == "step"]
    assert sum(s["prompt_tokens"] for s in dec) == total.prompt_tokens == sum(len(i["prompt_ids"]) - 1 for i in its)
    assert sum(s["prompt_seconds"] for s in dec) == pytest.approx(total.prompt_time, abs=1e-6 * len(dec) + 1e-9)
    assert total.prompt_time > 0
    assert sum(s["n_live"] for s in dec) == total.generation_tokens
    assert all(s["step_seconds"] >= 0 and s["call_seconds"] >= s["prompt_seconds"] - 1e-6 for s in dec)
    tree = ast.parse(Path(rg.__file__).read_text(encoding="utf-8"))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"_prompt_tokens_counter", "_prompt_time_counter"} and "stats" in attrs


# --------------------------------------------- private attributes the kit reads


def test_every_private_attribute_the_kit_reads_exists(model):
    """G1 item 2: _generation_batch, _prompt_batch, _unprocessed_sequences, GenerationBatch.prompt_cache,
    BatchKVCache._idx, BatchRotatingKVCache._offset and both batch caches' left_padding (read by run_cell's step
    log and in-flight count, _padded_len, gate/checks/g5_generation.max_live and tests/test_runner_tiny.py)."""
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache, BatchRotatingKVCache

    gen = rg.make_batch_generator(model, 3, EOS, 6, _greedy())
    try:
        assert hasattr(gen, "stats") and callable(gen.stats)
        gen.insert([tc.random_ids(n, seed=n) for n in (21, 9, 14, 30)], max_tokens=[6, 6, 6, 6])
        assert len(gen._unprocessed_sequences) == 4 and len(gen._prompt_batch) == 0 and len(gen._generation_batch) == 0
        for _ in range(20):
            gen.next()
            if len(gen._generation_batch):
                break
        gb = gen._generation_batch
        assert isinstance(gb, G.GenerationBatch) and len(gb) == 3 and len(gen._unprocessed_sequences) == 1
        assert isinstance(gen._prompt_batch, G.PromptProcessingBatch)
        caches = gb.prompt_cache
        kinds = {type(c) for c in caches}
        assert kinds == {BatchKVCache, BatchRotatingKVCache}
        for c in caches:
            assert isinstance(c.left_padding, mx.array) and c.left_padding.shape == (3,)
            if type(c) is BatchKVCache:
                assert isinstance(c._idx, int) and c._idx > 0
            else:
                assert isinstance(c._offset, int) and c._offset > 0
        full = next(c for c in caches if type(c) is BatchKVCache)
        assert rg._padded_len(gen) == full._idx
    finally:
        gen.close()


# ------------------------------------------------------- model_file loads (#1385)


@pytest.fixture
def k8_copy(tiny_real_val, tmp_path):
    d = tmp_path / "builds" / tiny_real_val.k8.name
    shutil.copytree(tiny_real_val.k8, d)
    return d


@pytest.fixture
def loader(monkeypatch, k8_copy):
    """runner/run.py Loaded.get on a tiny K8 conversion, with the chat checks stubbed and mlx_lm.load replaced by a
    spy that records its keyword arguments and loads strictly through mlx_lm.utils.load_model."""
    import mlx_lm
    from mlx_lm.utils import load_model

    run_mod = import_sibling("runner.run")
    monkeypatch.setattr(run_mod, "arm_dir", lambda arm: k8_copy)
    monkeypatch.setattr(run_mod.chat, "check_sampling", lambda fam, d: {"match": True})
    monkeypatch.setattr(run_mod.chat, "template_parity", lambda fam, d: None)
    monkeypatch.setattr(run_mod.chat, "eos_ids", lambda d: list(EOS))
    loads = []

    def spy_load(path, **kw):
        loads.append((path, dict(kw)))
        m, _ = load_model(Path(path), lazy=True, strict=True, **kw)
        return m, StubTok()

    monkeypatch.setattr(mlx_lm, "load", spy_load)
    return run_mod, loads


def test_a_kolibri_load_goes_through_model_file_trust(loader, k8_copy):
    run_mod, loads = loader
    from port import convert

    cfg = json.loads((k8_copy / "config.json").read_text())
    assert cfg["model_file"] == "kolibri1.py"
    model, tok, meta = run_mod.Loaded().get("K8")
    assert loads == [(str(k8_copy), {"trust_remote_code": True})]
    assert meta["port_sha256"] == convert.check_port_file(k8_copy) and meta["model_manifest_sha256"]
    # The model was built by the directory's own kolibri1.py (mlx-lm's model_file path), the kit's port.
    assert type(model).__name__ == "Model" and hasattr(model, "make_cache") and model.args.sliding_window > 0


def test_a_stale_port_file_is_refused_before_any_load(loader, k8_copy):
    run_mod, loads = loader
    with open(k8_copy / "kolibri1.py", "a", encoding="utf-8") as f:
        f.write("\n# edited after the conversion\n")
    with pytest.raises(SystemExit) as e:
        run_mod.Loaded().get("K8")
    assert "port file" in str(e.value) or "kolibri1.py" in str(e.value)
    assert loads == []
    # load_trust itself refuses the same directory, so no caller can load it.
    with pytest.raises(SystemExit):
        run_mod.load_trust("K8", k8_copy)


def test_built_in_model_types_load_without_the_flag_and_foreign_model_files_are_refused(tmp_path):
    run_mod = import_sibling("runner.run")
    peer = tmp_path / "peer"
    peer.mkdir()
    (peer / "config.json").write_text(json.dumps({"model_type": "qwen3_moe"}))
    assert run_mod.load_trust("Q36-8", peer) == {}
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "config.json").write_text(json.dumps({"model_type": "x", "model_file": "evil.py"}))
    (foreign / "evil.py").write_text("raise SystemExit('executed')\n")
    with pytest.raises(SystemExit) as e:
        run_mod.load_trust("G8", foreign)
    assert "evil.py" in str(e.value) and "executed" not in str(e.value)


# ------------------------------------------- samplers and logits processors


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    return mods


def test_the_sampler_imports_nothing_from_mlx_lm_sample_utils():
    """W0r item 6: mlx-lm 0.32.0's sampler changes (#1372, #1825, #1912) cannot reach the kit's sampling."""
    for f in ("sampler.py", "generate.py", "run.py"):
        mods = _imports(EXP_DIR / "runner" / f)
        assert not any("sample_utils" in m for m in mods), (f, sorted(m for m in mods if "sample_utils" in m))


def test_run_cell_passes_no_logits_processor(model, spy, tmp_path):
    """W0r item 4: #1777 (prompt tokens in the processors' history) cannot reach the runner: no processor is
    passed at construction or at insert, and the generator holds none."""
    tree = ast.parse(Path(rg.__file__).read_text(encoding="utf-8"))
    kws = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert "logits_processors" not in kws
    run(model, tmp_path / "c.jsonl", items(4), B=2, cap=8)
    s = spy[0]
    assert s.gen.logits_processors == [] and s.inserts and all(
        set(k) <= {"max_tokens"} for k in s.inserts), s.inserts
