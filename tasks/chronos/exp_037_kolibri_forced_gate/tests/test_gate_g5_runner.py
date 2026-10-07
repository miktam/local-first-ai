# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5c).
"""The G5 family's mechanics: gate/checks/g5_runner.py and the batched path of
gate/checks/g5_generation.py (DESIGN §3.12-§3.14, §2.9; G1 item 8).

* The 26-prompt layout of G5-BP-lean: the real one from thresholds G5BP.layout
  (16,916 prompt tokens, 792 generated positions) and the tiny one of §5.1.
* The admission loop on a scripted trace: first wave = the first B inserted,
  mid-run = inserted after the first finish, max_live counted over the
  prefill and decode batches.
* G1 item 8 on the tiny validation build (seed 29), K8 and K4: at B = 8 and 16
  with the tiny 26-prompt layout the labels are exactly that, admitted_mid_run
  >= 1 and max_live <= B; the runner's BatchGenerator is built by the one
  function runner.generate.make_batch_generator (an identity check, with the
  registered prefill_batch_size = min(B, 8)); with arms (K8, K4), every
  sequence that rules.py's R-anchor, R-greedy and (d) need for either arm has an
  R1 row in phase 6, and every log-prob file a phase JSON names exists with its
  sha256 (and is refused when it does not).
* Log-prob files and their sha256 round-trip; writers stay under
  work37_dir(); records hold relative paths only.
* G5-D32's series on tiny, in the shape rules.g5_d32 reads, with mutant 20
  caught by the decode leg.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import tiny_checkpoint as tc  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
RUN = "20261006T000000Z"


# ---------------------------------------------------------------------------
# The 26-prompt layout
# ---------------------------------------------------------------------------


def _real_like_textset():
    """A TextSet with the real profile and synthetic ids (T1-T8 of 1,536 tokens,
    T9 of 16,384): the layout only reads lengths and slices."""
    from gate import common, textset

    th = common.load_thresholds()
    rng = np.random.default_rng(0)
    ids = {t: [int(x) for x in rng.integers(0, 127000, 1536)] for t in textset.ALL8}
    t9 = [int(x) for x in rng.integers(0, 127000, 16384)]
    return textset.TextSet(textset.real_profile(th), ids, t9, [], {}, "k", {}), th


def test_prompts26_real_layout():
    from gate.checks import g5_runner as R

    ts, th = _real_like_textset()
    lay = th["G5BP"]["layout"]
    p = R.prompts26(ts, th)
    assert [e["seq"] for e in p] == ([f"A{j:02d}" for j in range(12)] + ["L1", "L2"] + [f"B{j:02d}" for j in range(12)])
    order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    L = [37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100]
    M = [48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44]
    assert lay["lengths"] == L and lay["max_tokens"] == M  # DESIGN §3.14, exp_036 run_gate.py l.488-489
    for j in range(12):
        a, b = p[j], p[14 + j]
        assert (a["text"], len(a["ids"]), a["max_tokens"]) == (order[j % 8], L[j], M[j])
        assert (b["text"], len(b["ids"]), b["max_tokens"]) == (order[(j + 4) % 8], L[j], M[j])
        assert a["ids"] == ts.ids[order[j % 8]][:L[j]] and b["ids"] == ts.ids[order[(j + 4) % 8]][:L[j]]
        assert a["text"] != b["text"]  # B_j is a different text from A_j
    l1, l2 = p[12], p[13]
    assert (l1["ids"], l1["max_tokens"]) == (ts.t9[0:2100], 24)
    assert (l2["ids"], l2["max_tokens"]) == (ts.t9[4096:7096], 40)
    assert sum(len(e["ids"]) for e in p) == 16916 == lay["prompt_tokens"]
    assert sum(e["max_tokens"] for e in p) == 792 == lay["generated_positions"]
    assert sum(len(e["ids"]) for e in p) + 792 == 17708  # §3.14: 17,708 tokens per configuration
    # The 2,048-token prefill chunk: L1 and L2 cross it, and B = 8's first wave is 8 x <= 1,100 rows wide.
    assert len(l1["ids"]) > 2048 and len(l2["ids"]) > 2048


def test_prompts26_tiny_layout():
    from gate import common, textset
    from gate.checks import g5_runner as R

    th = common.load_thresholds()
    ts = textset.make_tiny(513, 10, tc.VOCAB_SIZE)
    p = R.prompts26(ts, th)
    prof = ts.profile
    assert len(p) == 26 and [len(e["ids"]) for e in p[:12]] == list(prof.batch_lengths)
    assert [e["max_tokens"] for e in p[14:]] == list(prof.batch_max_tokens)
    l1, l2 = p[12], p[13]
    assert (l1["ids"], l1["max_tokens"]) == (ts.t9[0:100], 6)      # §5.1: crosses one 64-token chunk
    assert (l2["ids"], l2["max_tokens"]) == (ts.t9[128:300], 10)   # crosses two
    assert len(l1["ids"]) // prof.prefill_chunk == 1 and len(l2["ids"]) // prof.prefill_chunk == 2


def test_prompts26_refuses_a_short_text():
    from gate import textset
    from gate.checks import g5_runner as R

    ts, th = _real_like_textset()
    ts.t9 = ts.t9[:5000]  # L2 needs [4,096, 7,096)
    with pytest.raises(ValueError, match="L2"):
        R.prompts26(ts, th)
    ts2 = textset.make_tiny(513, 10, tc.VOCAB_SIZE)
    ts2.ids["T4"] = ts2.ids["T4"][:50]  # A_3 needs 110 tokens of T4
    with pytest.raises(ValueError, match="A03"):
        R.prompts26(ts2, th)


# ---------------------------------------------------------------------------
# The admission loop on a scripted trace
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, uid, token, lp, finish):
        self.uid, self.token, self.logprobs, self.finish_reason = uid, token, lp, finish


class FakeGen:
    """Scripted BatchGenerator: an inserted sequence spends `prefill` next()
    calls in the prompt batch, then produces one token per next() call until
    its max_tokens; responses carry numpy log-probs [V]."""

    V = 5

    def __init__(self, prefill: int = 1):
        self.prefill = prefill
        self._uid = 0
        self.seqs = {}
        self._generation_batch, self._prompt_batch = [], []
        self.closed = False
        self.inserts = []

    def insert(self, prompts, max_tokens):
        uids = list(range(self._uid, self._uid + len(prompts)))
        self._uid += len(prompts)
        for u, p, m in zip(uids, prompts, max_tokens):
            self.seqs[u] = {"left": self.prefill, "m": m, "n": 0, "plen": len(p)}
            self._prompt_batch.append(u)
        self.inserts.append(list(uids))
        return uids

    def next(self):
        resps = []
        for u in list(self._generation_batch):
            s = self.seqs[u]
            s["n"] += 1
            lp = np.log(np.full(self.V, 1.0 / self.V, dtype=np.float32))
            fin = "length" if s["n"] >= s["m"] else None
            resps.append(_Resp(u, (u + s["n"]) % self.V, lp, fin))
            if fin:
                self._generation_batch.remove(u)
        for u in list(self._prompt_batch):
            self.seqs[u]["left"] -= 1
            if self.seqs[u]["left"] <= 0:
                self._prompt_batch.remove(u)
                self._generation_batch.append(u)
        return [], resps

    def close(self):
        self.closed = True


def test_admission_labels_on_a_scripted_trace():
    """B = 3 over six prompts. Step 0 inserts 0, 1, 2 (the first wave); they
    decode from step 1. Prompt 1 (max_tokens 1) finishes at step 1; prompt 3 is
    inserted at step 2 (mid-run) as prompt 0 (2 tokens) finishes; prompt 4 is
    inserted at step 3 as prompt 3 finishes, prompt 5 at step 4; prompts 2, 4
    and 5 finish together at step 6."""
    from gate.checks import g5_generation as g5

    gen = FakeGen(prefill=1)
    prompts = [[1] * (3 + j) for j in range(6)]
    mt = [2, 1, 6, 1, 3, 2]
    seen = []
    res = g5.run_admission(gen, prompts, mt, 3, on_finish=lambda j, toks, lp: seen.append((j, len(toks), lp.shape)) or j)
    s = res["seqs"]
    assert [x["insert_step"] for x in s] == [0, 0, 0, 2, 3, 4]
    assert [x["insert_order"] for x in s] == [0, 1, 2, 3, 4, 5]
    assert [x["first_wave"] for x in s] == [True, True, True, False, False, False]
    assert [x["mid_run"] for x in s] == [False, False, False, True, True, True]
    assert [len(x["tokens"]) for x in s] == mt and all(x["finish_reason"] == "length" for x in s)
    assert [x["finish_step"] for x in s] == [2, 1, 6, 3, 6, 6]
    assert res["first_finish_step"] == 1 and res["admitted_mid_run"] == 3
    assert res["max_live"] == 3 and gen.closed
    assert gen.inserts == [[0, 1, 2], [3], [4], [5]]
    assert sorted(seen) == [(j, mt[j], (mt[j], FakeGen.V)) for j in range(6)]
    assert all(x["on_finish"] == j and x["logprobs"] is None for j, x in enumerate(s))


def test_admission_max_live_counts_the_prompt_batch():
    """With a three-call prefill, a mid-run admission sits in the prompt batch
    while the others decode: max_live counts both batches and stays <= B."""
    from gate.checks import g5_generation as g5

    gen = FakeGen(prefill=3)
    res = g5.run_admission(gen, [[1, 2]] * 5, [1, 4, 4, 2, 2], 3, keep_logprobs=True)
    assert res["max_live"] == 3 and res["admitted_mid_run"] == 2
    assert [x["first_wave"] for x in res["seqs"]] == [True, True, True, False, False]
    assert all(x["logprobs"].shape == (len(x["tokens"]), FakeGen.V) for x in res["seqs"])
    with pytest.raises(ValueError):
        g5.run_admission(FakeGen(), [[1]], [1, 2], 1)


def test_admission_at_B1_is_one_sequence_at_a_time():
    from gate.checks import g5_generation as g5

    gen = FakeGen(prefill=1)
    res = g5.run_admission(gen, [[1]] * 3, [2, 2, 2], 1)
    assert gen.inserts == [[0], [1], [2]] and res["max_live"] == 1
    assert [x["first_wave"] for x in res["seqs"]] == [True, False, False]
    assert [x["mid_run"] for x in res["seqs"]] == [False, True, True]


# ---------------------------------------------------------------------------
# Log-prob files
# ---------------------------------------------------------------------------


def test_logprob_files_round_trip(tmp_path, monkeypatch):
    from gate import common
    from gate.checks import g5_runner as R

    monkeypatch.setenv("EXP036_WORK", str(tmp_path / "work"))
    out = R.g5_out_dir(RUN, "K8", R.check_dir_name("g5_r1"))
    assert out == common.work37_dir() / "run" / RUN / "g5" / "K8" / "g5_r1"
    a = np.random.default_rng(1).standard_normal((7, 11)).astype(np.float32)
    e = R.write_logprobs(out, "G5_en_none_1", "runner", a)
    assert e["path"] == f"run/{RUN}/g5/K8/g5_r1/G5_en_none_1.runner.npy"
    assert e["shape"] == [7, 11] and e["dtype"] == "float32" and len(e["sha256"]) == 64
    assert not Path(e["path"]).is_absolute() and str(tmp_path) not in json.dumps(e)
    assert np.array_equal(R.load_logprobs(e), a)
    assert e["sha256"] == common.sha256_file(common.work37_dir() / e["path"])
    with pytest.raises(FileExistsError):  # never overwritten
        R.write_logprobs(out, "G5_en_none_1", "runner", a)
    p = common.work37_dir() / e["path"]
    raw = bytearray(p.read_bytes())
    raw[-1] ^= 1
    p.write_bytes(bytes(raw))
    with pytest.raises(R.G5DataError, match="sha256"):
        R.load_logprobs(e)
    p.unlink()
    with pytest.raises(R.G5DataError, match="missing"):
        R.load_logprobs(e)
    for bad in ("/abs/x.npy", "../x.npy", "run/../../x.npy"):
        with pytest.raises(R.G5DataError):
            R.logprob_path({"path": bad})
    with pytest.raises(ValueError, match="exp_037 writers stay under"):
        R.write_logprobs(tmp_path / "elsewhere", "s", "runner", a)
    with pytest.raises(ValueError):
        R.write_logprobs(out, "../s", "runner", a)
    with pytest.raises(ValueError):
        R.write_logprobs(out, "s", "logits", a)
    assert [R.check_dir_name("g5_r1", mutant=26), R.check_dir_name("g5_bp", B=8), R.check_dir_name("g5_bp", B=16),
            R.check_dir_name("g5_bp", mutant=27), R.check_dir_name("g5_bp", mutant=22)] == [
        "g5_r1_m26", "g5_bp_B8", "g5_bp_B16", "g5_bp_m27", "g5_bp_m22"]
    with pytest.raises(ValueError):
        R.g5_out_dir("..", "K8", "g5_r1")


def test_greedy_rows_measure_equals_greedy_vs_ref():
    from gate.checks import g5_generation as g5

    rng = np.random.default_rng(3)
    V, P = 40, 5
    conts, by_id, items = [], {}, []
    for k in range(3):
        n = 6 + k
        logits = rng.standard_normal((P + n, V)).astype(np.float32) * (1 + 3 * k)
        toks = [int(x) for x in logits[P - 1:P - 1 + n].argmax(-1)]
        toks[0] = (toks[0] + 1) % V
        conts.append({"id": f"p{k}", "prompt_tokens": P, "continuation": toks, "finished": "length"})
        by_id[f"p{k}"] = logits
        items.append((f"p{k}", logits[P - 1:P - 1 + n], toks, "length"))
    assert g5.greedy_rows_vs_ref(items, 2.0) == g5.greedy_vs_ref(conts, {}, by_id, 2.0)
    with pytest.raises(ValueError):
        g5.greedy_rows_vs_ref([("x", np.zeros((2, V)), [1, 2, 3], "length")], 2.0)


# ---------------------------------------------------------------------------
# One construction (G1 item 8, third bullet)
# ---------------------------------------------------------------------------


def _bg_calls(path: Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name == "BatchGenerator":
                out.append(node.lineno)
    return out


def test_no_batch_generator_is_built_under_gate_checks():
    calls = {p.name: _bg_calls(p) for p in sorted((EXP / "gate" / "checks").glob("*.py"))}
    assert not {k: v for k, v in calls.items() if v}, calls


def test_the_g5_modules_only_measure():
    """DESIGN §9.2: the decisions are gate/rules.py's. g5_generation's
    parity_bound and behaviour_verdict are rules.py's own functions (names kept
    for exp_036-era callers), and neither G5 module writes a "pass" key."""
    from gate import rules
    from gate.checks import g5_generation as g5

    assert g5.parity_bound is rules.parity_bound and g5.behaviour_verdict is rules.behaviour_verdict
    for name in ("g5_generation.py", "g5_runner.py"):
        tree = ast.parse((EXP / "gate" / "checks" / name).read_text(encoding="utf-8"))
        defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert not defs & {"parity_bound", "behaviour_verdict", "evaluate"}, name
        keys = [k.value for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys
                if isinstance(k, ast.Constant)]
        subs = [n.slice.value for n in ast.walk(tree) if isinstance(n, ast.Subscript)
                and isinstance(n.slice, ast.Constant)]
        assert "pass" not in keys + subs, name


# ---------------------------------------------------------------------------
# Tiny runs: K8 and K4 of the validation build (seed 29), shared by the tests below
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def g5_runs(tiny_real_val, tmp_path_factory):
    """Phase 4 (K8: greedy, G5-R1, mutant 26, G5-BP-lean B 8 and 16, mutant 27 (the
    control, decision (b)) and probe 22 (tiny mode))
    and phase 5 (K4: G5-R1, G5-BP-lean B 8 and 16) on the tiny validation build
    with the tiny profile (prefill chunk 64, fp32 activations as the tiny gate
    runs), no stop ids (every sequence runs to its max_tokens), each phase JSON
    through a JSON round trip, as the phase subprocesses hand them on. Every
    make_batch_generator call is recorded."""
    import mlx.core as mx

    from gate import common, textset
    from gate.checks import g5_generation as g5
    from gate.checks import g5_runner as R
    from runner import generate

    mp = pytest.MonkeyPatch()
    work = tmp_path_factory.mktemp("g5_work")
    mp.setenv("EXP036_WORK", str(work))
    calls = []
    real = generate.make_batch_generator

    def spy(model, B, eos, max_tokens, sampler, prefill_step_size=generate.PREFILL_STEP_SIZE):
        gen = real(model, B, eos, max_tokens, sampler, prefill_step_size=prefill_step_size)
        calls.append({"B": B, "eos": list(eos), "max_tokens": max_tokens, "prefill_step_size": prefill_step_size,
                      "completion_batch_size": gen.completion_batch_size, "prefill_batch_size": gen.prefill_batch_size,
                      "gen_prefill_step_size": gen.prefill_step_size, "max_kv_size": gen.max_kv_size})
        return gen

    mp.setattr(generate, "make_batch_generator", spy)
    try:
        cfg = common.read_json(tiny_real_val.bf16 / "config.json")
        ts = textset.make_tiny(cfg["sliding_window"], cfg["num_hidden_layers"], cfg["vocab_size"])
        step = ts.profile.prefill_chunk
        p26 = R.prompts26(ts)
        quiet = {"eos": (), "prefill_step_size": step, "log": lambda m: None}
        phases = {}
        for arm, d in (("K8", tiny_real_val.k8), ("K4", tiny_real_val.k4)):
            blocks = {}
            model, _, _ = common.load_port(d)
            model.set_dtype(mx.float32)
            if arm == "K8":
                conts = g5.greedy_continuations(model, ts.g5, ts.profile.g5_greedy_tokens, eos=())
                blocks["greedy"] = R.greedy_block(conts, ts.g5)
            blocks["g5_r1"] = R.g5_r1(model, ts.g5, {"floor_kl": 0.0, "floor_dis": 0.0},
                                      R.g5_out_dir(RUN, arm, "g5_r1"), arm=arm,
                                      max_tokens=ts.profile.g5_greedy_tokens, **quiet)
            for B in (8, 16):
                blocks[f"g5_bp_B{B}"] = R.g5_bp(model, p26, B, R.g5_out_dir(RUN, arm, f"g5_bp_B{B}"), arm=arm, **quiet)
            del model
            if arm == "K8":
                m26, _, _ = common.load_port(d, mutant="batch_decode_pos_frozen")
                m26.set_dtype(mx.float32)
                blocks["g5_r1_m26"] = R.g5_r1(m26, ts.g5, None, R.g5_out_dir(RUN, arm, "g5_r1_m26"), arm=arm,
                                              mutant=26, max_tokens=ts.profile.g5_greedy_tokens, **quiet)
                del m26
                m22, _, _ = common.load_port(d, mutant="batched_rope_positions_in_padded_frame")
                m22.set_dtype(mx.float32)
                blocks["g5_bp_m22"] = R.g5_bp(m22, p26, 8, R.g5_out_dir(RUN, arm, "g5_bp_m22"), arm=arm, mutant=22,
                                              **quiet)
                del m22
                m27, _, _ = common.load_port(d, mutant="batch_decode_pad_keys_visible")
                m27.set_dtype(mx.float32)
                blocks["g5_bp_m27"] = R.g5_bp(m27, p26, 8, R.g5_out_dir(RUN, arm, "g5_bp_m27"), arm=arm, mutant=27,
                                              **quiet)
                del m27
            mx.clear_cache()
            phase = 4 if arm == "K8" else 5
            phases[phase] = json.loads(common.dumps({"phase": phase, "data": {"g5": blocks}}))
        yield {"phases": phases, "calls": calls, "ts": ts, "p26": p26, "work": work, "bf16": tiny_real_val.bf16}
    finally:
        mp.undo()


def _blocks(runs, phase=None):
    ph = runs["phases"]
    return {b["id"]: b for p, j in ph.items() if phase in (None, p) for b in j["data"]["g5"].values()}


def test_g5_bp_first_wave_and_mid_run_at_B8_and_B16(g5_runs):
    """G1 item 8: first wave = the first B inserted, mid-run = admitted after the
    first finish, at B = 8 and 16 with the tiny 26-prompt layout; max_live <= B."""
    names = [e["seq"] for e in g5_runs["p26"]]
    blocks = _blocks(g5_runs)
    for arm in ("K8", "K4"):
        for B in (8, 16):
            b = blocks[f"{arm}/g5_bp_B{B}"]
            tr = b["trace"]
            assert tr["first_wave"] == names[:B] and tr["mid_run"] == names[B:] and tr["other"] == []
            assert tr["admitted_mid_run"] == 26 - B >= 1 and 1 <= tr["max_live"] <= B
            assert b["generator"] == {"completion_batch_size": B, "prefill_batch_size": min(B, 8),
                                      "prefill_step_size": 64, "max_kv_size": None}
            for s, e in zip(b["sequences"], g5_runs["p26"]):
                assert s["seq"] == e["seq"] and s["prompt_ids"] == e["ids"] and s["n"] == e["max_tokens"]
                assert s["label"] == ("first_wave" if e["seq"] in names[:B] else "mid_run")
            assert {x["insert_step"] for x in b["sequences"] if x["label"] == "first_wave"} == {0}
            mid = [s for s in b["sequences"] if s["label"] == "mid_run"]
            assert min(s["insert_step"] for s in mid) > tr["first_finish_step"]
    for mid in (27, 22):  # the control (decision (b)) and the probe
        m = blocks[f"K8/g5_bp_m{mid}"]
        assert m["mutant"] == mid and m["B"] == 8 and m["needs_r1"] and m["trace"]["first_wave"] == names[:8]


def test_g5_r1_runs_each_prompt_alone(g5_runs):
    blocks = _blocks(g5_runs)
    ts = g5_runs["ts"]
    for arm in ("K8", "K4"):
        b = blocks[f"{arm}/g5_r1"]
        assert b["B"] == 1 and b["trace"]["max_live"] == 1 and b["needs_r1"] and b["mutant"] is None
        assert b["generator"]["completion_batch_size"] == 1 and b["generator"]["prefill_batch_size"] == 1
        assert [s["seq"] for s in b["sequences"]] == [p["id"] for p in ts.g5]
        assert all(s["n"] == ts.profile.g5_greedy_tokens for s in b["sequences"])
        assert set(b["parity"]) == {"mean_kl", "top1_dis", "n"} and b["parity"]["n"] == 8 * ts.profile.g5_greedy_tokens
        assert b["parity"]["mean_kl"] < 1e-6 and b["parity"]["top1_dis"] == 0.0  # the runner's own path, fp32
    m26 = blocks["K8/g5_r1_m26"]
    assert m26["mutant"] == 26 and not m26["needs_r1"]
    assert m26["parity"]["mean_kl"] > 100 * 1e-4  # mutant 26 moves the runner, not the single path


def test_one_construction_with_the_registered_arguments(g5_runs):
    """G1 item 8: every G5 BatchGenerator came from runner.generate.make_batch_generator,
    with completion_batch_size B and prefill_batch_size min(B, 8)."""
    calls = g5_runs["calls"]
    # K8: G5-R1, B 8, B 16, mutant 26 (B 1), mutant 27 and probe 22 (B 8); K4: G5-R1, B 8, B 16
    assert sorted(c["B"] for c in calls) == sorted([1, 8, 16, 1, 8, 8] + [1, 8, 16])
    for c in calls:
        assert c["completion_batch_size"] == c["B"] and c["prefill_batch_size"] == min(c["B"], 8)
        assert c["prefill_step_size"] == c["gen_prefill_step_size"] == 64 and c["max_kv_size"] is None
        assert c["eos"] == []
    for b in _blocks(g5_runs).values():
        if b["check"] != "greedy":
            assert b["construction"].startswith("runner.generate.make_batch_generator")


def test_batch_parity_goes_through_make_batch_generator(tiny_real_val, monkeypatch):
    import mlx.core as mx

    from gate import common
    from gate.checks import g5_generation as g5
    from runner import generate

    calls = []
    real = generate.make_batch_generator
    monkeypatch.setattr(generate, "make_batch_generator",
                        lambda *a, **k: calls.append((a[1], k.get("prefill_step_size"))) or real(*a, **k))
    model, _, _ = common.load_port(tiny_real_val.k8)
    model.set_dtype(mx.float32)
    prompts = [tc.random_ids(n, seed=n) for n in (7, 30, 70, 12, 52)]
    res = g5.batch_parity(model, prompts, [5, 2, 4, 3, 6], 2, eos=())
    assert calls == [(2, 2048)] and res["path"] == g5.CONSTRUCTION
    assert res["max_live"] <= 2 and res["admitted_mid_run"] == 3 and res["n_positions"] == 20
    assert res["mean_kl"] < 1e-6


def test_anchor_covers_both_arms_and_the_reductions_feed_the_rules(g5_runs):
    """G1 item 8, last bullet: with arms (K8, K4), every sequence that R-anchor,
    R-greedy and (d) need for either arm has an R1 row in phase 6, and every
    log-prob file named in a phase JSON exists with its sha256."""
    from gate import common, rules
    from gate.checks import g5_runner as R
    from gate.checks import ref_drivers

    phases = [g5_runs["phases"][4], g5_runs["phases"][5]]
    blocks = _blocks(g5_runs)
    anc = R.anchor_sequences(phases)
    need = {(bid, s["seq"]): s for bid, b in blocks.items() if b["needs_r1"] for s in b["sequences"]}
    assert {b["id"] for b in blocks.values() if not b["needs_r1"]} == {"K8/g5_r1_m26"}
    assert {k[0] for k in need} == {"K8/greedy", "K8/g5_r1", "K8/g5_bp_B8", "K8/g5_bp_B16", "K8/g5_bp_m27",
                                    "K8/g5_bp_m22",
                                    "K4/g5_r1", "K4/g5_bp_B8", "K4/g5_bp_B16"}
    uses = {(u["block"], u["seq"]): u for u in anc["uses"]}
    assert set(uses) == set(need)
    for key, s in need.items():
        u = uses[key]
        fed = s["prompt_ids"] + s["tokens"][:-1]
        assert anc["seqs"][u["anchor"]] == fed and anc["keys"][u["anchor"]] == common.ids_sha256(fed)
        want = np.arange(len(s["prompt_ids"]) - 1, len(s["prompt_ids"]) - 1 + s["n"])
        assert np.isin(want, anc["rows"][u["anchor"]]).all()
    assert len(set(anc["keys"])) == anc["n_sequences"] <= len(need)  # identical sequences are scored once
    fv = R.verify_logprob_files(phases)
    n_files = 2 * sum(len(b["sequences"]) for b in blocks.values() if b["check"] != "greedy")
    assert fv == {"n_files": n_files, "problems": []}
    assert len(R.logprob_files(phases)) == n_files

    r1 = ref_drivers.r1_rows_pass(g5_runs["bf16"], anc["seqs"], anc["rows"], log=lambda m: None)
    red = R.reduce_with_r1(phases, r1, anc)
    # Mutant 26's files are not reloaded: its R-parity was reduced when it was generated.
    assert red["files_loaded"] == n_files - 2 * len(blocks["K8/g5_r1_m26"]["sequences"])
    red = json.loads(common.dumps(red))  # phase 6 hands phase 7 a JSON
    th = rules.load_thresholds()
    floor = {"floor_kl": 0.0, "floor_dis": 0.0}
    r1c = rules.g5_r1(red["g5_r1"], floor, red["g5_r1_mutants"], th)
    for arm in ("K8", "K4"):
        assert r1c[f"g5_r1_{arm}"]["state"] in (rules.PASS, rules.FAIL)  # measured, not MISSING
        a = r1c[f"g5_r1_{arm}"]["legs"]["R-anchor"]
        assert a["mean_kl_r1_runner"] > 0 and a["mean_kl_r1_single"] > 0
    assert set(r1c["g5_r1_K8"]["legs"]) == {"R-anchor", "R-greedy", "R-parity"}
    assert r1c["g5_r1_K8"]["legs"]["R-greedy"]["n"] == 8 * g5_runs["ts"].profile.g5_greedy_tokens
    assert r1c["g5_r1_K8"]["controls"]["G5R1/26"]["caught"] is True
    bp = rules.g5_bp(red["g5_bp"], red["g5_bp_mutants"], th)
    for arm in ("K8", "K4"):
        for B in (8, 16):
            c = bp["checks"][f"g5_bp_{arm}_B{B}"]
            assert c["legs"]["e"]["ok"] and c["legs"]["d:first_wave"]["n"] > 0 and c["legs"]["d:mid_run"]["n"] > 0
    assert set(red["g5_bp_mutants"]) == {"27", "22"}
    assert set(bp["controls"]) == {"G5BP/27"}  # probe 22 is reduced but never a control (DESIGN §4.2)
    assert bp["controls"]["G5BP/27"]["caught"] is not None  # judged (its tiny acceptance is W14's table)
    assert red["g5_greedy"]["n_positions"] == 8 * g5_runs["ts"].profile.g5_greedy_tokens
    assert rules.g5_greedy(red["g5_greedy"], th["G5"])["state"] in (rules.PASS, rules.FAIL)

    # The reductions are the direct computation from the files and R1's rows.
    b = blocks["K4/g5_bp_B8"]
    look = {(int(s), int(p)): j for j, (s, p) in enumerate(zip(r1["seq_index"], r1["positions"]))}
    kb, ks = [], []
    for s in b["sequences"]:
        if s["label"] != "mid_run":
            continue
        u = uses[(b["id"], s["seq"])]
        ref = r1["logits"][[look[(u["anchor"], p)] for p in range(u["start"], u["start"] + u["n"])]]
        kb.append(common.kl_rows(ref, R.load_logprobs(s["files"]["batched"])))
        ks.append(common.kl_rows(ref, R.load_logprobs(s["files"]["single"])))
    d = red["g5_bp"]["K4"]["8"]["mid_run"]
    assert d["n"] == sum(k.size for k in kb)
    assert d["mean_kl_r1_batched"] == pytest.approx(float(np.concatenate(kb).mean()), rel=1e-12)
    assert d["mean_kl_r1_single"] == pytest.approx(float(np.concatenate(ks).mean()), rel=1e-12)

    # No measurement emits a verdict.
    def keys(o):
        if isinstance(o, dict):
            return set(o) | set().union(*(keys(v) for v in o.values()))
        if isinstance(o, list):
            return set().union(*(keys(v) for v in o)) if o else set()
        return set()

    assert "pass" not in keys(phases) and "pass" not in keys(red)

    # Refusals: an R1 pass that misses a row, or ran on other sequences; an anchor of other phase JSONs.
    short = dict(r1, logits=r1["logits"][:-1], seq_index=r1["seq_index"][:-1], positions=r1["positions"][:-1])
    with pytest.raises(R.G5DataError, match="no R1 row"):
        R.reduce_with_r1(phases, short, anc)
    with pytest.raises(R.G5DataError):
        R.reduce_with_r1(phases, dict(r1, seq_lengths=r1["seq_lengths"][:-1]), anc)
    with pytest.raises(R.G5DataError):
        R.reduce_with_r1([g5_runs["phases"][4]], r1, anc)
    with pytest.raises(R.G5DataError, match="more than once"):
        R.anchor_sequences([g5_runs["phases"][4], g5_runs["phases"][4]])


def test_a_changed_logprob_file_is_refused(g5_runs, tmp_path):
    """A log-prob file that no longer matches its phase JSON's sha256 is found
    by verify_logprob_files and refused by the reduction (restored after)."""
    from gate.checks import g5_runner as R

    phases = [g5_runs["phases"][4], g5_runs["phases"][5]]
    e = _blocks(g5_runs)["K4/g5_r1"]["sequences"][0]["files"]["runner"]
    p = R.logprob_path(e)
    keep = p.read_bytes()
    try:
        p.write_bytes(keep[:-4] + b"\x00\x00\x80\x3f")
        fv = R.verify_logprob_files(phases)
        assert fv["problems"] == [f"{e['path']}: sha256 differs"]
        with pytest.raises(R.G5DataError, match="sha256"):
            R.load_logprobs(e)
    finally:
        p.write_bytes(keep)
    assert R.verify_logprob_files(phases)["problems"] == []


def test_greedy_block_checks_its_prompts():
    from gate import common, textset
    from gate.checks import g5_runner as R

    ts = textset.make_tiny(513, 10, tc.VOCAB_SIZE)
    conts = [{"id": p["id"], "prompt_tokens": len(p["ids"]), "prompt_ids_sha256": common.ids_sha256(p["ids"]),
              "continuation": [1, 2, 3], "finished": "length", "lang": p["lang"], "effort": p["effort"]} for p in ts.g5]
    b = R.greedy_block(conts, ts.g5)
    assert b["id"] == "K8/greedy" and b["needs_r1"] and [s["seq"] for s in b["sequences"]] == [p["id"] for p in ts.g5]
    conts[2]["prompt_ids_sha256"] = "0" * 64
    with pytest.raises(R.G5DataError):
        R.greedy_block(conts, ts.g5)


# ---------------------------------------------------------------------------
# G5-D32 (§3.12)
# ---------------------------------------------------------------------------


def test_g5_d32_series_and_mutant_20(tiny_real_val):
    """G5-D32's measured dict on the tiny K8 (fp32 activations) at a short range
    past 256 keys: the shape rules.g5_d32 reads, Csort and the two comparisons
    at the rounding level, and mutant 20 (decode window 256) caught by the
    decode leg with the §4.2 margin."""
    import mlx.core as mx

    from gate import common, rules, textset
    from gate.checks import g5_runner as R

    cfg = common.read_json(tiny_real_val.bf16 / "config.json")
    ts = textset.make_tiny(cfg["sliding_window"], cfg["num_hidden_layers"], cfg["vocab_size"])
    texts, ranges = R.d32_inputs(ts)
    assert set(texts) == set(ranges) == {"T1", "T3", "T9"} and ranges["T1"] == ts.profile.decode_ranges["T1"]
    short = {r: (lo, lo + 40) for r, (lo, hi) in ranges.items()}
    th = rules.load_thresholds()
    model, _, module = common.load_port(tiny_real_val.k8)
    model.set_dtype(mx.float32)
    base = R.g5_d32(model, module, texts, short, log=lambda m: None)
    for r, (lo, hi) in short.items():
        e = base[r]
        assert e["positions"] == [lo, hi] and e["n"] == 41 and e["chunk"] == 2048 and e["chunk_small"] == 64
        assert 0 <= e["csort"]["mean"] <= e["csort"]["max"] < 1e-6
        for comp in ("decode", "chunk64"):
            c = e[comp]
            assert c["n"] == 41 and 0 <= c["mean_kl"] <= c["max_kl"] < 1e-6
            assert set(c["top1_change_leads"]) == {"n", "top"} and c["top1_change_leads"]["n"] == c["n_top1_changes"]
    m20, _, m20_module = common.load_port(tiny_real_val.k8, mutant="decode_window_256")
    m20.set_dtype(mx.float32)
    mut = R.g5_d32(m20, m20_module, texts, short, comparisons=("decode",), log=lambda m: None)
    assert all(set(v) >= {"csort", "decode"} and "chunk64" not in v for v in mut.values())
    c = rules.g5_d32(base, {20: mut}, th)
    ctl = c["controls"]["G5D32/20"]
    assert ctl["caught"] is True and ctl["margin"]["met"]
    assert all(mut[r]["decode"]["mean_kl"] > 1e3 * max(base[r]["decode"]["mean_kl"], 1e-12) for r in short)
    with pytest.raises(ValueError):
        R.g5_d32(model, module, texts, {"T1": (5000, 5100)}, log=lambda m: None)
    with pytest.raises(ValueError):
        R.g5_d32(model, module, texts, short, comparisons=("prefill",), log=lambda m: None)
