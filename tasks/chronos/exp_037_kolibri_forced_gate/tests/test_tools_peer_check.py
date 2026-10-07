"""tools/peer_check.py (BUILD_SPEC §5.9; HYPOTHESIS C3, C9).

Weight-free: build-time tokenizer/template parity on the mini's peer folders
(skipped where they are absent), the committed parity record being current,
the static per-arm checks, and the decision arithmetic on synthetic numbers.
Amendment 5: the batched path's prompts start with the chat wrapper; run()
classifies a batched-path-check failure (parity or greedy flip) as B=1 and
anything else as fail, judges NLL/KL on the chat-wrapped texts, and still
records the raw-text NLL where the gate's G3 reads it; the record then works
downstream (G3, the runner's guard, the pilot's --without, the plan's B).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import peer_check as pc


def _mini_root() -> Path | None:
    from exp036_helpers import mini_asset_roots

    for c in mini_asset_roots():
        if (c / "mlx-community").is_dir() and (c / "upstream").is_dir():
            return c
    return None


@pytest.fixture(scope="module")
def mini():
    """The build host's peer and upstream folders (parity needs upstream; only the mini has it)."""
    from exp036_helpers import BUILD_HOST_ONLY

    root = _mini_root()
    if root is None:
        pytest.skip(f"{BUILD_HOST_ONLY} build-time peer and upstream files not found (set EXP036_MINI_ASSETS)")
    return root


@pytest.fixture(scope="module")
def peers():
    """The six peer folders: the mini's mlx-community/ or the run host's $EXP036_MODELS."""
    from exp036_helpers import peer_root

    root = peer_root(list(pc.ARM_DIRS.values()))
    if root is None:
        pytest.skip("peer folders not found (set EXP036_MINI_ASSETS on the mini, EXP036_MODELS on the run host)")
    return root


@pytest.fixture(scope="module")
def fresh_parity(mini):
    return pc.build_parity(mini / "mlx-community", mini / "upstream")


def test_parity_on_the_mini_folders(fresh_parity):
    fams = fresh_parity["families"]
    assert set(fams) == set(pc.FAMILIES)
    for fam, f in fams.items():
        assert "error" not in f, (fam, f.get("error"))
        assert f["committed_template"]["equal_to_upstream"], fam
        for arm in pc.FAMILIES[fam]["arms"]:
            p = f["peers"][arm]
            assert p["present"]
            assert p["token_ids"]["n"] == len(pc.FIXED_STRINGS)
            assert p["token_ids"]["mismatches"] == [], (arm, p["token_ids"]["mismatches"])
            assert p["template"]["render_mismatches"] == [], (arm, p["template"]["render_mismatches"])
            assert p["template"]["rendered_ids_mismatches"] == []
            assert p["byte_equal_to_upstream"]["chat_template.jinja"]


def test_committed_parity_record_is_current(fresh_parity):
    committed = json.loads(pc.PARITY_FILE.read_text())
    assert committed["schema"] == pc.PARITY_SCHEMA
    strip = lambda r: {k: v for k, v in r.items() if k != "utc"}  # noqa: E731
    assert strip(committed) == json.loads(json.dumps(strip(fresh_parity))), \
        "tools/peer_parity_build.json is stale: rerun tools/peer_check.py --parity"


def test_static_check_on_the_peer_folders(peers):
    for arm in pc.ARM_DIRS:
        r = pc.static_check(arm, peers)
        assert r["present"] and r["problems"] == [], (arm, r["problems"])
        assert all(r["same_files_as_build_parity"].values())
        assert r["config"]["ok"] and r["config"]["base"]["bits"] == pc.ARM_BITS[arm]
        assert r["eos"]["generation_config"]
        assert r["params_config"]["rel_diff_vs_published"] <= pc.PARAM_TOLERANCE
        if (pc.common.EXP_DIR / "runner" / "templates").is_dir():
            assert r["template_equals_committed"] is True


def test_static_check_flags_changed_files(peers, tmp_path):
    import shutil

    root = tmp_path / "peers"
    shutil.copytree(peers / pc.ARM_DIRS["G8"], root / pc.ARM_DIRS["G8"],
                    ignore=shutil.ignore_patterns("*.safetensors", ".cache"))
    with open(root / pc.ARM_DIRS["G8"] / "chat_template.jinja", "a") as f:
        f.write("\n{# edited #}\n")
    r = pc.static_check("G8", root)
    assert r["same_files_as_build_parity"]["chat_template.jinja"] is False
    assert any("differ from those of the build-time parity" in p for p in r["problems"])


def test_config_check_and_overrides():
    cfg = {"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                            "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8},
                            "language_model.model.layers.1.router.proj": {"group_size": 64, "bits": 8}}}
    r = pc.config_check(cfg, 4)
    assert r["ok"] and r["n_overrides"] == 2
    assert r["overrides"] == {"router.proj": [[8, 64]]}
    assert not pc.config_check(cfg, 8)["ok"]


class _Tok:
    def __init__(self, shift=0):
        self.shift = shift

    def encode(self, s, add_special_tokens=False):
        return [ord(c) + (self.shift if i == 3 else 0) for i, c in enumerate(s)]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_token_parity_reports_first_difference():
    same = pc.token_parity(_Tok(), _Tok(), strings=("abcdef",))
    assert same["mismatches"] == [] and same["decode_roundtrip_differs"] == 0
    diff = pc.token_parity(_Tok(), _Tok(shift=1), strings=("abcdef", "ab"))
    assert diff["mismatches"] == [{"string": 0, "len_a": 6, "len_b": 6, "first_diff": 3}]


def test_fidelity_rule_and_summaries():
    stats = {"T1": {"tokens": 100, "bytes": 400, "nll8": 200.0, "nll4": 199.0, "kl": 5.0},
             "T2": {"tokens": 100, "bytes": 450, "nll8": 220.0, "nll4": 220.0, "kl": 6.0}}
    r = pc.fidelity_rule(stats)
    assert r["nll_ok"] and r["kl_ok"]
    assert abs(r["nll8_per_token"] - 2.1) < 1e-12 and abs(r["kl_8v4_per_token"] - 0.055) < 1e-12
    stats["T2"]["nll8"] = 230.0
    assert not pc.fidelity_rule(stats)["nll_ok"]
    stats["T1"]["kl"] = 60.0
    assert not pc.fidelity_rule(stats)["kl_ok"]
    s = pc.nll_summary(stats, "nll8")
    assert set(s["per_text_bpb"]) == {"T1", "T2"}
    assert abs(s["per_text"]["T1"]["bpb"] - 200.0 / 0.6931471805599453 / 400) < 1e-9


def test_flip_rate_rule():
    a = ["A", "B", "C", None] + ["D"] * 46
    b = ["A", "B", "C", "A"] + ["D"] * 46
    r = pc.flip_rate(a, b)
    assert r == {"n": 50, "flips": 1, "rate": 0.02, "ok": True}
    assert not pc.flip_rate(a, ["X"] + b[1:3] + ["A"] + b[4:])["ok"]
    with pytest.raises(ValueError):
        pc.flip_rate(["A"], [])


def test_verdicts_and_exit_code():
    assert pc._verdict({"problems": ["x"]}) == "fail"
    assert pc._verdict({"problems": [], "batched_path": {"ok": False}}) == "B=1"
    assert pc._verdict({"problems": []}) == "ok"
    # Amendment 5: the greedy flip is part of the batched-path check, so a flip-only failure is B=1 ...
    flip = {"n": 30, "flips": 1, "rate": 1 / 30, "ok": False}
    assert pc._verdict({"problems": [], "batched_path": {"ok": True}, "greedy_flip": flip,
                        "batched_path_problems": ["greedy flip rate 0.033 > 0.02"]}) == "B=1"
    assert pc._verdict({"problems": [], "greedy_flip": flip}) == "B=1"
    assert pc._verdict({"problems": [], "batched_path": {"ok": False}, "greedy_flip": flip}) == "B=1"
    assert pc._verdict({"problems": [], "batched_path": {"ok": True}, "greedy_flip": dict(flip, ok=True),
                        "batched_path_problems": []}) == "ok"
    # ... while an NLL / KL failure, or a check that could not run, is still fail, flip or not.
    assert pc._verdict({"problems": ["NLL(8) / KL(8‖4) rule failed"], "greedy_flip": flip,
                        "batched_path_problems": ["greedy flip rate 0.033 > 0.02"]}) == "fail"
    assert pc._verdict({"problems": ["greedy flip rate could not run"], "batched_path": {"ok": True}}) == "fail"
    assert pc.exit_code({"arms": {"G8": {"verdict": "ok"}, "G4": {"verdict": "B=1"}}}) == 2
    assert pc.exit_code({"arms": {"G8": {"verdict": "fail"}}}) == 1
    assert pc.exit_code({"arms": {"G8": {"verdict": "ok"}}}) == 0


def test_upstream_pins_come_from_assets_json():
    pins = pc.upstream_pins()
    assert pins["google/gemma-4-26B-A4B-it"][0].startswith("20da991a")
    for fam in pc.FAMILIES.values():
        assert fam["upstream_repo"] in pins


# --------------------------------------------------------------------------
# Amendment 5: the chat-wrapped batched path and the run() classification
# --------------------------------------------------------------------------


def test_batched_prompts_start_with_the_wrapper_and_keep_the_registered_lengths():
    w = [900 + i for i in range(17)]
    stream = list(range(2000))
    base, prompts = pc.batched_prompts(w, stream)
    assert base == w + stream
    assert [len(p) for p in prompts] == list(pc.BATCH_LENGTHS)          # 37 ... 1,100, as registered
    for i, (p, n) in enumerate(zip(prompts, pc.BATCH_LENGTHS)):
        assert p[:17] == w and p[17:] == stream[i * 50:i * 50 + n - 17]
    lo, _ = pc.FLOOR_RANGE
    assert lo >= len(w)                                                  # the floor positions are text positions
    with pytest.raises(ValueError, match="shortest prompt"):
        pc.batched_prompts(list(range(37)), stream)
    with pytest.raises(ValueError, match="base tokens"):
        pc.batched_prompts(w, stream[:1000])
    need = max(i * 50 + n for i, n in enumerate(pc.BATCH_LENGTHS)) - len(w)
    pc.batched_prompts(w, stream[:need])                                 # exactly enough
    with pytest.raises(ValueError, match="base tokens"):
        pc.batched_prompts(w, stream[:need - 1])


def _stats(nll8, nll4, kl, tokens=100, nbytes=400):
    return {t: {"tokens": tokens, "bytes": nbytes, "nll8": nll8 * tokens, "nll4": nll4 * tokens, "kl": kl * tokens,
                "top1_agree": 0.5} for t in ("T1", "T2")}


RAW = _stats(10.2, 10.7, 4.6)       # Gemma-like raw text: the KL rule fails there
CHAT = _stats(2.0, 2.01, 0.05)      # chat-wrapped: a normal LM, the rule passes
WRAP = {"prompt_ids": [2, 105, 2364], "record": {"amendment": "Amendment 5", "kwargs": {"enable_thinking": False},
                                                 "prompt_ids": [2, 105, 2364], "template_sha256": "e" * 64}}


@pytest.fixture
def fake_run(monkeypatch, tmp_path):
    """run() with every weight-touching step stubbed: per-arm static checks pass, loads succeed, text_stats
    returns RAW for the raw text and CHAT for the chat-wrapped one, and the batched path / greedy flip
    return what the test sets."""
    from bench import kl_8v4 as K
    from tools import params as P

    state = {"bp_ok": True, "flip_ok": True, "chat": CHAT, "wrappers": {}, "calls": []}
    monkeypatch.setattr(pc, "static_check", lambda arm, root=None, parity=None: {
        "arm": arm, "family": pc.family_of(arm), "present": True, "problems": [],
        "params_config": {"total": 1000, "num_experts": None, "top_k": None, "tied_embeddings": False}})
    monkeypatch.setattr(pc, "_load", lambda d: f"model:{Path(d).name}")
    monkeypatch.setattr(P, "loaded_params", lambda *a, **k: {"total": 1000})
    monkeypatch.setattr(K, "gate_text_paths", lambda *a, **k: {})
    monkeypatch.setattr(K, "load_texts", lambda paths: {t: {"text": "x", "bytes": 400, "sha256": "0" * 64}
                                                        for t in ("T1", "T2")})

    def chat_wrapper(fam, d, tokenizer=None):
        state["wrappers"][Path(d).name] = fam
        return json.loads(json.dumps(WRAP))

    def text_stats(m8, m4, d8, texts, prompt_ids=None):
        state["calls"].append((m8, m4, prompt_ids))
        return RAW if prompt_ids is None else state["chat"]

    monkeypatch.setattr(K, "chat_wrapper", chat_wrapper)
    monkeypatch.setattr(pc, "text_stats", text_stats)
    monkeypatch.setattr(pc, "batched_path", lambda m, d, arm: {"ok": state["bp_ok"], "parity": {}, "wrapper": WRAP})
    monkeypatch.setattr(pc, "greedy_flip", lambda m, d, arm: (
        {"n": 30, "flips": 0, "rate": 0.0, "ok": True} if state["flip_ok"]
        else {"n": 30, "flips": 1, "rate": 1 / 30, "ok": False}))
    state["root"] = tmp_path / "models"
    return state


def test_run_judges_nll_kl_on_the_chat_wrapped_texts_and_keeps_raw_nll_for_g3(fake_run, tmp_path):
    from gate.checks import g3_oracle

    rec = pc.run(["G8", "G4", "Q36-8", "Q36-4"], root=fake_run["root"])
    # both runs: raw first (for G3), then the chat wrapper; 8-bit and 4-bit together
    assert [c[2] for c in fake_run["calls"]] == [None, WRAP["prompt_ids"]] * 2
    for fam, a8, a4 in (("gemma4", "G8", "G4"), ("qwen3_6", "Q36-8", "Q36-4")):
        f = rec["families"][fam]
        assert f["fidelity"]["kl_ok"] and f["fidelity"]["nll_ok"] and f["fidelity"]["texts"].startswith("chat")
        assert f["fidelity_raw_text"]["kl_ok"] is False and f["fidelity_raw_text"]["used_for_verdict"] is False
        assert f["wrapper"]["prompt_ids"] == WRAP["prompt_ids"] and f["wrapper"]["prompt_ids_4bit_identical"]
        assert f["wrapper"]["kwargs"] == {"enable_thinking": False} and len(f["wrapper"]["template_sha256"]) == 64
        for a, key in ((a8, "nll8"), (a4, "nll4")):
            assert rec["arms"][a]["nll"] == pc.nll_summary(RAW, key)          # raw text, where G3 reads it
            assert rec["arms"][a]["nll_chat"] == pc.nll_summary(CHAT, key)
            assert rec["arms"][a]["verdict"] == "ok" and rec["arms"][a]["problems"] == []
    # the gate's G3 reads the raw-text pooled bpb of G8 and Q36-8
    res = tmp_path / "results"
    res.mkdir()
    (res / "peers_20261004T160000Z.json").write_text(json.dumps(rec))
    got = g3_oracle.peer_bpb(res)
    raw_bpb = pc.nll_summary(RAW, "nll8")["mean_bpb"]
    assert got["peers"] == {"G8": raw_bpb, "Q36-8": raw_bpb}
    assert raw_bpb != pc.nll_summary(CHAT, "nll8")["mean_bpb"]


@pytest.mark.parametrize("bp_ok, flip_ok", [(True, False), (False, True), (False, False)])
def test_run_classifies_a_batched_path_check_failure_as_b1(fake_run, bp_ok, flip_ok):
    fake_run["bp_ok"], fake_run["flip_ok"] = bp_ok, flip_ok
    rec = pc.run(["G8", "G4", "Q36-8", "Q36-4"], root=fake_run["root"])
    for a8, a4 in (("G8", "G4"), ("Q36-8", "Q36-4")):
        assert rec["arms"][a8]["verdict"] == "B=1" and rec["arms"][a8]["problems"] == []
        assert len(rec["arms"][a8]["batched_path_problems"]) == (not bp_ok) + (not flip_ok)
        assert rec["arms"][a4]["verdict"] == "ok"
    assert pc.exit_code(rec) == 2


def test_run_still_fails_an_nll_kl_failure_whatever_the_flip(fake_run):
    fake_run["flip_ok"] = False
    fake_run["chat"] = _stats(2.0, 2.5, 0.4)           # chat-wrapped KL 0.4 > 0.2: a real fidelity failure
    rec = pc.run(["Q36-8", "Q36-4"], root=fake_run["root"])
    assert rec["arms"]["Q36-8"]["verdict"] == "fail" and rec["arms"]["Q36-4"]["verdict"] == "fail"
    assert rec["arms"]["Q36-8"]["problems"] == ["NLL(8) / KL(8‖4) rule failed"]
    assert pc.exit_code(rec) == 1


def test_a_b1_record_downstream_runs_every_cell_at_b1_and_is_never_a_drop(fake_run, tmp_path, monkeypatch):
    """The record run() writes for a flip-only failure: the guard lets the arm run, excluded_arms does not drop
    it, the pilot's --without refuses it, and the plan's own reader of the record (plan_fix.build_context, the
    only production path from results/peers_*.json into the plan) carries it into peer_b1, so fix() gives every
    cell of it (Tier A and B) B = 1 and names it in the amendment. A newer record that says "fail" moves the arm
    from peer_b1 to excluded_arms."""
    import importlib
    import shutil

    guard = importlib.import_module("runner.guard")
    plan_fix = importlib.import_module("runner.plan_fix")
    from runner import memory

    fake_run["flip_ok"] = False
    rec = pc.run(["G8", "G4", "Q36-8", "Q36-4"], root=fake_run["root"])
    # A scratch experiment directory for build_context: the plan rules, a scorers/ tree, and the record in results/.
    exp = tmp_path / "exp"
    (exp / "runner").mkdir(parents=True)
    shutil.copy(pc.common.EXP_DIR / "runner" / "plan_rules.json", exp / "runner" / "plan_rules.json")
    (exp / "scorers").mkdir()
    (exp / "scorers" / "x.py").write_text("X = 1\n")
    res = exp / "results"
    res.mkdir()
    (res / "peers_20261004T160000Z.json").write_text(json.dumps(rec))
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "no_models"))       # no arm weights: weight_bytes {}
    monkeypatch.setattr(memory, "effective_limit", lambda *a, **k: (115_448_725_504, {"chosen": "test"}))
    assert guard.require_peers("Q36-8", res)["verdict"] == "B=1"
    assert guard.excluded_arms(res) == {}
    built = plan_fix.build_context(exp, [])
    assert built["peer_b1"] == ["G8", "Q36-8"] and built["excluded_arms"] == {}
    assert built["peers_record"]["path"] == "results/peers_20261004T160000Z.json"
    run_mod = importlib.import_module("runner.run")

    class _Ctx:
        results = res

    with pytest.raises(guard.GuardError, match="do not exclude Q36-8"):
        run_mod.pilot_without("Q36-8", _Ctx())

    from test_runner_plan_fix import RULES, synthetic

    pilot, steps, s1, ctx = synthetic(0)
    ctx.update({k: built[k] for k in ("peer_b1", "excluded_arms", "peers_record")})   # what the record decides
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["status"] == "FIXED" and plan["peer_b1"] == ["G8", "Q36-8"] and plan["excluded_arms"] == {}
    for arm in ("G8", "Q36-8"):
        cells = [q for q in plan["queue"] if q["arm"] == arm]
        assert {q["tier"] for q in cells} == {"A", "B"} and all(q["B"] == 1 for q in cells), arm
    assert any(q["arm"] == "K8" and q["B"] > 1 for q in plan["queue"])      # other arms keep the memory-rule B
    assert md.split("Peers at B = 1")[1].splitlines()[0].endswith(": G8, Q36-8")
    base, _, _ = plan_fix.fix(*synthetic(0)[:3], RULES, [], context=synthetic(0)[3])
    assert plan["peers"] == base["peers"] == ["G8", "Q36-8"]                 # still the MoE peers for H3 / H6
    assert base["peer_b1"] == [] and "Peers at B = 1" not in plan_fix.amendment_md(base, 1, "x", None)
    assert all(q["B"] > 1 for q in base["queue"] if q["arm"] in ("G8", "Q36-8"))   # without the record: batched
    # A newer record where G8 fails anything else: the reader takes the newest record; G8 is dropped, not B = 1.
    rec2 = json.loads(json.dumps(rec))
    rec2["arms"]["G8"]["verdict"] = "fail"
    (res / "peers_20261004T170000Z.json").write_text(json.dumps(rec2))
    built2 = plan_fix.build_context(exp, [])
    assert built2["peer_b1"] == ["Q36-8"] and list(built2["excluded_arms"]) == ["G8"]
    assert built2["peers_record"]["path"] == "results/peers_20261004T170000Z.json"


# ---------------------------------------------------------------- exp_037: schemas and loads under mlx-lm 0.32.0

def test_schemas_peer_check_renamed_parity_kept_reference_renamed():
    """DESIGN §2.10: SCHEMA is renamed; the parity schema stays bound to the reused tools/peer_parity_build.json;
    the fidelity-reference schema is renamed with the pin's format (decision (c), §3.18: the record by rule)."""
    assert pc.SCHEMA == "exp037 peer check v1"
    assert pc.PARITY_SCHEMA == "exp036 peer parity v1" == json.loads(pc.PARITY_FILE.read_text())["schema"]
    assert pc.FIDELITY_REFERENCE_SCHEMA == "exp037 fidelity reference v1" == \
        json.loads(pc.FIDELITY_REFERENCE_FILE.read_text())["schema"]


@pytest.fixture(scope="module")
def tiny_k8(tmp_path_factory) -> Path:
    """A Kolibri-architecture 8-bit conversion by port/convert.py (model_file kolibri1.py, this kit's port):
    the shape of the dry run's peer stand-ins."""
    import contextlib
    import io

    import tiny_checkpoint as tc
    from tiny_real_layout import write_stub_tokenizer
    from port import convert

    root = tmp_path_factory.mktemp("peer_standin")
    src = tc.write_tiny_checkpoint(root / "src", seed=3, preset="vendor", copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg))
    write_stub_tokenizer(src)
    out = root / "standin-8bit"
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(src, out, 8, 64, guard=False)
    return out


@pytest.fixture
def load_spy(monkeypatch):
    import mlx_lm.utils

    calls = []

    def spy(path, **kw):
        calls.append((Path(path), kw))
        return "model", {}

    monkeypatch.setattr(mlx_lm.utils, "load_model", spy)
    return calls


def test_load_of_a_kolibri_architecture_standin_trusts_only_our_port(tiny_k8, load_spy):
    """mlx-lm 0.32.0 (#1385): our own kolibri1.py runs only with trust_remote_code=True, given after
    check_port_file has passed (DESIGN §6.5)."""
    assert pc._load(tiny_k8) == "model"
    assert load_spy == [(tiny_k8, {"strict": True, "trust_remote_code": True})]


def test_a_real_strict_load_of_the_standin_works_under_mlx_lm_032(tiny_k8):
    import mlx_lm

    model = pc._load(tiny_k8)
    assert mlx_lm.__version__ == "0.32.0"
    assert len(model.layers) == json.loads((tiny_k8 / "config.json").read_text())["num_hidden_layers"]


def test_load_of_a_builtin_peer_passes_no_trust(tmp_path, load_spy):
    """A real peer (a built-in mlx_lm model type, no model_file) loads without the flag."""
    d = tmp_path / "Qwen3.6-35B-A3B-8bit"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    pc._load(d)
    assert load_spy == [(d, {"strict": True})]


def test_load_refuses_a_stale_or_foreign_port_before_anything_runs(tiny_k8, tmp_path, load_spy):
    import shutil

    from port import convert

    stale = tmp_path / "stale"
    shutil.copytree(tiny_k8, stale)
    with open(stale / "kolibri1.py", "a") as f:
        f.write("\n# edited\n")
    with pytest.raises(convert.StalePortFile):
        pc._load(stale)
    foreign = tmp_path / "foreign"
    shutil.copytree(tiny_k8, foreign)
    cfg = json.loads((foreign / "config.json").read_text())
    cfg["model_file"] = "evil.py"
    (foreign / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(convert.UntrustedModelFile):
        pc._load(foreign)
    assert load_spy == []
