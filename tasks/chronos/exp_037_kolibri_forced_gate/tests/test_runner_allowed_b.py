"""allowed_B in the runner (exp_037 DESIGN §3.14, §6.3; HYPOTHESIS "Pilot
and the plan rule" and edge cases as amended for exp_037).

G5-BP-lean sets each Kolibri arm's allowed_B ({1}, {1, 2, 4, 8} or
{1, 2, 4, 8, 16}); the gate record carries it and runner/guard.require_gate
returns it. Then:

- run_cell refuses a B outside allowed_B when given it, before anything runs;
- runner/run.py (session, cell) refuses a queued Kolibri cell whose B is not
  allowed (a refusal, never a crash), and the pilot clips the memory-rule B
  to the largest allowed value;
- the crash fallback steps down within allowed_B;
- runner/plan_fix.py clips every Kolibri cell's memory-rule B to the largest
  allowed value <= it (the same path as a peer at B = 1), reads allowed_B
  from the newest gate record, and names it in the plan amendment.
"""

from __future__ import annotations

import copy
import json
import signal

import pytest

from runner import plan_fix
from runner.common import KOLIBRI_ARMS, clip_B, load_rules

RULES = load_rules()


# ------------------------------------------------------------------ clip_B


@pytest.mark.parametrize("B,allowed,want", [
    (16, [1, 2, 4, 8, 16], 16), (16, [1, 2, 4, 8], 8), (8, [1], 1), (4, [1, 2, 4, 8], 4), (2, [1, 2, 4, 8, 16], 2),
    (1, [1], 1), (0, [1, 2, 4, 8], 0), (8, None, 8), (0, None, 0),
])
def test_clip_B(B, allowed, want):
    assert clip_B(B, allowed) == want


# ----------------------------------------------------------- plan_fix (pure)

def _synthetic(s=0):
    from test_runner_plan_fix import synthetic

    return synthetic(s)


def _bs(plan):
    return {q["cell"]: q["B"] for q in plan["queue"] if q["arm"] != "bench"}


def test_plan_clips_kolibri_cells_to_allowed_B_and_leaves_peers_alone():
    pilot, steps, s1, ctx = _synthetic(0)
    base, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=copy.deepcopy(ctx))
    ctx["allowed_B"] = {"K8": [1, 2, 4, 8], "K4": [1]}
    plan, md, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["status"] == "FIXED" and plan["allowed_B"] == {"K4": [1], "K8": [1, 2, 4, 8]}
    got, want = _bs(plan), _bs(base)
    assert any(b == 16 for c, b in want.items() if c.startswith("K8:"))  # the memory rule gives K8 16 somewhere
    for cell, b in got.items():
        arm = cell.split(":")[0]
        if arm == "K8":
            assert b in (1, 2, 4, 8) and b == clip_B(want.get(cell, b), [1, 2, 4, 8])
        elif arm == "K4":
            assert b == 1
        elif cell in want:
            assert b == want[cell]  # peers: the memory rule's B, unchanged
    assert ("- Kolibri batch sizes within the gate's allowed_B (G5-BP-lean; each cell's memory-rule B is clipped to "
            "the largest allowed value): K4 {1}; K8 {1, 2, 4, 8}") in md


def test_full_allowed_B_changes_no_cell():
    pilot, steps, s1, ctx = _synthetic(0)
    base, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=copy.deepcopy(ctx))
    ctx["allowed_B"] = {a: [1, 2, 4, 8, 16] for a in KOLIBRI_ARMS}
    plan, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert _bs(plan) == _bs(base) and plan["plan"] == base["plan"] and plan["tier_b"] == base["tier_b"]


def test_k8_at_b1_is_re_simulated_at_b1():
    """DESIGN §8.2: if allowed_B(K8) = {1}, the rule projects K8 cells at B = 1 (slower), so the plan may fall. The
    synthetic pilot's step model is a constant aggregate rate (a = c = 0, where B does not matter); here K8's step
    model gets a per-step cost, so fewer rows per step cost more hours."""
    pilot, steps, s1, ctx = _synthetic(0)
    ctx["step_models_extra"] = {"K8": [0.02, 1.0 / 180.0, 0.0]}
    base, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=copy.deepcopy(ctx))
    ctx["allowed_B"] = {"K8": [1], "K4": [1, 2, 4, 8, 16]}
    plan, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert plan["k8_runs"] and all(b == 1 for c, b in _bs(plan).items() if c.startswith("K8:"))
    k8 = lambda p: sum(q["projected_h"] or 0 for q in p["queue"] if q["arm"] == "K8")  # noqa: E731
    first = "K8:gpqa_en:high"
    assert {q["cell"]: q["projected_h"] for q in plan["queue"]}[first] > {
        q["cell"]: q["projected_h"] for q in base["queue"]}[first]
    assert k8(plan) > 0


def test_a_peer_at_b1_and_an_allowed_B_take_the_same_path():
    pilot, steps, s1, ctx = _synthetic(0)
    ctx["peer_b1"] = ["G8"]
    ctx["allowed_B"] = {"K8": [1], "K4": [1]}
    plan, _, _ = plan_fix.fix(pilot, steps, s1, RULES, [], context=ctx)
    assert all(b == 1 for c, b in _bs(plan).items() if c.split(":")[0] in ("G8", "K8", "K4"))


def test_build_context_reads_allowed_B_from_the_newest_gate_record(tmp_path, monkeypatch):
    from runner import memory

    monkeypatch.setattr(memory, "effective_limit", lambda *a, **k: (115_448_725_504, {"chosen": "test"}))
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    exp = tmp_path / "exp"
    (exp / "runner").mkdir(parents=True)
    (exp / "runner" / "plan_rules.json").write_text(json.dumps(RULES))
    (exp / "scorers").mkdir()
    (exp / "scorers" / "x.py").write_text("X = 1\n")
    gate = exp / "results" / "gate"
    gate.mkdir(parents=True)
    ctx = plan_fix.build_context(exp, [])
    assert "allowed_B" not in ctx and "gate_record" not in ctx  # no gate record: nothing to clip to
    (gate / "gate_20261008T100000Z.json").write_text(json.dumps(
        {"mode": "real", "verdict": {"K8": "PASS", "K4": "PASS"}, "allowed_B": {"K8": [1, 2, 4, 8, 16], "K4": [1]}}))
    (gate / "gate_20261008T090000Z.json").write_text(json.dumps({"mode": "real", "allowed_B": {"K8": [1], "K4": [1]}}))
    ctx = plan_fix.build_context(exp, [])
    assert ctx["gate_record"]["path"] == "results/gate/gate_20261008T100000Z.json"
    assert ctx["allowed_B"] == {"K8": [1, 2, 4, 8, 16], "K4": [1]}
    (gate / "gate_20261008T110000Z.json").write_text(json.dumps({"mode": "real", "verdict": {"K8": "PASS"}}))
    assert plan_fix.build_context(exp, [])["allowed_B"] == {"K8": [1], "K4": [1]}  # none recorded: B = 1 only
    (gate / "gate_20261008T120000Z.json").write_text(json.dumps({"mode": "real", "allowed_B": {"K8": [1, 16]}}))
    with pytest.raises(SystemExit):
        plan_fix.build_context(exp, [])


# ------------------------------------------------------------ run_cell


def test_run_cell_refuses_a_B_outside_allowed_B_before_anything_runs(monkeypatch, tmp_path):
    from runner import generate as rg

    called = []
    monkeypatch.setattr(rg, "make_batch_generator", lambda *a, **k: called.append(a))
    it = [{"id": "q", "item_sha256": "0" * 64, "prompt_sha256": "1" * 64, "prompt_ids": [5, 6], "prompt_text": "u"}]
    for B, allowed in ((8, [1]), (16, [1, 2, 4, 8]), (2, [1])):
        with pytest.raises(ValueError, match="allowed_B"):
            rg.run_cell(object(), object(), it, "K8", "mmlu_en", "high", 8, B, 1, None, "kolibri",
                        eos=[2], allowed_B=allowed)
    assert called == []


# -------------------------------------------- run.py on the tiny checkpoint


@pytest.fixture
def world(tmp_path, monkeypatch, tiny_vendor_dir):
    pytest.importorskip("mlx.core")
    from exp036_helpers import import_sibling

    import tiny_checkpoint as tc

    run_mod = import_sibling("runner.run")
    harness = import_sibling("port_harness")
    model = harness.load_port(tiny_vendor_dir, float32=True)
    eos = [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]

    class StubTok:
        unk_token_id = None

        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"<{int(i)}>" for i in ids)

        def convert_tokens_to_ids(self, t):
            return {"<think>": 1000, "</think>": 1001}.get(t)

    monkeypatch.setenv("EXP036_PRIVATE", str(tmp_path / "private"))
    old = signal.getsignal(signal.SIGINT)
    ctx = run_mod.Ctx(tmp_path / "exp")
    ctx.raw.mkdir(parents=True)
    monkeypatch.setattr(run_mod, "Ctx", lambda exp_dir=None: ctx)
    monkeypatch.setattr(run_mod, "items_for_cell", lambda task, n: [
        {"id": f"{task}-{k}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k:064x}",
         "prompt_ids": tc.random_ids(9 + k, seed=k), "prompt_text": "u"} for k in range(n)])
    loads = []

    def get(self, arm):
        loads.append(arm)
        return model, StubTok(), {"eos_ids": eos}

    monkeypatch.setattr(run_mod.Loaded, "get", get)
    monkeypatch.setattr(run_mod.chat, "sampling_for", lambda fam: {"order": "vllm", "temperature": 0.0,
                                                                     "top_p": 1.0, "top_k": 0})
    for name in ("require_identity", "require_clean_tree", "require_signoff", "require_metal_limit",
                 "require_peers", "require_assets", "require_tier2", "require_environment"):
        monkeypatch.setattr(run_mod.guard, name, lambda *a, **k: None)
    state = {"plan": None, "allowed": {}}
    monkeypatch.setattr(run_mod.guard, "require_plan", lambda *a, **k: state["plan"])
    monkeypatch.setattr(run_mod.guard, "require_gate",
                        lambda arm, *a, **k: state["allowed"].get(arm) if arm in KOLIBRI_ARMS else None)
    yield run_mod, ctx, state, loads
    signal.signal(signal.SIGINT, old)


def q(arm, task, n, B, tier="A", proj=0.5):
    return {"cell": f"{arm}:{task}:high", "arm": arm, "task": task, "effort": "high", "pass": 0, "n": n, "cap": 6,
            "B": B, "tier": tier, "projected_h": proj, "session": "S2"}


def test_a_session_refuses_a_kolibri_cell_outside_allowed_B_and_it_is_not_a_crash(world, monkeypatch):
    from runner import jsonl

    run_mod, ctx, state, loads = world
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 2, B=2)]}
    state["allowed"] = {"K8": [1]}
    with pytest.raises(SystemExit) as e:
        run_mod.main(["session", "--name", "S2"])
    assert "allowed_B" in str(e.value) and loads == []
    assert not (ctx.raw / "S2" / "K8").exists()
    assert jsonl.read_jsonl(ctx.raw / "S2/.heartbeat")[0]["state"] == "stopped"
    assert run_mod.detect_crash(ctx, ctx.rules) is None
    # Within allowed_B the same queue runs, and run_cell gets allowed_B to assert.
    state["allowed"] = {"K8": [1, 2, 4, 8]}
    seen = []
    real = run_mod.run_cell

    def spy(*a, **k):
        seen.append(k.get("allowed_B"))
        return real(*a, **k)

    monkeypatch.setattr(run_mod, "run_cell", spy)
    assert run_mod.main(["session", "--name", "S2"]) == 0
    assert seen == [[1, 2, 4, 8]]
    recs = [r for r in jsonl.read_jsonl(ctx.raw / "S2/K8/mmlu_en_high.jsonl") if r["type"] == "record"]
    assert len(recs) == 2 and {r["batch_size"] for r in recs} == {2}


def test_peers_are_not_restricted(world):
    from runner import jsonl

    run_mod, ctx, state, _ = world
    state["plan"] = {"S1_hours": 6.0, "queue": [q("G8", "ifbench", 2, B=4)]}
    state["allowed"] = {"K8": [1], "K4": [1]}
    assert run_mod.main(["session", "--name", "S2"]) == 0
    recs = [r for r in jsonl.read_jsonl(ctx.raw / "S2/G8/ifbench_high.jsonl") if r["type"] == "record"]
    assert {r["batch_size"] for r in recs} == {4}


def test_the_cell_command_refuses_outside_allowed_B(world):
    run_mod, ctx, state, loads = world
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K4", "rgb_cb", 2, B=8)]}
    state["allowed"] = {"K4": [1]}
    with pytest.raises(SystemExit) as e:
        run_mod.main(["cell", "--arm", "K4", "--task", "rgb_cb", "--session", "S2"])
    assert "allowed_B" in str(e.value) and loads == [] and not (ctx.raw / "S2" / "K4").exists()


def test_the_crash_fallback_steps_down_within_allowed_B(world):
    run_mod, ctx, state, _ = world
    hb = {"arm": "K8", "task": "mmlu_en", "effort": "high", "pass": 0}
    rules = ctx.rules
    assert run_mod.plan_fallback(ctx, rules, {**hb, "B": 8}, [1, 2, 4, 8])["to"] == 4
    assert run_mod.plan_fallback(ctx, rules, {**hb, "B": 16}, [1, 2, 4, 8, 16])["to"] == 8
    assert run_mod.plan_fallback(ctx, rules, {**hb, "B": 8}, [1])["to"] == 1
    assert "no lower B" in run_mod.plan_fallback(ctx, rules, {**hb, "B": 1}, [1])["abort"]
    assert run_mod.plan_fallback(ctx, rules, {**hb, "B": 8})["to"] == 4  # a peer: the frozen B_choices


def test_a_session_after_a_crash_falls_back_within_allowed_B(world):
    from datetime import timedelta

    from runner import jsonl
    from runner.common import utc_iso, utc_now
    from runner.generate import write_heartbeat

    run_mod, ctx, state, _ = world
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 12, B=8)]}  # 12 > 8: the first wave stops it
    state["allowed"] = {"K8": [1, 2, 4, 8]}

    class StopAfter1:
        def __call__(self):
            p = ctx.raw / "S2/K8/mmlu_en_high.jsonl"
            return p.exists() and sum(1 for r in jsonl.read_jsonl(p) if r.get("type") == "record") >= 1

    cell = {k: v for k, v in state["plan"]["queue"][0].items() if k not in ("cell", "session")}
    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, StopAfter1(), allowed_B=[1, 2, 4, 8])
    write_heartbeat(ctx.raw / "S2" / ".heartbeat", {
        "state": "running", "pid": 2**22 + 4242, "cell": "K8:mmlu_en:high", "arm": "K8", "task": "mmlu_en",
        "effort": "high", "pass": 0, "B": 8, "utc": utc_iso(utc_now() - timedelta(hours=1))})
    assert run_mod.main(["session", "--name", "S2"]) == 0
    hs = [r for r in jsonl.read_jsonl(ctx.raw / "S2/K8/mmlu_en_high.jsonl") if r["type"] in ("header", "resume")]
    assert [h["B"] for h in hs] == [8, 4] and hs[-1]["b_fallback_from"] == 8
    assert len(jsonl.completed_keys("K8", "mmlu_en", "high", root=ctx.raw)) == 12


def test_the_pilot_clips_the_memory_rule_B_to_allowed_B(world, monkeypatch):
    run_mod, ctx, state, _ = world
    rules = copy.deepcopy(ctx.rules)
    rules["pilot"]["arms"] = ["K8", "G8"]
    rules["pilot"]["cells"] = {"K8": [{"task": "mmlu_full_en", "n": 2}], "G8": [{"task": "mmlu_full_en", "n": 2}]}
    rules["caps"] = {k: 6 for k in rules["caps"]}
    ctx.rules = rules
    monkeypatch.setattr(run_mod.memory, "effective_limit", lambda *a, **k: (1 << 36, {"chosen": "test"}))
    monkeypatch.setattr(run_mod.memory, "choose_B", lambda *a, **k: 8)
    state["allowed"] = {"K8": [1]}
    assert run_mod.main(["pilot"]) == 0
    s = json.loads(sorted(ctx.results.glob("pilot_summary_*.json"))[-1].read_text())
    assert {c["arm"]: c["B"] for c in s["cells"]} == {"K8": 1, "G8": 8}
    assert s["allowed_B"] == {"K8": [1]} and "environment" in s
