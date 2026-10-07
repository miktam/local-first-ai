"""runner/run.py session: the frozen queue, the 16 h admission rule, the end of
S3 (Tier B not started -> NOT RUN; Tier A outstanding -> S3b), the S3b overrun
ceiling, and status (HYPOTHESIS rule 6; BUILD_SPEC §5.4; RUNBOOK steps 14, 16).

The guards are replaced by no-ops (they have their own tests), the model by
the tiny checkpoint and the items by synthetic token prompts. Projections in
the fake plan are set so that the time rule bites without waiting hours.
"""

from __future__ import annotations

import signal

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from exp036_helpers import import_sibling  # noqa: E402

import tiny_checkpoint as tc  # noqa: E402
from runner import jsonl  # noqa: E402

EOS = [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]


class StubTok:
    unk_token_id = None

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": 1000, "</think>": 1001}.get(t)


def q(cell_arm, task, n, tier, proj, session="S2", B=2):
    return {"cell": f"{cell_arm}:{task}:high", "arm": cell_arm, "task": task, "effort": "high", "pass": 0, "n": n,
            "cap": 8, "B": B, "tier": tier, "projected_h": proj, "session": session}


@pytest.fixture
def sess(tmp_path, monkeypatch, tiny_vendor_dir):
    run_mod = import_sibling("runner.run")
    harness = import_sibling("port_harness")
    model = harness.load_port(tiny_vendor_dir, float32=True)
    monkeypatch.setenv("EXP036_PRIVATE", str(tmp_path / "private"))
    old_handler = signal.getsignal(signal.SIGINT)
    ctx = run_mod.Ctx(tmp_path / "exp")
    ctx.raw.mkdir(parents=True)
    monkeypatch.setattr(run_mod, "Ctx", lambda exp_dir=None: ctx)

    def items(task, n):
        return [{"id": f"{task}-{k}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k:064x}",
                 "prompt_ids": tc.random_ids(9 + k, seed=k), "prompt_text": "u"} for k in range(n)]

    monkeypatch.setattr(run_mod, "items_for_cell", items)
    monkeypatch.setattr(run_mod.Loaded, "get", lambda self, arm: (model, StubTok(), {"eos_ids": EOS}))
    for name in ("require_identity", "require_clean_tree", "require_signoff", "require_metal_limit",
                 "require_gate", "require_peers", "require_assets", "require_tier2", "require_environment"):
        monkeypatch.setattr(run_mod.guard, name, lambda *a, **k: None)
    state = {"plan": None}
    monkeypatch.setattr(run_mod.guard, "require_plan", lambda *a, **k: state["plan"])
    yield run_mod, ctx, state
    signal.signal(signal.SIGINT, old_handler)


def done(ctx, arm, task):
    return len(jsonl.completed_keys(arm, task, "high", root=ctx.raw))


def test_queue_runs_in_order_and_resumes_and_respects_the_16h_rule(sess, capsys):
    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 6.0, "_amendment_k": 7, "queue": [
        q("K8", "mmlu_en", 3, "A", 1.0),
        q("G8", "ifbench", 2, "A", 17.0),     # a new cell that would take S2 past 16 h
        q("Q36-8", "rgb_cb", 2, "B", 0.5, "S3"),
    ]}
    assert run_mod.main(["session", "--name", "S2"]) == 0
    out = capsys.readouterr().out
    assert done(ctx, "K8", "mmlu_en") == 3 and done(ctx, "G8", "ifbench") == 0
    assert "stops admitting new cells" in out
    events = [r["type"] for r in jsonl.read_jsonl(ctx.raw / "S2/session.jsonl")]
    assert events == ["start", "stop"]
    hb = jsonl.read_jsonl(ctx.raw / "S2/.heartbeat")[0]
    assert hb["state"] == "stopped"

    # S3: the 17 h Tier-A cell still does not fit 16 h; S3 ends, the Tier-B
    # cell that never started is NOT RUN, and S3b is announced.
    assert run_mod.main(["session", "--name", "S3"]) == 0
    out = capsys.readouterr().out
    assert "Tier-A outstanding — run S3b now" in out
    nr = run_mod.not_run_cells(ctx)
    assert set(nr) == {"Q36-8:rgb_cb:high"} and "Tier B" in nr["Q36-8:rgb_cb:high"]["reason"]

    # S3b admits Tier-A cells while S1 + all sessions + projection <= 44 h (6 + ~0 + 17).
    assert run_mod.main(["session", "--name", "S3b"]) == 0
    assert done(ctx, "G8", "ifbench") == 2
    assert done(ctx, "Q36-8", "rgb_cb") == 0  # Tier B never runs in S3b
    assert (ctx.raw / "S3b/G8/ifbench_high.jsonl").is_file()

    # Rerunning a finished queue is a no-op.
    assert run_mod.main(["session", "--name", "S3b"]) == 0
    assert done(ctx, "K8", "mmlu_en") == 3


def test_s3b_records_not_run_beyond_the_overrun_ceiling(sess):
    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 30.0, "queue": [q("K8", "mmlu_en", 2, "A", 15.0), q("K4", "ifbench", 2, "A", 1.0)]}
    assert run_mod.main(["session", "--name", "S3b"]) == 0
    nr = run_mod.not_run_cells(ctx)
    assert "K8:mmlu_en:high" in nr and "overrun ceiling" in nr["K8:mmlu_en:high"]["reason"]
    assert done(ctx, "K4", "ifbench") == 2  # 30 + 1 <= 44: admitted


def test_started_cells_finish_even_past_the_session_cap(sess):
    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 4, "A", 40.0)]}
    # Start the cell by hand (two records), as if S2 had been interrupted.
    cell = {k: v for k, v in state["plan"]["queue"][0].items() if k not in ("cell", "session")}

    class StopAfter2:
        def __call__(self):
            p = ctx.raw / "S2/K8/mmlu_en_high.jsonl"
            return p.exists() and sum(1 for r in jsonl.read_jsonl(p) if r.get("type") == "record") >= 2

    run_mod.execute_cell(ctx, "S2", cell, run_mod.Loaded(), None, StopAfter2())
    assert done(ctx, "K8", "mmlu_en") == 2
    assert run_mod.main(["session", "--name", "S3"]) == 0  # 40 h projection, but the cell was started
    assert done(ctx, "K8", "mmlu_en") == 4
    recs = jsonl.read_jsonl(ctx.raw / "S2/K8/mmlu_en_high.jsonl")
    assert [r["type"] for r in recs if r["type"] != "record"] == ["header", "resume"]


def test_status_reports_progress_without_accuracy(sess, capsys):
    run_mod, ctx, state = sess
    plan = {"plan": "P0", "queue": [q("K8", "mmlu_en", 3, "A", 1.0), q("G8", "ifbench", 2, "A", 2.0)]}
    state["plan"] = {"S1_hours": 1.0, "queue": plan["queue"][:1]}
    run_mod.main(["session", "--name", "S2"])
    capsys.readouterr()
    (ctx.results / "plan_fixed_20261004T180000Z.json").write_text(__import__("json").dumps(plan))
    assert run_mod.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "K8:mmlu_en:high" in out and "3/3" in out and "0/2" in out
    st = sorted(ctx.results.glob("status_*.json"))
    assert len(st) == 1
    rec = __import__("json").loads(st[0].read_text())
    cells = {c["cell"]: c for c in rec["cells"]}
    assert cells["K8:mmlu_en:high"]["done"] == 3 and cells["G8:ifbench:high"]["eta_h"] == pytest.approx(2.0)
    assert isinstance(cells["K8:mmlu_en:high"]["parse_fail"], int)  # counters, never accuracy
    assert not any("correct" in k or "accuracy" in k for c in rec["cells"] for k in c)


def test_pilot_writes_a_summary_plan_fix_can_read(sess, monkeypatch, capsys):
    import json

    run_mod, ctx, state = sess
    monkeypatch.setattr(run_mod.memory, "choose_B", lambda arm, task, cap, L, **k: 2)
    monkeypatch.setattr(run_mod.memory, "effective_limit", lambda: (100 * 2**30, {"chosen": "test"}))
    monkeypatch.setitem(ctx.rules, "caps", {k: 24 for k in ctx.rules["caps"]})  # tiny model: short completions
    assert run_mod.main(["pilot", "--cells", "K8:mmlu_full_en,K8:rgb_int_cb"]) == 0
    out = capsys.readouterr().out
    summ_path = sorted(ctx.results.glob("pilot_summary_*.json"))
    assert len(summ_path) == 1 and "pilot summary:" in out
    s = json.loads(summ_path[0].read_text())
    cells = {c["task"]: c for c in s["cells"]}
    assert set(cells) == {"mmlu_full_en", "rgb_int_cb"}
    for c in cells.values():
        assert c["n"] == len(c["lengths"]) == len(c["truncated"]) == len(c["prompt_tokens"])
        assert c["cap"] == ctx.rules["caps"]["mmlu" if c["task"].startswith("mmlu") else "rgb"]
        assert c["B"] == 2 and (ctx.exp / c["steps_file"]).is_file()
        assert "text" not in json.dumps(c)
    assert cells["mmlu_full_en"]["n"] == 8 and cells["rgb_int_cb"]["n"] == 16
    assert s["arms"]["K8"]["prefill_tokens"] > 0
    # Withheld pilot cells (RGB): hashes in the repo, text in $EXP036_PRIVATE/pilot/.
    repo_rgb = ctx.exp / cells["rgb_int_cb"]["file"]
    assert all("text" not in r for r in jsonl.read_jsonl(repo_rgb))
    assert list((ctx.private / "pilot").glob("*/K8/rgb_int_cb_high.jsonl"))
    # plan_fix reads the step logs back.
    from runner import plan_fix

    steps = plan_fix.load_steps(ctx.exp, plan_fix.merge_summaries([s]))
    assert len(steps["K8"]) > 0 and all(st["type"] == "step" for st in steps["K8"])
    man = jsonl.read_jsonl(ctx.evidence / "withheld_manifest.jsonl")
    assert any(r["path"].endswith("rgb_int_cb_high.jsonl") for r in man)


def test_an_exception_inside_a_cell_is_a_crash_and_the_next_start_falls_back(sess, monkeypatch):
    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 3, "A", 1.0, B=2)]}
    real = run_mod.run_cell

    def oom_after_first_step(*a, **k):
        def boom(gen):
            raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory")
        return real(*a, **k, on_step=boom)

    monkeypatch.setattr(run_mod, "run_cell", oom_after_first_step)
    with pytest.raises(RuntimeError):
        run_mod.main(["session", "--name", "S2"])
    hb = jsonl.read_jsonl(ctx.raw / "S2/.heartbeat")[0]
    assert hb["state"] == "failed" and hb["cell"] == "K8:mmlu_en:high" and hb["B"] == 2
    monkeypatch.setattr(run_mod, "run_cell", real)
    assert run_mod.main(["session", "--name", "S2"]) == 0
    recs = jsonl.read_jsonl(ctx.raw / "S2/K8/mmlu_en_high.jsonl")
    hs = [r for r in recs if r["type"] in ("header", "resume")]
    assert [h["B"] for h in hs] == [2, 1] and hs[-1]["b_fallback_from"] == 2
    assert all(r.get("b_fallback_from") == 2 for r in recs if r["type"] == "record")
    assert done(ctx, "K8", "mmlu_en") == 3


def test_a_refusal_is_not_a_crash(sess, monkeypatch):
    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 2, "A", 1.0), q("G8", "ifbench", 2, "A", 1.0)]}

    def refuse_g8(arm, *a, **k):
        if arm == "G8":
            run_mod.guard.refuse("peer check did not pass for G8")

    monkeypatch.setattr(run_mod.guard, "require_peers", refuse_g8)
    with pytest.raises(SystemExit):
        run_mod.main(["session", "--name", "S2"])
    assert jsonl.read_jsonl(ctx.raw / "S2/.heartbeat")[0]["state"] == "stopped"
    assert run_mod.detect_crash(ctx, ctx.rules) is None


def test_a_crash_between_cells_needs_no_fallback(sess):
    from datetime import timedelta

    from runner.common import utc_iso, utc_now
    from runner.generate import write_heartbeat

    run_mod, ctx, state = sess
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 2, "A", 1.0), q("G8", "ifbench", 2, "A", 1.0)]}
    assert run_mod.main(["session", "--name", "S2"]) == 0
    # A stale "running" heartbeat naming the finished first cell.
    write_heartbeat(ctx.raw / "S2/.heartbeat", {"state": "running", "pid": 2**22 + 777, "cell": "K8:mmlu_en:high",
                                                "arm": "K8", "task": "mmlu_en", "effort": "high", "pass": 0, "B": 2,
                                                "utc": utc_iso(utc_now() - timedelta(hours=2))})
    assert run_mod.main(["session", "--name", "S2"]) == 0
    assert run_mod.not_run_cells(ctx) == {}
    hs = [r for r in jsonl.read_jsonl(ctx.raw / "S2/K8/mmlu_en_high.jsonl") if r["type"] != "record"]
    assert len(hs) == 1  # untouched: no fallback resume header


def test_b4_without_tier2_is_skipped_not_fatal_and_not_run_at_the_end_of_s3(sess, monkeypatch, capsys):
    """Review fix 2026-10-03: a B4 ladder cell reached without the Tier-2 amendment at HEAD no longer ends the
    session (rc 1, later cells unrun, S3-end bookkeeping skipped). It is skipped, the queue continues, and at the
    end of S3 the unstarted B4 is NOT RUN with the reason."""
    run_mod, ctx, state = sess
    monkeypatch.setattr(run_mod.guard, "tier2_at_head", lambda *a, **k: False)
    bench_q = {"cell": "B4:ladder", "arm": "bench", "task": "ladder", "effort": "na", "pass": 0, "n": 0,
               "tier": "B", "projected_h": 0.01, "session": "S3", "needs": "tier2"}
    state["plan"] = {"S1_hours": 6.0, "queue": [q("K8", "mmlu_en", 2, "A", 0.5), bench_q,
                                                q("G8", "ifbench", 2, "B", 0.5, "S3")]}
    assert run_mod.main(["session", "--name", "S3"]) == 0
    assert done(ctx, "K8", "mmlu_en") == 2 and done(ctx, "G8", "ifbench") == 2
    assert "B4:ladder skipped: no Tier-2 analysis amendment at HEAD" in capsys.readouterr().err
    nr = run_mod.not_run_cells(ctx)
    assert "Tier-2" in nr["B4:ladder"]["reason"]


# ------------------------------------------------- pilot completeness (review fixes 2026-10-03)


@pytest.fixture
def pilot_env(sess, monkeypatch):
    run_mod, ctx, state = sess
    rules = __import__("copy").deepcopy(ctx.rules)
    rules["pilot"]["arms"] = ["K8"]
    rules["pilot"]["cells"] = {"K8": [{"task": "mmlu_full_en", "n": 2}, {"task": "ifbench_pilot", "n": 2}],
                               "K4": [{"task": "mmlu_full_en", "n": 2}]}
    rules["caps"] = {k: 8 for k in rules["caps"]}
    ctx.rules = rules
    monkeypatch.setattr(run_mod.memory, "effective_limit", lambda *a, **k: (1 << 36, {"chosen": "test"}))
    monkeypatch.setattr(run_mod.memory, "choose_B", lambda *a, **k: 2)
    return run_mod, ctx


def test_a_complete_pilot_writes_a_summary_marked_complete(pilot_env):
    run_mod, ctx = pilot_env
    assert run_mod.main(["pilot"]) == 0
    s = sorted(ctx.results.glob("pilot_summary_*.json"))
    import json as _json

    rec = _json.loads(s[-1].read_text())
    assert rec["complete"] is True and rec["requested"] == [["K8", "ifbench_pilot"], ["K8", "mmlu_full_en"]]
    assert [c["task"] for c in rec["cells"]] == ["mmlu_full_en", "ifbench_pilot"]


def test_an_interrupted_pilot_writes_no_summary_and_abort_pilot_moves_it(pilot_env, monkeypatch, capsys):
    run_mod, ctx = pilot_env

    class StopAfterFirstCell(run_mod.Stop):
        def install(self):
            return self

        def __call__(self):
            return any(ctx.results.glob("pilot/*/K8/mmlu_full_en_high.jsonl"))

    monkeypatch.setattr(run_mod, "Stop", StopAfterFirstCell)
    assert run_mod.main(["pilot"]) == 1
    assert not list(ctx.results.glob("pilot_summary_*.json"))
    err = capsys.readouterr().err
    stamp = err.split("abort-pilot --stamp ")[1].split(",")[0]
    assert run_mod.main(["abort-pilot", "--stamp", stamp]) == 0
    assert not (ctx.results / "pilot" / stamp).exists()
    note = (ctx.aborted / f"{stamp}-pilot" / "NOTE.md").read_text()
    assert "interrupted or crashed" in note and "mmlu_full_en_high.jsonl" in note


def test_repilot_cells_accept_the_cell_id_and_refuse_unknown_cells(pilot_env):
    run_mod, ctx = pilot_env
    rules = ctx.rules
    assert run_mod.parse_pilot_cells("K8:mmlu_full_en:high", rules) == {("K8", "mmlu_full_en")}
    assert run_mod.parse_pilot_cells("K8:mmlu_full_en", rules) == {("K8", "mmlu_full_en")}
    for bad in ("K8:gpqa_en", "K8:mmlu_full_en:low", "K8", ""):
        with pytest.raises(SystemExit):
            run_mod.parse_pilot_cells(bad, rules)


def test_pilot_without_an_arm_needs_a_record_that_excludes_it(pilot_env):
    run_mod, ctx = pilot_env
    with pytest.raises(SystemExit):
        run_mod.main(["pilot", "--without", "K4"])
    (ctx.results / "gate").mkdir(parents=True)
    (ctx.results / "gate" / "gate_20261004T120000Z.json").write_text(
        '{"mode": "real", "verdict": {"K8": "PASS", "K4": "FAIL"}}')
    assert run_mod.pilot_without("K4", ctx) == {"K4": "gate K4 FAIL (gate_20261004T120000Z.json)"}
