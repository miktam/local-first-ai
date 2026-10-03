# SPDX-License-Identifier: MIT
"""bench/run_bench.py: fixed cell order, guards before the cool-down, the
cool-down once before the speed cells, skip-if-complete, incomplete outputs
moved to aborted/, failure isolation and exit codes. The cells themselves are
replaced by fakes that write a complete output."""

from __future__ import annotations

import json

import pytest

from test_bench_support import bench_env  # noqa: F401  (fixture)

from bench import common, run_bench


def _fakes(log: list, fail: tuple = (), could_not: tuple = ()):
    def make(cell):
        def fn(results_dir, **kw):
            log.append((cell, kw.get("cool") is not None))
            if cell in fail:
                raise RuntimeError(f"{cell} broke")
            if cell in could_not:
                raise common.CouldNotRun(f"{cell} cannot")
            path = common.new_output_path(results_dir, cell)
            if path.suffix == ".json":
                common.write_json_new(path, {"complete": True})
            else:
                with common.JsonlCell(path, {"cell": cell}) as w:
                    w.finish({})
            return path

        return fn

    return {c: make(c) for c in run_bench.CELL_ORDER}


def test_parse_cells_fixed_order_and_unknown_names():
    assert run_bench.parse_cells("c1,tokenizer,speed,kl,fit,speed_desc") == ["speed", "speed_desc", "fit", "kl", "tokenizer", "c1"]
    assert run_bench.parse_cells("ladder,speed") == ["speed", "ladder"]
    for bad in ("", "speed,warp", "H1"):
        with pytest.raises(ValueError):
            run_bench.parse_cells(bad)


def test_runs_in_order_guards_first_cools_once(bench_env):
    log = []
    cells = run_bench.parse_cells("c1,tokenizer,speed_desc,speed,fit,kl")
    code, status = run_bench.run(cells, bench_env.results, _fakes(log))
    assert code == 0
    assert [c for c, _ in log] == ["speed", "speed_desc", "fit", "kl", "tokenizer", "c1"]
    assert [cool for c, cool in log if c in ("speed", "speed_desc")] == [True, True]
    assert bench_env.calls[0] == ("identity",)
    assert {c for c in bench_env.calls if c[0] == "gate"} == {("gate", "K4"), ("gate", "K8")}
    assert bench_env.sleeps == [600]  # one cool-down, before the speed cells
    assert all(s["status"] == "complete" for s in status.values())

    # a second run skips every complete cell and does not cool down again
    log.clear()
    bench_env.sleeps.clear()
    code, status = run_bench.run(cells, bench_env.results, _fakes(log))
    assert code == 0 and log == [] and bench_env.sleeps == []
    assert all(s["status"] == "skipped" for s in status.values())


def test_incomplete_output_is_moved_then_the_cell_reruns(bench_env):
    p = common.new_output_path(bench_env.results, "fit")
    with pytest.raises(RuntimeError):
        with common.JsonlCell(p, {"cell": "fit"}):
            raise RuntimeError
    log = []
    code, status = run_bench.run(["fit"], bench_env.results, _fakes(log))
    assert code == 0 and log == [("fit", False)]
    assert status["fit"]["moved_to_aborted"] == [p.name]
    moved = list((bench_env.root / "aborted").glob(f"*-bench-fit/{p.name}"))
    assert len(moved) == 1 and not common.is_complete(moved[0])
    assert common.is_complete(bench_env.results / "bench" / status["fit"]["file"])


def test_a_failing_cell_does_not_stop_the_others(bench_env):
    log = []
    code, status = run_bench.run(["fit", "kl", "tokenizer"], bench_env.results, _fakes(log, fail=("kl",)))
    assert code == 1
    assert [c for c, _ in log] == ["fit", "kl", "tokenizer"]
    assert status["kl"] == {"status": "failed", "error_type": "RuntimeError"}
    assert status["tokenizer"]["status"] == "complete"


def test_thermals_that_never_settle_mean_could_not_run(bench_env):
    bench_env.thermal = ["2"]
    log = []
    code, status = run_bench.run(["speed", "speed_desc", "fit"], bench_env.results, _fakes(log))
    assert code == 3
    assert status["speed"]["status"] == "could not run" and "serious" in status["speed"]["reason"]
    assert status["speed_desc"] == status["speed"]
    assert sum(bench_env.sleeps) == 600 + 1200  # waited once, not once per speed cell
    assert status["fit"]["status"] == "complete" and [c for c, _ in log] == ["fit"]


def test_missing_tier2_amendment_only_stops_the_ladder(bench_env, monkeypatch):
    def refuse():
        raise SystemExit("no Tier-2 amendment")

    monkeypatch.setattr(common, "require_tier2", refuse)
    log = []
    code, status = run_bench.run(["fit", "ladder"], bench_env.results, _fakes(log))
    assert code == 3 and [c for c, _ in log] == ["fit"]
    assert status["ladder"]["status"] == "could not run"


def test_a_refused_gate_stops_before_anything_runs(bench_env, monkeypatch):
    def refuse(arm):
        raise SystemExit(f"no gate PASS for {arm}")

    monkeypatch.setattr(common, "require_gate", refuse)
    log = []
    with pytest.raises(SystemExit):
        run_bench.run(["speed", "tokenizer"], bench_env.results, _fakes(log))
    assert log == [] and bench_env.sleeps == []


def test_main_prints_a_json_last_line(bench_env, monkeypatch, capsys):
    monkeypatch.setattr(run_bench, "runners", lambda: _fakes([]))
    code = run_bench.main(["--cells", "tokenizer", "--results-dir", str(bench_env.results)])
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and last["exit"] == 0 and last["cells"]["tokenizer"]["status"] == "complete"
    with pytest.raises(SystemExit):
        run_bench.main(["--cells", "nonsense"])


def test_cell_kolibri_arms_cover_what_the_cells_load():
    from bench import batch_flip, fit, ladder, speed

    assert set(run_bench.CELL_KOLIBRI_ARMS["speed"]) == {a for a in speed.H1_ARMS if common.is_kolibri(a)}
    assert set(run_bench.CELL_KOLIBRI_ARMS["speed_desc"]) == {a for a in speed.DESC_ARMS if common.is_kolibri(a)}
    assert run_bench.CELL_KOLIBRI_ARMS["fit"] == (fit.D1_ARM,)
    assert run_bench.CELL_KOLIBRI_ARMS["c1"] == (batch_flip.C1_ARM,)
    assert set(run_bench.CELL_KOLIBRI_ARMS["ladder"]) == {a for a in ladder.LADDER_ARMS if common.is_kolibri(a)}
    assert set(run_bench.CELL_KOLIBRI_ARMS["kl"]) == set(common.FAMILY_PAIRS["kolibri"])
