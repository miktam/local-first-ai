"""Cross-area contracts (integration of the exp_036 kit; BUILD_LOG.md).

Each test pins one place where two areas meet, so a later edit on one side cannot silently break the other:

- tools/peer_check.py's batched path calls the gate's G5 functions with their real signatures
  (floor_and_decode, parity_bound, batch_parity) and returns the bound and the mid-run admission count, on the
  Amendment 5 chat wrapper followed by the text stream;
- the plan's frozen category counts (runner/plan_fix.py "category_counts") feed the H2 post-stratification
  in analysis/verdicts.py;
- the pilot is defined consistently in runner/plan_rules.json (cells and n), tasks/selection_rules.json
  (manifest per task), runner/run.py (MANIFEST_SET, PILOT_TASK) and tasks/build_manifests.py (set sizes);
- every task the plan can queue has a manifest set, and scorers/score_all.py finds that manifest by name;
- the tests resolve the Kolibri tokenizer by BUILD_SPEC's name, $EXP036_TOK.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------------------- peer check x G5


def test_peer_batched_path_runs_through_the_gate_g5_functions(tiny_vendor_dir):
    pytest.importorskip("mlx.core")
    import tiny_checkpoint as tc
    from exp036_helpers import import_sibling

    from tools import peer_check as pc

    harness = import_sibling("port_harness")
    model = harness.load_port(tiny_vendor_dir, float32=True)
    base = tc.random_ids(1800, seed=7)
    wrapper = tc.random_ids(17, seed=11)
    res = pc.batched_path(model, tiny_vendor_dir, "G8", base_ids=base, wrapper_ids=wrapper)
    assert res["wrapper"]["prompt_ids"] == [int(i) for i in wrapper] and "Amendment 5" in res["text"]
    par = res["parity"]
    assert par["B"] == 8 and par["n_sequences"] == len(pc.BATCH_LENGTHS) > 8
    assert par["admitted_mid_run"] > 0 and par["max_live"] <= 8
    assert par["prompt_lengths"] == list(pc.BATCH_LENGTHS) and par["max_tokens"] == list(pc.BATCH_MAX_TOKENS)
    assert min(pc.BATCH_LENGTHS) == 37 and max(pc.BATCH_LENGTHS) == 1100  # HYPOTHESIS: mixed lengths 37-1,100
    fl = res["noise_floor"]
    assert fl["prefill_steps"] == [2048, 64] and fl["per_text"]["base"]["positions"] == list(pc.FLOOR_RANGE)
    th = json.loads((EXP / "gate" / "thresholds.json").read_text())["G5"]
    assert res["kl_bound"] == max(th["parity_kl_floor"], th["parity_kl_factor"] * fl["floor_kl"])
    assert res["eos_ids"] == [tc.EOS_TOKEN_ID, tc.PAD_TOKEN_ID]  # the peer's own EOS ids, not Kolibri's
    # fp32: batched equals single far inside the floor-derived bound
    assert res["ok"] is True, (par["mean_kl"], par["top1_dis"], res["kl_bound"], res["dis_bound"])


def test_peer_batched_path_refuses_a_short_base(tiny_vendor_dir):
    from tools import peer_check as pc

    with pytest.raises(ValueError, match="base tokens"):
        pc.batched_path(object(), tiny_vendor_dir, "G8", base_ids=list(range(100)), wrapper_ids=[1, 2, 3])


# ------------------------------------------------------------------------------------ plan x verdicts


def test_plan_category_counts_feed_the_post_stratification():
    from analysis import verdicts

    counts = {"en": {"math": 1351, "law": 1101}, "de": {"math": 1351, "law": 1100},
              "_meta": {"schema": "exp036 MMLU-ProX category counts v1"}}
    p = verdicts.normalise_plan({"plan": "P0", "category_counts": counts, "queue": []})
    assert p["post_strat_counts"] == {"en": counts["en"], "de": counts["de"]}
    ctx = verdicts.Context(None, {"plan": "P0", "category_counts": counts}, verdicts.Scores([]), {}, {}, {})
    assert ctx.post_strat("en") == {"math": 1351.0, "law": 1101.0}
    # an explicit post_strat_counts wins; an empty category_counts falls back to the manifest file
    p2 = verdicts.normalise_plan({"post_strat_counts": {"en": {"x": 1}}, "category_counts": counts})
    assert p2["post_strat_counts"] == {"en": {"x": 1}}
    assert "post_strat_counts" not in verdicts.normalise_plan({"category_counts": {}})


# ------------------------------------------------------------------------------ pilot and task names


def _rules():
    return json.loads((EXP / "runner" / "plan_rules.json").read_text(encoding="utf-8"))


def _selection():
    return json.loads((EXP / "tasks" / "selection_rules.json").read_text(encoding="utf-8"))


def test_pilot_is_defined_consistently():
    pytest.importorskip("mlx.core")
    from runner import run
    from tasks import build_manifests as bm

    rules, sel = _rules(), _selection()
    assert rules["pilot"]["arms"] == [a for a in sel["pilot"] if a != "rule"]
    for arm in rules["pilot"]["arms"]:
        cells = rules["pilot"]["cells"][arm]
        mine = [(run.PILOT_TASK[c["task"]], run.MANIFEST_SET[c["task"]]) for c in cells]
        theirs = [(e["task"], e["manifest"]) for e in sel["pilot"][arm]]
        assert mine == theirs, arm
        for c in cells:
            sd = bm.set_def(run.MANIFEST_SET[c["task"]])
            assert int(c["n"]) == sd.expected_n, (arm, c["task"], c["n"], sd.expected_n)
    # HYPOTHESIS "Pilot": K8 adds AIME (4, discarded); K4 has no forced-answer cell
    assert ("aime_en_pilot", 4) in [(c["task"], c["n"]) for c in rules["pilot"]["cells"]["K8"]]
    assert "rgb_int_forced" not in [c["task"] for c in rules["pilot"]["cells"]["K4"]]


def test_every_queued_task_has_a_manifest_the_scorers_find():
    pytest.importorskip("mlx.core")
    from runner import run
    from scorers import score_all
    from tasks import build_manifests as bm

    rules = _rules()
    queued = {c["task"] for c in rules["tier_a_queue"]}
    for cells in rules["tier_b_cells"].values():
        queued |= {c["task"] for c in cells if c["arm"] != "bench"}
    set_names = {sd.name for sd in bm.set_defs()}
    for task in sorted(queued):
        mset = run.MANIFEST_SET[task]
        assert mset in set_names, (task, mset)
        canon = score_all.canonical_task(task)
        tried = [canon] + [a for a, t in score_all.TASK_ALIASES.items() if t == canon]
        assert mset in tried, (task, mset, tried)
        assert score_all.TASKS[canon].withheld == bm.set_def(mset).withheld, task
    for task, mset in run.MANIFEST_SET.items():
        assert mset in set_names, (task, mset)


def test_tests_accept_the_build_spec_tokenizer_variable():
    # BUILD_SPEC §1/§6 name the variable EXP036_TOK; EXP036_TOKENIZER_DIR is the older alias. A test that reads
    # the alias directly must also read EXP036_TOK (or use exp036_helpers.kolibri_tok_env / conftest's fixture).
    bad = []
    for p in sorted((EXP / "tests").glob("*.py")):
        s = p.read_text(encoding="utf-8")
        if re.search(r"environ(?:\.get)?\(\s*[\"']EXP036_TOKENIZER_DIR|environ\[\s*[\"']EXP036_TOKENIZER_DIR", s) \
                and "EXP036_TOK\"" not in s and "EXP036_TOK'" not in s and "kolibri_tok_env" not in s:
            bad.append(p.name)
    assert not bad, bad


# ------------------------------------------------------------------------------- RUNBOOK x the CLIs

_PY_TOKENS = ('"$PY"', "$PY", "python", "python3")


def _runbook_commands() -> list[tuple[str, str | None, list[str]]]:
    """(script, subcommand, flags) for every kit script a RUNBOOK bash block runs with the kit's python."""
    import shlex

    text = (EXP / "RUNBOOK.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    blocks += re.findall(r'`("\$PY" [^`]+)`', text)  # inline commands, e.g. the mini's steps after step 19
    out = []
    for b in blocks:
        b = re.sub(r"\\\n\s*", " ", b)
        for line in b.splitlines():
            line = line.split(" #", 1)[0]
            for seg in re.split(r"&&|\|\||;|\|", line):
                try:
                    toks = shlex.split(seg, posix=False)
                except ValueError:
                    continue
                for i, t in enumerate(toks):
                    if t.endswith(".py") and i > 0 and (toks[i - 1] in _PY_TOKENS or toks[i - 1].endswith("python")):
                        rest = toks[i + 1:]
                        sub = rest[0] if rest and not rest[0].startswith("-") and toks[i].endswith("run.py") else None
                        flags = [x.split("=", 1)[0] for x in rest if x.startswith("--")]
                        out.append((toks[i], sub, flags))
    return out


def test_runbook_names_only_existing_scripts_and_flags():
    import os
    import subprocess
    import sys

    cmds = _runbook_commands()
    assert len(cmds) >= 20, cmds  # the parser found the runbook's commands
    seen = {}
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    problems = []
    for script, sub, flags in cmds:
        path = EXP / script
        if not path.is_file():
            problems.append(f"{script}: no such file")
            continue
        key = (script, sub)
        if key not in seen:
            argv = [sys.executable, str(path)] + ([sub] if sub else []) + ["--help"]
            p = subprocess.run(argv, cwd=str(EXP), env=env, capture_output=True, text=True, timeout=300)
            seen[key] = (p.returncode, p.stdout + p.stderr)
        rc, help_text = seen[key]
        if rc != 0:
            problems.append(f"{script} {sub or ''} --help exited {rc}")
            continue
        for f in flags:
            if f not in help_text:
                problems.append(f"{script} {sub or ''}: {f} is not in --help")
    assert not problems, problems


def test_runbook_bench_cells_parse():
    pytest.importorskip("mlx.core")
    from bench import run_bench

    text = (EXP / "RUNBOOK.md").read_text(encoding="utf-8")
    specs = re.findall(r"run_bench\.py --cells ([\w,]+)", text)
    assert specs
    for s in specs:
        assert run_bench.parse_cells(s)


# ------------------------------------------------------------------------------ runner projection


def test_step_model_fit_is_non_negative_and_unchanged_when_physical():
    import numpy as np

    from runner import simulate

    rng = np.random.default_rng(3)
    true = (0.02, 0.003, 4e-7)
    steps = []
    for _ in range(300):
        n, L = int(rng.integers(1, 9)), float(rng.integers(200, 9000))
        steps.append({"n_live": n, "padded_len": L,
                      "step_seconds": true[0] + true[1] * n + true[2] * n * L + float(rng.normal(0, 1e-5))})
    a, b, c = simulate.fit_step_model(steps)  # physical data: the plain least-squares answer
    A = np.array([[1.0, s["n_live"], s["n_live"] * s["padded_len"]] for s in steps])
    ls = np.linalg.lstsq(A, np.array([s["step_seconds"] for s in steps]), rcond=None)[0]
    assert np.allclose((a, b, c), ls, rtol=0, atol=0)
    # tiny-model noise (the dry run's pilot): a decreasing trend in padded_len must not give c < 0
    noisy = [{"n_live": 4, "padded_len": 100.0 + k, "step_seconds": 0.01 - 1e-5 * k + 1e-4 * ((k * 7) % 3)}
             for k in range(60)]
    a, b, c = simulate.fit_step_model(noisy)
    assert min(a, b, c) >= 0
    assert simulate.step_time((a, b, c), 14, 5000.0) > 0


# ------------------------------------------------------------------------------------ the dry run


def test_dry_run_scales_only_sizes_and_caps(tmp_path):
    from tools import dry_run

    rules = _rules()
    s = dry_run.scale_plan_rules(rules)
    assert set(s) == set(rules) | {"dry_run"}
    for k in ("version", "budget", "B_choices", "plans", "tier_b_order", "pilot", "rows", "projection",
              "crash_fallback", "kv_bytes_per_token", "fixed_window_bytes", "cap_raise"):
        if k != "cap_raise":
            assert s[k] == rules[k], k
    assert [(c["arm"], c["task"], c["effort"]) for c in s["tier_a_queue"]] == \
           [(c["arm"], c["task"], c["effort"]) for c in rules["tier_a_queue"]]
    assert all(c["n"] == "nM" or int(c["n"]) <= 6 for c in s["tier_a_queue"])
    assert set(s["caps"]) == set(rules["caps"]) and max(s["caps"].values()) < min(rules["caps"].values())
    assert {k: {kk: vv for kk, vv in v.items() if kk != "nM"} for k, v in s["plan_defs"].items()} == \
           {k: {kk: vv for kk, vv in v.items() if kk != "nM"} for k, v in rules["plan_defs"].items()}
    # Amendment 4: n_M need not be a multiple of 14; the post-stratified MMLU rows need every category, so each scaled
    # n_M must give every category an item under the Webster allocation over the registered Lite counts (16 is the
    # smallest such n; at 14, two categories would have none).
    from tasks import mmlu_prox

    lite = json.loads((EXP / "tasks" / "selection_rules.json").read_text(encoding="utf-8"))["mmlu_prox"]["lite_category_counts"]
    for n in {d["nM"] for d in s["plan_defs"].values()} | set(s["nM_ladder"]):
        assert min(mmlu_prox.allocation(lite, n).values()) >= 1, n
    assert min(mmlu_prox.allocation(lite, dry_run.DRY_NM_SMALL - 1).values()) == 0  # the smallest that works


def test_dry_run_rewords_text_but_not_layout(tmp_path):
    import tasks_synthetic as syn

    from tools import dry_run

    syn.write_all(tmp_path)
    before = {p: p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file()}
    dry_run.reword_synthetic(tmp_path)
    for p, old in before.items():
        new = p.read_text(encoding="utf-8")
        if p.suffix == ".jsonl":
            for a, b in zip(old.splitlines(), new.splitlines()):
                assert set(json.loads(a)) == set(json.loads(b)), p  # keys untouched
        if p.suffix == ".csv":
            assert old.splitlines()[0] == new.splitlines()[0], p  # header untouched
    assert "Synthetic question" not in (tmp_path / "gpqa" / "gpqa_diamond.csv").read_text(encoding="utf-8")
