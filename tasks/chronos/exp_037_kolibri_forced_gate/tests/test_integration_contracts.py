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
- the tests resolve the Kolibri tokenizer by BUILD_SPEC's name, $EXP036_TOK;
- exp_037 (DESIGN §12 W7): the gate's G1 leaves out exactly the heavy tiny tests
  (gate/checks/g1_synthetic.EXCLUDE, §3.3); phase 6 (gate/run_gate.py) scores, with R1, every
  sequence both arms' G5 blocks need (gate/checks/g5_runner.py; G1 item 8); the phase files keep
  a non-finite measurement exactly, so it reaches gate/rules.py as a NaN and fails there.

The RUNBOOK assertions read exp_037's RUNBOOK.md, which W12 writes (stage 5): until W12's
hand-off note switches DOC_ASSERTIONS_ON on they pass with a warning (as tests/test_gate_thresholds.py
does), and fail if the file is missing once switched on.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
# Switched to True after W12's hand-off note (DESIGN §12 W12, W15; final review W15-01).
DOC_ASSERTIONS_ON = True


def _runbook_text() -> str | None:
    p = EXP / "RUNBOOK.md"
    if p.is_file():
        return p.read_text(encoding="utf-8")
    if DOC_ASSERTIONS_ON:
        pytest.fail("RUNBOOK.md is missing, and the document assertions are switched on (W12 has handed off)")
    warnings.warn("RUNBOOK.md is not written yet (W12); its assertions are off until W12's hand-off note")
    return None


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

    text = _runbook_text()
    if text is None:
        return []
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

    if _runbook_text() is None:
        return
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

    text = _runbook_text()
    if text is None:
        return
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


# ------------------------------------------------------------------------------- the gate (exp_037, W7)


def test_g1_leaves_out_exactly_the_heavy_tiny_tests():
    """DESIGN §3.3: the gate's G1 run excludes the heavy tiny tests, and only them."""
    from gate.checks import g1_synthetic

    assert g1_synthetic.EXCLUDE == ("test_gate_drivers_tiny.py", "test_gate_end_to_end_tiny.py", "test_gate_ftiny.py")
    for name in g1_synthetic.EXCLUDE:
        assert (EXP / "tests" / name).is_file(), name


def test_phase_files_keep_a_nan_and_the_rules_fail_it(tmp_path):
    """A NaN the port produces is a FAIL in gate/rules.py, never a crash (exit 3): the
    phase files keep non-finite floats exactly, and phase 7 reads them back."""
    import numpy as np

    from gate import common, rules, run_gate

    data = {"T1-8": {"mean_kl": float("nan"), "top1_decisive": None}, "T9": {"mean_kl": np.float32(0.01)},
            "inf": [float("inf"), -float("inf")], "ok": np.float64(1.5)}
    p = common.write_new_json(tmp_path / "phase4.json", run_gate._wire({"data": data}))
    text = p.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text  # strict JSON on disk
    back = run_gate._unwire(common.read_json(p))["data"]
    assert math.isnan(back["T1-8"]["mean_kl"]) and back["inf"] == [math.inf, -math.inf] and back["ok"] == 1.5
    c = rules.g4_n(back, common.load_thresholds())
    assert c["state"] == rules.FAIL and c["legs"]["T1-8"]["ok"] is False and c["legs"]["T9"]["ok"] is True


def _logprobs(rng, n: int, vocab: int):
    import numpy as np

    x = rng.normal(size=(n, vocab)).astype(np.float64)
    return (x - np.log(np.exp(x).sum(axis=-1, keepdims=True))).astype(np.float32)


def test_phase6_scores_both_arms_sequences(tiny_pattern5_dir, tmp_path):
    """G1 item 8 (DESIGN §3.3, §3.13-§3.14): with arms (K8, K4), every sequence that
    gate/rules.py's R-anchor, R-greedy and (d) need for either arm gets an R1 row in phase 6,
    identical sequences are scored once, mutant 26's block (R-parity only) is not scored, and
    every log-prob file a phase JSON names exists with its sha256. G5-BP-lean's control is
    mutant 27 (decision (b)); phase 4 writes probe 22's block under "probes" (tiny mode), and
    phase 6 anchors and reduces it too, while rules.g5_bp reads only the required control."""
    pytest.importorskip("mlx.core")
    import numpy as np

    from gate import common, rules, run_gate
    from gate.checks import g5_runner as R

    models = tmp_path / "models"
    models.mkdir()
    (models / "Kolibri-1-BF16").symlink_to(tiny_pattern5_dir)
    vocab = common.read_json(tiny_pattern5_dir / "config.json")["vocab_size"]
    ctx = run_gate.GateContext(models_dir=models, work_dir=tmp_path / "work", thresholds=common.load_thresholds(),
                               tiny=True, results_dir=tmp_path / "results", builds_dir=models, subprocess_phases=False,
                               utc="20261006T120000Z")
    ctx.run_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    prompts = [{"id": f"G5_{j}", "lang": "en", "effort": "none", "ids": [int(x) for x in rng.integers(0, 1000, 20 + j)]}
               for j in range(3)]

    def seqs(arm, check, kind, items, labels):
        out_dir = R.g5_out_dir(ctx.utc, arm, check)
        out = []
        for (name, prompt, toks), label in zip(items, labels):
            files = {kind: R.write_logprobs(out_dir, name, kind, _logprobs(rng, len(toks), vocab)),
                     R.SINGLE: R.write_logprobs(out_dir, name, R.SINGLE, _logprobs(rng, len(toks), vocab))}
            out.append({"seq": name, "prompt_ids": prompt, "tokens": toks, "n": len(toks), "label": label,
                        "finish_reason": "length", "files": files})
        return out

    def block(arm, check, check_dir, kind, items, labels, mutant=None, B=1, **extra):
        return {"schema": R.BLOCK_SCHEMA, "id": f"{arm}/{check_dir}", "check": check, "arm": arm, "mutant": mutant,
                "needs_r1": not (check == "g5_r1" and mutant is not None), "B": B,
                "sequences": seqs(arm, check_dir, kind, items, labels), **extra}

    toks = {p["id"]: [int(x) for x in rng.integers(0, 1000, 4)] for p in prompts}
    shared = [(p["id"], p["ids"], toks[p["id"]]) for p in prompts]  # the runner's tokens equal greedy's
    conts = [{"id": p["id"], "lang": "en", "effort": "none", "prompt_tokens": len(p["ids"]),
              "prompt_ids_sha256": common.ids_sha256(p["ids"]), "continuation": toks[p["id"]], "finished": "length"}
             for p in prompts]
    bp_items = [(f"A{j:02d}", prompts[j % 3]["ids"][:10 + j], [int(x) for x in rng.integers(0, 1000, 3)]) for j in range(4)]
    k4_items = [(p["id"], p["ids"], [int(x) for x in rng.integers(0, 1000, 5)]) for p in prompts]
    labels = ["first_wave", "first_wave", "mid_run", "mid_run"]
    parity = {"mean_kl": 1e-9, "top1_dis": 0.0, "n": 12}
    trace = {"admitted_mid_run": 2, "max_live": 2}
    with run_gate.tiny_env(ctx):  # every log-prob file under the test's own $EXP036_WORK/exp037
        assert common.work37_dir() == ctx.work37
        ph4 = {"data": {
            "greedy": R.greedy_block(conts, prompts, arm="K8"),
            "g5_r1": block("K8", "g5_r1", "g5_r1", "runner", shared, [None] * 3, parity=parity),
            "g5_bp": {"8": block("K8", "g5_bp", "g5_bp_B8", "batched", bp_items, labels, B=8, trace=trace)},
            "mutants": {"g5_r1_K8": {"26": block("K8", "g5_r1", "g5_r1_m26", "runner", shared, [None] * 3, mutant=26,
                                                 parity={"mean_kl": 0.5, "top1_dis": 0.5, "n": 12})},
                        "g5_bp_K8_B8": {"27": block("K8", "g5_bp", "g5_bp_m27", "batched", bp_items, labels, mutant=27,
                                                    B=8, trace=trace)}},
            "probes": {"g5_bp_K8_B8": {"22": block("K8", "g5_bp", "g5_bp_m22", "batched", bp_items, labels, mutant=22,
                                                   B=8, trace=trace)}}}}
        ph5 = {"data": {"g5_r1": block("K4", "g5_r1", "g5_r1", "runner", k4_items, [None] * 3, parity=parity),
                        "g5_bp": {"8": block("K4", "g5_bp", "g5_bp_B8", "batched", bp_items, labels, B=8, trace=trace)}}}
        for n, ph in (("4", ph4), ("5", ph5)):
            common.write_new_json(run_gate.phase_path(ctx, n), run_gate._wire(ph))
        run_gate.run_phase("6", ctx)
    out = run_gate.read_phase(ctx, "6")["data"]
    red = out["reduced"]
    # files: K8 runner 3, K8 B8 4, mutant 26 3, mutant 27 4, probe 22 4, K4 runner 3, K4 B8 4 sequences, two files each
    assert out["logprob_files"] == {"n_files": 2 * (3 + 4 + 3 + 4 + 4 + 3 + 4), "problems": []}
    # scored once each: greedy = K8's runner (3), K8 B8 = mutant 27 = probe 22 = K4 B8 (4), K4's runner (3);
    # mutant 26 not at all
    assert out["anchor"]["n_sequences"] == 10 and out["anchor"]["n_rows"] == 12 + 12 + 15
    assert set(red["g5_r1"]) == {"K8", "K4"} and set(red["g5_bp"]) == {"K8", "K4"}
    assert set(red["g5_bp_mutants"]) == {"27", "22"} and set(red["g5_r1_mutants"]) == {"26"}
    assert red["g5_greedy"]["n_positions"] == 12 and red["g5_r1"]["K8"]["greedy"]["n"] == 12
    for arm, n in (("K8", 12), ("K4", 15)):
        assert red["g5_r1"][arm]["anchor"]["n"] == n
        p = red["g5_bp"][arm]["8"]
        assert p["first_wave"]["n"] == 2 * 3 and p["mid_run"]["n"] == 2 * 3  # two sequences of 3 tokens each
    th = common.load_thresholds()
    r1 = rules.g5_r1(red["g5_r1"], {"floor_kl": 0.0, "floor_dis": 0.0}, red["g5_r1_mutants"], th)
    assert set(r1) == {"g5_r1_K8", "g5_r1_K4"} and r1["g5_r1_K8"]["controls"]["G5R1/26"]["caught"] is True
    bp = rules.g5_bp(red["g5_bp"], red["g5_bp_mutants"], th)
    assert set(bp["allowed_B"]) == {"K8", "K4"} and bp["controls"]["G5BP/27"]["caught"] is not None
    assert set(bp["controls"]) == {"G5BP/27"}  # the probe is never a control (DESIGN §4.2)
