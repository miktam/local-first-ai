# SPDX-License-Identifier: MIT
"""exp_037's gate/common.py helpers (DESIGN §2.8-§2.10, §6.5; W0c).

* seed_from(*parts, prefix="exp037"): exp_037 draws under "exp037"; with
  prefix="exp036" it is exp_036's seed_from exactly (G1 item 13);
* the seed sites: the three registered fixture seeds (the tiny gate texts, the
  G0 digit lines, the tiny behaviour prompts) pass prefix="exp036" with the
  comment that says so; the G3b bootstrap, the router-margin bootstrap and the
  behaviour cells draw under exp037; nothing else in gate/ uses exp036;
* the tiny gate texts and the G0 digit lines are exp_036's, id for id, checked
  against exp_036's own code (run read-only in a subprocess) and pinned shas;
* work37_dir() = $EXP036_WORK/exp037 and builds_dir() = $EXP037_BUILDS, else
  models_dir(), with their fallbacks;
* load_port on a tiny conversion under mlx-lm 0.32.0 (decision F2): it passes
  trust_remote_code=True explicitly, and the flag is inert there, because
  model_config={"model_file": None} builds the model from exp_037's port module
  and never runs the directory's kolibri1.py;
* port modules are named exp037_gate_port_<n>.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint as tc
import tiny_real_layout as trl

EXP = Path(__file__).resolve().parents[1]
GATE = EXP / "gate"
E36 = EXP.parent / "exp_036_kolibri_local_eval"  # read-only (DESIGN: E36 found by layout, never by env)

FIXTURE_COMMENT = "registered fixture seed, not an exp_037 draw"
# (file, first argument) of every seed_from call that keeps exp_036's registered seed.
FIXTURE_SITES = {
    ("gate/textset.py", "tiny-gate-texts"),
    ("gate/tokenizer_lines.py", "tokenizer_lines"),
    ("gate/run_gate.py", "tiny-behaviour"),
}
# The unqualified exp_037 draws DESIGN §2.10 names.
EXP037_SITES = {
    ("gate/checks/g3_oracle.py", "G3"),       # the G3b bootstrap
    ("gate/checks/g2_layers.py", "G2"),       # the router-margin bootstrap
    ("gate/checks/g5_generation.py", "gate"),  # the behaviour cells
}

# exp_036's values (computed with exp_036's gate/common.py and gate/textset.py at 222c845).
E36_SEEDS = {
    "tiny-behaviour": 4835292403176067209,
    "tokenizer_lines": 14600968949785250915,
    "tiny-gate-texts|36": 12780284458726283677,
}
# TextSet.key of make_tiny (the sha256 of the ids, T9 and G5 ids; the reference-dump
# cache key). The text length is max(160, 2 * window + 30), so the windows 65 and 17
# give the same texts.
E36_TINY_KEYS = {
    "vendor": "f109d8a7fb562eafb29490e53cffbd3ff00f9a6f163a63eb19f890371c390da2",
    "pattern5": "f109d8a7fb562eafb29490e53cffbd3ff00f9a6f163a63eb19f890371c390da2",
    "w513": "c1909ea0fedbc587cf0a85d3fa536805a99a87ee43d1ac99f46dd118f4d64a72",
    "real_layout": "c1909ea0fedbc587cf0a85d3fa536805a99a87ee43d1ac99f46dd118f4d64a72",
}
# tokenizer_lines.group_digest(digit_lines()): the G0 "digits" group.
E36_DIGITS_DIGEST = "cbd30db570e2b90af24a77d75ebce5862d62ca99eb46107396eba97e095ef8f1"


def _old_rule(prefix: str, *parts) -> int:
    return int.from_bytes(hashlib.sha256((prefix + "|" + "|".join(map(str, parts))).encode()).digest()[:8], "big")


def _tiny_shapes() -> dict:
    """{name: (sliding_window, num_hidden_layers, vocab_size)} of the tiny checkpoints
    the gate's tiny runs use (run_gate.py: make_tiny(window, layers, vocab))."""
    out = {}
    for preset in ("vendor", "pattern5", "w513"):
        cfg = tc.preset_config(preset)
        out[preset] = (cfg.get("sliding_window") or 0, cfg["num_hidden_layers"], cfg["vocab_size"])
    cfg = tc.preset_config(trl.PRESET)
    cfg.update(trl.real_layout_overrides())
    out["real_layout"] = (cfg.get("sliding_window") or 0, cfg["num_hidden_layers"], cfg["vocab_size"])
    return out


def _texts(ts) -> dict:
    return {"ids": ts.ids, "t9": ts.t9, "g5": ts.g5, "key": ts.key}


# ---------------------------------------------------------------------------
# seed_from and the seed sites (G1 item 13)
# ---------------------------------------------------------------------------


def test_seed_from_default_is_exp037_and_exp036_is_the_registered_rule():
    from gate import common

    prefix = inspect.signature(common.seed_from).parameters["prefix"]
    assert prefix.kind is inspect.Parameter.KEYWORD_ONLY and prefix.default == "exp037"
    for parts in (("G3", "T1"), ("G2", "router_margin"), ("gate", "behaviour", "K8", "high"), ()):
        assert common.seed_from(*parts) == _old_rule("exp037", *parts)
        assert common.seed_from(*parts, prefix="exp036") == _old_rule("exp036", *parts)
        assert common.seed_from(*parts) != common.seed_from(*parts, prefix="exp036")
        assert 0 <= common.seed_from(*parts) < 2**64
    assert common.seed_from("tiny-behaviour", prefix="exp036") == E36_SEEDS["tiny-behaviour"]
    assert common.seed_from("tokenizer_lines", prefix="exp036") == E36_SEEDS["tokenizer_lines"]
    assert common.seed_from("tiny-gate-texts", 36, prefix="exp036") == E36_SEEDS["tiny-gate-texts|36"]


def _seed_calls() -> list[dict]:
    """Every seed_from(...) call in gate/ (common.seed_from or a bare seed_from)."""
    calls = []
    for path in sorted(GATE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(EXP).as_posix()
        src = path.read_text(encoding="utf-8")
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src, filename=rel)):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name != "seed_from":
                continue
            first = node.args[0] if node.args else None
            kw = {k.arg: k.value for k in node.keywords}
            calls.append({
                "file": rel,
                "line": node.lineno,
                "label": first.value if isinstance(first, ast.Constant) else None,
                "unpacked": any(isinstance(a, ast.Starred) for a in node.args) or None in kw,
                "has_prefix": "prefix" in kw,
                "prefix": kw["prefix"].value if isinstance(kw.get("prefix"), ast.Constant) else kw.get("prefix"),
                "text": "\n".join(lines[node.lineno - 1:node.end_lineno]),
            })
    return calls


def test_seed_sites_exp036_only_at_the_registered_fixtures():
    calls = _seed_calls()
    assert calls, "no seed_from call found in gate/"
    seen = {(c["file"], c["label"]): c for c in calls}
    for c in calls:
        where = f"{c['file']}:{c['line']}"
        assert not c["unpacked"], f"{where}: seed_from with *args/**kwargs cannot be checked"
        site = (c["file"], c["label"])
        if site in FIXTURE_SITES:
            assert c["prefix"] == "exp036", f"{where}: a registered fixture seed must pass prefix=\"exp036\""
            assert FIXTURE_COMMENT in c["text"], f"{where}: the fixture seed lacks the comment {FIXTURE_COMMENT!r}"
        else:
            # exp_037 draws: the default prefix, or "exp037" spelled out; never exp036.
            assert not c["has_prefix"] or c["prefix"] == "exp037", \
                f"{where}: seed_from(..., prefix={c['prefix']!r}) outside the registered fixture sites"
    assert sum((c["file"], c["label"]) in FIXTURE_SITES for c in calls) == len(FIXTURE_SITES), \
        "each registered fixture seed is drawn at exactly one site"
    for site in FIXTURE_SITES | EXP037_SITES:
        assert site in seen, f"seed site {site} not found"
    for site in EXP037_SITES:
        assert not seen[site]["has_prefix"], f"{site}: an exp_037 draw uses the default prefix"


# ---------------------------------------------------------------------------
# The tiny fixtures stay exp_036's, id for id
# ---------------------------------------------------------------------------

_E36_SCRIPT = r"""
import json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from gate import common, textset, tokenizer_lines
shapes = json.loads(sys.argv[2])
texts = {}
for name, (w, n, v) in shapes.items():
    ts = textset.make_tiny(w, n, v)
    texts[name] = {"ids": ts.ids, "t9": ts.t9, "g5": ts.g5, "key": ts.key}
print(json.dumps({
    "texts": texts,
    "digits": tokenizer_lines.digit_lines(),
    "seeds": {"tiny-behaviour": common.seed_from("tiny-behaviour"),
              "tokenizer_lines": common.seed_from("tokenizer_lines"),
              "tiny-gate-texts|36": common.seed_from("tiny-gate-texts", 36)},
}))
"""


def _tree_state(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*")}


@pytest.fixture(scope="module")
def e36_fixtures(tmp_path_factory) -> dict:
    """exp_036's tiny gate texts, digit lines and fixture seeds, computed by exp_036's
    own gate code in a subprocess (-B: no bytecode is written into exp_036)."""
    assert (E36 / "gate" / "textset.py").is_file(), f"exp_036's kit is not at {E36}"
    before = {d: _tree_state(E36 / d) for d in ("gate", "tools")}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-B", "-c", _E36_SCRIPT, str(E36), json.dumps(_tiny_shapes())],
                       cwd=tmp_path_factory.mktemp("e36_fixtures"), env=env,
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-4000:]
    after = {d: _tree_state(E36 / d) for d in ("gate", "tools")}
    assert after == before, "running exp_036's gate code changed files under exp_036"
    return json.loads(p.stdout)


def test_tiny_gate_texts_ids_sha_equals_exp036(e36_fixtures):
    from gate import common, textset

    for name, (w, n, v) in _tiny_shapes().items():
        ts = textset.make_tiny(w, n, v)
        mine, theirs = _texts(ts), e36_fixtures["texts"][name]
        assert mine["key"] == theirs["key"] == E36_TINY_KEYS[name], name
        for tid in textset.ALL8:
            assert common.ids_sha256(mine["ids"][tid]) == common.ids_sha256(theirs["ids"][tid]), (name, tid)
        assert common.ids_sha256(mine["t9"]) == common.ids_sha256(theirs["t9"]), (name, "T9")
        assert mine["g5"] == theirs["g5"], (name, "G5")
        assert json.loads(json.dumps(mine)) == theirs, name


def test_g0_digit_lines_and_fixture_seeds_equal_exp036(e36_fixtures):
    from gate import common, tokenizer_lines

    lines = tokenizer_lines.digit_lines()
    assert lines == e36_fixtures["digits"]
    assert tokenizer_lines.group_digest(lines) == E36_DIGITS_DIGEST
    assert e36_fixtures["seeds"] == E36_SEEDS
    # run_gate.py's tiny behaviour prompts draw from this seed (its call site is
    # checked by test_seed_sites_exp036_only_at_the_registered_fixtures).
    assert common.seed_from("tiny-behaviour", prefix="exp036") == e36_fixtures["seeds"]["tiny-behaviour"]


# ---------------------------------------------------------------------------
# work37_dir() and builds_dir()
# ---------------------------------------------------------------------------

_PATH_VARS = ("EXP036_MODELS", "EXP036_DATA", "EXP036_WORK", "EXP036_PRIVATE", "EXP036_TOK",
              "EXP036_TOKENIZER_DIR", "EXP037_BUILDS")


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for var in _PATH_VARS:
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    return home


def test_work37_dir_resolution_and_fallbacks(clean_env, monkeypatch, tmp_path):
    from gate import common

    home = clean_env
    # Nothing set: ~/models/exp036/work/exp037.
    assert common.work37_dir() == home / "models" / "exp036" / "work" / "exp037"
    # $EXP036_MODELS only: $EXP036_MODELS/work/exp037.
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    assert common.work37_dir() == tmp_path / "m" / "work" / "exp037"
    # $EXP036_WORK wins: $EXP036_WORK/exp037.
    monkeypatch.setenv("EXP036_WORK", str(tmp_path / "w"))
    assert common.work37_dir() == tmp_path / "w" / "exp037"
    assert common.work37_dir().parent == common.work_dir()
    # An empty value counts as unset; ~ is expanded.
    monkeypatch.setenv("EXP036_WORK", "")
    assert common.work37_dir() == tmp_path / "m" / "work" / "exp037"
    monkeypatch.setenv("EXP036_WORK", "~/w2")
    assert common.work37_dir() == home / "w2" / "exp037"
    # $EXP037_BUILDS does not move it; nothing is created.
    monkeypatch.setenv("EXP037_BUILDS", str(tmp_path / "b"))
    assert common.work37_dir() == home / "w2" / "exp037"
    assert not common.work37_dir().exists() and not (tmp_path / "w").exists()


def test_builds_dir_resolution_and_fallbacks(clean_env, monkeypatch, tmp_path):
    from gate import common

    home = clean_env
    # Nothing set: models_dir(), i.e. ~/models/exp036 (exp_036's arm directories).
    assert common.builds_dir() == common.models_dir() == home / "models" / "exp036"
    # $EXP036_MODELS only: the arms stay where exp_036 had them (the tiny tests, the dry run).
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    assert common.builds_dir() == tmp_path / "m"
    # $EXP037_BUILDS wins for the arms; the BF16 source, data and work stay under $EXP036_*.
    monkeypatch.setenv("EXP037_BUILDS", str(tmp_path / "m" / "exp037-builds"))
    assert common.builds_dir() == tmp_path / "m" / "exp037-builds"
    assert common.builds_dir() / common.ARM_DIRS["K8"] == tmp_path / "m" / "exp037-builds" / "Kolibri-1-MLX-8bit-g64"
    assert common.builds_dir() / common.ARM_DIRS["K4"] == tmp_path / "m" / "exp037-builds" / "Kolibri-1-MLX-4bit-g64"
    assert common.models_dir() == tmp_path / "m"
    assert common.data_dir() == tmp_path / "m" / "data"
    assert common.work_dir() == tmp_path / "m" / "work"
    assert common.tokenizer_dir() == tmp_path / "m" / common.BF16_DIR
    # An empty value counts as unset; ~ is expanded; nothing is created.
    monkeypatch.setenv("EXP037_BUILDS", "")
    assert common.builds_dir() == tmp_path / "m"
    monkeypatch.setenv("EXP037_BUILDS", "~/b")
    assert common.builds_dir() == home / "b"
    assert not (home / "b").exists()


# ---------------------------------------------------------------------------
# load_port under mlx-lm 0.32.0 (decision F2) and the module names
# ---------------------------------------------------------------------------


def test_port_modules_are_named_exp037():
    from gate import common

    assert common.PORT_MODULE_PREFIX == "exp037_gate_port_"
    a, b = common.exec_port_module(), common.exec_port_module()
    assert a.__name__.startswith("exp037_gate_port_") and b.__name__.startswith("exp037_gate_port_")
    assert a.__name__ != b.__name__ and a is not b
    assert "exp036_gate_port_" not in (GATE / "common.py").read_text(encoding="utf-8")


def _logits(model, ids) -> np.ndarray:
    import mlx.core as mx

    out = model(mx.array([ids]))
    return np.array(out.astype(mx.float32))


def test_load_port_on_a_tiny_conversion_passes_trust_remote_code(tiny_real_cal, monkeypatch, tmp_path):
    import mlx_lm
    import mlx_lm.utils as mu

    from gate import common
    from port import convert

    assert "trust_remote_code" in inspect.signature(mu.load_model).parameters, \
        f"mlx-lm {mlx_lm.__version__}: load_model has no trust_remote_code (exp_037 pins 0.32.0)"
    k8 = tiny_real_cal.k8
    assert json.loads((k8 / "config.json").read_text())["model_file"] == "kolibri1.py"
    # The arm directory as the gate resolves it: under builds_dir().
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("EXP037_BUILDS", str(tiny_real_cal.root))
    arm_dir = common.builds_dir() / common.ARM_DIRS["K8"]
    assert arm_dir == k8

    real = mu.load_model
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(mu, "load_model", spy)
    model, config, module = common.load_port(arm_dir)
    assert len(calls) == 1
    assert calls[0]["trust_remote_code"] is True
    assert calls[0]["model_config"] == {"model_file": None}
    assert calls[0]["strict"] is True
    assert config.get("model_file") is None
    assert module.__name__.startswith("exp037_gate_port_") and type(model) is module.Model
    assert config["quantization"]["bits"] == 8

    # The same model as mlx-lm 0.32.0's own model_file route (the directory's kolibri1.py,
    # trusted only after check_port_file): bitwise equal logits.
    own, _ = real(k8, **convert.model_file_trust(k8))
    ids = tc.random_ids(48, seed=37)
    assert np.array_equal(_logits(model, ids), _logits(own, ids))


def test_load_port_never_runs_the_directory_copy(tiny_real_cal, tmp_path):
    import mlx_lm.utils as mu

    from gate import common

    d = tmp_path / common.ARM_DIRS["K8"]
    shutil.copytree(tiny_real_cal.k8, d)  # the session fixture is shared: change a copy
    (d / "kolibri1.py").write_text('raise RuntimeError("the directory copy of kolibri1.py was executed")\n')
    # mlx-lm 0.32.0 refuses a model_file without the flag (#1385, CVE-2026-5843) ...
    with pytest.raises(ValueError, match="trust_remote_code"):
        mu.load_model(d)
    # ... and executes it with the flag:
    with pytest.raises(RuntimeError, match="was executed"):
        mu.load_model(d, trust_remote_code=True)
    # load_port passes the flag, and it is inert: model_config={"model_file": None}
    # builds the model from exp_037's port module.
    model, config, module = common.load_port(d)
    assert config.get("model_file") is None and type(model) is module.Model
    ref, _, _ = common.load_port(tiny_real_cal.k8)
    ids = tc.random_ids(32, seed=41)
    assert np.array_equal(_logits(model, ids), _logits(ref, ids))
