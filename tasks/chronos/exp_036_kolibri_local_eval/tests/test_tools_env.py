"""env/ (BUILD_SPEC §1, §3), the experiment .gitignore and README.md, and the
pre-push hook file."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import common

EXP = Path(__file__).resolve().parents[1]
ENV = EXP / "env"

BUILD_SPEC_PINS = {
    "mlx": "0.31.2", "mlx-metal": "0.31.2", "mlx-lm": "0.31.3", "numpy": "2.5.3", "safetensors": "0.8.0",
    "tokenizers": "0.23.2", "transformers": "5.18.0", "jinja2": "3.1.6", "pyyaml": "6.0.3",
}


def test_requirements_are_exact_pins_with_the_build_spec_block():
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    for k, v in BUILD_SPEC_PINS.items():
        assert pins[k] == v, k
    assert re.fullmatch(r"\d+\.\d+\.\d+", pins["pyarrow"]) and re.fullmatch(r"\d+\.\d+\.\d+", pins["pytest"])
    assert "torch" not in pins


def test_requirements_cover_the_mbp_venv():
    mbp = json.loads((EXP / "host" / "mbp_preflight_20261003T170152Z.json").read_text())["pip_freeze"]
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    for line in mbp:
        name, ver = line.split("==")
        assert pins.get(name.lower()) == ver, line


def test_versions_json_mirrors_the_requirements():
    v = json.loads((ENV / "versions.json").read_text())
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    assert v["pins"] == pins
    assert v["python"] == "3.12"
    assert set(v["core"]) == {"mlx", "mlx-metal", "mlx-lm", "numpy", "safetensors", "tokenizers", "transformers"}
    assert all(v["core"][k] == pins[k] for k in v["core"])


def test_setup_script_is_valid_and_never_sudo():
    assert os.access(ENV / "setup.sh", os.X_OK)
    subprocess.run(["bash", "-n", str(ENV / "setup.sh")], check=True)
    code = [ln for ln in (ENV / "setup.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not any(re.search(r"\bsudo\b", ln) for ln in code)
    assert "3.12" in (ENV / "setup.sh").read_text()


@pytest.mark.parametrize("shell", ["bash", "zsh"])
def test_env_file_defaults(shell, tmp_path):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not available")
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
    script = ('source env/exp036.env && printf "%s\\n" "$EXP036_MODELS" "$EXP036_DATA" "$EXP036_PRIVATE" '
              '"$EXP036_WORK" "$PY" "$HF_HUB_OFFLINE" "$TRANSFORMERS_OFFLINE" "$EXP" "$LFA"')
    out = subprocess.run([shell, "-c", script], cwd=EXP, env=env, capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    m = f"{tmp_path}/models/exp036"
    assert lines[:7] == [m, f"{m}/data", f"{m}/private", f"{m}/work", f"{m}/venv/bin/python", "1", "1"]
    assert Path(lines[7]).resolve() == EXP.resolve()
    assert Path(lines[8]).resolve() == EXP.parents[2].resolve()


def test_env_file_keeps_existing_values_and_reads_the_local_override(tmp_path):
    """On a temporary copy, so the real env/ is never written by a test."""
    exp = tmp_path / "exp"
    (exp / "env").mkdir(parents=True)
    (exp / "HYPOTHESIS.md").write_text("stub\n")
    shutil.copy(ENV / "exp036.env", exp / "env" / "exp036.env")
    (exp / "env" / "exp036.local.env").write_text('export EXP036_MODELS="$HOME/models/exp036-mini"\n')
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", ""), "EXP036_DATA": "/data/elsewhere"}
    out = subprocess.run(["bash", "-c", 'source env/exp036.env && echo "$EXP036_MODELS|$EXP036_DATA|$PY"'],
                         cwd=exp, env=env, capture_output=True, text=True, check=True)
    m = f"{tmp_path}/models/exp036-mini"
    assert out.stdout.strip() == f"{m}|/data/elsewhere|{m}/venv/bin/python"


def test_env_file_refuses_outside_the_experiment(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": os.environ.get("PATH", "")}
    r = subprocess.run(["bash", "-c", f'source "{ENV / "exp036.env"}"; echo "rc=$?"'], cwd=tmp_path, env=env,
                       capture_output=True, text=True)
    assert "rc=1" in r.stdout and "source this file from the exp_036 directory" in r.stderr


def test_gitignore_patterns():
    gi = (EXP / ".gitignore").read_text().splitlines()
    for pat in ("__pycache__/", ".pytest_cache/", "*.npy", "*.npz", "*.safetensors", "*.parquet", "*.pdf",
                "work/", "private/", "withheld/", "env/exp036.local.env"):
        assert pat in gi, pat


def test_gitignore_is_effective_in_git():
    if common.repo_root(EXP) is None:
        pytest.skip("not in a git work tree")
    probes = ["tools/__pycache__/x.pyc", ".pytest_cache/v", "results/x.npy", "private/a.txt",
              "results/work/x.json", "tasks/withheld/x.json", "env/exp036.local.env", "data.parquet"]
    for p in probes:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=EXP)
        assert r.returncode == 0, f"{p} is not ignored"
    r = subprocess.run(["git", "check-ignore", "-q", "tools/withheld_shingles.sha256"], cwd=EXP)
    assert r.returncode == 1, "the shingle file must be committable"


def test_readme_links_resolve():
    text = (EXP / "README.md").read_text()
    links = re.findall(r"\]\(\./([^)#]+)\)", text)
    assert {"HYPOTHESIS.md", "RUNBOOK.md", "BUILD_SPEC.md", "ASSETS.md"} <= set(links)
    for rel in links:
        assert (EXP / rel).exists(), rel


def test_pre_push_hook_file():
    hook = EXP / "tools" / "hooks" / "pre-push"
    assert os.access(hook, os.X_OK)
    subprocess.run(["bash", "-n", str(hook)], check=True)
    text = hook.read_text()
    assert "leak_check.py" in text and "--staged" in text and "@{u}..HEAD" in text
