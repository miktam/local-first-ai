"""env/ (exp_036 BUILD_SPEC §1, §3; exp_037 DESIGN §2.7, §2.11, decision F2), the experiment .gitignore and
README.md, and the pre-push hook file.

exp_037 keeps no copy of exp_036's environment file and no test reads it (DESIGN §2.5, §12 W10): the mbp sources
it in place from exp_036's directory. env/exp037.settings.sh is tested on a stub tree written by the test
(G1 item 12). The F2 pins (MLX 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0; every other pin as exp_036's) and E37's
own env/setup.sh, which builds $EXP037_VENV and never exp_036's venv, are checked here.
"""

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
SETTINGS = ENV / "exp037.settings.sh"

# DESIGN §12 W10: the README-link assertion is skipped until W12's hand-off note (README.md written);
# switched to True by the final review (W15-01).
W12_HANDOFF = True
W12_PENDING = "exp_037 README.md: switched on after W12's hand-off note (DESIGN §12 W10)"

# Decision F2 (2026-10-06T02:48:30Z): the three MLX pins move; every other pin is exp_036's.
F2_PINS = {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}
BUILD_SPEC_PINS = {
    **F2_PINS, "numpy": "2.5.3", "safetensors": "0.8.0",
    "tokenizers": "0.23.2", "transformers": "5.18.0", "jinja2": "3.1.6", "pyyaml": "6.0.3",
}


def test_requirements_are_exact_pins_with_the_build_spec_block():
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    for k, v in BUILD_SPEC_PINS.items():
        assert pins[k] == v, k
    assert re.fullmatch(r"\d+\.\d+\.\d+", pins["pyarrow"]) and re.fullmatch(r"\d+\.\d+\.\d+", pins["pytest"])
    assert "torch" not in pins


def test_requirements_cover_the_mbp_venv():
    """The mbp's freeze of 2026-10-03 (exp_036's venv) pin for pin, except the three F2 pins, which are
    exp_037's (decision F2: every other package keeps exp_036's pin)."""
    mbp = json.loads((EXP / "host" / "mbp_preflight_20261003T170152Z.json").read_text())["pip_freeze"]
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    seen = set()
    for line in mbp:
        name, ver = line.split("==")
        name = name.lower()
        seen.add(name)
        assert pins.get(name) == F2_PINS.get(name, ver), line
    assert set(F2_PINS) <= seen


def test_requirements_install_into_the_exp037_venv():
    text = (ENV / "requirements-mbp.txt").read_text()
    assert "$EXP037_VENV" in text.splitlines()[1]
    assert "$EXP036_MODELS/venv" not in text


def test_versions_json_mirrors_the_requirements():
    v = json.loads((ENV / "versions.json").read_text())
    pins = common.parse_requirements(ENV / "requirements-mbp.txt")
    assert v["pins"] == pins
    assert v["python"] == "3.12"
    assert set(v["core"]) == {"mlx", "mlx-metal", "mlx-lm", "numpy", "safetensors", "tokenizers", "transformers"}
    assert all(v["core"][k] == pins[k] for k in v["core"])
    assert {k: v["core"][k] for k in F2_PINS} == F2_PINS


def test_setup_script_is_valid_and_never_sudo():
    """E37's own env/setup.sh: executable, valid bash, Python 3.12, no sudo."""
    assert os.access(ENV / "setup.sh", os.X_OK)
    subprocess.run(["bash", "-n", str(ENV / "setup.sh")], check=True)
    code = [ln for ln in (ENV / "setup.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not any(re.search(r"\bsudo\b", ln) for ln in code)
    assert "3.12" in (ENV / "setup.sh").read_text()


def test_setup_script_builds_the_exp037_venv_never_exp036s():
    """Decision F2 (DESIGN §2.7): the venv defaults to $EXP037_VENV (exp037/venv beside $EXP036_MODELS), never
    $EXP036_MODELS/venv, and the script installs E37's requirements."""
    text = (ENV / "setup.sh").read_text()
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    m = re.search(r'^venv="(.+)"$', code, re.M)
    assert m and m.group(1) == '${EXP037_VENV:-$(dirname "$models")/exp037/venv}', m and m.group(1)
    assert "$EXP036_MODELS/venv" not in code and "$models/venv" not in code and "exp036/venv" not in code
    assert re.search(r'^req="\$here/requirements-mbp\.txt"$', code, re.M)


# ---------------------------------------------------------------- env/exp037.settings.sh (G1 item 12)

E36_NAME, E37_NAME = "exp_036_kolibri_local_eval", "exp_037_kolibri_forced_gate"


@pytest.fixture
def stub(tmp_path) -> dict:
    """A stub tree: chronos/{exp_036…,exp_037…} with HYPOTHESIS.md files and the settings file, a directory with
    the wrong name, and an exp_037 directory without an exp_036 sibling. Never exp_036's real files."""
    t = tmp_path.resolve()
    e36, e37 = t / "chronos" / E36_NAME, t / "chronos" / E37_NAME
    wrong, lone = t / "chronos" / "wrong", t / "lone" / E37_NAME
    for d in (e37, wrong, lone):
        (d / "env").mkdir(parents=True)
        (d / "HYPOTHESIS.md").write_text("stub\n")
        shutil.copy(SETTINGS, d / "env" / SETTINGS.name)
    e36.mkdir(parents=True)
    (e36 / "HYPOTHESIS.md").write_text("stub\n")
    return {"e36": e36, "e37": e37, "wrong": wrong, "lone": lone}


SHELLS = ["bash", "zsh"]


def _sh(shell: str, cwd: Path, script: str, env: dict) -> subprocess.CompletedProcess:
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not available")
    base = {"PATH": "/usr/bin:/bin"}
    return subprocess.run([shell, "-c", script], cwd=cwd, env={**base, **env}, capture_output=True, text=True)


SOURCE_TWICE = r'''
before="$(env | grep '^EXP036_' | grep -v '^EXP036_DIR=' | sort)"
source env/exp037.settings.sh; r1=$?
v1="$EXP036_DIR|$EXP|$EXP037_BUILDS|$EXP037_VENV|$PY"
source env/exp037.settings.sh; r2=$?
v2="$EXP036_DIR|$EXP|$EXP037_BUILDS|$EXP037_VENV|$PY"
after="$(env | grep '^EXP036_' | grep -v '^EXP036_DIR=' | sort)"
same=no; [ "$v1" = "$v2" ] && same=yes
keep=no; [ "$before" = "$after" ] && keep=yes
printf '%s\n' "$r1$r2" "$v1" "$same" "$keep" "${_e36-unset}"
'''


@pytest.mark.parametrize("shell", SHELLS)
def test_settings_sourced_twice_set_the_exp037_values(shell, stub):
    """EXP preset to E37 and PY preset to exp_036's venv (as exp_036's env file leaves them): the settings set
    EXP036_DIR, EXP, EXP037_BUILDS, EXP037_VENV and PY = $EXP037_VENV/bin/python, change no other EXP036_*
    value, and a second sourcing gives the same values."""
    env = {"HOME": "/nohome", "EXP": str(stub["e37"]), "EXP036_MODELS": "/m/models/exp036",
           "EXP036_DATA": "/m/data", "EXP036_WORK": "/m/work", "PY": "/m/models/exp036/venv/bin/python"}
    r = _sh(shell, stub["e37"], SOURCE_TWICE, env)
    assert r.returncode == 0, r.stderr
    rc, values, same, keep, leftover = r.stdout.splitlines()
    assert rc == "00"
    e36_dir, exp, builds, venv, py = values.split("|")
    assert Path(e36_dir) == stub["e36"] and Path(exp) == stub["e37"]
    assert (builds, venv, py) == ("/m/models/exp036/exp037-builds", "/m/models/exp037/venv",
                                  "/m/models/exp037/venv/bin/python")
    assert (same, keep, leftover) == ("yes", "yes", "unset")


@pytest.mark.parametrize("shell", SHELLS)
def test_settings_keep_preset_builds_and_venv(shell, stub):
    r = _sh(shell, stub["e37"], 'source env/exp037.settings.sh; echo "$?|$EXP037_BUILDS|$EXP037_VENV|$PY"',
            {"EXP036_MODELS": "/m/models/exp036", "EXP037_VENV": "/v", "EXP037_BUILDS": "/b"})
    assert r.stdout.strip() == "0|/b|/v|/v/bin/python"


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("case", ["no HYPOTHESIS.md", "wrong directory", "no exp_036 sibling", "no EXP036_MODELS"])
def test_settings_refuse(shell, case, stub):
    cwd, env = stub["e37"], {"EXP036_MODELS": "/m/models/exp036"}
    if case == "no HYPOTHESIS.md":
        (cwd / "HYPOTHESIS.md").unlink()
    elif case == "wrong directory":
        cwd = stub["wrong"]
    elif case == "no exp_036 sibling":
        cwd = stub["lone"]
    else:
        env = {}
    r = _sh(shell, cwd, 'source env/exp037.settings.sh; echo "$?|${EXP-unset}|${PY-unset}|${EXP037_VENV-unset}"', env)
    assert r.stdout.strip() == "1|unset|unset|unset", (r.stdout, r.stderr)
    assert "exp037.settings.sh:" in r.stderr


@pytest.mark.parametrize("shell", SHELLS)
def test_settings_executed_instead_of_sourced_exit_1(shell, stub):
    if shutil.which(shell) is None:
        pytest.skip(f"{shell} not available")
    r = subprocess.run([shell, "env/exp037.settings.sh"], cwd=stub["e37"], env={"PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True)
    assert r.returncode == 1


def test_settings_file_is_short_and_names_no_secret():
    text = SETTINGS.read_text()
    assert len(text.splitlines()) <= 25
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not any(re.search(r"\bsudo\b|/Users/|token|secret", ln, re.I) for ln in code)


# ---------------------------------------------------------------- .gitignore, README, hook

def test_gitignore_patterns():
    gi = (EXP / ".gitignore").read_text().splitlines()
    for pat in ("__pycache__/", ".pytest_cache/", "*.npy", "*.npz", "*.safetensors", "*.parquet", "*.pdf",
                "work/", "private/", "withheld/", "env/*.local.env"):
        assert pat in gi, pat


def test_gitignore_is_effective_in_git():
    if common.repo_root(EXP) is None:
        pytest.skip("not in a git work tree")
    probes = ["tools/__pycache__/x.pyc", ".pytest_cache/v", "results/x.npy", "private/a.txt",
              "results/work/x.json", "tasks/withheld/x.json", "env/exp036.local.env", "env/exp037.local.env",
              "data.parquet"]
    for p in probes:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=EXP)
        assert r.returncode == 0, f"{p} is not ignored"
    r = subprocess.run(["git", "check-ignore", "-q", "tools/withheld_shingles.sha256"], cwd=EXP)
    assert r.returncode == 1, "the shingle file must be committable"


@pytest.mark.skipif(not W12_HANDOFF, reason=W12_PENDING)
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
    assert 'exp="$top/tasks/chronos/exp_037_kolibri_forced_gate"' in text
    assert "exp_036_kolibri_local_eval" not in text
    code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "exp036/venv" not in code.replace("${EXP036_MODELS:-$HOME/models/exp036}", "")
