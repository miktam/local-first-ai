"""tools/version_record.py (exp_036 BUILD_SPEC §5.9): the exp_035-shape record,
identity refusal, no machine identifiers.

exp_037 (decision F2; DESIGN §2.8, §3.1 P2): schema "exp037 version record v1",
the core pins of E37's env/versions.json compared, mx.device_info() recorded
(P2 binds to its architecture), the Kolibri builds read under builds_dir() and
the exp_037 venv as the default interpreter."""

from __future__ import annotations

import json
import re

import pytest

from tools import common, version_record as vr
from tools.redact import HOST_LABEL, forbidden_keys


@pytest.fixture(autouse=True)
def _no_host_builds_or_venv(monkeypatch):
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    monkeypatch.delenv("EXP037_VENV", raising=False)


def test_record_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    rec = vr.record(with_params=False, with_assets=False)
    assert rec["schema"] == vr.SCHEMA == "exp037 version record v1"
    assert "mx_device_info" in rec and "architecture" in rec["metal"]
    assert rec["versions"]["source"] == "env/versions.json"
    for key in ("os", "python", "packages", "pip_freeze_sha256", "git", "trees", "converted", "metal"):
        assert key in rec, key
    for pkg in ("mlx", "mlx-metal", "mlx-lm", "numpy", "tokenizers", "safetensors", "transformers"):
        assert pkg in rec["packages"]
    assert rec["host"] == HOST_LABEL
    assert set(rec["trees"]["scopes"]) == {s for s in __import__("tools.hash_tree", fromlist=["x"]).BY_NAME}
    for s in rec["trees"]["scopes"].values():
        assert s["status"] in ("match", "MISMATCH", "unfilled", "missing")
    assert rec["converted"]["K8"]["present"] is False
    assert forbidden_keys(rec) == []
    assert "/Users/" not in json.dumps(rec)


def test_converted_manifest_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path))
    d = tmp_path / vr.ARM_DIRS["K8"]
    d.mkdir()
    (d / vr.CONVERT_RECORD).write_text(json.dumps({"manifest_sha256": "m" * 64, "port_sha256": "p" * 64,
                                                   "quantization": {"bits": 8}}))
    c = vr._converted()
    assert c["K8"]["manifest_sha256"] == "m" * 64 and c["K8"]["bits"] == 8
    assert c["K4"]["present"] is False


def test_write_refuses_without_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "x", "email": "y"})
    with pytest.raises(common.IdentityError):
        vr.write(out_dir=tmp_path / "results")
    assert not (tmp_path / "results").exists()
    assert vr.main(["--out-dir", str(tmp_path / "results")]) == 1


def test_write_with_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "Miktam", "email": "hello@localfirstai.eu"})
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(vr, "record", lambda **kw: {"utc": "2026-10-04T10:00:00Z", "schema": vr.SCHEMA})
    p = vr.write(out_dir=tmp_path / "results")
    assert re.fullmatch(r"version_record_20261004T100000Z\.json", p.name)


def test_arm_dirs_match_the_runner():
    try:
        from runner.common import ARMS
    except ModuleNotFoundError:
        pytest.skip("runner/common.py is not present yet")
    assert {a: v["dir"] for a, v in ARMS.items()} == vr.ARM_DIRS


# ---------------------------------------------------------------- exp_037

def _fake(monkeypatch, packages: dict, arch="applegpu_g17s"):
    from tools import preflight

    pins = preflight.load_versions()["pins"]
    monkeypatch.setattr(common, "freeze", lambda python=None: {"python": "3.12.13", "packages": {**pins, **packages}})
    monkeypatch.setattr(vr, "_device_info", lambda: {"architecture": arch, "device_name": "Apple M5 Max",
                                                     "max_recommended_working_set_size": 115_448_725_504})


@pytest.mark.parametrize("packages,core_ok", [
    ({}, True),
    ({"mlx": "0.31.2", "mlx-metal": "0.31.2", "mlx-lm": "0.31.3"}, False),
    ({"mlx-lm": "0.31.3"}, False),
])
def test_record_binds_the_f2_pins_and_the_architecture(tmp_path, monkeypatch, packages, core_ok):
    """The record P2 reads (DESIGN §3.1 P2): packages, the comparison with E37's env/versions.json, and
    mx_device_info.architecture."""
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    _fake(monkeypatch, packages)
    rec = vr.record(with_params=False, with_assets=False)
    assert rec["versions"]["core_ok"] is core_ok
    assert rec["versions"]["core_expected"]["mlx"] == "0.32.3" and rec["versions"]["core_expected"]["mlx-lm"] == "0.32.0"
    assert rec["packages"]["mlx-metal"] == {**{"mlx-metal": "0.32.3"}, **packages}.get("mlx-metal")
    assert rec["mx_device_info"]["architecture"] == rec["metal"]["architecture"] == "applegpu_g17s"
    assert forbidden_keys(rec) == []


def test_kolibri_builds_are_read_under_exp037_builds(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    builds = tmp_path / "builds"
    d = builds / vr.ARM_DIRS["K8"]
    d.mkdir(parents=True)
    (d / vr.CONVERT_RECORD).write_text(json.dumps({"manifest_sha256": "c" * 64, "port_sha256": "p" * 64,
                                                   "quantization": {"bits": 8}, "spec_version": "exp037-port-1"}))
    assert vr._converted()["K8"]["present"] is False
    monkeypatch.setenv("EXP037_BUILDS", str(builds))
    c = vr._converted()
    assert c["K8"]["manifest_sha256"] == "c" * 64 and c["K8"]["spec_version"] == "exp037-port-1"
    assert c["K8"]["path"].startswith("$") or "builds" in c["K8"]["path"]
    assert vr._arm_dir("G8") == tmp_path / "models" / vr.ARM_DIRS["G8"]          # peers stay under $EXP036_MODELS


def test_the_default_interpreter_is_the_exp037_venv(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    seen = []

    def freeze(python=None):
        seen.append(python)
        return {"python": "3.12.13", "packages": {}}

    monkeypatch.setattr(common, "freeze", freeze)
    monkeypatch.setattr(vr, "_device_info", lambda: None)
    for p in (tmp_path / "models" / "venv" / "bin" / "python", tmp_path / "exp037" / "venv" / "bin" / "python"):
        p.parent.mkdir(parents=True)
        p.write_text("")
    rec = vr.record(with_params=False, with_assets=False)
    assert seen == [str(tmp_path / "exp037" / "venv" / "bin" / "python")]
    assert rec["versions"]["core_ok"] is False and rec["mx_device_info"] is None
