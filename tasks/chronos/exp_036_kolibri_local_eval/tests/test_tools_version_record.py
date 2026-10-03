"""tools/version_record.py (BUILD_SPEC §5.9): the exp_035-shape record,
identity refusal, no machine identifiers."""

from __future__ import annotations

import json
import re

import pytest

from tools import common, version_record as vr
from tools.redact import HOST_LABEL, forbidden_keys


def test_record_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    rec = vr.record(with_params=False, with_assets=False)
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
