"""tools/redact.py (BUILD_SPEC §2 Privacy)."""

from __future__ import annotations

from pathlib import Path

from tools import redact


def test_redact_path_longest_prefix_wins(tmp_path, monkeypatch):
    models = tmp_path / "models"
    monkeypatch.setenv("EXP036_MODELS", str(models))
    monkeypatch.setenv("EXP036_DATA", str(models / "data"))
    monkeypatch.setenv("EXP036_PRIVATE", str(models / "private"))
    monkeypatch.setenv("EXP036_WORK", str(tmp_path / "scratch_work"))
    assert redact.redact_path(models / "data" / "gpqa" / "x.csv") == "$EXP036_DATA/gpqa/x.csv"
    assert redact.redact_path(models / "private" / "raw" / "a.jsonl") == "$EXP036_PRIVATE/raw/a.jsonl"
    assert redact.redact_path(models / "Kolibri-1-BF16") == "$EXP036_MODELS/Kolibri-1-BF16"
    assert redact.redact_path(tmp_path / "scratch_work" / "ref") == "$EXP036_WORK/ref"
    assert redact.redact_path(models) == "$EXP036_MODELS"
    assert redact.redact_path(str(models) + "-other/x") != "$EXP036_MODELS-other/x"


def test_home_and_experiment_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    assert redact.redact_path(Path.home() / "somewhere" / "f.txt") == "~/somewhere/f.txt"
    assert redact.redact_path(redact.common.EXP_DIR / "results" / "x.json") == "$EXP/results/x.json"
    assert redact.redact_path("/opt/homebrew/bin/python3") == "/opt/homebrew/bin/python3"


def test_redact_text(monkeypatch, tmp_path):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "m"))
    msg = f"cannot open {tmp_path / 'm' / 'venv' / 'bin' / 'python'} from {Path.home() / 'x'}"
    out = redact.redact_text(msg)
    assert "$EXP036_MODELS/venv/bin/python" in out and "~/x" in out
    assert str(Path.home()) not in out


def test_forbidden_keys_and_labels():
    obj = {"a": {"SerialNumber": 1}, "b": [{"hardware_uuid": 2}], "HostName": 3, "fine": {"chip": "x"}}
    assert sorted(redact.forbidden_keys(obj)) == ["HostName", "a.SerialNumber", "b[0].hardware_uuid"]
    assert redact.host_label("Apple M5 Max", 128 * 2**30) == redact.HOST_LABEL == "mbp (M5 Max, 128 GB)"
    assert redact.host_label("Apple M4 Pro", 64 * 2**30) == redact.MINI_LABEL
    assert redact.host_label("Apple M3", 16 * 2**30) == "other host (Apple M3, 16 GB)"
