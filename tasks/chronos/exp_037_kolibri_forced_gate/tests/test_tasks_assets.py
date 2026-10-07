"""tasks/assets.py against fake directories (BUILD_SPEC §5.5): env paths, lookup, revision checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tasks import assets


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for v in ("EXP036_MODELS", "EXP036_DATA", "EXP036_PRIVATE", "EXP036_WORK"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    return tmp_path


def test_env_defaults_and_overrides(env, monkeypatch):
    m = env / "models"
    assert assets.models_dir() == m
    assert assets.data_dir() == m / "data"
    assert assets.private_dir() == m / "private" and assets.work_dir() == m / "work"
    monkeypatch.setenv("EXP036_DATA", str(env / "elsewhere"))
    assert assets.resolve("data/gpqa") == env / "elsewhere" / "gpqa"
    assert assets.resolve("Kolibri-1-BF16") == m / "Kolibri-1-BF16"


def test_lookup_by_repo_path_or_basename(env):
    a = assets.asset("Idavidrein/gpqa")
    assert a.kind == "dataset" and a.revision.startswith("83022cef") and a.path == env / "models" / "data" / "gpqa"
    assert assets.asset("data/gpqa") == a == assets.asset("gpqa")
    rgb = assets.asset("chen700564/RGB")
    assert rgb.kind == "code" and rgb.revision.startswith("65ec39e4") and assets.asset("RGB-src") == rgb
    assert "data/en.json" in rgb.sha256
    with pytest.raises(KeyError):
        assets.asset("no/such-asset")


def _hf_meta(root: Path, rel: str, commit: str, etag: str = "e" * 64):
    p = root / ".cache" / "huggingface" / "download" / f"{rel}.metadata"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"{commit}\n{etag}\n1791051244.2\n", encoding="utf-8")


def test_verify_revision_hf_metadata(env):
    a = assets.asset("allenai/IFBench_test")
    a.path.mkdir(parents=True)
    assert assets.verify_revision(a) == (False, None)
    _hf_meta(a.path, "data/train-00000-of-00001.parquet", a.revision)
    _hf_meta(a.path, "README.md", a.revision)
    assert assets.verify_revision(a) == (True, a.revision)
    _hf_meta(a.path, "data/extra.parquet", "f" * 40)
    ok, found = assets.verify_revision(a)
    assert not ok and a.revision in found and "f" * 40 in found


def test_verify_revision_git_detached_ref_and_packed(env):
    a = assets.asset("chen700564/RGB")
    git = a.path / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text(a.revision + "\n", encoding="utf-8")
    assert assets.verify_revision(a) == (True, a.revision)
    (git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    assert assets.verify_revision(a) == (False, None)
    (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{a.revision} refs/heads/master\n", encoding="utf-8")
    assert assets.verify_revision(a) == (True, a.revision)
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "master").write_text("0" * 40 + "\n", encoding="utf-8")
    assert assets.verify_revision(a) == (False, "0" * 40)


def test_verify_files(env):
    a = assets.asset("HuggingFaceFW/fineweb-2")
    rel = next(iter(a.sha256))
    assert assets.verify_files(a) == [(rel, False, None)]
    p = a.path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not the parquet")
    [(r, ok, found)] = assets.verify_files(a)
    assert r == rel and not ok and found == hashlib.sha256(b"not the parquet").hexdigest()
