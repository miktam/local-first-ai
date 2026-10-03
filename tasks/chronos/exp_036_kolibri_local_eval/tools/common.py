"""Shared helpers for the exp_036 tools (BUILD_SPEC §1, §2, §5.9).

Paths come from the environment, never from hard-coded home directories:

    EXP036_MODELS   default ~/models/exp036
    EXP036_DATA     default $EXP036_MODELS/data
    EXP036_PRIVATE  default $EXP036_MODELS/private
    EXP036_WORK     default $EXP036_MODELS/work

An assets.json path that starts with "data/" resolves under $EXP036_DATA; any
other path resolves under $EXP036_MODELS. That is the one convention the tools
use, so moving the datasets only needs EXP036_DATA.

Everything here is stdlib only and makes no network call.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
EXP_DIR = TOOLS_DIR.parent

# BUILD_SPEC §2 Identity: every entry point refuses to write results otherwise.
REQUIRED_GIT_NAME = "Miktam"
REQUIRED_GIT_EMAIL = "hello@localfirstai.eu"


class IdentityError(RuntimeError):
    """git user.name / user.email are not the experiment's identity."""


# --------------------------------------------------------------------------
# Environment paths
# --------------------------------------------------------------------------

def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def models_dir() -> Path:
    return _env_path("EXP036_MODELS", Path.home() / "models" / "exp036")


def data_dir() -> Path:
    return _env_path("EXP036_DATA", models_dir() / "data")


def private_dir() -> Path:
    return _env_path("EXP036_PRIVATE", models_dir() / "private")


def work_dir() -> Path:
    return _env_path("EXP036_WORK", models_dir() / "work")


def asset_path(rel: str) -> Path:
    """Resolve an assets.json `path` (see the module docstring)."""
    rel = rel.strip("/")
    if rel == "data":
        return data_dir()
    if rel.startswith("data/"):
        return data_dir() / rel[len("data/"):]
    return models_dir() / rel


def exp_dir() -> Path:
    return EXP_DIR


def set_offline_env() -> None:
    """BUILD_SPEC §1: set before importing mlx_lm or transformers."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# --------------------------------------------------------------------------
# Hashing, time, JSON
# --------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_blob_sha1_file(path, chunk: int = 1 << 24) -> str:
    """The git blob id of a file ("blob <size>\\0" + content), which is the
    etag Hugging Face records for files that are not stored in LFS."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def utc_iso(t: _dt.datetime | None = None) -> str:
    """2026-10-03T17:01:52Z"""
    t = t or utc_now()
    return t.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp(t: _dt.datetime | None = None) -> str:
    """20261003T170152Z, for file names (the host/ records use the same form)."""
    t = t or utc_now()
    return t.astimezone(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dumps(obj) -> str:
    """BUILD_SPEC §2 Determinism: UTF-8 JSON with sorted keys."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_new_json(path: Path, obj) -> Path:
    """Write a new JSON file; never overwrite (BUILD_SPEC §2 Append-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(dumps(obj))
    return path


# --------------------------------------------------------------------------
# git (read-only; no network)
# --------------------------------------------------------------------------

def run(cmd, cwd=None, timeout: float = 30.0, input_text: str | None = None):
    """Run a command; return (returncode, stdout, stderr). Missing binary or
    timeout gives returncode 127 / 124 instead of raising."""
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, input=input_text,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"


def git(args, cwd=None, timeout: float = 60.0):
    return run(["git", "-c", "core.quotepath=off", *args], cwd=str(cwd or EXP_DIR), timeout=timeout)


def git_out(args, cwd=None) -> str | None:
    rc, out, _ = git(args, cwd=cwd)
    return out.strip() if rc == 0 else None


def repo_root(start=None) -> Path | None:
    out = git_out(["rev-parse", "--show-toplevel"], cwd=start or EXP_DIR)
    return Path(out) if out else None


def git_identity(cwd=None) -> dict:
    """user.name / user.email as git resolves them in the repo."""
    return {
        "name": git_out(["config", "--get", "user.name"], cwd=cwd),
        "email": git_out(["config", "--get", "user.email"], cwd=cwd),
    }


def identity_ok(ident: dict) -> bool:
    return ident.get("name") == REQUIRED_GIT_NAME and ident.get("email") == REQUIRED_GIT_EMAIL


def require_identity(cwd=None) -> dict:
    """Raise IdentityError unless git identity is Miktam <hello@localfirstai.eu>.

    runner.guard.require_identity() is the runner's entry-point form of this
    rule; the tools use this one so they do not depend on runner/."""
    ident = git_identity(cwd)
    if not identity_ok(ident):
        raise IdentityError(
            f"git identity is {ident.get('name')!r} <{ident.get('email')!r}>; "
            f"exp_036 writes results only as {REQUIRED_GIT_NAME} <{REQUIRED_GIT_EMAIL}>. "
            f"Run: git config user.name {REQUIRED_GIT_NAME} && "
            f"git config user.email {REQUIRED_GIT_EMAIL}"
        )
    return ident


def git_state(cwd=None) -> dict:
    """HEAD, upstream and dirty flag of the experiment directory."""
    cwd = cwd or EXP_DIR
    head = git_out(["rev-parse", "HEAD"], cwd=cwd)
    upstream_ref = git_out(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], cwd=cwd)
    upstream = git_out(["rev-parse", "@{u}"], cwd=cwd) if upstream_ref else None
    rc, out, _ = git(["status", "--porcelain", "--", "."], cwd=cwd)
    dirty_paths = [ln[3:] for ln in out.splitlines() if ln.strip()] if rc == 0 else []
    return {
        "head": head,
        "upstream_ref": upstream_ref,
        "upstream": upstream,
        "head_equals_upstream": bool(head and upstream and head == upstream),
        "dirty": bool(dirty_paths),
        "n_dirty_paths": len(dirty_paths),
    }


# --------------------------------------------------------------------------
# Package versions without pip (uv-built venvs have no pip)
# --------------------------------------------------------------------------

_FREEZE_SCRIPT = (
    "import importlib.metadata as m, json, sys;"
    "d={};"
    "[d.__setitem__(x.metadata['Name'].lower().replace('_','-'), x.version) "
    " for x in m.distributions() if x.metadata['Name']];"
    "print(json.dumps({'python': '.'.join(map(str, sys.version_info[:3])), 'packages': d}))"
)


def freeze(python: str | None = None) -> dict:
    """{'python': '3.12.13', 'packages': {name: version}} of an interpreter.

    Uses importlib.metadata in a subprocess, so it works in uv venvs without
    pip. With python=None it inspects the current interpreter in-process."""
    if python is None:
        import importlib.metadata as m
        import sys

        pkgs = {}
        for d in m.distributions():
            name = d.metadata["Name"]
            if name:
                pkgs[name.lower().replace("_", "-")] = d.version
        return {"python": ".".join(map(str, sys.version_info[:3])), "packages": pkgs}
    rc, out, err = run([python, "-c", _FREEZE_SCRIPT], timeout=120)
    if rc != 0:
        raise RuntimeError(f"cannot list packages of {python}: {err.strip()[:300]}")
    return json.loads(out)


def freeze_lines(packages: dict) -> list[str]:
    return sorted(f"{k}=={v}" for k, v in packages.items())


def parse_requirements(path) -> dict:
    """name==version pins of a requirements file (comments ignored)."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if not m:
            raise ValueError(f"{path}: not an exact pin: {line!r}")
        pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def load_assets(path=None) -> dict:
    return json.loads(Path(path or EXP_DIR / "assets.json").read_text(encoding="utf-8"))
