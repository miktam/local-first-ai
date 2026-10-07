"""Small helpers shared by the exp_036 test modules."""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest

# Top-level packages written by other parts of the kit. A test that needs one
# of them skips while the module does not exist yet; any other import error
# (a broken module, a missing third-party dependency) fails the test.
SIBLING_PACKAGES = ("port", "reference")


def import_sibling(modname: str):
    """Import port.* / reference.*, skipping the test if the file is missing."""
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing.split(".")[0] in SIBLING_PACKAGES:
            pytest.skip(f"{modname} is not importable yet ({missing} missing)")
        raise


def as_logits(out) -> np.ndarray:
    """Normalise a forward() result to a float32 [T, V] numpy array.

    Accepts a bare array or a (logits, ...) tuple, with or without a leading
    batch dimension of size 1.
    """
    if isinstance(out, (tuple, list)):
        out = out[0]
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 3:
        assert arr.shape[0] == 1, f"expected batch 1, got shape {arr.shape}"
        arr = arr[0]
    assert arr.ndim == 2, f"expected [T, V] logits, got shape {arr.shape}"
    return arr


def kolibri_tok_env() -> str | None:
    """The Kolibri tokenizer directory from the environment: $EXP036_TOK (BUILD_SPEC §1, §6), else the
    older name $EXP036_TOKENIZER_DIR, else $EXP036_MODELS/Kolibri-1-BF16 if it holds tokenizer.json. None
    otherwise."""
    for var in ("EXP036_TOK", "EXP036_TOKENIZER_DIR"):
        v = os.environ.get(var)
        if v:
            return os.path.expanduser(v)
    m = os.environ.get("EXP036_MODELS")  # the run host's BF16 snapshot (conftest's tokenizer_dir fallback)
    if m and os.path.isfile(os.path.join(os.path.expanduser(m), "Kolibri-1-BF16", "tokenizer.json")):
        return os.path.join(os.path.expanduser(m), "Kolibri-1-BF16")
    return None


# ------------------------------------------------------------------ test assets
# Two layouts hold the real (weight-free) files the tests read:
#   build host (the mini): ~/models/exp036-mini/{kolibri/Kolibri-1-BF16, mlx-community/<folder>, upstream/<repo@rev8>,
#                          data/, evalfw/, ifbench-venv, nltk_data}  ($EXP036_MINI_ASSETS or $EXP036_MINI overrides)
#   run host (the mbp):    $EXP036_MODELS/<folder> (the peer model folders) and $EXP036_DATA (datasets)
# A test that needs a file only the build host has (upstream peer files, the eval-framework checkout) skips with a
# reason starting BUILD_HOST_ONLY; conftest counts every other skip as a failure under EXP036_REQUIRE_ALL=run-host
# (RUNBOOK step 3 on the mbp) and every skip under EXP036_REQUIRE_ALL=1 (the pre-push run on the mini).

BUILD_HOST_ONLY = "build-host only:"


def mini_asset_roots():
    """Existing directories in the build host's layout, most specific first."""
    from pathlib import Path

    cands = []
    for var in ("EXP036_MINI_ASSETS", "EXP036_MINI"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]).expanduser())
    for var in ("EXP036_TOK", "EXP036_TOKENIZER_DIR", "EXP036_DATA"):
        if os.environ.get(var):
            p = Path(os.environ[var]).expanduser()
            cands += [p.parent, p.parent.parent]
    cands.append(Path("~/models/exp036-mini").expanduser())
    out = []
    for c in cands:
        if c.is_dir() and c not in out and any((c / d).is_dir() for d in ("mlx-community", "upstream", "data", "evalfw")):
            out.append(c)
    return out


def peer_folder(folder: str):
    """An mlx-community peer folder with its tokenizer/template/config files: <build root>/mlx-community/<folder>,
    $EXP036_PEER_TOK_ROOT/<folder>, else the run host's $EXP036_MODELS/<folder>. None if absent."""
    from pathlib import Path

    cands = [r / "mlx-community" / folder for r in mini_asset_roots()]
    for var in ("EXP036_PEER_TOK_ROOT", "EXP036_MODELS"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]).expanduser() / folder)
    for p in cands:
        if (p / "config.json").is_file() and (p / "tokenizer.json").is_file():
            return p
    return None


def peer_root(folders) -> "object | None":
    """A directory holding every folder in `folders` (the mini's mlx-community/ or the run host's $EXP036_MODELS)."""
    from pathlib import Path

    cands = [r / "mlx-community" for r in mini_asset_roots()]
    for var in ("EXP036_PEER_TOK_ROOT", "EXP036_MODELS"):
        if os.environ.get(var):
            cands.append(Path(os.environ[var]).expanduser())
    for c in cands:
        if all((c / f / "config.json").is_file() for f in folders):
            return c
    return None


def build_host_dir(*parts: str):
    """A directory that only the build host has (upstream peer files, the eval-framework checkout), or a
    build-host-only skip."""
    for r in mini_asset_roots():
        p = r.joinpath(*parts)
        if p.exists():
            return p
    pytest.skip(f"{BUILD_HOST_ONLY} {'/'.join(parts)} is a build-time asset of the mini (set EXP036_MINI_ASSETS)")
