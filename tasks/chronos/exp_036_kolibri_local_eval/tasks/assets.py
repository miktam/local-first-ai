"""Asset paths and revision checks (BUILD_SPEC §5.5 `tasks/assets.py`; §1 environment variables).

Paths come only from the environment, never from a hard-coded home directory:
- EXP036_MODELS  (default ~/models/exp036)
- EXP036_DATA    (default $EXP036_MODELS/data)
- EXP036_PRIVATE (default $EXP036_MODELS/private)
- EXP036_WORK    (default $EXP036_MODELS/work)

assets.json gives every local path relative to $EXP036_MODELS. A path under `data/` resolves under
$EXP036_DATA instead, so a host that keeps its datasets elsewhere (the mini's build-time copies) needs only
EXP036_DATA. `verify_revision` reads what is on disk (Hugging Face download metadata, or a git clone's
HEAD); it never contacts the network.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
ASSETS_JSON = EXP_DIR / "assets.json"


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


def models_dir() -> Path:
    return _env_path("EXP036_MODELS", Path("~/models/exp036").expanduser())


def data_dir() -> Path:
    return _env_path("EXP036_DATA", models_dir() / "data")


def private_dir() -> Path:
    return _env_path("EXP036_PRIVATE", models_dir() / "private")


def work_dir() -> Path:
    return _env_path("EXP036_WORK", models_dir() / "work")


def resolve(rel: str) -> Path:
    """An assets.json path ("data/gpqa", "Kolibri-1-BF16") on this host."""
    rel = rel.strip("/")
    if rel == "data":
        return data_dir()
    if rel.startswith("data/"):
        return data_dir() / rel[len("data/") :]
    return models_dir() / rel


@dataclass(frozen=True)
class Asset:
    name: str  # repo id, e.g. "Idavidrein/gpqa" or "chen700564/RGB"
    path: Path
    revision: str
    kind: str  # "model" | "dataset" | "code"
    rel: str
    files: tuple[str, ...] = ()
    sha256: dict = field(default_factory=dict)


def _repo_id(repo: str) -> str:
    for prefix in ("https://github.com/", "http://github.com/"):
        if repo.startswith(prefix):
            return repo[len(prefix) :].removesuffix(".git")
    return repo


def load_assets(assets_json: Path | None = None) -> list[Asset]:
    doc = json.loads(Path(assets_json or ASSETS_JSON).read_text(encoding="utf-8"))
    out = []
    for kind, key, rev_key in (("model", "models", "revision"), ("dataset", "datasets", "revision"), ("code", "code", "commit")):
        for entry in doc.get(key, []):
            out.append(
                Asset(
                    name=_repo_id(entry["repo"]),
                    path=resolve(entry["path"]),
                    revision=entry[rev_key],
                    kind=kind,
                    rel=entry["path"],
                    files=tuple(entry.get("files", ())),
                    sha256=dict(entry.get("sha256", {})),
                )
            )
    return out


def asset(name: str, assets_json: Path | None = None) -> Asset:
    """Look an asset up by repo id ("Idavidrein/gpqa"), its relative path ("data/gpqa") or the last path
    component ("gpqa", "RGB-src"). Exactly one match is required."""
    matches = [
        a for a in load_assets(assets_json)
        if name in (a.name, a.rel, a.rel.rsplit("/", 1)[-1], a.name.rsplit("/", 1)[-1])
    ]
    if len(matches) != 1:
        raise KeyError(f"asset {name!r}: {len(matches)} matches in assets.json")
    return matches[0]


def _git_head(repo_dir: Path) -> str | None:
    git = repo_dir / ".git"
    head_file = git / "HEAD"
    if not head_file.is_file():
        return None
    head = head_file.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head  # detached HEAD: the commit itself
    ref = head[4:].strip()
    loose = git / ref
    if loose.is_file():
        return loose.read_text(encoding="utf-8").strip()
    packed = git / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                sha, _, name = line.partition(" ")
                if name.strip() == ref:
                    return sha
    return None


def _hf_revisions(snapshot_dir: Path) -> set[str]:
    meta_root = snapshot_dir / ".cache" / "huggingface" / "download"
    revs = set()
    if meta_root.is_dir():
        for meta in sorted(meta_root.rglob("*.metadata")):
            lines = meta.read_text(encoding="utf-8").splitlines()
            if lines and lines[0].strip():
                revs.add(lines[0].strip())
    return revs


def _same_commit(found: str, pinned: str) -> bool:
    if found == pinned:
        return True
    short, full = sorted((found, pinned), key=len)
    return len(short) >= 7 and full.startswith(short)


def verify_revision(a: Asset) -> tuple[bool, str | None]:
    """(ok, found). Git clones: HEAD (detached or via refs/packed-refs). HF snapshots: the first line of every
    `.cache/huggingface/download/**/*.metadata`; ok only if all of them name the pinned revision."""
    if not a.path.exists():
        return False, None
    head = _git_head(a.path)
    if head is not None:
        return _same_commit(head, a.revision), head
    revs = _hf_revisions(a.path)
    if not revs:
        return False, None
    found = ",".join(sorted(revs))
    return (len(revs) == 1 and _same_commit(next(iter(revs)), a.revision)), found


def verify_files(a: Asset) -> list[tuple[str, bool, str | None]]:
    """For every file with a recorded sha256 in assets.json: (relpath, ok, found sha256 or None)."""
    out = []
    for rel, want in sorted(a.sha256.items()):
        p = a.path / rel
        if not p.is_file():
            out.append((rel, False, None))
            continue
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        out.append((rel, got == want, got))
    return out
