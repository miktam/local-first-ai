"""Shared helpers for the runner: directories, arms, hashing, UTC stamps,
deterministic JSON, and the environment record that goes into every header.

Paths come from the environment (BUILD_SPEC §1, "Environment variables") or
from this file's location; nothing here hard-codes a home directory. Nothing
here touches the network.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Every entry point sets these before mlx_lm / transformers are imported (§1).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RUNNER_DIR = Path(__file__).resolve().parent
EXP_DIR = RUNNER_DIR.parent
HOST_LABEL_FALLBACK = "mbp (M5 Max, 128 GB)"

if str(EXP_DIR) not in sys.path:  # so `import tools.redact`, `gate.run_gate` … resolve
    sys.path.insert(0, str(EXP_DIR))


# ---------------------------------------------------------------- directories


def _env_dir(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def models_dir() -> Path:
    return _env_dir("EXP036_MODELS", Path("~/models/exp036").expanduser())


def builds_dir() -> Path:
    """exp_037's Kolibri builds (DESIGN §2.8): $EXP037_BUILDS (the APFS clones that tools/refresh_builds.py
    refreshes) when set, else models_dir(), so the tiny tests and the dry run work unchanged. Peers stay under
    models_dir()."""
    return _env_dir("EXP037_BUILDS", models_dir())


def data_dir() -> Path:
    return _env_dir("EXP036_DATA", models_dir() / "data")


# exp_037's private records (raw, pilot and aborted) live in their own subdirectory of $EXP036_PRIVATE (DESIGN
# §2.9), so the runner's resume logic never sees an exp_036 file. The reused withheld manifests are read from
# $EXP036_PRIVATE/manifests/ by tasks/assets.py and tools/common.py, which keep the base directory.
PRIVATE_SUBDIR = "exp037"


def private_base_dir() -> Path:
    """$EXP036_PRIVATE itself (exp_036's private root)."""
    return _env_dir("EXP036_PRIVATE", models_dir() / "private")


def private_dir() -> Path:
    """$EXP036_PRIVATE/exp037/: every private file the exp_037 runner writes goes under it."""
    return private_base_dir() / PRIVATE_SUBDIR


def work_dir() -> Path:
    return _env_dir("EXP036_WORK", models_dir() / "work")


def results_dir(exp_dir: Path | None = None) -> Path:
    return Path(exp_dir or EXP_DIR) / "results"


def is_under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------- arms

# HYPOTHESIS "Arms". The directory names are the assets.json paths (peers) and
# the convert.py output names (Kolibri, BUILD_SPEC §5.1).
ARMS: dict[str, dict[str, Any]] = {
    "K8": {"family": "kolibri", "dir": "Kolibri-1-MLX-8bit-g64", "bits": 8},
    "K4": {"family": "kolibri", "dir": "Kolibri-1-MLX-4bit-g64", "bits": 4},
    "G8": {"family": "gemma4", "dir": "gemma-4-26b-a4b-it-8bit", "bits": 8},
    "G4": {"family": "gemma4", "dir": "gemma-4-26b-a4b-it-4bit", "bits": 4},
    "Q36-8": {"family": "qwen3_6", "dir": "Qwen3.6-35B-A3B-8bit", "bits": 8},
    "Q36-4": {"family": "qwen3_6", "dir": "Qwen3.6-35B-A3B-4bit", "bits": 4},
    "Q38-8": {"family": "qwen3_8", "dir": "Qwen3.8-27B-8bit", "bits": 8},
    "Q38-4": {"family": "qwen3_8", "dir": "Qwen3.8-27B-4bit", "bits": 4},
}

KOLIBRI_ARMS = ("K8", "K4")

# The decode batch sizes a cell may use (runner/plan_rules.json B_choices; DESIGN §3.14: allowed_B is one of
# {1}, {1, 2, 4, 8} and {1, 2, 4, 8, 16}).
B_CHOICES = (1, 2, 4, 8, 16)


def arm_dir(arm: str) -> Path:
    """A Kolibri arm resolves under builds_dir() (exp_037's refreshed clones, DESIGN §2.8), a peer under
    models_dir(). The directory names are unchanged."""
    root = builds_dir() if arm in KOLIBRI_ARMS else models_dir()
    return root / ARMS[arm]["dir"]


def clip_B(B: int, allowed: Iterable[int] | None) -> int:
    """The largest allowed batch size <= B (DESIGN §6.3: the memory rule's B clipped to the gate's allowed_B, the
    same path as a peer at B = 1). allowed None means no restriction; 0 when B < 1 or nothing allowed is <= B."""
    B = int(B)
    if allowed is None:
        return B
    fits = [int(a) for a in allowed if 1 <= int(a) <= B]
    return max(fits) if fits else 0


# Task keys used in cell ids, file names and plan_rules.json. The prefix before
# the first "_" (or the whole key) names the cap family of §7.2 "caps".
def task_kind(task: str) -> str:
    """gpqa | mmlu | aime | ifbench | rgb (the cap family of a task key)."""
    for kind in ("gpqa", "mmlu", "aime", "ifbench", "rgb"):
        if task == kind or task.startswith(kind + "_"):
            return kind
    raise ValueError(f"unknown task key {task!r}")


# Withheld sets: GPQA EN/DE (incl. the pilot's GPQA main rows), AIME-DE and all
# RGB cells. Their raw text goes to $EXP036_PRIVATE/exp037; the repo gets hashes only
# (BUILD_SPEC §2 Rights, §5.4 record schema).
def is_withheld(task: str) -> bool:
    return task.startswith("gpqa") or task.startswith("rgb") or task == "aime_de"


def cell_id(arm: str, task: str, effort: str, pass_: int = 0) -> str:
    base = f"{arm}:{task}:{effort}"
    return base if pass_ == 0 else f"{base}:p{pass_}"


def cell_file_stem(task: str, effort: str, pass_: int = 0) -> str:
    """<task>_<effort>.jsonl (results layout); pass 2 gets a _p1 suffix."""
    return f"{task}_{effort}" if pass_ == 0 else f"{task}_{effort}_p{pass_}"


# -------------------------------------------------------------------- hashing


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids: Iterable[int]) -> str:
    """sha256 of the token ids as compact JSON, e.g. "[1,2,3]"."""
    return sha256_text(json.dumps([int(i) for i in ids], separators=(",", ":")))


def tree_sha256(root: Path, exclude: Iterable[str] = ()) -> str:
    """sha256 over the sorted lines "relpath\\tsha256" of the files under root.

    Uses tools/hash_tree.py when it exists (the one definition the HYPOTHESIS
    hash table uses); this fallback implements the same rule for tests."""
    try:
        ht = importlib.import_module("tools.hash_tree")
        return ht.tree_sha256(root, exclude=list(exclude)) if exclude else ht.tree_sha256(root)
    except ModuleNotFoundError:
        pass
    root = Path(root)
    ex = tuple(exclude)
    lines = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.name == ".DS_Store":
            continue
        rel = p.relative_to(root).as_posix()
        if any(rel == e or rel.startswith(e.rstrip("/") + "/") for e in ex):
            continue
        lines.append(f"{rel}\t{sha256_file(p)}")
    return sha256_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- time, json


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso(dt: datetime | None = None) -> str:
    """2026-10-03T17:01:52.123Z (UTC from the clock, millisecond resolution)."""
    dt = dt or utc_now()
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def utc_stamp(dt: datetime | None = None) -> str:
    """20261003T170152Z, for file names."""
    dt = dt or utc_now()
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_utc(s: str) -> datetime:
    s = s.strip()
    if len(s) == 16 and s[8] == "T" and s.endswith("Z"):  # 20261003T170152Z
        return datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def dumps(obj: Any, indent: int | None = None) -> str:
    """Deterministic JSON: sorted keys, UTF-8 (no ASCII escaping)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=indent, separators=None if indent else (", ", ": "))


def write_new_json(path: Path, obj: Any) -> None:
    """Write a new JSON file; refuse to overwrite (BUILD_SPEC §2 Append-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(dumps(obj, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_new_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------ redaction


def redact_path(p: Any) -> str:
    """tools/redact.py:redact_path when present; else the same rewrite of
    $EXP036_MODELS/DATA/PRIVATE/WORK and $HOME (BUILD_SPEC §2 Privacy)."""
    try:
        return importlib.import_module("tools.redact").redact_path(p)
    except ModuleNotFoundError:
        pass
    s = str(p)
    for name, root in (
        (f"$EXP036_PRIVATE/{PRIVATE_SUBDIR}", private_dir()),
        ("$EXP036_PRIVATE", private_base_dir()),
        ("$EXP036_WORK", work_dir()),
        ("$EXP036_DATA", data_dir()),
        ("$EXP036_MODELS", models_dir()),
        ("~", Path.home()),
    ):
        r = str(root)
        if s == r or s.startswith(r.rstrip("/") + "/"):
            return name + s[len(r.rstrip("/")):]
    return s


def host_label(chip: str | None = None, memory_bytes: int | None = None) -> str:
    """tools/redact.py:host_label (the fixed run-host label on the mbp, a
    generic chip + memory label elsewhere); never a hostname."""
    try:
        return importlib.import_module("tools.redact").host_label(chip, memory_bytes)
    except (ModuleNotFoundError, AttributeError):
        return HOST_LABEL_FALLBACK


# ------------------------------------------------------- environment record


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def sysctl(name: str) -> str | None:
    return _run(["sysctl", "-n", name])


def git(args: list[str], cwd: Path | None = None) -> str | None:
    return _run(["git", *args], cwd=cwd or EXP_DIR)


def git_state(exp_dir: Path | None = None) -> dict:
    d = exp_dir or EXP_DIR
    head = git(["rev-parse", "HEAD"], d)
    porcelain = git(["status", "--porcelain"], d)
    return {"git_commit": head, "git_dirty": bool(porcelain) if porcelain is not None else None}


def _version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def power_mode() -> str | None:
    out = _run(["pmset", "-g"])
    if not out:
        return None
    keep = [ln.strip() for ln in out.splitlines() if any(k in ln.lower() for k in ("powermode", "lowpowermode"))]
    return "; ".join(keep) or None


def thermal_state() -> str | None:
    return _run(
        ["osascript", "-l", "JavaScript", "-e", 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState']
    )


def env_record(exp_dir: Path | None = None) -> dict:
    """Fields of the JSONL header that describe the host and software
    (BUILD_SPEC §5.4 jsonl header). A fixed host label, never a hostname."""
    chip = sysctl("machdep.cpu.brand_string")
    mem = int(sysctl("hw.memsize") or 0) or None
    rec = {
        "host": host_label(chip, mem),
        "chip": chip,
        "memory_bytes": mem,
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "mlx": _version("mlx"),
        "mlx_metal": _version("mlx-metal"),
        "mlx_lm": _version("mlx-lm"),
        "iogpu_wired_limit_mb": sysctl("iogpu.wired_limit_mb"),
        "power_mode": power_mode(),
    }
    rec.update(git_state(exp_dir))
    try:
        from runner import memory

        L, sources = memory.effective_limit()
        rec["metal_limit_bytes"] = L
        rec["metal_limit_sources"] = sources
    except Exception as e:  # mlx absent on a CPU-only checker
        rec["metal_limit_bytes"] = None
        rec["metal_limit_error"] = type(e).__name__
    return rec


def load_rules(path: Path | None = None) -> dict:
    return json.loads(Path(path or RUNNER_DIR / "plan_rules.json").read_text(encoding="utf-8"))
