# SPDX-License-Identifier: MIT
"""Shared plumbing for the exp_037 bench cells (BUILD_SPEC §5.7; exp_037
DESIGN §6.3).

Nothing here imports mlx, mlx_lm or transformers at module level: the H5
tokenizer cell is CPU only (BUILD_SPEC §5.7, "no mlx import") and uses this
module. MLX helpers live in bench/genutil.py.

What lives here:
- paths: $EXP036_MODELS, $EXP036_DATA, $EXP036_WORK (BUILD_SPEC §1 defaults),
  the exp_007 padding fixtures by their real names (hard fail if missing);
  the Kolibri arm directories resolve under builds_dir() ($EXP037_BUILDS, the
  refreshed clones, else $EXP036_MODELS; DESIGN §2.8), the peers under
  $EXP036_MODELS;
- the arm table (HYPOTHESIS "Arms");
- the guards every bench cell calls (runner/guard.py, imported lazily), the
  gate's allowed_B per Kolibri arm (DESIGN §3.14, §6.3) and the version
  binding run_bench.py does at its start (runner.guard.require_environment);
- result files: a new JSONL file per cell (header, records, end record),
  single JSON files written once, completion detection for run_bench, and
  moving incomplete files to aborted/ (never deleting them);
- system state: thermalState, power source and power mode (C17), the git
  commit and package versions for headers.

Record layout (for plan_fix's S1-hours sum and for analysis/): every JSONL
cell file starts with {"type": "header", "t_start": ...} and ends with
{"type": "end", "complete": true|false, "t_start": ..., "t_end": ...}; every
single-JSON output carries top-level "t_start", "t_end" and "complete".
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# BUILD_SPEC §1: every entry point sets these before mlx_lm or transformers is
# imported. bench/* imports this module first.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

EXP_DIR = Path(__file__).resolve().parent.parent
CHRONOS_DIR = EXP_DIR.parent
# HYPOTHESIS "Task sets": speed and context fixtures, loaded by their real names.
PADDING_DIR = CHRONOS_DIR / "exp_007_hardware_comparison" / "fixtures" / "padding"

# Patched by the tests; production code sleeps for real.
SLEEP: Callable[[float], None] = time.sleep


class BenchError(RuntimeError):
    """A bench cell cannot produce a valid record (the cell fails)."""


class CouldNotRun(RuntimeError):
    """A precondition outside the kit is not met (run_bench exit 3)."""


# ---------------------------------------------------------------------------
# Paths (BUILD_SPEC §1). Defaults are relative to $HOME, never hard-coded.
# ---------------------------------------------------------------------------


def _env_dir(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else default


def models_dir() -> Path:
    return _env_dir("EXP036_MODELS", Path("~/models/exp036").expanduser())


def data_dir() -> Path:
    return _env_dir("EXP036_DATA", models_dir() / "data")


def work_dir() -> Path:
    return _env_dir("EXP036_WORK", models_dir() / "work")


def builds_dir() -> Path:
    """Where the Kolibri arm directories resolve: $EXP037_BUILDS if set and
    non-empty (the APFS clones refreshed with exp_037's port, DESIGN §2.8),
    else models_dir(), which keeps the tiny tests and the dry run working
    unchanged. The peers and the BF16 source stay under models_dir()."""
    return _env_dir("EXP037_BUILDS", models_dir())


def default_results_dir() -> Path:
    return EXP_DIR / "results"


def fineweb_parquet() -> Path:
    """The pinned FineWeb-2 deu_Latn test file (BUILD_SPEC §8)."""
    return data_dir() / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet"


def fixture_path(name: str) -> Path:
    """An exp_007 padding fixture by its real file name; a missing fixture is a
    hard failure (BUILD_SPEC §2, the exp_011 lesson)."""
    p = PADDING_DIR / name
    if not p.is_file():
        raise BenchError(
            f"fixture {name} is missing (expected at "
            f"tasks/chronos/exp_007_hardware_comparison/fixtures/padding/{name}); "
            "fixtures are loaded by their real names and never substituted"
        )
    return p


def read_fixture(name: str) -> tuple[str, str]:
    """(text, sha256 of the file bytes)."""
    raw = fixture_path(name).read_bytes()
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Arms (HYPOTHESIS "Arms"; folder names under $EXP036_MODELS)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    name: str
    family: str  # kolibri | gemma4 | qwen3_6 | qwen3_8 (runner/plan_rules.json keys)
    bits: int
    folder: str


ARMS: dict[str, Arm] = {
    a.name: a
    for a in (
        Arm("K8", "kolibri", 8, "Kolibri-1-MLX-8bit-g64"),
        Arm("K4", "kolibri", 4, "Kolibri-1-MLX-4bit-g64"),
        Arm("G8", "gemma4", 8, "gemma-4-26b-a4b-it-8bit"),
        Arm("G4", "gemma4", 4, "gemma-4-26b-a4b-it-4bit"),
        Arm("Q36-8", "qwen3_6", 8, "Qwen3.6-35B-A3B-8bit"),
        Arm("Q36-4", "qwen3_6", 4, "Qwen3.6-35B-A3B-4bit"),
        Arm("Q38-8", "qwen3_8", 8, "Qwen3.8-27B-8bit"),
        Arm("Q38-4", "qwen3_8", 4, "Qwen3.8-27B-4bit"),
    )
}

# (8-bit arm, 4-bit arm) per family, in the order H8 reports them.
FAMILY_PAIRS: dict[str, tuple[str, str]] = {
    "kolibri": ("K8", "K4"),
    "gemma4": ("G8", "G4"),
    "qwen3_6": ("Q36-8", "Q36-4"),
    "qwen3_8": ("Q38-8", "Q38-4"),
}

# The BF16 source holds the Kolibri tokenizer used by H5 (raw `tokenizers`).
KOLIBRI_BF16_FOLDER = "Kolibri-1-BF16"


def is_kolibri(arm: str) -> bool:
    return ARMS[arm].family == "kolibri"


def arm_dir(arm: str) -> Path:
    """K8 and K4 under builds_dir() (DESIGN §2.8), every peer under models_dir()."""
    return (builds_dir() if is_kolibri(arm) else models_dir()) / ARMS[arm].folder


def weights_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))


def assets_json() -> dict:
    p = EXP_DIR / "assets.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def model_fingerprint(arm: str, model_dir: Path) -> dict:
    """Cheap identity of the model directory for record headers. The full
    per-file hashes are preflight --deep's and the convert record's job."""
    d = Path(model_dir)
    fp: dict[str, Any] = {"arm": arm, "weights_bytes": weights_bytes(d)}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        f = d / name
        fp[name.replace(".", "_") + "_sha256"] = sha256_file(f) if f.is_file() else None
    if is_kolibri(arm):
        rec = d / "exp036_convert_record.json"
        if rec.is_file():
            r = json.loads(rec.read_text())
            fp["convert_record_sha256"] = sha256_file(rec)
            fp["manifest_sha256"] = r.get("manifest_sha256")
            fp["port_sha256"] = r.get("port_sha256")
        else:
            fp["convert_record_sha256"] = None
    else:
        for m in assets_json().get("models", []):
            if m.get("path") == ARMS[arm].folder:
                fp["assets_revision"] = m.get("revision")
    return fp


# ---------------------------------------------------------------------------
# Guards (BUILD_SPEC §5.4 runner/guard.py; "Every bench cell requires
# guard.require_gate for Kolibri arms and guard.require_identity", §5.7)
# ---------------------------------------------------------------------------


def require_identity() -> None:
    from runner import guard

    guard.require_identity()


def require_gate(arm: str):
    from runner import guard

    return guard.require_gate(arm)


def require_tier2() -> None:
    from runner import guard

    guard.require_tier2()


def require_gates(arms: Iterable[str]) -> dict[str, Any]:
    """require_gate for every Kolibri arm in `arms` (a refusal exits); returns
    {arm: what the guard returned}, from which allowed_B() reads the batch
    sizes the gate allows. Peers are not gated and do not appear."""
    out: dict[str, Any] = {}
    for a in arms:
        if is_kolibri(a):
            out[a] = require_gate(a)
    return out


def allowed_B(arm: str, gate: Any) -> Optional[tuple[int, ...]]:
    """The batch sizes the gate allows for `arm` (DESIGN §3.14: allowed_B =
    {1} ∪ ({2, 4, 8} if pass(arm, 8)) ∪ ({16} if pass(arm, 8) and pass(arm, 16))),
    sorted; None for a peer (the gate does not judge peers: no restriction).

    `gate` is what require_gate returned for the arm. It is read by the
    runner's one implementation, runner.guard.allowed_B_of (the list
    require_gate returns, or a record, dict or GateRecord carrying allowed_B):
    a result without allowed_B gives {1}, the base set of §3.14, and a set
    G5-BP-lean cannot produce is refused (the guard exits)."""
    if not is_kolibri(arm):
        return None
    from runner import guard

    return tuple(int(b) for b in guard.allowed_B_of(gate, arm))


def clip_B(B: int, allowed: Optional[Iterable[int]]) -> int:
    """The largest allowed batch size <= B (DESIGN §6.3, the same clipping as
    runner/plan_fix.B_for); B itself when there is no restriction (a peer).
    B >= 1 is the caller's to check."""
    if allowed is None:
        return int(B)
    fit = [int(b) for b in allowed if int(b) <= int(B)]
    if not fit:
        raise CouldNotRun(f"no allowed batch size <= {B} in {sorted(allowed)}")
    return max(fit)


def latest_gate_record(results_dir: Path) -> Optional[dict]:
    """The newest results/gate/gate_<UTC>.json, by name, with its sha256."""
    files = sorted((Path(results_dir) / "gate").glob("gate_*.json"))
    files = [f for f in files if re.fullmatch(r"gate_\d{8}T\d{6}Z\.json", f.name)]
    if not files:
        return None
    f = files[-1]
    return {"file": f"results/gate/{f.name}", "sha256": sha256_file(f)}


def require_environment(results_dir: Path):
    """Version binding at the start of a bench run (DESIGN §6.3):
    runner.guard.require_environment(rec) with rec the newest gate record under
    `results_dir` (runner.guard.newest_gate_record; None when there is none).
    For a real-mode record (not promoted from a dry run) it checks P2's pins
    (MLX 0.32.3 / mlx-metal 0.32.3 / mlx-lm 0.32.0, the GPU family, macOS) and
    refuses on a mismatch, by exiting; for tiny and dry-run records it records
    the observed versions without judging them. Returns its binding record."""
    from runner import guard

    return guard.require_environment(guard.newest_gate_record(Path(results_dir)))


# ---------------------------------------------------------------------------
# Privacy (BUILD_SPEC §2): every path through tools/redact.py
# ---------------------------------------------------------------------------


def redact_path(p) -> str:
    from tools.redact import redact_path as _redact

    return _redact(p)


def host_label() -> str:
    """tools/redact.py's label for this machine: the fixed run-host label on
    the M5 Max, the mini's label on the mini (H5 re-run), from the chip name
    and memory size only."""
    from tools.redact import host_label as _label

    chip = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or None
    return _label(chip, sysctl_int("hw.memsize"))


# ---------------------------------------------------------------------------
# Time, hashing, JSON
# ---------------------------------------------------------------------------


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def utc_iso(t: Optional[_dt.datetime] = None) -> str:
    return (t or utc_now()).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp(t: Optional[_dt.datetime] = None) -> str:
    return (t or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_ids(ids: Iterable[int]) -> str:
    """Canonical hash of a token-id sequence: sha256 of "id,id,...,id" (UTF-8)."""
    return sha256_text(",".join(str(int(i)) for i in ids))


def sha256_file(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dumps(obj: Any) -> str:
    """UTF-8 JSON with sorted keys (BUILD_SPEC §2 Determinism); NaN refused."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)


# ---------------------------------------------------------------------------
# Result files. Append-only (BUILD_SPEC §2): every file is created new ("x");
# no existing results file is ever reopened for writing.
# ---------------------------------------------------------------------------

STAMP_RE = r"\d{8}T\d{6}Z"

# cell -> (subdirectory of results/, file prefix, suffix)
CELL_FILES: dict[str, tuple[str, str, str]] = {
    "speed": ("bench", "speed", ".jsonl"),
    "speed_desc": ("bench", "speed_desc", ".jsonl"),
    "fit": ("bench", "fit", ".jsonl"),
    "c1": ("bench", "c1", ".jsonl"),
    "ladder": ("bench", "ladder", ".jsonl"),
    "tokenizer": ("", "tokenizer", ".json"),
    "kl": ("", "kl_8v4", ".json"),
}


def new_output_path(results_dir: Path, cell: str) -> Path:
    sub, prefix, suffix = CELL_FILES[cell]
    d = Path(results_dir) / sub if sub else Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    while True:
        p = d / f"{prefix}_{utc_stamp()}{suffix}"
        if not p.exists() and not Path(str(p) + ".partial").exists():
            return p
        time.sleep(1.0)  # one file per second per cell; never overwrite


def cell_outputs(results_dir: Path, cell: str) -> list[Path]:
    sub, prefix, suffix = CELL_FILES[cell]
    d = Path(results_dir) / sub if sub else Path(results_dir)
    if not d.is_dir():
        return []
    pat = re.compile(rf"{re.escape(prefix)}_{STAMP_RE}{re.escape(suffix)}")
    return sorted(p for p in d.iterdir() if pat.fullmatch(p.name))


def is_complete(path: Path) -> bool:
    """A JSONL cell file is complete iff its last line is an end record with
    complete true; a JSON output iff its top-level "complete" is true."""
    path = Path(path)
    try:
        if path.suffix == ".jsonl":
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return rec.get("type") == "end" and rec.get("complete") is True
            return False
        return json.loads(path.read_text(encoding="utf-8")).get("complete") is True
    except (OSError, ValueError):
        return False


def complete_output(results_dir: Path, cell: str) -> Optional[Path]:
    done = [p for p in cell_outputs(results_dir, cell) if is_complete(p)]
    return done[-1] if done else None


def quarantine_incomplete(results_dir: Path, cell: str, aborted_dir: Optional[Path] = None) -> list[Path]:
    """Move incomplete outputs of a cell to aborted/<UTC>-bench-<cell>/ with a
    NOTE.md (HYPOTHESIS "What counts as evidence": aborted/ is never deleted).
    Bench files carry no withheld text (C1 records are hashed)."""
    bad = [p for p in cell_outputs(results_dir, cell) if not is_complete(p)]
    partials = []
    sub, prefix, _ = CELL_FILES[cell]
    d = Path(results_dir) / sub if sub else Path(results_dir)
    if d.is_dir():
        partials = sorted(d.glob(f"{prefix}_*.partial"))
    bad += partials
    if not bad:
        return []
    aborted_dir = Path(aborted_dir) if aborted_dir else Path(results_dir).parent / "aborted"
    dest = aborted_dir / f"{utc_stamp()}-bench-{cell}"
    dest.mkdir(parents=True, exist_ok=False)
    lines = [
        f"# Incomplete bench cell `{cell}`",
        "",
        f"Moved here by bench/run_bench.py at {utc_iso()} before the cell was rerun.",
        "The file(s) have no end record with `complete: true` (the run stopped or failed).",
        "",
        "| file | bytes | sha256 |",
        "|---|---|---|",
    ]
    moved = []
    for p in bad:
        target = dest / p.name
        lines.append(f"| {p.name} | {p.stat().st_size} | {sha256_file(p)} |")
        shutil.move(str(p), str(target))
        moved.append(target)
    (dest / "NOTE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return moved


class JsonlCell:
    """One new results JSONL file: a header record, records, an end record.

    Opened with mode "x", each line flushed and fsynced. Used as a context
    manager: leaving the block on an exception writes an end record with
    complete false and the exception's type name only (its message may hold
    paths; it is printed to stderr by the caller, never stored)."""

    def __init__(self, path: Path, header: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "x", encoding="utf-8")
        self.t_start = header.get("t_start") or utc_iso()
        self.n_records = 0
        self.finished = False
        self._write({**header, "type": "header", "t_start": self.t_start})

    def _write(self, obj: dict) -> None:
        self._f.write(dumps(obj) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def append(self, record: dict) -> None:
        self._write({**record, "type": "record"})
        self.n_records += 1

    def finish(self, summary: dict) -> None:
        self._write(
            {
                "type": "end",
                "complete": True,
                "t_start": self.t_start,
                "t_end": utc_iso(),
                "n_records": self.n_records,
                "summary": summary,
            }
        )
        self._close()

    def fail(self, error_type: str) -> None:
        self._write(
            {
                "type": "end",
                "complete": False,
                "t_start": self.t_start,
                "t_end": utc_iso(),
                "n_records": self.n_records,
                "error_type": error_type,
            }
        )
        self._close()

    def _close(self) -> None:
        self.finished = True
        self._f.close()

    def __enter__(self) -> "JsonlCell":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.finished:
            self.fail(exc_type.__name__ if exc_type else "NotFinished")
        return False


def write_json_new(path: Path, obj: dict) -> Path:
    """Write a single JSON result once: <path>.partial ("x"), fsync, rename.
    Refuses if the target exists."""
    path = Path(path)
    if path.exists():
        raise BenchError(f"refusing to overwrite {path.name}")
    tmp = Path(str(path) + ".partial")
    with open(tmp, "x", encoding="utf-8") as f:
        f.write(dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)
    return path


# ---------------------------------------------------------------------------
# System state: thermals, power (C17), sysctl, versions, git
# ---------------------------------------------------------------------------

THERMAL_CMD = [
    "osascript",
    "-l",
    "JavaScript",
    "-e",
    'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState',
]
# NSProcessInfoThermalState
THERMAL_LABELS = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def _run(cmd: list[str], timeout: float = 15.0) -> Optional[str]:
    """stdout of a short system command, or None. Patched by the tests."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def thermal_state() -> dict:
    out = _run(THERMAL_CMD)
    try:
        n = int((out or "").strip())
    except ValueError:
        return {"thermal_state": None, "thermal_label": "unavailable"}
    return {"thermal_state": n, "thermal_label": THERMAL_LABELS.get(n, "unknown")}


def power_state() -> dict:
    """Power source and the pmset power-mode fields, as raw values."""
    rec: dict[str, Any] = {"power_source": None, "powermode": None, "lowpowermode": None}
    ps = _run(["pmset", "-g", "ps"]) or ""
    m = re.search(r"drawing from '([^']+)'", ps)
    if m:
        rec["power_source"] = m.group(1)
    g = _run(["pmset", "-g"]) or ""
    for key in ("powermode", "lowpowermode"):
        m = re.search(rf"^\s*{key}\s+(\d+)\s*$", g, flags=re.MULTILINE)
        if m:
            rec[key] = int(m.group(1))
    return rec


def snapshot() -> dict:
    """thermalState, power source and power mode, with a UTC time (C17)."""
    return {"utc": utc_iso(), **thermal_state(), **power_state()}


def sysctl_int(name: str) -> Optional[int]:
    out = _run(["sysctl", "-n", name])
    try:
        return int((out or "").strip())
    except ValueError:
        return None


def idle(seconds: float) -> dict:
    t0 = utc_iso()
    SLEEP(seconds)
    return {"idle_s": seconds, "t_start": t0, "t_end": utc_iso()}


COOL_IDLE_S = 600  # BUILD_SPEC §5.7: speed cells start after a 10-minute idle
COOL_EXTRA_S = 1200  # then at most 20 more minutes waiting for "nominal"
COOL_POLL_S = 60


def wait_cool(min_idle_s: float = COOL_IDLE_S, extra_s: float = COOL_EXTRA_S, poll_s: float = COOL_POLL_S) -> dict:
    """Idle, then require thermalState nominal (BUILD_SPEC §5.7 run_bench).
    Raises CouldNotRun if the state cannot be read or never reaches nominal."""
    before = snapshot()
    print(f"[bench] idling {min_idle_s:.0f} s before the speed cells (thermalState {before['thermal_label']})", flush=True)
    SLEEP(min_idle_s)
    waited = float(min_idle_s)
    after = snapshot()
    while after["thermal_state"] != 0 and waited < min_idle_s + extra_s:
        if after["thermal_state"] is None:
            break
        SLEEP(poll_s)
        waited += poll_s
        after = snapshot()
    rec = {"idle_s": waited, "before": before, "after": after}
    if after["thermal_state"] is None:
        raise CouldNotRun("thermalState could not be read (osascript); the speed cells need it")
    if after["thermal_state"] != 0:
        raise CouldNotRun(f"thermalState is still {after['thermal_label']} after {waited:.0f} s idle")
    return rec


PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "numpy", "tokenizers", "transformers", "safetensors", "pyarrow")


def package_versions() -> dict:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def git_state() -> dict:
    commit = _run(["git", "-C", str(EXP_DIR), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(EXP_DIR), "status", "--porcelain", "--", "."])
    dirty = None
    if status is not None:
        dirty = any(
            line.strip() and not any(f"/{d}/" in line for d in ("results", "aborted", "evidence"))
            for line in status.splitlines()
        )
    return {"commit": commit.strip() if commit else None, "dirty": dirty}


def metal_limits() -> dict:
    """mx.device_info() fields and iogpu.wired_limit_mb (imports mlx)."""
    import mlx.core as mx

    info = mx.device_info()
    return {
        "device_name": info.get("device_name"),
        "memory_size": info.get("memory_size"),
        "max_recommended_working_set_size": info.get("max_recommended_working_set_size"),
        "iogpu_wired_limit_mb": sysctl_int("iogpu.wired_limit_mb"),
    }


def environment(include_mlx: bool = True) -> dict:
    env = {
        "host": host_label(),
        "macos": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
        "packages": package_versions(),
        "git": git_state(),
    }
    if include_mlx:
        env["metal"] = metal_limits()
    return env


def base_header(cell: str, spec: str, results_dir: Path, include_mlx: bool = True) -> dict:
    """Fields every bench header carries."""
    return {
        "cell": cell,
        "spec": spec,
        "t_start": utc_iso(),
        "environment": environment(include_mlx=include_mlx),
        "gate_record": latest_gate_record(results_dir),
    }
