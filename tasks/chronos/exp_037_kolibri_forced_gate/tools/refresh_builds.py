#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Clone exp_036's converted K8 / K4 builds and refresh their port file (exp_037 DESIGN §2.8;
RUNBOOK step 8, replacing convert).

    "$PY" tools/refresh_builds.py [--bits 8,4] [--results-dir DIR]

For each of Kolibri-1-MLX-8bit-g64 and Kolibri-1-MLX-4bit-g64 (in that order):

  1. Checks the source, $EXP036_MODELS/<build>: a real directory holding no symbolic link, whose
     kolibri1.py has exp_036's port sha256 (cd6153b8…, EXP036_PORT_SHA256) and whose
     exp036_convert_record.json hashes to the value recorded when it was first cloned (the
     `source_record_sha256` of every earlier refresh record of these bits, if any).
  2. Clones it with `cp -Rc` (APFS clonefile: a separate inode per file, so no write to the clone
     can reach exp_036's files) to $EXP037_BUILDS/<build>, through <build>.partial and a rename.
     If the target already exists, every model*.safetensors file and config.json must equal the
     source's by sha256 (and the shard sets must be equal); the clone is then skipped. Only a
     shard or config mismatch refuses, so a refresh after a port fix needs no manual delete.
  3. Refreshes the clone's port file with E37's port/convert.py refresh_port_file(): kolibri1.py
     and the record are rewritten through temporary files and os.replace, so only the clone
     changes, and its verification load passes trust_remote_code=True (mlx-lm 0.32.0, #1385;
     DESIGN §6.5). Afterwards convert.check_port_file(clone) must pass against E37's port.
  4. Re-checks the source (step 1, plus the size, mtime and inode of every file in it) and
     refuses if anything moved.
  5. Writes results/convert/refresh_<bits>bit_<UTC>.json: t_start, t_end, the source and the target
     (redacted), the shard count, the source record's sha256, previous_port_sha256 and
     port_sha256. runner/plan_fix.py's S1_GLOBS ("convert/*.json") counts it in S1 hours.

It is idempotent: a second run with the same port verifies the clone, finds the port current and
writes a new record whose previous_port_sha256 equals port_sha256. It refuses unless
$EXP037_BUILDS is set to a directory other than $EXP036_MODELS (the clones would otherwise be the
sources), and unless the git identity is Miktam <hello@localfirstai.eu>. The directory names stay
Kolibri-1-MLX-{8,4}bit-g64, and every Kolibri arm resolves under builds_dir() (gate, runner, bench,
tools). Weight-free: the verification load is lazy; nothing reads the network.

Exit codes: 0 every build refreshed; 1 refused or failed (the message names the step).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common
from tools.redact import redact_path

SCHEMA = "exp037 build refresh v1"
# exp_036's port/kolibri1.py, the port its converted builds carry (DESIGN §6.1).
EXP036_PORT_SHA256 = "cd6153b8b00b81f7058e786f33672a07b46c494be0203ad75b04372179dc5584"
BITS = (8, 4)
MODEL_FILE = "kolibri1.py"
CONVERT_RECORD = "exp036_convert_record.json"   # the record's name is kept (DESIGN §2.10)
SHARD_GLOB = "model*.safetensors"
BUILDS_LABEL = "$EXP037_BUILDS"
DEFAULT_RESULTS_DIR = common.EXP_DIR / "results" / "convert"


class RefreshError(RuntimeError):
    """A refusal: the step and the reason are in the message; nothing after it ran."""


def build_name(bits: int) -> str:
    return f"Kolibri-1-MLX-{bits}bit-g64"


def _convert():
    from port import convert  # E37's convert.py: refresh_port_file, check_port_file, PORT_FILE

    return convert


# --------------------------------------------------------------------------
# Step 1 / 4: the source
# --------------------------------------------------------------------------

def _stat_fingerprint(d: Path) -> dict:
    """{relpath: [size, mtime_ns, inode]} of every entry under d (files and directories)."""
    out = {}
    for p in sorted(d.rglob("*")):
        st = p.lstat()
        out[p.relative_to(d).as_posix()] = [st.st_size if p.is_file() else None, st.st_mtime_ns, st.st_ino]
    return out


def source_state(src: Path, expect_port: str) -> dict:
    """Step 1: the source's port and record sha256, its shard names and a stat fingerprint."""
    if src.is_symlink() or not src.is_dir():
        raise RefreshError(f"step 1: {redact_path(src)} is not a directory (or is a symbolic link)")
    links = [p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_symlink()]
    if links:
        raise RefreshError(f"step 1: {redact_path(src)} holds symbolic links {links[:5]}; a clone would share "
                           "their targets with exp_036")
    port = src / MODEL_FILE
    record = src / CONVERT_RECORD
    if not port.is_file() or not record.is_file():
        raise RefreshError(f"step 1: {redact_path(src)} lacks {MODEL_FILE} or {CONVERT_RECORD}; "
                           "not a build written by port/convert.py")
    port_sha = common.sha256_file(port)
    if port_sha != expect_port:
        raise RefreshError(f"step 1: {redact_path(port)} sha256 {port_sha[:16]}… is not the expected source "
                           f"port {expect_port[:16]}… (exp_036's port)")
    shards = sorted(p.name for p in src.glob(SHARD_GLOB) if p.is_file())
    if not shards or not (src / "config.json").is_file():
        raise RefreshError(f"step 1: {redact_path(src)} has no {SHARD_GLOB} shards or no config.json")
    return {"port_sha256": port_sha, "record_sha256": common.sha256_file(record), "shards": shards,
            "stat": _stat_fingerprint(src)}


def recorded_source_record_shas(results_dir: Path, bits: int) -> dict:
    """{source_record_sha256: record file name} of the earlier refresh records of these bits."""
    out = {}
    for p in sorted(Path(results_dir).glob(f"refresh_{bits}bit_*.json")):
        try:
            v = json.loads(p.read_text(encoding="utf-8")).get("source_record_sha256")
        except (OSError, json.JSONDecodeError):
            continue
        if v:
            out.setdefault(v, p.name)
    return out


# --------------------------------------------------------------------------
# Step 2: the clone
# --------------------------------------------------------------------------

def _hash_all(paths: list[Path], workers: int) -> list[str]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(common.sha256_file, paths))


def verify_existing(src: Path, dst: Path, shards: list[str], workers: int) -> list[str]:
    """Step 2 on an existing target: the mismatches (empty = the target's shards and config.json equal
    the source's by sha256, and the shard sets are equal)."""
    have = sorted(p.name for p in dst.glob(SHARD_GLOB) if p.is_file())
    problems = []
    if have != shards:
        problems.append(f"shard set differs (source {len(shards)}, target {len(have)}: "
                        f"missing {sorted(set(shards) - set(have))[:5]}, extra {sorted(set(have) - set(shards))[:5]})")
    names = [n for n in shards + ["config.json"] if (dst / n).is_file()]
    if not (dst / "config.json").is_file():
        problems.append("target has no config.json")
    a = _hash_all([src / n for n in names], workers)
    b = _hash_all([dst / n for n in names], workers)
    problems += [f"{n}: sha256 differs" for n, x, y in zip(names, a, b) if x != y]
    return problems


def clone(src: Path, dst: Path) -> None:
    """cp -Rc (APFS clonefile) through <dst>.partial and a rename; then every file of the clone must
    be a separate inode from its source file."""
    partial = dst.with_name(dst.name + ".partial")
    if partial.is_symlink():
        raise RefreshError(f"step 2: {partial.name} is a symbolic link; remove it by hand")
    if partial.exists():
        import shutil

        shutil.rmtree(partial)   # a leftover clone of an interrupted run: its own inodes only
    dst.parent.mkdir(parents=True, exist_ok=True)
    rc, _out, err = common.run(["cp", "-Rc", str(src), str(partial)], timeout=3600)
    if rc != 0:
        raise RefreshError(f"step 2: cp -Rc failed (exit {rc}): {err.strip()[:300]} (an APFS volume is required)")
    shared = [p.relative_to(partial).as_posix() for p in partial.rglob("*")
              if p.is_file() and p.stat().st_ino == (src / p.relative_to(partial)).stat().st_ino]
    if shared:
        raise RefreshError(f"step 2: the clone shares inodes with the source ({shared[:5]}); not a clone")
    os.rename(partial, dst)


# --------------------------------------------------------------------------
# One build, all steps
# --------------------------------------------------------------------------

def refresh_one(bits: int, models: Path, builds: Path, results_dir: Path, expect_port: str = EXP036_PORT_SHA256,
                port_file: Path | None = None, workers: int = 8, log=print) -> dict:
    """Steps 1-5 for one build; returns the record written (its path under "record_path")."""
    convert = _convert()
    port_file = Path(port_file or convert.PORT_FILE)
    t_start = common.utc_iso()
    name = build_name(bits)
    src, dst = models / name, builds / name

    # 1. the source
    s1 = source_state(src, expect_port)
    recorded = recorded_source_record_shas(results_dir, bits)
    other = {sha: f for sha, f in recorded.items() if sha != s1["record_sha256"]}
    if other:
        raise RefreshError(f"step 1: {redact_path(src / CONVERT_RECORD)} sha256 {s1['record_sha256'][:16]}… differs "
                           f"from the value recorded at its clone ({', '.join(sorted(other.values()))})")

    # 2. the clone
    if dst.is_symlink():
        raise RefreshError(f"step 2: {BUILDS_LABEL}/{name} is a symbolic link")
    if dst.exists():
        problems = verify_existing(src, dst, s1["shards"], workers)
        if problems:
            raise RefreshError(f"step 2: {BUILDS_LABEL}/{name} exists and does not equal the source: "
                               + "; ".join(problems[:5]))
        mode = "existing clone verified (model*.safetensors and config.json by sha256)"
        log(f"[refresh] {name}: existing clone verified ({len(s1['shards'])} shards + config.json)")
    else:
        clone(src, dst)
        mode = "cloned (cp -Rc)"
        log(f"[refresh] {name}: cloned to {BUILDS_LABEL}/{name}")

    # 3. the port file
    previous = common.sha256_file(dst / MODEL_FILE) if (dst / MODEL_FILE).is_file() else None
    try:
        rec = convert.refresh_port_file(dst, port_file=port_file)
        convert.check_port_file(dst, port_file=port_file)
    except SystemExit as e:   # refresh_port_file reports a failed load as SystemExit
        raise RefreshError(f"step 3: {e}") from None
    except convert.StalePortFile as e:
        raise RefreshError(f"step 3: {e}") from None
    port_sha = common.sha256_file(dst / MODEL_FILE)

    # 4. the source again
    s4 = source_state(src, expect_port)
    moved = [k for k in ("port_sha256", "record_sha256", "shards", "stat") if s4[k] != s1[k]]
    if moved:
        raise RefreshError(f"step 4: the source {redact_path(src)} changed during the refresh ({', '.join(moved)})")

    # 5. the record
    out = {
        "schema": SCHEMA, "experiment": "exp_037", "tool": "tools/refresh_builds.py",
        "bits": bits, "build": name, "mode": mode,
        "source": redact_path(src), "target": f"{BUILDS_LABEL}/{name}",
        "shard_count": len(s1["shards"]),
        "source_port_sha256": s1["port_sha256"], "source_record_sha256": s1["record_sha256"],
        "previous_port_sha256": previous, "port_sha256": port_sha,
        "port_refreshed": previous != port_sha,
        "spec_version": rec.get("spec_version"),
        "target_record_sha256": common.sha256_file(dst / CONVERT_RECORD),
        "manifest_sha256": rec.get("manifest_sha256"),
        "t_start": t_start,
    }
    path = _write_record(results_dir, bits, out)
    out["record_path"] = path
    log(f"[refresh] {name}: port {str(previous)[:12]} -> {port_sha[:12]}; record {path.name}")
    return out


def _write_record(results_dir: Path, bits: int, rec: dict) -> Path:
    """results/convert/refresh_<bits>bit_<UTC>.json; never overwrites (one name per UTC second)."""
    for _ in range(5):
        rec["t_end"] = rec["utc"] = common.utc_iso()
        path = Path(results_dir) / f"refresh_{bits}bit_{common.utc_stamp(common.utc_now())}.json"
        if path.exists():
            time.sleep(1.0)
            continue
        return common.write_new_json(path, rec)
    raise RefreshError(f"step 5: no free record name under {redact_path(results_dir)}")


def refresh(bits=BITS, results_dir=None, expect_port: str = EXP036_PORT_SHA256, port_file=None,
            workers: int = 8, log=print) -> list[dict]:
    """Every build in `bits` order. Refuses before any clone unless $EXP037_BUILDS names a directory
    other than $EXP036_MODELS and the git identity is the experiment's."""
    common.require_identity()
    models, builds = common.models_dir(), common.builds_dir()
    if not os.environ.get("EXP037_BUILDS"):
        raise RefreshError("EXP037_BUILDS is not set: source env/exp037.settings.sh (RUNBOOK block header)")
    if builds.resolve() == models.resolve():
        raise RefreshError("EXP037_BUILDS equals EXP036_MODELS: the clones would be exp_036's builds themselves")
    for b in bits:
        src = (models / build_name(b)).resolve()
        if src == builds.resolve() or src in builds.resolve().parents:
            raise RefreshError(f"EXP037_BUILDS lies inside the source {redact_path(src)}")
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    return [refresh_one(b, models, builds, results_dir, expect_port, port_file, workers, log) for b in bits]


def main(argv=None) -> int:
    sys.dont_write_bytecode = True      # no __pycache__ in a build directory
    common.set_offline_env()
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_037: clone exp_036's K8/K4 builds and refresh their port "
                                             "file (DESIGN §2.8; RUNBOOK step 8)")
    ap.add_argument("--bits", default="8,4", help="comma list of 8,4 (default both, K8 first)")
    ap.add_argument("--results-dir", default=None, help="where refresh_<bits>bit_<UTC>.json goes "
                                                        "(default results/convert)")
    ap.add_argument("--expect-source-port", default=EXP036_PORT_SHA256, metavar="SHA256",
                    help="the sources' kolibri1.py sha256 (default exp_036's port; for tests and dry runs only)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4), help="sha256 threads")
    args = ap.parse_args(argv)
    try:
        bits = tuple(int(b) for b in args.bits.split(",") if b.strip())
        if not bits or any(b not in BITS for b in bits):
            raise ValueError(args.bits)
    except ValueError:
        print(f"refresh_builds: --bits must be a comma list of 8 and 4, got {args.bits!r}", file=sys.stderr)
        return 2
    try:
        recs = refresh(bits, args.results_dir, args.expect_source_port, workers=args.workers)
    except (RefreshError, common.IdentityError) as e:
        print(f"refresh_builds: refused: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "records": [r["record_path"].name for r in recs],
                      "port_sha256": {r["build"]: r["port_sha256"] for r in recs}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
