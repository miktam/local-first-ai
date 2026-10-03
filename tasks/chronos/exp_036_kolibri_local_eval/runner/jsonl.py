"""Append-only raw JSONL (BUILD_SPEC §5.4 runner/jsonl.py; §2 Append-only).

- One JSON object per line, keys sorted, UTF-8. Every append is written,
  flushed and fsync'ed before the call returns, so a crash loses at most the
  line being written.
- open_run() never truncates. If the file does not end in "\\n" (a torn last
  line from a crash), it copies the torn bytes aside and appends a single
  "\\n", so the next record starts on its own line; the torn line stays in
  the file and is counted as unparsable by scan(). Torn bytes from a file
  under $EXP036_PRIVATE go to $EXP036_PRIVATE/aborted/; the repo
  aborted/<UTC>-<what>/NOTE.md then records only their sha256 and byte
  count. Torn bytes from a repo file go to aborted/<UTC>-<what>/.
- The first line of a new file is a "header" record; reopening an existing
  file (a resumed cell) appends a "resume" header instead.
- Duplicate keys are kept; only the first complete record of a key counts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from runner.common import (
    EXP_DIR,
    cell_file_stem,
    is_under,
    private_dir,
    redact_path,
    sha256_bytes,
    utc_iso,
    utc_stamp,
    write_new_text,
)

Key = tuple  # (item_id: str, pass_: int)


class Writer:
    """Append-only line writer with flush + fsync per record."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "ab")
        self.n_written = 0

    def append(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if "\n" in line:  # json.dumps escapes newlines; this is a guard, not a path
            raise ValueError("record serialises to more than one line")
        self._f.write(line.encode("utf-8") + b"\n")
        self._f.flush()
        os.fsync(self._f.fileno())
        self.n_written += 1

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _torn_tail(path: Path) -> bytes:
    """Bytes after the last newline, if the file does not end with one."""
    size = path.stat().st_size
    if size == 0:
        return b""
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return b""
        # Read backwards in chunks until a newline (or the start of the file).
        pos = size
        tail = b""
        while pos > 0:
            step = min(1 << 16, pos)
            pos -= step
            f.seek(pos)
            chunk = f.read(step)
            nl = chunk.rfind(b"\n")
            if nl >= 0:
                return chunk[nl + 1:] + tail
            tail = chunk + tail
        return tail


def _what(path: Path) -> str:
    parts = Path(path).parts[-2:]
    return "torn-" + "-".join(p.replace(".jsonl", "") for p in parts)


def _set_aside_torn(path: Path, torn: bytes, aborted_root: Path, private_root: Path) -> Path:
    """Copy torn bytes aside; return the repo note directory."""
    stamp = utc_stamp()
    what = _what(path)
    note_dir = Path(aborted_root) / f"{stamp}-{what}"
    digest = sha256_bytes(torn)
    if is_under(path, private_root):
        priv_dir = Path(private_root) / "aborted" / f"{stamp}-{what}"
        priv_dir.mkdir(parents=True, exist_ok=True)
        write_new_bytes(priv_dir / "torn.part", torn)
        note = (
            f"# Torn line in a private raw file ({stamp})\n\n"
            f"- File: `{redact_path(path)}`\n"
            f"- Torn bytes: {len(torn)}, sha256 `{digest}`\n"
            f"- Copy: `{redact_path(priv_dir / 'torn.part')}` (private; never in the repo)\n"
            "- The torn line stays in the file and is skipped as unparsable; the item is re-run on resume.\n"
        )
    else:
        note_dir.mkdir(parents=True, exist_ok=True)
        write_new_bytes(note_dir / "torn.part", torn)
        note = (
            f"# Torn line in a raw file ({stamp})\n\n"
            f"- File: `{redact_path(path)}`\n"
            f"- Torn bytes: {len(torn)}, sha256 `{digest}`, copied to `torn.part` here\n"
            "- The torn line stays in the file and is skipped as unparsable; the item is re-run on resume.\n"
        )
    write_new_text(note_dir / "NOTE.md", note)
    return note_dir


def write_new_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def open_run(
    path: Path,
    header: dict,
    *,
    aborted_root: Path | None = None,
    private_root: Path | None = None,
) -> Writer:
    """Open a raw JSONL file for appending and write its (resume) header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aborted_root = Path(aborted_root or EXP_DIR / "aborted")
    private_root = Path(private_root or private_dir())
    existing = path.exists() and path.stat().st_size > 0
    torn_note = None
    if existing:
        torn = _torn_tail(path)
        if torn:
            torn_note = _set_aside_torn(path, torn, aborted_root, private_root)
            with open(path, "ab") as f:
                f.write(b"\n")
                f.flush()
                os.fsync(f.fileno())
    w = Writer(path)
    hdr = dict(header)
    hdr["type"] = "resume" if existing else "header"
    hdr.setdefault("utc", utc_iso())
    if torn_note is not None:
        hdr["torn_line_note"] = redact_path(torn_note)
    w.append(hdr)
    return w


def scan(path: Path) -> tuple[list[dict], int]:
    """All parseable JSON objects in file order, and the unparsable line count."""
    records: list[dict] = []
    bad = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
    return records, bad


def record_key(rec: dict) -> Key:
    k = rec["key"]
    return (str(k["item"]), int(k.get("pass", 0)))


def is_complete(rec: dict) -> bool:
    return rec.get("type") == "record" and isinstance(rec.get("key"), dict) and "finish_reason" in rec


def first_records(records: Iterable[dict]) -> list[dict]:
    """First complete record per key, in file order (duplicates ignored)."""
    seen: set = set()
    out = []
    for r in records:
        if not is_complete(r):
            continue
        k = record_key(r)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def headers(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if r.get("type") in ("header", "resume")]


def raw_root(exp_dir: Path | None = None) -> Path:
    return Path(exp_dir or EXP_DIR) / "results" / "raw"


def cell_files(arm: str, task: str, effort: str, pass_: int = 0, *, root: Path | None = None) -> list[Path]:
    """results/raw/*/<arm>/<task>_<effort>.jsonl across sessions, sorted."""
    root = Path(root or raw_root())
    stem = cell_file_stem(task, effort, pass_)
    return sorted(root.glob(f"*/{arm}/{stem}.jsonl"))


def completed_keys(arm: str, task: str, effort: str, pass_: int = 0, *, root: Path | None = None) -> set:
    """Keys (item, pass) with a complete record in any session's file."""
    done: set = set()
    for p in cell_files(arm, task, effort, pass_, root=root):
        recs, _ = scan(p)
        done.update(record_key(r) for r in first_records(recs))
    return done


def fallback_count(path: Path) -> int:
    """How many crash fallbacks a cell has taken (resume headers marked
    crash_fallback; later resumes only carry b_fallback_from forward)."""
    if not Path(path).exists():
        return 0
    recs, _ = scan(path)
    return sum(1 for h in headers(recs) if h.get("crash_fallback"))


def last_header(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    recs, _ = scan(path)
    hs = headers(recs)
    return hs[-1] if hs else None


def read_jsonl(path: Path) -> list[dict]:
    recs, _ = scan(path)
    return recs

