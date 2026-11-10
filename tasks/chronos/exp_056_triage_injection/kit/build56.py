#!/usr/bin/env python3
"""exp_056 (Bodyguard) variant builder.

Combines EVERY benign base message with EVERY payload file into one
variant per pair. The payload directory and the output directory are
supplied by the operator as arguments; the payloads never live in this
workspace and the builder never prints their contents, only file names.

Variant layout (bytes):

    <base bytes with trailing CR/LF stripped>
    <blank line>
    <payload bytes, unchanged>

Variant name: <base stem>__<payload stem><base suffix>
(e.g. X003.eml + A1.txt -> X003__A1.eml; the base's extension is kept).

Standard library only. No network.
"""

import argparse
import json
import sys
from pathlib import Path


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _list_payloads(directory: Path) -> list:
    if not directory.is_dir():
        sys.exit(f"error: payloads directory not found: {directory}")
    files = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )
    if not files:
        sys.exit(f"error: no *.txt payload files in {directory}")
    return files


def _list_bases(directory: Path) -> list:
    if not directory.is_dir():
        sys.exit(f"error: base directory not found: {directory}")
    files = sorted(
        (p for p in directory.iterdir()
         if p.is_file() and p.suffix.lower() in (".eml", ".txt")),
        key=lambda p: p.name,
    )
    if not files:
        sys.exit(f"error: no *.eml/*.txt base files in {directory}")
    return files


def build(payloads_dir: Path, out_dir: Path, base_dir: Path) -> list:
    """Build one variant per (base, payload) pair and write them to out_dir.

    Returns the manifest (list of dicts: variant / payload / base).
    """
    payloads = _list_payloads(payloads_dir)
    bases = _list_bases(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for base in bases:
        base_bytes = _read_bytes(base)
        for payload in payloads:
            variant_bytes = base_bytes.rstrip(b"\r\n") + b"\n\n" + _read_bytes(payload)
            name = f"{base.stem}__{payload.stem}{base.suffix}"
            (out_dir / name).write_bytes(variant_bytes)
            manifest.append(
                {
                    "variant": name,
                    "payload": payload.stem,
                    "base": base.name,
                }
            )
            print(f"{name}  {payload.stem}  {base.name}")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    print(f"wrote {len(manifest)} variant(s) and manifest.json to {out_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, default=Path("fixtures/base"),
                        help="directory of benign base messages (default: fixtures/base)")
    parser.add_argument("payloads_dir", type=Path,
                        help="directory containing one *.txt file per payload")
    parser.add_argument("out_dir", type=Path,
                        help="directory to write variants (outside this workspace)")
    args = parser.parse_args()
    build(args.payloads_dir, args.out_dir, args.base_dir)


if __name__ == "__main__":
    main()
