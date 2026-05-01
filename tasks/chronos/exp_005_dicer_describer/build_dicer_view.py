#!/usr/bin/env python3
"""
build_dicer_view.py — Produce a trimmed manifest view for the Dicer prompt.

The full manifest.json is ~13K. Most of it is noise from the Dicer's
perspective: source lists, units, exact counts. The Dicer needs:

    - which record types exist
    - their date range (so it doesn't route to empty)
    - a rough volume hint (so it knows what 'big' looks like)
    - whether the type is numeric (affects aggregation choice)

This script reads index/manifest.json and writes index/dicer_view.json
with just those fields, ordered for legibility.

Run
---
    python3 build_dicer_view.py --index-dir index/

Idempotent. Re-run after rebuilding the index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def trim(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "generated_at": manifest.get("generated_at"),
        "record_count_total": manifest.get("record_count"),
        "record_types": {},
        "workouts": {
            "first": manifest["workouts"]["first"],
            "last": manifest["workouts"]["last"],
            "by_type": manifest["workouts"]["by_type"],
        },
        "ecg_count": manifest.get("ecg", {}).get("count", 0),
        "workout_routes_count": manifest.get("workout_routes", {}).get("count", 0),
    }

    # Order record types by count desc — Dicer sees high-volume types first,
    # which matches their importance for routing.
    types = manifest.get("record_types", {})
    ordered = sorted(types.items(), key=lambda kv: -kv[1]["count"])
    for name, stats in ordered:
        out["record_types"][name] = {
            "first": stats["first"][:10] if stats["first"] else None,
            "last":  stats["last"][:10]  if stats["last"]  else None,
            "count": stats["count"],
            "numeric": stats["numeric"],
        }

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index-dir", type=Path, default=Path("index"))
    args = p.parse_args()

    idx = args.index_dir.expanduser().resolve()
    src = idx / "manifest.json"
    dst = idx / "dicer_view.json"

    if not src.exists():
        print(f"error: {src} does not exist; run build_index.py first.",
              file=sys.stderr)
        return 2

    manifest = json.loads(src.read_text(encoding="utf-8"))
    view = trim(manifest)
    dst.write_text(json.dumps(view, indent=2, ensure_ascii=False), encoding="utf-8")

    size = dst.stat().st_size
    print(f"[build_dicer_view] wrote {dst} ({size:,} bytes, "
          f"{len(view['record_types'])} types, "
          f"{len(view['workouts']['by_type'])} workout types)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
