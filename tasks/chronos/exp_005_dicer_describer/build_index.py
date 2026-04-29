#!/usr/bin/env python3
"""
build_index.py — Build the deterministic index for Experiment 005 Phase 0.

Streams Apple Health export.xml once and emits five artefacts:

    index/manifest.json            — record types, counts, date ranges, sources, units
    index/monthly_aggregates.json  — small, Dicer-readable in a single prompt
    index/daily_aggregates.jsonl   — one row per (record_type, day); query target
    index/workouts.jsonl           — one row per workout (no GPX parsing)
    index/ecg_inventory.json       — filename + date per ECG (no content parsing)

Design notes
------------
The Dicer reads `manifest.json` and `monthly_aggregates.json` directly to plan a
route. It does NOT read raw XML or daily aggregates — those are the Describer's
substrate, fetched deterministically by a separate extractor based on the
Dicer's structured plan.

Numeric record types get count + min/max/mean per (type, day) and per (type, month).
Categorical types (e.g. SleepAnalysis) get count + sum-of-durations per bucket.
Other types fall back to count only — sufficient for routing, parsed properly
when extracted.

Memory: iterparse + element clearing keeps RAM bounded regardless of file size.
Tested target: ~3.5GB export.xml on Apple Silicon, single pass.

Run
---
    cd ~/REPOS/local-first-ai/tasks/chronos/exp_005_dicer_describer
    python3 build_index.py \\
        --export ~/REPOS/apple_health_export/export.xml \\
        --routes ~/REPOS/apple_health_export/workout-routes \\
        --ecg    ~/REPOS/apple_health_export/electrocardiograms \\
        --out    index/

Idempotent. Re-run after a new Apple Health export. Output is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import iterparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_apple_date(s: str) -> datetime | None:
    """Apple Health dates: '2019-05-14 07:23:11 +0200' or with 'Z'. Return naive UTC date-key source."""
    if not s:
        return None
    # Strip timezone — we only need date/month buckets, not absolute UTC accuracy.
    # Apple's local-time date is what the user means when they say "May 14".
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def try_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def short_type(t: str) -> str:
    """Strip 'HKQuantityTypeIdentifier' / 'HKCategoryTypeIdentifier' prefixes for readability."""
    for prefix in (
        "HKQuantityTypeIdentifier",
        "HKCategoryTypeIdentifier",
        "HKDataType",
    ):
        if t.startswith(prefix):
            return t[len(prefix):]
    return t


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

class NumericAgg:
    __slots__ = ("count", "total", "min_v", "max_v")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.min_v = float("inf")
        self.max_v = float("-inf")

    def add(self, v: float) -> None:
        self.count += 1
        self.total += v
        if v < self.min_v:
            self.min_v = v
        if v > self.max_v:
            self.max_v = v

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        return {
            "count": self.count,
            "min": round(self.min_v, 4),
            "max": round(self.max_v, 4),
            "mean": round(self.total / self.count, 4),
        }


class TypeStats:
    """Per-record-type rollup for manifest.json."""
    __slots__ = ("count", "first", "last", "sources", "units", "is_numeric")

    def __init__(self) -> None:
        self.count = 0
        self.first: datetime | None = None
        self.last: datetime | None = None
        self.sources: set[str] = set()
        self.units: set[str] = set()
        self.is_numeric: bool = False

    def update(self, dt: datetime | None, source: str | None, unit: str | None, numeric: bool) -> None:
        self.count += 1
        if dt is not None:
            if self.first is None or dt < self.first:
                self.first = dt
            if self.last is None or dt > self.last:
                self.last = dt
        if source:
            self.sources.add(source)
        if unit:
            self.units.add(unit)
        if numeric:
            self.is_numeric = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "first": self.first.isoformat() if self.first else None,
            "last": self.last.isoformat() if self.last else None,
            "sources": sorted(self.sources),
            "units": sorted(self.units),
            "numeric": self.is_numeric,
        }


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def build(export_path: Path, routes_dir: Path | None, ecg_dir: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    type_stats: dict[str, TypeStats] = defaultdict(TypeStats)
    # daily[(type, "YYYY-MM-DD")] -> NumericAgg (or count-only via .count)
    daily: dict[tuple[str, str], NumericAgg] = defaultdict(NumericAgg)
    monthly: dict[tuple[str, str], NumericAgg] = defaultdict(NumericAgg)

    workouts_path = out_dir / "workouts.jsonl"
    workouts_f = workouts_path.open("w", encoding="utf-8")
    workout_count = 0
    workout_by_type: dict[str, int] = defaultdict(int)
    workout_first: datetime | None = None
    workout_last: datetime | None = None

    record_count = 0
    started = time.time()

    print(f"[build_index] streaming {export_path} ({export_path.stat().st_size / 1e9:.2f} GB)", flush=True)

    # iterparse with end events; clear elements after handling to bound memory.
    context = iterparse(str(export_path), events=("end",))

    for _event, elem in context:
        tag = elem.tag

        if tag == "Record":
            t = elem.get("type", "")
            t_short = short_type(t)
            start = elem.get("startDate", "")
            dt = parse_apple_date(start)
            source = elem.get("sourceName")
            unit = elem.get("unit")
            v = try_float(elem.get("value"))

            type_stats[t_short].update(dt, source, unit, numeric=v is not None)

            if dt is not None:
                day_key = dt.strftime("%Y-%m-%d")
                month_key = dt.strftime("%Y-%m")
                if v is not None:
                    daily[(t_short, day_key)].add(v)
                    monthly[(t_short, month_key)].add(v)
                else:
                    # Count-only for non-numeric (categorical) records.
                    daily[(t_short, day_key)].count += 1
                    monthly[(t_short, month_key)].count += 1

            record_count += 1
            elem.clear()

        elif tag == "Workout":
            wtype = short_type(elem.get("workoutActivityType", ""))
            start = elem.get("startDate", "")
            end = elem.get("endDate", "")
            dt = parse_apple_date(start)
            row = {
                "type": wtype,
                "start": start,
                "end": end,
                "duration": try_float(elem.get("duration")),
                "duration_unit": elem.get("durationUnit"),
                "distance": try_float(elem.get("totalDistance")),
                "distance_unit": elem.get("totalDistanceUnit"),
                "energy": try_float(elem.get("totalEnergyBurned")),
                "energy_unit": elem.get("totalEnergyBurnedUnit"),
                "source": elem.get("sourceName"),
            }
            workouts_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            workout_count += 1
            workout_by_type[wtype] += 1
            if dt is not None:
                if workout_first is None or dt < workout_first:
                    workout_first = dt
                if workout_last is None or dt > workout_last:
                    workout_last = dt
            elem.clear()

        elif tag in ("Correlation", "ActivitySummary", "ClinicalRecord", "Audiogram", "VisionPrescription"):
            # Out of scope for Phase 0 — clear to keep memory bounded.
            elem.clear()

        # Progress heartbeat
        if record_count and record_count % 1_000_000 == 0:
            elapsed = time.time() - started
            print(f"[build_index]   {record_count:>12,} records  ({elapsed:6.1f}s)", flush=True)

    workouts_f.close()

    elapsed = time.time() - started
    print(f"[build_index] parse done: {record_count:,} records in {elapsed:.1f}s", flush=True)

    # ----- write manifest.json -----
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_xml": str(export_path),
        "source_xml_bytes": export_path.stat().st_size,
        "record_count": record_count,
        "record_types": {t: s.to_dict() for t, s in sorted(type_stats.items())},
        "workouts": {
            "count": workout_count,
            "first": workout_first.isoformat() if workout_first else None,
            "last": workout_last.isoformat() if workout_last else None,
            "by_type": dict(sorted(workout_by_type.items())),
        },
    }

    # ECG inventory (filenames + dates from name; no content parsing)
    ecg_inventory: list[dict[str, Any]] = []
    if ecg_dir and ecg_dir.exists():
        for p in sorted(ecg_dir.iterdir()):
            if p.is_file() and p.suffix.lower() == ".csv":
                ecg_inventory.append({
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                    "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                })
    manifest["ecg"] = {"count": len(ecg_inventory)}

    # Workout routes inventory (filename-only; no GPX parsing)
    if routes_dir and routes_dir.exists():
        route_files = [p.name for p in routes_dir.iterdir() if p.is_file()]
        manifest["workout_routes"] = {
            "count": len(route_files),
            "directory": str(routes_dir),
        }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ----- write monthly_aggregates.json (Dicer-readable, fits in a prompt) -----
    monthly_out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (t, mkey), agg in monthly.items():
        if type_stats[t].is_numeric:
            monthly_out[t][mkey] = agg.to_dict()
        else:
            monthly_out[t][mkey] = {"count": agg.count}
    (out_dir / "monthly_aggregates.json").write_text(
        json.dumps(monthly_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ----- write daily_aggregates.jsonl (larger, queried deterministically) -----
    with (out_dir / "daily_aggregates.jsonl").open("w", encoding="utf-8") as f:
        for (t, dkey), agg in sorted(daily.items()):
            row: dict[str, Any] = {"type": t, "date": dkey}
            if type_stats[t].is_numeric:
                row.update(agg.to_dict())
            else:
                row["count"] = agg.count
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ----- write ecg_inventory.json -----
    (out_dir / "ecg_inventory.json").write_text(
        json.dumps({"count": len(ecg_inventory), "files": ecg_inventory}, indent=2),
        encoding="utf-8",
    )

    # Summary to stderr-style stdout
    print(f"[build_index] wrote {out_dir}/manifest.json", flush=True)
    print(f"[build_index] wrote {out_dir}/monthly_aggregates.json"
          f"  ({len(monthly):,} type-month cells)", flush=True)
    print(f"[build_index] wrote {out_dir}/daily_aggregates.jsonl"
          f"   ({len(daily):,} type-day cells)", flush=True)
    print(f"[build_index] wrote {out_dir}/workouts.jsonl"
          f"           ({workout_count:,} workouts)", flush=True)
    print(f"[build_index] wrote {out_dir}/ecg_inventory.json"
          f"        ({len(ecg_inventory)} ECGs)", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, type=Path,
                   help="Path to export.xml")
    p.add_argument("--routes", type=Path, default=None,
                   help="Path to workout-routes/ (optional, inventoried by filename only)")
    p.add_argument("--ecg", type=Path, default=None,
                   help="Path to electrocardiograms/ (optional, inventoried by filename only)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for index artefacts")
    args = p.parse_args()

    export_path = args.export.expanduser().resolve()
    if not export_path.exists():
        print(f"error: {export_path} does not exist", file=sys.stderr)
        return 2

    routes_dir = args.routes.expanduser().resolve() if args.routes else None
    ecg_dir = args.ecg.expanduser().resolve() if args.ecg else None
    out_dir = args.out.expanduser().resolve()

    build(export_path, routes_dir, ecg_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
