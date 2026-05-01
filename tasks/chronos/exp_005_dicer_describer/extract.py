#!/usr/bin/env python3
"""
extract.py — Deterministic slice extractor for Experiment 005 Phase 0.

Reads a Dicer routing plan (validated against routing_plan.schema.json) and
produces the slice the Describer ingests. No LLM in this step. Pure parse + IO.

Inputs
------
  --plan        Path to a JSON file containing a routing plan, OR `-` for stdin.
  --index-dir   Directory containing manifest.json, monthly_aggregates.json,
                daily_aggregates.jsonl, workouts.jsonl. Default: ./index
  --export      Path to export.xml (only required if any slice uses
                aggregation_level='raw').
  --routes      Path to workout-routes/ (only required if any slice's workouts
                filter sets include_routes=true).

Output
------
  A single JSON object on stdout, matching the shape:

    {
      "kind": "slice_bundle" | "question",
      ... (see below)
    }

  For kind="slice_bundle":
    - "slices": one entry per Plan.slices entry, in the same order.
    - Each entry contains the resolved data plus metadata:
        {
          "label": "...",                        # passed through
          "record_types": [...],                  # echoed from plan
          "date_range": {"start": ..., "end": ...} | null,
          "aggregation_level": "monthly" | "daily" | "raw",
          "records": [ ... extracted rows ... ],
          "workouts": [ ... extracted workouts ... ] | null,
          "truncated": true | false,
          "row_count": <int>
        }

  For kind="question": passed through from the plan with no extraction.

Validation
----------
Strict. Plans that fail JSON Schema validation, reference unknown record types,
or use HKWorkoutActivityType values not in the manifest are rejected with a
non-zero exit code and a structured error to stderr. No repair, no fuzzy match.

Per ADR-001: the extractor is the enforcement boundary. Failure is data.

Run
---
  python3 extract.py --plan plan.json --index-dir index/
  cat plan.json | python3 extract.py --plan - --index-dir index/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from xml.etree.ElementTree import iterparse


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ExtractError(Exception):
    """Structured extractor failure. Emits to stderr as JSON, exit 2."""
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.details}


# ---------------------------------------------------------------------------
# Schema validation (lightweight, no external deps)
# ---------------------------------------------------------------------------
#
# We don't ship jsonschema as a dep for one validation. The schema is small
# enough to enforce its key invariants by hand. Missing fields, wrong types,
# enum violations, and the discriminated union are all checked here.

VALID_AGG = {"monthly", "daily", "raw"}
MAX_SLICES = 8
MAX_ROWS_CEILING = 5000


def _require(cond: bool, code: str, msg: str, **details: Any) -> None:
    if not cond:
        raise ExtractError(code, msg, **details)


def validate_plan(obj: Any) -> dict[str, Any]:
    """Validate against routing_plan.schema.json. Returns the validated dict."""
    _require(isinstance(obj, dict), "schema/not_object",
             "Top-level value is not a JSON object.")
    kind = obj.get("kind")
    _require(kind in ("plan", "question"), "schema/bad_kind",
             f"kind must be 'plan' or 'question', got {kind!r}.")

    if kind == "question":
        q = obj.get("question")
        _require(isinstance(q, str) and 1 <= len(q) <= 400,
                 "schema/bad_question",
                 "Question must be a string of length 1–400.")
        # 'reason' optional, string ≤300
        if "reason" in obj:
            _require(isinstance(obj["reason"], str) and len(obj["reason"]) <= 300,
                     "schema/bad_reason", "reason must be a string ≤300 chars.")
        # No other fields permitted
        _extra = set(obj) - {"kind", "question", "reason"}
        _require(not _extra, "schema/extra_fields",
                 f"Unexpected fields on question: {sorted(_extra)}")
        return obj

    # kind == "plan"
    slices = obj.get("slices")
    _require(isinstance(slices, list), "schema/no_slices",
             "Plan must have a 'slices' array.")
    _require(1 <= len(slices) <= MAX_SLICES, "schema/slice_count",
             f"Plan must contain 1–{MAX_SLICES} slices; got {len(slices)}.",
             count=len(slices))

    if "rationale" in obj:
        _require(isinstance(obj["rationale"], str) and len(obj["rationale"]) <= 500,
                 "schema/bad_rationale", "rationale must be a string ≤500 chars.")

    _extra = set(obj) - {"kind", "slices", "rationale"}
    _require(not _extra, "schema/extra_fields",
             f"Unexpected fields on plan: {sorted(_extra)}")

    for i, sl in enumerate(slices):
        _validate_slice(sl, i)
    return obj


def _validate_slice(sl: Any, i: int) -> None:
    _require(isinstance(sl, dict), "schema/slice_not_object",
             f"Slice {i} is not an object.")

    agg = sl.get("aggregation_level")
    _require(agg in VALID_AGG, "schema/bad_aggregation",
             f"Slice {i}: aggregation_level must be one of {sorted(VALID_AGG)}; "
             f"got {agg!r}.", slice_index=i)

    has_records = "record_types" in sl and sl["record_types"]
    has_workouts = "workouts" in sl
    _require(has_records or has_workouts, "schema/empty_slice",
             f"Slice {i} must specify record_types or workouts.", slice_index=i)

    if "record_types" in sl:
        rt = sl["record_types"]
        _require(isinstance(rt, list) and all(isinstance(t, str) for t in rt),
                 "schema/bad_record_types",
                 f"Slice {i}: record_types must be an array of strings.",
                 slice_index=i)
        _require(len(rt) == len(set(rt)), "schema/duplicate_record_types",
                 f"Slice {i}: record_types contains duplicates.",
                 slice_index=i)

    if "workouts" in sl:
        w = sl["workouts"]
        _require(isinstance(w, dict), "schema/bad_workouts",
                 f"Slice {i}: workouts must be an object.", slice_index=i)
        _extra_w = set(w) - {"activity_types", "include_routes"}
        _require(not _extra_w, "schema/extra_fields",
                 f"Slice {i}: unexpected workouts fields: {sorted(_extra_w)}",
                 slice_index=i)
        if "activity_types" in w:
            at = w["activity_types"]
            _require(isinstance(at, list) and all(isinstance(t, str) for t in at),
                     "schema/bad_activity_types",
                     f"Slice {i}: activity_types must be an array of strings.",
                     slice_index=i)

    if "date_range" in sl:
        dr = sl["date_range"]
        _require(isinstance(dr, dict), "schema/bad_date_range",
                 f"Slice {i}: date_range must be an object.", slice_index=i)
        for f in ("start", "end"):
            _require(f in dr, "schema/bad_date_range",
                     f"Slice {i}: date_range missing '{f}'.", slice_index=i)
            try:
                date.fromisoformat(dr[f])
            except (TypeError, ValueError):
                raise ExtractError("schema/bad_date_range",
                                   f"Slice {i}: date_range.{f} is not YYYY-MM-DD.",
                                   slice_index=i)
        _require(date.fromisoformat(dr["start"]) <= date.fromisoformat(dr["end"]),
                 "schema/bad_date_range",
                 f"Slice {i}: date_range start > end.", slice_index=i)

    if "max_rows" in sl:
        mr = sl["max_rows"]
        _require(isinstance(mr, int) and 1 <= mr <= MAX_ROWS_CEILING,
                 "schema/bad_max_rows",
                 f"Slice {i}: max_rows must be an integer 1–{MAX_ROWS_CEILING}.",
                 slice_index=i)

    if "label" in sl:
        _require(isinstance(sl["label"], str) and len(sl["label"]) <= 40,
                 "schema/bad_label",
                 f"Slice {i}: label must be a string ≤40 chars.", slice_index=i)

    _extra = set(sl) - {"label", "record_types", "workouts", "date_range",
                        "aggregation_level", "max_rows"}
    _require(not _extra, "schema/extra_fields",
             f"Slice {i}: unexpected fields: {sorted(_extra)}", slice_index=i)


# ---------------------------------------------------------------------------
# Manifest checks (semantic validation, beyond schema)
# ---------------------------------------------------------------------------

def validate_against_manifest(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reject plans that reference record types or activity types not in the corpus."""
    if plan["kind"] != "plan":
        return

    known_types = set(manifest.get("record_types", {}).keys())
    known_workout_types = set(manifest.get("workouts", {}).get("by_type", {}).keys())

    for i, sl in enumerate(plan["slices"]):
        for rt in sl.get("record_types", []):
            _require(rt in known_types, "manifest/unknown_record_type",
                     f"Slice {i}: record type {rt!r} not in manifest.",
                     slice_index=i, value=rt,
                     hint="See manifest.json record_types keys for valid values.")

        for at in sl.get("workouts", {}).get("activity_types", []):
            _require(at in known_workout_types, "manifest/unknown_activity_type",
                     f"Slice {i}: activity type {at!r} not in manifest.",
                     slice_index=i, value=at,
                     hint="See manifest.json workouts.by_type for valid values.")

        # raw extraction needs export.xml — caller must have provided it
        # (check happens at extract time when we know the path)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date | None:
    try:
        return date.fromisoformat(s[:10])
    except (TypeError, ValueError):
        return None


def _in_range(d: date | None, dr: dict[str, str] | None) -> bool:
    if dr is None:
        return True
    if d is None:
        return False
    return date.fromisoformat(dr["start"]) <= d <= date.fromisoformat(dr["end"])


def _short_type(t: str) -> str:
    for prefix in ("HKQuantityTypeIdentifier", "HKCategoryTypeIdentifier", "HKDataType"):
        if t.startswith(prefix):
            return t[len(prefix):]
    return t


# ---------------------------------------------------------------------------
# Extractors per aggregation_level
# ---------------------------------------------------------------------------

def extract_monthly(slice_spec: dict[str, Any], monthly: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull from monthly_aggregates.json. Filters to slice's record_types and date_range."""
    out: list[dict[str, Any]] = []
    types = slice_spec.get("record_types", [])
    dr = slice_spec.get("date_range")

    for t in types:
        per_month = monthly.get(t, {})
        for month_key in sorted(per_month.keys()):
            # month_key is "YYYY-MM"; treat first of month for range comparison
            try:
                d = date.fromisoformat(month_key + "-01")
            except ValueError:
                continue
            if not _in_range(d, dr):
                continue
            row = {"type": t, "month": month_key, **per_month[month_key]}
            out.append(row)
    return out


def extract_daily(slice_spec: dict[str, Any],
                  daily_path: Path) -> list[dict[str, Any]]:
    """Stream daily_aggregates.jsonl and filter."""
    out: list[dict[str, Any]] = []
    types = set(slice_spec.get("record_types", []))
    dr = slice_spec.get("date_range")

    with daily_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["type"] not in types:
                continue
            d = _parse_date(row["date"])
            if not _in_range(d, dr):
                continue
            out.append(row)
    return out


def extract_raw(slice_spec: dict[str, Any], export_xml: Path) -> list[dict[str, Any]]:
    """Stream export.xml and filter to matching records.

    Heavy operation. Only invoked when aggregation_level='raw'. Honors max_rows.
    """
    out: list[dict[str, Any]] = []
    types = set(slice_spec.get("record_types", []))
    dr = slice_spec.get("date_range")
    cap = slice_spec.get("max_rows", MAX_ROWS_CEILING)

    # Match against the SHORT form (manifest uses short names) by stripping
    # HK prefixes from the XML's full type strings on the fly.
    context = iterparse(str(export_xml), events=("end",))
    for _ev, elem in context:
        if elem.tag == "Record":
            t_full = elem.get("type", "")
            t_short = _short_type(t_full)
            if t_short in types:
                start = elem.get("startDate", "")
                d = _parse_date(start)
                if _in_range(d, dr):
                    out.append({
                        "type": t_short,
                        "start": start,
                        "end": elem.get("endDate", ""),
                        "value": elem.get("value"),
                        "unit": elem.get("unit"),
                        "source": elem.get("sourceName"),
                    })
                    if len(out) >= cap:
                        elem.clear()
                        return out
            elem.clear()
        elif elem.tag in ("Workout", "Correlation", "ActivitySummary",
                          "ClinicalRecord", "Audiogram", "VisionPrescription"):
            elem.clear()
    return out


def extract_workouts(slice_spec: dict[str, Any],
                     workouts_path: Path,
                     routes_dir: Path | None) -> list[dict[str, Any]]:
    """Stream workouts.jsonl and filter."""
    if "workouts" not in slice_spec:
        return []

    wf = slice_spec["workouts"]
    activity_types = set(wf.get("activity_types", []))  # empty = all
    include_routes = wf.get("include_routes", False)
    dr = slice_spec.get("date_range")
    cap = slice_spec.get("max_rows", MAX_ROWS_CEILING)

    # Routes inventory: filename → full path. Built once if needed.
    routes_index: dict[str, str] = {}
    if include_routes and routes_dir and routes_dir.exists():
        for p in routes_dir.iterdir():
            if p.is_file():
                routes_index[p.name] = str(p)

    out: list[dict[str, Any]] = []
    with workouts_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if activity_types and row["type"] not in activity_types:
                continue
            d = _parse_date(row.get("start", ""))
            if not _in_range(d, dr):
                continue
            if include_routes and routes_index:
                # Apple's routes are typically named route_YYYY-MM-DD_HH.MM.SSm.gpx
                start = row.get("start", "")[:10]
                hits = [v for k, v in routes_index.items() if start in k]
                row = {**row, "route_files": hits}
            out.append(row)
            if len(out) >= cap:
                break
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def execute_plan(plan: dict[str, Any],
                 index_dir: Path,
                 export_xml: Path | None,
                 routes_dir: Path | None) -> dict[str, Any]:
    if plan["kind"] == "question":
        return plan  # passthrough

    manifest_path = index_dir / "manifest.json"
    monthly_path = index_dir / "monthly_aggregates.json"
    daily_path = index_dir / "daily_aggregates.jsonl"
    workouts_path = index_dir / "workouts.jsonl"

    for p in (manifest_path, monthly_path, daily_path, workouts_path):
        if not p.exists():
            raise ExtractError("io/missing_index", f"Index file missing: {p}",
                               path=str(p))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_against_manifest(plan, manifest)

    monthly = json.loads(monthly_path.read_text(encoding="utf-8"))

    out_slices: list[dict[str, Any]] = []
    for i, sl in enumerate(plan["slices"]):
        agg = sl["aggregation_level"]
        cap = sl.get("max_rows", MAX_ROWS_CEILING)

        records: list[dict[str, Any]] = []
        if sl.get("record_types"):
            if agg == "monthly":
                records = extract_monthly(sl, monthly)
            elif agg == "daily":
                records = extract_daily(sl, daily_path)
            elif agg == "raw":
                if export_xml is None:
                    raise ExtractError("io/raw_needs_export",
                                       f"Slice {i} uses aggregation_level='raw' but "
                                       "--export was not provided.", slice_index=i)
                records = extract_raw(sl, export_xml)

        truncated_records = len(records) >= cap
        records = records[:cap]

        workouts = extract_workouts(sl, workouts_path, routes_dir) \
            if "workouts" in sl else None

        out_slices.append({
            "label": sl.get("label"),
            "record_types": sl.get("record_types", []),
            "date_range": sl.get("date_range"),
            "aggregation_level": agg,
            "records": records,
            "workouts": workouts,
            "truncated": truncated_records,
            "row_count": len(records) + (len(workouts) if workouts else 0),
        })

    return {
        "kind": "slice_bundle",
        "rationale": plan.get("rationale"),
        "slices": out_slices,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", required=True,
                   help="Path to routing plan JSON, or '-' for stdin.")
    p.add_argument("--index-dir", type=Path, default=Path("index"),
                   help="Directory with manifest.json etc. Default: ./index")
    p.add_argument("--export", type=Path, default=None,
                   help="Path to export.xml; required only for raw slices.")
    p.add_argument("--routes", type=Path, default=None,
                   help="Path to workout-routes/; required only when "
                        "include_routes=true.")
    args = p.parse_args()

    # Load plan
    if args.plan == "-":
        raw = sys.stdin.read()
    else:
        plan_path = Path(args.plan).expanduser().resolve()
        if not plan_path.exists():
            print(json.dumps({"error": "io/missing_plan",
                              "message": f"Plan file not found: {plan_path}"}),
                  file=sys.stderr)
            return 2
        raw = plan_path.read_text(encoding="utf-8")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": "schema/json_decode",
                          "message": f"Plan is not valid JSON: {e}"}),
              file=sys.stderr)
        return 2

    try:
        plan = validate_plan(obj)
        result = execute_plan(plan, args.index_dir.expanduser().resolve(),
                              args.export.expanduser().resolve() if args.export else None,
                              args.routes.expanduser().resolve() if args.routes else None)
    except ExtractError as e:
        print(json.dumps(e.to_dict()), file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
