#!/usr/bin/env python3
"""
extract.py — Deterministic slice extractor for Experiment 005 Phase 0.

Reads a Dicer routing plan (validated against routing_plan.schema.json) and
produces the slice the Describer ingests. No LLM in this step. Pure parse + IO.

Aggregation semantics
---------------------
aggregation_level affects BOTH record_types and workouts:

  - record_types:
      "monthly" → reads from monthly_aggregates.json
      "daily"   → reads from daily_aggregates.jsonl
      "raw"     → streams matching records from export.xml

  - workouts:
      "monthly" → groups by (activity_type, YYYY-MM); counts + duration/distance/energy sums
      "daily"   → groups by (activity_type, YYYY-MM-DD); same aggregations
      "raw"     → returns each workout row as-is from workouts.jsonl

Cliff-aware row caps (per ADR-002)
----------------------------------
Per Incident 003 (April 28, 2026), prefill goes super-quadratic between
25K and 35K on-the-wire tokens on the local stack (gemma4:26b on miktam02).
Slices that push the on-the-wire prompt into that zone cause prefill stalls
of many minutes and can wedge the Ollama runner.

To keep the architecture below the cliff, the extractor enforces caps that
bound any single slice's contribution to the bundle:

  - AGG_WORKOUT_ROW_CAP = 200    # ~15 activity types × ~13 months max
  - AGG_RECORD_DAILY_CAP = 1500  # ~4 years of one type, or proportional
  - RAW_DEFAULT_CAP via existing max_rows mechanism (already in schema)

These are tighter than the schema's MAX_ROWS_CEILING (5000) — they are the
*operational* ceilings for Phase 0, applied silently when a slice would
otherwise exceed them. The slice's `truncated` flag is set so the
Describer knows.

Per ADR-001: the extractor is the enforcement boundary. Failure is data.
Per ADR-002: cliff awareness is now part of that enforcement.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
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
# Schema validation
# ---------------------------------------------------------------------------

VALID_AGG = {"monthly", "daily", "raw"}
MAX_SLICES = 8
MAX_ROWS_CEILING = 5000  # absolute schema ceiling

# Cliff-aware operational caps (ADR-002). Applied silently when the
# Dicer's plan would otherwise exceed them.
AGG_WORKOUT_ROW_CAP = 200
AGG_RECORD_DAILY_CAP = 1500


def _require(cond: bool, code: str, msg: str, **details: Any) -> None:
    if not cond:
        raise ExtractError(code, msg, **details)


def validate_plan(obj: Any) -> dict[str, Any]:
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
        if "reason" in obj:
            _require(isinstance(obj["reason"], str) and len(obj["reason"]) <= 300,
                     "schema/bad_reason", "reason must be a string ≤300 chars.")
        _extra = set(obj) - {"kind", "question", "reason"}
        _require(not _extra, "schema/extra_fields",
                 f"Unexpected fields on question: {sorted(_extra)}")
        return obj

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


def validate_against_manifest(plan: dict[str, Any], manifest: dict[str, Any]) -> None:
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


# ---------------------------------------------------------------------------
# Helpers
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


def _safe_add(acc: float | None, v: Any) -> float | None:
    if v is None:
        return acc
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return acc
    if acc is None:
        return fv
    return acc + fv


# ---------------------------------------------------------------------------
# Records extractors
# ---------------------------------------------------------------------------

def extract_records_monthly(slice_spec: dict[str, Any],
                            monthly: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    types = slice_spec.get("record_types", [])
    dr = slice_spec.get("date_range")

    for t in types:
        per_month = monthly.get(t, {})
        for month_key in sorted(per_month.keys()):
            try:
                d = date.fromisoformat(month_key + "-01")
            except ValueError:
                continue
            if not _in_range(d, dr):
                continue
            row = {"type": t, "month": month_key, **per_month[month_key]}
            out.append(row)
    return out


def extract_records_daily(slice_spec: dict[str, Any],
                          daily_path: Path) -> list[dict[str, Any]]:
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


def extract_records_raw(slice_spec: dict[str, Any],
                        export_xml: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    types = set(slice_spec.get("record_types", []))
    dr = slice_spec.get("date_range")
    cap = slice_spec.get("max_rows", MAX_ROWS_CEILING)

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


# ---------------------------------------------------------------------------
# Workouts extractors
# ---------------------------------------------------------------------------

def _iter_workouts(workouts_path: Path,
                   activity_types: set[str],
                   dr: dict[str, str] | None):
    with workouts_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if activity_types and row["type"] not in activity_types:
                continue
            d = _parse_date(row.get("start", ""))
            if not _in_range(d, dr):
                continue
            yield row, d


def extract_workouts_aggregated(slice_spec: dict[str, Any],
                                workouts_path: Path,
                                granularity: str) -> list[dict[str, Any]]:
    if "workouts" not in slice_spec:
        return []
    wf = slice_spec["workouts"]
    activity_types = set(wf.get("activity_types", []))
    dr = slice_spec.get("date_range")

    bucket: dict[tuple[str, str], dict[str, Any]] = {}

    for row, d in _iter_workouts(workouts_path, activity_types, dr):
        if d is None:
            continue
        period = d.strftime("%Y-%m") if granularity == "monthly" else d.isoformat()
        key = (row["type"], period)
        agg = bucket.get(key)
        if agg is None:
            agg = {
                "activity_type": row["type"],
                "period": period,
                "count": 0,
                "duration_total": None,
                "distance_total": None,
                "energy_total": None,
                "duration_unit": row.get("duration_unit"),
                "distance_unit": row.get("distance_unit"),
                "energy_unit": row.get("energy_unit"),
            }
            bucket[key] = agg
        agg["count"] += 1
        agg["duration_total"] = _safe_add(agg["duration_total"], row.get("duration"))
        agg["distance_total"] = _safe_add(agg["distance_total"], row.get("distance"))
        agg["energy_total"] = _safe_add(agg["energy_total"], row.get("energy"))

    out: list[dict[str, Any]] = []
    for (_at, _period), agg in sorted(bucket.items()):
        for k in ("duration_total", "distance_total", "energy_total"):
            if isinstance(agg[k], float):
                agg[k] = round(agg[k], 3)
        out.append(agg)
    return out


def extract_workouts_raw(slice_spec: dict[str, Any],
                         workouts_path: Path,
                         routes_dir: Path | None) -> list[dict[str, Any]]:
    if "workouts" not in slice_spec:
        return []
    wf = slice_spec["workouts"]
    activity_types = set(wf.get("activity_types", []))
    include_routes = wf.get("include_routes", False)
    dr = slice_spec.get("date_range")
    cap = slice_spec.get("max_rows", MAX_ROWS_CEILING)

    routes_index: dict[str, str] = {}
    if include_routes and routes_dir and routes_dir.exists():
        for p in routes_dir.iterdir():
            if p.is_file():
                routes_index[p.name] = str(p)

    out: list[dict[str, Any]] = []
    for row, _d in _iter_workouts(workouts_path, activity_types, dr):
        if include_routes and routes_index:
            start = row.get("start", "")[:10]
            hits = [v for k, v in routes_index.items() if start in k]
            row = {**row, "route_files": hits}
        out.append(row)
        if len(out) >= cap:
            break
    return out


# ---------------------------------------------------------------------------
# Cliff-aware downsamplers (ADR-002)
# ---------------------------------------------------------------------------

def _coarsen_workout_aggregates_to_yearly(rows: list[dict[str, Any]]
                                          ) -> list[dict[str, Any]]:
    """If monthly workout aggregation is too dense, fall back to yearly.

    Groups by (activity_type, YYYY) and re-sums.
    """
    if not rows:
        return rows
    bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        period = r["period"][:4]  # YYYY
        key = (r["activity_type"], period)
        agg = bucket.get(key)
        if agg is None:
            agg = {
                "activity_type": r["activity_type"],
                "period": period,
                "count": 0,
                "duration_total": None,
                "distance_total": None,
                "energy_total": None,
                "duration_unit": r.get("duration_unit"),
                "distance_unit": r.get("distance_unit"),
                "energy_unit": r.get("energy_unit"),
            }
            bucket[key] = agg
        agg["count"] += r.get("count", 0)
        agg["duration_total"] = _safe_add(agg["duration_total"], r.get("duration_total"))
        agg["distance_total"] = _safe_add(agg["distance_total"], r.get("distance_total"))
        agg["energy_total"] = _safe_add(agg["energy_total"], r.get("energy_total"))
    out = []
    for (_at, _y), agg in sorted(bucket.items()):
        for k in ("duration_total", "distance_total", "energy_total"):
            if isinstance(agg[k], float):
                agg[k] = round(agg[k], 3)
        out.append(agg)
    return out


def _apply_cliff_caps(slice_out: dict[str, Any]) -> bool:
    """Apply ADR-002 caps to a slice's records and workouts. Returns True if
    anything was truncated or coarsened.
    """
    truncated = False

    # Records: cap daily at AGG_RECORD_DAILY_CAP. Monthly is naturally bounded
    # (~96 months × few types) so usually no cap needed; if it does exceed,
    # fall through to the same cap.
    records = slice_out.get("records") or []
    if len(records) > AGG_RECORD_DAILY_CAP:
        slice_out["records"] = records[:AGG_RECORD_DAILY_CAP]
        truncated = True

    # Workouts: aggregated form. If still over cap after monthly grouping,
    # coarsen to yearly. If still over, hard-truncate.
    workouts = slice_out.get("workouts")
    agg_level = slice_out.get("aggregation_level")
    if workouts and agg_level in ("monthly", "daily"):
        if len(workouts) > AGG_WORKOUT_ROW_CAP:
            coarsened = _coarsen_workout_aggregates_to_yearly(workouts)
            if len(coarsened) <= AGG_WORKOUT_ROW_CAP:
                slice_out["workouts"] = coarsened
                slice_out["_workout_coarsened_to"] = "yearly"
            else:
                slice_out["workouts"] = coarsened[:AGG_WORKOUT_ROW_CAP]
                slice_out["_workout_coarsened_to"] = "yearly+truncated"
            truncated = True

    if truncated:
        slice_out["row_count"] = (
            len(slice_out.get("records") or [])
            + len(slice_out.get("workouts") or [])
        )
    return truncated


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def execute_plan(plan: dict[str, Any],
                 index_dir: Path,
                 export_xml: Path | None,
                 routes_dir: Path | None) -> dict[str, Any]:
    if plan["kind"] == "question":
        return plan

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
                records = extract_records_monthly(sl, monthly)
            elif agg == "daily":
                records = extract_records_daily(sl, daily_path)
            elif agg == "raw":
                if export_xml is None:
                    raise ExtractError("io/raw_needs_export",
                                       f"Slice {i} uses aggregation_level='raw' but "
                                       "--export was not provided.", slice_index=i)
                records = extract_records_raw(sl, export_xml)
        truncated_records = len(records) >= cap
        records = records[:cap]

        workouts: list[dict[str, Any]] | None = None
        truncated_workouts = False
        if "workouts" in sl:
            if agg == "monthly":
                workouts = extract_workouts_aggregated(sl, workouts_path, "monthly")
            elif agg == "daily":
                workouts = extract_workouts_aggregated(sl, workouts_path, "daily")
            else:
                workouts = extract_workouts_raw(sl, workouts_path, routes_dir)
            truncated_workouts = len(workouts) >= cap
            workouts = workouts[:cap]

        slice_out = {
            "label": sl.get("label"),
            "record_types": sl.get("record_types", []),
            "date_range": sl.get("date_range"),
            "aggregation_level": agg,
            "records": records,
            "workouts": workouts,
            "truncated": truncated_records or truncated_workouts,
            "row_count": len(records) + (len(workouts) if workouts else 0),
        }

        # Cliff-aware caps (ADR-002): applied after extraction, before output.
        if _apply_cliff_caps(slice_out):
            slice_out["truncated"] = True

        out_slices.append(slice_out)

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
    p.add_argument("--index-dir", type=Path, default=Path("index"))
    p.add_argument("--export", type=Path, default=None,
                   help="Path to export.xml; required only for raw record_types slices.")
    p.add_argument("--routes", type=Path, default=None,
                   help="Path to workout-routes/; required only when include_routes=true.")
    args = p.parse_args()

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
