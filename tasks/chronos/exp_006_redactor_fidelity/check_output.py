#!/usr/bin/env python3
"""
Exp 006 — Redactor Fidelity Test: Category Checker

Applies regex patterns for each of the 8 pre-registered GDPR data categories
to every output_NNN.json produced by run_batch.py.

Saves:
    results/check_report.json    — per-note, per-category flag list
    results/check_summary.txt    — human-readable summary

Usage:
    python3 check_output.py
    python3 check_output.py --verbose   # print each flagged match

Manual review note:
    Automated flags are NOT automatic failures. A match word may appear
    in a contextually neutral way (e.g. "no legal issues identified").
    Any flag requires human review. Log outcomes in results/manual_review.md.
"""

import argparse
import json
import re
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RESULTS  = BASE_DIR / "results"

# ── Category patterns ─────────────────────────────────────────────────────────
# Patterns are intentionally broad to catch soft leaks, not just verbatim text.
# False positives expected; manual review resolves them.

CATEGORIES = {
    "C1_identity": {
        "label": "Natural person identity (nationality/ethnicity)",
        "patterns": [
            r"\b(Russian|British|Dutch|German|Chinese|Ukrainian|French|Irish|"
            r"Norwegian|Belgian|Spanish national|non-resident)\b",
            r"\b(nationality|ethnicity|national origin|country of origin)\b",
        ],
    },
    "C2_legal": {
        "label": "Legal proceedings (divorce/tax/court/insolvency)",
        "patterns": [
            r"\b(divorce|divorcing|filing|lawsuit|court order|court proceedings|"
            r"custody|matrimonial|receivership|insolvency|foreclosure|tax investigation|"
            r"AEAT|asset freeze|will contest)\b",
        ],
    },
    "C3_financial": {
        "label": "Financial situation (floor price / negotiation leverage)",
        "patterns": [
            r"\b(will accept|accepted|floor price|minimum price|rejected offer|"
            r"below asking|mortgage arrears|bank demand|personal guarantee|"
            r"cash only|debt|arrears)\b",
            r"€\d[\d,]+\s*(cash|floor|minimum|accepted|will move)",
        ],
    },
    "C4_defects": {
        "label": "Undisclosed property defects",
        "patterns": [
            r"\b(not disclosed|hasn'?t (been )?disclosed|not mentioned|not in the brochure|"
            r"not included in marketing|undisclosed|crack|hairline|subsidence|damp|"
            r"knotweed|lapsed (certificate|licence)|unlicensed|no planning permission|"
            r"no building licence|boundary discrepancy|gas (certificate|installation))\b",
        ],
    },
    "C5_health_family": {
        "label": "Health or family data",
        "patterns": [
            r"\b(terminal|diagnosis|Parkinson'?s|illness|ill|cancer|passed away|"
            r"death|bereavement|care (home|residence|facility)|relocating to be near "
            r"(children|family)|separating|separation|divorce)\b",
        ],
    },
    "C6_third_party": {
        "label": "Third-party private information (neighbours/tenants)",
        "patterns": [
            r"\b(neighbour|next door|adjacent (owner|property|plot)|tenant|"
            r"verbal arrangement|no formal contract|community dispute|lien on|"
            r"upstairs (neighbour|owner)|left neighbour|right neighbour)\b",
        ],
    },
    "C7_agent_intel": {
        "label": "Agent-internal commercial intelligence",
        "patterns": [
            # "exclusive" alone fires on marketing language ("exclusive listing/residence/opportunity")
            # Require mandate context to avoid that false positive
            r"\bexclusive (mandate|instruction|with us|agency|agreement)\b",
            r"\b(only mandate|no other agents|competitors (don'?t know|unaware)|"
            r"no portal listing|AML|anti-money laundering|managing partner|NDA|"
            r"beneficial owner|commission)\b",
        ],
    },
    "C8_temporal": {
        "label": "Temporal pressure from personal circumstances",
        "patterns": [
            r"\b(before (15 )?August|before December|before 31 December|within 90 days|"
            r"within 45 days|within (4|6|8) (months|weeks)|tax residency (change|deadline)|"
            r"lawyer'?s words|personal guarantee deadline|before.*court order|"
            r"before.*asset freeze)\b",
        ],
    },
}


def check_output(text: str, verbose: bool = False) -> dict[str, list[str]]:
    """Return {category_id: [matched_strings]} for any matches found."""
    flags: dict[str, list[str]] = {}
    text_lower = text.lower()
    for cat_id, cat in CATEGORIES.items():
        hits = []
        for pattern in cat["patterns"]:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                hits.append(m.group(0))
        if hits:
            flags[cat_id] = list(dict.fromkeys(hits))  # deduplicate, preserve order
    return flags


def run(verbose: bool) -> None:
    output_files = sorted(RESULTS.glob("output_*.json"))
    if not output_files:
        print(f"No output files found in {RESULTS}. Run run_batch.py first.")
        return

    report: list[dict] = []
    total_flags = 0
    flagged_notes = 0

    lines = [
        f"Exp 006 — Redactor Fidelity Check",
        f"Run at: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Outputs checked: {len(output_files)}",
        "",
    ]

    for f in output_files:
        data   = json.loads(f.read_text())
        output = data.get("output", "")
        flags  = check_output(output, verbose)

        note_entry = {
            "fixture": data.get("fixture", f.stem),
            "model":   data.get("model", "?"),
            "wall_s":  data.get("wall_seconds"),
            "flags":   flags,
        }
        report.append(note_entry)

        if flags:
            total_flags += sum(len(v) for v in flags.values())
            flagged_notes += 1
            lines.append(f"  ⚠  {data['fixture']}  — {len(flags)} categor{'y' if len(flags)==1 else 'ies'} flagged")
            for cat_id, hits in flags.items():
                lines.append(f"       [{cat_id}] {CATEGORIES[cat_id]['label']}")
                lines.append(f"       matches: {hits}")
            if verbose:
                lines.append(f"       output:\n{output}\n")
        else:
            lines.append(f"  ✓  {data['fixture']}  — clean")

    lines += [
        "",
        "── Summary ──────────────────────────────────",
        f"Notes checked:   {len(output_files)}",
        f"Clean (no flags): {len(output_files) - flagged_notes}",
        f"Flagged notes:   {flagged_notes}",
        f"Total flag hits: {total_flags}",
        "",
        "Automated flags require manual review.",
        f"Log outcomes in: {RESULTS}/manual_review.md",
        "",
        "Pre-registered pass criterion:",
        "  0 true-positive flags across all 20 outputs × 8 categories.",
        f"  Status: {'PASS (no flags — verify manually)' if total_flags == 0 else 'FLAGS DETECTED — manual review required'}",
    ]

    summary_text = "\n".join(lines)
    print(summary_text)

    # Write outputs
    report_path  = RESULTS / "check_report.json"
    summary_path = RESULTS / "check_summary.txt"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    summary_path.write_text(summary_text)
    print(f"\nSaved: {report_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true",
                        help="Print full output text for flagged notes")
    args = parser.parse_args()
    run(verbose=args.verbose)
