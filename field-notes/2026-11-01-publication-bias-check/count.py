#!/usr/bin/env python3
"""Count how many Chronos index rows record a refuted, failed or null headline outcome.

Reproduces the table in README.md. Reads tasks/chronos/README.md, matches the status
cell only, and prints each hit so the classification can be disputed row by row.

    python3 field-notes/2026-11-01-publication-bias-check/count.py
"""
import re
import sys
from pathlib import Path

ROW = re.compile(r"^\|\s*(Exp|Incident|Watcher)", re.I)

# Negative markers, matched against the status cell. "false refusals", "false spam" and
# "failover" are results about false-positive rates, not failures, so \bfail\b alone is
# too loose: a bare "fail" only counts when attached to a registered hypothesis (H1 fail).
NEGATIVE = re.compile(
    r"refuted|H\d[^.]{0,12}\bfail\b|/H\d fail|gate failed|no gain|not robust"
    r"|inconclusive|not resolved|no gate earns|\bstopped\b|all five fail",
    re.I,
)


def main() -> int:
    index = Path(__file__).resolve().parents[2] / "tasks" / "chronos" / "README.md"
    if not index.is_file():
        print(f"not found: {index}", file=sys.stderr)
        return 1

    rows = [l for l in index.read_text().splitlines() if ROW.match(l)]
    hits = []
    for row in rows:
        cells = row.split("|")
        if len(cells) < 4:
            continue
        entry, status = cells[1].strip(), cells[3].strip()
        if NEGATIVE.search(status):
            hits.append((entry, status))

    aborted = sorted(p.parent.name for p in index.parent.glob("exp_*/aborted"))

    print(f"index rows:                        {len(rows)}")
    print(f"refuted / failed / null outcomes:  {len(hits)}")
    print(f"experiments with an aborted/ dir:  {len(aborted)}  {', '.join(aborted)}")
    print()
    for entry, status in hits:
        print(f"  {entry:12} {status[:96]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
