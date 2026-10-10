"""exp_038 M0 (DESIGN §8.0): are there verbatim 12-word runs of the course materials (S3) in Andrei's guides (S1, S2)?

Run by Andrei on the mini. No agent opens S3: this script reads it, hashes its 12-word shingles in memory and prints
nothing from it. The output is counts, and S1/S2 file:line ranges whose 12-word shingles also occur in S3. Those are
lines of Andrei's own guide, safe to show. Flagged lines are excluded from every exp_038 writer input and replaced by
"[withheld]" in judge contexts (S1 itself is not edited, so the index hashes stay valid).

Words: Unicode \\w+ runs, lowercased. A shingle is 12 consecutive words, never crossing a file boundary. All-digit
shingles are skipped (article and number lists are not prose).

Usage:  python tools/m0_shingle_check.py [--casasol ~/REPOS/casasol] [--out runs/m0_<UTC>.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

N = 12
WORD = re.compile(r"\w+", re.UNICODE)
S3_LANGS = ("en", "es", "pl")
TEXT_SUFFIXES = {".txt", ".md"}


def words_with_lines(text: str) -> list[tuple[str, int]]:
    out = []
    for ln, line in enumerate(text.splitlines(), 1):
        out.extend((w.lower(), ln) for w in WORD.findall(line))
    return out


def shingles(ws: list[tuple[str, int]]):
    for i in range(len(ws) - N + 1):
        gram = [w for w, _ in ws[i:i + N]]
        if all(g.isdigit() for g in gram):
            continue
        yield hashlib.sha256(" ".join(gram).encode("utf-8")).digest(), ws[i][1], ws[i + N - 1][1]


def s3_index(root: Path) -> tuple[set[bytes], dict]:
    seen: set[bytes] = set()
    stats = {"files": 0, "words": 0}
    for lang in S3_LANGS:
        d = root / lang
        if not d.is_dir():
            raise SystemExit(f"S3 directory missing: {d}")
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
                ws = words_with_lines(p.read_text(encoding="utf-8", errors="replace"))
                stats["files"] += 1
                stats["words"] += len(ws)
                seen.update(h for h, _, _ in shingles(ws))
    stats["shingles"] = len(seen)
    return seen, stats


def check(guide_files: list[Path], s3: set[bytes], rel_to: Path) -> dict:
    out = {}
    for p in guide_files:
        ws = words_with_lines(p.read_text(encoding="utf-8"))
        hits = [(a, b) for h, a, b in shingles(ws) if h in s3]
        ranges: list[list[int]] = []
        for a, b in sorted(hits):
            if ranges and a <= ranges[-1][1] + 1:
                ranges[-1][1] = max(ranges[-1][1], b)
            else:
                ranges.append([a, b])
        out[str(p.relative_to(rel_to))] = {"words": len(ws), "shingle_hits": len(hits), "line_ranges": ranges}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casasol", type=Path, default=Path.home() / "REPOS/casasol")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    root = a.casasol
    guide = [root / "casasol.ai/guide/src" / lang / f"study_guide_modulo{m}.md" for lang in ("en", "pl") for m in range(1, 7)]
    missing = [str(p) for p in guide if not p.is_file()]
    if missing:
        raise SystemExit(f"guide files missing: {missing}")
    s3, s3_stats = s3_index(root / "tasks/coapi/materials/text")
    res = {
        "check": "exp_038 M0: 12-word shingles of S1/S2 found in S3 (en, es, pl)",
        "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "guide_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in guide},
        "s3_stats": s3_stats,
        "files": check(guide, s3, root),
    }
    res["total_shingle_hits"] = sum(f["shingle_hits"] for f in res["files"].values())
    res["flagged_line_ranges"] = sum(len(f["line_ranges"]) for f in res["files"].values())
    out = a.out or Path(__file__).resolve().parents[1] / "runs" / f"m0_{res['utc'].replace('-', '').replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"total_shingle_hits": res["total_shingle_hits"], "flagged_line_ranges": res["flagged_line_ranges"],
                      "s3_files": s3_stats["files"], "per_file": {k: v["shingle_hits"] for k, v in res["files"].items()},
                      "out": str(out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
