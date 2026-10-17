#!/usr/bin/env python3
"""exp_040 — writer/checker agreement (BUILD.md §3.3). Code only.

  python3.14 agree.py WRITER_OUT.jsonl CHECK_DIR   # CHECK_DIR/checker_out_*.jsonl → CHECK_DIR/agree.json, kept.jsonl, tiebreak.jsonl

Kept: checker's pick equals the label (numeric: within the item's tol, relative) and not ambiguous.
Dropped: ambiguous or undecidable. Tie-break: a clean pick that differs from the label (a third blind reader decides, 2 of 3).
A twin whose partner is not kept loses its twin link, not its place.
"""
import json
import sys
from collections import Counter
from pathlib import Path


def match(it, pick):
    if it["type"] == "numeric":
        try:
            v, p = float(it["numeric"]["value"]), float(pick)
        except (TypeError, ValueError):
            return False
        return abs(p - v) <= max(abs(v) * it["numeric"].get("tol", 0.005), 1e-9)
    return str(pick) == str(it["label"])


def main(src, chk):
    W = {json.loads(l)["item"]: json.loads(l) for l in Path(src).read_text().splitlines() if l.strip()}
    C = {}
    for p in sorted(chk.glob("checker_out_*.jsonl")):
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                C[r["item"]] = r
    missing = sorted(set(W) - set(C))
    out, kept, tie = Counter(), [], []
    for iid, it in W.items():
        c = C.get(iid)
        if c is None:
            out["missing"] += 1
            continue
        if c.get("pick") == "undecidable":
            out["undecidable"] += 1
        elif c.get("ambiguous"):
            out["ambiguous"] += 1
        elif match(it, c.get("pick")):
            out["agree"] += 1
            kept.append(it)
        else:
            out["disagree"] += 1
            tie.append(it)
    ids = {k["item"] for k in kept}
    for k in kept:
        if k.get("twin_of") and k["twin_of"] not in ids:
            k["twin_of"], k["flip"] = None, None
            out["twin_link_cleared"] += 1
    (chk / "kept.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept))
    (chk / "tiebreak.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in tie))
    rep = {"items": len(W), "checked": len(C), "missing": missing, **out,
           "kept_by_type": dict(Counter(k["type"] for k in kept)), "kept_twins": sum(1 for k in kept if k.get("twin_of")),
           "not_kept": {iid: ("undecidable" if C[iid].get("pick") == "undecidable" else "ambiguous" if C[iid].get("ambiguous") else "disagree")
                        for iid in W if iid in C and iid not in ids}}
    (chk / "agree.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "not_kept"}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
