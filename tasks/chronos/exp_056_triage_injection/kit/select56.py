#!/usr/bin/env python3
"""exp_056 base selection: the registered seeded draw of 24 base messages from exp_055's corpus.

Non-attack scaffolding. It chooses by exp_055's MANIFEST.json and its P and R results only, and copies each chosen
message by id, byte for byte (read_bytes -> write_bytes); it never prints or parses a message's text.

  python3.14 kit/select56.py --out DIR [--corpus DIR] [--p-results F] [--r-results F]

The procedure (Amendment 2's categories, the draw pinned by run 2026-11-10-10's brief): one random.Random(56); the
eight categories in Amendment 2's order, lures then genuine; for each, rng.sample(sorted(pool), k) over the ids
that meet it, minus the seven excluded ids (registered domains, Amendment 2) and those already chosen.

  L1  email lure, leaf points > 0, P suspicious, R benign      6
  L2  email lure, leaf points > 0, P suspicious, R suspicious  3
  L3  email lure, leaf points > 0, P likely-scam               1
  L4  chat lure (sms or whatsapp), leaf points > 0             2
  G1  genuine email, P suspicious, R benign                    4
  G2  genuine email, leaf points > 0, P benign                 4
  G3  genuine email, leaf points 0                             2
  G4  genuine chat                                             2

Writes DIR/<file> for each of the 24 and DIR/bases.json: a list, in draw order, of
{file, id, kind: lure|genuine, channel, category, sha256}. DIR must be new or empty.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

E055 = Path.home() / ".openclaw/workspace/tasks/exp055"
DEFAULT_CORPUS = E055 / "data"
DEFAULT_P = E055 / "runs/20261025T095614Z/P/results.jsonl"
DEFAULT_R = E055 / "runs/20261025T095614Z/R/results.jsonl"
SEED = 56
EXCLUDED = frozenset({"X011", "X067", "X085", "X114", "X148", "X161", "X174"})   # Amendment 2


def categories(P, R):
    email = lambda m: m["channel"] == "email"
    lure = lambda m: m["kind"] == "lure"
    genuine = lambda m: m["kind"] == "benign"                # exp_055's label for a genuine message
    pts = lambda i: P[i]["persona_points"]
    return [
        ("L1", 6, lambda i, m: lure(m) and email(m) and pts(i) > 0 and P[i]["band"] == "suspicious"
            and R[i]["band"] == "benign"),
        ("L2", 3, lambda i, m: lure(m) and email(m) and pts(i) > 0 and P[i]["band"] == "suspicious"
            and R[i]["band"] == "suspicious"),
        ("L3", 1, lambda i, m: lure(m) and email(m) and pts(i) > 0 and P[i]["band"] == "likely-scam"),
        ("L4", 2, lambda i, m: lure(m) and not email(m) and pts(i) > 0),
        ("G1", 4, lambda i, m: genuine(m) and email(m) and P[i]["band"] == "suspicious" and R[i]["band"] == "benign"),
        ("G2", 4, lambda i, m: genuine(m) and email(m) and pts(i) > 0 and P[i]["band"] == "benign"),
        ("G3", 2, lambda i, m: genuine(m) and email(m) and pts(i) == 0),
        ("G4", 2, lambda i, m: genuine(m) and not email(m)),
    ]


def draw(manifest: dict, P: dict, R: dict):
    """-> ([(id, category)] in draw order, {category: pool size})."""
    rng, chosen, out, pools = random.Random(SEED), set(), [], {}
    for name, k, pred in categories(P, R):
        pool = sorted(i for i, m in manifest.items() if i not in EXCLUDED and i not in chosen and pred(i, m))
        pools[name] = len(pool)
        for i in rng.sample(pool, k):
            chosen.add(i)
            out.append((i, name))
    return out, pools


def results(path: Path) -> dict:
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_056: the registered draw of 24 base messages")
    ap.add_argument("--out", required=True, help="new or empty folder for the 24 messages and bases.json")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="exp_055 data dir (MANIFEST.json, messages/)")
    ap.add_argument("--p-results", default=str(DEFAULT_P), help="exp_055 P arm results.jsonl")
    ap.add_argument("--r-results", default=str(DEFAULT_R), help="exp_055 R arm results.jsonl")
    a = ap.parse_args(argv)
    corpus, out = Path(a.corpus), Path(a.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        print(f"select56: {out}: exists and is not an empty folder; nothing written", file=sys.stderr)
        return 2
    manifest = json.loads((corpus / "MANIFEST.json").read_text(encoding="utf-8"))
    picks, pools = draw(manifest, results(Path(a.p_results)), results(Path(a.r_results)))
    out.mkdir(parents=True, exist_ok=True)
    listing = []
    for i, cat in picks:
        m = manifest[i]
        data = (corpus / "messages" / m["file"]).read_bytes()
        (out / m["file"]).write_bytes(data)
        listing.append({"file": m["file"], "id": i, "kind": "lure" if m["kind"] == "lure" else "genuine",
                        "channel": m["channel"], "category": cat, "sha256": hashlib.sha256(data).hexdigest()})
    (out / "bases.json").write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")
    print("pool sizes:", " ".join(f"{c}={n}" for c, n in pools.items()))
    print("drawn:", " ".join(f"{i}({c})" for i, c in picks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
