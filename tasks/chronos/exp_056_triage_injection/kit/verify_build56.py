#!/usr/bin/env python3
"""exp_056: check a build of the variants by its bytes, without showing any of them.

Written by the chief-of-staff session, 2026-11-10. It reads the base messages, the six payload files and
the built variants as bytes, compares them, and prints counts, hashes' agreement and yes/no. It never
prints, logs or returns message or payload text: where a separator is not pure whitespace it says so
and does not show it.

  python3.14 verify_build56.py --build BUILD --bases BASES --payloads PAYLOADS

BUILD     build56.py's output: the variants and manifest.json
BASES     select56.py's output: the 24 messages and bases.json
PAYLOADS  the folder with the texts to attach: A1.txt … A6.txt, or N1.txt for the neutral control

Exit 0 when every check passes, 1 otherwise.
"""
import argparse
import collections
import email
import hashlib
import json
import re
import sys
from pathlib import Path

REGISTRATION = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_056_triage_injection/HYPOTHESIS.md"
LABEL = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,39}$")
NAME = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,79}$")
WS = b" \t\r\n"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rows_of(raw):
    rows = raw["variants"] if isinstance(raw, dict) and "variants" in raw else raw
    out = []
    if isinstance(rows, list):
        for r in rows:
            out.append((r.get("variant"), r.get("payload"), r.get("base")))
    elif isinstance(rows, dict):
        for v, x in rows.items():
            if isinstance(x, dict):
                out.append((v, x.get("payload"), x.get("base")))
            elif isinstance(x, (list, tuple)) and len(x) >= 2:
                out.append((v, x[0], x[1]))
            else:
                out.append((v, None, None))
    return out


def shape(variant, base, payload):
    """How the variant is made of its base and its payload -> (name, separator description)."""
    b, p = base.rstrip(WS), payload.strip(WS)
    if variant.startswith(b) and variant.rstrip(WS).endswith(p):
        sep = variant.rstrip(WS)[len(b):len(variant.rstrip(WS)) - len(p)]
        kind = "base, then payload"
    elif variant.lstrip(WS).startswith(p) and variant.rstrip(WS).endswith(b):
        v = variant.lstrip(WS)
        sep = v[len(p):len(v.rstrip(WS)) - len(b)]
        kind = "payload, then base"
    else:
        return "neither order", "-"
    if sep.strip(WS):
        return kind, f"{len(sep)} bytes, not only whitespace (not shown)"
    return kind, repr(sep.decode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--bases", required=True)
    ap.add_argument("--payloads", required=True)
    a = ap.parse_args()
    build, bases_dir, pay_dir = Path(a.build), Path(a.bases), Path(a.payloads)
    ok = True

    def line(passed, text):
        nonlocal ok
        ok = ok and bool(passed)
        print(("ok   " if passed else "FAIL ") + text)

    # the bases
    listing = json.loads((bases_dir / "bases.json").read_text())
    base_bytes = {}
    for b in listing:
        data = (bases_dir / b["file"]).read_bytes()
        base_bytes[b["file"]] = data
        if sha(data) != b["sha256"]:
            line(False, f"base {b['id']}: bytes do not match bases.json")
    kinds = collections.Counter(b["kind"] for b in listing)
    line(len(listing) == 24 and kinds == {"lure": 12, "genuine": 12},
         f"bases: {len(listing)} listed, {kinds.get('lure', 0)} lures and {kinds.get('genuine', 0)} genuine, bytes as listed")
    multipart = sum(1 for b in listing if b["file"].endswith(".eml")
                    and email.message_from_bytes(base_bytes[b["file"]]).is_multipart())
    print(f"     e-mail bases with more than one part: {multipart} (text added at the end of such a file lands after the body)")
    by_id = {b["id"]: b["file"] for b in listing}

    # the payloads
    reg = REGISTRATION.read_text()
    pay = {p.name: p.read_bytes() for p in sorted(pay_dir.glob("*.txt"))}
    frozen = bool(pay) and all(f"| {n} | {sha(d)} |" in reg for n, d in pay.items())
    line(len(pay) in (1, 6) and frozen, f"texts to attach: {len(pay)} files, all matching the hashes frozen in the registration: {frozen}")

    # the manifest
    rows = rows_of(json.loads((build / "manifest.json").read_text()))
    plain = all(isinstance(v, str) and NAME.match(v) and isinstance(p, str) and LABEL.match(p)
                and isinstance(b, str) and NAME.match(b) for v, p, b in rows)
    line(plain, f"manifest: {len(rows)} rows, every variant name, payload label and base name a plain label: {plain}")
    if not plain:
        print("     stopping here: the manifest holds values that are not plain labels (not shown)")
        return 1
    labels = sorted({p for _, p, _ in rows})
    per_base = collections.defaultdict(set)
    for _, p, b in rows:
        per_base[by_id.get(b, b)].add(p)
    even = len(per_base) == 24 and all(s == set(labels) for s in per_base.values()) and set(per_base) == set(base_bytes)
    line(len(rows) == 24 * len(pay) and len(labels) == len(pay) and even,
         f"manifest: {len(per_base)} bases × {len(labels)} payload labels = {len(rows)} variants; every listed base has every label: {even}")
    print(f"     payload labels: {', '.join(labels)}")

    def payload_bytes(label):
        for n, d in pay.items():
            if label in (n, n.rsplit('.', 1)[0]):
                return d
        return None

    # the variants
    shapes, hashes, missing, unknown = collections.Counter(), {}, 0, 0
    for v, p, b in rows:
        f = build / v
        if not f.is_file() or f.is_symlink():
            missing += 1
            continue
        data = f.read_bytes()
        hashes[v] = sha(data)
        base, pb = base_bytes.get(by_id.get(b, b)), payload_bytes(p)
        if base is None or pb is None:
            unknown += 1
            continue
        shapes[shape(data, base, pb)] += 1
    line(missing == 0, f"variants: {len(hashes)} files present, {missing} missing or links")
    line(unknown == 0, f"variants: {unknown} whose base or payload label could not be matched to a file")
    distinct = len(set(hashes.values())) == len(hashes)
    not_base = not (set(hashes.values()) & {sha(d) for d in base_bytes.values()})
    line(distinct and not_base, f"variants: all different from each other: {distinct}; none equal to a base: {not_base}")
    print("     how each variant is made:")
    for (kind, sep), n in shapes.most_common():
        print(f"       {n:4d} × {kind}; separator {sep}")
    registered = sum(n for (kind, sep), n in shapes.items()
                     if kind == "base, then payload" and "not shown" not in sep and sep.count("\\n") >= 2)
    line(registered == len(rows), f"registered form (the payload after the body, at least one blank line between): {registered} of {len(rows)}")
    print("\n" + ("ALL CHECKS PASS" if ok else "NOT AS REGISTERED: see the FAIL lines"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
