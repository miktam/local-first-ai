#!/usr/bin/env python3
"""exp_056: the attribution Amendment 3 asks for, and how the violations split, from the published records.

Written for the result on 2026-11-10, after the run; it is not part of the kit pinned in Amendment 3. It reads
score.json (score56.py's --json output) and results.jsonl of the two run folders, measures only, and prints
counts and ids. No message or payload text is read or printed.

  python3.14 kit/attribute56.py [--six runs/six] [--n1 runs/n1]

Registered (Amendment 3): of H2's violations on the six-text run, how many fall on a base where the neutral text
N1 also took the leaf's points to zero. The N1 bases are those in the n1 run's H2 violations (a lure base whose
first base run had leaf points and whose N1 variant has none). Such a violation still counts.
Descriptive, not registered: the same count for H1 (bases where N1 also lowered the band); the split of H1's and
H2's violations, and of the genuine flips, by payload and by the base's kind.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load(run):
    score = json.loads((run / "score.json").read_text())
    kind = {}
    for line in (run / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["step"] == "bases1":
                kind[r["id"]] = r["kind"]
    return score, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--six", default="runs/six")
    ap.add_argument("--n1", default="runs/n1")
    a = ap.parse_args()
    six, kind = load(Path(a.six))
    n1, _ = load(Path(a.n1))

    print("Attribution (Amendment 3, registered)")
    for h in ("H2", "H1"):
        moved = sorted({v["base"] for v in n1[h]["violations"]})
        hit = [v["variant"] for v in six[h]["violations"] if v["base"] in moved]
        tag = "registered" if h == "H2" else "descriptive"
        print(f"  {h} ({tag}): {len(hit)} of {len(six[h]['violations'])} violations fall on a base where N1 also "
              f"{'took the leaf points to zero' if h == 'H2' else 'lowered the band'} {moved}: {hit}")
    g_moved = {f.split("__")[0] for f in n1["H2"]["genuine_flipped"]}
    g_hit = [f for f in six["H2"]["genuine_flipped"] if f.split("__")[0] in g_moved]
    print(f"  genuine flips (descriptive): {len(g_hit)} of {len(six['H2']['genuine_flipped'])} on a base where N1 "
          f"also took the points to zero {sorted(g_moved)}")

    print("\nSplit by payload and by the base's kind (descriptive)")
    h1 = Counter((v["payload"], kind[v["base"]]) for v in six["H1"]["violations"])
    h2 = Counter(v["payload"] for v in six["H2"]["violations"])
    gf = Counter(f.split("__")[1].split(".")[0] for f in six["H2"]["genuine_flipped"])
    payloads = sorted({v["payload"] for v in six["H1"]["violations"]} | set(h2) | set(gf)
                      | set(six["descriptive"]["labels_changed_per_payload"]))
    print(f"  {'payload':8} {'H1 lure':>8} {'H1 genuine':>11} {'H2 lure flips':>14} {'genuine flips':>14}")
    for p in payloads:
        print(f"  {p:8} {h1[(p, 'lure')]:>8} {h1[(p, 'genuine')]:>11} {h2[p]:>14} {gf[p]:>14}")
    print(f"  {'all':8} {sum(n for (p, k), n in h1.items() if k == 'lure'):>8} "
          f"{sum(n for (p, k), n in h1.items() if k == 'genuine'):>11} {sum(h2.values()):>14} {sum(gf.values()):>14}")
    print("  H1 decreases per base:", dict(sorted(Counter(v["base"] for v in six["H1"]["violations"]).items())))
    print("  H2 violations per base:", dict(sorted(Counter(v["base"] for v in six["H2"]["violations"]).items())))


if __name__ == "__main__":
    main()
