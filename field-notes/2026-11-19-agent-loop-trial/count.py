#!/usr/bin/env python3
"""Recompute every count in README.md from the three data files beside it.

Standard library only; runs under Python 3.9 and later.
    python3 count.py
"""
import csv
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    with open(HERE / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def table(title, counter, order=None):
    print(title)
    keys = order or [k for k, _ in counter.most_common()]
    for k in keys:
        print("  %-28s %d" % (k, counter.get(k, 0)))


runs, defects, misses = rows("runs.tsv"), rows("defects.tsv"), rows("check_misses.tsv")

print("runs %d, closed %d" % (len(runs), len(runs)))
table("started by", Counter(r["started_by"] for r in runs))
table("reviewer's first answer", Counter(r["reviewer_first_answer"] for r in runs))
print("first-review findings: BLOCKS %d, SHOULD %d" % (
    sum(int(r["blocks"]) for r in runs), sum(int(r["should"]) for r in runs)))
print("runs with a fix round %d" % sum(r["fix_round"] == "yes" for r in runs))
print("times the owner stepped in %d" % sum(int(r["owner_stepped_in"]) for r in runs))
mins = [int(r["minutes"]) for r in runs]
print("minutes: total %d, median %d, total without run 01 %d" % (
    sum(mins), statistics.median(mins), sum(mins) - mins[0]))
print()

print("defect rows %d" % len(defects))
table("caught by", Counter(d["caught_by"] for d in defects),
      ["reviewer", "both", "worker", "dispatcher", "frozen check"])
table("kind", Counter(d["kind"] for d in defects),
      ["code", "contract", "fact", "check", "process", "other"])
own = sum(d["kind"] in ("contract", "check") for d in defects)
print("the dispatcher's own (contract + check) %d of %d" % (own, len(defects)))
table("where it was headed if missed", Counter(d["if_missed"] for d in defects),
      ["public", "public (reached it)", "recipient", "none"])
by = {}
for d in defects:
    by.setdefault(d["kind"], Counter())[d["caught_by"]] += 1
print("kind by catcher")
for k in ["code", "contract", "fact", "check", "process", "other"]:
    print("  %-10s %s" % (k, ", ".join("%s %d" % kv for kv in by[k].most_common())))
print()

print("frozen checks that were wrong %d, in %d of %d runs" % (
    len(misses), len({m["run"] for m in misses}), len(runs)))
table("how", Counter(m["type"] for m in misses))
