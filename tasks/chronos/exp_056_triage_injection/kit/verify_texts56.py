#!/usr/bin/env python3
"""exp_056: check the published texts against what ran, by bytes, from the public tree alone.

Every variant the experiment ran is recorded by its sha256 in results.jsonl. A variant is its base message
without trailing line breaks, a blank line, and one text (Amendments 2 and 3). This script rebuilds each
variant in memory from the public base (exp_055's data/messages) and the public text (payloads/A*.txt,
control/N1.txt), hashes it, and compares with the record. It prints counts and yes or no, never content.

  python3.14 kit/verify_texts56.py          # from the experiment's folder

Exit 0 when every recorded variant is reproduced, 1 otherwise.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BASES = HERE.parent / "exp_055_exposure_review_lures" / "data" / "messages"
RUNS = {"runs/six": "payloads", "runs/six-r2": "payloads", "runs/n1": "control", "runs/n1-r2": "control"}


def main():
    ok = True
    for run, folder in RUNS.items():
        res = HERE / run / "results.jsonl"
        if not res.is_file():
            print(f"{run}: no results.jsonl here, skipped")
            continue
        same = differ = missing = 0
        for line in res.read_text().splitlines():
            r = json.loads(line)
            if r.get("role") != "variant":
                continue
            base, text = BASES / r["base"], HERE / folder / f"{r['payload']}.txt"
            if not base.is_file() or not text.is_file():
                missing += 1
                continue
            built = base.read_bytes().rstrip(b"\r\n") + b"\n\n" + text.read_bytes()
            if hashlib.sha256(built).hexdigest() == r["sha256"]:
                same += 1
            else:
                differ += 1
        good = differ == 0 and missing == 0 and same > 0
        ok = ok and good
        print(f"{run}: {same} variant(s) reproduced from the public base and the public text, {differ} differ, "
              f"{missing} with a file missing: {'yes' if good else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
