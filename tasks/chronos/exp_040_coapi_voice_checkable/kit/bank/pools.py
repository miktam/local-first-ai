#!/usr/bin/env python3
"""exp_040 — chunk pools for the training bank and v3 (BUILD.md §1). Deterministic, seed 40.

  python3.14 pools.py   # writes ~/REPOS/casasol/coapi_voice/data/bank40/pools.json; prints counts only

Pools by EN chunk id (kinds concepts/citations/formulas only):
  train   M1–M4, M5 T39–41 minus a 25 % holdout per tema   → writer/checker subagents
  valid   M5 T42–43                                         → writer/checker subagents (checkpoint selection)
  holdout the 25 % per tema                                 → the v3 session only
  m6      Módulo 6                                          → the v3 session only
Texts are never printed: pools.json holds ids and sha256 of each chunk text, nothing else.
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

CV = Path.home() / "REPOS/casasol/coapi_voice"
INDEX = {"en": ("f1e28120", CV / "index/guide_en.json"), "pl": ("9a422b0e", CV / "index/guide_pl.json")}
KINDS = {"concepts", "citations", "formulas"}
HOLD = 0.25
OUT = CV / "data/bank40/pools.json"


def load(lang):
    pin, p = INDEX[lang]
    assert hashlib.sha256(p.read_bytes()).hexdigest().startswith(pin), f"{p} is not {pin}"
    return json.loads(p.read_text())["chunks"]


def main():
    en = load("en")
    pl = {c["id"].split(":", 1)[1]: c for c in load("pl")}
    by = defaultdict(list)
    for c in en:
        if c["kind"] in KINDS:
            by[(c["module"], c["tema"])].append(c)
    rng = random.Random(40)
    pools = {"train": [], "valid": [], "holdout": [], "m6": []}
    for (m, t), xs in sorted(by.items()):
        xs = sorted(xs, key=lambda c: c["id"])
        if m == 6:
            pools["m6"] += xs
        elif m == 5 and t in (42, 43):
            pools["valid"] += xs
        else:
            rng.shuffle(xs)
            h = round(len(xs) * HOLD)
            pools["holdout"] += xs[:h]
            pools["train"] += xs[h:]
    rec = {"seed": 40, "hold": HOLD, "kinds": sorted(KINDS), "index": {k: v[0] for k, v in INDEX.items()}, "pools": {}}
    for name, xs in pools.items():
        rec["pools"][name] = [{"id": c["id"], "module": c["module"], "tema": c["tema"],
                               "pl_id": "pl:" + c["id"].split(":", 1)[1] if c["id"].split(":", 1)[1] in pl else None,
                               "sha256": hashlib.sha256(c["text"].encode()).hexdigest()} for c in sorted(xs, key=lambda c: c["id"])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    print({k: len(v) for k, v in rec["pools"].items()}, hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
