#!/usr/bin/env python3
"""
exp_041 — thresholds and the primary gate, from the dev set only (DESIGN §4, HYPOTHESIS). Code only.

  python rl41/select41.py        # reads runs/exp041_dev_G*/probs.jsonl; writes data/gate41/thresholds.json

A gate abstains on a row iff P(answerable) < τ. For each gate, τ is the largest value at which the dev
false-abstention rate (answerable rows with P < τ) is ≤ 0.03, searched over the observed probabilities. Reported per
gate on dev: abstention on not_answerable rows at τ, false abstention, AUROC, and the expected calibration error
(10 equal-width bins of P(answerable) against the label). The primary gate is the one with the highest dev
abstention at τ; ties → the faster gate (G1 < G2 < G3 < G4). Every gate with all dev rows is scored on v3; H1–H3
are decided on the primary gate only, the others are reported.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CV = Path(__file__).resolve().parent.parent
DEV = CV / "data" / "gate41" / "dev.jsonl"
OUT = CV / "data" / "gate41" / "thresholds.json"
FA_MAX_DEV = 0.03
ORDER = ["G1", "G2", "G3", "G4"]


def threshold(probs_ans: list[float], fa_max: float = FA_MAX_DEV) -> float:
    """Largest τ with (#answerable rows having p < τ) / n ≤ fa_max. τ is one of the observed values (or 0)."""
    s = sorted(probs_ans)
    k = int(fa_max * len(s))           # at most k answerable rows may fall below τ
    return s[k] if k < len(s) else 1.0


def ece(pairs: list[tuple[float, int]], bins: int = 10) -> float:
    n, tot = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        cell = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if cell:
            tot += len(cell) / n * abs(sum(p for p, _ in cell) / len(cell) - sum(y for _, y in cell) / len(cell))
    return tot


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of an answerable row > score of a not_answerable row), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    rows = {json.loads(l)["id"]: json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()}
    res = {}
    for gate in ORDER:
        path = CV / "runs" / f"exp041_dev_{gate}" / "probs.jsonl"
        if not path.exists():
            continue
        probs = {json.loads(l)["id"]: json.loads(l)["p"] for l in path.read_text().splitlines() if l.strip()}
        if set(probs) != set(rows):
            res[gate] = {"complete": False, "n": len(probs)}
            continue
        ans = [probs[i] for i, r in rows.items() if r["label"] == "answerable"]
        neg = [probs[i] for i, r in rows.items() if r["label"] == "not_answerable"]
        tau = threshold(ans)
        by = {}
        for key in ("lang", "src"):
            for v in sorted({r[key] for r in rows.values()}):
                a = [probs[i] for i, r in rows.items() if r[key] == v and r["label"] == "answerable"]
                n = [probs[i] for i, r in rows.items() if r[key] == v and r["label"] == "not_answerable"]
                by[f"{key}={v}"] = {"abstain": round(sum(p < tau for p in n) / len(n), 4) if n else None,
                                    "false_abstention": round(sum(p < tau for p in a) / len(a), 4) if a else None}
        res[gate] = {"complete": True, "n": len(probs), "tau": tau,
                     "abstain_dev": round(sum(p < tau for p in neg) / len(neg), 4),
                     "false_abstention_dev": round(sum(p < tau for p in ans) / len(ans), 4),
                     "auroc": round(auroc(ans, neg), 4),
                     "ece": round(ece([(probs[i], int(r["label"] == "answerable")) for i, r in rows.items()]), 4),
                     "by": by, "probs_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    done = [g for g in ORDER if res.get(g, {}).get("complete")]
    primary = max(done, key=lambda g: (res[g]["abstain_dev"], -ORDER.index(g))) if done else None
    out = {"rule": "abstain iff P(answerable) < tau; tau = largest value with dev false abstention <= 0.03; "
                   "primary = highest dev abstention at tau, ties -> faster gate",
           "dev_sha256": hashlib.sha256(DEV.read_bytes()).hexdigest(), "primary": primary, "gates": res}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({g: {k: v for k, v in r.items() if k not in ("by", "probs_sha256")} for g, r in res.items()}, indent=1),
          "\nprimary:", primary)


if __name__ == "__main__":
    main()
