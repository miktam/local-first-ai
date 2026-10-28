#!/usr/bin/env python3
"""exp_059: raw answers → band and action, through the frozen rev 4 knife, unchanged, for any arm.

`kev_knife_rev4.triage()` is called as it is. Its `ask()` is replaced by the arm's recorded answers for that enquiry, and
its `code_facts()` by the facts exp_044 recorded for the same text (language, budget, phone and e-mail flags are
deterministic functions of the text, and the live `code_facts` imports the CasaSol bot). Everything else — rounding,
the spam override, the "wants to own" rule, urgency and the next action — runs from the frozen bytes, so every arm's
decisions come from the same code and differ only in the model's answers.

    python3 finish59.py <arm> [--smoke]

Reads raw/<arm>.jsonl (or raw/smoke_<arm>.jsonl), writes pred/<arm>.jsonl (or pred/smoke_<arm>.jsonl) with the same
row shape as exp_044's pred files plus `ms` from the raw run. Prints the agreement of request type and action with
exp_044's published rev 4 run for the ids present (control C3 when the arm is mini-a).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
EXP044 = EXP.parent / "exp_044_kev_lead_triage_rev4"

sys.path.insert(0, str(HERE))
import render59  # noqa: E402


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test", type=Path, default=EXP044 / "data" / "test.jsonl")
    ap.add_argument("--pred044", type=Path, default=EXP044 / "results" / "test" / "pred_rev4.jsonl")
    a = ap.parse_args()

    if render59.sha256(a.test) != render59.TEST_SHA:
        sys.exit(f"REFUSED: {a.test} is not the sealed exp_044 test set")
    text = {r["id"]: r["text"] for r in load(a.test)}
    p044 = {r["id"]: r for r in load(a.pred044)}
    name = f"smoke_{a.arm}" if a.smoke else a.arm
    raw = load(EXP / "raw" / f"{name}.jsonl")

    K = render59.knife()
    current: dict = {}
    K.ask = lambda state, schema: current["answers"]            # the recorded answers stand in for the server
    K.code_facts = lambda t: dict(p044[current["id"]]["facts"])  # the facts exp_044 recorded for this text

    (EXP / "pred").mkdir(exist_ok=True)
    out = EXP / "pred" / f"{name}.jsonl"
    agree = {"intent": 0, "action": 0}
    with out.open("w", encoding="utf-8") as f:
        for r in raw:
            current.clear()
            current.update(r)
            row = K.triage(text[r["id"]])
            row["ms"] = r["ms"]
            f.write(json.dumps({"id": r["id"], **row}, ensure_ascii=False) + "\n")
            agree["intent"] += row["intent"]["choice"] == p044[r["id"]]["intent"]["choice"]
            agree["action"] += row["action"] == p044[r["id"]]["action"]
    n = len(raw)
    print(f"{out}: {n} rows; against exp_044's published rev 4 run on the same ids: "
          f"request {agree['intent']}/{n} ({agree['intent'] / n:.3f}), action {agree['action']}/{n} ({agree['action'] / n:.3f})")


if __name__ == "__main__":
    main()
