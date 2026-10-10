"""K4-MED reasoning cap (HYPOTHESIS "K4-MED reasoning cap"), from the pilot run. Code only.

Reads K4-MED's pilot records (the DE dev set and the 10 step0 questions, run with the maximum cap 16,384) and sets
cap = the smallest power of two >= 2 x the longest reasoning segment seen on an answer call, clamped to
[2,048, 16,384]. A pilot row whose reasoning hit 16,384 makes the cap 16,384. Writes runs/k4med_cap.json.

    python3 harness/pilot_cap.py <K4-MED run dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
LO, HI = 2048, 16384


def main(run_dir: Path) -> int:
    longest, n, capped = 0, 0, 0
    for s in ("dev", "step0"):
        f = run_dir / f"{s}.jsonl"
        if not f.is_file():
            sys.exit(f"REFUSED: {f} missing (the pilot runs dev and step0)")
        for l in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            for c in r.get("calls", []):
                m = c.get("mlx") or {}
                if m.get("call") == "answer" and m.get("effort") == "medium":
                    n += 1
                    longest = max(longest, int(m.get("reasoning_tokens") or 0))
                    capped += m.get("finish_reason") == "reasoning_cap"
    need = 2 * longest
    cap = LO
    while cap < need and cap < HI:
        cap *= 2
    cap = HI if capped else min(max(cap, LO), HI)
    res = {"pilot_answer_calls": n, "longest_reasoning_tokens": longest, "pilot_rows_at_cap": capped,
           "rule": "smallest power of two >= 2 x longest, clamped to [2048, 16384]", "cap": cap}
    (P38 / "runs" / "k4med_cap.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
