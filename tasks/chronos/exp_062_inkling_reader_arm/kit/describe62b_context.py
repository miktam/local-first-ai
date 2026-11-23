#!/usr/bin/env python3
"""exp_062 Addendum B: turns short of room in the context, per arm, from the token counts in the public rows.
Declared in ADDENDUM_B.md at the second stop ("the result will count those turns per arm from the token counts (a
prompt that leaves less than 2,048 tokens of room)"). Written 2026-11-23 after the run ended and before any arm was
scored. Descriptive: it decides nothing. Counts only.

    python3.14 -I kit/describe62b_context.py runs_b/<ts>          writes runs_b/<ts>/context.json

  tight turn      the prompt leaves less than the 2,048-token cap of room in the 8,192 context
  cut by context  a tight turn that ended on length with fewer than 2,048 tokens written
  cut by the cap  a turn that ended on length with 2,048 tokens written
Two controls ride along, since they come from the same rows:
  speed           median generation speed over turns of 100 tokens or more (the control that the pass count changed)
  no reasoning    turns recorded with no reasoning text, and the most tokens any of them wrote. A turn that wrote
                  many tokens and has no reasoning recorded would point at the kit's parser, not at the model.
"""
import json
import statistics
import sys
from pathlib import Path

CONTEXT, CAP = 8192, 2048


def main():
    run = Path(sys.argv[1])
    out = {"context": CONTEXT, "cap": CAP}
    for arm in ("G", "I", "V1", "V2", "V3", "V4", "V5"):
        p = run / arm / "results.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        turns = [(r["id"], t) for r in rows for t in r["turns"]]
        tight = [(i, t) for i, t in turns if CONTEXT - t["prompt_tokens"] < CAP]
        cut_ctx = [(i, t) for i, t in tight if t["finish_reason"] == "length" and t["completion_tokens"] < CAP]
        cut_cap = [(i, t) for i, t in turns if t["finish_reason"] == "length" and t["completion_tokens"] >= CAP]
        other_len = [(i, t) for i, t in turns if t["finish_reason"] == "length" and (i, t) not in cut_ctx and (i, t) not in cut_cap]
        fast = [t["tps"] for _, t in turns if t.get("tps") and t["completion_tokens"] >= 100]
        bare = [t for _, t in turns if not t["reasoning_chars"]]
        out[arm] = {
            "decisions": len(rows), "turns": len(turns),
            "largest_prompt_tokens": max(t["prompt_tokens"] for _, t in turns),
            "tight_turns": len(tight), "decisions_with_a_tight_turn": len({i for i, _ in tight}),
            "turns_cut_by_context": len(cut_ctx), "decisions_cut_by_context": sorted({i for i, _ in cut_ctx}),
            "turns_cut_by_the_cap": len(cut_cap), "decisions_cut_by_the_cap": sorted({i for i, _ in cut_cap}),
            "length_turns_not_classed": len(other_len),
            "decisions_ended_at_the_context_wall": sum(1 for r in rows if r["end"] == "context"),
            "median_tokens_per_second_turns_of_100_or_more": round(statistics.median(fast), 1) if fast else None,
            "turns_without_reasoning": len(bare),
            "most_tokens_in_a_turn_without_reasoning": max((t["completion_tokens"] for t in bare), default=None),
        }
    (run / "context.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
