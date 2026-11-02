"""exp_060: descriptive figures behind RESULT.md that score.json does not carry.

Written 2026-11-02, after the run and after scoring. Descriptive only: every verdict comes from score60.py.

    python3.14 -I kit/describe60.py runs/<ts>          # writes runs/<ts>/describe.json

The loop figures read each arm's thinking.jsonl, which is held back from the public repository (RESULT.md,
"Held back"). Without those files the script writes the rest and marks the loop section absent. It emits
counts only, never thinking text.
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP55 = HERE.parents[1] / "exp_055_exposure_review_lures"
INVENTED_IDS = ("C-0a9fbf", "C-0069db50")  # not in the footprint: A-0a9fbf and C-69db50 are
TAIL = 200


def rows(p):
    return {r["id"]: r for r in map(json.loads, open(p))}


def flagged(band):
    return band in ("suspicious", "likely-scam")


def median(xs):
    return round(statistics.median(xs), 1)


def tail_repeat(text):
    """Share of the last TAIL non-empty lines that repeat a line seen earlier in the text."""
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    seen, rep = set(lines[:-TAIL]), 0
    for s in lines[-TAIL:]:
        rep += s in seen
        seen.add(s)
    return round(rep / max(len(lines[-TAIL:]), 1), 2), len(lines)


def chunk_repeat(text, size=40, span=20000):
    """For text with few line breaks: distinct fixed-size chunks in the last `span` characters."""
    tail = text[-span:]
    chunks = [tail[i:i + size] for i in range(0, len(tail) - size + 1, size)]
    return {"distinct": len(set(chunks)), "of": len(chunks)}


def loops(run, arm, results, keep):
    path = run / arm / "thinking.jsonl"
    if not path.exists():
        return None
    out, invented = {}, Counter()
    for r in map(json.loads, open(path)):
        if not keep(r):
            continue
        t = r["thinking"] or ""
        done = results[r["id"]].get("done_reason")
        ran = done == "length"
        hit = any(x in t for x in INVENTED_IDS)
        if ran or done == "stop":
            invented[("runaway" if ran else "finished", hit)] += 1
        if ran:
            share, n = tail_repeat(t)
            row = {"lines": n, "tail_repeat": share, "invented_id_max": max(t.count(x) for x in INVENTED_IDS)}
            if n < 50:
                row["chunks"] = chunk_repeat(t)
            out[r["id"]] = row
    shares = sorted(v["tail_repeat"] for v in out.values())
    return {
        "runaways": out,
        "n": len(out),
        "tail_repeat_median": shares[len(shares) // 2],
        "tail_repeat_ge_092": sum(s >= 0.92 for s in shares),
        "invented_id_in_thinking": {"runaway": invented[("runaway", True)], "runaway_of": len(out),
                                    "finished": invented[("finished", True)],
                                    "finished_of": invented[("finished", True)] + invented[("finished", False)]},
    }


def main(run):
    man = json.load(open(EXP55 / "data/MANIFEST.json"))
    kind = {k: v["kind"] for k, v in man.items()}
    G = rows(EXP55 / "runs/20261025T095614Z/G/results.jsonl")
    T, F, P1, RT, C = (rows(run / p) for p in ("T/results.jsonl", "F/results.jsonl", "F/pass1.jsonl",
                                                 "F/retry.jsonl", "C/results.jsonl"))
    the22 = [i for i in G if not G[i].get("band")]
    right = lambda i, b: flagged(b) == (kind[i] == "lure")

    fb = [i for i in F if F[i].get("fallback")]
    rest = sorted(F[i]["seconds"] for i in F if i not in fb)
    fallback = {
        "n": len(fb),
        "median_s": median([F[i]["seconds"] for i in fb]),
        "pass1_median_s": median([P1[i]["seconds"] for i in fb]),
        "retry_median_s": median([RT[i]["seconds"] for i in fb]),
        "others_median_s": median(rest),
        "others_p95_s": rest[int(0.95 * len(rest))],
        "retry_band_equals_T": sum(RT[i]["band"] == T[i]["band"] for i in fb),
        "retry_items_equal_T": sum(RT[i]["used_items"] == T[i]["used_items"] for i in fb),
        "false_alarms": [i for i in fb if kind[i] != "lure" and flagged(F[i]["band"])],
        "misses": [i for i in fb if kind[i] == "lure" and not flagged(F[i]["band"])],
    }

    c_empty = [i for i in C if not C[i].get("band")]
    finished = [i for i in the22 if C[i].get("band")]
    new = [i for i in c_empty if i not in the22]
    both = [i for i in P1 if P1[i].get("done_reason") == "stop" and C[i].get("done_reason") == "stop"]
    arm_c = {
        "empty": len(c_empty),
        "former_empties_finished": finished,
        "former_empties_finished_eval_count": [min(C[i]["eval_count"] for i in finished),
                                               max(C[i]["eval_count"] for i in finished)],
        "former_empties_finished_right": sum(right(i, C[i]["band"]) for i in finished),
        "former_empties_still_empty": [i for i in c_empty if i in the22],
        "new_empties": {i: {"kind": kind[i], "done_reason": C[i].get("done_reason"),
                            "eval_count": C[i].get("eval_count"), "seconds": C[i].get("seconds"),
                            "error": C[i].get("error")} for i in new},
        "at_32k_wall_eval_count": [min(C[i]["eval_count"] for i in c_empty if C[i].get("done_reason") == "length"),
                                   max(C[i]["eval_count"] for i in c_empty if C[i].get("done_reason") == "length")],
        "at_32k_wall_median_s": median([C[i]["seconds"] for i in c_empty if C[i].get("done_reason") == "length"]),
        "finished_in_both_F_pass1_and_C": len(both),
        "same_generated_text": sum(P1[i]["content"] == C[i]["content"] for i in both),
    }

    d = [i for i in T if T[i]["band"] != F[i]["band"]]
    t_vs_f = {
        "differ": len(d),
        "only_F_right": dict(Counter(kind[i] for i in d if right(i, F[i]["band"]) and not right(i, T[i]["band"]))),
        "only_T_right": dict(Counter(kind[i] for i in d if right(i, T[i]["band"]) and not right(i, F[i]["band"]))),
        "strength_only": sum(right(i, T[i]["band"]) == right(i, F[i]["band"]) for i in d),
    }

    decode = {a: median([r["eval_count"] / (r["eval_duration"] / 1e9) for r in R.values() if r.get("eval_duration")])
              for a, R in (("T", T), ("F_pass1", P1), ("C", C))}
    thermal = Counter(line.split()[1] for line in open(run / "thermal.log") if line.strip())

    out = {
        "run": run.name,
        "F_fallbacks": fallback,
        "C_context_32k": arm_c,
        "T_vs_F": t_vs_f,
        "decode_tok_s_median": decode,
        "thermal_levels": dict(thermal),
        "loops": {"F_pass1": loops(run, "F", P1, lambda r: r.get("pass") == "pass1"),
                  "C": loops(run, "C", C, lambda r: True)},
    }
    (run / "describe.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "loops"}, indent=1))
    for k, v in out["loops"].items():
        print(k, None if v is None else {x: v[x] for x in v if x != "runaways"})


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
