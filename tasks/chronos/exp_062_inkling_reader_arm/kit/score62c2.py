#!/usr/bin/env python3
"""exp_062 Addendum C, Amendment 1: scores the release cells (runs_c2/<ts>) against Addendum C's own run folder.

    python3.14 -I kit/score62c2.py runs_c2/<ts> [dev_c2/toolcall-repro-b12676.txt]

Bars as registered in ADDENDUM_C_AMENDMENT_1.md: HD3 (read first), HD1, HD1b, HD4 with Addendum C's thresholds;
HD5 from the tool-call repro's SUMMARY lines. Writes <run>/score.json and prints it.
"""
import json
import re
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import score62c as s  # noqa: E402  (Addendum C's scorer: load, speed, same_text, capacity, the bars)

BASE = KIT.parent / "runs_c" / "20261123T091929Z"     # Addendum C's run: U0 (b12631), U2 (draft head), N2, C-U0, C-U2
SUMMARY = re.compile(r"^\[(?P<label>[^\]]+)\] SUMMARY: tool calls returned by the chat endpoint (?P<got>\d+)/(?P<of>\d+);"
                     r" raw output has a call marker (?P<marker>\d+)/\d+, with the tool's name before it (?P<named>\d+)/\d+")


def hd5(path: Path) -> dict:
    if not path.exists():
        return {"pass": None, "why": f"no repro output at {path.name}"}
    rows = {m["label"]: {k: int(m[k]) for k in ("got", "of", "marker", "named")}
            for m in (SUMMARY.match(ln) for ln in path.read_text(encoding="utf-8").splitlines()) if m}
    research = next((v for k, v in rows.items() if k.startswith("research")), None)
    minis = [v for k, v in rows.items() if k.startswith("Inkling 2 Mini")]
    if research is None or len(minis) != 2:
        return {"pass": False, "why": "a file did not complete the repro", "rows": rows}
    ok = research["got"] >= 4 and all(m["got"] == m["of"] == 5 for m in minis)
    return {"rows": rows, "bar": "research >= 4 of 5 returned; both Mini files 5 of 5", "pass": ok}


def main():
    run = Path(sys.argv[1]).resolve()
    repro = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else KIT.parent / "dev_c2" / "toolcall-repro-b12676.txt"
    cells = {c: s.load(BASE / "cells" / f"{c}.json") for c in ("U0", "U2", "N2")}
    cells.update({c: s.load(run / "cells" / f"{c}.json") for c in ("U3", "N3")})
    sp = {c: s.speed(v) for c, v in cells.items() if v}
    cap = {"C-U0": s.capacity(BASE, "C-U0"), "C-U2": s.capacity(BASE, "C-U2"), "C-U3": s.capacity(run, "C-U3")}
    st = {}
    for a, b in (("U0", "U3"), ("U2", "U3"), ("U0", "U2")):
        if cells.get(a) and cells.get(b):
            st[f"{a}_vs_{b}"] = s.same_text(cells[a], cells[b])

    v = {}
    if "U0_vs_U3" in st and cap["C-U3"]:
        eq, right = st["U0_vs_U3"]["first_token_equal"], cap["C-U3"]["right"][1]
        v["HD3"] = {"first_token_equal": f"{eq} of {st['U0_vs_U3']['texts']}",
                    "one_watcher_right": f"{right} of {cap['C-U3']['jobs']}", "pass": eq >= 7 and right >= 11}
    else:
        v["HD3"] = {"pass": None, "why": "needs cell U3 and Part 2 on the release"}
    readable = v["HD3"]["pass"] is True
    if "U3" in sp:
        v["HD1"] = {"U3_short_gen": sp["U3"]["short_gen"], "bar": s.HC1_BAR, "U2_short_gen_draft_head": sp["U2"]["short_gen"],
                    "U0_short_gen_b12631": sp["U0"]["short_gen"], "U3_long_gen": sp["U3"]["long_gen"],
                    "pass": (sp["U3"]["short_gen"] >= s.HC1_BAR) if readable else None}
    else:
        v["HD1"] = {"pass": False, "why": "does not run: the release did not complete its cell with file U"}
    if "U3" in sp and "N3" in sp:
        rs, rl = sp["U3"]["short_gen"] / sp["N3"]["short_gen"], sp["U3"]["long_gen"] / sp["N3"]["long_gen"]
        v["HD1b"] = {"short_ratio": round(rs, 3), "long_ratio": round(rl, 3), "bar": s.GAP_BAR,
                     "prefill_long_ratio": round(sp["U3"]["long_prefill"] / sp["N3"]["long_prefill"], 3),
                     "pass": (min(rs, rl) >= s.GAP_BAR) if readable else None}
    else:
        v["HD1b"] = {"pass": False if "U3" not in sp else None, "why": "needs cells U3 and N3"}
    if cap["C-U3"]:
        v["HD4"] = {"ratio_4_to_1": cap["C-U3"]["ratio_4_to_1"], "bar": s.QUEUE_BAR,
                    "draft_head_C-U2": cap["C-U2"]["ratio_4_to_1"] if cap["C-U2"] else None,
                    "pass": (cap["C-U3"]["ratio_4_to_1"] > s.QUEUE_BAR) if readable else None}
    else:
        v["HD4"] = {"pass": False if "U3" not in sp else None, "why": "Part 2 on the release was not run"}
    v["HD5"] = hd5(repro)

    out = {"verdicts": v,
           "speed": {c: {k: sp[c][k] for k in ("long_prefill", "long_gen", "short_prefill", "short_gen", "thermal_levels")}
                     for c in sp},
           "same_text": {k: {kk: x[kk] for kk in ("first_token_equal", "texts", "identical_to_the_end", "median_agreed_prefix")}
                         for k, x in st.items()},
           "capacity": {k: {kk: c[kk] for kk in ("jobs_per_minute", "right", "ratio_4_to_1", "ratio_2_to_1",
                                                   "tokens_per_second", "thermal_flagged")} for k, c in cap.items() if c},
           "placement": s.load(run / "placement.json"), "base_run": str(BASE.relative_to(KIT.parent))}
    if cap["C-U0"] and cap["C-U3"]:
        a, b = cap["C-U0"]["calls"][1], cap["C-U3"]["calls"][1]
        out["same_calls_one_watcher_U0_vs_U3"] = f"{sum(a[j] == b[j] for j in a)} of {len(a)}"
    (run / "score.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
