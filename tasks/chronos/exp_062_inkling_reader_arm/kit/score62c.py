#!/usr/bin/env python3
"""exp_062 Addendum C: reads a run folder of kit/speed62c.py and computes HC1 to HC4 as registered.

    python3.14 -I kit/score62c.py runs_c/<ts>

Writes <run>/score.json and prints the tables. HC3 is read first: if it fails, HC1, HC1b and HC4 are not readable.
A missing cell or Part 2 run makes every bar that needs it "not run", never a pass.
"""
import json
import statistics
import sys
from pathlib import Path

HC1_BAR, GAP_BAR, CONTROL_BAR, QUEUE_BAR = 22.8, 0.90, 0.10, 1.4
CELLS = ["U0", "N0", "U1", "U2", "N2", "U0r"]
CAPACITY = ["C-U0", "C-N0", "C-U2"]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def med(rows, key):
    return round(statistics.median(r[key] for r in rows), 2)


def speed(cell: dict) -> dict:
    return {"long_prefill": med(cell["long"], "prompt_per_second"), "long_gen": med(cell["long"], "predicted_per_second"),
            "short_prefill": med(cell["short"], "prompt_per_second"), "short_gen": med(cell["short"], "predicted_per_second"),
            "long_gen_all": [round(r["predicted_per_second"], 2) for r in cell["long"]],
            "short_gen_all": [round(r["predicted_per_second"], 2) for r in cell["short"]],
            "long_prefill_all": [round(r["prompt_per_second"], 1) for r in cell["long"]],
            "thermal_levels": sorted({r["thermal_before"] for r in cell["long"] + cell["short"]})}


def prefix(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def same_text(a: dict, b: dict) -> dict:
    """Two cells on the same eight texts: the first token, and the tokens agreed before the first difference."""
    ta, tb = {t["id"]: t for t in a["texts"]}, {t["id"]: t for t in b["texts"]}
    rows = []
    for i in ta:
        x, y = ta[i]["tokens"], tb[i]["tokens"]
        pa = {p["id"]: p for p in ta[i]["first_top5"]}
        pb = {p["id"]: p for p in tb[i]["first_top5"]}
        top = ta[i]["first_top5"][0]["id"] if ta[i]["first_top5"] else None
        key = "logprob" if top is not None and pa[top].get("logprob") is not None else "prob"
        d = (round(abs(pa[top][key] - pb[top][key]), 4) if top is not None and top in pb
             and pa[top].get(key) is not None and pb[top].get(key) is not None else None)
        rows.append({"id": i, "first_token_equal": bool(x and y and x[0] == y[0]), "agreed_prefix": prefix(x, y),
                     "length": min(len(x), len(y)), f"first_token_{key}_abs_diff": d})
    return {"first_token_equal": sum(r["first_token_equal"] for r in rows), "texts": len(rows),
            "identical_to_the_end": sum(r["agreed_prefix"] == r["length"] for r in rows),
            "median_agreed_prefix": statistics.median(r["agreed_prefix"] for r in rows), "rows": rows}


def capacity(run: Path, name: str):
    levels = {n: load(run / name / f"level-{n}.json") for n in (1, 2, 4)}
    if not all(levels.values()):
        return None
    s = {n: levels[n]["summary"] for n in levels}
    return {"jobs_per_minute": {n: s[n]["jobs_per_minute"] for n in s}, "right": {n: s[n]["right"] for n in s},
            "jobs": s[1]["jobs"], "median_job_s": {n: s[n]["median_job_s"] for n in s},
            "tokens_per_second": {n: s[n]["tokens_per_second"] for n in s},
            "server_rss_gb": {n: round(int(s[n]["server_rss_kb"]) / 1e6, 1) if str(s[n]["server_rss_kb"]).isdigit() else None for n in s},
            "ratio_4_to_1": round(s[4]["jobs_per_minute"] / s[1]["jobs_per_minute"], 2),
            "ratio_2_to_1": round(s[2]["jobs_per_minute"] / s[1]["jobs_per_minute"], 2),
            "thermal_flagged": any(s[n]["thermal"]["flagged"] for n in s),
            "calls": {n: {r["id"]: [c["name"] for c in r["calls"]] for r in levels[n]["rows"]} for n in levels}}


def main():
    run = Path(sys.argv[1]).resolve()
    cells = {c: load(run / "cells" / f"{c}.json") for c in CELLS}
    sp = {c: speed(cells[c]) for c in CELLS if cells[c]}
    cap = {c: capacity(run, c) for c in CAPACITY}
    out = {"run": run.name, "cells_complete": sorted(sp), "capacity_complete": sorted(c for c in cap if cap[c]),
           "speed": sp, "capacity": cap, "placement": load(run / "placement.json")}

    def need(*names):
        return all(n in sp for n in names)

    # ---- same text
    st = {}
    for a, b, what in (("U0", "U0r", "noise floor: U on b12631, first and last cell"), ("U0", "U2", "U: b12631 against the head"),
                       ("U0", "U1", "U: b12631 against the parent"), ("N0", "N2", "control, N: b12631 against the head"),
                       ("U2", "N2", "the two files on the head (not expected to agree)")):
        if need(a, b):
            st[f"{a}_vs_{b}"] = {"what": what, **same_text(cells[a], cells[b])}
    out["same_text"] = st
    # the same sequence across the six repeats of a workload, inside each cell (determinism)
    out["repeats_identical"] = {c: {"long": len({r["tokens_sha256"] for r in cells[c]["long"]}) == 1,
                                    "short": len({r["tokens_sha256"] for r in cells[c]["short"]}) == 1} for c in sp}

    v = {}
    # ---- HC3, read first
    if need("U0", "U2") and cap["C-U2"]:
        eq, right = st["U0_vs_U2"]["first_token_equal"], cap["C-U2"]["right"][1]
        v["HC3"] = {"first_token_equal": f"{eq} of {st['U0_vs_U2']['texts']}", "one_watcher_right": f"{right} of {cap['C-U2']['jobs']}",
                    "pass": eq >= 7 and right >= 11}
    else:
        v["HC3"] = {"pass": None, "why": "not run: needs cells U0 and U2 and Part 2 on the head"}
    readable = v["HC3"]["pass"] is True
    # ---- HC1, HC1b
    if need("U2"):
        v["HC1"] = {"U2_short_gen": sp["U2"]["short_gen"], "bar": HC1_BAR, "U0_short_gen": sp.get("U0", {}).get("short_gen"),
                    "U1_short_gen": sp.get("U1", {}).get("short_gen"), "U2_long_gen": sp["U2"]["long_gen"],
                    "pass": (sp["U2"]["short_gen"] >= HC1_BAR) if readable else None}
    else:
        v["HC1"] = {"pass": False, "why": "does not run: the head build did not complete its cell with file U"}
    if need("U2", "N2"):
        rs, rl = sp["U2"]["short_gen"] / sp["N2"]["short_gen"], sp["U2"]["long_gen"] / sp["N2"]["long_gen"]
        v["HC1b"] = {"short_ratio": round(rs, 3), "long_ratio": round(rl, 3), "bar": GAP_BAR,
                     "prefill_long_ratio": round(sp["U2"]["long_prefill"] / sp["N2"]["long_prefill"], 3),
                     "pass": (rs >= GAP_BAR and rl >= GAP_BAR) if readable else None}
    else:
        v["HC1b"] = {"pass": False if "U2" not in sp else None, "why": "needs cells U2 and N2"}
    # ---- HC2
    if need("N0", "N2"):
        ch = {k: round(sp["N2"][k] / sp["N0"][k] - 1, 3) for k in ("long_prefill", "long_gen", "short_prefill", "short_gen")}
        v["HC2"] = {"change": ch, "bar": CONTROL_BAR, "pass": all(abs(x) < CONTROL_BAR for x in ch.values())}
        if need("U0", "U0r"):
            v["HC2"]["noise_floor_U0_to_U0r"] = {k: round(sp["U0r"][k] / sp["U0"][k] - 1, 3)
                                                 for k in ("long_prefill", "long_gen", "short_prefill", "short_gen")}
    else:
        v["HC2"] = {"pass": None, "why": "not run: needs cells N0 and N2"}
    # ---- HC4
    if cap["C-U2"]:
        v["HC4"] = {"ratio_4_to_1": cap["C-U2"]["ratio_4_to_1"], "bar": QUEUE_BAR,
                    "baseline_C-U0": cap["C-U0"]["ratio_4_to_1"] if cap["C-U0"] else None,
                    "control_C-N0": cap["C-N0"]["ratio_4_to_1"] if cap["C-N0"] else None,
                    "pass": (cap["C-U2"]["ratio_4_to_1"] > QUEUE_BAR) if readable else None}
    else:
        v["HC4"] = {"pass": False if "U2" not in sp else None, "why": "Part 2 on the head was not run"}
    if not readable and v["HC3"]["pass"] is False:
        for k in ("HC1", "HC1b", "HC4"):
            v[k]["why"] = "not readable: HC3 failed"
    out["verdicts"] = v
    # ---- the same tool calls, one watcher, old build against the head
    if cap["C-U0"] and cap["C-U2"]:
        a, b = cap["C-U0"]["calls"][1], cap["C-U2"]["calls"][1]
        out["same_calls_one_watcher_U0_vs_U2"] = f"{sum(a[j] == b[j] for j in a)} of {len(a)}"
    for c in cap.values():
        if c:
            c.pop("calls")
    (run / "score.json").write_text(json.dumps(out, indent=1) + "\n")

    print("cells complete:", out["cells_complete"], " Part 2 complete:", out["capacity_complete"])
    print(f"{'cell':5} {'long prefill':>13} {'long gen':>9} {'short prefill':>14} {'short gen':>10}   thermal")
    for c in CELLS:
        if c in sp:
            s = sp[c]
            print(f"{c:5} {s['long_prefill']:13.1f} {s['long_gen']:9.2f} {s['short_prefill']:14.1f} {s['short_gen']:10.2f}   {s['thermal_levels']}")
    for c in CAPACITY:
        if cap[c]:
            print(c, "jobs a minute", cap[c]["jobs_per_minute"], "right", cap[c]["right"], "4:1", cap[c]["ratio_4_to_1"],
                  "GB", cap[c]["server_rss_gb"])
    for k, s in st.items():
        print(k, "first token equal", s["first_token_equal"], "of", s["texts"], "| identical to the end", s["identical_to_the_end"],
              "| median agreed prefix", s["median_agreed_prefix"])
    print("repeats identical:", out["repeats_identical"])
    for p in out["placement"] or []:
        print("placement", p.get("name"), p.get("splits_by_backend_last_graph"), p.get("mul_mat_nodes_by_op_and_backend_last_graph"), p.get("error", ""))
    for k in ("HC3", "HC1", "HC1b", "HC2", "HC4"):
        print(k, json.dumps(v[k]))
    if "same_calls_one_watcher_U0_vs_U2" in out:
        print("same tool calls, one watcher, U0 against U2:", out["same_calls_one_watcher_U0_vs_U2"])


if __name__ == "__main__":
    main()
