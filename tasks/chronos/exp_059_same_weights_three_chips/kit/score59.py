#!/usr/bin/env python3
"""exp_059: score the arms that exist — the hypotheses H1–H4, the controls C1–C3 and the descriptive values D1–D5
of HYPOTHESIS.md. Verdicts are printed only when both arms of a comparison are present and finished.

    python3 score59.py            # reads raw/<arm>.jsonl and pred/<arm>.jsonl for mini-a, mini-b, mbp, iphone, mbp-app
                                  # writes results/score.json and prints the tables
"""
from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
EXP044 = EXP.parent / "exp_044_kev_lead_triage_rev4"
ARMS = ("mini-a", "mini-b", "mbp", "iphone", "mbp-app")

H1_BAR, H2_TOL, H2_BAR, H3_X, H4_X = 0.98, 0.01, 0.99, 3.0, 1.2


def load(p: Path) -> dict[str, dict]:
    if not p.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}


def probs(ans: dict) -> list[float]:
    """one answer's probabilities as a list, in option order (noul: [1-p, p])"""
    if ans["type"] == "noul":
        return [1.0 - ans["noul"], ans["noul"]]
    return [ans["probabilities"][k] for k in sorted(ans["probabilities"])]   # by key: the arms may order keys differently


def pair_bands(pa: dict, pb: dict) -> dict | None:
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    ai = sum(pa[i]["intent"]["choice"] == pb[i]["intent"]["choice"] for i in ids)
    aa = sum(pa[i]["action"] == pb[i]["action"] for i in ids)
    flips = [{"id": i, "a": pa[i]["intent"]["choice"], "b": pb[i]["intent"]["choice"],
              "margin_a": margin(pa[i])} for i in ids if pa[i]["intent"]["choice"] != pb[i]["intent"]["choice"]]
    return {"n": len(ids), "intent": round(ai / len(ids), 4), "action": round(aa / len(ids), 4),
            "intent_flips": flips, "action_flips": sum(pa[i]["action"] != pb[i]["action"] for i in ids)}


def margin(row: dict) -> float | None:
    p = sorted((row.get("intent_kev") or row["intent"]).get("probabilities", {}).values(), reverse=True)
    return round(p[0] - p[1], 3) if len(p) >= 2 else None


def pair_probs(ra: dict, rb: dict) -> dict | None:
    ids = sorted(set(ra) & set(rb))
    if not ids:
        return None
    deltas, per_q = [], defaultdict(list)
    for i in ids:
        for qid, ans in ra[i]["answers"].items():
            if qid not in rb[i]["answers"]:
                continue
            d = max(abs(x - y) for x, y in zip(probs(ans), probs(rb[i]["answers"][qid])))
            deltas.append(d)
            per_q[qid].append(d)
    deltas.sort()
    within = sum(d <= H2_TOL for d in deltas) / len(deltas)
    return {"pairs": len(deltas), "within_tol": round(within, 4), "median": deltas[len(deltas) // 2],
            "p99": deltas[int(len(deltas) * 0.99)], "max": deltas[-1],
            "per_question_max": {q: max(v) for q, v in per_q.items()}}


def ms_of(raw: dict) -> list[float]:
    return [r["ms"] for r in raw.values()]


def tok_s(raw: dict) -> float | None:
    """prefill tokens per second over the arm: tokens from the server's usage.input_tokens or the app's n_tokens"""
    toks, secs = 0, 0.0
    for r in raw.values():
        n = (r.get("usage") or {}).get("input_tokens") or r.get("n_tokens")
        if n:
            toks += n
            secs += r["ms"] / 1000
    return round(toks / secs, 1) if secs else None


def verdict(ok: bool | None) -> str:
    return "—" if ok is None else ("PASS" if ok else "FAIL")


def main():
    raw = {a: load(EXP / "raw" / f"{a}.jsonl") for a in ARMS}
    pred = {a: load(EXP / "pred" / f"{a}.jsonl") for a in ARMS}
    have = [a for a in ARMS if raw[a] and pred[a]]
    test = load(EXP044 / "data" / "test.jsonl")
    p044 = load(EXP044 / "results" / "test" / "pred_rev4.jsonl")
    out: dict = {"arms": {a: {"n_raw": len(raw[a]), "n_pred": len(pred[a])} for a in ARMS}, "comparisons": {}, "hypotheses": {},
                 "controls": {}, "descriptive": {}}

    def cmp(a: str, b: str) -> dict | None:
        if a not in have or b not in have:
            return None
        c = {"bands": pair_bands(pred[a], pred[b]), "probs": pair_probs(raw[a], raw[b])}
        out["comparisons"][f"{a} vs {b}"] = c
        return c

    # controls
    c1 = cmp("mini-a", "mini-b")
    c2 = cmp("mbp-app", "mbp")
    c3 = pair_bands(pred["mini-a"], p044) if "mini-a" in have else None
    out["controls"] = {"C1_floor": c1, "C2_readout": c2, "C3_continuity_vs_exp044": c3}
    c1_bands_ok = c1 and c1["bands"]["intent"] >= H1_BAR and c1["bands"]["action"] >= H1_BAR
    c1_probs_ok = c1 and c1["probs"]["within_tol"] >= H2_BAR
    c2_ok = c2 and c2["bands"]["intent"] >= H1_BAR and c2["bands"]["action"] >= H1_BAR and c2["probs"]["within_tol"] >= H2_BAR

    # hypotheses: the phone against mini-a
    ph = cmp("iphone", "mini-a")
    h = out["hypotheses"]
    if ph:
        b, p = ph["bands"], ph["probs"]
        h["H1"] = {"intent": b["intent"], "action": b["action"], "bar": H1_BAR,
                   "pass": b["intent"] >= H1_BAR and b["action"] >= H1_BAR,
                   "separable": bool(c1_bands_ok) if c1 else None, "void_without_C2": not bool(c2_ok)}
        h["H2"] = {"within_0.01": p["within_tol"], "bar": H2_BAR, "pass": p["within_tol"] >= H2_BAR,
                   "separable": bool(c1_probs_ok) if c1 else None, "void_without_C2": not bool(c2_ok)}
    if raw["iphone"] and raw["mini-a"]:
        mi, mm = st.median(ms_of(raw["iphone"])), st.median(ms_of(raw["mini-a"]))
        h["H3"] = {"iphone_median_ms": mi, "mini_median_ms": mm, "ratio": round(mi / mm, 2), "bar": H3_X, "pass": mi <= H3_X * mm}
    if len(raw["iphone"]) >= 200:
        order = ms_of(raw["iphone"])
        first, last = st.median(order[:100]), st.median(order[-100:])
        h["H4"] = {"first100_median_ms": first, "last100_median_ms": last, "ratio": round(last / first, 2), "bar": H4_X,
                   "pass": last <= H4_X * first}

    # descriptive
    d = out["descriptive"]
    d["D1_prefill_tok_s"] = {a: tok_s(raw[a]) for a in ARMS if raw[a]}
    d["D1_median_ms"] = {a: st.median(ms_of(raw[a])) for a in ARMS if raw[a]}
    if ph:
        by_lang = defaultdict(lambda: [0, 0])
        for i in set(pred["iphone"]) & set(pred["mini-a"]):
            by_lang[test[i]["lang"]][1] += 1
            by_lang[test[i]["lang"]][0] += pred["iphone"][i]["intent"]["choice"] == pred["mini-a"][i]["intent"]["choice"]
        d["D4_intent_agreement_by_language"] = {l: round(v[0] / v[1], 3) for l, v in sorted(by_lang.items())}
        d["D4_flips"] = ph["bands"]["intent_flips"]
    d["D5_mbp_vs_mini"] = cmp("mbp", "mini-a")
    if raw["iphone"]:
        th = defaultdict(int)
        for r in raw["iphone"].values():
            th[r.get("thermal", "?")] += 1
        d["D2_iphone_thermal_states"] = dict(th)
        bl = [r["battery"] for r in raw["iphone"].values() if r.get("battery") is not None]
        d["D2_iphone_battery_first_last"] = [bl[0], bl[-1]] if bl else None
        run = EXP / "raw" / "iphone.run.json"
        d["D3_iphone_run"] = json.loads(run.read_text()) if run.exists() else None

    (EXP / "results").mkdir(exist_ok=True)
    (EXP / "results" / "score.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print("arms:", ", ".join(f"{a} {len(raw[a])}/{len(pred[a])}" for a in ARMS))
    print("\n| comparison | n | request | action | probs within 0.01 | median Δp | max Δp |\n|---|---|---|---|---|---|---|")
    for k, c in out["comparisons"].items():
        b, p = c["bands"], c["probs"]
        print(f"| {k} | {b['n']} | {b['intent']} | {b['action']} | {p['within_tol']} | {p['median']:.4f} | {p['max']:.4f} |")
    if c3:
        print(f"| mini-a vs exp_044 (2026-10-19) | {c3['n']} | {c3['intent']} | {c3['action']} | — | — | — |")
    print("\n| hypothesis | value | bar | verdict |\n|---|---|---|---|")
    for k, v in h.items():
        note = ""
        if v.get("void_without_C2"):
            note = " (void: C2 not passed)"
        elif v.get("separable") is False:
            note = " (not separable from run-to-run noise, C1)"
        val = {kk: vv for kk, vv in v.items() if kk not in ("pass", "bar", "separable", "void_without_C2")}
        print(f"| {k} | {json.dumps(val)} | {v['bar']} | {verdict(v['pass'])}{note} |")
    print("\nD1 prefill tok/s:", d["D1_prefill_tok_s"], " median ms:", d["D1_median_ms"])
    print(f"written: {EXP / 'results' / 'score.json'}")


if __name__ == "__main__":
    main()
