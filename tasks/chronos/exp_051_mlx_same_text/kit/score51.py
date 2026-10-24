#!/usr/bin/env python3
"""exp_051: score the registered hypotheses from one run directory (runs/<ts>/{O,ON,X,XN,F,O_drift}).

    python3.14 -I kit/score51.py runs/<ts>
"""
import json
import sys
from pathlib import Path

EXP011 = {"15k": 1.650, "35k": 2.161}          # exp_011 Phase A, repeat 1 (evidence/20260609T064038Z-phase_a-mini)


def load(run: Path, arm: str) -> dict:
    s = json.loads((run / arm / "summary.json").read_text())
    return {r["size"]: r for r in s["sizes_summary"]}


def val(d, key, field):
    r = d.get(key)
    return None if r is None or r["clean_reps"] < 2 else r[field]


def interp(d: dict, tokens: float):
    pts = sorted((r["prompt_tokens"], r["prefill_ms_per_tok_median"]) for r in d.values() if r["clean_reps"] >= 2)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return None


def verdict(ok):
    return "not decided" if ok is None else ("pass" if ok else "fail")


def main():
    run = Path(sys.argv[1])
    O, ON, X, XN, F = (load(run, a) for a in ("O", "ON", "X", "XN", "F"))
    Od = load(run, "O_drift")
    res = {}

    base = val(X, "15k", "prefill_ms_per_tok_median")
    over = {k: r["prefill_ms_per_tok_median"] / base for k, r in X.items() if base and r["prefill_ms_per_tok_median"]}
    res["H1_mlx_no_cliff"] = {"x_15k": base, "ratio_to_15k": over,
                             "verdict": verdict(None if not base else all(v <= 2.0 for v in over.values()))}

    def ratio(a, b, k, f):
        va, vb = val(a, k, f), val(b, k, f)
        return None if va is None or vb is None else va / vb

    r15, r35 = ratio(X, O, "15k", "prefill_ms_per_tok_median"), ratio(X, O, "35k", "prefill_ms_per_tok_median")
    res["H2_mlx_prefill_not_slower"] = {"x_over_o_15k": r15, "x_over_o_35k": r35,
                                        "verdict": verdict(None if None in (r15, r35) else (r15 <= 1.05 and r35 <= 1.05))}
    g4 = ratio(X, O, "4k", "gen_tps_median")
    res["H3_mlx_generation_lead_4k"] = {"x_over_o_gen_4k": g4, "verdict": verdict(None if g4 is None else g4 >= 1.10)}

    f = {k: val(F, k, "prefill_ms_per_tok_median") for k in EXP011}
    dev = {k: (f[k] / EXP011[k] - 1) if f[k] else None for k in EXP011}
    res["H4_replicates_exp011_on_its_input"] = {"f": f, "exp011": EXP011, "rel_diff": dev,
                                                "verdict": verdict(None if None in dev.values() else all(abs(v) <= 0.10 for v in dev.values()))}

    m = {}
    for k in ("15k", "35k"):
        fv, ft = val(F, k, "prefill_ms_per_tok_median"), val(F, k, "prompt_tokens")
        xi = interp(X, ft) if ft else None
        m[k] = {"f": fv, "f_tokens": ft, "x_at_same_tokens": xi, "rel_diff": (fv / xi - 1) if fv and xi else None}
    res["H5_text_does_not_change_mlx_prefill"] = {
        "matched": m, "verdict": verdict(None if any(v["rel_diff"] is None for v in m.values())
                                         else all(abs(v["rel_diff"]) <= 0.10 for v in m.values()))}

    n15, n35 = ratio(XN, ON, "15k", "prefill_ms_per_tok_median"), ratio(XN, ON, "35k", "prefill_ms_per_tok_median")
    res["H6_mlx_prefill_not_slower_natural_text"] = {"xn_over_on_15k": n15, "xn_over_on_35k": n35,
                                                    "verdict": verdict(None if None in (n15, n35) else (n15 <= 1.05 and n35 <= 1.05))}

    ob = val(O, "15k", "prefill_ms_per_tok_median")
    res["descriptive"] = {
        "o_ratio_to_15k": {k: r["prefill_ms_per_tok_median"] / ob for k, r in O.items() if ob and r["prefill_ms_per_tok_median"]},
        "x_over_o_prefill": {k: ratio(X, O, k, "prefill_ms_per_tok_median") for k in O},
        "x_over_o_gen": {k: ratio(X, O, k, "gen_tps_median") for k in O},
        "xn_over_on_prefill": {k: ratio(XN, ON, k, "prefill_ms_per_tok_median") for k in ON},
        "xn_over_on_gen": {k: ratio(XN, ON, k, "gen_tps_median") for k in ON}}
    d0, d1 = val(O, "15k", "prefill_ms_per_tok_median"), val(Od, "15k", "prefill_ms_per_tok_median")
    res["drift_check"] = {"o_15k_first": d0, "o_15k_last": d1, "rel": (d1 / d0 - 1) if d0 and d1 else None,
                          "within_5pct": None if not (d0 and d1) else abs(d1 / d0 - 1) <= 0.05}
    res["tables"] = {a: d for a, d in (("O", O), ("ON", ON), ("X", X), ("XN", XN), ("F", F), ("O_drift", Od))}
    out = run / "score.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        if k.startswith("H"):
            print(k, v["verdict"])
    print("drift", res["drift_check"])
    print("wrote", out)


if __name__ == "__main__":
    main()
