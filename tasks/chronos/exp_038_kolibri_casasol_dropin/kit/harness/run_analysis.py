"""exp_038: the registered analysis end to end (HYPOTHESIS "Hypotheses", "Verdict map"). Code only.

    python3 harness/run_analysis.py <stage1.json> <collected_judged.json> [<collected_redeal.json>] --out <results.json>

Inputs: stage 1's deterministic report; the unblinded judge scores {arm: {"<set>:<id>": fields}}; optionally the
re-dealt blocks' scores; the sealed sets (substance flags, row types); exp_035's v1 substance flags; E38-D1's result.
Floor scope (Andrei, 2026-10-10): categorical floors per language, the numeric and citation floors pooled per arm,
plus 0 fabricated citations kept (judge).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P38 / "harness"))
import analysis as A  # noqa: E402

E35_JUDGE = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_035_coapi_voice/results/designA_A12_judge_20260922.json"
CATEGORICAL = ("language", "plain_text", "oos_declined", "handoff")


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def floors(st: dict, judge: dict) -> tuple[bool, list[str]]:
    fails = []
    for lang, f in sorted(st["floors"].items()):
        for k in CATEGORICAL:
            if f["rates"][k] < 1.0:
                fails.append(f"{lang.upper()} {k}")
        if f["scope_false_out"]:
            fails.append(f"{lang.upper()} scope-gate false-OUT")
        if f["failed_calls"]:
            fails.append(f"{lang.upper()} failed calls")
    pooled = st["pooled_ratio_floors"]
    if pooled["numeric"] < 0.98:
        fails.append("numeric preservation (pooled)")
    if pooled["citation"] < 0.94:
        fails.append("citation-in-context (pooled)")
    if sum(len(v.get("fabricated_citations_kept") or []) for v in judge.values()):
        fails.append("fabricated citations kept")
    return not fails, fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage1", type=Path)
    ap.add_argument("judged", type=Path)
    ap.add_argument("redeal", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    st = json.loads(a.stage1.read_text())
    J = json.loads(a.judged.read_text())["scores"]
    sets = P38 / "private/sets"
    v1 = jl(P38 / "vendor/coapi_voice/data/eval/coapi_eval_v1.jsonl")
    de = jl(sets / "de_slice.jsonl")
    v1_flag = {r["id"]: bool(r["substance_in_context"]) for r in json.loads(E35_JUDGE.read_text())["rows"]}
    corr = lambda arm, key: J[arm][key]["correctness"]  # noqa: E731

    # H1 on P = v1
    ids = [r["id"] for r in v1]
    d = [corr("K4", f"v1:{i}") - corr("G26", f"v1:{i}") for i in ids]
    h1 = A.h1_test(d, [r["lang"] for r in v1])
    # headroom
    C = A.ceiling([{"type": r["type"], "substance_in_context": v1_flag.get(r["id"], False)} for r in v1])
    g26_mean = sum(corr("G26", f"v1:{i}") for i in ids) / len(ids)
    ceiling_limited = C - g26_mean < 0.30
    # H3 on the 33 in-scope DE rows, fixed sequence
    de_in = [r for r in de if r["type"] != "oos"]
    d3 = [corr("K4", f"de:{r['id']}") - corr("G26", f"de:{r['id']}") for r in de_in]
    h3 = A.h3_test(d3, tested=h1["state"] == "CONFIRMED")
    # D4b interaction (DE vs twin)
    d4b = A.interaction([corr("K4", f"de:{r['id']}") for r in de_in], [corr("G26", f"de:{r['id']}") for r in de_in],
                        [corr("K4", f"twins:{r['twin_id']}") for r in de_in], [corr("G26", f"twins:{r['twin_id']}") for r in de_in])
    # harm guard on P
    harm = A.harm_guard({i: J["K4"][f"v1:{i}"] for i in ids}, {i: J["G26"][f"v1:{i}"] for i in ids}, ids)
    # floors for K4 and E38-D1
    ok, failing = floors(st["K4"], J["K4"])
    d1 = json.loads((P38 / "runs/e38_d1/e38_d1_result.json").read_text())["state"]
    verdict = A.verdict(h1, ok, failing, harm, d1, None, ceiling_limited, h3, d4b)

    def mean(arm, keys):
        xs = [corr(arm, k) for k in keys if k in J.get(arm, {})]
        return sum(xs) / len(xs) if xs else None

    v1k = [f"v1:{i}" for i in ids]
    descriptive = {
        "D1_by_language": {lang: A.h1_test([x for x, r in zip(d, v1) if r["lang"] == lang], [lang] * sum(r["lang"] == lang for r in v1),
                                           tag=f"exp038|D1|{lang}") for lang in ("en", "pl", "es")},
        "D2_substance_rows": A.h1_test([x for x, r in zip(d, v1) if v1_flag.get(r["id"])],
                                       [r["lang"] for r in v1 if v1_flag.get(r["id"])], tag="exp038|D2"),
        "D2_from_weights": {arm: sum(bool(v.get("from_weights")) for v in J[arm].values()) for arm in J},
        "D3_harmful": {arm: sum(v["usefulness"] <= 1 for k, v in J[arm].items() if k.startswith("v1:")) for arm in J},
        "D4_german_penalty": {arm: (mean(arm, [f"twins:{r['twin_id']}" for r in de_in]) or 0) - (mean(arm, [f"de:{r['id']}" for r in de_in]) or 0)
                              for arm in J if f"de:{de_in[0]['id']}" in J[arm]},
        "D4b_interaction": d4b,
        "D5_K4_minus_A12": A.h1_test([corr("K4", k) - corr("A12", k) for k in v1k], [r["lang"] for r in v1], tag="exp038|D5") if "A12" in J else None,
        "D8_K8": {"K8_minus_K4": A.h1_test([corr("K8", k) - corr("K4", k) for k in v1k], [r["lang"] for r in v1], tag="exp038|D8a"),
                  "K8_minus_G26": A.h1_test([corr("K8", k) - corr("G26", k) for k in v1k], [r["lang"] for r in v1], tag="exp038|D8b")} if "K8" in J else None,
        "D9_K4MED": {"MED_minus_K4": A.h1_test([corr("K4-MED", k) - corr("K4", k) for k in v1k], [r["lang"] for r in v1], tag="exp038|D9a"),
                     "MED_minus_G26": A.h1_test([corr("K4-MED", k) - corr("G26", k) for k in v1k], [r["lang"] for r in v1], tag="exp038|D9b")} if "K4-MED" in J else None,
        "means_v1": {arm: mean(arm, v1k) for arm in J},
    }
    rel = None
    if a.redeal:
        R = json.loads(a.redeal.read_text())["scores"]
        first = {f"{arm}|{k}": J[arm][k]["correctness"] for arm in R for k in R[arm]}
        second = {f"{arm}|{k}": R[arm][k]["correctness"] for arm in R for k in R[arm]}
        rel = A.reliability(first, second)
        sub = [k for k in R.get("K4", {}) if k.startswith("v1:") and k in R.get("G26", {})]
        if sub:
            rel["H1_subset"] = {"first": A.h1_test([J["K4"][k]["correctness"] - J["G26"][k]["correctness"] for k in sub], ["x"] * len(sub), tag="exp038|rel1")["state"],
                                "second": A.h1_test([R["K4"][k]["correctness"] - R["G26"][k]["correctness"] for k in sub], ["x"] * len(sub), tag="exp038|rel2")["state"]}
            rel["double_judging_required"] = rel["double_judging_required"] or rel["H1_subset"]["first"] != rel["H1_subset"]["second"]
    res = {"H1": h1, "H3": h3, "headroom": {"C": C, "G26_mean": g26_mean, "ceiling_limited": ceiling_limited},
           "harm_guard": harm, "K4_floors": {"met": ok, "failing": failing}, "E38_D1": d1, "verdict": verdict,
           "descriptive": descriptive, "reliability": rel}
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"verdict": verdict, "H1": {k: h1[k] for k in ("point", "p", "p_rev", "state")}, "H3": h3["state"],
                      "reliability": rel and {k: rel[k] for k in ("exact_agreement", "double_judging_required")}}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
