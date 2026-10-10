"""exp_038: the public results export (HYPOTHESIS "Published and private"). Numbers, flags and ids only; no text.

    python3 tools/export_results.py <public_dir>

writes <public_dir>/results/:
  per_row_scores.jsonl   one line per scored row (set, id): per arm the judge's scores and flags, stage 1's checks
                          and the call's timing; claim and citation lists become counts, judge notes are left out
  redeal_scores.jsonl    the re-dealt blocks' second scores, same fields
  judge_export.json      block -> row key -> answer code -> {arm, set, id} and the scores (both passes)
  arm_summaries.json     per arm: means by set and language, floors, probes, judge flags, timing
  analysis.json          the registered analysis output (runs/results.json) and the registered verdict line
  checks.json            R1-R7, E38-D1 and the K4-MED cap, short labels and numbers only
  blindness/<label>.json every audited subagent: file lists, counts, flags, the reviewed verdict
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
HOME = str(Path.home())
JUDGE = P38 / "private/w6/judge"
ARMS = ["G26", "K4", "K8", "K4-MED", "A12"]
SETS = ["v1", "de", "twins"]
SCORE_FIELDS = ["correctness", "usefulness", "language_ok", "language_native", "tail_degradation", "from_weights"]
STAGE1_FIELDS = ["q_lang", "answer_lang", "lang_ok", "refused", "handoff", "plain_text", "cites_in_context", "cites_total",
                 "nums_total", "nums_in_context", "nums_in_gold", "nums_derived", "spanish_term_ok", "retrieval_hit",
                 "output_tokens", "truncated", "scope", "failed_call"]
RAW_FIELDS = ["wall_s", "prompt_tokens", "output_tokens", "repaired", "failed_call"]
WALL_LIMIT_S = 120.0
SCRATCH = re.compile(r"/private/tmp/claude-\d+/[^/\s]+/[^/\s]+/scratchpad")  # the session scratchpad (carries the account)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def short(x):
    """Keep numbers, booleans, hashes and labels; drop any string of six words or more; never a home path."""
    if isinstance(x, dict):
        return {k: short(v) for k, v in x.items()}
    if isinstance(x, list):
        return [short(v) for v in x]
    if isinstance(x, str):
        if len(x.split()) >= 6:
            return "[text omitted]"
        return x.replace(HOME, "~")
    return x


def scores(f: dict) -> dict:
    out = {k: f.get(k) for k in SCORE_FIELDS}
    out["n_unsupported_claims"] = len(f.get("unsupported_claims") or [])
    out["n_fabricated_citations_kept"] = len(f.get("fabricated_citations_kept") or [])
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def rows_meta() -> dict:
    v1_flag = {r["id"]: bool(r["substance_in_context"]) for r in json.loads(
        (Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_035_coapi_voice/results/designA_A12_judge_20260922.json")
        .read_text())["rows"]}
    meta = {}
    for r in jl(P38 / "vendor/coapi_voice/data/eval/coapi_eval_v1.jsonl"):
        meta[f"v1:{r['id']}"] = {"set": "v1", "id": r["id"], "lang": r["lang"], "type": r["type"],
                                 "substance_in_context": v1_flag.get(r["id"], False)}
    for r in jl(P38 / "private/sets/de_slice.jsonl"):
        meta[f"de:{r['id']}"] = {"set": "de", "id": r["id"], "lang": r["lang"], "type": r["type"],
                                 "substance_in_context": bool(r["substance_in_context"]), "twin_id": r["twin_id"]}
    for r in jl(P38 / "private/sets/de_twins.jsonl"):
        meta[f"twins:{r['id']}"] = {"set": "twins", "id": r["id"], "lang": r["lang"], "type": r["type"],
                                    "substance_in_context": bool(r["substance_in_context"]), "twin_of": r["twin_of"]}
    return meta


def raw_records() -> dict:
    out = {}
    for arm in ARMS:
        for s in SETS + ["probes"]:
            p = P38 / "runs/arms" / arm / f"{s}.jsonl"
            if p.is_file():
                for r in jl(p):
                    rec = {k: r.get(k) for k in RAW_FIELDS}
                    rec["n_citations_dropped"] = len(r.get("citations_dropped") or [])
                    rec["n_flagged_claims"] = len(r.get("flagged_claims") or [])
                    out[(arm, f"{s}:{r['id']}")] = rec
    return out


def r6_identity() -> dict:
    out = {}
    for arm in ["G26", "K4", "K8", "K4-MED"]:
        main = {r["id"]: r["answer"] for r in jl(P38 / "runs/arms" / arm / "v1.jsonl")}
        rerun = jl(P38 / "runs/arms" / f"{arm}-R6" / "v1.jsonl")
        out[arm] = {"n": len(rerun), "identical": sum(main.get(r["id"]) == r["answer"] for r in rerun)}
    return out


def main(public: Path) -> int:
    res = public / "results"
    (res / "blindness").mkdir(parents=True, exist_ok=True)
    J = json.loads((JUDGE / "collected_judged.json").read_text())
    R = json.loads((JUDGE / "collected_judged_redeal.json").read_text())
    assert not J["errors"], J["errors"]
    st = json.loads((P38 / "runs/stage1.json").read_text())
    meta, raw = rows_meta(), raw_records()

    # per-row scores
    lines = []
    for key in sorted(meta, key=lambda k: (SETS.index(k.split(":")[0]), k)):
        row = dict(meta[key])
        row["arms"] = {}
        for arm in ARMS:
            if key not in J["scores"].get(arm, {}):
                continue
            a = {"judge": scores(J["scores"][arm][key])}
            s1 = st[arm]["rows"].get(key)
            if s1:
                a["stage1"] = {k: s1.get(k) for k in STAGE1_FIELDS}
                a["stage1"]["n_nums_unsupported"] = len(s1.get("nums_unsupported") or [])
            a["call"] = raw.get((arm, key))
            row["arms"][arm] = a
        lines.append(json.dumps(row, ensure_ascii=False))
    (res / "per_row_scores.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    red = []
    for arm in ARMS:
        for key, f in sorted(R["scores"].get(arm, {}).items()):
            red.append(json.dumps({"arm": arm, "set": key.split(":")[0], "id": key.split(":", 1)[1],
                                   "first": scores(J["scores"][arm][key]), "second": scores(f)}))
    (res / "redeal_scores.jsonl").write_text("\n".join(red) + "\n", encoding="utf-8")

    # structured judge export: block -> key -> code -> arm/set/id + scores
    unblind = json.loads((JUDGE / "private_unblind.json").read_text())["map"]
    export = {"note": "Answer codes and row keys are the judges' blind labels; arm, set and id come from the unblind map.",
              "judged": {}, "redeal": {}}
    for kind, d in (("judged", "judged"), ("redeal", "judged_redeal")):
        for jf in sorted((JUDGE / d).glob("B*.json")):
            j = json.loads(jf.read_text())
            export[kind][jf.stem] = {r["key"]: {code: {**unblind[f"{r['key']}:{code}"], **scores(f)}
                                                for code, f in r["answers"].items()} for r in j["rows"]}
    (res / "judge_export.json").write_text(json.dumps(export, indent=1) + "\n")

    # per-arm summaries
    summ = {}
    for arm in ARMS:
        S = J["scores"].get(arm, {})
        a = {"n_judged": len(S)}
        for s in SETS:
            ks = [k for k in S if k.startswith(f"{s}:")]
            if ks:
                a[f"{s}_correctness_mean"] = mean(S[k]["correctness"] for k in ks)
                a[f"{s}_usefulness_mean"] = mean(S[k]["usefulness"] for k in ks)
                a[f"{s}_language_native_rate"] = mean(float(bool(S[k]["language_native"])) for k in ks)
                a[f"{s}_harmful"] = sum(S[k]["usefulness"] <= 1 for k in ks)
        a["v1_correctness_by_lang"] = {lg: mean(S[k]["correctness"] for k in S if k.startswith("v1:") and meta[k]["lang"] == lg)
                                       for lg in ("en", "pl", "es")}
        a["from_weights"] = sum(bool(v.get("from_weights")) for v in S.values())
        a["tail_degradation"] = sum(bool(v.get("tail_degradation")) for v in S.values())
        a["unsupported_claims"] = sum(len(v.get("unsupported_claims") or []) for v in S.values())
        a["fabricated_citations_kept"] = sum(len(v.get("fabricated_citations_kept") or []) for v in S.values())
        a["floors"] = short(st[arm]["floors"])
        a["pooled_ratio_floors"] = short(st[arm]["pooled_ratio_floors"])
        a["probes"] = short(st[arm]["probes"])
        a["probes_passed"] = sum(bool(p.get("pass")) for p in st[arm]["probes"])
        walls = {s: [raw[(arm, k)]["wall_s"] for (ar, k) in raw if ar == arm and k.startswith(f"{s}:")] for s in SETS + ["probes"]}
        a["wall_s"] = {s: {"n": len(w), "median": round(statistics.median(w), 2), "max": round(max(w), 2),
                           "over_120s": sum(x > WALL_LIMIT_S for x in w)} for s, w in walls.items() if w}
        toks = [raw[(arm, k)]["output_tokens"] for (ar, k) in raw if ar == arm and raw[(arm, k)]["output_tokens"] is not None]
        a["output_tokens_median"] = statistics.median(toks) if toks else None
        a["truncated"] = sum(bool(r.get("truncated")) for r in st[arm]["rows"].values())
        summ[arm] = a
    (res / "arm_summaries.json").write_text(json.dumps(summ, indent=1) + "\n")

    # analysis and the registered verdict line (HYPOTHESIS: "It also ends '; ceiling-limited' when the headroom rule fires")
    an = json.loads((P38 / "runs/results.json").read_text())
    line = an["verdict"] + ("; ceiling-limited" if an["headroom"]["ceiling_limited"] and not an["verdict"].endswith("; ceiling-limited") else "")
    (res / "analysis.json").write_text(json.dumps({"verdict_registered": line, "verdict_code": an["verdict"], **an},
                                                  indent=1, default=float) + "\n")

    # checks
    r1 = {p.parent.name: short(json.loads(p.read_text())) for p in sorted((P38 / "runs").glob("r1_*/r1_result.json"))}
    r2 = {p.parent.name: short(json.loads(p.read_text())) for p in sorted((P38 / "runs").glob("r2_*/r2_result.json"))}
    checks = {"R1": r1, "R2": r2, "R3": short(json.loads((P38 / "runs/r3_result.json").read_text())),
              "R4": short({k: v for k, v in json.loads(sorted((P38 / "runs").glob("r4_*.json"))[-1].read_text()).items()
                           if k != "files"}),
              "R6": {"ids": json.loads((P38 / "runs/r6_ids.json").read_text())["ids"], "identity": r6_identity()},
              "R7": short(json.loads((P38 / "runs/r7_result.json").read_text())),
              "E38_D1": short(json.loads((P38 / "runs/e38_d1/e38_d1_result.json").read_text())),
              "K4_MED_cap": short(json.loads((P38 / "runs/k4med_cap.json").read_text()))}
    (res / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")

    # blindness audits: file lists only (bash bodies stay private: they carry the judges' own notes)
    reviews = {
        "w4_v1_checker": "the marker is only a source-name value inside the JSON the agent wrote; its reads are its allowed input",
        "w4_repl_reviewer": "the flagged string is prose inside the agent's own review JSON; no such file exists",
        "w6_judge_": "the flags come from the judge's own directories (mkdir of its output dir, or cd into the judge dir with "
                     "relative paths to its own two files); no other file opened, listed or named in any bash command",
    }
    for p in sorted((P38 / "blindness").glob("*.json")):
        d = json.loads(p.read_text())
        rel = lambda x: SCRATCH.sub("<session scratchpad>", x).replace(str(P38) + "/", "").replace(HOME, "~")  # noqa: E731
        note = next((v for k, v in reviews.items() if d["label"] == k or (k.endswith("_") and d["label"].startswith(k))), None)
        if d["verdict"] != "CLEAN" and note is None:
            sys.exit(f"{d['label']}: REVIEW without a recorded review")
        pub = {"label": d["label"], "allowed": [rel(a) for a in d["allowed"]], "reads": [rel(x) for x in d["reads"]],
               "writes": [rel(x) for x in d["writes"]], "n_searches": len(d["searches"]), "n_bash": len(d["bash"]),
               "other_tools": d["other_tools"], "flags": [rel(f) for f in d["flags"]], "audit_verdict": d["verdict"],
               "reviewed_verdict": "CLEAN" if d["verdict"] == "CLEAN" else "CLEAN after manual review",
               **({"review": note} if d["verdict"] != "CLEAN" else {})}
        (res / "blindness" / p.name).write_text(json.dumps(pub, indent=1) + "\n")
    print(json.dumps({"rows": len(lines), "redeal": len(red), "blocks": len(export["judged"]),
                      "audits": len(list((res / "blindness").glob("*.json"))), "verdict": line}))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
