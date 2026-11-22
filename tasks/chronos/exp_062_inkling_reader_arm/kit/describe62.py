#!/usr/bin/env python3
"""exp_062: the descriptive tables the registration promised and score62.py does not write.

Written on 2026-11-22, AFTER the run and after its score was read. It changes no verdict: it reads
the complete run and the two published baselines, and writes runs/<ts>/describe.json.

    python3.14 -I kit/describe62.py runs/<ts>
"""
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXP55 = HERE.parent / "exp_055_exposure_review_lures"
EXP60 = HERE.parent / "exp_060_reader_empty_answers"
sys.path.insert(0, str(EXP55 / "kit"))
import score55  # noqa: E402

G_RESULTS = EXP55 / "runs/20261025T095614Z/G/results.jsonl"
F_RESULTS = EXP60 / "runs/20261101T152955Z/F/results.jsonl"

# Phrases counted in the model's stated reason ("why") on the genuine messages it flagged. Chosen after reading
# eight of them. The first row is the prompt's own definition of the band "suspicious" ("she should verify it
# through a separate, known channel before acting"), so it shows the model restating the band, not a cause.
REASONS = {
    "restates_the_band_definition": r"(?:separate|another|independent)[, ]+(?:and )?(?:known |trusted )?channel|known channel",
    "says_something_cannot_be_verified": r"unverif|not verifi|(?:cannot|can't|could not|can not|not) be (?:verified|confirmed)"
                                         r"|unable to (?:verify|confirm)|no way to (?:verify|confirm)",
    "says_sender_or_detail_is_not_in_her_footprint": r"footprint|appears nowhere|absent from|not (?:listed|present|found) in",
    "cites_a_domain_date": r"\b20\d\d-\d\d-\d\d\b|registered|created|weeks old|days old|newly",
    "says_unsolicited_or_unknown_sender": r"unsolicited|unknown sender|sender is unknown|unfamiliar",
}


def seq(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def flagged(r) -> bool:
    return (r or {}).get("band") in score55.FLAGGED


def correct(m: dict, r) -> bool:
    return flagged(r) if m["kind"] == "lure" else not flagged(r)


def bands(res: dict, man: dict) -> dict:
    c = Counter((man[i]["kind"], str(r.get("band"))) for i, r in res.items())
    return {f"{k}/{b}": n for (k, b), n in sorted(c.items())}


def lure_age(m):       # exp_055's labels, score55.py main()
    return (("old" if any(c < "2026-01-01" for c in m["domains"].values()) else "new") if m["domains"] else "no domain") \
        if m["kind"] == "lure" else None


def genuine_age(m):    # exp_055's labels, score55.py main()
    return (("young" if any(c >= "2026-06-01" for c in m["domains"].values()) else "old") if m["domains"] else "no domain") \
        if m["kind"] == "benign" else None


def paired_by_kind(a: dict, b: dict, man: dict) -> dict:
    t = Counter()
    for i, m in man.items():
        x, y = correct(m, a.get(i)), correct(m, b.get(i))
        if x != y:
            t[("only_I" if x else "only_other") + "/" + m["kind"]] += 1
    return dict(sorted(t.items()))


def q95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.95 * len(xs)))]


def main():
    run = Path(sys.argv[1])
    man = json.loads((EXP55 / "data/MANIFEST.json").read_text())
    i_seq, f_seq, g_seq = seq(run / "I" / "results.jsonl"), seq(F_RESULTS), seq(G_RESULTS)
    I, F, G = ({r["id"]: r for r in s} for s in (i_seq, f_seq, g_seq))
    assert set(I) == set(man) and len(i_seq) == len(man), "the run is not complete"

    ans = [i for i in I if I[i]["band"] is not None]
    emp = [i for i in I if I[i]["band"] is None]
    lures = [i for i in man if man[i]["kind"] == "lure"]
    gen = [i for i in man if man[i]["kind"] == "benign"]
    g_emp = {i for i in G if G[i].get("band") is None}
    fl_gen = sorted(i for i in gen if flagged(I[i]))

    reasons = {k: sum(1 for i in fl_gen if re.search(p, I[i].get("why") or "", re.I)) for k, p in REASONS.items()}
    cause = [k for k in REASONS if k != "restates_the_band_definition"]
    reasons["none_of_the_four_after_the_first"] = sum(
        1 for i in fl_gen if not any(re.search(REASONS[k], I[i].get("why") or "", re.I) for k in cause))
    f_gen = sorted(i for i in gen if flagged(F[i]))
    rc = sum(I[i].get("reasoning_chars") or 0 for i in I)
    cc = sum(len(I[i].get("content") or "") for i in I)

    out = {
        "run": str(run), "written": "after the run and after the score was read; descriptive only",
        "order": {"same_as_F": [r["id"] for r in i_seq] == [r["id"] for r in f_seq],
                  "same_as_G": [r["id"] for r in i_seq] == [r["id"] for r in g_seq],
                  "sorted_by_id": [r["id"] for r in i_seq] == sorted(I),
                  "manifest_key_order": [r["id"] for r in i_seq] == list(man)},
        "bands_by_class": {"I": bands(I, man), "F": bands(F, man), "G": bands(G, man)},
        "answered_only": {
            "answered": len(ans), "flagged": sum(1 for i in ans if flagged(I[i])),
            "lures_answered": sum(1 for i in lures if i in ans),
            "lures_flagged": sum(1 for i in lures if flagged(I[i])),
            "genuine_answered": sum(1 for i in gen if i in ans),
            "genuine_flagged": len(fl_gen),
            "genuine_called_benign": sum(1 for i in gen if I[i]["band"] == "benign"),
            "genuine_not_flagged_because_no_answer": sum(1 for i in gen if i in emp)},
        "domain_age": {
            "I": {"lure": score55.breakdown(I, man, lure_age), "genuine": score55.breakdown(I, man, genuine_age)},
            "F": {"lure": score55.breakdown(F, man, lure_age), "genuine": score55.breakdown(F, man, genuine_age)}},
        "empties": {
            "n": len(emp), "by_kind": dict(Counter(man[i]["kind"] for i in emp)),
            "finish_reason": dict(Counter(I[i]["finish_reason"] for i in emp)),
            "prompt_plus_completion_tokens": dict(Counter(I[i]["prompt_tokens"] + I[i]["completion_tokens"] for i in emp)),
            "completion_tokens_min_max": [min(I[i]["completion_tokens"] for i in emp), max(I[i]["completion_tokens"] for i in emp)],
            "content_chars": dict(Counter(len(I[i].get("content") or "") for i in emp)),
            "median_seconds": round(statistics.median(I[i]["seconds"] for i in emp), 1),
            "ids": {i: man[i]["kind"] for i in sorted(emp)},
            "G_empties": len(g_emp), "shared_with_G": sorted(g_emp & set(emp)),
            "F_band_on_the_same": dict(Counter(f"{man[i]['kind']}/{F[i].get('band')}" for i in emp))},
        "paired_by_kind": {"vs_F": paired_by_kind(I, F, man), "vs_G": paired_by_kind(I, G, man)},
        "lures_missed": {i: {"band": I[i]["band"], "finish_reason": I[i]["finish_reason"]}
                         for i in sorted(lures) if not flagged(I[i])},
        "genuine_at_likely_scam": sorted(i for i in gen if I[i]["band"] == "likely-scam"),
        "genuine_called_benign": sorted(i for i in gen if I[i]["band"] == "benign"),
        "flagged_genuine_reasons": {
            "n": len(fl_gen), "phrases_counted_in_why": REASONS, "messages_matching": reasons,
            "F_flagged_genuine": len(f_gen), "flagged_by_both_I_and_F": len(set(f_gen) & set(fl_gen)),
            "note": "regex counts over the model's one-sentence reason; a message can match several; "
                    "they describe the wording, not the cause"},
        "time": {
            "answered": {"n": len(ans), "median_s": round(statistics.median(I[i]["seconds"] for i in ans), 1),
                         "p95_s": round(q95([I[i]["seconds"] for i in ans]), 1),
                         "max_s": round(max(I[i]["seconds"] for i in ans), 1)},
            "median_tokens_per_second": round(statistics.median(
                I[i]["predicted_per_second"] for i in I if I[i].get("predicted_per_second")), 1),
            "median_prompt_tokens": statistics.median(I[i]["prompt_tokens"] for i in I),
            "median_completion_tokens": {"answered": statistics.median(I[i]["completion_tokens"] for i in ans),
                                         "lure": statistics.median(I[i]["completion_tokens"] for i in lures),
                                         "genuine": statistics.median(I[i]["completion_tokens"] for i in gen)},
            "total_completion_tokens": sum(I[i]["completion_tokens"] for i in I),
            "hours_spent_on_empties": round(sum(I[i]["seconds"] for i in emp) / 3600, 2)},
        "reasoning": {
            "note": "by characters, not tokens: the server reports one completion-token count for reasoning and answer together",
            "reasoning_chars": rc, "answer_chars": cc, "share_by_chars": round(rc / (rc + cc), 3),
            "median_reasoning_chars": {"all": statistics.median(I[i].get("reasoning_chars") or 0 for i in I),
                                       "answered": statistics.median(I[i].get("reasoning_chars") or 0 for i in ans)}},
    }
    (run / "describe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("order", "bands_by_class", "answered_only", "domain_age", "paired_by_kind",
                                           "flagged_genuine_reasons", "time", "reasoning")}, indent=1, ensure_ascii=False))
    e = out["empties"]
    print("empties:", {k: e[k] for k in e if k != "ids"})


if __name__ == "__main__":
    main()
