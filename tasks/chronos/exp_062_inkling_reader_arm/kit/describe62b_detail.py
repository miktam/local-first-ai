#!/usr/bin/env python3
"""exp_062 Addendum B: detail behind the registered figures. POST HOC: written 2026-11-23 after the arms were scored,
to say where the differences are. Nothing here was registered and nothing here decides anything. Counts only, from the
public rows and the scorer's own verdicts.

    python3.14 -I kit/describe62b_detail.py runs_b/<ts>          writes runs_b/<ts>/detail.json

  wrong            every wrong decision of every arm: id, tier, case, the scorer's reasons
  hb1_by_tier      the discordant pairs of G and I per tier, and the sign test on the 40 single jobs of tiers 2 and 3
                   alone (the 36 stream decisions sit in three conversations and are not independent of each other)
  missing          the lookups the scorer found missing, per arm, by tool name
  written          reasoning written per decision at each pass count: the registered median, the same median over the
                   decisions every research arm got right with no turn cut at the cap, turns per decision, tokens
                   generated per decision, and the reasoning of the turns cut at the cap
  v2_speed         arm V2's generation speed for its first 15 decisions (before the power cut) and the rest
  pass_pairs       each pass count against 1 pass and against the count before it: decisions only one of the two got
                   right, with the two-sided sign test; pass_pairs_single_jobs is the same against 1 pass on the 60
                   single jobs alone
  cut_turns        (only with the private folder as a second argument) for every turn cut at the cap: how many
                   sentences its reasoning has and how many of them are distinct. A low share of distinct sentences
                   means the turn was repeating itself. No text is printed.
"""
import json
import re
import statistics
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import score62b as S  # noqa: E402

ARMS = ("G", "I", "V1", "V2", "V3", "V4", "V5")


def med(xs):
    return statistics.median(xs) if xs else None


def main():
    run = Path(sys.argv[1])
    meta = json.loads((run / "meta.json").read_text())
    data = json.loads((KIT / meta.get("jobs_file", "test62b.json")).read_text())
    jobs = {**{j["id"]: j for j in data["jobs"]}, **{e["id"]: e for s in data["streams"] for e in s["events"]}}
    res = {a: S.rows(run / a / "results.jsonl") for a in ARMS}
    verd = {a: S.summary(jobs, res[a], data)[1] for a in ARMS}
    out = {"note": "post hoc; see the docstring of kit/describe62b_detail.py"}

    out["wrong"] = {a: [{"id": i, "tier": str(jobs[i]["tier"]), "case": jobs[i]["case"], "why": v["why"]}
                        for i, v in verd[a].items() if not v["right"]] for a in ARMS}

    by = {}
    for i in jobs:
        g, k = verd["G"][i]["right"], verd["I"][i]["right"]
        t = by.setdefault(str(jobs[i]["tier"]), {"only_I_right": 0, "only_G_right": 0, "both_right": 0, "neither": 0})
        t["only_I_right" if k and not g else "only_G_right" if g and not k else "both_right" if g else "neither"] += 1
    singles = [by[t] for t in ("2", "3")]
    a, b = sum(t["only_I_right"] for t in singles), sum(t["only_G_right"] for t in singles)
    out["hb1_by_tier"] = {**by, "single_jobs_of_tiers_2_and_3": {"only_I_right": a, "only_G_right": b, "sign_p_two_sided": round(S.sign_p(a, b), 4)}}

    miss = {}
    for arm in ARMS:
        m = {}
        for v in verd[arm].values():
            for w in v["why"]:
                if w.startswith("missing lookup"):
                    for name in json.loads(w.split(":", 1)[1].strip().replace("'", '"')):
                        m[name] = m.get(name, 0) + 1
        miss[arm] = m
    out["missing"] = miss

    vs = [a for a in ARMS if a.startswith("V")]
    capped = {a: {i for i, r in res[a].items() if any(t["finish_reason"] == "length" for t in r["turns"])} for a in ARMS}
    common = [i for i in jobs if all(verd[a][i]["right"] and i not in capped[a] for a in vs)]
    written = {"decisions_right_and_uncut_in_every_research_arm": len(common)}
    for arm in ARMS:
        per = {i: sum(t["reasoning_chars"] for t in r["turns"]) for i, r in res[arm].items()}
        toks = {i: sum(t["completion_tokens"] for t in r["turns"]) for i, r in res[arm].items()}
        cut = [t for r in res[arm].values() for t in r["turns"] if t["finish_reason"] == "length"]
        written[arm] = {
            "median_reasoning_chars_per_decision": med(list(per.values())),
            "the_same_over_the_common_decisions": med([per[i] for i in common]) if arm in vs else None,
            "median_reasoning_chars_right_decisions": med([per[i] for i in per if verd[arm][i]["right"]]),
            "median_turns_per_decision": med([len(r["turns"]) for r in res[arm].values()]),
            "turns": sum(len(r["turns"]) for r in res[arm].values()),
            "median_tokens_generated_per_decision": med(list(toks.values())),
            "tokens_generated": sum(toks.values()),
            "turns_cut_at_the_cap": len(cut),
            "reasoning_chars_of_those_turns": sorted(t["reasoning_chars"] for t in cut),
            "content_chars_of_those_turns": sorted(t["content_chars"] for t in cut),
            "tool_calls_in_those_turns": sum(t.get("tool_calls") or 0 for t in cut),
            "decisions_with_a_cut_turn": sorted(capped[arm]),
            "of_them_ended_final": sorted(i for i in capped[arm] if res[arm][i]["end"] == "final"),
        }
    out["written"] = written

    order = list(res["V2"])
    def speed(ids):
        x = [t["tps"] for i in ids for t in res["V2"][i]["turns"] if t.get("tps") and t["completion_tokens"] >= 100]
        return round(statistics.median(x), 1) if x else None
    out["v2_speed"] = {"first_15_decisions": speed(order[:15]), "the_other_81": speed(order[15:]),
                       "minutes_first_15": round(sum(res["V2"][i]["seconds"] for i in order[:15]) / 60, 1),
                       "minutes_the_other_81": round(sum(res["V2"][i]["seconds"] for i in order[15:]) / 60, 1)}

    pairs = {}
    for x, y in (("V1", "V2"), ("V1", "V3"), ("V1", "V4"), ("V1", "V5"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5")):
        a = sum(1 for i in jobs if verd[y][i]["right"] and not verd[x][i]["right"])
        b = sum(1 for i in jobs if verd[x][i]["right"] and not verd[y][i]["right"])
        pairs[f"{y}_against_{x}"] = {f"only_{y}_right": a, f"only_{x}_right": b, "sign_p_two_sided": round(S.sign_p(a, b), 4)}
    out["pass_pairs"] = pairs
    single = [i for i in jobs if str(jobs[i]["tier"]) != "stream"]      # added 2026-11-23: the 60 jobs that are their own conversation
    alone = {}
    for x, y in (("V1", "V2"), ("V1", "V3"), ("V1", "V4"), ("V1", "V5")):
        a = sum(1 for i in single if verd[y][i]["right"] and not verd[x][i]["right"])
        b = sum(1 for i in single if verd[x][i]["right"] and not verd[y][i]["right"])
        alone[f"{y}_against_{x}"] = {f"only_{y}_right": a, f"only_{x}_right": b, "sign_p_two_sided": round(S.sign_p(a, b), 4)}
    out["pass_pairs_single_jobs"] = {"jobs": len(single), **alone}

    if len(sys.argv) > 2:
        private, sent = Path(sys.argv[2]).expanduser(), re.compile(r"[.!?]+(?:\s|$)|\n+")
        cuts = {}
        for arm in ARMS:
            want = {(i, t["turn"]) for i, r in res[arm].items() for t in r["turns"] if t["finish_reason"] == "length"}
            rows = []
            for l in open(private / f"reasoning-{arm}.jsonl", encoding="utf-8"):
                r = json.loads(l)
                if (r["id"], r["turn"]) in want:
                    ss = [x.strip().lower() for x in sent.split(r["reasoning"]) if x.strip()]
                    rows.append({"id": r["id"], "turn": r["turn"], "sentences": len(ss), "distinct": len(set(ss)),
                                 "distinct_share": round(len(set(ss)) / len(ss), 2) if ss else None})
            cuts[arm] = rows
        out["cut_turns"] = cuts

    (run / "detail.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "wrong"}, indent=1))
    for arm in ARMS:
        print(arm, [(w["id"], w["case"], w["why"]) for w in out["wrong"][arm]])


if __name__ == "__main__":
    main()
