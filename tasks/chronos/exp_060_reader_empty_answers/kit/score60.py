#!/usr/bin/env python3
"""exp_060: score a run against exp_055's labels and its G baseline; controls C1–C4, hypotheses H1–H4, the decision.

    python3.14 kit/score60.py runs/<ts>            writes runs/<ts>/score.json and prints the verdicts

Reads exp_055's MANIFEST and G results in place. Metrics (catch, false alarms, balanced accuracy, attribution,
intervals, breakdowns) are exp_055's own functions, imported from its kit.
"""
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXP55 = HERE.parent / "exp_055_exposure_review_lures"
sys.path.insert(0, str(EXP55 / "kit"))
import score55  # noqa: E402

G_RUN = EXP55 / "runs/20261025T095614Z"
CHANGED = {"X011", "X067", "X085", "X114", "X148", "X161", "X174"}   # the published copy's .example domains
OPUS_BA = 0.950
BASELINE = {"balanced_accuracy": 0.882, "attribution_rate": 0.85, "answered": 198}
BUDGETS = (2000, 4000, 6000, 7000)
ARMS = ("T", "F", "C")


def rows(path: Path) -> dict:
    """id -> last row (a resumed run may repeat an id)."""
    out = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def p95(xs):
    xs = sorted(xs)
    return xs[max(0, math.ceil(0.95 * len(xs)) - 1)] if xs else None


def timing(res: dict) -> dict:
    secs = [r["seconds"] for r in res.values() if r.get("seconds") is not None]
    errors = [i for i, r in res.items() if r.get("error")]
    timeouts = [i for i in errors if "timed out" in res[i]["error"].lower() or "timeout" in res[i]["error"].lower()]
    return {"n": len(secs), "median_s": round(statistics.median(secs), 1) if secs else None,
            "p95_s": round(p95(secs), 1) if secs else None, "max_s": round(max(secs), 1) if secs else None,
            "errors": len(errors), "timeouts": len(timeouts), "error_ids": sorted(errors)}


def agreement(a: dict, b: dict, ids) -> dict:
    ids = [i for i in ids if i in a and i in b]
    same = [i for i in ids if a[i].get("band") == b[i].get("band")]
    return {"n": len(ids), "same": len(same), "rate": round(len(same) / len(ids), 3) if ids else None,
            "differ": sorted(set(ids) - set(same))}


def counts(res: dict, key: str) -> dict:
    out = {}
    for r in res.values():
        k = str(r.get(key))
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def quantiles(xs) -> dict:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return {"n": 0}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "min": xs[0], "p50": q(0.5), "p90": q(0.9), "max": xs[-1]}


def lure_age(m):
    if not m["domains"]:
        return "no domain"
    return "old" if any(c < "2026-01-01" for c in m["domains"].values()) else "new"


def genuine_age(m):
    if not m["domains"]:
        return "no domain"
    return "young" if any(c >= "2026-06-01" for c in m["domains"].values()) else "old"


def age_class(m):
    return lure_age(m) if m["kind"] == "lure" else genuine_age(m)


def correct(m, band) -> bool:
    flagged = band in score55.FLAGGED
    return flagged if m["kind"] == "lure" else not flagged


def audit_check(path: Path) -> dict:
    """C3: every connect goes to 127.0.0.1:11434; every resolution is of the loopback name; subprocesses are ollama."""
    if not path.exists():
        return {"present": False, "pass": False, "note": "no audit log"}
    connects, resolves, spawns, bad = 0, 0, 0, []
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("kind") == "getaddrinfo":
            resolves += 1
            if str(r.get("host")) not in ("127.0.0.1", "localhost"):
                bad.append(f"resolve {r.get('host')}")
        elif r.get("kind") == "audit":
            ev, args = r.get("event"), r.get("args") or []
            if ev == "socket.connect":
                connects += 1
                addr = args[1] if len(args) > 1 else ""
                if not re.search(r"'127\.0\.0\.1',\s*11434\)", addr):
                    bad.append(f"connect {addr[:120]}")
            elif ev in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"):
                spawns += 1
                if "ollama" not in " ".join(args) and "sw_vers" not in " ".join(args):
                    bad.append(f"{ev} {' '.join(args)[:120]}")
    return {"present": True, "connects": connects, "resolutions": resolves, "spawns": spawns, "violations": bad[:20],
            "pass": connects > 0 and not bad}


def daemon_ctx(path: Path, model="gemma4:26b"):
    if not path.exists():
        return None, None
    d = json.loads(path.read_text())
    for m in (d.get("api_ps") or {}).get("models") or []:
        if m.get("name") == model or m.get("model") == model:
            return m.get("context_length"), m.get("size_vram")
    return None, None


def main():
    run = Path(sys.argv[1])
    man = json.loads((EXP55 / "data/MANIFEST.json").read_text())
    G = rows(G_RUN / "G" / "results.jsonl")
    g_empty = sorted(i for i, r in G.items() if r.get("band") is None)
    unchanged_answered = sorted(i for i, r in G.items() if r.get("band") is not None and i not in CHANGED)
    unchanged_empty = [i for i in g_empty if i not in CHANGED]
    arms = {a: rows(run / a / "results.jsonl") for a in ARMS}
    arms = {a: r for a, r in arms.items() if r}
    pass1 = rows(run / "F" / "pass1.jsonl")
    score = {"run": str(run), "baseline": {"G_answered": len(G) - len(g_empty), "G_empty": len(g_empty),
                                             "unchanged_answered": len(unchanged_answered),
                                             "unchanged_empty": len(unchanged_empty), **BASELINE},
             "arms": {}, "controls": {}, "hypotheses": {}, "descriptive": {}}

    # per-arm metrics, timing, counters
    met = {}
    for a, res in arms.items():
        m = score55.arm_metrics(res, man)
        met[a] = m
        score["arms"][a] = {k: v for k, v in m.items() if not k.startswith("_")}
        score["arms"][a]["answered"] = len(res) - m["unanswered"]
        score["arms"][a]["timing"] = timing(res)
        score["arms"][a]["done_reason"] = counts(res, "done_reason")
        score["arms"][a]["eval_count"] = quantiles(r.get("eval_count") for r in res.values() if r.get("band"))
        score["arms"][a]["agreement_with_G_on_unchanged_answered"] = agreement(res, G, unchanged_answered)
        score["arms"][a]["opus_gap"] = round(OPUS_BA - m["balanced_accuracy"], 3)
        if a == "F":
            score["arms"][a]["fallbacks"] = sum(1 for r in res.values() if r.get("fallback"))

    # C1: the baseline reproduces (F pass 1 against exp_055's G)
    if pass1:
        ag = agreement(pass1, G, unchanged_answered)
        empties_again = [i for i in unchanged_empty if i in pass1 and pass1[i].get("band") is None]
        score["controls"]["C1_baseline_reproduces"] = {
            "agreement": ag, "unchanged_empties_empty_again": len(empties_again), "of": len(unchanged_empty),
            "pass1_empty_ids": sorted(i for i, r in pass1.items() if r.get("band") is None),
            "pass": (ag["rate"] or 0) >= 0.95 and len(empties_again) >= 18}
        # C2: the wall is the cause
        emp = {i: r for i, r in pass1.items() if r.get("band") is None}
        at_wall = {i: (r.get("done_reason") == "length" or
                       ((r.get("prompt_eval_count") or 0) + (r.get("eval_count") or 0)) >= 8000) for i, r in emp.items()}
        score["controls"]["C2_wall_is_the_cause"] = {
            "empty_pass1": len(emp), "at_wall": sum(at_wall.values()),
            "not_at_wall": sorted(i for i, ok in at_wall.items() if not ok),
            "tokens": {i: {"done_reason": r.get("done_reason"), "prompt_eval_count": r.get("prompt_eval_count"),
                           "eval_count": r.get("eval_count"), "error": r.get("error")} for i, r in emp.items()},
            "pass": bool(emp) and all(at_wall.values())}
    # C3: nothing leaves the machine
    c3 = {a: audit_check(run / a / "audit.jsonl") for a in arms}
    score["controls"]["C3_loopback_only"] = {"per_arm": c3, "pass": bool(c3) and all(v["pass"] for v in c3.values())}
    # C4: arm C ran at 32K
    if "C" in arms:
        ctx_during, vram_during = daemon_ctx(run / "C" / "daemon_during.json")
        ctx_restored, _ = daemon_ctx(run / "C" / "daemon_restored.json")
        score["controls"]["C4_arm_C_at_32K"] = {"context_during": ctx_during, "context_restored": ctx_restored,
                                                "pass": ctx_during == 32768}

    # hypotheses, per arm
    H = {}
    for a, m in met.items():
        t = score["arms"][a]["timing"]
        ag = score["arms"][a]["agreement_with_G_on_unchanged_answered"]
        h = {"H1_loss_closed": {"answered": score["arms"][a]["answered"], "pass": score["arms"][a]["answered"] >= 218},
             "H3_no_worse_than_G": {"balanced_accuracy": m["balanced_accuracy"], "attribution": m["attribution_rate"],
                                    "pass": m["balanced_accuracy"] >= 0.882 and (m["attribution_rate"] or 0) >= 0.85},
             "H4_affordable": {"median_s": t["median_s"], "p95_s": t["p95_s"], "errors": t["errors"],
                               "pass": t["median_s"] is not None and t["median_s"] <= 45 and t["p95_s"] <= 180
                               and t["errors"] == 0}}
        if a == "C":
            h["H2_moves_nothing_it_did_not_have_to"] = {"agreement": ag["rate"], "differ": ag["differ"],
                                                        "pass": (ag["rate"] or 0) >= 0.95}
        elif a == "F":
            h["H2_moves_nothing_it_did_not_have_to"] = {"inherited_from": "C1", "agreement": ag["rate"],
                                                        "pass": score["controls"].get("C1_baseline_reproduces", {}).get("pass")}
        else:
            h["H2_moves_nothing_it_did_not_have_to"] = {"descriptive_only": True, "agreement": ag["rate"]}
        H[a] = h
    score["hypotheses"] = H

    # decision rule
    def eligible(a):
        h = H[a]
        ok = h["H1_loss_closed"]["pass"] and h["H3_no_worse_than_G"]["pass"] and h["H4_affordable"]["pass"]
        if a == "C":
            ok = ok and bool(h["H2_moves_nothing_it_did_not_have_to"]["pass"])
        return ok
    cands = [a for a in met if eligible(a)]
    chosen, why = None, "no arm passes H1, H3 and H4 (and H2 for C): A1 is registered with the loss declared"
    if cands:
        best = max(met[a]["balanced_accuracy"] for a in cands)
        near = [a for a in cands if met[a]["balanced_accuracy"] >= best - 0.01]
        chosen = min(near, key=lambda a: score["arms"][a]["timing"]["median_s"])
        why = (f"eligible {cands}; highest balanced accuracy {best}; within 0.01: {near}; "
               f"chosen by lower median wall-clock" if len(near) > 1 else f"eligible {cands}; highest balanced accuracy")
    score["decision"] = {"eligible": cands, "chosen": chosen, "why": why,
                         "thinking_earned_its_place": None if chosen is None else chosen != "T"}

    # descriptive
    fate = {}
    for i in g_empty:
        m = man[i]
        fate[i] = {"kind": m["kind"], "age": age_class(m), "ask": m["ask"], "changed_in_published_copy": i in CHANGED}
        for a, res in arms.items():
            b = res.get(i, {}).get("band")
            fate[i][a] = {"band": b, "correct": correct(m, b) if b else None}
    score["descriptive"]["fate_of_the_22"] = fate
    score["descriptive"]["recovered_summary"] = {
        a: {"answered": sum(1 for i in g_empty if arms[a].get(i, {}).get("band")),
            "correct": sum(1 for i in g_empty if arms[a].get(i, {}).get("band") and correct(man[i], arms[a][i]["band"])),
            "lures_caught": sum(1 for i in g_empty if man[i]["kind"] == "lure" and arms[a].get(i, {}).get("band") in score55.FLAGGED),
            "genuine_flagged": sum(1 for i in g_empty if man[i]["kind"] == "benign" and arms[a].get(i, {}).get("band") in score55.FLAGGED)}
        for a in arms}
    if pass1:
        answered = [r["eval_count"] for r in pass1.values() if r.get("band") and r.get("eval_count") is not None]
        score["descriptive"]["budget_curve_from_F_pass1"] = {
            "answered_with_counts": len(answered),
            **{f"within_{b}": {"share": round(sum(1 for x in answered if x <= b) / len(answered), 3) if answered else None,
                               "a_budget_here_would_have_cut": sum(1 for x in answered if x > b)} for b in BUDGETS}}
    if "C" in arms:
        former = {i: arms["C"][i] for i in g_empty if i in arms["C"]}
        score["descriptive"]["C_former_empties"] = {
            i: {"band": r.get("band"), "done_reason": r.get("done_reason"), "prompt_eval_count": r.get("prompt_eval_count"),
                "eval_count": r.get("eval_count"), "seconds": r.get("seconds"),
                "beyond_8192": ((r.get("prompt_eval_count") or 0) + (r.get("eval_count") or 0)) > 8192}
            for i, r in former.items()}
        ctx_before, vram_before = daemon_ctx(run / "C" / "daemon_before.json")
        ctx_during, vram_during = daemon_ctx(run / "C" / "daemon_during.json")
        score["descriptive"]["memory_gemma4_26b"] = {"before": {"context": ctx_before, "size_vram_gb": round(vram_before / 1e9, 2) if vram_before else None},
                                                    "during_C": {"context": ctx_during, "size_vram_gb": round(vram_during / 1e9, 2) if vram_during else None}}
    pairs = [("T", "F"), ("T", "C"), ("F", "C")]
    dis = {}
    for x, y in pairs:
        if x in arms and y in arms:
            ids = [i for i in man if i in arms[x] and i in arms[y] and arms[x][i].get("band") and arms[y][i].get("band")]
            d = [i for i in ids if arms[x][i]["band"] != arms[y][i]["band"]]
            dis[f"{x}_vs_{y}"] = {"compared": len(ids), "differ": len(d),
                                  "by_kind": counts({i: man[i] for i in d}, "kind"),
                                  "by_age": {k: sum(1 for i in d if age_class(man[i]) == k) for k in ("new", "old", "young", "no domain")},
                                  "ids": d}
    score["descriptive"]["disagreements"] = dis
    score["descriptive"]["breakdowns"] = {a: {
        "lure_by_domain_age": score55.breakdown(r, man, lambda m: lure_age(m) if m["kind"] == "lure" else None),
        "lure_by_ask": score55.breakdown(r, man, lambda m: m["ask"] if m["kind"] == "lure" else None),
        "genuine_by_domain_age": score55.breakdown(r, man, lambda m: genuine_age(m) if m["kind"] == "benign" else None),
        "by_channel_and_kind": score55.breakdown(r, man, lambda m: f"{m['channel']}/{m['kind']}")} for a, r in arms.items()}

    (run / "score.json").write_text(json.dumps(score, indent=2, ensure_ascii=False, default=str) + "\n")
    mark = lambda ok: "PASS" if ok else ("FAIL" if ok is False else "n/a")
    print(f"exp_060 score — {run}")
    for k, v in score["controls"].items():
        print(f"  {k:28s} {mark(v.get('pass'))}")
    for a in arms:
        s = score["arms"][a]
        print(f"  arm {a}: answered {s['answered']}/220  catch {s['catch_rate']}  false alarms {s['false_alarm_rate']}  "
              f"BA {s['balanced_accuracy']}  attribution {s['attribution_rate']}  median {s['timing']['median_s']} s  "
              f"p95 {s['timing']['p95_s']} s  errors {s['timing']['errors']}  agree-with-G {s['agreement_with_G_on_unchanged_answered']['rate']}")
        for h, v in H[a].items():
            print(f"      {h:36s} {mark(v.get('pass'))}")
    print(f"  decision: {score['decision']['chosen']}  ({score['decision']['why']})")
    if "recovered_summary" in score["descriptive"]:
        print(f"  the 22: {score['descriptive']['recovered_summary']}")


if __name__ == "__main__":
    main()
