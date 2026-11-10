#!/usr/bin/env python3
"""exp_056: the first run beside the re-run (Amendment 4), from the published records only.

Written for the re-run's result on 2026-11-10, after the run; it is not part of the kit pinned in Amendment 3. It
reads results.jsonl, meta.json, score.json and the cut audit logs of the four run folders, and exp_055's two
records (its first run of 2026-10-25 and the re-run with the fixed check). Measures only: bands, scores, points,
the model step's closed labels, indicator ids, hashes, seconds and request counts. No message or payload text is
read or printed.

  python3.14 kit/compare56.py            # from tasks/chronos/exp_056_triage_injection
"""
import argparse
import json
import re
import statistics as st
from collections import Counter
from datetime import datetime
from pathlib import Path

ORDER = {"benign": 0, "suspicious": 1, "likely-scam": 2}
FLAGGED = {"suspicious", "likely-scam"}
MODEL_STEPS = ("bases1", "variants", "bases2")
STEPS = ("bases1", "variants", "bases2", "nollm", "dead")
URL = re.compile(r"https?://127\.0\.0\.1:\d+/api/([a-z]+)")


def jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def records(run):
    return {r["key"]: r for r in jsonl(run / "results.jsonl")}


def verdict(item):
    if item.get("void"):
        return "void"
    return "holds" if item.get("holds") is True else "does not hold"


def chat_calls(run, step):
    """Model chat requests in one step's audit log (the URL survives the cut to 160 characters)."""
    n = 0
    for r in jsonl(run / "audit" / f"{step}.jsonl"):
        if r.get("event") == "urllib.Request":
            m = URL.search(str((r.get("args") or [""])[0]))
            n += bool(m and m.group(1) == "chat")
    return n


def duration(run):
    m = json.loads((run / "meta.json").read_text())
    a, b = (datetime.fromisoformat(m[k].replace("Z", "+00:00")) for k in ("started", "finished"))
    s = int((b - a).total_seconds())
    return f"{m['started']} to {m['finished']} ({s // 60} min {s % 60} s)"


def mean_median(xs):
    return f"mean {st.mean(xs):.2f} s, median {st.median(xs):.2f} s" if xs else "-"


def same(a, b):
    return (a["band"], a["score"], a["persona_points"], a["leaf_output"], a["indicators"]) == \
           (b["band"], b["score"], b["persona_points"], b["leaf_output"], b["indicators"])


def exp056(first, rerun, label):
    old, new = records(first), records(rerun)
    so, sn = (json.loads((r / "score.json").read_text()) for r in (first, rerun))
    print(f"== {label}: {first.name} (first run) against {rerun.name} (re-run)")
    print(f"   times (UTC): first {duration(first)}; re-run {duration(rerun)}")
    for item in ("H1", "H2", "H5", "C2", "C3"):
        o, n = so[item], sn[item]
        cnt = lambda x: f"{len(x['violations'])} of {x['n']}" if "n" in x else ""
        print(f"   {item}: first {cnt(o)} {verdict(o)}; re-run {cnt(n)} {verdict(n)}")
    print(f"   failing: first {so['failing']}; re-run {sn['failing']}")

    print("   records: re-run keys equal to the first run's:", set(old) == set(new),
          "| sha256 equal per key:", sum(1 for k in new if old.get(k, {}).get("sha256") == new[k]["sha256"]),
          "of", len(new))
    digests = Counter(r["model"]["digest"] for r in new.values() if r.get("model"))
    print("   model records:", sum(digests.values()), "digests:", dict(digests))
    mo, mn = (json.loads((r / "meta.json").read_text()) for r in (first, rerun))
    print("   meta tools: first", mo["tools"], "re-run", mn["tools"])
    print("   meta kit, bases, manifest, payload hashes, hook equal to the first run's:",
          all(mo[k] == mn[k] for k in ("kit", "bases", "manifest_sha256", "payloads", "hook_sha256")))

    print("   bases, re-run against the first run (band, score, points, labels, indicators):")
    for step in ("bases1", "bases2"):
        keys = [k for k in new if new[k]["step"] == step]
        eq = sum(1 for k in keys if same(old[k], new[k]))
        print(f"     {step}: {eq} of {len(keys)} identical")
    for lab, recs in (("first", old), ("re-run", new)):
        sus = sorted(r["id"] for r in recs.values() if r["step"] == "bases1" and r["kind"] == "genuine"
                     and r["band"] in FLAGGED)
        print(f"     genuine bases flagged ({lab}): {len(sus)} {sus}")

    print("   variants against their base (bases1 of the same run):")
    for lab, recs in (("first", old), ("re-run", new)):
        c = Counter()
        bands_up = []
        for r in recs.values():
            if r["step"] != "variants":
                continue
            b = recs["bases1:" + r["base"]]
            d = ORDER[r["band"]] - ORDER[b["band"]]
            c["band up" if d > 0 else "band down" if d < 0 else "band equal"] += 1
            c["score up" if r["score"] > b["score"] else "score down" if r["score"] < b["score"] else "score equal"] += 1
            c["points up" if r["persona_points"] > b["persona_points"] else
              "points down" if r["persona_points"] < b["persona_points"] else "points equal"] += 1
            if d > 0:
                bands_up.append(r["id"])
            new_ind = sorted(set(r["indicators"]) - set(b["indicators"]))
            if new_ind:
                c["indicator added: " + ",".join(new_ind)] += 1
            if set(b["indicators"]) - set(r["indicators"]):
                c["indicator lost"] += 1
        print(f"     {lab}: {dict(sorted(c.items()))}")
        print(f"       band up: {sorted(bands_up)}")
    down_first = sorted(r["id"] for r in old.values() if r["step"] == "variants"
                        and r["persona_points"] < old["bases1:" + r["base"]]["persona_points"])
    back = sum(1 for k, r in new.items() if r["step"] == "variants" and r["id"] in down_first
               and r["persona_points"] == new["bases1:" + r["base"]]["persona_points"])
    print(f"     the {len(down_first)} variants whose points fell in the first run: in the re-run {back} have "
          f"their base's points")
    unchanged = sum(1 for k, r in new.items() if r["step"] == "variants" and same(old[k], r))
    print(f"     variants identical to the first run's record: {unchanged} of "
          f"{sum(1 for r in new.values() if r['step'] == 'variants')}")
    up = sorted(r["id"] for r in new.values() if r["step"] == "variants"
                and r["score"] > new["bases1:" + r["base"]]["score"])
    print(f"     re-run variants scoring above their base: {len(up)}")

    print("   cost per message (model calls from the audit logs' chat requests; seconds = the runner's wall time "
          "of one triage process):")
    for step in STEPS:
        no = sum(1 for r in old.values() if r["step"] == step)
        nn = sum(1 for r in new.values() if r["step"] == step)
        co, cn = chat_calls(first, step), chat_calls(rerun, step)
        so_ = [r["seconds"] for r in old.values() if r["step"] == step]
        sn_ = [r["seconds"] for r in new.values() if r["step"] == step]
        print(f"     {step:8} calls {co}/{no} = {co / no:.2f} -> {cn}/{nn} = {cn / nn:.2f}; "
              f"seconds {mean_median(so_)} -> {mean_median(sn_)}")
    so_ = [r["seconds"] for r in old.values() if r["step"] in MODEL_STEPS]
    sn_ = [r["seconds"] for r in new.values() if r["step"] in MODEL_STEPS]
    co = sum(chat_calls(first, s) for s in MODEL_STEPS)
    cn = sum(chat_calls(rerun, s) for s in MODEL_STEPS)
    print(f"     all model-step runs: calls {co}/{len(so_)} = {co / len(so_):.2f} -> {cn}/{len(sn_)} = "
          f"{cn / len(sn_):.2f}; seconds {mean_median(so_)} -> {mean_median(sn_)}; "
          f"total {sum(so_):.0f} s -> {sum(sn_):.0f} s")


def exp055(first, rerun, manifest):
    man = json.loads(Path(manifest).read_text())
    print(f"== exp_055's arms: {first} (first run) against {rerun} (re-run with the fixed check)")
    for arm in ("R", "P"):
        a = {r["id"]: r for r in jsonl(first / arm / "results.jsonl")}
        b = {r["id"]: r for r in jsonl(rerun / arm / "results.jsonl")}
        gen = [i for i in sorted(man) if man[i]["kind"] == "benign"]
        lur = [i for i in sorted(man) if man[i]["kind"] == "lure"]
        fa = [{i for i in gen if x[i]["band"] in FLAGGED} for x in (a, b)]
        ca = [{i for i in lur if x[i]["band"] in FLAGGED} for x in (a, b)]
        print(f"   {arm}: lures caught {len(ca[0])} of {len(lur)} -> {len(ca[1])} of {len(lur)}; "
              f"genuine flagged {len(fa[0])} of {len(gen)} -> {len(fa[1])} of {len(gen)}")
        print(f"      genuine newly flagged {sorted(fa[1] - fa[0])}; no longer flagged {sorted(fa[0] - fa[1])}")
        print(f"      lures newly caught {sorted(ca[1] - ca[0])}; no longer caught {sorted(ca[0] - ca[1])}")
        moved = [(i, man[i]["kind"], a[i]["band"], b[i]["band"], a[i]["score"], b[i]["score"],
                  a[i]["persona_points"], b[i]["persona_points"]) for i in sorted(man) if a[i]["band"] != b[i]["band"]]
        print(f"      band changed on {len(moved)}: (id, kind, band, band, score, score, points, points) {moved}")
        pts = Counter("up" if (b[i]["persona_points"] or 0) > (a[i]["persona_points"] or 0) else
                      "down" if (b[i]["persona_points"] or 0) < (a[i]["persona_points"] or 0) else "equal" for i in man)
        ups = sorted(i for i in man if (b[i]["persona_points"] or 0) > (a[i]["persona_points"] or 0))
        print(f"      model-step points {dict(pts)}; up on {ups}")
        print(f"      seconds per message {mean_median([a[i]['seconds'] for i in man])} -> "
              f"{mean_median([b[i]['seconds'] for i in man])}; total {sum(a[i]['seconds'] for i in man):.0f} s -> "
              f"{sum(b[i]['seconds'] for i in man):.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--e55", default="../exp_055_exposure_review_lures")
    a = ap.parse_args()
    runs, e55 = Path(a.runs), Path(a.e55)
    exp056(runs / "six", runs / "six-r2", "six texts")
    exp056(runs / "n1", runs / "n1-r2", "neutral control N1")
    first55 = e55 / "runs" / (e55 / "runs/CURRENT").read_text().strip()
    exp055(first55, runs / "exp055-views", e55 / "data/MANIFEST.json")


if __name__ == "__main__":
    main()
