"""exp_048: the middle-sized decision model.

exp_047 found that Clef 27B, behind rev 4.1's unchanged questions and rules, lifts the weak triage fields over their bars,
at 13.5 s and 29 GB per enquiry. Does Clef-Flash 9B (Cloudflare's clef-flash, Q8_0, about 9.7 GB, llama.cpp b11711
`/v1/systemone`) keep most of that gain at a fraction of the cost? Only the decision server changes (KEV_URL); everything
else is rev 4.1 as frozen. The Kev-4B and Clef 27B arms are exp_047's recorded predictions on the same 566 enquiries.

usage: flash48.py run dev|test [--limit N] [--max-seconds N]   (resumes; the test pass runs once)
       flash48.py score dev|test
The server: llama-server -m <clef-flash gguf> --port 18448 -c 16384 -b 8192 -ub 8192 (exp_041's settings).
"""
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import leads44 as L  # noqa: E402
from scripts.clef47 import _props, _rows  # noqa: E402   (exp_047, as run: server props; the "discard" = "review" label)

D = ROOT / "data" / "leads46"
RUNS = ROOT / "runs"
SERVER = "http://127.0.0.1:18448"
FLASH_GGUF = Path.home() / "models/exp041/clef-flash/Cloudflare_clef-flash-Q8_0.gguf"
FLASH_SHA_PREFIX = "55c222df"            # exp_041's record of this file
RECORDED = {"kev": "exp047_{set}_kev", "clef27b": "exp047_{set}_clef"}   # exp_047's arms, same set, same rev 4.1
FIELDS = ("intent", "action", "qualified", "timeline", "property_type", "financing", "foreign_buyer")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _guard(set_: str, out: Path) -> dict:
    from scripts.translate46 import SETS as FULL
    if hashlib.sha256((D / f"{set_}.jsonl").read_bytes()).hexdigest() != FULL[set_]:
        sys.exit(f"REFUSED: {set_}.jsonl is not the registered set")
    path, frozen = L.VERSIONS["rev41"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
        sys.exit("REFUSED: kev_knife.py differs from the frozen rev 4.1")
    try:
        props = _props(SERVER)
    except Exception as e:
        sys.exit(f"REFUSED: no decision server at {SERVER} ({type(e).__name__})")
    if not str(props.get("model_path", "")).endswith(FLASH_GGUF.name):
        sys.exit(f"REFUSED: the server on {SERVER} is not serving {FLASH_GGUF.name}: {props.get('model_path')}")
    if set_ == "test":
        if (out / "done.json").exists():
            sys.exit("REFUSED: the test set has had its one pass for Clef-Flash")
        if _git("status", "--porcelain", "scripts/flash48.py"):
            sys.exit("REFUSED: commit flash48.py before the test pass")
        for arm, name in RECORDED.items():
            if not (RUNS / name.format(set=set_) / "done.json").exists():
                sys.exit(f"REFUSED: exp_047's recorded {arm} arm is missing")
    return props


def cmd_run(set_: str, limit: int, max_s: float):
    out = RUNS / f"exp048_{set_}_flash"
    props = _guard(set_, out)
    K = L.knife("rev41")
    K.KEV_URL = SERVER                       # the only change: which decision model answers
    rows = L.load(D / f"{set_}.jsonl")
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]      # exp_047's dev spread, so the ids match
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    t_start = time.time()
    with pred.open("a", encoding="utf-8") as f:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            if max_s and time.time() - t_start > max_s:
                print(f"stopped after {max_s:.0f} s; run again to resume", flush=True)
                return
            f.write(json.dumps({"id": r["id"], **K.triage(r["text"])}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} flash {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        sha = hashlib.sha256(FLASH_GGUF.read_bytes()).hexdigest() if set_ == "test" else None
        if sha and not sha.startswith(FLASH_SHA_PREFIX):
            print(f"WARNING: {FLASH_GGUF.name} sha256 {sha[:12]} is not exp_041's {FLASH_SHA_PREFIX}", flush=True)
        (out / "done.json").write_text(json.dumps({
            "n": len(rows), "server": SERVER, "props": props, "gguf_sha256": sha,
            "set_sha256": hashlib.sha256((D / f"{set_}.jsonl").read_bytes()).hexdigest(),
            "kev_knife_rev41_sha256": L.VERSIONS["rev41"][1],
            "flash48_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "casasol_head": _git("rev-parse", "HEAD"),
            "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest()}) + "\n")
    print("done")


# The hybrid (registered before the test, from the dev diagnosis): Kev answers the request side (its own intent reading
# with probabilities, the business-pitch and wants-to-own questions), Clef-Flash answers every other question, and rev
# 4.1's own policy() turns the merged answers into the final request, urgency and next step. Offline, from both arms'
# recorded answers; no extra model call.
KEV_SIDE = ("b2b", "wants_to_own")
FLASH_SIDE = ("property_type", "timeline", "financing", "foreign_buyer", "qualified", "hot", "specific_listing",
              "call_request", "asks_worth")


def recombine(K, req: dict, facts_from: dict) -> dict:
    """policy() over merged answers: the request side from `req`, everything else from `facts_from`."""
    m = {"intent": dict(req["intent_kev"]), **{k: req[k] for k in KEV_SIDE},
         **{k: facts_from[k] for k in FLASH_SIDE}, "facts": dict(req["facts"])}
    m["urgency"], m["action"] = K.policy(m, m["facts"])
    return m


def cmd_score(set_: str):
    rows = _rows(set_)
    arms = {"flash": {p["id"]: p for p in L.load(RUNS / f"exp048_{set_}_flash" / "pred.jsonl")}}
    for arm, name in RECORDED.items():
        arms[arm] = {p["id"]: p for p in L.load(RUNS / name.format(set=set_) / "pred.jsonl")}
    ids = [i for i in rows if all(i in a for a in arms.values())]
    K = L.knife("rev41")
    self_check = [i for i in ids if (lambda m: (m["intent"]["choice"], m["action"], m["urgency"]))(recombine(K, arms["kev"][i], arms["kev"][i]))
                  != (arms["kev"][i]["intent"]["choice"], arms["kev"][i]["action"], arms["kev"][i]["urgency"])]
    if self_check:
        sys.exit(f"recombine() does not reproduce Kev's recorded decisions on {len(self_check)} enquiries: {self_check[:5]}")
    arms["hybrid"] = {i: recombine(K, arms["kev"][i], arms["flash"][i]) for i in ids}

    def acc(field, arm, sub):
        sub = [i for i in sub if rows[i]["labels"].get(field) not in ("disagree", None)]
        return L.acc([rows[i]["labels"][field] == L.predicted(arms[arm][i])[field] for i in sub]), len(sub)

    def vs_kev(field, sub):
        sub = [i for i in sub if rows[i]["labels"].get(field) not in ("disagree", None)]
        rf = {i: rows[i]["labels"][field] == L.predicted(arms["flash"][i])[field] for i in sub}
        rk = {i: rows[i]["labels"][field] == L.predicted(arms["kev"][i])[field] for i in sub}
        b, c = sum(rf[i] and not rk[i] for i in sub), sum(rk[i] and not rf[i] for i in sub)
        return {"flash_only_right": b, "kev_only_right": c, "p_one_sided_flash_better": round(L.mcnemar_one_sided(b, c), 4)}

    res = {"set": set_, "n": len(ids), "fields": {}}
    for f in FIELDS:
        a = {arm: acc(f, arm, ids)[0] for arm in arms}
        k, c, fl = a["kev"] or 0, a["clef27b"] or 0, a["flash"] or 0
        res["fields"][f] = {"n": acc(f, "flash", ids)[1], **a, "share_of_clef27b_gain": round((fl - k) / (c - k), 2) if c != k else None,
                            **vs_kev(f, ids)}
    res["urgency_within_1"] = {arm: L.acc([abs(rows[i]["labels"]["urgency"] - L.predicted(p[i])["urgency"]) <= 1 for i in ids
                                           if rows[i]["labels"]["urgency"] not in (None, "disagree")]) for arm, p in arms.items()}
    res["spam"] = {arm: {k: s[k] for k in ("caught", "spam_n", "recall", "false_spam", "genuine_n", "false_spam_ids")}
                   for arm, s in ((arm, L.spam_stats(rows, p, ids)) for arm, p in arms.items())}
    res["by_language_request"] = {lang: {arm: acc("intent", arm, [i for i in ids if rows[i]["lang"] == lang])[0] for arm in arms}
                                  for lang in sorted({rows[i]["lang"] for i in ids})}
    conf = Counter((rows[i]["labels"]["intent"], arms["flash"][i]["intent"]["choice"]) for i in ids
                   if arms["flash"][i]["intent"]["choice"] != rows[i]["labels"]["intent"])
    res["confusions_flash"] = [f"{x}→{y}: {n}" for (x, y), n in conf.most_common(8)]
    res["median_ms"] = {arm: sorted(p[i].get("ms", 0) for i in ids)[len(ids) // 2] for arm, p in arms.items()}
    F, s = res["fields"], res["spam"]
    share = [F[f]["share_of_clef27b_gain"] for f in ("intent", "qualified", "timeline")]
    res["hypotheses"] = {
        "H1_qualified_showable": {"flash": F["qualified"]["flash"], "pass": (F["qualified"]["flash"] or 0) >= 0.80},
        "H2_timeline_showable": {"flash": F["timeline"]["flash"], "pass": (F["timeline"]["flash"] or 0) >= 0.75},
        "H3_no_regression_vs_kev": {"request": round((F["intent"]["flash"] or 0) - (F["intent"]["kev"] or 0), 3),
                                    "next_step": round((F["action"]["flash"] or 0) - (F["action"]["kev"] or 0), 3),
                                    "pass": (F["intent"]["flash"] or 0) - (F["intent"]["kev"] or 0) >= -0.02
                                            and (F["action"]["flash"] or 0) - (F["action"]["kev"] or 0) >= -0.02},
        "H4_spam_safety": {"false_spam": s["flash"]["false_spam"], "kev": s["kev"]["false_spam"], "recall": s["flash"]["recall"],
                           "pass": s["flash"]["false_spam"] <= s["kev"]["false_spam"] + 1 and (s["flash"]["recall"] or 0) >= 0.85},
        "H5_most_of_the_gain": {"share_request_qualified_timeline": share,
                                "pass": all(x is not None and x >= 0.5 for x in share)},
        "H6_hybrid": {"request": F["intent"]["hybrid"], "next_step": F["action"]["hybrid"], "kev_next_step": F["action"]["kev"],
                      "false_spam": s["hybrid"]["false_spam"],
                      "pass": (F["action"]["hybrid"] or 0) - (F["action"]["kev"] or 0) >= -0.02
                              and (F["intent"]["hybrid"] or 0) - (F["intent"]["kev"] or 0) >= -0.02
                              and s["hybrid"]["false_spam"] <= s["kev"]["false_spam"] + 1},
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("run", "score") or sys.argv[2] not in ("dev", "test"):
        sys.exit(__doc__)
    cmd, set_, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    opt = lambda k, d: type(d)(rest[rest.index(k) + 1]) if k in rest else d
    if cmd == "score":
        cmd_score(set_)
    else:
        cmd_run(set_, opt("--limit", 0), opt("--max-seconds", 0.0))
