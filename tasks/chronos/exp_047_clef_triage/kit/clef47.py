"""exp_047: a bigger decision model behind the same rules.

Clef 27B (Cloudflare's clef, Q8_0, llama.cpp b11711 `/v1/systemone`, as in exp_041/042) answers rev 4.1's questions in
Kev-4B's place. Everything else is rev 4.1 as frozen (`kev_knife.py`, leads44 VERSIONS["rev41"]): the questions, the code
facts, the agency's rules and the cuts tuned for Kev. Only the decision server changes, through the module's KEV_URL.
Is Kev the ceiling, or are the rules? The weak fields (qualified 0.78, timeline 0.73 in exp_043) are the target.

usage: clef47.py run dev|test --arm kev|clef [--limit N] [--max-seconds N]   (resumes; the test pass runs once per arm)
       clef47.py score dev|test
The Clef server: llama-server -m <clef gguf> --port 18447 -c 16384 -b 8192 -ub 8192 (exp_041's settings).
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import leads44 as L  # noqa: E402

D = ROOT / "data" / "leads46"          # exp_043 + exp_044 dev/test, public copies (build_leads46.py), every language
RUNS = ROOT / "runs"
SERVERS = {"kev": "http://127.0.0.1:18442", "clef": "http://127.0.0.1:18447"}
CLEF_GGUF = Path.home() / "models/exp041/clef-27b/Cloudflare_clef-Q8_0.gguf"
CLEF_SHA_PREFIX = "9fe953f0"             # exp_041's record of this file
PAIRED = ("intent", "action", "qualified", "timeline", "property_type")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _props(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/props", timeout=10) as r:
        p = json.load(r)
    return {k: p.get(k) for k in ("model_path", "build_info", "total_slots") if k in p}


def _knife(arm: str):
    K = L.knife("rev41")
    K.KEV_URL = SERVERS[arm]                 # the only change: which decision model answers
    return K


def _guard(set_: str, arm: str, out: Path) -> dict:
    setp = D / f"{set_}.jsonl"
    from scripts.translate46 import SETS as FULL
    if hashlib.sha256(setp.read_bytes()).hexdigest() != FULL[set_]:
        sys.exit(f"REFUSED: {setp.name} is not the registered set")
    path, frozen = L.VERSIONS["rev41"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
        sys.exit("REFUSED: kev_knife.py differs from the frozen rev 4.1")
    try:
        props = _props(SERVERS[arm])
    except Exception as e:
        sys.exit(f"REFUSED: no decision server for {arm} at {SERVERS[arm]} ({type(e).__name__})")
    if arm == "clef" and not str(props.get("model_path", "")).endswith(CLEF_GGUF.name):
        sys.exit(f"REFUSED: the server on {SERVERS[arm]} is not serving {CLEF_GGUF.name}: {props.get('model_path')}")
    if set_ == "test":
        if (out / "done.json").exists():
            sys.exit(f"REFUSED: the test set has had its one pass for the {arm} arm")
        if _git("status", "--porcelain", "scripts/clef47.py"):
            sys.exit("REFUSED: commit clef47.py before the test pass")
    return props


def cmd_run(set_: str, arm: str, limit: int, max_s: float):
    out = RUNS / f"exp047_{set_}_{arm}"
    props = _guard(set_, arm, out)
    K = _knife(arm)
    rows = L.load(D / f"{set_}.jsonl")
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]      # an even spread across sources and languages
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
                print(f"{set_} {arm} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        gguf_sha = hashlib.sha256(CLEF_GGUF.read_bytes()).hexdigest() if arm == "clef" and set_ == "test" else None
        if gguf_sha and not gguf_sha.startswith(CLEF_SHA_PREFIX):
            print(f"WARNING: {CLEF_GGUF.name} sha256 {gguf_sha[:12]} is not exp_041's {CLEF_SHA_PREFIX}", flush=True)
        (out / "done.json").write_text(json.dumps({
            "n": len(rows), "arm": arm, "server": SERVERS[arm], "props": props, "gguf_sha256": gguf_sha,
            "set_sha256": hashlib.sha256((D / f"{set_}.jsonl").read_bytes()).hexdigest(),
            "kev_knife_rev41_sha256": L.VERSIONS["rev41"][1],
            "clef47_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "casasol_head": _git("rev-parse", "HEAD"),
            "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest()}) + "\n")
    print("done")


def _rows(set_: str) -> dict:
    """exp_043's labels call spam's next step "discard"; since 2026-10-18 it is "review" (never discard). Same decision."""
    rows = {r["id"]: r for r in L.load(D / f"{set_}.jsonl")}
    for r in rows.values():
        if r["labels"].get("action") == "discard":
            r["labels"]["action"] = "review"
    return rows


def cmd_score(set_: str):
    rows = _rows(set_)
    kev = {p["id"]: p for p in L.load(RUNS / f"exp047_{set_}_kev" / "pred.jsonl")}
    clef = {p["id"]: p for p in L.load(RUNS / f"exp047_{set_}_clef" / "pred.jsonl")}
    ids = [i for i in rows if i in kev and i in clef]

    def paired(field, sub, own=False):
        sub = [i for i in sub if rows[i]["labels"].get(field) not in ("disagree", None) or field == "budget_eur"]
        pick = (lambda p: p["intent_kev"]["choice"]) if own else (lambda p: L.predicted(p)[field])
        rk = {i: rows[i]["labels"][field] == pick(kev[i]) for i in sub}
        rc = {i: rows[i]["labels"][field] == pick(clef[i]) for i in sub}
        b, c = sum(rc[i] and not rk[i] for i in sub), sum(rk[i] and not rc[i] for i in sub)
        ak, ac = L.acc(list(rk.values())), L.acc(list(rc.values()))
        return {"n": len(sub), "kev": ak, "clef": ac, "diff": round((ac or 0) - (ak or 0), 3), "clef_only_right": b,
                "kev_only_right": c, "p_one_sided_clef_better": round(L.mcnemar_one_sided(b, c), 4)}

    res = {"set": set_, "n": len(ids), "fields": {f: paired(f, ids) for f in PAIRED}}
    res["fields"]["intent_before_rules"] = paired("intent", ids, own=True)
    for f in ("financing", "foreign_buyer"):
        res["fields"][f] = paired(f, ids)
    res["urgency_within_1"] = {a: L.acc([abs(rows[i]["labels"]["urgency"] - L.predicted(p[i])["urgency"]) <= 1 for i in ids
                                         if rows[i]["labels"]["urgency"] not in (None, "disagree")]) for a, p in (("kev", kev), ("clef", clef))}
    res["spam"] = {a: {k: s[k] for k in ("caught", "spam_n", "recall", "false_spam", "genuine_n", "false_spam_ids")}
                   for a, s in (("kev", L.spam_stats(rows, kev, ids)), ("clef", L.spam_stats(rows, clef, ids)))}
    res["by_language_request"] = {lang: paired("intent", [i for i in ids if rows[i]["lang"] == lang])
                                  for lang in sorted({rows[i]["lang"] for i in ids})}
    res["by_source_request"] = {s: paired("intent", [i for i in ids if rows[i]["source"] == s]) for s in ("exp_043", "exp_044")}
    for a, p in (("kev", kev), ("clef", clef)):
        conf = Counter((rows[i]["labels"]["intent"], p[i]["intent"]["choice"]) for i in ids if p[i]["intent"]["choice"] != rows[i]["labels"]["intent"])
        res[f"confusions_{a}"] = [f"{x}→{y}: {n}" for (x, y), n in conf.most_common(8)]
    ms = lambda p: sorted(p[i].get("ms", 0) for i in ids)[len(ids) // 2] if ids else None
    res["median_ms"] = {"kev": ms(kev), "clef": ms(clef)}
    q, t, r, n = (res["fields"][k] for k in ("qualified", "timeline", "intent", "action"))
    s = res["spam"]
    res["hypotheses"] = {
        "H1_qualified": {"kev": q["kev"], "clef": q["clef"], "pass": (q["clef"] or 0) >= 0.80 and q["diff"] >= 0.05},
        "H2_timeline": {"kev": t["kev"], "clef": t["clef"], "pass": (t["clef"] or 0) >= 0.75 and t["diff"] >= 0.05},
        "H3_no_regression": {"request": r["diff"], "next_step": n["diff"], "pass": r["diff"] >= -0.02 and n["diff"] >= -0.02},
        "H4_spam_safety": {"false_spam": [s["kev"]["false_spam"], s["clef"]["false_spam"]], "recall_clef": s["clef"]["recall"],
                           "pass": s["clef"]["false_spam"] <= s["kev"]["false_spam"] + 1 and (s["clef"]["recall"] or 0) >= 0.85},
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("run", "score") or sys.argv[2] not in ("dev", "test"):
        sys.exit(__doc__)
    cmd, set_, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if cmd == "score":
        cmd_score(set_)
    else:
        arm = rest[rest.index("--arm") + 1] if "--arm" in rest else ""
        if arm not in SERVERS:
            sys.exit(f"--arm must be one of {tuple(SERVERS)}")
        opt = lambda k, d: type(d)(rest[rest.index(k) + 1]) if k in rest else d
        cmd_run(set_, arm, opt("--limit", 0), opt("--max-seconds", 0.0))
