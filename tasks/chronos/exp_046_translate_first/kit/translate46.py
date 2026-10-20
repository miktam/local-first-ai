"""exp_046: translate before triage.

Kev-4B's model card says English only, and Polish is its weakest language. Here a local translator (gemma4:26b on the
mini's Ollama) turns each non-English enquiry into English, and Kev answers its questions on the translation. Everything
else is rev 4 as frozen (kev_knife_rev4.py): contacts are masked before the translator sees the text, and the code facts
(language, budget, phone) and the agency's rules still come from the original. Compared with rev 4 on the original,
enquiry by enquiry, on the non-English enquiries of exp_043's and exp_044's test sets (public copies).

Built to the deterministic-glue invariants except one: Kev reads the translator's output (invariant 1, "no model call
on another model's prose"). Adoption would need Andrei's explicit exception (2026-10-20). Every translation is written
to disk (translations.jsonl), the translator answers in a strict JSON contract, and code checks that every number and
every [phone]/[email] marker survives; on any failure Kev reads the original instead (recorded as a fallback).

usage: translate46.py check dev                       (the re-assembled triage equals rev 4's on 10 dev enquiries)
       translate46.py run dev|test --arm native|translated [--max-seconds N]   (resumes; the test pass runs once per arm)
       translate46.py score dev|test
"""
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import leads44 as L  # noqa: E402

D = ROOT / "data" / "leads46"
RUNS = ROOT / "runs"
ARMS = ("native", "translated")
OLLAMA = "http://127.0.0.1:11434/api/chat"
TRANSLATOR = "gemma4:26b"
TRANSLATOR_DIGEST = "777b927e3b6b1452e44c31a7b6d136dbefe34826b0528e0d8cd67c73dd9eef29"
SYSTEM = ("You translate estate-agency enquiries into English. Translate faithfully: keep every number, amount, currency, "
          "date, place name and the markers [phone] and [email] exactly as written. Do not add, explain, summarise or "
          'answer. Reply with JSON: {"english": "<the translation>"}.')
FORMAT = {"type": "object", "properties": {"english": {"type": "string"}}, "required": ["english"]}
OPTIONS = {"temperature": 0, "seed": 46, "num_predict": 800}
SETS = {"dev": "91fb8e0cde3824ebfaa9ea19ab59adbfb8c5ebfe63f937b4ae9987389d309f9e",    # exp_043 + exp_044 dev, public copies
        "test": "a5f823d9c98f674d24493031e5846ac0cc3a570500fe11608c17c32fd90fa3b2"}   # exp_043 + exp_044 test, public copies (local-first-ai 9d87aa7)


def _digits(s: str) -> Counter:
    """The numbers that must survive a translation: 3+ digits (amounts, years, sizes), a thousands-grouped number read as
    one ("1.200.000" = "1,200,000" = "1 200 000"). Small numbers ("2 bedrooms") may become words."""
    nums = (re.sub(r"[.,\s]", "", m) for m in re.findall(r"\d{1,3}(?:[.,\s]\d{3})+(?!\d)|\d+", s))
    return Counter(n for n in nums if len(n) >= 3)


def translate(src: str) -> dict:
    body = {"model": TRANSLATOR, "think": False, "stream": False, "format": FORMAT, "options": OPTIONS,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": src}]}
    t0 = time.time()
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = {"english": None, "ok": False, "reason": None}
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            content = json.load(r)["message"]["content"]
        en = json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip()).strip())["english"]
        if not isinstance(en, str) or not en.strip():
            out["reason"] = "empty"
        elif any(en.count(t) != src.count(t) for t in ("[phone]", "[email]")):
            out["reason"] = "marker lost or added"
        elif _digits(src) - _digits(en):
            out["reason"] = f"number lost: {sorted((_digits(src) - _digits(en)).elements())[:3]}"
        else:
            out.update(english=en.strip(), ok=True)
        if out["english"] is None and isinstance(en, str):
            out["rejected"] = en
    except Exception as e:                           # invalid output fails the leaf; it never reaches Kev
        out["reason"] = f"{type(e).__name__}"
    out["ms"] = round((time.time() - t0) * 1000)
    return out


def triage_on(K, text: str, kev_text: str) -> dict:
    """rev 4's triage() with one change: Kev reads `kev_text`. The facts and the rules come from the original `text`."""
    t0 = time.time()
    a = K.ask(kev_text, K.LEAD_SCHEMA)
    out = {"facts": K.code_facts(text)}
    for k, v in a.items():
        if v["type"] == "choice":
            out[k] = {"choice": v["choice"], "confidence": round(v["confidence"], 2),
                      "probabilities": {o: round(pr, 3) for o, pr in (v.get("probabilities") or {}).items()}}
        elif v["type"] == "noul":
            out[k] = round(v["noul"], 2)
        else:
            out[k] = round(v["score"], 2)
    out["urgency"], out["action"] = K.policy(out, out["facts"])
    out["action_by"] = "rules"
    out["ms"] = round((time.time() - t0) * 1000)
    return out


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _translator_digest() -> str:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=10) as r:
        return next((m["digest"] for m in json.load(r)["models"] if m["name"] == TRANSLATOR), "")


def _rows(set_: str) -> list[dict]:
    return [r for r in L.load(D / f"{set_}.jsonl") if r["lang"] != "en"]


def _guard(set_: str, arm: str, out: Path):
    setp = D / f"{set_}.jsonl"
    if SETS.get(set_) and hashlib.sha256(setp.read_bytes()).hexdigest() != SETS[set_]:
        sys.exit(f"REFUSED: {setp.name} is not the registered set")
    path, frozen = L.VERSIONS["rev4"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
        sys.exit("REFUSED: kev_knife_rev4.py differs from the frozen rev 4")
    if set_ == "test":
        if not SETS.get("test"):
            sys.exit("REFUSED: the test set's hash is not registered in SETS")
        if (out / "done.json").exists():
            sys.exit(f"REFUSED: the test set has had its one pass for the {arm} arm")
        if _git("status", "--porcelain", "scripts/translate46.py"):
            sys.exit("REFUSED: commit translate46.py before the test pass")
        if arm == "translated" and _translator_digest() != TRANSLATOR_DIGEST:
            sys.exit(f"REFUSED: {TRANSLATOR} is not the registered build ({TRANSLATOR_DIGEST[:12]})")


def _diffs(a, b, path=""):
    """Differences between two triage results: any string or bool, or a number by more than 0.005."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in set(a) | set(b) for d in _diffs(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return [] if abs(a - b) <= 0.005 else [(path, a, b)]
    return [] if a == b else [(path, a, b)]


def cmd_check(set_: str, n: int = 10):
    """triage_on(text, masked text) must equal rev 4's own triage: same decisions, probabilities within 0.005. Kev's
    probabilities move by about ±0.001 between identical calls (llama.cpp's prompt cache), so exact equality is wrong."""
    K = L.knife("rev4")
    for r in _rows(set_)[:n]:
        a, b = K.triage(r["text"]), triage_on(K, r["text"], K.masked(r["text"]))
        a.pop("ms"), b.pop("ms")
        if _diffs(a, b):
            sys.exit(f"MISMATCH on {r['id']}: {_diffs(a, b)[:3]}")
    print(f"triage_on == rev 4 triage on {n} {set_} enquiries (decisions exact, probabilities within 0.005)")


def cmd_run(set_: str, arm: str, max_s: float):
    out = RUNS / f"exp046_{set_}_{arm}"
    _guard(set_, arm, out)
    K = L.knife("rev4")
    rows = _rows(set_)
    out.mkdir(parents=True, exist_ok=True)
    pred, trans = out / "pred.jsonl", out / "translations.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    t_start = time.time()
    with pred.open("a", encoding="utf-8") as f, trans.open("a", encoding="utf-8") as g:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            if max_s and time.time() - t_start > max_s:
                print(f"stopped after {max_s:.0f} s; run again to resume", flush=True)
                return
            if arm == "native":
                p = K.triage(r["text"])
            else:
                src = K.masked(r["text"])
                tr = translate(src)
                g.write(json.dumps({"id": r["id"], "lang": r["lang"], "source": src, **tr}, ensure_ascii=False) + "\n")
                g.flush()
                p = triage_on(K, r["text"], tr["english"] if tr["ok"] else src)
                p.update(translated=tr["ok"], translate_ms=tr["ms"])
            f.write(json.dumps({"id": r["id"], **p}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} {arm} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        (out / "done.json").write_text(json.dumps({
            "n": len(rows), "arm": arm, "set_sha256": hashlib.sha256((D / f"{set_}.jsonl").read_bytes()).hexdigest(),
            "kev_knife_rev4_sha256": L.VERSIONS["rev4"][1],
            "translate46_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "casasol_head": _git("rev-parse", "HEAD"),
            "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(),
            "translations_sha256": hashlib.sha256(trans.read_bytes()).hexdigest() if arm == "translated" else None,
            "translator": {"model": TRANSLATOR, "digest": _translator_digest(), "options": OPTIONS} if arm == "translated" else None}) + "\n")
    print("done")


def cmd_score(set_: str):
    rows = {r["id"]: r for r in _rows(set_)}
    nat = {p["id"]: p for p in L.load(RUNS / f"exp046_{set_}_native" / "pred.jsonl")}
    tra = {p["id"]: p for p in L.load(RUNS / f"exp046_{set_}_translated" / "pred.jsonl")}
    ids = [i for i in rows if i in nat and i in tra]

    def paired(field, sub):
        rn = {i: L.correct(field, rows[i]["labels"][field], L.predicted(nat[i])[field]) for i in sub}
        rt = {i: L.correct(field, rows[i]["labels"][field], L.predicted(tra[i])[field]) for i in sub}
        b, c = sum(rt[i] and not rn[i] for i in sub), sum(rn[i] and not rt[i] for i in sub)
        return {"n": len(sub), "native": L.acc(list(rn.values())), "translated": L.acc(list(rt.values())),
                "diff": round((L.acc(list(rt.values())) or 0) - (L.acc(list(rn.values())) or 0), 3),
                "translated_only_right": b, "native_only_right": c, "p_one_sided": round(L.mcnemar_one_sided(b, c), 4)}

    res = {"set": set_, "n": len(ids), "request": paired("intent", ids), "next_step": paired("action", ids)}
    res["spam"] = {k: {key: s[key] for key in ("caught", "spam_n", "recall", "false_spam", "genuine_n", "false_spam_ids")}
                   for k, s in (("native", L.spam_stats(rows, nat, ids)), ("translated", L.spam_stats(rows, tra, ids)))}
    res["by_language"] = {lang: {"request": paired("intent", g), "next_step": paired("action", g)}
                          for lang in sorted({rows[i]["lang"] for i in ids}) for g in [[i for i in ids if rows[i]["lang"] == lang]]}
    res["by_source"] = {src: paired("intent", g) for src in sorted({rows[i]["source"] for i in ids})
                        for g in [[i for i in ids if rows[i]["source"] == src]]}
    trans = L.load(RUNS / f"exp046_{set_}_translated" / "translations.jsonl")
    res["translation"] = {"ok": sum(t["ok"] for t in trans), "fallbacks": Counter(t["reason"] for t in trans if not t["ok"]),
                          "median_ms": sorted(t["ms"] for t in trans)[len(trans) // 2] if trans else None}
    conf = Counter((rows[i]["labels"]["intent"], tra[i]["intent"]["choice"]) for i in ids if tra[i]["intent"]["choice"] != rows[i]["labels"]["intent"])
    res["translated_confusions_top"] = [f"{a}→{b}: {c}" for (a, b), c in conf.most_common(8)]
    res["flips"] = [f"{i} [{rows[i]['lang']}]: {rows[i]['labels']['intent']} | native {nat[i]['intent']['choice']} | translated {tra[i]['intent']['choice']}"
                    for i in ids if nat[i]["intent"]["choice"] != tra[i]["intent"]["choice"]]
    r, n, s = res["request"], res["next_step"], res["spam"]
    pl = res["by_language"].get("pl", {}).get("request", {})
    res["hypotheses"] = {
        "H1_request_gain": {"diff": r["diff"], "p": r["p_one_sided"], "pass": r["diff"] >= 0.02 and r["p_one_sided"] < 0.05},
        "H2_next_step_no_harm": {"diff": n["diff"], "pass": n["diff"] >= -0.01},
        "H3_spam_safety": {"false_spam": [s["native"]["false_spam"], s["translated"]["false_spam"]],
                           "recall_translated": s["translated"]["recall"],
                           "pass": s["translated"]["false_spam"] <= s["native"]["false_spam"] + 1 and (s["translated"]["recall"] or 0) >= 0.85},
        "H4_polish": {"diff": pl.get("diff"), "pass": (pl.get("diff") or 0) >= 0.05},
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("check", "run", "score") or sys.argv[2] not in ("dev", "test"):
        sys.exit(__doc__)
    cmd, set_, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if cmd == "check":
        cmd_check(set_)
    elif cmd == "score":
        cmd_score(set_)
    else:
        arm = rest[rest.index("--arm") + 1] if "--arm" in rest else ""
        if arm not in ARMS:
            sys.exit(f"--arm must be one of {ARMS}")
        cmd_run(set_, arm, float(rest[rest.index("--max-seconds") + 1]) if "--max-seconds" in rest else 0.0)
