#!/usr/bin/env python3
"""
exp_049 — lead triage rev 5 vs rev 4.2 on a fresh synthetic set (workspace tasks/exp049/). Synthetic only.

Data, as exp_044 but with the lessons of the 2026-10-20 post applied: hard strata written from rev 4.2's recorded error
patterns; a blind checker of a different model (Opus, not the Fable writers); a third reader of a different family
(gemma4:26b, local), reported, never used to drop; phone numbers only in Ofcom's range, checked at writing time.

  .venv/bin/python3 scripts/leads49.py validate        # writer files → counts, schema errors, realistic contacts
  .venv/bin/python3 scripts/leads49.py checker [N]     # blind checker batches (id + text only)
  .venv/bin/python3 scripts/leads49.py agree           # writer × checker → data/leads49/agreed.jsonl
  .venv/bin/python3 scripts/leads49.py third           # gemma4:26b reads every agreed lead (resumes) → third.jsonl
  .venv/bin/python3 scripts/leads49.py third-score     # how often the third reader agrees with the kept labels
  .venv/bin/python3 scripts/leads49.py split           # seeded dev / test by language × stratum + READ30.md

Agreement (fixed before any checker output, as in exp_043/044): a lead is dropped if writer and checker disagree on
intent, or the checker flags doubt. Any other field where they disagree is excluded for that lead; budget_eur agrees
within 1 %.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.leads44 import budget_agree, dump, load, realistic_contacts, valid_budget  # noqa: E402

D = ROOT / "data" / "leads49"
LABELS_MD = Path.home() / ".openclaw/workspace/tasks/exp049/LABELS.md"
SEED = 49
DEV_N = 165
FIELDS = ("intent", "property_type", "timeline", "financing", "foreign_buyer", "on_behalf", "qualified", "urgency",
          "action", "budget_eur")
STRATA = ("standard", "hard_intermediary", "hard_valuation", "hard_timing")
VALUES = {"intent": {"buy", "rent_long", "rent_holiday", "sell", "valuation", "invest", "info", "spam"},
          "property_type": {"apartment", "penthouse", "townhouse", "villa", "plot", "commercial", "unspecified"},
          "timeline": {"now", "months", "later", "unspecified"},
          "financing": {True, False, None}, "foreign_buyer": {True, False, None}, "on_behalf": {True, False},
          "qualified": {True, False}, "urgency": {0, 1, 2, 3},
          "action": {"call_now", "send_listings", "ask_details", "book_valuation", "answer_question", "review"}}
TRI = ("financing", "foreign_buyer", "budget_eur", "on_behalf")     # disagreement → "disagree", not None


def writer_rows() -> list[dict]:
    rows = []
    for p in sorted((D / "raw").glob("y*.jsonl")):
        rows += load(p)
    return rows


def bad_labels(labels: dict) -> list[str]:
    bad = [f for f in FIELDS if f != "budget_eur" and (f not in labels or labels[f] not in VALUES[f])]
    if "budget_eur" not in labels or not valid_budget(labels["budget_eur"]):
        bad.append("budget_eur")
    return bad


def cmd_validate():
    rows = writer_rows()
    ids = Counter(r["id"] for r in rows)
    errs = [(r["id"], bad_labels(r["labels"])) for r in rows if bad_labels(r["labels"])]
    contacts = [(r["id"], c) for r in rows for c in realistic_contacts(r["text"])]
    print(json.dumps({"n": len(rows), "dup_ids": [i for i, c in ids.items() if c > 1],
                      "by_file": dict(Counter(r["id"].split("-")[0] for r in rows)),
                      "by_lang": dict(Counter(r["lang"] for r in rows)),
                      "by_stratum": dict(Counter(r.get("stratum") for r in rows)),
                      "standard_by_intent": dict(Counter(r["labels"]["intent"] for r in rows if r.get("stratum") == "standard")),
                      "hard_by_stratum_and_intent": {s: dict(Counter(r["labels"]["intent"] for r in rows if r.get("stratum") == s))
                                                     for s in STRATA[1:]},
                      "hard_timing_timeline": dict(Counter(r["labels"].get("timeline") for r in rows if r.get("stratum") == "hard_timing")),
                      "bad_stratum": [r["id"] for r in rows if r.get("stratum") not in STRATA][:10],
                      "label_errors": errs[:10], "n_label_errors": len(errs),
                      "realistic_contacts": contacts[:20], "n_realistic_contacts": len(contacts)}, ensure_ascii=False, indent=1))


def cmd_checker(n: int = 70):
    rows = [r for r in writer_rows() if not bad_labels(r["labels"]) and r.get("stratum") in STRATA]
    rng = random.Random(f"{SEED}:chk")
    rng.shuffle(rows)                                   # each checker batch mixes languages, writers and strata
    out = D / "chk"
    out.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(rows), n):
        (out / f"c_{b // n + 1:02d}.jsonl").write_text(
            "".join(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n" for r in rows[b:b + n]),
            encoding="utf-8")
    print(f"{len(rows)} leads → {-(-len(rows) // n)} checker batches in {out}")


def cmd_agree():
    w = {r["id"]: r for r in writer_rows() if not bad_labels(r["labels"]) and r.get("stratum") in STRATA}
    c = {}
    for p in sorted((D / "chk").glob("out_*.jsonl")):
        for r in load(p):
            c[r["id"]] = r
    missing = sorted(set(w) - set(c))
    if missing:
        sys.exit(f"REFUSED: {len(missing)} leads have no checker labels (first {missing[:3]})")
    kept, drop, field_agree = [], Counter(), Counter()
    drop_by = defaultdict(Counter)
    for i, r in w.items():
        cl = c[i]["labels"]
        reason = ("checker_schema" if bad_labels(cl) else "doubt" if c[i].get("doubt")
                  else "intent_disagree" if cl["intent"] != r["labels"]["intent"] else None)
        if reason:
            drop[reason] += 1
            drop_by[r["stratum"]][reason] += 1
            continue
        labels = {}
        for f in FIELDS:
            same = budget_agree(cl[f], r["labels"][f]) if f == "budget_eur" else cl[f] == r["labels"][f]
            field_agree[f] += same
            labels[f] = r["labels"][f] if same else ("disagree" if f in TRI else None)
        kept.append({"id": i, "lang": r["lang"], "stratum": r["stratum"], "channel": r["channel"], "text": r["text"],
                     "labels": labels})
    h = dump(D / "agreed.jsonl", kept)
    print(json.dumps({"writer": len(w), "kept": len(kept), "dropped": dict(drop),
                      "dropped_by_stratum": {k: dict(v) for k, v in drop_by.items()},
                      "kept_by_stratum": dict(Counter(r["stratum"] for r in kept)),
                      "field_agreement_among_kept": {f: round(field_agree[f] / max(1, len(kept)), 3) for f in FIELDS},
                      "sha256": h[:16]}, indent=1))


THIRD_MODEL = "gemma4:26b"
THIRD_SCHEMA = {"type": "object", "required": list(FIELDS), "properties": {
    "intent": {"enum": sorted(VALUES["intent"])}, "property_type": {"enum": sorted(VALUES["property_type"])},
    "timeline": {"enum": sorted(VALUES["timeline"])}, "financing": {"type": ["boolean", "null"]},
    "foreign_buyer": {"type": ["boolean", "null"]}, "on_behalf": {"type": "boolean"}, "qualified": {"type": "boolean"},
    "urgency": {"enum": [0, 1, 2, 3]}, "action": {"enum": sorted(VALUES["action"])}, "budget_eur": {"type": ["number", "null"]}}}


def cmd_third():
    """gemma4:26b (Google; neither the writers', the checker's nor the triage models' family) labels every agreed lead
    blind, from LABELS.md and the text. Reported only: it never drops a lead."""
    rules = LABELS_MD.read_text(encoding="utf-8")
    system = ("You label inbound messages to a Costa del Sol estate agency. Apply these label definitions exactly, from the "
              "message text alone. Reply with JSON only.\n\n" + rules)
    out = D / "third.jsonl"
    done = {r["id"] for r in load(out)} if out.exists() else set()
    rows = [r for r in load(D / "agreed.jsonl") if r["id"] not in done]
    with out.open("a", encoding="utf-8") as f:
        for n, r in enumerate(rows, 1):
            body = {"model": THIRD_MODEL, "think": False, "stream": False, "format": THIRD_SCHEMA,
                    "options": {"temperature": 0, "seed": SEED, "num_predict": 400},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": r["text"]}]}
            req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    labels = json.loads(json.load(resp)["message"]["content"])
            except Exception as e:
                labels = {"error": type(e).__name__}
            f.write(json.dumps({"id": r["id"], "labels": labels, "ms": round((time.time() - t0) * 1000)}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"third {len(done) + n}/{len(done) + len(rows)}", flush=True)
    print("done")


def cmd_third_score():
    kept = {r["id"]: r for r in load(D / "agreed.jsonl")}
    third = {r["id"]: r["labels"] for r in load(D / "third.jsonl")}
    res = {"n": len(third), "errors": sum("error" in t for t in third.values())}
    for f in FIELDS:
        pairs = [(kept[i]["labels"][f], t.get(f)) for i, t in third.items() if i in kept and "error" not in t
                 and kept[i]["labels"][f] not in (None, "disagree")]
        ok = [budget_agree(a, b) if f == "budget_eur" and (a is None or isinstance(b, (int, float))) else a == b for a, b in pairs]
        res[f] = {"n": len(ok), "agree": round(sum(ok) / max(1, len(ok)), 3)}
    res["intent_by_stratum"] = {s: round(sum(third[i].get("intent") == kept[i]["labels"]["intent"] for i in kept
                                             if kept[i]["stratum"] == s and i in third) /
                                         max(1, sum(1 for i in kept if kept[i]["stratum"] == s and i in third)), 3) for s in STRATA}
    print(json.dumps(res, indent=1))


def cmd_split():
    rows = load(D / "agreed.jsonl")
    rng = random.Random(f"{SEED}:split")
    cells = defaultdict(list)
    for r in rows:
        cells[(r["lang"], r["stratum"])].append(r)
    dev, test = [], []
    share = DEV_N / len(rows)
    for key in sorted(cells):
        rs = sorted(cells[key], key=lambda r: r["id"])
        rng.shuffle(rs)
        k = round(share * len(rs))
        dev += rs[:k]
        test += rs[k:]
    hd, ht = dump(D / "dev.jsonl", dev), dump(D / "test.jsonl", test)
    (D / "dev.sha256").write_text(hd + "\n")
    (D / "test.sha256").write_text(ht + "\n")
    sample = random.Random(f"{SEED}:read30").sample(sorted(test, key=lambda r: r["id"]), 30)
    lines = ["# exp_049: 30 test leads for Andrei's read", "",
             "Check each label against the message and LABELS.md (today is 21 October 2026). Mark any lead you would label differently.", ""]
    for r in sample:
        L = dict(r["labels"])
        lines += [f"**{r['id']}** ({r['lang']}, {r['channel']}, {r['stratum']})", "", f"> {r['text']}", "",
                  "intent `{intent}` · type `{property_type}` · timeline `{timeline}` · financing `{financing}` · "
                  "foreign `{foreign_buyer}` · on behalf `{on_behalf}` · qualified `{qualified}` · urgency `{urgency}` · "
                  "action `{action}` · budget `{budget_eur}`".format(**L), ""]
    (D / "READ30.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dev": len(dev), "test": len(test), "dev_sha": hd[:16], "test_sha": ht[:16],
                      "test_by_lang": dict(Counter(r["lang"] for r in test)),
                      "test_by_stratum": dict(Counter(r["stratum"] for r in test)),
                      "test_standard_by_intent": dict(Counter(r["labels"]["intent"] for r in test if r["stratum"] == "standard"))},
                     indent=1))


# ── the experiment: rev 4.2 (live) against rev 5 ─────────────────────────────────────────────────────────────────
import datetime as _dt
import hashlib
import importlib.util
import itertools

RUNS = ROOT / "runs"
TODAY = _dt.date(2026, 10, 21)          # the day every message "arrives" (LABELS.md)
VERSIONS = {"rev42": (ROOT / "scripts" / "kev_knife.py", "5b79950f35e3248399ccfbeca9a1d56cfd144a7f28d228953a851185c24e09cf"),
            "rev5": (ROOT / "scripts" / "kev_knife_rev5.py", "8d268660bc6c5cd82f16f8f7cbf7ec4681233560c4fa9835fe4e3fbc7f0bd7a6")}   # frozen 2026-10-21 after dev tuning
BASE_REQUEST_SIDE = ("intent", "b2b", "wants_to_own")
NEW_Q = ("on_behalf", "price_only")


def knife(ver: str):
    path, _ = VERSIONS[ver]
    spec = importlib.util.spec_from_file_location(f"kev_knife49_{ver}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if ver == "rev5":
        mod.TODAY, mod.KEEP_RAW = TODAY, True
    return mod


def _flash_up() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:18448/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def cmd_run(set_: str, ver: str, max_s: float = 0.0):
    path, frozen = VERSIONS[ver]
    if not _flash_up():
        sys.exit("REFUSED: Clef-Flash is not answering on :18448 (rev 4.2 and rev 5 both need it)")
    if set_ == "test":
        if frozen is None:
            sys.exit(f"REFUSED: {ver} is not frozen yet")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
            sys.exit(f"REFUSED: {path.name} differs from the version frozen for the test pass")
    sha = (D / f"{set_}.sha256").read_text().strip()
    if hashlib.sha256((D / f"{set_}.jsonl").read_bytes()).hexdigest() != sha:
        sys.exit(f"REFUSED: {set_}.jsonl differs from its sealed hash")
    out = RUNS / f"exp049_{set_}_{ver}"
    if set_ == "test" and (out / "done.json").exists():
        sys.exit(f"REFUSED: the test set has had its one pass for {ver}")
    K = knife(ver)
    rows = load(D / f"{set_}.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    t0 = time.time()
    with pred.open("a", encoding="utf-8") as f:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            if max_s and time.time() - t0 > max_s:
                print(f"stopped after {max_s:.0f} s; run again to resume", flush=True)
                return
            p = K.triage(r["text"])
            if (p.get("models") or {}).get("facts") != "clef-flash-9b":
                sys.exit(f"STOPPED: Flash did not answer {r['id']}; nothing written for it")
            f.write(json.dumps({"id": r["id"], **p}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} {ver} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        (out / "done.json").write_text(json.dumps({"n": len(rows), "version": ver,
                                                   "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(),
                                                   "set_sha256": sha, "today": str(TODAY)}) + "\n")
    print("done")


def _pred_fields(p: dict) -> dict:
    from scripts.leads44 import predicted
    out = predicted(p)
    out["on_behalf"] = (p.get("on_behalf") or 0) >= 0.5 if "on_behalf" in p else None
    return out


def _ok(field, label, pred) -> bool:
    return budget_agree(pred, label) if field == "budget_eur" else label == pred


def _acc(rows, preds, field, ids):
    xs = [_ok(field, rows[i]["labels"][field], _pred_fields(preds[i])[field]) for i in ids
          if i in preds and rows[i]["labels"][field] not in ("disagree",) and not (rows[i]["labels"][field] is None and field != "budget_eur")]
    return (round(sum(xs) / len(xs), 3) if xs else None), len(xs)


def _paired(rows, a, b, field, ids):
    ids = [i for i in ids if rows[i]["labels"][field] not in ("disagree", None)]
    ra = {i: _ok(field, rows[i]["labels"][field], _pred_fields(a[i])[field]) for i in ids}
    rb = {i: _ok(field, rows[i]["labels"][field], _pred_fields(b[i])[field]) for i in ids}
    bw, cw = sum(rb[i] and not ra[i] for i in ids), sum(ra[i] and not rb[i] for i in ids)
    from scripts.leads44 import mcnemar_one_sided
    acc = lambda d: round(sum(d.values()) / len(d), 3) if d else None
    return {"n": len(ids), "rev42": acc(ra), "rev5": acc(rb), "diff": round((acc(rb) or 0) - (acc(ra) or 0), 3),
            "rev5_only_right": bw, "rev42_only_right": cw, "p_one_sided": round(mcnemar_one_sided(bw, cw), 4)}


def _spam(rows, preds, ids):
    genuine = [i for i in ids if rows[i]["labels"]["intent"] != "spam"]
    spam = [i for i in ids if rows[i]["labels"]["intent"] == "spam"]
    fs = [i for i in genuine if preds[i]["intent"]["choice"] == "spam"]
    return {"false_spam": len(fs), "genuine": len(genuine), "caught": sum(preds[i]["intent"]["choice"] == "spam" for i in spam),
            "spam_n": len(spam), "recall": round(sum(preds[i]["intent"]["choice"] == "spam" for i in spam) / max(1, len(spam)), 3),
            "false_spam_ids": fs}


def cmd_compare(set_: str):
    rows = {r["id"]: r for r in load(D / f"{set_}.jsonl")}
    a = {p["id"]: p for p in load(RUNS / f"exp049_{set_}_rev42" / "pred.jsonl")}
    b = {p["id"]: p for p in load(RUNS / f"exp049_{set_}_rev5" / "pred.jsonl")}
    ids = [i for i in rows if i in a and i in b]
    by = lambda st: [i for i in ids if rows[i]["stratum"] == st]
    std, hi, hv, ht = by("standard"), by("hard_intermediary"), by("hard_valuation"), by("hard_timing")
    res = {"set": set_, "n": len(ids),
           "standard": {f: _paired(rows, a, b, f, std) for f in ("intent", "action", "timeline")},
           "hard_intermediary": {"intent": _paired(rows, a, b, "intent", hi),
                                 "spam_rev42": _spam(rows, a, hi), "spam_rev5": _spam(rows, b, hi)},
           "hard_valuation": {"intent": _paired(rows, a, b, "intent", hv)},
           "hard_timing": {"timeline": _paired(rows, a, b, "timeline", ht)},
           "all": {f: _paired(rows, a, b, f, ids) for f in ("intent", "action", "timeline", "qualified", "financing", "property_type")},
           "spam_all": {"rev42": _spam(rows, a, ids), "rev5": _spam(rows, b, ids)},
           "on_behalf_rev5": _acc(rows, b, "on_behalf", ids)[0],
           "by_language_intent": {lang: _paired(rows, a, b, "intent", [i for i in ids if rows[i]["lang"] == lang])
                                  for lang in sorted({rows[i]["lang"] for i in ids})}}
    S, I, V, T = res["standard"], res["hard_intermediary"], res["hard_valuation"], res["hard_timing"]
    f42, f5 = I["spam_rev42"]["false_spam"], I["spam_rev5"]["false_spam"]
    sa = res["spam_all"]
    res["hypotheses"] = {
        "H1_no_regression_standard": {"request": S["intent"]["diff"], "next_step": S["action"]["diff"],
                                      "pass": S["intent"]["diff"] >= -0.02 and S["action"]["diff"] >= -0.02},
        "H2_intermediaries": {"false_spam_hard": [f42, f5], "false_spam_all": [sa["rev42"]["false_spam"], sa["rev5"]["false_spam"]],
                              "recall_all": [sa["rev42"]["recall"], sa["rev5"]["recall"]],
                              "pass": (f5 <= f42 // 2 if f42 >= 2 else f5 <= f42)
                                      and sa["rev5"]["false_spam"] <= sa["rev42"]["false_spam"]
                                      and sa["rev5"]["recall"] >= 0.85 and sa["rev5"]["recall"] >= sa["rev42"]["recall"] - 0.05},
        "H3_valuation": {"diff": V["intent"]["diff"], "p": V["intent"]["p_one_sided"],
                         "pass": V["intent"]["diff"] >= 0.10 and V["intent"]["p_one_sided"] < 0.05},
        "H4_timing": {"diff": T["timeline"]["diff"], "p": T["timeline"]["p_one_sided"], "standard_timeline": S["timeline"]["diff"],
                      "pass": T["timeline"]["diff"] >= 0.10 and T["timeline"]["p_one_sided"] < 0.05 and S["timeline"]["diff"] >= -0.02},
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))


def recombine(K, p: dict, ob_src: str, po_src: str) -> dict:
    """rev 5's policy over recorded raw answers, with on_behalf and price_only taken from the chosen model."""
    kev, flash = p["raw"]["kev"], p["raw"]["flash"]
    m = {**kev, **{k: v for k, v in flash.items() if k not in BASE_REQUEST_SIDE + NEW_Q}, "facts": dict(p["facts"])}
    m["on_behalf"] = (kev if ob_src == "kev" else flash)["on_behalf"]
    m["price_only"] = (kev if po_src == "kev" else flash)["price_only"]
    m["intent"] = dict(kev["intent"])
    m["urgency"], m["action"] = K.policy(m, m["facts"])
    return m


def cmd_tune():
    """Offline on dev: which model answers on_behalf and price_only, and the cuts. Best dev request accuracy with no more
    false spam than rev 4.2 on dev and spam recall within 0.05 of it; ties → the most cautious (higher cuts, fewer rules)."""
    rows = {r["id"]: r for r in load(D / "dev.jsonl")}
    a = {p["id"]: p for p in load(RUNS / "exp049_dev_rev42" / "pred.jsonl")}
    raw = {p["id"]: p for p in load(RUNS / "exp049_dev_rev5" / "pred.jsonl")}
    ids = [i for i in rows if i in a and i in raw]
    K = knife("rev5")
    base_spam = _spam(rows, a, ids)
    base_req = round(sum(rows[i]["labels"]["intent"] == a[i]["intent"]["choice"] for i in ids) / len(ids), 3)
    results = []
    for ob_src, obc, po_src, poc, rev in itertools.product(("kev", "flash"), (None, 0.5, 0.6, 0.7, 0.8), ("kev", "flash"),
                                                           (None, 0.5, 0.6, 0.7, 0.8), (False, True)):
        if poc is None and rev:
            continue
        K.ON_BEHALF_CUT, K.PRICE_CUT, K.PRICE_REVERSE = obc, poc, rev
        preds = {i: recombine(K, raw[i], ob_src, po_src) for i in ids}
        req = round(sum(rows[i]["labels"]["intent"] == preds[i]["intent"]["choice"] for i in ids) / len(ids), 3)
        sp = _spam(rows, preds, ids)
        ok = sp["false_spam"] <= base_spam["false_spam"] and sp["recall"] >= base_spam["recall"] - 0.05
        caution = (obc is None, poc is None, obc or 0, poc or 0, not rev)
        results.append((ok, req, caution, {"on_behalf": [ob_src, obc], "price_only": [po_src, poc], "reverse": rev,
                                           "request": req, "false_spam": sp["false_spam"], "recall": sp["recall"]}))
    results.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
    print(json.dumps({"rev42_dev": {"request": base_req, **{k: base_spam[k] for k in ("false_spam", "recall")}},
                      "best": results[0][3], "top10": [r[3] for r in results[:10]],
                      "rules_off": next(r[3] for r in results if r[3]["on_behalf"][1] is None and r[3]["price_only"][1] is None)},
                     indent=1))


if __name__ == "__main__":
    cmds = {"validate": cmd_validate, "checker": cmd_checker, "agree": cmd_agree, "third": cmd_third,
            "third-score": cmd_third_score, "split": cmd_split, "tune": cmd_tune}
    if len(sys.argv) >= 3 and sys.argv[1] == "run":
        rest = sys.argv[4:]
        cmd_run(sys.argv[2], sys.argv[3], float(rest[rest.index("--max-seconds") + 1]) if "--max-seconds" in rest else 0.0)
        sys.exit()
    if len(sys.argv) >= 3 and sys.argv[1] == "compare":
        cmd_compare(sys.argv[2])
        sys.exit()
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    if sys.argv[1] == "checker" and len(sys.argv) > 2:
        cmd_checker(int(sys.argv[2]))
    else:
        cmds[sys.argv[1]]()
