#!/usr/bin/env python3
"""
exp_044 — lead triage rev 4 vs rev 3.1 on a fresh synthetic set (workspace tasks/exp044/HYPOTHESIS.md). Synthetic only.

  .venv/bin/python3 scripts/leads44.py validate              # writer files → counts per language / stratum / intent, schema errors
  .venv/bin/python3 scripts/leads44.py checker [N]           # blind checker batches (id + text only), N per batch
  .venv/bin/python3 scripts/leads44.py agree                 # writer × checker → data/leads44/agreed.jsonl
  .venv/bin/python3 scripts/leads44.py split                 # seeded dev / test split by language × stratum + READ30.md
  .venv/bin/python3 scripts/leads44.py run dev|test rev31|rev4    # Kev triage → runs/exp044_<set>_<ver>/pred.jsonl (test once each)
  .venv/bin/python3 scripts/leads44.py score dev|test rev31|rev4  # per-field, per-stratum scores
  .venv/bin/python3 scripts/leads44.py compare dev|test      # the pre-registered hypotheses H1–H5, paired
  .venv/bin/python3 scripts/leads44.py regress <kev_knife.py>  # standing guard: a candidate vs the adopted rev 4 on the sealed test (~9 min)

Agreement (fixed before any checker output, as in exp_043): a lead is dropped if writer and checker disagree on intent,
or the checker flags doubt. Any other field where they disagree is excluded for that lead; budget_eur agrees within 1 %.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D = ROOT / "data" / "leads44"
RUNS = ROOT / "runs"
SEED = 44
DEV_N = 150
FIELDS = ("intent", "property_type", "timeline", "financing", "foreign_buyer", "qualified", "urgency", "action", "budget_eur")
STRATA = ("standard", "hard_b2b", "hard_holiday", "hard_budget")
VALUES = {"intent": {"buy", "rent_long", "rent_holiday", "sell", "valuation", "invest", "info", "spam"},
          "property_type": {"apartment", "penthouse", "townhouse", "villa", "plot", "commercial", "unspecified"},
          "timeline": {"now", "months", "later", "unspecified"},
          "financing": {True, False, None}, "foreign_buyer": {True, False, None}, "qualified": {True, False},
          "urgency": {0, 1, 2, 3},
          "action": {"call_now", "send_listings", "ask_details", "book_valuation", "answer_question", "review"}}

# The two versions. rev 3.1 is frozen by pre-registration; rev 4 is frozen after dev tuning (None until then).
VERSIONS = {"rev31": (ROOT / "scripts" / "kev_knife_rev31.py",
                      "bc2ae14723530d6893206bf80dcb16844e79034eff77b93fb3de776a7bf9b16c"),
            "rev4": (ROOT / "scripts" / "kev_knife_rev4.py",
                     "b19402352632b676861477284fbbde06fdd2e4a65b8139509c2b199f11c9a815"),   # frozen 2026-10-19 after dev tuning
            # rev 4.1 (2026-10-20): code-fact fixes from exp_045 only (budget reader, amounts vs phones, spelled-out
            # phones masked); questions, rules and cuts unchanged. Gate passed 2026-10-20 (identical on every check).
            "rev41": (ROOT / "scripts" / "kev_knife_rev41.py",
                      "2acc7a1db1296e5284cd264762166c4a61d139376291e6002b8b6c1fd6ada956"),
            # rev 4.2 (2026-10-21, exp_048's verdict, Andrei's choice): Kev-4B answers the request side, Clef-Flash 9B the
            # rest, rev 4.1's rules combine them. Gate passed 2026-10-21 (request 0.912 = rev 4, next step 0.945 vs 0.933,
            # false spam 2 = rev 4; Flash answered all 331).
            "rev42": (ROOT / "scripts" / "kev_knife_rev42.py",
                      "5b79950f35e3248399ccfbeca9a1d56cfd144a7f28d228953a851185c24e09cf"),
            # rev 5.1 (2026-10-21, exp_049's verdict): rev 5 with F2 off. Gate passed 2026-10-21: request 0.912 (= rev 4),
            # next step 0.952 (0.933), false spam 0 (2), spam recall 0.918 (0.939: X2-02, a scam claiming a client).
            "rev51": (ROOT / "scripts" / "kev_knife.py",
                      "897c0ac2a06c935fc1eb4d57c40e5b0e39152d1e8e76c0d34bcd70645ab3d1e9")}
EXP043_STANDARD = {"intent": 0.894, "action": 0.898}     # exp_043 test, for the replication check


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(p: Path, rows: list[dict]) -> str:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    p.with_suffix(".sha256").write_text(h + "\n")
    return h


def writer_rows() -> list[dict]:
    rows = []
    for p in sorted((D / "raw").glob("x*.jsonl")):
        rows += load(p)
    return rows


def valid_budget(v) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0)


def bad_labels(labels: dict) -> list[str]:
    bad = [f for f in FIELDS if f != "budget_eur" and (f not in labels or labels[f] not in VALUES[f])]
    if "budget_eur" not in labels or not valid_budget(labels["budget_eur"]):
        bad.append("budget_eur")
    return bad


def budget_agree(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= 0.01 * max(a, b)


_PHONE_LIKE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_EMAIL_LIKE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
FAKE_PHONE = re.compile(r"^(\+|00)?(44\s?|0)7700\s?900\d{3}\b")  # Ofcom's drama range, never allocated (07700 900000–900999)
# Padded is not fake: "+34 600 000 1xx" (the exp_043/044 brief's example) is a valid-format Spanish mobile, and Spain has
# no reserved fiction range we could find, so it is flagged like any other number (2026-10-19).
FAKE_EMAIL = re.compile(r"@example\.[a-z]{2,3}$", re.I)


def _ofcom(span: str) -> bool:
    """In Ofcom's drama range in any spacing ("07700 900 466", "+44 77 00 90 01 45"): compare the digits (2026-10-21)."""
    return re.fullmatch(r"7700900\d{3}", re.sub(r"^(0044|44|0)", "", re.sub(r"\D", "", span))) is not None


def realistic_contacts(text: str) -> list[str]:
    """Contact details outside the reserved ranges: Ofcom's 07700 900xxx for phones, example.com for e-mail. exp_044's
    writers invented 26 realistic numbers despite "clearly fake"; anything listed here must be replaced before a
    dataset is published (exp_044 so-what, 2026-10-19)."""
    out = [m.group(0).strip() for m in _PHONE_LIKE.finditer(text) if not _ofcom(m.group(0))
           and sum(c.isdigit() for c in m.group(0)) >= 9 and not re.search(r"\d\.\d{3}\b", m.group(0))]   # "290.000" is a price
    out += [e for e in (m.group(0).rstrip(".,;:") for m in _EMAIL_LIKE.finditer(text)) if not FAKE_EMAIL.search(e)]
    return out


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
                      "bad_stratum": [r["id"] for r in rows if r.get("stratum") not in STRATA][:10],
                      "label_errors": errs[:10], "n_label_errors": len(errs),
                      "realistic_contacts": contacts[:20], "n_realistic_contacts": len(contacts)}, ensure_ascii=False, indent=1))


def cmd_checker(n: int = 73):
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
            labels[f] = r["labels"][f] if same else ("disagree" if f in ("financing", "foreign_buyer", "budget_eur") else None)
        kept.append({"id": i, "lang": r["lang"], "stratum": r["stratum"], "channel": r["channel"], "text": r["text"],
                     "labels": labels})
    h = dump(D / "agreed.jsonl", kept)
    print(json.dumps({"writer": len(w), "kept": len(kept), "dropped": dict(drop),
                      "dropped_by_stratum": {k: dict(v) for k, v in drop_by.items()},
                      "kept_by_stratum": dict(Counter(r["stratum"] for r in kept)),
                      "field_agreement_among_kept": {f: round(field_agree[f] / max(1, len(kept)), 3) for f in FIELDS},
                      "sha256": h[:16]}, indent=1))


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
    sample = random.Random(f"{SEED}:read30").sample(sorted(test, key=lambda r: r["id"]), 30)
    lines = ["# exp_044: 30 test leads for Andrei's read", "",
             "Check each label against the message and LABELS.md. Mark any lead you would label differently.", ""]
    for r in sample:
        L = dict(r["labels"])
        lines += [f"**{r['id']}** ({r['lang']}, {r['channel']}, {r['stratum']})", "", f"> {r['text']}", "",
                  "intent `{intent}` · type `{property_type}` · timeline `{timeline}` · financing `{financing}` · "
                  "foreign `{foreign_buyer}` · qualified `{qualified}` · urgency `{urgency}` · action `{action}` · "
                  "budget `{budget_eur}`".format(**L), ""]
    (D / "READ30.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dev": len(dev), "test": len(test), "dev_sha": hd[:16], "test_sha": ht[:16],
                      "test_by_lang": dict(Counter(r["lang"] for r in test)),
                      "test_by_stratum": dict(Counter(r["stratum"] for r in test)),
                      "test_standard_by_intent": dict(Counter(r["labels"]["intent"] for r in test if r["stratum"] == "standard"))},
                     indent=1))


def knife(ver: str):
    """Load a frozen kev_knife version from its file (both versions can run in one process)."""
    path, _ = VERSIONS[ver]
    spec = importlib.util.spec_from_file_location(f"kev_knife_{ver}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cmd_run(set_: str, ver: str):
    path, frozen = VERSIONS[ver]
    if set_ == "test":
        if frozen is None:
            sys.exit(f"REFUSED: {ver} is not frozen yet (set its sha256 in VERSIONS after dev tuning)")
        if hashlib.sha256(path.read_bytes()).hexdigest() != frozen:
            sys.exit(f"REFUSED: {path.name} differs from the version frozen for the test pass")
    out = RUNS / f"exp044_{set_}_{ver}"
    if set_ == "test" and (out / "done.json").exists():
        sys.exit(f"REFUSED: the test set has had its one pass for {ver}")
    K = knife(ver)
    rows = load(D / f"{set_}.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    with pred.open("a", encoding="utf-8") as f:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            f.write(json.dumps({"id": r["id"], **K.triage(r["text"])}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} {ver} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in pred.open()) >= len(rows):
        (out / "done.json").write_text(json.dumps({"n": len(rows), "version": ver,
                                                   "code_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   "pred_sha256": hashlib.sha256(pred.read_bytes()).hexdigest(),
                                                   "set_sha256": (D / f"{set_}.sha256").read_text().strip()}) + "\n")
    print("done")


def predicted(p: dict, threshold: float = 0.5) -> dict:
    return {"intent": p["intent"]["choice"], "property_type": p["property_type"]["choice"], "timeline": p["timeline"]["choice"],
            "financing": p["financing"] >= threshold, "foreign_buyer": p["foreign_buyer"] >= threshold,
            "qualified": p["qualified"] >= threshold, "urgency": int(round(p["urgency"])), "action": p["action"],
            "budget_eur": (p.get("facts") or {}).get("budget_eur")}


def correct(field: str, label, pred) -> bool:
    return budget_agree(pred, label) if field == "budget_eur" else label == pred


def scored(rows: dict, preds: dict, field: str, ids=None) -> list[bool]:
    ids = ids if ids is not None else list(rows)
    return [correct(field, rows[i]["labels"][field], predicted(preds[i])[field]) for i in ids
            if i in preds and rows[i]["labels"][field] not in ("disagree",)
            and not (rows[i]["labels"][field] is None and field != "budget_eur")]


def acc(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 3) if xs else None


def spam_stats(rows: dict, preds: dict, ids=None) -> dict:
    ids = [i for i in (ids if ids is not None else rows) if i in preds]
    spam = [i for i in ids if rows[i]["labels"]["intent"] == "spam"]
    genuine = [i for i in ids if rows[i]["labels"]["intent"] != "spam"]
    flagged = [i for i in genuine if preds[i]["intent"]["choice"] == "spam"]
    return {"spam_n": len(spam), "caught": sum(preds[i]["intent"]["choice"] == "spam" for i in spam),
            "recall": acc([preds[i]["intent"]["choice"] == "spam" for i in spam]),
            "genuine_n": len(genuine), "false_spam": len(flagged),
            "false_spam_rate": round(len(flagged) / max(1, len(genuine)), 4), "false_spam_ids": flagged}


def bands(rows: dict, preds: dict) -> dict:
    out = {}
    for name, lo, hi in (("sure", 0.75, 1.01), ("fairly_sure", 0.5, 0.75), ("unsure", 0.0, 0.5)):
        ids = [i for i in rows if i in preds and lo <= preds[i]["intent"]["confidence"] < hi]
        out[name] = {"n": len(ids), "accuracy": acc([rows[i]["labels"]["intent"] == preds[i]["intent"]["choice"] for i in ids])}
    pairs = [(preds[i]["intent"]["confidence"], rows[i]["labels"]["intent"] == preds[i]["intent"]["choice"]) for i in rows if i in preds]
    bins = defaultdict(list)
    for conf, ok in pairs:
        bins[min(9, int(conf * 10))].append((conf, ok))
    ece = sum(len(b) / len(pairs) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b)) for b in bins.values()) if pairs else None
    out["ece"] = round(ece, 3) if ece is not None else None
    return out


def score_set(set_: str, ver: str) -> dict:
    rows = {r["id"]: r for r in load(D / f"{set_}.jsonl")}
    preds = {p["id"]: p for p in load(RUNS / f"exp044_{set_}_{ver}" / "pred.jsonl")}
    res = {"n": len(preds), "by_stratum": {}}
    for st in STRATA + ("all",):
        ids = [i for i, r in rows.items() if st == "all" or r["stratum"] == st]
        res["by_stratum"][st] = {f: {"n": len(scored(rows, preds, f, ids)), "accuracy": acc(scored(rows, preds, f, ids))} for f in FIELDS}
        res["by_stratum"][st]["urgency_within_1"] = acc([abs(rows[i]["labels"]["urgency"] - predicted(preds[i])["urgency"]) <= 1
                                                        for i in ids if i in preds and rows[i]["labels"]["urgency"] is not None])
    res["spam"] = spam_stats(rows, preds)
    res["bands"] = bands(rows, preds)
    std = [i for i, r in rows.items() if r["stratum"] == "standard"]
    res["standard_by_lang_intent"] = {lang: acc([rows[i]["labels"]["intent"] == preds[i]["intent"]["choice"]
                                                 for i in std if rows[i]["lang"] == lang and i in preds])
                                      for lang in sorted({rows[i]["lang"] for i in std})}
    conf = Counter((rows[i]["labels"]["intent"], preds[i]["intent"]["choice"]) for i in rows if i in preds)
    res["intent_confusions_top"] = [f"{a}→{b}: {c}" for (a, b), c in conf.most_common() if a != b][:10]
    ms = sorted(p.get("ms", 0) for p in preds.values())
    res["median_ms"] = ms[len(ms) // 2] if ms else None
    return res


def cmd_score(set_: str, ver: str):
    print(json.dumps(score_set(set_, ver), indent=1, ensure_ascii=False))


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b wins out of b + c discordant pairs | p = 0.5): rev 4 right & rev 3.1 wrong = b."""
    n = b + c
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n if n else 1.0


def cmd_compare(set_: str):
    rows = {r["id"]: r for r in load(D / f"{set_}.jsonl")}
    p3 = {p["id"]: p for p in load(RUNS / f"exp044_{set_}_rev31" / "pred.jsonl")}
    p4 = {p["id"]: p for p in load(RUNS / f"exp044_{set_}_rev4" / "pred.jsonl")}
    std = [i for i, r in rows.items() if r["stratum"] == "standard"]
    a = lambda preds, f, ids: acc(scored(rows, preds, f, ids))
    h1 = {f: {"rev31": a(p3, f, std), "rev4": a(p4, f, std)} for f in ("intent", "action")}
    for f in h1:
        h1[f]["diff"] = round(h1[f]["rev4"] - h1[f]["rev31"], 3)
    s3, s4 = spam_stats(rows, p3), spam_stats(rows, p4)
    hol = [i for i, r in rows.items() if r["stratum"] == "hard_holiday" and i in p3 and i in p4]
    r3 = {i: rows[i]["labels"]["intent"] == p3[i]["intent"]["choice"] for i in hol}
    r4 = {i: rows[i]["labels"]["intent"] == p4[i]["intent"]["choice"] for i in hol}
    b = sum(r4[i] and not r3[i] for i in hol)
    c = sum(r3[i] and not r4[i] for i in hol)
    bud = [i for i, r in rows.items() if r["stratum"] == "hard_budget" and r["labels"]["budget_eur"] not in (None, "disagree")]
    band = bands(rows, p4)
    out = {
        "H1_no_regression_standard": {**h1, "pass": all(h1[f]["diff"] >= -0.02 for f in h1)},
        "H2_safer_spam": {"rev31": {k: s3[k] for k in ("caught", "spam_n", "recall", "false_spam", "genuine_n", "false_spam_rate")},
                          "rev4": {k: s4[k] for k in ("caught", "spam_n", "recall", "false_spam", "genuine_n", "false_spam_rate")},
                          "rev4_false_spam_ids": s4["false_spam_ids"],
                          "pass": s4["false_spam_rate"] <= 0.01 and s4["false_spam_rate"] <= s3["false_spam_rate"]
                                  and (s4["recall"] or 0) >= 0.85},
        "H3_holiday": {"n": len(hol), "rev31": acc(list(r3.values())), "rev4": acc(list(r4.values())),
                       "rev4_only_right": b, "rev31_only_right": c, "p_one_sided": round(mcnemar_one_sided(b, c), 4)},
        "H4_budget_words": {"n": len(bud), "rev31": a(p3, "budget_eur", bud), "rev4": a(p4, "budget_eur", bud)},
        "H5_bands_rev4": band,
        "replication_rev31_standard_vs_exp043": {"exp043": EXP043_STANDARD, "exp044_rev31": {f: h1[f]["rev31"] for f in h1}},
    }
    h3 = out["H3_holiday"]
    h3["diff"] = round((h3["rev4"] or 0) - (h3["rev31"] or 0), 3)
    h3["pass"] = h3["diff"] >= 0.10 and h3["p_one_sided"] < 0.05
    out["H4_budget_words"]["pass"] = (out["H4_budget_words"]["rev4"] or 0) >= 0.90
    s, f_, u = (band[k]["accuracy"] for k in ("sure", "fairly_sure", "unsure"))
    out["H5_bands_rev4"]["pass"] = (s is not None and s >= 0.90 and f_ is not None and u is not None and s > f_ > u)
    print(json.dumps(out, indent=1, ensure_ascii=False))


REGRESS_TOLERANCE = {"intent": 0.01, "action": 0.01, "spam_recall": 0.05}


def regress_verdict(rows: dict, base: dict, cand: dict) -> dict:
    """A candidate against the adopted rev 4 on exp_044's sealed test set (a standing guard, not a new measurement):
    request and next step on all leads within 0.01, no more genuine leads flagged as spam, spam caught within 0.05,
    budgets in words still ≥ 0.90."""
    b_sp, c_sp = spam_stats(rows, base), spam_stats(rows, cand)
    bud = [i for i, r in rows.items() if r["stratum"] == "hard_budget" and r["labels"]["budget_eur"] not in (None, "disagree")]
    checks = {
        "intent": (acc(scored(rows, base, "intent")), acc(scored(rows, cand, "intent"))),
        "action": (acc(scored(rows, base, "action")), acc(scored(rows, cand, "action"))),
        "false_spam": (b_sp["false_spam"], c_sp["false_spam"]),
        "spam_recall": (b_sp["recall"], c_sp["recall"]),
        "budget_words": (acc(scored(rows, base, "budget_eur", bud)), acc(scored(rows, cand, "budget_eur", bud))),
    }
    ok = {"intent": checks["intent"][1] >= checks["intent"][0] - REGRESS_TOLERANCE["intent"],
          "action": checks["action"][1] >= checks["action"][0] - REGRESS_TOLERANCE["action"],
          "false_spam": checks["false_spam"][1] <= checks["false_spam"][0],
          "spam_recall": (checks["spam_recall"][1] or 0) >= (checks["spam_recall"][0] or 0) - REGRESS_TOLERANCE["spam_recall"],
          "budget_words": (checks["budget_words"][1] or 0) >= 0.90}
    return {"checks": {k: {"rev4": v[0], "candidate": v[1], "ok": ok[k]} for k, v in checks.items()},
            "pass": all(ok.values()), "candidate_false_spam_ids": c_sp["false_spam_ids"]}


def cmd_regress(path: str):
    """Run a candidate kev_knife file on exp_044's sealed test set and compare it with the adopted rev 4. ~9 min on
    the mini. Exit code 1 if it regresses. Use before deploying any change to the triage (TRIAGE.md)."""
    cand_path = Path(path).resolve()
    sha = hashlib.sha256(cand_path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("kev_knife_candidate", cand_path)
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    rows = {r["id"]: r for r in load(D / "test.jsonl")}
    out = RUNS / f"exp044_regress_{sha[:12]}"
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in pred.read_text().splitlines()} if pred.exists() else set()
    with pred.open("a", encoding="utf-8") as f:
        for n, (i, r) in enumerate([(i, r) for i, r in rows.items() if i not in done], 1):
            f.write(json.dumps({"id": i, **K.triage(r["text"])}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"regress {len(done) + n}/{len(rows)}", flush=True)
    cand = {p["id"]: p for p in load(pred)}
    base = {p["id"]: p for p in load(RUNS / "exp044_test_rev4" / "pred.jsonl")}
    verdict = {"candidate": str(cand_path), "sha256": sha, **regress_verdict(rows, base, cand)}
    print(json.dumps(verdict, indent=1, ensure_ascii=False))
    sys.exit(0 if verdict["pass"] else 1)


def cmd_tune():
    """rev 4 cuts on dev only, offline: re-apply policy() to Kev's recorded answers (intent_kev) for each candidate.
    Rule (pre-registered): no genuine dev lead flagged as spam; then most spam caught; then request, then next-step accuracy."""
    import copy
    rows = {r["id"]: r for r in load(D / "dev.jsonl")}
    preds = {p["id"]: p for p in load(RUNS / "exp044_dev_rev4" / "pred.jsonl")}
    K = knife("rev4")
    results = []
    for b2b_cut in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        for agree in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3):
            for own_cut in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.01):
                for reverse in (False, True):
                    K.B2B_CUT, K.SPAM_AGREE, K.OWN_CUT, K.OWN_REVERSE = b2b_cut, agree, own_cut, reverse
                    new = {}
                    for i, p in preds.items():
                        o = copy.deepcopy(p)
                        o["intent"] = copy.deepcopy(p["intent_kev"])
                        o["urgency"], o["action"] = K.policy(o, p["facts"])
                        new[i] = o
                    sp = spam_stats(rows, new)
                    results.append({"B2B_CUT": b2b_cut, "SPAM_AGREE": agree, "OWN_CUT": own_cut, "OWN_REVERSE": reverse,
                                    "false_spam": sp["false_spam"], "spam_recall": sp["recall"],
                                    "intent": acc(scored(rows, new, "intent")), "action": acc(scored(rows, new, "action")),
                                    "holiday_intent": acc(scored(rows, new, "intent", [i for i, r in rows.items() if r["stratum"] == "hard_holiday"]))})
    results.sort(key=lambda x: (x["false_spam"] > 0, -(x["spam_recall"] or 0), -(x["intent"] or 0), -(x["action"] or 0),
                                x["OWN_REVERSE"], -x["OWN_CUT"]))
    print(json.dumps({"best": results[0], "top10": results[:10], "n_candidates": len(results)}, indent=1))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "tune":
        cmd_tune()
        sys.exit()
    if cmd == "regress":
        cmd_regress(rest[0])
    if cmd == "validate":
        cmd_validate()
    elif cmd == "checker":
        cmd_checker(int(rest[0]) if rest else 73)
    elif cmd == "agree":
        cmd_agree()
    elif cmd == "split":
        cmd_split()
    elif cmd == "run":
        cmd_run(rest[0], rest[1])
    elif cmd == "score":
        cmd_score(rest[0], rest[1])
    elif cmd == "compare":
        cmd_compare(rest[0])
    else:
        sys.exit(__doc__)
