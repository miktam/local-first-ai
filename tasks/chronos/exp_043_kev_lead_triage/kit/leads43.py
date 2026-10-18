#!/usr/bin/env python3
"""
exp_043 — lead-triage measurement for the Kev knife (scripts/kev_knife.py). Synthetic leads only.

  .venv/bin/python3 scripts/leads43.py validate            # writer files → counts, schema errors
  .venv/bin/python3 scripts/leads43.py checker [N]         # blind checker batches (id + text only), N per batch
  .venv/bin/python3 scripts/leads43.py agree               # writer × checker → agreed labels (data/leads43/agreed.jsonl)
  .venv/bin/python3 scripts/leads43.py split               # seeded dev / test split + READ30.md sample of test
  .venv/bin/python3 scripts/leads43.py run dev|test        # Kev triage → runs/exp043_<set>/pred.jsonl (test once)
  .venv/bin/python3 scripts/leads43.py score dev|test      # per-field accuracy against agreed labels

Agreement (fixed before any checker output): a lead is dropped if writer and checker disagree on intent, or the
checker flags doubt. Any other field where they disagree becomes null for that lead (excluded from that field's
score). Label definitions: workspace tasks/exp043/LABELS.md.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
D = ROOT / "data" / "leads43"
RUNS = ROOT / "runs"
SEED = 43
DEV_N = 120
FIELDS = ("intent", "property_type", "timeline", "financing", "foreign_buyer", "qualified", "urgency", "action")
VALUES = {"intent": {"buy", "rent_long", "rent_holiday", "sell", "valuation", "invest", "info", "spam"},
          "property_type": {"apartment", "penthouse", "townhouse", "villa", "plot", "commercial", "unspecified"},
          "timeline": {"now", "months", "later", "unspecified"},
          "financing": {True, False, None}, "foreign_buyer": {True, False, None}, "qualified": {True, False},
          "urgency": {0, 1, 2, 3},
          "action": {"call_now", "send_listings", "ask_details", "book_valuation", "answer_question", "discard"}}


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
    for p in sorted((D / "raw").glob("w*.jsonl")):
        rows += load(p)
    return rows


def bad_labels(labels: dict) -> list[str]:
    return [f for f in FIELDS if f not in labels or labels[f] not in VALUES[f]]


def cmd_validate():
    rows = writer_rows()
    ids = Counter(r["id"] for r in rows)
    errs = [(r["id"], bad_labels(r["labels"])) for r in rows if bad_labels(r["labels"])]
    print(json.dumps({"n": len(rows), "dup_ids": [i for i, c in ids.items() if c > 1],
                      "by_file": dict(Counter(r["id"].split("-")[0] for r in rows)),
                      "by_lang": dict(Counter(r["lang"] for r in rows)),
                      "by_intent": dict(Counter(r["labels"]["intent"] for r in rows)),
                      "label_errors": errs[:10], "n_label_errors": len(errs)}, ensure_ascii=False, indent=1))


def cmd_checker(n: int = 55):
    rows = [r for r in writer_rows() if not bad_labels(r["labels"])]
    rng = random.Random(f"{SEED}:chk")
    rng.shuffle(rows)                                   # each checker batch mixes languages and writers
    out = D / "chk"
    out.mkdir(parents=True, exist_ok=True)
    for b in range(0, len(rows), n):
        (out / f"c_{b // n + 1:02d}.jsonl").write_text(
            "".join(json.dumps({"id": r["id"], "text": r["text"]}, ensure_ascii=False) + "\n" for r in rows[b:b + n]),
            encoding="utf-8")
    print(f"{len(rows)} leads → {-(-len(rows) // n)} checker batches in {out}")


def cmd_agree():
    w = {r["id"]: r for r in writer_rows() if not bad_labels(r["labels"])}
    c = {}
    for p in sorted((D / "chk").glob("out_*.jsonl")):
        for r in load(p):
            c[r["id"]] = r
    missing = sorted(set(w) - set(c))
    if missing:
        sys.exit(f"REFUSED: {len(missing)} leads have no checker labels (first {missing[:3]})")
    kept, drop = [], Counter()
    field_agree = Counter()
    for i, r in w.items():
        cl = c[i]["labels"]
        if bad_labels(cl):
            drop["checker_schema"] += 1
            continue
        if c[i].get("doubt"):
            drop["doubt"] += 1
            continue
        if cl["intent"] != r["labels"]["intent"]:
            drop["intent_disagree"] += 1
            continue
        labels = {}
        for f in FIELDS:
            same = cl[f] == r["labels"][f]
            field_agree[f] += same
            labels[f] = r["labels"][f] if same else None
            if not same and f in ("financing", "foreign_buyer"):
                labels[f] = "disagree"
        kept.append({"id": i, "lang": r["lang"], "channel": r["channel"], "text": r["text"], "labels": labels})
    h = dump(D / "agreed.jsonl", kept)
    n = len(w) - sum(drop.values())
    print(json.dumps({"writer": len(w), "kept": len(kept), "dropped": dict(drop),
                      "field_agreement_among_kept": {f: round(field_agree[f] / max(1, len(kept)), 3) for f in FIELDS},
                      "sha256": h[:16]}, indent=1))


def cmd_split():
    rows = load(D / "agreed.jsonl")
    rng = random.Random(f"{SEED}:split")
    by = defaultdict(list)
    for r in rows:
        by[r["lang"]].append(r)
    dev, test = [], []
    share = DEV_N / len(rows)
    for lang in sorted(by):
        rs = sorted(by[lang], key=lambda r: r["id"])
        rng.shuffle(rs)
        k = round(share * len(rs))
        dev += rs[:k]
        test += rs[k:]
    hd, ht = dump(D / "dev.jsonl", dev), dump(D / "test.jsonl", test)
    sample = random.Random(f"{SEED}:read30").sample(sorted(test, key=lambda r: r["id"]), 30)
    lines = ["# exp_043: 30 test leads for Andrei's read", "",
             "Check each label against the message and LABELS.md. Mark any lead you would label differently.", ""]
    for r in sample:
        L = r["labels"]
        lines += [f"**{r['id']}** ({r['lang']}, {r['channel']})", "", f"> {r['text']}", "",
                  "intent `{intent}` · type `{property_type}` · timeline `{timeline}` · financing `{financing}` · "
                  "foreign `{foreign_buyer}` · qualified `{qualified}` · urgency `{urgency}` · action `{action}`".format(**L), ""]
    (D / "READ30.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dev": len(dev), "test": len(test), "dev_sha": hd[:16], "test_sha": ht[:16],
                      "test_by_lang": dict(Counter(r["lang"] for r in test)),
                      "test_by_intent": dict(Counter(r["labels"]["intent"] for r in test))}, indent=1))


KEV_KNIFE_FROZEN = "98f3009053f542763ee6042632551cb24bdbf08d58dbbd8c672f9bb63a07fb83"   # kev_knife.py rev 3, casasol d581267 — the test pass must use exactly this


def cmd_run(set_: str):
    import kev_knife as K
    if set_ == "test" and hashlib.sha256((ROOT / "scripts" / "kev_knife.py").read_bytes()).hexdigest() != KEV_KNIFE_FROZEN:
        sys.exit("REFUSED: scripts/kev_knife.py differs from the version frozen for the test pass")
    out = RUNS / f"exp043_{set_}"
    if set_ == "test" and (out / "done.json").exists():
        sys.exit("REFUSED: the test set has had its one pass")
    rows = load(D / f"{set_}.jsonl")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pred.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate([r for r in rows if r["id"] not in done], 1):
            f.write(json.dumps({"id": r["id"], **K.triage(r["text"])}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{set_} {len(done) + i}/{len(rows)}", flush=True)
    if sum(1 for _ in path.open()) >= len(rows):
        (out / "done.json").write_text(json.dumps({"n": len(rows), "pred_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   "set_sha256": (D / f"{set_}.sha256").read_text().strip()}) + "\n")
    print("done")


def predicted(p: dict, threshold: float = 0.5) -> dict:
    return {"intent": p["intent"]["choice"], "property_type": p["property_type"]["choice"], "timeline": p["timeline"]["choice"],
            "financing": p["financing"] >= threshold, "foreign_buyer": p["foreign_buyer"] >= threshold,
            "qualified": p["qualified"] >= threshold, "urgency": int(round(p["urgency"])), "action": p["action"]}


def cmd_score(set_: str):
    rows = {r["id"]: r for r in load(D / f"{set_}.jsonl")}
    preds = {p["id"]: p for p in load(RUNS / f"exp043_{set_}" / "pred.jsonl")}
    res = {}
    for f in FIELDS:
        pairs = [(rows[i]["labels"][f], predicted(preds[i])[f]) for i in rows if i in preds
                 and rows[i]["labels"][f] not in (None, "disagree")]
        res[f] = {"n": len(pairs), "accuracy": round(sum(a == b for a, b in pairs) / max(1, len(pairs)), 3)}
        if f == "urgency":
            res[f]["within_1"] = round(sum(abs(a - b) <= 1 for a, b in pairs) / max(1, len(pairs)), 3)
    spam = [(rows[i]["labels"]["intent"] == "spam", preds[i]["intent"]["choice"] == "spam") for i in rows if i in preds]
    res["spam"] = {"recall": round(sum(a and b for a, b in spam) / max(1, sum(a for a, _ in spam)), 3),
                   "false_spam_rate": round(sum(b and not a for a, b in spam) / max(1, sum(not a for a, _ in spam)), 3)}
    conf = Counter((rows[i]["labels"]["intent"], preds[i]["intent"]["choice"]) for i in rows if i in preds)
    res["intent_confusions_top"] = [f"{a}→{b}: {c}" for (a, b), c in conf.most_common() if a != b][:8]
    res["by_lang_intent_acc"] = {}
    for lang in sorted({r["lang"] for r in rows.values()}):
        ids = [i for i, r in rows.items() if r["lang"] == lang and i in preds]
        res["by_lang_intent_acc"][lang] = round(sum(rows[i]["labels"]["intent"] == preds[i]["intent"]["choice"] for i in ids) / max(1, len(ids)), 3)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "validate":
        cmd_validate()
    elif cmd == "checker":
        cmd_checker(int(rest[0]) if rest else 55)
    elif cmd == "agree":
        cmd_agree()
    elif cmd == "split":
        cmd_split()
    elif cmd in ("run", "score"):
        (cmd_run if cmd == "run" else cmd_score)(rest[0])
    else:
        sys.exit(__doc__)
