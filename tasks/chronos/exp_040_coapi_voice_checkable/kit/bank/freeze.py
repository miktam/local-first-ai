#!/usr/bin/env python3
"""exp_040 — freeze the training bank and the validation bank (BUILD.md §3, §6.4). Code only.

  python3.14 freeze.py   # writes data/bank40/BANK_FROZEN.jsonl, VALID_FROZEN.jsonl, their .sha256, FREEZE_MANIFEST.json

Training = the checker-kept items of pilot + waves 1–3 (each */kept.jsonl); validation = validw_chk/kept.jsonl.
A final same-label near-duplicate pass runs over the whole training bank (trigram >= 0.45 on q.en, first kept wins),
then validation items that duplicate a training item are dropped from validation. Yes/no is capped at 25 % of the
training bank (seeded; pairs are dropped together so no twin is orphaned). Twin links to dropped items are cleared.
The manifest also lists exp_039's arithmetic templates (id, tema, guide quote) so the v3 session can exclude their rules.
"""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

CV = Path.home() / "REPOS/casasol/coapi_voice"
B = CV / "data/bank40"
sys.path.insert(0, str(CV / "data/eval_v2"))
sys.path.insert(0, str(CV / "rl"))
import dedupe as dd  # noqa: E402

TRAIN_SRC = ["pilot_chk", "wave1_chk", "wave2_chk", "wave3_chk"]
VALID_SRC = ["validw_chk"]
YESNO_CAP = 0.25


def load(names):
    out = []
    for n in names:
        out += [json.loads(l) for l in (B / n / "kept.jsonl").read_text().splitlines() if l.strip()]
    return out


def key(it):
    return str(it.get("label") if it["type"] != "numeric" else it["numeric"]["value"])


def dedup(items, prior=()):
    seen = [(key(p), dd.grams(p["q"]["en"])) for p in prior]
    kept, dropped = [], []
    for it in items:
        g = dd.grams(it["q"]["en"])
        if any(k == key(it) and dd.jaccard(g, sg) >= dd.THRESHOLD for k, sg in seen):
            dropped.append(it["item"])
            continue
        seen.append((key(it), g))
        kept.append(it)
    return kept, dropped


def clear_twins(items):
    ids = {i["item"] for i in items}
    n = 0
    for i in items:
        if i.get("twin_of") and i["twin_of"] not in ids:
            i["twin_of"], i["flip"] = None, None
            n += 1
    return n


def cap_yesno(items):
    yn = [i for i in items if i["type"] == "yesno"]
    limit = int(YESNO_CAP * len(items))
    if len(yn) <= limit:
        return items, []
    rng = random.Random("40:yesno-cap")
    # drop whole groups (an item and its twins) until under the cap
    groups = {}
    for i in yn:
        root = i.get("twin_of") or i["item"]
        groups.setdefault(root, []).append(i["item"])
    order = sorted(groups)
    rng.shuffle(order)
    drop = set()
    for g in order:
        if len(yn) - len(drop) <= int(YESNO_CAP * (len(items) - len(drop))):
            break
        drop.update(groups[g])
    return [i for i in items if i["item"] not in drop], sorted(drop)


def write(name, items):
    p = B / name
    p.write_text("".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items))
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    (B / (name.rsplit(".", 1)[0] + ".sha256")).write_text(h + "\n")
    return h


def main():
    train, d_train = dedup(load(TRAIN_SRC))
    train, d_yesno = cap_yesno(train)
    t_cleared = clear_twins(train)
    valid, d_valid = dedup(load(VALID_SRC), prior=train)
    v_cleared = clear_twins(valid)
    h_train, h_valid = write("BANK_FROZEN.jsonl", train), write("VALID_FROZEN.jsonl", valid)
    import arith_templates as at
    man = {
        "train": {"n": len(train), "sha256": h_train, "by_type": dict(Counter(i["type"] for i in train)),
                  "by_module": dict(Counter(i["chunk"].split(":")[1] for i in train)),
                  "twins": sum(1 for i in train if i.get("twin_of")), "chunks": len({i["chunk"] for i in train}),
                  "dropped_dup": d_train, "dropped_yesno_cap": d_yesno, "twin_links_cleared": t_cleared},
        "valid": {"n": len(valid), "sha256": h_valid, "by_type": dict(Counter(i["type"] for i in valid)),
                  "dropped_dup_vs_train": d_valid, "twin_links_cleared": v_cleared},
        "arith_templates": [{"id": t["id"], "tema": t["tema"], "split": t["split"], "guide_quote": t.get("guide_quote")}
                            for t in at.TEMPLATES],
        "sources": {"train": TRAIN_SRC, "valid": VALID_SRC},
    }
    (B / "FREEZE_MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("dropped_dup", "dropped_yesno_cap", "dropped_dup_vs_train")}
                      for k, v in man.items() if k in ("train", "valid")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
