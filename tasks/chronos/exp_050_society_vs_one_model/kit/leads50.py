#!/usr/bin/env python3
"""
exp_050 — the front desk as a society of small models (G) against one model doing it all (M).
Workspace tasks/exp050/ (HYPOTHESIS.md, LABELS.md v4). Synthetic only: invented enquiries, an invented inventory.

  .venv/bin/python3 scripts/leads50.py inventory       # the agency's 60 listings (seed 50) → data + tasks/exp050/inventory.md
  .venv/bin/python3 scripts/leads50.py validate        # writer files → counts, schema errors, contacts, fitting sets
  .venv/bin/python3 scripts/leads50.py checker [N]     # blind checker batches (id + text only)
  .venv/bin/python3 scripts/leads50.py agree           # writer × checker → data/leads50/agreed.jsonl
  .venv/bin/python3 scripts/leads50.py split           # seeded dev / test by language × stratum + READ30.md
  .venv/bin/python3 scripts/leads50.py extra           # run 3's two added listings: fit no kept enquiry (checked)
  .venv/bin/python3 scripts/leads50.py langcheck       # lang_of() against the writers' language on every kept enquiry
  .venv/bin/python3 scripts/leads50.py run SET ARM K   # SET dev|test, ARM G|M|Gs|Ms|M31, K run number (resumes)
  .venv/bin/python3 scripts/leads50.py compare SET     # E1–E4, wrong sent, H1–H5 with the registered bars → JSON

Agreement (fixed before any checker output): a lead is dropped if writer and checker disagree on intent, action or any
criterion (property_type, areas, min_bedrooms, budget_eur, features), or the checker flags doubt, because the fitting set
depends on every criterion. Any other field where they disagree is excluded for that lead; budget_eur agrees within 1 %.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.leads44 import budget_agree, dump, load, realistic_contacts, valid_budget  # noqa: E402

D = ROOT / "data" / "leads50"
WS = Path.home() / ".openclaw/workspace/tasks/exp050"
SEED = 50
DEV_N = 80
AREAS = ("san_pedro", "nueva_andalucia", "guadalmina", "marbella_centre", "puerto_banus", "elviria", "estepona", "benahavis")
AREA_NAMES = {"san_pedro": "San Pedro de Alcántara", "nueva_andalucia": "Nueva Andalucía", "guadalmina": "Guadalmina",
              "marbella_centre": "Marbella centre", "puerto_banus": "Puerto Banús", "elviria": "Elviria",
              "estepona": "Estepona", "benahavis": "Benahavís"}
FEATURES = ("sea_view", "pool", "garden", "terrace", "parking", "walk_to_beach", "near_schools", "golf")
TYPES = ("apartment", "penthouse", "townhouse", "villa", "plot", "commercial")
CRITERIA = ("property_type", "areas", "min_bedrooms", "budget_eur", "features")
FIELDS = ("intent", "property_type", "timeline", "financing", "foreign_buyer", "on_behalf", "qualified", "urgency",
          "action", "budget_eur", "areas", "min_bedrooms", "features")
STRATA = ("standard", "no_match", "seller", "question", "spam")
VALUES = {"intent": {"buy", "rent_long", "rent_holiday", "sell", "valuation", "invest", "info", "spam"},
          "property_type": {"apartment", "penthouse", "townhouse", "villa", "plot", "commercial", "unspecified"},
          "timeline": {"now", "months", "later", "unspecified"},
          "financing": {True, False, None}, "foreign_buyer": {True, False, None}, "on_behalf": {True, False},
          "qualified": {True, False}, "urgency": {0, 1, 2, 3},
          "action": {"call_now", "send_listings", "ask_details", "book_valuation", "answer_question", "review"}}
TRI = ("financing", "foreign_buyer", "on_behalf")     # disagreement → "disagree", not None
SEEKS = {"buy", "invest", "rent_long", "rent_holiday"}

# ---------------------------------------------------------------------------------------------------------------------
# The agency's inventory: invented, plausible. Prices per m² by area, type multipliers and feature odds are set by hand
# here (2026-10-23), not taken from any portal or agency.

_SALE_M2 = {"puerto_banus": 7500, "marbella_centre": 6500, "nueva_andalucia": 6000, "guadalmina": 5500,
            "benahavis": 5500, "elviria": 4800, "san_pedro": 4500, "estepona": 4000}
_LET_M2 = {"puerto_banus": 25, "marbella_centre": 22, "nueva_andalucia": 20, "guadalmina": 19, "benahavis": 18,
           "elviria": 17, "san_pedro": 16, "estepona": 14}            # € per m² per month
_TYPE_X = {"apartment": 1.0, "penthouse": 1.2, "townhouse": 0.9, "villa": 1.1}
_BEDS = {"apartment": (1, 3), "penthouse": (2, 4), "townhouse": (2, 4), "villa": (3, 6)}
_BEACH = {"san_pedro": .4, "guadalmina": .4, "puerto_banus": .6, "marbella_centre": .6, "elviria": .5, "estepona": .5,
          "nueva_andalucia": .05, "benahavis": 0}
_SCHOOLS = {"nueva_andalucia": .5, "san_pedro": .6, "guadalmina": .4, "estepona": .5, "marbella_centre": .5}
_GOLF = {"nueva_andalucia": .6, "guadalmina": .6, "benahavis": .5}
_MIX = {"sale": {"apartment": 14, "penthouse": 6, "townhouse": 8, "villa": 14, "plot": 2, "commercial": 2},
        "long_let": {"apartment": 7, "penthouse": 2, "townhouse": 3, "villa": 2}}


def _listing(rng: random.Random, op: str, typ: str, area: str) -> dict:
    if typ == "plot":
        m2 = rng.randrange(800, 2600, 50)
        return {"operation": op, "area": area, "type": typ, "bedrooms": None, "m2": m2,
                "price_eur": round(m2 * _SALE_M2[area] * rng.uniform(.12, .2), -4),
                "features": sorted(f for f, p in (("sea_view", .5), ("golf", _GOLF.get(area, .1))) if rng.random() < p)}
    if typ == "commercial":
        m2 = rng.randrange(60, 220, 5)
        return {"operation": op, "area": area, "type": typ, "bedrooms": None, "m2": m2,
                "price_eur": round(m2 * _SALE_M2[area] * rng.uniform(.5, .7), -4),
                "features": ["parking"] if rng.random() < .5 else []}
    beds = rng.randint(*_BEDS[typ])
    m2 = {"apartment": 60 + 30 * beds, "penthouse": 100 + 35 * beds, "townhouse": 110 + 35 * beds,
          "villa": 200 + 80 * (beds - 3) + rng.randrange(0, 121, 5)}[typ] + rng.randrange(-15, 16, 5)
    if op == "sale":
        price = round(m2 * _SALE_M2[area] * _TYPE_X[typ] * rng.uniform(.85, 1.15), -4)
    else:
        price = round(m2 * _LET_M2[area] * _TYPE_X[typ] * rng.uniform(.9, 1.1) / 50) * 50
    odds = {"sea_view": {"penthouse": .7, "villa": .5, "apartment": .35, "townhouse": .3}[typ] + (.15 if area in ("estepona", "elviria") else 0),
            "pool": .9 if typ == "villa" else .6,
            "garden": {"villa": .9, "townhouse": .6, "apartment": .2, "penthouse": 0}[typ],
            "terrace": {"penthouse": 1, "apartment": .8, "townhouse": .7, "villa": .7}[typ],
            "parking": .7, "walk_to_beach": _BEACH[area], "near_schools": _SCHOOLS.get(area, .2), "golf": _GOLF.get(area, .1)}
    return {"operation": op, "area": area, "type": typ, "bedrooms": beds, "m2": m2, "price_eur": int(price),
            "features": sorted(f for f in FEATURES if rng.random() < odds[f])}


def make_inventory(seed=SEED) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for op, mix in _MIX.items():
        for typ, n in mix.items():
            for _ in range(n):
                rows.append(_listing(rng, op, typ, rng.choice(AREAS)))
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["ref"] = f"EX-{101 + i}"
        r["price_eur"] = int(r["price_eur"])
    return [{"ref": r.pop("ref"), **r} for r in rows]


def inventory_md(rows: list[dict]) -> str:
    out = ["# exp_050: the agency's inventory (invented)", "",
           "*Generated by `scripts/leads50.py inventory` (seed 50). Every listing, ref and price is invented. The agency sells "
           "and lets long-term (6 months or more) in eight areas; it does no holiday lets. Sale prices are totals; long-let "
           "prices are per month. For plots, m² is the plot; otherwise it is built area.*", "",
           "| ref | operation | area | type | bedrooms | m² | price (€) | features |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['ref']} | {r['operation'].replace('_', ' ')} | {AREA_NAMES[r['area']]} | {r['type']} | "
                   f"{r['bedrooms'] if r['bedrooms'] is not None else '–'} | {r['m2']} | "
                   f"{r['price_eur']:,}{' /month' if r['operation'] == 'long_let' else ''} | {', '.join(r['features']) or '–'} |")
    return "\n".join(out) + "\n"


def inventory() -> list[dict]:
    return load(D / "inventory.jsonl")


def cmd_inventory():
    rows = make_inventory()
    D.mkdir(parents=True, exist_ok=True)
    h = dump(D / "inventory.jsonl", rows)
    (WS / "inventory.md").write_text(inventory_md(rows), encoding="utf-8")
    print(json.dumps({"n": len(rows), "sha256": h[:16], "by_op": dict(Counter(r["operation"] for r in rows)),
                      "by_area": dict(Counter(r["area"] for r in rows)), "by_type": dict(Counter(r["type"] for r in rows)),
                      "features": dict(Counter(f for r in rows for f in r["features"]))}, indent=1))


# ---------------------------------------------------------------------------------------------------------------------
# The fit rule (LABELS.md v4), applied to criteria: the labels for scoring, G's extracted answers inside the chain.

_TYPE_OK = {"apartment": {"apartment", "penthouse"}, "unspecified": {"apartment", "penthouse", "townhouse", "villa"}}


def fits(c: dict, r: dict) -> bool:
    """c: intent, property_type, areas, min_bedrooms, budget_eur, features. r: one listing."""
    op = {"buy": "sale", "invest": "sale", "rent_long": "long_let"}.get(c["intent"])
    if op is None or r["operation"] != op:
        return False
    if r["type"] not in _TYPE_OK.get(c["property_type"], {c["property_type"]}):
        return False
    areas = set(c.get("areas") or [])
    if areas and r["area"] not in areas:
        return False
    if c.get("budget_eur") and r["price_eur"] > c["budget_eur"] * 1.10:
        return False
    if c.get("min_bedrooms") and (r["bedrooms"] or 0) < c["min_bedrooms"]:
        return False
    return set(c.get("features") or []) <= set(r["features"])


def fitting(c: dict, inv: list[dict]) -> list[str]:
    return [r["ref"] for r in inv if fits(c, r)]


# ---------------------------------------------------------------------------------------------------------------------
# Data: writers, checker, agreement, split

def writer_rows() -> list[dict]:
    rows = []
    for p in sorted((D / "raw").glob("y*.jsonl")):
        rows += load(p)
    return rows


def bad_labels(labels: dict) -> list[str]:
    bad = [f for f in VALUES if f not in labels or labels[f] not in VALUES[f]]
    if "budget_eur" not in labels or not valid_budget(labels["budget_eur"]):
        bad.append("budget_eur")
    a, b, f = labels.get("areas"), labels.get("min_bedrooms"), labels.get("features")
    if not isinstance(a, list) or not set(a) <= set(AREAS) | {"other"} or len(set(a)) != len(a):
        bad.append("areas")
    if not (b is None or (isinstance(b, int) and not isinstance(b, bool) and 1 <= b <= 8)):
        bad.append("min_bedrooms")
    if not isinstance(f, list) or not set(f) <= set(FEATURES) or len(set(f)) != len(f):
        bad.append("features")
    if labels.get("intent") not in SEEKS and (a or b is not None or f):
        bad.append("criteria_for_non_seeker")
    return bad


def _canon(f: str, v):
    return sorted(v) if f in ("areas", "features") and isinstance(v, list) else v


def _contacts_ok(r: dict) -> list[str]:
    """Labelled contacts must appear in the text; spelled-out ones are checked as words."""
    return [c for c in r.get("contacts", []) if c not in r["text"]]


def cmd_validate():
    rows = writer_rows()
    inv = inventory()
    ids = Counter(r["id"] for r in rows)
    errs = [(r["id"], bad_labels(r["labels"])) for r in rows if bad_labels(r["labels"])]
    contacts = [(r["id"], c) for r in rows for c in realistic_contacts(r["text"])]
    lost = [(r["id"], _contacts_ok(r)) for r in rows if _contacts_ok(r)]
    stratum_vs_fit = Counter()
    for r in rows:
        if bad_labels(r["labels"]):
            continue
        n = len(fitting(r["labels"], inv))
        if r.get("stratum") == "standard":
            stratum_vs_fit["standard_with_fit" if n else "standard_NO_FIT"] += 1
        elif r.get("stratum") == "no_match":
            stratum_vs_fit["no_match_ok" if not n else "no_match_HAS_FIT"] += 1
    print(json.dumps({"n": len(rows), "dup_ids": [i for i, c in ids.items() if c > 1],
                      "by_file": dict(Counter(r["id"].split("-")[0] for r in rows)),
                      "by_lang": dict(Counter(r["lang"] for r in rows)),
                      "by_stratum": dict(Counter(r.get("stratum") for r in rows)),
                      "by_intent": dict(Counter(r["labels"].get("intent") for r in rows)),
                      "stratum_vs_fit": dict(stratum_vs_fit),
                      "fit_sizes_standard": dict(Counter(min(len(fitting(r["labels"], inv)), 5) for r in rows
                                                         if r.get("stratum") == "standard" and not bad_labels(r["labels"]))),
                      "bad_stratum": [r["id"] for r in rows if r.get("stratum") not in STRATA][:10],
                      "label_errors": errs[:10], "n_label_errors": len(errs),
                      "with_contacts": sum(bool(r.get("contacts")) for r in rows),
                      "contacts_not_in_text": lost[:10],
                      "realistic_contacts": contacts[:20], "n_realistic_contacts": len(contacts)}, ensure_ascii=False, indent=1))


def cmd_checker(n: int = 60):
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
    inv = inventory()
    kept, drop, field_agree = [], Counter(), Counter()
    drop_by = defaultdict(Counter)
    for i, r in w.items():
        cl = c[i]["labels"]
        same = {f: (budget_agree(cl.get(f), r["labels"][f]) if f == "budget_eur" else _canon(f, cl.get(f)) == _canon(f, r["labels"][f]))
                for f in FIELDS}
        reason = ("checker_schema" if bad_labels(cl) else "doubt" if c[i].get("doubt")
                  else "intent_disagree" if not same["intent"] else "action_disagree" if not same["action"]
                  else next((f"{f}_disagree" for f in CRITERIA if not same[f]), None))
        if reason:
            drop[reason] += 1
            drop_by[r["stratum"]][reason] += 1
            continue
        labels = {}
        for f in FIELDS:
            field_agree[f] += same[f]
            labels[f] = r["labels"][f] if same[f] else ("disagree" if f in TRI else None)
        kept.append({"id": i, "lang": r["lang"], "stratum": r["stratum"], "channel": r["channel"], "text": r["text"],
                     "contacts": r.get("contacts", []), "labels": labels, "fitting": fitting(labels, inv)})
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
    frac = DEV_N / len(rows)
    for k in sorted(cells):
        cell = cells[k]
        rng.shuffle(cell)
        nd = round(len(cell) * frac)
        dev += cell[:nd]
        test += cell[nd:]
    hd, ht = dump(D / "dev.jsonl", dev), dump(D / "test.jsonl", test)
    for name, h in (("dev", hd), ("test", ht)):
        (D / f"{name}.sha256").write_text(h + "\n")
    read = random.Random(f"{SEED}:read30").sample(test, 30)
    lines = ["# exp_050: 30 test leads for Andrei to read before signing", "",
             "*Drawn at random (seed 50:read30) from the test set. Each shows the message, the agreed labels and the homes "
             "that fit under the fit rule. Note anything you would label differently.*", ""]
    inv = {r["ref"]: r for r in inventory()}
    for r in read:
        L = r["labels"]
        fit = ", ".join(f"{x} ({inv[x]['type']}, {AREA_NAMES[inv[x]['area']]}, {inv[x]['price_eur']:,})" for x in r["fitting"][:4])
        lines += [f"## {r['id']} · {r['lang']} · {r['stratum']}", "", "> " + r["text"].replace("\n", "\n> "), "",
                  f"- **{L['intent']}** · {L['property_type']} · timeline {L['timeline']} · qualified {L['qualified']} · "
                  f"urgency {L['urgency']} · **{L['action']}**",
                  f"- budget {L['budget_eur']} · areas {L['areas'] or '–'} · bedrooms {L['min_bedrooms']} · "
                  f"features {L['features'] or '–'}",
                  f"- fits: {len(r['fitting'])}{' — ' + fit if fit else ''}{' …' if len(r['fitting']) > 4 else ''}", ""]
    (WS / "data" / "READ30.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"dev": len(dev), "test": len(test), "dev_sha": hd[:16], "test_sha": ht[:16],
                      "test_by_stratum": dict(Counter(r["stratum"] for r in test)),
                      "test_by_lang": dict(Counter(r["lang"] for r in test))}, indent=1))


def cmd_extra():
    """Run 3 adds two listings that fit no test enquiry under the fit rule: the inventory's 'version 2'."""
    kept = load(D / "agreed.jsonl")                   # dev and test both: dev rehearses run 3 too
    rng = random.Random(f"{SEED}:extra")
    picked = []
    while len(picked) < 2:
        op = rng.choice(["sale", "long_let"])
        typ = rng.choice(list(_MIX[op]))
        r = _listing(rng, op, typ, rng.choice(AREAS))
        if not any(fits({**t["labels"]}, r) for t in kept if t["labels"]["intent"] in SEEKS):
            picked.append({"ref": f"EX-{161 + len(picked)}", **r, "price_eur": int(r["price_eur"])})
    h = dump(D / "extra.jsonl", picked)
    print(json.dumps({"extra": picked, "sha256": h[:16]}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------------------------------------
# Runs

ARMS = {"G": ("society", "gemma4:26b"), "M": ("monolith", "gemma4:26b"), "Gs": ("society", "gemma4:e4b"),
        "Ms": ("monolith", "gemma4:e4b")}            # M-31 cut (Andrei 2026-10-23): ~14 h with thinking
# Set on dev (HYPOTHESIS "As built"): M thinks (30 dev leads: wrong sent 10 → 1); temperature 0 tied the default (24 vs
# 24 of 74), so the default stays; G's writer stays as it is (its decisions are not the writer's); unsure cut 0.5.
THINK = {"G": False, "M": True, "Gs": False, "Ms": True}
TEMP = {"G": None, "M": None, "Gs": None, "Ms": None}                     # None = the model's default (1.0 as shipped)


def run_inventory(k: int) -> list[dict]:
    """Run k sees the inventory in its own order; run 3 also gets the two extra listings (the inventory's version 2)."""
    inv = inventory() + (load(D / "extra.jsonl") if k == 3 else [])
    random.Random(f"{SEED}:order:{k}").shuffle(inv)
    return inv


def _services_up() -> bool:
    import urllib.request
    from scripts import kev_knife as K
    try:
        for u in (K.KEV_URL, K.FLASH_URL):
            with urllib.request.urlopen(f"{u}/health", timeout=5) as r:
                if json.load(r).get("status") != "ok":
                    return False
        return True
    except Exception:
        return False


def cmd_run(set_: str, arm: str, k: int, max_s: float = 0.0, tag: str = ""):
    """Dev only: a tag names a tuning variant, and EXP50_THINK / EXP50_TEMP / EXP50_UNSURE / EXP50_IDS override the
    settings. A test run takes the settings frozen in this file, with no tag and no override."""
    import os
    import time
    from scripts import desk50
    over = {k_: os.environ[k_] for k_ in ("EXP50_THINK", "EXP50_TEMP", "EXP50_UNSURE", "EXP50_IDS") if k_ in os.environ}
    if set_ == "test" and (tag or over):
        sys.exit("REFUSED: no tags or overrides on the test set")
    if arm.startswith("G") and not _services_up():
        sys.exit("REFUSED: Kev (:18442) and Flash (:18448) must both be up")
    think = over["EXP50_THINK"] == "1" if "EXP50_THINK" in over else THINK[arm]
    temp = float(over["EXP50_TEMP"]) if "EXP50_TEMP" in over else TEMP[arm]
    if "EXP50_UNSURE" in over:
        desk50.UNSURE_CUT = float(over["EXP50_UNSURE"])
    rows = load(D / f"{set_}.jsonl")
    if "EXP50_IDS" in over:
        keep = set(over["EXP50_IDS"].split(","))
        rows = [r for r in rows if r["id"] in keep]
    out = ROOT / "runs" / f"exp050_{set_}_{arm}_r{k}{'_' + tag if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    pred_p, log_p = out / "pred.jsonl", out / "prompts.jsonl"
    done = {r["id"] for r in load(pred_p)} if pred_p.exists() else set()
    if set_ == "test" and len(done) >= len(rows):     # never repeated; skipping keeps the tmux block restartable
        print(f"{out.name} is complete: skipped (a test pass is not repeated)")
        return
    inv = run_inventory(k)
    fn, model = ARMS[arm]
    t0 = time.time()
    for i, r in enumerate(x for x in rows if x["id"] not in done):
        for attempt in range(1, 6):                      # a serving failure is retried, never recorded as an answer
            desk50.LOG = []
            try:
                p = getattr(desk50, fn)(r["text"], inv, model, think=think, temperature=temp)
                break
            except desk50.InfraError as e:
                print(f"[{arm} r{k} {set_}] INFRA attempt {attempt}/5 on {r['id']}: {e}", flush=True)
                if attempt == 5:
                    sys.exit(f"STOPPED on infrastructure failures at {r['id']}; nothing recorded for it. Run again to resume.")
                time.sleep(60)
        p["id"] = r["id"]
        with log_p.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": r["id"], "calls": desk50.LOG}, ensure_ascii=False) + "\n")
        with pred_p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(p, ensure_ascii=False, default=str) + "\n")
        n = len(done) + i + 1
        print(f"[{arm} r{k} {set_}] {n}/{len(rows)} {r['id']} {'READY' if p['ready'] else 'NEEDS YOU: ' + str(p['reason'])} "
              f"{p.get('ms')} ms", flush=True)
        if max_s and time.time() - t0 > max_s:
            print(f"stopped after {max_s:.0f} s; run again to resume")
            return


# ---------------------------------------------------------------------------------------------------------------------
# Scoring (the same code for every arm)

SHOULD_READY = ("call_now", "ask_details", "book_valuation")

# Scorer correction, 2026-10-24, after the test (EXP50_SCORER=corrected; the registered scoring stays the default).
# Two false-positive classes in E3, found by reading every flagged reply, one against each desk:
#   (1) budget_words reads English slang "grand" (= 1,000) inside "grand plaisir" (French): G, 2 replies;
#   (2) the sender's own phone, spelled out in words in the enquiry and written in digits in the reply, was not in the
#       record because numbers() reads no spelled digits: M 2, M-small 1.
# Corrected: "grand" is not a number in a reply, and a reply number of 4+ digits that is part of a phone in the
# enquiry (in digits, or spelled out in any of the 7 languages) is the sender's own contact, not a fact.
import os as _os
SCORER = _os.environ.get("EXP50_SCORER", "registered")
_DW = {}
for _d, _ws in {0: "zero oh o cero noll null zéro ноль", 1: "one uno una jeden eins ett un один", 2: "two dos dwa zwei två deux два",
                3: "three tres trzy drei tre trois три", 4: "four cuatro cztery vier fyra quatre четыре",
                5: "five cinco pięć fünf fem cinq пять", 6: "six seis sześć sechs sex шесть", 7: "seven siete siedem sieben sju sept семь",
                8: "eight ocho osiem acht åtta huit восемь", 9: "nine nueve dziewięć neun nio neuf девять"}.items():
    for _w in _ws.split():
        _DW[_w] = str(_d)
_TENS = {"forty": "4", "cuarenta": "4", "czterdzieści": "4", "vierzig": "4", "fyrtio": "4", "quarante": "4", "сорок": "4"}
_HUNDREDS = {"hundred", "cientos", "sto", "hundert", "hundra", "cent"}


def spelled_digits(text: str) -> list[str]:
    """Runs of 7+ spelled-out digits in 7 languages: 'nine hundred' → 900, 'forty four' / 'cuarenta y cuatro' → 44."""
    out, cur, toks, i = [], "", re.findall(r"[^\W\d_]+", text.lower()), 0
    while i < len(toks):
        w = toks[i]
        if w in _DW:
            cur += _DW[w]
        elif w in _TENS:
            j = i + 2 if i + 2 < len(toks) and toks[i + 1] in ("y", "und", "et") else i + 1
            cur += _TENS[w] + _DW.get(toks[j] if j < len(toks) else "", "0")
            i = j
        elif w in ("novecientos", "девятьсот"):
            cur += "900"
        elif w in _HUNDREDS and cur:
            cur += "00"
        elif w not in ("plus", "más", "плюс", "double"):
            if len(cur) >= 7:
                out.append(cur)
            cur = ""
        i += 1
    return out + ([cur] if len(cur) >= 7 else [])


def _e3(reply: str, enquiry: str, rows: list[dict]) -> list[float]:
    from scripts.desk50 import record_values, ungrounded
    if SCORER != "corrected":
        return ungrounded(reply, record_values(enquiry, rows))
    from scripts import kev_knife as K
    phones = [re.sub(r"\D", "", m.group(0)) for m in K._phones(enquiry)] + spelled_digits(enquiry)
    reply = re.sub(r"\bgrand\b", " ", reply, flags=re.I)
    return [v for v in ungrounded(reply, record_values(enquiry, rows))
            if not (v >= 1000 and v == int(v) and any(str(int(v)) in ph for ph in phones))]


def _inv_by_ref(k: int) -> dict:
    return {r["ref"]: r for r in run_inventory(k)}


def errors(r: dict, p: dict, inv: dict) -> dict:
    from scripts.desk50 import lang_of, record_values, ungrounded
    L = r["labels"]
    fit = set(r["fitting"])
    offered = [x for x in p.get("offered") or []]
    rows = [inv[x] for x in offered if x in inv]
    reply = p.get("reply") or ""
    e1 = p.get("action") != L["action"]
    e2 = bool(set(offered) - fit) or (p["ready"] and p.get("action") == "send_listings" and not offered)
    e3 = _e3(reply, r["text"], rows) if reply else []
    e4 = bool(reply) and lang_of(reply) != r["lang"]
    return {"E1": e1, "E2": e2, "E3": bool(e3), "E3_values": e3[:5], "E4": e4}


def outcome(r: dict, p: dict, inv: dict) -> dict:
    e = errors(r, p, inv)
    wrong = e["E1"] or e["E2"] or e["E3"] or e["E4"]
    should = r["labels"]["action"] in SHOULD_READY or (r["labels"]["action"] == "send_listings" and bool(r["fitting"]))
    return {**e, "ready": p["ready"], "wrong": wrong, "wrong_sent": p["ready"] and wrong, "should_ready": should,
            "right_ready": p["ready"] and not wrong, "decision": [p["ready"], p.get("action"), sorted(p.get("offered") or [])]}


def gate_M(r: dict, p: dict, inv: dict) -> dict:
    """M+audit, offline: G's code checks applied to M's whole reply. The gate sees only what code sees (the enquiry and
    the offered rows), never the labels: an ungrounded number, a reply not in the enquiry's language, or a contact that
    is not the sender's own holds the reply."""
    from scripts import kev_knife as K
    from scripts.desk50 import lang_of, record_values, ungrounded
    q = dict(p)
    reply = p.get("reply") or ""
    rows = [inv[x] for x in p.get("offered") or [] if x in inv]
    foreign = [m.group(0) for m in K._EMAIL.finditer(reply) if m.group(0) not in r["text"]]
    if p["ready"] and (_e3(reply, r["text"], rows) or lang_of(reply) != lang_of(r["text"]) or foreign):
        q["ready"], q["reason"] = False, "audit"
    return q


def _mcnemar_one_sided(b: int, c: int) -> float:
    """P(X ≥ b) for X ~ Bin(b + c, 0.5): b = M wrong and G not, c = G wrong and M not."""
    from math import comb
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(b, n + 1)) / 2 ** n


def _contact_hits(contacts: list[str], prompts: list) -> list[str]:
    """A labelled contact reached a model if its digits appear as one run in a prompt (separators between digits
    ignored), a run of 6+ of them survived masking, or, spelled out or an e-mail, it appears as written."""
    text = json.dumps(prompts, ensure_ascii=False).lower()
    runs = re.findall(r"\d+", re.sub(r"(?<=\d)[\s().\-]+(?=\d)", "", text))
    hits = []
    for c in contacts:
        d = re.sub(r"\D", "", c)
        if len(d) >= 7:
            if any(d in x or (len(x) >= 6 and x in d) for x in runs):
                hits.append(c)
        elif c.lower() in text:
            hits.append(c)
    return hits


def cmd_compare(set_: str):
    rows = {r["id"]: r for r in load(D / f"{set_}.jsonl")}

    def preds(arm, k):
        p = ROOT / "runs" / f"exp050_{set_}_{arm}_r{k}" / "pred.jsonl"
        return {x["id"]: x for x in load(p)} if p.exists() else {}

    res, outs = {}, {}
    for arm, k in (("G", 1), ("G", 2), ("G", 3), ("M", 1), ("M", 2), ("M", 3), ("Gs", 1), ("Ms", 1)):
        P = preds(arm, k)
        if not P:
            continue
        inv = _inv_by_ref(k)
        outs[(arm, k)] = {i: outcome(rows[i], P[i], inv) for i in rows if i in P}
        if (arm, k) == ("M", 1):
            outs[("M+audit", 1)] = {i: outcome(rows[i], gate_M(rows[i], P[i], inv), inv) for i in rows if i in P}
    for (arm, k), O in outs.items():
        n = len(O)
        sr = [o for o in O.values() if o["should_ready"]]
        res[f"{arm} r{k}"] = {
            "n": n, "ready": round(sum(o["ready"] for o in O.values()) / max(1, n), 3),
            "wrong_sent": round(sum(o["wrong_sent"] for o in O.values()) / max(1, n), 3),
            "wrong_sent_n": sum(o["wrong_sent"] for o in O.values()),
            "should_ready_n": len(sr), "right_ready_of_should": round(sum(o["right_ready"] for o in sr) / max(1, len(sr)), 3),
            "errors_when_sent": {e: sum(o[e] for o in O.values() if o["wrong_sent"]) for e in ("E1", "E2", "E3", "E4")},
            "wrong_sent_by_stratum": {s: sum(o["wrong_sent"] for i, o in O.items() if rows[i]["stratum"] == s) for s in STRATA},
            "wrong_sent_without_questions": round(sum(o["wrong_sent"] for i, o in O.items() if rows[i]["stratum"] != "question")
                                                  / max(1, sum(rows[i]["stratum"] != "question" for i in O)), 3),
            "right_ready_by_lang": {l: round(sum(o["right_ready"] for i, o in O.items() if rows[i]["lang"] == l and o["should_ready"])
                                             / max(1, sum(rows[i]["lang"] == l and o["should_ready"] for i, o in O.items())), 3)
                                    for l in sorted({r["lang"] for r in rows.values()})}}
    H = {}
    G1, M1 = outs.get(("G", 1)), outs.get(("M", 1))
    if G1 and M1 and set(G1) == set(M1) == set(rows):
        wg, wm = res["G r1"]["wrong_sent"], res["M r1"]["wrong_sent"]
        b = sum(M1[i]["wrong_sent"] and not G1[i]["wrong_sent"] for i in rows)
        c = sum(G1[i]["wrong_sent"] and not M1[i]["wrong_sent"] for i in rows)
        pval = _mcnemar_one_sided(b, c)
        H["H1"] = {"G": wg, "M": wm, "b_M_only": b, "c_G_only": c, "p": round(pval, 4),
                   "pass": (wg <= wm) if wm < 0.04 else (wg <= 0.5 * wm and pval < 0.05)}
        sg, sm = res["G r1"]["right_ready_of_should"], res["M r1"]["right_ready_of_should"]
        H["H2"] = {"G": sg, "M": sm, "pass": sg >= 0.60 and sg >= sm - 0.10}
    if all(outs.get((a, k)) for a in ("G", "M") for k in (1, 2, 3)):
        st = {a: round(sum(outs[(a, 1)][i]["decision"] == outs[(a, 2)][i]["decision"] == outs[(a, 3)][i]["decision"]
                           for i in rows) / len(rows), 3) for a in ("G", "M")}
        H["H3"] = {**st, "pass": st["G"] >= 0.95 and st["G"] - st["M"] >= 0.10}
    if G1 and outs.get(("Gs", 1)):
        d_w = res["Gs r1"]["wrong_sent"] - res["G r1"]["wrong_sent"]
        d_r = res["Gs r1"]["right_ready_of_should"] - res["G r1"]["right_ready_of_should"]
        H["H4"] = {"wrong_sent_rise": round(d_w, 3), "right_ready_fall": round(-d_r, 3), "pass": d_w <= 0.02 and -d_r <= 0.10}
    hits, scanned = [], 0
    for arm in ("G", "Gs"):
        for k in (1, 2, 3):
            lp = ROOT / "runs" / f"exp050_{set_}_{arm}_r{k}" / "prompts.jsonl"
            if lp.exists():
                for x in load(lp):
                    scanned += 1
                    h = _contact_hits(rows[x["id"]]["contacts"], x["calls"]) if x["id"] in rows else []
                    hits += [(arm, k, x["id"], c) for c in h]
    if scanned:
        H["H5"] = {"prompts_scanned": scanned, "contacts_found": len(hits), "examples": hits[:5], "pass": not hits}
    print(json.dumps({"set": set_, "arms": res, "hypotheses": H}, ensure_ascii=False, indent=1))


def cmd_tune_unsure():
    """Dev only, offline: G's unsure cut. The leads G held as unsure in r1 were re-run with no cut (tag unsure0); at cut c,
    a lead with intent confidence below c is held, otherwise it takes the unsure0 result. Choice: the highest share of
    should-be-ready leads sent right, with no more wrong replies sent than at 0.5; ties → the higher (more cautious) cut."""
    rows = {r["id"]: r for r in load(D / "dev.jsonl")}
    base = {x["id"]: x for x in load(ROOT / "runs/exp050_dev_G_r1/pred.jsonl")}
    free = {x["id"]: x for x in load(ROOT / "runs/exp050_dev_G_r1_unsure0/pred.jsonl")}
    inv = _inv_by_ref(1)
    grid = []
    for c in (0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.0):
        P = {i: (free[i] if i in free and free[i]["intent_conf"] >= c else base[i]) for i in rows}
        O = [outcome(rows[i], P[i], inv) for i in rows]
        sr = [o for o in O if o["should_ready"]]
        grid.append({"cut": c, "wrong_sent": sum(o["wrong_sent"] for o in O),
                     "right_ready_of_should": round(sum(o["right_ready"] for o in sr) / len(sr), 3)})
    w05 = grid[0]["wrong_sent"]
    best = max((g for g in grid if g["wrong_sent"] <= w05), key=lambda g: (g["right_ready_of_should"], g["cut"]))
    print(json.dumps({"grid": grid, "chosen": best}, indent=1))


def cmd_langcheck():
    from scripts.desk50 import lang_of
    rows = load(D / "agreed.jsonl")
    miss = [(r["id"], r["lang"], lang_of(r["text"])) for r in rows if lang_of(r["text"]) != r["lang"]]
    print(json.dumps({"n": len(rows), "wrong": len(miss), "by_lang": dict(Counter(m[1] for m in miss)), "examples": miss[:10]}, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = {"inventory": cmd_inventory, "validate": cmd_validate, "agree": cmd_agree, "split": cmd_split, "extra": cmd_extra,
            "langcheck": cmd_langcheck, "tune-unsure": cmd_tune_unsure}
    if a and a[0] in cmds:
        cmds[a[0]]()
    elif a and a[0] == "run" and len(a) >= 4:
        cmd_run(a[1], a[2], int(a[3]), float(a[4]) if len(a) > 4 else 0.0, a[5] if len(a) > 5 else "")
    elif a and a[0] == "compare" and len(a) > 1:
        cmd_compare(a[1])
    elif a and a[0] == "checker":
        cmd_checker(int(a[1]) if len(a) > 1 else 60)
    else:
        sys.exit(__doc__)
