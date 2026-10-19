#!/usr/bin/env python3
"""
Kev knife — a Swiss-army set of closed decisions for a property agency, answered by one local decision model
(Kev-4B on the mini's llama.cpp /v1/systemone, service ai.casasol.gate-kev4b). Prototype, 2026-10-18.

Blade 1: lead triage. One call per inbound message answers every typed question below; Kev returns a probability
per option and never writes text. Exact facts (language, budget figure, phone/e-mail presence) come from code, and
contact details are never sent to Kev or stored: they are masked before the call.

  .venv/bin/python3 scripts/kev_knife.py demo             # synthetic leads only
  .venv/bin/python3 scripts/kev_knife.py lead "<text>"     # one message → JSON

Rev 2 (exp_043 dev tuning, 2026-10-18): sharper intent descriptions; Kev answers plain yes/no facts (hot,
specific listing, call request, asks worth; rev 3 added b2b) and code derives urgency and the next action from the agency's written
rules (workspace tasks/exp043/LABELS.md) — Kev no longer guesses the action or the urgency.
After exp_043 (Andrei): spam is never discarded — its action is `review` (a human review folder). The test measured
`discard`; the label is renamed, the decision is the same, nothing is thrown away.

Unmeasured: accuracy and calibration on real leads are unknown until a labelled set exists.

Rev 4 (exp_044, 2026-10-19; rev 3.1 frozen as kev_knife_rev31.py): a "wants to own" question for the holiday-home
confusion; the business-pitch answer overrides Kev only when Kev's own spam probability agrees; budgets are the largest
amount stated, with digits or in words. Cuts are tuned on the exp_044 dev set only.

Full reference — questions, rules, thresholds, the /triage bot demo, voice, privacy rules, exp_043 results, known
weaknesses, how to change it safely: TRIAGE.md.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root, for scripts.telegram_bot

KEV_URL = "http://127.0.0.1:18442"

LEAD_SCHEMA = {
    "intent": {"type": "choice", "instructions": "What does the sender mainly want?",
               "criteria": {"buy": "wants to buy a home to live in or use themselves",
                            "rent_long": "wants a rental of six months or more (or offers their property for a long let)",
                            "rent_holiday": "wants a rental of a few weeks or a holiday let",
                            "sell": "wants to sell or list their property, even if they also ask what it is worth",
                            "valuation": "only wants to know what their property is worth, not asking to sell",
                            "invest": "buying not to live in: rental yield, development, land banking or several units",
                            "info": "asks a question about law, taxes, process or an area, without asking for a transaction",
                            "spam": "not a client looking for a property: advertising, a supplier, a scam, another agency prospecting"}},
    "property_type": {"type": "choice", "instructions": "Which kind of property is it about?",
                      "criteria": {"apartment": "flat or apartment", "penthouse": None, "townhouse": "townhouse or semi-detached",
                                   "villa": "detached villa or house", "plot": "land or plot", "commercial": "shop, office or commercial premises",
                                   "unspecified": "no type mentioned"}},
    "timeline": {"type": "choice", "instructions": "When does the sender want to act?",
                 "criteria": {"now": "within about a month, urgently, or already visiting or ready to sign",
                              "months": "within the next months or this year", "later": "next year or later",
                              "unspecified": "no timing mentioned"}},
    "financing": {"type": "noul", "instructions": "Will the sender need a mortgage or financing?"},
    "foreign_buyer": {"type": "noul", "instructions": "Is the sender a foreigner or non-resident in Spain?"},
    "qualified": {"type": "noul", "instructions": "Does the message state enough to act on: a clear goal plus at least two of budget, area and timing?",
                  "criteria": {"true": "clear goal and at least two of budget, area, timing", "false": "vague or missing most details"}},
    "hot": {"type": "noul", "instructions": "Is the sender visiting today or tomorrow, about to make an offer or sign, or talking to another agency at the same time?"},
    "specific_listing": {"type": "noul", "instructions": "Does the sender ask about one specific listing (a reference number, link or address)?"},
    "call_request": {"type": "noul", "instructions": "Does the sender ask to be phoned or called back?"},
    "asks_worth": {"type": "noul", "instructions": "Does the sender ask what their property is worth?"},
    "b2b": {"type": "noul", "instructions": "Is this a business writing to the agency to sell it a service or propose a deal "
                                            "(marketing, leads, a portal, software, another agency asking to share commission), "
                                            "rather than a private person with their own property need?"},
    # rev 4 (exp_044): the holiday-home buyer vs holiday renter confusion (10 of 25 intent errors in exp_043)
    "wants_to_own": {"type": "noul", "instructions": "Does the sender want to buy a property to own, even a holiday or "
                                                     "second home, rather than rent one?"},
}

_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_MONEY = re.compile(r"(?:€\s?|eur\s?)?(\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)\s?(k|m|mln|million|millones|mio|tys|млн|тыс|€|eur|euros?|евро)(?!\w)", re.I)


def _figures(text: str) -> list[float]:
    """Every amount written with digits and a unit ("1.2M", "800k", "450.000 €", "3 млн")."""
    out = []
    for m in _MONEY.finditer(text):
        num = float(re.sub(r"[.,\s]", "", m.group(1)) if re.search(r"[.,\s]\d{3}", m.group(1)) else m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        if unit == "m" and num >= 50:     # "300 m de la plage" is metres, not €300M (found in exp_044 writer data, 2026-10-19)
            continue
        out.append(num * (1e6 if unit in ("m", "mln", "million", "millones", "mio", "млн") else 1e3 if unit in ("k", "tys", "тыс") else 1))
    return out


def code_facts(text: str) -> dict:
    """Deterministic facts; contact details are detected, never kept. rev 4: the budget is the largest amount stated,
    with digits or in words (scripts/budget_words.py), as LABELS.md defines it."""
    from scripts.budget_words import amounts
    found = [a for a in _figures(text) + amounts(text) if 0 < a < 1e9]
    budget = max(found) if found else None
    from scripts.telegram_bot import detect_language          # the bot's 7-language detector
    return {"language": detect_language(text), "budget_eur": budget,
            "has_phone": bool(_PHONE.search(text)), "has_email": bool(_EMAIL.search(text))}


def masked(text: str) -> str:
    return _EMAIL.sub("[email]", _PHONE.sub("[phone]", text))


def ask(state: str, schema: dict) -> dict:
    body = json.dumps({"state": state, "questions": schema}).encode()
    req = urllib.request.Request(f"{KEV_URL}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]


# rev 4 cuts, tuned on the exp_044 dev set only (workspace tasks/exp044/HYPOTHESIS.md, as built). Frozen 2026-10-19.
B2B_CUT = 0.6       # the business-pitch answer at or above this ... (dev: tied best; the most cautious of the ties)
SPAM_AGREE = 0.15   # ... overrides Kev's reading only if Kev's own spam probability also reaches this (rev 3: no second condition)
OWN_CUT = 0.65      # "wants to own" at or above this turns a holiday-rent reading into buy (dev request 0.903 off → 0.923)
OWN_REVERSE = False  # the reverse (buy → holiday rent) changed nothing on dev, so it stays off


def policy(o: dict, facts: dict) -> tuple[int, str]:
    """Urgency and next action from the agency's written rules (tasks/exp044/LABELS.md), applied by code to Kev's answers."""
    o["intent_kev"] = dict(o["intent"])      # Kev's own reading (with its probabilities), before any rule
    p_spam = (o["intent"].get("probabilities") or {}).get("spam", 0.0)
    if o["intent"]["choice"] != "spam" and o["b2b"] >= B2B_CUT and p_spam >= SPAM_AGREE:
        o["intent"] = {"choice": "spam", "confidence": o["b2b"], "by": "b2b rule"}    # a business pitching the agency
    own = o.get("wants_to_own")
    if own is not None:
        if o["intent"]["choice"] == "rent_holiday" and own >= OWN_CUT:
            o["intent"] = {**o["intent"], "choice": "buy", "by": "own rule"}
        elif OWN_REVERSE and o["intent"]["choice"] == "buy" and own <= 1 - OWN_CUT:
            o["intent"] = {**o["intent"], "choice": "rent_holiday", "by": "own rule"}
    intent, tl = o["intent"]["choice"], o["timeline"]["choice"]
    q, hot, spec = o["qualified"] >= 0.5, o["hot"] >= 0.5, o["specific_listing"] >= 0.5
    call = facts["has_phone"] or o["call_request"] >= 0.5
    if intent in ("info", "spam"):
        urgency = 0
    elif hot:
        urgency = 3
    elif (q and tl == "now") or spec:
        urgency = 2
    elif tl in ("now", "months"):
        urgency = 1
    else:
        urgency = 0
    if intent == "spam":
        action = "review"          # never discard (Andrei, 2026-10-18): spam goes to a human review folder
    elif intent == "info":
        action = "answer_question"
    elif intent == "valuation" or (intent == "sell" and o["asks_worth"] >= 0.5):
        action = "book_valuation"
    elif q and tl == "now" and call:
        action = "call_now"
    elif q:
        action = "book_valuation" if intent == "sell" else "send_listings"
    else:
        action = "ask_details"
    return urgency, action


def triage(text: str) -> dict:
    t0 = time.time()
    a = ask(masked(text), LEAD_SCHEMA)
    out = {"facts": code_facts(text)}
    for k, v in a.items():
        if v["type"] == "choice":
            out[k] = {"choice": v["choice"], "confidence": round(v["confidence"], 2),
                      "probabilities": {o: round(pr, 3) for o, pr in (v.get("probabilities") or {}).items()}}
        elif v["type"] == "noul":
            out[k] = round(v["noul"], 2)
        else:
            out[k] = round(v["score"], 2)
    out["urgency"], out["action"] = policy(out, out["facts"])
    out["action_by"] = "rules"
    out["ms"] = round((time.time() - t0) * 1000)
    return out


DEMO = [
    "Hi, we are a Swedish couple looking for a 3-bed villa in Benahavís, budget around 1.5M, cash buyers, visiting next week. Can someone call me? +46 70 123 4567",
    "Dzień dobry, szukamy mieszkania do wynajęcia na dłużej w Marbelli, 2 sypialnie, do 2000 € miesięcznie, od stycznia.",
    "Hola, quiero vender mi ático en Nueva Andalucía y saber cuánto vale. ¿Pueden venir a verlo esta semana?",
    "Is it true that non-residents pay a 3% withholding when they sell? Just curious, not selling yet.",
    "Boost your agency's Google ranking! SEO packages from 99€/month, reply YES for a free audit.",
    "Hallo, wir suchen ein Grundstück an der Costa del Sol zum Bauen, vielleicht nächstes Jahr. Finanzierung über eine deutsche Bank.",
]


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "lead":
        print(json.dumps(triage(sys.argv[2]), ensure_ascii=False, indent=1))
        return
    for text in DEMO:
        r = triage(text)
        f = r["facts"]
        print(f"\n{text[:95]}…")
        print(f"  [{f['language']}] budget={f['budget_eur']} phone={f['has_phone']} · intent={r['intent']['choice']}({r['intent']['confidence']}) "
              f"type={r['property_type']['choice']} timeline={r['timeline']['choice']} qualified={r['qualified']} "
              f"financing={r['financing']} foreign={r['foreign_buyer']} urgency={r['urgency']}"
              f" · action={r['action']} (by {r['action_by']}) · {r['ms']} ms")


if __name__ == "__main__":
    main()
