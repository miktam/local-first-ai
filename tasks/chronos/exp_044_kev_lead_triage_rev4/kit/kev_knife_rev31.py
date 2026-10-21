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
}

_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_MONEY = re.compile(r"(?:€\s?|eur\s?)?(\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)\s?(k|m|mln|million|millones|mio|tys|млн|тыс|€|eur|euros?|евро)(?!\w)", re.I)


def code_facts(text: str) -> dict:
    """Deterministic facts; contact details are detected, never kept."""
    budget = None
    m = _MONEY.search(text)
    if m:
        num = float(re.sub(r"[.,\s]", "", m.group(1)) if re.search(r"[.,\s]\d{3}", m.group(1)) else m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        budget = num * (1e6 if unit in ("m", "mln", "million", "millones", "mio", "млн") else 1e3 if unit in ("k", "tys", "тыс") else 1)
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


B2B_CUT = 0.6   # set on exp_043 dev: no client above 0.55, 16/21 spam above 0.6 (0.5 flagged 5 clients as spam)


def policy(o: dict, facts: dict) -> tuple[int, str]:
    """Urgency and next action from the agency's written rules (tasks/exp043/LABELS.md), applied by code to Kev's answers."""
    if o["b2b"] >= B2B_CUT and o["intent"]["choice"] != "spam":   # a business pitching the agency is spam by LABELS.md
        o["intent"] = {"choice": "spam", "confidence": o["b2b"], "by": "b2b rule"}
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
    "Hi, we are a Swedish couple looking for a 3-bed villa in Benahavís, budget around 1.5M, cash buyers, visiting next week. Can someone call me? +44 7700 900123",
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
