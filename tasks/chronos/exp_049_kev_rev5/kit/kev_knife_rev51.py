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

Rev 4.1 (2026-10-20; rev 4 frozen as kev_knife_rev4.py): code facts only, from exp_045 — a currency symbol before full
digits ("€350,000"), bare grouped amounts after a budget or price cue ("up to 850,000", "Preisvorstellung 295.000"),
amounts per month ("до 2000 в месяц"), amounts grouped like phones no longer taken for phones ("1 600 000 euro"), and
phone numbers spelled out in words masked. The questions, rules and cuts are rev 4's.

Rev 4.2 (2026-10-21, exp_048's verdict, Andrei's choice; rev 4.1 frozen as kev_knife_rev41.py): two decision models.
Kev-4B answers the request side (the request, the business-pitch and wants-to-own questions); Clef-Flash 9B
(ai.casasol.triage-flash9b, :18448) answers every other question. Both get the full schema, as measured, at the same time;
rev 4.1's rules combine the answers. Flash unavailable → Kev answers everything, which is exactly rev 4.1.

Rev 5.1 (2026-10-21, exp_049's verdict: F1, F3 and F4 adopted, F2 off; rev 4.2 frozen as kev_knife_rev42.py, rev 5 as
kev_knife_rev5.py). The rev 5 changes, as measured:
- F1, intermediaries: an `on_behalf` question. A spam reading of someone writing for a client falls back to the best
  non-spam reading.
- F2, price only: a `price_only` question. A "sell" reading of an owner who only wants a price becomes valuation.
- F3, the date: the timeline question is asked with today's date (TODAY; None = the real date).
- F4, financing: the financing question asks "use", not "need" (exp_047: approved mortgages read as no need).
Cuts are tuned on exp_049's dev set only.

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
FLASH_URL = "http://127.0.0.1:18448"     # rev 4.2: Clef-Flash 9B answers the facts (exp_048)
REQUEST_SIDE = ("intent", "b2b", "wants_to_own")   # Kev's answers; the rest are Flash's, on_behalf and price_only included
# (exp_049 dev: Flash read on_behalf 161/163 against Kev's 156, and price_only 24/24 on the valuation stratum against 18)

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
    "financing": {"type": "noul", "instructions": "Will the purchase or rental use a mortgage or loan, already approved or not?"},
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
    # rev 5 (exp_049)
    "on_behalf": {"type": "noul", "instructions": "Is the sender writing on behalf of someone else who is the actual client "
                                                  "(a lawyer, notary, family office, relocation firm, employer, assistant or "
                                                  "relative), rather than for themselves or to sell the agency a service?"},
    "price_only": {"type": "noul", "instructions": "Does the sender only want to know what their property is worth, "
                                                   "without having decided to sell it?"},
}

TODAY = None   # rev 5 F3: the date the timeline question is asked with; None = today. exp_049 fixes it to 2026-10-21.
KEEP_RAW = False   # exp_049: record both models' raw answers in the output


def lead_schema() -> dict:
    """LEAD_SCHEMA with the timeline question dated (rev 5 F3): a bare month means nothing without today's date."""
    import datetime as dt
    d = TODAY or dt.date.today()
    day = f"{d.day} {d.strftime('%B')} {d.year}"
    s = dict(LEAD_SCHEMA)
    s["timeline"] = {"type": "choice", "instructions": f"Today is {day}. When does the sender want to act?",
                     "criteria": {"now": f"within about a month of {day}, urgently, or already visiting or ready to sign",
                                  "months": f"later in {d.year}, by 31 December {d.year}",
                                  "later": f"in {d.year + 1} or later, even if that is only weeks away",
                                  "unspecified": "no timing mentioned"}}
    return s

_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_GROUPED = re.compile(r"^\d{1,3}(?:[ .]\d{3})+$")          # "1 600 000", "3.500.000": shaped like an amount
_CURRENCY_AFTER = re.compile(r"^\s?(?:€|eur\b|euros?\b|евро|zł|złotych|kr\b|sek\b|\$|£)", re.I)
_CUE = (r"budget|budżet|бюджет|presupuesto|max(?:imum|imal)?|up\s+to|around|about|approx\.?|hasta|alrededor\s+de|unos|"
        r"bis(?:\s+zu)?|etwa|ungefähr|jusqu'?à|environ|około|do|до|около|upp\s+till|runt|cirka|"
        r"price|precio|prix|preis\w*|cena|цена|pris|asking|pide|vendo\s+por")
_CUE_BEFORE = re.compile(rf"(?:{_CUE})\b[^\n.!?]{{0,20}}$", re.I)       # "around 700 to 800,000" keeps its cue
_DIGIT_WORDS = (r"zero|oh|one|two|three|four|five|six|seven|eight|nine|hundred|double|triple|cero|uno|dos|tres|cuatro|"
                r"cinco|seis|siete|ocho|nueve|zéro|un|deux|trois|quatre|cinq|sept|huit|neuf|null|eins|zwei|drei|vier|"
                r"fünf|sechs|sieben|acht|neun|jeden|dwa|trzy|cztery|pięć|sześć|siedem|osiem|dziewięć|ноль|один|два|три|"
                r"четыре|пять|шесть|семь|восемь|девять|noll|ett|två|tre|fyra|fem|sex|sju|åtta|nio")
_SPELLED_PHONE = re.compile(rf"(?:\b(?:plus|más|плюс)[\s,]+)?\b(?:{_DIGIT_WORDS})\b(?:[\s,.-]+(?:{_DIGIT_WORDS})\b){{6,}}", re.I)
_MONEY = re.compile(r"(?:€\s?|eur\s?)?(\d{1,3}(?:[.,\s]\d{3})+|\d+(?:[.,]\d+)?)\s?(k|m|mln|million|millones|mio|tys|млн|тыс|€|eur|euros?|евро)(?!\w)", re.I)
# rev 4.1: a currency symbol before full digits with no unit after ("€350,000", "up to €2,000 a month")
_MONEY_PREFIX = re.compile(r"(?:€|£|\$)\s?(\d{1,3}(?:[.,\s]\d{3})+|\d{3,})(?![\d.,]*\s?(?:k|m|mln|million|millones|mio|tys|млн|тыс)\b)", re.I)
# rev 4.1: bare grouped amounts after a budget/price cue, and amounts per month, as Whisper writes spoken money
_BARE = re.compile(r"(?<![\d+])(\d{1,3}(?:[.,\s]\d{3})+)(?![\d]|[.,]\d|\s?(?:m²|m2|sqm|sq\s?m|metros|meter|mètres|кв|kvm|ha\b|k?m\b(?!€)))")
_PER_MONTH = re.compile(r"(?<![\d+])(\d{3,5})\s?(?:€|eur|euros?|евро)?\s?(?:/\s?(?:mes|month|mo|mån|monat|mois|m-c|мес)\b|a\s+month|per\s+month|al\s+mes|"
                        r"im\s+monat|par\s+mois|в\s+месяц|miesięcznie|i\s+månaden|pm\b)", re.I)


def _figures(text: str) -> list[float]:
    """Every amount written with digits and a unit ("1.2M", "800k", "450.000 €", "3 млн")."""
    out = []
    for m in _MONEY.finditer(text):
        num = float(re.sub(r"[.,\s]", "", m.group(1)) if re.search(r"[.,\s]\d{3}", m.group(1)) else m.group(1).replace(",", "."))
        unit = m.group(2).lower()
        if unit == "m" and num >= 50:     # "300 m de la plage" is metres, not €300M (found in exp_044 writer data, 2026-10-19)
            continue
        out.append(num * (1e6 if unit in ("m", "mln", "million", "millones", "mio", "млн") else 1e3 if unit in ("k", "tys", "тыс") else 1))
    for m in _MONEY_PREFIX.finditer(text):
        out.append(_plain(m.group(1)))
    for m in _BARE.finditer(text):
        if _CUE_BEFORE.search(text[max(0, m.start() - 40):m.start()]) and not _is_phone_context(text, m.start()):
            out.append(_plain(m.group(1)))
    for m in _PER_MONTH.finditer(text):
        out.append(float(m.group(1)))
    return out


def _plain(s: str) -> float:
    return float(re.sub(r"[.,\s]", "", s))


def _is_phone_context(text: str, start: int) -> bool:
    return bool(re.search(r"(?:\+|\b(?:tel|phone|móvil|movil|telefon\w*|телефон\w*|nummer|numéro|numer|whatsapp)\b)[^\n]{0,12}$",
                          text[max(0, start - 20):start], re.I))


def _phones(text: str) -> list[re.Match]:
    """Phone-like digit runs, minus amounts: a run shaped like "1 600 000" with a currency right after it or a price or
    budget cue right before it is money, not a phone (exp_045: "1 600 000 euro", "3.500.000 €")."""
    out = []
    for m in _PHONE.finditer(text):
        span = m.group(0).strip()
        if _GROUPED.match(span) and (_CURRENCY_AFTER.match(text[m.end():m.end() + 9])
                                     or _CUE_BEFORE.search(text[max(0, m.start() - 40):m.start()])):
            continue
        out.append(m)
    return out


def code_facts(text: str) -> dict:
    """Deterministic facts; contact details are detected, never kept. rev 4: the budget is the largest amount stated,
    with digits or in words (scripts/budget_words.py), as LABELS.md defines it."""
    from scripts.budget_words import amounts
    found = [a for a in _figures(text) + amounts(text) if 0 < a < 1e9]
    budget = max(found) if found else None
    from scripts.telegram_bot import detect_language          # the bot's 7-language detector
    return {"language": detect_language(text), "budget_eur": budget,
            "has_phone": bool(_phones(text) or _SPELLED_PHONE.search(text)), "has_email": bool(_EMAIL.search(text))}


def masked(text: str) -> str:
    """Contacts out before Kev: phone numbers in digits (amounts kept) and spelled out in words, then e-mails."""
    for m in reversed(_phones(text)):
        text = text[:m.start()] + "[phone]" + text[m.end():]
    text = _SPELLED_PHONE.sub("[phone]", text)
    return _EMAIL.sub("[email]", text)


def ask(state: str, schema: dict, url: str | None = None) -> dict:
    body = json.dumps({"state": state, "questions": schema}).encode()
    req = urllib.request.Request(f"{url or KEV_URL}/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["answers"]


# rev 4 cuts, tuned on the exp_044 dev set only (workspace tasks/exp044/HYPOTHESIS.md, as built). Frozen 2026-10-19.
B2B_CUT = 0.6       # the business-pitch answer at or above this ... (dev: tied best; the most cautious of the ties)
SPAM_AGREE = 0.15   # ... overrides Kev's reading only if Kev's own spam probability also reaches this (rev 3: no second condition)
OWN_CUT = 0.65      # "wants to own" at or above this turns a holiday-rent reading into buy (dev request 0.903 off → 0.923)
OWN_REVERSE = False  # the reverse (buy → holiday rent) changed nothing on dev, so it stays off
# rev 5 cuts, tuned on exp_049's dev set only (163 enquiries, offline over both models' recorded answers; the best dev
# request accuracy with no more false spam than rev 4.2 and spam recall within 0.05; ties → the most cautious). Frozen
# 2026-10-21: dev request 0.933 (rev 4.2) → 0.969, false spam 1 → 0, recall 1.00 both.
ON_BEHALF_CUT = 0.8     # F1: Flash's on_behalf at or above this overturns a spam reading
PRICE_CUT = None        # F2 OFF in rev 5.1: exp_049's H3 fell short (+0.074, p 0.0625; 6 corrections, 0 errors). Was 0.7.
PRICE_REVERSE = False   # F2 reverse (valuation → sell): tied on dev, off as the more cautious


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
    # rev 5 F1: someone writing for a client is not spam; fall back to the best non-spam reading.
    ob = o.get("on_behalf")
    if ON_BEHALF_CUT is not None and ob is not None and o["intent"]["choice"] == "spam" and ob >= ON_BEHALF_CUT:
        probs = (o.get("intent_kev") or {}).get("probabilities") or {}
        best = max(((c, p) for c, p in probs.items() if c != "spam"), key=lambda kv: kv[1], default=None)
        if best:
            o["intent"] = {"choice": best[0], "confidence": round(best[1], 2), "probabilities": probs, "by": "on-behalf rule"}
    # rev 5 F2: an owner who only wants a price is a valuation, not a sale.
    po = o.get("price_only")
    if PRICE_CUT is not None and po is not None:
        if o["intent"]["choice"] == "sell" and po >= PRICE_CUT:
            o["intent"] = {**o["intent"], "choice": "valuation", "by": "price-only rule"}
        elif PRICE_REVERSE and o["intent"]["choice"] == "valuation" and po <= 1 - PRICE_CUT:
            o["intent"] = {**o["intent"], "choice": "sell", "by": "price-only rule"}
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


def _answers(a: dict) -> dict:
    out = {}
    for k, v in a.items():
        if v["type"] == "choice":
            out[k] = {"choice": v["choice"], "confidence": round(v["confidence"], 2),
                      "probabilities": {o: round(pr, 3) for o, pr in (v.get("probabilities") or {}).items()}}
        elif v["type"] == "noul":
            out[k] = round(v["noul"], 2)
        else:
            out[k] = round(v["score"], 2)
    return out


def triage(text: str) -> dict:
    """rev 4.2: Kev and Flash answer the full schema at the same time; Kev's request side, Flash's everything else, rev 4.1's
    rules over the merged answers (exp_048's hybrid). If Flash fails, Kev's answers are used throughout (rev 4.1)."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    state = masked(text)
    with ThreadPoolExecutor(max_workers=2) as pool:
        schema = lead_schema()
        f_kev = pool.submit(ask, state, schema, KEV_URL)
        f_flash = pool.submit(ask, state, schema, FLASH_URL) if FLASH_URL else None
        _answers_raw_kev = _answers(f_kev.result())
        out = {"facts": code_facts(text), **_answers_raw_kev}
        flash = None
        if f_flash is not None:
            try:
                flash = _answers(f_flash.result())
            except Exception:
                flash = None
    if flash:
        out.update({k: v for k, v in flash.items() if k not in REQUEST_SIDE})
    out["models"] = {"request": "kev-4b", "facts": "clef-flash-9b" if flash else "kev-4b"}
    if KEEP_RAW:                         # exp_049 dev: both models' answers, so tuning can choose sources offline
        out["raw"] = {"kev": _answers_raw_kev, "flash": flash}
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
