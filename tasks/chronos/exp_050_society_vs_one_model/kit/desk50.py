#!/usr/bin/env python3
"""
exp_050 — the two front desks (workspace tasks/exp050/HYPOTHESIS.md). Synthetic only.

G, the society: every model answers only what it reads well; code holds everything between them.
  enquiry → code (language, budget, contacts masked) → Kev-4B + Clef-Flash 9B (triage rev 5.1, unchanged)
          → code (the agency's rules → next step) → Clef-Flash 9B (what the sender needs: areas, bedrooms, 8 features)
          → code (the fit rule over the agency's inventory → up to 3 homes, or none)
          → writer model (1–3 sentences in the sender's language, no numbers)
          → code (listing cards + closing line rendered from the record; audit of the writer's sentences)
          → READY or NEEDS YOU, with the reason

M, one model: the enquiry, the agency's rules and its whole inventory in one call → intent, next step, listings,
needs_human, reason and the reply.

Every prompt sent to any model is logged (invariant 3 of library/patterns/deterministic-glue-pipeline.md; H5).

  .venv/bin/python3 scripts/desk50.py G "<enquiry>"     # one enquiry through G, as JSON
  .venv/bin/python3 scripts/desk50.py M "<enquiry>"     # one enquiry through M
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts import kev_knife as K  # noqa: E402
from scripts.budget_words import amounts  # noqa: E402
from scripts.leads50 import AREAS, AREA_NAMES, FEATURES, SEEKS, fitting, inventory, inventory_md  # noqa: E402

TODAY = dt.date(2026, 10, 23)            # the label date (LABELS.md v4); both desks are told it
K.TODAY = TODAY
K.KEEP_RAW = True
OLLAMA = "http://127.0.0.1:11434"
NUM_CTX = 16384     # dev: M's prompt (3.8K) plus its thinking (up to ~4.4K) filled the server's 8,192 and stopped on
                    # 'length' with no answer (4 of 30). Every exp_050 call uses 16K, so no arm reloads the model.
WRITER = "gemma4:26b"
SENDABLE = ("send_listings", "call_now", "ask_details", "book_valuation")
NEEDS_CUT = 0.5                          # yes/no answers in the listing join, as in policy()
UNSURE_CUT = 0.5                         # intent confidence below this → NEEDS YOU (rev 5.1's TRIAGE_SURE); set on dev
AGENCY_TERMS = [6, 8]                    # numbers the agency itself states to both desks: lets of 6 months+, eight areas
LOG: list[dict] | None = None            # set by the runner: every prompt sent to a model is appended here


def _log(model: str, prompt):
    if LOG is not None:
        LOG.append({"model": model, "prompt": prompt})


_ask = K.ask


def _logged_ask(state, schema, url=None):
    """Every decision-model call, Kev's and Flash's, as sent: triage() and needs() both go through K.ask."""
    _log({K.KEV_URL: "kev-4b", K.FLASH_URL: "clef-flash-9b"}.get(url or K.KEV_URL, url), state)
    return _ask(state, schema, url)


K.ask = _logged_ask


# ---------------------------------------------------------------------------------------------------------------------
# Language (code): stopwords plus letters only one language uses. Used by G for the target language and its audit,
# and by the scorer for E4.

_SW = {
    "en": "the and to of in is for we you my our are with on have be this that it at from would looking like please "
          "thanks hello can do will your not or but me us am want need there what",
    "es": "el la los las que y en un una para por con mi nos es estamos buscando gustaría hola gracias su lo del al se no "
          "muy busco tengo queremos le les somos sería pero también usted",
    "pl": "w na z do się że jest nie jak dla po czy jestem szukam szukamy chcemy mam bardzo dzień dobry witam proszę "
          "mieszkanie który które oraz jesteśmy chcielibyśmy pan pani mój nasz",
    "de": "der die das und ich wir sie ist nicht ein eine mit für zu auf den dem von suchen haben möchten bitte hallo "
          "guten tag gerne es auch noch uns unsere wäre sind",
    "sv": "och att det är ett jag vi på för med som till av har inte vill hej tack söker letar efter skulle kan om men "
          "oss vår vårt hälsningar gärna",
    "fr": "le la les et des un une je nous vous est pour dans avec sur pas que qui bonjour merci cherchons recherche "
          "voudrions notre mon au du serait aussi madame monsieur",
    "ru": "и в не на я что с мы по для это как у вы но мне нас есть хотим ищем здравствуйте спасибо пожалуйста",
}
_SW = {k: set(v.split()) for k, v in _SW.items()}
_ONLY = {"pl": "ąęłńśźż", "sv": "å", "de": "ßü", "es": "ñ¿¡", "fr": "çèêàùœ"}
LANG_NAME = {"en": "English", "es": "Spanish", "pl": "Polish", "de": "German", "sv": "Swedish", "fr": "French", "ru": "Russian"}


def lang_of(text: str) -> str:
    low = text.lower()
    for name in AREA_NAMES.values():                 # Spanish place names would count as Spanish words
        low = low.replace(name.lower(), " ")
    if sum("а" <= c <= "я" or c == "ё" for c in low) >= 3:
        return "ru"
    toks = re.findall(r"[^\W\d_]+", low)
    score = {k: sum(t in sw for t in toks) for k, sw in _SW.items() if k != "ru"}
    for k, chars in _ONLY.items():
        score[k] += 2 * sum(c in low for c in set(chars))
    best = max(score, key=lambda k: (score[k], k == "en"))
    return best if score[best] > 0 else "en"


# ---------------------------------------------------------------------------------------------------------------------
# The kernel: fixed text, rendered by code from the record, in 7 languages.

_TYPE = {"en": ("apartment", "penthouse", "townhouse", "villa", "plot", "commercial premises"),
         "es": ("apartamento", "ático", "adosado", "villa", "parcela", "local comercial"),
         "pl": ("apartament", "penthouse", "szeregowiec", "willa", "działka", "lokal użytkowy"),
         "de": ("Wohnung", "Penthouse", "Reihenhaus", "Villa", "Grundstück", "Gewerbefläche"),
         "sv": ("lägenhet", "takvåning", "radhus", "villa", "tomt", "lokal"),
         "fr": ("appartement", "penthouse", "maison mitoyenne", "villa", "terrain", "local commercial"),
         "ru": ("квартира", "пентхаус", "таунхаус", "вилла", "участок", "коммерческое помещение")}
_BEDS = {"en": lambda n: f"{n} bedroom{'s' if n != 1 else ''}", "es": lambda n: f"{n} dormitorio{'s' if n != 1 else ''}",
         "pl": lambda n: f"sypialnie: {n}", "de": lambda n: f"{n} Schlafzimmer", "sv": lambda n: f"{n} sovrum",
         "fr": lambda n: f"{n} chambre{'s' if n != 1 else ''}", "ru": lambda n: f"спален: {n}"}
_SEP = {"en": ",", "es": ".", "de": ".", "pl": " ", "sv": " ", "fr": " ", "ru": " "}
_MONTH = {"en": "a month", "es": "al mes", "pl": "miesięcznie", "de": "pro Monat", "sv": "i månaden", "fr": "par mois",
          "ru": "в месяц"}
CLOSING = {
    "send_listings": {
        "en": "Would you like to see any of them? Reply with the reference and we'll arrange a viewing.",
        "es": "¿Le gustaría ver alguna? Responda con la referencia y organizamos una visita.",
        "pl": "Chętnie umówimy oglądanie: wystarczy odpisać z numerem referencyjnym.",
        "de": "Möchten Sie eine davon besichtigen? Antworten Sie einfach mit der Referenz, dann vereinbaren wir einen Termin.",
        "sv": "Vill du titta på någon av dem? Svara med referensen så bokar vi en visning.",
        "fr": "Souhaitez-vous en visiter une ? Répondez avec la référence et nous organiserons une visite.",
        "ru": "Хотите посмотреть какой-нибудь из них? Ответьте с номером объекта, и мы организуем просмотр."},
    "call_now": {
        "en": "An agent will call you shortly.", "es": "Un agente le llamará en breve.",
        "pl": "Nasz agent wkrótce zadzwoni.", "de": "Ein Makler ruft Sie in Kürze an.",
        "sv": "En mäklare ringer dig inom kort.", "fr": "Un agent vous appellera très bientôt.",
        "ru": "Наш агент скоро вам позвонит."},
    "ask_details": {
        "en": "As soon as we have these details, we'll send you a selection of homes.",
        "es": "En cuanto tengamos estos datos, le enviaremos una selección de viviendas.",
        "pl": "Gdy tylko otrzymamy te informacje, prześlemy wybór ofert.",
        "de": "Sobald wir diese Angaben haben, schicken wir Ihnen eine Auswahl passender Immobilien.",
        "sv": "Så snart vi har uppgifterna skickar vi ett urval av bostäder.",
        "fr": "Dès que nous aurons ces informations, nous vous enverrons une sélection de biens.",
        "ru": "Как только мы получим эти данные, мы пришлём вам подборку объектов."},
    "book_valuation": {
        "en": "Let us know a day and time that suits you, and an agent will visit to value your property.",
        "es": "Indíquenos qué día y hora le vienen bien, y un agente visitará la vivienda para valorarla.",
        "pl": "Prosimy o informację, jaki dzień i godzina Państwu odpowiadają, a nasz agent przyjedzie wycenić nieruchomość.",
        "de": "Nennen Sie uns einen passenden Tag und eine Uhrzeit, dann kommt ein Makler zur Bewertung vorbei.",
        "sv": "Hör av dig med en dag och tid som passar, så kommer en mäklare och värderar bostaden.",
        "fr": "Indiquez-nous le jour et l'heure qui vous conviennent, et un agent viendra estimer votre bien.",
        "ru": "Сообщите удобные день и время, и наш агент приедет оценить вашу недвижимость."}}
SIGN = {"en": "Kind regards, the team", "es": "Un saludo, el equipo", "pl": "Pozdrawiamy, zespół",
        "de": "Viele Grüße, Ihr Team", "sv": "Vänliga hälsningar, teamet", "fr": "Cordialement, l'équipe",
        "ru": "С уважением, команда"}


def _price(v: int, lang: str) -> str:
    s = f"{v:,}".replace(",", _SEP[lang])
    return f"€{s}" if lang == "en" else f"{s} €"


def card(r: dict, lang: str) -> str:
    parts = [r["ref"], _TYPE[lang][("apartment", "penthouse", "townhouse", "villa", "plot", "commercial").index(r["type"])],
             AREA_NAMES[r["area"]]]
    if r["bedrooms"]:
        parts.append(_BEDS[lang](r["bedrooms"]))
    parts += [f"{r['m2']} m²", _price(r["price_eur"], lang) + (f" {_MONTH[lang]}" if r["operation"] == "long_let" else "")]
    return " · ".join(parts)


# ---------------------------------------------------------------------------------------------------------------------
# Numbers (code): the audit forbids them in the writer's sentences; the scorer's E3 checks every number a reply states.

_NUM_WORDS = set((
    "two three four five six seven eight nine ten eleven twelve twenty thirty hundred thousand million millions "
    "dos tres cuatro cinco seis siete ocho nueve diez once doce veinte treinta cien ciento mil millón millones "
    "dwa dwie dwóch trzy trzech cztery czterech pięć sześć siedem osiem dziewięć dziesięć dwadzieścia sto tysiąc tysięcy "
    "milion miliony milionów zwei drei vier fünf sechs sieben acht neun zehn elf zwölf zwanzig hundert tausend million "
    "millionen två tre fyra fem sex sju åtta nio tio elva tolv tjugo hundra tusen miljon miljoner deux trois quatre cinq "
    "sept huit dix onze douze vingt trente cent mille millions два две двух три трёх трех четыре четырёх пять шесть семь "
    "восемь девять десять двадцать сто тысяча тысяч тысячи миллион миллиона миллионов").split())
_SMALL = {w: v for v, ws in {
    2: "two dos dwa dwie dwóch zwei två deux два две двух", 3: "three tres trzy trzech drei tre trois три трёх трех",
    4: "four cuatro cztery czterech vier fyra quatre четыре четырёх", 5: "five cinco pięć fünf fem cinq пять",
    6: "six seis sześć sechs sex шесть", 7: "seven siete siedem sieben sju sept семь", 8: "eight ocho osiem acht åtta huit восемь",
    9: "nine nueve dziewięć neun nio девять", 10: "ten diez dziesięć zehn tio dix десять"}.items() for w in ws.split()}
_REF = re.compile(r"\bEX-?\s?\d{3}\b", re.I)
_NUM = re.compile(r"(?<![\w.,])(\d{1,3}(?:[.,   ]\d{3})+|\d+(?:[.,]\d+)?)(?:\s?(k|m|mln|mio|million(?:s|en)?|millones|"
                  r"miljon(?:er)?|milion(?:y|ów)?|млн|миллион(?:а|ов)?|тыс|tys|thousand|mil)(?![\w²]))?", re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "mil": 1e3, "tys": 1e3, "тыс": 1e3}


def numbers(text: str) -> list[float]:
    """Every number stated: digits with their multiplier, amounts in words, and small counts in words (2–10)."""
    t = _REF.sub(" ", text)
    out = []
    for m in _NUM.finditer(t):
        s, mult = m.group(1), (m.group(2) or "").lower()
        if re.fullmatch(r"\d{1,3}(?:[.,   ]\d{3})+", s):
            v = float(re.sub(r"\D", "", s))
        else:
            v = float(s.replace(",", "."))
        out.append(v * (_MULT.get(mult, 1e6 if mult else 1)))
    out += amounts(t)
    out += [_SMALL[w] for w in re.findall(r"[^\W\d_]+", t.lower()) if w in _SMALL]
    return out


def ungrounded(reply: str, record: list[float]) -> list[float]:
    """E3: numbers in the reply that are in neither the enquiry nor the offered listings (0 and 1 are not facts)."""
    def ok(v):
        return v in (0, 1) or any(abs(v - r) <= max(0.005 * abs(r), 0.0) for r in record)
    return [v for v in numbers(reply) if not ok(v)]


def record_values(enquiry: str, offered: list[dict]) -> list[float]:
    vals = numbers(enquiry) + [len(offered)] + AGENCY_TERMS
    for r in offered:
        vals += [r["price_eur"], r["m2"]] + ([r["bedrooms"]] if r["bedrooms"] else [])
    return vals


# ---------------------------------------------------------------------------------------------------------------------
# Models

class InfraError(RuntimeError):
    """The serving stack failed, not the model: no tokens generated, a cut-off response, a dropped connection, or Flash
    missing from the triage. Never recorded as an arm's answer; the runner retries the lead or stops (2026-10-23)."""


def ollama(model: str, messages: list[dict], schema: dict, think: bool = False, temperature: float | None = None) -> tuple[dict, dict]:
    body = {"model": model, "messages": messages, "format": schema, "stream": False, "think": think,
            "options": {"num_ctx": NUM_CTX, **({} if temperature is None else {"temperature": temperature})}}
    _log(model, messages)
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
    except Exception as e:
        raise InfraError(f"{model}: {type(e).__name__}: {e}") from e
    if not out.get("done") or not out.get("eval_count") or out.get("done_reason") not in ("stop", "length"):
        raise InfraError(f"{model}: incomplete response (done={out.get('done')}, done_reason={out.get('done_reason')}, "
                         f"eval_count={out.get('eval_count')})")
    meta = {k: out.get(k) for k in ("total_duration", "prompt_eval_count", "eval_count", "done_reason")}
    return json.loads(out["message"]["content"]), meta     # invalid JSON from a finished call is the model's own


MARBELLA = ("san_pedro", "nueva_andalucia", "guadalmina", "marbella_centre", "puerto_banus", "elviria")
NEEDS_SCHEMA = {
    **{f"area_{a}": {"type": "noul", "instructions": f"Does the sender name {AREA_NAMES[a]} (or say near {AREA_NAMES[a]}) "
                                                      f"as a place where they want the home?"} for a in AREAS},
    "area_marbella": {"type": "noul", "instructions": "Does the sender ask for Marbella in general, without naming any "
                                                       "district or area of it?"},
    "area_other": {"type": "noul", "instructions": "Does the sender name a place to live that is outside these areas: "
                   + ", ".join(AREA_NAMES.values()) + " (and Marbella in general)?"},
    "min_bedrooms": {"type": "choice", "instructions": "What is the smallest number of bedrooms the sender would accept?",
                     "criteria": {"none": "no number of bedrooms is mentioned", "1": "one", "2": "two", "3": "three",
                                  "4": "four", "5": "five or more"}},
    **{f"feat_{f}": {"type": "noul", "instructions": q} for f, q in {
        "sea_view": "Does the sender require a sea view (not just 'ideally' or 'a bonus')?",
        "pool": "Does the sender require a pool, private or communal (not just 'ideally' or 'a bonus')?",
        "garden": "Does the sender require a garden (not just 'ideally' or 'a bonus')?",
        "terrace": "Does the sender require a terrace or balcony (not just 'ideally' or 'a bonus')?",
        "parking": "Does the sender require parking or a garage (not just 'ideally' or 'a bonus')?",
        "walk_to_beach": "Does the sender require being within walking distance of the beach (not just 'ideally')?",
        "near_schools": "Does the sender require schools nearby for children (not just 'ideally')?",
        "golf": "Does the sender require being on or next to a golf course (not just 'ideally')?"}.items()}}


def needs(state: str) -> dict:
    return K._answers(K.ask(state, NEEDS_SCHEMA, K.FLASH_URL))


def criteria(t: dict, n: dict) -> dict:
    named = {a for a in AREAS if n[f"area_{a}"] >= NEEDS_CUT} | (set(MARBELLA) if n["area_marbella"] >= NEEDS_CUT else set())
    areas = [a for a in AREAS if a in named] + (["other"] if n["area_other"] >= NEEDS_CUT else [])   # LABELS v4: Marbella = six
    b = n["min_bedrooms"]["choice"]
    return {"intent": t["intent"]["choice"], "property_type": t["property_type"]["choice"], "areas": areas,
            "min_bedrooms": None if b == "none" else int(b), "budget_eur": t["facts"]["budget_eur"],
            "features": [f for f in FEATURES if n[f"feat_{f}"] >= NEEDS_CUT]}


def rank(c: dict, inv: list[dict]) -> list[dict]:
    """Fitting homes in a fixed order, whatever the inventory's order: within budget first (closest below it), then the
    10 % over; with no budget, cheapest first; ties by ref."""
    refs = set(fitting(c, inv))
    rows = [r for r in inv if r["ref"] in refs]
    b = c.get("budget_eur")
    key = (lambda r: (r["price_eur"] > b, -r["price_eur"] if r["price_eur"] <= b else r["price_eur"], r["ref"])) if b \
        else (lambda r: (r["price_eur"], r["ref"]))
    return sorted(rows, key=key)


WRITER_SYSTEM = """You write the opening of a reply from a Costa del Sol estate agency to an enquiry.
Write 1 to 3 short, warm, professional sentences in {language}. Reply to this sender: you may refer to what they wrote.

Rules:
- No numbers of any kind: no digits, no numbers written as words, no prices, sizes, counts, dates, times or references.
- No phone numbers, e-mail addresses or links.
- No promises: no times, discounts, availability or guarantees.
- Do not describe any home; the agency adds the listings and the closing line after your text.
- Do not sign the message.

Your task: {task}

Return JSON: {{"text": "..."}}"""
_TASK = {"send_listings": "Thank them, and say we have picked {homes} what they asked for; {follow} below.",
         "call_now": "Thank them, and say an agent will call them to talk it through.",
         "ask_details": "Thank them, and ask in one friendly sentence for: {missing}.",
         "book_valuation": "Thank them, and say we would be glad to value their property."}
_MISSING = {"budget": "their budget", "area": "the area they prefer", "timing": "when they plan to buy or move",
            "property": "where their property is and what kind of property it is"}


def audit(text: str, lang: str) -> str | None:
    """Can only hold a reply back, never approve one the rules held (asymmetric authority)."""
    if not text.strip():
        return "empty"
    if len(text) > 600:
        return "too long"
    if re.search(r"\d", text) or set(re.findall(r"[^\W\d_]+", text.lower())) & _NUM_WORDS:
        return "a number in the writer's text"
    if K._PHONE.search(text) or K._EMAIL.search(text) or re.search(r"\[(phone|email)\]|https?://|www\.", text, re.I):
        return "a contact or link in the writer's text"
    if lang_of(text) != lang:
        return f"writer's text not in {LANG_NAME[lang]}"
    return None


def society(text: str, inv: list[dict], writer: str = WRITER, think: bool = False, temperature: float | None = None) -> dict:
    t0 = time.time()
    try:
        t = K.triage(text)
    except Exception as e:
        raise InfraError(f"triage: {type(e).__name__}: {e}") from e
    if t["models"]["facts"] != "clef-flash-9b":          # rev 5.1 falls back to Kev alone; G is defined with Flash
        raise InfraError("triage: Flash did not answer")
    lang = lang_of(text)
    action = t["action"]
    out = {"arm": "G", "lang": lang, "action": action, "intent": t["intent"]["choice"],
           "intent_conf": t["intent"]["confidence"], "triage": {k: v for k, v in t.items() if k != "raw"},
           "triage_raw": t.get("raw"), "offered": [], "needs": None, "criteria": None, "ready": False, "reason": None,
           "writer_text": None, "audit": None, "reply": None}

    def hold(reason):
        out["reason"] = reason
        out["ms"] = round((time.time() - t0) * 1000)
        return out

    if action == "review":
        return hold("spam: review folder")
    if action == "answer_question":
        return hold("a question for the agent")
    if t["intent"]["confidence"] < UNSURE_CUT:
        return hold("unsure of the request")
    offered, missing = [], []
    if out["intent"] in SEEKS:
        try:
            n = needs(K.masked(text))
        except Exception as e:
            raise InfraError(f"needs: {type(e).__name__}: {e}") from e
        c = criteria(t, n)
        out["needs"], out["criteria"] = n, c
        if action == "send_listings":
            offered = rank(c, inv)[:3]
            out["offered"] = [r["ref"] for r in offered]
            if not offered:
                return hold("no home fits")
        if action == "ask_details":
            missing = [k for k, absent in (("budget", c["budget_eur"] is None), ("area", not c["areas"]),
                                           ("timing", t["timeline"]["choice"] == "unspecified")) if absent]
    elif action == "ask_details":                           # a seller or owner without enough to act on
        missing = ["property"]
    task = _TASK[action].format(homes="a home that matches" if len(offered) == 1 else "homes that match",
                                follow="it follows" if len(offered) == 1 else "they follow", missing=", ".join(_MISSING[m] for m in missing) or "a little more about what they are looking for")
    msgs = [{"role": "system", "content": WRITER_SYSTEM.format(language=LANG_NAME[lang], task=task)},
            {"role": "user", "content": K.masked(text)}]
    try:
        w, meta = ollama(writer, msgs, {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}}},
                         think=think, temperature=temperature)
        wt = str(w.get("text", "")).strip()
    except InfraError:
        raise
    except Exception as e:                                   # the model's own bad output holds the reply
        return hold(f"writer failed: {type(e).__name__}")
    out["writer_text"], out["writer_meta"] = wt, meta
    out["audit"] = audit(wt, lang)
    body = [wt] + (["\n".join(card(r, lang) for r in offered)] if offered else []) + [CLOSING[action][lang], SIGN[lang]]
    out["reply"] = "\n\n".join(body)
    if out["audit"]:
        return hold(f"audit: {out['audit']}")
    out["ready"] = True
    out["ms"] = round((time.time() - t0) * 1000)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# M: one model does it all

M_SYSTEM = """You are the front desk of a Costa del Sol estate agency. Today is Friday 23 October 2026.
The agency sells homes and lets them long-term (6 months or more) in eight areas: San Pedro de Alcántara, Nueva Andalucía,
Guadalmina, Marbella centre, Puerto Banús, Elviria, Estepona and Benahavís. It does no holiday lets.

For the enquiry you receive, decide the sender's intent and the agency's next step, choose the homes to offer, and write
the reply. Follow the agency's rules exactly.

INTENT (the sender's main goal):
- buy: a home to own, including a holiday or second home they want to own.
- rent_long: a rental of 6 months or more. rent_holiday: a rental of weeks.
- sell: has decided to sell or list, even if also asking what it is worth.
- valuation: wants a price estimate and has not decided to sell.
- invest: buying for yield, development, land banking or several units.
- info: a question about law, taxes, process or an area, with no transaction asked for.
- spam: suppliers, advertising, scams, other agencies prospecting, anything that is not a client. A business or person
  acting for a genuine client (a lawyer, family office, relocation firm, assistant, relative) is a client, not spam.

QUALIFIED: the goal is clear and at least two of budget, area and timing are stated. Never for info or spam.
TIMELINE: now = within about a month (by about 23 November 2026), urgent, or already visiting or ready to sign;
months = later in 2026; later = 2027 or beyond, even if only weeks away; unspecified = no timing given.

NEXT STEP (apply in order; the first that fits decides):
1. spam → review
2. info → answer_question
3. valuation, or sell when they ask what the property is worth → book_valuation
4. qualified, timeline now, and a phone number or a request to call → call_now
5. qualified → send_listings (for sell without a valuation request → book_valuation)
6. otherwise → ask_details

WHICH HOMES FIT (only for buy and invest: homes for sale; rent_long: long lets; no other intent fits any home):
- type, if stated: apartment accepts an apartment or a penthouse; any other type only itself; if no type is stated,
  an apartment, penthouse, townhouse or villa (never a plot or commercial premises);
- area, if named: the home is in one of the areas named ("Marbella" with no district means San Pedro de Alcántara,
  Nueva Andalucía, Guadalmina, Marbella centre, Puerto Banús and Elviria); a place outside the eight areas fits nothing;
- price at most the budget plus 10 % (the monthly rent for a long let), if a budget is stated;
- at least the bedrooms asked for, if stated;
- every feature the sender requires (not ones they call "ideally" or "a bonus").

WHEN A PERSON MUST HANDLE IT: set needs_human to true if the next step is review or answer_question, if you are unsure
of the request, or if the next step is send_listings and no home fits. Otherwise set it to false.

THE REPLY (always write one, even when needs_human is true):
- in the language the enquiry is written in;
- for send_listings, offer up to 3 homes that fit, giving for each its ref, type, area, bedrooms, m² and price exactly
  as in the inventory;
- state no other numbers, except ones the sender wrote;
- no promises of times, discounts or availability; no phone numbers, e-mails or links;
- close to fit the next step: call_now, an agent will call; ask_details, ask for the missing budget, area or timing;
  book_valuation, ask for a day and time to visit and value the property;
- sign as "the team".

THE INVENTORY:
{inventory}
Return JSON with intent, action (the next step), listing_refs, needs_human, reason (one short sentence) and reply."""

M_SCHEMA = {"type": "object", "required": ["intent", "action", "listing_refs", "needs_human", "reason", "reply"],
            "properties": {"intent": {"enum": ["buy", "rent_long", "rent_holiday", "sell", "valuation", "invest", "info", "spam"]},
                           "action": {"enum": ["call_now", "send_listings", "ask_details", "book_valuation", "answer_question", "review"]},
                           "listing_refs": {"type": "array", "items": {"type": "string"}},
                           "needs_human": {"type": "boolean"}, "reason": {"type": "string"}, "reply": {"type": "string"}}}


def monolith(text: str, inv: list[dict], model: str = WRITER, think: bool = False, temperature: float | None = None) -> dict:
    t0 = time.time()
    table = inventory_md(inv).split("\n", 4)[-1]                 # the table only, in this run's order
    msgs = [{"role": "system", "content": M_SYSTEM.format(inventory=table)}, {"role": "user", "content": text}]
    out = {"arm": "M", "model": model, "ready": False, "offered": [], "reason": None, "reply": None}
    try:
        m, meta = ollama(model, msgs, M_SCHEMA, think=think, temperature=temperature)
    except InfraError:
        raise
    except Exception as e:
        out.update(reason=f"model failed: {type(e).__name__}", ms=round((time.time() - t0) * 1000))
        return out
    reply = str(m.get("reply", ""))
    refs = sorted({f"EX-{d}" for s in [*(m.get("listing_refs") or []), reply] for d in re.findall(r"EX-?\s?(\d{3})", str(s), re.I)})
    out.update(intent=m.get("intent"), action=m.get("action"), offered=refs, needs_human=bool(m.get("needs_human")),
               model_reason=m.get("reason"), reply=reply, meta=meta)
    out["ready"] = (not out["needs_human"]) and out["action"] in SENDABLE
    out["reason"] = None if out["ready"] else ("model: needs a person" if out["needs_human"] else f"model: {out['action']}")
    out["ms"] = round((time.time() - t0) * 1000)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("G", "M"):
        sys.exit(__doc__)
    inv = inventory()
    f = society if sys.argv[1] == "G" else monolith
    print(json.dumps(f(sys.argv[2], inv), ensure_ascii=False, indent=1, default=str))
