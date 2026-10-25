"""exp_052: the code checks, taken from the bot's own rules (SYSTEM_PROMPT in scripts/telegram_bot.py).

An answer is flagged when any check fires:
  V1 language  the answer is not in the question's language (kit/langcheck.py)
  V2 price     a euro amount of 1,000 or more that is in neither the retrieved context nor the question
               (amounts are compared after normalising 450k / 450.000 € / €450,000 / 1,2 M / 2 millones; ±1 %)
  V3 place     a place from PLACES named in the answer but in neither the context nor the question
  V4 contact   a phone number or e-mail address in the answer ("Do not reveal seller contact details")
  V5 failure   the bot's error message, an empty answer, a stop for length, or thinking markup in the answer
V2 also counts amounts the model computed (a tax worked out from a price); it is a check on numbers the record
does not hold, read the same way for both arms.
"""
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import langcheck  # noqa: E402

PLACES = ["Marbella", "Estepona", "Benahavís", "San Pedro", "Nueva Andalucía", "Puerto Banús", "Golden Mile",
          "Milla de Oro", "Elviria", "Casares", "Sotogrande", "La Zagaleta", "Mijas", "Fuengirola", "Benalmádena",
          "Torremolinos", "Málaga", "Manilva", "Cancelada", "Guadalmina", "Nerja", "Ronda", "Vélez-Málaga",
          "Rincón de la Victoria", "Alhaurín", "Coín", "Ojén", "Istán", "La Cala", "Calahonda", "Cabopino",
          "Los Monteros", "Río Real", "Sierra Blanca", "Las Chapas", "Aloha", "Las Brisas", "La Quinta",
          "El Paraíso", "Atalaya", "Duquesa", "Gibraltar", "Tarifa", "Cádiz", "Sevilla", "Granada", "Valle Romano",
          "Alcaidesa", "San Roque", "Los Arqueros", "Lomas de Marbella", "Bel Air", "El Rosario", "Artola",
          "Puerto de Cabopino", "Benalmádena Pueblo", "Mijas Pueblo", "Torre del Mar", "Almuñécar", "Frigiliana"]
LLM_ERROR = "__LLM_ERROR__"
MARKUP = ("<|channel>", "<channel|>", "<|turn>", "<turn|>", "<think>", "</think>")


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


_AMT = re.compile(
    r"(?:€|eur\b|euros?\b)?\s*(\d{1,3}(?:[.,\s  ]\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(k\b|m\b|mln\b|mill(?:ones|ion|ions|iones)?\b|millones\b|mio\b|млн\b|тыс\b)?\s*(€|eur\b|euros?\b)?",
    re.IGNORECASE)


def amounts(text: str) -> set:
    """Euro amounts of 1,000 or more, as integers. A bare number counts only with a € sign, a currency word or a
    k / M multiplier next to it."""
    out = set()
    for m in _AMT.finditer(text):
        num, mult, cur_after = m.group(1), (m.group(2) or "").lower(), m.group(3)
        cur_before = re.search(r"(€|eur|euros?)\s*$", text[max(0, m.start() - 6):m.start(1)], re.IGNORECASE)
        if not (mult or cur_after or cur_before):
            continue
        digits = re.sub(r"[\s  ]", "", num)
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", digits):
            val = float(re.sub(r"[.,]", "", digits))
        else:
            val = float(digits.replace(",", "."))
        if mult in ("k", "тыс"):
            val *= 1_000
        elif mult:
            val *= 1_000_000
        if val >= 1_000:
            out.add(round(val))
    return out


def _found(v: int, pool: set) -> bool:
    return any(abs(v - p) <= 0.01 * p for p in pool)


_PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,12}\d")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def check(answer: str, lang: str, question: str, context: str, finish: str | None) -> dict:
    v = {}
    if answer == LLM_ERROR or not answer.strip() or finish == "length" or any(t in answer for t in MARKUP):
        v["V5"] = "failure"
    if "V5" not in v:
        got = langcheck.detect(answer)
        if got != lang:
            v["V1"] = got
        pool = amounts(context) | amounts(question)
        extra = sorted(a for a in amounts(answer) if not _found(a, pool))
        if extra:
            v["V2"] = extra
        known = fold(context + " " + question)
        places = sorted(p for p in PLACES if fold(p) in fold(answer) and fold(p) not in known)
        if places:
            v["V3"] = places
        contacts = [m.group(0) for m in _EMAIL.finditer(answer)] + \
                   [m.group(0) for m in _PHONE.finditer(answer) if len(re.sub(r"\D", "", m.group(0))) >= 9
                    and not amounts(m.group(0))]
        if contacts:
            v["V4"] = contacts
    return {"flagged": bool(v), "violations": v}
