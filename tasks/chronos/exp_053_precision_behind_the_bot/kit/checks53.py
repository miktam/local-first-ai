"""exp_053: the code checks, taken from the bot's own rules (SYSTEM_PROMPT in scripts/telegram_bot.py).

An answer is flagged when any check fires:
  V1 language  the answer is not in the question's language (kit/langcheck.py)
  V2 price     a euro amount of 1,000 or more that is in neither the retrieved context nor the question
               (amounts are compared after normalising 450k / 450.000 € / €450,000 / 1,2 M / 2 millones /
               300 mil / 280 tys / 300 тыс, and the low end of a range takes the high end's multiplier:
               "400-450k", "entre 300 y 400 mil"; ±1 %). Changed from exp_052, whose checker missed these.
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
    r"(k\b|m\b|mln\b|mill(?:ones|ion|ions|iones)?\b|millones\b|mio\b|млн\b|тыс\b|mil\b|tys\b|tusen\b|tsd\b|mille\b)?"
    r"\s*(€|eur\b|euros?\b)?",
    re.IGNORECASE)
_THOUSAND = ("k", "тыс", "mil", "tys", "tusen", "tsd", "mille")
_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)(\s*(?:-|–|—|a|y|to|and|bis|do|à|till|och|und|et|i|до|и)\s*)"
                    r"(\d+(?:[.,]\d+)?\s*)(k|m|mil|tys|тыс|млн|mln|mio|millones|million|millions|tusen|tsd|mille)\b",
                    re.IGNORECASE)


def amounts(text: str) -> set:
    """Euro amounts of 1,000 or more, as integers. A bare number counts only with a € sign, a currency word or a
    k / M multiplier next to it."""
    out = set()
    text = _RANGE.sub(lambda m: m.group(1) + m.group(4) + m.group(2) + m.group(3) + m.group(4), text)
    for m in _AMT.finditer(text):
        num, mult, cur_after = m.group(1), (m.group(2) or "").lower(), m.group(3)
        cur_before = re.search(r"(€|eur|euros?)\s*$", text[max(0, m.start() - 6):m.start(1)], re.IGNORECASE)
        if not (mult or cur_after or cur_before):
            continue
        if mult == "m" and m.group(2) == "m" and not (cur_after or cur_before):
            continue                                 # "121 m construidos" is metres, not millions
        digits = re.sub(r"[\s  ]", "", num)
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", digits):
            val = float(re.sub(r"[.,]", "", digits))
        else:
            val = float(digits.replace(",", "."))
        if mult in _THOUSAND:
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


def check_v2(answer: str, lang: str, question: str, context: str, finish: str | None) -> dict:
    """exp_053 correction, applied after the run and reported beside the registered check:
    V1 uses langcheck.detect_v2 (place names ignored); V3 matches a place only as a whole word, so 'coincida' is
    not Coín and 'distancia' is not Istán. V2, V4 and V5 are unchanged."""
    r = check(answer, lang, question, context, finish)
    v = dict(r["violations"])
    if "V5" in v:
        return r
    got = langcheck.detect_v2(answer)
    v.pop("V1", None)
    if got != lang:
        v["V1"] = got
    known = fold(context + " " + question)
    fa = fold(answer)
    word = lambda p, s: re.search(r"(?<!\w)" + re.escape(fold(p)) + r"(?!\w)", s) is not None
    places = sorted(p for p in PLACES if word(p, fa) and not word(p, known))
    v.pop("V3", None)
    if places:
        v["V3"] = places
    return {"flagged": bool(v), "violations": v}

