"""
Budgets written as words, in the 7 triage languages (exp_044, rev 4 of the lead triage; no model, a fixed word list).

  amounts("un millón y medio")            → [1500000.0]
  amounts("rund neunhunderttausend Euro")   → [900000.0]
  amounts("около девятисот тысяч евро")     → [900000.0]

A phrase counts only if it contains "thousand" or "million" (so "two bedrooms" is never a budget). German and Swedish
compounds are split into known parts ("tvåtusenfemhundra" → två|tusen|fem|hundra). Polish and Russian are matched by the
inflected forms listed below. Word lists are tuned on the exp_044 dev set only.
"""
from __future__ import annotations

import re

# value words: number → value; "H" hundred, "K" thousand, "M" million; "HALF" before a multiplier ("half a million");
# "PLUSHALF" after one ("y medio", "et demi", "с половиной"); "ONEHALF" = 1.5 ("полтора"); "POINT" decimals (en).
_W: dict[str, object] = {}


def _add(value, *words):
    for w in words:
        _W[w] = value


# English
_add(1, "one", "a", "an"); _add(2, "two"); _add(3, "three"); _add(4, "four"); _add(5, "five"); _add(6, "six"); _add(7, "seven")
_add(8, "eight"); _add(9, "nine"); _add(10, "ten"); _add(11, "eleven"); _add(12, "twelve"); _add(15, "fifteen"); _add(20, "twenty")
_add(25, "twentyfive"); _add(30, "thirty"); _add(40, "forty"); _add(50, "fifty"); _add(60, "sixty"); _add(70, "seventy")
_add(80, "eighty"); _add(90, "ninety"); _add("H", "hundred"); _add("K", "thousand", "grand"); _add("M", "million", "millions")
_add("HALF", "half"); _add("POINT", "point")
# Spanish
_add(1, "un", "uno", "una"); _add(2, "dos"); _add(3, "tres"); _add(4, "cuatro"); _add(5, "cinco"); _add(6, "seis"); _add(7, "siete")
_add(8, "ocho"); _add(9, "nueve"); _add(10, "diez"); _add(15, "quince"); _add(20, "veinte"); _add(25, "veinticinco"); _add(30, "treinta")
_add(40, "cuarenta"); _add(50, "cincuenta"); _add(60, "sesenta"); _add(70, "setenta"); _add(80, "ochenta"); _add(90, "noventa")
_add(100, "cien", "ciento"); _add(200, "doscientos", "doscientas"); _add(300, "trescientos", "trescientas")
_add(400, "cuatrocientos", "cuatrocientas"); _add(500, "quinientos", "quinientas"); _add(600, "seiscientos", "seiscientas")
_add(700, "setecientos", "setecientas"); _add(800, "ochocientos", "ochocientas"); _add(900, "novecientos", "novecientas")
_add("K", "mil"); _add("M", "millón", "millon", "millones"); _add("HALF", "medio")
# Polish
_add(1, "jeden", "jedna", "jednego"); _add(2, "dwa", "dwie", "dwóch"); _add(3, "trzy", "trzech"); _add(4, "cztery", "czterech")
_add(5, "pięć", "pięciu"); _add(6, "sześć", "sześciu"); _add(7, "siedem", "siedmiu"); _add(8, "osiem", "ośmiu"); _add(9, "dziewięć", "dziewięciu")
_add(10, "dziesięć"); _add(15, "piętnaście"); _add(20, "dwadzieścia"); _add(30, "trzydzieści"); _add(40, "czterdzieści")
_add(50, "pięćdziesiąt"); _add(60, "sześćdziesiąt"); _add(70, "siedemdziesiąt"); _add(80, "osiemdziesiąt"); _add(90, "dziewięćdziesiąt")
_add(100, "sto"); _add(200, "dwieście"); _add(300, "trzysta"); _add(400, "czterysta"); _add(500, "pięćset"); _add(600, "sześćset")
_add(700, "siedemset"); _add(800, "osiemset"); _add(900, "dziewięćset")
_add("K", "tysiąc", "tysiące", "tysięcy", "tysiąca", "tys"); _add("M", "milion", "miliony", "milionów", "miliona", "mln")
_add("HALF", "pół"); _add("ONEHALF", "półtora", "półtorej")
# German (also the parts of compounds)
_add(1, "ein", "eine", "einen", "eins"); _add(2, "zwei"); _add(3, "drei"); _add(4, "vier"); _add(5, "fünf"); _add(6, "sechs")
_add(7, "sieben"); _add(8, "acht"); _add(9, "neun"); _add(10, "zehn"); _add(20, "zwanzig"); _add(30, "dreißig"); _add(40, "vierzig")
_add(50, "fünfzig"); _add(60, "sechzig"); _add(70, "siebzig"); _add(80, "achtzig"); _add(90, "neunzig"); _add("H", "hundert")
_add("K", "tausend"); _add("M", "millionen", "mio"); _add("HALF", "halbe", "halb"); _add("PLUSHALF", "einhalb")
_add("ONEHALF", "anderthalb", "eineinhalb")
# Swedish
_add(1, "en", "ett"); _add(2, "två"); _add(3, "tre"); _add(4, "fyra"); _add(5, "fem"); _add(6, "sex"); _add(7, "sju"); _add(8, "åtta")
_add(9, "nio"); _add(10, "tio"); _add(20, "tjugo"); _add(30, "trettio"); _add(40, "fyrtio"); _add(50, "femtio"); _add(60, "sextio")
_add(70, "sjuttio"); _add(80, "åttio"); _add(90, "nittio"); _add("H", "hundra"); _add("K", "tusen"); _add("M", "miljon", "miljoner")
_add("HALF", "halv", "halva")
# French
_add(1, "un", "une"); _add(2, "deux"); _add(3, "trois"); _add(4, "quatre"); _add(5, "cinq"); _add(6, "six"); _add(7, "sept"); _add(8, "huit")
_add(9, "neuf"); _add(10, "dix"); _add(20, "vingt", "vingts"); _add(30, "trente"); _add(40, "quarante"); _add(50, "cinquante")
_add(60, "soixante"); _add(80, "quatrevingt", "quatrevingts"); _add("H", "cent", "cents"); _add("K", "mille")
_add("M", "million", "millions"); _add("HALF", "demi")
# Russian
_add(1, "один", "одна", "одного"); _add(2, "два", "две", "двух"); _add(3, "три", "трёх", "трех"); _add(4, "четыре", "четырёх", "четырех")
_add(5, "пять", "пяти"); _add(6, "шесть", "шести"); _add(7, "семь", "семи"); _add(8, "восемь", "восьми"); _add(9, "девять", "девяти")
_add(10, "десять", "десяти"); _add(20, "двадцать"); _add(30, "тридцать"); _add(40, "сорок"); _add(50, "пятьдесят"); _add(60, "шестьдесят")
_add(70, "семьдесят"); _add(80, "восемьдесят"); _add(90, "девяносто"); _add(100, "сто"); _add(200, "двести", "двухсот")
_add(300, "триста", "трёхсот", "трехсот"); _add(400, "четыреста", "четырёхсот", "четырехсот"); _add(500, "пятьсот", "пятисот")
_add(600, "шестьсот", "шестисот"); _add(700, "семьсот", "семисот"); _add(800, "восемьсот", "восьмисот"); _add(900, "девятьсот", "девятисот")
_add("K", "тысяча", "тысячи", "тысяч", "тысячу", "тыс"); _add("M", "миллион", "миллиона", "миллионов", "млн")
_add("ONEHALF", "полтора", "полторы"); _add("HALF", "пол")

# Articles that mean "one" only when a multiplier follows ("a million", "un millón", "en miljon", "eine halbe Million").
_ARTICLES = {"a", "an", "un", "una", "une", "ein", "eine", "einen", "en", "ett"}
# Words that may sit inside a number phrase without ending it.
_GLUE = {"and", "y", "i", "und", "och", "et", "и", "с", "з", "e"}
# Phrases for "+ half" after a multiplier or a number.
_PLUSHALF = [r"\by medio\b", r"\bet demi\b", r"\bet demie\b", r"\boch en halv\b", r"\bс половиной\b", r"\bi pół\b",
             r"\band a half\b"]
# Compound parts for German and Swedish, longest first.
_PARTS = sorted([w for w in _W if re.fullmatch(r"[a-zäöüßå]+", w) and len(w) > 2] + ["und", "och"], key=len, reverse=True)


def _split_compound(tok: str) -> list[str] | None:
    """Split 'neunhunderttausend' into ['neun', 'hundert', 'tausend']; None if it isn't made of known parts."""
    out, i = [], 0
    while i < len(tok):
        for part in _PARTS:
            if tok.startswith(part, i):
                out.append(part)
                i += len(part)
                break
        else:
            return None
    return out if len(out) > 1 else None


def _tokens(text: str) -> list[str]:
    t = text.lower().replace("quatre-vingt", "quatrevingt").replace("twenty-five", "twentyfive")
    t = re.sub(r"\bпол(?=миллион)", "пол ", t)                 # полмиллиона → пол миллиона
    for pat in _PLUSHALF:
        t = re.sub(pat, " PLUSHALF ", t)
    t = re.sub(r"(?<=\d)[.\s](?=\d{3}\b)", "", t)             # 1.200 / 1 200 → 1200
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)                     # 1,5 → 1.5
    raw = re.findall(r"PLUSHALF|\d+(?:\.\d+)?|[^\W\d_]+", t)
    out = []
    for tok in raw:
        if tok == "PLUSHALF" or re.fullmatch(r"\d+(?:\.\d+)?", tok) or tok in _W or tok in _GLUE:
            out.append(tok)
        else:
            parts = _split_compound(tok)
            out.extend(parts if parts else [tok])
    return out


def _value(tok: str):
    if tok == "PLUSHALF":
        return "PLUSHALF"
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return float(tok)
    return _W.get(tok)


def amounts(text: str) -> list[float]:
    """Every money amount in the text that is written (at least partly) in words and contains thousand or million."""
    toks = _tokens(text)
    found, i = [], 0
    while i < len(toks):
        if _value(toks[i]) is None or toks[i] in ("a", "an", "en", "e", "i", "y"):
            # an article or a glue word can't start a phrase on its own unless a multiplier follows soon
            if not (toks[i] in ("a", "an", "en", "un", "une", "ein", "eine") and i + 1 < len(toks)
                    and _value(toks[i + 1]) in ("M", "K", "HALF")):
                i += 1
                continue
        total = current = 0.0
        factor, last_mult, has_mult, has_word, decimals = 1.0, None, False, False, None
        j = i
        while j < len(toks):
            tok, v = toks[j], _value(toks[j])
            if tok in _ARTICLES and not (j + 1 < len(toks) and _value(toks[j + 1]) in ("M", "K", "HALF", "H")):
                break                                         # "a month", "un piso": an article is "one" only before a multiplier
            if v is None:
                if tok in _GLUE:
                    j += 1
                    continue
                break
            if not isinstance(v, float) or tok in _W:
                has_word = has_word or tok in _W or tok == "PLUSHALF"
            if v == "POINT":
                decimals = ""
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                if decimals is not None and isinstance(_W.get(tok), int) and _W[tok] < 10:
                    decimals += str(_W[tok])
                else:
                    current += float(v)
            elif v == "ONEHALF":
                current += 1.5
            elif v == "HALF":
                factor = 0.5
            elif v == "PLUSHALF":
                if current:
                    current += 0.5
                elif last_mult:
                    total += 0.5 * last_mult
            elif v == "H":
                current = (current or 1) * 100
            elif v in ("K", "M"):
                mult = 1e3 if v == "K" else 1e6
                base = (current or 1) + (float("0." + decimals) if decimals else 0)
                total += base * mult * factor
                current, factor, decimals, last_mult, has_mult = 0.0, 1.0, None, mult, True
            j += 1
        if has_mult and has_word:
            found.append(total + current)
        i = max(j, i + 1)
    return [a for a in found if 0 < a < 1e9]
