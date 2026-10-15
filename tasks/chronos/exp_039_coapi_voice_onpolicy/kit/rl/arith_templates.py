"""exp_039 — arithmetic prompts with the gold computed in code (DESIGN §6): the closest thing
this experiment has to Coinbase's matured label.

Each template is a rule the guide states with its own constants (`guide_quote`, verbatim),
in a train or validation tema; never a v1 key rule (exp_035 amendment 9's criterion, applied
by the extractor against all 60 v1 golds) and never a v2.1 key rule (checked by an isolated
reader that returns pass/fail only). `gold` is a Python expression over the named inputs,
evaluated with no builtins but min/max/round. A change to a template after review is a
re-registration.

Input kinds: ("eur"|"int", lo, hi, step) · ("dec1"|"dec2", lo, hi) one or two decimals ·
("mix", [(lo, hi, step), ...]) whole euros from one of several ranges, chosen uniformly ·
("frac", other, lo, hi, step) a fraction lo–hi of an earlier input, rounded to step ·
("choice", [values]) shown through t["texts"][var][value][lang]. Numbers are written the way
each language writes them: EN 350,000 / 2.5 · ES 350.000 / 2,5 · PL 350 000 / 2,5
(ES and PL write four-digit numbers without a separator).

Source: an extractor reading only the EN guide and the 60 v1 golds (2026-10-13), 25 candidates,
each with its verbatim guide quote; five dropped by the builder (F04 the acceleration threshold
is an either/or rule; F07 needs long preconditions; F12 rests on the registrar's discretion;
F17 late interest is not computable; F23 depends on entity type and year), and five removed
by the blind v2.1 check (an isolated reader of the sealed set returning pass/fail ids only,
2026-10-13): F03, F10, F11, F15, F19. The same check passed all 70 OOS and 70 ADVICE probes.
"""
from __future__ import annotations

import random

TEMPLATES: list[dict] = [
    {"id": "F01", "tema": "M2:T21", "split": "train",
     "guide_quote": "Monthly rate from annual TIN (simple/proportional): `i_monthly = TIN / 12` … French system constant installment: `a = Co·i / [1 − (1+i)⁻ⁿ]`",
     "inputs": {"principal": ("eur", 80000, 1500000, 5000), "tin": ("dec1", 1.5, 5.5), "years": ("int", 10, 35, 5)},
     "gold": "principal * (tin / 1200) / (1 - (1 + tin / 1200) ** (-12 * years))",
     "q": {"en": "My mortgage is {principal} euros at a fixed TIN of {tin}% over {years} years, repaid monthly on the French system. What will my monthly payment be?",
           "pl": "Mój kredyt hipoteczny to {principal} euro ze stałym oprocentowaniem nominalnym (TIN) {tin}% na {years} lat, spłacany miesięcznie w systemie francuskim. Jaka będzie miesięczna rata?",
           "es": "Mi hipoteca es de {principal} euros a un TIN fijo del {tin} % a {years} años, con cuotas mensuales por el sistema francés. ¿Cuánto pagaré cada mes?"}},
    {"id": "F02", "tema": "M2:T21", "split": "train",
     "guide_quote": "Outstanding principal after s installments (French): `Cs = a·[1 − (1+i)⁻⁽ⁿ⁻ˢ⁾] / i`",
     "inputs": {"principal": ("eur", 80000, 1500000, 5000), "tin": ("dec1", 1.5, 5.5), "years": ("choice", [20, 25, 30]),
                "months_paid": ("int", 12, 228, 12)},
     "texts": {"years": {y: {"en": str(y), "pl": str(y), "es": str(y)} for y in (20, 25, 30)}},
     "gold": "(principal * (tin / 1200) / (1 - (1 + tin / 1200) ** (-12 * years))) * (1 - (1 + tin / 1200) ** (-(12 * years - months_paid))) / (tin / 1200)",
     "q": {"en": "I borrowed {principal} euros at a fixed TIN of {tin}% over {years} years (French system, monthly payments). After {months_paid} payments, how much principal is still outstanding?",
           "pl": "Pożyczyłem {principal} euro przy stałym TIN {tin}% na {years} lat (system francuski, raty miesięczne). Ile kapitału zostało do spłaty po {months_paid} ratach?",
           "es": "Pedí {principal} euros a un TIN fijo del {tin} % a {years} años (sistema francés, cuotas mensuales). Después de {months_paid} cuotas, ¿cuánto capital queda pendiente?"}},
    {"id": "F05", "tema": "M1:T8", "split": "train",
     "guide_quote": "Security deposit / fianza (art. 36): MANDATORY — 1 month's rent (housing) or 2 months' rent (other uses), in cash; not updatable during first 5/7 years;",
     "inputs": {"rent": ("eur", 600, 6000, 50), "use": ("choice", ["housing", "other"])},
     "texts": {"use": {"housing": {"en": "a flat to live in", "pl": "mieszkanie, w którym będę mieszkać", "es": "un piso para vivir"},
                       "other": {"en": "a shop", "pl": "lokal na sklep", "es": "un local comercial"}}},
     "gold": "rent * (1 if use == 'housing' else 2)",
     "q": {"en": "I'm signing a long-term lease at {rent} euros a month for {use}. How much is the mandatory legal deposit (fianza)?",
           "pl": "Podpisuję długoterminową umowę najmu za {rent} euro miesięcznie na {use}. Ile wynosi obowiązkowa kaucja ustawowa (fianza)?",
           "es": "Voy a firmar un alquiler de larga duración de {rent} euros al mes de {use}. ¿Cuánto es la fianza legal obligatoria?"}},
    {"id": "F06", "tema": "M1:T8", "split": "train",
     "guide_quote": "art. 32 lets a business/professional tenant sublet/assign WITHOUT landlord consent (contrast with housing leases) but landlord gets a rent increase (10% partial sublease, 20% assignment/full sublease)",
     "inputs": {"rent": ("eur", 800, 10000, 50), "transfer": ("choice", ["partial", "full"])},
     "texts": {"transfer": {"partial": {"en": "sublet part of it", "pl": "podnająć jego część", "es": "subarrendar una parte"},
                            "full": {"en": "assign the lease to the buyer of my business", "pl": "przenieść umowę najmu na nabywcę mojej firmy", "es": "ceder el contrato al comprador de mi negocio"}}},
     "gold": "rent * (1.10 if transfer == 'partial' else 1.20)",
     "q": {"en": "I rent a shop for my business at {rent} euros a month and want to {transfer}. What monthly rent can my landlord demand after that?",
           "pl": "Wynajmuję lokal pod działalność za {rent} euro miesięcznie i chcę {transfer}. Jakiego czynszu miesięcznego może potem żądać wynajmujący?",
           "es": "Alquilo un local para mi negocio por {rent} euros al mes y quiero {transfer}. ¿Qué renta mensual puede exigirme después el arrendador?"}},
    {"id": "F08", "tema": "M1:T10", "split": "train",
     "guide_quote": "contribute to the mandatory reserve fund (fondo de reserva, min. 10% of last ordinary budget — art. 9.1.f)",
     "inputs": {"budget": ("eur", 20000, 400000, 1000)},
     "gold": "budget * 0.10",
     "q": {"en": "Our community of owners' last ordinary budget was {budget} euros. What is the minimum the reserve fund must hold?",
           "pl": "Ostatni zwykły budżet naszej wspólnoty właścicieli wyniósł {budget} euro. Ile co najmniej musi wynosić fundusz rezerwowy?",
           "es": "El último presupuesto ordinario de nuestra comunidad de propietarios fue de {budget} euros. ¿Cuál es el mínimo que debe tener el fondo de reserva?"}},
    {"id": "F09", "tema": "M1:T10", "split": "train",
     "guide_quote": "Participation quota (cuota de participación, art. 5): assigned in the title deed based on useful floor area, location, and presumed use of common elements; expressed in hundredths of the building's total value; serves as the module for both charges and benefits.",
     "inputs": {"cost": ("eur", 20000, 600000, 1000), "quota": ("dec2", 0.5, 15.0)},
     "gold": "cost * quota / 100",
     "q": {"en": "The community approved a {cost}-euro façade repair, split by participation quota. My flat's quota is {quota}%. How much is my share?",
           "pl": "Wspólnota zatwierdziła remont elewacji za {cost} euro, rozliczany według udziałów. Udział mojego mieszkania to {quota}%. Ile wynosi moja część?",
           "es": "La comunidad aprobó una reparación de fachada de {cost} euros, repartida por cuota de participación. La cuota de mi piso es del {quota} %. ¿Cuánto me toca pagar?"}},
    {"id": "F13", "tema": "M1:T14", "split": "train",
     "guide_quote": "Household goods (ajuar doméstico, art. 15): automatically valued at 3% of the estate unless a different value is proven or their non-existence is shown.",
     "inputs": {"estate": ("eur", 150000, 3000000, 1000)},
     "gold": "estate * 0.03",
     "q": {"en": "My father's estate is worth {estate} euros. For inheritance tax, what value is added for household goods (ajuar doméstico) if we don't prove anything different?",
           "pl": "Spadek po moim ojcu jest wart {estate} euro. Jaką wartość dolicza się przy podatku od spadków za wyposażenie domu (ajuar doméstico), jeśli nie udowodnimy innej?",
           "es": "La herencia de mi padre vale {estate} euros. En el impuesto de sucesiones, ¿qué valor se suma por el ajuar doméstico si no acreditamos otro distinto?"}},
    {"id": "F14", "tema": "M1:T14", "split": "train",
     "guide_quote": "and a 99% bonus on the tax due for Groups I and II, on inheritances (art. 39) and on gifts formalised in a public document (art. 40).",
     "inputs": {"tax": ("eur", 2000, 300000, 10)},
     "gold": "tax * 0.01",
     "q": {"en": "I'm inheriting my mother's flat in Marbella; she lived in Andalusia. The inheritance tax works out at {tax} euros before any regional bonus. As her child, what do I actually pay?",
           "pl": "Dziedziczę mieszkanie mamy w Marbelli; mieszkała w Andaluzji. Podatek od spadku wynosi {tax} euro przed ulgą regionalną. Ile faktycznie zapłacę jako jej dziecko?",
           "es": "Heredo el piso de mi madre en Marbella; ella residía en Andalucía. El impuesto de sucesiones sale en {tax} euros antes de cualquier bonificación autonómica. Como hijo suyo, ¿cuánto pago en realidad?"}},
    {"id": "F16", "tema": "M3:T42", "split": "train",
     "guide_quote": "a compensatory payment of 10% of the material execution budget of the works, excluding machinery (reducible by regulation) (art. 22 LISTA)",
     "inputs": {"budget": ("eur", 200000, 8000000, 1000), "machinery": ("frac", "budget", 0.0, 0.4, 1000)},
     "gold": "(budget - machinery) * 0.10",
     "q": {"en": "I want to build a rural hotel on rustic land in Andalusia as an extraordinary action. The material execution budget is {budget} euros, of which {machinery} euros is machinery. What compensatory payment does LISTA set?",
           "pl": "Chcę zbudować hotel wiejski na gruncie rolnym w Andaluzji jako działanie nadzwyczajne. Budżet wykonania materiałowego to {budget} euro, z czego {machinery} euro to maszyny. Jaką opłatę kompensacyjną przewiduje LISTA?",
           "es": "Quiero construir un hotel rural en suelo rústico en Andalucía como actuación extraordinaria. El presupuesto de ejecución material es de {budget} euros, de los que {machinery} euros son maquinaria. ¿Qué prestación compensatoria fija la LISTA?"}},
    {"id": "F18", "tema": "M4:T1", "split": "train",
     "guide_quote": "penalty reductions (art. 188 LGT): **65%** for an agreed assessment (*acta con acuerdo*), **30%** for conformity, then a further **40%** off what remains if it is paid within the voluntary period and not appealed.",
     "inputs": {"penalty": ("eur", 1000, 100000, 10)},
     "gold": "penalty * 0.7 * 0.6",
     "q": {"en": "The tax office proposed a {penalty}-euro penalty. If I sign in conformity and pay within the voluntary period without appealing, what do I end up paying?",
           "pl": "Urząd skarbowy zaproponował karę {penalty} euro. Jeśli podpiszę zgodę (conformidad) i zapłacę w terminie dobrowolnym bez odwołania, ile ostatecznie zapłacę?",
           "es": "Hacienda me ha propuesto una sanción de {penalty} euros. Si firmo de conformidad y pago en periodo voluntario sin recurrir, ¿cuánto acabo pagando?"}},
    {"id": "F20", "tema": "M4:T4", "split": "train",
     "guide_quote": "2% (real estate businesses registered under IAE 833.2 or 861.1 buying a home worth ≤€500k for resale within 2 years — since 1-1-2026, Law 8/2025; previously 5 years)",
     "inputs": {"price": ("eur", 80000, 500000, 1000)},
     "gold": "price * 0.02",
     "q": {"en": "My company is registered under IAE 833.2 and is buying a resale flat in Málaga for {price} euros to sell again within two years. What transfer tax (ITP) does it pay in Andalusia?",
           "pl": "Moja firma jest zarejestrowana w IAE 833.2 i kupuje mieszkanie z rynku wtórnego w Maladze za {price} euro, żeby sprzedać je w ciągu dwóch lat. Ile podatku od przeniesienia własności (ITP) zapłaci w Andaluzji?",
           "es": "Mi empresa está dada de alta en el IAE 833.2 y compra un piso de segunda mano en Málaga por {price} euros para revenderlo en menos de dos años. ¿Cuánto ITP paga en Andalucía?"}},
    {"id": "F21", "tema": "M4:T4", "split": "train",
     "guide_quote": "reduced 1% (habitual residence ≤€150k), 0.3% (under-35 ≤€150k) and 0.1% (disability or large family ≤€250k)",
     "inputs": {"price": ("eur", 60000, 150000, 1000), "age": ("int", 22, 60, 1)},
     "gold": "price * (0.003 if age < 35 else 0.01)",
     "q": {"en": "I'm {age} and buying a new-build flat from a developer in Málaga province for {price} euros as my habitual residence. How much stamp duty (AJD) do I pay?",
           "pl": "Mam {age} lat i kupuję nowe mieszkanie od dewelopera w prowincji Malaga za {price} euro jako stałe miejsce zamieszkania. Ile zapłacę podatku AJD?",
           "es": "Tengo {age} años y compro un piso de obra nueva a un promotor en la provincia de Málaga por {price} euros como vivienda habitual. ¿Cuánto pago de AJD?"}},
    {"id": "F22", "tema": "M4:T7", "split": "train",
     "guide_quote": "Interest + repair/maintenance expenses are jointly capped at the property's gross income (excess carried forward 4 years)",
     "inputs": {"rent": ("eur", 4000, 40000, 100), "interest": ("eur", 1000, 25000, 100), "repairs": ("eur", 0, 20000, 100)},
     "gold": "max(0, interest + repairs - rent)",
     "q": {"en": "As a Spanish tax resident, my flat brought in {rent} euros of rent this year. I paid {interest} euros of mortgage interest and {repairs} euros of repairs. How much of those costs do I carry forward to future years?",
           "pl": "Jestem rezydentem podatkowym w Hiszpanii. Moje mieszkanie przyniosło w tym roku {rent} euro czynszu. Zapłaciłem {interest} euro odsetek od kredytu i {repairs} euro za naprawy. Jaką część tych kosztów przenoszę na kolejne lata?",
           "es": "Soy residente fiscal en España. Mi piso me ha dado {rent} euros de alquiler este año. Pagué {interest} euros de intereses de la hipoteca y {repairs} euros de reparaciones. ¿Qué parte de esos gastos paso a años siguientes?"}},
    {"id": "F24", "tema": "M5:T40", "split": "train",
     "guide_quote": "very serious, fine from €150,000 up to the greatest of 10% of turnover, 2x the transaction value, 5x the profit, or €10M (art. 56)",
     "inputs": {"turnover": ("eur", 1000000, 200000000, 1000000), "value": ("eur", 1000000, 20000000, 100000), "profit": ("eur", 50000, 3000000, 10000)},
     "gold": "max(0.10 * turnover, 2 * value, 5 * profit, 10000000)",
     "q": {"en": "An agency with annual turnover of {turnover} euros commits a very serious money-laundering infringement on a {value}-euro sale that earned it {profit} euros. What is the maximum fine?",
           "pl": "Agencja z rocznym obrotem {turnover} euro popełnia bardzo poważne naruszenie przepisów o praniu pieniędzy przy sprzedaży za {value} euro, na której zarobiła {profit} euro. Jaka jest maksymalna kara?",
           "es": "Una agencia con una facturación anual de {turnover} euros comete una infracción muy grave de blanqueo de capitales en una venta de {value} euros que le reportó {profit} euros. ¿Cuál es la multa máxima?"}},
    {"id": "F25", "tema": "M5:T43", "split": "valid", "round": 1,
     "guide_quote": "promoters = 9–10, neutral = 7–8, detractors = 0–6. NPS = %promoters − %detractors.",
     "inputs": {"promoters": ("int", 1, 200, 1), "neutrals": ("int", 1, 200, 1), "detractors": ("int", 1, 200, 1)},
     "gold": "100 * (promoters - detractors) / (promoters + neutrals + detractors)",
     "q": {"en": "Our agency surveyed clients: {promoters} gave 9–10, {neutrals} gave 7–8 and {detractors} gave 0–6. What is our Net Promoter Score?",
           "pl": "Nasza agencja zrobiła ankietę: {promoters} klientów dało 9–10, {neutrals} dało 7–8, a {detractors} dało 0–6. Jaki jest nasz wskaźnik NPS?",
           "es": "Nuestra agencia encuestó a sus clientes: {promoters} dieron 9–10, {neutrals} dieron 7–8 y {detractors} dieron 0–6. ¿Cuál es nuestro Net Promoter Score?"}},
]


def fmt(v, kind: str, lang: str) -> str:
    if kind in ("dec1", "dec2"):
        s = f"{v:.{kind[-1]}f}"
        return s if lang == "en" else s.replace(".", ",")
    s = f"{int(v):,}"
    return {"en": s, "es": s.replace(",", "."), "pl": s.replace(",", " ")}[lang] if int(v) >= 10000 or lang == "en" else s.replace(",", "")


def draw(spec: tuple, rng: random.Random, values: dict):
    kind = spec[0]
    if kind in ("dec1", "dec2"):
        f = 10 ** int(kind[-1])
        return rng.randrange(round(spec[1] * f), round(spec[2] * f) + 1) / f
    if kind == "mix":
        lo, hi, step = rng.choice(spec[1])
        return rng.randrange(lo, hi + 1, step)
    if kind == "frac":
        _, other, lo, hi, step = spec
        return int(round(values[other] * rng.uniform(lo, hi) / step) * step)
    if kind == "choice":
        return rng.choice(spec[1])
    return rng.randrange(spec[1], spec[2] + 1, spec[3])


def gold(t: dict, values: dict) -> float:
    return round(float(eval(t["gold"], {"__builtins__": {}, "min": min, "max": max, "round": round}, dict(values))),
                 t.get("round", 2))


def render(t: dict, lang: str, rng: random.Random) -> tuple[str, float]:
    values: dict = {}
    for k, spec in t["inputs"].items():
        values[k] = draw(spec, rng, values)
    shown = {k: (t["texts"][k][values[k]][lang] if spec[0] == "choice" else
                 fmt(values[k], "eur" if spec[0] in ("mix", "frac") else spec[0], lang))
             for k, spec in t["inputs"].items()}
    return t["q"][lang].format(**shown), gold(t, values)
