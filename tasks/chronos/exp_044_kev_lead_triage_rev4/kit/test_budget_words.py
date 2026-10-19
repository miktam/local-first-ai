"""Budgets written as words (exp_044 rev 4): one or two forms per language, plus the cases that must not be budgets."""
import pytest

from scripts.budget_words import amounts

CASES = [
    ("Our budget is half a million euros", 500000), ("around one point two million", 1200000),
    ("about nine hundred thousand euros", 900000), ("two and a half thousand a month", 2500),
    ("tenemos un millón y medio", 1500000), ("hasta medio millón", 500000), ("novecientos mil euros", 900000),
    ("dos mil quinientos euros al mes", 2500), ("pół miliona euro", 500000), ("milion dwieście tysięcy", 1200000),
    ("około dziewięćset tysięcy euro", 900000), ("półtora miliona", 1500000), ("eine halbe Million", 500000),
    ("zwei Millionen Euro", 2000000), ("rund neunhunderttausend Euro", 900000), ("zweieinhalbtausend im Monat", 2500),
    ("en halv miljon euro", 500000), ("två miljoner euro", 2000000), ("runt niohundratusen euro", 900000),
    ("tvåtusenfemhundra euro i månaden", 2500), ("un demi-million", 500000), ("deux millions d'euros", 2000000),
    ("environ huit cent mille euros", 800000), ("deux mille cinq cents euros par mois", 2500),
    ("полмиллиона евро", 500000), ("полтора миллиона", 1500000), ("около девятисот тысяч евро", 900000),
    ("две с половиной тысячи в месяц", 2500), ("два миллиона евро", 2000000),
]


@pytest.mark.parametrize("text,value", CASES)
def test_budget_words(text, value):
    assert value in amounts(text), (text, amounts(text))


@pytest.mark.parametrize("text", ["three bedrooms near the beach", "una casa con dos baños", "we are two adults and a dog",
                                  "dwa pokoje, blisko plaży", "zwei Schlafzimmer", "trois chambres"])
def test_no_budget_without_thousand_or_million(text):
    assert amounts(text) == []
