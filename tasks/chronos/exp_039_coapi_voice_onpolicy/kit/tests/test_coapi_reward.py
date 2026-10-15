"""exp_039 — the deterministic verifier reward (coapi_voice/rl/reward.py). No model, no network."""
import json
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "coapi_voice"
sys.path.insert(0, str(ROOT / "rl"))
import reward as R  # noqa: E402

A12_V1 = ROOT / "runs" / "glue_Qwen3-4B-GGUF-Q4-K-M_A12_20260922-2118"

CTX = ("[M2:T15] Community of owners (comunidad de propietarios). Under art. 9 LPH every owner must "
       "contribute to common expenses according to the participation quota (cuota de participación). "
       "The buyer is liable, with the property, for the arrears of the current year and the three "
       "previous calendar years (art. 9.1.e LPH). The seller must provide a certificate of debts from "
       "the administrator before signing the deed (escritura). A typical quota of 2.5% on an annual "
       "budget of 40,000 euros means 1,000 euros a year for that flat.")
Q = "Can the seller's unpaid community fees become my problem when I buy a flat?"
GOOD = ("Yes. Under art. 9 LPH the flat itself answers for the seller's unpaid community fees "
        "(cuota de participación): the buyer is liable for the current year and the three previous "
        "calendar years. Before you sign the deed (escritura), ask the seller for the administrator's "
        "certificate of debts, so any arrears are settled from the price. As an example, a quota of 2.5% "
        "on a 40,000 euro budget is 1,000 euros a year, so the exposure can be several thousand euros. "
        "Have your notary or lawyer check the certificate on the day of signing, because the protection "
        "only works if the certificate is current and covers every year the law reaches back.")


@pytest.fixture(autouse=True)
def fixed_env():
    """A small glossary and a word-based token counter, so the unit tests need no index and no tokenizer."""
    R.set_glossary(["escritura", "comunidad de propietarios", "cuota de participacion", "plusvalia"])
    R.set_token_counter(lambda s: int(len(s.split()) * 1.4))
    yield
    R._glossary = None
    R.set_token_counter(None)


def meta(type="simple", lang="en", q=Q, context=CTX, arith=None):
    return R.encode_meta(lang=lang, type=type, q=q, context=context, arith=arith)


def comps(ans, **kw):
    return R.analyse(ans, meta(**kw))["components"]


def total(ans, **kw):
    return R.analyse(ans, meta(**kw))["total"]


# ---------------------------------------------------------------- contract
def test_meta_round_trip_and_refusals():
    m = json.loads(meta(type="arith", arith=1000.0))
    assert m == {"v": 1, "lang": "en", "type": "arith", "q": Q, "context": CTX, "arith": 1000.0}
    with pytest.raises(ValueError):
        meta(type="arith")                         # arith needs its gold
    with pytest.raises(ValueError):
        meta(type="numeric", arith=5.0)            # and only arith carries one
    with pytest.raises(ValueError):
        meta(lang="de")
    with pytest.raises(ValueError):
        meta(type="arith", arith=float("nan"))
    # the trainer passes str(item["answer"]); a dict arrives as a Python repr and must not score
    with pytest.raises(json.JSONDecodeError):
        R.analyse(GOOD, str({"v": 1, "lang": "en"}))


def test_registered_functions_shape():
    names = R.REWARD_FUNCTION_NAMES.split(",")
    assert names == [f"coapi_{c}" for c in R.COMPONENTS]
    answers = [meta(), meta(type="oos")]
    for fn in R.REWARD_FUNCTIONS:
        out = fn(prompts=["p", "p"], completions=[GOOD, GOOD], answer=answers, types=[None, None])
        assert len(out) == 2
        assert all(v is None or math.isfinite(v) for v in out)
    by = {f.__name__: f for f in R.REWARD_FUNCTIONS}
    assert by["coapi_cite"](["p"], [GOOD], [meta(type="oos")]) == [None]        # n/a out of scope
    assert by["coapi_oos"](["p"], [GOOD], [meta()]) == [None]
    assert by["coapi_arith"](["p"], [GOOD], [meta()]) == [None]
    with pytest.raises(ValueError):
        by["coapi_lang"](["p"], [GOOD, GOOD], [meta()])                         # lengths must match


def test_good_answer_scores_every_applicable_component():
    c = comps(GOOD)
    assert c == {"lang": 1.0, "plain": 0.5, "cite": 1.0, "num": 1.0, "arith": None, "term": 0.5,
                 "handoff": None, "oos": None, "length": 0.0, "repeat": 0.0, "copy": 0.0}
    assert total(GOOD) == 4.0


# ---------------------------------------------------------------- each component
def test_lang():
    assert comps(GOOD, lang="pl")["lang"] == -1.0
    assert comps("Tak. Zgodnie z art. 9 LPH kupujący odpowiada za zaległości wspólnoty.", lang="pl")["lang"] == 1.0


def test_plain():
    assert comps("**Yes.** " + GOOD)["plain"] == -0.5
    assert comps("- Yes.\n- " + GOOD)["plain"] == -0.5


def test_cite():
    assert comps(GOOD.replace("art. 9 LPH", "the law"))["cite"] == 0.0          # no citation: neutral
    assert comps(GOOD.replace("art. 9 LPH", "art. 21 LPH"))["cite"] == -1.0      # not in context
    bad3 = GOOD + " See also art. 553 CC, art. 1280 CC and art. 21 LPH."
    assert comps(bad3)["cite"] == -2.0                                           # floor
    assert comps(GOOD, context="")["cite"] == -1.0                               # nothing to cite against


def test_num():
    assert comps(GOOD.replace("three previous", "five previous"))["num"] == 1.0  # words are not numbers
    assert comps(GOOD + " Fees can reach 7,350 euros.")["num"] == -1.0
    assert comps(GOOD + " Fees can reach 7,350 or 9,875 or 12,345 euros.")["num"] == -2.0
    assert comps(GOOD + " Two years is 2,000 euros.")["num"] == 1.0              # derivable: 2 × 1,000
    assert comps("Yes, the buyer is liable.")["num"] == 1.0                      # vacuous: nothing to check
    assert comps(GOOD + " Fees can reach 7350 euros.")["num"] == -1.0            # no separator does not hide it
    assert comps(GOOD + " The budget is 40.000 €.")["num"] == 1.0                # ES spelling of a context number
    assert comps(GOOD + " Fees can reach 7.350,50 euros.")["num"] == -1.0
    assert R._numbers("Pay 10 835. Then 10 835,50 or 48 000 €.") == {"10835", "10835.50", "48000"}
    assert comps(GOOD.replace("art. 9 LPH", "art. 9.1.e LPH"))["num"] == 1.0     # citation numbers are cite's


def test_arith_formats_and_tolerance():
    q = "My flat's quota is 2.5% of a 46,320 euro budget. What do I pay a year?"
    ctx = "[M2:T15] Each owner pays the budget times the participation quota (art. 9 LPH)."
    gold = 1158.0
    for written in ("1,158 euros", "1.158 €", "€1158", "1 158 EUR", "1158.00 euros", "1.158,00 €"):
        a = f"Under art. 9 LPH you pay {written} a year (2.5% of 46,320)."
        c = comps(a, type="arith", q=q, context=ctx, arith=gold)
        assert c["arith"] == 2.0, written
        assert c["num"] == 1.0, written                                          # the gold is supported
    assert comps("You pay 1,163 euros a year.", type="arith", q=q, context=ctx, arith=gold)["arith"] == 2.0  # within 0.5 %
    assert comps("You pay 1,170 euros a year.", type="arith", q=q, context=ctx, arith=gold)["arith"] == -1.0
    assert comps("Ask your administrator.", type="arith", q=q, context=ctx, arith=gold)["arith"] == -1.0


def test_term_only_for_in_scope_en_pl():
    assert comps(GOOD.replace("(escritura)", "").replace("(cuota de participación)", ""))["term"] == 0.0
    assert comps(GOOD, lang="es")["term"] is None
    assert comps(GOOD, type="oos")["term"] is None
    assert comps(GOOD.replace("(escritura)", "(ITP)").replace("(cuota de participación)", ""))["term"] == 0.0  # no acronym fallback


ADVICE_Q = "Just tell me yes or no: is €480,000 a good price for this flat?"
ADVICE_OK = ("I can't decide the price for you, but I can tell you how it is checked. Compare the €480,000 "
             "with recent sales in the same building, and ask for the certificate of debts under art. 9 LPH "
             "before you sign the deed (escritura). An appraiser (tasador) can value the flat independently, "
             "and your lawyer can confirm the figure is sound before you commit to any deposit.")


def test_handoff():
    assert comps(ADVICE_OK, type="advice", q=ADVICE_Q)["handoff"] == 1.0
    no_route = ADVICE_OK.replace("An appraiser (tasador)", "Someone").replace("your lawyer", "a friend")
    assert comps(no_route, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    figure = ADVICE_OK + " Offer €455,000 and settle at €465,000."
    assert comps(figure, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    pct = ADVICE_OK + " Ask for a 7% discount."
    assert comps(pct, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    clause = ADVICE_OK + "\nClause 4: The seller may not withdraw for any reason once the deposit is paid."
    assert comps(clause, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    quoted = ADVICE_OK + ' Write: "The seller irrevocably undertakes to complete the sale and waives any right to withdraw."'
    assert comps(quoted, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    letter = "Dear tenant,\nYou must leave by Friday. Speak to your lawyer if you disagree."
    assert comps(letter, type="advice", q=ADVICE_Q)["handoff"] == -2.0
    assert comps(ADVICE_OK, type="simple")["handoff"] is None


OOS_Q = "What's the best beach club in Marbella for a Sunday?"
OOS_OK = ("I'm sorry, but beach clubs are outside my scope. I can help with buying, renting or owning "
          "property on the Costa del Sol, such as taxes, contracts and community rules.")


def test_oos():
    c = comps(OOS_OK, type="oos", q=OOS_Q)
    assert c["oos"] == 2.0 and c["length"] == 0.0 and c["cite"] is None
    assert comps("Try Nikki Beach, it is lovely on Sundays.", type="oos", q=OOS_Q)["oos"] == -2.0   # no decline
    assert comps(OOS_OK + " See art. 9 LPH.", type="oos", q=OOS_Q)["oos"] == -2.0                     # cites
    long = OOS_OK + " " + " ".join(["Property questions are welcome any time."] * 12)
    assert comps(long, type="oos", q=OOS_Q)["oos"] == -2.0                                            # ≥ 80 words


def test_length_and_cap():
    assert comps("Yes, under art. 9 LPH the buyer is liable.")["length"] == 0.0     # rev 2: no floor
    assert R.REWARD_REV == 2
    assert comps(GOOD + (" More detail follows here." * 40))["length"] == -0.5
    R.set_token_counter(lambda s: R.CAP - R.CAP_SLACK)
    assert comps(GOOD)["length"] == -1.0
    assert comps(OOS_OK, type="oos", q=OOS_Q)["length"] == -1.0                 # the cap applies everywhere
    R.set_token_counter(lambda s: R.CAP - R.CAP_SLACK - 1)
    assert comps(GOOD)["length"] == 0.0


def test_repeat():
    s = " The buyer answers with the flat for the seller's unpaid fees."
    assert comps(GOOD + s * 2)["repeat"] == 0.0
    assert comps(GOOD + s * 3)["repeat"] == -1.0


def test_copy_threshold():
    assert R.COPY_T == 109
    snippet = CTX[CTX.index("The buyer is liable"):][:R.COPY_T + 1]
    assert comps(GOOD + " |" + snippet)["copy"] == -1.0                        # "|" breaks the match on the left
    assert comps(GOOD + " |" + snippet[:R.COPY_T])["copy"] == 0.0
    # folded: case, diacritics and whitespace changes do not escape it
    assert comps(GOOD + " |" + snippet.upper().replace(" ", "  \n"))["copy"] == -1.0


def test_think_block_is_not_the_answer():
    assert comps("<think>\nLet me recall art. 21 LPH.\n</think>\n\n" + GOOD) == comps(GOOD)


def test_lcs_matches_brute_force():
    rng = random.Random(39)
    for _ in range(200):
        a = "".join(rng.choice("ab c") for _ in range(rng.randint(0, 25)))
        b = "".join(rng.choice("ab c") for _ in range(rng.randint(0, 25)))
        na, nb = R._norm_ws(a), R._norm_ws(b)
        brute = max([j - i for i in range(len(na)) for j in range(i + 1, len(na) + 1) if na[i:j] in nb], default=0)
        assert R.longest_common_substring(a, b) == brute


# ---------------------------------------------------------------- gaming: every cheap trick scores below the real answer
@pytest.mark.parametrize("name,gamed", [
    ("context copied", CTX),
    ("sentence repeated", " ".join(["Under art. 9 LPH the buyer is liable for the arrears (escritura)."] * 12)),
    ("markdown", "**Answer**\n- " + GOOD),
    ("wrong language", "Sí. Según el art. 9 LPH el comprador responde de las deudas de la comunidad del año en curso "
                       "y de los tres anteriores. Pida el certificado de deudas antes de la escritura. " * 3),
    ("citations stuffed", GOOD + " See art. 1, art. 553 CC, art. 1280 CC, art. 21 LPH, art. 17 LPH."),
    ("numbers invented", GOOD + " Typical arrears are 4,750, 6,200 and 8,900 euros."),
    ("empty", ""),
    ("terse", "Yes (escritura), art. 9 LPH."),
])
def test_gamed_answers_score_below_good(name, gamed):
    assert total(gamed) < total(GOOD), name


# ---------------------------------------------------------------- A12's v1 answers (private run, mini only)
needs_a12 = pytest.mark.skipif(not (A12_V1 / "outputs.jsonl").exists(), reason="A12 v1 run not on this machine")


@needs_a12
def test_calibration_reproduces_frozen_copy_t():
    c = R.calibrate(A12_V1)
    assert (c["n"], c["p95"], c["above_p95"]) == (60, R.COPY_T, 3)


@needs_a12
def test_a12_v1_answers_score_finite_and_sound():
    try:
        R._glossary = None
        R.glossary()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"guide index not available: {e}")
    R.set_token_counter(None)
    try:
        R.token_count("probe")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qwen3-4B tokenizer not cached: {e}")
    recs = [json.loads(l) for l in (A12_V1 / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {r["id"]: R.analyse(r["answer"], R.encode_meta(lang=r["lang"], type=r["type"], q=r["q"], context=r["context"]))
           for r in recs}
    totals = [x["total"] for x in res.values()]
    assert all(math.isfinite(t) for t in totals)
    # the reference answers are what a good rollout looks like: none is pushed below zero
    assert min(totals) >= 0.0
    assert all(x["components"]["lang"] == 1.0 for x in res.values())
    assert all(x["components"]["oos"] == 2.0 for r, x in zip(recs, res.values()) if r["type"] == "oos")
    assert all(x["components"]["handoff"] == 1.0 for r, x in zip(recs, res.values()) if r["type"] == "advice")
    assert not any(x["facts"]["at_cap"] for x in res.values())
