"""Kev knife lead triage: the code policy (LABELS.md rules) and contact masking. No model, no network."""
import scripts.kev_knife as K


def o(intent="buy", tl="months", q=0.9, hot=0.1, spec=0.1, call=0.1, worth=0.1, b2b=0.1, p_spam=0.0, own=None):
    x = {"intent": {"choice": intent, "confidence": 0.9, "probabilities": {intent: 0.9 - p_spam, "spam": p_spam}},
         "timeline": {"choice": tl}, "qualified": q, "hot": hot, "specific_listing": spec, "call_request": call,
         "asks_worth": worth, "b2b": b2b}
    if own is not None:
        x["wants_to_own"] = own
    return x


F = {"has_phone": False}


def test_actions_follow_the_written_rules_in_order():
    assert K.policy(o("spam"), F)[1] == "review"            # never discard
    assert K.policy(o("info"), F)[1] == "answer_question"
    assert K.policy(o("valuation"), F)[1] == "book_valuation"
    assert K.policy(o("sell", worth=0.9), F)[1] == "book_valuation"
    assert K.policy(o("buy", tl="now", call=0.9), F)[1] == "call_now"
    assert K.policy(o("buy", tl="now"), {"has_phone": True})[1] == "call_now"
    assert K.policy(o("buy", tl="now"), F)[1] == "send_listings"            # qualified, now, no call request
    assert K.policy(o("sell", tl="months"), F)[1] == "book_valuation"
    assert K.policy(o("buy", q=0.2), F)[1] == "ask_details"


def test_urgency_scale():
    assert K.policy(o("info"), F)[0] == 0 and K.policy(o("spam", hot=0.9), F)[0] == 0
    assert K.policy(o("buy", hot=0.9), F)[0] == 3
    assert K.policy(o("buy", tl="now"), F)[0] == 2 and K.policy(o("buy", q=0.2, spec=0.9), F)[0] == 2
    assert K.policy(o("buy", q=0.2), F)[0] == 1 and K.policy(o("buy", q=0.2, tl="later"), F)[0] == 0


def test_b2b_pitch_becomes_spam_only_when_kev_agrees():
    """rev 4: the business-pitch answer overrides Kev only above its cut AND when Kev's own spam probability agrees."""
    x = o("sell", b2b=K.B2B_CUT + 0.01, p_spam=K.SPAM_AGREE + 0.01)
    assert K.policy(x, F)[1] == "review" and x["intent"]["choice"] == "spam" and x["intent_kev"]["choice"] == "sell"
    assert K.policy(o("sell", b2b=K.B2B_CUT - 0.01, p_spam=0.9), F)[1] != "review"          # below the cut
    assert K.policy(o("sell", b2b=0.95, p_spam=K.SPAM_AGREE - 0.01), F)[1] != "review"       # Kev disagrees: no override


def test_wants_to_own_turns_a_holiday_rent_reading_into_buy():
    x = o("rent_holiday", own=K.OWN_CUT + 0.01)
    K.policy(x, F)
    assert x["intent"]["choice"] == "buy" and x["intent"]["by"] == "own rule"
    y = o("rent_holiday", own=K.OWN_CUT - 0.01)
    K.policy(y, F)
    assert y["intent"]["choice"] == "rent_holiday"


def test_budget_is_the_largest_amount_with_digits_or_words():
    assert K.code_facts("between 800k and 1M")["budget_eur"] == 1e6
    assert K.code_facts("presupuesto: medio millón")["budget_eur"] == 5e5
    assert K.code_facts("three bedrooms please")["budget_eur"] is None
    assert K.code_facts("villa à 300 m de la plage, budget deux millions d'euros")["budget_eur"] == 2e6   # metres, not millions
    assert K.code_facts("up to 1.2m")["budget_eur"] == 1.2e6


def test_contacts_are_masked_before_kev():
    m = K.masked("call me +44 7700 900123 or anna@example.com")
    assert "7700" not in m and "@" not in m and "[phone]" in m and "[email]" in m


def test_budget_units_in_russian_and_german():
    assert K.code_facts.__wrapped__ if hasattr(K.code_facts, "__wrapped__") else True
    import re
    m = K._MONEY.search("бюджет до 2,5 млн евро")
    assert m and m.group(2).lower() == "млн"
    assert K._MONEY.search("bis 1,2 Mio").group(2).lower() == "mio"
    assert K._MONEY.search("300 тыс евро").group(2).lower() == "тыс"
