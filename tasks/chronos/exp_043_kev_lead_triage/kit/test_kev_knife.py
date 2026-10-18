"""Kev knife lead triage: the code policy (LABELS.md rules) and contact masking. No model, no network."""
import scripts.kev_knife as K


def o(intent="buy", tl="months", q=0.9, hot=0.1, spec=0.1, call=0.1, worth=0.1, b2b=0.1):
    return {"intent": {"choice": intent, "confidence": 0.9}, "timeline": {"choice": tl}, "qualified": q, "hot": hot,
            "specific_listing": spec, "call_request": call, "asks_worth": worth, "b2b": b2b}


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


def test_b2b_pitch_becomes_spam_only_above_the_cut():
    x = o("sell", b2b=K.B2B_CUT + 0.01)
    assert K.policy(x, F)[1] == "review" and x["intent"]["choice"] == "spam"
    assert K.policy(o("sell", b2b=K.B2B_CUT - 0.01), F)[1] != "review"


def test_contacts_are_masked_before_kev():
    m = K.masked("call me +34 600 000 123 or anna@example.com")
    assert "600" not in m and "@" not in m and "[phone]" in m and "[email]" in m


def test_budget_units_in_russian_and_german():
    assert K.code_facts.__wrapped__ if hasattr(K.code_facts, "__wrapped__") else True
    import re
    m = K._MONEY.search("бюджет до 2,5 млн евро")
    assert m and m.group(2).lower() == "млн"
    assert K._MONEY.search("bis 1,2 Mio").group(2).lower() == "mio"
    assert K._MONEY.search("300 тыс евро").group(2).lower() == "тыс"
