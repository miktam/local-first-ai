"""exp_040 — the correctness-checkable reward (coapi_voice/rl40/reward40.py). No model, no network."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "coapi_voice"
sys.path[:0] = [str(ROOT / "rl"), str(ROOT / "rl40"), str(ROOT)]
import reward as R  # noqa: E402
import reward40 as W  # noqa: E402

CTX = ("[M4:T6] Plusvalía municipal (IIVTNU). On a transfer for consideration the seller pays the IIVTNU (art. 106 "
       "TRLRHL), unless the seller is a non-resident individual, in which case the buyer pays as substitute taxpayer. "
       "On an inheritance the acquirer files within 6 months, extendable to 1 year.")
Q = "I am a Spanish resident selling my flat to a private buyer. Who pays the plusvalía?"
CANDS = {"seller": {"en": ["seller", "vendor"], "pl": ["sprzedający"], "es": ["vendedor"]},
         "buyer": {"en": ["buyer", "purchaser"], "pl": ["kupujący"], "es": ["comprador"]},
         "both": {"en": ["both parties", "both"], "pl": ["obie strony"], "es": ["ambas partes"]}}
BODY = "On a sale the seller pays the plusvalía municipal (IIVTNU) under art. 106 TRLRHL, since the seller is resident."


@pytest.fixture(autouse=True)
def fixed_env():
    R.set_token_counter(lambda s: int(len(s.split()) * 1.4))
    W.analyse.cache_clear()
    yield
    R.set_token_counter(None)


def meta_cat(label="seller", lang="en", ctx=CTX, q=Q):
    return W.encode_meta40(lang=lang, kind="categorical", q=q, context=ctx, label=label, candidates=CANDS)


def meta_num(value, ctx=CTX, q="Within how many months must an heir file the plusvalía?", tol=0.005):
    return W.encode_meta40(lang="en", kind="numeric", q=q, context=ctx, value=value, tol=tol)


def comp(text, meta):
    return W.analyse(text, meta)


# ---------------------------------------------------------------- answer component
def test_right_label_scores_plus_three():
    r = comp(f"{BODY}\nAnswer: the seller", meta_cat())
    assert r["components"]["answer"] == 3.0 and r["facts"]["verdict"] == "right"


def test_wrong_label_scores_minus_two():
    assert comp(f"{BODY}\nAnswer: the buyer", meta_cat())["components"]["answer"] == -2.0


def test_hedge_two_candidates_scores_minus_two():
    r = comp(f"{BODY}\nAnswer: the seller or the buyer", meta_cat())
    assert r["components"]["answer"] == -2.0 and r["facts"]["verdict"] == "hedge"


def test_missing_final_line_scores_minus_one():
    assert comp(BODY, meta_cat())["components"]["answer"] == -1.0


def test_final_line_without_any_candidate_is_missing():
    assert comp(f"{BODY}\nAnswer: it depends", meta_cat())["components"]["answer"] == -1.0


def test_last_answer_line_wins():
    assert comp(f"Answer: buyer\n{BODY}\nAnswer: seller", meta_cat())["components"]["answer"] == 3.0


def test_localised_keys_and_forms_accent_folded():
    pl = W.encode_meta40(lang="pl", kind="categorical", q="Kto płaci?", context=CTX, label="seller", candidates=CANDS)
    es = W.encode_meta40(lang="es", kind="categorical", q="¿Quién paga?", context=CTX, label="seller", candidates=CANDS)
    assert comp("Płaci sprzedający.\nOdpowiedź: sprzedający", pl)["components"]["answer"] == 3.0
    assert comp("Paga el vendedor.\nOdpowiedz: sprzedajacy", pl)["components"]["answer"] == 3.0
    assert comp("Paga el vendedor.\nRespuesta: el vendedor", es)["components"]["answer"] == 3.0


def test_markdown_wrapped_key_is_read():
    assert comp(f"{BODY}\n**Answer:** seller", meta_cat())["components"]["answer"] == 3.0


def test_whole_word_only():
    # "both" must not fire inside "bother"; "seller" inside "bestseller" must not fire either
    assert comp(f"{BODY}\nAnswer: a bestseller bother", meta_cat())["components"]["answer"] == -1.0


def test_nested_span_prefers_longer_candidate():
    cands = {"built": {"en": ["built area"], "pl": ["x"], "es": ["y"]},
             "common": {"en": ["built area with common elements"], "pl": ["z"], "es": ["w"]}}
    m = W.encode_meta40(lang="en", kind="categorical", q="Which area?", context=CTX, label="common", candidates=cands)
    assert comp("It includes common areas.\nAnswer: built area with common elements", m)["components"]["answer"] == 3.0


def test_abstain_is_a_candidate_everywhere():
    r = comp(f"{BODY}\nAnswer: not in the material", meta_cat())
    assert r["components"]["answer"] == -2.0 and r["facts"]["verdict"] == "wrong"


def test_abstain_item_rewards_abstention_and_spanish_no_is_not_a_hedge():
    yn = {"yes": {"en": ["yes"], "pl": ["tak"], "es": ["sí"]}, "no": {"en": ["no"], "pl": ["nie"], "es": ["no"]}}
    m = W.encode_meta40(lang="es", kind="abstain", q="¿Puedo?", context="otro texto", label="ABSTAIN")
    assert comp("El material no lo cubre.\nRespuesta: no está en el material", m)["components"]["answer"] == 3.0
    m2 = W.encode_meta40(lang="es", kind="yesno", q="¿Puedo?", context=CTX, label="no", candidates=yn)
    assert comp("No.\nRespuesta: no está en el material", m2)["facts"]["verdict"] == "wrong"   # abstain, not a no+abstain hedge
    assert comp("No.\nRespuesta: no", m2)["components"]["answer"] == 3.0


# ---------------------------------------------------------------- numeric
def test_numeric_right_with_unit():
    assert comp("The heir has 6 months.\nAnswer: 6 months", meta_num(6))["components"]["answer"] == 3.0


@pytest.mark.parametrize("line,value", [("122,606.47 €", 122606.47), ("122.606,47 €", 122606.47), ("€ 26 256", 26256),
                                        ("0.01%", 0.01), ("1,5 %", 1.5), ("0 EUR", 0.0), ("-45 points", -45)])
def test_numeric_formats(line, value):
    assert comp(f"Body.\nAnswer: {line}", meta_num(value))["components"]["answer"] == 3.0


def test_numeric_two_values_is_hedge():
    r = comp("Body.\nAnswer: 5 or 7 years", meta_num(5))
    assert r["components"]["answer"] == -2.0 and r["facts"]["verdict"] == "hedge"


def test_numeric_wrong_and_out_of_tolerance():
    assert comp("Body.\nAnswer: 4 months", meta_num(6))["components"]["answer"] == -2.0
    assert comp("Body.\nAnswer: 100.6", meta_num(100))["components"]["answer"] == -2.0
    assert comp("Body.\nAnswer: 100.4", meta_num(100))["components"]["answer"] == 3.0


def test_numeric_citation_on_line_is_not_a_second_number():
    assert comp("Body.\nAnswer: 9 days (art. 1522 CC)", meta_num(9))["components"]["answer"] == 3.0


def test_numeric_abstain_plus_number_is_hedge():
    assert comp("Body.\nAnswer: 6, not in the material", meta_num(6))["facts"]["verdict"] == "hedge"


# ---------------------------------------------------------------- form terms and dominance
def test_form_terms():
    r = comp(f"{BODY}\nAnswer: seller", meta_cat())["components"]
    assert r["lang"] == 0.5 and r["plain"] == 0.25 and r["cite"] == 0.25 and r["num"] == 0.0
    bad = comp("**Seller** pays under art. 999 LH and 77 777 euros.\nAnswer: seller", meta_cat())["components"]
    assert bad["plain"] == -0.25 and bad["cite"] == -0.5 and bad["num"] == -0.5


def test_correctness_dominates_form():
    """A plain right answer always beats the best-formed wrong one."""
    right_plain = comp("**Seller.**\nAnswer: seller", meta_cat())["total"]
    wrong_polished = comp(f"{BODY}\nAnswer: buyer", meta_cat())["total"]
    assert right_plain > wrong_polished


def test_meta_validation():
    with pytest.raises(ValueError):
        W.encode_meta40(lang="en", kind="categorical", q="q", context="c", label="x", candidates=CANDS)
    with pytest.raises(ValueError):
        W.encode_meta40(lang="en", kind="numeric", q="q", context="c")
    with pytest.raises(ValueError):
        W.encode_meta40(lang="en", kind="abstain", q="q", context="c", label="seller")


def test_registered_component_signature():
    m = meta_cat()
    out = W.REWARD_FUNCTIONS[0](["p"], [f"{BODY}\nAnswer: seller"], [m])
    assert out == [3.0] and W.REWARD_FUNCTION_NAMES.split(",")[0] == "coapi40_answer"


# ---------------------------------------------------------------- the renderer (train = filter = serve)
def test_system40_ends_with_final_line_and_keeps_advice_suffix():
    import prompt40 as P
    for lang in ("en", "pl", "es"):
        assert P.system40("Who pays the plusvalía?", lang).endswith(W.FINAL_LINE[lang])
        assert P.system40("Who pays the plusvalía?", lang).startswith(P.G.SYSTEM[lang])


def test_source_retrieved_pl_is_section_level():
    import prompt40 as P
    ids = ["pl:M2:T12:concepts:5", "pl:M2:T13:vocab:1"]
    assert P.source_retrieved("en:M2:T12:concepts:3", "pl", ids)
    assert not P.source_retrieved("en:M2:T12:citations:3", "pl", ids)
    assert P.source_retrieved("en:M2:T12:concepts:3", "es", ["en:M2:T12:concepts:3"])
    assert not P.source_retrieved("en:M2:T12:concepts:3", "en", ["en:M2:T12:concepts:4"])


def test_inline_final_sentence_is_the_final_line():
    """Qwen3-4B writes one paragraph and puts the answer last (smoke run 2026-10-15)."""
    assert comp(f"{BODY} Answer: seller.", meta_cat())["components"]["answer"] == 3.0
    assert comp("Cap is in art. 1454 CC. Answer: €122,606.47.", meta_num(122606.47))["components"]["answer"] == 3.0
    pl = W.encode_meta40(lang="pl", kind="categorical", q="Kto?", context=CTX, label="seller", candidates=CANDS)
    assert comp("Płaci sprzedający (art. 106). ODPOWIEDŹ: sprzedający.", pl)["components"]["answer"] == 3.0


def test_key_inside_a_sentence_is_not_a_final_line():
    assert comp("I think the answer: seller is likely here", meta_cat())["components"]["answer"] == -1.0
    # the last valid key wins; a mid-sentence one after it is ignored
    assert comp("Answer: buyer\nso the answer: seller", meta_cat())["components"]["answer"] == -2.0


def test_options_line_is_localised_seeded_and_complete():
    import prompt40 as P
    en = P.options_line(CANDS, "en", "M1-x-01")
    assert en.startswith("Options: ") and en.endswith(".")
    assert sorted(en[len("Options: "):-1].split(" / ")) == ["both parties", "buyer", "seller"]
    assert en == P.options_line(CANDS, "en", "M1-x-01")                       # stable
    assert P.options_line(CANDS, "pl", "M1-x-01").startswith("Opcje: ") and "sprzedający" in P.options_line(CANDS, "pl", "k")
    assert P.options_line(CANDS, "es", "k").startswith("Opciones: ")


def test_options_shown_after_question_but_not_used_for_retrieval():
    import prompt40 as P

    class Idx:
        def __init__(self):
            self.queries = []

        def retrieve(self, q, k, budget):
            self.queries.append(q)
            return []
    idx = {"en": Idx(), "pl": Idx()}
    r = P.render40("Who pays?", "en", idx, "Options: seller / buyer.")
    assert idx["en"].queries == ["Who pays?"]
    assert r["user"].endswith("Who pays?\nOptions: seller / buyer.") and r["shown_q"] == "Who pays?\nOptions: seller / buyer."
    assert P.render40("Who pays?", "en", idx)["user"].endswith("Who pays?")


def test_withheld_chunks_are_removed_from_the_index():
    import prompt40 as P
    import guide_bm25 as gb
    idx = {"en": gb.load_index("en"), "pl": gb.load_index("pl")}
    q = "Who pays the plusvalía municipal when a resident sells a flat?"
    top = P.render40(q, "en", idx)["chunk_ids"]
    rule = next(c for c in top if ":vocab:" not in c)
    r = P.render40(q, "en", idx, withhold=[rule])
    assert rule not in r["chunk_ids"] and r["context"]
    assert P.render40(q, "es", idx, withhold=[rule])["chunk_ids"] and rule not in P.render40(q, "es", idx, withhold=[rule])["chunk_ids"]
    sec = P.pl_section(rule, idx)
    assert sec and all(s.startswith("pl:") for s in sec)
    assert not set(sec) & set(P.render40("Kto płaci plusvalía?", "pl", idx, withhold=sec)["chunk_ids"])
    with pytest.raises(ValueError):
        P.render40(q, "pl", idx, withhold=[rule])          # no PL id to withhold: refused
    both = P.render40("Kto płaci plusvalía?", "pl", idx, withhold=[rule] + sec)   # v3 lists both languages
    assert not set(sec) & set(both["chunk_ids"])
    assert rule not in P.render40(q, "en", idx, withhold=[rule] + sec)["chunk_ids"]


def test_answering_an_abstain_row_is_wrong_not_missing():
    m = W.encode_meta40(lang="en", kind="abstain", q="Who pays?", context="other text", label="ABSTAIN")
    r = comp(f"{BODY}\nAnswer: the seller", m)
    assert r["components"]["answer"] == -2.0 and r["facts"]["verdict"] == "wrong"
    assert comp("Answer: 6 months", m)["components"]["answer"] == -2.0
    assert comp(BODY, m)["components"]["answer"] == -1.0                          # no final line: still missing
    assert comp(f"{BODY} Answer: not in the material.", m)["components"]["answer"] == 3.0


# ---------------------------------------------------------------- the v3 harness and the statistics
def test_mcnemar_and_paired_bootstrap():
    import compare40 as C
    assert C.mcnemar_one_sided(0, 0) == 1.0
    assert abs(C.mcnemar_one_sided(10, 2) - sum(__import__("math").comb(12, k) for k in range(10, 13)) / 4096) < 1e-12
    units = [("en/single", 1, 0)] * 30 + [("en/single", 1, 1)] * 60 + [("en/single", 0, 1)] * 5 + [("en/single", 0, 0)] * 55
    r = C.paired(units)
    assert r["diff"] == round(25 / 150, 4) and r["ci95"][0] > 0 and r["shown"]
    null = C.paired([("s", 1, 0)] * 10 + [("s", 0, 1)] * 10 + [("s", 1, 1)] * 50)
    assert not null["shown"] and null["diff"] == 0


def test_harness_row_helpers():
    import run_arm40 as A
    rows = [{"id": "a", "type": "categorical", "twin_of": None, "source_chunk": "x"},
            {"id": "b", "type": "categorical", "twin_of": "a", "source_chunk": "x"},
            {"id": "c", "type": "numeric", "twin_of": None, "source_chunk": "x"},
            {"id": "d", "type": "abstain", "twin_of": None, "source_chunk": "x"},
            {"id": "e", "type": "abstain", "twin_of": None, "source_chunk": None}]
    twins = {r["twin_of"] for r in rows if r["twin_of"]}
    assert [A.kind_of(r, twins) for r in rows] == ["pair", "pair", "single", "abstain", "oos"]
    row = {"id": "b", "lang": "en", "type": "categorical", "twin_of": "a", "candidates": CANDS}
    assert A.options_for(row) == A.P.options_line(CANDS, "en", "a")           # twins share the order
    assert A.options_for({**row, "type": "abstain", "candidates": None}) is None
    m = json.loads(A.meta_of({**row, "label": "seller", "numeric": None}, "Who?\nOptions: x.", "ctx"))
    assert m["label"] == "seller" and m["q"].endswith("Options: x.")
    m = json.loads(A.meta_of({"id": "e", "lang": "pl", "type": "abstain"}, "Q", "ctx"))
    assert m["label"] == W.ABSTAIN and m["candidates"] is None


def test_glue_patches_add_final_line_and_strip_options():
    import run_arm40 as A
    seen = {}
    G = A.G
    orig_chat, orig_scope = G.ev.ollama_chat, G.in_scope
    G.ev.ollama_chat = lambda model, messages, opts: seen.setdefault("sys", messages[0]["content"]) and {"message": {"content": ""}}
    G.in_scope = lambda q, model: seen.setdefault("scope_q", q) and (True, "in")
    try:
        with A.glue40("en", "Options: a / b."):
            G.ev.ollama_chat("m", [{"role": "system", "content": G.SYSTEM["en"] + " extra"}], {})
            G.in_scope("Who pays?\nOptions: a / b.", "m")
    finally:
        G.ev.ollama_chat, G.in_scope = orig_chat, orig_scope
    assert seen["sys"].endswith(W.FINAL_LINE["en"]) and seen["scope_q"] == "Who pays?"
    assert G.ev.ollama_chat is orig_chat                                      # restored

    class Idx:
        def retrieve(self, q, k=5, budget=None):
            return [q]
    assert A._BareIndex(Idx(), "Options: a / b.").retrieve("Who pays?\nOptions: a / b.") == ["Who pays?"]
