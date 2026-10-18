"""exp_041 — the answerability gate harness (coapi_voice/rl41). No model, no network."""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "coapi_voice"
sys.path[:0] = [str(ROOT / "rl41"), str(ROOT / "rl40"), str(ROOT / "rl"), str(ROOT)]
import gate41 as GT     # noqa: E402
import score41 as S     # noqa: E402
import select41 as SEL  # noqa: E402


def test_p_yes_from_top_logprobs():
    tops = [{"token": "yes", "logprob": math.log(0.6)}, {"token": " No", "logprob": math.log(0.2)},
            {"token": "Yes", "logprob": math.log(0.1)}, {"token": "maybe", "logprob": math.log(0.1)}]
    assert abs(GT.p_yes(tops) - 0.7 / 0.9) < 1e-9
    assert GT.p_yes([{"token": "maybe", "logprob": 0.0}]) == 0.5


def test_threshold_keeps_false_abstention_at_or_under_the_cap():
    ans = [i / 100 for i in range(100)]            # 100 answerable rows, p = 0.00 … 0.99
    tau = SEL.threshold(ans, 0.03)
    assert sum(p < tau for p in ans) == 3 and tau == 0.03
    assert sum(p < SEL.threshold(ans, 0.0) for p in ans) == 0


def test_threshold_with_ties_never_exceeds_the_cap():
    ans = [0.1] * 5 + [0.9] * 95
    tau = SEL.threshold(ans, 0.03)
    assert sum(p < tau for p in ans) <= 3


def test_ece_and_auroc():
    assert SEL.ece([(1.0, 1), (0.0, 0)]) == 0.0
    assert abs(SEL.ece([(0.9, 0), (0.9, 0)]) - 0.9) < 1e-9
    assert SEL.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert SEL.auroc([0.5], [0.5]) == 0.5


def test_gated_rows():
    base = {"a": {"kind": "abstain", "verdict": "wrong", "abstained": False, "lang": "en"},
            "b": {"kind": "single", "verdict": "right", "abstained": False, "lang": "en"},
            "c": {"kind": "oos", "verdict": "right", "abstained": True, "lang": "pl"}}
    g = S.gated(base, {"a": 0.1, "b": 0.2, "c": 0.9}, tau=0.5)
    assert g["a"]["verdict"] == "right" and g["a"]["abstained"]          # blocked abstention row: now right
    assert g["b"]["verdict"] == "wrong" and g["b"]["abstained"]          # blocked answerable row: false abstention
    assert g["c"] == {**base["c"], "gate_blocked": False}                # passed: B0i's own verdict stands


def test_spanish_questions_are_searched_in_english(monkeypatch):
    import xling as X
    monkeypatch.setattr(X, "translate_es_en", lambda q, model: "EN: " + q)
    assert X.query_for_index("¿Quién paga?", "es", "m") == "EN: ¿Quién paga?"
    assert X.query_for_index("Who pays?", "en", "m") == "Who pays?"
    assert X.query_for_index("Kto płaci?", "pl", "m") == "Kto płaci?"


def test_glue_clean_trims_an_unpunctuated_ending_only_when_truncated():
    import glue as G
    text = "The landlord pays for the repair (art. 21 LAU).\n\nIn short: the landlord, art. 21 LAU"
    assert G.clean(text, truncated=False).endswith("In short: the landlord, art. 21 LAU")
    assert not G.clean(text, truncated=True).endswith("21 LAU")            # truncated: the unpunctuated tail goes
    assert G.clean(text) == G.clean(text, truncated=True)                 # default keeps the old behaviour
    cut = "The deposit is one month. The landlord must return it within one month after the keys are"
    assert G.clean(cut, truncated=True) == "The deposit is one month."
