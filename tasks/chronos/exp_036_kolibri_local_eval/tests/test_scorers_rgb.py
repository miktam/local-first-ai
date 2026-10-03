"""scorers/rgb.py: golden synthetic cases, the frozen lexicon, and parity with RGB's own code run in place.

`test_checkanswer_matches_rgb_in_place` (BUILD_SPEC §5.6) reads RGB's evalue.py from the local checkout
($EXP036_DATA/RGB-src, RGB @ 65ec39e4), compiles only its `checkanswer` and `predict` functions (the file's
heavy imports are never executed) and compares them with our implementation on 500 seeded synthetic cases.
Nothing of RGB is copied into the repo; the source is read at test time only.
"""

from __future__ import annotations

import ast
import builtins
import json
import os
import random
from pathlib import Path

import pytest

from scorers import rgb

EXP_DIR = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((EXP_DIR / "scorers" / "golden" / "rgb_golden.json").read_text(encoding="utf-8"))
RGB_COMMIT = "65ec39e40e7dc9abb50e9bf1b4f32be3f6f16615"


def _rgb_src() -> Path | None:
    cands = []
    if os.environ.get("EXP036_DATA"):
        cands.append(Path(os.environ["EXP036_DATA"]).expanduser())
    if os.environ.get("EXP036_MODELS"):
        cands.append(Path(os.environ["EXP036_MODELS"]).expanduser() / "data")
    # Default layouts: the mini's build-time assets (BUILD_SPEC §8) and the run host (ASSETS.md).
    cands += [Path("~/models/exp036-mini/data").expanduser(), Path("~/models/exp036/data").expanduser()]
    for d in cands:
        if (d / "RGB-src" / "evalue.py").is_file():
            return d / "RGB-src"
    return None


def _rgb_functions(src: Path) -> dict:
    """RGB's checkanswer and predict, compiled from evalue.py without running its imports."""
    tree = ast.parse((src / "evalue.py").read_text(encoding="utf-8"))
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("checkanswer", "predict")]
    assert sorted(n.name for n in wanted) == ["checkanswer", "predict"]
    ns: dict = {"__builtins__": builtins}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(src / "evalue.py"), "exec"), ns)
    return ns


class _EchoModel:
    """Stands in for RGB's model: generate() returns the synthetic prediction."""

    def __init__(self, prediction: str):
        self.prediction = prediction

    def generate(self, text, temperature, system=None):
        return self.prediction


# ---------------------------------------------------------------------------------------------------
# Golden


@pytest.mark.parametrize("case", GOLDEN["closed_book"], ids=lambda c: c["why"][:50])
def test_golden_closed_book(case):
    assert rgb.classify_cb(case["answer"], case["gold"]) == case["expected"]


@pytest.mark.parametrize("case", GOLDEN["negative"], ids=lambda c: c["answer"][:40])
def test_golden_negative(case):
    assert rgb.classify_neg(case["answer"]) == case["expected"]


@pytest.mark.parametrize("case", GOLDEN["fact_check"], ids=lambda c: c["answer"][:40])
def test_golden_fact_check(case):
    assert rgb.classify_fact(case["answer"], case["true"], case["fake"]) == case["expected"]


def test_overlap_cases_named_in_build_spec():
    # "gold string plus a lexicon hedge" -> correct
    assert rgb.classify_cb("I'm not sure, but it is Lunaport.", ["Lunaport"]) == "correct"
    assert rgb.has_lexicon_hit("I'm not sure, but it is Lunaport.", "abstain")
    # "gold string plus 'insufficient information'" -> not correct
    assert rgb.classify_cb("Lunaport, though there is insufficient information.", ["Lunaport"]) != "correct"


def test_lexicon_never_changes_correct():
    rng = random.Random(2333)
    phrases = rgb.lexicon()["abstain"]
    for _ in range(200):
        hedge = rng.choice(phrases)
        answer = f"{hedge.capitalize()}, yet my answer is Lunaport."
        if rgb.REJECT_STRING in answer:
            continue
        assert rgb.classify_cb(answer, ["Lunaport"]) == "correct"


# ---------------------------------------------------------------------------------------------------
# The frozen lexicon


def test_lexicon_shape():
    lex = rgb.lexicon()
    assert lex["schema"] == "exp036 abstain lexicon v1"
    for kind in ("abstain", "fact_error"):
        phrases = lex[kind]
        assert phrases and len(set(phrases)) == len(phrases)
        assert all(p == rgb.normalise(p) and p.strip() == p and p for p in phrases)
    assert not set(lex["abstain"]) & set(lex["fact_error"])
    assert "insufficient information" in lex["abstain"]
    assert "factual errors" in lex["fact_error"]
    assert len(rgb.lexicon_sha256()) == 64


def test_lexicon_matching_rules():
    assert rgb.lexicon_hits("I DON’T   know", "abstain") == ["i don't know"]
    assert rgb.lexicon_hits("We cannot determine it.", "abstain") == ["cannot determine"]
    assert rgb.lexicon_hits("cannot determined", "abstain") == []  # a letter follows the phrase
    assert rgb.lexicon_hits("xcannot determine", "abstain") == []  # a letter precedes it
    assert rgb.lexicon_hits("The documents contain factual errors.", "fact_error") == ["factual errors"]
    with pytest.raises(ValueError):
        rgb.lexicon_hits("x", "other")


def test_rejection_string_is_case_sensitive_but_lexicon_is_not():
    assert rgb.REJECT_STRING not in "Insufficient information"
    assert rgb.classify_neg("Insufficient information") == "rejected"  # via the lexicon
    assert rgb.is_correct("Insufficient information, Lunaport", ["Lunaport"])  # RGB's rule: not rejected
    assert not rgb.is_correct("insufficient information, Lunaport", ["Lunaport"])


# ---------------------------------------------------------------------------------------------------
# Parity with RGB's own code, run in place


_WORDS = ["Lunaport", "Sandmere", "quorvia", "ADA", "Brightwell", "2031", "Straße", "Élan", "x", "",
          "Luna Port", "zeta-9", "Kjøge", "OMEGA prize", "the", "of"]


def _rand_alias(rng: random.Random) -> str:
    w = rng.choice(_WORDS)
    return rng.choice([w, w.lower(), w.upper(), w.title()])


def _rand_gold(rng: random.Random):
    shape = rng.randrange(6)
    if shape == 0:
        return _rand_alias(rng)  # bare string: one slot
    if shape == 1:
        return [_rand_alias(rng) for _ in range(rng.randint(1, 3))]  # flat list: every string a slot
    if shape == 2:
        return [[_rand_alias(rng) for _ in range(rng.randint(1, 3))] for _ in range(rng.randint(1, 3))]
    if shape == 3:
        return [[_rand_alias(rng) for _ in range(rng.randint(0, 2))], _rand_alias(rng)]  # mixed, maybe []
    if shape == 4:
        return []
    return [[_rand_alias(rng)]]


def _rand_prediction(rng: random.Random, gold) -> str:
    flat: list[str] = []
    for slot in (gold if isinstance(gold, list) else [gold]):
        flat.extend(slot if isinstance(slot, list) else [slot])
    parts = [_rand_alias(rng) for _ in range(rng.randint(0, 4))]
    parts += [rng.choice([a, a.lower(), a.upper()]) for a in flat if rng.random() < 0.6]
    extra = rng.random()
    if extra < 0.1:
        parts.append("insufficient information")
    elif extra < 0.15:
        parts.append("Insufficient Information")
    elif extra < 0.25:
        parts.append("there are factual errors")
    rng.shuffle(parts)
    return rng.choice([" ", ", ", "", "\n"]).join(parts)


def test_checkanswer_matches_rgb_in_place():
    src = _rgb_src()
    if src is None:
        pytest.skip("RGB checkout not found (set EXP036_DATA to the directory holding RGB-src)")
    head = (src / ".git" / "HEAD").read_text().strip() if (src / ".git" / "HEAD").is_file() else ""
    if head.startswith("ref: "):
        head = (src / ".git" / head[5:]).read_text().strip()
    assert head == RGB_COMMIT, f"RGB checkout is at {head!r}, not the pinned {RGB_COMMIT}"
    ns = _rgb_functions(src)
    rng = random.Random(36)
    n_correct = n_rejected = n_fact = 0
    for _ in range(500):
        gold = _rand_gold(rng)
        pred = _rand_prediction(rng, gold)
        assert rgb.checkanswer(pred, gold) == ns["checkanswer"](pred, gold), (pred, gold)
        # RGB's own decision path for the closed-book setting (passage_num = 0, no documents).
        labels, prediction, factlabel = ns["predict"](
            "q", gold, [], _EchoModel(pred), "system", "{QUERY}{DOCS}", 0.7, "en")
        assert prediction == pred
        theirs = (0 not in labels and 1 in labels)
        assert rgb.is_correct(pred, gold) == theirs, (pred, gold, labels)
        assert (rgb.FACT_STRING in pred) == bool(factlabel)
        n_correct += theirs
        n_rejected += labels == [-1]
        n_fact += factlabel
    # The generator must exercise every branch.
    assert 50 < n_correct < 450 and n_rejected > 10 and n_fact > 10
