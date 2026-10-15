"""exp_039 — arithmetic templates and the prompt builder's invariants. No model, no network."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "coapi_voice"
sys.path[:0] = [str(ROOT), str(ROOT / "rl")]
import arith_templates as at  # noqa: E402
import reward as R  # noqa: E402

BY_ID = {t["id"]: t for t in at.TEMPLATES}


def test_templates_are_complete():
    assert len(at.TEMPLATES) == 15 and len(BY_ID) == 15
    assert not {"F03", "F10", "F11", "F15", "F19"} & set(BY_ID)              # failed the blind v2.1 check
    for t in at.TEMPLATES:
        assert t["split"] in ("train", "valid") and t["guide_quote"]
        assert not t["tema"].startswith("M6:")                       # Módulo 6 is never trained
        for lang in ("en", "pl", "es"):
            for var in t["inputs"]:
                assert "{" + var + "}" in t["q"][lang], (t["id"], lang, var)
    assert [t["id"] for t in at.TEMPLATES if t["split"] == "valid"] == ["F25"]


@pytest.mark.parametrize("tid,values,expected", [
    ("F01", {"principal": 200000, "tin": 3.0, "years": 25}, 948.42),         # standard annuity
    ("F05", {"rent": 1200, "use": "other"}, 2400.0),
    ("F16", {"budget": 1000000, "machinery": 200000}, 80000.0),
    ("F18", {"penalty": 10000}, 4200.0),                                     # 30 % then 40 % off
    ("F21", {"price": 100000, "age": 34}, 300.0),
    ("F21", {"price": 100000, "age": 35}, 1000.0),
    ("F22", {"rent": 10000, "interest": 4000, "repairs": 3000}, 0.0),
    ("F24", {"turnover": 50000000, "value": 3000000, "profit": 100000}, 10000000.0),
    ("F25", {"promoters": 60, "neutrals": 30, "detractors": 10}, 50.0),
])
def test_gold_values(tid, values, expected):
    assert at.gold(BY_ID[tid], values) == pytest.approx(expected, abs=0.01)


def test_number_format_per_language():
    assert [at.fmt(350000, "eur", l) for l in ("en", "es", "pl")] == ["350,000", "350.000", "350 000"]
    assert [at.fmt(4500, "eur", l) for l in ("en", "es", "pl")] == ["4,500", "4500", "4500"]
    assert [at.fmt(2.5, "dec1", l) for l in ("en", "es", "pl")] == ["2.5", "2,5", "2,5"]


def test_rendered_gold_is_found_by_the_reward_when_written_back():
    """The arithmetic reward must recognise its own gold in each language's number style."""
    rng = random.Random(39)
    for t in at.TEMPLATES:
        for lang in ("en", "es", "pl"):
            q, g = at.render(t, lang, rng)
            written = at.fmt(g, "dec2", lang) if g != int(g) else at.fmt(int(g), "eur", lang)
            assert R._arith_hit(f"The result is {written}.", g), (t["id"], lang, written)


def test_render_is_deterministic():
    a = [at.render(t, "es", random.Random(7)) for t in at.TEMPLATES]
    b = [at.render(t, "es", random.Random(7)) for t in at.TEMPLATES]
    assert a == b


def test_expected_trainable_matches_qwen3_4b_qkvo_r16():
    src = (ROOT / "rl" / "train_r1.py").read_text(encoding="utf-8")
    assert "EXPECTED_TRAINABLE = 16 * ((2560 + 4096) + (2560 + 1024) * 2 + (4096 + 2560)) * 36" in src
    cfg = json.loads(json.dumps(__import__("yaml").safe_load((ROOT / "rl" / "grpo_config.yaml").read_text())))
    assert cfg["lora_parameters"]["keys"] == [f"self_attn.{p}_proj" for p in "qkvo"]
    assert (cfg["lora_parameters"]["rank"], cfg["lora_parameters"]["scale"]) == (16, 2.0)
    assert (cfg["group_size"], cfg["beta"], cfg["epsilon"], cfg["max_completion_length"]) == (6, 0.02, 0.2, R.CAP)
    assert cfg["reward_functions"] == R.REWARD_FUNCTION_NAMES


def test_measurement_prompt_is_the_training_prompt(monkeypatch):
    """rl/run_arm.py's prompt mode (arms R1, B0r) must send byte-for-byte what R1 was trained on."""
    import run_arm as RA
    RA.use_index(RA.INDEX_LIVE)
    idx = {lang: RA.g.load_index(lang) for lang in RA.g.LANGS}
    sent = []
    monkeypatch.setattr(RA.ev, "ollama_chat", lambda model, msgs, opts: sent.append((msgs, opts)) or {"message": {"content": ""}})
    rows = [json.loads(l) for l in (ROOT / "data/rl/train.jsonl").read_text(encoding="utf-8").splitlines()[:60]]
    for r in rows:
        RA.prompt_answer(json.loads(r["answer"])["q"], RA.BASE_MODEL, idx)
        msgs, opts = sent[-1]
        assert msgs == [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}], r["id"]
        assert opts == RA.ev.OPTIONS
