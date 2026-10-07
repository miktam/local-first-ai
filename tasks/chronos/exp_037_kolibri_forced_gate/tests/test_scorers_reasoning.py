"""scorers/reasoning.py: the reasoning split (BUILD_SPEC §4 item 28, §5.6; HYPOTHESIS pilot rule 3).

- golden text cases per family and prompt state;
- the prompt state of every pinned template's actual render equals `expected_prompt_state`;
- split from token ids == split from the decoded text with special tokens kept, on real tokenizers;
- Gemma 4 "none" is legitimate, Kolibri / Qwen "none" with thinking on is a defect.

Tokenizers: Kolibri from the conftest `tokenizer_dir` fixture; the peers from $EXP036_PEER_TOK_DIR,
$EXP036_MODELS, or the default layouts ~/models/exp036-mini/mlx-community and ~/models/exp036.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scorers import reasoning as R

EXP_DIR = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((EXP_DIR / "scorers" / "golden" / "reasoning_text_golden.json").read_text(encoding="utf-8"))
KOLIBRI_TEMPLATE = EXP_DIR / "tests" / "fixtures" / "kolibri1_chat_template.vendor.jinja"

PEER_FOLDERS = {
    "gemma4": "gemma-4-26b-a4b-it-8bit",
    "qwen3_6": "Qwen3.6-35B-A3B-8bit",
    "qwen3_8": "Qwen3.8-27B-8bit",
}
RUNNER_TEMPLATES = {"gemma4": "gemma4.jinja", "qwen3_6": "qwen3_6.jinja", "qwen3_8": "qwen3_8.jinja"}

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# ---------------------------------------------------------------------------------------------------
# Pure rule


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"{c['family']}-{c['state']}-{c['completion'][:20]}")
def test_golden_text(case):
    assert list(R.split_text(case["family"], case["state"], case["completion"])) == case["expected"]


@pytest.mark.parametrize("case", GOLDEN["prompt_tails"], ids=lambda c: f"{c['family']}-{c['state']}")
def test_golden_prompt_state(case):
    assert R.prompt_state_text(case["family"], case["prompt"]) == case["state"]


def test_expected_prompt_states():
    assert R.expected_prompt_state("kolibri", effort="none") == "closed"
    for e in ("minimal", "low", "medium", "high", "xhigh"):
        assert R.expected_prompt_state("kolibri", effort=e) == "plain"
    assert R.expected_prompt_state("gemma4", thinking=True) == "plain"
    assert R.expected_prompt_state("gemma4", thinking=False) == "closed"
    assert R.expected_prompt_state("qwen3_6", thinking=True) == "open"
    assert R.expected_prompt_state("qwen3_8", thinking=False) == "closed"
    assert R.expected_prompt_state("qwen3_6", effort="high") == "open"
    assert R.expected_prompt_state("qwen3_6", effort="off") == "closed"
    with pytest.raises(ValueError):
        R.expected_prompt_state("kolibri")


def test_defects():
    assert R.reasoning_defect("kolibri", "none", effort="high")
    assert not R.reasoning_defect("kolibri", "none", effort="none")
    assert not R.reasoning_defect("kolibri", "closed", effort="high")
    assert R.reasoning_defect("qwen3_6", "none", thinking=True)
    assert not R.reasoning_defect("qwen3_6", "none", thinking=False)
    assert not R.reasoning_defect("gemma4", "none", thinking=True)  # Gemma 4 may skip the channel


def test_runner_prefill_labels():
    assert R.prompt_state_from_prefill("open_think") == "open"
    assert R.prompt_state_from_prefill("empty_think") == "closed"
    assert R.prompt_state_from_prefill("empty_channel") == "closed"
    assert R.prompt_state_from_prefill("none") == "plain"
    with pytest.raises(ValueError):
        R.prompt_state_from_prefill("other")


def test_family_names():
    assert R.family_of_arm("K8") == R.family_of_arm("K4") == "kolibri"
    assert R.family_of_arm("G8") == "gemma4"
    assert R.family_of_arm("Q36-8") == "qwen3_6"
    assert R.family_of_arm("Q38-4") == "qwen3_8"
    assert R.family_spec("Qwen3.6").name == "qwen3_6"
    with pytest.raises(ValueError):
        R.family_spec("llama")
    with pytest.raises(ValueError):
        R.family_of_arm("X1")


def test_split_ids_rule_without_tokenizer():
    spec = R.FAMILIES["kolibri"]
    ws = {10}
    is_ws = ws.__contains__
    o, c, e = spec.open_id, spec.close_id, spec.eos_ids[0]
    assert R.split_ids("kolibri", "plain", [10, o, 1, 2, c, 3, e], is_ws) == ([1, 2], [3], "closed")
    assert R.split_ids("kolibri", "plain", [5, o, 1, c, 3], is_ws) == ([], [5, o, 1, c, 3], "none")
    assert R.split_ids("kolibri", "plain", [o, 1, 2, e], is_ws) == ([1, 2], [], "unclosed")
    assert R.split_ids("kolibri", "closed", [o, 1, c, 2], is_ws) == ([], [o, 1, c, 2], "none")
    assert R.split_ids("kolibri", "open", [1, c, 2, e, e], is_ws) == ([1], [2], "closed")
    with pytest.raises(ValueError):
        R.split_ids("kolibri", "weird", [], is_ws)


# ---------------------------------------------------------------------------------------------------
# Real tokenizers and template renders


def _peer_dir(family: str) -> Path | None:
    folder = PEER_FOLDERS[family]
    cands = []
    if os.environ.get("EXP036_PEER_TOK_DIR"):
        cands.append(Path(os.environ["EXP036_PEER_TOK_DIR"]).expanduser())
    if os.environ.get("EXP036_MODELS"):
        m = Path(os.environ["EXP036_MODELS"]).expanduser()
        cands += [m, m / "mlx-community"]
    cands += [Path("~/models/exp036-mini/mlx-community").expanduser(), Path("~/models/exp036").expanduser()]
    for c in cands:
        if (c / folder / "tokenizer.json").is_file():
            return c / folder
    return None


def _load(family: str, kolibri_dir: Path | None):
    """(hf tokenizer for rendering, tokenizers.Tokenizer, template text) or skip."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    if family == "kolibri":
        d, template = kolibri_dir, KOLIBRI_TEMPLATE.read_text(encoding="utf-8")
    else:
        d = _peer_dir(family)
        if d is None:
            pytest.skip(f"no {PEER_FOLDERS[family]} tokenizer (set EXP036_PEER_TOK_DIR or EXP036_MODELS)")
        committed = EXP_DIR / "runner" / "templates" / RUNNER_TEMPLATES[family]
        template = (committed if committed.is_file() else d / "chat_template.jinja").read_text(encoding="utf-8")
    hf = transformers.AutoTokenizer.from_pretrained(str(d))
    raw = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
    return hf, raw, template


SETTINGS = [
    ("kolibri", {"reasoning_effort": "none"}, "closed"),
    ("kolibri", {"reasoning_effort": "low"}, "plain"),
    ("kolibri", {"reasoning_effort": "medium"}, "plain"),
    ("kolibri", {"reasoning_effort": "high"}, "plain"),
    ("gemma4", {"enable_thinking": True}, "plain"),
    ("gemma4", {"enable_thinking": False}, "closed"),
    ("qwen3_6", {"enable_thinking": True}, "open"),
    ("qwen3_6", {"enable_thinking": False}, "closed"),
    ("qwen3_8", {"enable_thinking": True}, "open"),
    ("qwen3_8", {"enable_thinking": True, "reasoning_effort": "xhigh"}, "open"),  # runner/chat.py's kwargs
    ("qwen3_8", {"enable_thinking": False}, "closed"),
]

ANSWER = "The answer is 4. Daher ist die Antwort (B)."


def _completions(family: str, state: str) -> list[tuple[str, str]]:
    """(completion text with special tokens, expected status) for a prompt state."""
    s = R.family_spec(family)
    eos = s.eos[0]
    think = "Two plus two.\nCheck: 4 = 4, fine; äöü ß 中."
    if state == "closed":
        return [(ANSWER + eos, "none")]
    if state == "open":
        return [(think + s.close + "\n\n" + ANSWER + eos, "closed"), (think, "unclosed"), (think + eos, "unclosed")]
    return [
        ("\n" + s.open + think + s.close + "\n\n" + ANSWER + eos, "closed"),
        (s.open + think, "unclosed"),
        (ANSWER + eos, "none"),
    ]


@pytest.mark.parametrize("family,kwargs,state", SETTINGS, ids=lambda x: str(x))
def test_template_render_state_and_id_text_parity(family, kwargs, state, request):
    kdir = request.getfixturevalue("tokenizer_dir") if family == "kolibri" else None
    hf, raw, template = _load(family, kdir)
    messages = [{"role": "user", "content": "What is 2 + 2? Antworte kurz."}]
    prompt_text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         chat_template=template, **kwargs)
    prompt_ids = raw.encode(prompt_text, add_special_tokens=False).ids
    effort = kwargs.get("reasoning_effort")
    thinking = kwargs.get("enable_thinking")
    assert R.expected_prompt_state(family, effort=effort, thinking=thinking) == state
    assert R.prompt_state_text(family, prompt_text) == state
    assert R.prompt_state_ids(family, prompt_ids) == state

    for completion, status in _completions(family, state):
        ids = raw.encode(completion, add_special_tokens=False).ids
        text = raw.decode(ids, skip_special_tokens=False)
        assert text == completion  # the stored text (special tokens kept) round-trips
        for tok in (raw, hf):
            from_ids = R.split_reasoning(family, prompt_ids, ids, tok)
            from_text = R.split_reasoning_text(family, prompt_text, text)
            assert from_ids == from_text, (family, kwargs, completion)
            assert from_ids[2] == status
        if status == "closed":
            assert from_ids[1].strip() == ANSWER
        r_n, a_n, st = R.split_token_counts(family, prompt_ids, ids, raw)
        assert st == status and r_n + a_n <= len(ids)
        r_ids, a_ids, st2 = R.split_reasoning_ids(family, prompt_ids, ids, raw)
        assert (len(r_ids), len(a_ids), st2) == (r_n, a_n, st)
        assert raw.decode(a_ids, skip_special_tokens=False) == from_ids[1]


def test_wrong_tokenizer_is_refused(tokenizer_dir):
    tokenizers = pytest.importorskip("tokenizers")
    raw = tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    with pytest.raises(ValueError):
        R.split_reasoning("gemma4", [], [1, 2], raw)
    R.check_tokenizer("kolibri", raw)  # the right one passes
