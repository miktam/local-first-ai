"""runner/chat.py: per-family thinking kwargs, rendering with the committed
templates, EOS ids and sampling (BUILD_SPEC §5.4; HYPOTHESIS Reasoning, C9).

Tokenizers (no weights) come from the environment:
  Kolibri  $EXP036_TOK (or $EXP036_TOKENIZER_DIR) or $EXP036_MODELS/Kolibri-1-BF16 (conftest fixture)
  peers    $EXP036_PEER_TOK_ROOT/<mlx-community folder>, else $EXP036_MODELS/<folder>,
           else <$EXP036_TOK>/../../mlx-community/<folder> (the mini's layout)
  upstream $EXP036_UPSTREAM_DIR, else <$EXP036_TOK>/../../upstream (mini only)
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from exp036_helpers import kolibri_tok_env

from runner import chat

PEER_DIRS = {"gemma4": "gemma-4-26b-a4b-it-8bit", "qwen3_6": "Qwen3.6-35B-A3B-8bit", "qwen3_8": "Qwen3.8-27B-8bit"}
UPSTREAM = {"gemma4.jinja": "gemma-4-26B-A4B-it@20da991a", "qwen3_6.jinja": "Qwen3.6-35B-A3B@995ad96e",
            "qwen3_8.jinja": "Qwen3.8-27B@1d4bf0f2"}
MSGS = [{"role": "user", "content": "Wie viele Primzahlen liegen zwischen 10 und 20? Antworte kurz."}]
MSGS_SYS = [{"role": "system", "content": "Answer in one word."}, {"role": "user", "content": "Capital of Austria?"}]


def _peer_root() -> Path | None:
    cands = []
    if os.environ.get("EXP036_PEER_TOK_ROOT"):
        cands.append(Path(os.environ["EXP036_PEER_TOK_ROOT"]).expanduser())
    if os.environ.get("EXP036_MODELS"):
        cands.append(Path(os.environ["EXP036_MODELS"]).expanduser())
    if kolibri_tok_env():
        cands.append(Path(kolibri_tok_env()).resolve().parents[1] / "mlx-community")
    for c in cands:
        if all((c / d / "tokenizer.json").is_file() for d in PEER_DIRS.values()):
            return c
    return None


def _upstream_root() -> Path | None:
    cands = []
    if os.environ.get("EXP036_UPSTREAM_DIR"):
        cands.append(Path(os.environ["EXP036_UPSTREAM_DIR"]).expanduser())
    if kolibri_tok_env():
        cands.append(Path(kolibri_tok_env()).resolve().parents[1] / "upstream")
    for c in cands:
        if all((c / d / "chat_template.jinja").is_file() for d in UPSTREAM.values()):
            return c
    return None


@pytest.fixture(scope="module")
def peer_root() -> Path:
    r = _peer_root()
    if r is None:
        pytest.skip("no peer tokenizer folders: set EXP036_PEER_TOK_ROOT or EXP036_MODELS")
    return r


def _tok(path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


@pytest.fixture(scope="module")
def kolibri_tok(tokenizer_dir):
    return _tok(tokenizer_dir)


# ------------------------------------------------------------- weight-free


def test_template_kwargs_are_explicit_per_family():
    assert chat.template_kwargs("kolibri", "high") == {"reasoning_effort": "high"}
    assert chat.template_kwargs("kolibri", "none") == {"reasoning_effort": "none"}
    assert chat.template_kwargs("gemma4", "high") == {"enable_thinking": True}
    assert chat.template_kwargs("qwen3_6", "high") == {"enable_thinking": True}
    assert chat.template_kwargs("qwen3_8", "high") == {"enable_thinking": True, "reasoning_effort": "xhigh"}
    assert chat.template_kwargs("gemma4", "none") == {"enable_thinking": False}
    for fam, eff in (("kolibri", "xhigh"), ("gemma4", "low"), ("qwen3_6", "medium")):
        with pytest.raises(ValueError):
            chat.template_kwargs(fam, eff)


def test_family_of_and_sampling():
    assert [chat.family_of(a) for a in ("K8", "K4", "G8", "G4", "Q36-8", "Q38-4")] == [
        "kolibri", "kolibri", "gemma4", "gemma4", "qwen3_6", "qwen3_8"]
    with pytest.raises(ValueError):
        chat.family_of("X9")
    assert chat.sampling_for("kolibri") == {"order": "vllm", "temperature": 1.0, "top_p": 0.97, "top_k": 128}
    assert chat.SAMPLING["gemma4"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    assert chat.SAMPLING["qwen3_6"] == chat.SAMPLING["qwen3_8"] == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_committed_templates_match_their_manifest():
    man = json.loads((chat.TEMPLATES_DIR / "MANIFEST.json").read_text())
    assert set(man["files"]) == set(chat.TEMPLATE_FILES.values())
    for name, e in man["files"].items():
        b = (chat.TEMPLATES_DIR / name).read_bytes()
        assert len(b) == e["bytes"] and hashlib.sha256(b).hexdigest() == e["sha256"], name
    up = _upstream_root()
    if up is not None:  # on the mini: byte-identical to the fetched upstream files
        for name, folder in UPSTREAM.items():
            assert (chat.TEMPLATES_DIR / name).read_bytes() == (up / folder / "chat_template.jinja").read_bytes()


def test_eos_ids_and_sampling_check_from_generation_config(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": [127906, 127901], "temperature": 1.0, "top_p": 0.97, "top_k": 128}))
    assert chat.eos_ids(tmp_path) == [127906, 127901]
    assert chat.check_sampling("kolibri", tmp_path)["match"] is True
    assert chat.check_sampling("gemma4", tmp_path)["match"] is False
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    assert chat.eos_ids(tmp_path) == [1]
    assert chat.check_sampling("kolibri", tmp_path)["match"] is False


def test_prompt_prefill_classification():
    assert chat.prompt_prefill("kolibri", "x<|im_start|>assistant\n<think>\n\n</think>\n\n") == "empty_think"
    assert chat.prompt_prefill("qwen3_6", "x<|im_start|>assistant\n<think>\n") == "open_think"
    assert chat.prompt_prefill("kolibri", "x<|im_start|>assistant\n") == "none"
    assert chat.prompt_prefill("gemma4", "x<|turn>model\n<|channel>thought\n<channel|>") == "empty_channel"
    assert chat.prompt_prefill("gemma4", "x<|turn>model\n") == "none"


# ---------------------------------------------------------------- Kolibri

SENTENCES = {
    "none": "Reasoning is disabled. Proceed straight to answering according to the user's instructions.",
    "low": "Reasoning effort is set to low.",
    "medium": "Reasoning effort is set to medium.",
    "high": "Reasoning effort is set to high.",
}


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high"])
def test_kolibri_effort_sentences(kolibri_tok, effort):
    text, ids, sha = chat.render("kolibri", kolibri_tok, MSGS, effort)
    assert SENTENCES[effort] in text
    others = [s for e, s in SENTENCES.items() if e != effort]
    assert not any(s in text for s in others)
    if effort == "none":
        assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert ids[-4:] == [127907, 263, 127908, 263]
    else:
        # The pinned template does not prefill <think> at low / medium / high.
        assert text.endswith("<|im_start|>assistant\n")
    assert sha == hashlib.sha256(text.encode()).hexdigest()
    assert ids == kolibri_tok.encode(text, add_special_tokens=False)
    assert 127906 in ids  # <|im_end|> closes the system and user turns; no BOS (spec item 15)


def test_kolibri_explicit_high_equals_template_default(kolibri_tok):
    hf_default = kolibri_tok.apply_chat_template(MSGS, tokenize=False, add_generation_prompt=True)
    assert chat.render("kolibri", kolibri_tok, MSGS, "high")[0] == hf_default


# ------------------------------------------------------------------ peers


def test_gemma4_thinking_toggle(peer_root):
    tok = _tok(peer_root / PEER_DIRS["gemma4"])
    on, ids_on, _ = chat.render("gemma4", tok, MSGS, "high")
    off, _, _ = chat.render("gemma4", tok, MSGS, "none")
    assert on.startswith("<bos><|turn>system\n<|think|>")
    assert on.endswith("<|turn>model\n") and "<|channel>" not in on
    assert off.endswith("<|turn>model\n<|channel>thought\n<channel|>") and "<|think|>" not in off
    assert ids_on[0] == tok.bos_token_id and ids_on.count(tok.bos_token_id) == 1  # one BOS, from the template
    with_sys, _, _ = chat.render("gemma4", tok, MSGS_SYS, "high")
    assert with_sys.startswith("<bos><|turn>system\n<|think|>\nAnswer in one word.")
    par = chat.template_parity("gemma4", peer_root / PEER_DIRS["gemma4"])
    assert par["equal"] is True
    assert chat.eos_ids(peer_root / PEER_DIRS["gemma4"]) == [1, 106, 50]


@pytest.mark.parametrize("family", ["qwen3_6", "qwen3_8"])
def test_qwen_thinking_toggle(peer_root, family):
    tok = _tok(peer_root / PEER_DIRS[family])
    on, ids, _ = chat.render(family, tok, MSGS, "high")
    off, _, _ = chat.render(family, tok, MSGS, "none")
    assert on.endswith("<|im_start|>assistant\n<think>\n")
    assert off.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert chat.prompt_prefill(family, on) == "open_think"
    assert ids[-2:] == tok.encode("<think>\n", add_special_tokens=False)
    assert chat.template_parity(family, peer_root / PEER_DIRS[family])["equal"] is True
    assert chat.eos_ids(peer_root / PEER_DIRS[family]) == [248046, 248044]


def test_qwen38_explicit_xhigh_equals_template_default(peer_root):
    tok = _tok(peer_root / PEER_DIRS["qwen3_8"])
    explicit = chat.render("qwen3_8", tok, MSGS, "high")[0]
    default = tok.apply_chat_template(MSGS, chat_template=chat.template_text("qwen3_8"), tokenize=False,
                                      add_generation_prompt=True, enable_thinking=True)
    assert explicit == default
    assert "Reasoning effort is set to xhigh." in explicit


def test_mlx_lm_wrapper_default_is_not_used(peer_root):
    """The TokenizerWrapper would inject enable_thinking itself; render() goes
    through the HF tokenizer with the explicit kwarg, so a wrapper and a bare
    tokenizer give the same prompt."""
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tok = _tok(peer_root / PEER_DIRS["gemma4"])
    w = TokenizerWrapper(tok)
    assert chat.render("gemma4", w, MSGS, "none")[0] == chat.render("gemma4", tok, MSGS, "none")[0]
