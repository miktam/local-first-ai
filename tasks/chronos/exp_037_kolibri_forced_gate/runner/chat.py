"""Per-family chat rendering, thinking kwargs, sampling and EOS ids
(BUILD_SPEC §5.4 runner/chat.py; HYPOTHESIS "Fixed before any run",
Reasoning and Sampling; C9).

Every thinking/effort kwarg is passed explicitly. mlx_lm's TokenizerWrapper
injects enable_thinking=<has_thinking> when the caller does not pass it, which
turns Gemma 4's thinking on silently; this module therefore renders through
the underlying Hugging Face tokenizer with an explicit kwarg set.

Templates:
  kolibri  the chat_template in the model's tokenizer_config.json (gate G0
           checks it against the vendor jinja, 120/120 byte-identical).
  gemma4   runner/templates/gemma4.jinja  (google/gemma-4-26B-A4B-it @ 20da991a)
  qwen3_6  runner/templates/qwen3_6.jinja (Qwen/Qwen3.6-35B-A3B @ 995ad96e)
  qwen3_8  runner/templates/qwen3_8.jinja (Qwen/Qwen3.8-27B @ 1d4bf0f2)
The peer files are byte-for-byte upstream copies; templates/MANIFEST.json
holds their sha256 and provenance. render() returns whether the committed
template equals the local mlx-community one (template_parity()).

The prompt is tokenised with add_special_tokens=False, as vLLM does for chat
requests: the templates emit their own BOS (Gemma) or none (Kolibri, Qwen).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runner.common import ARMS, RUNNER_DIR, sha256_bytes, sha256_text

FAMILIES = ("kolibri", "gemma4", "qwen3_6", "qwen3_8")
TEMPLATES_DIR = RUNNER_DIR / "templates"
TEMPLATE_FILES = {"gemma4": "gemma4.jinja", "qwen3_6": "qwen3_6.jinja", "qwen3_8": "qwen3_8.jinja"}

# Each model's pinned generation_config (HYPOTHESIS Sampling): T / top-p / top-k.
SAMPLING = {
    "kolibri": {"temperature": 1.0, "top_p": 0.97, "top_k": 128},
    "gemma4": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "qwen3_6": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
    "qwen3_8": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
}

KOLIBRI_EFFORTS = ("none", "low", "medium", "high")
# Peers have one scored setting, thinking on ("high" in cell keys); "none" is
# thinking off (peer check flip rate, C1-style controls only).
PEER_EFFORTS = ("high", "none")
# Qwen3.8's template default effort when thinking is on (jinja line 47).
QWEN38_DEFAULT_EFFORT = "xhigh"


def family_of(arm: str) -> str:
    try:
        return ARMS[arm]["family"]
    except KeyError:
        raise ValueError(f"unknown arm {arm!r}; known: {sorted(ARMS)}") from None


def template_kwargs(family: str, effort: str) -> dict[str, Any]:
    """The explicit thinking / effort kwargs for one family and cell effort."""
    if family == "kolibri":
        if effort not in KOLIBRI_EFFORTS:
            raise ValueError(f"Kolibri effort must be one of {KOLIBRI_EFFORTS}, got {effort!r}")
        return {"reasoning_effort": effort}
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    if effort not in PEER_EFFORTS:
        raise ValueError(f"{family}: effort must be one of {PEER_EFFORTS} (thinking on/off), got {effort!r}")
    if effort == "none":
        return {"enable_thinking": False}
    if family == "qwen3_8":
        return {"enable_thinking": True, "reasoning_effort": QWEN38_DEFAULT_EFFORT}
    return {"enable_thinking": True}


def template_text(family: str) -> str | None:
    """Committed peer template, or None for Kolibri (its tokenizer's own)."""
    if family == "kolibri":
        return None
    return (TEMPLATES_DIR / TEMPLATE_FILES[family]).read_text(encoding="utf-8")


def template_sha256(family: str) -> str | None:
    if family == "kolibri":
        return None
    return sha256_bytes((TEMPLATES_DIR / TEMPLATE_FILES[family]).read_bytes())


def hf_tokenizer(tokenizer):
    """The Hugging Face tokenizer under an mlx_lm TokenizerWrapper; a bare
    transformers tokenizer is returned as is (its own `_tokenizer` is the raw
    tokenizers.Tokenizer, which has no chat template)."""
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        return inner
    return tokenizer


_hf = hf_tokenizer


def render(family: str, tokenizer, messages: list[dict], effort: str) -> tuple[str, list[int], str]:
    """Render one conversation with add_generation_prompt=True.

    Returns (text, ids, sha256 of the rendered text)."""
    hf = _hf(tokenizer)
    kwargs = template_kwargs(family, effort)
    text = hf.apply_chat_template(
        messages,
        chat_template=template_text(family),
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(text, str):
        raise TypeError(f"apply_chat_template returned {type(text).__name__}, expected str")
    ids = [int(i) for i in hf.encode(text, add_special_tokens=False)]
    return text, ids, sha256_text(text)


# The generation-prompt suffixes the templates can prefill (item 28).
PREFILL_SUFFIXES = {
    "kolibri": {"empty_think": "<think>\n\n</think>\n\n", "open_think": "<think>\n"},
    "qwen3_6": {"empty_think": "<think>\n\n</think>\n\n", "open_think": "<think>\n"},
    "qwen3_8": {"empty_think": "<think>\n\n</think>\n\n", "open_think": "<think>\n"},
    "gemma4": {"empty_channel": "<|channel>thought\n<channel|>"},
}


def prompt_prefill(family: str, rendered_text: str) -> str:
    """What the rendered prompt prefilled after the assistant header:
    empty_think | open_think | empty_channel | none. Stored in every record so
    the reasoning split can be recomputed without the (possibly withheld)
    prompt text."""
    for name, suffix in PREFILL_SUFFIXES[family].items():
        if rendered_text.endswith(suffix):
            return name
    return "none"


# ------------------------------------------------------------ model dir facts


def _generation_config(model_dir: Path) -> dict:
    p = Path(model_dir) / "generation_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def eos_ids(model_dir: Path) -> list[int]:
    """EOS ids from generation_config.json (falls back to config.json)."""
    gen = _generation_config(model_dir)
    eos = gen.get("eos_token_id")
    if eos is None:
        cfg_p = Path(model_dir) / "config.json"
        if cfg_p.is_file():
            cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
            eos = cfg.get("eos_token_id", cfg.get("text_config", {}).get("eos_token_id"))
    if eos is None:
        raise ValueError(f"no eos_token_id in {model_dir}/generation_config.json or config.json")
    return [int(e) for e in (eos if isinstance(eos, list) else [eos])]


def check_sampling(family: str, model_dir: Path) -> dict:
    """Compare the frozen SAMPLING with the model's generation_config.json.

    Returns {"frozen": …, "generation_config": …, "match": bool}; the runner
    refuses a cell when match is False (the pre-registered values are the
    pinned generation_config values)."""
    gen = _generation_config(model_dir)
    found = {k: gen.get(k) for k in ("temperature", "top_p", "top_k")}
    frozen = SAMPLING[family]
    match = all(found[k] is not None and float(found[k]) == float(frozen[k]) for k in frozen)
    return {"frozen": dict(frozen), "generation_config": found, "match": match}


def local_template_bytes(model_dir: Path) -> bytes | None:
    """The template a model directory ships: chat_template.jinja, else the
    tokenizer_config.json chat_template string."""
    p = Path(model_dir) / "chat_template.jinja"
    if p.is_file():
        return p.read_bytes()
    tc = Path(model_dir) / "tokenizer_config.json"
    if tc.is_file():
        t = json.loads(tc.read_text(encoding="utf-8")).get("chat_template")
        if isinstance(t, str):
            return t.encode("utf-8")
    return None


def template_parity(family: str, model_dir: Path) -> dict:
    """Does the committed template equal the model directory's, byte for byte?"""
    if family == "kolibri":
        return {"family": family, "committed": None, "local_sha256": None, "equal": None,
                "note": "Kolibri renders with its tokenizer_config template (gate G0 parity)"}
    committed = (TEMPLATES_DIR / TEMPLATE_FILES[family]).read_bytes()
    local = local_template_bytes(model_dir)
    return {
        "family": family,
        "committed": TEMPLATE_FILES[family],
        "committed_sha256": sha256_bytes(committed),
        "local_sha256": sha256_bytes(local) if local is not None else None,
        "equal": local == committed if local is not None else None,
    }


def sampling_for(family: str) -> dict:
    return {"order": "vllm", **SAMPLING[family]}
