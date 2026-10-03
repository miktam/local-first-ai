"""Chat template parity and tokenizer sanity (spec items 15 and 25a, 25g).

fixtures/ holds two copies of the Kolibri 1 chat template:

  kolibri1_chat_template.tokenizer_config.jinja  the "chat_template" string of
      the released tokenizer_config.json, written out verbatim
  kolibri1_chat_template.vendor.jinja            the vendor plugin's copy
      (aleph-alpha-inference tests/kolibri1_chat_template.jinja), which adds
      a leading {#- ... -#} comment

Both are rendered in a jinja2 sandbox configured the way transformers
configures it (ImmutableSandboxedEnvironment, trim_blocks, lstrip_blocks,
loopcontrols, and the tojson / raise_exception / strftime_now helpers) and
must give byte-identical prompts for every message set and template kwarg the
eval harness can send.

With a tokenizer directory available ($EXP036_TOKENIZER_DIR or
$EXP036_MODELS/Kolibri-1-BF16), the tokenizer checks also run: the
AutoTokenizer that mlx_lm uses must produce the same ids as the raw
tokenizers.Tokenizer. That settles the "fix_mistral_regex" warning
transformers prints for this tokenizer (spec item 25a): it is a false alarm
triggered by config.json having no transformers_version, and the default
load leaves the pre-tokenizer alone.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOKCFG_TEMPLATE = FIXTURES / "kolibri1_chat_template.tokenizer_config.jinja"
VENDOR_TEMPLATE = FIXTURES / "kolibri1_chat_template.vendor.jinja"

# The four system-message sentences, verbatim from the template.
SENTENCE_NONE = "Reasoning is disabled. Proceed straight to answering according to the user's instructions."
SENTENCE_LOW = (
    "Reasoning effort is set to low. Think briefly through only the essential steps "
    "in the user's language, then proceed directly to the answer."
)
SENTENCE_MEDIUM = (
    "Reasoning effort is set to medium. Think through the task methodically in the "
    "user's language, check key assumptions, and provide a well-supported answer."
)
SENTENCE_HIGH = (
    "Reasoning effort is set to high. Think carefully through the task in the user's "
    "language, validate key assumptions, consider plausible alternatives, and "
    "prioritize correctness and clarity."
)
EMPTY_THINK = "<think>\n\n</think>\n\n"

MESSAGE_SETS = {
    "single_user": [
        {"role": "user", "content": "What is 2+2?"},
    ],
    "system_user": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Summarise the plot of Hamlet in one sentence."},
    ],
    "multi_turn": [
        {"role": "user", "content": "Pick a prime between 10 and 20."},
        # Reasoning inside the content: the template splits it at </think> and
        # drops it for turns before the last user query.
        {"role": "assistant", "content": "<think>\n11, 13, 17 and 19 qualify.\n</think>\n\n13."},
        {"role": "user", "content": "And one between 20 and 30?"},
    ],
    "german": [
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Wie heißt die Hauptstadt von Österreich? Antworte kurz — danke! Grüße aus Köln, Straße 12."},
    ],
}

TEMPLATE_KWARGS = {
    "none": {},
    "thinking_off": {"enable_thinking": False},
    "thinking_on": {"enable_thinking": True},
    "effort_none": {"reasoning_effort": "none"},
    "effort_low": {"reasoning_effort": "low"},
    "effort_medium": {"reasoning_effort": "medium"},
    "effort_high": {"reasoning_effort": "high"},
    "thinking_off_effort_high": {"enable_thinking": False, "reasoning_effort": "high"},
}

# kwargs name -> (system sentence, generation prompt opens with an empty think block)
EXPECTED_MODE = {
    "none": (SENTENCE_HIGH, False),
    "thinking_off": (SENTENCE_NONE, True),
    "thinking_on": (SENTENCE_HIGH, False),
    "effort_none": (SENTENCE_NONE, True),
    "effort_low": (SENTENCE_LOW, False),
    "effort_medium": (SENTENCE_MEDIUM, False),
    "effort_high": (SENTENCE_HIGH, False),
    # reasoning_effort wins over enable_thinking.
    "thinking_off_effort_high": (SENTENCE_HIGH, False),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Wetter für eine Stadt abrufen",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "Stadt, z. B. Köln"}},
                "required": ["city"],
            },
        },
    }
]
TOOL_MESSAGES = [
    {"role": "user", "content": "Wie ist das Wetter in Köln?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Köln"}}}],
    },
    {"role": "tool", "content": '{"temp_c": 14, "sky": "bewölkt"}'},
]


def _make_env():
    """jinja2 environment configured as transformers 5.x configures it
    (transformers/utils/chat_template_utils.py, _cached_compile_jinja_template).
    The {% generation %} tag extension is left out: Kolibri's template does not
    use it."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # Not jinja's built-in tojson, which HTML-escapes.
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        return datetime.now().strftime(format)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


@pytest.fixture(scope="module")
def templates():
    env = _make_env()
    return {
        "tokenizer_config": env.from_string(TOKCFG_TEMPLATE.read_text(encoding="utf-8")),
        "vendor": env.from_string(VENDOR_TEMPLATE.read_text(encoding="utf-8")),
    }


def _render(template, messages, tools=None, **kwargs) -> str:
    # transformers passes messages, tools, documents and add_generation_prompt
    # explicitly, then the template kwargs.
    return template.render(messages=messages, tools=tools, documents=None, add_generation_prompt=True, **kwargs)


# --- the two copies are the same template -----------------------------------


def test_fixture_copies_differ_only_by_the_vendor_comment():
    tok = TOKCFG_TEMPLATE.read_text(encoding="utf-8")
    vendor = VENDOR_TEMPLATE.read_text(encoding="utf-8")
    assert vendor.startswith("{#-")
    comment_end = vendor.index("-#}") + len("-#}")
    assert vendor[comment_end:].lstrip("\n") == tok


@pytest.mark.parametrize("kw_name", list(TEMPLATE_KWARGS))
@pytest.mark.parametrize("msg_name", list(MESSAGE_SETS))
def test_render_parity(templates, msg_name, kw_name):
    messages = MESSAGE_SETS[msg_name]
    kwargs = TEMPLATE_KWARGS[kw_name]
    a = _render(templates["tokenizer_config"], messages, **kwargs)
    b = _render(templates["vendor"], messages, **kwargs)
    assert a.encode("utf-8") == b.encode("utf-8")

    sentence, empty_think = EXPECTED_MODE[kw_name]
    system_block = a.split("<|im_end|>\n", 1)[0]
    assert system_block.startswith("<|im_start|>system\n")
    assert system_block.endswith("# Reasoning effort\n\n" + sentence)
    if messages[0]["role"] == "system":
        assert system_block == "<|im_start|>system\n" + messages[0]["content"] + "\n\n# Reasoning effort\n\n" + sentence
    if empty_think:
        assert a.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
    else:
        assert a.endswith("<|im_start|>assistant\n")
    # Every user turn appears verbatim.
    for m in messages:
        if m["role"] == "user":
            assert "<|im_start|>user\n" + m["content"] + "<|im_end|>\n" in a


def test_render_parity_with_tools(templates):
    a = _render(templates["tokenizer_config"], TOOL_MESSAGES, tools=TOOLS)
    b = _render(templates["vendor"], TOOL_MESSAGES, tools=TOOLS)
    assert a.encode("utf-8") == b.encode("utf-8")
    # tojson must not escape non-ASCII (transformers' override, not jinja's built-in).
    assert '"description": "Stadt, z. B. Köln"' in a
    assert '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Köln"}}\n</tool_call>' in a
    assert "<|im_start|>user\n<tool_response>\n" in a


def test_pinned_prompts_none_and_high(templates):
    """Exact prompts for the two modes the eval uses most."""
    t = templates["tokenizer_config"]
    user = [{"role": "user", "content": "What is 2+2?"}]
    high = (
        "<|im_start|>system\n# Reasoning effort\n\n" + SENTENCE_HIGH + "<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    off = (
        "<|im_start|>system\n# Reasoning effort\n\n" + SENTENCE_NONE + "<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    # No kwargs means HIGH effort, not "thinking off".
    assert _render(t, user) == high
    assert _render(t, user, reasoning_effort="high") == high
    assert _render(t, user, reasoning_effort="none") == off
    assert _render(t, user, enable_thinking=False) == off


def test_multi_turn_drops_earlier_reasoning(templates):
    a = _render(templates["tokenizer_config"], MESSAGE_SETS["multi_turn"])
    assert "11, 13, 17 and 19 qualify." not in a
    assert "<|im_start|>assistant\n13.<|im_end|>\n" in a


def test_env_matches_transformers_renderer():
    """Our sandbox renders exactly as transformers' own renderer does."""
    cu = pytest.importorskip("transformers.utils.chat_template_utils")
    tmpl = TOKCFG_TEMPLATE.read_text(encoding="utf-8")
    ours = _make_env().from_string(tmpl)
    for msg_name, messages in MESSAGE_SETS.items():
        for kw_name, kwargs in TEMPLATE_KWARGS.items():
            rendered, _ = cu.render_jinja_template(
                conversations=[messages], chat_template=tmpl, add_generation_prompt=True, **kwargs
            )
            assert rendered[0] == _render(ours, messages, **kwargs), (msg_name, kw_name)
    rendered, _ = cu.render_jinja_template(
        conversations=[TOOL_MESSAGES], tools=TOOLS, chat_template=tmpl, add_generation_prompt=True
    )
    assert rendered[0] == _render(ours, TOOL_MESSAGES, tools=TOOLS)


# --- tokenizer (needs the real tokenizer files) -----------------------------

# EN/DE samples: digits (single-digit pre-tokenisation), contractions,
# umlauts, ß / ẞ, camelCase, punctuation, newlines and the chat specials.
TOKENIZER_SAMPLES = [
    "Hello, world! I can't believe it's 2026: 3.14159 and 1,000,000.",
    "They'll say we'd've known; you're right, I'm sure. DON'T STOP.",
    "Grüße aus Köln: Die Straße ist 12,5 km lang. Wir haben's geschafft!",
    "ÄÖÜ äöü ß ẞ — naïve café, Maßstab 1:25.000, Größe 42",
    "iPhone XMLHttpRequest lowerUPPER camelCaseWord",
    "Zeile eins\nZeile zwei\n\n  eingerückt\tTab",
    "<|im_start|>user\nWie geht's?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
]


@pytest.fixture(scope="module")
def tokenizers_pair(tokenizer_dir):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    raw = tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    # The same call mlx_lm.tokenizer_utils.load makes (no extra kwargs).
    auto = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    return raw, auto


@pytest.mark.parametrize("text", TOKENIZER_SAMPLES)
def test_autotokenizer_matches_raw_tokenizer(tokenizers_pair, text):
    raw, auto = tokenizers_pair
    want = raw.encode(text, add_special_tokens=False).ids
    got = auto.encode(text, add_special_tokens=False)
    assert got == want
    assert auto.decode(got) == text


def test_digits_are_single_tokens(tokenizers_pair):
    raw, auto = tokenizers_pair
    ids = auto.encode("1234567890", add_special_tokens=False)
    assert len(ids) == 10
    assert ids == raw.encode("1234567890", add_special_tokens=False).ids


def test_special_token_ids(tokenizers_pair):
    """Spec item 15."""
    _, auto = tokenizers_pair
    ids = {t: auto.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>")}
    assert ids == {"<|im_start|>": 127904, "<|im_end|>": 127906, "<|endoftext|>": 127901, "<think>": 127907, "</think>": 127908}
    assert auto.eos_token_id == 127906
    assert auto.pad_token_id == 127901
    assert auto.bos_token_id is None
    # No BOS is prepended by default (add_bos_token false).
    assert auto.encode("Hallo") == auto.encode("Hallo", add_special_tokens=False)


def test_tokenizer_chat_template_is_the_fixture(tokenizers_pair, templates):
    """The template the runtime will use is the fixture, and renders the same."""
    _, auto = tokenizers_pair
    assert auto.chat_template == TOKCFG_TEMPLATE.read_text(encoding="utf-8")
    for msg_name, messages in MESSAGE_SETS.items():
        for kw_name, kwargs in TEMPLATE_KWARGS.items():
            text = auto.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
            assert text == _render(templates["vendor"], messages, **kwargs), (msg_name, kw_name)
            ids = auto.encode(text, add_special_tokens=False)
            # Chat specials are single ids, never split into pieces.
            assert ids.count(127904) == text.count("<|im_start|>")
            assert ids.count(127906) == text.count("<|im_end|>")


def test_fix_mistral_regex_would_break_tokenisation(tokenizer_dir):
    """Following the warning's advice changes Kolibri's ids, so do not."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    raw = tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    fixed = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, fix_mistral_regex=True)
    text = "iPhone XMLHttpRequest lowerUPPER"
    assert fixed.encode(text, add_special_tokens=False) != raw.encode(text, add_special_tokens=False).ids
