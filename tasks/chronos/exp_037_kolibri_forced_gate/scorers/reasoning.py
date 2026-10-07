"""Split a completion into its reasoning segment and its answer (BUILD_SPEC §4 item 28, §5.6).

Only the answer (the text after the reasoning segment) is ever scored. The split is a pure function of
the rendered prompt, the completion and the family's delimiter tokens, so it can be recomputed from the
stored record: from token ids (`split_reasoning`) or from the raw completion text with special tokens
kept (`split_reasoning_text`). Both apply the same rule and must agree (tests/test_scorers_reasoning.py).

The rule, per family (delimiters below):

1. Look at the rendered prompt after its last turn-start token (the generation prompt).
   - It ends inside an open reasoning block ("open": Qwen with thinking on, whose template prefills
     `<think>\\n`): the reasoning runs from the start of the completion to the first close delimiter.
   - It holds a closed block ("closed": Kolibri at effort none prefills `<think>\\n\\n</think>\\n\\n`;
     Qwen and Gemma with thinking off prefill an empty block): the whole completion is the answer,
     status "none".
   - It holds neither ("plain": Kolibri at effort low / medium / high, Gemma 4 with thinking on): if the
     completion's first non-whitespace token is the open delimiter, the reasoning runs to the first
     close delimiter; otherwise the whole completion is the answer, status "none".
2. A reasoning block with no close delimiter is "unclosed": the answer is empty. At the cap
   (finish_reason "length") that is a truncation and is wrong; ended by EOS it is a parse failure.
3. Trailing EOS tokens are not part of the answer or the reasoning.

Status "none" for Kolibri at effort != none, and for Qwen with thinking on, is recorded as a defect
(`reasoning_defect`); for Gemma 4 with thinking on it is legitimate (its template does not prefill the
thought channel) and the whole completion is scored (HYPOTHESIS, pilot rule 3).

Note on the pinned Kolibri template (tests/fixtures/kolibri1_chat_template.vendor.jinja): at effort
low / medium / high it does NOT prefill `<think>`; the generation prompt ends `<|im_start|>assistant\\n`
and the model opens the block itself. The "plain" branch handles that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

STATUSES = ("closed", "unclosed", "none")
PROMPT_STATES = ("open", "closed", "plain")


@dataclass(frozen=True)
class FamilySpec:
    """Delimiter tokens of one model family, with the ids they have in the pinned tokenizers."""

    name: str
    turn_start: str
    open: str
    close: str
    eos: tuple[str, ...]
    turn_start_id: int
    open_id: int
    close_id: int
    eos_ids: tuple[int, ...]


# Token strings and ids checked against the pinned tokenizers (Kolibri-1-BF16 @ 7a8f290e; mlx-community
# gemma-4-26b-a4b-it, Qwen3.6-35B-A3B and Qwen3.8-27B at their assets.json revisions; the upstream copies
# carry the same ids). EOS ids are each model's generation_config eos_token_id.
_QWEN = dict(
    turn_start="<|im_start|>", open="<think>", close="</think>", eos=("<|im_end|>", "<|endoftext|>"),
    turn_start_id=248045, open_id=248068, close_id=248069, eos_ids=(248046, 248044),
)
FAMILIES: dict[str, FamilySpec] = {
    "kolibri": FamilySpec(
        name="kolibri", turn_start="<|im_start|>", open="<think>", close="</think>",
        eos=("<|im_end|>", "<|endoftext|>"),
        turn_start_id=127904, open_id=127907, close_id=127908, eos_ids=(127906, 127901),
    ),
    "gemma4": FamilySpec(
        name="gemma4", turn_start="<|turn>", open="<|channel>", close="<channel|>",
        eos=("<eos>", "<turn|>", "<|tool_response>"),
        turn_start_id=105, open_id=100, close_id=101, eos_ids=(1, 106, 50),
    ),
    "qwen3_6": FamilySpec(name="qwen3_6", **_QWEN),
    "qwen3_8": FamilySpec(name="qwen3_8", **_QWEN),
}

_ALIASES = {
    "kolibri": "kolibri", "kolibri1": "kolibri", "kolibri-1": "kolibri",
    "gemma4": "gemma4", "gemma-4": "gemma4", "gemma": "gemma4",
    "qwen3_6": "qwen3_6", "qwen3.6": "qwen3_6", "qwen36": "qwen3_6",
    "qwen3_8": "qwen3_8", "qwen3.8": "qwen3_8", "qwen38": "qwen3_8",
}

# Arm prefix -> family (HYPOTHESIS "Arms"). runner/chat.py:family_of is the authority at run time; this
# table is the same mapping for the scorers, so scoring needs no runner import.
_ARM_FAMILY = (("K", "kolibri"), ("G", "gemma4"), ("Q36", "qwen3_6"), ("Q38", "qwen3_8"))


def family_spec(family: str) -> FamilySpec:
    key = _ALIASES.get(str(family).strip().lower())
    if key is None:
        raise ValueError(f"unknown model family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[key]


def family_of_arm(arm: str) -> str:
    """K8/K4 -> kolibri, G8/G4 -> gemma4, Q36-8/Q36-4 -> qwen3_6, Q38-8/Q38-4 -> qwen3_8."""
    a = str(arm).strip().upper()
    for prefix, fam in sorted(_ARM_FAMILY, key=lambda p: -len(p[0])):
        if a.startswith(prefix):
            return fam
    raise ValueError(f"unknown arm {arm!r}")


def expected_prompt_state(family: str, *, effort: str | None = None, thinking: bool | None = None) -> str:
    """The prompt state the pinned templates produce for an arm's settings.

    Kolibri: effort "none" prefills a closed block; every other effort leaves the prompt plain.
    Qwen3.6 / Qwen3.8: thinking on prefills an open `<think>`; off prefills a closed block.
    Gemma 4: thinking on leaves the prompt plain (the model may or may not open the channel); off
    prefills a closed empty channel.
    Checked against real renders in tests/test_scorers_reasoning.py.
    """
    fam = family_spec(family).name
    if fam == "kolibri":
        if effort is None:
            raise ValueError("Kolibri needs an explicit effort")
        return "closed" if str(effort) == "none" else "plain"
    if thinking is None:
        # Scored peer cells always run with thinking on (HYPOTHESIS "Fixed before any run").
        thinking = _thinking_from_effort(effort)
    if fam == "gemma4":
        return "plain" if thinking else "closed"
    return "open" if thinking else "closed"


def reasoning_defect(family: str, status: str, *, effort: str | None = None, thinking: bool | None = None) -> bool:
    """True where status "none" cannot be legitimate: Kolibri at effort != none, Qwen with thinking on."""
    if status != "none":
        return False
    fam = family_spec(family).name
    if fam == "kolibri":
        return effort is not None and str(effort) != "none"
    if fam in ("qwen3_6", "qwen3_8"):
        if thinking is None:
            thinking = _thinking_from_effort(effort)
        return bool(thinking)
    return False  # Gemma 4: "none" is legitimate with thinking on


def _thinking_from_effort(effort) -> bool:
    """A peer's thinking switch from the effort label of its cell: off only for none / off / false."""
    if effort is None:
        return True
    return str(effort).strip().lower() not in ("none", "off", "false", "0")


# ---------------------------------------------------------------------------------------------------
# Prompt state


# runner/chat.py:prompt_prefill labels, stored in every raw record as "prompt_prefill".
_PREFILL_STATE = {"open_think": "open", "empty_think": "closed", "empty_channel": "closed", "none": "plain"}


def prompt_state_from_prefill(prefill: str) -> str:
    """The prompt state for a runner `prompt_prefill` label (open_think | empty_think | empty_channel | none)."""
    if prefill not in _PREFILL_STATE:
        raise ValueError(f"unknown prompt_prefill {prefill!r}; known: {sorted(_PREFILL_STATE)}")
    return _PREFILL_STATE[prefill]


def prompt_state_ids(family: str, prompt_ids: Sequence[int]) -> str:
    spec = family_spec(family)
    ids = list(prompt_ids)
    start = _last_index(ids, spec.turn_start_id)
    tail = ids[start:] if start is not None else ids
    return _state(_last_index(tail, spec.open_id), _last_index(tail, spec.close_id))


def prompt_state_text(family: str, prompt_text: str) -> str:
    spec = family_spec(family)
    start = prompt_text.rfind(spec.turn_start)
    tail = prompt_text[start:] if start >= 0 else prompt_text
    o, c = tail.rfind(spec.open), tail.rfind(spec.close)
    return _state(o if o >= 0 else None, c if c >= 0 else None)


def _state(last_open: int | None, last_close: int | None) -> str:
    if last_open is not None and (last_close is None or last_close < last_open):
        return "open"
    if last_close is not None:
        return "closed"
    return "plain"


def _last_index(seq: Sequence[int], value: int) -> int | None:
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] == value:
            return i
    return None


# ---------------------------------------------------------------------------------------------------
# Split on token ids


def split_ids(
    family: str,
    prompt_state: str,
    completion_ids: Sequence[int],
    is_whitespace: Callable[[int], bool],
    eos_ids: Iterable[int] | None = None,
) -> tuple[list[int], list[int], str]:
    """(reasoning_ids, answer_ids, status) without the delimiter tokens and trailing EOS ids."""
    spec = family_spec(family)
    if prompt_state not in PROMPT_STATES:
        raise ValueError(f"prompt_state must be one of {PROMPT_STATES}, got {prompt_state!r}")
    eos = set(spec.eos_ids if eos_ids is None else eos_ids)
    body = list(completion_ids)
    while body and body[-1] in eos:
        body.pop()

    if prompt_state == "closed":
        return [], body, "none"
    if prompt_state == "open":
        start = 0
    else:
        i = 0
        while i < len(body) and body[i] != spec.open_id and is_whitespace(body[i]):
            i += 1
        if i >= len(body) or body[i] != spec.open_id:
            return [], body, "none"
        start = i + 1
    for j in range(start, len(body)):
        if body[j] == spec.close_id:
            return body[start:j], body[j + 1:], "closed"
    return body[start:], [], "unclosed"


def split_reasoning(
    family: str,
    rendered_prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    tokenizer,
    eos_ids: Iterable[int] | None = None,
) -> tuple[str, str, str]:
    """BUILD_SPEC §5.6: (reasoning, answer, status) from token ids.

    `tokenizer` is a `tokenizers.Tokenizer`, a transformers tokenizer or an mlx_lm TokenizerWrapper.
    Its delimiter ids are checked against the family table (a mismatch raises). Text is decoded with
    special tokens kept and no clean-up, so the result equals `split_reasoning_text` on the decoded
    completion.
    """
    spec = family_spec(family)
    check_tokenizer(spec.name, tokenizer)
    state = prompt_state_ids(spec.name, rendered_prompt_ids)
    r_ids, a_ids, status = split_ids(
        spec.name, state, completion_ids, lambda t: decode(tokenizer, [t]).strip() == "", eos_ids
    )
    return decode(tokenizer, r_ids), decode(tokenizer, a_ids), status


def split_reasoning_ids(
    family: str,
    rendered_prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    tokenizer,
    eos_ids: Iterable[int] | None = None,
) -> tuple[list[int], list[int], str]:
    """(reasoning_ids, answer_ids, status): the same split as `split_reasoning`, as token ids (for the
    runner's record fields reasoning_tokens / answer_tokens). Delimiters and trailing EOS are in neither."""
    spec = family_spec(family)
    check_tokenizer(spec.name, tokenizer)
    state = prompt_state_ids(spec.name, rendered_prompt_ids)
    return split_ids(spec.name, state, completion_ids, lambda t: decode(tokenizer, [t]).strip() == "", eos_ids)


def split_token_counts(
    family: str, rendered_prompt_ids: Sequence[int], completion_ids: Sequence[int], tokenizer,
    eos_ids: Iterable[int] | None = None,
) -> tuple[int, int, str]:
    """(reasoning_tokens, answer_tokens, status); delimiters and trailing EOS are in neither count."""
    spec = family_spec(family)
    check_tokenizer(spec.name, tokenizer)
    state = prompt_state_ids(spec.name, rendered_prompt_ids)
    r_ids, a_ids, status = split_ids(
        spec.name, state, completion_ids, lambda t: decode(tokenizer, [t]).strip() == "", eos_ids
    )
    return len(r_ids), len(a_ids), status


# ---------------------------------------------------------------------------------------------------
# Split on text (special tokens kept)


def split_text(family: str, prompt_state: str, completion_text: str) -> tuple[str, str, str]:
    """The same rule on the raw completion text with special tokens kept."""
    spec = family_spec(family)
    if prompt_state not in PROMPT_STATES:
        raise ValueError(f"prompt_state must be one of {PROMPT_STATES}, got {prompt_state!r}")
    body = strip_trailing_eos(spec.name, completion_text)
    if prompt_state == "closed":
        return "", body, "none"
    if prompt_state == "open":
        start = 0
    else:
        lead = len(body) - len(body.lstrip())
        if not body.startswith(spec.open, lead):
            return "", body, "none"
        start = lead + len(spec.open)
    j = body.find(spec.close, start)
    if j < 0:
        return body[start:], "", "unclosed"
    return body[start:j], body[j + len(spec.close):], "closed"


def split_reasoning_text(family: str, rendered_prompt_text: str, completion_text: str) -> tuple[str, str, str]:
    """(reasoning, answer, status) from the rendered prompt text and the raw completion text."""
    spec = family_spec(family)
    return split_text(spec.name, prompt_state_text(spec.name, rendered_prompt_text), completion_text)


def strip_trailing_eos(family: str, text: str) -> str:
    spec = family_spec(family)
    changed = True
    while changed:
        changed = False
        for e in spec.eos:
            if text.endswith(e):
                text = text[: -len(e)]
                changed = True
    return text


# ---------------------------------------------------------------------------------------------------
# Tokenizer helpers (duck-typed: tokenizers.Tokenizer, transformers, mlx_lm TokenizerWrapper)


def decode(tokenizer, ids: Sequence[int]) -> str:
    ids = [int(i) for i in ids]
    if not ids:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        # tokenizers.Tokenizer.decode has no clean-up argument (and never cleans up).
        return tokenizer.decode(ids, skip_special_tokens=False)


def token_id(tokenizer, token: str) -> int | None:
    for name in ("token_to_id", "convert_tokens_to_ids"):
        fn = getattr(tokenizer, name, None)
        if fn is not None:
            try:
                i = fn(token)
            except Exception:  # noqa: BLE001 - an unknown token is simply absent
                continue
            if isinstance(i, int):
                return i  # an unknown token mapped to <unk> simply fails the comparison in check_tokenizer
    return None


_CHECKED: set[tuple[str, int]] = set()


def check_tokenizer(family: str, tokenizer) -> None:
    """Raise if the tokenizer's delimiter ids differ from the family table (wrong family or tokenizer)."""
    spec = family_spec(family)
    key = (spec.name, id(tokenizer))
    if key in _CHECKED:
        return
    want = {spec.turn_start: spec.turn_start_id, spec.open: spec.open_id, spec.close: spec.close_id}
    want.update(zip(spec.eos, spec.eos_ids))
    got = {tok: token_id(tokenizer, tok) for tok in want}
    bad = {t: (got[t], want[t]) for t in want if got[t] != want[t]}
    if bad:
        raise ValueError(f"tokenizer does not match family {spec.name!r}: (found, expected) = {bad}")
    _CHECKED.add(key)
