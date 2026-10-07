# SPDX-License-Identifier: MIT
"""Rendering of the vendor chat template, independently of transformers.

The vendor jinja (tests/fixtures/kolibri1_chat_template.vendor.jinja, the
aleph-alpha-inference copy) is rendered in a jinja2 sandbox configured as
transformers 5.x configures it (ImmutableSandboxedEnvironment, trim_blocks,
lstrip_blocks, loopcontrols, a non-escaping tojson, raise_exception,
strftime_now). tests/test_chat_template.py::test_env_matches_transformers_renderer
checks that this environment renders exactly as transformers does.

Used by gate/build_gate_text.py (T7, T8 and the G5 prompts) and by the G0
template-parity check, which compares this render with the runtime path
(mlx_lm's TokenizerWrapper.apply_chat_template).
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from gate.common import VENDOR_JINJA


def make_env():
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # transformers' override; jinja's built-in tojson HTML-escapes.
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


@lru_cache(maxsize=4)
def _template(path: str):
    return make_env().from_string(Path(path).read_text(encoding="utf-8"))


def render(messages, tools=None, add_generation_prompt: bool = True, template_path: Path = VENDOR_JINJA, **kwargs) -> str:
    """The vendor template's render. kwargs are the template kwargs exactly as
    given (reasoning_effort, enable_thinking); nothing is injected."""
    return _template(str(template_path)).render(
        messages=messages, tools=tools, documents=None, add_generation_prompt=add_generation_prompt, **kwargs
    )
