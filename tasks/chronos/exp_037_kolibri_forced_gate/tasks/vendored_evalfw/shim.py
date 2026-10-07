"""Load the vendored eval-framework v0.14.2 source files and expose what the kit uses (BUILD_SPEC §5.5).

The vendored files import the eval-framework package and its dependencies (pydantic-based
template_formatting, sympy, dill, llm_sandbox, the metric classes, dataset loading). None of that is
installed in the kit's environment and none of it is needed to build a prompt or extract an answer. This
shim therefore re-implements nothing: it executes the vendored source text as it is, with

- every `eval_framework.*` module that is vendored here resolved to the vendored file (loaded once, under a
  private name in sys.modules, so an installed eval_framework could never be picked up by accident);
- every other `eval_framework.*`, `template_formatting.*`, `sympy*`, `dill` and `llm_sandbox*` import
  resolved to an inert stub module.

A stub stands in for a class or function the kit never relies on (metrics, ComposedBenchmark, dataset
policies, Message/Role, ...). Stubs are classes, so they can be subclassed, unioned in annotations and
called. Calling a stub returns an inert instance that keeps the call's keyword arguments as attributes,
and attribute access on a stub returns another stub, memoised per name so `Language.DEU` is the same
object each time. That is how the upstream benchmark factories (e.g. `gpqa_diamond_cot_v2()`) can be
called unchanged: they return a stub record whose `.kind` and `.answer` are the real vendored `Cot` /
`Generative` kind and `ExtractFromCompletion` answer policy, composed exactly as upstream composes them.

Everything is loaded lazily on first use; importing this module does no work.
"""

from __future__ import annotations

import builtins
import functools
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
_PREFIX = "_exp036_vendored_evalfw."

# Module name -> vendored file (relative to SRC).
VENDORED_MODULES = {
    "eval_framework.answer": "eval_framework/answer.py",
    "eval_framework.choices": "eval_framework/choices.py",
    "eval_framework.eval_kind": "eval_framework/eval_kind.py",
    "eval_framework.benchmarks.cot": "eval_framework/benchmarks/cot.py",
    "eval_framework.benchmarks.gpqa": "eval_framework/benchmarks/gpqa.py",
    "eval_framework.benchmarks.gpqa_ellamind": "eval_framework/benchmarks/gpqa_ellamind.py",
    "eval_framework.benchmarks.mmlu_pro": "eval_framework/benchmarks/mmlu_pro.py",
    "eval_framework.benchmarks.math_reasoning": "eval_framework/benchmarks/math_reasoning.py",
    "eval_framework.tasks.task_style": "eval_framework/tasks/task_style.py",
    "eval_framework.tasks.utils": "eval_framework/tasks/utils.py",
    "eval_framework.metrics.completion.minerva_math_utils": "eval_framework/metrics/completion/minerva_math_utils.py",
}
# Top-level packages whose non-vendored modules are stubbed.
STUB_ROOTS = ("eval_framework", "template_formatting", "sympy", "dill", "llm_sandbox")


# --------------------------------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------------------------------


class _StubMeta(type):
    """Metaclass of stub classes: unknown attributes are child stubs, memoised per name."""

    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        children = cls.__dict__.get("_stub_children")
        if children is None:  # a real class deriving from a stub: behave normally
            raise AttributeError(name)
        if name not in children:
            children[name] = _make_stub(f"{cls.__name__}.{name}")
        return children[name]


class _StubBase(metaclass=_StubMeta):
    _stub_children: dict = {}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        object.__setattr__(self, "_stub_args", args)
        object.__setattr__(self, "_stub_kwargs", dict(kwargs))
        object.__setattr__(self, "_stub_attr_children", {})
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        children = object.__getattribute__(self, "_stub_attr_children")
        if name not in children:
            children[name] = _make_stub(f"{type(self).__name__}().{name}")
        return children[name]

    def __repr__(self) -> str:
        return f"<stub {type(self).__name__}>"


def _make_stub(name: str) -> type:
    return _StubMeta(name, (_StubBase,), {"_stub_children": {}, "__module__": "stub"})


class _StubModule(types.ModuleType):
    """A module whose every attribute is a (memoised) stub class."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.__dict__["_stub_members"] = {}

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        members = self.__dict__["_stub_members"]
        if name not in members:
            members[name] = _make_stub(f"{self.__name__}.{name}")
        return members[name]


def is_stub(obj: Any) -> bool:
    return isinstance(obj, _StubBase) or (isinstance(obj, type) and issubclass(obj, _StubBase))


# --------------------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------------------


class _Loader:
    def __init__(self) -> None:
        self.modules: dict[str, types.ModuleType] = {}
        self.stubs: dict[str, _StubModule] = {}
        self.builtins = dict(vars(builtins))
        self.builtins["__import__"] = self._import

    def stub_module(self, name: str) -> _StubModule:
        if name not in self.stubs:
            self.stubs[name] = _StubModule(_PREFIX + "stub." + name)
        return self.stubs[name]

    def load(self, name: str) -> types.ModuleType:
        if name in self.modules:
            return self.modules[name]
        rel = VENDORED_MODULES[name]
        path = SRC / rel
        source = path.read_text(encoding="utf-8")
        module = types.ModuleType(_PREFIX + name)
        module.__file__ = str(path)
        module.__dict__["__builtins__"] = self.builtins
        # Register before executing, as the import system does (also lets dataclasses find the module).
        self.modules[name] = module
        sys.modules[_PREFIX + name] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException:
            del self.modules[name]
            sys.modules.pop(_PREFIX + name, None)
            raise
        return module

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in VENDORED_MODULES:
            if not fromlist:
                raise ImportError(f"vendored module {name} must be imported with 'from … import …'")
            return self.load(name)
        if level == 0 and name.split(".")[0] in STUB_ROOTS:
            return self.stub_module(name if fromlist else name.split(".")[0])
        return builtins.__import__(name, globals, locals, fromlist, level)


@functools.lru_cache(maxsize=1)
def _loader() -> _Loader:
    return _Loader()


def module(name: str) -> types.ModuleType:
    """The vendored module `eval_framework.<…>` (e.g. "eval_framework.benchmarks.gpqa")."""
    return _loader().load(name)


# --------------------------------------------------------------------------------------------------
# What the kit uses (BUILD_SPEC §5.5 list, plus the composed upstream benchmark kinds)
# --------------------------------------------------------------------------------------------------


def _cot():
    return module("eval_framework.benchmarks.cot")


def tulu3_cot_prompt(raw_question: str, choices: list[str]) -> str:
    return _cot().tulu3_cot_prompt(raw_question, choices)


def tulu3_cot_prompt_de(raw_question: str, choices: list[str]) -> str:
    return module("eval_framework.benchmarks.gpqa_ellamind").tulu3_cot_prompt_de(raw_question, choices)


def tulu_answer_v2(n_options: int):
    """The vendored ExtractFromCompletion policy (lenient English extractor, last match wins)."""
    return _cot().tulu_answer_v2(n_options)


def tulu_answer_de():
    """The vendored German GPQA extractor policy (A–D)."""
    return module("eval_framework.benchmarks.gpqa_ellamind").tulu_answer_de()


def extract(policy, completion_text: str) -> str:
    """Apply a vendored answer policy to a completion: the extracted answer or "[invalid]"/"[no_answer]"."""
    return policy.extract_answer(completion_text, context=None, ground_truth=None, messages=[])


def gpqa_reader():
    """eval-framework's English GPQA reader (option shuffle seeded by the option texts)."""
    return module("eval_framework.benchmarks.gpqa").GpqaReader()


def gpqa_ellamind_reader():
    """eval-framework's German GPQA reader (shuffle_correct_with_distractors, seed question + answer)."""
    return module("eval_framework.benchmarks.gpqa_ellamind").GpqaReader()


def shuffle_correct_with_distractors(correct: str, distractors: list[str], seed_text: str):
    return module("eval_framework.tasks.task_style").shuffle_correct_with_distractors(correct, distractors, seed_text)


def get_n_letters(n: int) -> list[str]:
    return module("eval_framework.tasks.utils").get_n_letters(n)


def aime_query_template() -> str:
    """`_AIME_QUERY_TEMPLATE` (NeMo-Skills llama3-instruct math wrapper), unformatted."""
    return module("eval_framework.benchmarks.math_reasoning")._AIME_QUERY_TEMPLATE


def extract_boxed(text: str) -> str | None:
    return module("eval_framework.benchmarks.math_reasoning")._extract_boxed(text)


def strip_string_with_bug(text: str) -> str:
    return module("eval_framework.benchmarks.math_reasoning")._strip_string_with_bug(text)


def overlong_question_sha256() -> str:
    """The patched constant of the vendored gpqa.py (sha256 of the excluded question, UTF-8)."""
    return module("eval_framework.benchmarks.gpqa")._OVERLONG_QUESTION_SHA256


@functools.lru_cache(maxsize=None)
def gpqa_overlong_filter():
    """The vendored gpqa.py dataset filter (row -> keep?), taken from the GPQA_DIAMOND_COT_V2 benchmark."""
    bench = module("eval_framework.benchmarks.gpqa").gpqa_diamond_cot_v2()
    subset_call = bench.dataset_policy  # pinned_by_framework(...).subset(<filter>, description=...)
    return subset_call._stub_args[0]


def mmlu_pro_subjects() -> list[str]:
    return list(module("eval_framework.benchmarks.mmlu_pro").MMLU_PRO_SUBJECTS)


# Upstream benchmark compositions. Each returns (kind, answer policy, upstream id).


@functools.lru_cache(maxsize=None)
def gpqa_diamond_cot_v2():
    b = module("eval_framework.benchmarks.gpqa").gpqa_diamond_cot_v2()
    return b.kind, b.answer, b.id


@functools.lru_cache(maxsize=None)
def gpqa_ellamind_diamond_cot_de():
    b = module("eval_framework.benchmarks.gpqa_ellamind").gpqa_ellamind_diamond_cot_de()
    return b.kind, b.answer, b.id


@functools.lru_cache(maxsize=None)
def mmlu_pro_cot_v2():
    b = module("eval_framework.benchmarks.mmlu_pro").mmlu_pro_cot_v2()
    return b.kind, b.answer, b.id


@functools.lru_cache(maxsize=None)
def aime2026():
    b = module("eval_framework.benchmarks.math_reasoning").aime2026()
    return b.kind, b.answer, b.id


@functools.lru_cache(maxsize=None)
def mmlu_prox_de_kind():
    """Our composition for MMLU-ProX DE (an R row, see tasks/prompts/mmlu_prox_de_note.md): the vendored
    Cot kind with the vendored MmluProReader and the vendored German Tülu prompt, no preamble."""
    cot = _cot()
    reader = module("eval_framework.benchmarks.mmlu_pro").MmluProReader()
    return cot.Cot(reader, build_prompt=tulu3_cot_prompt_de)


@dataclass(frozen=True)
class Rendered:
    messages: list[dict]  # [{"role": "system"|"user"|"assistant", "content": str}, ...]
    ground_truth: Any  # what upstream scores against (a letter for the CoT kinds)
    n_choices: int | None


def _role_name(role) -> str:
    formatter = _loader().stub_module("template_formatting.formatter")
    for name in ("SYSTEM", "USER", "ASSISTANT"):
        if role is getattr(formatter.Role, name):
            return name.lower()
    raise ValueError(f"unknown message role {role!r}")


def to_dicts(messages) -> list[dict]:
    """Upstream Message records (stub instances keeping role/content) -> plain dicts."""
    return [{"role": _role_name(m.role), "content": m.content} for m in messages]


def render(kind, item: dict, subject_label: str = "no_subject") -> Rendered:
    """Exactly upstream's prompt path for one item, 0-shot: kind.samples(item) -> kind.messages(...)."""
    samples = kind.samples(item)
    if len(samples) != 1:
        raise ValueError(f"expected one sample per item, got {len(samples)}")
    body = samples[0]
    messages = to_dicts(kind.messages(body, fewshot=[], subject_label=subject_label))
    reader = getattr(kind, "_reader", None)
    n_choices = len(reader.read(item).choices) if reader is not None else None
    return Rendered(messages=messages, ground_truth=body.ground_truth, n_choices=n_choices)
