# Vendored eval-framework v0.14.2

- **Source:** <https://github.com/Aleph-Alpha-Research/eval-framework>, tag `v0.14.2`, commit `3532e9f16e19e396fc2b4d4543878cdccc4d04da`.
- **Licence:** Apache-2.0, © 2025 Aleph Alpha Research GmbH. The upstream `LICENSE` is copied here unchanged. eval-framework v0.14.2 ships no NOTICE file, so there is none to reproduce. Each file is named in the experiment's `NOTICE`.
- **Why it is here:** HYPOTHESIS "Sources & Rights" and C10. The V rows (GPQA Diamond EN/DE, AIME 2026 EN) use eval-framework's prompts, option shuffles and answer extractors verbatim, and the R rows (MMLU-ProX EN/DE) reuse them.

## Files

Every file keeps its upstream relative path. All but one are byte-for-byte copies. `MANIFEST.json` records each file's sha256, its upstream sha256 and whether it was modified.

| File | What the kit uses from it |
|---|---|
| `LICENSE` | — |
| `src/eval_framework/benchmarks/cot.py` | `tulu3_cot_prompt`, `tulu_answer_v2`, `Cot` |
| `src/eval_framework/benchmarks/gpqa.py` (**modified**) | `GpqaReader` (EN option shuffle), `gpqa_diamond_cot_v2`, the overlong-question filter |
| `src/eval_framework/benchmarks/gpqa_ellamind.py` | `tulu3_cot_prompt_de`, `tulu_answer_de`, `GpqaReader` (DE), `gpqa_ellamind_diamond_cot_de` |
| `src/eval_framework/benchmarks/mmlu_pro.py` | `mmlu_pro_cot_v2` (preamble and prompt), `MmluProReader`, `MMLU_PRO_SUBJECTS` |
| `src/eval_framework/benchmarks/math_reasoning.py` | `_AIME_QUERY_TEMPLATE`, `aime2026`, `_extract_boxed`, `_strip_string_with_bug` |
| `src/eval_framework/answer.py` | `ExtractFromCompletion`, `last_match`, `first_match` |
| `src/eval_framework/choices.py` | `ChoiceFields`, `ChoiceReader` |
| `src/eval_framework/eval_kind.py` | `assemble_messages` (how the MMLU-Pro preamble is folded into the user turn), `SampleBody`, `EvalKind` |
| `src/eval_framework/tasks/task_style.py` | `shuffle_correct_with_distractors` |
| `src/eval_framework/tasks/utils.py` | `get_n_letters` |
| `src/eval_framework/metrics/completion/minerva_math_utils.py` | the `_fix_*` and `_remove_right_units` helpers called by `_strip_string_with_bug` |

`math_reasoning.py` itself credits two upstream sources in its comments: OpenAI simple-evals (MIT) for `_QUERY_TEMPLATE`, which the kit does not use, and NVIDIA NeMo-Skills (Apache-2.0) for `_AIME_QUERY_TEMPLATE`. Our German AIME wrapper, `tasks/prompts/aime_de.txt`, translates the latter.

## The one modification (Apache-2.0 §4(b))

Upstream `gpqa.py` holds the full text of one GPQA question as the string literal `_OVERLONG_QUESTION`, and its dataset filter drops the row whose `Question` equals it. GPQA's terms forbid revealing examples online, so the text cannot enter this public repository. `vendor.py` applies this patch in memory:

1. The `_OVERLONG_QUESTION = (…)` assignment becomes `_OVERLONG_QUESTION_SHA256 = "04e898b3dfc33cd195ea9efa91d1ffa6c3faf1fcdc3d751f16fe9c16b9150241"`, the sha256 of the UTF-8 bytes of that string. The string has no leading or trailing whitespace, so this is also the sha256 of the stripped question.
2. The filter becomes `hashlib.sha256(row["Question"].encode()).hexdigest() != _OVERLONG_QUESTION_SHA256`. Upstream compared the raw `row["Question"]` for equality, so the predicate is the same.
3. Four comment lines go at the top: the SPDX identifier, the upstream copyright and the "Modified by Miktam for Chronos exp_036" note.

No other byte changes. Upstream sha256 `0b7e50fe08716fa7ac9e9f045307838a22807095a1ec929a21558977469f5a5a`; the patched sha256 is in `MANIFEST.json`. The question text was never written to disk, and the upstream file is never staged.

## Reproduce or check

```bash
git clone https://github.com/Aleph-Alpha-Research/eval-framework EF && git -C EF checkout v0.14.2
"$PY" tasks/vendored_evalfw/vendor.py --upstream EF --check     # exit 0: every byte and MANIFEST.json re-derived
```

## How the kit runs it

The originals import the eval-framework package (pydantic, sympy, dill, llm_sandbox, …), and none of that is installed here. `shim.py` executes the vendored source text unchanged:
- vendored modules resolve to each other;
- every other eval-framework, template_formatting, sympy, dill and llm_sandbox import resolves to an inert stub.

The shim re-implements nothing. The upstream benchmark factories, e.g. `gpqa_diamond_cot_v2()`, are called as they are, and the kit takes their composed `kind` and `answer` policy from the result (see the `shim.py` docstring).
