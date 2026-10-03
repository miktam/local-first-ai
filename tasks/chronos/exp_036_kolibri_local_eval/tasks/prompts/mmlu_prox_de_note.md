# MMLU-ProX DE prompt (R row) and the other prompt files

## MMLU-ProX DE

eval-framework v0.14.2 has no MMLU-ProX task, so the vendor's German MMLU prompt is not public. HYPOTHESIS H2 labels this row R and fixes the protocol: "eval-framework German Tülu CoT prompt with A–J, German lenient extractor extended to A–J". `tasks/mmlu_prox.py` builds it from the vendored parts alone (`shim.mmlu_prox_de_kind()`):

- Kind: the vendored `Cot`, 0-shot, with no assistant cue. The reader is the vendored `MmluProReader`, which strips the question and keeps the options in dataset order without shuffling; the correct letter is the one at `answer_index`.
- Prompt: the vendored `tulu3_cot_prompt_de` of `gpqa_ellamind.py`. Its option letters come from `get_n_letters(len(options))`, so a 10-option question is labelled (A)–(J). The answer phrase stays German: "Daher ist die Antwort (ANTWORTBUCHSTABE)".
- No preamble. The German GPQA CoT has none, and translating MMLU-Pro's English subject line would mean writing our own text.
- Extractor (scorers/mc.py): `tulu_answer_de` with the letter class widened from A–D to A–J, and nothing else changed (BUILD_SPEC §5.6).

The English row (MMLU-Pro CoT EN on MMLU-ProX-Lite EN text) uses the eval-framework `MMLU_PRO_COT_V2` composition unchanged: the subject preamble, then `tulu3_cot_prompt`, with `tulu_answer_v2(10)`. MMLU-ProX's 14 categories carry the English MMLU-Pro names, which fill the preamble's `<category>`.

## `aime_de.txt`

A German translation of eval-framework's `_AIME_QUERY_TEMPLATE`, which in turn comes from NVIDIA NeMo-Skills `llama3-instruct/math.yaml` (Apache-2.0; named in NOTICE). The R row AIME 2026 DE uses it. It is filled with `str.format(Question=<problem>)` exactly as upstream fills the English template, so `{{answer}}` renders as `{answer}` and the model is asked for `$\boxed{answer}$`. The answer is extracted with the same boxed extractor as AIME EN.

Translation choices:
- the indentation, the line structure and `## Schritt N:` follow the English template;
- "I hope it is correct." becomes "Ich hoffe, die Antwort ist korrekt.", eval-framework's own German counterpart (`END_SEQ_DE` in `minerva_math_utils.py`);
- `answer` inside `\boxed{}` and `[answer]` stay unchanged, so the boxed format is the same in both languages.

The file has no trailing newline. The English template has none either, and the problem is the last thing in the prompt.

## `rgb_forced_en.txt`

Our own text. No RGB text is used. It is the forced-answer condition of HYPOTHESIS E2 and C22, filled with `str.format(QUERY=<question>)` and sent as the only user message, with no system message (as in the closed-book condition). It has no trailing newline. The sentence "Answer with a short phrase. Always give your best answer." is the wording pre-registered in E2.
