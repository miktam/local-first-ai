"""exp_036 scorers (BUILD_SPEC §5.6).

Deterministic, pure functions from a raw completion (and gold) to a score. No model, no network, no
cloud: every verdict is code (deterministic-glue invariants, BUILD_SPEC §2).

- reasoning.py        split a completion into reasoning and answer, per model family (spec item 28)
- mc.py               GPQA / MMLU letter extraction with the vendored eval-framework regexes
- aime.py             AIME boxed-integer extraction and integer compare
- rgb.py              our implementation of RGB's answer rule, plus the abstain / reject / fact-check splits
- lang_tag.py         frozen stopword language tagger (gate behaviour check, E4)
- abstain_lexicon.json  frozen phrase lists used by rgb.py
- ifbench_adapter.py  runs allenai/IFBench's own checkers in its own venv (subprocess, offline)
- ifbench_driver.py   the script that adapter runs inside the IFBench venv
- score_all.py        raw JSONL -> results/scores/<arm>/<task>_<effort>.jsonl
- golden/             hand-written golden fixtures (synthetic text only; no withheld or RGB text)

Every scorer scores only the text after the reasoning segment. Truncated, unclosed and unparseable
outputs are wrong and are never re-asked (HYPOTHESIS "Common rules").
"""
