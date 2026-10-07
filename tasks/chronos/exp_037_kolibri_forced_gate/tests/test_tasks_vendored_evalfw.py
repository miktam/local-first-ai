"""tasks/vendored_evalfw: manifest, the gpqa.py patch, the shim, and known-good renders (BUILD_SPEC §5.5).

Synthetic items only. The expected prompts are written out by hand from the upstream templates; the option
orders are computed by an independent transcription of eval-framework's documented shuffles.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import pytest

import tasks_synthetic as syn

EXP = Path(__file__).resolve().parent.parent
VEND = EXP / "tasks" / "vendored_evalfw"
GPQA_PY = VEND / "src" / "eval_framework" / "benchmarks" / "gpqa.py"

from tasks.vendored_evalfw import shim  # noqa: E402

sys.path.insert(0, str(VEND))
import vendor  # noqa: E402

# String constants of >= 8 words allowed in the patched gpqa.py (upstream docstrings and the GPQA preamble).
GPQA_LONG_STRING_ALLOWLIST = {
    "f0fd040115baf5a26a842bcf0c6021992575d42ebb68625ade97e25f89c60a90",  # module docstring
    "ffb71eba6916f55770111e42d83c7a5b29e51af5c5487bc74dc46188120479a9",  # _gpqa_preamble text
    "8d7b8eceac4526922d763f7269305b0554b158265e6a24d3a069589e2d01bbbd",  # _preprocess docstring
    "3c2e159d2966ff38b8a526dc666aa5e0efd36fb79c381149458a48323d083162",  # GpqaReader docstring
}

TULU_EN = (
    "Answer the following multiple-choice question by giving the correct answer letter in parentheses. "
    "Provide CONCISE reasoning for the answer, and make sure to finish the response with "
    '"Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
    "\n\nQuestion: {q}\n{opts}\n\n"
    'Answer the above question and REMEMBER to finish your response with the exact phrase "Therefore, the '
    'answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D), (E), etc.'
)
TULU_DE = (
    "Beantworte die folgende Multiple-Choice-Frage, indem du den Buchstaben der richtigen Antwort in Klammern "
    'angibst. Begründe deine Antwort KURZ und beende deine Antwort unbedingt mit "Daher ist die Antwort '
    '(ANTWORTBUCHSTABE)", wobei (ANTWORTBUCHSTABE) einer von (A), (B), (C), (D), (E) usw. ist.'
    "\n\nFrage: {q}\n{opts}\n\n"
    'Beantworte die obige Frage und DENKE DARAN, deine Antwort mit genau dem Satz "Daher ist die Antwort '
    '(ANTWORTBUCHSTABE)" abzuschließen, wobei (ANTWORTBUCHSTABE) einer von (A), (B), (C), (D), (E) usw. ist.'
)
AIME_EN = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "    - For simple problems (2 steps or fewer):\n"
    "    Provide a concise solution with minimal explanation.\n\n"
    "    - For complex problems (3 steps or more):\n"
    "    Use this step-by-step format:\n\n"
    "    ## Step 1: [Concise description]\n"
    "    [Brief explanation and calculations]\n\n"
    "    ## Step 2: [Concise description]\n"
    "    [Brief explanation and calculations]\n\n"
    "    ...\n\n"
    "    Regardless of the approach, always conclude with:\n\n"
    "    Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "    Where [answer] is just the final number or expression that solves the problem.\n\n"
    "    Problem: {p}"
)


def opts(choices):
    return "\n".join(f"({'ABCDEFGHIJ'[i]}) {c}" for i, c in enumerate(choices))


def oracle_gpqa_en(row):
    """Independent transcription of eval-framework's GpqaReader.read (preprocess, sha-seeded shuffle)."""
    def prep(t):
        t = t.strip().replace(" [title]", ". ")
        return re.sub(r"\[.*?\]", "", t).replace("  ", " ")

    choices = [prep(row[f"Incorrect Answer {i}"]) for i in (1, 2, 3)]
    correct = prep(row["Correct Answer"])
    rng = random.Random(int(hashlib.sha256(f"{choices} {correct}".encode()).hexdigest(), 16))
    rng.shuffle(choices)
    ci = rng.randint(0, 3)
    choices.insert(ci, correct)
    return row["Question"].strip(), choices, "ABCD"[ci]


def oracle_shuffle_de(correct, distractors, seed_text):
    choices = [*distractors, correct]
    rng = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest(), 16))
    order = list(range(len(choices)))
    rng.shuffle(order)
    return [choices[i] for i in order], order.index(len(choices) - 1)


# --------------------------------------------------------------------------------------------------


def test_manifest_matches_files():
    assert vendor.check_local() == []
    manifest = json.loads((VEND / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source"]["tag"] == "v0.14.2"
    assert manifest["source"]["commit"].startswith("3532e9f1")
    gp = manifest["files"]["src/eval_framework/benchmarks/gpqa.py"]
    assert gp["modified"] is True and gp["sha256"] != gp["upstream_sha256"]
    assert gp["upstream_sha256"] == "0b7e50fe08716fa7ac9e9f045307838a22807095a1ec929a21558977469f5a5a"
    assert sum(1 for e in manifest["files"].values() if e["modified"]) == 1
    assert manifest["overlong_question_sha256"] == shim.overlong_question_sha256()


def test_gpqa_patch_removed_the_question_text():
    src = GPQA_PY.read_text(encoding="utf-8")
    assert "_OVERLONG_QUESTION = (" not in src
    assert re.search(r"_OVERLONG_QUESTION(?!_SHA256)", src.split('"""', 1)[1]) is None
    assert re.search(r'^_OVERLONG_QUESTION_SHA256 = "[0-9a-f]{64}"$', src, re.M)
    assert "Modified by Miktam for Chronos exp_036" in src and src.startswith("# SPDX-License-Identifier: Apache-2.0\n")
    long_strings = [n.value for n in ast.walk(ast.parse(src))
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value.split()) >= 8]
    unexpected = [s[:40] for s in long_strings if hashlib.sha256(s.encode()).hexdigest() not in GPQA_LONG_STRING_ALLOWLIST]
    assert unexpected == []


def test_patched_filter_compares_hashes():
    keep = shim.gpqa_overlong_filter()
    assert keep({"Question": "Synthetic question that is not the excluded one?"}) is True
    # The filter is a hash comparison: a row whose question hashes to the constant is dropped.
    src = GPQA_PY.read_text(encoding="utf-8")
    assert 'hashlib.sha256(row["Question"].encode()).hexdigest() != _OVERLONG_QUESTION_SHA256' in src


def _upstream_dir() -> Path | None:
    cands = [os.environ.get("EXP036_EVALFW_SRC"), str(Path.home() / "models/exp036-mini/evalfw")]
    for c in cands:
        if c and (Path(c).expanduser() / vendor.PATCHED_FILE).is_file():
            return Path(c).expanduser()
    return None


def test_rederived_from_upstream_byte_for_byte():
    up = _upstream_dir()
    if up is None:
        pytest.skip("build-host only: no eval-framework v0.14.2 checkout (set EXP036_EVALFW_SRC); the local manifest check still ran")
    assert vendor.check_against_upstream(up) == []
    value, _, _ = vendor.overlong_literal((up / vendor.PATCHED_FILE).read_text(encoding="utf-8"))
    assert hashlib.sha256(value.encode()).hexdigest() == shim.overlong_question_sha256()
    assert hashlib.sha256(value.strip().encode()).hexdigest() == shim.overlong_question_sha256()


def test_shim_does_not_register_eval_framework():
    for name in shim.VENDORED_MODULES:
        shim.module(name)
    assert "eval_framework" not in sys.modules and "template_formatting" not in sys.modules
    assert all(not k.startswith("eval_framework") for k in sys.modules)


def test_gpqa_en_render_known_good():
    row = syn.gpqa_row(3)
    kind, answer, bench_id = shim.gpqa_diamond_cot_v2()
    assert bench_id == "GPQA_DIAMOND_COT_V2"
    r = shim.render(kind, row, "gpqa_diamond")
    q, choices, letter = oracle_gpqa_en(row)
    assert r.messages == [{"role": "user", "content": TULU_EN.format(q=q, opts=opts(choices))}]
    assert r.ground_truth == letter and r.n_choices == 4
    # the [note] markup leaves upstream's trailing space; the [title] markup becomes ". "
    assert "Widget value 3D " in choices and "Widget value 3B. extra" in choices


def test_gpqa_de_render_known_good():
    row = syn.gpqa_de_row(4, True)
    kind, answer, bench_id = shim.gpqa_ellamind_diamond_cot_de()
    assert bench_id == "GPQA_ELLAMIND_DIAMOND_COT_DE"
    r = shim.render(kind, row, "deu")
    choices, ci = oracle_shuffle_de(row["correct_answer"], row["incorrect_answers"], row["question"] + row["correct_answer"])
    assert shim.shuffle_correct_with_distractors(row["correct_answer"], row["incorrect_answers"],
                                                 row["question"] + row["correct_answer"]) == (choices, ci)
    assert r.messages == [{"role": "user", "content": TULU_DE.format(q=row["question"], opts=opts(choices))}]
    assert r.ground_truth == "ABCD"[ci]


def test_mmlu_en_and_de_render_known_good():
    row = {"question": "  Synthetic question?  ", "options": [f"o{k}" for k in range(10)], "answer_index": 9,
           "category": "computer science"}
    kind, answer, bench_id = shim.mmlu_pro_cot_v2()
    assert bench_id == "MMLU_PRO_COT_V2"
    r = shim.render(kind, row, "computer science")
    pre = "The following are multiple choice questions (with answers) about computer science.\n\n"
    assert r.messages == [{"role": "user", "content": pre + TULU_EN.format(q="Synthetic question?", opts=opts(row["options"]))}]
    assert r.ground_truth == "J" and r.n_choices == 10
    rd = shim.render(shim.mmlu_prox_de_kind(), row, "computer science")
    assert rd.messages == [{"role": "user", "content": TULU_DE.format(q="Synthetic question?", opts=opts(row["options"]))}]
    assert rd.ground_truth == "J"


def test_aime_en_render_known_good():
    kind, answer, bench_id = shim.aime2026()
    assert bench_id == "AIME2026"
    r = shim.render(kind, {"problem": "Synthetic problem: compute 2 plus 3.", "answer": "5"})
    assert r.messages == [{"role": "user", "content": AIME_EN.replace("{p}", "Synthetic problem: compute 2 plus 3.")}]
    assert r.ground_truth == "5" and r.n_choices is None
    assert shim.aime_query_template().endswith("Problem: {Question}")


def test_extractors():
    v2_4 = shim.tulu_answer_v2(4)
    assert shim.extract(v2_4, "So the answer is (C).") == "C"
    assert shim.extract(v2_4, "the answer is: b") == "B"
    assert shim.extract(v2_4, "answer is (A) ... Therefore, the answer is (D)") == "D"  # last match wins
    assert shim.extract(v2_4, "Therefore, the answer is (E)") == "[invalid]"
    assert shim.extract(shim.tulu_answer_v2(10), "Therefore, the answer is (J)") == "J"
    de = shim.tulu_answer_de()
    assert shim.extract(de, "Daher ist die Antwort (D).") == "D"
    assert shim.extract(de, "Antwort: a") == "A"
    assert shim.extract(de, "The answer is (B)") == "B"
    assert shim.extract(de, "Antwort D in der Begründung") == "[invalid]"
    assert shim.extract(de, "Daher ist die Antwort (E)") == "[invalid]"
    _, aime_answer, _ = shim.aime2026()
    assert shim.extract(aime_answer, r"so $\boxed{042}$") == "42"
    assert shim.extract(aime_answer, "no box") == "[no_answer]"
    assert shim.extract_boxed(r"x \boxed{\frac{1}{2}} y") == r"\frac{1}{2}"
    assert shim.strip_string_with_bug("0.5") == r"\frac{1}{2}"


def test_get_n_letters_and_subjects():
    assert shim.get_n_letters(4) == ["A", "B", "C", "D"]
    assert shim.get_n_letters(10)[-1] == "J"
    assert shim.mmlu_pro_subjects() == syn.MMLU_CATS
