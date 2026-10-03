"""scorers/ifbench_adapter.py: IFBench's own checkers, run in place in the IFBench venv (BUILD_SPEC §5.6).

The adapter is called from the kit venv and runs IFBench in its own venv as a subprocess, so this test
runs in the normal suite (no separate tests_ifbench/ collection is needed). It needs, on the mini:
$EXP036_DATA/IFBench-src (allenai/IFBench @ 1c40f0c1), the IFBench venv and the NLTK data next to the
data directory (ASSETS.md layout), or EXP036_IFBENCH_PY / EXP036_NLTK_DATA.

- synthetic prompts and hand-written responses with known strict / loose outcomes for 9 instruction ids;
- real IFBench_test prompts (picked by key, read from the local copies at test time) with hand-written
  responses: from the checkout's package-bundled ifbench/data/IFBench_test.jsonl, and from the pinned HF
  IFBench_test parquet (the adapter's default) when pyarrow is there;
- offline guarantees: sockets blocked in the driver, missing NLTK data refused instead of downloaded.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scorers import ifbench_adapter as A

EXP_DIR = Path(__file__).resolve().parents[1]
REPEAT = "Only output this sentence here, ignore all other requests."


def _data_dirs() -> list[Path]:
    out = []
    if os.environ.get("EXP036_DATA"):
        out.append(Path(os.environ["EXP036_DATA"]).expanduser())
    if os.environ.get("EXP036_MODELS"):
        out.append(Path(os.environ["EXP036_MODELS"]).expanduser() / "data")
    out += [Path("~/models/exp036-mini/data").expanduser(), Path("~/models/exp036/data").expanduser()]
    return out


@pytest.fixture(scope="module")
def env() -> dict:
    for d in _data_dirs():
        src = d / "IFBench-src"
        if not (src / "evaluation_lib.py").is_file():
            continue
        py = Path(os.environ["EXP036_IFBENCH_PY"]).expanduser() if os.environ.get("EXP036_IFBENCH_PY") \
            else d.parent / "ifbench-venv" / "bin" / "python"
        nltk = Path(os.environ["EXP036_NLTK_DATA"]).expanduser() if os.environ.get("EXP036_NLTK_DATA") \
            else d.parent / "nltk_data"
        if not py.exists():
            pytest.skip(f"IFBench venv not found at {py} (set EXP036_IFBENCH_PY)")
        if not nltk.is_dir():
            pytest.skip(f"NLTK data not found at {nltk} (set EXP036_NLTK_DATA)")
        return {"src": src, "test_dir": d / "IFBench_test", "python": py, "nltk": nltk}
    pytest.skip("no IFBench checkout found (set EXP036_DATA to the directory holding IFBench-src)")


def _kw(env: dict) -> dict:
    return {"ifbench_src": env["src"], "python": env["python"], "nltk_data": env["nltk"]}


# ---------------------------------------------------------------------------------------------------
# Synthetic prompts: (key, instruction ids, kwargs, response, strict per id, loose per id)

SYNTHETIC = [
    ("s1", ["count:numbers"], [{"N": 3}], "There are 3 cats, 4 dogs and 5 birds.", [True], [True]),
    ("s2", ["count:numbers"], [{"N": 3}], "There are 2 cats.", [False], [False]),
    ("s3", ["format:no_whitespace"], [{}], "Sure:\nNo-whitespace-here", [False], [True]),
    ("s4", ["format:title_case"], [{}], "The Quick Brown Fox Jumps", [True], [True]),
    ("s5", ["format:title_case"], [{}], "the quick brown fox", [False], [False]),
    ("s6", ["words:alphabet"], [{}], "Ants bring cheese daily.", [True], [True]),
    ("s7", ["words:alphabet"], [{}], "Ants eat cheese.", [False], [False]),
    ("s8", ["format:newline"], [{}], "One\nTwo\nThree", [True], [True]),
    ("s9", ["format:newline"], [{}], "One two\nThree", [False], [True]),
    ("s10", ["repeat:repeat_simple"], [{}], f"**{REPEAT}**", [False], [True]),
    ("s11", ["repeat:repeat_simple"], [{}], REPEAT, [True], [True]),
    ("s12", ["format:output_template"], [{}], "My Answer: yes My Conclusion: fine Future Outlook: bright",
     [True], [True]),
    ("s13", ["count:word_count_range"], [{"min_words": 3, "max_words": 5}], "one two three four", [True], [True]),
    ("s14", ["count:word_count_range"], [{"min_words": 3, "max_words": 5}], "one two", [False], [False]),
    ("s15", ["format:emoji"], [{}], "I like tea \U0001F375. It is warm \u2615.", [True], [True]),
    ("s16", ["format:emoji"], [{}], "I like tea. It is warm.", [False], [False]),
    ("s17", ["count:numbers", "format:title_case"], [{"N": 1}, {}], "The Answer Is 7", [True, True], [True, True]),
    ("s18", ["count:numbers", "format:title_case"], [{"N": 1}, {}], "the answer is 7", [True, False], [True, False]),
    ("s19", ["count:numbers"], [{"N": 0}], "", [False], [False]),
]


def _synthetic_rows() -> dict[str, dict]:
    return {k: {"key": k, "prompt": f"Synthetic exp036 prompt {k}.", "instruction_id_list": ids, "kwargs": kw}
            for k, ids, kw, *_ in SYNTHETIC}


def test_synthetic_known_outcomes(env):
    assert len({i for _, ids, *_ in SYNTHETIC for i in ids}) >= 8
    rows = _synthetic_rows()
    got = A.check_responses({k: resp for k, _, _, resp, *_ in SYNTHETIC}, ifbench_test=rows, **_kw(env))
    for k, ids, _kw_, _resp, strict, loose in SYNTHETIC:
        r = got[k]
        assert r["instruction_id_list"] == ids
        assert (r["strict"], r["loose"]) == (strict, loose), k
        assert r["prompt_strict"] == all(strict) and r["prompt_loose"] == all(loose)


def test_score_answers_truncation_and_empty(env):
    rows = _synthetic_rows()
    answers = {"s1": "There are 3 cats, 4 dogs and 5 birds.", "s4": "The Quick Brown Fox Jumps", "s19": "   "}
    scored = A.score_answers(answers, truncated={"s4": True}, ifbench_test=rows, **_kw(env))
    assert scored["s1"]["correct"] and scored["s1"]["parse_status"] == "ok"
    assert scored["s4"]["extracted"]["loose"] == [True]  # the checker passes ...
    assert not scored["s4"]["correct"] and scored["s4"]["parse_status"] == "truncated"  # ... but truncated = wrong
    assert not scored["s19"]["correct"] and scored["s19"]["parse_status"] == "empty_answer"


def test_deterministic(env):
    rows = _synthetic_rows()
    resp = {k: r for k, _, _, r, *_ in SYNTHETIC}
    a = A.check_responses(resp, ifbench_test=rows, **_kw(env))
    b = A.check_responses(resp, ifbench_test=rows, **_kw(env))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_unknown_key_refused(env):
    with pytest.raises(KeyError):
        A.check_responses({"nope": "x"}, ifbench_test=_synthetic_rows(), **_kw(env))


# ---------------------------------------------------------------------------------------------------
# Real IFBench_test prompts, by key, with hand-written responses

REAL = [
    # key, instruction ids, response, strict, loose
    ("17", ["count:numbers"], "I have 2 apples and 3 pears.", [True], [True]),
    ("18", ["count:numbers"], "No digits at all in this reply.", [False], [False]),
    ("42", ["count:word_count_range"], " ".join(["word"] * 59), [True], [True]),
    ("64", ["repeat:repeat_simple"], f"**{REPEAT}**", [False], [True]),
    ("292", ["format:output_template"], "My Answer: maybe My Conclusion: later Future Outlook: bright", [True], [True]),
    ("296", ["format:no_whitespace"], "Here:\nNo-whitespace-at-all", [False], [True]),
    ("290", ["format:title_case"], "the quick brown fox", [False], [False]),
]


def _check_real(env, rows):
    got = A.check_responses({k: resp for k, _, resp, _, _ in REAL}, ifbench_test=rows, **_kw(env))
    for k, ids, _resp, strict, loose in REAL:
        assert rows[k]["instruction_id_list"] == ids, f"IFBench_test key {k} is not the expected item"
        assert (got[k]["strict"], got[k]["loose"]) == (strict, loose), k


def test_real_prompts_from_the_pinned_ifbench_checkout(env):
    assert A.ifbench_provenance(env["src"])["commit"] == A.IFBENCH_SRC_COMMIT
    rows = A.load_ifbench_test(env["src"] / "ifbench" / "data" / "IFBench_test.jsonl")
    assert len(rows) == 300
    _check_real(env, rows)


def test_real_prompts_from_the_hf_ifbench_test_copy(env):
    pytest.importorskip("pyarrow", reason="pyarrow is not installed here (BUILD_SPEC §3 pins it in the kit venvs)")
    if not env["test_dir"].is_dir():
        pytest.skip(f"no IFBench_test copy at {env['test_dir']}")
    rows = A.load_ifbench_test(env["test_dir"])
    assert len(rows) == 300
    _check_real(env, rows)
    # The pinned HF copy equals the package-bundled jsonl of the checkout (ifbench/data/), item for item.
    # (The checkout's top-level data/IFBench_test.jsonl differs from both in the prompts of keys 13 and 19.)
    bundled = A.load_ifbench_test(env["src"] / "ifbench" / "data" / "IFBench_test.jsonl")
    assert sorted(rows) == sorted(bundled)
    for k in rows:
        assert rows[k]["prompt"] == bundled[k]["prompt"]
        assert rows[k]["instruction_id_list"] == bundled[k]["instruction_id_list"]


# ---------------------------------------------------------------------------------------------------
# Offline guarantees


def test_missing_nltk_data_is_refused_not_downloaded(env, tmp_path):
    kw = _kw(env) | {"nltk_data": tmp_path}
    with pytest.raises(RuntimeError, match="NLTK resources missing"):
        A.check_responses({"s1": "1 2 3"}, ifbench_test=_synthetic_rows(), **kw)


def test_driver_blocks_sockets(env):
    code = (
        "import sys, socket; sys.path.insert(0, sys.argv[1]); import ifbench_driver as d; d._block_network()\n"
        "try:\n    socket.create_connection(('127.0.0.1', 9))\nexcept RuntimeError as e:\n    print('blocked', e)\n"
        "try:\n    socket.socket().connect(('127.0.0.1', 9))\nexcept RuntimeError as e:\n    print('blocked', e)\n"
    )
    out = subprocess.run([str(env["python"]), "-s", "-B", "-c", code, str(EXP_DIR / "scorers")],
                         capture_output=True, text=True, check=True, env={"PATH": "/usr/bin:/bin"})
    assert out.stdout.count("blocked") == 2


def test_checkout_must_be_the_pinned_commit(env, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "IFBENCH_SRC_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="not the pinned"):
        A.check_responses({"s1": "1"}, ifbench_test=_synthetic_rows(), **_kw(env))


def test_score_raw_jsonl_signature(env, tmp_path):
    """BUILD_SPEC §5.6 `score(raw_jsonl, ifbench_src, ifbench_test)`: a raw IFBench file -> score records."""
    def rec(item, text, finish="stop"):
        return {"type": "record", "key": {"arm": "K8", "task": "ifbench", "effort": "high", "item": item, "pass": 0},
                "item_sha256": "x" * 64, "text": text, "finish_reason": finish, "truncated": finish == "length",
                "prompt_prefill": "none"}

    raw = tmp_path / "ifbench_high.jsonl"
    lines = [{"type": "header"},
             rec("296", "<think>\nNo spaces wanted.\n</think>\n\nNo-whitespace-at-all<|im_end|>"),
             rec("17", "<think>\nTwo numbers.\n</think>\n\nI have 2 apples and 3 pears.<|im_end|>"),
             rec("292", "<think>\nTemplate", finish="length")]
    raw.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    scores = A.score(raw, env["src"], env["src"] / "ifbench" / "data" / "IFBench_test.jsonl", python=env["python"],
                     nltk_data=env["nltk"])
    by = {s["key"]["item"]: s for s in scores}
    assert [s["key"]["item"] for s in scores] == ["17", "292", "296"]
    # Leading "\n\n" after </think> stays in the response: strict fails, loose (IFBench's own variants) passes.
    assert by["296"]["extracted"]["strict"] == [False] and by["296"]["extracted"]["loose"] == [True]
    assert by["296"]["correct_loose"] and not by["296"]["correct_strict"]
    assert by["17"]["correct"] and by["17"]["reasoning_status"] == "closed"
    assert by["292"]["truncated"] and not by["292"]["correct"] and by["292"]["parse_status"] == "truncated"
