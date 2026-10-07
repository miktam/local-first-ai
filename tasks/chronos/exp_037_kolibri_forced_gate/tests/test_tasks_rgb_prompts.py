"""tasks/rgb.py on synthetic RGB-shaped items with stub instruction strings (BUILD_SPEC §5.5).

No RGB text appears here. One test runs RGB's own `processdata` in place (parsed out of the local
evalue.py, never copied) on the synthetic rows and checks our document choice against it.
"""

from __future__ import annotations

import ast
import math
import os
import random
from pathlib import Path

import pytest

import tasks_synthetic as syn
from tasks import rgb
from tasks.common import json_sha256


@pytest.fixture()
def rgb_dir(tmp_path):
    return syn.write_rgb(tmp_path, n_en=9, n_fact=6, n_int=20)


@pytest.fixture()
def instr(rgb_dir):
    return rgb.load_instruction(rgb_dir)


def test_instruction_loader(instr, tmp_path):
    assert instr == {"system": syn.RGB_SYSTEM, "instruction": syn.RGB_INSTRUCTION}
    bad = tmp_path / "bad"
    (bad / "config").mkdir(parents=True)
    (bad / "config" / "instruction.yaml").write_text('en:\n  system: "s"\n  instruction: "no placeholders"\n', encoding="utf-8")
    with pytest.raises(ValueError):
        rgb.load_instruction(bad)


def test_closed_book_has_no_system_message_and_empty_docs(rgb_dir, instr):
    items = rgb.cb_items(rgb_dir, instr)
    assert len(items) == 15 and [it.id for it in items][:2] == ["en-0", "en-1"] and items[-1].id == "en_fact-5"
    for it in items:
        assert it.messages == [{"role": "user", "content": syn.RGB_INSTRUCTION.format(QUERY=it.source["query"], DOCS="")}]
        assert all(m["role"] != "system" for m in it.messages)
        assert it.gold == it.source["answer"]
    assert items[-1].private["gold_fake"] == "widget 5" and "gold_fake" not in items[0].private


def test_forced_prompt(rgb_dir):
    items = rgb.forced_items(rgb_dir)
    q = items[0].source["query"]
    assert items[0].messages == [{"role": "user", "content": f"Question: {q}\n\nAnswer with a short phrase. Always give your best answer."}]
    assert [it.id for it in items] == [it.id for it in rgb.cb_items(rgb_dir)]


def test_negative_noise_one_has_no_positive_passage(rgb_dir, instr):
    for it in rgb.negative_items(rgb_dir, instr):
        sys_msg, user = it.messages
        assert sys_msg == {"role": "system", "content": syn.RGB_SYSTEM}
        assert "POS" not in user["content"] and "NEG" in user["content"]
        n_neg = len(it.source["negative"])
        assert sorted(it.public["doc_indices"]) == list(range(min(5, n_neg)))
        docs = [it.source["negative"][i] for i in it.public["doc_indices"]]
        assert user["content"] == syn.RGB_INSTRUCTION.format(QUERY=it.source["query"], DOCS="\n".join(docs))


def test_fact_uses_positive_wrong(rgb_dir, instr):
    for it in rgb.fact_items(rgb_dir, instr):
        user = it.messages[1]["content"]
        assert "WRONG" in user and "POS" not in user and "NEG" not in user
        assert sorted(it.public["doc_indices"]) == list(range(min(5, len(it.source["positive"]))))


def test_indices_deterministic_and_replayable(rgb_dir, instr):
    a = rgb.negative_items(rgb_dir, instr)
    b = rgb.negative_items(rgb_dir, instr)
    assert [it.public["doc_indices"] for it in a] == [it.public["doc_indices"] for it in b]
    recorded = {it.id: it.public["doc_indices"] for it in a}
    c = rgb.negative_items(rgb_dir, instr, indices=recorded)
    assert [it.prompt_sha256 for it in a] == [it.prompt_sha256 for it in c]
    # different recorded indices give a different prompt: the run host renders from the manifest, not the RNG
    first = a[0].id
    swapped = dict(recorded, **{first: list(reversed(recorded[first]))})
    d = rgb.negative_items(rgb_dir, instr, indices=swapped)
    assert d[0].prompt_sha256 != a[0].prompt_sha256 and d[0].prompt_sha256 == json_sha256(d[0].messages)


def _rgb_src() -> Path | None:
    for c in (os.environ.get("EXP036_DATA"), str(Path.home() / "models/exp036-mini/data")):
        if c and (Path(c).expanduser() / "RGB-src" / "evalue.py").is_file():
            return Path(c).expanduser() / "RGB-src"
    return None


def _rgb_processdata():
    src = _rgb_src()
    if src is None:
        pytest.skip("no local RGB checkout (set EXP036_DATA); RGB's processdata cannot be run in place")
    tree = ast.parse((src / "evalue.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "processdata")
    ns = {"math": math, "random": random}
    exec(compile(ast.Module([fn], []), str(src / "evalue.py"), "exec"), ns)
    return ns["processdata"]


def test_document_choice_matches_rgb_processdata_in_place(rgb_dir, instr):
    processdata = _rgb_processdata()
    state = random.getstate()
    try:
        for row in rgb.load_en(rgb_dir):
            random.seed(2333)
            _, _, docs = processdata(dict(row), 1.0, 5, "en")
            _, idx = rgb.negative(row, instr)
            assert docs == [row["negative"][i] for i in idx]
        for row in rgb.load_fact(rgb_dir):
            random.seed(2333)
            _, _, docs = processdata(dict(row), 0.0, 5, "en_fact", 0)
            _, idx = rgb.fact(row, instr)
            assert docs == [row["positive_wrong"][i] for i in idx]
    finally:
        random.setstate(state)


def test_pilot_from_en_int_only(rgb_dir, instr):
    cb, fo = rgb.pilot_items(rgb_dir, n_cb=16, n_forced=8, instr=instr)
    assert len(cb) == 16 and len(fo) == 8
    assert all(it.id.startswith("en_int-") for it in cb)
    assert [it.id for it in fo] == [it.id for it in cb[:8]]
    assert [it.id for it in cb] == rgb.pilot_ids(rgb_dir, 16)
    assert all(it.task == "rgb_forced" for it in fo) and all(it.task == "rgb_cb" for it in cb)


def test_ids_carry_the_file(rgb_dir):
    ids = [it.id for it in rgb.forced_items(rgb_dir)]
    assert len(ids) == len(set(ids)) and "en-0" in ids and "en_fact-0" in ids
