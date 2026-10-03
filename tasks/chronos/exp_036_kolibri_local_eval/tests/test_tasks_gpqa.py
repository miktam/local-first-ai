"""tasks/gpqa.py on synthetic GPQA-shaped rows (BUILD_SPEC §5.5; no real GPQA text anywhere)."""

from __future__ import annotations

import hashlib

import pytest

import tasks_synthetic as syn
from tasks import gpqa
from tasks.common import json_sha256
from tasks.vendored_evalfw import shim


@pytest.fixture()
def data(tmp_path):
    syn.write_gpqa_en(tmp_path, n_diamond=6, n_main_extra=12)
    syn.write_gpqa_de(tmp_path, n_diamond=6, n_other=12)
    return tmp_path


def test_diamond_en_items(data):
    items = gpqa.load_diamond_en(data / "gpqa")
    assert [it.id for it in items] == [f"recSYN{i:04d}" for i in range(6)]
    for it in items:
        assert it.task == "gpqa_en" and it.n_options == 4 and it.gold in "ABCD"
        assert len(it.messages) == 1 and it.messages[0]["role"] == "user"
        assert it.prompt_sha256 == json_sha256(it.messages)
        assert it.item_sha256 == json_sha256({k: it.source[k] for k in gpqa.EN_FIELDS})
        assert it.prompt_text.endswith("(D), (E), etc.")


def test_shuffle_matches_vendored_reader_and_is_stable(data):
    a = gpqa.load_diamond_en(data / "gpqa")
    b = gpqa.load_diamond_en(data / "gpqa")
    assert [(x.prompt_sha256, x.gold) for x in a] == [(x.prompt_sha256, x.gold) for x in b]
    reader = shim.gpqa_reader()
    for it in a:
        fields = reader.read(it.source)
        assert it.gold == "ABCD"[fields.correct_index]
        for letter, choice in zip("ABCD", fields.choices):
            assert f"({letter}) {choice}\n" in it.prompt_text + "\n"
    # the gold letter is not constant across items (the shuffle does something)
    assert len({it.gold for it in a}) > 1


def test_primary_excludes_by_hash_only(data):
    items = gpqa.load_diamond_en(data / "gpqa")
    target = hashlib.sha256(items[2].source["Question"].encode()).hexdigest()
    kept = gpqa.primary_197(items, overlong_sha256=target)
    assert [it.id for it in kept] == [it.id for i, it in enumerate(items) if i != 2]
    with pytest.raises(ValueError, match="expected 1"):
        gpqa.primary_197(items, overlong_sha256="0" * 64)
    # the real constant matches none of the synthetic rows, so the production call refuses
    with pytest.raises(ValueError, match="expected 1"):
        gpqa.primary_197(items)


def test_primary_refuses_a_strip_only_match(data):
    items = gpqa.load_diamond_en(data / "gpqa")
    items[1].source = dict(items[1].source, Question="  " + items[1].source["Question"] + " ")
    target = hashlib.sha256(items[1].source["Question"].strip().encode()).hexdigest()
    with pytest.raises(ValueError, match="only after stripping"):
        gpqa.primary_197(items, overlong_sha256=target)


def test_diamond_de_filters_is_diamond(data):
    items = gpqa.load_diamond_de(data / "gpqa-multilingual")
    assert len(items) == 6 and all(it.source["is_diamond"] for it in items)
    assert [it.id for it in items] == [f"recSYN{i:04d}" for i in range(6)]
    assert {it.public["id_source"] for it in items} == {"Record ID"}
    for it in items:
        assert it.prompt_text.startswith("Beantworte die folgende Multiple-Choice-Frage")
        choices, ci = shim.shuffle_correct_with_distractors(
            it.source["correct_answer"], list(it.source["incorrect_answers"]), it.source["question"] + it.source["correct_answer"])
        assert it.gold == "ABCD"[ci]


def test_de_ids_fall_back_to_row_index(tmp_path):
    syn.write_gpqa_de(tmp_path, n_diamond=3, n_other=2, with_record_ids=False)
    items = gpqa.load_diamond_de(tmp_path / "gpqa-multilingual")
    assert [it.id for it in items] == ["0", "1", "2"]
    assert {it.public["id_source"] for it in items} == {"_exp036_row_index"}


def test_pilot_disjoint_from_diamond_and_seeded(data):
    diamond = gpqa.diamond_record_ids(data / "gpqa")
    en = gpqa.load_pilot_main(data / "gpqa", diamond, n=8)
    de = gpqa.load_pilot_main(data / "gpqa-multilingual", diamond, n=8, lang="de")
    assert len(en) == 8 and len(de) == 8
    assert not ({it.id for it in en} & diamond) and not ({it.id for it in de} & diamond)
    assert all(not it.source["is_diamond"] for it in de)
    again = gpqa.load_pilot_main(data / "gpqa", diamond, n=8)
    assert [it.id for it in en] == [it.id for it in again]
    other_seed = gpqa.load_pilot_main(data / "gpqa", diamond, n=8, seed=37)
    assert [it.id for it in en] != [it.id for it in other_seed]
    # the over-long question is excluded from the pilot pool too
    target = hashlib.sha256(en[0].source["Question"].encode()).hexdigest()
    shifted = gpqa.load_pilot_main(data / "gpqa", diamond, n=8, overlong_sha256=target)
    assert en[0].id not in {it.id for it in shifted}


def test_missing_column_is_an_error(tmp_path):
    d = tmp_path / "gpqa"
    d.mkdir()
    (d / "gpqa_diamond.csv").write_text("Record ID,Question\nrec1,Synthetic?\n", encoding="utf-8")
    with pytest.raises(KeyError):
        gpqa.load_diamond_en(d)


def test_csv_fields_are_read_as_pandas_would(tmp_path):
    """HF datasets loads the CSV with pandas: an exact NA string ("None", "N/A", "") arrives as None, and
    eval-framework's _preprocess renders a None option as " "."""
    syn.write_gpqa_en(tmp_path, n_diamond=2, n_main_extra=0)
    p = tmp_path / "gpqa" / "gpqa_diamond.csv"
    p.write_text(p.read_text(encoding="utf-8").replace("Widget value 1D [note]", "None"), encoding="utf-8")
    rows = gpqa.en_rows(tmp_path / "gpqa")
    assert rows[1]["Incorrect Answer 3"] is None and rows[0]["Incorrect Answer 3"] == "Widget value 0D [note]"
    item = gpqa.load_diamond_en(tmp_path / "gpqa")[1]
    fields = shim.gpqa_reader().read(item.source)
    assert " " in fields.choices and "None" not in fields.choices
    assert item.item_sha256 == json_sha256({k: item.source[k] for k in gpqa.EN_FIELDS})
