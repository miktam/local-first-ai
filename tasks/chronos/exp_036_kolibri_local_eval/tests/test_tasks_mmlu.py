"""tasks/mmlu_prox.py on small synthetic MMLU-ProX-shaped rows (BUILD_SPEC §5.5)."""

from __future__ import annotations

import re

import pytest

import tasks_synthetic as syn
from tasks import mmlu_prox


@pytest.fixture()
def dirs(tmp_path):
    return syn.write_mmlu(tmp_path, lite_per_cat=3, full_extra_per_cat=11)


def test_normalise_drops_trailing_padding_and_checks_gold():
    row = syn.mmlu_row(7, "law", "en", n_opt=7)
    n = mmlu_prox.normalise_row(row)
    assert len(n["options"]) == 7 and n["answer_index"] == 0 and n["id"] == "7"
    listed = mmlu_prox.normalise_row(syn.mmlu_row(7, "law", "en", n_opt=7, style="list"))
    assert listed == n
    with pytest.raises(ValueError, match="disagrees"):
        mmlu_prox.normalise_row(dict(row, answer="C"))
    with pytest.raises(ValueError, match="outside"):
        mmlu_prox.normalise_row(dict(row, answer_index=9, answer="J"))
    gap = dict(row, option_2=None)
    with pytest.raises(ValueError, match="gaps"):
        mmlu_prox.normalise_row(gap)


def test_parallel_ids_per_category_and_seeded(dirs):
    lite, _ = dirs
    per_cat = mmlu_prox.parallel_ids(lite)
    assert list(per_cat) == syn.MMLU_CATS
    assert all(len(v) == 3 for v in per_cat.values())
    assert "9999" not in {i for v in per_cat.values() for i in v}  # EN-only id is not parallel
    assert per_cat == mmlu_prox.parallel_ids(lite)
    assert per_cat != mmlu_prox.parallel_ids(lite, seed=37)


def test_select_ids_nested_and_balanced(dirs):
    lite, _ = dirs
    per_cat = mmlu_prox.parallel_ids(lite)
    s14, s28, s42 = (mmlu_prox.select_ids(per_cat, n) for n in (14, 28, 42))
    assert s28[:14] == s14 and s42[:28] == s28  # nested: smaller sets are prefixes
    for sel, k in ((s14, 1), (s28, 2), (s42, 3)):
        by_cat = {}
        for i in sel:
            cat = next(c for c, ids in per_cat.items() if i in ids)
            by_cat.setdefault(cat, []).append(i)
        assert set(by_cat) == set(syn.MMLU_CATS) and all(len(v) == k for v in by_cat.values())
        assert all(v == per_cat[c][:k] for c, v in by_cat.items())
    with pytest.raises(ValueError, match="multiple"):
        mmlu_prox.select_ids(per_cat, 15)
    with pytest.raises(ValueError, match="fewer"):
        mmlu_prox.select_ids(per_cat, 56)


def test_lite_items_en_de(dirs):
    lite, _ = dirs
    ids = mmlu_prox.select_ids(mmlu_prox.parallel_ids(lite), 28)
    en = mmlu_prox.load_lite("en", ids, lite)
    de = mmlu_prox.load_lite("de", ids, lite)
    assert [m for it in en for m in mmlu_prox.render_en(it)] == [m for it in en for m in it.messages]
    assert mmlu_prox.render_de(de[0]) == de[0].messages
    assert [it.id for it in en] == ids == [it.id for it in de]
    for e, d in zip(en, de):
        assert e.category == d.category and e.gold == d.gold and e.n_options == d.n_options
        assert e.prompt_text.startswith(f"The following are multiple choice questions (with answers) about {e.category}.\n\n")
        assert d.prompt_text.startswith("Beantworte die folgende Multiple-Choice-Frage")
        block = e.prompt_text.split("\n\nAnswer the above")[0].split("Question: ")[1]
        assert re.findall(r"^\(([A-J])\) ", block, re.M) == list("ABCDEFGHIJ"[: e.n_options])
        assert "Question: Synthetic en question" in e.prompt_text  # question stripped by MmluProReader
        assert e.gold == "ABCDEFGHIJ"[int(e.id) % e.n_options]
    assert {it.n_options for it in en} == {7, 10}


def test_category_counts(dirs):
    _, full = dirs
    counts = mmlu_prox.category_counts(full, "en")
    assert list(counts) == syn.MMLU_CATS and set(counts.values()) == {14}
    assert mmlu_prox.category_counts(full, "de") == counts


def test_german_category_names_map_through_english(tmp_path):
    lite, full = syn.write_mmlu(tmp_path, lite_per_cat=1, full_extra_per_cat=0)
    p = full / "de" / "test-00000-of-00001.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace('"category": "law"', '"category": "Recht"'), encoding="utf-8")
    assert mmlu_prox.category_counts(full, "de")["law"] == 1


def test_full_pool_disjoint_from_lite_and_slices_disjoint(dirs):
    lite, full = dirs
    pool = mmlu_prox.full_pool_ids(full, lite)
    lite_ids = {i for v in mmlu_prox.parallel_ids(lite).values() for i in v} | {"9999"}
    assert not set(pool) & lite_ids and len(pool) == 14 * 11
    pilot_en = mmlu_prox.load_full_pilot("en", 8, full, lite)
    pilot_de = mmlu_prox.load_full_pilot("de", 8, full, lite)
    c1 = mmlu_prox.c1_items(full, lite, n=100)
    peer = mmlu_prox.peer_check_items(full, lite, n=30)
    assert [it.id for it in pilot_en] == [it.id for it in pilot_de] == pool[:8]
    ids = [it.id for it in pilot_en] + [it.id for it in c1] + [it.id for it in peer]
    assert len(ids) == len(set(ids)) == 8 + 100 + 30


def test_full_pool_excludes_lite_question_text_under_other_ids(tmp_path):
    lite, full = syn.write_mmlu(tmp_path, lite_per_cat=1, full_extra_per_cat=2)
    # renumber one full row so its id differs from Lite while its English text is a Lite question
    lite_first = mmlu_prox.rows_by_id(lite, "en")
    some_id, some_row = next(iter(lite_first.items()))
    for lang in ("en", "de"):
        p = full / lang / "test-00000-of-00001.jsonl"
        p.write_text(p.read_text(encoding="utf-8").replace(f'"question_id": {some_id},', '"question_id": 777777,'), encoding="utf-8")
    assert "777777" in mmlu_prox.rows_by_id(full, "en")
    assert "777777" not in mmlu_prox.full_pool_ids(full, lite)


def test_validation_split_ignored(dirs):
    lite, _ = dirs
    assert "1" not in mmlu_prox.rows_by_id(lite, "en")
