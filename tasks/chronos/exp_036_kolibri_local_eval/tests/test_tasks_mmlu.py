"""tasks/mmlu_prox.py on small synthetic MMLU-ProX-shaped rows (BUILD_SPEC §5.5).

Amendment 4 (2026-10-04): MMLU-ProX-Lite is not 42 per category, so the n_M allocation is a house-monotone Webster
allocation proportional to Lite's own counts (tested here on the registered real counts, which are counts only),
and a full-split row whose answer letter disagrees with answer_index is left out of the full pool and listed, while
Lite stays strict.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

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


def test_select_ids_on_a_balanced_lite_is_the_old_rank_interleave(dirs):
    """On equal category counts, Webster with ties to the subject order gives every category its k-th item before
    any category its (k+1)-th: the pre-Amendment-4 listing (n_M/14 per category, interleaved by rank) exactly."""
    lite, _ = dirs
    per_cat = mmlu_prox.parallel_ids(lite)
    s14, s28, s42 = (mmlu_prox.select_ids(per_cat, n) for n in (14, 28, 42))
    assert s28[:14] == s14 and s42[:28] == s28  # nested: smaller sets are prefixes
    for sel, k in ((s14, 1), (s28, 2), (s42, 3)):
        assert sel == [per_cat[c][r] for r in range(k) for c in syn.MMLU_CATS]
    # n_M need not be a multiple of 14 any more (Amendment 4); it must lie within the parallel ids available
    s15 = mmlu_prox.select_ids(per_cat, 15)
    assert s15 == s42[:15] and s15[14] == per_cat[syn.MMLU_CATS[0]][1]  # the tie goes to the first subject
    with pytest.raises(ValueError, match="outside"):
        mmlu_prox.select_ids(per_cat, 43)
    with pytest.raises(ValueError, match="outside"):
        mmlu_prox.select_ids(per_cat, 0)


# The registered Lite counts (selection_rules.json, Amendment 4) in the registered subject order; counts only.
RULES = json.loads((Path(__file__).resolve().parents[1] / "tasks" / "selection_rules.json").read_text(encoding="utf-8"))
LITE = {c: RULES["mmlu_prox"]["lite_category_counts"][c] for c in syn.MMLU_CATS}
LADDER = RULES["mmlu_prox"]["nM_ladder"]

# The per-category allocation at every ladder n_M on the real Lite counts (the table Amendment 4 registers).
ALLOCATION_TABLE = {
    #                 bio bus chem cs econ eng health hist law math other phil phys psych
    588: dict(zip(sorted(LITE), (36, 40, 56, 20, 42, 48, 35, 19, 48, 68, 46, 25, 65, 40))),
    504: dict(zip(sorted(LITE), (31, 34, 48, 17, 36, 41, 30, 16, 41, 58, 40, 22, 56, 34))),
    406: dict(zip(sorted(LITE), (25, 27, 39, 14, 29, 33, 24, 13, 33, 47, 32, 17, 45, 28))),
    350: dict(zip(sorted(LITE), (21, 24, 33, 12, 25, 29, 21, 11, 29, 40, 27, 15, 39, 24))),
    294: dict(zip(sorted(LITE), (18, 20, 28, 10, 21, 24, 17, 9, 24, 34, 23, 13, 33, 20))),
    252: dict(zip(sorted(LITE), (15, 17, 24, 9, 18, 21, 15, 8, 20, 29, 20, 11, 28, 17))),
    196: dict(zip(sorted(LITE), (12, 13, 19, 7, 14, 16, 12, 6, 16, 23, 15, 8, 22, 13))),
    154: dict(zip(sorted(LITE), (9, 10, 15, 5, 11, 13, 9, 5, 13, 18, 12, 7, 17, 10))),
}


def _cumulative(seats):
    """Allocation after each seat: [a(1), a(2), ...] as dicts."""
    a = dict.fromkeys(LITE, 0)
    out = []
    for c in seats:
        a[c] += 1
        out.append(dict(a))
    return out


def test_registered_lite_counts_are_the_real_composition():
    assert sum(LITE.values()) == 588 and list(LITE) == mmlu_prox.subjects() == syn.MMLU_CATS
    assert LITE == syn.MMLU_LITE_COUNTS
    assert sorted(set(LITE.values())) != [42]  # not the 42 x 14 the pre-registration assumed


def test_webster_allocation_table_on_the_registered_counts():
    for n, want in ALLOCATION_TABLE.items():
        assert mmlu_prox.allocation(LITE, n) == {c: want[c] for c in LITE}, n
    assert set(ALLOCATION_TABLE) == set(LADDER)
    assert min(min(a.values()) for a in ALLOCATION_TABLE.values()) == 5  # every category present at every n_M


def test_webster_is_house_monotone_and_a_webster_allocation_at_every_n():
    """For every n from 1 to 588: the allocation at n + 1 adds exactly one item to the allocation at n (nested), and
    the allocation satisfies the Webster divisor condition max c/(2a+1) <= min c/(2a-1) (among the categories that can
    still take an item, resp. that have one), checked exactly; so the sequential build is Webster at every n."""
    seats = mmlu_prox.webster_order(LITE)
    assert len(seats) == 588
    cum = _cumulative(seats)
    for n in range(1, 589):
        a = cum[n - 1]
        if n > 1:
            prev = cum[n - 2]
            assert sum(a[c] - prev[c] for c in LITE) == 1 and all(a[c] >= prev[c] for c in LITE)
        open_ = [Fraction(LITE[c], 2 * a[c] + 1) for c in LITE if a[c] < LITE[c]]
        held = [Fraction(LITE[c], 2 * a[c] - 1) for c in LITE if a[c] > 0]
        if open_ and held:
            assert max(open_) <= min(held), n
    assert cum[-1] == LITE  # n_M = 588 is all of Lite
    for n in (1, 15, 16, 154, 300, 587, 588):
        assert mmlu_prox.allocation(LITE, n) == cum[n - 1]


def test_webster_is_within_one_item_of_the_exact_quota_at_every_n():
    cum = _cumulative(mmlu_prox.webster_order(LITE))
    worst = Fraction(0)
    for n in range(1, 589):
        for c in LITE:
            dev = abs(cum[n - 1][c] - Fraction(n * LITE[c], 588))
            assert dev < 1, (n, c)
            worst = max(worst, dev)
    assert worst > Fraction(1, 2)  # a real constraint: rounding alone does not keep every n within half an item


def test_webster_ties_go_to_the_registered_subject_order():
    assert mmlu_prox.webster_order({"b": 2, "a": 2}) == ["b", "a", "b", "a"]
    # seat 2: a 1/1 = b 3/3, a tie, so a (first in order) gets it; a is then full and b takes the rest
    assert mmlu_prox.webster_order({"a": 1, "b": 3}) == ["b", "a", "b", "b"]
    seats = mmlu_prox.webster_order(LITE)
    # engineering and law both hold 48: engineering, earlier in the subject order, gets each tied seat first
    eng = [i for i, c in enumerate(seats) if c == "engineering"]
    law = [i for i, c in enumerate(seats) if c == "law"]
    assert all(e < l for e, l in zip(eng, law))
    with pytest.raises(ValueError, match="outside"):
        mmlu_prox.allocation(LITE, 589)


def test_select_ids_on_the_real_composition(tmp_path):
    """Synthetic Lite with the real counts: every n_M set is a prefix of the 588 listing (nested), takes a prefix of
    each category's seed-36 order, has the table's per-category n, and is the same for EN and DE."""
    lite, _ = syn.write_mmlu(tmp_path, lite_counts=syn.MMLU_LITE_COUNTS, full_extra_per_cat=0)
    per_cat = mmlu_prox.parallel_ids(lite)
    assert {c: len(v) for c, v in per_cat.items()} == LITE
    full = mmlu_prox.select_ids(per_cat, 588)
    assert sorted(full) == sorted(i for v in per_cat.values() for i in v)  # all 588, each once
    cat_of = {i: c for c, ids in per_cat.items() for i in ids}
    for n in range(1, 589):
        sel = mmlu_prox.select_ids(per_cat, n)
        assert sel == full[:n]
        if n in ALLOCATION_TABLE:
            by_cat = {c: [i for i in sel if cat_of[i] == c] for c in LITE}
            assert {c: len(v) for c, v in by_cat.items()} == ALLOCATION_TABLE[n]
            assert all(v == per_cat[c][:len(v)] for c, v in by_cat.items())
    en = mmlu_prox.load_lite("en", full[:154], lite)
    de = mmlu_prox.load_lite("de", full[:154], lite)
    assert [it.id for it in en] == [it.id for it in de] == full[:154]
    assert [it.category for it in en] == [it.category for it in de] == [cat_of[i] for i in full[:154]]


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
    """Every full-split row counts, the planted gold-mismatch row included (Amendment 4: its category is valid)."""
    _, full = dirs
    counts = mmlu_prox.category_counts(full, "en")
    assert list(counts) == syn.MMLU_CATS
    assert counts == {c: 14 + (c == syn.MMLU_GOLD_MISMATCH_CATEGORY) for c in syn.MMLU_CATS}
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
    assert str(syn.MMLU_GOLD_MISMATCH_ID) not in pool
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
    assert "777777" in mmlu_prox.rows_by_id(full, "en", keep_gold_mismatch=True)  # the full split: Amendment 4
    assert "777777" not in mmlu_prox.full_pool_ids(full, lite)


def test_validation_split_ignored(dirs):
    lite, _ = dirs
    assert "1" not in mmlu_prox.rows_by_id(lite, "en")


# ------------------------------------------------------------------------------ gold disagreement (Amendment 4)


def test_full_gold_mismatch_is_left_out_of_every_pool_and_listed(tmp_path):
    lite, full = syn.write_mmlu(tmp_path, lite_per_cat=3, full_extra_per_cat=11, gold_mismatch=True)
    bad = str(syn.MMLU_GOLD_MISMATCH_ID)
    assert mmlu_prox.gold_mismatch_ids(full) == {"en": [bad], "de": [bad]}
    assert mmlu_prox.rows_by_id(full, "en", keep_gold_mismatch=True)[bad]["gold_mismatch"] is True
    with pytest.raises(ValueError, match="disagrees"):
        mmlu_prox.rows_by_id(full, "en")  # strict reading still refuses it
    pool = mmlu_prox.full_pool_ids(full, lite)
    assert bad not in pool
    ids = ([it.id for it in mmlu_prox.load_full_pilot("en", 8, full, lite)]
           + [it.id for it in mmlu_prox.load_full_pilot("de", 8, full, lite)]
           + [it.id for it in mmlu_prox.c1_items(full, lite, n=100)]
           + [it.id for it in mmlu_prox.peer_check_items(full, lite, n=30)])
    assert bad not in ids
    with pytest.raises(ValueError, match="cannot be items"):
        mmlu_prox.load_full("en", [bad], full)
    # Leaving the row out shifts no other id: the pool equals the pool of the same world without the row.
    lite2, full2 = syn.write_mmlu(tmp_path / "clean", lite_per_cat=3, full_extra_per_cat=11, gold_mismatch=False)
    assert mmlu_prox.gold_mismatch_ids(full2) == {"en": [], "de": []}
    assert mmlu_prox.full_pool_ids(full2, lite2) == pool


def test_full_gold_mismatch_in_one_language_leaves_the_parallel_id_out(tmp_path):
    lite, full = syn.write_mmlu(tmp_path, lite_per_cat=1, full_extra_per_cat=3, gold_mismatch=False)
    rows = mmlu_prox.rows_by_id(full, "de")
    victim = next(i for i in mmlu_prox.full_pool_ids(full, lite))
    p = full / "de" / "test-00000-of-00001.jsonl"
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    for r in lines:
        if str(r["question_id"]) == victim:
            r["answer"] = "ABCDEFGHIJ"[(rows[victim]["answer_index"] + 1) % 10]
    p.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    assert mmlu_prox.gold_mismatch_ids(full) == {"en": [], "de": [victim]}
    assert victim not in mmlu_prox.full_pool_ids(full, lite)
    assert mmlu_prox.category_counts(full, "de") == mmlu_prox.category_counts(full, "en")  # still counted


def test_lite_gold_mismatch_still_stops(tmp_path):
    lite, _ = syn.write_mmlu(tmp_path, lite_per_cat=2, full_extra_per_cat=0)
    p = lite / "en" / "test-00000-of-00001.jsonl"
    first = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    wrong = "ABCDEFGHIJ"[(first["answer_index"] + 1) % 10]
    p.write_text(p.read_text(encoding="utf-8").replace(f'"answer": "{first["answer"]}"', f'"answer": "{wrong}"', 1),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees with answer_index"):
        mmlu_prox.parallel_ids(lite)
    with pytest.raises(ValueError, match="disagrees with answer_index"):
        mmlu_prox.load_lite("en", [str(first["question_id"])], lite)
