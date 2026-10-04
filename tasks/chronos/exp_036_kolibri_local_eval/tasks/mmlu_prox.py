"""MMLU-ProX-Lite EN/DE (main rows) and MMLU-ProX full (pilot, C1, peer check, category weights).

BUILD_SPEC §5.5 `tasks/mmlu_prox.py`; HYPOTHESIS H2 (MMLU rows, post-stratified), H3, H7, C14, C19, Control C1.

Public data (MIT): manifests may carry id, category and gold letter, never the text.

Rows. MMLU-ProX stores the options either as a list `options` or as columns `option_0` … `option_9`. Questions
with fewer than ten options pad the trailing columns; trailing options that are null, empty or "N/A" are
dropped (selection_rules.json `mmlu_prox.options_rule`). `answer_index` must point inside the options and agree
with the `answer` letter. Categories are the 14 English MMLU-Pro names; a German row's category is taken from
the English row with the same `question_id` when it is not one of them.

Gold disagreement (Amendment 4, 2026-10-04; selection_rules.json `mmlu_prox.gold_rule`). In MMLU-ProX-Lite a row
whose `answer` letter disagrees with `answer_index` stops the build, as before. In the full test split such a row
(question_id 3787 at 8e6106a6, in EN and DE, the only one of 11,759 per language) is kept with the flag
`gold_mismatch` so that the post-stratification category counts still count it (its category is valid), and it is
left out of the full pool (pilot, Control C1, peer check); `gold_mismatch_ids` lists it per language for the
manifests and the build's checks, and no item is ever rendered from it.

Prompts (C10, R rows):
- EN: eval-framework MMLU_PRO_COT_V2 — subject preamble "The following are multiple choice questions (with
  answers) about <category>." folded into the user turn, then tulu3_cot_prompt; extractor tulu_answer_v2(10).
- DE: the vendored Cot kind with MmluProReader and tulu3_cot_prompt_de, no preamble
  (tasks/prompts/mmlu_prox_de_note.md); the German extractor is extended to A–J in scorers/mc.py.

Selection (selection_rules.json `nM_rule`, Amendment 4): ids present in both languages, per category in one seed-36
order. MMLU-ProX-Lite is not 42 per category (at e82aafb9: biology 36, business 40, chemistry 56, computer science
20, economics 42, engineering 48, health 35, history 19, law 48, math 68, other 46, philosophy 25, physics 65,
psychology 40; 588 per language), so the per-category allocation at n_M is a house-monotone Webster
(Sainte-Laguë) allocation proportional to those counts, built seat by seat (`webster_order`): each next item goes
to the category with the highest c_k / (2 a_k + 1) among those with a_k < c_k, ties to the earlier category in the
registered subject order, and within a category the items are a prefix of its seed-36 order. Every smaller set is
a subset of every larger one, n_M = 588 is all of Lite, and n_M need not be a multiple of 14. The manifest lists
the items in seat order, so the n_M set is the first n_M entries. (Until Amendment 4: n_M/14 ids per category.)
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

from tasks.common import Item, default_path, make_item, read_split, require_unique, seeded_order, sha256_hex
from tasks.vendored_evalfw import shim

LITE_REPO = "li-lab/MMLU-ProX-Lite"
FULL_REPO = "li-lab/MMLU-ProX"
LANGS = ("en", "de")
SPLIT = "test"
LETTERS = "ABCDEFGHIJ"
_PAD_OPTIONS = ("", "N/A")


def subjects() -> list[str]:
    """The 14 MMLU-Pro categories, in eval-framework's order."""
    return shim.mmlu_pro_subjects()


def gold_disagrees(row: dict) -> bool:
    """True when the raw row's `answer` is one letter A–J that is not the letter at `answer_index`."""
    letter = row.get("answer")
    if not (isinstance(letter, str) and len(letter.strip()) == 1 and letter.strip() in LETTERS):
        return False  # no single-letter answer: answer_index alone is the gold, as before
    idx = int(row["answer_index"])
    return not 0 <= idx < len(LETTERS) or letter.strip() != LETTERS[idx]


def normalise_row(row: dict, keep_gold_mismatch: bool = False) -> dict:
    """{id, question, options, answer_index, category} from an MMLU-ProX row (see the module docstring).

    A row whose answer letter disagrees with answer_index raises (Lite; strict), unless `keep_gold_mismatch`
    (the full split, Amendment 4): the row is then returned with `gold_mismatch: True` and must not be rendered."""
    if "question_id" not in row or "question" not in row or "answer_index" not in row or "category" not in row:
        raise KeyError(f"MMLU-ProX row lacks one of question_id/question/answer_index/category: {sorted(row)}")
    if isinstance(row.get("options"), (list, tuple)):
        options = list(row["options"])
    else:
        options = [row[f"option_{k}"] for k in range(10) if f"option_{k}" in row]
    while options and (options[-1] is None or str(options[-1]).strip() in _PAD_OPTIONS):
        options.pop()
    if not options or any(o is None for o in options):
        raise ValueError(f"MMLU-ProX {row['question_id']}: options are empty or have gaps")
    idx = int(row["answer_index"])
    if not 0 <= idx < len(options):
        raise ValueError(f"MMLU-ProX {row['question_id']}: answer_index {idx} outside {len(options)} options")
    mismatch = gold_disagrees(row)
    if mismatch and not keep_gold_mismatch:
        raise ValueError(f"MMLU-ProX {row['question_id']}: answer {row.get('answer')!r} disagrees with answer_index {idx}")
    out = {
        "id": str(row["question_id"]),
        "question": row["question"],
        "options": [str(o) for o in options],
        "answer_index": idx,
        "category": row["category"],
    }
    if mismatch:
        out["gold_mismatch"] = True
    return out


def _lite(d: Path | None) -> Path:
    return Path(d) if d is not None else default_path(LITE_REPO)


def _full(d: Path | None) -> Path:
    return Path(d) if d is not None else default_path(FULL_REPO)


def rows_by_id(root: Path, lang: str, keep_gold_mismatch: bool = False) -> dict[str, dict]:
    """{id: normalised row} of the test split; `keep_gold_mismatch` for the full split only (Amendment 4)."""
    rows = [normalise_row(r, keep_gold_mismatch) for r in read_split(root, config=lang, split=SPLIT)]
    require_unique([r["id"] for r in rows], f"MMLU-ProX {lang} {SPLIT}")
    return {r["id"]: r for r in rows}


def gold_mismatch_ids(full_dir: Path | None = None) -> dict[str, list[str]]:
    """{lang: sorted ids of the full test split whose answer letter disagrees with answer_index} (Amendment 4)."""
    full_dir = _full(full_dir)
    return {lang: sorted(i for i, r in rows_by_id(full_dir, lang, keep_gold_mismatch=True).items()
                         if r.get("gold_mismatch")) for lang in LANGS}


def _english_category(row: dict, en_rows: dict[str, dict] | None) -> str:
    cats = subjects()
    if row["category"] in cats:
        return row["category"]
    if en_rows is not None and row["id"] in en_rows and en_rows[row["id"]]["category"] in cats:
        return en_rows[row["id"]]["category"]
    raise ValueError(f"MMLU-ProX {row['id']}: category {row['category']!r} is not an MMLU-Pro category")


def parallel_ids(lite_dir: Path | None = None, seed: int = 36) -> dict[str, list[str]]:
    """{category: ids present in both en and de (Lite test), in the seed-`seed` order}, in subject order."""
    lite_dir = _lite(lite_dir)
    en, de = rows_by_id(lite_dir, "en"), rows_by_id(lite_dir, "de")
    shared = sorted(set(en) & set(de))
    by_cat: dict[str, list[str]] = {c: [] for c in subjects()}
    for i in shared:
        by_cat[_english_category(en[i], None)].append(i)
    return {c: seeded_order(ids, seed, f"mmlu_prox_lite|{c}") for c, ids in by_cat.items()}


def webster_order(counts: Mapping[str, int]) -> list[str]:
    """The seat order of the house-monotone Webster (Sainte-Laguë) allocation over `counts` (Amendment 4).

    Seat by seat, the next item goes to the category with the highest c_k / (2 a_k + 1) among the categories with
    a_k < c_k (a_k = the items it has so far); a tie goes to the category that comes first in `counts` (the
    registered subject order). The comparison is exact (integer cross-multiplication). One entry per item,
    sum(counts) long; its first n entries are the allocation at n, for every n, so the sets are nested."""
    cats = list(counts)
    c = [int(counts[k]) for k in cats]
    if any(x < 0 for x in c):
        raise ValueError(f"negative category count in {dict(counts)}")
    a = [0] * len(c)
    seats: list[str] = []
    for _ in range(sum(c)):
        best = -1
        for k in range(len(c)):
            if a[k] < c[k] and (best < 0 or c[k] * (2 * a[best] + 1) > c[best] * (2 * a[k] + 1)):
                best = k
        a[best] += 1
        seats.append(cats[best])
    return seats


def allocation(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Items per category at n (the first n seats of `webster_order`), in the order of `counts`."""
    total = sum(int(v) for v in counts.values())
    if not 0 <= n <= total:
        raise ValueError(f"n={n} outside 0..{total}")
    got = Counter(webster_order(counts)[:n])
    return {k: got.get(k, 0) for k in counts}


def select_ids(per_category: dict[str, list[str]], n_m: int) -> list[str]:
    """The n_M set in listing order (Amendment 4): the first n_M seats of `webster_order` over the categories'
    parallel-id counts, each seat taking the next id of its category's seed-36 order (so a category's items are a
    prefix of that order). n_M = the Lite total lists all of Lite, and every smaller n_M set is a prefix of it."""
    counts = {c: len(ids) for c, ids in per_category.items()}
    total = sum(counts.values())
    if not 0 < n_m <= total:
        raise ValueError(f"n_M={n_m}: outside 1..{total} (the parallel ids available)")
    taken = dict.fromkeys(per_category, 0)
    out = []
    for c in webster_order(counts)[:n_m]:
        out.append(per_category[c][taken[c]])
        taken[c] += 1
    return out


def category_counts(full_dir: Path | None = None, lang: str = "en") -> dict[str, int]:
    """Items per category in the full MMLU-ProX test split of `lang` (post-stratification weights, H2). Every row
    counts, including a gold-mismatch row (Amendment 4: its category is valid; only its gold is ambiguous)."""
    full_dir = _full(full_dir)
    rows = rows_by_id(full_dir, lang, keep_gold_mismatch=True)
    en_rows = rows if lang == "en" else rows_by_id(full_dir, "en", keep_gold_mismatch=True)
    counts = Counter(_english_category(r, en_rows) for r in rows.values())
    return {c: counts.get(c, 0) for c in subjects()}


def rendered_en(row: dict) -> shim.Rendered:
    """MMLU_PRO_COT_V2 on a normalised row whose category is an MMLU-Pro name (the preamble's subject)."""
    kind, _answer, _id = shim.mmlu_pro_cot_v2()
    return shim.render(kind, row, subject_label=row["category"])


def rendered_de(row: dict) -> shim.Rendered:
    return shim.render(shim.mmlu_prox_de_kind(), row, subject_label=row["category"])


def render_en(row_or_item) -> list[dict]:
    """Messages of a normalised EN row, or of an Item (from its source row and English category)."""
    if isinstance(row_or_item, Item):
        return rendered_en(dict(row_or_item.source, category=row_or_item.category)).messages
    return rendered_en(row_or_item).messages


def render_de(row_or_item) -> list[dict]:
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return rendered_de(row).messages


def _item(row: dict, lang: str, category: str) -> Item:
    r = rendered_en(dict(row, category=category)) if lang == "en" else rendered_de(row)
    src = {"question": row["question"], "options": row["options"], "answer_index": row["answer_index"],
           "category": category}
    item = make_item(f"mmlu_{lang}", row["id"], src, r.messages, gold=r.ground_truth,
                     n_options=r.n_choices, category=category)
    item.source = dict(row)
    return item


def load_items(root: Path, lang: str, ids: list[str], en_rows: dict[str, dict] | None = None,
               keep_gold_mismatch: bool = False) -> list[Item]:
    """Items for `ids` (in that order) from one dataset (Lite or full) in one language. With `keep_gold_mismatch`
    (the full split) a gold-mismatch row is read but never rendered: asking for one is an error (Amendment 4)."""
    rows = rows_by_id(root, lang, keep_gold_mismatch)
    if en_rows is None and lang != "en":
        en_rows = rows_by_id(root, "en", keep_gold_mismatch)
    missing = [i for i in ids if i not in rows]
    if missing:
        raise KeyError(f"MMLU-ProX {lang}: {len(missing)} ids missing, e.g. {missing[:3]}")
    bad = [i for i in ids if rows[i].get("gold_mismatch")]
    if bad:
        raise ValueError(f"MMLU-ProX {lang}: ids whose answer letter disagrees with answer_index cannot be items "
                         f"(Amendment 4): {bad[:3]}")
    return [_item(rows[i], lang, _english_category(rows[i], en_rows if lang != "en" else None)) for i in ids]


def load_lite(lang: str, ids: list[str], lite_dir: Path | None = None) -> list[Item]:
    """Lite items: strict, a gold-mismatch row stops the read (Amendment 4 keeps Lite strict)."""
    return load_items(_lite(lite_dir), lang, ids)


def load_full(lang: str, ids: list[str], full_dir: Path | None = None) -> list[Item]:
    """Items of the full test split (pilot, C1, peer check); gold-mismatch rows are read, never rendered."""
    return load_items(_full(full_dir), lang, ids, keep_gold_mismatch=True)


# --------------------------------------------------------------------------------------------------
# Full MMLU-ProX, non-Lite: pilot (8 + 8, parallel), Control C1 (100 EN), peer check (30 EN)
# --------------------------------------------------------------------------------------------------


def _question_key(question: str) -> str:
    return sha256_hex(" ".join(question.split()).lower())


def full_pool_ids(full_dir: Path | None = None, lite_dir: Path | None = None, seed: int = 36) -> list[str]:
    """Ids of the full test split present in both languages, not in Lite and with a consistent gold in both
    languages, in one seed-`seed` order.

    "Not in Lite" excludes Lite ids of either language and, in case the two datasets number questions
    differently, any full row whose normalised English question equals a Lite English question. A row whose answer
    letter disagrees with answer_index in EN or DE is left out (Amendment 4; `gold_mismatch_ids` lists them). The
    order key of an id does not depend on the other ids, so leaving one out shifts no other id's relative order.
    """
    full_dir, lite_dir = _full(full_dir), _lite(lite_dir)
    full_en = rows_by_id(full_dir, "en", keep_gold_mismatch=True)
    full_de = rows_by_id(full_dir, "de", keep_gold_mismatch=True)
    lite_en, lite_de = rows_by_id(lite_dir, "en"), rows_by_id(lite_dir, "de")
    lite_ids = set(lite_en) | set(lite_de)
    lite_questions = {_question_key(r["question"]) for r in lite_en.values()}
    mismatch = {i for rows in (full_en, full_de) for i, r in rows.items() if r.get("gold_mismatch")}
    pool = [i for i in sorted(set(full_en) & set(full_de))
            if i not in lite_ids and i not in mismatch and _question_key(full_en[i]["question"]) not in lite_questions]
    return seeded_order(pool, seed, "mmlu_prox_full")


def load_full_pilot(lang: str, n: int = 8, full_dir: Path | None = None, lite_dir: Path | None = None,
                    seed: int = 36) -> list[Item]:
    """Pilot items (non-Lite, parallel: the same ids in EN and DE), positions 0..n-1 of the pool."""
    return load_full(lang, full_pool_ids(full_dir, lite_dir, seed)[:n], full_dir)


def c1_items(full_dir: Path | None = None, lite_dir: Path | None = None, n: int = 100, offset: int = 8,
             seed: int = 36) -> list[Item]:
    """Control C1: n EN items of the non-Lite pool, after the pilot items (disjoint from them)."""
    return load_full("en", full_pool_ids(full_dir, lite_dir, seed)[offset:offset + n], full_dir)


def peer_check_items(full_dir: Path | None = None, lite_dir: Path | None = None, n: int = 30, offset: int = 108,
                     seed: int = 36) -> list[Item]:
    """Peer check greedy flip rate: n EN items of the non-Lite pool, after the pilot and C1 items."""
    return load_full("en", full_pool_ids(full_dir, lite_dir, seed)[offset:offset + n], full_dir)
