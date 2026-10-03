"""MMLU-ProX-Lite EN/DE (main rows) and MMLU-ProX full (pilot, C1, peer check, category weights).

BUILD_SPEC §5.5 `tasks/mmlu_prox.py`; HYPOTHESIS H2 (MMLU rows, post-stratified), H3, H7, C14, C19, Control C1.

Public data (MIT): manifests may carry id, category and gold letter, never the text.

Rows. MMLU-ProX stores the options either as a list `options` or as columns `option_0` … `option_9`. Questions
with fewer than ten options pad the trailing columns; trailing options that are null, empty or "N/A" are
dropped (selection_rules.json `mmlu_prox.options_rule`). `answer_index` must point inside the options and agree
with the `answer` letter. Categories are the 14 English MMLU-Pro names; a German row's category is taken from
the English row with the same `question_id` when it is not one of them.

Prompts (C10, R rows):
- EN: eval-framework MMLU_PRO_COT_V2 — subject preamble "The following are multiple choice questions (with
  answers) about <category>." folded into the user turn, then tulu3_cot_prompt; extractor tulu_answer_v2(10).
- DE: the vendored Cot kind with MmluProReader and tulu3_cot_prompt_de, no preamble
  (tasks/prompts/mmlu_prox_de_note.md); the German extractor is extended to A–J in scorers/mc.py.

Selection (selection_rules.json): ids present in both languages, per category in one seed-36 order; n_M
(a multiple of 14) takes the first n_M/14 ids of every category, so smaller sets are nested and balanced.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

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


def normalise_row(row: dict) -> dict:
    """{id, question, options, answer_index, category} from an MMLU-ProX row (see the module docstring)."""
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
    letter = row.get("answer")
    if isinstance(letter, str) and len(letter.strip()) == 1 and letter.strip() in LETTERS and letter.strip() != LETTERS[idx]:
        raise ValueError(f"MMLU-ProX {row['question_id']}: answer {letter!r} disagrees with answer_index {idx}")
    return {
        "id": str(row["question_id"]),
        "question": row["question"],
        "options": [str(o) for o in options],
        "answer_index": idx,
        "category": row["category"],
    }


def _lite(d: Path | None) -> Path:
    return Path(d) if d is not None else default_path(LITE_REPO)


def _full(d: Path | None) -> Path:
    return Path(d) if d is not None else default_path(FULL_REPO)


def rows_by_id(root: Path, lang: str) -> dict[str, dict]:
    rows = [normalise_row(r) for r in read_split(root, config=lang, split=SPLIT)]
    require_unique([r["id"] for r in rows], f"MMLU-ProX {lang} {SPLIT}")
    return {r["id"]: r for r in rows}


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


def select_ids(per_category: dict[str, list[str]], n_m: int) -> list[str]:
    """The n_M set: n_M/14 ids per category (a prefix of each category's order), interleaved by rank."""
    cats = list(per_category)
    if n_m % len(cats):
        raise ValueError(f"n_M={n_m} is not a multiple of {len(cats)}")
    k = n_m // len(cats)
    short = [c for c in cats if len(per_category[c]) < k]
    if short:
        raise ValueError(f"n_M={n_m}: categories with fewer than {k} parallel ids: {short}")
    return [per_category[c][r] for r in range(k) for c in cats]


def category_counts(full_dir: Path | None = None, lang: str = "en") -> dict[str, int]:
    """Items per category in the full MMLU-ProX test split of `lang` (post-stratification weights, H2)."""
    full_dir = _full(full_dir)
    rows = rows_by_id(full_dir, lang)
    en_rows = rows if lang == "en" else rows_by_id(full_dir, "en")
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


def load_items(root: Path, lang: str, ids: list[str], en_rows: dict[str, dict] | None = None) -> list[Item]:
    """Items for `ids` (in that order) from one dataset (Lite or full) in one language."""
    rows = rows_by_id(root, lang)
    if en_rows is None and lang != "en":
        en_rows = rows_by_id(root, "en")
    missing = [i for i in ids if i not in rows]
    if missing:
        raise KeyError(f"MMLU-ProX {lang}: {len(missing)} ids missing, e.g. {missing[:3]}")
    return [_item(rows[i], lang, _english_category(rows[i], en_rows if lang != "en" else None)) for i in ids]


def load_lite(lang: str, ids: list[str], lite_dir: Path | None = None) -> list[Item]:
    return load_items(_lite(lite_dir), lang, ids)


# --------------------------------------------------------------------------------------------------
# Full MMLU-ProX, non-Lite: pilot (8 + 8, parallel), Control C1 (100 EN), peer check (30 EN)
# --------------------------------------------------------------------------------------------------


def _question_key(question: str) -> str:
    return sha256_hex(" ".join(question.split()).lower())


def full_pool_ids(full_dir: Path | None = None, lite_dir: Path | None = None, seed: int = 36) -> list[str]:
    """Ids of the full test split present in both languages and not in Lite, in one seed-`seed` order.

    "Not in Lite" excludes Lite ids of either language and, in case the two datasets number questions
    differently, any full row whose normalised English question equals a Lite English question.
    """
    full_dir, lite_dir = _full(full_dir), _lite(lite_dir)
    full_en, full_de = rows_by_id(full_dir, "en"), rows_by_id(full_dir, "de")
    lite_en, lite_de = rows_by_id(lite_dir, "en"), rows_by_id(lite_dir, "de")
    lite_ids = set(lite_en) | set(lite_de)
    lite_questions = {_question_key(r["question"]) for r in lite_en.values()}
    pool = [i for i in sorted(set(full_en) & set(full_de))
            if i not in lite_ids and _question_key(full_en[i]["question"]) not in lite_questions]
    return seeded_order(pool, seed, "mmlu_prox_full")


def load_full_pilot(lang: str, n: int = 8, full_dir: Path | None = None, lite_dir: Path | None = None,
                    seed: int = 36) -> list[Item]:
    """Pilot items (non-Lite, parallel: the same ids in EN and DE), positions 0..n-1 of the pool."""
    return load_items(_full(full_dir), lang, full_pool_ids(full_dir, lite_dir, seed)[:n])


def c1_items(full_dir: Path | None = None, lite_dir: Path | None = None, n: int = 100, offset: int = 8,
             seed: int = 36) -> list[Item]:
    """Control C1: n EN items of the non-Lite pool, after the pilot items (disjoint from them)."""
    return load_items(_full(full_dir), "en", full_pool_ids(full_dir, lite_dir, seed)[offset:offset + n])


def peer_check_items(full_dir: Path | None = None, lite_dir: Path | None = None, n: int = 30, offset: int = 108,
                     seed: int = 36) -> list[Item]:
    """Peer check greedy flip rate: n EN items of the non-Lite pool, after the pilot and C1 items."""
    return load_items(_full(full_dir), "en", full_pool_ids(full_dir, lite_dir, seed)[offset:offset + n])
