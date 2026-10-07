"""GPQA Diamond EN (Idavidrein/gpqa) and DE (ellamind/gpqa-multilingual): loaders and renderers.

BUILD_SPEC §5.5 `tasks/gpqa.py`; HYPOTHESIS "Task sets", H2 rows GPQA Diamond EN (V) and DE (V), C14, C19.

The CSV is read the way eval-framework's `load_dataset` reads it (pandas defaults: exact NA strings become
None, which upstream's `_preprocess` renders as " ").

Withheld: no item text, option text or gold ever leaves memory except into $EXP036_PRIVATE. Prompts are
built by eval-framework's own composition (tasks/vendored_evalfw/shim.py):
- EN: GPQA_DIAMOND_COT_V2 — GpqaReader (option shuffle seeded by sha256 of the option texts) + tulu3_cot_prompt.
- DE: GPQA_ELLAMIND_DIAMOND_COT_DE — GpqaReader (shuffle_correct_with_distractors, seed = question + answer)
  + tulu3_cot_prompt_de.

The EN primary set is all 198 Diamond items (Amendment 3, 2026-10-04). eval-framework's dataset filter drops
one over-long question, identified here only by the sha256 constant of the patched vendored gpqa.py; that question
is in gpqa_extended.csv only, not in gpqa_diamond.csv or gpqa_main.csv, so the filter removes no Diamond item and
the vendor's GPQA_DIAMOND_COT scored 198 too. The filter stays in force and `primary_en` asserts the number it
excludes (0), so a change in the data cannot slip through. The pre-registered 197-primary / 198-sensitivity split
(HYPOTHESIS C14) assumed the question was in Diamond; Amendment 3 drops that sensitivity check.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tasks.common import Item, default_path, make_item, read_rows, read_split, require_unique, seeded_order
from tasks.vendored_evalfw import shim

EN_REPO = "Idavidrein/gpqa"
DE_REPO = "ellamind/gpqa-multilingual"
EN_FIELDS = ("Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
DE_FIELDS = ("question", "correct_answer", "incorrect_answers", "is_diamond")
# Candidate id columns of the German dataset, in order of preference; the first one present is used and
# recorded in the manifest as `id_source`. Without any, the 0-based row index in file order is the id.
DE_ID_FIELDS = ("Record ID", "record_id", "id", "question_id", "original_id")
DE_CONFIG = "deu"
DE_SPLIT = "train"


def _csv(root: Path | None, name: str) -> Path:
    root = Path(root) if root is not None else default_path(EN_REPO)
    if root.is_file():
        if root.name != name:
            raise FileNotFoundError(f"expected {name}, got {root.name}")
        return root
    direct = root / name
    if direct.is_file():
        return direct
    hits = [p for p in sorted(root.rglob(name)) if not any(s.startswith(".") for s in p.relative_to(root).parts)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: {len(hits)} matches under {root.name}/")
    return hits[0]


def _en_source(row: dict) -> dict:
    missing = [k for k in EN_FIELDS + ("Record ID",) if k not in row]
    if missing:
        raise KeyError(f"GPQA EN row lacks {missing}")
    return {k: row[k] for k in EN_FIELDS}


def rendered_en(row: dict) -> shim.Rendered:
    """GPQA_DIAMOND_COT_V2, 0-shot: messages, the gold letter after the option shuffle, the option count."""
    kind, _answer, _id = shim.gpqa_diamond_cot_v2()
    return shim.render(kind, row, subject_label="gpqa_diamond")


def rendered_de(row: dict) -> shim.Rendered:
    """GPQA_ELLAMIND_DIAMOND_COT_DE, 0-shot."""
    kind, _answer, _id = shim.gpqa_ellamind_diamond_cot_de()
    return shim.render(kind, row, subject_label=DE_CONFIG)


def render_en(row_or_item) -> list[dict]:
    """The message list of a GPQA EN row (or of an Item, from its source row)."""
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return rendered_en(row).messages


def render_de(row_or_item) -> list[dict]:
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return rendered_de(row).messages


def _en_item(row: dict, task: str) -> Item:
    r = rendered_en(row)
    src = _en_source(row)
    item = make_item(task, row["Record ID"], src, r.messages, gold=r.ground_truth, n_options=r.n_choices)
    item.source = dict(row)
    return item


def en_rows(path: Path | None = None, name: str = "gpqa_diamond.csv") -> list[dict]:
    """Rows of a GPQA CSV as eval-framework sees them (loaded through Hugging Face `datasets`, i.e. pandas:
    a field that is exactly a pandas NA string such as "None" or "" arrives as None)."""
    rows = read_rows(_csv(path, name), csv_pandas_na=True)
    for row in rows:
        _en_source(row)
    require_unique([r["Record ID"] for r in rows], name)
    return rows


def load_diamond_en(path: Path | None = None, task: str = "gpqa_en") -> list[Item]:
    """All rows of gpqa_diamond.csv (198), in file order."""
    return [_en_item(row, task) for row in en_rows(path)]


def diamond_record_ids(path: Path | None = None) -> set[str]:
    return {r["Record ID"] for r in en_rows(path)}


def is_overlong(row: dict, overlong_sha256: str | None = None) -> bool:
    """eval-framework's exclusion predicate, from the vendored (patched) gpqa.py filter. A different hash can
    be passed for tests on synthetic rows; the predicate is then the same comparison."""
    if overlong_sha256 is None:
        return not shim.gpqa_overlong_filter()(row)
    return hashlib.sha256(row["Question"].encode()).hexdigest() == overlong_sha256


# Amendment 3 (2026-10-04): the number of gpqa_diamond.csv rows the vendored over-long filter excludes.
DIAMOND_OVERLONG_EXCLUDED = 0


def primary_en(items: list[Item], overlong_sha256: str | None = None,
               expect_excluded: int = DIAMOND_OVERLONG_EXCLUDED) -> list[Item]:
    """The eval-framework set: the Diamond items the vendored over-long filter keeps (by hash only).

    Amendment 3: the filter excludes no Diamond item, so this is all 198 items, and exactly `expect_excluded`
    (default 0) items may go; any other count is an error, so a change in the data cannot slip through.
    Upstream compares the raw `Question`; its text has no surrounding whitespace, so the stripped question gives
    the same hash (BUILD_SPEC wording). If a row matched only after stripping, upstream would not drop it, so that
    case is an error rather than a silent choice.
    """
    target = overlong_sha256 or shim.overlong_question_sha256()
    keep, dropped = [], []
    for it in items:
        q = it.source["Question"]
        raw_hit = is_overlong(it.source, overlong_sha256)
        strip_hit = hashlib.sha256(q.strip().encode()).hexdigest() == target
        if strip_hit and not raw_hit:
            raise ValueError(f"item {it.id}: matches the over-long hash only after stripping; upstream would keep it")
        (dropped if raw_hit else keep).append(it)
    if len(dropped) != expect_excluded:
        raise ValueError(f"expected {expect_excluded} over-long item(s) excluded, found {len(dropped)}")
    return keep


# --------------------------------------------------------------------------------------------------
# German GPQA (ellamind)
# --------------------------------------------------------------------------------------------------


def _truthy(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        if v.strip().lower() in ("true", "1", "yes"):
            return True
        if v.strip().lower() in ("false", "0", "no", ""):
            return False
    raise ValueError(f"is_diamond value {v!r} is not a boolean")


def de_rows(path: Path | None = None) -> tuple[list[dict], str]:
    """All rows of the deu config (train split, as eval-framework samples it) and the id field used."""
    path = Path(path) if path is not None else default_path(DE_REPO)
    try:
        rows = read_split(path, config=DE_CONFIG, split=DE_SPLIT)
    except FileNotFoundError:
        rows = read_split(path, config=DE_CONFIG)
    for row in rows:
        missing = [k for k in DE_FIELDS if k not in row]
        if missing:
            raise KeyError(f"German GPQA row lacks {missing}")
    id_field = next((f for f in DE_ID_FIELDS if all(f in r and r[f] not in (None, "") for r in rows)), None)
    if id_field is None:
        for i, row in enumerate(rows):
            row["_exp036_row_index"] = str(i)
        id_field = "_exp036_row_index"
    require_unique([str(r[id_field]) for r in rows], "German GPQA ids")
    return rows, id_field


def _de_source(row: dict) -> dict:
    return {
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "incorrect_answers": list(row["incorrect_answers"]),
    }


def _de_item(row: dict, id_field: str, task: str) -> Item:
    r = rendered_de(row)
    item = make_item(task, row[id_field], _de_source(row), r.messages, gold=r.ground_truth, n_options=r.n_choices)
    item.public["id_source"] = id_field
    item.source = dict(row)
    return item


def load_diamond_de(path: Path | None = None, task: str = "gpqa_de") -> list[Item]:
    """The `is_diamond` rows of the deu config (198), in file order."""
    rows, id_field = de_rows(path)
    return [_de_item(r, id_field, task) for r in rows if _truthy(r["is_diamond"])]


# --------------------------------------------------------------------------------------------------
# Pilot (HYPOTHESIS "Pilot"; C19: disjoint from the Diamond items)
# --------------------------------------------------------------------------------------------------


def load_pilot_main(path: Path | None, diamond_ids: set[str], n: int = 8, seed: int = 36, lang: str = "en",
                    overlong_sha256: str | None = None) -> list[Item]:
    """n non-Diamond items in the seed-`seed` order.

    EN: rows of gpqa_main.csv whose Record ID is not in gpqa_diamond.csv (and not the over-long question,
    which eval-framework drops from every config; on the pinned data it is in gpqa_extended.csv only, so this
    removes nothing, Amendment 3). DE: `is_diamond` false rows of the deu config, also
    excluding any row whose id is a Diamond Record ID when the German data carries Record IDs.
    """
    if lang == "en":
        rows = [r for r in en_rows(path, "gpqa_main.csv")
                if r["Record ID"] not in diamond_ids and not is_overlong(r, overlong_sha256)]
        by_id = {r["Record ID"]: r for r in rows}
        order = seeded_order(by_id, seed, "gpqa_pilot_en")[:n]
        return [_en_item(by_id[i], "gpqa_en") for i in order]
    if lang == "de":
        rows, id_field = de_rows(path)
        rows = [r for r in rows if not _truthy(r["is_diamond"]) and str(r[id_field]) not in diamond_ids]
        by_id = {str(r[id_field]): r for r in rows}
        order = seeded_order(by_id, seed, "gpqa_pilot_de")[:n]
        return [_de_item(by_id[i], id_field, "gpqa_de") for i in order]
    raise ValueError(f"lang must be 'en' or 'de', not {lang!r}")
