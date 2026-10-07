"""AIME 2026 EN (math-ai/aime26, V row) and DE (ellamind/aime26-multilingual `deu`, R row).

BUILD_SPEC §5.5 `tasks/aime.py`; HYPOTHESIS H2 secondary rows (B1), E4, Tier B B1/B2.

- EN: eval-framework AIME2026 — Generative kind, prompt `_AIME_QUERY_TEMPLATE.format(Question=problem)`, gold
  `answer`. Public: manifests may carry the gold integer; raw outputs are published.
- DE: tasks/prompts/aime_de.txt, our German translation of the same NeMo-Skills wrapper, filled the same way.
  Withheld (no licence stated): no problem text, gold or raw output leaves $EXP036_PRIVATE.
Both are scored with the vendored boxed extractor (scorers/aime.py).
"""

from __future__ import annotations

from pathlib import Path

from tasks.common import Item, default_path, make_item, read_split, require_unique, seeded_order, user_message
from tasks.vendored_evalfw import shim

EN_REPO = "math-ai/aime26"
DE_REPO = "ellamind/aime26-multilingual"
PROMPTS = Path(__file__).resolve().parent / "prompts"
AIME_DE_TEMPLATE = PROMPTS / "aime_de.txt"
EN_SPLIT = "test"  # eval-framework aime2026: sample_split="test"
DE_CONFIG = "deu"
ID_FIELDS = ("id", "problem_id", "ID")


def _rows(path: Path, config: str | None, split: str | None) -> list[dict]:
    try:
        return read_split(path, config=config, split=split)
    except FileNotFoundError:
        if split is None:
            raise
        return read_split(path, config=config)


def _with_ids(rows: list[dict], what: str) -> tuple[list[dict], str]:
    id_field = next((f for f in ID_FIELDS if all(f in r and r[f] not in (None, "") for r in rows)), None)
    if id_field is None:
        for i, r in enumerate(rows):
            r["_exp036_row_index"] = str(i)
        id_field = "_exp036_row_index"
    require_unique([str(r[id_field]) for r in rows], what)
    return rows, id_field


def gold_int(answer) -> int:
    """AIME answers are integers 0–999; a non-integer gold is a data error."""
    s = str(answer).strip()
    if not s.isdigit() or not 0 <= int(s) <= 999:
        raise ValueError(f"AIME gold {answer!r} is not an integer in 0..999")
    return int(s)


def rendered_en(row: dict) -> shim.Rendered:
    kind, _answer, _id = shim.aime2026()
    return shim.render(kind, row)


def render_en(row_or_item) -> list[dict]:
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return rendered_en(row).messages


def de_template() -> str:
    text = AIME_DE_TEMPLATE.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def render_de(row_or_item) -> list[dict]:
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return user_message(de_template().format(Question=row["problem"]))


def load_en(path: Path | None = None) -> list[Item]:
    path = Path(path) if path is not None else default_path(EN_REPO)
    rows, id_field = _with_ids(_rows(path, None, EN_SPLIT), "AIME26 EN")
    items = []
    for r in rows:
        if "problem" not in r or "answer" not in r:
            raise KeyError(f"AIME26 EN row lacks problem/answer: {sorted(r)}")
        rendered = rendered_en(r)
        gold = gold_int(r["answer"])
        it = make_item("aime_en", r[id_field], {"problem": r["problem"], "answer": str(r["answer"])},
                       rendered.messages, gold=gold)
        it.public["id_source"] = id_field
        it.source = dict(r)
        items.append(it)
    return items


def load_de(path: Path | None = None) -> list[Item]:
    path = Path(path) if path is not None else default_path(DE_REPO)
    rows, id_field = _with_ids(_rows(path, DE_CONFIG, EN_SPLIT), "AIME26 DE")
    items = []
    for r in rows:
        if "problem" not in r and "question" in r:
            r = dict(r, problem=r["question"])
        if "problem" not in r or "answer" not in r:
            raise KeyError(f"AIME26 DE row lacks problem/answer: {sorted(r)}")
        it = make_item("aime_de", r[id_field], {"problem": r["problem"], "answer": str(r["answer"])},
                       render_de(r), gold=gold_int(r["answer"]))
        it.public["id_source"] = id_field
        it.source = dict(r)
        items.append(it)
    return items


def pilot_en(items: list[Item], n: int = 4, seed: int = 36) -> list[Item]:
    """The AIME EN pilot items (K8 only; outputs discarded, C19)."""
    by_id = {it.id: it for it in items}
    return [by_id[i] for i in seeded_order(by_id, seed, "aime_pilot_en")[:n]]
