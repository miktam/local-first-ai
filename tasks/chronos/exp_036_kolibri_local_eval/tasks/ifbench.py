"""IFBench (allenai/IFBench_test @ 2e8a48de): loader and renderer.

BUILD_SPEC §5.5 `tasks/ifbench.py`; HYPOTHESIS H2 (IFBench loose-prompt, R), H4, H7.

300 items (key, prompt, instruction_id_list, kwargs). The prompt is the user message, verbatim. Public data
(ODC-BY 1.0): manifests carry ids and hashes; raw outputs are published. Scoring is the official checker
(scorers/ifbench_adapter.py, in the IFBench venv on the mini).

`kwargs` comes from parquet as one struct per instruction holding the union of all keys, with nulls for the
keys an instruction does not take; the null entries are dropped, which gives the same dicts as IFBench's
own JSONL. The item hash is over the cleaned record, so a parquet and a JSONL copy of the same data agree.
"""

from __future__ import annotations

from pathlib import Path

from tasks.common import Item, default_path, find_data_files, make_item, read_rows, require_unique, seeded_order, user_message

REPO = "allenai/IFBench_test"
FIELDS = ("key", "prompt", "instruction_id_list", "kwargs")


def clean_kwargs(kwargs) -> list[dict]:
    if kwargs is None:
        return []
    return [{k: v for k, v in (kw or {}).items() if v is not None} for kw in kwargs]


def data_file(path: Path | None = None) -> Path:
    """The IFBench_test data file: the dataset's single parquet (or one JSONL file)."""
    path = Path(path) if path is not None else default_path(REPO)
    if path.is_file():
        return path
    files = find_data_files(path)
    if len(files) != 1:
        raise FileNotFoundError(f"IFBench_test: expected one data file under {path.name}/, found {len(files)}")
    return files[0]


def render(row_or_item) -> list[dict]:
    row = row_or_item.source if isinstance(row_or_item, Item) else row_or_item
    return user_message(row["prompt"])


def load(path: Path | None = None) -> list[Item]:
    rows = read_rows(data_file(path))
    items = []
    for r in rows:
        missing = [k for k in FIELDS if k not in r]
        if missing:
            raise KeyError(f"IFBench row lacks {missing}")
        src = {"key": str(r["key"]), "prompt": r["prompt"],
               "instruction_id_list": list(r["instruction_id_list"]), "kwargs": clean_kwargs(r["kwargs"])}
        it = make_item("ifbench", src["key"], src, render(r))
        it.public["n_instructions"] = len(src["instruction_id_list"])
        items.append(it)
    require_unique([it.id for it in items], "IFBench keys")
    return items


def pilot(items: list[Item], n: int = 8, seed: int = 36) -> list[Item]:
    """IFBench pilot items (outputs discarded, C19)."""
    by_id = {it.id: it for it in items}
    return [by_id[i] for i in seeded_order(by_id, seed, "ifbench_pilot")[:n]]
