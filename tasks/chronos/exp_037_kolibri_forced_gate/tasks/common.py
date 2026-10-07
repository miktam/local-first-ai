"""Shared helpers for the task loaders (BUILD_SPEC §5.5): hashing, seeded orders, local file reading.

Conventions used by every loader and by build_manifests.py:
- `canonical_json(x)`: UTF-8 JSON, sorted keys, no whitespace, non-ASCII kept. All sha256 below are over it.
- `item_sha256`: sha256 of canonical_json of the source fields the loader reads for that item (question,
  options, gold, ...; for RGB the query and answers, not the documents, which are bound by prompt_sha256).
- `prompt_sha256`: sha256 of canonical_json of the rendered message list [{"role", "content"}, ...], i.e.
  the task prompt before any chat template. The runner's record carries the rendered-template sha
  separately.
- Seeded orders (`seeded_order`): ids sorted by sha256("exp036|<pool>|<seed>|<id>"). This is independent
  of any RNG implementation, so the same ids in give the same order on every host and Python version.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def json_sha256(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def user_message(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


def seeded_order(ids: Iterable[Any], seed: int, pool: str) -> list[str]:
    """The ids (as strings) in the frozen seeded order of `pool` (see the module docstring)."""
    ids = [str(i) for i in ids]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate ids in pool {pool!r}")
    return sorted(ids, key=lambda i: hashlib.sha256(f"exp036|{pool}|{seed}|{i}".encode("utf-8")).hexdigest())


@dataclass
class Item:
    """One task item in memory. `source` (the raw row) and `messages`/`gold` of a withheld set never
    leave memory except into $EXP036_PRIVATE; `public` holds the extra fields a public manifest may carry."""

    id: str
    task: str
    item_sha256: str
    messages: list[dict]
    prompt_sha256: str
    gold: Any = None
    n_options: int | None = None
    category: str | None = None
    public: dict = field(default_factory=dict)  # extra fields a public manifest may carry
    private: dict = field(default_factory=dict, repr=False)  # extra fields for the private manifest only
    source: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        """The runner's item dict (runner/generate.py: id, item_sha256, prompt_sha256, messages)."""
        return {"id": self.id, "task": self.task, "item_sha256": self.item_sha256,
                "prompt_sha256": self.prompt_sha256, "messages": self.messages}

    @property
    def prompt_text(self) -> str:
        """The user turn (every exp_036 prompt has exactly one)."""
        users = [m["content"] for m in self.messages if m["role"] == "user"]
        if len(users) != 1:
            raise ValueError(f"item {self.id}: expected one user message, found {len(users)}")
        return users[0]


def make_item(task: str, item_id: Any, source_fields: dict, messages: list[dict], **kw) -> Item:
    return Item(
        id=str(item_id),
        task=task,
        item_sha256=json_sha256(source_fields),
        messages=messages,
        prompt_sha256=json_sha256(messages),
        source=source_fields,
        **kw,
    )


def default_path(repo: str) -> Path:
    """The local path of an assets.json entry on this host (tasks/assets.py; EXP036_* environment)."""
    from tasks import assets

    return assets.asset(repo).path


# --------------------------------------------------------------------------------------------------
# Local data files
# --------------------------------------------------------------------------------------------------

DATA_SUFFIXES = (".parquet", ".jsonl", ".json", ".csv")

# pandas.read_csv's default NA strings (pandas >= 2.0, incl. "None"). Hugging Face `datasets` loads a CSV
# through pandas with these defaults, so a field that is exactly one of them reaches eval-framework as None
# (gpqa.py's `_preprocess` turns a None option into " "). `read_rows(..., csv_pandas_na=True)` reproduces
# that for the CSV-backed GPQA EN files.
PANDAS_NA_STRINGS = frozenset({
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
    "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null",
})


def read_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq  # lazy: only parquet inputs need it
    except ModuleNotFoundError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(f"pyarrow is required to read {path.name} (pip install pyarrow, pinned in env/)") from e
    return pq.read_table(path).to_pylist()


def read_rows(path: Path, csv_pandas_na: bool = False) -> list[dict]:
    """Rows of one local data file, in file order: .parquet, .jsonl, .json (array or JSON lines), .csv.

    CSV fields are strings (a UTF-8 BOM is dropped); with csv_pandas_na a field exactly equal to one of
    PANDAS_NA_STRINGS becomes None, as Hugging Face `datasets` (pandas) would load it."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return read_parquet(path)
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        if csv_pandas_na:
            rows = [{k: (None if v in PANDAS_NA_STRINGS else v) for k, v in r.items()} for r in rows]
        return rows
    if suffix in (".jsonl", ".json"):
        text = path.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if suffix == ".json" and stripped.startswith("["):
            return list(json.loads(text))
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raise ValueError(f"unsupported data file type: {path.name}")


def find_data_files(root: Path, *, config: str | None = None, split: str | None = None) -> list[Path]:
    """Data files of a local Hugging Face dataset snapshot, sorted by relative path.

    A file qualifies when its suffix is a data suffix, it is not under a dot-directory (e.g. .cache), and
    the config name and the split name (each if given) are each either a parent directory name or a token
    of the file stem split at "-", "_" and "." (e.g. `de/test-00000-of-00001.parquet`, `test.jsonl`,
    `deu/train-00000-of-00001.parquet`).
    """
    root = Path(root)
    if root.is_file():
        return [root]
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in DATA_SUFFIXES:
            continue
        rel = p.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        dirs = set(rel.parts[:-1])
        tokens = set(re.split(r"[-_.]", p.stem))
        if config is not None and config not in dirs and config not in tokens:
            continue
        if split is not None and split not in dirs and split not in tokens:
            continue
        out.append(p)
    return out


def read_split(root: Path, *, config: str | None = None, split: str | None = None) -> list[dict]:
    """All rows of one (config, split) of a local dataset snapshot; a hard error if no file matches."""
    files = find_data_files(root, config=config, split=split)
    if not files:
        raise FileNotFoundError(f"no data file for config={config!r} split={split!r} under {Path(root).name}/")
    rows: list[dict] = []
    for f in files:
        rows.extend(read_rows(f))
    return rows


def require_unique(ids: list[str], what: str) -> None:
    seen = set()
    dups = sorted({i for i in ids if i in seen or seen.add(i)})
    if dups:
        raise ValueError(f"{what}: {len(dups)} duplicate ids, e.g. {dups[:3]}")
