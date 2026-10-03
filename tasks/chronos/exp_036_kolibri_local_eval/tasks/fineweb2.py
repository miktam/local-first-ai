"""FineWeb-2 deu_Latn test shard 000_00000 (af9c1333): documents for H5 and the gate texts T5, T6, T9.

BUILD_SPEC §5.5 `tasks/fineweb2.py`; HYPOTHESIS H5 (first 5,000 documents in file order), Phase 0 gate text
(T5–T6: the first two documents with ≥ 1,536 Kolibri tokens at file index ≥ 5,000; disjoint from H5).

No FineWeb text is ever written to the repo; callers keep it in memory or under $EXP036_WORK. pyarrow is
imported lazily; `reader` can be injected (tests) as any callable returning the `text` column in file order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

TEXT_COLUMN = "text"


def _pyarrow_texts(parquet: Path) -> Iterator[str]:
    import pyarrow.parquet as pq  # lazy

    pf = pq.ParquetFile(str(parquet))
    for batch in pf.iter_batches(columns=[TEXT_COLUMN], batch_size=1024):
        yield from batch.column(0).to_pylist()


def iter_docs(parquet: Path, start: int = 0, n: int = 5000,
              reader: Callable[[Path], Iterator[str]] | None = None) -> Iterator[tuple[int, str]]:
    """(file row index, text) for rows start … start+n-1, in file order. H5 uses start=0, n=5000."""
    if start < 0 or n < 0:
        raise ValueError("start and n must be non-negative")
    texts = (reader or _pyarrow_texts)(Path(parquet))
    stop = start + n
    for i, text in enumerate(texts):
        if i >= stop:
            break
        if i >= start:
            yield i, text


def gate_docs(parquet: Path, start: int = 5000, min_tokens: int = 1536, k: int = 2,
              count_tokens: Callable[[str], int] | None = None,
              reader: Callable[[Path], Iterator[str]] | None = None) -> list[tuple[int, str, int]]:
    """The first k documents at file index ≥ start with ≥ min_tokens tokens: [(index, text, n_tokens)].

    count_tokens (required) is the Kolibri tokenizer's count without special tokens, supplied by the caller
    (gate/build_gate_text.py), so this module never loads a tokenizer itself.
    """
    if count_tokens is None:
        raise ValueError("count_tokens is required (the Kolibri tokenizer's token count)")
    out = []
    texts = (reader or _pyarrow_texts)(Path(parquet))
    for i, text in enumerate(texts):
        if i < start:
            continue
        n_tok = count_tokens(text)
        if n_tok >= min_tokens:
            out.append((i, text, n_tok))
            if len(out) == k:
                break
    if len(out) < k:
        raise ValueError(f"only {len(out)} documents with ≥ {min_tokens} tokens at index ≥ {start}")
    return out


def iter_from(parquet: Path, start: int, reader: Callable[[Path], Iterator[str]] | None = None) -> Iterator[tuple[int, str]]:
    """(index, text) for every row at index ≥ start, in file order (T9 continues with these)."""
    texts = (reader or _pyarrow_texts)(Path(parquet))
    for i, text in enumerate(texts):
        if i >= start:
            yield i, text
