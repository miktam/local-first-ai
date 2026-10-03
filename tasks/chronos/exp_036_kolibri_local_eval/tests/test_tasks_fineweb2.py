"""tasks/fineweb2.py with an injected reader (no web text), plus a pyarrow round trip when available."""

from __future__ import annotations

import pytest

from tasks import fineweb2

DOCS = [f"synthetic doc {i} " + "wort " * (i % 7) * 300 for i in range(40)]


def reader(_path):
    yield from DOCS


def count(text):
    return len(text.split())


def test_iter_docs_window():
    got = list(fineweb2.iter_docs("x.parquet", start=5, n=3, reader=reader))
    assert got == [(5, DOCS[5]), (6, DOCS[6]), (7, DOCS[7])]
    assert len(list(fineweb2.iter_docs("x.parquet", start=0, n=5000, reader=reader))) == 40


def test_gate_docs_first_k_long_enough_at_or_after_start():
    got = fineweb2.gate_docs("x.parquet", start=10, min_tokens=1500, k=2, count_tokens=count, reader=reader)
    idx = [i for i, _, _ in got]
    expected = [i for i in range(10, 40) if count(DOCS[i]) >= 1500][:2]
    assert idx == expected and all(n >= 1500 for _, _, n in got)
    with pytest.raises(ValueError):
        fineweb2.gate_docs("x.parquet", start=10, min_tokens=10**6, k=2, count_tokens=count, reader=reader)
    with pytest.raises(ValueError, match="count_tokens"):
        fineweb2.gate_docs("x.parquet", reader=reader)


def test_iter_from():
    assert [i for i, _ in fineweb2.iter_from("x.parquet", 37, reader=reader)] == [37, 38, 39]


def test_pyarrow_round_trip(tmp_path):
    pa = pytest.importorskip("pyarrow", reason="pyarrow not installed in this environment")
    import pyarrow.parquet as pq

    p = tmp_path / "t.parquet"
    pq.write_table(pa.table({"text": DOCS, "id": list(range(len(DOCS)))}), p, row_group_size=7)
    assert list(fineweb2.iter_docs(p, start=3, n=9)) == list(fineweb2.iter_docs("x", start=3, n=9, reader=reader))
