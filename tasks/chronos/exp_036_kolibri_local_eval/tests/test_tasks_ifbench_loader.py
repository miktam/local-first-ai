"""tasks/ifbench.py on a 3-row synthetic file, plus a shape check of the local IFBench data if present."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import tasks_synthetic as syn
from tasks import ifbench
from tasks.common import json_sha256


def test_three_rows(tmp_path):
    d = syn.write_ifbench(tmp_path, n=3)
    items = ifbench.load(d)
    assert [it.id for it in items] == ["0", "1", "2"]
    rows = syn.ifbench_rows(3)
    for it, row in zip(items, rows):
        assert it.messages == [{"role": "user", "content": row["prompt"]}]  # verbatim
        assert it.gold is None and it.task == "ifbench"
        assert it.source["kwargs"] == ifbench.clean_kwargs(row["kwargs"])
        assert all(v is not None for kw in it.source["kwargs"] for v in kw.values())
        assert it.item_sha256 == json_sha256(it.source)


def test_null_kwargs_do_not_change_the_hash(tmp_path):
    rows = syn.ifbench_rows(2)
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    import json

    a.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    stripped = [dict(r, kwargs=ifbench.clean_kwargs(r["kwargs"])) for r in rows]
    b.write_text("".join(json.dumps(r) + "\n" for r in stripped), encoding="utf-8")
    assert [it.item_sha256 for it in ifbench.load(a)] == [it.item_sha256 for it in ifbench.load(b)]


def test_duplicate_keys_refused(tmp_path):
    rows = syn.ifbench_rows(2)
    rows[1]["key"] = rows[0]["key"]
    p = tmp_path / "dup.jsonl"
    import json

    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        ifbench.load(p)


def test_pilot_is_a_seeded_subset(tmp_path):
    items = ifbench.load(syn.write_ifbench(tmp_path, n=12))
    p = ifbench.pilot(items, n=8)
    assert len(p) == 8 and [i.id for i in p] == [i.id for i in ifbench.pilot(items, n=8)]


def _data_dir() -> Path | None:
    for c in (os.environ.get("EXP036_DATA"), str(Path.home() / "models/exp036-mini/data")):
        if c and Path(c).expanduser().is_dir():
            return Path(c).expanduser()
    return None


def test_local_ifbench_data_shape():
    d = _data_dir()
    if d is None or not (d / "IFBench_test").is_dir():
        pytest.skip("no local IFBench_test copy (set EXP036_DATA)")
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        pytest.skip("pyarrow not installed: the IFBench_test parquet cannot be read in this environment")
    items = ifbench.load(d / "IFBench_test")
    assert len(items) == 300 and len({it.id for it in items}) == 300
