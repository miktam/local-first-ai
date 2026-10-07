"""runner/jsonl.py: append-only raw JSONL (BUILD_SPEC §5.4; §2 Append-only, Rights).

Torn line then resume (the next record parses); duplicates (the first
complete record counts); fsync per record; the S2 -> S3 hand-over (a cell
started in S2 keeps appending to its S2 file with a "resume" header); a torn
private line leaves no text of a withheld task anywhere in the repo tree.
"""

from __future__ import annotations

import json
import os

import pytest

from runner import jsonl


def _rec(item, pass_=0, **kw):
    r = {"type": "record", "key": {"arm": "K8", "task": "mmlu_en", "effort": "high", "item": item, "pass": pass_},
         "finish_reason": "stop", "truncated": False, "completion_tokens": 3}
    r.update(kw)
    return r


@pytest.fixture
def roots(tmp_path):
    repo = tmp_path / "repo"
    priv = tmp_path / "private"
    (repo / "aborted").mkdir(parents=True)
    priv.mkdir()
    return repo, priv


def test_header_then_records_sorted_keys_one_line_each(roots):
    repo, priv = roots
    p = repo / "results/raw/S2/K8/mmlu_en_high.jsonl"
    w = jsonl.open_run(p, {"cell": "K8:mmlu_en:high", "B": 8}, aborted_root=repo / "aborted", private_root=priv)
    w.append(_rec("a", text="line one\nline two"))
    w.close()
    lines = p.read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3
    hdr, rec = (json.loads(x) for x in lines[:2])
    assert hdr["type"] == "header" and hdr["B"] == 8 and "utc" in hdr
    assert list(rec) == sorted(rec)
    assert rec["text"] == "line one\nline two"


def test_fsync_is_called_per_record(roots, monkeypatch):
    repo, priv = roots
    calls = []
    real = os.fsync
    monkeypatch.setattr(jsonl.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    w = jsonl.open_run(repo / "x.jsonl", {"h": 1}, aborted_root=repo / "aborted", private_root=priv)
    n0 = len(calls)
    w.append(_rec("a"))
    w.append(_rec("b"))
    w.close()
    assert n0 >= 1 and len(calls) == n0 + 2


def test_torn_line_then_resume_next_record_parses(roots):
    repo, priv = roots
    p = repo / "results/raw/S2/K8/mmlu_en_high.jsonl"
    w = jsonl.open_run(p, {"h": 1}, aborted_root=repo / "aborted", private_root=priv)
    w.append(_rec("a"))
    w.close()
    with open(p, "ab") as f:  # a crash mid-write
        f.write(b'{"type": "record", "key": {"item": "b", "pa')
    w = jsonl.open_run(p, {"h": 2}, aborted_root=repo / "aborted", private_root=priv)
    w.append(_rec("b"))
    w.close()
    recs, bad = jsonl.scan(p)
    assert bad == 1
    assert [r["type"] for r in recs] == ["header", "record", "resume", "record"]
    assert jsonl.record_key(recs[-1]) == ("b", 0)
    assert "torn" in recs[2]["torn_line_note"]
    notes = list((repo / "aborted").glob("*/NOTE.md"))
    assert len(notes) == 1 and (notes[0].parent / "torn.part").read_bytes().startswith(b'{"type": "record"')
    assert jsonl.completed_keys("K8", "mmlu_en", "high", root=repo / "results/raw") == {("a", 0), ("b", 0)}


def test_duplicates_are_kept_and_the_first_complete_record_counts(roots):
    repo, priv = roots
    p = repo / "f.jsonl"
    w = jsonl.open_run(p, {"h": 1}, aborted_root=repo / "aborted", private_root=priv)
    w.append(_rec("a", completion_tokens=5))
    w.append({"type": "record", "key": {"item": "b", "pass": 0}})  # incomplete: no finish_reason
    w.append(_rec("a", completion_tokens=9))
    w.append(_rec("b", completion_tokens=2))
    w.close()
    recs, _ = jsonl.scan(p)
    first = jsonl.first_records(recs)
    assert [(jsonl.record_key(r), r["completion_tokens"]) for r in first] == [(("a", 0), 5), (("b", 0), 2)]
    assert sum(1 for r in recs if r.get("type") == "record") == 4  # nothing dropped from the file


def test_s2_to_s3_handover_appends_to_the_s2_file(roots, monkeypatch):
    from runner import run

    repo, priv = roots
    monkeypatch.setenv("EXP036_PRIVATE", str(priv))
    ctx = run.Ctx(repo)
    cell = {"arm": "K8", "task": "gpqa_en", "effort": "high", "pass": 0}
    r2, p2, s2 = run.cell_paths(ctx, "S2", cell)
    assert r2 == repo / "results/raw/S2/K8/gpqa_en_high.jsonl"
    # exp_037's private records live under $EXP036_PRIVATE/exp037 (runner.common.private_dir(); DESIGN §2.9).
    assert ctx.private == priv / "exp037"
    assert p2 == priv / "exp037/raw/S2/K8/gpqa_en_high.jsonl"
    w = jsonl.open_run(r2, {"session": "S2"}, aborted_root=repo / "aborted", private_root=ctx.private)
    w.append(_rec("q1"))
    w.close()
    r3, p3, s3 = run.cell_paths(ctx, "S3", cell)
    assert r3 == r2 and p3 == p2 and s3 == s2  # resumed in S3, still the S2 files
    w = jsonl.open_run(r3, {"session": "S3"}, aborted_root=repo / "aborted", private_root=ctx.private)
    w.append(_rec("q2"))
    w.close()
    recs, _ = jsonl.scan(r2)
    assert [r["type"] for r in recs] == ["header", "record", "resume", "record"]
    assert recs[2]["session"] == "S3"
    assert not (repo / "results/raw/S3/K8").exists()
    # A new cell started in S3 lives in S3.
    r_new, _, _ = run.cell_paths(ctx, "S3", {"arm": "K8", "task": "gpqa_de", "effort": "high", "pass": 0})
    assert r_new.parent == repo / "results/raw/S3/K8"
    # pass 2 (B10) is its own file
    assert run.cell_paths(ctx, "S3", {**cell, "pass": 1})[0].name == "gpqa_en_high_p1.jsonl"


def test_torn_private_line_leaves_no_withheld_text_in_the_repo(roots, monkeypatch):
    from runner.common import private_dir

    repo, priv = roots
    monkeypatch.setenv("EXP036_PRIVATE", str(priv))
    proot = private_dir()   # $EXP036_PRIVATE/exp037, as runner/run.py passes it (DESIGN §2.9)
    assert proot == priv / "exp037"
    secret = "SECRET WITHHELD QUESTION TEXT 7f3a"
    p = proot / "raw/S2/K8/gpqa_en_high.jsonl"
    w = jsonl.open_run(p, {"h": 1}, aborted_root=repo / "aborted", private_root=proot)
    w.append(_rec("q1", text="fine"))
    w.close()
    with open(p, "ab") as f:
        f.write(json.dumps({"type": "record", "text": secret + " and more"})[:-5].encode())
    w = jsonl.open_run(p, {"h": 2}, aborted_root=repo / "aborted", private_root=proot)
    w.close()
    # The torn bytes went to $EXP036_PRIVATE/exp037/aborted, the repo got a note with hash and size only.
    priv_copies = list((proot / "aborted").glob("*/torn.part"))
    assert not list((priv / "aborted").glob("*/torn.part"))   # nothing in exp_036's private aborted/
    assert len(priv_copies) == 1 and secret.encode() in priv_copies[0].read_bytes()
    notes = list((repo / "aborted").glob("*/NOTE.md"))
    assert len(notes) == 1
    note = notes[0].read_text()
    assert "sha256" in note and "Torn bytes" in note
    for f in repo.rglob("*"):
        if f.is_file():
            data = f.read_bytes()
            assert secret.encode() not in data, f
            assert b'"text"' not in data, f
    assert not list((repo / "aborted").glob("*/torn.part"))


def test_fallback_count_and_last_header(roots):
    repo, priv = roots
    p = repo / "c.jsonl"
    jsonl.open_run(p, {"B": 8, "b_fallback_from": None}, aborted_root=repo / "aborted", private_root=priv).close()
    jsonl.open_run(p, {"B": 4, "b_fallback_from": 8, "crash_fallback": True}, aborted_root=repo / "aborted",
                   private_root=priv).close()
    jsonl.open_run(p, {"B": 4, "b_fallback_from": 8, "crash_fallback": False}, aborted_root=repo / "aborted",
                   private_root=priv).close()  # a plain resume after the fallback
    assert jsonl.fallback_count(p) == 1
    assert jsonl.last_header(p)["B"] == 4
    assert jsonl.fallback_count(repo / "missing.jsonl") == 0
