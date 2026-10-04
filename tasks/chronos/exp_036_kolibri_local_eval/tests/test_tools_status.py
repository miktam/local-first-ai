"""tools/status.py (BUILD_SPEC §5.9): only the Status line changes in place;
run-record blocks are appended; amendment sync is ordered and idempotent."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools import status

EXP = Path(__file__).resolve().parents[1]


@pytest.fixture
def exp(tmp_path) -> Path:
    d = tmp_path / "exp"
    d.mkdir()
    shutil.copy(EXP / "HYPOTHESIS.md", d / "HYPOTHESIS.md")
    return d


def _status_index(lines):
    return next(i for i, ln in enumerate(lines) if status.STATUS_RE.match(ln))


def test_set_status_changes_only_that_line(exp):
    """Independent of the run's state (review fix 2026-10-03): the copy's own
    current Status is read first, whatever it is by then."""
    hyp = exp / "HYPOTHESIS.md"
    before = hyp.read_text().split("\n")
    i = _status_index(before)
    current = status.STATUS_RE.match(before[i]).group(2)
    old = status.set_status(hyp, "a test status that no real run writes")
    after = hyp.read_text().split("\n")
    assert old == current
    assert len(after) == len(before)
    assert [ln for j, ln in enumerate(after) if j != i] == [ln for j, ln in enumerate(before) if j != i]
    assert after[i].endswith("· Status: a test status that no real run writes*")


def test_record_block_appends_and_updates_status(exp):
    hyp = exp / "HYPOTHESIS.md"
    res = exp / "results" / "gate"
    res.mkdir(parents=True)
    (res / "gate_20261004T120000Z.json").write_text(json.dumps({"verdict": {"K8": "PASS", "K4": "PASS"}}))
    (res / "gate_20261004T120000Z_mutants.json").write_text("{}")
    before = hyp.read_text().split("\n")
    block = status.record_block("gate", hyp, exp_dir=exp)
    after = hyp.read_text().split("\n")
    i = _status_index(before)
    assert after[: len(before)][:i] == before[:i]
    assert after[i + 1: len(before) - 1] == before[i + 1: len(before) - 1]
    assert "## gate — run record (" in block
    assert "`results/gate/gate_20261004T120000Z.json`" in block and "mutants" not in block
    assert "- Gate verdict: K4 PASS, K8 PASS" in block
    assert "gate K4 PASS, K8 PASS" in after[i]
    assert hyp.read_text().rstrip().endswith(block.rstrip())


def test_record_block_with_explicit_files_and_status(exp):
    f = exp / "results" / "preflight_x.json"
    f.parent.mkdir(parents=True)
    f.write_text("{}")
    block = status.record_block("preflight", exp / "HYPOTHESIS.md", files=[f], status="custom status", exp_dir=exp)
    assert "`results/preflight_x.json`" in block
    assert "· Status: custom status*" in (exp / "HYPOTHESIS.md").read_text()


def _amend(exp: Path, name: str, k: int, body: str) -> None:
    d = exp / "amendments"
    d.mkdir(exist_ok=True)
    (d / name).write_text(f"## Amendment {k} — gate fix (2026-10-0{k}T10:00:00Z)\n\n{body}\n")


def test_sync_amendments_in_number_order_and_idempotent(exp):
    hyp = exp / "HYPOTHESIS.md"
    _amend(exp, "b_first_by_name.md", 2, "second")
    _amend(exp, "z_last_by_name.md", 1, "first")
    done = status.sync_amendments(hyp, exp)
    assert [h.split(" — ")[0] for h in done] == ["## Amendment 1", "## Amendment 2"]
    text = hyp.read_text()
    assert text.index("## Amendment 1") < text.index("## Amendment 2")
    assert status.sync_amendments(hyp, exp) == []
    assert hyp.read_text() == text


def test_sync_with_empty_or_missing_amendments_is_a_noop(exp):
    hyp = exp / "HYPOTHESIS.md"
    before = hyp.read_text()
    assert status.sync_amendments(hyp, exp) == []
    (exp / "amendments").mkdir()
    (exp / "amendments" / ".gitkeep").write_text("")
    assert status.sync_amendments(hyp, exp) == []
    assert hyp.read_text() == before


def test_bad_amendment_heading_is_refused(exp):
    d = exp / "amendments"
    d.mkdir()
    (d / "x.md").write_text("# not an amendment\n")
    with pytest.raises(ValueError):
        status.sync_amendments(exp / "HYPOTHESIS.md", exp)


def test_no_status_line_is_an_error(tmp_path):
    p = tmp_path / "H.md"
    p.write_text("# nothing\n")
    with pytest.raises(ValueError):
        status.set_status(p, "x")


def test_append_amendment_appends_once_and_refuses_bad_or_repeated_headings(exp, tmp_path):
    """Review fix 2026-10-03: RUNBOOK step 13 appends the plan amendment with --append-amendment, not `cat >>`."""
    hyp = exp / "HYPOTHESIS.md"
    am = tmp_path / "AMENDMENT_5_20261004T180000Z.md"
    am.write_text("## Amendment 5 — plan (2026-10-04T18:00:00.000Z)\n\n- Status: FIXED\n")
    before = hyp.read_text()
    assert status.append_amendment(am, hyp, exp) == "## Amendment 5 — plan (2026-10-04T18:00:00.000Z)"
    after = hyp.read_text()
    assert after.startswith(before) and after.rstrip().endswith("- Status: FIXED")
    with pytest.raises(ValueError):
        status.append_amendment(am, hyp, exp)                # already there
    bad = tmp_path / "x.md"
    bad.write_text("# not an amendment\n")
    with pytest.raises(ValueError):
        status.append_amendment(bad, hyp, exp)


def test_verify_private_follows_latest_hashes_and_moves(tmp_path):
    exp = tmp_path / "exp"
    priv = tmp_path / "private"
    (exp / "evidence").mkdir(parents=True)
    (priv / "raw" / "S2" / "K8").mkdir(parents=True)
    (priv / "manifests").mkdir()
    (priv / "aborted" / "x").mkdir(parents=True)
    f = priv / "raw" / "S2" / "K8" / "gpqa_en_high.jsonl"
    f.write_text("a\nb\n")
    (priv / "manifests" / "gpqa_diamond_en.jsonl").write_text("m\n")
    moved = priv / "aborted" / "x" / "S2-rgb_cb_high.jsonl"
    moved.write_text("r\n")
    import hashlib

    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    recs = [
        {"type": "private_file", "path": "$EXP036_PRIVATE/raw/S2/K8/gpqa_en_high.jsonl", "sha256": sha(b"a\n")},
        {"type": "private_file", "path": "$EXP036_PRIVATE/raw/S2/K8/gpqa_en_high.jsonl", "sha256": sha(b"a\nb\n")},
        {"type": "private_file", "path": "$EXP036_PRIVATE/raw/S2/K8/rgb_cb_high.jsonl", "sha256": sha(b"r\n")},
        {"type": "private_moved", "path": "$EXP036_PRIVATE/raw/S2/K8/rgb_cb_high.jsonl",
         "to": "$EXP036_PRIVATE/aborted/x/S2-rgb_cb_high.jsonl", "sha256": sha(b"r\n")},
    ]
    (exp / "evidence" / "withheld_manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    res = status.verify_private(exp, priv)
    assert res["ok"] and res["verified"] == 2 and res["expected_extras"] == 1 and res["other_extras"] == []
    f.write_text("a\n")                                       # the mirror holds an older version
    res = status.verify_private(exp, priv)
    assert not res["ok"] and res["sha256_mismatch"] == ["$EXP036_PRIVATE/raw/S2/K8/gpqa_en_high.jsonl"]


def test_an_amendment_with_an_unfilled_placeholder_is_never_appended(exp, tmp_path):
    """Review 2026-10-04 (Amendment 6): a drafted amendment that still holds {{ANDREI_DECISION}} is refused by both
    entry points, and nothing at all is appended, not even an earlier complete amendment of the same sync."""
    hyp = exp / "HYPOTHESIS.md"
    before = hyp.read_text()
    _amend(exp, "1_ok.md", 1, "complete")
    d = exp / "amendments"
    (d / "2_draft.md").write_text("## Amendment 2 — draft (2026-10-04T20:00:00Z)\n\n**Andrei's decision.** "
                                  "{{ANDREI_DECISION}}\n")
    with pytest.raises(ValueError, match=r"\{\{ANDREI_DECISION\}\} is not filled in"):
        status.sync_amendments(hyp, exp)
    assert hyp.read_text() == before
    with pytest.raises(ValueError, match="not filled in"):
        status.append_amendment(d / "2_draft.md", hyp, exp)
    assert hyp.read_text() == before
    (d / "2_draft.md").write_text("## Amendment 2 — draft (2026-10-04T20:00:00Z)\n\n**Andrei's decision.** "
                                  "\"his words\"\n")
    assert [h.split(" — ")[0] for h in status.sync_amendments(hyp, exp)] == ["## Amendment 1", "## Amendment 2"]
