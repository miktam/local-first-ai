"""tools/status.py (exp_036 BUILD_SPEC §5.9): only the Status line changes in place;
run-record blocks are appended; amendment sync is ordered and idempotent.

exp_037: the tests run on a copy of a HYPOTHESIS.md. Until W12's hand-off note
(DESIGN §12 W10) that is a stub with the pre-registration's header lines and a
hash table with one row per scope; afterwards (since W15-01) it is exp_037's real
HYPOTHESIS.md. The refresh
phase (RUNBOOK step 8) and the mirror check over $EXP036_PRIVATE/exp037/ are
exp_037's additions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tools import common, hash_tree, status

EXP = Path(__file__).resolve().parents[1]

# DESIGN §12 W10: switched on (True) after W12's hand-off note (final review W15-01), so the tests run on the real HYPOTHESIS.md.
W12_HANDOFF = True


def _stub_hypothesis() -> str:
    rows = {}
    for s in hash_tree.SCOPES:
        rows.setdefault(s.row, []).append(s)
    lines = ["# Experiment 037: Kolibri forced gate (stub for tests/test_tools_status.py)", "",
             "*Pre-registered: {{PREREG_UTC}} · Status: pre-registered, awaiting Andrei's sign-off; no scored run*",
             "", "## Fixed before any run (hashes)", "", "| Item | Value |", "|---|---|"]
    lines += [f"| {row} | " + "; ".join(f"sha256 `{{{{{s.placeholder}}}}}`" for s in scopes) + " |"
              for row, scopes in rows.items()]
    lines += ["", "## Sign-off", "", "- Signed off by:", ""]
    return "\n".join(lines)


@pytest.fixture
def exp(tmp_path) -> Path:
    d = tmp_path / "exp"
    d.mkdir()
    if W12_HANDOFF:
        shutil.copy(EXP / "HYPOTHESIS.md", d / "HYPOTHESIS.md")
    else:
        (d / "HYPOTHESIS.md").write_text(_stub_hypothesis(), encoding="utf-8")
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


# ---------------------------------------------------------------- exp_037

def test_refresh_record_block_names_both_refresh_records(exp):
    """RUNBOOK step 8 (DESIGN §2.8): `status.py --record-block refresh` lists the newest refresh record of each
    bit width, and no convert record."""
    conv = exp / "results" / "convert"
    conv.mkdir(parents=True)
    for name in ("refresh_8bit_20261007T090000Z.json", "refresh_4bit_20261007T090001Z.json",
                 "convert_8bit_20261004T133423Z.json"):
        (conv / name).write_text("{}")
    block = status.record_block("refresh", exp / "HYPOTHESIS.md", exp_dir=exp)
    assert "`results/convert/refresh_8bit_20261007T090000Z.json`" in block
    assert "`results/convert/refresh_4bit_20261007T090001Z.json`" in block
    assert "convert_8bit" not in block
    assert "· Status: K8/K4 builds cloned and refreshed; Session 1 in progress; no scored run*" in \
        (exp / "HYPOTHESIS.md").read_text()


def test_verify_private_on_exp037_paths(tmp_path):
    """DESIGN §2.9: exp_037's private records live under $EXP036_PRIVATE/exp037/. exp_036's own files and the
    reused withheld manifests are expected extras (counted, not listed); so are exp_037's aborted copies; any other
    unlisted file under exp037/ is reported."""
    import hashlib

    exp = tmp_path / "exp"
    priv = tmp_path / "private"
    (exp / "evidence").mkdir(parents=True)
    f = priv / "exp037" / "raw" / "S2" / "K8" / "gpqa_en_high.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text("a\n")
    for rel in ("manifests/gpqa_diamond_en.jsonl", "raw/S2/K8/gpqa_en_high.jsonl", "exp037/aborted/x/y.jsonl"):
        (priv / rel).parent.mkdir(parents=True, exist_ok=True)
        (priv / rel).write_text("old\n")
    rec = {"type": "private_file", "path": "$EXP036_PRIVATE/exp037/raw/S2/K8/gpqa_en_high.jsonl",
           "sha256": hashlib.sha256(b"a\n").hexdigest()}
    (exp / "evidence" / "withheld_manifest.jsonl").write_text(json.dumps(rec) + "\n")
    res = status.verify_private(exp, priv)
    assert res["ok"] and res["verified"] == 1
    assert res["expected_extras"] == 3 and res["expected_extras_exp036"] == 2 and res["other_extras"] == []
    stray = priv / "exp037" / "pilot" / "K8" / "unlisted.jsonl"
    stray.parent.mkdir(parents=True)
    stray.write_text("x\n")
    res = status.verify_private(exp, priv)
    assert res["ok"] and res["other_extras"] == ["$EXP036_PRIVATE/exp037/pilot/K8/unlisted.jsonl"]
    f.write_text("changed\n")
    res = status.verify_private(exp, priv)
    assert not res["ok"] and res["sha256_mismatch"] == ["$EXP036_PRIVATE/exp037/raw/S2/K8/gpqa_en_high.jsonl"]


def test_cli_description_names_exp037(capsys):
    with pytest.raises(SystemExit):
        status.main(["--help"])
    assert "exp_037" in capsys.readouterr().out


# ---------------------------------------------------------------- pre-freeze review 2026-10-06: what the blocks hash

def _touch(exp: Path, rel: str, text: str = "x\n", mtime: float | None = None) -> Path:
    p = exp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _listed(block: str) -> dict[str, str]:
    """The block's "Result files (sha256)" lines: relpath -> sha256."""
    return dict(re.findall(r"^  - `([^`]+)` `([0-9a-f]{64})`$", block, re.M))


def _want(exp: Path, paths) -> dict[str, str]:
    return {p.relative_to(exp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_s3_record_block_covers_s3b_and_s2_at_its_final_state(exp):
    """RUNBOOK step 17's S3 block (with or without the mbp scores) also hashes S3b, which runs before it, and every
    results/raw/S2/ file at its final state (a cell started in S2 finishes in S3 there, runner/run.py cell_paths),
    step logs and the step logs of aborted cells included. The S2 and S3b blocks keep their own session."""
    hyp = exp / "HYPOTHESIS.md"
    s2 = [_touch(exp, "results/raw/S2/K8/gpqa_en_high.jsonl", "partial\n"),
          _touch(exp, "results/raw/S2/K8/gpqa_en_high.steps.jsonl"), _touch(exp, "results/raw/S2/session.jsonl"),
          _touch(exp, "aborted/20261009T010000Z-K4-rgb_cb_high/S2-rgb_cb_high.steps.jsonl")]
    s2_block = status.record_block("S2", hyp, exp_dir=exp)
    assert _listed(s2_block) == _want(exp, s2)
    s2[0].write_text("partial\nfinished in S3\n")
    s3 = [_touch(exp, "results/raw/S3/K4/mmlu_en_high.jsonl"),
          _touch(exp, "results/raw/S3/K4/mmlu_en_high.steps.jsonl"),
          _touch(exp, "aborted/20261010T030000Z-K8-aime_de_high/S3-aime_de_high.steps.jsonl")]
    s3b = [_touch(exp, "results/raw/S3b/K8/gpqa_de_high.jsonl"),
           _touch(exp, "results/raw/S3b/K8/gpqa_de_high.steps.jsonl")]
    score = _touch(exp, "results/scores/K8/gpqa_en_high.jsonl")
    _touch(exp, "aborted/20261010T030000Z-K8-aime_de_high/S3-aime_de_high.jsonl")   # committed: git holds it
    _touch(exp, "results/raw/S3/.heartbeat")
    for phase, want in (("S3, mbp scores", s2 + s3 + s3b + [score]), ("S3", s2 + s3 + s3b), ("S3b", s3b),
                        ("S2", s2)):
        assert _listed(status.record_block(phase, hyp, exp_dir=exp)) == _want(exp, want), phase
    assert _listed(s2_block)["results/raw/S2/K8/gpqa_en_high.jsonl"] != _want(exp, s2[:1])[
        "results/raw/S2/K8/gpqa_en_high.jsonl"]   # step 15 hashed the part-written cell; only S3's block has it final


def test_verdicts_record_block_names_the_minis_files(exp):
    """RUNBOOK step 19 (the mini): the newest results/verdicts_<UTC>.json and .md (analysis/verdicts.py write()), the
    newest results/rescore_mini_<UTC>.json (scorers/score_all.py --rescore-compare) and every IFBench score file
    (score_all.py --ifbench, the mini only); no other score file, no plan file."""
    for rel in ("results/verdicts_20261011T090000Z.json", "results/verdicts_20261011T090000Z.md",
                "results/rescore_mini_20261011T085000Z.json"):
        _touch(exp, rel, mtime=1_000_000_000)
    want = [_touch(exp, "results/verdicts_20261011T100000Z.json"), _touch(exp, "results/verdicts_20261011T100000Z.md"),
            _touch(exp, "results/rescore_mini_20261011T095000Z.json"),
            _touch(exp, "results/scores/G8/ifbench_high.jsonl"), _touch(exp, "results/scores/K8/ifbench_high.jsonl")]
    _touch(exp, "results/scores/K8/gpqa_en_high.jsonl")
    _touch(exp, "results/plan_fixed_20261008T120000Z.json")
    block = status.record_block("verdicts", exp / "HYPOTHESIS.md", exp_dir=exp)
    assert _listed(block) == _want(exp, want)
    assert "- Result files: none" not in block


def test_plan_record_block_covers_the_pilot_and_its_step_logs(exp):
    """RUNBOOK step 13 commits the pilot (step 12 has no commit), so its plan block also hashes every pilot summary
    (a re-pilot's plan reads both), every pilot cell file and step log, and an aborted pilot's step logs; the plan
    files stay the newest ones. `--record-block pilot` gives the pilot's files alone."""
    hyp = exp / "HYPOTHESIS.md"
    _touch(exp, "results/plan_fixed_20261008T100000Z.json", mtime=1_000_000_000)
    plan = [_touch(exp, "results/plan_fixed_20261008T120000Z.json"),
            _touch(exp, "results/AMENDMENT_7_20261008T120000Z.md")]
    pilot = [_touch(exp, "results/pilot_summary_20261008T080000Z.json", mtime=1_000_000_000),
             _touch(exp, "results/pilot_summary_20261008T110000Z.json"),
             _touch(exp, "results/pilot/20261008T060000Z/K8/gpqa_main_en_high.jsonl"),
             _touch(exp, "results/pilot/20261008T060000Z/K8/gpqa_main_en_high.steps.jsonl"),
             _touch(exp, "results/pilot/20261008T103000Z/K8/gpqa_main_en_high.jsonl"),
             _touch(exp, "results/pilot/20261008T103000Z/K8/gpqa_main_en_high.steps.jsonl"),
             _touch(exp, "aborted/20261008T050000Z-pilot/pilot/K8/gpqa_main_en_high.steps.jsonl")]
    for rel in ("results/pilot/20261008T060000Z/.heartbeat", "aborted/20261008T050000Z-pilot/NOTE.md",
                "aborted/20261008T050000Z-pilot/pilot/K8/gpqa_main_en_high.jsonl"):
        _touch(exp, rel)
    assert _listed(status.record_block("plan", hyp, exp_dir=exp)) == _want(exp, plan + pilot)
    assert _listed(status.record_block("pilot", hyp, exp_dir=exp)) == _want(exp, pilot)


def test_session_step_logs_are_git_ignored_pilot_ones_committed_and_all_hashed(exp):
    """Pre-freeze decision on the step logs (about 190 B a decode step): a session's, under results/raw/ and as an
    aborted cell's aborted/<UTC>-<arm>-<stem>/<session>-<stem>.steps.jsonl copy (runner/run.py abort_cell; about
    190 MB for a K8 cell at B = 1), are git-ignored working files; the pilot's, under results/pilot/ and an aborted
    pilot's aborted/<UTC>-pilot/ (with the rest of that pilot), are committed (the plan rule fits its step model on a
    summarised pilot's, runner/plan_fix.py load_steps), as are the cell files beside both. tools/status.py reads the filesystem, so the S2, S3b and plan
    blocks give every step log's sha256, ignored or committed."""
    ignored = ["results/raw/S2/K8/gpqa_en_high.steps.jsonl", "results/raw/S3b/K8/gpqa_de_high.steps.jsonl",
               "aborted/20261009T010000Z-K4-rgb_cb_high/S2-rgb_cb_high.steps.jsonl",
               "aborted/20261010T030000Z-K8-aime_de_high/S3b-aime_de_high.steps.jsonl"]
    pilot = ["results/pilot/20261008T060000Z/K8/gpqa_main_en_high.steps.jsonl",
             "aborted/20261008T050000Z-pilot/pilot/K8/gpqa_main_en_high.steps.jsonl"]
    kept = ["results/raw/S2/K8/gpqa_en_high.jsonl", "results/raw/S2/session.jsonl",
            "results/pilot/20261008T060000Z/K8/gpqa_main_en_high.jsonl",
            "aborted/20261009T010000Z-K4-rgb_cb_high/S2-rgb_cb_high.jsonl",
            "aborted/20261008T050000Z-pilot/pilot/K8/gpqa_main_en_high.jsonl"]
    if common.repo_root(EXP) is None:
        pytest.skip("not in a git work tree")
    for rel in ignored + pilot + kept:
        r = subprocess.run(["git", "check-ignore", "-q", "--no-index", rel], cwd=EXP)
        assert r.returncode == (0 if rel in ignored else 1), rel
    for rel in ignored + pilot + kept:
        _touch(exp, rel, f"{rel}\n")
    blocks = {}
    for phase in ("S2", "S3b", "plan"):
        blocks.update(_listed(status.record_block(phase, exp / "HYPOTHESIS.md", exp_dir=exp)))
    # Aborted cell and pilot files other than step logs are in no block: git holds them.
    assert blocks == _want(exp, [exp / rel for rel in ignored + pilot + kept[:3]])
