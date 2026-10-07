"""tools/refresh_builds.py (exp_037 DESIGN §2.8; G1 item 11; RUNBOOK step 8).

On copies of the tiny real-layout validation set's K8/K4 builds whose kolibri1.py is exp_036's port (the state of
the real sources): after the clone and the refresh the sources' kolibri1.py and records are byte-identical and the
clones carry this kit's port; a second refresh with a different port version succeeds and leaves the sources
byte-identical; a re-run is idempotent. Refusals: another source port, a source record changed since its clone, an
existing clone whose shards differ, a source that moves during the refresh, a symbolic link in a source, no
$EXP037_BUILDS (or one equal to $EXP036_MODELS), no git identity. The record is redacted and counts in S1 hours.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from port import convert
from tools import common
from tools import refresh_builds as rb

EXP = Path(__file__).resolve().parents[1]
E36_PORT = EXP.parent / "exp_036_kolibri_local_eval" / "port" / "kolibri1.py"   # read-only


def _tree(d: Path) -> dict:
    """{relpath: sha256} of every file under d."""
    return {p.relative_to(d).as_posix(): common.sha256_file(p) for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture
def world(tmp_path, monkeypatch, tiny_real_val):
    """$EXP036_MODELS with copies of the validation set's builds, set to exp_036's port; $EXP037_BUILDS beside
    them; the experiment identity; results/convert in tmp (never the kit's own results/)."""
    assert common.sha256_file(E36_PORT) == rb.EXP036_PORT_SHA256
    models = tmp_path / "models"
    for bits in rb.BITS:
        shutil.copytree(tiny_real_val.build(bits), models / rb.build_name(bits))
        convert.refresh_port_file(models / rb.build_name(bits), port_file=E36_PORT)
    monkeypatch.setenv("EXP036_MODELS", str(models))
    monkeypatch.setenv("EXP037_BUILDS", str(models / "exp037-builds"))
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "Miktam", "email": "hello@localfirstai.eu"})
    return {"models": models, "builds": models / "exp037-builds", "results": tmp_path / "results" / "convert",
            "sources": {b: _tree(models / rb.build_name(b)) for b in rb.BITS}}


def _quiet(*a, **k):
    pass


def test_clone_and_refresh_leave_the_sources_byte_identical(world):
    recs = rb.refresh(results_dir=world["results"], log=_quiet)
    port = common.sha256_file(convert.PORT_FILE)
    assert [r["bits"] for r in recs] == [8, 4]
    for r in recs:
        name = rb.build_name(r["bits"])
        src, dst = world["models"] / name, world["builds"] / name
        assert _tree(src) == world["sources"][r["bits"]]                         # byte-identical
        assert common.sha256_file(src / "kolibri1.py") == rb.EXP036_PORT_SHA256
        assert common.sha256_file(dst / "kolibri1.py") == port == r["port_sha256"]
        assert convert.check_port_file(dst) == port
        assert r["previous_port_sha256"] == rb.EXP036_PORT_SHA256 and r["port_refreshed"] is True
        assert r["mode"].startswith("cloned")
        shards = sorted(p.name for p in src.glob("model*.safetensors"))
        assert r["shard_count"] == len(shards) > 0
        for s in shards + ["config.json"]:
            assert common.sha256_file(dst / s) == common.sha256_file(src / s)
            assert (dst / s).stat().st_ino != (src / s).stat().st_ino            # a separate inode per file
        assert r["source_record_sha256"] == world["sources"][r["bits"]][rb.CONVERT_RECORD]
        assert r["target_record_sha256"] == common.sha256_file(dst / rb.CONVERT_RECORD)
        assert r["spec_version"] == json.loads((dst / rb.CONVERT_RECORD).read_text())["spec_version"]
        assert not (world["builds"] / (name + ".partial")).exists()


def test_the_record_is_redacted_and_counts_in_s1_hours(world):
    from runner import plan_fix

    rb.refresh(results_dir=world["results"], log=_quiet)
    files = sorted(world["results"].glob("refresh_*bit_*.json"))
    assert [p.name.split("_")[1] for p in files] == ["4bit", "8bit"]
    for p in files:
        text = p.read_text()
        rec = json.loads(text)
        assert rec["schema"] == rb.SCHEMA and rec["experiment"] == "exp_037"
        assert rec["source"] == f"$EXP036_MODELS/{rec['build']}" and rec["target"] == f"$EXP037_BUILDS/{rec['build']}"
        for key in ("t_start", "t_end", "shard_count", "source_record_sha256", "previous_port_sha256", "port_sha256"):
            assert rec.get(key) is not None, key
        assert str(world["models"]) not in text and "/Users/" not in text
    phases = [r["phase"] for r in plan_fix.load_s1_records(world["results"].parent)]
    assert sorted(phases) == sorted(f"convert/{p.name}" for p in files)


def test_a_second_refresh_with_another_port_version_succeeds(world, tmp_path):
    """G1 item 11: the clone exists, its shards are verified, the new port goes in; the sources stay put."""
    rb.refresh(results_dir=world["results"], log=_quiet)
    v1 = common.sha256_file(convert.PORT_FILE)
    v2_file = tmp_path / "port_v2" / "kolibri1.py"
    v2_file.parent.mkdir()
    v2_file.write_text(convert.PORT_FILE.read_text() + "\n# a later port version (test)\n")
    v2 = common.sha256_file(v2_file)
    recs = rb.refresh(results_dir=world["results"], port_file=v2_file, log=_quiet)
    for r in recs:
        name = rb.build_name(r["bits"])
        assert r["mode"].startswith("existing clone verified")
        assert (r["previous_port_sha256"], r["port_sha256"]) == (v1, v2)
        assert common.sha256_file(world["builds"] / name / "kolibri1.py") == v2
        assert convert.check_port_file(world["builds"] / name, port_file=v2_file) == v2
        assert _tree(world["models"] / name) == world["sources"][r["bits"]]


def test_a_rerun_with_the_same_port_is_idempotent(world):
    rb.refresh(results_dir=world["results"], log=_quiet)
    clones = {b: _tree(world["builds"] / rb.build_name(b)) for b in rb.BITS}
    recs = rb.refresh(results_dir=world["results"], log=_quiet)
    for r in recs:
        assert r["previous_port_sha256"] == r["port_sha256"] and r["port_refreshed"] is False
        assert _tree(world["builds"] / rb.build_name(r["bits"])) == clones[r["bits"]]
    assert len(list(world["results"].glob("refresh_8bit_*.json"))) == 2


def test_refuses_a_source_with_another_port(world):
    convert.refresh_port_file(world["models"] / rb.build_name(8))            # this kit's port, not exp_036's
    with pytest.raises(rb.RefreshError, match="step 1"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    assert not world["builds"].exists() and not world["results"].exists()


def test_refuses_an_existing_clone_whose_shard_differs(world):
    rb.refresh(results_dir=world["results"], log=_quiet)
    dst = world["builds"] / rb.build_name(8)
    shard = sorted(dst.glob("model*.safetensors"))[0]
    with open(shard, "ab") as f:
        f.write(b"\0")
    port_before = common.sha256_file(dst / "kolibri1.py")
    with pytest.raises(rb.RefreshError, match=r"step 2: .*sha256 differs"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    assert common.sha256_file(dst / "kolibri1.py") == port_before
    assert _tree(world["models"] / rb.build_name(8)) == world["sources"][8]


def test_refuses_a_source_record_changed_since_its_clone(world):
    rb.refresh(results_dir=world["results"], log=_quiet)
    rec = world["models"] / rb.build_name(8) / rb.CONVERT_RECORD
    rec.write_text(rec.read_text().rstrip() + "\n\n")
    with pytest.raises(rb.RefreshError, match=r"step 1: .*recorded at its clone \(refresh_8bit_"):
        rb.refresh(results_dir=world["results"], log=_quiet)


def test_refuses_a_source_that_moves_during_the_refresh(world, monkeypatch):
    real = convert.refresh_port_file

    def and_touch_the_source(out, port_file=convert.PORT_FILE):
        rec = real(out, port_file=port_file)
        src_rec = world["models"] / Path(out).name / rb.CONVERT_RECORD
        src_rec.write_text(src_rec.read_text() + " ")
        return rec

    monkeypatch.setattr(convert, "refresh_port_file", and_touch_the_source)
    with pytest.raises(rb.RefreshError, match="step 4"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    assert not world["results"].exists()                                       # no record for a refused build


def test_refuses_a_symbolic_link_in_a_source(world, tmp_path):
    (tmp_path / "elsewhere.txt").write_text("x\n")
    (world["models"] / rb.build_name(8) / "linked.txt").symlink_to(tmp_path / "elsewhere.txt")
    with pytest.raises(rb.RefreshError, match="symbolic links"):
        rb.refresh(results_dir=world["results"], log=_quiet)


def test_refuses_without_a_separate_builds_dir_or_identity(world, monkeypatch):
    monkeypatch.delenv("EXP037_BUILDS")
    with pytest.raises(rb.RefreshError, match="EXP037_BUILDS is not set"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    monkeypatch.setenv("EXP037_BUILDS", str(world["models"]))
    with pytest.raises(rb.RefreshError, match="equals EXP036_MODELS"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    monkeypatch.setenv("EXP037_BUILDS", str(world["models"] / rb.build_name(8) / "inside"))
    with pytest.raises(rb.RefreshError, match="inside the source"):
        rb.refresh(results_dir=world["results"], log=_quiet)
    monkeypatch.setenv("EXP037_BUILDS", str(world["builds"]))
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "x", "email": "y"})
    with pytest.raises(common.IdentityError):
        rb.refresh(results_dir=world["results"], log=_quiet)
    assert not world["builds"].exists() and not world["results"].exists()


def test_cli(world, capsys):
    assert rb.main(["--bits", "9", "--results-dir", str(world["results"])]) == 2
    assert rb.main(["--bits", "8", "--results-dir", str(world["results"])]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] and len(out["records"]) == 1 and out["records"][0].startswith("refresh_8bit_")
    assert not (world["builds"] / rb.build_name(4)).exists()
    assert rb.main(["--results-dir", str(world["results"]), "--expect-source-port", "0" * 64]) == 1
    assert "refused: step 1" in capsys.readouterr().err
