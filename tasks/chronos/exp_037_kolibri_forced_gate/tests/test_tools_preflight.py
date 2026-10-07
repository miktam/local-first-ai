"""tools/preflight.py and tools/assetcheck.py (BUILD_SPEC §5.9; HYPOTHESIS
"K8 working-set rule").

System calls are monkeypatched (FakeSystem); one test runs the real preflight
on this machine in "not run host" mode and checks it records no identifier.
"""

from __future__ import annotations

import getpass
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tools import assetcheck, common, preflight
from tools.redact import HOST_LABEL, forbidden_keys

GiB = 2**30
L_MBP = 115_448_725_504          # measured on the mbp, 2026-10-03 (107.52 GiB)
L_MINI = int(51.84 * GiB)


class FakeSystem(preflight.System):
    def __init__(self, L=L_MBP, wired_mb=0, free=3_000 * 10**9, packages=None, power=("AC", "0", "2"), thermal=0):
        self.L, self.wired_mb, self.free, self.thermal = L, wired_mb, free, thermal
        self.power = power
        pins = preflight.load_versions()["pins"]
        self.packages = dict(pins) if packages is None else packages

    def sysctl(self, name):
        return {"machdep.cpu.brand_string": "Apple M5 Max", "hw.model": "Mac17,7",
                "hw.memsize": str(128 * GiB), "iogpu.wired_limit_mb": str(self.wired_mb),
                "vm.swapusage": "total = 0.00M  used = 0.00M  free = 0.00M  (encrypted)"}.get(name)

    def sw_vers(self):
        return {"product_version": "27.0", "build": "26A428"}

    def gpu_cores(self):
        return 40

    def pmset(self):
        return {"source": self.power[0], "lowpowermode": self.power[1], "powermode": self.power[2]}

    def thermal_state(self):
        return self.thermal

    def device_info(self):
        return {"device_name": "Apple M5 Max", "max_recommended_working_set_size": self.L,
                "memory_size": 128 * GiB, "architecture": "applegpu_g17s"}

    def disk_free(self, path):
        return self.free

    def freeze(self, python):
        return {"python": "3.12.13", "packages": dict(self.packages)}


def _complete_assets(**kw):
    return {"models": [], "datasets": [], "code": [], "environments": [], "problems": [], "complete": True}


@pytest.fixture
def run_host(tmp_path, monkeypatch):
    """Models folder in tmp; assets reported complete unless a test says otherwise. No $EXP037_BUILDS or
    $EXP037_VENV from the host: the builds resolve under $EXP036_MODELS (builds_dir()'s fallback)."""
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    monkeypatch.delenv("EXP036_DATA", raising=False)
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    monkeypatch.delenv("EXP037_VENV", raising=False)
    monkeypatch.setattr(assetcheck, "check_all", _complete_assets)
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "Miktam", "email": "hello@localfirstai.eu"})
    return tmp_path


def _cells(rec):
    return {c["task"]: c["B"] for c in rec["memory_rule"]["cells"]}


def test_clean_run_host_record(run_host):
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")
    assert rec["role"] == "run host" and rec["host_label"] == HOST_LABEL
    assert rec["problems"] == [], rec["problems"]
    assert rec["metal"]["L_bytes"] == L_MBP and rec["metal"]["L_source"] == "max_recommended_working_set_size"
    assert rec["machine"]["chip"] == "Apple M5 Max" and rec["machine"]["gpu_cores"] == 40
    assert rec["memory_rule"]["weight_source"].startswith("predicted")
    assert forbidden_keys(rec) == []
    json.dumps(rec)   # serialisable


def test_k8_working_set_at_the_measured_limit_needs_no_sysctl(run_host):
    """HYPOTHESIS rule 1 at L = 107.5 GiB: GPQA and MMLU B = 8, IFBench and RGB B = 16, no line."""
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")
    cells = _cells(rec)
    assert (cells["gpqa"], cells["mmlu"], cells["ifbench"], cells["rgb"]) == (8, 8, 16, 16)
    assert 1 <= cells["aime"] <= 8
    assert rec["memory_rule"]["advice"] is None and rec["memory_rule"]["c1_B"] == 8
    assert rec["memory_rule"]["k8_fits"]


def test_fallback_rule_agrees_with_runner_memory(run_host, monkeypatch):
    a = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")["memory_rule"]
    monkeypatch.setitem(sys.modules, "runner.memory", None)      # import fails -> fallback
    b = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")["memory_rule"]
    assert b["source"] == "tools.preflight fallback"
    assert {c["task"]: c["B"] for c in a["cells"] if c["task"] != "aime"} == \
           {c["task"]: c["B"] for c in b["cells"] if c["task"] != "aime"}
    assert a["advice"] is None and b["advice"] is None


@pytest.mark.parametrize("fallback", [False, True])
def test_advice_text_when_the_limit_is_low(run_host, monkeypatch, fallback):
    if fallback:
        monkeypatch.setitem(sys.modules, "runner.memory", None)
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(L=90 * GiB), venv_python="x")
    adv = rec["memory_rule"]["advice"]
    assert adv and adv.startswith("sudo sysctl iogpu.wired_limit_mb=114688")
    assert any("sysctl advice" in w for w in rec["warnings"])
    assert preflight.exit_code(rec) == 2
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(L=L_MINI), venv_python="x")
    assert not rec["memory_rule"]["k8_fits"] and "does not fit" in rec["memory_rule"]["advice"]


def test_wired_limit_raises_L(run_host):
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(wired_mb=114688), venv_python="x")
    assert rec["metal"]["L_bytes"] == 114688 * 2**20 and rec["metal"]["L_source"] == "iogpu.wired_limit_mb"


def test_not_run_host_never_prints_a_sysctl_line(run_host):
    rec = preflight.gather("quick", run_host=False, system=FakeSystem(L=L_MINI), venv_python="x")
    assert rec["memory_rule"]["advice"] is None
    assert "advice_not_applicable" in rec["memory_rule"]
    assert rec["role"] == "not run host"


def test_identity_refusal(run_host, monkeypatch, tmp_path):
    monkeypatch.setattr(common, "git_identity", lambda cwd=None: {"name": "Someone", "email": "x@y.z"})
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")
    assert any("git identity" in p for p in rec["problems"])
    with pytest.raises(common.IdentityError):
        preflight.write_record(rec, tmp_path / "results")
    assert not (tmp_path / "results").exists()
    monkeypatch.setattr(preflight, "System", lambda: FakeSystem())
    assert preflight.main(["--quick", "--out-dir", str(tmp_path / "results"), "--python", "x"]) == 1
    assert not (tmp_path / "results").exists()


def test_writes_record_and_never_overwrites(run_host, monkeypatch, tmp_path):
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(), venv_python="x")
    p = preflight.write_record(rec, tmp_path / "results")
    assert re.fullmatch(r"preflight_\d{8}T\d{6}Z\.json", p.name)
    assert json.loads(p.read_text())["schema"] == preflight.SCHEMA
    with pytest.raises(FileExistsError):
        preflight.write_record(rec, tmp_path / "results")


@pytest.mark.parametrize("free_gb,converted,ok", [
    (250, False, False), (300, False, True), (110, True, False), (130, True, True),
])
def test_disk_thresholds(run_host, free_gb, converted, ok):
    models = run_host / "models"
    if converted:
        for d in (preflight.K8_DIR, preflight.K4_DIR):
            (models / d).mkdir(parents=True)
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(free=free_gb * 10**9), venv_python="x")
    assert rec["disk"]["ok"] is ok
    assert rec["disk"]["required_bytes"] == (120 if converted else 260) * 10**9
    assert any(p.startswith("disk:") for p in rec["problems"]) is (not ok)


def test_missing_downloads_raise_the_disk_requirement(run_host, monkeypatch):
    res = _complete_assets()
    res["models"] = [{"repo": "r", "bytes_manifest": 50 * 10**9, "bytes_on_disk": 10 * 10**9}]
    monkeypatch.setattr(assetcheck, "check_all", lambda **kw: res)
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(free=280 * 10**9), venv_python="x")
    assert rec["disk"]["missing_download_bytes"] == 40 * 10**9
    assert not rec["disk"]["ok"]


def test_versions_core_mismatch_is_a_problem_other_is_recorded(run_host):
    pins = preflight.load_versions()["pins"]
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(packages=dict(pins, mlx="0.31.1")),
                           venv_python="x")
    assert any(p.startswith("core versions") and "mlx 0.31.1" in p for p in rec["problems"])
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(packages=dict(pins, rich="1.0", extra="2")),
                           venv_python="x")
    assert rec["problems"] == []
    assert "rich" in rec["versions"]["other_mismatch"] and "extra" in rec["versions"]["extras"]


def test_power_and_thermal_warnings(run_host):
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(power=("battery", "1", "0"), thermal=2),
                           venv_python="x")
    text = " ".join(rec["warnings"])
    assert "not AC" in text and "Low Power Mode" in text and "High Power" in text and "serious" in text


def test_exp007_fixtures_are_found_by_real_name():
    fx = preflight.fixtures()
    assert set(fx) == {"pad_4k.txt", "pad_15k.txt", "pad_120k.txt"}
    for name, f in fx.items():
        assert f["present"], f"{name} missing at {f['path']}"
        assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])


def test_next_step_follows_the_runbook(run_host, monkeypatch):
    st = {"identity_ok": True, "versions_ok": True, "assets_ok": True, "advice": None}
    assert "step 2" in preflight.next_step(common.EXP_DIR, dict(st, identity_ok=False))
    assert "step 3" in preflight.next_step(common.EXP_DIR, dict(st, versions_ok=False))
    assert "step 0" in preflight.next_step(common.EXP_DIR, dict(st, assets_ok=False))
    assert "step 5" in preflight.next_step(common.EXP_DIR, dict(st, advice="sudo ..."))


# ---------------------------------------------------------------- assetcheck

def _hf_dir(root: Path, files: dict, commit: str, etags: dict | None = None, incomplete: bool = False) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        m = root / ".cache" / "huggingface" / "download" / (rel + ".metadata")
        m.parent.mkdir(parents=True, exist_ok=True)
        etag = (etags or {}).get(rel) or common.sha256_bytes(data)
        m.write_text(f"{commit}\n{etag}\n1791051258.0\n")
    if incomplete:
        (root / ".cache" / "huggingface" / "download" / "x.abc.incomplete").write_bytes(b"")
    return root


def test_metadata_parsing_and_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path))
    monkeypatch.setenv("EXP036_DATA", str(tmp_path / "data"))  # the run host's environment file sets it
    pin = "a" * 40
    _hf_dir(tmp_path / "data" / "ds", {"README.md": b"r", "data/train.parquet": b"PAR1"}, pin)
    meta = assetcheck.hf_metadata(tmp_path / "data" / "ds")
    assert meta["data/train.parquet"]["commit"] == pin
    entry = {"repo": "x/ds", "revision": pin, "path": "data/ds"}
    r = assetcheck._hf_entry(entry, "dataset", deep=True, workers=2, lfs=None)
    assert r["revision_ok"] and r["complete"] and r["deep"]["mismatches"] == []
    r = assetcheck._hf_entry(dict(entry, revision="b" * 40), "dataset", deep=False, workers=1, lfs=None)
    assert not r["revision_ok"] and not r["complete"]
    _hf_dir(tmp_path / "data" / "gated", {"README.md": b"only the card"}, pin)
    r = assetcheck._hf_entry({"repo": "x/g", "revision": pin, "path": "data/gated"}, "dataset", False, 1, None)
    assert "no data files (only README/attributes)" in r["problems"]
    _hf_dir(tmp_path / "m", {"a.safetensors": b"w"}, pin, incomplete=True)
    r = assetcheck._hf_entry({"repo": "x/m", "revision": pin, "path": "m", "bytes": 1}, "model", False, 1, None)
    assert r["incomplete_files"] == 1 and not r["complete"]


def test_data_paths_follow_exp036_data(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("EXP036_DATA", str(tmp_path / "elsewhere"))
    assert common.asset_path("data/RGB-src") == tmp_path / "elsewhere" / "RGB-src"
    assert common.asset_path("Kolibri-1-BF16") == tmp_path / "models" / "Kolibri-1-BF16"


def test_deep_hash_against_lfs_and_git_blob_etags(tmp_path, monkeypatch):
    monkeypatch.setenv("EXP036_MODELS", str(tmp_path))
    pin = "c" * 40
    shard, small = b"shard bytes" * 100, b'{"a": 1}\n'
    blob = __import__("hashlib").sha1(f"blob {len(small)}\0".encode() + small).hexdigest()
    _hf_dir(tmp_path / "K", {"model-00001-of-00001.safetensors": shard, "config.json": small}, pin,
            etags={"config.json": blob})
    lfs = {"revision": pin, "shards": {"model-00001-of-00001.safetensors":
                                       {"sha256": common.sha256_bytes(shard), "bytes": len(shard)}}}
    entry = {"repo": "Aleph-Alpha/Kolibri-1-BF16", "revision": pin, "path": "K"}
    r = assetcheck._hf_entry(entry, "model", deep=True, workers=2, lfs=lfs)
    assert r["complete"], r["problems"]
    assert r["deep"]["sha256"]["model-00001-of-00001.safetensors"] == common.sha256_bytes(shard)
    assert r["lfs"]["deep_mismatches"] == [] and r["lfs"]["etag_mismatch"] == []
    bad = {"revision": pin, "shards": {"model-00001-of-00001.safetensors": {"sha256": "0" * 64, "bytes": len(shard)}}}
    r = assetcheck._hf_entry(entry, "model", deep=True, workers=1, lfs=bad)
    assert r["lfs"]["etag_mismatch"] and r["lfs"]["deep_mismatches"] and not r["complete"]


def test_git_head_detached_and_packed_ref(tmp_path):
    d = tmp_path / "clone"
    (d / ".git").mkdir(parents=True)
    (d / ".git" / "HEAD").write_text("1" * 40 + "\n")
    assert assetcheck.git_head(d) == "1" * 40
    (d / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (d / ".git" / "packed-refs").write_text("# pack-refs\n" + "2" * 40 + " refs/heads/main\n")
    assert assetcheck.git_head(d) == "2" * 40


def test_committed_lfs_oids_match_the_index():
    oids = assetcheck.load_lfs_oids()
    assert oids["revision"] == "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
    assert oids["shard_count"] == 32 == len(oids["shards"])
    assets = common.load_assets()
    kol = next(m for m in assets["models"] if m["repo"] == "Aleph-Alpha/Kolibri-1-BF16")
    total = oids["shard_bytes_total"] + sum(v["bytes"] for v in oids["non_lfs_files"].values())
    assert total == kol["bytes"]
    for d in _kolibri_dirs():
        idx = json.loads((d / "model.safetensors.index.json").read_text())
        assert sorted(set(idx["weight_map"].values())) == sorted(oids["shards"])
        meta = assetcheck.hf_metadata(d)
        for name, v in oids["non_lfs_files"].items():
            if name in meta:
                assert meta[name]["etag"] == v["git_blob_sha1"], name


def _kolibri_dirs():
    import os

    out = []
    for var in ("EXP036_TOKENIZER_DIR", "EXP036_TOK"):
        v = os.environ.get(var)
        if v and (Path(v) / "model.safetensors.index.json").is_file():
            out.append(Path(v))
    return out


# ---------------------------------------------------------------- the real machine

def test_real_not_run_host_preflight_records_no_identifiers(tmp_path):
    out = tmp_path / "preflight.json"
    code = preflight.main(["--quick", "--not-run-host", "--out", str(out)])
    assert code in (0, 2), code
    rec = json.loads(out.read_text())
    assert rec["role"] == "not run host"
    assert forbidden_keys(rec) == []
    text = out.read_text()
    user = getpass.getuser()
    assert not re.search(r"(?<![A-Za-z0-9_])" + re.escape(user) + r"(?![A-Za-z0-9_])", text)
    host = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True).stdout.strip()
    if len(host) >= 4:
        assert host.lower() not in text.lower()
    assert not re.search(r"(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}", text)
    assert "/Users/" not in text
    assert rec["memory_rule"].get("advice") is None


# ---------------------------------------------------------------- exp_037 (DESIGN §2.8, §3.1 P2, decision F2)

def test_schema_and_f2_core_pins():
    assert preflight.SCHEMA == "exp037 preflight v1"
    v = preflight.load_versions()
    assert preflight.VERSIONS_FILE == common.EXP_DIR / "env" / "versions.json"
    assert {k: v["core"][k] for k in ("mlx", "mlx-metal", "mlx-lm")} == \
        {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}


@pytest.mark.parametrize("pkgs,ok", [
    ({"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}, True),
    ({"mlx": "0.31.2", "mlx-metal": "0.31.2", "mlx-lm": "0.31.3"}, False),     # exp_036's venv
    ({"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.31.3"}, False),
])
def test_an_exp036_venv_is_a_core_version_problem(run_host, pkgs, ok):
    pins = preflight.load_versions()["pins"]
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(packages=dict(pins, **pkgs)), venv_python="x")
    assert (rec["problems"] == []) is ok, rec["problems"]
    assert rec["versions"]["versions_file_sha256"] == common.sha256_file(preflight.VERSIONS_FILE)
    assert rec["mx_device_info"]["architecture"] == "applegpu_g17s"


def test_builds_resolve_under_exp037_builds(run_host, monkeypatch):
    """DESIGN §2.8: the K8 weight and the conversions present are read under $EXP037_BUILDS when it is set."""
    builds = run_host / "builds"
    for d in (preflight.K8_DIR, preflight.K4_DIR):
        (builds / d).mkdir(parents=True)
    (builds / preflight.K8_DIR / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 1000)
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(free=130 * 10**9), venv_python="x")
    assert rec["disk"]["conversions_present"] == {"K8": False, "K4": False}
    assert rec["memory_rule"]["weight_source"].startswith("predicted")
    monkeypatch.setenv("EXP037_BUILDS", str(builds))
    rec = preflight.gather("quick", run_host=True, system=FakeSystem(free=130 * 10**9), venv_python="x")
    assert rec["disk"]["conversions_present"] == {"K8": True, "K4": True} and rec["disk"]["ok"]
    assert rec["memory_rule"]["weight_source"].startswith("measured") and rec["memory_rule"]["weight_bytes"] == 1000


def test_the_default_venv_is_the_exp037_one(run_host, monkeypatch):
    """Decision F2: with no --python, the run host compares $EXP037_VENV's interpreter, never exp_036's venv."""
    seen = []

    class Spy(FakeSystem):
        def freeze(self, python):
            seen.append(python)
            return super().freeze(python)

    old = run_host / "models" / "venv" / "bin" / "python"            # exp_036's venv layout
    new = run_host / "exp037" / "venv" / "bin" / "python"            # exp037/venv beside $EXP036_MODELS
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_text("")
    preflight.gather("quick", run_host=True, system=Spy())
    assert seen == [str(new)]
    other = run_host / "elsewhere" / "bin" / "python"
    other.parent.mkdir(parents=True)
    other.write_text("")
    monkeypatch.setenv("EXP037_VENV", str(run_host / "elsewhere"))
    preflight.gather("quick", run_host=True, system=Spy())
    assert seen[-1] == str(other)


def test_next_step_8_is_the_refresh(run_host, monkeypatch, tmp_path):
    exp = tmp_path / "exp"
    (exp / "tools").mkdir(parents=True)
    (exp / "tools" / "withheld_shingles.sha256").write_text("# x\n")
    st = {"identity_ok": True, "versions_ok": True, "assets_ok": True, "advice": None}
    assert "step 8" in preflight.next_step(exp, st) and "refresh_builds" in preflight.next_step(exp, st)
    builds = tmp_path / "builds"
    for d in (preflight.K8_DIR, preflight.K4_DIR):
        (builds / d).mkdir(parents=True)
    monkeypatch.setenv("EXP037_BUILDS", str(builds))
    assert "step 8" in preflight.next_step(exp, st)                  # the clones exist, the records do not
    (exp / "results" / "convert").mkdir(parents=True)
    for b in (8, 4):
        (exp / "results" / "convert" / f"refresh_{b}bit_20261007T090000Z.json").write_text("{}")
    assert "step 9" in preflight.next_step(exp, st)


def test_phase_plan_is_exp037s_s1():
    assert "gate 3.0/4.4" in preflight.PHASE_PLAN and "total 6.2/9.2 h" in preflight.PHASE_PLAN
    assert "port refresh" in preflight.PHASE_PLAN and "convert" not in preflight.PHASE_PLAN
