"""tools/peer_check.py (BUILD_SPEC §5.9; HYPOTHESIS C3, C9).

Weight-free: build-time tokenizer/template parity on the mini's peer folders
(skipped where they are absent), the committed parity record being current,
the static per-arm checks, and the decision arithmetic on synthetic numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools import peer_check as pc


def _mini_root() -> Path | None:
    from exp036_helpers import mini_asset_roots

    for c in mini_asset_roots():
        if (c / "mlx-community").is_dir() and (c / "upstream").is_dir():
            return c
    return None


@pytest.fixture(scope="module")
def mini():
    """The build host's peer and upstream folders (parity needs upstream; only the mini has it)."""
    from exp036_helpers import BUILD_HOST_ONLY

    root = _mini_root()
    if root is None:
        pytest.skip(f"{BUILD_HOST_ONLY} build-time peer and upstream files not found (set EXP036_MINI_ASSETS)")
    return root


@pytest.fixture(scope="module")
def peers():
    """The six peer folders: the mini's mlx-community/ or the run host's $EXP036_MODELS."""
    from exp036_helpers import peer_root

    root = peer_root(list(pc.ARM_DIRS.values()))
    if root is None:
        pytest.skip("peer folders not found (set EXP036_MINI_ASSETS on the mini, EXP036_MODELS on the run host)")
    return root


@pytest.fixture(scope="module")
def fresh_parity(mini):
    return pc.build_parity(mini / "mlx-community", mini / "upstream")


def test_parity_on_the_mini_folders(fresh_parity):
    fams = fresh_parity["families"]
    assert set(fams) == set(pc.FAMILIES)
    for fam, f in fams.items():
        assert "error" not in f, (fam, f.get("error"))
        assert f["committed_template"]["equal_to_upstream"], fam
        for arm in pc.FAMILIES[fam]["arms"]:
            p = f["peers"][arm]
            assert p["present"]
            assert p["token_ids"]["n"] == len(pc.FIXED_STRINGS)
            assert p["token_ids"]["mismatches"] == [], (arm, p["token_ids"]["mismatches"])
            assert p["template"]["render_mismatches"] == [], (arm, p["template"]["render_mismatches"])
            assert p["template"]["rendered_ids_mismatches"] == []
            assert p["byte_equal_to_upstream"]["chat_template.jinja"]


def test_committed_parity_record_is_current(fresh_parity):
    committed = json.loads(pc.PARITY_FILE.read_text())
    assert committed["schema"] == pc.PARITY_SCHEMA
    strip = lambda r: {k: v for k, v in r.items() if k != "utc"}  # noqa: E731
    assert strip(committed) == json.loads(json.dumps(strip(fresh_parity))), \
        "tools/peer_parity_build.json is stale: rerun tools/peer_check.py --parity"


def test_static_check_on_the_peer_folders(peers):
    for arm in pc.ARM_DIRS:
        r = pc.static_check(arm, peers)
        assert r["present"] and r["problems"] == [], (arm, r["problems"])
        assert all(r["same_files_as_build_parity"].values())
        assert r["config"]["ok"] and r["config"]["base"]["bits"] == pc.ARM_BITS[arm]
        assert r["eos"]["generation_config"]
        assert r["params_config"]["rel_diff_vs_published"] <= pc.PARAM_TOLERANCE
        if (pc.common.EXP_DIR / "runner" / "templates").is_dir():
            assert r["template_equals_committed"] is True


def test_static_check_flags_changed_files(peers, tmp_path):
    import shutil

    root = tmp_path / "peers"
    shutil.copytree(peers / pc.ARM_DIRS["G8"], root / pc.ARM_DIRS["G8"],
                    ignore=shutil.ignore_patterns("*.safetensors", ".cache"))
    with open(root / pc.ARM_DIRS["G8"] / "chat_template.jinja", "a") as f:
        f.write("\n{# edited #}\n")
    r = pc.static_check("G8", root)
    assert r["same_files_as_build_parity"]["chat_template.jinja"] is False
    assert any("differ from those of the build-time parity" in p for p in r["problems"])


def test_config_check_and_overrides():
    cfg = {"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
                            "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8},
                            "language_model.model.layers.1.router.proj": {"group_size": 64, "bits": 8}}}
    r = pc.config_check(cfg, 4)
    assert r["ok"] and r["n_overrides"] == 2
    assert r["overrides"] == {"router.proj": [[8, 64]]}
    assert not pc.config_check(cfg, 8)["ok"]


class _Tok:
    def __init__(self, shift=0):
        self.shift = shift

    def encode(self, s, add_special_tokens=False):
        return [ord(c) + (self.shift if i == 3 else 0) for i, c in enumerate(s)]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_token_parity_reports_first_difference():
    same = pc.token_parity(_Tok(), _Tok(), strings=("abcdef",))
    assert same["mismatches"] == [] and same["decode_roundtrip_differs"] == 0
    diff = pc.token_parity(_Tok(), _Tok(shift=1), strings=("abcdef", "ab"))
    assert diff["mismatches"] == [{"string": 0, "len_a": 6, "len_b": 6, "first_diff": 3}]


def test_fidelity_rule_and_summaries():
    stats = {"T1": {"tokens": 100, "bytes": 400, "nll8": 200.0, "nll4": 199.0, "kl": 5.0},
             "T2": {"tokens": 100, "bytes": 450, "nll8": 220.0, "nll4": 220.0, "kl": 6.0}}
    r = pc.fidelity_rule(stats)
    assert r["nll_ok"] and r["kl_ok"]
    assert abs(r["nll8_per_token"] - 2.1) < 1e-12 and abs(r["kl_8v4_per_token"] - 0.055) < 1e-12
    stats["T2"]["nll8"] = 230.0
    assert not pc.fidelity_rule(stats)["nll_ok"]
    stats["T1"]["kl"] = 60.0
    assert not pc.fidelity_rule(stats)["kl_ok"]
    s = pc.nll_summary(stats, "nll8")
    assert set(s["per_text_bpb"]) == {"T1", "T2"}
    assert abs(s["per_text"]["T1"]["bpb"] - 200.0 / 0.6931471805599453 / 400) < 1e-9


def test_flip_rate_rule():
    a = ["A", "B", "C", None] + ["D"] * 46
    b = ["A", "B", "C", "A"] + ["D"] * 46
    r = pc.flip_rate(a, b)
    assert r == {"n": 50, "flips": 1, "rate": 0.02, "ok": True}
    assert not pc.flip_rate(a, ["X"] + b[1:3] + ["A"] + b[4:])["ok"]
    with pytest.raises(ValueError):
        pc.flip_rate(["A"], [])


def test_verdicts_and_exit_code():
    assert pc._verdict({"problems": ["x"]}) == "fail"
    assert pc._verdict({"problems": [], "batched_path": {"ok": False}}) == "B=1"
    assert pc._verdict({"problems": []}) == "ok"
    assert pc.exit_code({"arms": {"G8": {"verdict": "ok"}, "G4": {"verdict": "B=1"}}}) == 2
    assert pc.exit_code({"arms": {"G8": {"verdict": "fail"}}}) == 1
    assert pc.exit_code({"arms": {"G8": {"verdict": "ok"}}}) == 0


def test_upstream_pins_come_from_assets_json():
    pins = pc.upstream_pins()
    assert pins["google/gemma-4-26B-A4B-it"][0].startswith("20da991a")
    for fam in pc.FAMILIES.values():
        assert fam["upstream_repo"] in pins
