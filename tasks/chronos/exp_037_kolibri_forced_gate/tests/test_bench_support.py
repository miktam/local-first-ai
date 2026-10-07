# SPDX-License-Identifier: MIT
"""Shared helpers and fixtures for the bench tests (tests/test_bench_*.py).

No tests live here. The other bench test modules import the fixtures by name.

- write_byte_tokenizer(): a byte-level BPE tokenizer (one token per UTF-8
  byte, ids 0-255; <|endoftext|> 1021 and <|im_end|> 1022 as in the tiny
  checkpoints), so mlx_lm.load gives a working TokenizerWrapper offline.
- tiny_model_dir(): a tiny Kolibri checkpoint (tests/tiny_checkpoint.py) plus
  that tokenizer. Its config.json names model_file "kolibri1.py" and it has no
  convert record, so the tests load it with mlx_lm.load(...,
  trust_remote_code=True) directly (mlx-lm 0.32.0, #1385; exp_037 DESIGN
  §6.5): the kit's own writers made it. bench/genutil.load_arm instead goes
  through port.convert.model_file_trust (tested on a tiny conversion).
- bench_env: stubs for the sibling modules the bench calls (runner.guard,
  tools.redact when absent), no-op sleeps, fake thermal/power/sysctl output,
  and $EXP036_MODELS / $EXP036_WORK / $EXP036_DATA in a temporary directory
  ($EXP037_BUILDS unset). The gate stub returns a gate-record-like dict whose
  allowed_B is stubs.allowed_B (all of 1, 2, 4, 8, 16 unless a test narrows
  it); the version-binding stub records ("environment",) and returns
  stubs.binding.
- mini_assets(): the mini's build-time asset root (~/models/exp036-mini, or
  $EXP036_MINI), for the tests with real tokenizers and the real parquet.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint

EOS_ID = tiny_checkpoint.EOS_TOKEN_ID  # 1022
PAD_ID = tiny_checkpoint.PAD_TOKEN_ID  # 1021


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte -> printable character map (what ByteLevel uses)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def write_byte_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    m = _bytes_to_unicode()
    vocab = {m[b]: b for b in range(256)}
    # Inert fillers keep the id range contiguous (mlx_lm's streaming
    # detokenizer indexes a list by id).
    vocab.update({f"<|f{i}|>": i for i in range(256, PAD_ID)})
    vocab["<|endoftext|>"] = PAD_ID
    vocab["<|im_end|>"] = EOS_ID
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_end|>"])
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<|im_end|>",
                "pad_token": "<|endoftext|>",
                "bos_token": None,
                "model_max_length": 10_000_000,
            }
        )
    )


def tiny_model_dir(out: Path, seed: int = 0, preset: str = "vendor", edit_tensors=None) -> Path:
    d = tiny_checkpoint.write_tiny_checkpoint(out, seed=seed, preset=preset, edit_tensors=edit_tensors)
    write_byte_tokenizer(d)
    return d


def eos_first_model_dir(out: Path) -> Path:
    """A tiny model whose greedy choice is always an EOS id: lm_head is zero,
    so every logit ties at 0 and argmax picks id 0, and id 0 is declared an
    EOS id in config.json and generation_config.json. With EOS masked, greedy
    picks id 1 and the run goes on."""

    def zero_head(t: dict) -> None:
        t["lm_head.weight"] = np.zeros_like(t["lm_head.weight"])

    d = tiny_model_dir(out, seed=0, edit_tensors=zero_head)
    eos = [0, EOS_ID, PAD_ID]
    for name in ("config.json", "generation_config.json"):
        cfg = json.loads((d / name).read_text())
        cfg["eos_token_id"] = eos
        (d / name).write_text(json.dumps(cfg, indent=2) + "\n")
    return d


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory) -> dict[str, Path]:
    """Two tiny models with different weights ("a", "b")."""
    return {
        "a": tiny_model_dir(tmp_path_factory.mktemp("bench_tiny_a"), seed=0),
        "b": tiny_model_dir(tmp_path_factory.mktemp("bench_tiny_b"), seed=1),
    }


def tiny_loader(mapping: dict[str, Path], calls: list | None = None):
    """A loader for the bench cells: arm -> (model, tokenizer, dir) of a tiny
    directory, through mlx_lm.load (the real loader also checks the port file
    of a converted Kolibri build, which tiny directories do not have)."""

    def load(arm: str):
        from mlx_lm import load as mlx_load

        if calls is not None:
            calls.append(arm)
        d = mapping[arm]
        model, tok = mlx_load(str(d), trust_remote_code=True)  # mlx-lm 0.32.0: our own tiny kolibri1.py
        return model, tok, d

    return load


ALL_B = [1, 2, 4, 8, 16]


class Stubs:
    def __init__(self):
        self.calls: list[tuple] = []
        self.sleeps: list[float] = []
        self.thermal = ["0"]  # successive osascript outputs; the last repeats
        self.allowed_B = {"K8": list(ALL_B), "K4": list(ALL_B)}  # the gate record's allowed_B (DESIGN §3.14)
        self.binding = {"observed": {"mlx": "stub"}, "judged": False}  # what require_environment returns

    def gate(self, arm: str) -> dict:
        self.calls.append(("gate", arm))
        return {"verdict": {arm: "PASS"}, "allowed_B": {a: list(v) for a, v in self.allowed_B.items()}}

    def environment(self, results_dir=None):
        self.calls.append(("environment",))
        return self.binding


def _fake_run(stubs: Stubs):
    def run(cmd, timeout=15.0):
        if cmd[0] == "osascript":
            out = stubs.thermal.pop(0) if len(stubs.thermal) > 1 else stubs.thermal[0]
            return None if out is None else out + "\n"
        if cmd[:3] == ["pmset", "-g", "ps"]:
            return "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged\n"
        if cmd[:2] == ["pmset", "-g"]:
            return "System-wide power settings:\nCurrently in use:\n lowpowermode         0\n powermode            2\n"
        if cmd[:2] == ["sysctl", "-n"]:
            return {"machdep.cpu.brand_string": "Apple M5 Max\n", "hw.memsize": f"{128 * 2**30}\n"}.get(cmd[2], "0\n")
        if cmd[0] == "git":
            return "0123456789abcdef0123456789abcdef01234567\n" if "rev-parse" in cmd else ""
        return None

    return run


def _sibling(modname: str):
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == modname.split(".")[0]:
            return None
        raise


@pytest.fixture
def bench_env(monkeypatch, tmp_path):
    """Isolate a bench test from siblings, the clock's sleeps and the host."""
    from bench import common

    stubs = Stubs()
    monkeypatch.setattr(common, "require_identity", lambda: stubs.calls.append(("identity",)))
    monkeypatch.setattr(common, "require_gate", stubs.gate)
    monkeypatch.setattr(common, "require_environment", stubs.environment)
    monkeypatch.setattr(common, "require_tier2", lambda: stubs.calls.append(("tier2",)))
    monkeypatch.setattr(common, "SLEEP", lambda s: stubs.sleeps.append(s))
    monkeypatch.setattr(common, "_run", _fake_run(stubs))
    redact = _sibling("tools.redact")
    if redact is None:
        home = str(Path.home())
        monkeypatch.setattr(common, "redact_path", lambda p: str(p).replace(home, "~"))
        monkeypatch.setattr(common, "host_label", lambda: "mbp (M5 Max, 128 GB)")
    for var, sub in (("EXP036_MODELS", "models"), ("EXP036_WORK", "work"), ("EXP036_DATA", "data")):
        p = tmp_path / sub
        p.mkdir(exist_ok=True)
        monkeypatch.setenv(var, str(p))
    monkeypatch.delenv("EXP037_BUILDS", raising=False)
    stubs.results = tmp_path / "results"
    stubs.root = tmp_path
    return stubs


def mini_assets() -> Path:
    return Path(os.environ.get("EXP036_MINI") or "~/models/exp036-mini").expanduser()


def mini_dir(*parts: str) -> Path:
    """A directory of the build-time assets, or skip. Peer folders ("mlx-community", <folder>) are also found
    in the run host's layout ($EXP036_MODELS/<folder>); "upstream" and "evalfw" exist only on the build host,
    so their absence is a "build-host only:" skip (tests/conftest.py)."""
    from exp036_helpers import BUILD_HOST_ONLY, peer_folder

    if len(parts) == 2 and parts[0] == "mlx-community":
        found = peer_folder(parts[1])
        if found is not None:
            return found
    p = mini_assets().joinpath(*parts)
    if not p.exists():
        if parts and parts[0] in ("upstream", "evalfw"):
            pytest.skip(f"{BUILD_HOST_ONLY} mini build asset {p} not present (set EXP036_MINI)")
        pytest.skip(f"mini build asset {p} not present (set EXP036_MINI, or EXP036_MODELS on the run host)")
    return p


def kolibri_tok_dir() -> Path:
    """The Kolibri tokenizer directory: $EXP036_TOK (BUILD_SPEC §6's name), $EXP036_TOKENIZER_DIR (the older
    name), $EXP036_MODELS/Kolibri-1-BF16 (the run host), else the mini's BF16 folder."""
    from exp036_helpers import kolibri_tok_env

    v = kolibri_tok_env()
    if v and (Path(v) / "tokenizer.json").is_file():
        return Path(v)
    return mini_dir("kolibri", "Kolibri-1-BF16")


def mini_parquet() -> Path:
    d = os.environ.get("EXP036_DATA")
    candidates = [Path(d).expanduser()] if d else []
    if os.environ.get("EXP036_MODELS"):
        candidates.append(Path(os.environ["EXP036_MODELS"]).expanduser() / "data")
    candidates.append(mini_assets() / "data")
    for c in candidates:
        p = c / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet"
        if p.is_file():
            return p
    pytest.skip(f"FineWeb-2 parquet not found under {[str(c) for c in candidates]}")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def assert_sorted_keys_jsonl(p: Path) -> None:
    for line in Path(p).read_text().splitlines():
        obj = json.loads(line)
        assert line == json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)
