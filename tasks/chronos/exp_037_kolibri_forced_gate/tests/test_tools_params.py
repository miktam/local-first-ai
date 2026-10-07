"""tools/params.py (HYPOTHESIS C16, "Arms"; BUILD_SPEC §5.9 peer check):
config arithmetic and the logical count of loaded (quantised) models."""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

import tiny_checkpoint as tc
from exp036_helpers import import_sibling
from tools import params

KOLIBRI_TOTAL = 78_103_074_560          # HYPOTHESIS "Arms", model card
KOLIBRI_ACTIVE_CARD = 3_457_573_120


def _kolibri_config_dir() -> Path | None:
    for var in ("EXP036_TOKENIZER_DIR", "EXP036_TOK"):
        v = os.environ.get(var)
        if v and (Path(v) / "config.json").is_file():
            return Path(v)
    v = os.environ.get("EXP036_MODELS")
    if v and (Path(v) / "Kolibri-1-BF16" / "config.json").is_file():
        return Path(v) / "Kolibri-1-BF16"
    return None


def test_kolibri_config_arithmetic():
    d = _kolibri_config_dir()
    if d is None:
        pytest.skip("no Kolibri config.json (set EXP036_TOKENIZER_DIR or EXP036_TOK)")
    import_sibling("port.kolibri1")
    p = params.config_params(d)
    assert p["total"] == KOLIBRI_TOTAL
    assert p["num_experts"] == 384 and p["top_k"] == 6
    # Our convention excludes the input embedding; the card's figure differs by one 2,560 vector.
    assert abs(p["active_excl_input_embedding"] - KOLIBRI_ACTIVE_CARD) <= 2560


def test_tiny_config_count_equals_the_checkpoint(tiny_vendor_dir):
    import_sibling("port.kolibri1")
    cfg = tc.preset_config("vendor")
    tensors = tc.make_tensors(cfg, seed=0)
    total = sum(int(np.prod(t.shape)) for t in tensors.values())
    p = params.config_params(tiny_vendor_dir)
    assert p["total"] == total
    n_layers, E, I, H = cfg["num_hidden_layers"], cfg["num_experts"], cfg["moe_intermediate_size"], cfg["hidden_size"]
    assert p["routed_expert_params"] == n_layers * E * 3 * I * H
    assert p["active"] == total - p["routed_expert_params"] + p["routed_expert_params"] * cfg["num_experts_per_tok"] // E


def _minimal_tokenizer(d: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(tc.VOCAB_SIZE)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "PreTrainedTokenizerFast",
                                                         "eos_token": f"t{tc.EOS_TOKEN_ID}"}))


@pytest.mark.parametrize("bits", [8, 4])
def test_loaded_quantised_count_equals_config(tmp_path, bits):
    convert = import_sibling("port.convert")
    src = tc.write_tiny_checkpoint(tmp_path / "src", seed=0, preset="vendor", copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg))
    _minimal_tokenizer(src)
    out = tmp_path / f"tiny-{bits}bit"
    with contextlib.redirect_stdout(io.StringIO()):
        convert.convert(src, out, bits, 64, guard=False)
    from mlx_lm.utils import load_model

    # mlx-lm 0.32.0 (#1385; decision F2): the conversion names model_file kolibri1.py, our own port, which
    # mlx_lm executes only with trust_remote_code=True.
    model, _ = load_model(out, strict=True, trust_remote_code=True)
    want = params.config_params(src)
    got = params.loaded_params(model, want["num_experts"], want["top_k"], want["tied_embeddings"])
    assert got["total"] == want["total"]
    assert got["routed_expert_params"] == want["routed_expert_params"]


@pytest.mark.parametrize("folder,published", [
    ("gemma-4-26b-a4b-it-8bit", 25.2e9), ("Qwen3.6-35B-A3B-8bit", 34.7e9), ("Qwen3.8-27B-8bit", 27.0e9),
])
def test_peer_config_arithmetic_is_within_2_percent_of_the_published_totals(folder, published):
    from exp036_helpers import peer_folder

    d = peer_folder(folder)
    if d is None:
        pytest.skip("peer configs not found (set EXP036_MINI_ASSETS on the mini, EXP036_MODELS on the run host)")
    p = params.config_params(d)
    assert abs(p["total"] - published) / published <= 0.02
