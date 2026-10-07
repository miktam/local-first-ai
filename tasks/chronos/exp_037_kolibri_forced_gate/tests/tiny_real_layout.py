# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06). Lifted from exp_036's
# diagnostics/gate1/mini/bughunt/verify/build_ck.py.
"""Tiny real-layout Kolibri 1 checkpoints for the exp_037 calibrators (DESIGN §5.1).

The released model's layer and head layout at a tiny width, on tiny_checkpoint.py's
"w513" preset (sliding_window 513):
  - 10 layers, full attention at i % 5 == 4;
  - 48 q / 4 kv heads of head_dim 128, so the q width (6144) differs from hidden (256);
  - 64 experts, top-6; moe and shared intermediate 64;
  - max_position_embeddings 32768;
  - the q_norm and k_norm weights multiplied by `sharpen` (3; 1 for the whole tiny
    gate's validation build, below) before the BF16 cast.

write_real_layout(out, seed, sharpen=3.0) is build_ck.py's recipe as a function. At
seed 23 it reproduces exp_036's bughunt build ckv_s3 byte for byte: shards, index,
config and generation config, against the sha256 values in exp_036's
diagnostics/gate1/mini/MANIFEST.json (tests/test_tiny_real_layout.py). The factor
is what makes the routing chaos visible on tiny weights: the port against itself
(prefill chunks of 2,048 vs 64, 16,384 positions) had no router flip on the
unsharpened twin ckv_s1 and 16,239 on ckv_s3 (exp_036
diagnostics/gate1/mini/bughunt/verify/v4_ckv_s{1,3}_16384.json).

build_real_layout_set(root, seed) writes the set the calibrators use, laid out as
the gate's models dir:
  root/Kolibri-1-BF16           the source layout of the released Kolibri-1-BF16:
                                no model_file and no kolibri1.py, plus a stub
                                word-level tokenizer over t0..t1023, which mlx_lm
                                needs to convert
  root/Kolibri-1-MLX-8bit-g64   port/convert.py, 8 bits, group 64
  root/Kolibri-1-MLX-4bit-g64   port/convert.py, 4 bits, group 64
Both conversions use the pre-registered head policy: embedding and head quantised
at the arm's bits.

Which build serves which test (DESIGN §5.1; decision (a), 2026-10-06). conftest.py
provides each as a session fixture:
  tiny_real_cal          seed 23, sharpen 3: F_tiny's calibration checkpoint, and the
                         fp32 forced checks' own tests (G4-F32, ForcedTrunk, the
                         reference drivers)
  tiny_real_val          seed 29, sharpen 3: F_tiny's out-of-sample validation
                         (tests/test_gate_ftiny.py)
  tiny_real_val_unsharp  seed 29, sharpen 1 (build_gate_validation_set): the
                         validation build of the whole tiny gate (the end-to-end
                         test, the drivers test's port run, tools/dry_run.py's gate
                         stage). The x 3 sharpening makes 8-bit quantisation of random
                         tiny weights unrepresentative (KL(R1||R2F) 0.21 with routing
                         held, top-1 agreement 12 %), so on the sharpened build the
                         checks with a quantised or bf16 arm measure fixture noise, not
                         code (DESIGN §11.13, the stress finding).
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import tiny_checkpoint as tc

SEED_CAL = 23  # F_tiny calibration (DESIGN §5.1)
SEED_VAL = 29  # out-of-sample validation (DESIGN §5.1)
SHARPEN = 3.0  # q/k-norm factor (build_ck.py's ckv_s3): F_tiny's calibration and validation builds
SHARPEN_GATE = 1.0  # the whole tiny gate's validation build: unsharpened (DESIGN §5.1, decision (a))
PRESET = "w513"
GROUP_SIZE = 64
BITS = (8, 4)
BF16_NAME = "Kolibri-1-BF16"

# build_ck.py's `base`, value for value and in its order.
_REAL_LAYOUT = {
    "num_hidden_layers": 10,
    "layer_types": tc._layer_types(10, 5),
    "sliding_window": 513,
    "max_position_embeddings": 32768,
    "num_attention_heads": 48,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "hidden_size": 256,
    "num_experts": 64,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64,
}


def real_layout_overrides() -> dict:
    """The config overrides on the "w513" preset (a fresh copy)."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _REAL_LAYOUT.items()}


def real_layout_config(hf_layout: bool = False) -> dict:
    """The config.json write_real_layout writes (a fresh copy)."""
    cfg = tc.preset_config(PRESET)
    cfg.update(real_layout_overrides())
    if hf_layout:
        cfg.pop("model_file")
    return cfg


def sharpen_qk_norms(factor: float) -> Callable[[dict[str, np.ndarray]], None]:
    """build_ck.py's sharpen(f): multiply every q_norm and k_norm weight by
    `factor` in float32, before write_tiny_checkpoint's BF16 cast."""

    def edit(t: dict[str, np.ndarray]) -> None:
        for k in list(t):
            if k.endswith("q_norm.weight") or k.endswith("k_norm.weight"):
                t[k] = t[k] * factor

    return edit


def write_real_layout(out, seed: int, sharpen: float = SHARPEN, *, hf_layout: bool = False) -> Path:
    """Write a tiny real-layout BF16 checkpoint into `out` and return its path.

    hf_layout=False (build_ck.py's form): config.json names model_file
    "kolibri1.py" and the directory holds a copy of port/kolibri1.py
    (tiny_checkpoint.PORT_MODEL_FILE). mlx-lm 0.32.0 loads it with
    trust_remote_code=True only.
    hf_layout=True: no model_file and no kolibri1.py, as in the released
    Kolibri-1-BF16 (the input convert.py expects). The shards and the index are
    the same bytes in both forms."""
    out = tc.write_tiny_checkpoint(
        out,
        seed=seed,
        preset=PRESET,
        copy_port=not hf_layout,
        overrides=real_layout_overrides(),
        edit_tensors=sharpen_qk_norms(sharpen),
    )
    if hf_layout:
        cfg = json.loads((out / "config.json").read_text())
        cfg.pop("model_file", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return out


def write_stub_tokenizer(d) -> None:
    """A word-level tokenizer over t0..t1023 (EOS t1022, PAD t1021), so that
    mlx_lm, which loads a tokenizer while it converts, has one. Offline. It
    plays no part in any logit."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    d = Path(d)
    tok = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(tc.VOCAB_SIZE)}, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": f"t{tc.EOS_TOKEN_ID}",
                "pad_token": f"t{tc.PAD_TOKEN_ID}",
            }
        )
    )


@dataclass(frozen=True)
class RealLayoutSet:
    """One tiny real-layout checkpoint and its two conversions. Shared by a test
    session: copy a directory before changing anything in it."""

    seed: int
    sharpen: float
    root: Path  # the models dir: BF16_NAME and the two Kolibri-1-MLX-*bit-g64 builds
    bf16: Path
    k8: Path
    k4: Path

    def build(self, bits: int) -> Path:
        return {8: self.k8, 4: self.k4}[bits]


def build_dir_name(bits: int) -> str:
    return f"Kolibri-1-MLX-{bits}bit-g{GROUP_SIZE}"


def build_real_layout_set(root, seed: int, sharpen: float = SHARPEN) -> RealLayoutSet:
    """root/Kolibri-1-BF16 (write_real_layout, hf_layout=True, plus the stub
    tokenizer) and its 8-bit and 4-bit group-64 conversions by port/convert.py."""
    from port import convert  # the experiment directory is on sys.path (conftest.py)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    bf16 = write_real_layout(root / BF16_NAME, seed, sharpen, hf_layout=True)
    write_stub_tokenizer(bf16)
    builds = {}
    for bits in BITS:
        out = root / build_dir_name(bits)
        with contextlib.redirect_stdout(io.StringIO()):
            convert.convert(bf16, out, bits, GROUP_SIZE, guard=False)
        builds[bits] = out
    return RealLayoutSet(seed=seed, sharpen=sharpen, root=root, bf16=bf16, k8=builds[8], k4=builds[4])


def build_gate_validation_set(root) -> RealLayoutSet:
    """The validation build of the whole tiny gate (DESIGN §5.1, decision (a)): seed 29
    with the q/k norms unsharpened (sharpen = 1), BF16 and its 8-bit and 4-bit group-64
    conversions. Every check of the tiny gate is judged on it, those with a quantised or
    bf16 arm (G4-N(i), G4-F16 with R3, G4 K4, G5-R1, G5-BP-lean, behaviour, the noise
    floor, free decode-vs-prefill) and that run's fp32 forced checks and control margins
    alike. One definition for conftest.py's tiny_real_val_unsharp and tools/dry_run.py."""
    return build_real_layout_set(root, SEED_VAL, SHARPEN_GATE)
