# SPDX-License-Identifier: MIT
# Chronos exp_036: numpy unpacker for MLX affine-quantised checkpoints.
# Written by Miktam, 2026-10-03, from the storage format (below); it imports
# neither mlx nor the port, so the reference's dequantised mode (G2q) shares
# no code with the quantised forward it checks.
"""Read an MLX affine-quantised checkpoint (a directory written by
port/convert.py) as float32 weights under the original checkpoint names.

Storage format of an affine-quantised tensor W [..., out, in] with b bits and
group size g (MLX `mx.quantize(..., mode="affine")`):

    <module>.weight  uint32 [..., out, in * b / 32]   packed codes q in [0, 2^b - 1]
    <module>.scales  bf16   [..., out, in / g]         one scale per group of g inputs
    <module>.biases  bf16   [..., out, in / g]         one offset per group

For b in {2, 4, 8} each uint32 word holds 32 / b consecutive codes, the
first in the least significant bits: code j of word w is
(w >> (b * j)) & (2^b - 1). The value is w = scale * q + bias.

How the consuming MLX op uses that value matters at the last bit, so the
dequantisation is done the way each op does it (checked on mlx 0.31.2,
tests/test_port_ref_reference_modes.py):

* quantized_matmul / gather_qmm (QuantizedLinear, QuantizedSwitchLinear):
  scale * q + bias in fp32 without rounding to bf16; the result of an fp32
  input against bf16 scales is fp32 (spec item 14) and matches x @ W_fp32^T
  to fp32 accumulation error. -> dequantize(..., out="float32").
* mx.dequantize and QuantizedEmbedding: the same fp32 value rounded once,
  to nearest even, to the scales' dtype (bf16) on the GPU (Metal).
  -> dequantize(..., out="bfloat16"). (mlx's CPU kernel instead rounds the
  product and the sum separately in bf16; the experiment runs on the GPU.)

numpy computes scale * q and the + bias as two fp32 operations, as the Metal
kernel does; the tests show 0 differing bits against mx.dequantize on both
paths at 8 and 4 bits.
"""

from __future__ import annotations

import json
import os

import numpy as np

try:  # package import (python -m reference.kolibri_ref, tests)
    from .safetensors_np import open_checkpoint
except ImportError:  # direct script execution from inside reference/
    from safetensors_np import open_checkpoint

SUPPORTED_BITS = (2, 4, 8)


def round_bf16(x: np.ndarray) -> np.ndarray:
    """float32 -> nearest bfloat16 (ties to even) -> float32, in numpy.

    The bf16 value is the upper 16 bits of the float32; adding 0x7FFF plus the
    lowest kept bit and truncating rounds to nearest even. NaN stays NaN;
    overflow goes to +-inf as in a hardware conversion. Subnormals are
    rounded, not flushed (IEEE; mlx's CPU conversion does the same, Metal
    flushes them to zero). The uint32 sum can wrap only for NaN patterns,
    which are restored afterwards; working in uint32 keeps the temporaries at
    the input's size (the emulation rounds [16384, 6144] arrays on T9)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits >> 16) & np.uint32(1)
    rounded += np.uint32(0x7FFF)
    rounded += bits
    rounded &= np.uint32(0xFFFF0000)
    out = rounded.view(np.float32)
    nan = np.isnan(x)
    if nan.any():
        out[nan] = x[nan]
    return out


def unpack(packed: np.ndarray, bits: int) -> np.ndarray:
    """uint32 words [..., n] -> codes uint32 [..., n * 32 / bits]."""
    if bits not in SUPPORTED_BITS:
        raise NotImplementedError(f"{bits}-bit packing (supported: {SUPPORTED_BITS})")
    packed = np.asarray(packed)
    if packed.dtype != np.uint32:
        raise TypeError(f"packed weight must be uint32, got {packed.dtype}")
    per_word = 32 // bits
    mask = np.uint32((1 << bits) - 1)
    shifts = (np.arange(per_word, dtype=np.uint32) * np.uint32(bits))
    codes = (packed[..., None] >> shifts) & mask  # [..., n, per_word]
    return codes.reshape(packed.shape[:-1] + (packed.shape[-1] * per_word,))


def dequantize(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray, bits: int,
               group_size: int, out: str = "float32") -> np.ndarray:
    """w = scale * q + bias per group of `group_size` inputs, in fp32.

    scales / biases: float32 arrays holding the stored (bf16) values, as
    safetensors_np returns them. out="float32" keeps the fp32 value (what
    quantized_matmul multiplies with); out="bfloat16" rounds it to nearest
    even bf16 (mx.dequantize and QuantizedEmbedding). Returns float32."""
    codes = unpack(packed, bits)
    n_in = codes.shape[-1]
    if n_in % group_size:
        raise ValueError(f"{n_in} inputs are not a multiple of group size {group_size}")
    groups = n_in // group_size
    scales = np.asarray(scales, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    if scales.shape[-1] != groups or biases.shape != scales.shape:
        raise ValueError(f"scales {scales.shape} / biases {biases.shape} do not match {groups} groups")
    q = codes.astype(np.float32).reshape(codes.shape[:-1] + (groups, group_size))
    w = q * scales[..., None]
    w += biases[..., None]
    w = w.reshape(codes.shape)
    if out == "bfloat16":
        return round_bf16(w)
    if out != "float32":
        raise ValueError(f"out must be 'float32' or 'bfloat16', not {out!r}")
    return w


# ---------------------------------------------------------------------------
# A converted directory under the original (HF) tensor names
# ---------------------------------------------------------------------------


class ConvertedCheckpoint:
    """The tensors of a directory written by port/convert.py, presented under
    the names of the BF16 source checkpoint, so KolibriReference can run on
    them unchanged (its dequantised mode, HYPOTHESIS "Reference"):

      model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight
          -> row E of model.layers.N.mlp.switch_mlp.{...}_proj (stacked by sanitize)
      model.layers.N.moe.router.expert_bias -> model.layers.N.mlp.gate.expert_bias
      every other name -> the same name.

    Quantised tensors are dequantised on read: model.embed_tokens rounded to
    bf16 (QuantizedEmbedding), every linear weight kept in fp32 (quantized
    matmul); see the module docstring. Unquantised tensors are returned as
    stored (bf16 or fp32, widened exactly). Implements the subset of
    safetensors_np.Checkpoint the reference uses: names, __contains__,
    shape_of, dtype_of, get, get_rows, get_raw."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json")) as f:
            self.config = json.load(f)
        quant = self.config.get("quantization")
        if not isinstance(quant, dict):
            raise ValueError(f"{model_dir}: config.json has no quantization block; not a converted model")
        if quant.get("mode", "affine") != "affine":
            raise ValueError(f"{model_dir}: quantization mode {quant.get('mode')!r}, expected affine")
        self.quantization = quant
        self.stored = open_checkpoint(model_dir)
        n_experts = int(self.config["num_experts"])
        # HF name -> (stored key, quantised, expert row or None). The stored
        # key is the module path for a quantised weight, else the tensor name.
        self._entries: dict[str, tuple[str, bool, int | None]] = {}
        for stored_name in self.stored.names():
            module, param = stored_name.rsplit(".", 1)
            if param in ("scales", "biases"):
                continue
            quantised = param == "weight" and f"{module}.scales" in self.stored
            key = module if quantised else stored_name
            if ".mlp.switch_mlp." in module and param == "weight":
                prefix, proj = module.split(".mlp.switch_mlp.")
                n = self._stored_shape(key, quantised)[0]
                if n != n_experts:
                    raise ValueError(f"{stored_name}: {n} experts, config says {n_experts}")
                for e in range(n_experts):
                    self._entries[f"{prefix}.mlp.experts.{e}.{proj}.weight"] = (key, quantised, e)
            elif module.endswith(".mlp.gate") and param == "expert_bias":
                hf = module[: -len(".mlp.gate")] + ".moe.router.expert_bias"
                self._entries[hf] = (key, False, None)
            else:
                self._entries[stored_name] = (key, quantised, None)

    def _params(self, module: str) -> tuple[int, int]:
        """(bits, group_size) of a quantised module, honouring per-path
        overrides in config["quantization"]."""
        override = self.quantization.get(module)
        if isinstance(override, dict):
            return int(override["bits"]), int(override["group_size"])
        return int(self.quantization["bits"]), int(self.quantization["group_size"])

    def _stored_shape(self, key: str, quantised: bool) -> tuple[int, ...]:
        if not quantised:
            return tuple(self.stored.shape_of(key))
        bits, _ = self._params(key)
        shape = tuple(self.stored.shape_of(f"{key}.weight"))
        return shape[:-1] + (shape[-1] * 32 // bits,)

    def _entry(self, name: str):
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"tensor not in converted checkpoint: {name}") from None

    def is_quantised(self, name: str) -> bool:
        return self._entry(name)[1]

    # ---- Checkpoint interface ----------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def shape_of(self, name: str) -> tuple[int, ...]:
        key, quantised, row = self._entry(name)
        shape = self._stored_shape(key, quantised)
        return shape[1:] if row is not None else shape

    def dtype_of(self, name: str) -> str:
        key, quantised, _ = self._entry(name)
        return "F32" if quantised else self.stored.dtype_of(key)

    def _read(self, name: str, rows=None) -> np.ndarray:
        key, quantised, row = self._entry(name)
        # Rows of the stored tensor to read: one expert of a stacked tensor,
        # or the requested rows of an ordinary one.
        sel = slice(row, row + 1) if row is not None else rows

        def read(tensor):
            return self.stored.get(tensor) if sel is None else self.stored.get_rows(tensor, sel)

        if quantised:
            bits, group_size = self._params(key)
            out = "bfloat16" if key.endswith("embed_tokens") else "float32"
            value = dequantize(read(f"{key}.weight"), read(f"{key}.scales"), read(f"{key}.biases"),
                               bits, group_size, out=out)
        else:
            value = read(key)
        if row is not None:
            value = value[0]
            if rows is not None:
                value = value[rows]
        return value

    def get(self, name: str) -> np.ndarray:
        return self._read(name)

    def get_rows(self, name: str, rows) -> np.ndarray:
        return self._read(name, rows)

    def get_raw(self, name: str) -> np.ndarray:
        key, quantised, _ = self._entry(name)
        return self.stored.get_raw(f"{key}.weight" if quantised else key)


def open_converted(model_dir: str) -> ConvertedCheckpoint:
    return ConvertedCheckpoint(model_dir)
