# SPDX-License-Identifier: MIT
# Chronos exp_036: minimal safetensors reader for the independent numpy reference.
# Written by Miktam, 2026-10-03, from the safetensors format description only
# (no code taken from the safetensors package, MLX or torch).
"""Read safetensors checkpoints with numpy only.

Format (https://github.com/huggingface/safetensors, "Format" section):

    [8 bytes]  N, unsigned 64-bit little-endian: length of the JSON header
    [N bytes]  UTF-8 JSON: {"__metadata__": {...}?,
                            "<name>": {"dtype": "BF16", "shape": [...],
                                       "data_offsets": [begin, end]}, ...}
    [rest]     raw little-endian tensor bytes; offsets are relative to the
               first byte after the header (byte 8 + N of the file)

Each tensor is memory-mapped on demand (np.memmap over exactly its byte
range) and decoded into a fresh float32 array, after which the mapping is
dropped. Nothing else in the file is read, so a 156 GB checkpoint can be
consumed one tensor at a time.

BF16 decoding is exact: a bfloat16 value is the upper 16 bits of the
float32 with the same sign, exponent and leading mantissa bits, so
float32_bits = uint16_bits << 16.
"""

import json
import os

import numpy as np

# safetensors dtype string -> (numpy storage dtype, bytes per element).
# BF16 has no numpy dtype; it is stored as raw uint16 bit patterns.
_STORAGE = {
    "BF16": (np.dtype("<u2"), 2),
    "F16": (np.dtype("<f2"), 2),
    "F32": (np.dtype("<f4"), 4),
    "I64": (np.dtype("<i8"), 8),
    # Packed codes of MLX affine-quantised tensors (converted checkpoints;
    # see mlx_affine_np.py). Returned as uint32, never as float.
    "U32": (np.dtype("<u4"), 4),
}

# A header larger than this is treated as corruption rather than parsed.
_MAX_HEADER_BYTES = 256 * 1024 * 1024

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """Exact bfloat16 -> float32: widen the uint16 bit pattern and shift it
    into the high half of a uint32, then reinterpret. Returns a new array."""
    wide = bits.astype(np.uint32)  # copy
    wide <<= 16
    return wide.view(np.float32)


def parse_header_bytes(buf: bytes) -> tuple[dict, dict, int]:
    """Parse the 8-byte length prefix and JSON header from the start of a
    safetensors file. Returns (tensors, metadata, data_start) where
    tensors maps name -> {"dtype", "shape", "begin", "end"} with begin/end
    relative to data_start (absolute file offset of the data section)."""
    if len(buf) < 8:
        raise ValueError("safetensors: file shorter than the 8-byte header length")
    n = int.from_bytes(buf[:8], "little", signed=False)
    if n > _MAX_HEADER_BYTES:
        raise ValueError(f"safetensors: implausible header length {n}")
    if len(buf) < 8 + n:
        raise ValueError(f"safetensors: header claims {n} bytes, only {len(buf) - 8} present")
    raw = json.loads(bytes(buf[8 : 8 + n]).decode("utf-8"))
    metadata = raw.pop("__metadata__", {}) or {}
    tensors = {}
    for name, info in raw.items():
        dtype = info["dtype"]
        shape = tuple(int(s) for s in info["shape"])
        begin, end = (int(o) for o in info["data_offsets"])
        if end < begin:
            raise ValueError(f"safetensors: {name}: data_offsets end < begin")
        if dtype in _STORAGE:
            itemsize = _STORAGE[dtype][1]
            expected = itemsize * int(np.prod(shape, dtype=np.int64))
            if end - begin != expected:
                raise ValueError(
                    f"safetensors: {name}: {dtype}{list(shape)} needs {expected} bytes, "
                    f"header gives {end - begin}"
                )
        tensors[name] = {"dtype": dtype, "shape": shape, "begin": begin, "end": end}
    return tensors, metadata, 8 + n


def read_header(path: str) -> tuple[dict, dict, int]:
    """Read and parse only the header of one safetensors file."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) < 8:
            raise ValueError(f"{path}: not a safetensors file (too short)")
        n = int.from_bytes(prefix, "little", signed=False)
        if n > _MAX_HEADER_BYTES:
            raise ValueError(f"{path}: implausible header length {n}")
        body = f.read(n)
    tensors, metadata, data_start = parse_header_bytes(prefix + body)
    size = os.path.getsize(path)
    for name, t in tensors.items():
        if data_start + t["end"] > size:
            raise ValueError(f"{path}: tensor {name} runs past end of file")
    return tensors, metadata, data_start


class Checkpoint:
    """A set of safetensors files viewed as one name -> tensor mapping.

    Use open_checkpoint(model_dir) to construct. All accessors read only the
    bytes of the requested tensor (or rows of it)."""

    def __init__(self, files: list[str]):
        self._entries = {}  # name -> (path, data_start, info)
        self.metadata = {}
        for path in files:
            tensors, metadata, data_start = read_header(path)
            self.metadata[os.path.basename(path)] = metadata
            for name, info in tensors.items():
                if name in self._entries:
                    raise ValueError(f"tensor {name} appears in more than one file")
                self._entries[name] = (path, data_start, info)

    # ---- metadata -------------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def dtype_of(self, name: str) -> str:
        return self._entry(name)[2]["dtype"]

    def shape_of(self, name: str) -> tuple[int, ...]:
        return self._entry(name)[2]["shape"]

    def file_of(self, name: str) -> str:
        return self._entry(name)[0]

    # ---- data -----------------------------------------------------------

    def get(self, name: str) -> np.ndarray:
        """Return the whole tensor as a new float32 array (BF16/F16/F32).
        I64 tensors are returned as a new int64 array instead, since a cast
        to float32 would lose integers above 2**24, and U32 tensors (packed
        quantised codes) as a new uint32 array."""
        return self._decode(self._map(name), self.dtype_of(name))

    def get_rows(self, name: str, rows) -> np.ndarray:
        """Return rows of a tensor along axis 0 as a new array (float32, or
        int64 for I64, uint32 for U32). `rows` is a slice or an integer index array; only the
        pages holding those rows are read (used for the embedding gather and
        the chunked LM head)."""
        mm = self._map(name)
        if isinstance(rows, slice):
            part = mm[rows]
        else:
            part = mm[np.asarray(rows, dtype=np.int64)]
        return self._decode(part, self.dtype_of(name))

    def get_raw(self, name: str) -> np.ndarray:
        """Return a copy of the stored values in their storage dtype (uint16
        bit patterns for BF16). For bit-level comparisons."""
        return np.array(self._map(name), copy=True)

    # ---- internals ------------------------------------------------------

    def _entry(self, name: str):
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(f"tensor not in checkpoint: {name}") from None

    def _map(self, name: str) -> np.ndarray:
        """Memory-map exactly one tensor's bytes (read-only). The mapping is
        released when the returned array and its views are garbage collected."""
        path, data_start, info = self._entry(name)
        dtype = info["dtype"]
        if dtype not in _STORAGE:
            raise TypeError(f"{name}: dtype {dtype} not supported (BF16, F16, F32, I64, U32 only)")
        storage, _ = _STORAGE[dtype]
        shape = info["shape"]
        if info["end"] == info["begin"]:  # zero-size tensor: nothing to map
            return np.zeros(shape, dtype=storage)
        return np.memmap(path, dtype=storage, mode="r", offset=data_start + info["begin"], shape=shape)

    @staticmethod
    def _decode(stored: np.ndarray, dtype: str) -> np.ndarray:
        if dtype == "BF16":
            return bf16_bits_to_f32(stored)
        if dtype == "I64":
            return np.array(stored, dtype=np.int64, copy=True)
        if dtype == "U32":
            return np.array(stored, dtype=np.uint32, copy=True)
        # F16 widens exactly; F32 is copied out of the mapping.
        return np.array(stored, dtype=np.float32, copy=True)


def open_checkpoint(model_dir: str) -> Checkpoint:
    """Open model.safetensors.index.json (sharded) if present, otherwise a
    single model.safetensors, in model_dir."""
    index_path = os.path.join(model_dir, INDEX_FILE)
    single_path = os.path.join(model_dir, SINGLE_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        files = sorted(set(weight_map.values()))
        ckpt = Checkpoint([os.path.join(model_dir, fn) for fn in files])
        # The index and the shard headers must agree on where each tensor is.
        for name, fn in weight_map.items():
            if name not in ckpt:
                raise ValueError(f"index lists {name} in {fn}, but no shard header has it")
            if os.path.basename(ckpt.file_of(name)) != fn:
                raise ValueError(f"index puts {name} in {fn}, header in {ckpt.file_of(name)}")
        if len(weight_map) != len(ckpt.names()):
            extra = sorted(set(ckpt.names()) - set(weight_map))[:5]
            raise ValueError(f"shard headers hold tensors the index does not list, e.g. {extra}")
        return ckpt
    if os.path.isfile(single_path):
        return Checkpoint([single_path])
    raise FileNotFoundError(f"neither {INDEX_FILE} nor {SINGLE_FILE} in {model_dir}")
