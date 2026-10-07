# SPDX-License-Identifier: MIT
"""reference/safetensors_np.py, the reference's own safetensors reader
(BUILD_SPEC 5.2, `test_safetensors_np.py`; named test_port_ref_* here):

* header parsing: length prefix, metadata, offsets, and the refusals;
* bf16 widening is exact for all 65,536 bit patterns;
* tensors spread over several shards are found through the index, with
  offsets relative to each shard's data section;
* packed uint32 tensors (converted checkpoints) come back as uint32.

The files are written here by hand from the format description, plus one
written by mlx, so the reader is checked against two independent writers.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from exp036_helpers import import_sibling


def _write_safetensors(path, tensors: dict, metadata=None, pad_header: int = 0) -> None:
    """Minimal writer from the format description. tensors: name -> (dtype
    string, shape, raw little-endian bytes)."""
    header, blobs, offset = {}, [], 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    text = json.dumps(header).encode("utf-8") + b" " * pad_header
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(text)))
        f.write(text)
        for raw in blobs:
            f.write(raw)


def test_bf16_widening_is_exact_for_every_bit_pattern():
    import mlx.core as mx

    st = import_sibling("reference.safetensors_np")
    bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
    got = st.bf16_bits_to_f32(bits)
    assert got.dtype == np.float32
    # Definition: the float32 whose upper half is the bf16 pattern.
    np.testing.assert_array_equal(got.view(np.uint32), bits.astype(np.uint32) << 16)
    # Independent check against mlx's own bf16 -> f32 conversion (NaNs compared by bits).
    want = np.array(mx.array(bits).view(mx.bfloat16).astype(mx.float32))
    np.testing.assert_array_equal(got.view(np.uint32), want.view(np.uint32))


def test_header_parsing_and_offsets(tmp_path):
    st = import_sibling("reference.safetensors_np")
    a = np.arange(6, dtype="<f4").reshape(2, 3)
    b = np.array([0x3F80, 0x0001, 0x8000, 0x7F80], dtype="<u2")  # bf16: 1.0, a subnormal, -0.0, +inf
    c = np.array([2**40, -3], dtype="<i8")
    d = np.array([7, 0xFFFFFFFF], dtype="<u4")
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {
        "a": ("F32", a.shape, a.tobytes()),
        "b": ("BF16", b.shape, b.tobytes()),
        "c": ("I64", c.shape, c.tobytes()),
        "d": ("U32", d.shape, d.tobytes()),
    }, metadata={"format": "pt"}, pad_header=5)
    tensors, metadata, data_start = st.read_header(str(path))
    assert metadata == {"format": "pt"}
    assert set(tensors) == {"a", "b", "c", "d"}
    n = struct.unpack("<Q", path.read_bytes()[:8])[0]
    assert data_start == 8 + n
    assert tensors["a"]["begin"] == 0 and tensors["a"]["end"] == 24 and tensors["b"]["begin"] == 24

    ck = st.open_checkpoint(str(tmp_path))
    np.testing.assert_array_equal(ck.get("a"), a)
    np.testing.assert_array_equal(ck.get_rows("a", [1]), a[[1]])
    np.testing.assert_array_equal(ck.get_rows("a", slice(0, 1)), a[:1])
    np.testing.assert_array_equal(ck.get("b").view(np.uint32), b.astype(np.uint32) << 16)
    np.testing.assert_array_equal(ck.get_raw("b"), b)
    assert ck.get("c").dtype == np.int64 and ck.get("c").tolist() == [2**40, -3]
    got = ck.get("d")
    assert got.dtype == np.uint32 and got.tolist() == [7, 0xFFFFFFFF]
    assert ck.dtype_of("d") == "U32" and ck.shape_of("a") == (2, 3)


def test_header_refusals(tmp_path):
    st = import_sibling("reference.safetensors_np")
    with pytest.raises(ValueError, match="shorter than"):
        st.parse_header_bytes(b"\x01\x00")
    with pytest.raises(ValueError, match="implausible"):
        st.parse_header_bytes(struct.pack("<Q", 1 << 40))
    with pytest.raises(ValueError, match="claims"):
        st.parse_header_bytes(struct.pack("<Q", 100) + b"{}")
    bad = json.dumps({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}).encode()
    with pytest.raises(ValueError, match="needs 12 bytes"):
        st.parse_header_bytes(struct.pack("<Q", len(bad)) + bad)
    # A tensor running past the end of the file (a truncated download).
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, {"x": ("F32", (4,), np.zeros(4, "<f4").tobytes())})
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="past end of file"):
        st.read_header(str(path))
    # FP8 is not a dtype the reader decodes.
    _write_safetensors(path, {"w": ("F8_E4M3", (2,), b"\x00\x01")})
    with pytest.raises(TypeError, match="not supported"):
        st.open_checkpoint(str(tmp_path)).get("w")


def test_offsets_across_shards(tmp_path):
    """Tensors in two shards, found through the index; each tensor's offsets
    are relative to its own shard's data section."""
    import mlx.core as mx

    st = import_sibling("reference.safetensors_np")
    rng = np.random.default_rng(0)
    first = {"model.embed_tokens.weight": rng.standard_normal((5, 4)).astype(np.float32),
             "model.layers.0.w": rng.standard_normal((3,)).astype(np.float32)}
    second = {"lm_head.weight": rng.standard_normal((5, 4)).astype(np.float32)}
    # Shard 1 by mlx (bf16), shard 2 by hand (f32), with a long metadata block.
    mx.save_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"),
                        {k: mx.array(v).astype(mx.bfloat16) for k, v in first.items()}, metadata={"format": "mlx"})
    _write_safetensors(tmp_path / "model-00002-of-00002.safetensors",
                       {k: ("F32", v.shape, v.tobytes()) for k, v in second.items()},
                       metadata={"note": "x" * 1000})
    weight_map = {k: "model-00001-of-00002.safetensors" for k in first}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in second})
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    ck = st.open_checkpoint(str(tmp_path))
    assert ck.names() == sorted(weight_map)
    for name, value in first.items():
        want = np.array(mx.array(value).astype(mx.bfloat16).astype(mx.float32))
        np.testing.assert_array_equal(ck.get(name), want)
    np.testing.assert_array_equal(ck.get("lm_head.weight"), second["lm_head.weight"])
    np.testing.assert_array_equal(ck.get_rows("lm_head.weight", slice(2, 4)), second["lm_head.weight"][2:4])
    # The index must agree with the shard headers.
    weight_map["lm_head.weight"] = "model-00001-of-00002.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="index"):
        st.open_checkpoint(str(tmp_path))
